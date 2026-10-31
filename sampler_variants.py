"""Native sampler adapters that record per-step data for deferred influence maps.

Sampling runs exactly once. Every positive model call is stored on CPU together
with its state, so mask/mode/capture/threshold-style parameters live in the
separate ``KleinInfluenceMap`` node and never trigger another diffusion pass.
"""
import copy
import json
import time
import torch
import nodes
import comfy.samplers
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SamplerCustom, Guider_Basic
from comfy.model_patcher import create_model_options_clone
from .neutral_conditioning import neutral_embedding, encode_neutral_conditioning
from .sampling_record import SamplingRecord, SOURCE_REFERENCES  # noqa: F401  (SOURCE_REFERENCES re-exported)

RECORD_TYPE = 'KLEIN_SAMPLING_RECORD'
RECORD_OUT = ('KLEIN_SAMPLING_RECORD', 'STRING')
RECORD_NAMES = ('sampling_record', 'diagnostics')


def options():
    return {
        'record_neutral': ('BOOLEAN', {
            'default': False, 'label_on': '记录中性预测', 'label_off': '不记录',
            'tooltip': 'Off: only zero-cost modes (source_latent, adjacent_prediction). On: also run one neutral forward per call and store it, so neutral modes can be applied later without resampling. Cost is one extra forward per call.'}),
        'clip': ('CLIP',),
        'source_latent': ('LATENT',),
        'neutral_conditioning': ('CONDITIONING',),
    }


def positive_replay_args(args, neutral):
    """Replay only the positive CFG block for native, unpatched conditioning.

    Unknown batch-shaped values fail closed. Transformer patches can define their
    own batch semantics, so retain the original full-batch path for those.
    """
    flags = args['cond_or_uncond']
    x = args['input']; t = args['timestep']; c = dict(args['c'])
    if flags.count(0) != 1 or x.shape[0] % len(flags):
        raise ValueError('Unsupported conditioning batching for positive-only neutral replay.')
    pos = flags.index(0)
    if c.get('control') is not None:
        raise ValueError('Neutral replay with ControlNet is unsupported.')
    cross = c.get('c_crossattn')
    if cross is None:
        raise ValueError('No c_crossattn in guider prediction; incompatible guider.')
    to = dict(c.get('transformer_options', {}))
    replay_chunks = len(flags)
    if replay_chunks > 1 and not any(to.get(k) for k in ('patches', 'patches_replace', 'wrappers')):
        rows = x.shape[0]; batch = rows // len(flags); sl = slice(pos * batch, (pos + 1) * batch)

        def select(value):
            if torch.is_tensor(value):
                if value.ndim == 0:
                    return value
                if value.shape[0] == rows:
                    return value[sl]
                if value.shape[0] == 1:
                    return value
                raise ValueError('Unsupported conditioning batch shape for neutral replay: ' + str(tuple(value.shape)))
            if isinstance(value, list):
                return [select(v) for v in value]
            if isinstance(value, tuple):
                return tuple(select(v) for v in value)
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            raise ValueError('Unsupported conditioning type for positive-only neutral replay: ' + type(value).__name__)
        c = {k: (v if k == 'transformer_options' else select(v)) for k, v in c.items()}
        x = x[sl]; t = select(t)
        to['cond_or_uncond'] = [0]
        if 'uuids' in to:
            to['uuids'] = [to['uuids'][pos]]
        replay_chunks = 1; pos = 0
    c['c_crossattn'] = neutral.to(cross).expand(x.shape[0], -1, -1)
    c['transformer_options'] = to
    return x, t, c, replay_chunks, pos


class Recorder:
    """Records diffusion calls; produces a SamplingRecord, not masks."""

    def __init__(self, model, latent, total, model_options=None, **kw):
        opts = model.model_options if model_options is None else model_options
        if opts.get('model_function_wrapper') or model.model_options.get('model_function_wrapper'):
            raise ValueError('Existing model_function_wrapper is unsupported; use an unwrapped model.')
        if opts.get('sampler_calc_cond_batch_function'):
            raise ValueError('Custom conditioning executor is unsupported.')
        # Keep scalar settings, not CLIP/LATENT objects, in metadata.
        self.kw = {k: v for k, v in kw.items() if k not in ('clip', 'source_latent', 'neutral_conditioning')}
        dm = model.get_model_object('diffusion_model')
        if type(dm).__module__ != 'comfy.ldm.flux.model' or getattr(dm.params, 'context_in_dim', 0) != 12288:
            raise ValueError('Native Flux2 Klein9B required; third-party/KV engines not validated.')
        src = kw.get('source_latent', latent)['samples']
        if src.is_nested or src.ndim != 4:
            raise ValueError('Only regular image latents [B,C,H,W] are supported, not nested/video latents.')
        src_origin = 'source_latent_input' if 'source_latent' in kw else 'latent_image'
        nm = latent.get('noise_mask')
        self.record = SamplingRecord(
            src_input=model.model.process_latent_in(src.detach().cpu().clone()).float(),
            src_origin=src_origin,
            noise_mask=None if nm is None else nm.detach().cpu(),
            total_calls=total)
        self.record_neutral = bool(kw.get('record_neutral', False))
        self.neutral = None; self.neutral_source = 'unused'
        if self.record_neutral:
            neutral = kw.get('neutral_conditioning')
            if neutral is not None:
                self.neutral = neutral_embedding(neutral)
                self.neutral_source = 'conditioning'
            else:
                clip = kw.get('clip')
                if clip is None:
                    raise ValueError('record_neutral needs neutral_conditioning (KleinNeutralConditioning) or a CLIP input.')
                self.neutral = neutral_embedding(encode_neutral_conditioning(clip))
                self.neutral_source = 'legacy_clip'
                print('[KleinMaskAdapter] Legacy CLIP encoding inside sampler; connect KleinNeutralConditioning to avoid repeated text-encoder loads.')
        self.sigmas = []
        self.neutral_calls = []
        self.reference_found = False
        self.t = time.perf_counter()
        # Clone exactly once from the cached upstream patcher. A temporary parent
        # plus a bound-method cycle can be GC'd together; LoadedModel then loses
        # both weakrefs even while the raw model stays cached on the GPU.
        self.patched = model.clone()
        if model_options is not None:
            self.patched.model_options = create_model_options_clone(opts)
        self.patched.set_model_unet_function_wrapper(self.wrap)

    def close(self):
        """Break wrapper ownership on success, errors and cancellation (idempotent)."""
        patched = self.patched
        if patched is not None:
            wrapper = patched.model_options.get('model_function_wrapper')
            if getattr(wrapper, '__self__', None) is self:
                patched.model_options.pop('model_function_wrapper', None)
        self.patched = None
        self.neutral = None
        self.kw.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _capture_reference(self, args, flags, edited):
        """Store the edit's own reference image once, in prediction coordinates."""
        if self.reference_found:
            return
        refs = args['c'].get('ref_latents')
        if not isinstance(refs, (list, tuple)) or not refs:
            return
        r = refs[0]
        if not torch.is_tensor(r) or r.ndim != 4:
            return
        rows = len(flags); pos = flags.index(0)
        if r.shape[0] == edited.shape[0] * rows:
            r = r.chunk(rows)[pos]
        if r.shape != edited.shape:
            return
        self.record.reference = r.detach().float().cpu().clone()
        self.reference_found = True

    def wrap(self, apply, args):
        out = apply(args['input'], args['timestep'], **args['c'])
        flags = args.get('cond_or_uncond', [])
        if 0 not in flags:
            return out
        if flags.count(0) != 1 or out.shape[0] % len(flags):
            raise ValueError('Unsupported conditioning batching for internal mask collection.')
        idx = len(self.sigmas)
        self.sigmas.append(float(args['timestep'].flatten()[0]))
        edited = out.chunk(len(flags))[flags.index(0)]
        self._capture_reference(args, flags, edited)
        expected = self.record.reference if self.reference_found else self.record.src_input
        if edited.shape != expected.shape:
            raise ValueError('Source latent and positive prediction shapes differ. Connect matching source_latent; regional conditioning is not supported.')
        neutral = None
        if self.record_neutral:
            x, t, c, chunks, pos = positive_replay_args(args, self.neutral)
            with torch.no_grad():
                neutral = apply(x, t, **c).chunk(chunks)[pos].detach().cpu()
            del x, t, c
            self.neutral_calls.append(idx)
        self.record.add_step(idx, float(args['timestep'].flatten()[0]),
                             edited.detach().cpu().clone(), neutral)
        return out

    def finish(self):
        info = json.dumps({
            'record': self.record.describe(), 'neutral_source': self.neutral_source,
            'recorded_calls': len(self.sigmas), 'extra_neutral_forwards': len(self.neutral_calls),
            'neutral_calls': self.neutral_calls,
            'elapsed_seconds': round(time.perf_counter() - self.t, 3),
            'note': 'influence maps are computed by KleinInfluenceMap from this record',
        }, ensure_ascii=False)
        print('[KleinMaskAdapter] ' + info)
        return self.record, info


class KleinMaskSamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'noise': ('NOISE',), 'guider': ('GUIDER',), 'sampler': ('SAMPLER',),
                             'sigmas': ('SIGMAS',), 'latent_image': ('LATENT',)}, 'optional': options()}

    RETURN_TYPES = ('LATENT', 'LATENT') + RECORD_OUT
    RETURN_NAMES = ('output', 'denoised_output') + RECORD_NAMES
    FUNCTION = 'sample'; CATEGORY = 'KleinMask/sampler variants'

    def sample(self, noise, guider, sampler, sigmas, latent_image, **kw):
        import inspect
        from pathlib import Path
        native_file = Path(inspect.getfile(Guider_Basic)).resolve()
        native_basic = type(guider).__name__ == 'Guider_Basic' and Path(inspect.getfile(type(guider))).resolve() == native_file
        if type(guider) is not comfy.samplers.CFGGuider and not native_basic:
            raise ValueError('Only native BasicGuider and CFGGuider supported; DualCFG/custom guiders have different branch semantics.')
        if sigmas.numel() < 2:
            raise ValueError('At least two sigma values required for a sampling record.')
        with Recorder(guider.model_patcher, latent_image, sigmas.numel() - 1, model_options=guider.model_options, **kw) as rec:
            # Preserve guider-specific options, but do not create another patcher.
            g = copy.copy(guider)
            g.model_patcher = rec.patched; g.model_options = rec.patched.model_options
            g.original_conds = {k: [dict(c) for c in conds] for k, conds in guider.original_conds.items()}
            try:
                result = SamplerCustomAdvanced.execute(noise, g, sampler, sigmas, latent_image)
                record, info = rec.finish()
                return (result[0], result[1], record, info)
            finally:
                # A traceback may keep this private guider alive after an error.
                # Never retain the temporary patcher/conditioning through it.
                g.__dict__.clear()


class KleinMaskKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        d = copy.deepcopy(nodes.KSampler.INPUT_TYPES()); d['optional'] = options(); return d

    RETURN_TYPES = ('LATENT',) + RECORD_OUT
    RETURN_NAMES = ('samples',) + RECORD_NAMES
    FUNCTION = 'sample'; CATEGORY = 'KleinMask/sampler variants'

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1., **kw):
        with Recorder(model, latent_image, steps, **kw) as rec:
            result = nodes.KSampler().sample(rec.patched, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            record, info = rec.finish()
            return (*result, record, info)


class KleinMaskKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        d = copy.deepcopy(nodes.KSamplerAdvanced.INPUT_TYPES()); d['optional'] = options(); return d

    RETURN_TYPES = ('LATENT',) + RECORD_OUT
    RETURN_NAMES = ('samples',) + RECORD_NAMES
    FUNCTION = 'sample'; CATEGORY = 'KleinMask/sampler variants'

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, start_at_step, end_at_step, return_with_leftover_noise, **kw):
        total = max(1, min(steps, end_at_step) - start_at_step)
        with Recorder(model, latent_image, total, **kw) as rec:
            result = nodes.KSamplerAdvanced().sample(rec.patched, add_noise, noise_seed, steps, cfg, sampler_name,
                                                     scheduler, positive, negative, latent_image, start_at_step,
                                                     end_at_step, return_with_leftover_noise)
            record, info = rec.finish()
            return (*result, record, info)


class KleinMaskSamplerCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'add_noise': ('BOOLEAN', {'default': True}),
                             'noise_seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
                             'cfg': ('FLOAT', {'default': 1., 'min': 0., 'max': 100., 'step': .1}),
                             'positive': ('CONDITIONING',), 'negative': ('CONDITIONING',),
                             'sampler': ('SAMPLER',), 'sigmas': ('SIGMAS',), 'latent_image': ('LATENT',)}, 'optional': options()}

    RETURN_TYPES = ('LATENT', 'LATENT') + RECORD_OUT
    RETURN_NAMES = ('output', 'denoised_output') + RECORD_NAMES
    FUNCTION = 'sample'; CATEGORY = 'KleinMask/sampler variants'

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, **kw):
        with Recorder(model, latent_image, max(1, sigmas.numel() - 1), **kw) as rec:
            result = SamplerCustom.execute(rec.patched, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)
            record, info = rec.finish()
            return (result[0], result[1], record, info)
