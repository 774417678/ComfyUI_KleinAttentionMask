"""Native sampler adapters that record what the chosen mode needs.

``mode`` lives here because *which* data must be recorded has to be decided
before diffusion: a neutral prediction only exists while the model sits at that
exact noisy state. Recording is exactly as expensive as the old single-node
build of the same mode:

    source_latent / adjacent_prediction   no extra forward
    fixed_neutral                         one extra forward (call 0)
    sparse_neutral                        probe_steps extra forwards
    dense_neutral / compare_all           one extra forward per call in the window

A finished record is cached in this module, keyed by the identities of every
input that shaped the sampling. Re-running the node with a cheaper mode (or a
narrower window) reuses that record and skips diffusion entirely: coverage is
downward compatible. Changing an upstream input changes the key, so diffusion
runs again.
"""
import copy
import json
import time
import weakref

import torch
import nodes
import comfy.samplers
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SamplerCustom, Guider_Basic
from comfy.model_patcher import create_model_options_clone

from .neutral_conditioning import neutral_embedding, encode_neutral_conditioning
from .klein_support import flux2_variant
from .internal_sampler import MODES as MODE_ORDER
from .sampling_record import (SamplingRecord, SOURCE_REFERENCES, NEUTRAL_MODES,  # noqa: F401
                              required_neutral)

RECORD_TYPE = 'KLEIN_SAMPLING_RECORD'
RECORD_OUT = ('KLEIN_SAMPLING_RECORD', 'STRING')
RECORD_NAMES = ('sampling_record', 'diagnostics')

_MAX_RECORDS = 3
_CACHE = {}


def options():
    return {
        'mode': (list(MODE_ORDER), {'default': 'source_latent'}),
        'capture_start': ('INT', {'default': 0, 'min': 0, 'max': 10000,
                                  'tooltip': 'First recorded model call used by the chosen mode (0 = first).'}),
        'capture_end': ('INT', {'default': -1, 'min': -1, 'max': 10000,
                                'tooltip': '-1 means the last call. Together with the mode this decides what gets recorded.'}),
        'probe_steps': ('INT', {'default': 2, 'min': 1, 'max': 100,
                                'tooltip': 'Only sparse_neutral: how many captures inside the window get a neutral pair.'}),
        'clip': ('CLIP',),
        'source_latent': ('LATENT',),
        'neutral_conditioning': ('CONDITIONING',),
    }


def _cache_key(objects):
    return tuple(id(o) for o in objects)


def _ref(obj):
    """Describe one key object: None, a weakref when possible, else a strong ref.

    Lists (conditioning) are not weak-referenceable; keeping a strong reference is
    safe here because the cache holds at most a few entries and the identity check
    is what prevents stale reuse.
    """
    if obj is None:
        return (0, None)
    try:
        return (1, weakref.ref(obj))
    except TypeError:
        return (2, obj)


def _cache_get(key, objects):
    entry = _CACHE.get(key)
    if entry is None:
        return None
    refs, record = entry
    if len(refs) != len(objects):
        _CACHE.pop(key, None); return None
    for (kind, holder), obj in zip(refs, objects):
        if kind == 0:
            if obj is not None:
                _CACHE.pop(key, None); return None
        elif kind == 1:
            if holder() is not obj:      # an input object was replaced upstream
                _CACHE.pop(key, None); return None
        elif holder is not obj:
            _CACHE.pop(key, None); return None
    return record


def _cache_put(key, objects, record):
    while len(_CACHE) >= _MAX_RECORDS:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = ([_ref(o) for o in objects], record)


def _upstream_objects(model, noise, sampler, sigmas, latent, source_latent, conditioning):
    items = [model, noise, sampler, sigmas, latent['samples']]
    items.append(None if source_latent is None else source_latent['samples'])
    items.extend(conditioning)
    return items


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
    """Runs diffusion while recording exactly the calls the mode requires."""

    def __init__(self, model, latent, total, model_options=None, **kw):
        opts = model.model_options if model_options is None else model_options
        if opts.get('model_function_wrapper') or model.model_options.get('model_function_wrapper'):
            raise ValueError('Existing model_function_wrapper is unsupported; use an unwrapped model.')
        if opts.get('sampler_calc_cond_batch_function'):
            raise ValueError('Custom conditioning executor is unsupported.')
        self.kw = {k: v for k, v in kw.items() if k not in ('clip', 'source_latent', 'neutral_conditioning')}
        self.mode = kw.get('mode', 'source_latent')
        if self.mode not in MODE_ORDER:
            raise ValueError('Unknown mask mode: ' + str(self.mode))
        dm = model.get_model_object('diffusion_model')
        self.context_in_dim = flux2_variant(dm)   # native Klein 9B (12288) or 4B (7680)
        src = kw.get('source_latent', latent)['samples']
        if src.is_nested or src.ndim != 4:
            raise ValueError('Only regular image latents [B,C,H,W] are supported, not nested/video latents.')
        src_origin = 'source_latent_input' if 'source_latent' in kw else 'latent_image'
        nm = latent.get('noise_mask')
        self.capture_start = int(kw.get('capture_start', 0))
        self.capture_end = int(kw.get('capture_end', -1))
        self.probe_steps = int(kw.get('probe_steps', 2))
        self.record = SamplingRecord(
            src_input=model.model.process_latent_in(src.detach().cpu().clone()).float(),
            src_origin=src_origin,
            noise_mask=None if nm is None else nm.detach().cpu(),
            total_calls=total, mode=self.mode,
            capture_start=self.capture_start, capture_end=self.capture_end,
            probe_steps=self.probe_steps)
        self.required = required_neutral(self.mode, self.capture_start, self.capture_end,
                                         self.probe_steps, total)
        self.neutral = None; self.neutral_source = 'unused'
        if self.required:
            neutral = kw.get('neutral_conditioning')
            if neutral is not None:
                self.neutral = neutral_embedding(neutral)
                self.neutral_source = 'conditioning'
            else:
                clip = kw.get('clip')
                if clip is None:
                    raise ValueError('Mode %s needs neutral predictions: connect KleinNeutralConditioning to neutral_conditioning (or a CLIP).' % self.mode)
                self.neutral = neutral_embedding(encode_neutral_conditioning(clip))
                self.neutral_source = 'legacy_clip'
                print('[KleinMaskAdapter] Legacy CLIP encoding inside sampler; connect KleinNeutralConditioning to avoid repeated text-encoder loads.')
        self.sigmas = []
        self.neutral_calls = []
        self.reference_found = False
        self.t = time.perf_counter()
        self.patched = model.clone()
        if model_options is not None:
            self.patched.model_options = create_model_options_clone(opts)
        self.patched.set_model_unet_function_wrapper(self.wrap)

    def close(self):
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
        if idx in self.required:
            x, t, c, chunks, pos = positive_replay_args(args, self.neutral)
            with torch.no_grad():
                neutral = apply(x, t, **c).chunk(chunks)[pos].detach().cpu()
            del x, t, c
            self.neutral_calls.append(idx)
        self.record.add_step(idx, float(args['timestep'].flatten()[0]), edited.detach().cpu().clone(), neutral)
        return out

    def finish(self):
        self.record.neutral_calls = sorted(self.neutral_calls)
        info = json.dumps({
            'record': self.record.describe(), 'neutral_source': self.neutral_source,
            'recorded_calls': len(self.sigmas), 'extra_neutral_forwards': len(self.neutral_calls),
            'neutral_calls': self.neutral_calls,
            'elapsed_seconds': round(time.perf_counter() - self.t, 3),
            'note': 'influence maps are computed by KleinInfluenceMap from this record',
        }, ensure_ascii=False)
        print('[KleinMaskAdapter] ' + info)
        return self.record, info


def _reuse(record, mode, capture_start, capture_end, probe_steps):
    """Adopt a cheaper mode on an existing record when coverage allows it."""
    if record is None or not record.covers(mode, capture_start, capture_end, probe_steps):
        return None
    record.adopt(mode, capture_start, capture_end, probe_steps)
    info = json.dumps({
        'record': record.describe(), 'neutral_source': 'reused_recording',
        'recorded_calls': len(record.steps), 'extra_neutral_forwards': 0, 'neutral_calls': record.neutral_calls,
        'reused': True, 'note': 'diffusion skipped: stored record already covers this mode',
    }, ensure_ascii=False)
    print('[KleinMaskAdapter] reuse ' + info)
    return record, info


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
        mode = kw.get('mode', 'source_latent')
        cs, ce, probe = int(kw.get('capture_start', 0)), int(kw.get('capture_end', -1)), int(kw.get('probe_steps', 2))
        objects = _upstream_objects(guider.model_patcher, noise, sampler, sigmas, latent_image,
                                    kw.get('source_latent'),
                                    [guider, kw.get('neutral_conditioning'), kw.get('clip')])
        key = _cache_key(objects)
        hit = _reuse(_cache_get(key, objects), mode, cs, ce, probe)
        if hit is not None:
            record, info = hit
            return (record.output, record.denoised_output, record, info)
        with Recorder(guider.model_patcher, latent_image, sigmas.numel() - 1, model_options=guider.model_options, **kw) as rec:
            g = copy.copy(guider)
            g.model_patcher = rec.patched; g.model_options = rec.patched.model_options
            g.original_conds = {k: [dict(c) for c in conds] for k, conds in guider.original_conds.items()}
            try:
                result = SamplerCustomAdvanced.execute(noise, g, sampler, sigmas, latent_image)
                rec.record.output, rec.record.denoised_output = result[0], result[1]
                record, info = rec.finish()
                _cache_put(key, objects, record)
                return (result[0], result[1], record, info)
            finally:
                g.__dict__.clear()


class KleinMaskKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        d = copy.deepcopy(nodes.KSampler.INPUT_TYPES()); d['optional'] = options(); return d

    RETURN_TYPES = ('LATENT',) + RECORD_OUT
    RETURN_NAMES = ('samples',) + RECORD_NAMES
    FUNCTION = 'sample'; CATEGORY = 'KleinMask/sampler variants'

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1., **kw):
        mode = kw.get('mode', 'source_latent')
        cs, ce, probe = int(kw.get('capture_start', 0)), int(kw.get('capture_end', -1)), int(kw.get('probe_steps', 2))
        objects = [model, sampler_name, scheduler, float(cfg), int(seed), int(steps), float(denoise),
                   latent_image['samples'], positive, negative, kw.get('neutral_conditioning'), kw.get('clip')]
        key = _cache_key(objects)
        hit = _reuse(_cache_get(key, objects), mode, cs, ce, probe)
        if hit is not None:
            record, info = hit
            return (record.output, record, info)
        with Recorder(model, latent_image, steps, **kw) as rec:
            result = nodes.KSampler().sample(rec.patched, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            rec.record.output = result[0]
            record, info = rec.finish()
            _cache_put(key, objects, record)
            return (result[0], record, info)


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
        mode = kw.get('mode', 'source_latent')
        cs, ce, probe = int(kw.get('capture_start', 0)), int(kw.get('capture_end', -1)), int(kw.get('probe_steps', 2))
        objects = [model, sampler_name, scheduler, float(cfg), int(noise_seed), int(steps), bool(add_noise),
                   int(start_at_step), int(end_at_step), str(return_with_leftover_noise),
                   latent_image['samples'], positive, negative, kw.get('neutral_conditioning'), kw.get('clip')]
        key = _cache_key(objects)
        hit = _reuse(_cache_get(key, objects), mode, cs, ce, probe)
        if hit is not None:
            record, info = hit
            return (record.output, record, info)
        with Recorder(model, latent_image, total, **kw) as rec:
            result = nodes.KSamplerAdvanced().sample(rec.patched, add_noise, noise_seed, steps, cfg, sampler_name,
                                                     scheduler, positive, negative, latent_image, start_at_step,
                                                     end_at_step, return_with_leftover_noise)
            rec.record.output = result[0]
            record, info = rec.finish()
            _cache_put(key, objects, record)
            return (result[0], record, info)


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
        mode = kw.get('mode', 'source_latent')
        cs, ce, probe = int(kw.get('capture_start', 0)), int(kw.get('capture_end', -1)), int(kw.get('probe_steps', 2))
        objects = [model, add_noise, int(noise_seed), float(cfg), sampler, sigmas, latent_image['samples'],
                   positive, negative, kw.get('neutral_conditioning'), kw.get('clip')]
        key = _cache_key(objects)
        hit = _reuse(_cache_get(key, objects), mode, cs, ce, probe)
        if hit is not None:
            record, info = hit
            return (record.output, record.denoised_output, record, info)
        with Recorder(model, latent_image, max(1, sigmas.numel() - 1), **kw) as rec:
            result = SamplerCustom.execute(rec.patched, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)
            rec.record.output, rec.record.denoised_output = result[0], result[1]
            record, info = rec.finish()
            _cache_put(key, objects, record)
            return (result[0], result[1], record, info)
