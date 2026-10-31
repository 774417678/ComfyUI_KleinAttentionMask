"""Sampling record: per-step data captured during diffusion.

Which data must be recorded is decided *before* sampling, because a neutral
prediction can only be produced while the model is at that exact noisy state.
The sampler therefore owns ``mode`` (plus capture window and probe count) and
records exactly what that mode needs:

    source_latent / adjacent_prediction   no extra forward
    fixed_neutral                         one extra forward (call 0)
    sparse_neutral                        probe_steps extra forwards
    dense_neutral / compare_all           one extra forward per call in the window

Coverage is *downward compatible*: a record that already holds every neutral it
could need also serves the cheaper modes, so switching dense -> fixed/source
reuses the stored sampling instead of running diffusion again.

Pure CPU arithmetic here; nothing in this module touches the model.
"""
import json
import torch
import torch.nn.functional as F

from .internal_sampler import MODES, METRICS, influence_map

SOURCE_REFERENCES = ['auto', 'reference_latent', 'source_latent_input']
NEUTRAL_MODES = ('fixed_neutral', 'sparse_neutral', 'dense_neutral', 'compare_all')


def pick_capture_steps(start, end, probe_steps, total):
    """Indices used for sparse neutral capture."""
    if end < start:
        raise ValueError('capture_end < capture_start')
    last = max(total - 1, 0)
    lo = min(start, last)
    hi = last if end > last else min(end, last)
    span = torch.linspace(lo, hi, min(max(probe_steps, 1), hi - lo + 1))
    return set(span.round().long().tolist())


def required_neutral(mode, capture_start, capture_end, probe_steps, total):
    """Model calls that need a neutral prediction recorded for this mode."""
    if mode not in MODES:
        raise ValueError('Unknown mask mode: ' + str(mode))
    if mode not in NEUTRAL_MODES:
        return set()
    end = total - 1 if capture_end < 0 else min(capture_end, total - 1)
    if mode == 'fixed_neutral':
        return {0} if capture_start <= 0 else set()
    if mode == 'sparse_neutral':
        return pick_capture_steps(capture_start, end, probe_steps, total)
    return set(range(capture_start, end + 1))


class SamplingRecord:
    """One sampling run, held by ComfyUI's node output cache."""

    TYPE = 'KLEIN_SAMPLING_RECORD'

    def __init__(self, src_input, src_origin, noise_mask, total_calls, mode, capture_start, capture_end, probe_steps):
        self.src_input = src_input
        self.src_origin = src_origin
        self.reference = None
        self.noise_mask = noise_mask
        self.total_calls = total_calls
        self.mode = mode
        self.capture_start = capture_start
        self.capture_end = capture_end
        self.probe_steps = probe_steps
        self.steps = []
        self.output = None            # sampled LATENT, reused when the record is reused
        self.denoised_output = None
        self.reused = False

    def add_step(self, idx, timestep, denoised, neutral=None):
        self.steps.append({'idx': idx, 'timestep': timestep, 'sigma': timestep,
                           'denoised': denoised, 'neutral': neutral})

    @property
    def neutral_indices(self):
        return {s['idx'] for s in self.steps if s.get('neutral') is not None}

    @property
    def neutral_enabled(self):
        return bool(self.neutral_indices)

    def step(self, idx):
        for record in self.steps:
            if record['idx'] == idx:
                return record
        return None

    def required_for(self, mode, capture_start, capture_end, probe_steps):
        total = max(self.total_calls, len(self.steps))
        return required_neutral(mode, capture_start, capture_end, probe_steps, total)

    def covers(self, mode, capture_start, capture_end, probe_steps):
        """True when every neutral this mode needs is already stored."""
        return self.required_for(mode, capture_start, capture_end, probe_steps) <= self.neutral_indices

    def adopt(self, mode, capture_start, capture_end, probe_steps):
        """Serve a cheaper mode from the stored data without resampling."""
        self.mode = mode
        self.capture_start = capture_start
        self.capture_end = capture_end
        self.probe_steps = probe_steps
        self.reused = True

    def source_for(self, source_reference):
        if source_reference not in SOURCE_REFERENCES:
            raise ValueError('Unknown source_reference: ' + str(source_reference))
        if source_reference == 'source_latent_input':
            return self.src_input, 'source_latent_input'
        if self.reference is not None:
            return self.reference, 'reference_latent'
        if source_reference == 'reference_latent':
            raise ValueError('source_reference=reference_latent, but no ReferenceLatent was present in the positive conditioning.')
        return self.src_input, self.src_origin

    def describe(self):
        return {'mode': self.mode, 'steps': len(self.steps), 'total_calls': self.total_calls,
                'capture_window': [self.capture_start, self.capture_end],
                'probe_steps': self.probe_steps,
                'neutral_recorded_calls': sorted(self.neutral_indices),
                'source_origin': self.src_origin, 'has_reference': self.reference is not None,
                'reused_without_resampling': self.reused,
                'sigmas': [round(float(s['sigma']), 6) for s in self.steps]}


def build_maps(record, common_influence, source_reference):
    """Recompute influence maps from a record. Mode/window come from the record."""
    mode = record.mode
    if mode not in MODES:
        raise ValueError('Unknown mask mode: ' + str(mode))
    active = METRICS if mode == 'compare_all' else [mode]
    total = max(record.total_calls, len(record.steps))
    capture_start = record.capture_start
    capture_end = total - 1 if record.capture_end < 0 else min(record.capture_end, total - 1)
    if capture_end < capture_start:
        raise ValueError('capture_end < capture_start')
    source, source_origin = record.source_for(source_reference)
    sparse = pick_capture_steps(capture_start, capture_end, record.probe_steps, total)
    missing = record.required_for(mode, record.capture_start, record.capture_end, record.probe_steps) - record.neutral_indices
    if missing:
        raise ValueError('Mode %s needs neutral predictions at calls %s, which were not recorded. Re-sample with this mode (or with dense_neutral/compare_all) once; the record is then reusable for cheaper modes.'
                         % (mode, sorted(missing)))
    combine = 'min' if common_influence else 'mean'
    height, width = source.shape[-2] * 16, source.shape[-1] * 16
    caps = {m: [] for m in active}
    at = {m: [] for m in active}
    for step in record.steps:
        idx = step['idx']
        if not (capture_start <= idx <= capture_end):
            continue
        edited = step['denoised']
        if edited.shape != source.shape:
            raise ValueError('Source latent and positive prediction shapes differ; connect a matching source_latent.')
        for m in active:
            ref = None
            if m == 'source_latent':
                ref = source
            elif m == 'fixed_neutral':
                first = record.step(0)
                ref = None if first is None else first.get('neutral')
            elif m == 'adjacent_prediction':
                prev = record.step(idx - 1)
                ref = None if prev is None else prev['denoised']
            elif m == 'sparse_neutral' and idx in sparse:
                ref = step.get('neutral')
            elif m == 'dense_neutral':
                ref = step.get('neutral')
            if ref is not None:
                caps[m].append(_rms(edited, ref))
                at[m].append(idx)
    maps = {}
    for m in active:
        if not caps[m]:
            raise ValueError('No collected predictions for ' + m + '. Widen capture_start/capture_end (adjacent needs at least two calls).')
        heat = influence_map(caps[m], height, width, combine)
        if record.noise_mask is not None:
            heat = _apply_noise_mask(heat, record.noise_mask)
        maps[m] = heat
    info = {'mode': mode, 'combine': 'common_min' if common_influence else 'mean',
            'source_reference': source_reference, 'source_origin': source_origin,
            'comparison_order': active, 'captured_calls': at,
            'nominal_capture_window': [capture_start, capture_end],
            'sparse_capture_steps': sorted(sparse),
            'index_semantics': 'positive model invocations, not scheduler steps; CFG raw positive branch, not guided result',
            'positive_calls': len(record.steps),
            'neutral_calls': sorted(record.neutral_indices),
            'noise_mask_applied': record.noise_mask is not None,
            'reused_without_resampling': record.reused, 'experimental': True}
    return maps, info


def _rms(a, b):
    return (a.float() - b.float()).square().mean(1).sqrt().detach().cpu()


def _apply_noise_mask(heat, noise_mask):
    nm = noise_mask.detach().cpu().reshape(-1, 1, *noise_mask.shape[-2:]).float()
    nm = F.interpolate(nm, size=heat.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    if nm.shape[0] == 1:
        nm = nm.expand(heat.shape[0], -1, -1)
    if nm.shape[0] != heat.shape[0]:
        raise ValueError('noise_mask batch differs from source batch')
    return heat * nm.clamp(0, 1)
