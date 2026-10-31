"""Sampling record: per-step data captured during diffusion, plus deferred maps.

The sampler runs diffusion exactly once and stores, for every positive model call:
  * x_in / timestep (the state that call saw)
  * the edited denoised prediction
  * optionally the neutral prediction for the same state (``record_neutral``)

All influence-map parameters (mode, capture window, probe_steps, common influence,
which latent counts as "original") are applied later by ``KleinInfluenceMap``.
Changing them therefore never repeats sampling, and recomputing an already
recorded neutral branch is free.

This module never touches the model; it only does CPU arithmetic.
"""
import json
import torch

from .internal_sampler import MODES, METRICS, influence_map

SOURCE_REFERENCES = ['auto', 'reference_latent', 'source_latent_input']


class SamplingRecord:
    """Container for one sampling run. Held by ComfyUI's node output cache."""

    TYPE = 'KLEIN_SAMPLING_RECORD'

    def __init__(self, src_input, src_origin, noise_mask, total_calls):
        self.src_input = src_input            # process_latent_in(source_latent input or latent_image)
        self.src_origin = src_origin          # 'source_latent_input' | 'latent_image'
        self.reference = None                 # first ReferenceLatent seen, already process_latent_in'ed
        self.noise_mask = noise_mask
        self.total_calls = total_calls
        self.steps = []                       # dicts: idx, timestep, sigma, x_in, denoised, neutral
        self.neutral_enabled = False
        self._neutral_by_index = {}

    # ---- recording -------------------------------------------------------
    def add_step(self, idx, timestep, denoised, neutral=None):
        """Store one positive call. Only the denoised prediction is needed:
        every mode compares predictions against another prediction (source,
        previous step, or the neutral one), never against the noisy input."""
        record = {'idx': idx, 'timestep': timestep, 'sigma': timestep,
                  'denoised': denoised, 'neutral': None}
        self.steps.append(record)
        if neutral is not None:
            record['neutral'] = neutral
            self._neutral_by_index[idx] = neutral
            self.neutral_enabled = True

    def set_neutral(self, idx, neutral):
        """Attach a neutral prediction (also used to fill in after the fact)."""
        for record in self.steps:
            if record['idx'] == idx:
                record['neutral'] = neutral
        self._neutral_by_index[idx] = neutral
        self.neutral_enabled = True

    # ---- accessors -------------------------------------------------------
    def step(self, idx):
        for record in self.steps:
            if record['idx'] == idx:
                return record
        return None

    def source_for(self, source_reference):
        """Resolve which latent counts as the untouched original."""
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
        return {'steps': len(self.steps), 'total_calls': self.total_calls,
                'source_origin': self.src_origin, 'has_reference': self.reference is not None,
                'neutral_recorded': self.neutral_enabled,
                'sigmas': [round(float(s['sigma']), 6) for s in self.steps]}


def pick_capture_steps(start, end, probe_steps, total):
    """Indices used for sparse neutral capture, matching the original Recorder."""
    if end < start:
        raise ValueError('capture_end < capture_start')
    last = total - 1
    lo = min(start, last)
    hi = last if end > last else min(end, last)
    span = torch.linspace(lo, hi, min(max(probe_steps, 1), hi - lo + 1))
    return set(span.round().long().tolist())


def build_maps(record, mode, capture_start, capture_end, probe_steps, common_influence, source_reference):
    """Recompute influence maps from a record. Pure post-processing.

    Returns (maps dict mode -> heat, diagnostics dict).
    """
    if mode not in MODES:
        raise ValueError('Unknown mask mode: ' + str(mode))
    active = METRICS if mode == 'compare_all' else [mode]
    total = max(record.total_calls, len(record.steps))
    if capture_end < 0:                       # -1 means the last recorded call
        capture_end = total - 1
    if capture_end < capture_start:
        raise ValueError('capture_end < capture_start')
    source, source_origin = record.source_for(source_reference)
    sparse = pick_capture_steps(capture_start, capture_end, probe_steps, total)
    needs_neutral = bool(set(active) & {'fixed_neutral', 'sparse_neutral', 'dense_neutral'})
    if needs_neutral and not record.neutral_enabled:
        raise ValueError('This sampling record has no neutral prediction. Enable record_neutral on the sampler and connect neutral_conditioning, then run once more; later parameter changes reuse it.')
    combine = 'min' if common_influence else 'mean'
    height = source.shape[-2] * 16
    width = source.shape[-1] * 16
    caps = {m: [] for m in active}
    at = {m: [] for m in active}
    masked_at = []
    for step in record.steps:
        idx = step['idx']
        within = capture_start <= idx <= capture_end
        if not within:
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
                masked_at.append(idx)
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
            'neutral_calls': sorted(record._neutral_by_index),
            'noise_mask_applied': record.noise_mask is not None,
            'reused_without_resampling': True, 'experimental': True}
    return maps, info


def _rms(a, b):
    return (a.float() - b.float()).square().mean(1).sqrt().detach().cpu()


def _apply_noise_mask(heat, noise_mask):
    import torch.nn.functional as F
    nm = noise_mask.detach().cpu().reshape(-1, 1, *noise_mask.shape[-2:]).float()
    nm = F.interpolate(nm, size=heat.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    if nm.shape[0] == 1:
        nm = nm.expand(heat.shape[0], -1, -1)
    if nm.shape[0] != heat.shape[0]:
        raise ValueError('noise_mask batch differs from source batch')
    return heat * nm.clamp(0, 1)
