"""Constrain Klein sampling with planned regions (experimental).

Every denoising step the guided prediction ``denoised`` (x0) is blended toward
the clean source latent outside the editable area:

    x0' = w * x0 + (1 - w) * source

``w`` is the edit weight, 1 inside the target region and 0 in the protected
region. Between them the *release* rule decides how much a protected area
yields when the edit really needs to enter it:

* intrusion = the model's own measured change at that step (normalised RMS of
  the positive prediction against the source, the same quantity as
  ``source_latent`` influence), so it costs no extra forward.
* ``release`` scales how strongly intrusion overrides protection; ``hard``
  keeps protected areas fixed regardless, ``soft`` grants
  w = max(target, release * intrusion) clamped by ``max_release``.
* ``guard_start`` / ``guard_end`` bound the constraint to a fraction of the
  sampling schedule, so late steps can unify lighting and detail.

The hook is ``sampler_post_cfg_function``: it sees the guided x0 of every
native sampler without extra model calls and composes with the recording
wrapper used by the KleinMask samplers.
"""
import json

import torch
import torch.nn.functional as F

RELEASE_MODES = ('soft', 'hard', 'off')
SHAPE_MODES = ('influence', 'box')


def _to_latent_map(mask, shape, device, dtype):
    """[B?,H,W] pixel mask -> [B,1,h,w] latent map."""
    m = mask.float()
    if m.ndim == 2:
        m = m[None]
    if m.ndim == 4:
        m = m[:, 0]
    m = F.interpolate(m[:, None], size=shape[-2:], mode='area')
    if m.shape[0] == 1 and shape[0] > 1:
        m = m.expand(shape[0], -1, -1, -1)
    if m.shape[0] != shape[0]:
        raise ValueError('Region mask batch %d differs from latent batch %d.' % (m.shape[0], shape[0]))
    return m.to(device=device, dtype=dtype).clamp(0, 1)


def normalised_change(pred, source):
    """Per-pixel channel RMS change, normalised median->0, q99.5->1 (as influence maps)."""
    change = (pred.float() - source.float()).square().mean(1, keepdim=True).sqrt()
    flat = change.flatten(1)
    lo = flat.median(1).values.view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 0.995, dim=1).view(-1, 1, 1, 1)
    return ((change - lo) / (hi - lo).clamp_min(1e-7)).clamp(0, 1)


def edit_weight(target, protect, intrusion, release_mode, release, max_release):
    """Combine soft region maps and measured intrusion into the per-pixel edit weight.

        w = t + (1 - t) * ((1 - p) + p * granted)

    target t is fully editable; outside it, unprotected area is free and the
    protected share only gets the granted release. Exact for soft maps.
    """
    if release_mode == 'off':
        return torch.ones_like(target)
    if release_mode == 'hard':
        granted = torch.zeros_like(target)
    else:
        granted = (release * intrusion).clamp(0, max_release)
    return (target + (1.0 - target) * ((1.0 - protect) + protect * granted)).clamp(0, 1)


class RegionGuard:
    """Post-CFG blend toward the source latent, honouring planned regions."""

    def __init__(self, source, target, protect, release_mode, release, max_release,
                 start, end, sigmas, smooth_px, shape_mode='box', shape_floor=0.35):
        if release_mode not in RELEASE_MODES:
            raise ValueError('release_mode must be one of %s' % (RELEASE_MODES,))
        if shape_mode not in SHAPE_MODES:
            raise ValueError('shape_mode must be one of %s' % (SHAPE_MODES,))
        self.shape_mode = shape_mode
        self.shape_floor = float(shape_floor)
        self.shape = None
        self.source = source
        self.target_px = target
        self.protect_px = protect
        self.release_mode = release_mode
        self.release = float(release)
        self.max_release = float(max_release)
        self.start = float(start); self.end = float(end)
        self.sigma_hi = None if sigmas is None else float(sigmas[0])
        self.sigma_lo = None if sigmas is None else float(sigmas[-1])
        self.smooth = int(smooth_px)
        self.steps = None if sigmas is None else max(1, len(sigmas) - 1)
        self.sigma_list = None if sigmas is None else [float(s) for s in sigmas]
        self.calls = 0
        self.applied = 0
        self.scale = None          # intrusion scale fixed at the first guarded step
        self.mean_release = []
        self.strengths = []

    def _progress(self, sigma):
        """Fraction of the schedule by *step index* (sigma spacing is very uneven at 4 steps)."""
        if self.sigma_list is None or self.steps is None:
            return None
        idx = min(range(len(self.sigma_list) - 1), key=lambda i: abs(self.sigma_list[i] - sigma))
        return idx / self.steps

    def _strength(self, progress):
        """Guard strength: full at the start, fading linearly to zero at guard_end."""
        if progress is None:
            return 1.0
        span = max(self.end - self.start, 1e-6)
        return max(0.0, min(1.0, 1.0 - (progress - self.start) / span))

    def __call__(self, args):
        denoised = args['denoised']
        self.calls += 1
        sigma = float(args['sigma'].flatten()[0])
        progress = self._progress(sigma)
        if progress is not None and not (self.start <= progress < self.end):
            return denoised
        src = self.source.to(device=denoised.device, dtype=denoised.dtype)
        if src.shape[-2:] != denoised.shape[-2:] or src.shape[1] != denoised.shape[1]:
            raise ValueError('Region guard source latent %s does not match prediction %s.'
                             % (tuple(src.shape), tuple(denoised.shape)))
        if src.shape[0] != denoised.shape[0]:
            src = src[:1].expand(denoised.shape[0], -1, -1, -1)
        target = _to_latent_map(self.target_px, denoised.shape, denoised.device, denoised.dtype)
        protect = _to_latent_map(self.protect_px, denoised.shape, denoised.device, denoised.dtype)
        cond = args.get('cond_denoised')
        probe = cond if torch.is_tensor(cond) and cond.shape == denoised.shape else denoised
        change = (probe.float() - src.float()).square().mean(1, keepdim=True).sqrt()
        if self.scale is None:
            # Fix the normalisation at the first step: later steps change less, and
            # re-normalising each step would inflate noise into fake intrusion.
            flat = change.flatten(1)
            self.scale = (flat.median(1).values.view(-1, 1, 1, 1),
                          torch.quantile(flat, 0.995, dim=1).view(-1, 1, 1, 1))
        lo, hi = (t.to(change.device) for t in self.scale)
        intrusion = ((change - lo) / (hi - lo).clamp_min(1e-7)).clamp(0, 1).to(denoised.dtype)
        if self.shape_mode == 'influence':
            if self.shape is None:
                # The VL box says *where*; Klein's own first-step influence says *what shape*.
                # Inside the box keep at least shape_floor so the edit is never fully blocked.
                self.shape = intrusion.detach()
            shaped = self.shape_floor + (1.0 - self.shape_floor) * self.shape
            boxed = target
            target = boxed * shaped
            # The rest of the box is held like a protected area (still subject to release).
            protect = torch.maximum(protect, boxed)
        w = edit_weight(target, protect, intrusion, self.release_mode, self.release, self.max_release)
        if self.smooth > 0:
            k = 2 * self.smooth + 1
            w = F.avg_pool2d(w, k, 1, self.smooth, count_include_pad=False)
        strength = self._strength(progress)
        w = 1.0 - strength * (1.0 - w)       # fade the constraint out toward guard_end
        self.applied += 1
        self.strengths.append(round(strength, 3))
        held = (protect > 0.5).float()
        if held.sum() > 0:
            self.mean_release.append(round(float((w * held).sum() / held.sum()), 4))
        print('[KleinRegionGuard] ' + json.dumps(self.report(), ensure_ascii=False))
        return w * denoised + (1.0 - w) * src

    def report(self):
        return {'guard_calls': self.calls, 'guard_applied': self.applied,
                'release_mode': self.release_mode, 'release': self.release, 'max_release': self.max_release,
                'window': [self.start, self.end], 'strength_per_step': self.strengths,
                'mean_edit_weight_in_protected_per_step': self.mean_release}


class KleinRegionGuard:
    """Attach region-constrained sampling to a MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'source_latent': ('LATENT', {'tooltip': 'Clean VAE-encoded source image (same latent as ReferenceLatent).'}),
                'target_region': ('MASK',),
                'protected_region': ('MASK',),
                'release_mode': (list(RELEASE_MODES), {'default': 'soft', 'tooltip': 'soft: protected areas yield where the edit pushes into them; hard: never yield; off: disable the guard.'}),
                'release': ('FLOAT', {'default': 0.6, 'min': 0.0, 'max': 4.0, 'step': 0.05, 'tooltip': 'How strongly measured intrusion overrides protection.'}),
                'max_release': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'Upper bound of the edit weight granted inside protected areas.'}),
                'guard_start': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'Schedule fraction where the guard starts (0 = first step).'}),
                'guard_end': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'Fraction of steps (by index) where the guard has faded to zero. With 4 steps, 0.5 guards steps 0-1 (full, then half) and leaves steps 2-3 free to harmonise.'}),
                'smooth_px': ('INT', {'default': 1, 'min': 0, 'max': 16, 'tooltip': 'Box blur radius of the edit weight in latent pixels (1 latent px = 16 image px).'}),
            },
            'optional': {
                'sigmas': ('SIGMAS', {'tooltip': 'Schedule used by the sampler; required for guard_start/guard_end. Without it the guard runs on every step.'}),
                'shape_mode': (list(SHAPE_MODES), {'default': 'influence', 'tooltip': 'influence: inside the VL box, Klein\'s own first-step influence shapes the editable area (object outline instead of a rectangle). box: use the planned mask as is.'}),
                'shape_floor': ('FLOAT', {'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'Minimum edit weight kept everywhere inside the box, so weak first-step influence cannot block the edit.'}),
            },
        }

    RETURN_TYPES = ('MODEL', 'STRING')
    RETURN_NAMES = ('model', 'guard_info')
    FUNCTION = 'patch'
    CATEGORY = 'KleinMask/experimental'
    DESCRIPTION = ('Blend each guided prediction toward the source outside the target region. Protected areas '
                   'release editing only where the edit actually intrudes, bounded by max_release.')

    def patch(self, model, source_latent, target_region, protected_region, release_mode, release,
              max_release, guard_start, guard_end, smooth_px, sigmas=None, shape_mode='influence',
              shape_floor=0.35):
        if guard_end < guard_start:
            raise ValueError('guard_end < guard_start')
        samples = source_latent['samples']
        if samples.is_nested or samples.ndim != 4:
            raise ValueError('source_latent must be a regular image latent [B,C,H,W].')
        source = model.model.process_latent_in(samples.detach().cpu().clone()).float()
        guard = RegionGuard(source, target_region.detach().cpu().float(), protected_region.detach().cpu().float(),
                            release_mode, release, max_release, guard_start, guard_end,
                            None if sigmas is None else sigmas.detach().cpu(), smooth_px,
                            shape_mode, shape_floor)
        m = model.clone()
        if release_mode != 'off':
            m.set_model_sampler_post_cfg_function(guard)
        info = json.dumps({'release_mode': release_mode, 'release': release, 'max_release': max_release,
                           'window': [guard_start, guard_end], 'uses_schedule': sigmas is not None,
                           'note': 'per-step statistics are printed after sampling'}, ensure_ascii=False)
        return (m, info)
