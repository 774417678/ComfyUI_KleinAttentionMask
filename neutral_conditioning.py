"""Fail-closed neutral text preparation for native Flux2 Klein9B (ComfyUI 0.42.3).

Integration::

    # Prefer an upstream cached CONDITIONING; CLIP is only a legacy fallback.
    cond = supplied_neutral if supplied_neutral is not None else encode_neutral_conditioning(clip)
    neutral = neutral_embedding(cond)

The preparation node returns the *complete* CONDITIONING, never ``[0][0]``.
Only the tensor consumer extracts an embedding, after validating all metadata.
The node owns neither CLIP nor an embedding cache: use ComfyUI's node cache.

Core contract checked against comfy/sd.py, sd1_clip.py, text_encoders/flux.py,
text_encoders/llama.py and model_base.py:
* vanilla Qwen3 has pooled_output=None;
* its TE attention_mask is a binary [1,T] tensor, not a DiT/image mask;
* Flux2.extra_conds left-zero-pads cross attention to at least 512 tokens;
* Flux only forwards attention_mask to the DiT with attention_mask_img_shape.
  That image-mask field is deliberately forbidden here. A vanilla TE mask may
  therefore be retained in CONDITIONING without silently discarding DiT masks.

Tensor validation uses only PyTorch and never initializes GPU services. The
preparation node lazily calls core's targeted encoder unload after encoding;
it does not globally unload models or clear execution caches. CPU tests stub
that targeted operation and never load or execute a text encoder model.
"""

import torch
import torch.nn.functional as F

NEUTRAL_PROMPT = "Keep the image unchanged."
_CONTEXT_DIM = 12288
_MIN_TOKENS = 512
_ALLOWED_METADATA = frozenset(("pooled_output", "attention_mask"))

__all__ = [
    "NEUTRAL_PROMPT",
    "KleinNeutralConditioning",
    "encode_neutral_conditioning",
    "neutral_embedding",
    "prepare_neutral_conditioning",
]


def _error(message):
    return ValueError("Klein neutral conditioning: " + message)


def _regular_cpu_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise _error(name + " must be a torch.Tensor.")
    # Nested tensors must be rejected before consulting shape/ndim.
    if value.is_nested or value.layout != torch.strided or value.is_quantized:
        raise _error(name + " must be a regular dense, non-nested tensor.")
    if value.device.type == "meta":
        raise _error(name + " cannot be a meta tensor without data.")
    return value


def _cpu_copy(value):
    # A copy avoids retaining GPU storage, autograd graphs, or mutable upstream
    # cache storage. Never alter the caller's tensor/dict while normalizing.
    return value.detach().to(device="cpu", copy=True).contiguous()


def _normalized_conditioning(conditioning):
    """Validate the entire CONDITIONING and return an independent CPU copy."""
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) != 1:
        raise _error("expected exactly one unscheduled CONDITIONING entry; multiple conditions/schedules are unsupported.")
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise _error("the single entry must be [embedding, metadata_dict].")
    embedding, metadata = entry
    if not isinstance(metadata, dict):
        raise _error("conditioning metadata must be a dict.")
    unknown = [key for key in metadata if key not in _ALLOWED_METADATA]
    if unknown:
        fields = ", ".join(sorted(repr(key) for key in unknown))
        raise _error("unsupported metadata field(s): " + fields + ". Only vanilla pooled_output=None and a TE attention_mask are allowed; hooks, schedules, regions, masks, controls, references and guidance cannot be dropped.")
    if "pooled_output" in metadata and metadata["pooled_output"] is not None:
        raise _error("pooled_output must be None for vanilla Klein/Qwen3; a non-None pooled output cannot be silently discarded.")

    embedding = _regular_cpu_tensor(embedding, "embedding")
    if embedding.ndim != 3 or embedding.shape[0] != 1 or embedding.shape[2] != _CONTEXT_DIM or embedding.shape[1] < 1:
        raise _error("embedding must have shape [1,T,12288] with T >= 1 (native Klein9B, not Klein4B or a batched context).")
    if not embedding.is_floating_point():
        raise _error("embedding must have a floating-point dtype, not integer/bool/complex.")
    tokens = embedding.shape[1]
    cpu_embedding = _cpu_copy(embedding)
    if not torch.isfinite(cpu_embedding).all().item():
        raise _error("embedding must contain only finite values (no NaN/Inf).")

    cpu_metadata = {}
    for key, value in metadata.items():
        if key == "pooled_output":
            cpu_metadata[key] = None
            continue
        mask = _regular_cpu_tensor(value, "attention_mask")
        if mask.ndim != 2 or tuple(mask.shape) != (1, tokens):
            raise _error("attention_mask must be the TE binary mask with shape [1,T] matching the *input* embedding; DiT/image/additive masks are unsupported.")
        if mask.is_complex():
            raise _error("attention_mask must be a binary bool/integer/real tensor.")
        cpu_mask = _cpu_copy(mask)
        if not ((cpu_mask == 0) | (cpu_mask == 1)).all().item():
            raise _error("attention_mask must contain only binary 0/1 values (no additive masks or NaN/Inf).")
        if not (cpu_mask == 1).any().item():
            raise _error("attention_mask must include at least one real text token.")
        cpu_metadata[key] = F.pad(cpu_mask, (_MIN_TOKENS - tokens, 0)) if tokens < _MIN_TOKENS else cpu_mask

    if tokens < _MIN_TOKENS:
        cpu_embedding = F.pad(cpu_embedding, (0, 0, _MIN_TOKENS - tokens, 0))
    return [[cpu_embedding, cpu_metadata]]


def neutral_embedding(conditioning) -> torch.Tensor:
    """Return detached CPU floating [1,T,12288], left-zero-padded to T>=512.

    Accept exactly one complete vanilla CONDITIONING. Validate metadata even
    though this tensor-only consumer does not return it; never pick the first
    entry of a schedule or discard semantic modifiers. The source is unchanged.
    """
    return _normalized_conditioning(conditioning)[0][0]


def prepare_neutral_conditioning(conditioning) -> torch.Tensor:
    """Compatibility name for the tensor-only :func:`neutral_embedding` helper."""
    return neutral_embedding(conditioning)


def _check_plain_clip(clip):
    if clip is None:
        raise _error("connect neutral_conditioning or provide CLIP for the legacy neutral encoding fallback.")
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise _error("CLIP has no patcher; cannot verify that forced_hooks are absent.")
    if getattr(patcher, "forced_hooks", None) is not None:
        raise _error("CLIP.patcher.forced_hooks is unsupported; use an unhooked, unscheduled CLIP.")
    # Check presence, not truthiness: an empty/false-y HookGroup is still not a
    # vanilla CLIP, and must not get through a tensor-only replay path.
    if getattr(clip, "apply_hooks_to_conds", None) is not None:
        raise _error("CLIP.apply_hooks_to_conds is unsupported; conditioning hooks cannot be discarded.")
    if getattr(clip, "use_clip_schedule", False):
        raise _error("CLIP.use_clip_schedule is enabled; scheduled neutral encoding is unsupported.")
    if not callable(getattr(clip, "tokenize", None)) or not callable(getattr(clip, "encode_from_tokens_scheduled", None)):
        raise _error("CLIP must provide tokenize and encode_from_tokens_scheduled; no metadata-dropping encoder fallback is supported.")


def encode_neutral_conditioning(clip):
    """Encode the fixed prompt and return full, validated CPU CONDITIONING.

    Shared by the node and the legacy sampler CLIP fallback. Hook/schedule
    checks happen before tokenization/encoding, not after executing a model.
    Retains and CPU-copies vanilla metadata rather than slicing ``[0][0]``.
    Exceptions from the CLIP (including cancellation) propagate unchanged.
    """
    _check_plain_clip(clip)
    tokens = clip.tokenize(NEUTRAL_PROMPT)
    # A tokenizer wrapper must not enable hooks between the check and encode.
    _check_plain_clip(clip)
    with torch.no_grad():
        conditioning = clip.encode_from_tokens_scheduled(tokens)
    return _normalized_conditioning(conditioning)


class KleinNeutralConditioning:
    """Cacheable upstream neutral encoding, without strong CLIP ownership."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",)}}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("neutral_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "KleinMask/experimental"
    DESCRIPTION = "Prepare cached vanilla Klein9B neutral text before sampling: Keep the image unchanged."

    def encode(self, clip):
        # Release only this encoder's GPU residency after its output is copied
        # to CPU. Dynamic VRAM otherwise keeps TE pages alongside diffusion,
        # even across a node boundary. Keep execution caches and other models.
        _check_plain_clip(clip)
        try:
            return (encode_neutral_conditioning(clip),)
        finally:
            offload_text_encoder(clip)


def offload_text_encoder(clip):
    """Use core's targeted unload API, without moving shared weights by hand."""
    import comfy.model_management
    comfy.model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False)
