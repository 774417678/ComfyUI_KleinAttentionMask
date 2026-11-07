"""Which native Flux2 Klein variants are supported, and how to recognise them.

Klein 4B and 9B share the Flux2 denoiser (axes_dim [32, 32, 32, 32], 128 latent
channels, the same ``ref_latents`` conditioning). They differ only in the text
encoder, hence in the width of the cross-attention context:

    Klein 9B   qwen3_8b   hidden 4096 x 3 tapped layers = 12288
    Klein 4B   qwen3_4b   hidden 2560 x 3 tapped layers =  7680

Everything else this plugin relies on -- ``process_latent_in``, ``ref_latents``
slicing of the CFG batch, the left-zero-padded 512-token context, the
``model_function_wrapper`` contract -- is identical, so the samplers, records,
influence maps and region nodes work on both without further changes.
"""
from .neutral_conditioning import CONTEXT_DIMS

FLUX2_MODULE = 'comfy.ldm.flux.model'
CONTEXT_DIM_NAMES = {12288: 'Klein 9B', 7680: 'Klein 4B'}
LATENT_CHANNELS = 128

__all__ = ['CONTEXT_DIMS', 'CONTEXT_DIM_NAMES', 'FLUX2_MODULE', 'flux2_variant', 'variant_name']


def variant_name(context_in_dim):
    return CONTEXT_DIM_NAMES.get(context_in_dim, 'unknown Flux2 (%s)' % (context_in_dim,))


def flux2_variant(diffusion_model):
    """Validate a native Flux2 Klein denoiser and return its context width.

    Fails closed on anything else: other architectures, third-party/KV engines
    and unknown text encoders have different conditioning contracts.
    """
    module = type(diffusion_model).__module__
    if module != FLUX2_MODULE:
        raise ValueError('Native ComfyUI Flux2 Klein required (4B or 9B); got %s.' % module)
    params = getattr(diffusion_model, 'params', None)
    if params is None:
        raise ValueError('Diffusion model exposes no params; native Flux2 Klein required.')
    dim = getattr(params, 'context_in_dim', None)
    if dim not in CONTEXT_DIMS:
        raise ValueError('Unsupported Flux2 Klein context width %s; expected %s (%s).'
                         % (dim, ' or '.join(str(d) for d in CONTEXT_DIMS),
                            ', '.join('%d = %s' % kv for kv in sorted(CONTEXT_DIM_NAMES.items()))))
    channels = getattr(params, 'out_channels', None)
    if channels is not None and channels != LATENT_CHANNELS:
        raise ValueError('Unsupported latent channel count %s; Klein uses %d.' % (channels, LATENT_CHANNELS))
    return dim
