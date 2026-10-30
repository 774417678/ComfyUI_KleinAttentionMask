from .mask_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .internal_sampler import KleinInternalEditMaskSampler
NODE_CLASS_MAPPINGS['KleinInternalEditMaskSampler'] = KleinInternalEditMaskSampler
NODE_DISPLAY_NAME_MAPPINGS['KleinInternalEditMaskSampler'] = 'Klein 内部指令影响采样器（实验）'

from .sampler_variants import KleinMaskSamplerCustomAdvanced, KleinMaskSamplerCustom, KleinMaskKSampler, KleinMaskKSamplerAdvanced
from .neutral_conditioning import KleinNeutralConditioning
for _cls, _label in [
    (KleinNeutralConditioning, 'Klein 中性条件准备（缓存）'),
    (KleinMaskSamplerCustomAdvanced, 'Klein 内部掩码 · SamplerCustomAdvanced'),
    (KleinMaskSamplerCustom, 'Klein 内部掩码 · SamplerCustom'),
    (KleinMaskKSampler, 'Klein 内部掩码 · KSampler'),
    (KleinMaskKSamplerAdvanced, 'Klein 内部掩码 · KSamplerAdvanced'),
]:
    NODE_CLASS_MAPPINGS[_cls.__name__] = _cls
    NODE_DISPLAY_NAME_MAPPINGS[_cls.__name__] = _label
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
