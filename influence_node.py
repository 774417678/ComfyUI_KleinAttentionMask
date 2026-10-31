"""Influence maps computed from a sampling record (no diffusion here).

``mode``, the capture window and ``probe_steps`` live on the sampler, because
they decide what had to be recorded while the model was running. This node only
does CPU arithmetic on data that already exists, so changing its parameters
never re-runs sampling, and the sampler, VAE decode and composites stay cached.
"""
import json

import torch

from .internal_sampler import MODES
from .sampling_record import build_maps, SOURCE_REFERENCES


class KleinInfluenceMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {'record': ('KLEIN_SAMPLING_RECORD',)},
            'optional': {
                'common_influence': ('BOOLEAN', {
                    'default': False, 'label_on': '共同影响（逐点最小）', 'label_off': '平均',
                    'tooltip': 'On: keep only positions strong in EVERY captured call (elementwise min of per-call normalized maps). Off: average.'}),
                'source_reference': (SOURCE_REFERENCES, {
                    'default': 'auto',
                    'tooltip': 'Which latent counts as the untouched original. auto: the edit reference image when present, else the sampler source_latent input / latent_image.'}),
            },
        }

    RETURN_TYPES = ('MASK', 'STRING', 'MASK')
    RETURN_NAMES = ('influence_heatmap', 'diagnostics', 'comparison_heatmaps')
    FUNCTION = 'compute'
    CATEGORY = 'KleinMask/influence'

    def compute(self, record, common_influence=False, source_reference='auto'):
        if not hasattr(record, 'steps'):
            raise ValueError('Connect a sampling_record output from a Klein mask sampler.')
        if getattr(record, 'mode', None) not in MODES:
            raise ValueError('The sampling record has no valid mode; re-run the sampler once.')
        if record.output is None:
            raise ValueError('The sampling record has no sampled latent; re-run the sampler once.')
        maps, info = build_maps(record, common_influence, source_reference)
        active = info['comparison_order']
        selected = 'source_latent' if record.mode == 'compare_all' else record.mode
        info = dict(info)
        info['record'] = record.describe()
        text = json.dumps(info, ensure_ascii=False)
        print('[KleinInfluenceMap] ' + text)
        return (maps[selected], text, torch.cat([maps[m] for m in active]))


NODE_CLASS_MAPPINGS = {'KleinInfluenceMap': KleinInfluenceMap}
NODE_DISPLAY_NAME_MAPPINGS = {'KleinInfluenceMap': 'Klein 影响图（由采样记录计算）'}
