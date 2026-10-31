"""Influence maps computed from a sampling record (no diffusion here).

Every parameter on this node feeds only CPU arithmetic over an existing
``KLEIN_SAMPLING_RECORD``. Changing mode, capture window, probe_steps, common
influence or which latent counts as "original" therefore re-runs this node alone;
the sampler, VAE decode and any composite stay cached.
"""
import json

from .internal_sampler import MODES
from .sampling_record import build_maps, SOURCE_REFERENCES


class KleinInfluenceMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {'record': ('KLEIN_SAMPLING_RECORD',)},
            'optional': {
                'mode': (MODES, {'default': 'source_latent'}),
                'capture_start': ('INT', {'default': 0, 'min': 0, 'max': 10000,
                                         'tooltip': 'First recorded model call to compare (0 = first).'}),
                'capture_end': ('INT', {'default': -1, 'min': -1, 'max': 10000,
                                        'tooltip': '-1 means the last recorded call.'}),
                'probe_steps': ('INT', {'default': 2, 'min': 1, 'max': 100,
                                        'tooltip': 'Only sparse_neutral: how many captures inside the window get a neutral pair.'}),
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

    def compute(self, record, mode='source_latent', capture_start=0, capture_end=-1,
                probe_steps=2, common_influence=False, source_reference='auto'):
        if not hasattr(record, 'steps'):
            raise ValueError('Connect a sampling_record output from a Klein mask sampler.')
        total = max(record.total_calls, len(record.steps))
        end = total - 1 if capture_end < 0 else capture_end
        maps, info = build_maps(record, mode, capture_start, end, probe_steps,
                                common_influence, source_reference)
        active = info['comparison_order']
        selected = 'source_latent' if mode == 'compare_all' else mode
        info = dict(info)
        info['record'] = record.describe()
        import torch
        text = json.dumps(info, ensure_ascii=False)
        print('[KleinInfluenceMap] ' + text)
        return (maps[selected], text, torch.cat([maps[m] for m in active]))


NODE_CLASS_MAPPINGS = {'KleinInfluenceMap': KleinInfluenceMap}
NODE_DISPLAY_NAME_MAPPINGS = {'KleinInfluenceMap': 'Klein 影响图（由采样记录计算）'}
