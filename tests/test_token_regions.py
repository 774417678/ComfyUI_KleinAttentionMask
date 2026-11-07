"""CPU tests for token-region attention bias (no model)."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F

PLUGIN = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('klein_token_regions', PLUGIN / 'token_regions.py')
TR = importlib.util.module_from_spec(spec); spec.loader.exec_module(TR)


def sdpa(q, k, v, heads, mask=None, **kw):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class Bias(unittest.TestCase):
    def setUp(self):
        self.layout = {'txt': 6, 'target': 4, 'ref': 4, 'h': 2, 'w': 2}
        self.protect = torch.zeros(1, 32, 32); self.protect[:, :, 16:] = 1.0   # right column protected

    def test_suppresses_only_edit_text_for_protected_tokens(self):
        b = TR.build_bias(self.layout, self.protect, (2, 5), 1.0, 0.0, 1, 'cpu', torch.float32)[0, 0]
        # target tokens are row-major: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1); right column = 1,3
        for tok, protected in ((0, False), (1, True), (2, False), (3, True)):
            row = b[6 + tok]
            self.assertEqual(bool((row[2:5] < -1).all()), protected)
            self.assertTrue((row[0:2] == 0).all() and (row[5:6] == 0).all())   # template untouched
            self.assertTrue((row[6:] == 0).all())                               # image keys untouched
        self.assertTrue((b[2:5, 7] < -1).all() and (b[2:5, 6] == 0).all())     # text -> protected keys

    def test_self_bias_on_matching_reference(self):
        b = TR.build_bias(self.layout, self.protect, (2, 5), 0.0, 2.0, 1, 'cpu', torch.float32)[0, 0]
        self.assertEqual(float(b[6 + 1, 10 + 1]), 2.0)
        self.assertEqual(float(b[6 + 0, 10 + 0]), 0.0)

    def test_grid_mismatch(self):
        with self.assertRaises(ValueError):
            TR.build_bias({'txt': 2, 'target': 5, 'ref': 0, 'h': 2, 'w': 2}, self.protect, (0, 1), 1, 0, 1,
                          'cpu', torch.float32)

    def test_attention_effect_and_fallback(self):
        state = TR.TokenRegionBias(self.protect, 1.0, 0.0, 3, 1, 0)
        ids = torch.zeros(1, 8, 4)
        ids[0, :4, 1] = torch.tensor([0., 0., 1., 1.]); ids[0, :4, 2] = torch.tensor([0., 1., 0., 1.])
        ids[0, 4:, 0] = 10; ids[0, 4:, 1:3] = ids[0, :4, 1:3]
        state.post_input({'img_ids': ids, 'txt': torch.zeros(1, 6, 8),
                          'transformer_options': {'reference_image_num_tokens': [4]}})
        self.assertEqual(state.layout, {'txt': 6, 'target': 4, 'ref': 4, 'h': 2, 'w': 2})
        self.assertEqual(state.edit_span(6), (2, 5))
        q = torch.randn(1, 2, 14, 8); k = torch.randn(1, 2, 14, 8); v = torch.zeros(1, 2, 14, 8)
        v[:, :, 2:5] = 1.0                                       # edit text carries a signal
        mod = types.ModuleType('comfy.ldm.modules.attention'); mod.attention_pytorch = sdpa
        saved = {n: sys.modules.get(n) for n in ('comfy', 'comfy.ldm', 'comfy.ldm.modules', 'comfy.ldm.modules.attention')}
        for n in ('comfy', 'comfy.ldm', 'comfy.ldm.modules'):
            sys.modules[n] = types.ModuleType(n)
        sys.modules['comfy.ldm.modules.attention'] = mod
        try:
            out = state.attention(sdpa, q, k, v, 2, transformer_options={'block_type': 'double', 'block_index': 0})
            short = state.attention(sdpa, q[:, :, :10], k[:, :, :10], v[:, :, :10], 2, transformer_options={})
        finally:
            for n, mval in saved.items():
                if mval is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = mval
        self.assertLess(float(out[0, :, 6 + 1].abs().max()), 1e-3)       # protected token got no edit-text signal
        self.assertGreater(float(out[0, :, 6 + 0].abs().max()), 1e-2)    # free token did
        self.assertEqual(state.biased, 1); self.assertEqual(state.skipped, 1)
        self.assertEqual(tuple(short.shape), (1, 2, 10, 8))

    def test_template_split(self):
        class Clip:
            def tokenize(self, text):
                core = [5, 6, 7][:len(text)] if text else []
                return {'q': [[(PAD,), (1,), (2,)] + [(t,) for t in core] + [(8,), (9,)]]}
        PAD = TR.PAD_ID
        self.assertEqual(TR.template_split(Clip(), 'abc'), (3, 2))

    def test_node_installs_hooks_and_rejects_double(self):
        class Patcher:
            def __init__(self): self.model_options = {'transformer_options': {}}; self.post = []
            def clone(self):
                p = Patcher(); p.model_options = {'transformer_options': dict(self.model_options['transformer_options'])}
                return p
            def set_model_post_input_patch(self, fn): self.post.append(fn)

        class Clip:
            def tokenize(self, text):
                return {'q': [[(1,)] + [(5,)] * len(text) + [(9,)]]}
        node = TR.KleinTokenRegionAttention()
        m, info = node.patch(Patcher(), Clip(), 'abcd', self.protect, 1.0, 0.0, 0)
        self.assertEqual(len(m.post), 1)
        self.assertIsNotNone(m.model_options['transformer_options']['optimized_attention_override'])
        with self.assertRaises(ValueError):
            node.patch(m, Clip(), 'abcd', self.protect, 1.0, 0.0, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
