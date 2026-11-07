"""CPU-only tests for VL region planning and the region guard (no model loads)."""
import json
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

PLUGIN = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location('klein_' + name, PLUGIN / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RP = _load('region_planner')
RG = _load('region_guard')

CLAUSES = ['亚洲女孩', '将人物的上衣换成红色', '将裙子变成裤子', '给她戴一顶帽子']
REPLY = ('<think>looking</think>```json\n'
         '{"instructions":['
         '{"id":1,"kind":"style","target":"woman","present":true,"boxes":[[100,0,900,1000]]},'
         '{"id":2,"kind":"edit","target":"top","present":true,"boxes":[[200,400,800,700]]},'
         '{"id":3,"kind":"edit","target":"skirt","present":false,"boxes":[]},'
         '{"id":4,"kind":"add","target":"hat","present":true,"boxes":[[300,0,700,150]]}],'
         '"protected":[{"label":"face","boxes":[[350,150,650,400]]}]}\n```')


class Planning(unittest.TestCase):
    def test_split(self):
        self.assertEqual(RP.split_instructions('亚洲女孩，将人物的上衣换成红色，将裙子变成裤子'),
                         ['亚洲女孩', '将人物的上衣换成红色', '将裙子变成裤子'])
        self.assertEqual(RP.split_instructions('a\n\nb;c'), ['a', 'b', 'c'])

    def test_parse_and_decide_drops_absent_target(self):
        plan, protected = RP.parse_plan(REPLY, CLAUSES)
        RP.decide(plan, ['auto'] * 4)
        keep = {e['clause']: e['keep'] for e in plan}
        self.assertTrue(keep['亚洲女孩'])
        self.assertTrue(keep['将人物的上衣换成红色'])
        self.assertFalse(keep['将裙子变成裤子'])
        self.assertTrue(keep['给她戴一顶帽子'])          # additions are never auto-dropped
        self.assertEqual(RP.filtered_prompt(plan), '亚洲女孩，将人物的上衣换成红色，给她戴一顶帽子')
        self.assertEqual(protected[0]['label'], 'face')

    def test_overrides(self):
        plan, _ = RP.parse_plan(REPLY, CLAUSES)
        RP.decide(plan, RP.parse_overrides('auto\nauto\nkeep\ndrop', 4))
        self.assertTrue(plan[2]['keep']); self.assertFalse(plan[3]['keep'])
        with self.assertRaises(ValueError):
            RP.parse_overrides('maybe', 4)

    def test_missing_answer_is_kept(self):
        reply = '{"instructions":[{"id":1,"kind":"edit","present":false,"boxes":[]}]}'
        plan, _ = RP.parse_plan(reply, ['a', 'b'])
        RP.decide(plan, ['auto', 'auto'])
        self.assertFalse(plan[0]['keep']); self.assertTrue(plan[1]['keep'])
        self.assertIn('no answer', plan[1]['reason'])

    def test_boxes_normalised_and_unit_scale(self):
        self.assertEqual(RP._clean_box([0.1, 0.2, 0.5, 0.6]), [0.1, 0.2, 0.5, 0.6])
        self.assertEqual(RP._clean_box([800, 600, 200, 100]), [0.2, 0.1, 0.8, 0.6])
        self.assertIsNone(RP._clean_box([1, 2, 3]))
        self.assertIsNone(RP._clean_box([5, 5, 5, 5]))

    def test_invalid_reply(self):
        with self.assertRaises(ValueError):
            RP.parse_plan('no json here', ['a'])
        with self.assertRaises(ValueError):
            RP.parse_plan('{"other":1}', ['a'])

    def test_protect_clause_kept_and_protected(self):
        reply = ('{"instructions":[{"id":1,"kind":"edit","present":true,"boxes":[[0,0,500,1000]]},'
                 '{"id":2,"kind":"edit","present":true,"boxes":[[500,500,1000,1000]]}]}')
        plan, protected = RP.parse_plan(reply, ['将上衣换成红色', '不要修改人物的大腿'])
        RP.decide(plan, ['auto', 'auto'])
        self.assertEqual(plan[1]['kind'], 'protect'); self.assertTrue(plan[1]['keep'])
        ex, tg, pr, grounded = RP.build_regions(plan, protected, 100, 100, 0, 0, False)
        self.assertTrue(grounded)
        self.assertEqual(float(tg[75, 75]), 0.0)      # protect clause is not an edit target
        self.assertEqual(float(pr[75, 75]), 1.0)      # its box is protected even without protect_rest

    def test_protect_clause_dropped_when_target_absent(self):
        reply = ('{"instructions":[{"id":1,"kind":"edit","present":true,"boxes":[[0,0,500,500]]},'
                 '{"id":2,"kind":"protect","present":false,"boxes":[]},'
                 '{"id":3,"kind":"protect","present":true,"boxes":[[0,500,500,1000]]}]}')
        clauses = ['将上衣换成红色', '不要修改人物的大腿', '不要修改人物的脸']
        plan, protected = RP.parse_plan(reply, clauses)
        RP.decide(plan, ['auto'] * 3)
        self.assertFalse(plan[1]['keep'])                 # no thigh in the image -> dropped
        self.assertIn('not found', plan[1]['reason'])
        self.assertTrue(plan[2]['keep'])                  # face present -> kept and protected
        self.assertEqual(RP.filtered_prompt(plan), '将上衣换成红色，不要修改人物的脸')
        _, _, pr, _ = RP.build_regions(plan, protected, 100, 100, 0, 0, False)
        self.assertEqual(float(pr[75, 25]), 1.0)
        # an unanswered protect clause is kept, and keep/drop overrides still win
        plan2, _ = RP.parse_plan('{"instructions":[]}', clauses)
        RP.decide(plan2, ['auto', 'auto', 'drop'])
        self.assertTrue(plan2[1]['keep']); self.assertFalse(plan2[2]['keep'])

    USER_CLAUSES = ['人物胸前的衣服半敞', '不要修改人物的大腿', '不要删除人物的丝袜', '漏出胸部和粉色的乳头',
                    '人物的隐私部位都是白皙粉嫩的', '移除人物的bra', '移除人物的裙子、内裤']
    _BIG = '[0,410,1000,1000]'
    _CHEST = '[275,538,575,750]'

    def _user_reply(self, framing):
        items = [(1, 'edit', self._CHEST), (2, 'protect', self._BIG), (3, 'protect', self._BIG),
                 (4, 'edit', self._CHEST), (5, 'edit', self._BIG), (6, 'edit', self._CHEST), (7, 'edit', self._BIG)]
        body = ','.join('{"id":%d,"kind":"%s","present":true,"boxes":[%s]}' % i for i in items)
        head = '"framing":"%s",' % framing if framing else ''
        return '{%s"visible":["white top"],"instructions":[%s]}' % (head, body)

    def _run_user(self, framing_reply, framing_widget='auto'):
        node = RP.KleinVLRegionPlanner()
        image = torch.rand(1, 64, 40, 3)
        text, ex, tg, pr, report, raw = node.plan(None, image, '，'.join(self.USER_CLAUSES), 0, 0, True, 128,
                                                  '', vl_reply=self._user_reply(framing_reply),
                                                  refine='box', framing=framing_widget)
        return text, json.loads(report)

    def test_chest_framing_drops_lower_body_clauses(self):
        text, report = self._run_user('chest')
        self.assertEqual(report['framing'], 'chest')
        self.assertEqual(text, '人物胸前的衣服半敞，漏出胸部和粉色的乳头，移除人物的bra')
        dropped = [e['clause'] for e in report['instructions'] if not e['keep']]
        self.assertIn('人物的隐私部位都是白皙粉嫩的', dropped)
        self.assertIn('不要修改人物的大腿', dropped)
        self.assertIn('移除人物的裙子、内裤', dropped)

    def test_style_label_does_not_rescue_out_of_frame_part(self):
        reply = ('{"framing":"chest","instructions":[{"id":1,"kind":"edit","present":true,"boxes":[[0,400,900,1000]]},'
                 '{"id":2,"kind":"style","target":"skin tone","present":true,"boxes":[]}]}')
        plan, _ = RP.parse_plan(reply, ['人物胸前的衣服半敞', '人物的隐私部位都是白皙粉嫩的'])
        RP.apply_checks(plan, 'chest'); RP.decide(plan, ['auto', 'auto'])
        self.assertTrue(plan[0]['keep']); self.assertFalse(plan[1]['keep'])
        RP.decide(plan, ['auto', 'keep'])                 # manual keep still wins
        self.assertTrue(plan[1]['keep'])

    def test_placeholder_box_ignored_even_without_framing(self):
        text, report = self._run_user(None)
        self.assertIsNone(report['framing'])
        self.assertTrue(any('reused' in w for w in report['warnings']))
        for e in report['instructions']:
            for b in e['boxes']:
                self.assertNotEqual([round(v, 3) for v in b], [0.0, 0.41, 1.0, 1.0])

    def test_manual_framing_overrides_vl(self):
        text, report = self._run_user('full', framing_widget='chest')
        self.assertEqual(report['vl_framing'], 'full'); self.assertEqual(report['framing'], 'chest')
        self.assertNotIn('大腿', text)

    def test_framing_rules(self):
        self.assertEqual(RP.normalise_framing('Upper body'), 'chest')
        self.assertEqual(RP.normalise_framing('mid-thigh shot'), 'thighs')
        self.assertTrue(RP.framing_excludes('不要修改人物的大腿', 'chest'))
        self.assertFalse(RP.framing_excludes('不要修改人物的大腿', 'thighs'))
        self.assertFalse(RP.framing_excludes('将上衣换成红色', 'head'))   # unknown part: not judged
        self.assertTrue(RP.framing_excludes('移除人物的裙子', 'waist'))

    def test_no_grounding_does_not_lock_whole_image(self):
        reply = '{"instructions":[],"protected":[{"label":"face","boxes":[[0,0,200,200]]}]}'
        plan, protected = RP.parse_plan(reply, ['a', 'b'])
        RP.decide(plan, ['auto', 'auto'])
        ex, tg, pr, grounded = RP.build_regions(plan, protected, 100, 100, 0, 0, True)
        self.assertFalse(grounded)
        self.assertEqual(float(tg.amax()), 0.0)
        self.assertEqual(float(pr[90, 90]), 0.0)      # rest is NOT protected
        self.assertEqual(float(pr[5, 5]), 1.0)        # named face still protected
        warnings = RP.plan_warnings(plan, grounded, True)
        self.assertTrue(any('none of the instructions' in w for w in warnings))

    def test_guided_refine_follows_image_edge_within_radius(self):
        img = torch.zeros(1, 64, 64, 3); img[:, :, 40:] = 1.0           # bright object on the right
        box = torch.zeros(64, 64); box[:, 37:] = 1.0                     # box edge 3 px off the object
        q = RP.guided_refine(box, img, radius=4, eps=1e-4, work_edge=64)
        self.assertLess(float(q[32, 38]), 0.7)                           # leak past the edge reduced (was 1.0)
        self.assertGreater(float(q[32, 50]), 0.9)
        self.assertTrue(torch.equal(RP.guided_refine(torch.zeros(8, 8), img), torch.zeros(8, 8)))

    def test_vl_image_downscaled(self):
        img = torch.rand(1, 4608, 2592, 3)
        small = RP.vl_image(img)
        self.assertLessEqual(max(small.shape[1:3]), RP.VL_MAX_EDGE)
        self.assertEqual(small.shape[1] % 32, 0); self.assertEqual(small.shape[2] % 32, 0)
        tiny = torch.rand(1, 64, 48, 3)
        self.assertIs(RP.vl_image(tiny), tiny)

    def test_regions(self):
        plan, protected = RP.parse_plan(REPLY, CLAUSES)
        RP.decide(plan, ['auto'] * 4)
        ex, tg, pr, grounded = RP.build_regions(plan, protected, 100, 100, 0, 0, True)
        self.assertTrue(grounded)
        # existing: only the top (style + addition excluded)
        self.assertEqual(float(ex[50, 50]), 1.0); self.assertEqual(float(ex[5, 50]), 0.0)
        # target: top + hat + style box (covers x 10..90)
        self.assertEqual(float(tg[5, 50]), 1.0); self.assertEqual(float(tg[50, 50]), 1.0)
        # protected: outside target only; face overlaps the style box -> stays editable
        self.assertEqual(float(pr[50, 2]), 1.0)
        self.assertEqual(float(pr[50, 50]), 0.0)
        self.assertTrue(((tg * pr) == 0).all())

    def test_node_with_supplied_reply(self):
        node = RP.KleinVLRegionPlanner()
        image = torch.rand(1, 64, 80, 3)
        out = node.plan(None, image, '，'.join(CLAUSES), 0, 0, False, 128, '', vl_reply=REPLY)
        text, ex, tg, pr, report, raw = out
        self.assertNotIn('裙子', text)
        self.assertEqual(tuple(ex.shape), (1, 64, 80))
        self.assertEqual(json.loads(report)['filtered_prompt'], text)

    def test_ask_vl_uses_greedy_generate_and_offloads(self):
        calls = {}

        class FakeClip:
            patcher = object()
            def tokenize(self, text, image=None, min_length=None):
                calls['image'] = image.shape; calls['text'] = text; return {'k': [[1]]}
            def generate(self, tokens, **kw):
                calls['kw'] = kw; return [1, 2]
            def decode(self, ids):
                return REPLY
        saved = {k: sys.modules.get(k) for k in ('comfy', 'comfy.model_management')}
        pkg = types.ModuleType('comfy')
        mm = types.ModuleType('comfy.model_management')
        mm.unload_model_and_clones = lambda p, unload_additional_models=False: calls.setdefault('unloaded', True)
        pkg.model_management = mm
        sys.modules['comfy'] = pkg; sys.modules['comfy.model_management'] = mm
        try:
            reply = RP.ask_vl(FakeClip(), torch.rand(1, 8, 8, 3), 'q', 64)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertEqual(reply, REPLY)
        self.assertFalse(calls['kw']['do_sample'])
        self.assertTrue(calls.get('unloaded'))


class Guard(unittest.TestCase):
    def setUp(self):
        self.src = torch.zeros(1, 4, 8, 8)
        self.target = torch.zeros(1, 128, 128); self.target[:, :, :64] = 1   # left half
        self.protect = 1 - self.target

    def _guard(self, mode, release=0.6, max_release=0.5, start=0.0, end=1.0, sigmas=None):
        return RG.RegionGuard(self.src, self.target, self.protect, mode, release, max_release,
                              start, end, sigmas, 0)

    def _args(self, denoised, sigma=0.5):
        return {'denoised': denoised, 'cond_denoised': denoised, 'sigma': torch.tensor([sigma])}

    def test_hard_keeps_protected_exactly(self):
        pred = torch.ones(1, 4, 8, 8)
        out = self._guard('hard')(self._args(pred))
        self.assertTrue(torch.allclose(out[..., :4], torch.ones(1, 4, 8, 4)))
        self.assertTrue(torch.allclose(out[..., 4:], torch.zeros(1, 4, 8, 4)))

    def test_soft_release_follows_intrusion_and_cap(self):
        pred = torch.zeros(1, 4, 8, 8); pred[..., 6:] = 1.0     # strong change inside protected half
        out = self._guard('soft', release=1.0, max_release=0.5)(self._args(pred))
        self.assertAlmostEqual(float(out[0, 0, 0, 7]), 0.5, places=5)   # capped release
        self.assertAlmostEqual(float(out[0, 0, 0, 5]), 0.0, places=5)   # no intrusion, no release

    def test_off_and_window_by_step_index_with_fade(self):
        pred = torch.ones(1, 4, 8, 8)
        self.assertTrue(torch.equal(self._guard('off')(self._args(pred)), pred))
        # Klein 4-step schedule: sigma spacing is uneven, the window must use step index.
        sig = torch.tensor([1.0, 0.9045, 0.7595, 0.5128, 0.0])
        g = self._guard('hard', start=0.0, end=0.5, sigmas=sig)
        s0 = g(self._args(pred, sigma=1.0))          # step 0: full strength
        s1 = g(self._args(pred, sigma=0.9045))       # step 1: half strength
        s2 = g(self._args(pred, sigma=0.7595))       # step 2: free
        s3 = g(self._args(pred, sigma=0.5128))       # step 3: free
        self.assertAlmostEqual(float(s0[0, 0, 0, 7]), 0.0, places=5)
        self.assertAlmostEqual(float(s1[0, 0, 0, 7]), 0.5, places=5)
        self.assertTrue(torch.equal(s2, pred)); self.assertTrue(torch.equal(s3, pred))
        self.assertEqual(g.strengths, [1.0, 0.5])

    def test_influence_shapes_box_and_keeps_floor(self):
        g = RG.RegionGuard(self.src, self.target, self.protect, 'hard', 0.6, 0.5, 0.0, 1.0, None, 0,
                           'influence', 0.35)
        pred = torch.zeros(1, 4, 8, 8); pred[..., 2] = 1.0       # Klein only changes column 2 inside the box
        out = g(self._args(pred))
        self.assertAlmostEqual(float(out[0, 0, 0, 2]), 1.0, places=4)    # influenced column fully editable
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 0.0, places=4)    # pred there is 0 anyway
        pred2 = torch.ones(1, 4, 8, 8)
        out2 = g(self._args(pred2))                                      # shape fixed from step 1
        self.assertAlmostEqual(float(out2[0, 0, 0, 0]), 0.35, places=3)  # floor inside the box
        self.assertAlmostEqual(float(out2[0, 0, 0, 2]), 1.0, places=3)
        self.assertAlmostEqual(float(out2[0, 0, 0, 7]), 0.0, places=3)   # outside box stays protected

    def test_intrusion_scale_fixed_at_first_step(self):
        g = self._guard('soft', release=1.0, max_release=1.0)
        strong = torch.zeros(1, 4, 8, 8); strong[..., 6:] = 1.0
        g(self._args(strong))
        weak = torch.zeros(1, 4, 8, 8); weak[..., 6:] = 0.1     # later, smaller change
        out = g(self._args(weak))
        # with a fixed scale the weak change grants little release; per-step
        # re-normalisation would have inflated it to the maximum
        self.assertLess(float(out[0, 0, 0, 7]), 0.05)

    def test_shape_mismatch_fails_closed(self):
        g = self._guard('hard')
        with self.assertRaises(ValueError):
            g(self._args(torch.ones(1, 3, 8, 8)))

    def test_node_attaches_post_cfg(self):
        class Raw:
            def process_latent_in(self, x): return x
        class Patcher:
            def __init__(self): self.model = Raw(); self.fns = []
            def clone(self):
                p = Patcher(); p.fns = list(self.fns); return p
            def set_model_sampler_post_cfg_function(self, fn): self.fns.append(fn)
        base = Patcher()
        m, info = RG.KleinRegionGuard().patch(base, {'samples': self.src}, self.target, self.protect,
                                              'soft', 0.6, 0.5, 0.0, 0.75, 1)
        self.assertEqual(len(m.fns), 1); self.assertEqual(base.fns, [])
        m2, _ = RG.KleinRegionGuard().patch(base, {'samples': self.src}, self.target, self.protect,
                                            'off', 0.6, 0.5, 0.0, 0.75, 1)
        self.assertEqual(m2.fns, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
