"""Real-torch CPU tests; no ComfyUI import, model files, GPU, server or API.

Run: python -B test_neutral_conditioning.py
Optional: set CORE to the ComfyUI root to exercise extracted *real* core methods
against tiny local stubs as well. Only source text/AST is read from that root.
"""

import ast
import contextlib
import gc
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
import warnings
import weakref
from unittest.mock import patch

sys.dont_write_bytecode = True
import torch

# Do not import the plugin's __init__ (which would import sampler/GPU modules).
_spec = importlib.util.spec_from_file_location("_klein_neutral_cpu_test", Path(__file__).resolve().parent.parent / "neutral_conditioning.py")
neutral = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(neutral)


def embedding(tokens=3, dtype=torch.float32):
    return torch.arange(tokens, dtype=dtype, device="cpu").reshape(1, tokens, 1).expand(1, tokens, 12288).clone()


def conditioning(tokens=3, metadata=None):
    if metadata is None:
        metadata = {"pooled_output": None, "attention_mask": torch.ones((1, tokens), dtype=torch.long, device="cpu")}
    return [[embedding(tokens), metadata]]


class FakeClip:
    """Only a CLIP interface stub: all embeddings/masks are real CPU tensors."""

    def __init__(self, result=None):
        self.patcher = types.SimpleNamespace(forced_hooks=None)
        self.apply_hooks_to_conds = None
        self.use_clip_schedule = False
        self.result = conditioning() if result is None else result
        self.calls = []

    def tokenize(self, text):
        self.calls.append(("tokenize", text))
        return {"qwen3_8b": "stub tokens"}

    def encode_from_tokens_scheduled(self, tokens):
        self.calls.append(("encode", tokens))
        return self.result


class CpuOnlyTestCase(unittest.TestCase):
    def setUp(self):
        self.cuda_before = torch.cuda.is_initialized()
        # Fail the test if any implementation path tries to initialize CUDA.
        self.cuda_guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU test attempted CUDA initialization"))
        self.cuda_guard.start()
        self.offload_guard = patch.object(neutral, 'offload_text_encoder')
        self.offload_mock = self.offload_guard.start()

    def tearDown(self):
        self.offload_guard.stop()
        self.cuda_guard.stop()
        self.assertEqual(torch.cuda.is_initialized(), self.cuda_before)


class NeutralEmbeddingTests(CpuOnlyTestCase):
    def test_short_context_left_zero_pad_and_no_mutation(self):
        source = conditioning(3)
        original = source[0][0].clone()
        result = neutral.neutral_embedding(source)
        self.assertEqual(tuple(result.shape), (1, 512, 12288))
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, source[0][0].dtype)
        self.assertFalse(result.is_nested)
        self.assertTrue(result.is_contiguous())
        self.assertTrue(torch.equal(result[:, 509:], original))
        self.assertEqual(torch.count_nonzero(result[:, :509]).item(), 0)
        self.assertTrue(torch.equal(source[0][0], original))
        self.assertEqual(tuple(source[0][1]["attention_mask"].shape), (1, 3))
        self.assertIsNone(source[0][1]["pooled_output"])

    def test_512_and_long_context_not_truncated_or_reordered(self):
        for tokens in (512, 513):
            with self.subTest(tokens=tokens):
                source = conditioning(tokens)
                result = neutral.neutral_embedding(source)
                self.assertTrue(torch.equal(result, source[0][0]))
                self.assertNotEqual(result.data_ptr(), source[0][0].data_ptr())

    def test_dtypes_and_noncontiguous_tensor(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                source = embedding(2, dtype).transpose(1, 2).contiguous().transpose(1, 2)
                result = neutral.neutral_embedding([[source, {}]])
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(result.is_contiguous())
                self.assertTrue(torch.equal(result[:, -2:], source))

    def test_detaches_autograd_and_does_not_share_storage(self):
        source = embedding(512).requires_grad_(True)
        result = neutral.neutral_embedding([[source, {}]])
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)
        result[0, 0, 0] = 111
        self.assertEqual(source[0, 0, 0].item(), 0)

    def test_alias_helper(self):
        source = conditioning()
        self.assertTrue(torch.equal(neutral.prepare_neutral_conditioning(source), neutral.neutral_embedding(source)))

    def test_tuple_conditioning_and_absent_optional_metadata(self):
        result = neutral.neutral_embedding(((embedding(), {}),))
        self.assertEqual(tuple(result.shape), (1, 512, 12288))

    def test_malformed_or_multiple_conditioning_rejected(self):
        valid = conditioning()[0]
        cases = (None, {}, [], [valid, valid], [embedding()], [[embedding()]], [[embedding(), {}, None]], [[embedding(), None]])
        for source in cases:
            with self.subTest(source_type=type(source).__name__):
                with self.assertRaisesRegex(ValueError, "Klein neutral conditioning"):
                    neutral.neutral_embedding(source)

    def test_shape_mismatch_rejected(self):
        for shape in ((1, 12288), (2, 1, 12288), (1, 0, 12288), (1, 1, 12289), (1, 1, 1, 12288), (1, 1, 2560)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, r"\[1,T,D\]"):
                    neutral.neutral_embedding([[torch.zeros(shape, device="cpu"), {}]])

    def test_klein4b_context_width_accepted(self):
        # Klein 4B: qwen3_4b, 2560 hidden x 3 tapped layers = 7680
        result = neutral.neutral_embedding([[torch.zeros((1, 9, 7680), device="cpu"), {}]])
        self.assertEqual(tuple(result.shape), (1, 512, 7680))
        self.assertEqual(neutral.CONTEXT_DIMS, (12288, 7680))

    def test_integer_bool_complex_and_non_tensor_embedding_rejected(self):
        for dtype in (torch.int64, torch.bool, torch.complex64):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "floating-point"):
                    neutral.neutral_embedding([[torch.zeros((1, 1, 12288), dtype=dtype, device="cpu"), {}]])
        with self.assertRaisesRegex(ValueError, "torch.Tensor"):
            neutral.neutral_embedding([[[1, 2, 3], {}]])

    def test_nan_inf_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            source = embedding()
            source[0, 0, 0] = bad
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "finite"):
                    neutral.neutral_embedding([[source, {}]])

    def test_nested_sparse_and_meta_rejected_cleanly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            nested = torch.nested.nested_tensor([torch.zeros((1, 12288), device="cpu"), torch.zeros((2, 12288), device="cpu")])
        for value in (nested, embedding().to_sparse()):
            with self.assertRaisesRegex(ValueError, "non-nested"):
                neutral.neutral_embedding([[value, {}]])
        with self.assertRaisesRegex(ValueError, "meta tensor"):
            neutral.neutral_embedding([[torch.empty((1, 3, 12288), device="meta"), {}]])


class MetadataTests(CpuOnlyTestCase):
    def test_all_semantic_and_unknown_keys_fail_closed_even_if_falsey(self):
        keys = (
            "start_percent", "end_percent", "clip_start_percent", "clip_end_percent", "timestep_start", "timestep_end",
            "hooks", "schedule", "area", "strength", "mask", "mask_strength", "set_area_to_bounds",
            "control", "control_apply_to_uncond", "reference_latents", "ref_latents", "reference_latents_method",
            "guidance", "attention_mask_img_shape", "gligen", "model_conds", "uuid", "future_unknown", 7,
        )
        source = embedding()
        for key in keys:
            for value in (None, False):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, "unsupported metadata field"):
                        neutral.neutral_embedding([[source, {key: value}]])

    def test_non_none_pooled_rejected_instead_of_dropping(self):
        for value in (torch.zeros((1, 4096), device="cpu"), False, 0, [], "pooled"):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "pooled_output must be None"):
                    neutral.neutral_embedding([[embedding(), {"pooled_output": value}]])

    def test_te_mask_preserved_cpu_detached_copied_and_left_padded(self):
        for dtype in (torch.bool, torch.int32, torch.int64, torch.float32):
            with self.subTest(dtype=dtype):
                mask = torch.tensor([[1, 1, 0]], dtype=dtype, device="cpu")
                source = conditioning(3, {"pooled_output": None, "attention_mask": mask})
                clip = FakeClip(source)
                output = neutral.encode_neutral_conditioning(clip)
                outmask = output[0][1]["attention_mask"]
                self.assertEqual(set(output[0][1]), {"pooled_output", "attention_mask"})
                self.assertIsNone(output[0][1]["pooled_output"])
                self.assertEqual(tuple(outmask.shape), (1, 512))
                self.assertEqual(outmask.dtype, dtype)
                self.assertEqual(outmask.device.type, "cpu")
                self.assertEqual(torch.count_nonzero(outmask[:, :-3]).item(), 0)
                self.assertTrue(torch.equal(outmask[:, -3:], mask))
                self.assertFalse(outmask.requires_grad)
                self.assertNotEqual(outmask.data_ptr(), mask.data_ptr())
                self.assertEqual(tuple(mask.shape), (1, 3))
                self.assertIsNot(output[0][1], source[0][1])

    def test_mask_must_match_original_not_padded_length(self):
        for shape in ((3,), (3, 1), (2, 3), (1, 512), (1, 1, 3), (1, 1, 3, 3)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "TE binary mask.*shape"):
                    neutral.neutral_embedding(conditioning(3, {"attention_mask": torch.ones(shape, device="cpu")}))

    def test_mask_values_and_kinds_rejected(self):
        cases = (None, [[1, 1, 0]], torch.zeros((1, 3), device="cpu"),
                 torch.tensor([[1.0, 0.5, 0.0]], device="cpu"), torch.tensor([[1.0, -1.0, 0.0]], device="cpu"),
                 torch.tensor([[1.0, float("nan"), 0.0]], device="cpu"), torch.tensor([[1.0, float("inf"), 0.0]], device="cpu"),
                 torch.ones((1, 3), dtype=torch.complex64, device="cpu"), torch.ones((1, 3), device="cpu").to_sparse())
        for mask in cases:
            with self.subTest(mask_type=type(mask).__name__):
                with self.assertRaisesRegex(ValueError, "attention_mask"):
                    neutral.neutral_embedding(conditioning(3, {"attention_mask": mask}))


class EncodingAndNodeTests(CpuOnlyTestCase):
    def test_node_contract_and_fixed_prompt(self):
        node = neutral.KleinNeutralConditioning()
        self.assertEqual(node.INPUT_TYPES(), {"required": {"clip": ("CLIP",)}})
        self.assertEqual(node.RETURN_TYPES, ("CONDITIONING",))
        self.assertEqual(node.FUNCTION, "encode")
        self.assertEqual(neutral.NEUTRAL_PROMPT, "Keep the image unchanged.")
        clip = FakeClip()
        result = node.encode(clip)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0][0][1], dict)
        self.assertEqual(tuple(result[0][0][0].shape), (1, 512, 12288))
        self.assertEqual(clip.calls[0], ("tokenize", "Keep the image unchanged."))
        self.assertEqual([call[0] for call in clip.calls], ["tokenize", "encode"])
        self.assertEqual(vars(node), {})

    def test_prepare_offloads_only_after_cpu_copy_and_on_error(self):
        clip=FakeClip()
        result=neutral.KleinNeutralConditioning().encode(clip)
        self.assertEqual(result[0][0][0].device.type,'cpu')
        self.offload_mock.assert_called_once_with(clip)
        self.offload_mock.reset_mock()
        clip.encode_from_tokens_scheduled=lambda tokens: (_ for _ in ()).throw(RuntimeError('cancelled encoding'))
        with self.assertRaisesRegex(RuntimeError,'cancelled encoding'):
            neutral.KleinNeutralConditioning().encode(clip)
        self.offload_mock.assert_called_once_with(clip)
    def test_all_clip_hooks_and_schedules_rejected_before_tokenize(self):
        for field, value in (("forced_hooks", object()), ("forced_hooks", []), ("apply_hooks_to_conds", object()),
                             ("apply_hooks_to_conds", []), ("apply_hooks_to_conds", False), ("use_clip_schedule", True)):
            with self.subTest(field=field, value_type=type(value).__name__):
                clip = FakeClip()
                setattr(clip.patcher if field == "forced_hooks" else clip, field, value)
                with self.assertRaisesRegex(ValueError, field):
                    neutral.encode_neutral_conditioning(clip)
                self.assertEqual(clip.calls, [])

    def test_missing_clip_patcher_or_encoder_is_clear(self):
        with self.assertRaisesRegex(ValueError, "connect neutral_conditioning"):
            neutral.encode_neutral_conditioning(None)
        with self.assertRaisesRegex(ValueError, "no patcher"):
            neutral.encode_neutral_conditioning(types.SimpleNamespace())
        clip = FakeClip()
        clip.encode_from_tokens_scheduled = None
        with self.assertRaisesRegex(ValueError, "no metadata-dropping encoder fallback"):
            neutral.encode_neutral_conditioning(clip)
        self.assertEqual(clip.calls, [])

    def test_tokenizer_cannot_enable_hooks_before_encode(self):
        clip = FakeClip()
        def tokenize(text):
            clip.patcher.forced_hooks = []
            return "tokens"
        clip.tokenize = tokenize
        with self.assertRaisesRegex(ValueError, "forced_hooks"):
            neutral.encode_neutral_conditioning(clip)
        self.assertEqual(clip.calls, [])

    def test_encoder_multiple_or_semantic_conditioning_never_sliced(self):
        for output in (conditioning() * 2, conditioning(metadata={"guidance": 3.5}), conditioning(metadata={"hooks": None})):
            with self.subTest(count=len(output)):
                with self.assertRaisesRegex(ValueError, "exactly one|unsupported metadata"):
                    neutral.KleinNeutralConditioning().encode(FakeClip(output))

    def test_encoder_exceptions_and_cancellation_propagate(self):
        class Cancelled(Exception):
            pass
        for exception in (Cancelled("stop"), RuntimeError("encoder failure")):
            clip = FakeClip()
            def encode(tokens):
                raise exception
            clip.encode_from_tokens_scheduled = encode
            with self.assertRaises(type(exception)) as caught:
                neutral.encode_neutral_conditioning(clip)
            self.assertIs(caught.exception, exception)

    def test_no_clip_ownership_global_cache_or_implicit_unload(self):
        node = neutral.KleinNeutralConditioning()
        clip = FakeClip()
        clip_ref = weakref.ref(clip)
        first = node.encode(clip)[0]
        clip.result[0][0].fill_(8)
        second = node.encode(clip)[0]
        self.assertEqual(first[0][0][0, -1, 0].item(), 2)
        self.assertEqual(second[0][0][0, -1, 0].item(), 8)
        self.assertEqual(sum(call[0] == "encode" for call in clip.calls), 2)
        self.offload_mock.reset_mock()  # Test mock call history must not own CLIP.
        del clip
        gc.collect()
        self.assertIsNone(clip_ref())
        self.assertEqual(vars(node), {})
        self.assertFalse(any(name.startswith("comfy") for name in sys.modules))


def core_method(relative_path, class_name, method_name, namespace=None):
    """Compile one source AST method, never import a GPU-aware core module."""
    root = Path(os.environ["CORE"])
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {"torch": torch}
    if namespace:
        scope.update(namespace)
    exec(compile(module, relative_path, "exec"), scope)
    return scope[method_name]


@unittest.skipUnless(os.environ.get("CORE"), "CORE not set; optional real-core AST contract tests skipped")
class RealCoreContractTests(CpuOnlyTestCase):
    def test_real_sd_clip_unscheduled_roundtrip_keeps_metadata(self):
        fake_mm = types.SimpleNamespace(cuda_device_context=lambda device: contextlib.nullcontext())
        methods = {name: core_method("comfy/sd.py", "CLIP", name, {"model_management": fake_mm})
                   for name in ("encode_from_tokens_scheduled", "encode_from_tokens", "add_hooks_to_dict")}
        class CoreClip(FakeClip):
            def load_model(self, tokens):
                pass
        for name, method in methods.items():
            setattr(CoreClip, name, method)
        clip = CoreClip()
        clip.layer_idx = None
        clip.patcher.load_device = "cpu"
        raw = clip.result[0]
        clip.cond_stage_model = types.SimpleNamespace(reset_clip_options=lambda: None, set_clip_options=lambda options: None,
                                                     encode_token_weights=lambda tokens: (raw[0], None, {"attention_mask": raw[1]["attention_mask"]}))
        result = neutral.encode_neutral_conditioning(clip)
        self.assertEqual(set(result[0][1]), {"pooled_output", "attention_mask"})
        self.assertIsNone(result[0][1]["pooled_output"])
        self.assertEqual(tuple(result[0][1]["attention_mask"].shape), (1, 512))

    def test_real_sdclip_qwen_style_pair_produces_none_pooled_and_te_mask(self):
        forward = core_method("comfy/sd1_clip.py", "SDClipModel", "forward")
        hidden = torch.zeros((1, 3, 3, 4096), device="cpu")
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long, device="cpu")
        encoder = types.SimpleNamespace(execution_device="cpu", enable_attention_masks=True, layer=[9, 18, 27],
                                        layer_idx=None, layer_norm_hidden_state=False, zero_out_masked=False,
                                        return_projected_pooled=True, return_attention_masks=True,
                                        process_tokens=lambda tokens, device: (torch.zeros((1, 3, 4096), device="cpu"), mask, [2], []),
                                        transformer=lambda *args, **kwargs: (hidden[:, -1], hidden))
        result = forward(encoder, [[1, 2, 151643]])
        self.assertIsNone(result[1])
        self.assertTrue(torch.equal(result[2]["attention_mask"], mask))
        self.assertEqual(tuple(result[2]["attention_mask"].shape), (1, 3))

    def test_real_token_weight_encoder_flattens_te_mask_not_attention_matrix(self):
        encode = core_method("comfy/sd1_clip.py", "ClipTokenWeightEncoder", "encode_token_weights",
                             {"model_management": types.SimpleNamespace(intermediate_device=lambda: "cpu")})
        hidden = torch.zeros((2, 3, 12288), device="cpu")
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long, device="cpu")
        encoder = types.SimpleNamespace(encode=lambda tokens: (hidden, None, {"attention_mask": mask}))
        result = encode(encoder, [[(1, 1.0)] * 3, [(2, 1.0)] * 3])
        self.assertEqual(tuple(result[0].shape), (1, 6, 12288))
        self.assertEqual(tuple(result[2]["attention_mask"].shape), (1, 6))
        self.assertIsNone(result[1])
        prepared = neutral.neutral_embedding([[result[0], {"pooled_output": result[1], **result[2]}]])
        self.assertEqual(tuple(prepared.shape), (1, 512, 12288))

    def test_real_flux2_extra_conds_left_pad_equivalence(self):
        root = Path(os.environ["CORE"])
        tree = ast.parse((root / "comfy/model_base.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Flux2")
        class FluxStub:
            def extra_conds(self, **kwargs):
                return {}
        scope = {"torch": torch, "Flux": FluxStub,
                 "comfy": types.SimpleNamespace(conds=types.SimpleNamespace(CONDRegular=lambda tensor: tensor))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "model_base.py", "exec"), scope)
        for tokens in (3, 512, 513):
            source = embedding(tokens)
            expected = scope["Flux2"]().extra_conds(cross_attn=source)["c_crossattn"]
            self.assertTrue(torch.equal(neutral.neutral_embedding([[source, {}]]), expected))


if __name__ == "__main__":
    print("Python:", sys.version.split()[0], "torch:", torch.__version__, "CUDA initialized:", torch.cuda.is_initialized(), flush=True)
    unittest.main(verbosity=2)
