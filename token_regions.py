"""Token-region attention for Flux2 Klein (experimental, prototype).

Each block of the Klein forward runs one joint attention over

    [ text (left-padded to >= 512) | target image tokens | reference image tokens ]

This patch adds an additive bias to that attention so tokens are treated by
region:

* protected target tokens do not attend to the *edit text* tokens (and the
  edit text does not attend to them), so instructions cannot pull those areas;
* optionally, a protected target token is biased toward the reference token at
  the same position, which helps it copy the source image;
* target-region tokens and the chat-template tokens are untouched.

Hooks: ``post_input`` (records the token layout of the current call) and
``optimized_attention_override`` (adds the bias and calls PyTorch SDPA). No
core file is changed.

Prototype limits:
* exactly one reference image of the same latent size as the output;
* one (T x T) bias per batch row is materialised: ~9.7k tokens at 1024^2 is
  about 190 MB in bf16 per row;
* biased calls use PyTorch SDPA instead of sage/flash kernels (slower).
"""
import json

import torch
import torch.nn.functional as F

NEG = -1e4
PAD_ID = 151643


def _to_tokens(mask, h, w):
    m = mask.float()
    if m.ndim == 2:
        m = m[None]
    if m.ndim == 4:
        m = m[:, 0]
    m = F.interpolate(m[:, None], size=(h, w), mode='area')[:, 0]
    return m.flatten(1).clamp(0, 1)            # [B, h*w]


def build_bias(layout, protect, edit_span, text_suppress, self_bias, batch, device, dtype):
    """Additive attention bias [B,1,T,T] for one sequence layout."""
    txt, n, ref, h, w = layout['txt'], layout['target'], layout['ref'], layout['h'], layout['w']
    if h * w != n:
        raise ValueError('Token grid %dx%d does not match %d target tokens.' % (h, w, n))
    T = txt + n + ref
    prot = _to_tokens(protect, h, w).to(device)
    if prot.shape[0] != batch:
        prot = prot[:1].expand(batch, -1)
    bias = torch.zeros(batch, T, T, device=device, dtype=torch.float32)
    t0, t1 = txt, txt + n
    lo, hi = edit_span
    if text_suppress > 0 and hi > lo:
        bias[:, t0:t1, lo:hi] = (NEG * text_suppress) * prot[:, :, None]
        bias[:, lo:hi, t0:t1] = (NEG * text_suppress) * prot[:, None, :]
    if self_bias != 0 and ref == n:
        idx = torch.arange(n, device=device)
        bias[:, t0 + idx, t1 + idx] += self_bias * prot
    return bias.to(dtype)[:, None]


class TokenRegionBias:
    """State for one patched model: layout per call + cached bias."""

    def __init__(self, protect, text_suppress, self_bias, edit_tokens, tail_tokens, start_block):
        self.protect = protect
        self.text_suppress = float(text_suppress)
        self.self_bias = float(self_bias)
        self.edit_tokens = int(edit_tokens)
        self.tail_tokens = int(tail_tokens)
        self.start_block = int(start_block)
        self.layout = None
        self.cache_key = None
        self.cache_bias = None
        self.calls = 0
        self.biased = 0
        self.skipped = 0

    def edit_span(self, txt_len):
        hi = max(0, txt_len - self.tail_tokens)
        lo = max(0, hi - self.edit_tokens)
        return lo, hi

    def post_input(self, args):
        img_ids, txt = args['img_ids'], args['txt']
        to = args.get('transformer_options', {})
        ref = int(sum(to.get('reference_image_num_tokens', []) or []))
        n = img_ids.shape[1] - ref
        ids = img_ids[0, :n]
        h = int(round(float(ids[:, 1].max() - ids[:, 1].min()))) + 1 if n else 0
        w = int(round(float(ids[:, 2].max() - ids[:, 2].min()))) + 1 if n else 0
        self.layout = {'txt': int(txt.shape[1]), 'target': int(n), 'ref': ref, 'h': h, 'w': w}
        return args

    def bias(self, batch, device, dtype):
        L = self.layout
        key = (batch, L['txt'], L['target'], L['ref'], str(device), str(dtype))
        if key != self.cache_key:
            self.cache_bias = build_bias(L, self.protect, self.edit_span(L['txt']), self.text_suppress,
                                         self.self_bias, batch, device, dtype)
            self.cache_key = key
        return self.cache_bias

    def attention(self, func, q, k, v, heads, mask=None, transformer_options=None, **kwargs):
        self.calls += 1
        to = transformer_options or {}
        L = self.layout
        early = to.get('block_type') == 'double' and to.get('block_index', 0) < self.start_block
        if L is None or early or q.ndim != 4 or q.shape[2] != L['txt'] + L['target'] + L['ref']:
            self.skipped += 1
            return func(q, k, v, heads, mask=mask, transformer_options=transformer_options, **kwargs)
        b = self.bias(q.shape[0], q.device, q.dtype)
        if mask is not None:
            m = mask
            while m.ndim < 4:
                m = m.unsqueeze(0)
            b = b + m.to(b.dtype)
        self.biased += 1
        from comfy.ldm.modules.attention import attention_pytorch
        return attention_pytorch(q, k, v, heads, mask=b, transformer_options=transformer_options, **kwargs)

    def report(self):
        return {'attention_calls': self.calls, 'biased_calls': self.biased, 'skipped_calls': self.skipped,
                'layout': self.layout, 'edit_span': None if self.layout is None else self.edit_span(self.layout['txt']),
                'text_suppress': self.text_suppress, 'self_bias': self.self_bias}


def template_split(clip, prompt):
    """(instruction tokens, template tokens after the instruction) in Klein's chat template."""
    def ids(text):
        rows = next(iter(clip.tokenize(text).values()))
        return [t[0] for row in rows for t in row if isinstance(t[0], int) and t[0] != PAD_ID]
    full, empty = ids(prompt), ids('')
    tail = 0
    while tail < min(len(full), len(empty)) and full[-1 - tail] == empty[-1 - tail]:
        tail += 1
    head = 0
    while head < min(len(full), len(empty)) - tail and full[head] == empty[head]:
        head += 1
    return max(0, len(full) - head - tail), tail


class KleinTokenRegionAttention:
    """Region-aware joint attention inside the Klein forward (prototype)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'clip': ('CLIP', {'tooltip': 'Klein text encoder; only used to locate the instruction tokens.'}),
                'prompt': ('STRING', {'forceInput': True, 'tooltip': 'The exact edit prompt sent to CLIPTextEncode (e.g. filtered_prompt).'}),
                'protected_region': ('MASK',),
                'text_suppress': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                                            'tooltip': '1 = protected tokens cannot attend to the edit text at all.'}),
                'self_bias': ('FLOAT', {'default': 0.0, 'min': -4.0, 'max': 8.0, 'step': 0.25,
                                        'tooltip': 'Extra logit for a protected token attending to its own source-image token.'}),
                'start_block': ('INT', {'default': 0, 'min': 0, 'max': 64,
                                        'tooltip': 'Skip the bias on double blocks before this index.'}),
            },
        }

    RETURN_TYPES = ('MODEL', 'STRING')
    RETURN_NAMES = ('model', 'info')
    FUNCTION = 'patch'
    CATEGORY = 'KleinMask/experimental'
    DESCRIPTION = 'Prototype: protected image tokens ignore the edit text inside the Klein joint attention.'

    def patch(self, model, clip, prompt, protected_region, text_suppress, self_bias, start_block):
        edit_tokens, tail = template_split(clip, prompt)
        state = TokenRegionBias(protected_region.detach().cpu().float(), text_suppress, self_bias,
                                edit_tokens, tail, start_block)
        m = model.clone()
        to = m.model_options.setdefault('transformer_options', {})
        if to.get('optimized_attention_override') is not None:
            raise ValueError('Another attention override is already installed on this model.')
        m.set_model_post_input_patch(state.post_input)
        to['optimized_attention_override'] = state.attention
        m.model_options['klein_token_region'] = state
        return (m, json.dumps({'edit_tokens': edit_tokens, 'tail_template_tokens': tail,
                               'text_suppress': text_suppress, 'self_bias': self_bias}, ensure_ascii=False))
