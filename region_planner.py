"""Vision-language region planning for Klein edits (experimental).

A separate VL model (native ComfyUI Qwen3-VL ``clip.generate``) answers, before
sampling, which parts of the source image each instruction refers to. The
answer is a strict JSON plan; this module parses it and rasterises three maps:

* ``existing_region``  - where the edit targets already are in the source image
* ``target_region``    - where the edit is allowed to happen (existing region,
                         plus generous boxes for instructions that add things)
* ``protected_region`` - elements that must stay as they are (face, hands ...)

It also rewrites the prompt: instructions whose target is absent from the
source image are dropped, so the diffusion model never chases a missing object
(the typical cause of half-body edits collapsing or inventing new content).

The VL model never feeds Klein directly; only text and masks cross over, so any
VL size works. Parsing and rasterising are pure CPU code.
"""
import hashlib
import json
import re
import time
import weakref
from collections import OrderedDict

import torch
import torch.nn.functional as F

SPLIT_RE = re.compile(r'[\n\r，,。；;！!？?]+')
ADD_HINTS = ('添加', '增加', '加上', '戴上', '戴', '画上', '放上', '拿着', '手持', '叼着', '叼',
             ' add ', 'adding', ' put ', 'wear', 'holding')
PROTECT_HINTS = ('不要', '别', '勿', '保持', '保留', '不变', '不改', '维持', '不动',
                 ' keep ', ' do not ', " don't ", ' preserve ', ' unchanged')
VL_MAX_EDGE = 1024   # the VL only needs to recognise elements; full-resolution input costs minutes

PLAN_INSTRUCTIONS = """You are a precise visual grounding assistant for an image-editing pipeline.
First look at the image WITHOUT reading the instructions and report:
- "framing": how much of the main person is inside the frame, exactly one of: head, shoulders, chest, waist, hips, thighs, knees, full. Judge by the LOWEST body part that is visible. If the image is cut at the chest, answer chest even if the instructions talk about legs.
- "visible": short English names of the clothing items and body parts you can actually see.
Do not report anything as present just because an instruction mentions it. Many instructions name things that are NOT in this image; for those answer present=false with no boxes. Never use one large box for several different targets.
Then, for EACH numbered instruction below (answer every id, even if unsure), decide:
- "kind": "edit" if it changes something that should already be in the image, "add" if it introduces a new object, "style" if it describes the whole subject or image (e.g. ethnicity, art style), "protect" if it asks to keep something unchanged.
- "target": a short English noun phrase naming the image element it refers to.
- "present": true only if that element is clearly visible in THIS image (for "add", whether a suitable place exists).
- "boxes": bounding boxes of that element as [x1, y1, x2, y2], integers normalised to 0-1000 relative to image width/height. Empty list if not present. For "add", the box where the new object would go.
Also list "protected": at most 4 visible elements that none of the instructions ask to change but that are easily damaged by edits (face, hands, text, logos), each with "label" and "boxes".
Reply with ONLY one compact JSON object on a single line, without spaces, line breaks or markdown, in this exact shape:
{"framing":"chest","visible":["..."],"instructions":[{"id":1,"kind":"edit","target":"...","present":true,"boxes":[[x1,y1,x2,y2]]}],"protected":[{"label":"...","boxes":[[x1,y1,x2,y2]]}]}

Instructions:
"""

ACTIONS = ('auto', 'keep', 'drop')

# How far down the body the frame reaches. A clause about a body part below the
# framing cannot have a visible target, whatever the VL claims for that clause.
FRAMINGS = ('head', 'shoulders', 'chest', 'waist', 'hips', 'thighs', 'knees', 'full')
FRAMING_ALIASES = {
    'close-up': 'head', 'closeup': 'head', 'face': 'head', 'portrait': 'shoulders', 'headshot': 'head',
    'bust': 'chest', 'upper body': 'chest', 'upper-body': 'chest', 'half': 'waist', 'half body': 'waist',
    'half-body': 'waist', 'medium': 'waist', 'hip': 'hips', 'thigh': 'thighs', 'mid-thigh': 'thighs',
    'american': 'thighs', 'cowboy': 'thighs', 'knee': 'knees', 'full body': 'full', 'full-body': 'full',
    'whole body': 'full', 'long': 'full', 'feet': 'full',
}
# (keywords, lowest framing at which the part is visible); first match wins, most specific first.
BODY_PART_LEVELS = (
    (('小腿', '脚', '足', '鞋', '袜子', 'shoe', 'feet', 'foot', 'ankle', 'calf', 'calves', 'sock'), 'full'),
    (('膝', 'knee'), 'knees'),
    (('大腿', '腿', '丝袜', '长袜', 'thigh', 'leg', 'stocking', 'pantyhose', 'tights'), 'thighs'),
    (('裙', '内裤', '臀', '屁股', '胯', '裆', '隐私', '私处', '阴', '下体', '裤',
      'skirt', 'panties', 'underwear', 'hip', 'butt', 'crotch', 'pubic', 'genital', 'pussy', 'vagina',
      'pants', 'shorts', 'trousers'), 'hips'),
    (('腰', '腹', '肚', '脐', 'waist', 'belly', 'navel', 'abdomen', 'stomach'), 'waist'),
    (('胸', '乳', '奶', 'bra', '内衣', '文胸', 'breast', 'nipple', 'chest', 'boob'), 'chest'),
    (('肩', 'shoulder'), 'shoulders'),
)
FRAMING_CHOICES = ('auto',) + FRAMINGS


def normalise_framing(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in FRAMINGS:
        return v
    if v in FRAMING_ALIASES:
        return FRAMING_ALIASES[v]
    for alias, name in FRAMING_ALIASES.items():
        if alias in v:
            return name
    for name in FRAMINGS:
        if name in v:
            return name
    return None


def required_framing(clause):
    text = clause.lower()
    for words, level in BODY_PART_LEVELS:
        if any(w in text for w in words):
            return level
    return None


def framing_excludes(clause, framing):
    """True when the clause names a body part that lies below the framing."""
    need = required_framing(clause)
    if need is None or framing is None:
        return False
    return FRAMINGS.index(need) > FRAMINGS.index(framing)


def placeholder_boxes(plan, min_area=0.35, min_repeats=3):
    """Boxes the VL reused for several different targets: a sign it was guessing."""
    seen = {}
    for e in plan:
        for b in e['boxes']:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area >= min_area:
                seen.setdefault(tuple(round(v, 3) for v in b), set()).add(e['id'])
    return {box for box, ids in seen.items() if len(ids) >= min_repeats}


def split_instructions(prompt):
    """Split a prompt into instruction clauses on punctuation and line breaks."""
    parts = [p.strip() for p in SPLIT_RE.split(prompt or '')]
    return [p for p in parts if p]


def looks_additive(clause):
    text = ' ' + clause.lower() + ' '
    return any(h in text for h in ADD_HINTS)


def looks_protective(clause):
    text = ' ' + clause.lower() + ' '
    return any(h in text for h in PROTECT_HINTS)


def parse_overrides(text, count):
    """Optional per-clause override, one action per line: auto / keep / drop."""
    actions = ['auto'] * count
    if not text or not text.strip():
        return actions
    tokens = [t.strip().lower() for t in re.split(r'[\n,，\s]+', text) if t.strip()]
    for i, token in enumerate(tokens[:count]):
        if token not in ACTIONS:
            raise ValueError('instruction_overrides only accepts auto/keep/drop, got: ' + token)
        actions[i] = token
    return actions


def build_prompt(clauses):
    lines = [PLAN_INSTRUCTIONS]
    for i, clause in enumerate(clauses, 1):
        lines.append('%d. %s' % (i, clause))
    return '\n'.join(lines)


def _extract_json(text):
    """Return the first balanced JSON object in a VL reply."""
    if text is None:
        raise ValueError('VL model returned no text.')
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    cleaned = cleaned.replace('```json', '```')
    start = cleaned.find('{')
    while start >= 0:
        depth = 0; in_str = False; esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = cleaned.find('{', start + 1)
    raise ValueError('VL reply contains no valid JSON object: ' + cleaned[:300])


def _clean_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    scale = 1.0 if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5 else 1000.0
    x1, x2 = sorted((x1 / scale, x2 / scale)); y1, y2 = sorted((y1 / scale, y2 / scale))
    x1, y1 = max(0.0, x1), max(0.0, y1); x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
        return None
    return [x1, y1, x2, y2]


def parse_plan(text, clauses):
    """Normalise a VL reply into one entry per clause plus protected elements."""
    data = _extract_json(text)
    raw = data.get('instructions') if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ValueError('VL JSON has no "instructions" list.')
    by_id = {}
    for pos, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            key = int(item.get('id', pos + 1))
        except (TypeError, ValueError):
            key = pos + 1
        by_id.setdefault(key, item)
    plan = []
    for i, clause in enumerate(clauses, 1):
        item = by_id.get(i, {})
        kind = str(item.get('kind', 'edit')).lower()
        if kind not in ('edit', 'add', 'style', 'protect'):
            kind = 'edit'
        if looks_protective(clause):
            kind = 'protect'
        boxes = [b for b in (_clean_box(x) for x in item.get('boxes') or []) if b]
        present = item.get('present')
        present = bool(present) if isinstance(present, bool) else bool(boxes)
        plan.append({'id': i, 'clause': clause, 'kind': kind,
                     'target': str(item.get('target', '')), 'present': present,
                     'boxes': boxes, 'answered': i in by_id})
    protected = []
    for item in (data.get('protected') or []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        boxes = [b for b in (_clean_box(x) for x in item.get('boxes') or []) if b]
        if boxes:
            protected.append({'label': str(item.get('label', '')), 'boxes': boxes})
    return plan, protected


def parse_meta(text):
    """framing / visible list from a VL reply (absent in older-style replies)."""
    try:
        data = _extract_json(text)
    except ValueError:
        return None, []
    if not isinstance(data, dict):
        return None, []
    visible = data.get('visible') or []
    return normalise_framing(data.get('framing')), [str(v) for v in visible if isinstance(v, (str, int, float))]


def apply_checks(plan, framing):
    """Deterministic checks the VL cannot talk its way around.

    * A clause about a body part below the framing has no visible target.
    * Boxes reused for several different targets are guesses; they are dropped
      from the region maps (the clause keeps the VL's present flag otherwise).
    """
    guesses = placeholder_boxes(plan)
    for e in plan:
        e['vl_present'] = e['present']
        e['checks'] = []
        if framing_excludes(e['clause'], framing):
            e['present'] = False
            e['boxes'] = []
            e['checks'].append('framing=%s excludes %s' % (framing, required_framing(e['clause'])))
            continue
        if guesses:
            kept = [b for b in e['boxes'] if tuple(round(v, 3) for v in b) not in guesses]
            if len(kept) != len(e['boxes']):
                e['checks'].append('discarded placeholder box shared by several targets')
                e['boxes'] = kept
    return guesses


def decide(plan, overrides):
    """Decide which clauses stay. Additions and style text are never auto-dropped."""
    for entry, action in zip(plan, overrides):
        entry.setdefault('checks', [])
        additive = entry['kind'] == 'add' or (entry['kind'] != 'protect' and looks_additive(entry['clause']))
        if action == 'keep':
            keep, reason = True, 'forced keep'
        elif action == 'drop':
            keep, reason = False, 'forced drop'
        elif entry['kind'] == 'protect':
            # A protection clause only makes sense when its target is in the image;
            # naming an absent body part or object invites the model to draw it.
            if not entry['answered']:
                keep, reason = True, 'protection instruction; VL gave no answer, kept to be safe'
            elif entry['present']:
                keep, reason = True, 'protected target present (its area joins the protected region)'
            else:
                keep, reason = False, 'protected target not found in image; dropped to avoid prompt pollution'
        elif entry['checks'] and entry['checks'][0].startswith('framing='):
            # The clause names a body part outside the frame; whatever kind the VL
            # assigned (it may call such text "style"), keeping it pollutes the prompt.
            keep, reason = False, 'body part outside the frame (' + entry['checks'][0] + ')'
        elif entry['kind'] == 'style':
            keep, reason = True, 'style/global description'
        elif additive:
            keep, reason = True, 'adds a new element'
        elif not entry['answered']:
            keep, reason = True, 'VL gave no answer; kept to be safe'
        elif entry['present']:
            keep, reason = True, 'target present'
        else:
            keep, reason = False, 'target not found in image'
        entry['keep'] = keep
        entry['reason'] = reason
        entry['additive'] = additive
    return plan


def rasterise(boxes, height, width, grow_px=0, feather_px=0):
    """Union of normalised boxes as a [H,W] float map, optionally grown and feathered."""
    mask = torch.zeros(height, width)
    for x1, y1, x2, y2 in boxes:
        c1 = int(round(x1 * width)); c2 = int(round(x2 * width))
        r1 = int(round(y1 * height)); r2 = int(round(y2 * height))
        mask[r1:max(r2, r1 + 1), c1:max(c2, c1 + 1)] = 1.0
    if grow_px > 0:
        mask = F.max_pool2d(mask[None, None], 2 * grow_px + 1, 1, grow_px)[0, 0]
    if feather_px > 0:
        mask = F.avg_pool2d(mask[None, None], 2 * feather_px + 1, 1, feather_px, count_include_pad=False)[0, 0]
    return mask.clamp(0, 1)


def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, 1, r, count_include_pad=False)


def guided_refine(mask, image, radius=8, eps=1e-3, work_edge=768):
    """Edge-aware refinement of a [H,W] map with the source image as guide.

    Classic guided filter (He et al.): a box edge snaps to an image edge only
    when that edge lies within about ``radius`` pixels; in flat areas it just
    softens the box. It is not segmentation. The object shape itself comes from
    the Klein influence measured by KleinRegionGuard (shape_mode=influence).
    """
    if mask.amax() <= 0 or mask.amin() >= 1:
        return mask
    h, w = mask.shape
    scale = min(1.0, work_edge / max(h, w))
    sh, sw = max(8, round(h * scale)), max(8, round(w * scale))
    guide = image[0, ..., :3].float().mean(-1)[None, None]
    guide = F.interpolate(guide, size=(sh, sw), mode='area')
    p = F.interpolate(mask[None, None].float(), size=(sh, sw), mode='area')
    r = max(1, int(radius))
    mean_i = _box(guide, r); mean_p = _box(p, r)
    var_i = _box(guide * guide, r) - mean_i * mean_i
    cov_ip = _box(guide * p, r) - mean_i * mean_p
    a = cov_ip / (var_i + eps); b = mean_p - a * mean_i
    q = _box(a, r) * guide + _box(b, r)
    q = F.interpolate(q, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
    return q.clamp(0, 1)


def build_regions(plan, protected, height, width, grow_px, feather_px, protect_rest, refine=None):
    """Rasterise the three region maps from a decided plan.

    * existing  = boxes of kept clauses whose target is present (not additions)
    * target    = existing + boxes of kept additions (+ kept style boxes)
    * protected = protected elements, minus the target region; when
      ``protect_rest`` everything outside the target region is protected as well.
    """
    existing_boxes, target_boxes, keep_boxes = [], [], []
    for entry in plan:
        if not entry.get('keep'):
            continue
        if entry['kind'] == 'protect':
            keep_boxes.extend(entry['boxes'])
        elif entry['kind'] == 'style':
            # Style text describes the whole subject: editable, but not an edit target.
            target_boxes.extend(entry['boxes'])
        elif entry['additive']:
            target_boxes.extend(entry['boxes'])
        elif entry['present']:
            existing_boxes.extend(entry['boxes'])
            target_boxes.extend(entry['boxes'])
    existing = rasterise(existing_boxes, height, width, grow_px, feather_px)
    target = rasterise(target_boxes, height, width, grow_px, feather_px)
    named = rasterise([b for p in protected for b in p['boxes']] + keep_boxes, height, width, 0, feather_px)
    if refine is not None:
        existing, target, named = refine(existing), refine(target), refine(named)
    # Without any grounded target, "protect everything else" would lock the whole
    # image and silently cancel the edit. Fall back to the named elements only.
    grounded = bool(target.amax() > 0)
    rest = (1.0 - target) if (protect_rest and grounded) else torch.zeros_like(target)
    protect = torch.maximum(named * (1.0 - target), rest).clamp(0, 1)
    return existing, target, protect, grounded


def filtered_prompt(plan, joiner='，'):
    return joiner.join(e['clause'] for e in plan if e.get('keep'))


class KleinVLRegionPlanner:
    """Ask a VL model where each instruction lands, before Klein samples."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'vl_clip': ('CLIP', {'tooltip': 'Qwen3-VL loaded with Load CLIP. Any size works; it is only asked questions, Klein never reads its features.'}),
                'image': ('IMAGE', {'tooltip': 'The source image that will be edited.'}),
                'prompt': ('STRING', {'multiline': True, 'default': '', 'tooltip': 'Editing instruction; split into clauses on punctuation and line breaks.'}),
                'grow_px': ('INT', {'default': 24, 'min': 0, 'max': 512, 'tooltip': 'Expand target boxes (pixels).'}),
                'feather_px': ('INT', {'default': 16, 'min': 0, 'max': 512}),
                'protect_rest': ('BOOLEAN', {'default': True, 'tooltip': 'Treat everything outside the target region as protected, not only the listed elements.'}),
                'max_new_tokens': ('INT', {'default': 768, 'min': 64, 'max': 4096}),
            },
            'optional': {
                'instruction_overrides': ('STRING', {'multiline': True, 'default': '', 'tooltip': 'One per clause, in order: auto / keep / drop.'}),
                'vl_reply': ('STRING', {'forceInput': True, 'tooltip': 'Use an existing VL reply (e.g. from another service) instead of running vl_clip.'}),
                'refine': (['guided', 'box'], {'default': 'guided', 'tooltip': 'guided: box edges follow image edges (guided filter, no extra model). box: raw rectangles.'}),
                'refine_radius': ('INT', {'default': 8, 'min': 1, 'max': 64, 'tooltip': 'Guided filter radius at the 768 px working size.'}),
                'framing': (list(FRAMING_CHOICES), {'default': 'auto', 'tooltip': 'How far down the body the image reaches. auto uses the VL answer. Clauses about body parts below it (e.g. thighs in a chest-up crop) are treated as absent.'}),
            },
        }

    RETURN_TYPES = ('STRING', 'MASK', 'MASK', 'MASK', 'STRING', 'STRING')
    RETURN_NAMES = ('filtered_prompt', 'existing_region', 'target_region', 'protected_region', 'plan_json', 'vl_raw')
    FUNCTION = 'plan'
    CATEGORY = 'KleinMask/experimental'
    DESCRIPTION = ('Before sampling, a VL model locates each instruction target. Instructions whose target is '
                   'absent are dropped; existing / target / protected regions are produced for constrained sampling.')

    def plan(self, vl_clip, image, prompt, grow_px, feather_px, protect_rest, max_new_tokens,
             instruction_overrides='', vl_reply=None, refine='guided', refine_radius=8, framing='auto'):
        clauses = split_instructions(prompt)
        if not clauses:
            raise ValueError('prompt is empty.')
        overrides = parse_overrides(instruction_overrides, len(clauses))
        height, width = int(image.shape[1]), int(image.shape[2])
        raw = vl_reply if vl_reply else ask_vl(vl_clip, vl_image(image[:1]), build_prompt(clauses), max_new_tokens)
        plan, protected = parse_plan(raw, clauses)
        vl_framing, visible = parse_meta(raw)
        if framing not in FRAMING_CHOICES:
            raise ValueError('framing must be one of %s' % (FRAMING_CHOICES,))
        used_framing = vl_framing if framing == 'auto' else framing
        guesses = apply_checks(plan, used_framing)
        decide(plan, overrides)
        if refine not in ('guided', 'box'):
            raise ValueError('refine must be guided or box')
        refiner = (lambda m: guided_refine(m, image[:1].cpu(), refine_radius)) if refine == 'guided' else None
        existing, target, protect, grounded = build_regions(plan, protected, height, width, grow_px,
                                                            feather_px, protect_rest, refiner)
        text = filtered_prompt(plan)
        warnings = plan_warnings(plan, grounded, protect_rest)
        if guesses:
            warnings.append('VL reused %d large box(es) for several different targets; those boxes were ignored.'
                            % len(guesses))
        if used_framing is None:
            warnings.append('VL reported no framing; body-part checks were skipped (set framing manually).')
        report = json.dumps({'framing': used_framing, 'vl_framing': vl_framing, 'visible': visible,
                             'instructions': plan, 'protected': protected, 'warnings': warnings,
                             'grounded_target': grounded,
                             'filtered_prompt': text, 'image_size': [width, height]},
                            ensure_ascii=False, indent=1)
        print('[KleinVLRegionPlanner] ' + json.dumps({'kept': [e['clause'] for e in plan if e['keep']],
                                                       'dropped': [e['clause'] for e in plan if not e['keep']],
                                                       'answered': sum(e['answered'] for e in plan),
                                                       'warnings': warnings}, ensure_ascii=False))
        return (text, existing[None], target[None], protect[None], report, raw)


def plan_warnings(plan, grounded, protect_rest):
    out = []
    answered = sum(e['answered'] for e in plan)
    if plan and answered == 0:
        out.append('VL answered none of the instructions (it may decline some content). All instructions '
                   'were kept unfiltered; no edit region could be located.')
    elif answered < len(plan):
        out.append('VL skipped %d of %d instructions; skipped ones were kept unfiltered.'
                   % (len(plan) - answered, len(plan)))
    if not grounded and protect_rest:
        out.append('No target region was located, so protect_rest was not applied: only named elements '
                   'are protected and the rest of the image stays editable.')
    return out


def vl_image(image):
    """Downscale for the VL: element recognition does not need the full resolution."""
    h, w = int(image.shape[1]), int(image.shape[2])
    scale = VL_MAX_EDGE / max(h, w)
    if scale >= 1:
        return image
    nh, nw = max(32, round(h * scale / 32) * 32), max(32, round(w * scale / 32) * 32)
    return F.interpolate(image.movedim(-1, 1).float(), size=(nh, nw), mode='area').movedim(1, -1)


_VL_CACHE = OrderedDict()
_VL_CACHE_SIZE = 8


def _image_digest(image):
    small = F.interpolate(image.movedim(-1, 1).float(), size=(64, 64), mode='area')
    data = (small.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha1(data + repr(tuple(image.shape)).encode()).hexdigest()


def ask_vl(clip, image, question, max_new_tokens):
    """Greedy, deterministic VL answer through ComfyUI's native generate API.

    Answers are cached by (VL model, image content, question): batch reruns of
    the same image and prompt skip the slow token-by-token decoding.
    """
    if not callable(getattr(clip, 'generate', None)):
        raise ValueError('vl_clip cannot generate text; load Qwen3-VL with Load CLIP (a VL type, not flux2).')
    key = (id(getattr(clip, 'patcher', clip)), _image_digest(image), question, int(max_new_tokens))
    hit = _VL_CACHE.get(key)
    if hit is not None and hit[0]() is getattr(clip, 'patcher', clip):
        _VL_CACHE.move_to_end(key)
        print('[KleinVLRegionPlanner] VL answer reused from cache (same image and prompt).')
        return hit[1]
    tokens = clip.tokenize(question, image=image, min_length=1)
    t = time.perf_counter()
    try:
        ids = clip.generate(tokens, do_sample=False, max_length=max_new_tokens, temperature=1.0, top_k=1,
                            top_p=1.0, min_p=0.0, repetition_penalty=1.0, seed=0)
    finally:
        _offload(clip)
    reply = clip.decode(ids)
    print('[KleinVLRegionPlanner] VL generated %d tokens in %.1fs' % (len(ids) if hasattr(ids, '__len__') else -1,
                                                                    time.perf_counter() - t))
    try:
        _VL_CACHE[key] = (weakref.ref(getattr(clip, 'patcher', clip)), reply)
        while len(_VL_CACHE) > _VL_CACHE_SIZE:
            _VL_CACHE.popitem(last=False)
    except TypeError:
        pass
    return reply


def _offload(clip):
    try:
        import comfy.model_management
        comfy.model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False)
    except Exception as exc:  # unloading is best-effort; never mask the real result
        print('[KleinVLRegionPlanner] VL offload skipped: %s' % exc)
