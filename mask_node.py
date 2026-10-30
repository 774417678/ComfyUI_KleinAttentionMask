"""把采集到的内部注意力聚合为 MASK 的节点"""

import torch

from . import probe


class KleinAttentionMaskExtract:
    """IMAGE 直通 + 输出内部注意力掩码。

    必须接在 KSampler → VAE Decode 之后（保证探针已采集完成）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.90, "min": 0.5, "max": 0.999, "step": 0.005,
                                        "tooltip": "按注意力分位取阈值（0.90=取最亮的10%区域）"}),
                "grow": ("INT", {"default": 6, "min": 0, "max": 64}),
                "feather": ("INT", {"default": 8, "min": 0, "max": 64}),
                "aggregation": (["mean", "max"],),
                "metric": (["reference", "text"], {"default": "reference", "tooltip": "reference=对参考图token注意力；text=对文本token注意力"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "extract"
    CATEGORY = "KleinMask"

    def extract(self, image, threshold, grow, feather, aggregation, metric):
        maps = probe.STORE.get("maps", [])
        probe.STORE["armed"] = False
        if not maps:
            raise RuntimeError(
                "KleinAttentionMask: 未采集到任何注意力图。请确认探针节点"
                "（Klein Attention Mask Probe）接在 MODEL 链路上且位于 KSampler 之前。"
            )

        if isinstance(maps[0], dict):
            selected = [x.get(metric, x.get("text")) for x in maps]
        else:
            selected = maps
        stack = torch.stack(selected, 0)        # [N,B,T]（兼容旧版 [N,T]）
        if stack.ndim == 2:
            stack = stack[:, None, :]
        agg = stack.mean(0) if aggregation == "mean" else stack.max(0).values  # [B,T]

        # 由图像推断 token 网格（Flux2 VAE+2x patch 后约为像素/16）
        b, ih, iw, _ = image.shape
        masks = []
        for bi in range(b):
            a = agg[min(bi, agg.shape[0] - 1)]
            a = a - a.min()
            if a.max() > 0:
                a = a / a.max()
            tok_h = max(1, round(ih / 16))
            tok_w = max(1, round(iw / 16))
            tokens = a.numel()
            if tokens == tok_h * tok_w:
                grid = a.view(tok_h, tok_w)
            else:
                # 兜底：按图像长宽比推网格
                r = ih / iw
                gh = max(1, round(math_sqrt(tokens * r)))
                gw = max(1, tokens // gh)
                usable = gh * gw
                grid = a[:usable].view(gh, gw)

            m = grid[None, None]
            m = torch.nn.functional.interpolate(m, size=(ih, iw), mode="bilinear", align_corners=False)[0, 0]
            t = torch.quantile(m.flatten(), threshold).item()
            mask = (m > t).float()
            mask = _largest_component(mask)
            if grow > 0:
                mask = _dilate(mask, grow)
            if feather > 0:
                mask = _blur(mask, feather)
            masks.append(torch.clamp(mask, 0, 1))

        mask_out = torch.stack(masks, 0).to(image.device, image.dtype)
        return (image, mask_out)


def math_sqrt(x):
    import math
    return math.sqrt(max(x, 1.0))


def _dilate(mask, px):
    k = px * 2 + 1
    m = mask[None, None]
    m = torch.nn.functional.max_pool2d(m, k, stride=1, padding=px)
    return m[0, 0]


def _blur(mask, px):
    sigma = max(px / 2.0, 0.5)
    radius = int(sigma * 3)
    if radius < 1:
        return mask
    xs = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
    g = torch.exp(-(xs * xs) / (2 * sigma * sigma))
    g = (g / g.sum())
    m = mask[None, None]
    m = torch.nn.functional.conv2d(m, g.view(1, 1, 1, -1), padding=(0, radius))
    m = torch.nn.functional.conv2d(m, g.view(1, 1, -1, 1), padding=(radius, 0))
    return m[0, 0]


def _largest_component(mask):
    """保留最大的连通区域（纯 torch 实现，不依赖 cv2/scipy）。"""
    import numpy as np
    try:
        from scipy import ndimage
        m = mask.cpu().numpy() > 0.5
        if not m.any():
            return mask
        lab, n = ndimage.label(m)
        if n <= 1:
            return mask
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = (sizes.argmax() + 1)
        out = (lab == keep).astype(np.float32)
        return torch.from_numpy(out).to(mask.device)
    except Exception:
        return mask  # scipy 缺失时退化为不滤连通域


class KleinAttentionMaskProbe:
    """挂载内部注意力探针。MODEL 直通。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "img_tokens": ("INT", {"default": 0, "min": 0, "max": 100000,
                                         "tooltip": "生成图 token 数覆盖值；0=从 Klein reference_image_num_tokens 自动推导"}),
                "capture_every": ("INT", {"default": 8, "min": 1, "max": 128,
                                          "tooltip": "每 N 次块调用采集一次（覆盖不同块/步）"}),
                "max_captures": ("INT", {"default": 24, "min": 1, "max": 256}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"
    CATEGORY = "KleinMask"

    def attach(self, model, img_tokens, capture_every, max_captures):
        m = model.clone()
        probe.reset_store({"capture_every": capture_every, "max_captures": max_captures,
                           "img_tokens": img_tokens})
        m.set_model_attn1_patch(probe.make_patch(capture_every=capture_every, max_captures=max_captures))
        return (m,)


NODE_CLASS_MAPPINGS = {
    "KleinAttentionMaskProbe": KleinAttentionMaskProbe,
    "KleinAttentionMaskExtract": KleinAttentionMaskExtract,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KleinAttentionMaskProbe": "Klein 内部注意力探针 (MODEL)",
    "KleinAttentionMaskExtract": "Klein 注意力转掩码 (IMAGE→MASK)",
}
