"""Klein 内部注意力掩码探针
=========================

不依赖 SAM / 外部分割，直接挂进 FLUX.2 Klein（Flux 双流块）推理过程：
利用 ComfyUI 原生 ``attn1_patch`` 钩子截获 rope 前的 q/k，按 chunk 自行计算
注意力权重，统计每个图像 token 分配给文本 token 的注意力质量
（"图像这块区域有多听文本的话"）。编辑目标区域的该指标显著偏高，
聚合多个块/步后取阈值即可得到目标重绘区域掩码。
"""

import math
import torch

from comfy.ldm.flux.math import apply_rope

# 全局采集仓库（同一次采样内共享）
STORE = {
    "maps": [],          # list[Tensor 1D] 每次采集的 per-image-token 文本注意力
    "call_index": 0,
    "armed": False,
    "params": {},
}

CHUNK_ROWS = 256  # 分块计算行数，控制显存


def _sigma_fraction(extra_options):
    """尽力获取当前步的 sigma 归一化进度；拿不到返回 None。"""
    to = extra_options.get("transformer_options", extra_options) or {}
    sigmas = to.get("sigmas") or extra_options.get("sigmas")
    sigma = to.get("sigma") or extra_options.get("sigma")
    try:
        if sigma is not None:
            s = float(sigma if torch.is_tensor(sigma) else sigma)
            return s  # 原始 sigma，由调用方与阈值比较
        if torch.is_tensor(sigmas) and sigmas.numel() >= 2:
            s = float(sigmas.flatten()[0])
            m = float(sigmas.flatten()[-2]) if sigmas.numel() > 1 else 1.0
            return s / max(m, 1e-6)
    except Exception:
        pass
    return None


def make_patch(img_tokens=0, capture_every=8, max_captures=24,
               sigma_threshold=None, use_sigma_gate=False):
    """构造 attn1_patch。参数在探针节点执行时固化。"""

    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        p = STORE["params"]
        if not STORE.get("armed"):
            return {}
        # 门控：按调用序号均匀撒点，覆盖不同块与步
        STORE["call_index"] += 1
        if (STORE["call_index"] - 1) % max(1, int(p.get("capture_every", capture_every))) != 0:
            return {}
        if len(STORE["maps"]) >= int(p.get("max_captures", max_captures)):
            return {}

        img_slice = extra_options.get("img_slice") if extra_options else None
        if img_slice is None:
            return {}
        txt_len, total_len = int(img_slice[0]), int(img_slice[1])
        if total_len - txt_len < 16:
            return {}

        try:
            with torch.no_grad():
                if pe is not None:
                    qr, kr = apply_rope(q.clone(), k.clone(), pe)  # rope 后副本，不动原张量
                else:
                    qr, kr = q, k
                heads = qr.shape[1]
                rows_total = total_len - txt_len
                refs = (extra_options or {}).get('reference_image_num_tokens', [])
                inferred = rows_total - sum(refs) if refs else rows_total
                keep = int(p.get("img_tokens", img_tokens)) or inferred
                keep = min(keep, rows_total)
                # Flux/Kontext 将原始待生成 img tokens 放在 img 段前部，参考图 tokens 追加在其后。
                # 只取前 keep 行，排除追加的 reference/Kontext tokens。
                q_img = qr[:, :, txt_len: txt_len + keep, :]
                k_all = kr
                dim = q_img.shape[-1]
                scale = 1.0 / math.sqrt(dim)

                acc_t = torch.zeros((q_img.shape[0], keep), dtype=torch.float32, device=q_img.device)
                acc_r = torch.zeros_like(acc_t)
                ref_c0, ref_c1 = txt_len + keep, k_all.shape[2]
                for s0 in range(0, keep, CHUNK_ROWS):
                    s1 = min(s0 + CHUNK_ROWS, keep)
                    qc = q_img[:, :, s0:s1, :].to(torch.float32)          # b,h,r,d
                    kc = k_all.to(torch.float32)                           # b,h,L,d
                    logits = torch.einsum("bhrd,bhld->bhrl", qc, kc) * scale
                    probs = logits.softmax(dim=-1)
                    # 对文本 key 的注意力质量（背景 token 往往偏高）
                    acc_t[:, s0:s1] += probs[:, :, :, :txt_len].sum(dim=-1).mean(dim=1)
                    # 对参考图(Kontext) key 的注意力质量：编辑区域显著偏高（首选指标）
                    if ref_c1 > ref_c0:
                        acc_r[:, s0:s1] += probs[:, :, :, ref_c0:ref_c1].sum(dim=-1).mean(dim=1)
            STORE["maps"].append({"text": acc_t.detach().float().cpu(),
                                  "reference": acc_r.detach().float().cpu()})
        except Exception:
            pass  # 任何异常都不能影响正常采样
        return {}

    return patch


def reset_store(params=None):
    STORE["maps"] = []
    STORE["call_index"] = 0
    STORE["armed"] = True
    STORE["params"] = dict(params or {})
