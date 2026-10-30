# Klein 内部编辑掩码 · 旧版节点使用说明

> 本节描述旧版单节点 `KleinInternalEditMaskSampler`。新项目建议改用 `sampler_variants.py` 中的四个采样器变体（见根目录 README 与 `docs/SAMPLER_VARIANTS_ZH.md`），它们不固定采样算法，并保留 ComfyUI 原生采样行为。

## 接线

```
UNETLoader ─┬─ KleinInternalEditMaskSampler ─ KSampler ─ VAEDecode ─┬─ SaveImage
CLIPLoader ─┤                                                      └─ KleinAttentionMaskExtract
PainterFluxImageEdit ─ positive / negative / latent ───────────────┘
```

`PainterFluxImageEdit` 需连接 VAE 与 image1，**不要连接 `image1_mask`**（第一遍用于发现掩码，不能自带掩码）。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `mode` | `source_latent` | 五种对比基准，含义见根目录 README |
| `threshold` | `0.20` | 热图二值化绝对阈值，调大缩小选区 |
| `grow` | `2` | 掩码外扩像素 |
| `feather` | `4` | 边缘羽化像素 |
| `probe_steps` | `2` | 仅 `sparse_neutral`：中性配对次数 |
| `start_step` | `1` | 采集窗口起点，**预测序号**，0 起编号 |
| `end_step` | `-1` | 采集窗口终点，`-1` 为最后一步 |

此节点内部固定 `euler` / `simple` / `cfg=1.0` / `denoise=1.0`，且 `cond_or_uncond` 必须为 `[0]`。若需其他采样算法或 CFG，请使用采样器变体。

## 输出

```
samples, edit_mask, influence_heatmap, diagnostics, comparison_masks, comparison_heatmaps
```

`comparison_*` 仅在 `mode=compare_all` 时含五组结果，单模式时只含该模式一组。

## 用法

`edit_mask` 接 `ImageCompositeMasked`：destination = 原图，source = 编辑结果。白色采用编辑结果，黑色保留原图。

该节点在采样**之后**输出掩码，用于合成，不会约束刚才那次采样。若要局部二次重绘，请把掩码接到下一遍 inpaint。

## 遗留注意力探针（不推荐用于批量）

`KleinAttentionMaskProbe`（挂 MODEL）+ `KleinAttentionMaskExtract`（接 VAE 解码后）提供原始注意力观察。`img_tokens=0` 时会用 `reference_image_num_tokens` 自动扣除参考图 token。`reference` 为对参考图 key 的注意力总量，`text` 为对文本 key 的注意力总量。

**这两个指标都不是稳健的编辑目标度量**：实测中 `reference` 圈出整个人、`text` 偏向背景。且采集用全局状态存储，不适合批量或多个探针同时使用。保留仅为诊断对照。
