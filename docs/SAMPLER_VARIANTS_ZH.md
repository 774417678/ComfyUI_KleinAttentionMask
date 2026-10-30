# Klein 内部编辑掩码 · 采样器变体

四个采样器适配节点的接线、输出顺序与兼容范围。核心逻辑、模式与参数说明见根目录 `README.md`。

## 节点与输出

| 节点 | 原生采样接口 | 前几个输出 |
|---|---|---|
| `KleinMaskSamplerCustomAdvanced` | NOISE + GUIDER + SAMPLER + SIGMAS + LATENT | `output`, `denoised_output` |
| `KleinMaskSamplerCustom` | MODEL / add_noise / noise_seed / cfg / positive / negative / SAMPLER / SIGMAS / LATENT | `output`, `denoised_output` |
| `KleinMaskKSampler` | 标准 KSampler（含 `denoise`） | `samples` |
| `KleinMaskKSamplerAdvanced` | KSamplerAdvanced（加噪开关、起止步、保留剩余噪声） | `samples` |

之后固定追加 5 个输出：

```
edit_mask, influence_heatmap, diagnostics, comparison_masks, comparison_heatmaps
```

采样完全委托当前 ComfyUI 原生节点执行，噪声生成、seed、sigma、`noise_mask`、预览 callback、x0 输出、`downscale_ratio_spacial/temporal` 均由原生路径处理。掩码额外乘以输入 `noise_mask`，避免在不可重绘区域给出建议。

## 可选输入

| 输入 | 说明 |
|---|---|
| `clip` | `fixed_neutral` / `sparse_neutral` / `dense_neutral` / `compare_all` **必须接**；`source_latent` / `adjacent_prediction` 不需要 |
| `source_latent` | 干净原图 latent。默认以 `latent_image["samples"]` 为基准；**续采的带噪 latent 必须显式提供** |
| `mode` | 见 README「模式」一节 |
| `threshold` / `grow` / `feather` / `probe_steps` | 见 README「参数详解」 |
| `capture_start` / `capture_end` | **模型调用序号**，0 起编号，`-1` 表示到结束 |

## 调用序号 ≠ 步号

Heun、`dpmpp_2m`、ER-SDE 等算法会在一个 sigma 区间内多次调用模型，`capture_*` 统计的是模型调用次数，不是 scheduler 步数。自适应算法（如 `dpm_adaptive`）步数不固定，请填确定的 `capture_end`，并以 `diagnostics` 里的 `sigmas` 列表核对实际采集位置。

## 采集语义

掩码基于 **CFG 混合之前的原始正向分支预测**（`cond_or_uncond` flag `0`），不是 guidance 加权输出。支持正负条件一起打包或分开调用；负向分支绝不能被误当中性基准。

## 兼容范围

支持：原生 `BasicGuider`、`CFGGuider`；普通二维图像 latent 与 `noise_mask`。

显式拒绝：DualCFG / 未知自定义引导器、已有 `model_function_wrapper`、自定义 cond 执行器、nested / video latent、空间形状不匹配的 source latent、ControlNet 条件下的中性重放。

未验证：batch > 2、多 GPU、第三方量化执行引擎、带状态的自定义采样插件。

模型克隆仅隔离 patch 与 options，不是独立模型副本，不宣称并发安全。

## 显存与生命周期

- 2026-10-31 修复：高级采样器只克隆一次，上游 guider 的 options 独立复制；四种接口均在成功、异常、取消路径执行 `Recorder.close()`，解除包装器和临时模型之间的引用环。无需通过清空执行缓存释放权重。
- 普通“卸载模型”对应 `POST /free` 的 `{"unload_models": true}`；“同时清执行缓存”还会加 `"free_memory": true`。检查卸载问题时先测前者，否则可能掩盖模型跟踪丢失。
- 中性文本按照原生 Flux2 约定左侧补零到至少 512 token；不带自定义 transformer 补丁时，CFG 重放只计算正向条件块。存在自定义 transformer 补丁时保留整包重放，避免猜测补丁的 batch 语义。
- `source_latent` / 固定基准 / 前次预测 / 捕获差分在 CPU 上存储和归约，避免每次将基准上传 GPU；这只是小量临时分配优化，不是大幅降低模型权重显存的手段。
- 修复前后热图可能因中性文本填充和 CPU 归约略有变化，旧阈值请重新预览；采样输出仍由原生节点决定。
- 区分采样峰值、执行结束的模型缓存、普通卸载后的残留三个时点。中性配对仍有额外前向成本，不承诺峰值等同原生。
