# ComfyUI_KleinAttentionMask

从 FLUX.2 [klein] 9B **模型内部预测**中估计图像编辑的目标区域，输出 MASK。不需要 SAM、GroundingDINO 或任何外部分割模型，也不需要额外填写"要分割什么物体"——只靠编辑指令本身。

![overview](images/demo_overview.png)

> 原图 → 模型内部自动掩码 → 编辑结果（头发改为蓝色）。全程未使用任何分割提示词。

---

## 许可证 / License

本项目采用 **MIT License**，详见 [LICENSE](LICENSE)。

第三方作品与本项目的关系见下方[技术参考与致谢](#技术参考与致谢)。引用请注明出处。

---

## 它解决什么问题

指令式图像编辑模型（Qwen-Image-Edit、Step1X-Edit、FLUX Kontext、Klein 等）本身**不输出"我要改哪里"**。批量跑图时你只有两个选择：

1. **手动画掩码** —— 每张图都要画，批量场景不可行；
2. **外部分割模型**（SAM / GroundingDINO / Florence2 + 提示词）—— 需要为每类编辑目标写提示词，遇到"新增物体""换背景""改材质"就失效，而且分割结果和编辑模型真实意图是两套系统。

本插件走第三条路：**直接问模型自己**。在去噪过程中截取模型的内部预测，量化"这条编辑指令让模型在哪些位置产生了不同的预测"。

---

## 快速开始

### 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/774417678/ComfyUI_KleinAttentionMask.git
```

无第三方 Python 依赖，只用 ComfyUI 自带 PyTorch。重启 ComfyUI 即可。

### 推荐配置

在节点面板搜索 `Klein`，接线为 **采样器 → 影响图 → 外部遮罩** 两个节点：

```
CLIP → Klein 中性条件准备（缓存） ─┐
                                   ↓
SamplerCustomAdvanced(掩码版) → sampling_record → KleinInfluenceMap → influence_heatmap
                                                                     → ThresholdMask(0.20)
                                                                     → GrowMask(2)
                                                                     → FeatherMask(4)
```

`KleinInfluenceMap` 按下面设置：

| 参数 | 推荐值 |
|---|---|
| `common_influence` | 开 |
| `source_reference` | `auto` |

采样器上设 `mode`（局部编辑用 `source_latent`；多物体改不动再换 `fixed_neutral`）、`capture_start=0`、`capture_end=-1`。

**`source_latent` + 共同影响是零额外前向的组合**，采样耗时与原生一致；`fixed_neutral` 定位更稳，只多一次中性前向（第 0 次调用）。

可直接导入 [`workflows/03_example_self_mask.json`](workflows/03_example_self_mask.json)（即原先的「klein自遮罩示例工作流」）。该示例：

- `LoadImage` 指向 `example_input.png`，请先把 [`images/example_input.png`](images/example_input.png) 复制到 ComfyUI 的 `input/` 目录；
- 模型与 LoRA 用的是作者实际使用的文件名（`klein9b\flux-2-klein-9b-kv-fp8.safetensors`、`klein9b\编辑一致性f2k_9B_lcs_consist_20260415.safetensors`，LoRA **已启用**，强度 1）；没有对应文件时改成自己的权重，或把 LoRA 右键旁路；
- `Image Comparer` 里保留了作者的三联对比快照，这些只是展示用的临时预览，重新运行后会被自己的结果覆盖；
- 目标分辨率由工作流内的 `GetImageSize` + 数学节点按原图比例自动计算（默认 1024² 面积预算），不需要手填尺寸；
- 采样器为 `fixed_neutral`、窗口 `0–1`、`probe_steps=1`，影响图开启共同影响、`source_reference=auto`；
- 遮罩链为 `ThresholdMask → GrowMask → 高斯模糊 → ImageCompositeMasked`，与编辑结果合成后三方对比。

---

## 技术核心

这一节说明它到底在做什么，以及为什么这样做。

### 1. 切入点：ComfyUI 官方的 `model_function_wrapper`

ComfyUI 的 `ModelPatcher` 提供 `set_model_unet_function_wrapper()`，允许在每次 `apply_model` 调用前后插入逻辑。本插件通过这个官方扩展点挂载采集器，**不修改 ComfyUI 核心代码**，也不替换采样器、噪声、sigma 调度——原采样流程由 ComfyUI 原生节点原样执行，采集只是旁路观察。

### 2. 关键：比较「内部去噪预测」，不是「最终输出图像」

每个去噪步，模型输出的其实是对**干净 latent** 的预测（x0-prediction）。本插件的核心操作是：

```
同一份带噪 latent、同一个噪声等级 σ
    ├─ 用「编辑指令」跑一次  → 预测 A
    └─ 用「保持原图不变」跑一次 → 预测 B
差值 = RMS(A - B) 在通道维上聚合 → 空间响应图
```

**为什么不直接对最终两张 RGB 图求差？** 因为 VAE 编解码误差、像素级噪声、JPEG 压缩伪影的量级远大于真实编辑区域，差异会被彻底淹没。而在 latent 预测层面比较，信号干净得多，且天然对齐模型"打算改哪里"。

这个"用同条件下的预测差异定位编辑区域"的思路，沿袭自 [DiffEdit](https://arxiv.org/abs/2210.11427) 与 [Watch Your Steps](https://arxiv.org/abs/2308.08947)（见致谢）。

### 3. 为什么强调「同状态」（same-state）

模型预测同时取决于 **latent 状态**和**噪声等级**。如果把第 0 步的中性预测固定下来，拿去和第 3 步的编辑预测比，差值里会混入"模型逐渐看清图像"这个正常去噪过程的变化——那是去噪，不是编辑指令的影响。

所以 `sparse_neutral` / `dense_neutral` 严格在**同一 latent、同一 σ** 上配对运行。这也是 `fixed_neutral` 模式被标记为"含跨状态差异"的原因：它省掉重复计算，但代价是语义不纯。

### 4. latent 坐标对齐（容易踩的坑）

ComfyUI 的 Flux2 latent 是 **128 通道 packed 表示，空间分辨率是像素的 1/16**。更关键的是：VAE 直接 encode 出来的 latent 和模型内部使用的 latent **不在同一坐标系**——模型内部会做 batch-norm 风格的缩放。

所以原图 latent 必须先经过 `model.model.process_latent_in()` 转换，才能和模型预测比较。跳过这一步，差异图会被 VAE 的 scale/bias 主导，得到毫无意义的掩码。

同理， latent 空间尺寸是 1/16 而非 1/8，掩码上采样必须 ×16。

### 5. 为什么不是直接用注意力

实测中，朴素的注意力总量指标在这个任务上不可靠：

| 指标 | 实测表现 |
|---|---|
| 图像 token 对**文本** token 的注意力总量 | 偏向背景空白区 |
| 图像 token 对**参考图** token 的注意力总量 | 圈出整个人，漏掉部分目标衣物 |

"哪里注意力高"和"编辑指令影响了哪里"不是一回事。指令影响差分直接测量后者，实测对"衣服换色""头发换色"都能正确迁移热区。本仓库保留了两个注意力探针节点（`KleinAttentionMaskProbe` / `KleinAttentionMaskExtract`），但**仅作诊断用途，不建议用于批量**（它们使用全局状态存储，且上述指标不稳健）。

### 6. 取 CFG 之前的原始正向预测

CFG 混合后的输出是 guidance 加权结果，不是模型对正向条件的纯预测。采集器从 `cond_or_uncond` flags 中定位正向分支（flag `0`），在 CFG 混合**之前**取原始预测。负向分支绝不能被误当中性基准——两者语义完全不同。

---

## 节点清单

| 节点 | 采样接口 | 额外前向（默认配置） |
|---|---|---:|
| `KleinMaskSamplerCustomAdvanced` | **NOISE + GUIDER + SAMPLER + SIGMAS + LATENT** | 0（`record_neutral` 开则每步 +1） |
| `KleinMaskSamplerCustom` | MODEL / add_noise / noise_seed / cfg / positive / negative / SAMPLER / SIGMAS / LATENT | 同上 |
| `KleinMaskKSampler` | 标准 KSampler（含 `denoise`） | 同上 |
| `KleinMaskKSamplerAdvanced` | KSamplerAdvanced（含加噪开关、起止步、保留剩余噪声） | 同上 |
| **`KleinInfluenceMap`** | 从采样记录计算影响图（纯 CPU 后处理） | 0 |
| `KleinInternalEditMaskSampler` | 旧版单节点（内置 euler/simple/CFG=1） | 2 |
| `KleinAttentionMaskProbe` / `KleinAttentionMaskExtract` | 注意力诊断（**遗留，不推荐**） | 0 |

四个采样器变体的输出前缀为原生输出，之后追加 2 个输出：

```
[原生输出...] , sampling_record , diagnostics
```

`sampling_record` 接到 **`KleinInfluenceMap`**，由它产生三个输出：

```
influence_heatmap , diagnostics , comparison_heatmaps
```

`influence_heatmap` 是连续 0–1 的 MASK 影响图，**不做**二值化、扩张或羽化（阈值用外部 `ThresholdMask` / `GrowMask` / `FeatherMask`）。

### 为什么 mode 在采样器上

中性预测只在模型处于那个具体的带噪状态时才存在，所以**记录什么必须在采样前决定**。因此 `mode`、`capture_start` / `capture_end`、`probe_steps` 留在采样器上，每种模式只记它需要的部分，额外前向与旧版单节点完全一致：

| mode | 额外前向（4 步 Euler） |
|---|---:|
| `source_latent` / `adjacent_prediction` | 0 |
| `fixed_neutral` | 1（仅第 0 次调用） |
| `sparse_neutral` | `probe_steps`（窗口内） |
| `dense_neutral` / `compare_all` | 窗口内的调用数 |

记录完成后会按上游输入的身份缓存。**覆盖范围向下兼容**：跑过 `dense_neutral` / `compare_all` 之后，把 mode 改成 `fixed_neutral` 或 `source_latent` 不会重跑扩散，直接复用已记录的数据；采样器节点仍会重新执行，但内部跳过采样（实测约 1 秒完成）。只有把 mode 收窄到“记录里没有的中性步骤”时才需要重采一次。

上游任何影响采样的输入（模型、噪声、sigmas、latent、条件）发生变化，缓存键就不同，会正常重跑。

`KleinInfluenceMap` 上只剩两个纯后处理参数：`common_influence` 与 `source_reference`，改它们永远不触发采样。

> 2026-10-31 起 `mode` 等参数从采样器移到影响图节点，并移除了旧的 `edit_mask` / `comparison_masks` 输出与 `threshold` / `grow` / `feather` 参数。旧工作流需要按新接线重连（示例工作流已迁移）；旧版本仍可从 Releases 页面按 tag 取得。

---

## 参数详解

### `mode` —— 五种对比基准（在采样器上）

| 模式 | 比较对象 | 额外前向 | 含义与适用 |
|---|---|---:|---|
| **`source_latent`** | 每步干净预测 ↔ 原图 latent（默认自动取编辑条件里的参考图，见 `source_reference`） | 0 | "模型预测会把原图改成什么样"。最快，包含模型复原误差；配合 `common_influence` 取多步共同区域后背景很干净，**绝大多数情况够用** |
| **`fixed_neutral`** | 每步预测 ↔ 第 0 步那次中性预测 | 1 | 目前定位最稳，但含跨噪声状态差异，需一次中性前向 |
| **`adjacent_prediction`** | 第 N 步预测 ↔ 第 N-1 步预测 | 0 | "哪里还在变"。已稳定下来的编辑信号会消失，后期纹理抖动反而明显。适合观察采样动态 |
| **`sparse_neutral`** ⭐ | 选定调用的编辑预测 ↔ 同状态中性预测 | `probe_steps` | **纯净的指令影响**。推荐默认 |
| **`dense_neutral`** | 窗口内每步都配对 | 窗口步数 | 最完整，也最慢 |
| **`compare_all`** | 以上五种同时输出 | 窗口步数 | 一次采样对比全部模式，用于调参 |

`compare_all` 的 `influence_heatmap` 采用 `source_latent`，五组完整影响图从 `comparison_heatmaps` 按 batch 读取，顺序固定为：

```
source_latent → fixed_neutral → adjacent_prediction → sparse_neutral → dense_neutral
```

### 多物体编辑时的模式选择（实测结论）

一条指令同时改多个物体时，模型的注意力会被摊薄，影响图会出现**明显的空间偏移**：主体响应变弱、区域之间互相拉扯，单一阈值下容易把不相干的物体连成一片。

| 情况 | 建议 |
|---|---|
| 单物体、局部换色/换材质 | `source_latent` + 共同影响，零额外开销，干净够用 |
| 多物体，且能接受额外前向 | `fixed_neutral`，目前定位最稳；每步多一次中性前向 |
| 多物体但要求速度 | 先用 `source_latent` 看 `influence_heatmap`，再决定是否值得换模式重采 |

注意：记录向下兼容。跑过 `dense_neutral` / `compare_all` 之后切到 `fixed_neutral` / `source_latent` 不重跑扩散（约 1 秒返回）；反过来从 `sparse_neutral` 收窄到更宽的中性窗口才需要重采一次。所以“多物体先用 source 看、不满意再换 fixed”最省时间。

### 采样器参数（mode 与窗口决定记录内容）

| 参数 | 默认 | 说明 |
|---|---|---|
| `mode` | `source_latent` | 见上表。它决定采样时要记录哪些中性预测 |
| `capture_start` | `0` | 第几次**模型调用**开始参与比较/记录，0 起编号 |
| `capture_end` | `-1` | 到第几次调用为止；`-1` 表示到结束 |
| `probe_steps` | `2` | 仅 `sparse_neutral`：窗口内取几个调用做中性配对 |
| `source_latent` | — | 无参考图条件时的原图基准。**续采的带噪 latent 必须显式提供**，并把影响图的 `source_reference` 设为 `source_latent_input` |
| `neutral_conditioning` | — | 中性模式需要：接 `KleinNeutralConditioning`；旧 `clip` 入口保留兼容但会在采样器内部编码，可能抬高峰值 |

### 影响图参数（纯后处理，改它不触发采样）

| 参数 | 默认 | 说明 |
|---|---|---|
| `common_influence` | 关 | **共同影响**。开：每次采集先各自归一化（背景中位数→0，99.5% 分位→1），再逐点取**最小值**，只保留每次都强的区域。关：平均 |
| `source_reference` | `auto` | 哪份 latent 算"原图"。`auto`：优先用正向条件里第一张 `ReferenceLatent`（采样时自动记录），没有时退回采样器的 `source_latent` 输入或 `latent_image`；`reference_latent`：必须用参考图，没有就报错；`source_latent_input`：始终用 `source_latent` 输入（旧行为） |

### 遮罩改为外部生成

影响图已归一化到约 0–1（每次采集背景中位数→0、99.5% 分位→1），外部阈值是**绝对值，不是"取最亮的百分之几"**，调大选区缩小、调小选区扩大。旧参数对应的核心节点：

| 旧参数 | 外部节点 |
|---|---|
| `threshold` | `ThresholdMask`（value，严格大于；旧版为大于等于，连续影响图上实测像素级一致） |
| `grow` | `GrowMask`（expand） |
| `feather` | `FeatherMask`（四边设同一值） |

示例工作流已按旧参数自动换接成上述链路，下游合成节点保持连接。

> ⚠️ **`capture_start` / `capture_end` 计的是「模型调用序号」，不是 scheduler 步号。** Heun、dpmpp_2m、ER-SDE 等算法会在一个 sigma 区间内多次调用模型，调用数会多于步数。自适应算法请填确定的 `capture_end`。`diagnostics` 输出里有真实的调用序号与 σ 列表，可据此核对。

### 可选输入

| 输入 | 何时需要 |
|---|---|
| `neutral_conditioning` | 采样器 `record_neutral` 打开时接 `KleinNeutralConditioning`；不开中性模式可以不接 |
| `clip` | 旧图兼容入口，仅在没有 `neutral_conditioning` 时用于内部编码，可能抬高峰值 |
| `source_latent` | 无参考图条件时的原图基准，默认退回 `latent_image["samples"]`。**若输入是中途续采的带噪 latent，必须显式提供干净原图 latent**，并把 `source_reference` 设为 `source_latent_input` |

### 局部编辑推荐：对比原图 + 共同影响

影响图上设 `mode=source_latent`、`common_influence=开`、`capture_start=0`、`capture_end=-1`（4 步 Euler 即第 1–4 步），外部 `ThresholdMask(0.20)`。**不需要中性条件，也没有额外前向**，耗时与原生采样相同，采样器的 `record_neutral` 保持关闭。

修复说明：旧版 `source_latent` 在示例工作流里拿 `EmptyFlux2LatentImage`（全零）作"原图"，实际比较的是"预测 vs 空 latent"，并非对比原图。现在 `auto` 会直接使用 `ReferenceLatent` 里已经过 `process_latent_in` 的参考图 latent，与预测处于同一坐标系。`diagnostics` 的 `source_origin` 会显示实际来源（`reference_latent` / `source_latent_input` / `latent_image`）。

实测（2026-10-31，Klein 9B FP8 + LoRA、Euler 4 步、CFG=1、832×1248，"将衣服变成红色"）：影响图经外部 `ThresholdMask(0.2)` 得到的遮罩与离线逐步数据算出的"对比原图·第 1–4 步共同 ≥0.2"掩码 IoU **0.995**，选区 11.2%，对应红色连体衣。改姿势、移动物体这类结构性编辑，模型本身会重绘大面积背景，共同影响如实反映这一点，不能把它当作噪声去掉。

> 与上一版（单节点、采样时直接出图）相比，同一图的遮罩 IoU 为 **0.9963**（差 431 像素 / 约 10 万），影响图最大差 32/255、仅 0.4% 像素差超过 2 级。两个版本各自重复运行都是逐像素一致的，所以这是两条记录路径的微小数值差异，不是阈值或算法改变；需要与旧版完全一致时可按 tag 取回旧版本。

---

## 版本与旧版获取

| 标签 | 内容 |
|---|---|
| `v2.0.1` | 当前版本：`mode` 与采集窗口在采样器上（记录什么必须在采样前决定），影响图节点只做纯后处理；覆盖向下兼容，跑过更全的模式后切便宜模式不再重采样 |
| `v2.0.0` | 曾把 `mode` 放在影响图节点：切换模式很方便，但 `record_neutral` 会对每一步都记录中性预测，4 步采样前向从 5 次变成 8 次，采样时间接近翻倍。不建议使用 |
| `legacy-single-node` | 2.0 之前的单节点版本：采样时直接算影响图/掩码，参数在采样器上，改参数会整节点重跑 |

需要旧版时按标签取回即可，不影响现有安装：

```bash
git clone --branch legacy-single-node https://github.com/774417678/ComfyUI_KleinAttentionMask.git
# 或在已有目录里： git checkout legacy-single-node
```

两个版本的节点结构不兼容（旧工作流的参数在采样器上、输出直接给影响图），混用会缺少应有的输入/输出，请整套切换。

---

## 推荐配置为什么是 `sparse_neutral` + `0` + `1`

去噪早期的步骤（高噪声）承载最强的**语义定位**信号——模型此时还在决定"改哪里"；到后期预测趋于稳定，差异变小且容易混入纹理级噪声。

只在前两步做配对：
- 信号最强的位置已覆盖；
- 额外前向仅 2 次，4 步采样总开销 4→6；
- 实测「头发改蓝」掩码干净集中在头发，无明显溢出。

需要更高精度时再逐步放宽 `capture_end`；批量筛图追求速度时用 `source_latent`（零额外前向）。

---

## 工作流

| 文件 | 用途 |
|---|---|
| [`workflows/01_recommended_sparse_neutral.json`](workflows/01_recommended_sparse_neutral.json) | 推荐配置，五输入采样器 + 掩码/结果/合成三路输出 |
| [`workflows/02_compare_all_modes.json`](workflows/02_compare_all_modes.json) | 一次采样输出五组掩码与热图，用于对比调参 |
| [`workflows/03_example_self_mask.json`](workflows/03_example_self_mask.json) | 完整实战示例（原 `klein自遮罩示例工作流`）：自动分辨率 + 原生 `ReferenceLatent` 条件 + 热图二次阈值 + 模糊合成 + 三联对比 |

### 工作流 03 说明

功能更完整的实战图，与 01 的区别：

- **条件路径**用原生 `CLIPTextEncode → ReferenceLatent`（配 `ConditioningZeroOut` 做负向），不依赖 `PainterFluxImageEdit`；
- **分辨率自适应**：`easy imageSize` + `ComfyMathExpression` 按面积把任意原图缩放到约 1MP；
- **外部遮罩链路**：`influence_heatmap`（第 3 个输出 `out2`）→ `ThresholdMask 0.06` → `GrowMask 15` → 高斯模糊 → `ImageCompositeMasked`。热图是连续值，低阈值 + 大扩张适合边缘柔和的头发/布料；
- `LoraLoaderModelOnly` 挂编辑一致性 LoRA，**已置为旁路模式**：没有该 LoRA 也能跑，需要时右键启用并换成自己的权重；
- 用 rgthree 的 `Set/Get Node` 做连线复用，`Image Comparer` 做前后对比。

**额外依赖插件**（01 / 02 不需要；**03 需要下表全部四项**）：

| 插件 | 用到的节点 |
|---|---|
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | `SetNode` / `GetNode` / `Image Comparer` / `Any Switch` |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | `easy imageSize` |
| [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) | `LayerFilter: GaussianBlur` / `LayerMask: MaskPreview` |
| [ComfyUI-LogicUtils](https://github.com/aria1th/ComfyUI-LogicUtils) | `MaxNode` |

> `SetNode` / `GetNode` / `Image Comparer` / `Any Switch` / `Note` 属**前端虚拟节点**，不会出现在 `/object_info`，查不到是正常的。

该图默认 `sparse_neutral` + `capture_start=0` + `capture_end=1` + `probe_steps=1`。KV 权重（`flux-2-klein-9b-kv-fp8`）与非 KV 权重均实测可用。

01/02 两个工作流的 `LoadImage` 指向 `example_input.png`，使用前请把 [`images/example_input.png`](images/example_input.png) 放进 ComfyUI `input/` 目录，并把 `UNETLoader` 改成你的 Klein 9B 权重。编辑指令写在 `PainterFluxImageEdit` 的 `prompt` 中，示例为「Change the hair to blue. Keep everything else unchanged.」。

掩码的典型用法：`ImageCompositeMasked`，destination = 原图，source = 编辑结果，mask = 外部 `ThresholdMask` / `GrowMask` / `FeatherMask` 处理后的影响图。白色区域采用编辑结果，黑色保留原图。

**注意**：掩码是在采样**之后**输出的，用于合成；它不会反过来约束刚才那次采样。若要让掩码局部二次重绘，请把它接到下一遍 inpaint。

---

## 已验证环境

- ComfyUI 0.42.3（V3 节点 API）
- PyTorch 2.13.0+cu130 / Python 3.14.7
- FLUX.2 [klein] 9B（`moodyDesireMixFlux2_v30_pruned_fp8`，FP8，896×1152）
- 编码器 Qwen3-8B fp8mixed + `flux2-vae`

**已验证**：五种模式、batch=2 张量维度、CFG 正负分支乱序、`noise_mask` 传递、原生采样输出不被修改、四种采样器接口真实跑通、三个发布工作流端到端跑通（含「头发改蓝」示例），KV 与非 KV 权重均可用。

**未验证**：batch > 2、视频/nested latent、DualCFG 引导器、ControlNet 条件下的中性重放、多 GPU、第三方量化执行引擎。

---

## 共同影响与对比原图修复（2026-10-31）

新增 `common_influence` 与 `source_reference` 两个可选输入，追加在已有输入末尾，旧工作流控件顺序不变，默认值保持原行为（`common_influence` 关；`source_reference=auto` 只在存在参考图时改变 `source_latent` 的基准）。四种采样器变体共用同一实现。CPU 回归 22 + 28 项通过；GPU 实测见上文"局部编辑推荐"。

## 显存释放修复（2026-10-31）

修复了 `KleinMaskSamplerCustomAdvanced` 在普通“卸载模型”后仍占用大量显存的问题。旧实现的双层临时 `ModelPatcher` 克隆与绑定采集器形成引用环，垃圾回收后 ComfyUI 的 `LoadedModel` 丢失克隆及其临时父对象的弱引用；底层模型仍在工作流缓存中，因而无法通过正常模型卸载路径释放。

现在只从上游模型克隆一次，并在四种采样器变体正常结束、异常或取消时显式拆除采集包装器和采集数据。保留输入 guider 的独立 options 副本，不修改上游对象，也不通过清空整个执行缓存绕过问题。

同一份原生 Klein 9B FP8 + 一致性 LoRA、Euler 4 步、CFG=1 工作流，仅替换采样器的实测：

| 场景 | ComfyUI 核心统计：普通卸载后 |
|---|---:|
| 原生采样器 | 约 1.20 GiB |
| 修复前自遮罩 | 约 9.67 GiB |
| 修复后自遮罩 | 约 1.20 GiB |

另一次使用与 Crystools 相同的 NVIDIA NVML 全卡统计，修复版自遮罩结束后约 75%，普通卸载后约 13.5%。NVML 包含桌面及其他应用，不能和 ComfyUI 核心的进程侧数字直接混用。这项修复解决的是卸载残留，**不保证采样峰值降到某个百分比**。随后在用户预留的空闲窗口，用 WebSocket 单独标记采样器阶段：同图原生/修复版的 NVML 峰值分别为 **94.84% / 94.04%**，普通卸载后为 **13.99% / 14.13%**；本组未见额外峰值抬升。不同输入图、缓存和其他工作流仍需各自对照。完整条件和验证边界见 [显存回归记录](docs/MEMORY_FIX_ZH.md)。

同时，中性文本现在遵循原生 Flux2 的左侧零填充到至少 512 token 的约定；差分归约移到 CPU，普通条件下的 CFG 中性重放仅计算正向块。中性填充会改变旧版本的热图，CPU/GPU 归约也可能有末位数值差别，建议更新后重新预览掩码。实测修复版原生/自遮罩/`source_latent` 的解码编辑图逐像素一致。成功、异常、取消、条件切片等 11 项 CPU 回归通过，CFG=2 实际推理通过；其他三种接口本次只验证了清理异常路径，未重新完成 GPU 回归。

更新后需要重启 ComfyUI，使旧的运行中类被替换。

## 已知限制

1. **实验性区域估计，不是精确语义分割。** 边界可能漏选或包含周边变化。批量无人值守前请先看 `influence_heatmap`。
2. **仅支持 ComfyUI 原生 Flux2 Klein 9B。** 会显式拒绝：已有 `model_function_wrapper` 的模型、DualCFG / 未知引导器、第三方 cond 执行器、nested/video latent、形状不匹配的 source latent。
3. **中性重放不支持 ControlNet。** 节点会显式报错而非静默出错。
4. **模型克隆只隔离 patch/options，不是独立模型副本。** 不宣称并发安全。
5. **`KleinAttentionMaskProbe` / `KleinAttentionMaskExtract` 为遗留诊断节点**，使用全局状态存储，不适合批量或多个探针同时使用。

---

## 技术参考与致谢

本项目受以下工作启发，并明确声明与其的关系：

| 项目 | 关系 |
|---|---|
| [**DiffEdit**](https://arxiv.org/abs/2210.11427)（arXiv:2210.11427） | "用文本条件下的预测差异自动生成编辑掩码"这一核心思想的方法论源头。本项目将同一思想迁移到 DiT 指令编辑器的 latent 预测层 |
| [**Watch Your Steps**](https://arxiv.org/abs/2308.08947)（ECCV 2024，[代码](https://github.com/SamsungLabs/WatchYourSteps)） | 用「有/无编辑指令的模型预测差异」定位编辑区域。本项目的 same-state 配对设计直接受此启发 |
| [**TAL: Rethinking Where to Edit**](https://arxiv.org/abs/2604.20258)（ACM MM 2026，[代码](https://github.com/Jessie459/TAL)) | 面向 DiT 指令编辑器的编辑定位研究，启发了本项目的动机。**本项目不是 TAL 的复现**——TAL 走"注意力 + 特征质心 + 任务相关掩码构造"路线，本项目走"内部预测差分"路线，两者方法与实现均不同 |
| [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) | 提供 `model_function_wrapper`、`SamplerCustomAdvanced` 等官方扩展点与原生采样实现，本项目全部逻辑为旁路挂载 |
| [**FLUX.2 [klein]**](https://github.com/black-forest-labs/flux2) | Black Forest Labs 的开源权重与推理代码，本项目的目标模型家族 |

未使用上述任何项目的代码；引用仅为学术脉络说明。

---

## 目录结构

```
ComfyUI_KleinAttentionMask/
├── __init__.py                 节点注册
├── sampler_variants.py         四种采样器变体 + 采集器
├── internal_sampler.py         旧版单节点 + 五种模式核心逻辑
├── mask_node.py                旧版注意力探针节点
├── probe.py                    旧版注意力探针采集器
├── docs/
│   ├── USAGE_ZH.md             旧版节点使用说明
│   └── SAMPLER_VARIANTS_ZH.md  采样器变体详细说明
├── workflows/
│   ├── 01_recommended_sparse_neutral.json
│   ├── 02_compare_all_modes.json
│   └── 03_example_self_mask.json
└── images/
    ├── example_input.png       工作流所需输入图（896×1152）
    ├── demo_overview.png       三联演示图
    ├── demo_edit.png           编辑结果
    └── demo_influence.png      内部影响热图
```

示例图片为原创绘制的动漫风格角色，可自由使用。
