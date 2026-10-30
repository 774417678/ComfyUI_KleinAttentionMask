# 显存卸载问题回归记录（2026-10-31）

## 环境与隔离条件

- Windows，RTX 3080 Ti Laptop 16GB，ComfyUI 0.42.3，Python 3.14.7，PyTorch 2.13.0+cu130。
- 原生 FLUX.2 Klein 9B FP8，编辑一致性 LoRA，Euler 4 步，CFG=1，单图约 1MP，原生 ReferenceLatent 条件。
- 从实际执行历史取得 API 图，只替换采样器；保留模型、LoRA、seed、提示、参考图、尺寸与调度。输出为 VAE 解码预览与掩码预览，不执行另存原图或后处理合成。
- `sparse_neutral`，capture 0..1，probe_steps=1（与问题图一致）；推荐的 probe_steps=2 另做了 CFG=2 功能验证。
- 先等待采样成功、再等待 11~12 秒触发垃圾回收；用普通卸载模型操作，不先清执行缓存。`/free` 是异步标志，发送后额外等 5 秒才采读数。

## 1. 主缺陷：普通卸载后约六成残留

| 场景 | ComfyUI 核心统计，普通卸载后 MiB |
|---|---:|
| 修复前原生 | 1230.75 |
| 修复前自遮罩 | 9902.80 |
| 修复后原生 | 1230.75 |
| 修复后自遮罩 | 1230.75 |

旧版再清执行缓存也能回落，因此只测试 `unload_models + free_memory` 会掩盖缺陷。根因是双层临时 ModelPatcher 与采集器绑定方法形成循环引用，GC 后 ComfyUI LoadedModel 的模型/父 clone 弱引用同时失效；底层模型仍在缓存中，is_dead 条目却不能进入正常卸载路径。

修复不改核心：单层 clone、独立 options、上下文 finally 显式断开 wrapper 与 recorder，并清理私有 guider 及 CPU 采集状态。4 种接口都走这一生命周期。

## 2. 对齐前端 NVIDIA NVML 统计

| 场景 | 全流程采样期间观察到的最大占用 | 结束并等待12秒 | 普通卸载后 |
|---|---:|---:|---:|
| 原生 | 93.34% | 76.62% | 14.91% |
| 修复版自遮罩 | 93.25% | 75.05% | 13.54% |

NVML 包含桌面和其他程序，不能与上表 ComfyUI 核心统计直接混比。此前这轮有用户任务交错，只作为显示口径参考，不作为严格峰值结论。

## 3. 单独隔离采样器阶段（空闲窗口对照）

通过 ComfyUI WebSocket `executing` 事件标记节点，每约250ms同时采集 NVML 与核心统计，排除 CLIP/VAE 期间峰值。

| 场景 | sampler 节点峰值（NVML） | sampler 峰值 MiB | 普通卸载后 |
|---|---:|---:|---:|
| 原生 | 94.84% | 15539.00 | 13.99% |
| 修复版自遮罩 | 94.04% | 15407.31 | 14.13% |

这组只对应上述同图配置，采样频率也不是硬件级精确峰值。不能据此承诺所有工作流降至75%，亦不能否定不同输入图/缓存状态下的高峰值报告。

## 4. 其他回归与行为变化

- 11 项 CPU 生命周期测试通过：成功/异常/取消均解除 wrapper，LoadedModel 追踪返回上游，原 guider options 不变，其他三种接口的异常清理，CFG 正负乱序、batch=2、参考条件切片。
- 真实 CFG=2 sparse_neutral 推理成功，普通卸载回落约14.3% NVML；其余三种接口此次未重新跑 GPU 全流程，不据 CPU 测试宣称全部 GPU 回归完成。
- 固定 seed 原生/修复自遮罩/修复 source_latent 的解码编辑图逐像素一致。
- 中性文本补齐到至少512 token以遵守原生 Flux2 extra_conds；热图可能不同于旧版。CPU/GPU RMS 归约顺序也可能产生末位差异，旧阈值应重新预览。
- CFG中性重放在普通条件下只计算正向条件块；自定义 transformer patch 存在时保留整包重放，不猜测其batch语义。
- 尚未运行 GPU 强制 OOM / 取消恢复测试，生命周期错误路径以 CPU 测试覆盖。不得以普通推理成功代替 OOM 恢复保证。

## 复跑 CPU 测试

在本项目根目录运行，使用 ComfyUI 对应 Python；独立 checkout 用环境变量指定兼容核心：

```powershell
$env:COMFYUI_ROOT = "<ComfyUI目录>"
& "<ComfyUI Python>" tests/test_memory_lifecycle.py
```

没有额外测试依赖，仅用 Python 标准库和现有 PyTorch。测试从本地核心 AST 提取真实 LoadedModel 类，不加载 GPU 模型。更新插件后重启 ComfyUI；不需要修改已有工作流。
