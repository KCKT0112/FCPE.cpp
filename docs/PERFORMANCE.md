# FCPE 加速研究与性能记录

2026-09-06，第二轮优化，Windows x64 实测。**11 秒输入的严格 F32 Vulkan 网络中位耗时为 6.362 ms，完整 WAV→F0 为 11.474 ms。** 同轮 PyTorch CUDA 为 6.472 ms，ONNX Runtime CUDA 为 7.703 ms。相对上一轮配置的本次复测 94.198 ms，再加速 **14.81 倍**；原图本次复测 1110.981 ms，相差 **174.62 倍**。

这是完整 F32 操作数与累加、相同整段模型语义下的结果。与 PyTorch CUDA 的约 1.7% 差距处于本机计时波动范围，应理解为达到同一速度档位，不能据此宣称对所有 CUDA 实现具有稳定优势。

## 测量条件

| 项目 | 设置 |
| --- | --- |
| CPU | AMD Ryzen 7 5700X，8 核 16 线程；推理线程 4 |
| GPU | NVIDIA RTX 4060，8 GiB；驱动 620.02 |
| 系统／编译器 | Windows build 29648；MSVC 19.51，Release、AVX2 |
| ggml | v0.19.0，commit `30bf8685ed4eb0a47f2b06229543327749904150` |
| Vulkan | SDK `C:\VulkanSDK\1.4.341.1`；strict F32；F16／cooperative-matrix 路径关闭 |
| Python / NumPy | 3.11.4 / 1.25.1 |
| PyTorch | 2.7.1+cu118；CUDA 11.8，cuDNN 9.1；eager inference mode |
| ONNX / ORT | ONNX 1.18.0、opset 17；onnxruntime-gpu 1.22.0 |
| ORT CUDA DLL | 系统 CUDA 12.8 + PyTorch 安装中的 cuDNN 9，显式加载 |
| 模型 | 指定 torchfcpe 0.0.4 wheel 的 checkpoint；F32 GGUF；同模型导出的 ONNX |
| 数值设置 | TF32 关闭，无 F16 激活、权重量化或音频分段近似 |
| 输入 | batch=1；0.1 / 1 / 3 / 11 秒，11 / 101 / 301 / 1101 帧 |
| 计时 | 首次调用另记；预热 3 次，再测 20 次；中位数、线性插值 P95 |

输入沿用上一轮的文件：JFK 语音从 1 秒处截取，不足长度时重复音频。所有引擎使用完全相同的 host F32 Mel，返回 host F32 概率。计时包含 GPU 上传、计算、下载、同步；ggml 还包括 API 有限值检查和输出分配。完整语音精度测试使用原 JFK 全长文件，与该 benchmark 截取方式不同。

模型／session 复用。ggml 每个长度独立进程；PyTorch／ORT 每个引擎独立进程，依次测四种长度。ggml CPU 也使用同一 Vulkan 构建的 `--backend cpu`。PyTorch inter-op=1；ORT intra-op=4、inter-op=1、顺序执行、全部图优化。PyTorch CUDA 开启 cuDNN 算法搜索，成本计入首次调用；ORT 使用 HEURISTIC 搜索。

全部 11 个引擎按顺序重新测量，没有将探索扫描的最佳样本混入正式报告，没有同时运行本任务的构建、测试或其他 benchmark。profiler 独立运行。**44 个引擎／长度组合全部通过概率误差 ≤1e-5、F0 误差 ≤0.02 Hz、UV 差异为零的检查。** CPU 和 GPU 的频率、桌面负载仍有波动；20 次样本不是硬实时或跨设备保证。

## 同轮实测

下表为网络/API 中位耗时，单位 ms；不含 Mel 前端、解码和文件 I/O。

| 引擎 | 0.1 秒 | 1 秒 | 3 秒 | 11 秒 |
| --- | ---: | ---: | ---: | ---: |
| ggml CPU，原 im2col 图 | 5.705 | 54.294 | 157.966 | 577.831 |
| ggml CPU，直接深度卷积 | 3.060 | 19.811 | 58.774 | 222.404 |
| Vulkan，原图／上游严格 F32／host-visible 显存 | 19.649 | 95.739 | 326.126 | 1110.981 |
| Vulkan，直接深度卷积／上游严格 F32／host-visible 显存 | 15.026 | 49.417 | 166.772 | 618.063 |
| Vulkan，上一轮图／手写 tile／host-visible 显存 | 9.712 | 16.566 | 34.175 | 94.198 |
| Vulkan，上一轮图／上游严格 F32／device-local 优先 | 1.114 | 3.462 | 3.062 | 7.314 |
| Vulkan，当前全部优化 | 0.936 | 1.406 | 2.763 | 6.362 |
| PyTorch CPU | 24.199 | 29.490 | 65.404 | 206.740 |
| PyTorch CUDA | 3.280 | 4.129 | 3.855 | 6.472 |
| ONNX Runtime CPU | 3.675 | 13.301 | 43.121 | 136.578 |
| ONNX Runtime CUDA | 1.274 | 1.679 | 3.163 | 7.703 |

| 引擎，11 秒输入 | 中位数 ms | P95 ms |
| --- | ---: | ---: |
| ggml CPU，原 im2col 图 | 577.831 | 607.944 |
| ggml CPU，直接深度卷积 | 222.404 | 260.822 |
| Vulkan，原图／上游严格 F32／host-visible 显存 | 1110.981 | 1226.722 |
| Vulkan，直接深度卷积／上游严格 F32／host-visible 显存 | 618.063 | 685.634 |
| Vulkan，上一轮图／手写 tile／host-visible 显存 | 94.198 | 119.480 |
| Vulkan，上一轮图／上游严格 F32／device-local 优先 | 7.314 | 7.592 |
| Vulkan，当前全部优化 | 6.362 | 6.757 |
| PyTorch CPU | 206.740 | 228.326 |
| PyTorch CUDA | 6.472 | 7.026 |
| ONNX Runtime CPU | 136.578 | 178.601 |
| ONNX Runtime CUDA | 7.703 | 8.036 |

当前 11 秒网络实时因子为 0.0005784，约 1729 倍实时。当前 Vulkan 的短输入也比本机 ggml CPU 更快。上游默认 tile 在 101 帧的表现较差，因此该旧配置出现 1 秒比 3 秒还慢的现象；本轮的矩阵参数选择改善了这个长度。

第一轮历史数据保持在 [原归档](benchmarks/2026-09-06-windows/benchmark.json)：当时最终 Vulkan 为 99.331 ms，CPU 为 227.175 ms，PyTorch CPU/CUDA 为 142.820/6.598 ms，ORT CPU/CUDA 为 115.955/7.905 ms。相对历史 99.331 ms，本轮约快 15.61 倍。两轮 CPU 参考耗时有明显波动，不能把跨轮差值当作代码收益；本文主表采用同轮重测结果。

全部逐次样本、P95、准确性、输入与二进制／DLL 哈希：[benchmark.json](benchmarks/2026-09-06-windows-extreme/benchmark.json)。[all-results.md](benchmarks/2026-09-06-windows-extreme/all-results.md) 列出所有 44 组分位数。代码哈希和检查记录：[provenance.json](benchmarks/2026-09-06-windows-extreme/provenance.json)。

## 完整音频、首次调用和内存

`fcpe-bench --wav` 现在实际重复调用 `Model::infer()`，测量已读入的 WAV 样本→Mel→网络→F0；文件读取和模型加载在循环外。它不是把三个独立阶段的均值相加。PyTorch／ORT 主表仍比较相同 Mel 的网络/API 耗时；没有将其与 ggml 的完整音频时间混比。

| 输入时长 | 网络中位 ms | Mel 前端均值 ms | WAV→F0 中位 ms | WAV→F0 P95 ms |
| --- | ---: | ---: | ---: | ---: |
| 0.1 秒 | 0.936 | 0.206 | 1.234 | 1.908 |
| 1 秒 | 1.406 | 0.870 | 2.293 | 2.825 |
| 3 秒 | 2.763 | 1.483 | 4.453 | 4.888 |
| 11 秒 | 6.362 | 4.293 | 11.474 | 11.833 |

11 秒前端正式均值为 4.293 ms，相对历史 56.464 ms 约快 13.15 倍。探索时同一最终代码曾测得 2.889 ms，保留在 tuning 归档，不替换主表的正式值。F0 解码独立均值为 0.392 ms。阶段计时与整条链路会受到频率、分配和缓存状态影响，不能期待分位数可直接相加。

当前 Vulkan 的 11 秒模型加载为 337.927 ms，首次网络调用为 540.469 ms；首次调用含构图、分配和驱动 pipeline 创建。驱动／磁盘缓存未清空，这不是整机冷启动测量。常驻 `Model` 才能获得表中的预热性能。

| 11 秒，scheduler 计算缓冲区 | 原图 | 上一轮直接卷积图 | 当前 |
| --- | ---: | ---: | ---: |
| CPU | 139.775 MiB | 21.504 MiB | 21.504 MiB |
| Vulkan | 140.313 MiB | 22.042 MiB | 17.741 MiB |

当前 Vulkan 有 162 个图节点，其中 110 个计算节点分配在 GPU，CPU 计算节点为 0，调度分区为 1。融合后实际派发数量更少：本次 profiler 最后一次调用列出 89 个操作组，其中 22 个融合组；图节点数不能当作 shader dispatch 次数。上述内存不包括约 41.56 MiB 权重、驱动／pipeline、宿主输出、staging 和前端 scratch，也不是整个进程或显卡总占用。

ORT 独立 profiler 仍确认 **81 个 CUDA 计算节点、0 个 CPU 计算节点**，见 [ort-placement.json](benchmarks/2026-09-06-windows-extreme/ort-placement.json)。这些带 instrumentation 的耗时不参与主表。

## 主要突破：显存类型

这台显卡上最大的瓶颈来自 ggml Vulkan 的 host-visible device-local 分配偏好。保持旧图和旧手写矩阵内核不变，独立 3 次预热＋20 次采样的对照结果为：

| 配置，11 秒网络 | 中位 ms |
| --- | ---: |
| 旧图＋旧 tile＋host-visible device-local | 91.652 |
| 旧图＋旧 tile＋device-local 优先 | 10.716 |
| 旧图＋上游严格 F32＋device-local 优先 | 6.924 |

仅显存分配方式约带来 8.55 倍收益。它也改变了内核选择的结论：显存瓶颈消除后，上游 F32 shader 比上一轮手写 tile 更快。因此当前默认是在上游 shader 上调参数，保留旧内核用于对照。

分配日志给出实际属性：

| 用途 | memory type | heap | property flags |
| --- | ---: | ---: | --- |
| 原先权重／计算缓冲区 | 4 | 0 | 7：DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT |
| 当前权重／计算缓冲区 | 1 | 0 | 1：DEVICE_LOCAL |
| 上传／下载 staging | 3 | 1 | 14：HOST_VISIBLE + HOST_COHERENT + HOST_CACHED |

**两种权重／计算分配都在 GPU heap 0，并非从系统 RAM 搬回显存。** 观察与 host-visible/ReBAR 路径有关，但还没有证据把底层原因确定为某个驱动缓存机制。日志与所有 A/B 样本在 [memory-ablation.json](benchmarks/2026-09-06-windows-extreme/memory-ablation.json)、[device-local-memory.log](benchmarks/2026-09-06-windows-extreme/device-local-memory.log) 和 [host-visible-memory.log](benchmarks/2026-09-06-windows-extreme/host-visible-memory.log)。单次带日志数据只用来确认类型。

实现复用 ggml 的 device-local 分配路径；未承诺所有驱动都选择相同的 memory type。UMA 分支仍由上游先处理，保留统一内存行为。`FCPE_VK_HOST_VISIBLE=1` 恢复旧偏好；上游 `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` 仍优先禁用它。此策略仅在本机 RTX 4060 实测，应在其他设备做同样 A/B。

## 算子、融合与矩阵调优

深度卷积沿用上一轮成果：把输入表示成 `[T,1,C]`，使用已有 `CONV_2D_DW`，避免 kernel=31 的 im2col 展开；设备不支持时保留旧路径。整段 GroupNorm 语义不变。

本轮新增 [fcpe-fused.comp](../src/vulkan/fcpe-fused.comp) 和 [融合识别／派发](../src/vulkan/fcpe-fusion.inc)，处理六类严格 F32 子图：LayerNorm＋仿射、乘法＋bias、sigmoid-GLU、bias＋sigmoid、bias＋SiLU、bias＋leaky-ReLU。FCPE 的 GLU 是 `a * sigmoid(b)`，没有替换成 SWIGLU。Vulkan 图直接读取带行 stride 的两半切片，去掉六层中的 12 个 CONT；其他后端保留原连续路径。

融合复用 ggml 的消费者、别名和依赖检查，保留被外部使用的中间结果，拒绝危险的输入／输出重叠。F32 二维行连续输入可带行 stride；输出连续。归一化归约用 512 线程，shader 的仿射中间值使用 `precise` 避免多余 FMA 合并。设备工作组能力不足时不创建／选用融合 pipeline。180 组独立 double 参考用例通过，最大误差 3.013e-6；包括显式保留中间输出、非对齐 GLU 切片和非整组长度。

**不需要新增 ggml 算子枚举。** 本轮增加的是现有操作的后端融合 kernel 与识别规则；缺失的 sigmoid-GLU 语义通过精确子图识别解决。实际图排序／别名限制下，并非每一个可想象的 bias／SiLU／残差组合都已融合：最终 profiler 仍有 20 个 ADD 和 6 个独立 SiLU 操作组。

矩阵乘法当前使用上游 `matmul_f32_f32_fp32` shader，扫描六组 warp/register tile 参数，只改变 specialization。dense F32 shader 的 BK 固定为 32。默认仅对 NVIDIA、device-local 路径、subgroup=32 且满足工作组／共享内存／buffer 约束的单批连续 F32 矩阵启用：

| N（该模型为帧数） | 默认路径 |
| --- | --- |
| N ≤ 32 | 上游默认，包括单帧 matvec |
| 32 < N ≤ 128 | variant 3：128 线程，64×64 输出块 |
| N > 128 | variant 1：128 线程，64×128 输出块 |

要求 K 为 4 的倍数；其他形状／类型／硬件沿用上游。没有 Tensor Core、TF32 或 F16 操作数。六组扫描及三轮交替布局复测均归档于 [tuning.json](benchmarks/2026-09-06-windows-extreme/tuning.json)，属于探索数据。CWHN 深度卷积布局需要额外权重重排，复测没有稳定收益，因此保留 WHCN；单次提交整个图的试验也没有稳定收益，相关试验补丁已移除。

所有后端改动通过 `cmake/VulkanPrecision.cmake`、`VulkanMatmul.cmake` 和 `VulkanFusion.cmake` 生成构建目录中的 ggml 源文件副本；替换锚点不匹配会在配置时报错。依赖 checkout 未改动。GLSL 在构建时用 glslc 编译并嵌入 DLL；运行时无需 shader 编译器或 Python。这些开关会影响同一构建中共享该 Vulkan 后端的其他模型。

## Mel 前端

CPU 前端现在只累加每个三角 Mel 滤波器的非零支撑区间，保留原有 F32 累加顺序；启用线程安全的 pocketfft plan 缓存；按最多 64 帧批量 FFT，让 pocketfft 跨帧向量化；按配置线程数并行独立帧，每个 worker 至少约 64 帧，小输入只用一个 worker。每个 worker 的 FFT scratch 有界。

这里只并行独立的特征帧，没有把网络切成块；跨全时间轴的 GroupNorm 仍在完整计算图中执行。F32 CPU／Vulkan 的 14 组对照全部通过：线性 Mel 最大误差 2.385e-6，完整 F0 最大误差 0.001313 Hz，UV 差异为零。F16 存储仍有已知的第 885 帧阈值差异，详见 [VALIDATION.md](VALIDATION.md)。

## 开关和复现

构建选项默认均 ON，仅在 `FCPE_VULKAN=ON` 时适用：

| CMake 选项 | 含义 |
| --- | --- |
| `FCPE_VULKAN_STRICT_F32` | 保持 F32 操作数；同时应用显存策略。关闭后其余本项目 Vulkan 补丁不生效，不能保证当前精度 |
| `FCPE_VULKAN_TILED_F32` | 编译上游 F32 tile 调优及旧手写 kernel；要求 strict F32 |
| `FCPE_VULKAN_FUSION` | 编译六类融合，与 tile 选项独立；要求 strict F32 |

下列布尔型运行时开关以“变量存在”为开启，写 `0` 也算开启；恢复默认应删除变量。`FCPE_VK_MM_VARIANT` 例外，它读取具体值：

| 开关 | 用途 |
| --- | --- |
| `FCPE_VK_HOST_VISIBLE=1` | 恢复旧显存偏好，同时按旧策略优先使用手写 tile |
| `FCPE_VK_TILED_F32=1` | 强制选择旧手写 tile（仍检查形状／设备） |
| `FCPE_VK_DISABLE_TILED_F32=1` | 禁用旧 tile 和自动调优，使用上游选择；显式 variant 覆盖仍可生效 |
| `FCPE_VK_MM_VARIANT=0..5` | 显式扫描上游 F32 tile；`upstream` 禁止自动选择，未设置旧 tile 时用上游 |
| `FCPE_VK_DISABLE_FCPE_FUSION=1` | 禁用本项目六类融合，保留 ggml 原有融合 |
| `FCPE_VK_LEGACY_GRAPH=1` | 恢复 GLU 连续拷贝图；不改变深度卷积选择 |
| `FCPE_VK_LOG_MEMORY_TYPE=1` | 记录实际显存 type／heap／flags |
| `--reference-graph` | 恢复原 im2col 深度卷积图；单独使用不能恢复所有旧后端设置 |

先按 README 下载、解包、转换指定 wheel 并安装 benchmark 依赖：

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.341.1'
$env:FCPE_BUILD_DIR = 'build-vulkan'
.\cmake\build-msvc.cmd -DFCPE_VULKAN=ON
Remove-Item Env:FCPE_BUILD_DIR
python scripts/benchmark.py --prepare --wav validation/jfk.wav --output-dir validation/benchmark-extreme
python scripts/benchmark.py --output-dir validation/benchmark-extreme
# 仅汇总、重验已有数据，不重测：
python scripts/benchmark.py --summarize --output-dir validation/benchmark-extreme
# 独立 profiler，不覆盖正式样本：
python scripts/benchmark.py --worker ort_cuda --profile-ort --warmup 0 --repeats 1 --output-dir validation/benchmark-extreme
.\build-vulkan\bin\fcpe-fusion-tests.exe Vulkan0
```

默认脚本自动清理各 native worker 的 FCPE_VK／GGML_VK 环境变量，并为 `reference/direct/previous/local/current` 分别设置可复现配置。没有 CUDA 参考环境时用 `--engines` 选择可用引擎。ORT GPU 包可按 `scripts/requirements-benchmark.txt` 安装，本机通过项目内 `.ort-deps` 优先加载；`--cuda-dir` 指定 CUDA 12 路径。cu118 PyTorch 与 ORT CUDA 12 共存会有 preload 警告；实际 provider 初始化、节点放置及精度都已核验，未静默回退 CPU。

同一图／内核的显存对照可将归档 [diagnostic-recipe.py](benchmarks/2026-09-06-windows-extreme/diagnostic-recipe.py) 复制到仓库 `build/run_extreme_diagnostics.py` 后执行；它使用刚生成的 benchmark 输入，结果写入单独 diagnostics 目录。不要与正式测速同时运行。

## 还可以继续优化什么

最终独立 profiler 的 11 秒调用约 4.842 ms GPU 操作组时间，其中矩阵乘法约 3.188 ms、深度卷积约 0.573 ms、CONT 约 0.153 ms；该次带 instrumentation 的 host API 为 6.672 ms。两种计时边界不同，不能简单把差值全部归因于某一种开销。日志见 [final-profile.log](benchmarks/2026-09-06-windows-extreme/final-profile.log)。

下一步的可测方向是矩阵乘法 epilogue 融合 bias／激活、深度卷积＋SiLU、减少剩余布局拷贝，以及更低开销的前端任务复用／GPU FFT。CPU 网络仍慢于 ORT CPU，CPU GEMM／线程调度也有空间。短序列更多受提交、同步和框架成本影响；更低精度路径需单独给出误差与阈值边界，不能称为严格 F32。

Metal 在 Windows 下仍未编译或实测，不能套用这里的显存策略或数字；已保留接入、源码精度检查点和 Mac 验证命令，见 [METAL.md](METAL.md)。原生 ggml CUDA 仍受 CUDA 12.8 与 MSVC 19.51 不兼容限制，没有使用 unsupported-compiler override；本文 CUDA 数据仅来自 PyTorch／ORT。当前结果不表示已经达到硬件上限。
