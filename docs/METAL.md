# macOS / Metal：严格 F32 与专项优化

2026-09-06，Apple M4 / macOS 27.0 (26A5425a) / Apple Clang 21，ggml v0.19.0 原始 SHA256 校验归档，官方 torchfcpe 0.0.4 wheel，PyTorch / torchaudio 2.10.0。设备名称是 `MTL0`，以 `--list-backends` 的实际输出为准。

第一轮完成 F32 适配；第二轮补充转置、卷积、归约和融合 kernel，逐项测量并删除没有稳定收益的方案。质量门槛、整段 GroupNorm 语义、F32 操作数及参考模型保持一致，未采用半精度激活、TF32、分段近似或阈值放宽。第一轮原始数据保留于 [macOS 初始归档](benchmarks/2026-09-06-macos/README.md)，第二轮见 [专项优化归档](benchmarks/2026-09-06-macos-extreme/README.md)。

## 实现与精度

`FCPE_METAL_STRICT_F32=ON` 默认启用。上游 F32 矩阵路径内部使用 half：M=N=K=64 的独立 double 参考测得最大误差 0.000653370、RMSE 0.000171388；相对两个操作数先舍入为 half 的参考仅 8.68701e-8。修复在已有 SIMD-group 算法中使用 float 操作数和 float 累加，并将共享内存改为 12 KiB；19 组矩阵测试最大误差 1.69602e-5。`GGML_PREC_F32` 本身不能阻止上游舍入。

本轮保留的优化：

| 路径 | 最终实现 | 质量与回退约束 |
| --- | --- | --- |
| CONT 转置／切片拷贝 | 32×32 共享内存转置、连续行拷贝 | F32 二维形状／stride 检查；逐位一致；其余布局走上游 |
| 两层普通卷积的 im2col | 合并时间与通道 tile，复用 3-tap 输入，保持 F32 展开顺序 | kernel=3、stride=1、pad=1 的一维路径；35 组逐位检查 |
| 六层深度卷积 | 通道连续的输入与输出，不再来回转置；共享内存复用相邻帧与 31-tap 权重 | 保持逻辑 WHCN shape，显式 stride；原空间连续图和 kernel 均可恢复 |
| GroupNorm | 小输入单工作组，大输入三阶段整段归约，scratch 随输出缓冲区分配 | 方差使用中心化平方差；精确除法修复常量输入误差；所有阶段显式 barrier |
| sigmoid-GLU | 移除六层的 12 次连续拷贝；融合 stride 视图上的 sigmoid 与乘法 | 消费者与内存重叠检查；保留外部使用的中间量 |
| 线性投影＋bias＋GLU | 在同一 F32 SIMD-group tile 中计算两半通道，直接输出 GLU | 矩阵、视图、偏移、对齐、GPU 能力及别名检查；不能融合时回退 |
| bias 激活 | 合并 bias 与 sigmoid／SiLU／leaky-ReLU | 上游消费者与 hazard 检查；暴露中间输出时不删除它 |

没有新增 ggml 枚举或自定义 CPU 回调。神经网络仍由 ggml 图和 scheduler 执行。构图优化只用于本项目配置的 Metal 后端；CPU、Vulkan 和严格 F32 关闭时保留原构图。`src/metal/kernels.metal`、`dispatch.inc` 及 `cmake/MetalPrecision.cmake` 实现本轮代码，ggml checkout 不修改。构建目录中的副本保留上游许可，shader 嵌入动态库，运行时由系统 Metal 编译。共享此 Metal 后端的其他模型也可能匹配新的 kernel／融合规则。

默认通用矩阵仍采用第一轮 F32 修复后的上游算法。扫描四种分块、标量和向量化加载、矩阵 bias／残差／sigmoid epilogue 后，新通用实现没有稳定胜过它，已删除。隐式普通卷积收益不稳定；深度卷积后处理融合受到分配器别名限制，没有把未实际命中的融合当作收益。向量化广播加法、关闭并发、图优化、共享缓冲区、residency，以及单次提交整个图也未得到稳定的默认收益。探索结果保留，未用最小单次值替换正式样本。

## 质量验证

完整 14 组对照包含 13 组边界／重采样／合成输入和 11 秒 JFK，分别比较相同 Mel 的网络输出与完整 WAV→F0。最终 F32 严格通过，逐帧 UV 一致。所有比较使用原有概率、log-Mel、线性 Mel、F0 及 UV 门槛。

| 最终 F32，14 组最大误差 | 数值 |
| --- | ---: |
| 相同 Mel 概率误差 | 0.000001758 |
| WAV 概率误差 | 0.000411808 |
| WAV F0 误差 | 0.001388550 Hz |
| UV 差异 | 0 |


C++ 共 7 项 CTest，覆盖解码、19 组矩阵、动态长度模型、180 组融合／回退，以及 60 组逐位拷贝、144 组双布局深度卷积、108 组 GroupNorm、32 组投影 GLU、35 组逐位 im2col。包括奇数／尾块尺寸、单帧、常量和低方差输入、批次、暴露中间输出、复用图及确定性。Metal Shader Validation 下也运行同一套测试，相关日志单独保存，耗时不参与测速。

F16 只是存储近似，权重在图中仍扩展到 F32。JFK 第 885 帧跨越 0.006 阈值的问题保留；严格失败和原始 F0 差异如实归档。不能把 F16 解释为清浊判断完全等价或峰值内存减半。

## 测速方法

`benchmark.py` 比较同一 host F32 Mel→host 概率，包含 GPU 传输和同步；4 个 CPU 线程、3 次预热、20 次测量。native 单独测量已读入 WAV 样本→F0，排除读盘与模型加载。PyTorch MPS 是独立的实际 MPS worker，显式同步并拒绝 CPU fallback，没有将 CUDA 名称替换为 MPS。

`benchmark_metal.py` 对 0.1／1／3／11 秒做 3 轮交替顺序消融：每个配置每轮独立预热与 20 次采样，总计 132 组。原二进制对照显式固定其 dylib 搜索路径，并从 dyld 日志核实未加载当前动态库。每组都做严格输出检查。跨轮运行状态会波动，因此同时保留全部样本、轮次中位数和合并分位数。

诊断 `FCPE_METAL_PROFILE=1` 为每个原始算子创建独立 command buffer，读取 GPU 时间戳；该模式关闭融合、改变提交边界，只用于定位单算子瓶颈，不能直接视为生产图的逐算子耗时，也不能混进正式测速。两轮诊断显示，CONT＋im2col 总计从约 13.7 ms 降至约 0.21 ms，GroupNorm 从约 2.2 ms 降至约 0.06 ms；剩余主要算术工作是矩阵和深度卷积。精确日志见归档。

## 本机结果

三轮交替对照，每项合并 60 个样本；单位 ms：

| 输入 | 原二进制 | 当前 | 加速 |
| --- | ---: | ---: | ---: |
| 0.1s | 1.475 | 1.105 | 1.34× |
| 1s | 3.713 | 2.169 | 1.71× |
| 3s | 9.620 | 3.962 | 2.43× |
| 11s | 32.344 | 12.442 | 2.60× |

11 秒当前实际 WAV→F0 合并中位 14.068 ms、P95 17.975 ms。98 个网络计算节点均在 GPU，CPU 回退为 0，1 个调度分区。计算缓冲区由第一轮的 22.042 MiB 降到 17.741 MiB；这是 scheduler 计算缓冲区，不含权重、前端、驱动和整个进程内存。

另一次完整框架对比包含 10 种长度。下表为该轮网络中位耗时，GPU 与 CPU 使用相同 host Mel；其运行状态与消融轮不同，不混用数值。

| 引擎 | 0.01 秒 | 0.1 秒 | 1 秒 | 3 秒 | 11 秒 | 60 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ggml_cpu | 0.759 | 2.117 | 18.150 | 53.575 | 208.414 | 1142.889 |
| ggml_metal | 0.903 | 1.152 | 2.469 | 5.661 | 16.814 | 73.477 |
| torch_cpu | 73.043 | 78.073 | 84.351 | 96.886 | 120.388 | 271.164 |
| torch_mps | 3.765 | 3.953 | 4.374 | 8.012 | 17.691 | 84.889 |
| ort_cpu | 1.071 | 3.320 | 8.371 | 16.353 | 59.843 | 374.429† |

全部 FCPE 原生 CPU／Metal（40 组）和 PyTorch CPU／MPS（20 组）通过门槛；60 秒 Metal 网络 73.477 ms，实际 WAV→F0 81.675 ms。60 秒完整 WAV 对照最大 F0 误差约 0.000305 Hz、UV 差异为零。

† ORT CPU 的 30／60 秒概率最大误差分别为 2.72095e-5／0.000110090，超过既有 1e-5 门槛；F0/UV 检查通过，但该两项整体记 FAIL。总计 68/70 通过，完整框架脚本如实非零退出。失败样本和耗时保留，并未把它们当作同精度性能胜负；复现默认完整列表时也应保留该结果。

当前 Metal 对短输入比本机 MPS 快，11 秒同轮差距较小，60 秒约快 1.16 倍。剩余主要耗时仍是矩阵和深度卷积；没有证据宣称达到硬件理论极限。完整 P95、输入／动态库哈希、失败记录和消融见 [专项归档](benchmarks/2026-09-06-macos-extreme/README.md)。

## 构建与复现

先按 README 下载、校验并转换模型，安装 `scripts/requirements-benchmark.txt`；保持 torch 与 torchaudio 匹配。

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DFCPE_METAL=ON \
  -DFCPE_TEST_MODEL="$PWD/models/fcpe-f32.gguf" -DFCPE_TEST_BACKEND=MTL0
cmake --build build-metal --parallel 4
./build-metal/bin/fcpe-cli --list-backends
ctest --test-dir build-metal --output-on-failure
MTL_SHADER_VALIDATION=1 ctest --test-dir build-metal --output-on-failure
python tests/test_tools.py --cli build-metal/bin/fcpe-cli --model models/fcpe-f32.gguf -v
python scripts/validate.py --cli build-metal/bin/fcpe-cli --model models/fcpe-f32.gguf \
  --backend MTL0 --wav validation/jfk.wav --output-dir validation/macos-extreme/metal-f32
python scripts/validate.py --cli build-metal/bin/fcpe-cli --model models/fcpe-f16.gguf --type f16 \
  --backend MTL0 --wav validation/jfk.wav --output-dir validation/macos-extreme/metal-f16
python scripts/benchmark.py --prepare --wav validation/jfk.wav \
  --seconds 0.01 0.08 0.32 0.64 0.1 1 3 11 30 60 --output-dir validation/macos-extreme/benchmark
python scripts/benchmark.py --bench build-metal/bin/fcpe-bench --output-dir validation/macos-extreme/benchmark
python scripts/benchmark_metal.py --bench build-metal/bin/fcpe-bench \
  --inputs validation/macos-extreme/benchmark --cases 0.1s 1s 3s 11s --output-dir validation/metal-ablation
```

F16 命令应保留 JFK 的已知严格失败。语音来自 whisper.cpp 的 samples/jfk.wav。本机目录为 `build-macos-metal`；`GGML_METAL_EMBED_LIBRARY=ON` 是要求，也是 ggml Metal 的默认值。本机 ccache 缺少 libfmt.11.dylib，使用 `-DGGML_CCACHE=OFF`；OpenMP 未安装。受限沙箱可能无法访问 GPU，本次实机验证在获准访问 Metal 的进程中运行。

布尔环境开关按“存在”启用，写 `0` 也开启；清除变量恢复默认。正式 `benchmark.py` 会清除所有 FCPE_METAL／GGML_METAL 开关，以免把 profiler 或消融误当默认配置；消融请使用 `benchmark_metal.py` 或直接调用 `fcpe-bench`。

| 开关 | 对照用途 |
| --- | --- |
| `FCPE_METAL_LEGACY_COPY` | 恢复上游 CONT kernel |
| `FCPE_METAL_LEGACY_GRAPH` | 恢复 GLU 两半的连续拷贝 |
| `FCPE_METAL_LEGACY_DW_LAYOUT` | 恢复空间连续深度卷积图与转置 |
| `FCPE_METAL_LEGACY_DW` | 恢复上游深度卷积 kernel；可与上一项组合 |
| `FCPE_METAL_LEGACY_GN` | 恢复上游 GroupNorm |
| `FCPE_METAL_LEGACY_IM2COL` | 恢复上游展开 kernel |
| `FCPE_METAL_DISABLE_GLU_PROJECT` | 禁用投影＋GLU 融合 |
| `FCPE_METAL_DISABLE_FUSION` | 禁用本项目融合，保留 ggml 原有融合 |
| `FCPE_METAL_SINGLE_SUBMIT` | 主线程一次编码整图的诊断路径，默认关闭 |
| `FCPE_METAL_PROFILE` | 独立节点 GPU 时间戳，非正式测速 |
| `--reference-graph` | 恢复原 im2col 深度卷积图，但 F32 修复仍启用 |

macOS CI 编译 Metal 并做 CPU 对照；托管 runner 不保证 GPU 可访问，不能代替实机 Metal 验证。本轮只实测 Apple M4；Intel Mac、其他 Apple GPU、旧 macOS 和 M5 未验证。数据支持的是本机已测候选中的选择，不是全局理论最优证明。
