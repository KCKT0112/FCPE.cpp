# FCPE.cpp

> **语言：** [English](README.md) | [中文](README_CN.md)

[FCPE](https://github.com/CNChTu/FCPE) 的 C++17 / ggml 推理实现，参考 [game.cpp](https://github.com/KakaruHayate/game.cpp) 与 [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) 的工程组织方式。输入 WAV 或 Mel，输出逐帧 F0、置信度和清浊音标记。原生推理不依赖 Python、PyTorch 或 ONNX Runtime。

移植目标为用户指定的官方发布包：

```text
https://github.com/CNChTu/FCPE/releases/download/v0.0.4/torchfcpe-0.0.4-py3-none-any.whl
└── torchfcpe/assets/fcpe_c_v001.pt
```

本项目是独立移植，非 FCPE 官方发行版。项目代码使用 **[MPL-2.0](LICENSE)**；FCPE 上游 MIT 声明和第三方许可完整保留，模型资产不因 GGUF 转换而改用 MPL。来源、分发要求和使用提醒见 **[NOTICE.md](NOTICE.md)**。

发布模型使用 `CFNaiveMelPE` 的 `conv_only=true` 架构：16 kHz、hop 160、FFT 1024、128 Mel bins、512 隐藏维、6 个 Conformer 卷积模块、360 个音高 bins，音高范围约 32.7–1975.5 Hz。

## 下载、解包与转换

转换需要 Python 3.10+。建议在自己的虚拟环境中安装依赖；已有 PyTorch 时保留与其版本匹配的 torchaudio。

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/download_model.py
python scripts/convert_fcpe.py models/torchfcpe-0.0.4-py3-none-any.whl models/fcpe-f32.gguf
python scripts/convert_fcpe.py models/torchfcpe-0.0.4-py3-none-any.whl models/fcpe-f16.gguf --type f16
```

也可以直接转换解包后的 checkpoint：

```powershell
python scripts/convert_fcpe.py models/torchfcpe-0.0.4/torchfcpe/assets/fcpe_c_v001.pt models/fcpe-f32.gguf
```

`download_model.py` 验证 wheel 的 SHA256，再解包源码及权重。原始 wheel 的 SHA256：

```text
f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d
```

| 文件 | 大小 | 用途 |
| --- | ---: | --- |
| `models/fcpe-f32.gguf` | 41.56 MiB | 精度基准，所有权重 F32 |
| `models/fcpe-f16.gguf` | 21.33 MiB | 卷积和线性矩阵存储为 F16；归一化、深度卷积、Mel 常量及 cents 表保留 F32 |

每个 GGUF 包含 61 个张量，旁边的同名 `.json` 记录来源、模型配置、文件大小与 SHA256。转换器融合输出层的 weight normalization，移除推理不用的参数，检查张量形状并拒绝未支持的模型架构。转换不导入或修改原版模型实现。

F16 是较小的**存储格式**：推理时将这些矩阵转换为 F32，并保留 F32 中间计算，避免输入激活再次被舍入。它不保证降低峰值内存。需要尽可能复现原版时使用 F32；F16 的权重舍入仍可能改变置信度阈值附近的清浊判断，见下方精度说明。

## 编译

使用 CMake 3.18+ 和 C++17 编译器。ggml 固定为 v0.19.0：若存在 `third_party/ggml`，使用本地源码；否则 CMake 下载带 SHA256 校验的固定版本。也可传入 `-DFCPE_GGML_SOURCE_DIR=/path/to/ggml`。

Windows / MSVC：

```powershell
.\cmake\build-msvc.cmd
```

脚本用 `vswhere` 查找 Visual Studio，初始化 x64 编译环境后调用 Ninja。也可以在 Developer PowerShell 中自行构建：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Linux / macOS：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

默认编译 CPU 后端。`GGML_NATIVE` 使用 ggml 的默认值 ON，生成适合当前 CPU 的指令；要发布到其他 CPU，可设置 `-DGGML_NATIVE=OFF` 并选择合适的 ggml 指令集选项。

可选 GPU 后端通过 `FCPE_VULKAN`、`FCPE_CUDA`、`FCPE_METAL` 开启。Vulkan 构建需要 Vulkan SDK（含 glslc 和 SPIRV-Headers 的 CMake 配置），CUDA 需要与宿主编译器兼容的 CUDA Toolkit，Metal 需要 macOS 的开发工具链。例如：

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.341.1' # 替换为本机路径
$env:FCPE_BUILD_DIR = 'build-vulkan'
.\cmake\build-msvc.cmd -DFCPE_VULKAN=ON
Remove-Item Env:FCPE_BUILD_DIR
```

```sh
cmake -S . -B build-cuda -DFCPE_CUDA=ON
cmake --build build-cuda -j
# macOS:
cmake -S . -B build-metal -DFCPE_METAL=ON
cmake --build build-metal -j
```

## 使用 CLI

```powershell
.\build\bin\fcpe-cli.exe -m models/fcpe-f32.gguf -i input.wav -o pitch.csv -t 4
.\build-vulkan\bin\fcpe-cli.exe --list-backends
.\build-vulkan\bin\fcpe-cli.exe -m models/fcpe-f32.gguf -i input.wav -o pitch.csv --backend Vulkan0
```

Linux / macOS 的可执行文件为 `build/bin/fcpe-cli`。Windows 运行时将对应构建目录 `bin` 中的 ggml DLL 与 exe 放在一起。

CSV 格式：

```csv
time,f0,confidence,unvoiced
0,0,0.0012,1
0.01,220.15,0.82,0
```

`time` 单位为秒，`f0` 单位为 Hz；默认帧距 10 ms。低于置信度阈值的帧输出 F0=0。`confidence` 是 360 个音高 bins 中的最大概率，`unvoiced=1` 表示清音或低于设定最小音高的帧。

| 参数 | 含义 |
| --- | --- |
| `--backend cpu/auto/名称` | 默认 CPU；auto 优先可用 GPU，明确指定不存在的设备会报错 |
| `-t, --threads N` | CPU 推理和 Mel 前端线程；0 自动选择，自动选择最多 8 |
| `--threshold 0.006` | 与 wheel 推理接口一致的默认阈值，比较使用严格大于 |
| `--decoder local_argmax/argmax` | 默认使用峰值附近 9 个 bins 的加权 cents；原版的 argmax 名称实际上表示全 bins 加权平均 |
| `--interp-uv` | 对未发声区间的 F0 插值，保留原始 UV 标记 |
| `--f0-min HZ` | 用于 UV 分类，不会直接把低于该值的 F0 截断 |
| `--f0-max HZ` | 输出 F0 上限，0 不截断 |
| `--output-frames N` | 对输出进行最近邻插值 |
| `--mel PATH` | 替代 WAV 输入，读取小端 float32、行主序 `[frames,128]`，帧数从文件大小计算 |
| `--dump-prefix PATH` | 保存 `.mel.f32`、`.probabilities.f32`、`.f0.f32`，用于对照分析 |
| `--reference-graph` | 使用优化前的 im2col 深度卷积图，供 A/B 对照 |

WAV 支持 PCM 8/16/24/32 位、IEEE float 32/64 位及相应 WAVE_FORMAT_EXTENSIBLE 格式，多声道取平均。非 16 kHz 输入使用与原版 `torchaudio.transforms.Resample(lowpass_filter_width=128)` 对应的 sinc/Hann 重采样。

## C++ 集成

```cmake
add_subdirectory(path/to/FCPE.cpp)
target_link_libraries(your_app PRIVATE fcpe::fcpe)
```

```cpp
#include <fcpe/fcpe.h>

fcpe::Model model("models/fcpe-f32.gguf", {"cpu", 4});
fcpe::Audio audio = fcpe::read_wav("input.wav");
fcpe::DecodeOptions options;
options.threshold = 0.006f;
auto result = model.infer(audio.samples, audio.sample_rate, options);
// result.f0, result.confidence, result.unvoiced
```

公共接口在 `include/fcpe/fcpe.h`，也支持分别调用 `mel()`、`probabilities()`、`infer_mel()`。模型对象复用已分配的同长度计算图，输入长度变化时重建；并发调用请使用不同模型实例。`examples/` 提供可单独用 CMake 构建的外部消费者示例。

## 精度与测试

验证脚本实际导入指定 wheel 的源码和 checkpoint，逐项比较 PyTorch 与 C++ 的 Mel、360 维输出概率、F0 和 UV。13 组内建用例包括单样本、padding/hop 边界、静音、谐波扫频、噪声、清浊转换、立体声及 8/16/22.05/44.1/48 kHz 输入。可以用 `--wav recording.wav` 增加真实录音。

```powershell
cmake -S . -B build -DFCPE_TEST_MODEL="${PWD}/models/fcpe-f32.gguf"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python tests/test_tools.py --cli build/bin/fcpe-cli.exe --model models/fcpe-f32.gguf -v
python scripts/validate.py --model models/fcpe-f32.gguf
python scripts/validate.py --model models/fcpe-f16.gguf --type f16 --output-dir validation/f16
python scripts/validate.py --cli build-vulkan/bin/fcpe-cli.exe --model models/fcpe-f32.gguf --backend Vulkan0 --output-dir validation/vulkan-f32
```

详细指标写入输出目录的 `report.json`。C++ 测试额外覆盖解码阈值、音高边界的重复 gather、UV 插值、动态长度和连续调用。工具回归测试覆盖 12 种 WAV 编码/容器组合、截断文件、无效 GGUF 元数据、非有限音频和 CLI 参数错误。

本机 Windows / Ryzen 7 5700X / RTX 4060 验证：F32 CPU 与 Vulkan 均通过 13 组内建用例和 11 秒 JFK 真实语音，全部清浊判定一致；完整音频链路 F0 最大绝对误差为 **0.001313 Hz**。F16 通过 13 组内建用例，但在 JFK 的 1101 帧中有 1 帧因权重舍入跨过 `0.006` 阈值，严格对照脚本如实返回失败。详见 [docs/VALIDATION.md](docs/VALIDATION.md)，其中保留 F16 的原始 F0 差异，未用只统计有声帧的结果替代。

数值实现说明：

- 普通卷积显式使用 F32 im2col，避免 ggml 默认 `ggml_conv_1d` 将输入中间量降至 F16。神经网络计算均由 ggml 图执行，GPU 后端使用 ggml scheduler；不支持的算子可由 CPU 后端执行。
- 深度卷积默认使用 ggml 已有的 `CONV_2D_DW` 直接实现，消除 kernel=31 的 im2col 展开；所选设备不支持时保留原路径。CPU 和 Vulkan 均已实测。
- Vulkan 默认启用 `FCPE_VULKAN_STRICT_F32=ON`。ggml v0.19.0 的 cooperative-matrix/F16 路径会舍入 F32 输入，`GGML_PREC_F32` 仅保证累加精度，不能阻止该行为。`cmake/VulkanPrecision.cmake` 在构建目录生成一份源文件副本，选择 ggml 已有的完整 F32 shader 路径，不修改依赖源码或进程环境。该设置对本构建中的 Vulkan 后端整体生效，会影响共用此后端的其他模型；关闭它可恢复上游性能路径，但不再保证这里的 F32 对照精度。
- Mel 使用 pocketfft。不同 F32 FFT 实现在接近 `1e-5` 截断值处的微小幅值差会被 log 放大，因此验证同时检查 log-Mel、线性 Mel 和直接输入相同 Mel 时的网络误差。并非逐位相同的 FFT 输出。
- `local_argmax` 在边界重复使用被 clamp 的 bin，与原版 `gather` 保持一致；不能用缩短窗口替代。
- wheel 的 Wav2Mel 用**重采样前**的样本数修正帧数，导致低于 16 kHz 的输入被截断。此实现根据**重采样后的**长度输出 `floor(samples / hop) + 1` 帧；验证时先由原版 torchaudio 重采样，再调用 wheel 的 16 kHz 路径。
- 全清音的 UV 插值返回全零，避免原版在没有任何有声点时索引空张量。

## 加速与性能对比

Vulkan 默认启用严格 F32、device-local 显存优先、上游 F32 矩阵内核参数调优，以及 LayerNorm／仿射／sigmoid-GLU／bias 激活融合。GLU 直接读取带 stride 的切片，去掉中间拷贝。所有后端补丁只生成在构建目录中，不改 ggml checkout，也不需要运行时编译 shader。CPU Mel 前端加入稀疏投影、FFT 缓存和批量并行。

本机 Ryzen 7 5700X / RTX 4060，同一 Mel 输入、4 个 CPU 线程、预热 3 次、测量 20 次，11 秒网络中位耗时：

| 引擎 | 本轮实测 |
| --- | ---: |
| ggml CPU，直接深度卷积 | 222.404 ms |
| ggml Vulkan，上一轮配置复测 | 94.198 ms |
| **ggml Vulkan，当前完整 F32** | **6.362 ms** |
| PyTorch CPU / CUDA | 206.740 / 6.472 ms |
| ONNX Runtime CPU / CUDA | 136.578 / 7.703 ms |

当前 Vulkan 相对上一轮配置再快约 **14.81 倍**；相对历史记录 99.331 ms 约快 15.61 倍。与本次 PyTorch CUDA 的差距很小，应视为同一速度档位。表中包含 GPU 传输与同步；**实际 WAV→F0 中位为 11.474 ms，P95 为 11.833 ms**，不含读盘与模型加载。当前 Vulkan 的 0.1 / 1 / 3 秒网络中位为 0.936 / 1.406 / 2.763 ms。

显存类型是本机最大的突破：两种分配都在 GPU 显存堆，host-visible 路径却显著较慢；已经记录仅切换分配方式的独立 A/B，未把尚未证实的驱动原因当结论。旧手写矩阵 kernel 保留用于复现，在当前显存配置下优先使用更快的上游 F32 shader 调优版本。计算缓冲区从原图 140.31 MiB／上一轮 22.04 MiB 降到 **17.74 MiB**，1 个调度分区、110 个 GPU 计算节点、0 个 CPU 计算节点。

44 组引擎／长度组合的精度检查全部通过。完整数据、P95、原始样本、显存属性、开关组合、六类融合、剩余瓶颈和复现命令见 [docs/PERFORMANCE.md](docs/PERFORMANCE.md)。测速工具为 `fcpe-bench` 和 `scripts/benchmark.py`，ONNX 从原 wheel 模型导出。第一轮数据归档保留不变。

## 范围与限制

此实现对应发布包中的纯卷积推理模型。转换器会拒绝 attention、harmonic embedding 或 STFT-only 等其他训练配置。F32 与混合 F16 模型受支持，尚未实现整数权重量化。

输入堆栈的 GroupNorm 同时跨时间和组内通道归一化，因此分段音频的结果不等价于整段推理。当前保留整段语义，计算图内存随音频长度增长，没有隐式切片或流式近似。原生 ggml CUDA、Metal 开关接入上游 ggml，但未在本机验证；PyTorch／ORT CUDA 的测试结果不代表 ggml CUDA。Metal 的源码检查、精度注意点和 Mac 验证步骤单独记录在 [docs/METAL.md](docs/METAL.md)。

## 许可与声明

本项目自有代码、构建集成、工具和文档采用 **[Mozilla Public License 2.0](LICENSE)**，文件另有声明的除外。MPL 的文件级开源要求、可执行文件分发时对应源代码的提供方式、第三方版权与免责说明见 [NOTICE.md](NOTICE.md)。

FCPE 上游代码及指定官方 wheel 标明 MIT；原始 CN_ChiTu 版权和许可文本保存在 [licenses/FCPE-MIT.txt](licenses/FCPE-MIT.txt)。MPL 不替代 ggml 的 MIT、pocketfft 的 BSD-3-Clause、librosa 的 ISC 或上游模型许可。权重转换不会改变模型的许可归属；再分发 GGUF 时应附上来源、精度、转换信息及适用的上游许可。

依赖版本与来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。分发构建产物时请一并保留 LICENSE、NOTICE.md、THIRD_PARTY.md 和适用的 licenses/ 文件；CMake 默认将其安装到 share/doc/fcpe。软件按现状提供，不附带保证。本项目未获 FCPE 或其他上游作者的官方背书。模型权重、下载的参考仓库、依赖 checkout、构建结果及验证音频不纳入 Git 源码文件列表。
