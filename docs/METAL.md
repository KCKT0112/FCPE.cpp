# Metal 后端预留与验证计划

本轮新增的 Vulkan 显存选择、矩阵乘法参数和六类融合均限定在 Vulkan 后端；Metal 仍使用原有 GLU 视图加连续拷贝路径。不能将离散显卡的非 host-visible 显存策略套用到 Apple 统一内存。CPU Mel 前端共享本轮的稀疏 Mel 投影、FFT plan 缓存、64 帧批处理和按线程数并行，macOS 上仍需核验其编译、数值及端到端收益。CMake 已显式链接 `Threads::Threads`。

当前已提供 `FCPE_METAL=ON`，通过 ggml 注册设备、分配模型和调度计算图。**本次环境是 Windows，没有运行或编译验证 Metal，也没有 Metal 性能数据。** Vulkan 的专用 F32 内核只加入 Vulkan 构建，不会被当作 Metal 实现使用。

在 macOS 上先构建并枚举设备，后续命令中的 `Metal` 应替换为实际输出的设备名称：

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DFCPE_METAL=ON
cmake --build build-metal -j
./build-metal/bin/fcpe-cli --list-backends
./build-metal/bin/fcpe-cli -m models/fcpe-f32.gguf -i input.wav -o pitch.csv --backend Metal
./build-metal/bin/fcpe-backend-tests Metal
./build-metal/bin/fcpe-tests models/fcpe-f32.gguf Metal
python scripts/validate.py --cli build-metal/bin/fcpe-cli --model models/fcpe-f32.gguf \
  --backend Metal --wav validation/jfk.wav --output-dir validation/metal-f32
./build-metal/bin/fcpe-bench --model models/fcpe-f32.gguf \
  --mel validation/benchmark/11s.mel.f32 --backend Metal --threads 4 \
  --warmup 3 --repeats 20 --output validation/metal-bench.json
```

`benchmark.py` 的自动对比列表目前针对本机 CPU、Vulkan、PyTorch CUDA 和 ORT CUDA；Metal 可先使用支持任意设备名的 `fcpe-bench`，再补齐 Apple 平台的自动对比。不能直接把脚本中的 CUDA 测试替换名称当作 MPS 测试。

对 ggml v0.19.0 源码的检查结果：

| FCPE 路径 | 上游 Metal 源码情况 | 仍需实机确认 |
| --- | --- | --- |
| 两层普通卷积 | `IM2COL` 接受 F32 输入和 F32 输出，随后执行 `MUL_MAT` | 中间量和矩阵乘法实际精度 |
| 六层深度卷积 | `CONV_2D_DW` 声明支持 F32 输入／权重／输出 | `[T,1,C]` 布局、单帧和 padding 的正确性与速度 |
| GroupNorm / LayerNorm | 支持检测依赖 SIMD-group reduction 及连续行 | 跨时间的归一化、不同序列长度的误差 |
| GLU、SiLU、残差与仿射变换 | 由视图、拷贝、sigmoid、乘法、加法等基本算子组成 | 调度是否回退 CPU，融合收益 |
| GGUF F16 存储 | 模型图先将普通矩阵权重转为 F32 | 转换成本和峰值内存；阈值附近的既有差异 |

深度卷积构图会先询问所选设备是否支持直接卷积，若不支持则保留原来的 im2col 路径。scheduler 仍可对其他不支持的算子使用 CPU。运行后通过 `Model::execution_info()` 或 benchmark JSON 的 `cpu_compute_nodes`、`accelerator_compute_nodes`、`graph_splits` 核实实际分配；仅列出 Metal 设备不足以证明全图加速。

实机工作顺序：先完成 F32 的逐算子及 14 组端到端精度对比，再比较直接卷积与 `--reference-graph` 的预热耗时和内存，最后决定是否需要专用 Metal F32 矩阵乘法或 sigmoid-GLU 融合。不要假定 F32 GGUF 或 `GGML_PREC_F32` 足以保证所有后端中间量精度；Vulkan 已验证这两者与实际乘法操作数精度并不等价。

源码中还有一个具体的精度检查点：`ggml-metal.metal` 的 `kernel_mul_mm_f32_f32` 实例化使用 `half` 和 `simdgroup_half8x8` 作为内部矩阵类型。名称中的 F32 描述存储类型，不能据此认定乘法操作数始终为 F32。此路径尚未做本模型实测，应优先确认选择条件及误差；如果被本模型使用且超出误差界限，需要补充真正的 F32 Metal 路径或使用经过验证的 CPU 后端。当前没有宣称 Metal 达到 Windows F32 的精度。
