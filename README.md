# FCPE.cpp

> **Languages:** [English](README.md) | [中文](README_CN.md)

A **C++17 / ggml implementation of [FCPE](https://github.com/CNChTu/FCPE)** for frame-level fundamental frequency (F0) estimation. Read WAV audio or Mel features and produce pitch, confidence and voiced/unvoiced flags. Native inference does not require Python, PyTorch or ONNX Runtime.

This is an independent port, with project organization and ggml/GGUF integration informed by [game.cpp](https://github.com/KakaruHayate/game.cpp) and [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp). It is not an official FCPE release or an upstream-endorsed product.

**Project code: [MPL-2.0](LICENSE).** Upstream FCPE notices and third-party licenses are preserved. Model assets retain their upstream terms; conversion to GGUF does not make them MPL-licensed. See **[NOTICE.md](NOTICE.md)** for provenance, distribution requirements and usage notes.

## Features

- Native WAV loading, resampling, Mel extraction, ggml inference and F0 decoding.
- F32 GGUF for reference accuracy; optional F16 weight storage with documented threshold differences.
- Validated Windows CPU and Vulkan paths, with direct depthwise convolution, strict F32 Vulkan fusion and tuned matrix multiplication.
- Reusable C++ model sessions, a CLI, conversion tools, numerical tests and reproducible PyTorch / ONNX Runtime benchmarks.
- Metal on Apple M4: strict F32, optimized kernels/fusion and real-model validation; native ggml CUDA remains unvalidated.

## Model and upstream source

The numerical reference is the exact official **torchfcpe 0.0.4** release:

```text
https://github.com/CNChTu/FCPE/releases/download/v0.0.4/torchfcpe-0.0.4-py3-none-any.whl
└── torchfcpe/assets/fcpe_c_v001.pt
```

The wheel declares MIT and includes the original `Copyright (c) 2023 CN_ChiTu` notice, retained in [licenses/FCPE-MIT.txt](licenses/FCPE-MIT.txt). The checkpoint is bundled in that distribution; no separate checkpoint license was present in the inspected wheel. The wheel, extracted package and model weights are downloaded locally and are not committed to this repository.

The released `CFNaiveMelPE` model uses `conv_only=true`: 16 kHz audio, a 160-sample hop, FFT size 1024, 128 Mel bins, hidden width 512, six Conformer convolution modules and 360 pitch bins spanning approximately 32.7–1975.5 Hz. Other FCPE training configurations are outside this port's scope.

## Download and convert

Conversion requires Python 3.10+. Use a virtual environment where practical; if PyTorch is already installed, keep a matching torchaudio version.

```sh
python -m pip install -r scripts/requirements.txt
python scripts/download_model.py
python scripts/convert_fcpe.py models/torchfcpe-0.0.4-py3-none-any.whl models/fcpe-f32.gguf
python scripts/convert_fcpe.py models/torchfcpe-0.0.4-py3-none-any.whl models/fcpe-f16.gguf --type f16
```

You can also convert the extracted checkpoint:

```sh
python scripts/convert_fcpe.py models/torchfcpe-0.0.4/torchfcpe/assets/fcpe_c_v001.pt models/fcpe-f32.gguf
```

`download_model.py` verifies this wheel SHA256 before extracting it:

```text
f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d
```

| File | Size | Purpose |
| --- | ---: | --- |
| `models/fcpe-f32.gguf` | 41.56 MiB | Reference model; all weights stored as F32 |
| `models/fcpe-f16.gguf` | 21.33 MiB | Convolution / linear matrices stored as F16; normalization, depthwise weights and frontend / decoder constants remain F32 |

Each GGUF contains 61 tensors. Its adjacent `.json` records the source, model configuration, file size and hashes. The converter folds output weight normalization, removes training-only parameters and rejects unsupported architectures or tensor shapes. See [the model format](docs/MODEL_FORMAT.md).

**F16 is a storage option.** These matrices are expanded to F32 for inference; it does not halve peak inference memory. Weight rounding can still change a voiced/unvoiced decision near the confidence threshold. Use F32 to match the original model more closely, and retain the upstream attribution and license when redistributing converted weights.

## Build

Requirements: CMake 3.18+ and a C++17 compiler. ggml is pinned to **v0.19.0**. CMake uses `third_party/ggml` if present, otherwise downloads a SHA256-checked archive. To use another checkout, set `-DFCPE_GGML_SOURCE_DIR=/path/to/ggml`; the Vulkan integration expects the pinned version.

### Windows / MSVC

```powershell
.\cmake\build-msvc.cmd
```

The helper locates Visual Studio with `vswhere`, initializes the x64 compiler environment and builds with Ninja. Alternatively, in a Developer PowerShell:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Linux / macOS

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CPU is the default backend. ggml's `GGML_NATIVE` defaults to ON and targets the build machine's CPU. For distribution to other CPUs, use `-DGGML_NATIVE=OFF` and select appropriate ggml instruction-set options.

### GPU backends

Enable `FCPE_VULKAN`, `FCPE_CUDA` or `FCPE_METAL`. Vulkan needs a Vulkan SDK with glslc and the SPIRV-Headers CMake package. CUDA needs a toolkit compatible with the host compiler. Metal requires macOS development tools.

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.341.1' # Adjust for your installation.
$env:FCPE_BUILD_DIR = 'build-vulkan'
.\cmake\build-msvc.cmd -DFCPE_VULKAN=ON
Remove-Item Env:FCPE_BUILD_DIR
```

```sh
cmake -S . -B build-cuda -DFCPE_CUDA=ON
cmake --build build-cuda -j
# On macOS:
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DFCPE_METAL=ON
cmake --build build-metal -j
./build-metal/bin/fcpe-cli --list-backends
```

The Vulkan build defaults to strict F32, device-local memory preference, matrix-kernel tuning and FCPE fusion. It generates a build-local ggml source copy; the dependency checkout is not modified. Shaders are compiled and embedded at build time. These settings affect other models sharing the same built Vulkan backend. Options and device-specific limitations are described in [the performance report](docs/PERFORMANCE.md) (Chinese).

## CLI

```powershell
.\build\bin\fcpe-cli.exe -m models/fcpe-f32.gguf -i input.wav -o pitch.csv -t 4
.\build-vulkan\bin\fcpe-cli.exe --list-backends
.\build-vulkan\bin\fcpe-cli.exe -m models/fcpe-f32.gguf -i input.wav -o pitch.csv --backend Vulkan0 -t 4
```

On Linux / macOS, use `build/bin/fcpe-cli` without `.exe`. On Windows, keep the corresponding build's ggml DLLs alongside the executable.

```csv
time,f0,confidence,unvoiced
0,0,0.0012,1
0.01,220.15,0.82,0
```

`time` is in seconds and `f0` in Hz; the default frame interval is 10 ms. Confidence is the maximum probability over the 360 pitch bins. Confidence must be strictly greater than the threshold for a nonzero decoded pitch. `unvoiced=1` also marks pitches below an explicitly configured minimum F0.

| Option | Meaning |
| --- | --- |
| `--backend cpu/auto/NAME` | Default CPU; `auto` prefers an available GPU. An unavailable explicit device is an error |
| `-t, --threads N` | CPU inference and Mel frontend threads; 0 chooses automatically, up to 8 in automatic mode |
| `--threshold 0.006` | Default confidence threshold; uses strict greater-than, matching the wheel |
| `--decoder local_argmax/argmax` | Default: weighted cents over nine clamped bins around the peak. Upstream's `argmax` mode is actually a weighted average over all bins |
| `--interp-uv` | Interpolate F0 through unvoiced regions while retaining the original UV flags |
| `--f0-min HZ` | Used for UV classification; does not directly clamp low decoded F0 to zero |
| `--f0-max HZ` | Upper F0 clamp; 0 disables it |
| `--output-frames N` | Resize output with nearest-neighbor interpolation |
| `--mel PATH` | Read little-endian F32, row-major `[frames,128]` instead of WAV |
| `--dump-prefix PATH` | Write `.mel.f32`, `.probabilities.f32` and `.f0.f32` for comparison |
| `--reference-graph` | Restore the original im2col depthwise graph for A/B testing; backend settings are controlled separately |

WAV input supports PCM 8/16/24/32-bit, IEEE float 32/64-bit and their WAVE_FORMAT_EXTENSIBLE forms. Multiple channels are averaged. Non-16-kHz audio uses sinc/Hann resampling corresponding to upstream `torchaudio.transforms.Resample(lowpass_filter_width=128)`.

## C++ integration

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

The public API is in [include/fcpe/fcpe.h](include/fcpe/fcpe.h), including separate `mel()`, `probabilities()` and `infer_mel()` calls. A model reuses its allocated graph for the same input length and rebuilds it when the length changes. Use separate instances for concurrent calls. [examples/](examples/) contains a standalone CMake consumer.

## Accuracy and tests

The validation harness executes the original extracted wheel and compares Mel features, all 360 output probabilities, F0 and UV flags. Thirteen built-in recordings cover one-sample inputs, hop/padding boundaries, silence, chirps, noise, voiced/unvoiced transitions, stereo and 8/16/22.05/44.1/48 kHz input. Add a recording with `--wav recording.wav`.

```powershell
cmake -S . -B build -DFCPE_TEST_MODEL="${PWD}/models/fcpe-f32.gguf"
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
python tests/test_tools.py --cli build/bin/fcpe-cli.exe --model models/fcpe-f32.gguf -v
python scripts/validate.py --model models/fcpe-f32.gguf
python scripts/validate.py --model models/fcpe-f16.gguf --type f16 --output-dir validation/f16
python scripts/validate.py --cli build-vulkan/bin/fcpe-cli.exe --model models/fcpe-f32.gguf --backend Vulkan0 --output-dir validation/vulkan-f32
.\build-vulkan\bin\fcpe-fusion-tests.exe Vulkan0
```

Results are written to `report.json` in the chosen directory. C++ tests also cover decoder boundaries, duplicate clamped bins, UV interpolation, dynamic lengths, repeated inference, matrix shapes and fused/fallback operations. File-format regressions exercise 12 WAV encoding/container combinations and malformed input handling.

On Windows / Ryzen 7 5700X / RTX 4060, **F32 CPU and Vulkan pass all 13 built-in recordings plus the 11-second JFK speech sample**. All voiced/unvoiced decisions match; the maximum full-path F0 error is approximately **0.001313 Hz**. F16 passes the built-in recordings but crosses the `0.006` threshold at one of JFK's 1101 frames; the strict validator deliberately reports failure when that recording is included. See [the validation record](docs/VALIDATION.md).

Implementation details that matter for parity:

- Ordinary convolutions use F32 im2col explicitly; depthwise layers use ggml's existing direct `CONV_2D_DW` where supported, with the original graph as a fallback.
- `FCPE_VULKAN_STRICT_F32=ON` avoids upstream cooperative-matrix paths that can round F32 operands to F16. `GGML_PREC_F32` alone does not prevent that conversion. Disabling strict mode invalidates the precision claims recorded here.
- pocketfft and PyTorch need not produce bit-identical FFT outputs. Small magnitudes near the log floor amplify differences, so validation checks both linear and log Mel as well as network and pitch outputs.
- The local decoder preserves repeated clamped bins at pitch boundaries. Shortening the nine-bin window would change the result.
- Frame count is based on the **resampled** length. This corrects the wheel's use of the original length for non-16-kHz input; the reference harness resamples first for an equivalent comparison.
- Interpolating an entirely unvoiced recording returns zeros instead of indexing an empty set of voiced frames.

On Apple M4 / macOS 27, CPU and the corrected Metal F32 backend pass all 14 recordings, with maximum full-path F0 error **0.001389 Hz** and no voiced/unvoiced differences. The device name is `MTL0`. Default `FCPE_METAL_STRICT_F32=ON` uses float operands instead of upstream's internal half rounding. F16 retains the known JFK frame-885 threshold difference. See [Metal validation and commands](docs/METAL.md). Three alternating rounds reduce the 11-second Metal network median from **32.344 to 12.442 ms (2.60×)** with no CPU fallback. Final Metal passes 15 full-audio cases, including 60 seconds, and 7 CTests under Metal Shader Validation. The wider framework run retains two ORT CPU long-input probability failures; all native FCPE cases pass.

## Performance

The optimized path combines direct depthwise convolution, Vulkan memory selection, F32 matrix specialization, LayerNorm / affine / sigmoid-GLU / bias-activation fusion, and strided GLU views. The CPU frontend uses sparse Mel projection, cached FFT plans and batched parallel FFTs. It preserves whole-recording normalization semantics.

**Measured on 2026-09-06**, Ryzen 7 5700X / RTX 4060, batch 1, four CPU threads, three warmups and 20 timed calls. Values below are network/API medians in **milliseconds**, using identical host F32 Mel and returning host F32 probabilities. GPU transfers and synchronization are included; frontend, decoder and file I/O are excluded.

| Engine | 0.1 s audio | 1 s audio | 3 s audio | 11 s audio |
| --- | ---: | ---: | ---: | ---: |
| ggml CPU, current | 3.060 | 19.811 | 58.774 | 222.404 |
| ggml Vulkan, previous configuration retested | 9.712 | 16.566 | 34.175 | 94.198 |
| **ggml Vulkan, current strict F32** | **0.936** | **1.406** | **2.763** | **6.362** |
| PyTorch CPU | 24.199 | 29.490 | 65.404 | 206.740 |
| PyTorch CUDA | 3.280 | 4.129 | 3.855 | 6.472 |
| ONNX Runtime CPU | 3.675 | 13.301 | 43.121 | 136.578 |
| ONNX Runtime CUDA | 1.274 | 1.679 | 3.163 | 7.703 |

Current Vulkan is **14.81× faster** than the previous configuration retested in the same round. Against the earlier archived 99.331 ms result, it is 15.61× faster. The 11-second difference from PyTorch CUDA is small enough to be within timing variability; treat them as the same performance tier on this setup, not a general speed guarantee.

The separately measured **WAV-samples-to-F0 median is 11.474 ms, P95 11.833 ms** for 11 seconds of audio, excluding disk I/O and model loading. The native Mel frontend averages 4.293 ms. The first network call is 540.469 ms on that run, so reuse a resident `Model` for warm performance.

Vulkan scheduler compute buffers decreased from 140.31 MiB in the original graph, through 22.04 MiB in the previous configuration, to **17.74 MiB**. This excludes weights, driver resources and frontend/staging memory. The current graph has one scheduling split, 110 GPU compute nodes and no CPU compute nodes; fusion reduces actual dispatch work further.

All **44 engine/length combinations** in the full study passed numerical checks. The largest improvement on this machine came from avoiding the slower host-visible device-local memory preference. Both allocation types were in the GPU heap; the underlying driver/cache mechanism has not been established. This result should be rechecked on other GPUs.

See [the full performance report](docs/PERFORMANCE.md) (Chinese), [portable raw measurements](docs/benchmarks/2026-09-06-windows-extreme/README.md) (English) and [all 44 results](docs/benchmarks/2026-09-06-windows-extreme/all-results.md). The earlier archive is preserved. Benchmark setup:

```sh
python -m pip install -r scripts/requirements-benchmark.txt
# Supply a mono 16 kHz WAV. The recorded study used validation/jfk.wav.
python scripts/benchmark.py --prepare --wav validation/jfk.wav --output-dir validation/benchmark-extreme
python scripts/benchmark.py --output-dir validation/benchmark-extreme
# CPU references only, when CUDA reference dependencies are unavailable:
python scripts/benchmark.py --output-dir validation/benchmark-extreme --engines ggml_cpu torch_cpu ort_cpu
```

The default benchmark executable path is `build-vulkan/bin/fcpe-bench.exe`; use `--bench` for another build. PyTorch CUDA and ORT CUDA require their own compatible dependencies. Their successful tests do not validate the native ggml CUDA backend.

## Scope and limitations

This port targets the released convolution-only model. Attention, harmonic embedding, STFT-only training configurations and integer weight quantization are not supported.

Input GroupNorm spans both time and channels within each group. Processing independent chunks is not equivalent to processing the complete recording. This implementation keeps whole-recording semantics; graph memory grows with input length.

Native ggml CUDA remains unvalidated: CUDA 12.8 rejected the installed MSVC 19.51 compiler; no unsupported-compiler override was used. Metal has now been built and validated on Apple M4. Other Apple GPUs, Intel Macs and older macOS releases remain untested. See [Metal results and verification steps](docs/METAL.md) (Chinese).

## License and attribution

First-party code, build integration, tools and documentation are licensed under **[Mozilla Public License 2.0](LICENSE)** unless otherwise marked. MPL-2.0 provides file-level copyleft; distributors of covered executables must make the corresponding covered source available and tell recipients how to obtain it, as required by the license.

Upstream material retains its own terms: **FCPE / ggml: MIT; pocketfft: BSD-3-Clause; librosa: ISC**. The original FCPE copyright and MIT text are preserved in [licenses/FCPE-MIT.txt](licenses/FCPE-MIT.txt). The project MPL does not supersede upstream model licensing, and converting a checkpoint does not change its license.

Keep `LICENSE`, `NOTICE.md`, `THIRD_PARTY.md` and the applicable `licenses/` files with redistributions. CMake installs these under `share/doc/fcpe` by default. The software is provided **as is**, without warranty. Full attribution, model provenance, source-availability reminders and validation caveats are in **[NOTICE.md](NOTICE.md)**; inspected upstream commits are in [THIRD_PARTY.md](THIRD_PARTY.md).
