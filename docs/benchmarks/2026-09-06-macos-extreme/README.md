# Metal optimization round, 2026-09-06

Apple M4, macOS 27.0, strict F32. See [METAL.md](../../METAL.md) for kernel choices, reproduction commands and limitations. The [initial macOS archive](../2026-09-06-macos/README.md) is unchanged.

The final implementation retains tiled copies and im2col, channel-contiguous depthwise convolution, parallel whole-recording GroupNorm, strided sigmoid-GLU, projection-plus-GLU and bias activation fusion. It does not introduce reduced precision, audio chunking or a changed confidence threshold.

| Artifact | Contents |
| --- | --- |
| `benchmark.json` / `all-results.md` | Same-input framework benchmark over 10 lengths, from 10 ms through 60 seconds; raw samples, numerical checks, hashes and actual native end-to-end samples |
| `ablation.json` / `ablation.md` | 132 runs: 11 configurations × 4 lengths × 3 alternating rounds, 20 samples each; all numerical checks retained |
| `validation.json` | Final original-wheel comparisons, including the known F16 JFK threshold failure |
| `tuning.json` | Exploratory configuration, tile, fusion, convolution and memory probes, separate from formal timings |
| `profile-baseline.log` / `profile-current.log` | Isolated original-node GPU timestamps, fusion disabled; diagnostic only |
| `shader-validation.log` / `ctest.log` | Metal Shader Validation and ordinary final CTest results |
| `provenance.json` | Final source/binary/dependency hashes, toolchain and validation evidence |

The copied baseline binary is identified by the initial archive's Metal dylib hash. The ablation runner explicitly sets its dylib directory and verifies actual dyld loads: a copied executable's original absolute CMake rpath alone would have loaded the current backend. `scripts/benchmark_metal.py` reproduces the native ablations and records every run.

All performance comparisons measure host F32 Mel to host probabilities with GPU synchronization and transfers. Native WAV-sample-to-F0 timing is a separate loop, excluding disk I/O and model loading. Instrumented tests and GPU profiles are excluded. Cross-round frequency/scheduling variation is visible in the raw samples; neither a minimum sample nor historical platform timings replace the controlled baseline comparison.

The exploratory custom generic matmul variants, epilogues, implicit ordinary convolution and broadcast kernel were removed when they did not show stable benefit. The old spatial depthwise layout/kernel and other useful ablation switches remain available. The independent backend tests and model comparisons gate quality; the results do not prove a global hardware optimum or validate other Mac models.

The framework-wide benchmark deliberately returns a nonzero status: ORT CPU at 30/60 seconds has probability maximum errors 2.72095e-5/0.000110090, above the original 1e-5 gate. Its F0/UV checks pass. All 40 native FCPE and 20 PyTorch CPU/MPS combinations pass; ORT passes its other 8 cases. No tolerance is widened. The native 132-run ablation and 15-recording full-WAV parity suite both pass.
