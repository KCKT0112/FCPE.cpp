# Apple M4 macOS validation, 2026-09-06

See [Metal adaptation and reproduction](../../METAL.md) for the implementation and test commands. The existing Windows archives are unchanged.

| Artifact | Contents |
| --- | --- |
| [benchmark.json](benchmark.json) | 7 engines × 4 input lengths; 3 warmups, 20 raw timing samples; correctness, model/input and native dylib hashes; native WAV-to-F0 samples and scheduler placement |
| [all-results.md](all-results.md) | All network medians, P95 and correctness results |
| [validation.json](validation.json) | All numerical metrics for CPU/Metal × F32/F16; known F16 JFK failures retained |
| [matmul-upstream.log](matmul-upstream.log) | Strict mode OFF: first F32 matrix failure and comparison with half-rounded operands |
| [matmul-strict.log](matmul-strict.log) | Strict mode ON: all 19 matrix shapes pass |
| [ctest-cpu.log](ctest-cpu.log), [ctest-metal.log](ctest-metal.log) | Final CPU and GPU-enabled test runs |
| [provenance.json](provenance.json) | Source hashes, toolchain, unchanged upstream checks, build settings and validation limits |

Network measurements start with identical host float32 Mel and end with host probabilities. GPU transfers and synchronization are included. Native end-to-end samples separately time already-loaded WAV samples to F0. They exclude disk I/O and model loading. Independent timing loops can differ because of device frequency, scheduling and memory state; their medians must not be added or subtracted to infer a stage cost.

All 28 benchmark combinations passed the existing probability/F0/UV gates. PyTorch MPS is measured with explicit device synchronization and CPU fallback disabled; it is separate from native ggml Metal. The CPU build had no OpenMP. This is an M4 baseline, not a claim of optimal hardware utilization or support verified on other GPUs.

Model assets, WAVs, ONNX, output tensors and verbose validation pipeline logs remain in ignored models/ and validation/ directories. The exact upstream wheel and benchmark WAV are identified by SHA256. Source changes implementing this result were uncommitted at measurement time; source hashes identify the implementation.
