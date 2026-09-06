# Validation record

Date: 2026-09-06, second optimization round. Native Windows x64, MSVC 19.51, ggml v0.19.0, PyTorch 2.7.1+cu118 reference running on CPU. CPU: AMD Ryzen 7 5700X, 4 inference threads. GPU: NVIDIA RTX 4060, driver 620.02, Vulkan SDK 1.4.341.1. These results include direct depthwise convolution, device-local memory, tuned upstream F32 matmul, strided GLU, Vulkan graph fusion and the batched/parallel native Mel frontend. `FCPE_VULKAN_STRICT_F32`, `FCPE_VULKAN_TILED_F32` and `FCPE_VULKAN_FUSION` are ON. F32 operands and accumulation are retained; TF32, F16 activation rounding and audio chunking are not used.

## Reference and coverage

The reference executes the **unmodified Python files extracted from the user's exact torchfcpe 0.0.4 wheel**. The validation harness loads the trusted checkpoint with `weights_only=True`, sets inference dropout to zero like the bundled loader, and puts the model in evaluation mode.

The 13 built-in recordings cover silence, 1-sample input, short input, hop/padding boundaries, voiced/unvoiced regions, stereo mixing, noise, harmonic chirps and 8/16/22.05/44.1/48 kHz resampling. A fourteenth recording is the 11-second [JFK speech sample](https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav) from whisper.cpp (1101 model frames).

Each recording is run twice: first with identical PyTorch-generated Mel input to isolate the ggml graph, then from WAV through the complete native frontend. The comparator checks frame counts, finiteness, log-Mel, linear Mel, all 360 probabilities, decoded F0 and every unvoiced flag. Output artifacts include the original reference arrays, C++ arrays, CSVs and JSON metrics.

## F32 results

| Metric, maximum over all 14 recordings | CPU | Vulkan / RTX 4060 |
| --- | ---: | ---: |
| Probability absolute error, identical Mel input | 0.000003755 | 0.000002921 |
| Probability absolute error, complete WAV path | 0.000381351 | 0.000381649 |
| F0 absolute error, complete WAV path | 0.001312 Hz | 0.001312 Hz |
| Unvoiced flag differences | 0 | 0 |
| Strict comparison | PASS | PASS |

For the JFK speech recording alone, complete-path F0 maximum error is approximately 0.000305 Hz on both CPU and Vulkan. The largest full-suite probability difference comes from synthetic harmonic signals near the frontend's logarithmic floor. Across the suite, **linear Mel** maximum absolute error is 0.000002385; maximum log-Mel error is approximately 0.01369. Noise Mel agrees within 0.000000954. The validator separately gates on linear magnitude error, log error/RMSE, direct-Mel network error, final F0 and UV; it does not demand bitwise equivalence between pocketfft and PyTorch's FFT.

Raw reports:

- `validation/extreme-cpu-f32/report.json`
- `validation/extreme-vulkan-f32/report.json`

Portable metric snapshots for both precisions are retained in [benchmarks/2026-09-06-windows-extreme/validation.json](benchmarks/2026-09-06-windows-extreme/validation.json). The [first optimization round](benchmarks/2026-09-06-windows/validation.json) remains unchanged, as do the earlier pre-optimization reports in `validation/f32-speech` and `validation/vulkan-f32-strict`.

## F16 storage results and threshold boundary

F16 GGUF reduces the file from 41.56 MiB to 21.33 MiB. Normalization, depthwise kernels, the Mel basis, Hann window and cents table remain F32. All other matrix weights are expanded to F32 in the graph, so stored weight rounding is the only intentional precision reduction; peak inference memory is not halved.

The final CPU implementation passes all 13 synthetic cases. On the JFK recording, one of 1101 frames changes its voiced/unvoiced decision:

| Frame 885, 8.85 seconds | Original / F32 | F16 storage, CPU |
| --- | ---: | ---: |
| Maximum probability | 0.0059890384 | 0.0060026571 |
| Threshold | 0.006 | 0.006 |
| Decoded F0 | 0 Hz | 199.213074 Hz |

The weight rounding crosses the strict threshold. Consequently the raw maximum CPU F0 difference is **199.213074 Hz**, and `validate.py --type f16 --wav validation/jfk.wav` deliberately returns failure under the same zero-UV-mismatch requirement. On frames that both versions classify as voiced, the maximum JFK F0 difference is 0.006378 Hz on CPU and 0.006439 Hz on Vulkan. The largest probability difference across all 14 recordings is 0.00196144. The new Vulkan F16 WAV path exhibits the same one-frame threshold boundary (confidence 0.0060026580, F0 199.213074 Hz). With identical Mel, its confidence is 0.0060026660 and F0 is 199.213181 Hz. Both backends pass the 13 synthetic cases and fail the two JFK comparisons (identical Mel and complete WAV), as expected from the existing storage approximation.

Use the F32 model when matching the original model's decisions matters. Do not interpret the small error on jointly voiced frames as a guarantee that F16 preserves all voiced/unvoiced decisions. We retain the strict failing F16 report rather than widening the threshold or silently omitting this frame.

Raw reports:

- `validation/extreme-cpu-f16/report.json`
- `validation/extreme-vulkan-f16/report.json`

## Other checks

- CTest: decoder behavior and dynamic-length model tests passed. Repeated inference with the same model instance is deterministic for each tested length.
- The dynamic-length model suite also passed on Vulkan, including graph resizing down to a single frame and reuse of an allocated graph.
- The backend test compares 12 matrix shapes against an independent double-precision scalar oracle: odd M/N/K, tile boundaries, the model's projection shapes, batch fallback and a transposed input made contiguous in F32. CPU and Vulkan passed; the final Vulkan maximum error was 0.000018134.
- `fcpe-fusion-tests Vulkan0` passes 180 fused/fallback cases against a separate double scalar oracle: six operations, five channel counts (7/32/360/512/1024), three row counts (1/3/65), and intermediates retained as outputs or not. This covers strided and misaligned GLU views, partial workgroups, LayerNorm with more channels than threads, and output-consumer checks that prevent fusion. Maximum absolute error: 0.000003013. Both project SPIR-V shaders pass `spirv-val --target-env vulkan1.2`.
- Python file-format regressions passed: PCM 8/16/24/32, float 32/64, each in ordinary and extensible WAV; odd unknown RIFF chunks; invalid CLI arguments; NaN audio; malformed GGUF metadata; truncated models; protection against overwriting the input/model.
- The external CMake consumer in `examples/` builds as a separate project and processes the 0.1-second WAV successfully, including the explicit `Threads::Threads` link used by the parallel frontend.
- Native ggml CUDA and Metal build options are connected to upstream ggml, but **not validated here**. The available CUDA 12.8 toolkit rejects the installed MSVC 19.51 compiler; no unsupported-compiler override was used. No macOS host was available. PyTorch CUDA and ORT CUDA were separately benchmarked and checked for numerical agreement; those successful runs do not validate ggml CUDA. See [METAL.md](METAL.md) for the Metal source review and future validation commands.
- A Windows/Linux CPU CI workflow is provided in `.github/workflows/ci.yml`. It has not been run on GitHub from this workspace.

## Timing limits

Controlled warm-run benchmarking is recorded in [PERFORMANCE.md](PERFORMANCE.md), with full per-call samples in its linked JSON archive. It compares the original and optimized ggml graphs, memory/kernel settings, PyTorch CPU/CUDA and ORT CPU/CUDA using identical Mel inputs. All 44 engine/length combinations pass probability error ≤1e-5, F0 error ≤0.02 Hz and zero UV differences. Vulkan first calls include pipeline creation; neither those cold-path diagnostics nor profiler-instrumented runs are mixed into the warm timing table. The native benchmark now also measures an actual WAV-samples-to-F0 loop; it is separate from the host-Mel-to-host-probabilities comparison and excludes file I/O.

## Behavioral corrections

- The original wheel bases its final frame-count adjustment on the input length before resampling. This port uses the resampled length; the reference harness resamples with original torchaudio first and then calls the 16 kHz wheel path to compare equivalent audio.
- Fully unvoiced interpolation returns zeros. The original interpolation code indexes an empty set of voiced points in this case.
- No chunking is used: input GroupNorm normalizes over the complete time axis, so chunking would change the model output.
