# Second optimization round, 2026-09-06

Ryzen 7 5700X / RTX 4060, Windows, strict F32. The full analysis and reproduction commands are in [PERFORMANCE.md](../../PERFORMANCE.md).

| Artifact | Contents |
| --- | --- |
| [benchmark.json](benchmark.json) | All 11 engines × 4 lengths, 3 warmups + 20 timed samples, numerical checks, input hashes, executable/DLL hashes, actual native end-to-end samples |
| [all-results.md](all-results.md) | Network median, P95, realtime factor and accuracy for every engine/length |
| [validation.json](validation.json) | Four complete 14-recording suites: F32 CPU/Vulkan pass; F16 CPU/Vulkan retain the known JFK frame-885 UV failures |
| [memory-ablation.json](memory-ablation.json) | Same-graph/same-kernel memory comparison; separate instrumented diagnostics explicitly marked |
| [device-local-memory.log](device-local-memory.log) / [host-visible-memory.log](host-visible-memory.log) | Actual allocation memory types, heaps and property flags |
| [final-profile.log](final-profile.log) | Final Vulkan operation groups and instrumented timing; excluded from fair benchmark |
| [ort-placement.json](ort-placement.json) | Separate ORT profiler: 81 CUDA nodes, no CPU compute nodes |
| [tuning.json](tuning.json) | Exploratory tile/layout scans and raw samples; not the formal benchmark, not all experiments retained |
| [provenance.json](provenance.json) | Source hashes, environment, command and verification record |
| [diagnostic-recipe.py](diagnostic-recipe.py) | Copy to the repository's `build/run_extreme_diagnostics.py` and execute to rerun the isolated memory/profile experiments |

The [first-round archive](../2026-09-06-windows/benchmark.json) is preserved unchanged. Native baseline configurations in this new archive reproduce their old graph/kernel/memory behavior using the current binary; their WAV end-to-end timings use the current faster frontend. The formal comparison across frameworks is host F32 Mel to host F32 probabilities, with GPU transfers and synchronization included. Only native engines have separate WAV-samples-to-F0 timings; neither comparison includes disk I/O.

Large models, WAVs, output tensors and the raw ONNX profiler remain under `models/` and `validation/` rather than this portable metrics archive. Input generation from the original wheel and benchmark source WAV is documented; hashes identify the exact inputs used here.

The source and binary hashes in `provenance.json` / `benchmark.json` identify the files at measurement time, before publication added MPL headers, distribution notice installation and Git line-ending normalization. Historical hashes and timing samples are retained rather than rewritten to imply a new benchmark run. Those publication edits do not change inference algorithms; see the repository's commit history for the published sources.
