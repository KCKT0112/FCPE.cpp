# Sources and licenses

FCPE.cpp's own contributions are licensed under [MPL-2.0](LICENSE). The following upstream grants and notices remain in force; see [NOTICE.md](NOTICE.md) for model provenance and distribution requirements. Full notices are included in `licenses/` so they remain available when dependency checkouts are absent.

- [CNChTu/FCPE](https://github.com/CNChTu/FCPE), MIT, copyright 2023 CN_ChiTu. The architecture, preprocessing and decoder behavior are ported from the official `torchfcpe-0.0.4` wheel. Its MIT license is retained verbatim from the wheel in [licenses/FCPE-MIT.txt](licenses/FCPE-MIT.txt). The checkpoint is downloaded from the release, not committed to this source tree. The wheel declares MIT and contains the checkpoint; no separate checkpoint license was present in the inspected wheel. Converted GGUF assets retain the upstream terms rather than this port's MPL.
- [KakaruHayate/game.cpp](https://github.com/KakaruHayate/game.cpp) and [KakaruHayate/pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp), MPL-2.0: project organization, CMake / ggml / GGUF integration references. Their inference implementations and backend patches are not copied into this project.
- [ggml](https://github.com/ggml-org/ggml), MIT. Pinned to v0.19.0, commit `30bf8685ed4eb0a47f2b06229543327749904150`. CMake can download the pinned archive or use a local source checkout. The default Vulkan build generates a source copy selecting full-F32 arithmetic, preferring non-host-visible device memory on discrete GPUs, tuning the upstream F32 `MUL_MAT` shader and recognizing FCPE fusion patterns. Integration lives in `cmake/VulkanPrecision.cmake`, `cmake/VulkanMatmul.cmake`, `cmake/VulkanFusion.cmake` and `src/vulkan/fcpe-fusion.inc`. The project GLSL sources are `src/vulkan/fcpe-fused.comp` and the retained legacy `src/vulkan/fcpe-matmul-f32.comp`. No ggml checkout files are changed. See `third_party/ggml/LICENSE` in the fetched checkout.
- [pocketfft](https://github.com/mreineck/pocketfft), BSD-3-Clause. `third_party/pocketfft_hdronly.h` is the unmodified header distributed by the pc-nsf-hifigan.cpp reference checkout; its complete copyright and license notice is at the top of that file and in [licenses/pocketfft-BSD-3-Clause.txt](licenses/pocketfft-BSD-3-Clause.txt). This header is not MPL-licensed.
- [librosa](https://github.com/librosa/librosa), ISC. FCPE's `mel_fn_librosa.py` identifies `librosa.filters` as its source; the converter implements the corresponding Slaney filterbank. The license from librosa 0.10.1 is retained in [licenses/librosa-ISC.txt](licenses/librosa-ISC.txt).
- The optional real-speech validation sample is [whisper.cpp/samples/jfk.wav](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav); it is a test artifact in `validation/`, not part of the source distribution.

Python dependencies are used only to convert and validate the model. Native inference links ggml and embeds pocketfft; it does not invoke Python, PyTorch, or ONNX Runtime.

The pinned ggml MIT notice is also retained in [licenses/ggml-MIT.txt](licenses/ggml-MIT.txt). Generated Vulkan source copies retain upstream notices. This project's build patches and fusion code use MPL-2.0; the reference vocoder's separately dual-licensed patch directory and its model-specific non-commercial conditions are not imported here.

Reference checkouts inspected on 2026-09-06:

| Repository | Commit |
| --- | --- |
| game.cpp | `ffd3e20b307a6bf9cc1797a6551d68955dc0b656` |
| pc-nsf-hifigan.cpp | `e46c93308f0bbec36e1d08707f94250c79abab56` |
| FCPE | `6a149c1afb1c7e7821b71869dfb31ad50c95b516` |

The released wheel, rather than the latest FCPE checkout, is the numerical reference.
