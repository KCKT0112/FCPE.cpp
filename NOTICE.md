# Notices, provenance and distribution

FCPE.cpp is an independent C++17 / ggml port of [CNChTu/FCPE](https://github.com/CNChTu/FCPE). It is not an official FCPE release and is not endorsed by the upstream authors. FCPE, ggml and other project names identify their respective upstream projects; this repository grants no trademark rights.

## Project license

The first-party source code, build integration, tools, tests and documentation in this repository are made available under the **Mozilla Public License 2.0**, except where a file or the third-party notices below state otherwise. The full text is in [LICENSE](LICENSE).

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at <https://mozilla.org/MPL/2.0/>.

MPL-2.0 applies to this port's contributions. It does not replace or revoke the original MIT / BSD / ISC grants on upstream material. Preserve the relevant copyright notices, license terms and warranty disclaimers when redistributing it.

## FCPE source and model provenance

| Item | Source |
| --- | --- |
| Original project / author | [CNChTu/FCPE](https://github.com/CNChTu/FCPE), CN_ChiTu |
| Numerical reference | Official `torchfcpe` **0.0.4** wheel, not an arbitrary checkout of the latest training code |
| Release | [FCPE v0.0.4](https://github.com/CNChTu/FCPE/releases/tag/v0.0.4) |
| Exact download | [torchfcpe-0.0.4-py3-none-any.whl](https://github.com/CNChTu/FCPE/releases/download/v0.0.4/torchfcpe-0.0.4-py3-none-any.whl) |
| Checkpoint within the wheel | `torchfcpe/assets/fcpe_c_v001.pt` |
| Upstream copyright | Copyright (c) 2023 CN_ChiTu |
| Upstream license | MIT; retained verbatim in [licenses/FCPE-MIT.txt](licenses/FCPE-MIT.txt) from the wheel's `.dist-info/LICENSE` |
| Wheel SHA256 | `f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d` |

The model architecture, Mel preprocessing and pitch decoder were ported from that release. Python scripts load the original extracted implementation for parity tests. The native runtime does not invoke Python or import the upstream package. Implementation differences and numerical limits are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

The official wheel declares `License: MIT` and includes the checkpoint in that distribution; no separate checkpoint license was present in the inspected wheel. The checkpoint and converted GGUF files remain **upstream model assets**, not MPL-licensed project code. Converting a checkpoint to GGUF does not create a new license for it. Retain the FCPE attribution and MIT notice when redistributing these assets, and follow any applicable asset-specific terms for other models you choose to convert. This port's license does not grant rights to unrelated models, recordings or datasets.

Weights, the original wheel, extracted Python package and validation audio are downloaded locally and are **not committed to this repository**. The converter records provenance and hashes. If distributing converted weights separately, include their provenance, precision, conversion settings and applicable upstream license alongside the assets.

## Third-party material

| Component | License | Relationship and retained notice |
| --- | --- | --- |
| FCPE | MIT | Architecture / preprocessing / decoder port; [FCPE-MIT.txt](licenses/FCPE-MIT.txt) |
| ggml v0.19.0 | MIT | Tensor engine and backend source used by generated Vulkan build copies; [ggml-MIT.txt](licenses/ggml-MIT.txt) |
| pocketfft | BSD-3-Clause | Unmodified vendored `third_party/pocketfft_hdronly.h`; its header retains the notice, also copied to [pocketfft-BSD-3-Clause.txt](licenses/pocketfft-BSD-3-Clause.txt) for binary distributions |
| librosa | ISC | Slaney Mel filterbank reference through FCPE's `mel_fn_librosa.py`; [librosa-ISC.txt](licenses/librosa-ISC.txt) |

The MPL does not relicense the unmodified pocketfft header or downloaded dependency checkouts. Vulkan integration retains the ggml license in generated source copies; this project's changes and fusion kernels are covered by MPL-2.0. There is no separately dual-licensed patch directory in this repository.

[KakaruHayate/game.cpp](https://github.com/KakaruHayate/game.cpp) and [KakaruHayate/pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) were references for project organization, ggml/GGUF integration and attribution structure. Their inference implementations and backend patch sets are not vendored here. The pocketfft header was obtained from the latter's vendored copy and retains pocketfft's BSD license. The NSF-HiFiGAN / DiffSinger model assets and their non-commercial terms are not used or imported into FCPE.cpp.

See [THIRD_PARTY.md](THIRD_PARTY.md) for exact inspected commits and dependency sources. Python conversion / validation packages are separate development dependencies; if you bundle them, preserve their own licenses too.

## Distribution reminders

- Keep [LICENSE](LICENSE), this notice, [THIRD_PARTY.md](THIRD_PARTY.md) and the applicable files in `licenses/` with redistributions. CMake installs these files under `share/doc/fcpe` by default.
- When distributing MPL-covered executable code, make the corresponding MPL-covered Source Code Form, including your modifications, available as required by MPL-2.0 §3.2 and tell recipients how to obtain it. This project's source is <https://github.com/KCKT0112/FCPE.cpp>; modified builds must identify the corresponding modified source, not just this upstream URL.
- MPL-2.0 uses file-level copyleft. It permits a Larger Work under other terms provided that the MPL-covered files and notices continue to meet the license requirements. The license text controls the exact obligations.
- Preserve upstream terms independently of the code license when distributing weights or other assets.

## Warranty and validation scope

The software is provided **as is**, without warranty, subject to the disclaimers and limitations in the applicable licenses. Accuracy and speed figures describe the tested model, inputs, hardware and software versions, not a guarantee for all recordings or devices. F16 storage can change a voiced/unvoiced decision near the confidence threshold. CPU and Vulkan were validated on Windows; Metal and native ggml CUDA were not validated in that environment. Consult the recorded [accuracy results](docs/VALIDATION.md), [benchmarks](docs/PERFORMANCE.md) and [Metal notes](docs/METAL.md) before relying on another configuration.
