# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

"""One-off diagnostic collection, run after the uninstrumented formal suite."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / "validation/benchmark-extreme"
dest = out / "diagnostics"
dest.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("benchmark", root / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
base = {k: v for k, v in os.environ.items() if not k.startswith(("FCPE_VK_", "GGML_VK_"))}
legacy = {"FCPE_VK_LEGACY_GRAPH": "1", "FCPE_VK_DISABLE_FCPE_FUSION": "1"}
reference = np.fromfile(out / "11s.reference.f32", dtype="<f4").reshape(-1, 360)
cents = np.fromfile(out / "cents.f32", dtype="<f4")
results = []
configs = [
    ("host-visible-legacy", {**legacy, "FCPE_VK_HOST_VISIBLE": "1"}, 3, 20),
    ("device-local-legacy", {**legacy, "FCPE_VK_TILED_F32": "1"}, 3, 20),
    ("device-local-upstream", {**legacy, "FCPE_VK_DISABLE_TILED_F32": "1"}, 3, 20),
    ("device-local-memory", {"FCPE_VK_LOG_MEMORY_TYPE": "1"}, 0, 1),
    ("host-visible-memory", {"FCPE_VK_LOG_MEMORY_TYPE": "1", "FCPE_VK_HOST_VISIBLE": "1"}, 0, 1),
    ("final-profile", {"GGML_VK_PERF_LOGGER": "1"}, 1, 1),
]
for name, overrides, warmup, repeats in configs:
    cmd = [str(root / "build-vulkan/bin/fcpe-bench.exe"), "--model", str(root / "models/fcpe-f32.gguf"),
           "--mel", str(out / "11s.mel.f32"), "--output", str(dest / f"{name}.json"),
           "--dump", str(dest / f"{name}.f32"), "--backend", "Vulkan0", "--threads", "4",
           "--warmup", str(warmup), "--repeats", str(repeats)]
    proc = subprocess.run(cmd, env={**base, **overrides}, capture_output=True, text=True, check=True)
    (dest / f"{name}.log").write_text(proc.stderr + proc.stdout, encoding="utf-8")
    report = json.loads((dest / f"{name}.json").read_text())
    accuracy = benchmark.compare(np.fromfile(dest / f"{name}.f32", dtype="<f4").reshape(-1, 360), reference, cents)
    assert accuracy["probability_max_abs"] <= 1e-5 and accuracy["uv_mismatch"] == 0 and accuracy["f0_max_abs_hz"] <= .02
    results.append({"name": name, "environment": overrides, "accuracy": accuracy,
                    "instrumented": warmup != 3, **report})
    print(name, report["median_ms"], accuracy, flush=True)
summary = {"purpose": "Isolate memory type / matmul choice and inspect final GPU dispatches; separate from formal suite",
           "mel_sha256": hashlib.sha256((out / "11s.mel.f32").read_bytes()).hexdigest(), "results": results}
(dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
