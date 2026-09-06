#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

"""Export FCPE to ONNX and benchmark ggml, PyTorch and ONNX Runtime identically.

All network measurements start/end with host float32 arrays. GPU timings include
H2D + forward + D2H and synchronize. Each engine runs in a separate process.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ["ggml_reference_cpu", "ggml_cpu", "ggml_reference_vulkan", "ggml_direct_vulkan",
           "ggml_previous_vulkan", "ggml_local_vulkan", "ggml_vulkan",
           "torch_cpu", "torch_cuda", "ort_cpu", "ort_cuda",
           "ggml_reference_metal", "ggml_metal", "torch_mps"]
DEFAULT_ENGINES = (["ggml_reference_cpu", "ggml_cpu", "ggml_reference_metal", "ggml_metal",
                    "torch_cpu", "torch_mps", "ort_cpu"] if sys.platform == "darwin" else ENGINES[:-3])


def setup_paths(out):
    os.environ.setdefault("NUMBA_CACHE_DIR", str(out / "numba-cache"))
    sys.path.insert(0, str(ROOT / "models/torchfcpe-0.0.4"))
    if (ROOT / ".python-deps").is_dir():
        sys.path.insert(1, str(ROOT / ".python-deps"))


def reference_model(threads):
    import torch
    import torchfcpe
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ckpt = torch.load(ROOT / "models/torchfcpe-0.0.4/torchfcpe/assets/fcpe_c_v001.pt", map_location="cpu", weights_only=True)
    ckpt["config_dict"]["model"].update(conv_dropout=0.0, atten_dropout=0.0)
    model = torchfcpe.models_infer.InferCFNaiveMelPE(torchfcpe.tools.DotDict(ckpt["config_dict"]), ckpt["model"]).eval()
    return model


def prepare(args):
    import torch
    import onnx
    import soundfile as sf
    from validate import write_wav
    reference = reference_model(args.threads)
    audio, sr = sf.read(args.wav, dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("Benchmark source must be mono 16 kHz WAV")
    cases = []
    for duration in args.seconds:
        n = int(duration * 16000)
        source = np.resize(audio, max(n + 16000, len(audio)))
        x = np.ascontiguousarray(source[16000:16000 + n], dtype=np.float32)
        name = f"{duration:g}s"
        wav_path = args.output_dir / f"{name}.wav"
        write_wav(wav_path, x, sr)
        with torch.inference_mode():
            mel = reference.wav2mel(torch.from_numpy(x).reshape(1, -1, 1), sr)
            p = reference.model(mel)[0].numpy()
            f0 = reference.model.cent_to_f0(reference.model.latent2cents_local_decoder(
                torch.from_numpy(p)[None], threshold=0.006))[0].numpy().reshape(-1)
        mel[0].numpy().tofile(args.output_dir / f"{name}.mel.f32")
        p.tofile(args.output_dir / f"{name}.reference.f32")
        f0.tofile(args.output_dir / f"{name}.f0.f32")
        cases.append({"name": name, "seconds": duration, "frames": int(mel.shape[1]),
                      "mel_sha256": hashlib.sha256((args.output_dir / f"{name}.mel.f32").read_bytes()).hexdigest()})
    onnx_path = args.output_dir / "fcpe.onnx"
    with torch.inference_mode():
        torch.onnx.export(reference.model, torch.zeros(1, 101, 128), str(onnx_path),
                          input_names=["mel"], output_names=["probabilities"],
                          dynamic_axes={"mel": {1: "frames"}, "probabilities": {1: "frames"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(onnx_path))
    np.asarray(reference.model.cent_table).tofile(args.output_dir / "cents.f32")
    manifest = {"source_wav": str(args.wav), "source_sha256": hashlib.sha256(args.wav.read_bytes()).hexdigest(),
                "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(), "cases": cases}
    (args.output_dir / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(cases)} cases and checked {onnx_path}", flush=True)


def stats(samples):
    return {"mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples}


def worker(args):
    import torch
    manifest = json.loads((args.output_dir / "inputs.json").read_text())
    engine = args.worker
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__}
    versions["torch_cuda_runtime"] = torch.version.cuda
    versions["cudnn"] = torch.backends.cudnn.version()
    start = time.perf_counter()
    handles = []
    if engine.startswith("torch"):
        reference = reference_model(args.threads)
        device = engine.removeprefix("torch_")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS unavailable")
        if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for an MPS benchmark")
        model = reference.model.to(device)
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        load_ms = (time.perf_counter() - start) * 1000
        def forward(x):
            tensor = torch.from_numpy(x).to(device)
            out = model(tensor).cpu().numpy()
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            return out
    else:
        if (ROOT / ".ort-deps").is_dir():
            sys.path.insert(0, str(ROOT / ".ort-deps"))
        import onnxruntime as ort
        versions["onnxruntime"] = ort.__version__
        if os.name == "nt" and engine.endswith("cuda"):
            cuda = Path(args.cuda_dir)
            torch_lib = Path(torch.__file__).parent / "lib"
            for directory in (cuda / "bin", torch_lib):
                handles.append(os.add_dll_directory(str(directory)))
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(cuda=True, cudnn=False, directory=str(cuda / "bin"))
                ort.preload_dlls(cuda=False, cudnn=True, directory=str(torch_lib))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if args.profile_ort:
            opts.enable_profiling = True
            opts.profile_file_prefix = str(args.output_dir / f"{engine}-profile")
        providers = ["CPUExecutionProvider"]
        if engine.endswith("cuda"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime CUDA provider unavailable")
            providers = [("CUDAExecutionProvider", {"use_tf32": "0", "cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        session = ort.InferenceSession(str(args.output_dir / "fcpe.onnx"), opts, providers=providers)
        if engine.endswith("cuda") and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime silently fell back from CUDA")
        versions["providers"] = session.get_providers()
        versions["provider_options"] = session.get_provider_options()
        load_ms = (time.perf_counter() - start) * 1000
        def forward(x):
            return session.run(None, {"mel": x})[0]
    rows = []
    with torch.inference_mode():
        for case in manifest["cases"]:
            x = np.fromfile(args.output_dir / f"{case['name']}.mel.f32", dtype="<f4").reshape(1, -1, 128)
            start = time.perf_counter(); output = forward(x); first_ms = (time.perf_counter() - start) * 1000
            for _ in range(args.warmup):
                output = forward(x)
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter(); output = forward(x); samples.append((time.perf_counter() - start) * 1000)
            if not args.profile_ort:
                output.tofile(args.output_dir / f"{engine}-{case['name']}.f32")
            rows.append({**case, **stats(samples), "first_ms": first_ms})
            print(engine, case["name"], f"{rows[-1]['median_ms']:.3f} ms", flush=True)
    report = {"engine": engine, "versions": versions, "load_ms": load_ms, "cases": rows,
              "threads": args.threads, "warmup": args.warmup, "repeats": args.repeats, "tf32": False}
    if engine.startswith("ort") and args.profile_ort:
        path = Path(session.end_profiling())
        nodes = {}
        for event in json.loads(path.read_text()):
            provider = event.get("args", {}).get("provider")
            if provider and event.get("cat") == "Node":
                nodes.setdefault(provider, set()).add(event["name"])
        report["profile"] = {"file": str(path), "unique_nodes_by_provider": {k: len(v) for k, v in nodes.items()},
                             "nodes_by_provider": {k: sorted(v) for k, v in nodes.items()}}
        # Keep instrumented runs out of the fair benchmark data.
        (args.output_dir / f"{engine}-profile-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    (args.output_dir / f"{engine}.json").write_text(json.dumps(report, indent=2) + "\n")


def compare(output, reference, cents):
    if output.shape != reference.shape or not np.isfinite(output).all():
        raise ValueError("Invalid benchmark output")
    diff = output.astype(np.float64) - reference
    def f0(p):
        peak = p.argmax(-1)
        indices = np.clip(peak[:, None] + np.arange(-4, 5), 0, len(cents) - 1)
        local = np.take_along_axis(p, indices, -1)
        value = 10 * np.exp2((np.sum(cents[indices] * local, axis=-1) / local.sum(-1)) / 1200)
        return np.where(p.max(-1) > 0.006, value, 0)
    a, b = f0(reference), f0(output)
    return {"probability_max_abs": float(np.max(abs(diff))), "probability_rmse": float(np.sqrt(np.mean(diff ** 2))),
            "f0_max_abs_hz": float(np.max(abs(a - b))), "uv_mismatch": int(np.sum((a == 0) != (b == 0)))}


def run(args):
    manifest = json.loads((args.output_dir / "inputs.json").read_text())
    results = []
    for engine in args.engines:
        if args.summarize:
            report = json.loads((args.output_dir / f"{engine}.json").read_text())
            if (report["threads"], report["warmup"], report["repeats"]) != (args.threads, args.warmup, args.repeats):
                raise ValueError(f"Mismatched measurement settings in {engine}.json")
            if [(c["name"], c["mel_sha256"]) for c in report["cases"]] != [(c["name"], c["mel_sha256"]) for c in manifest["cases"]]:
                raise ValueError(f"Mismatched inputs in {engine}.json")
            results.append(report)
            continue
        if engine.startswith("ggml"):
            rows = []
            for case in manifest["cases"]:
                name = case["name"]
                output = args.output_dir / f"{engine}-{name}.json"
                command = [str(args.bench.resolve()), "--model", str(args.model.resolve()),
                           "--mel", str(args.output_dir / f"{name}.mel.f32"), "--output", str(output),
                           "--dump", str(args.output_dir / f"{engine}-{name}.f32"),
                           "--backend", ("Vulkan0" if engine.endswith("vulkan") else
                                         args.metal_backend if engine.endswith("metal") else "cpu"),
                           "--threads", str(args.threads), "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                           "--wav", str(args.output_dir / f"{name}.wav")]
                if "reference" in engine:
                    command.append("--reference-graph")
                environment = os.environ.copy()
                for key in list(environment):
                    if key.startswith(("FCPE_VK_", "GGML_VK_", "FCPE_METAL_", "GGML_METAL_")):
                        environment.pop(key)
                if engine in ("ggml_reference_vulkan", "ggml_direct_vulkan", "ggml_previous_vulkan"):
                    environment["FCPE_VK_HOST_VISIBLE"] = "1"
                if engine in ("ggml_reference_vulkan", "ggml_direct_vulkan", "ggml_previous_vulkan", "ggml_local_vulkan"):
                    environment["FCPE_VK_LEGACY_GRAPH"] = "1"
                    environment["FCPE_VK_DISABLE_FCPE_FUSION"] = "1"
                if engine in ("ggml_reference_vulkan", "ggml_direct_vulkan", "ggml_local_vulkan"):
                    environment["FCPE_VK_DISABLE_TILED_F32"] = "1"
                proc = subprocess.run(command, capture_output=True, text=True, timeout=600, env=environment)
                (args.output_dir / f"{engine}-{name}.log").write_text(proc.stderr + proc.stdout)
                if proc.returncode:
                    raise RuntimeError(proc.stderr + proc.stdout)
                row = json.loads(output.read_text())
                if engine.endswith("metal") and not row["accelerator_compute_nodes"]:
                    raise RuntimeError("Metal benchmark did not execute any accelerator nodes")
                row.update(stats(row["samples_ms"]))
                if row.get("end_to_end_samples_ms"):
                    row["end_to_end"] = stats(row["end_to_end_samples_ms"])
                rows.append({**case, **row})
                print(engine, name, f"{row['median_ms']:.3f} ms", flush=True)
            report = {"engine": engine, "cases": rows, "threads": args.threads, "warmup": args.warmup, "repeats": args.repeats,
                      "graph": "im2col" if "reference" in engine else "strided_glu_fused" if engine=="ggml_vulkan" else "direct_depthwise",
                      "vulkan_memory": "host_visible_device_local" if engine in ("ggml_reference_vulkan","ggml_direct_vulkan","ggml_previous_vulkan") else "device_local" if engine.endswith("vulkan") else None,
                      "vulkan_matmul": "tiled_f32" if engine=="ggml_previous_vulkan" else "upstream_strict_f32_tuned_on_nvidia" if engine=="ggml_vulkan" else "upstream_strict_f32" if engine.endswith("vulkan") else None,
                      "gguf_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      "executable_sha256": hashlib.sha256(args.bench.read_bytes()).hexdigest(),
                      "runtime_library_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in sorted({p.resolve() for pattern in ("*.dll", "*.dylib", "*.so*")
                                                                  for p in args.bench.parent.glob(pattern) if p.is_file()})}}
            (args.output_dir / f"{engine}.json").write_text(json.dumps(report, indent=2) + "\n")
        else:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", engine,
                            "--output-dir", str(args.output_dir), "--threads", str(args.threads),
                            "--warmup", str(args.warmup), "--repeats", str(args.repeats), "--cuda-dir", args.cuda_dir], check=True)
            report = json.loads((args.output_dir / f"{engine}.json").read_text())
        results.append(report)
    cents = np.fromfile(args.output_dir / "cents.f32", dtype="<f4")
    for report in results:
        for row in report["cases"]:
            name = row["name"]
            output = np.fromfile(args.output_dir / f"{report['engine']}-{name}.f32", dtype="<f4").reshape(-1, 360)
            reference = np.fromfile(args.output_dir / f"{name}.reference.f32", dtype="<f4").reshape(-1, 360)
            row["accuracy"] = compare(output, reference, cents)
            row["realtime_factor"] = row["median_ms"] / (row["seconds"] * 1000)
    summary = {"measurement": "host float32 Mel -> probabilities on host; GPU transfers and synchronization included",
               "platform": platform.platform(), "processor": platform.processor(),
               "threads": args.threads, "warmup": args.warmup, "repeats": args.repeats,
               "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(), "inputs": manifest, "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| Engine | Audio seconds | Median ms | P95 ms | RTF | Max probability error | UV differences |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for report in results:
        for r in report["cases"]:
            lines.append(f"| {report['engine']} | {r['seconds']:g} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} | "
                         f"{r['realtime_factor']:.5f} | {r['accuracy']['probability_max_abs']:.3g} | {r['accuracy']['uv_mismatch']} |")
    (args.output_dir / "table.md").write_text("\n".join(lines) + "\n")
    failures = [f"{r['engine']}/{c['name']}" for r in results for c in r["cases"]
                if c["accuracy"]["probability_max_abs"] > 1e-5 or c["accuracy"]["uv_mismatch"] or
                c["accuracy"]["f0_max_abs_hz"] > 0.02]
    if failures:
        raise SystemExit("Benchmark correctness check failed (reports retained): " + ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="Recheck and aggregate existing engine reports, without timing again")
    parser.add_argument("--worker", choices=("torch_cpu", "torch_cuda", "torch_mps", "ort_cpu", "ort_cuda"))
    parser.add_argument("--profile-ort", action="store_true", help="Instrument an ORT worker in a separate diagnostic run")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation/benchmark")
    parser.add_argument("--wav", type=Path, default=ROOT / "validation/jfk.wav")
    parser.add_argument("--model", type=Path, default=ROOT / "models/fcpe-f32.gguf")
    parser.add_argument("--bench", type=Path, default=ROOT / ("build-metal/bin/fcpe-bench" if sys.platform == "darwin" else
                                                           "build-vulkan/bin/fcpe-bench.exe" if os.name == "nt" else
                                                           "build-vulkan/bin/fcpe-bench"))
    parser.add_argument("--metal-backend", default="MTL0", help="Metal device name from fcpe-cli --list-backends")
    parser.add_argument("--cuda-dir", default="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    parser.add_argument("--seconds", type=float, nargs="+", default=[0.1, 1, 3, 11])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=DEFAULT_ENGINES)
    args = parser.parse_args()
    if args.profile_ort and (not args.worker or not args.worker.startswith("ort")):
        parser.error("--profile-ort requires --worker ort_cpu or ort_cuda")
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1 or any(s <= 0 for s in args.seconds):
        parser.error("repeats, threads and durations must be positive; warmup must be nonnegative")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_paths(args.output_dir)
    if args.prepare:
        prepare(args)
    elif args.worker:
        worker(args)
    else:
        run(args)
