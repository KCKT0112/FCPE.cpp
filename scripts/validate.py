#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

"""Execute the original wheel and compare its mel / probabilities / F0 with the C++ CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import torch


def write_wav(path, audio, sr):
    audio = np.asarray(audio, dtype="<f4")
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    raw = audio.tobytes()
    fmt = struct.pack("<HHIIHH", 3, channels, sr, sr * channels * 4, channels * 4, 32)
    path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt " +
                     struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(raw)) + raw)


def error(a, b):
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} versus {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Non-finite comparison input")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cli", type=Path, default=Path("build/bin/fcpe-cli.exe" if os.name == "nt" else "build/bin/fcpe-cli"))
    parser.add_argument("--wheel-dir", type=Path, default=Path("models/torchfcpe-0.0.4"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation/f32"))
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--type", choices=("f32", "f16"), default="f32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wav", type=Path, action="append", default=[], help="Additional real audio recordings")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Keep third-party JIT caches in the writable test output, not site-packages.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(out / "numba-cache"))
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.wheel_dir.resolve()))
    local_deps = Path(__file__).resolve().parents[1] / ".python-deps"
    if local_deps.is_dir():
        sys.path.insert(1, str(local_deps))
    print("Loading the original torchfcpe wheel...", flush=True)
    import torchfcpe
    from torchaudio.transforms import Resample
    checkpoint = torch.load(args.wheel_dir / "torchfcpe/assets/fcpe_c_v001.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config_dict"]
    config["model"].update(conv_dropout=0.0, atten_dropout=0.0)
    reference = torchfcpe.models_infer.InferCFNaiveMelPE(torchfcpe.tools.DotDict(config), checkpoint["model"]).eval()
    print(f"Reference loaded from {torchfcpe.__file__}", flush=True)
    rng = np.random.default_rng(20260906)
    cases = []
    for name, n in (("one_sample", 1), ("short", 159), ("hop_boundary", 160),
                    ("pad_boundary", 433), ("silence", 8000)):
        cases.append((name, np.zeros(n, np.float32), 16000))
    for sr in (8000, 16000, 22050, 44100, 48000):
        t = np.arange(int(sr * 0.75) + 37, dtype=np.float64) / sr
        phase = 2 * np.pi * (130 * t + 65 * t * t)
        x = 0.45 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.07 * np.sin(3 * phase)
        x *= np.minimum(1, t * 30) * np.minimum(1, (t[-1] - t) * 30)
        cases.append((f"harmonic_{sr}", x.astype(np.float32), sr))
    cases.append(("noise", (rng.standard_normal(8147) * 0.03).astype(np.float32), 16000))
    t = np.arange(48037) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    tone[:8000] = 0
    tone[32000:] = 0
    cases.append(("voiced_unvoiced", tone, 16000))
    cases.append(("stereo", np.stack((tone[:12000], tone[:12000] * 0.3), axis=1), 16000))
    for i, path in enumerate(args.wav):
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        cases.append((f"recording_{i}_{path.stem}", audio, sr))

    report = {"model": str(args.model), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "torch": torch.__version__, "backend": args.backend, "type": args.type, "cases": [],
              "note": "Non-16k input is resampled before the original Wav2Mel call to correct the wheel's original-length frame-count bug."}
    failed = []
    prob_limit = 5e-4 if args.type == "f32" else 0.005
    direct_prob_limit = 1e-5 if args.type == "f32" else 0.005
    f0_limit = 0.02 if args.type == "f32" else 0.25
    for name, audio, sr in cases:
        wav = out / f"{name}.wav"
        write_wav(wav, audio, sr)
        mono = audio if audio.ndim == 1 else np.mean(audio.astype(np.float64), axis=1).astype(np.float32)
        with torch.inference_mode():
            x = torch.from_numpy(mono).reshape(1, -1, 1)
            if sr != 16000:
                x = Resample(sr, 16000, lowpass_filter_width=128)(x.squeeze(-1)).unsqueeze(-1)
            ref_mel = reference.wav2mel(x, 16000)
            start = time.perf_counter()
            ref_probs = reference.model(ref_mel)
            ref_ms = (time.perf_counter() - start) * 1000
            ref_f0 = reference.model.cent_to_f0(reference.model.latent2cents_local_decoder(ref_probs, threshold=0.006))
        m = ref_mel[0].numpy()
        p = ref_probs[0].numpy()
        f0 = ref_f0.numpy().reshape(-1)
        m.tofile(out / f"{name}.reference.mel.f32")
        p.tofile(out / f"{name}.reference.probabilities.f32")
        f0.tofile(out / f"{name}.reference.f0.f32")
        row = {"name": name, "sample_rate": sr, "samples": len(mono), "frames": len(m), "torch_ms": ref_ms}
        for mode in ("mel", "wav"):
            prefix = out / f"{name}.{mode}"
            command = [str(args.cli.resolve()), "-m", str(args.model.resolve()), "--backend", args.backend,
                       "-t", str(args.threads), "-o", str(prefix) + ".csv", "--dump-prefix", str(prefix)]
            command += ["--mel", str(out / f"{name}.reference.mel.f32")] if mode == "mel" else ["-i", str(wav)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if process.returncode:
                raise RuntimeError(f"{name} {mode}: {process.stderr}\n{process.stdout}")
            got_m = np.fromfile(str(prefix) + ".mel.f32", dtype="<f4").reshape(-1, 128)
            got_p = np.fromfile(str(prefix) + ".probabilities.f32", dtype="<f4").reshape(-1, 360)
            got_f0 = np.fromfile(str(prefix) + ".f0.f32", dtype="<f4")
            voiced = (f0 > 0) & (got_f0 > 0)
            uv_indices = np.flatnonzero((f0 == 0) != (got_f0 == 0))
            row[mode] = {"mel": error(m, got_m), "mel_magnitude": error(np.exp(m), np.exp(got_m)),
                         "probabilities": error(p, got_p), "f0": error(f0, got_f0),
                         "jointly_voiced_f0": error(f0[voiced], got_f0[voiced]) if np.any(voiced) else None,
                         "uv_mismatch": len(uv_indices),
                         "uv_mismatch_details": [{"frame": int(i), "reference_confidence": float(p[i].max()),
                                                  "cpp_confidence": float(got_p[i].max()),
                                                  "reference_f0": float(f0[i]), "cpp_f0": float(got_f0[i])} for i in uv_indices],
                         "timing": process.stderr.strip()}
            check = row[mode]
            # Float32 FFT implementations differ around the 1e-5 log floor.
            # Also gate on LINEAR mel error and direct-mel network accuracy, so
            # low-energy log amplification cannot hide a frontend or graph bug.
            if (check["mel"]["max_abs"] > 0.02 or check["mel"]["rmse"] > 0.002 or
                    check["mel_magnitude"]["max_abs"] > 1e-5 or
                    check["probabilities"]["max_abs"] > (direct_prob_limit if mode == "mel" else prob_limit) or
                    check["f0"]["max_abs"] > f0_limit or check["uv_mismatch"]):
                failed.append(f"{name}/{mode}")
        report["cases"].append(row)
        print(f"{name}: frames={len(m)} mel={row['wav']['mel']['max_abs']:.3g} "
              f"p={row['wav']['probabilities']['max_abs']:.3g} f0={row['wav']['f0']['max_abs']:.3g} Hz", flush=True)
    report["failed"] = failed
    report["passed"] = not failed
    report["limits"] = {"log_mel_max_abs": 0.02, "log_mel_rmse": 0.002, "linear_mel_max_abs": 1e-5,
                        "direct_probabilities_max_abs": direct_prob_limit, "wav_probabilities_max_abs": prob_limit,
                        "f0_max_abs_hz": f0_limit, "uv_mismatch": 0}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
    print(f"PASS: {len(cases)} cases, direct mel and end-to-end WAV. Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
