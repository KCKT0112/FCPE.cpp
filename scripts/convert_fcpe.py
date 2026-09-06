#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.
# Portions derived from FCPE: Copyright (c) 2023 CN_ChiTu.
# Upstream MIT notice: licenses/FCPE-MIT.txt; see NOTICE.md.

"""Convert the official torchfcpe 0.0.4 wheel (or its checkpoint) to GGUF."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import gguf
import numpy as np
import torch

ASSET = "torchfcpe/assets/fcpe_c_v001.pt"


def mel_basis(sr, n_fft, n_mels, fmin, fmax):
    # Slaney, matching torchfcpe/mel_fn_librosa.py (including float32 rounding).
    def hz_to_mel(hz):
        return hz / (200.0 / 3) if hz < 1000 else 15 + np.log(hz / 1000) / (np.log(6.4) / 27)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = (200.0 / 3) * mels
    idx = mels >= 15
    hz[idx] = 1000 * np.exp((np.log(6.4) / 27) * (mels[idx] - 15))
    ramps = hz[:, None] - np.fft.rfftfreq(n_fft, 1.0 / sr)[None, :]
    diffs = np.diff(hz)
    basis = np.maximum(0, np.minimum(-ramps[:-2] / diffs[:-1, None],
                                     ramps[2:] / diffs[1:, None])).astype(np.float32)
    basis *= (2 / (hz[2:] - hz[:-2]))[:, None]
    return basis


def convert(source, output, dtype):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the checkpoint / wheel")
    raw = source.read_bytes()
    if source.suffix == ".whl":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            checkpoint_bytes = archive.read(ASSET)
    else:
        checkpoint_bytes = raw
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    config, state = checkpoint["config_dict"], checkpoint["model"]
    model, mel = config["model"], config["mel"]
    if model["type"] != "CFNaiveMelPE" or not model.get("conv_only", False):
        raise ValueError("Only the released CFNaiveMelPE conv_only architecture is supported")
    if model.get("use_harmonic_emb", False) or mel.get("type", "default") not in (None, "default"):
        raise ValueError("Harmonic embeddings and STFT-only models are not supported")
    if mel["n_fft"] != mel["win_size"] or mel["win_size"] < mel["hop_size"]:
        raise ValueError("Expected n_fft == win_size >= hop_size")
    hidden, layers, bins = model["hidden_dims"], model["n_layers"], model["out_dims"]
    tensors = {}
    used = set()

    def take(name, shape, target=None, squeeze=False):
        value = state[name].detach().float().cpu()
        if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid tensor {name}: {tuple(value.shape)}, expected {shape}")
        used.add(name)
        tensors[target or name] = value.squeeze(-1).numpy() if squeeze else value.numpy()

    for i, channels in ((0, mel["num_mels"]), (3, hidden)):
        take(f"input_stack.{i}.weight", (hidden, channels, 3))
        take(f"input_stack.{i}.bias", (hidden,))
    for suffix in ("weight", "bias"):
        take(f"input_stack.1.{suffix}", (hidden,))
        take(f"norm.{suffix}", (hidden,))
    for i in range(layers):
        src = f"net.encoder_layers.{i}.conformer.net."
        dst = f"block.{i}."
        for suffix in ("weight", "bias"):
            take(src + "0." + suffix, (hidden,), dst + "norm." + suffix)
        take(src + "2.weight", (4 * hidden, hidden, 1), dst + "in.weight", squeeze=True)
        take(src + "2.bias", (4 * hidden,), dst + "in.bias")
        take(src + "4.conv.weight", (2 * hidden, 1, 31), dst + "dw.weight")
        take(src + "4.conv.bias", (2 * hidden,), dst + "dw.bias")
        take(src + "6.weight", (hidden, 2 * hidden, 1), dst + "out.weight", squeeze=True)
        take(src + "6.bias", (hidden,), dst + "out.bias")
    take("output_proj.bias", (bins,))
    if "output_proj.weight_g" in state:
        g_name, v_name = "output_proj.weight_g", "output_proj.weight_v"
    else:
        g_name = "output_proj.parametrizations.weight.original0"
        v_name = "output_proj.parametrizations.weight.original1"
    g, v = state[g_name].float(), state[v_name].float()
    if tuple(g.shape) != (bins, 1) or tuple(v.shape) != (bins, hidden):
        raise ValueError("Invalid output weight normalization shapes")
    weight = torch._weight_norm(v, g, 0)  # identical to the original weight_norm forward
    if not torch.isfinite(weight).all():
        raise ValueError("Output weight normalization produced non-finite values")
    tensors["output_proj.weight"] = weight.numpy()
    used.update((g_name, v_name))
    take("cent_table", (bins,))
    if not np.all(np.diff(tensors["cent_table"]) > 0):
        raise ValueError("cent_table must be strictly increasing")
    ignored = {"gaussian_blurred_cent_mask"}
    ignored.update(f"net.encoder_layers.{i}.norm.{s}" for i in range(layers) for s in ("weight", "bias"))
    unexpected = set(state) - used - ignored
    if unexpected:
        raise ValueError(f"Unexpected weights (unsupported architecture): {sorted(unexpected)}")
    tensors["mel.basis"] = mel_basis(mel["sr"], mel["n_fft"], mel["num_mels"],
                                     mel.get("fmin", 0), mel.get("fmax", mel["sr"] / 2))
    tensors["mel.window"] = torch.hann_window(mel["win_size"], periodic=True).numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    writer = gguf.GGUFWriter(str(temp), "fcpe")
    writer.add_name("FCPE conv-only v001")
    writer.add_uint32("fcpe.version", 1)
    writer.add_string("fcpe.source.sha256", hashlib.sha256(raw).hexdigest())
    writer.add_string("fcpe.checkpoint.sha256", hashlib.sha256(checkpoint_bytes).hexdigest())
    writer.add_string("fcpe.config", json.dumps(config, ensure_ascii=False))
    writer.add_string("fcpe.weight_type", dtype)
    for name, value in {"sample_rate": mel["sr"], "hop_size": mel["hop_size"],
                        "n_fft": mel["n_fft"], "win_size": mel["win_size"], "mel_bins": mel["num_mels"],
                        "hidden_dims": hidden, "layers": layers, "bins": bins}.items():
        writer.add_uint32("fcpe." + name, value)
    for name in ("f0_min", "f0_max"):
        writer.add_float32("fcpe." + name, model[name])
    for name, value in tensors.items():
        # Keep normalization, depthwise kernels and DSP constants in F32.
        half = dtype == "f16" and name.endswith("weight") and value.ndim >= 2 and ".dw." not in name
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float16 if half else np.float32))
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    temp.replace(output)
    report = {"source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
              "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
              "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "bytes": output.stat().st_size, "type": dtype, "tensors": len(tensors),
              "config": {"mel": mel, "model": model}}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}: {len(tensors)} tensors, {report['bytes'] / 1048576:.2f} MiB ({dtype})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official wheel or extracted .pt checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--type", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    convert(args.source, args.output, args.type)
