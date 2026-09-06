#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

"""Regression checks for WAV encodings, CLI failures, and malformed model metadata."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import gguf
import numpy as np

CLI = None
MODEL = None


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fcpe-test-")
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args, good=True):
        process = subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True, timeout=60)
        if good:
            self.assertEqual(process.returncode, 0, process.stderr)
        else:
            # A clean C++ exception, not a ggml assertion or access violation.
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertIn("fcpe:", process.stderr)
        return process

    def wav(self, name, payload, bits, format=1, extensible=False):
        fmt = struct.pack("<HHIIHH", 0xfffe if extensible else format, 1, 16000, 16000 * bits // 8, bits // 8, bits)
        if extensible:
            fmt += struct.pack("<HHI", 22, bits, 4) + struct.pack("<IHH", format, 0, 16) + bytes.fromhex("800000aa00389b71")
        body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        # Exercise an odd-length unknown chunk and its pad byte.
        body += b"JUNK" + struct.pack("<I", 3) + b"abc\x00"
        body += b"data" + struct.pack("<I", len(payload)) + payload
        if len(payload) % 2:
            body += b"\x00"
        path = self.path / name
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
        return path

    def test_wav_encodings(self):
        # Exact multiples of 1/128 are representable in every supported encoding.
        samples = np.tile(np.array([-64, -32, 0, 32, 64], dtype=np.int32), 161)
        floating = samples.astype(np.float32) / 128
        encodings = [("u8", (samples + 128).astype("u1").tobytes(), 8, 1),
                     ("s16", (samples * 256).astype("<i2").tobytes(), 16, 1),
                     ("s24", b"".join(struct.pack("<i", int(x) * 65536)[:3] for x in samples), 24, 1),
                     ("s32", (samples * 16777216).astype("<i4").tobytes(), 32, 1),
                     ("f32", floating.astype("<f4").tobytes(), 32, 3),
                     ("f64", floating.astype("<f8").tobytes(), 64, 3)]
        reference = None
        for name, payload, bits, fmt in encodings:
            for ext in (False, True):
                prefix = self.path / f"{name}_{ext}"
                wav = self.wav(f"{name}_{ext}.wav", payload, bits, fmt, ext)
                self.run_cli("-m", MODEL, "-i", wav, "-o", str(prefix) + ".csv", "--dump-prefix", prefix, "-t", 2)
                mel = Path(str(prefix) + ".mel.f32").read_bytes()
                if reference is None:
                    reference = mel
                self.assertEqual(mel, reference, f"Incorrect decoding of {name}, extensible={ext}")

    def test_cli_errors(self):
        self.run_cli("--not-an-option", good=False)
        self.run_cli("-m", good=False)
        self.run_cli("-t", "2junk", good=False)
        wav = self.wav("ok.wav", bytes(640), 16)
        self.run_cli("-m", MODEL, "-i", wav, "--threshold", "nan", good=False)
        self.run_cli("-m", MODEL, "-i", wav, "--threshold", "2", good=False)
        self.run_cli("-m", MODEL, "-i", wav, "--backend", "missing-device", good=False)
        self.run_cli("-m", MODEL, "-i", wav, "-o", wav, good=False)
        self.run_cli("-m", MODEL, "-i", wav, "-o", wav.parent / ".." / wav.parent.name / wav.name, good=False)
        broken = self.path / "broken.wav"
        broken.write_bytes(wav.read_bytes()[:-1])
        self.run_cli("-m", MODEL, "-i", broken, good=False)
        nan = self.wav("nan.wav", np.array([np.nan], dtype="<f4").tobytes(), 32, 3)
        self.run_cli("-m", MODEL, "-i", nan, good=False)
        mel = self.path / "input.f32"
        np.zeros(128, dtype="<f4").tofile(mel)
        result = self.run_cli("-m", MODEL, "--mel", mel, "-t", 2)
        self.assertTrue(result.stdout.startswith("time,f0,confidence,unvoiced\n"))

    def test_malformed_gguf(self):
        mel = self.path / "input.f32"
        np.zeros(128, dtype="<f4").tofile(mel)
        for arch, version in (("wrong", 1), ("fcpe", 999), ("fcpe", "wrong type")):
            model = self.path / "invalid.gguf"
            writer = gguf.GGUFWriter(model, arch)
            if isinstance(version, str):
                writer.add_string("fcpe.version", version)
            else:
                writer.add_uint32("fcpe.version", version)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            self.run_cli("-m", model, "--mel", mel, good=False)
        truncated = self.path / "truncated.gguf"
        truncated.write_bytes(MODEL.read_bytes()[:-128])
        self.run_cli("-m", truncated, "--mel", mel, good=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    CLI, MODEL = args.cli.resolve(), args.model.resolve()
    unittest.main(argv=[__file__, *remaining])
