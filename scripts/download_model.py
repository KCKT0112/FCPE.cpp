#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

"""Download and safely unpack the exact official torchfcpe release used by this port."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = "https://github.com/CNChTu/FCPE/releases/download/v0.0.4/torchfcpe-0.0.4-py3-none-any.whl"
SHA256 = "f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d"


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / URL.rsplit("/", 1)[1]
    if not wheel.exists():
        temp = wheel.with_suffix(".download")
        print(f"Downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, temp.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Downloaded wheel SHA256 mismatch")
        temp.replace(wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"Wheel SHA256 mismatch: {wheel}")
    destination = (directory / "torchfcpe-0.0.4").resolve()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {name}")
        archive.extractall(destination)
    print(f"Verified wheel: {wheel}\nCheckpoint: {destination / 'torchfcpe/assets/fcpe_c_v001.pt'}")
    return wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    prepare(args.output_dir)
