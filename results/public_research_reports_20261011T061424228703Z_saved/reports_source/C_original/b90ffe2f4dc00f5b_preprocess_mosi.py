"""Create small, fixed-size MOSI features from the verified MMSA aligned_50.pkl.

The input pickle must match the SHA-256 published by the MMSA project. Loading
unverified pickle data is unsafe; this script checks the digest before reading.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np


EXPECTED_SHA256 = "d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9"


def masked_moments(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    w = mask.astype(np.float32)[..., None]
    count = np.maximum(w.sum(axis=1), 1)
    mean = (x * w).sum(axis=1) / count
    var = (((x - mean[:, None, :]) ** 2) * w).sum(axis=1) / count
    return np.concatenate((mean, np.sqrt(np.maximum(var, 0))), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected source SHA-256: {digest}")

    with args.source.open("rb") as stream:
        data = pickle.load(stream)

    output: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        part = data[split]
        text_mask = np.asarray(part["text_bert"][:, 1, :], dtype=bool)
        output[f"{split}_text"] = masked_moments(part["text"], text_mask)
        for modality in ("audio", "vision"):
            values = np.asarray(part[modality])
            mask = np.linalg.norm(values, axis=-1) > 1e-8
            output[f"{split}_{modality}"] = masked_moments(values, mask)
        output[f"{split}_y"] = np.asarray(part["regression_labels"], dtype=np.float32)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.target, **output)
    for key, value in output.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
