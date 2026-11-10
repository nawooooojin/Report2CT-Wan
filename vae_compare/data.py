"""CT-RATE volume reading, the 448^3 grid, volume ids and abnormality labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from monai import transforms

GRID = (448, 448, 448)
Z_PAD = 1

_LOAD = None


def load_volume(path: str | Path) -> np.ndarray:
    """Read a CT-RATE volume as RAS HU resized to the 448^3 grid, (X, Y, Z) float32."""
    global _LOAD
    if _LOAD is None:
        _LOAD = transforms.Compose([
            transforms.LoadImaged(keys="image"),
            transforms.EnsureChannelFirstd(keys="image"),
            transforms.Orientationd(keys="image", axcodes="RAS"),
            transforms.EnsureTyped(keys="image", dtype=torch.float32),
            transforms.Resized(keys="image", spatial_size=GRID, mode="trilinear"),
        ])
    return _LOAD({"image": str(path)})["image"].numpy()[0]


def pad_z(a: np.ndarray, k: int = Z_PAD) -> np.ndarray:
    """Repeat the last axial slice of an (X, Y, Z) array k times."""
    return np.concatenate([a, np.repeat(a[:, :, -1:], k, axis=2)], axis=2)


def drop_z(rec: np.ndarray, k: int = Z_PAD) -> np.ndarray:
    """Drop the k padded slices from a (Z + k, X, Y) reconstruction."""
    if rec.shape[0] != GRID[2] + k:
        raise RuntimeError(f"decoder returned {rec.shape[0]} slices, expected {GRID[2] + k}")
    return rec[:GRID[2]]


def list_volumes(folder: str | Path) -> dict[str, Path]:
    """Map every CT-RATE volume id under a folder to its path, sorted by id."""
    paths = {p.name[:-len(".nii.gz")]: p for p in Path(folder).rglob("*.nii.gz")}
    return dict(sorted(paths.items()))


def volume_path(folder: str | Path, scan_id: str) -> Path:
    """Path of a CT-RATE volume id under a train_fixed or valid_fixed folder."""
    series = scan_id.rsplit("_", 1)[0]
    patient = series.rsplit("_", 1)[0]
    return Path(folder) / patient / series / f"{scan_id}.nii.gz"


def read_split(split_json: str | Path, split: str) -> list[str]:
    """Volume ids of one split from a {"train": [...], "val": [...]} json, in file order."""
    return list(json.loads(Path(split_json).read_text())[split])


def load_labels(csv_path: str | Path) -> tuple[list[str], dict[str, list[int]]]:
    """Label names and per-volume 0/1 labels from a CT-RATE multi_abnormality_labels CSV."""
    with open(csv_path) as f:
        r = csv.reader(f)
        names = next(r)[1:]
        return names, {row[0].replace(".nii.gz", ""): [int(v) for v in row[1:]] for row in r}
