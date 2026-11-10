"""Reconstruct CT-RATE volumes with one VAE and write per-volume PSNR and SSIM on the 448^3 grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from vae_compare import data, vaes

COLS = ["scan_id", "psnr", "ssim"]


def unit(hu: np.ndarray) -> np.ndarray:
    """Clip HU to [-1000, 1000] and scale to [0, 1] float32."""
    return ((np.clip(hu, -vaes.HU_CLIP, vaes.HU_CLIP) + vaes.HU_CLIP) / (2 * vaes.HU_CLIP)).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB for data range 1."""
    return float(10 * np.log10(1.0 / np.mean((a.astype(np.float64) - b) ** 2)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """3D SSIM for data range 1."""
    return float(structural_similarity(a, b, data_range=1.0))


def read_rows(path: Path) -> list[dict]:
    """Rows of an existing metrics CSV."""
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Reconstruct every volume of a folder that is not yet in the CSV, then print mean and sd."""
    ap = argparse.ArgumentParser(description=__doc__)
    vaes.add_args(ap)
    ap.add_argument("--ct-dir", required=True, help="CT-RATE valid_fixed folder")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["scan_id"] for r in read_rows(out)}
    todo = {k: v for k, v in data.list_volumes(args.ct_dir).items() if k not in done}
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if todo:
        vae = vaes.build(args)
        with torch.inference_mode():
            for k, (sid, path) in enumerate(todo.items(), 1):
                x = data.load_volume(path)
                ref = np.ascontiguousarray(x.transpose(2, 0, 1))
                rec = vae.dec(vae.enc(vae.prep(x)))
                r, g = unit(rec), unit(ref)
                row = {"scan_id": sid, "psnr": psnr(r, g), "ssim": ssim(g, r)}
                new = not out.is_file()
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=COLS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"{k}/{len(todo)} {sid} psnr {row['psnr']:.3f} ssim {row['ssim']:.4f}", flush=True)

    rows = read_rows(out)
    for key in ("psnr", "ssim"):
        v = np.array([float(r[key]) for r in rows])
        print(f"{args.vae} {key}: {v.mean():.4f} +- {v.std(ddof=1):.4f} (n = {len(v)})")


if __name__ == "__main__":
    main()
