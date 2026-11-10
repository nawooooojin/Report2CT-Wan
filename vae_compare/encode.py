"""Encode CT-RATE volumes and save pooled latent features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from vae_compare import data, vaes

POOL = 8


def _bins(length: int, out: int) -> list[tuple[int, int]]:
    """Bin edges for adaptive average pooling."""
    return [((i * length) // out, -((-(i + 1) * length) // out)) for i in range(out)]


def pool8(cdhw: np.ndarray, out: int = POOL) -> np.ndarray:
    """Average-pool a (C, D, H, W) latent to 8x8x8 and flatten."""
    c = cdhw.shape[0]
    res = np.empty((c, out, out, out), np.float64)
    b0, b1, b2 = (_bins(n, out) for n in cdhw.shape[1:])
    for i, (s0, e0) in enumerate(b0):
        for j, (s1, e1) in enumerate(b1):
            for k, (s2, e2) in enumerate(b2):
                res[:, i, j, k] = cdhw[:, s0:e0, s1:e1, s2:e2].reshape(c, -1).mean(axis=1, dtype=np.float64)
    return res.reshape(-1).astype(np.float32)


def main() -> None:
    """Encode one split and write <out-dir>/<vae>_<split>.npy (N, C*512) and <vae>_<split>_ids.txt."""
    ap = argparse.ArgumentParser(description=__doc__)
    vaes.add_args(ap)
    ap.add_argument("--ct-dir", required=True, help="CT-RATE train_fixed or valid_fixed folder")
    ap.add_argument("--split", required=True, help="split name used in the output file names")
    ap.add_argument("--split-json", help='{"train": [...], "val": [...]}; without it every volume in --ct-dir is used')
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.split_json:
        ids = data.read_split(args.split_json, args.split)
        paths = {sid: data.volume_path(args.ct_dir, sid) for sid in ids}
    else:
        paths = data.list_volumes(args.ct_dir)
        ids = list(paths)
    out = Path(args.out_dir)
    cache = out / f"{args.vae}_{args.split}"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [sid for sid in ids if not (cache / f"{sid}.npy").is_file()]
    print(f"{len(ids) - len(todo)} done, {len(todo)} to go", flush=True)
    if todo:
        vae = vaes.build(args)
        with torch.inference_mode():
            for k, sid in enumerate(todo, 1):
                z = vae.enc(vae.prep(data.load_volume(paths[sid])))
                zc = z.float().cpu().numpy()
                feat = pool8(np.ascontiguousarray(zc[0], dtype=np.float32))
                tmp = cache / f"{sid}.tmp.npy"
                np.save(tmp, feat)
                os.replace(tmp, cache / f"{sid}.npy")
                print(f"{k}/{len(todo)} {sid} latent {tuple(zc.shape)}", flush=True)

    feats = np.stack([np.load(cache / f"{sid}.npy") for sid in ids])
    np.save(out / f"{args.vae}_{args.split}.npy", feats)
    (out / f"{args.vae}_{args.split}_ids.txt").write_text("\n".join(ids) + "\n")
    print(f"{args.vae} {args.split}: features {feats.shape}")


if __name__ == "__main__":
    main()
