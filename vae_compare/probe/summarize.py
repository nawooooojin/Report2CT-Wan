"""Print mAUROC, mAUPRC and macro F1 per VAE, averaged over probe seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vae_compare.vaes import VAES


def main() -> None:
    """Read <results-dir>/<vae>/seed*.json and print one row per VAE."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--vaes", nargs="+", default=list(VAES))
    args = ap.parse_args()
    keys = ("auroc", "auprc", "f1")
    print(f"{'VAE':<10}{'mAUROC':>18}{'mAUPRC':>18}{'macro F1':>18}{'seeds':>7}")
    for vae in args.vaes:
        rs = [json.loads(p.read_text()) for p in sorted((Path(args.results_dir) / vae).glob("seed*.json"))]
        if not rs:
            continue
        cells = []
        for k in keys:
            v = np.array([r["test"]["macro"][k] for r in rs])
            sd = f" +- {v.std(ddof=1):.4f}" if len(v) > 1 else ""
            cells.append(f"{v.mean():.4f}{sd}")
        print(f"{vae:<10}" + "".join(f"{c:>18}" for c in cells) + f"{len(rs):>7}")


if __name__ == "__main__":
    main()
