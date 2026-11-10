"""Train and test a linear probe on pooled VAE features."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from vae_compare.data import load_labels

LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 128, 100, 5


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC of one label."""
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision of one label."""
    y = y.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(ys)[last].astype(np.float64)
    fp = (last + 1) - tp
    prec, rec = tp / (tp + fp), tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def f1_score(t: np.ndarray, pred: np.ndarray) -> float:
    """F1 of one label from boolean truth and prediction, 0 when undefined."""
    tp, fp, fn = int((pred & t).sum()), int((pred & ~t).sum()), int((~pred & t).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def val_thresholds(logits: np.ndarray, y: np.ndarray) -> list[float | None]:
    """Per-label thresholds that maximize validation F1."""
    out: list[float | None] = []
    for i in range(y.shape[1]):
        t, s = y[:, i].astype(bool), logits[:, i]
        n_pos = int(t.sum())
        if n_pos == 0:
            out.append(None)
            continue
        order = np.argsort(-s, kind="mergesort")
        ts, ss = t[order], s[order]
        last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
        tp = np.cumsum(ts)[last].astype(np.float64)
        fp = (last + 1) - tp
        f1 = 2 * tp / (tp + fp + n_pos)
        out.append(float(ss[last[int(np.argmax(f1))]]))
    return out


def f1_at_thresholds(logits: np.ndarray, y: np.ndarray, thr: list[float | None]) -> list[float]:
    """Per-label F1 for logit >= thr[i], or logit > 0 where thr[i] is None."""
    return [f1_score(y[:, i].astype(bool), logits[:, i] >= th if th is not None else logits[:, i] > 0)
            for i, th in enumerate(thr)]


def load_split(feature_dir: Path, vae: str, split: str, table: dict[str, list[int]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Ids, float32 features and float32 labels of one split."""
    ids = (feature_dir / f"{vae}_{split}_ids.txt").read_text().split()
    x = np.load(feature_dir / f"{vae}_{split}.npy")
    y = np.array([table[s] for s in ids], dtype=np.float32)
    return ids, x, y


def fit_zscore(x: np.ndarray):
    """Fit a per-feature z-score."""
    mu, sd = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)
    sd[sd == 0] = 1.0
    mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)
    return lambda a: (a - mu32) / sd32


def batches(x: np.ndarray, y: np.ndarray, order: np.ndarray, bs: int, drop_last: bool):
    """Yield mini-batches in the given order."""
    for k in range(0, len(order), bs):
        idx = np.sort(order[k : k + bs])
        if drop_last and len(idx) < bs:
            return
        yield np.asarray(x[idx], dtype=np.float32), y[idx]


def run(args: argparse.Namespace) -> dict:
    """Train with early stopping on validation macro AUROC and evaluate the best state once on the test split."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    labels = Path(args.labels_dir)
    names, tab_tr = load_labels(labels / "train_predicted_labels.csv")
    names_te, tab_te = load_labels(labels / "valid_predicted_labels.csv")
    if names != names_te or len(names) != 18:
        raise SystemExit("label CSVs do not share the 18 CT-RATE labels")
    fdir = Path(args.feature_dir)
    tr_ids, tr_x, tr_y = load_split(fdir, args.vae, "train", tab_tr)
    va_ids, va_x, va_y = load_split(fdir, args.vae, "val", tab_tr)
    zscore = fit_zscore(tr_x)
    tr_x, va_x = zscore(tr_x), zscore(va_x)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    probe = nn.Linear(tr_x.shape[1], len(names)).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    def predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        probe.eval()
        outs = []
        with torch.no_grad():
            for xb, _ in batches(x, y, np.arange(len(x)), BATCH_SIZE, False):
                outs.append(probe(torch.from_numpy(xb).to(dev)).double().cpu().numpy())
        return np.concatenate(outs)

    best_auroc, best_epoch, best_state, since = -1.0, 0, None, 0
    for ep in range(args.max_epochs):
        probe.train()
        for xb, yb in batches(tr_x, tr_y, rng.permutation(len(tr_ids)), BATCH_SIZE, True):
            xt, yt = torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)
            opt.zero_grad()
            loss = loss_fn(probe(xt), yt)
            loss.backward()
            opt.step()
        sched.step()
        logits = predict(va_x, va_y)
        auroc = float(np.nanmean([roc_auc(va_y[:, i].astype(int), logits[:, i]) for i in range(va_y.shape[1])]))
        print(f"epoch {ep + 1} val macro AUROC {auroc:.4f}", flush=True)
        if auroc > best_auroc:
            best_auroc, best_epoch, since = auroc, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    probe.load_state_dict(best_state)
    thr = val_thresholds(predict(va_x, va_y), va_y.astype(int))

    te_ids, te_x, te_y = load_split(fdir, args.vae, "test", tab_te)
    logits = predict(zscore(te_x), te_y)
    yt = te_y.astype(int)
    per = {"auroc": [roc_auc(yt[:, i], logits[:, i]) for i in range(18)],
           "auprc": [average_precision(yt[:, i], logits[:, i]) for i in range(18)],
           "f1": f1_at_thresholds(logits, yt, thr)}
    return {"vae": args.vae, "seed": args.seed, "max_epochs": args.max_epochs, "best_epoch": best_epoch,
            "best_val_macro_auroc": best_auroc, "n_train": len(tr_ids), "n_val": len(va_ids), "n_test": len(te_ids),
            "d_in": int(tr_x.shape[1]), "test": {"macro": {k: float(np.nanmean(v)) for k, v in per.items()},
                                                 "per_label": {k: dict(zip(names, v)) for k, v in per.items()}},
            "val_thresholds": dict(zip(names, thr))}


def main() -> None:
    """Parse arguments, run the probe and write <out-dir>/<vae>/seed<k>.json."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--feature-dir", required=True, help="output folder of vae_compare.encode")
    ap.add_argument("--labels-dir", required=True, help="CT-RATE multi_abnormality_labels folder")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()
    res = run(args)
    out = Path(args.out_dir) / args.vae
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed{args.seed}.json").write_text(json.dumps(res, indent=1))
    m = res["test"]["macro"]
    print(f"{args.vae} seed {args.seed}: best epoch {res['best_epoch']} mAUROC {m['auroc']:.4f} "
          f"mAUPRC {m['auprc']:.4f} macro F1 {m['f1']:.4f}")


if __name__ == "__main__":
    main()
