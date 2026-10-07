"""walk-forward readout helpers: per-year feature extraction (teacher / online / random / aux) and walk-forward ridge.

Reuses expanded_eval.peer_groups + cached_encoder_features on a date-sliced panel view so that one
year's memmap cache stays small, and evaluation.RidgeReadout statistics for the annual expanding ridge.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .evaluation import RidgeReadout, date_block_interval, purged_training_dates, rank_ic
from .expanded_eval import _normalized_view, cached_encoder_features, peer_groups


def window_view(panel, first_index, last_index, context=60):
    """Copy of `panel` restricted to rows [first_index - context + 1, last_index]; returns (view, offset)."""
    lo = max(0, int(first_index) - context + 1); hi = int(last_index) + 1
    view = copy.copy(panel)
    for name, value in vars(panel).items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(panel.dates):
            setattr(view, name, value[lo:hi])
    return view, lo


class TeacherView:
    """Exposes the EMA teacher through the `encode` interface cached_encoder_features expects."""
    def __init__(self, model):
        self.model = model; self.config = model.config
    def to(self, device):
        self.model.to(device); return self
    def eval(self):
        self.model.eval(); return self
    def encode(self, stock, market, stock_obs, market_obs, eligibility):
        return self.model.teacher(stock, market, stock_obs, market_obs, eligibility)


def extract_year_features(panel, model, random_model, normalizer, year_mask, eligible, cache_dir, *, device="cpu",
                          batch_size=16, peer_seed=42, max_stocks=32, context=60, signature="", on_progress=None):
    """Rows = eligible stock-dates in `year_mask`. Returns dict of arrays indexed by row:
    t_idx, k_idx, online[n,2w], teacher[n,2w], random[n,2w], aux[n,H]."""
    rows_t = np.flatnonzero(year_mask & eligible.any(axis=1))
    if not len(rows_t):
        return {"t_idx": np.zeros(0, int), "k_idx": np.zeros(0, int)}
    view, offset = window_view(panel, rows_t[0], rows_t[-1], context)
    normalized = _normalized_view(view, normalizer, "book", cutoff_limit=None)
    elig_view = np.zeros(normalized.universe.shape, bool)
    elig_view[rows_t - offset] = eligible[rows_t]
    groups, _ = peer_groups(normalized, elig_view, peer_seed=peer_seed, max_stocks=max_stocks)
    out = {}
    cache_dir = Path(cache_dir)
    for kind, encoder in (("online", model), ("teacher", TeacherView(model)), ("random", random_model)):
        kind_dir = cache_dir / kind
        if (kind_dir / "manifest.json").exists() and not (kind_dir / "features.f32").exists():
            shutil.rmtree(kind_dir, ignore_errors=True)  # a manifest restored without its memmaps: start that kind afresh
        feats, _ = cached_encoder_features(normalized, encoder, elig_view, groups, cache_dir / kind, f"{signature}|{kind}",
                                           device=device, batch_size=batch_size, max_stocks=max_stocks, on_progress=on_progress, phase=f"features_{kind}")
        t_v, k_v = np.nonzero(elig_view)
        out[kind] = np.asarray(feats[t_v, k_v], dtype=np.float16)  # halves disk; the ridge casts chunks to float64
        del feats
    out["t_idx"], out["k_idx"] = t_v + offset, k_v
    width = model.config.width
    with torch.no_grad():
        head = model.aux_head.to("cpu")
        out["aux"] = head(torch.as_tensor(out["online"][:, :width].astype(np.float32))).numpy().astype(np.float32)
    shutil.rmtree(cache_dir, ignore_errors=True)
    return out


def walk_forward_ridge(X, y_rank, y_raw, t_idx, dates, ends, year, *, purge=20, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Annual expanding ridge for test year `year`: alpha chosen by mean daily rank IC on year-1 (fit on data before year-1),
    refit on everything permitted before `year`. Returns (predictions for rows of `year` as {row_index: value}, fold record)."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    year_of_row = dates[t_idx].astype("datetime64[Y]").astype(int) + 1970
    finite = np.isfinite(y_rank) & np.isfinite(y_raw)
    train = purged_training_dates(dates, ends, f"{year}-01-01", purge)[t_idx] & finite
    inner = purged_training_dates(dates, ends, f"{year - 1}-01-01", purge)[t_idx] & finite
    val = (year_of_row == year - 1) & train
    test = year_of_row == year
    record = {"year": int(year), "train_rows": int(train.sum()), "inner_rows": int(inner.sum()), "validation_rows": int(val.sum()), "test_rows": int(test.sum())}
    if train.sum() < 200 or inner.sum() < 200 or val.sum() < 50 or test.sum() == 0:
        record["status"] = "insufficient"
        return {}, record
    inner_stats = RidgeReadout.statistics(X[inner], y_rank[inner])
    val_t = t_idx[val]; val_dates = np.unique(val_t); y_val = y_raw[val]
    scores = {}
    for alpha in alphas:
        pv = RidgeReadout(alpha).fit_statistics(inner_stats).predict(X[val])
        scores[alpha] = float(np.nanmean([rank_ic(pv[val_t == t], y_val[val_t == t]) or np.nan for t in val_dates]))
    best = max(scores, key=lambda a: (scores[a], a))
    model = RidgeReadout(best).fit_statistics(RidgeReadout.statistics(X[train], y_rank[train]))
    pred = model.predict(X[test])
    record.update(status="ok", alpha=float(best), alpha_scores=scores)
    return {int(i): float(v) for i, v in zip(np.flatnonzero(test), pred)}, record


def daily_ic(pred, y_raw, t_idx, dates):
    """Mean cross-sectional rank IC per date for rows where pred is finite."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    good = np.isfinite(pred) & np.isfinite(y_raw)
    out = {}
    for t in np.unique(t_idx[good]):
        sel = good & (t_idx == t)
        ic = rank_ic(pred[sel], y_raw[sel])
        if ic is not None:
            out[str(dates[t])] = float(ic)
    return out


def window_means(daily, windows):
    return {name: (float(np.mean([v for k, v in daily.items() if lo <= int(k[:4]) <= hi])) if any(lo <= int(k[:4]) <= hi for k in daily) else None)
            for name, (lo, hi) in windows.items()}


def paired_block_difference(a, b, lo, hi):
    keys = [k for k in a if k in b and lo <= int(k[:4]) <= hi]
    diffs = [a[k] - b[k] for k in keys]
    if len(diffs) < 25:
        return None
    ci = date_block_interval(diffs)
    return {"mean": float(np.mean(diffs)), "ci95": [float(ci[0]), float(ci[1])], "dates": len(diffs)}
