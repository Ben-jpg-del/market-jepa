"""Chronological, frozen-encoder evaluation for the Colab pilot.

No model is fitted to 2019+ during representation learning. Readouts expand once
per calendar year, with a 20-trading-row gap and explicit label-end checks. The
2022–2025 period is historical evaluation, not an untouched discovery holdout.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Mapping

import numpy as np


def _dates(values):
    result = np.asarray(values, dtype="datetime64[D]")
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("dates must be a nonempty one-dimensional array")
    if np.any(np.isnat(result)) or np.any(result[1:] <= result[:-1]):
        raise ValueError("dates must be unique, sorted, and nonmissing")
    return result


def make_forward_labels(panel, horizon=5):
    """Origin t -> next open t+1 to close t+h; stock return minus SPY.

    Future quote availability only controls label validity, never context
    eligibility. Both stock and benchmark prices must use consistent adjustments.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    dates = _dates(panel.dates)
    opened, closed = np.asarray(panel.open), np.asarray(panel.close)
    bo, bc = np.asarray(panel.benchmark_open), np.asarray(panel.benchmark_close)
    if opened.shape != closed.shape or opened.shape[0] != len(dates):
        raise ValueError("Stock price shapes do not match dates")
    if bo.shape != dates.shape or bc.shape != dates.shape:
        raise ValueError("Benchmark open and close prices are required")
    labels = np.full(opened.shape, np.nan, dtype=np.float32)
    entry_dates = np.full(len(dates), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    end_dates = entry_dates.copy()
    n = len(dates) - horizon
    if n > 0:
        entries, exits = opened[1:n + 1], closed[horizon:]
        b_entries, b_exits = bo[1:n + 1], bc[horizon:]
        valid = (np.isfinite(entries) & np.isfinite(exits) & (entries > 0)
                 & (exits > 0) & np.isfinite(b_entries[:, None])
                 & np.isfinite(b_exits[:, None]) & (b_entries[:, None] > 0)
                 & (b_exits[:, None] > 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            forward = exits / entries - b_exits[:, None] / b_entries[:, None]
        labels[:n] = np.where(valid, forward, np.nan)
        entry_dates[:n], end_dates[:n] = dates[1:n + 1], dates[horizon:]
    return labels, entry_dates, end_dates


def purged_training_dates(dates, label_end_dates, boundary, purge=20):
    """Past origins only; reserve at least `purge` rows before boundary."""
    dates = _dates(dates)
    ends = np.asarray(label_end_dates, dtype="datetime64[D]")
    if ends.shape != dates.shape or purge < 0:
        raise ValueError("Invalid label ends or purge")
    boundary = np.datetime64(boundary, "D")
    boundary_index = int(np.searchsorted(dates, boundary))
    cutoff_index = max(0, boundary_index - purge)
    return ((np.arange(len(dates)) < cutoff_index) & ~np.isnat(ends)
            & (ends < boundary) & (dates < boundary))


def _rank(values):
    """Average ranks, including ties, without a SciPy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def rank_ic(prediction, target):
    """Return None when a cross-section has fewer than 3 stocks or no ranks."""
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    good = np.isfinite(prediction) & np.isfinite(target)
    if good.sum() < 3:
        return None
    p, y = _rank(prediction[good]), _rank(target[good])
    p, y = p - p.mean(), y - y.mean()
    denominator = np.linalg.norm(p) * np.linalg.norm(y)
    return float(np.dot(p, y) / denominator) if denominator > 0 else None


def date_block_interval(daily_values, samples=500, block_length=20, seed=42):
    """Moving-date-block CI; all stocks on a date are kept together upstream."""
    x = np.asarray([np.nan if v is None else v for v in daily_values], dtype=float)
    if len(x) < 8 or np.isfinite(x).sum() < 8 or samples < 1:
        return None
    block = min(max(1, int(block_length)), max(1, len(x) // 2))
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(samples):
        starts = rng.integers(0, len(x) - block + 1, size=math.ceil(len(x) / block))
        sample = np.concatenate([x[s:s + block] for s in starts])[:len(x)]
        if np.isfinite(sample).any():
            estimates.append(float(np.nanmean(sample)))
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])] if estimates else None


class RidgeReadout:
    """Training-only mean imputation, standardization, and intercept ridge.

    A sum-of-squares objective is used (the usual sklearn Ridge alpha scale).
    Constant and entirely missing columns receive zero weight.
    """
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)

    @staticmethod
    def statistics(x, y, row_mask=None, chunk_size=4096):
        """Two passes over bounded row chunks; no N-by-D float64 copy.

        x may be a disk-backed [date,stock,feature] array. The standardized
        normal equations are algebraically identical to the dense fit. Stats
        can be reused across alpha candidates and matching expanding folds.
        """
        x, y = np.asarray(x), np.asarray(y, dtype=np.float64)
        if x.ndim < 2 or x.shape[:-1] != y.shape or chunk_size < 1:
            raise ValueError("Feature/target shapes or chunk size are invalid")
        mask = np.isfinite(y)
        if row_mask is not None:
            if np.shape(row_mask) != y.shape:
                raise ValueError("Row mask must match target shape")
            mask &= row_mask
        indices = np.flatnonzero(mask.reshape(-1))
        if len(indices) < 2:
            raise ValueError("At least two valid training examples are required")
        dimension = x.shape[-1]
        flat_x, flat_y = x.reshape(-1, dimension), y.reshape(-1)
        count, sums = np.zeros(dimension, dtype=np.int64), np.zeros(dimension)
        target_sum = 0.0
        for start in range(0, len(indices), chunk_size):
            ids = indices[start:start + chunk_size]
            chunk = np.asarray(flat_x[ids], dtype=np.float64)
            finite = np.isfinite(chunk)
            count += finite.sum(axis=0)
            sums += np.where(finite, chunk, 0).sum(axis=0)
            target_sum += flat_y[ids].sum()
        mean = np.divide(sums, count, out=np.zeros(dimension), where=count > 0)
        intercept = float(target_sum / len(indices))
        gram, cross = np.zeros((dimension, dimension)), np.zeros(dimension)
        for start in range(0, len(indices), chunk_size):
            ids = indices[start:start + chunk_size]
            chunk = np.asarray(flat_x[ids], dtype=np.float64)
            centered = np.where(np.isfinite(chunk), chunk - mean, 0)
            gram += centered.T @ centered
            cross += centered.T @ (flat_y[ids] - intercept)
        variance = np.diag(gram) / len(indices)
        scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        return {"mean": mean, "scale": scale, "intercept": intercept,
                "gram": gram / scale[:, None] / scale[None, :],
                "cross": cross / scale, "examples": len(indices)}

    def fit_statistics(self, stats):
        self.mean_, self.scale_ = stats["mean"], stats["scale"]
        self.intercept_ = stats["intercept"]
        gram = stats["gram"].copy()
        gram.flat[::gram.shape[0] + 1] += self.alpha
        self.coef_ = np.linalg.solve(gram, stats["cross"])
        self.training_examples_ = stats["examples"]
        return self

    def fit(self, x, y):
        return self.fit_statistics(self.statistics(x, y))

    def predict(self, x, chunk_size=4096):
        x = np.asarray(x)
        flat = x.reshape(-1, x.shape[-1])
        output = np.empty(len(flat), dtype=np.float64)
        for start in range(0, len(flat), chunk_size):
            chunk = np.asarray(flat[start:start + chunk_size], dtype=np.float64)
            z = np.where(np.isfinite(chunk), chunk - self.mean_, 0) / self.scale_
            output[start:start + chunk_size] = z @ self.coef_ + self.intercept_
        return output.reshape(x.shape[:-1])

    def predict_masked(self, x, row_mask, chunk_size=4096):
        x, row_mask = np.asarray(x), np.asarray(row_mask, dtype=bool)
        if x.shape[:-1] != row_mask.shape:
            raise ValueError("Prediction mask must match feature rows")
        indices = np.flatnonzero(row_mask.reshape(-1))
        flat_x = x.reshape(-1, x.shape[-1])
        output = np.full(row_mask.size, np.nan)
        for start in range(0, len(indices), chunk_size):
            ids = indices[start:start + chunk_size]
            output[ids] = self.predict(flat_x[ids], chunk_size)
        return output.reshape(row_mask.shape)


def _select_rows(x, labels, eligible, date_mask):
    mask = date_mask[:, None] & eligible & np.isfinite(labels)
    # Missing features are handled using statistics from these training rows.
    return x[mask], labels[mask], mask


def _score_matrix(predictions, labels, eligible, date_mask):
    daily = []
    for t in np.flatnonzero(date_mask):
        good = eligible[t] & np.isfinite(labels[t])
        daily.append(rank_ic(predictions[t, good], labels[t, good]))
    valid = [v for v in daily if v is not None]
    return float(np.mean(valid)) if valid else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.datetime64, Path)):
        return str(value)
    return value


def portfolio_costs(rows, horizon=5, costs_bps=(0, 5, 10, 20), quantile=0.2):
    """Nonoverlapping five-day trades, flat overnight between cycles.

    At each scheduled origin: 0.5 notional long top quintile, 0.5 short bottom
    quintile; fully exit at h-th close. Entry plus exit turnover is 2.0 per
    completed trade cycle. No overlap, execution-at-signal-close, or free netting.
    Rows must belong to one model/cohort. Missing scheduled cycles stay in cash.
    """
    by_date = {}
    for row in rows:
        by_date.setdefault(int(row["origin_index"]), []).append(row)
    if not by_date:
        return {"cycles": 0, "scenarios": [], "trades": []}
    first, last = min(by_date), max(by_date)
    trades, missing_return_cycles = [], 0
    for t in range(first, last + 1, horizon):
        group = by_date.get(t, [])
        if len(group) < 5:
            continue
        # A constant forecast has no ranking; do not let symbol order create trades.
        predictions = np.asarray([r["prediction"] for r in group])
        if not np.isfinite(predictions).all() or np.ptp(predictions) == 0:
            continue
        group = sorted(group, key=lambda r: (r["prediction"], r["symbol"]))
        q = max(1, int(len(group) * quantile))
        chosen = group[-q:] + group[:q]
        if any(r["target"] is None or not np.isfinite(r["target"]) for r in chosen):
            # Do not re-rank around a future missing/delisted price or pretend
            # the planned portfolio stayed in cash. Aggregate P&L is unidentified.
            missing_return_cycles += 1
            continue
        gross = 0.5 * (np.mean([r["target"] for r in group[-q:]])
                       - np.mean([r["target"] for r in group[:q]]))
        trades.append({"origin_index": t, "origin_date": group[0]["origin_date"],
                       "entry_date": group[0]["entry_date"], "exit_date": group[0]["exit_date"],
                       "gross_return": float(gross), "entry_turnover": 1.0,
                       "exit_turnover": 1.0, "long_count": q, "short_count": q})
    span = last - first + horizon
    scenarios = []
    for bps in costs_bps:
        r = np.asarray([trade["gross_return"] - 2 * bps / 10000 for trade in trades])
        # Scheduled cash cycles count in risk/return; unavailable data is disclosed.
        schedule_n = len(range(first, last + 1, horizon))
        schedule_returns = np.zeros(schedule_n, dtype=float)
        for trade, net in zip(trades, r):
            schedule_returns[(trade["origin_index"] - first) // horizon] = net
        std = np.std(schedule_returns, ddof=1) if len(schedule_returns) > 1 else 0.0
        annual_return = (np.prod(1 + schedule_returns) ** (252 / span) - 1
                         if np.all(schedule_returns > -1) else None)
        scenarios.append({"cost_bps_per_side": bps,
                          "mean_completed_cycle_net_return": float(r.mean()) if len(r) else None,
                          "annualized_net_return": annual_return if not missing_return_cycles else None,
                          "net_sharpe": (float(schedule_returns.mean() / std * np.sqrt(252 / horizon))
                                         if std > 0 and not missing_return_cycles else None),
                          "total_turnover": float(2 * len(r)),
                          "annualized_turnover": float(2 * len(r) * 252 / span)})
    return _json_safe({"cycles": len(trades), "scheduled_cycles": len(range(first, last + 1, horizon)),
                       "skipped_cycles": len(range(first, last + 1, horizon)) - len(trades),
                       "missing_selected_return_cycles": missing_return_cycles,
                       "portfolio_return_coverage_complete": missing_return_cycles == 0,
                       "gross_notional": 1.0, "round_trip_turnover_per_cycle": 2.0,
                       "scenarios": scenarios, "trades": trades})


def summarize_predictions(rows, horizon=5, bootstrap_samples=500, seed=42):
    models = sorted({r["model"] for r in rows})
    report = {}
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        report[model] = {}
        for cohort in ("development", "historical_evaluation"):
            selected = [r for r in model_rows if r["cohort"] == cohort]
            by_date = {}
            for row in selected:
                by_date.setdefault(row["origin_date"], []).append(row)
            daily = {d: rank_ic([r["prediction"] for r in group], [r["target"] for r in group])
                     for d, group in sorted(by_date.items())}
            valid = [v for v in daily.values() if v is not None]
            yearly = {}
            for year in sorted({d[:4] for d in daily}):
                values = [v for d, v in daily.items() if d.startswith(year) and v is not None]
                yearly[year] = {"mean_rank_ic": float(np.mean(values)) if values else None,
                                "valid_dates": len(values)}
            report[model][cohort] = {
                "mean_daily_rank_ic": float(np.mean(valid)) if valid else None,
                "rank_ic_95ci_date_blocks": date_block_interval(list(daily.values()),
                                                                bootstrap_samples, 20, seed),
                "valid_ic_dates": len(valid), "prediction_rows": len(selected),
                "yearly": yearly, "portfolio": portfolio_costs(selected, horizon),
            }
    return _json_safe(report)


def evaluate_arrays(features: Mapping[str, np.ndarray], labels, dates, *,
                    label_end_dates, entry_dates=None, eligible=None, symbols=None,
                    fixed_predictions=None, years=range(2019, 2026),
                    alphas=(0.1, 1.0, 10.0, 100.0), purge=20, horizon=5,
                    bootstrap_samples=500, seed=42):
    """Evaluate [date,stock,feature] matrices with matched chronological probes.

    Per outer year, select alpha on the previous calendar year using an earlier
    purged fit; then refit on all permitted past data. Training transforms never
    inspect validation/evaluation rows. fixed_predictions is for momentum, etc.
    """
    started = time.perf_counter()
    years = tuple(years)
    dates = _dates(dates)
    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 2 or labels.shape[0] != len(dates):
        raise ValueError("labels must have shape [date, stock]")
    if purge < max(20, horizon):
        raise ValueError("The pilot requires at least 20 trading rows of purging")
    ends = np.asarray(label_end_dates, dtype="datetime64[D]")
    entries = (np.asarray(entry_dates, dtype="datetime64[D]") if entry_dates is not None
               else np.r_[dates[1:], np.datetime64("NaT", "D")])
    if ends.shape != dates.shape or entries.shape != dates.shape:
        raise ValueError("Entry/exit dates must have one item per origin")
    eligible = np.ones(labels.shape, bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if eligible.shape != labels.shape:
        raise ValueError("eligible must match labels")
    symbols = list(symbols) if symbols is not None else [str(i) for i in range(labels.shape[1])]
    if len(symbols) != labels.shape[1]:
        raise ValueError("symbols must match stock axis")
    year_numbers = dates.astype("datetime64[Y]").astype(int) + 1970
    predictions, folds = [], []

    def append_rows(name, p, mask):
        good = (mask[:, None] & eligible & np.isfinite(p)
                & ~np.isnat(ends)[:, None] & ~np.isnat(entries)[:, None])
        for t, k in zip(*np.nonzero(good)):
            predictions.append({"model": name, "origin_index": int(t),
                                "origin_date": str(dates[t]), "entry_date": str(entries[t]),
                                "exit_date": str(ends[t]), "symbol": symbols[k],
                                "prediction": float(p[t, k]),
                                "target": float(labels[t, k]) if np.isfinite(labels[t, k]) else None,
                                "cohort": "development" if year_numbers[t] <= 2021 else "historical_evaluation"})

    for name, values in features.items():
        x = np.asarray(values)
        if x.ndim != 3 or x.shape[:2] != labels.shape or x.shape[2] == 0:
            raise ValueError(f"{name}: features must have shape [date,stock,dimension]")
        statistic_cache = {}
        for year in years:
            outer = year_numbers == year
            if not outer.any():
                continue
            boundary = np.datetime64(f"{year}-01-01")
            train = purged_training_dates(dates, ends, boundary, purge)
            inner_train = purged_training_dates(dates, ends, f"{year - 1}-01-01", purge)
            # Keep the inner validation labels wholly before the outer boundary.
            inner_val = (year_numbers == year - 1) & train
            tm = train[:, None] & eligible & np.isfinite(labels)
            im = inner_train[:, None] & eligible & np.isfinite(labels)
            vm = inner_val[:, None] & eligible & np.isfinite(labels)
            train_count, inner_count = int(tm.sum()), int(im.sum())
            if train_count < 50 or inner_count < 50 or inner_val.sum() < 8:
                folds.append({"model": name, "year": int(year), "status": "insufficient_history",
                              "training_rows": train_count, "inner_training_rows": inner_count})
                continue
            if vm.sum() < 3:
                continue
            # The previous year's outer training split is this year's inner
            # training split. Cache only O(D^2) statistics, not training rows.
            for key, mask in ((year - 1, im), (year, tm)):
                if key not in statistic_cache:
                    statistic_cache[key] = RidgeReadout.statistics(x, labels, mask)
            scores = []
            for alpha in alphas:
                inner = RidgeReadout(alpha).fit_statistics(statistic_cache[year - 1])
                matrix = inner.predict_masked(x, vm)
                ic = _score_matrix(matrix, labels, eligible, inner_val)
                scores.append({"alpha": float(alpha), "rank_ic": ic,
                               "negative_mse": -float(np.mean((matrix[vm] - labels[vm]) ** 2))})
            metric = "rank_ic" if any(s["rank_ic"] is not None for s in scores) else "negative_mse"
            selected = max(scores, key=lambda s: (s[metric] if s[metric] is not None else -np.inf,
                                                  s["alpha"]))
            readout = RidgeReadout(selected["alpha"]).fit_statistics(statistic_cache[year])
            origin_mask = outer[:, None] & eligible
            p = readout.predict_masked(x, origin_mask)
            append_rows(name, p, outer)
            folds.append({"model": name, "year": int(year), "status": "ok",
                          "alpha": selected["alpha"], "selection_metric": metric,
                          "inner_validation_year": int(year - 1), "alpha_scores": scores,
                          "training_rows": train_count, "training_dates": int(train.sum()),
                          "last_training_origin": str(dates[np.flatnonzero(train)[-1]]),
                          "latest_training_label_end": str(np.max(ends[train])),
                          "first_evaluation_origin": str(dates[np.flatnonzero(outer)[0]])})
    outer_all = np.isin(year_numbers, list(years))
    for name, p in (fixed_predictions or {}).items():
        p = np.asarray(p, dtype=float)
        if p.shape != labels.shape:
            raise ValueError(f"{name}: fixed predictions must match labels")
        append_rows(name, p, outer_all)
    label_missing = outer_all[:, None] & eligible & ~np.isfinite(labels)
    report = {
        "protocol": {"primary_horizon_trading_days": horizon,
                     "entry": "next trading session open", "exit": "h-th trading session close",
                     "target": "arithmetic stock return minus SPY return",
                     "readout": "shared ridge, training-only imputation and standardization",
                     "alpha_grid": list(alphas), "purge_trading_rows": purge,
                     "head_refit": "annual expanding, alpha selected on previous calendar year",
                     "development": "2019-2021", "historical_evaluation": "2022-2025",
                     "evaluated_years": list(years),
                     "historical_evaluation_previously_examined": True,
                     "ci": "moving blocks of 20 dates; stocks never resampled independently",
                     "costs": "0/5/10/20 bps each side, fully entered and exited nonoverlapping cycles"},
        "folds": folds, "metrics": summarize_predictions(predictions, horizon, bootstrap_samples, seed),
        "eligible_origins_with_missing_label": int(label_missing.sum()),
        "limitations": ["Missing future prices are excluded from IC, not context eligibility or portfolio ranking; incomplete selected returns invalidate aggregate portfolio results.",
                        "Cost model excludes borrow fees, market impact and execution uncertainty.",
                        "Readout uncertainty does not include architecture selection or encoder-training uncertainty."],
        "evaluation_seconds": time.perf_counter() - started,
    }
    return {"report": _json_safe(report), "predictions": predictions}


def _field(panel, preferred, alternative):
    return np.asarray(getattr(panel, preferred, getattr(panel, alternative, None)))


def context_eligibility(panel, context=60):
    """Origin-known universe and fully observed basic price/volume channels."""
    stock_obs = _field(panel, "stock_obs", "stock_observed").astype(bool)
    universe = np.asarray(panel.universe, dtype=bool)
    eligible = np.zeros(universe.shape, dtype=bool)
    carried = getattr(panel, "origin_observed", None)
    required = np.asarray(carried, dtype=bool) if carried is not None else stock_obs[..., :4].all(axis=-1)
    counts = np.vstack([np.zeros((1, universe.shape[1]), dtype=int), np.cumsum(required, axis=0)])
    for t in range(context - 1, len(universe)):
        eligible[t] = universe[t] & ((counts[t + 1] - counts[t + 1 - context]) == context)
    return eligible


def _feature_storage(shape, output_path=None):
    if output_path is None:
        return np.full(shape, np.nan, np.float32)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
    # Preserve unavailable-row semantics without allocating a dense fill array.
    for start in range(0, shape[0], 16):
        result[start:start + 16] = np.nan
    return result


def raw_feature_baseline(panel, context=60, output_path=None):
    """66 dimensions: latest + trailing60 mean + observed fraction, stock14/market8."""
    stock, market = np.asarray(panel.stock), np.asarray(panel.market)
    so = _field(panel, "stock_obs", "stock_observed").astype(bool)
    mo = _field(panel, "market_obs", "market_observed").astype(bool)
    t_count, k_count, _ = stock.shape
    result = _feature_storage((t_count, k_count, 3 * (stock.shape[-1] + market.shape[-1])), output_path)
    for t in range(context - 1, t_count):
        lo = t - context + 1
        sm, mm = so[lo:t + 1], mo[lo:t + 1]
        sc, mc = sm.sum(0), mm.sum(0)
        smean = np.divide(np.where(sm, stock[lo:t + 1], 0).sum(0), sc,
                          out=np.full(sc.shape, np.nan, dtype=float), where=sc > 0)
        mmean = np.divide(np.where(mm, market[lo:t + 1], 0).sum(0), mc,
                          out=np.full(mc.shape, np.nan, dtype=float), where=mc > 0)
        sf = np.concatenate([np.where(so[t], stock[t], np.nan), smean, sc / context], axis=-1)
        mf = np.concatenate([np.where(mo[t], market[t], np.nan), mmean, mc / context])
        result[t] = np.concatenate([sf, np.broadcast_to(mf, (k_count, len(mf)))], axis=-1)
    return result


def extract_encoder_features(panel, encoder, *, device="cpu", context=60,
                             max_stocks=32, batch_size=16, seed=42, eligible=None,
                             maximum_date=None, output_path=None):
    """Frozen causal inference, fixed symbol-hash peers, recorded each date.

    Peer assignments use only origin-eligible stocks. No target availability or
    future membership is consulted. Each market embedding is conditional on that
    partition and is concatenated with its stocks' embeddings.
    """
    import hashlib
    import torch

    started = time.perf_counter()
    dates = _dates(panel.dates)
    eligible = context_eligibility(panel, context) if eligible is None else np.asarray(eligible, bool)
    so, mo = _field(panel, "stock_obs", "stock_observed"), _field(panel, "market_obs", "market_observed")
    symbols = list(panel.symbols)
    permutation = sorted(range(len(symbols)), key=lambda k: hashlib.sha256(f"{seed}:{symbols[k]}".encode()).digest())
    groups, records = [], []
    for t in range(context - 1, len(dates)):
        if maximum_date is not None and dates[t] > np.datetime64(maximum_date, "D"):
            break
        indices = [k for k in permutation if eligible[t, k]]
        for start in range(0, len(indices), max_stocks):
            group = indices[start:start + max_stocks]
            groups.append((t, group))
            records.append({"date": str(dates[t]), "partition": start // max_stocks,
                            "symbols": [symbols[k] for k in group]})
    encoder = encoder.to(device)
    encoder.eval()
    output, valid_output = None, np.zeros(eligible.shape, bool)
    with torch.inference_mode():
        for offset in range(0, len(groups), batch_size):
            items = groups[offset:offset + batch_size]
            b = len(items)
            sx = np.zeros((b, max_stocks, context, panel.stock.shape[-1]), np.float32)
            mx = np.zeros((b, context, panel.market.shape[-1]), np.float32)
            sxm, mxm, valid = np.zeros_like(sx, bool), np.zeros_like(mx, bool), np.zeros((b, max_stocks), bool)
            for j, (t, indices) in enumerate(items):
                sl = slice(t - context + 1, t + 1)
                n = len(indices)
                sx[j, :n] = np.asarray(panel.stock)[sl][:, indices].transpose(1, 0, 2)
                sxm[j, :n] = so[sl][:, indices].transpose(1, 0, 2)
                mx[j], mxm[j], valid[j, :n] = panel.market[sl], mo[sl], True
            tensors = [torch.as_tensor(a, device=device) for a in (sx, mx, sxm, mxm, valid)]
            encoded = encoder.encode(*tensors)
            stock_z = encoded["stock_embeddings"].detach().float().cpu().numpy()
            market_z = encoded["market_embedding"].detach().float().cpu().numpy()
            stock_valid = encoded["valid_stock_mask"].detach().cpu().numpy().astype(bool)
            if output is None:
                output = _feature_storage((*eligible.shape, stock_z.shape[-1] + market_z.shape[-1]), output_path)
            for j, (t, indices) in enumerate(items):
                n = len(indices)
                good = stock_valid[j, :n]
                chosen = np.asarray(indices)[good]
                output[t, chosen] = np.concatenate([stock_z[j, :n][good],
                                                    np.broadcast_to(market_z[j], (good.sum(), market_z.shape[-1]))], axis=-1)
                valid_output[t, chosen] = True
    if output is None:
        raise ValueError("No origin-eligible context windows")
    if isinstance(output, np.memmap):
        output.flush()
    return output, valid_output, records, time.perf_counter() - started


def evaluate_encoder(panel, encoder, *, random_encoder=None, device="cpu", output_dir=None,
                     horizon=5, max_stocks=32, batch_size=16, seed=42,
                     bootstrap_samples=500, normalized=False, years=(2019, 2020, 2021),
                     encoder_cutoff="2018-12-31", feature_set="book"):
    """Memory-bounded evaluation using local temporary disk for embeddings.

    Public API is unchanged. Temporary files are removed on success or failure;
    persistent outputs contain reports/predictions, not multi-gigabyte features.
    """
    with tempfile.TemporaryDirectory(prefix="fjepa-evaluation-") as scratch:
        return _evaluate_encoder_impl(
            panel, encoder, random_encoder=random_encoder, device=device,
            output_dir=output_dir, horizon=horizon, max_stocks=max_stocks,
            batch_size=batch_size, seed=seed, bootstrap_samples=bootstrap_samples,
            normalized=normalized, years=years, encoder_cutoff=encoder_cutoff,
            feature_set=feature_set, scratch_dir=Path(scratch))


def _evaluate_encoder_impl(panel, encoder, *, random_encoder=None, device="cpu", output_dir=None,
                           horizon=5, max_stocks=32, batch_size=16, seed=42,
                           bootstrap_samples=500, normalized=False, years=(2019, 2020, 2021),
                           encoder_cutoff="2018-12-31", feature_set="book", scratch_dir):
    """Full train/random/raw comparison; model must be trained only through 2018.

    Pass the exact training-normalized panel with normalized=True; otherwise fit
    the data module's normalizer through 2018-12-31. The caller must verify the
    checkpoint's cutoff and dataset fingerprint before using this function.
    """
    import copy

    years = tuple(years)
    if not years or min(years) < 2019 or max(years) > 2025:
        raise ValueError("Evaluation years must be within 2019–2025")
    if np.datetime64(encoder_cutoff, "D") > np.datetime64("2018-12-31"):
        raise ValueError("Encoder training must end no later than 2018-12-31")
    if random_encoder is None:
        raise ValueError("Pass a freshly initialized random_encoder with the identical model configuration")
    if feature_set not in ("basic", "book"):
        raise ValueError("feature_set must be 'basic' or 'book'")
    if not normalized:
        from .data import fit_normalizer, normalize_panel
        panel = normalize_panel(panel, fit_normalizer(panel, cutoff="2018-12-31"))
    # The development screen cannot inspect historical evaluation input/labels.
    # Its final h origin rows consequently have no usable forward label.
    cutoff_rows = int(np.searchsorted(_dates(panel.dates),
                                     np.datetime64(f"{max(years)}-12-31"), side="right"))
    panel = copy.copy(panel)
    for field in ("dates", "stock", "market", "stock_obs", "market_obs", "open", "close",
                  "volume", "volume_baseline20", "benchmark_open", "benchmark_close", "universe"):
        if hasattr(panel, field):
            setattr(panel, field, np.asarray(getattr(panel, field))[:cutoff_rows])
    if feature_set == "basic":
        # Preserve caller-owned panel and make inference match A/B training.
        panel = copy.copy(panel)
        panel.stock, panel.market = panel.stock.copy(), panel.market.copy()
        panel.stock_obs = _field(panel, "stock_obs", "stock_observed").copy()
        panel.market_obs = _field(panel, "market_obs", "market_observed").copy()
        panel.stock[..., 9:], panel.stock_obs[..., 9:] = 0, False
        panel.market[..., 2:], panel.market_obs[..., 2:] = 0, False
    eligible = context_eligibility(panel)
    # Both encoder matrices plus the raw baseline are disk-backed. The stock
    # union can be much larger than the 128 eligible names on any one date.
    width = getattr(getattr(encoder, "config", None), "width", 128)
    feature_width = 4 * width + 3 * (panel.stock.shape[-1] + panel.market.shape[-1])
    scratch_bytes = int(np.prod(eligible.shape)) * feature_width * np.dtype("float32").itemsize
    if shutil.disk_usage(scratch_dir).free < scratch_bytes * 1.1 + 64 * 1024**2:
        raise RuntimeError(f"Evaluation requires about {scratch_bytes / 1024**3:.1f} GiB of free local scratch disk")
    trained, trained_valid, peers, seconds = extract_encoder_features(
        panel, encoder, device=device, max_stocks=max_stocks,
        batch_size=batch_size, seed=seed, eligible=eligible,
        maximum_date=f"{max(years)}-12-31", output_path=scratch_dir / "trained.f32")
    random_z, random_valid, random_peers, random_seconds = extract_encoder_features(
        panel, random_encoder, device=device, max_stocks=max_stocks,
        batch_size=batch_size, seed=seed, eligible=eligible,
        maximum_date=f"{max(years)}-12-31", output_path=scratch_dir / "random.f32")
    if random_peers != peers:
        raise RuntimeError("Trained and random models used different peer cohorts")
    labels, entries, ends = make_forward_labels(panel, horizon)
    matched = eligible & trained_valid & random_valid
    features = {"trained_encoder": trained, "random_encoder": random_z,
                "raw_features": raw_feature_baseline(panel, output_path=scratch_dir / "raw.f32")}
    # These are monotonic standardized versions of the corresponding return.
    # Missing indicators stay missing, not zero-return signals.
    observed = _field(panel, "stock_obs", "stock_observed")
    momentum = np.where(observed[..., 5], np.asarray(panel.stock)[..., 5], np.nan)
    reversal = -np.where(observed[..., 4], np.asarray(panel.stock)[..., 4], np.nan)
    evaluated = evaluate_arrays(features, labels, panel.dates, label_end_dates=ends,
                                entry_dates=entries, eligible=matched, symbols=panel.symbols,
                                fixed_predictions={"momentum_20d": momentum, "reversal_5d": reversal},
                                horizon=horizon, bootstrap_samples=bootstrap_samples, seed=seed,
                                years=years)
    report = evaluated["report"]
    report["feature_definitions"] = {
        "feature_set": feature_set,
        "trained_encoder": "frozen stock embedding concatenated with its peer-partition market embedding",
        "random_encoder": "identical architecture/feature masks, fresh random parameters, same peer partitions",
        "raw_features": "latest, trailing60 observed mean, observed fraction for all stock14 and market8 = 66 dimensions",
        "momentum_20d": "latest 20-day return, positive direction",
        "reversal_5d": "negative latest 5-day return",
    }
    report["inference"] = {"peer_seed": seed, "maximum_stocks_per_partition": max_stocks,
                           "feature_storage": "temporary local disk-backed float32 arrays",
                           "estimated_scratch_bytes": scratch_bytes,
                           "ridge_chunk_rows": 4096,
                           "encoder_cutoff_asserted_by_caller": str(encoder_cutoff),
                           "trained_seconds": seconds, "random_seconds": random_seconds,
                           "origin_eligible_stock_dates": int(eligible.sum()),
                           "matched_stock_dates": int(matched.sum())}
    report["data_metadata"] = getattr(panel, "metadata", {})
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        prediction_path = directory / "predictions.csv"
        with prediction_path.open("w", newline="") as handle:
            fields = ["model", "origin_index", "origin_date", "entry_date", "exit_date", "symbol", "prediction", "target", "cohort"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(evaluated["predictions"])
        peer_path = directory / "peer_partitions.json"
        peer_path.write_text(json.dumps(peers, indent=2))
        report["artifacts"] = {"predictions": str(prediction_path), "peer_partitions": str(peer_path)}
        report = _json_safe(report)
        (directory / "evaluation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    else:
        report["predictions"] = evaluated["predictions"]
        report["peer_partitions"] = peers
    return _json_safe(report)
