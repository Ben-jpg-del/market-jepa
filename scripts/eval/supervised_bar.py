"""Supervised bar: ridge vs LightGBM on the indicators (+ families), same walk-forward protocol.

For each evaluation year Y (default 2015–2021, so 2015–2018 is a second holdout next to the
2019–2021 development window): train on origins whose labels end before Y-01-01 minus a purge of
max(20, horizon) sessions; choose ridge alpha / LightGBM rounds on year Y-1 (itself fit on data
before Y-1); refit on everything permitted; score year Y. Metric: mean daily rank IC of the
SPY-relative forward return; paired per-date differences with moving-block intervals.

Feature sets: ind (66 cached raw indicator features), ind+fund (+7 fundamentals, per-date normal
scores), ind+fund+macro (+15 macro, date-constant), all (+7 events, ranked).
Target: per-date normal scores of the forward return (rank target) for fitting; IC is computed on
the raw forward return.

Usage:
  work/colab-test-env/bin/python scripts/eval/supervised_bar.py --cohort common --horizon 5 \
      --cache-dir outputs/colab-tokenization-10k/local-eval/cache --out outputs/supervised-bar
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb  # noqa: E402  (must be imported before torch on macOS: two OpenMP runtimes otherwise segfault)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import fit_normalizer, load_panel  # noqa: E402
from src.evaluation import date_block_interval, make_forward_labels, purged_training_dates, rank_ic  # noqa: E402
from src.expanded_eval import (FORMAT_VERSION, _crop, _hash_array, _hash_json, _write_json,  # noqa: E402
                                       cached_raw_features, context_eligibility, shared_origin_mask)
from src.runner import source_fingerprint  # noqa: E402


def log(**info):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **info}), flush=True)


def normal_scores_by_group(values, group_ids, min_n=5):
    """Per-group normal scores (rank transform) column by column; NaN preserved."""
    from scipy.stats import norm
    out = np.full_like(values, np.nan, dtype=np.float32)
    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    bounds = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.r_[0, bounds]; stops = np.r_[bounds, len(order)]
    for s, e in zip(starts, stops):
        idx = order[s:e]
        block = values[idx]
        for j in range(block.shape[1]):
            col = block[:, j]; ok = np.isfinite(col); n = int(ok.sum())
            if n >= min_n:
                r = np.empty(n); r[np.argsort(col[ok], kind="stable")] = np.arange(n)
                out[idx[ok], j] = norm.ppf((r + 0.5) / n)
    return out


def own_feature_book(panel, cache, years_max=2025):
    """Single-panel mode: cohort = the panel's own origin-eligible universe; raw-66 features cached per panel."""
    panel = _crop(panel, years_max)
    eligible = context_eligibility(panel, 60)
    eligible &= (panel.dates >= np.datetime64("2006-01-01"))[:, None]
    eligible &= panel.market_obs[:, 0, None]
    base = {"format": FORMAT_VERSION, "source": source_fingerprint(), "dataset": panel.metadata["dataset_fingerprint"],
            "last_input_date": str(panel.dates[-1]), "readout_start": "2006-01-01", "eligibility": _hash_array(eligible),
            "symbols": list(panel.symbols), "context": 60, "mode": "own"}
    normalizer_path = cache / f"own_raw_normalizer_{_hash_json(base)}.json"
    if normalizer_path.exists():
        normalizer = json.loads(normalizer_path.read_text())
    else:
        normalizer = fit_normalizer(panel, cutoff="2018-12-31", start="2006-01-01")
        _write_json(normalizer_path, normalizer)
    raw_id = _hash_json({**base, "raw_normalizer": normalizer, "layout": "latest_mean60_fraction_66"})
    raw = cached_raw_features(panel, normalizer, eligible, cache / f"raw_book_{raw_id}", raw_id, context=60)
    return panel, {"own": eligible}, raw


def raw_feature_book(expanded_panel, original_panel, cache, years_max=2021):
    """Same raw-66 features and cohort masks as the incremental scripts (reuses the cache)."""
    expanded, original = _crop(expanded_panel, years_max), _crop(original_panel, years_max)
    cohorts = shared_origin_mask(expanded, original, context=60)
    eligible = cohorts["expanded"]
    meta = json.loads((ROOT / "outputs/colab-tokenization/live-run/artifacts/F0_s42_seed42/run_metadata.json").read_text())
    alignment = meta.get("panel_metadata", {}).get("training_date_alignment")

    def training_dates(panel):
        valid = context_eligibility(panel, 60).sum(axis=1) >= 2
        valid &= panel.market_obs[:, 0]
        valid &= (panel.dates >= np.datetime64("2006-01-01")) & (panel.dates <= np.datetime64("2018-12-31"))
        return panel.dates[valid]
    shared = np.intersect1d(training_dates(expanded), training_dates(original)) if alignment else None
    base = {"format": FORMAT_VERSION, "source": source_fingerprint(), "expanded_dataset": expanded_panel.metadata["dataset_fingerprint"],
            "last_input_date": str(expanded.dates[-1]), "readout_start": "2006-01-01",
            "eligibility": _hash_array(eligible), "symbols": list(expanded.symbols),
            "context": 60, "peer_seed": 42, "max_stocks": 32, "training_date_alignment": alignment}
    normalizer_path = cache / f"expanded_raw_normalizer_{_hash_json(base)}.json"
    if normalizer_path.exists():
        normalizer = json.loads(normalizer_path.read_text())
    else:
        fit_panel = copy.copy(expanded); start = "2006-01-01"
        if shared is not None:
            fit_panel.universe = expanded.universe.copy()
            rows = (expanded.dates >= np.datetime64("2006-01-01")) & (expanded.dates <= np.datetime64("2018-12-31"))
            fit_panel.universe[rows & ~np.isin(expanded.dates, shared)] = False
            start = alignment["start"]
        normalizer = fit_normalizer(fit_panel, cutoff="2018-12-31", start=start)
        _write_json(normalizer_path, normalizer)
    raw_id = _hash_json({**base, "raw_normalizer": normalizer, "layout": "latest_mean60_fraction_66"})
    raw = cached_raw_features(expanded, normalizer, eligible, cache / f"raw_book_{raw_id}", raw_id, context=60)
    return expanded, cohorts, raw


def ridge_fit_predict(xtr, ytr, xte, alpha):
    mu = np.nanmean(xtr, axis=0); sd = np.nanstd(xtr, axis=0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    def prep(x):
        z = (x - mu) / sd
        return np.where(np.isfinite(z), z, 0.0)
    a = prep(xtr); b = prep(xte)
    d = a.shape[1]
    g = a.T @ a + alpha * np.eye(d); w = np.linalg.solve(g, a.T @ (ytr - ytr.mean()))
    return b @ w


def lgbm_params(seed):
    return dict(objective="regression", learning_rate=0.03, num_leaves=31, max_depth=6, min_data_in_leaf=500,
                feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1,
                num_threads=12, seed=seed, deterministic=True, force_row_wise=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs/supervised-bar"))
    parser.add_argument("--cohort", choices=("common", "expanded", "own"), default="common")
    parser.add_argument("--panel", default=None, help="single-panel mode (use with --cohort own): evaluate on this panel's own universe")
    parser.add_argument("--tag", default="", help="output folder suffix")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--years", default="2015,2016,2017,2018,2019,2020,2021")
    parser.add_argument("--feature-sets", default="ind,ind+fund,ind+fund+macro,all")
    parser.add_argument("--models", default="ridge,lgbm")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-rounds", type=int, default=800)
    parser.add_argument("--families", default=str(ROOT / "work/fjepa-families/family_channels.npz"))
    parser.add_argument("--fundamentals", default=str(ROOT / "work/fjepa-families/fundamentals_channels.npz"))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    years = [int(y) for y in args.years.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out) / f"{args.cohort}{args.tag}_h{args.horizon}"; out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    if args.panel:
        assert args.cohort == "own", "--panel requires --cohort own"
        expanded, cohorts, raw = own_feature_book(load_panel(args.panel), cache, years_max=max(years))
    else:
        expanded_panel = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
        original_panel = load_panel(ROOT / "work/fjepa-expanded-data/original_panel.npz")
        expanded, cohorts, raw = raw_feature_book(expanded_panel, original_panel, cache)
    mask = cohorts[args.cohort]
    T, K = mask.shape
    dates = expanded.dates.astype("datetime64[D]")
    labels, entry_dates, ends = make_forward_labels(expanded, args.horizon)
    log(stage="features_ready", cohort=args.cohort, rows=int(mask.sum()), elapsed=round(time.perf_counter() - started))

    # ---- row table ---------------------------------------------------------------------------
    t_idx, k_idx = np.nonzero(mask)
    n = len(t_idx)
    X_ind = np.asarray(raw[t_idx, k_idx], dtype=np.float32)  # (n, 66)
    fam = np.load(args.families, allow_pickle=False)
    ti = np.searchsorted(fam["dates"].astype("datetime64[D]"), dates)
    macro = np.where(fam["macro_obs"][ti], fam["macro"][ti], np.nan).astype(np.float32)[t_idx]  # (n, 15)
    events = np.where(fam["events_obs"][ti], fam["events"][ti], np.nan).astype(np.float32)[t_idx, k_idx]  # (n, 7)
    fz = np.load(args.fundamentals, allow_pickle=False)
    fund = np.where(fz["fundamentals_obs"][ti], fz["fundamentals"][ti], np.nan).astype(np.float32)[t_idx, k_idx]  # (n, 7)
    fund_r = normal_scores_by_group(fund, t_idx)
    events_r = normal_scores_by_group(events, t_idx)
    y_raw = labels[t_idx, k_idx].astype(np.float64)
    y_rank = normal_scores_by_group(y_raw[:, None], t_idx)[:, 0].astype(np.float64)
    names = {"ind": [f"ind_{i}" for i in range(X_ind.shape[1])],
             "fund": list(fz["fundamental_names"]), "macro": list(fam["macro_names"]), "events": list(fam["event_names"])}
    sets = {"ind": (X_ind, names["ind"]),
            "ind+fund": (np.concatenate([X_ind, fund_r], 1), names["ind"] + names["fund"]),
            "ind+fund+macro": (np.concatenate([X_ind, fund_r, macro], 1), names["ind"] + names["fund"] + names["macro"]),
            "ind+fund+events": (np.concatenate([X_ind, fund_r, events_r], 1), names["ind"] + names["fund"] + names["events"]),
            "all": (np.concatenate([X_ind, fund_r, macro, events_r], 1), names["ind"] + names["fund"] + names["macro"] + names["events"])}
    feature_sets = [s for s in args.feature_sets.split(",") if s in sets]
    models = args.models.split(",")
    year_of_row = dates[t_idx].astype("datetime64[Y]").astype(int) + 1970
    purge = max(20, args.horizon)
    finite = np.isfinite(y_raw)
    log(stage="table_ready", n=n, features={k: v[0].shape[1] for k, v in sets.items()}, elapsed=round(time.perf_counter() - started))

    # ---- walk-forward ------------------------------------------------------------------------
    preds = {}  # (model, set) -> array (n,) of predictions on evaluated years
    folds = []
    for fs in feature_sets:
        X, cols = sets[fs]
        for model in models:
            key = f"{model}|{fs}"
            p = np.full(n, np.nan)
            for Y in years:
                train_dates = purged_training_dates(dates, ends, f"{Y}-01-01", purge)
                inner_dates = purged_training_dates(dates, ends, f"{Y - 1}-01-01", purge)
                tr = train_dates[t_idx] & finite
                itr = inner_dates[t_idx] & finite
                iva = (year_of_row == Y - 1) & tr
                te = year_of_row == Y
                if tr.sum() < 1000 or itr.sum() < 1000 or iva.sum() < 200 or te.sum() == 0:
                    folds.append({"model": model, "set": fs, "year": Y, "status": "insufficient"}); continue
                t0 = time.perf_counter()
                if model == "ridge":
                    best, best_ic = None, -np.inf
                    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
                        pv = ridge_fit_predict(X[itr], y_rank[itr], X[iva], alpha)
                        ic = np.nanmean([rank_ic(pv[t_idx[iva] == t], y_raw[iva][t_idx[iva] == t]) or np.nan for t in np.unique(t_idx[iva])])
                        if ic > best_ic: best, best_ic = alpha, ic
                    p[te] = ridge_fit_predict(X[tr], y_rank[tr], X[te], best)
                    folds.append({"model": model, "set": fs, "year": Y, "status": "ok", "alpha": best, "inner_ic": round(float(best_ic), 4),
                                  "train_rows": int(tr.sum()), "seconds": round(time.perf_counter() - t0, 1)})
                else:
                    acc = np.zeros(te.sum()); rounds = []
                    grid = [g for g in (25, 50, 100, 200, 400, 800) if g <= args.max_rounds] or [args.max_rounds]
                    iva_t = t_idx[iva]; iva_dates = np.unique(iva_t); y_iva = y_raw[iva]
                    for seed in seeds:
                        params = lgbm_params(seed)
                        d_itr = lgb.Dataset(X[itr], y_rank[itr], feature_name=cols, free_raw_data=False)
                        inner = lgb.train(params, d_itr, num_boost_round=max(grid))
                        # select rounds by inner-year mean daily rank IC (L2 early stopping stops after 1-5 rounds on this SNR)
                        best_k, best_ic = grid[0], -np.inf
                        for k in grid:
                            pv = inner.predict(X[iva], num_iteration=k)
                            ic = np.nanmean([rank_ic(pv[iva_t == t], y_iva[iva_t == t]) or np.nan for t in iva_dates])
                            if ic > best_ic:
                                best_k, best_ic = k, ic
                        full = lgb.train(params, lgb.Dataset(X[tr], y_rank[tr], feature_name=cols), num_boost_round=best_k)
                        acc += full.predict(X[te]); rounds.append(int(best_k))
                        if seed == seeds[0] and Y == years[-1]:
                            imp = full.feature_importance(importance_type="gain"); order = np.argsort(imp)[::-1][:20]
                            folds.append({"model": model, "set": fs, "year": Y, "status": "importance",
                                          "top_gain": [(cols[i], round(float(imp[i] / imp.sum()), 4)) for i in order]})
                    p[te] = acc / len(seeds)
                    folds.append({"model": model, "set": fs, "year": Y, "status": "ok", "best_rounds": rounds,
                                  "train_rows": int(tr.sum()), "seconds": round(time.perf_counter() - t0, 1)})
                log(stage="fold", model=model, set=fs, year=Y, seconds=round(time.perf_counter() - t0, 1))
            preds[key] = p

    # ---- scoring -----------------------------------------------------------------------------
    eval_rows = np.isin(year_of_row, years) & finite
    uniq_t = np.unique(t_idx[eval_rows])
    daily = {}
    for key, p in preds.items():
        d = {}
        for t in uniq_t:
            sel = (t_idx == t) & eval_rows
            ic = rank_ic(p[sel], y_raw[sel])
            if ic is not None:
                d[str(dates[t])] = ic
        daily[key] = d
    def summary(d, yrs):
        vals = [v for k, v in d.items() if int(k[:4]) in yrs]
        return float(np.mean(vals)) if vals else None
    dev, hold = [y for y in years if 2019 <= y <= 2021], [y for y in years if y < 2019]
    recent = [y for y in years if y >= 2022]
    def paired(a, b, yrs):
        keys = [k for k in a if k in b and int(k[:4]) in yrs]
        diffs = [a[k] - b[k] for k in keys]
        if not diffs: return None
        lo, hi = date_block_interval(diffs)
        return {"mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)], "dates": len(diffs)}
    result = {"cohort": args.cohort, "horizon": args.horizon, "years": years, "seeds": seeds, "rows": n,
              "mean_daily_rank_ic": {k: {"dev_2019_2021": summary(d, dev), "holdout_2015_2018": summary(d, hold),
                                         "recent_2022_plus": summary(d, recent),
                                         "yearly": {str(y): summary(d, [y]) for y in years}} for k, d in daily.items()},
              "paired": {}, "folds": folds, "seconds": round(time.perf_counter() - started, 1)}
    for fs in feature_sets:
        if f"lgbm|{fs}" in daily and f"ridge|{fs}" in daily:
            result["paired"][f"lgbm_minus_ridge|{fs}"] = {"dev": paired(daily[f"lgbm|{fs}"], daily[f"ridge|{fs}"], dev),
                                                          "holdout": paired(daily[f"lgbm|{fs}"], daily[f"ridge|{fs}"], hold),
                                                          "recent": paired(daily[f"lgbm|{fs}"], daily[f"ridge|{fs}"], recent)}
    for model in models:
        for fs in feature_sets:
            if fs != "ind" and f"{model}|{fs}" in daily and f"{model}|ind" in daily:
                result["paired"][f"{model}:{fs}_minus_ind"] = {"dev": paired(daily[f"{model}|{fs}"], daily[f"{model}|ind"], dev),
                                                                "holdout": paired(daily[f"{model}|{fs}"], daily[f"{model}|ind"], hold),
                                                                "recent": paired(daily[f"{model}|{fs}"], daily[f"{model}|ind"], recent)}
    (out / "results.json").write_text(json.dumps(result, indent=1))
    (out / "daily_ic.json").write_text(json.dumps(daily))
    rnd = lambda x: round(x, 4) if x is not None else None
    print("SUPERVISED_BAR " + json.dumps({"cohort": args.cohort + args.tag, "h": args.horizon,
          "ic(hold15-18,dev19-21,recent22+)": {k: (rnd(v["holdout_2015_2018"]), rnd(v["dev_2019_2021"]), rnd(v["recent_2022_plus"])) for k, v in result["mean_daily_rank_ic"].items()},
          "seconds": result["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
