"""Phase-0 gate for ridge and LightGBM on the 26 assembled channels (latest / mean60 / fraction = 78 features),
walk-forward 2015-2025 on a assembled panel's own origin-eligible universe, windows 2015-18 / 2019-21 / 2022-25.

  work/colab-test-env/bin/python scripts/eval/raw_feature_gate.py --panel work/fjepa-assembled-data/panel_russell2000.npz --tag russell2000 \
      [--cache-dir outputs/raw-feature-gate/cache] [--horizons 5,20] [--seeds 42,43,44] [--years 2015,...,2025]

Writes outputs/raw-feature-gate/<tag>_h<h>/{results.json,daily_ic.json} and prints one RAW_GATE line.
The gate is informational (decided 2026-10-06): it sets the raw bar that the encoder must beat on 2022-25.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb  # noqa: E402  (before torch on macOS)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from src.data import fit_normalizer, load_panel  # noqa: E402
from src.evaluation import context_eligibility, date_block_interval, make_forward_labels, purged_training_dates, rank_ic  # noqa: E402
from src.expanded_eval import FORMAT_VERSION, _crop, _hash_array, _hash_json, _write_json, cached_raw_features  # noqa: E402
from src.runner import source_fingerprint  # noqa: E402
from supervised_bar import lgbm_params, normal_scores_by_group, ridge_fit_predict  # noqa: E402


def log(**info):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **info}), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs/raw-feature-gate"))
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/raw-feature-gate/cache"))
    parser.add_argument("--horizons", default="5,20")
    parser.add_argument("--years", default=",".join(str(y) for y in range(2015, 2026)))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-rounds", type=int, default=800)
    parser.add_argument("--normalizer-cutoff", default="2014-12-31", help="raw-feature normalizer fit window end (before the first test year)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    years = [int(y) for y in args.years.split(",")]; seeds = [int(s) for s in args.seeds.split(",")]
    cache = Path(args.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    panel = _crop(load_panel(args.panel), max(years))
    if panel.metadata.get("feature_layout") != "v2":
        raise SystemExit("expected a assembled panel (feature_layout == 'v2')")
    dates = panel.dates.astype("datetime64[D]")
    eligible = context_eligibility(panel, 60) & (dates >= np.datetime64("2006-01-01"))[:, None] & panel.market_obs[:, 0, None]
    base = {"format": FORMAT_VERSION, "source": source_fingerprint(), "dataset": panel.metadata["dataset_fingerprint"], "last_input_date": str(dates[-1]),
            "eligibility": _hash_array(eligible), "symbols": list(panel.symbols), "context": 60, "mode": "raw-gate", "cutoff": args.normalizer_cutoff}
    norm_path = cache / f"v2_raw_normalizer_{_hash_json(base)}.json"
    if norm_path.exists():
        normalizer = json.loads(norm_path.read_text())
    else:
        normalizer = fit_normalizer(panel, cutoff=args.normalizer_cutoff, start="2006-01-01"); _write_json(norm_path, normalizer)
    raw_id = _hash_json({**base, "raw_normalizer": normalizer, "layout": "latest_mean60_fraction"})
    raw = cached_raw_features(panel, normalizer, eligible, cache / f"raw_book_{raw_id}", raw_id, context=60)
    t_idx, k_idx = np.nonzero(eligible); n = len(t_idx)
    X = np.asarray(raw[t_idx, k_idx], dtype=np.float32)
    year_of_row = dates[t_idx].astype("datetime64[Y]").astype(int) + 1970
    log(stage="features_ready", rows=n, dims=X.shape[1], elapsed=round(time.perf_counter() - started))
    dev, hold, recent = [y for y in years if 2019 <= y <= 2021], [y for y in years if y < 2019], [y for y in years if y >= 2022]
    gate = {}
    for h in [int(x) for x in args.horizons.split(",")]:
        labels, _, ends = make_forward_labels(panel, h)
        y_raw = labels[t_idx, k_idx].astype(np.float64); finite = np.isfinite(y_raw)
        y_rank = normal_scores_by_group(y_raw[:, None], t_idx)[:, 0].astype(np.float64)
        purge = max(20, h); preds = {"ridge|v2raw": np.full(n, np.nan), "lgbm|v2raw": np.full(n, np.nan)}; folds = []
        for Y in years:
            tr = purged_training_dates(dates, ends, f"{Y}-01-01", purge)[t_idx] & finite
            itr = purged_training_dates(dates, ends, f"{Y - 1}-01-01", purge)[t_idx] & finite
            iva = (year_of_row == Y - 1) & tr; te = year_of_row == Y
            if tr.sum() < 1000 or itr.sum() < 1000 or iva.sum() < 200 or te.sum() == 0:
                folds.append({"year": Y, "status": "insufficient"}); continue
            iva_t = t_idx[iva]; iva_dates = np.unique(iva_t); y_iva = y_raw[iva]
            def inner_ic(pv):
                return float(np.nanmean([rank_ic(pv[iva_t == t], y_iva[iva_t == t]) or np.nan for t in iva_dates]))
            best, best_ic = None, -np.inf
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
                ic = inner_ic(ridge_fit_predict(X[itr], y_rank[itr], X[iva], alpha))
                if ic > best_ic: best, best_ic = alpha, ic
            preds["ridge|v2raw"][te] = ridge_fit_predict(X[tr], y_rank[tr], X[te], best)
            acc = np.zeros(te.sum()); rounds = []
            grid = [g for g in (25, 50, 100, 200, 400, 800) if g <= args.max_rounds]
            for seed in seeds:
                params = lgbm_params(seed)
                inner = lgb.train(params, lgb.Dataset(X[itr], y_rank[itr]), num_boost_round=max(grid))
                best_k, best_k_ic = grid[0], -np.inf
                for k in grid:
                    ic = inner_ic(inner.predict(X[iva], num_iteration=k))
                    if ic > best_k_ic: best_k, best_k_ic = k, ic
                full = lgb.train(params, lgb.Dataset(X[tr], y_rank[tr]), num_boost_round=best_k)
                acc += full.predict(X[te]); rounds.append(best_k)
            preds["lgbm|v2raw"][te] = acc / len(seeds)
            folds.append({"year": Y, "status": "ok", "alpha": best, "rounds": rounds, "train_rows": int(tr.sum())})
            log(stage="fold", h=h, year=Y, alpha=best, rounds=rounds)
        eval_rows = np.isin(year_of_row, years) & finite; uniq = np.unique(t_idx[eval_rows])
        daily = {}
        for key, p in preds.items():
            d = {}
            for t in uniq:
                sel = (t_idx == t) & eval_rows; ic = rank_ic(p[sel], y_raw[sel])
                if ic is not None: d[str(dates[t])] = ic
            daily[key] = d
        def mean_over(d, yrs):
            v = [x for k, x in d.items() if int(k[:4]) in yrs]; return float(np.mean(v)) if v else None
        def paired(a, b, yrs):
            keys = [k for k in a if k in b and int(k[:4]) in yrs]; diffs = [a[k] - b[k] for k in keys]
            if not diffs: return None
            lo, hi = date_block_interval(diffs); return {"mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)], "dates": len(diffs)}
        result = {"panel": args.panel, "tag": args.tag, "horizon": h, "years": years, "rows": n, "feature_dims": int(X.shape[1]), "seeds": seeds,
                  "mean_daily_rank_ic": {k: {"holdout_2015_2018": mean_over(d, hold), "dev_2019_2021": mean_over(d, dev), "recent_2022_plus": mean_over(d, recent),
                                            "yearly": {str(y): mean_over(d, [y]) for y in years}} for k, d in daily.items()},
                  "paired_lgbm_minus_ridge": {w: paired(daily["lgbm|v2raw"], daily["ridge|v2raw"], yrs) for w, yrs in (("holdout", hold), ("dev", dev), ("recent", recent))},
                  "folds": folds, "seconds": round(time.perf_counter() - started, 1)}
        out = Path(args.out) / f"{args.tag}_h{h}"; out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(result, indent=1)); (out / "daily_ic.json").write_text(json.dumps(daily))
        # per-row out-of-sample predictions (rows = eligible stock-days of the panel), for stacking / incremental-value tests
        np.savez_compressed(out / "predictions.npz", t_idx=t_idx, k_idx=k_idx, dates=dates.astype("datetime64[D]").astype(str),
                            **{k.replace("|", "_"): v.astype(np.float32) for k, v in preds.items()})
        gate[f"h{h}"] = {k: [round(v["holdout_2015_2018"] or 0, 4), round(v["dev_2019_2021"] or 0, 4), round(v["recent_2022_plus"] or 0, 4)]
                         for k, v in result["mean_daily_rank_ic"].items()}
    passed = all(v[2] > 0 for h in gate.values() for v in h.values())
    print("RAW_GATE " + json.dumps({"tag": args.tag, "ic(hold15-18,dev19-21,recent22-25)": gate, "recent_positive_everywhere": passed,
                                   "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
