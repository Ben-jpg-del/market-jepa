"""Incremental value of new information families on top of the raw indicators (common cohort, 5-day).

Readouts (ridge, identical protocol to the expanded evaluator):
  raw                       66 raw indicator features (the bar)
  raw+macro                 + 12 market-priced macro channels (same for every stock on a date)
  raw+events                + 7 earnings-event channels (stock-specific)
  raw+macro+events          + both
  macro_only, events_only   for reference
Reports mean daily rank IC and paired per-date differences vs raw with moving-block intervals.

Usage: work/colab-test-env/bin/python scripts/eval/incremental_value_families.py --cache-dir outputs/colab-tokenization-10k/local-eval/cache
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from src.data import fit_normalizer, load_panel  # noqa: E402
from src.evaluation import evaluate_arrays, make_forward_labels  # noqa: E402
from src.expanded_eval import (FORMAT_VERSION, _crop, _hash_array, _hash_json, _write_json,  # noqa: E402
                                       cached_raw_features, context_eligibility, shared_origin_mask)
from src.runner import source_fingerprint  # noqa: E402
from incremental_value_eval import daily_ics, paired  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "incremental-value" / "families"))
    parser.add_argument("--families", default=str(ROOT / "work/fjepa-families/family_channels.npz"))
    parser.add_argument("--fundamentals", default=None, help="fundamentals_channels.npz; adds raw+fundamentals, raw+all, fundamentals_only readouts")
    parser.add_argument("--cohort", choices=("common", "expanded"), default="common", help="readout fit/score cohort (features are always computed with the expanded peer pool)")
    parser.add_argument("--readouts", default=None, help="comma-separated subset of readout names to run (default: all available)")
    parser.add_argument("--rank-families", action="store_true",
                        help="replace fundamentals / TimesFM / event values by per-date normal scores over the scoring cohort (cross-sectional rank transform)")
    parser.add_argument("--timesfm", default=None, help="timesfm_channels.npz; adds raw+timesfm, raw+fundamentals+timesfm, raw+everything, timesfm_only readouts")
    args = parser.parse_args(argv)
    out, cache = Path(args.out), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    expanded_panel = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
    original_panel = load_panel(ROOT / "work/fjepa-expanded-data/original_panel.npz")
    years = (2019, 2020, 2021)
    expanded, original = _crop(expanded_panel, max(years)), _crop(original_panel, max(years))
    cohorts = shared_origin_mask(expanded, original, context=60)
    eligible, common = cohorts["expanded"], cohorts[args.cohort]  # `common` below = the scoring cohort
    T, K = eligible.shape
    # shared raw baseline exactly as the evaluator builds it (reuses the existing cache when identities match)
    meta = json.loads((ROOT / "outputs/colab-tokenization/live-run/artifacts/F0_s42_seed42/run_metadata.json").read_text())
    alignment = meta.get("panel_metadata", {}).get("training_date_alignment")
    def training_dates(panel):
        valid = context_eligibility(panel, 60).sum(axis=1) >= 2
        valid &= panel.market_obs[:, 0]
        valid &= (panel.dates >= np.datetime64("2006-01-01")) & (panel.dates <= np.datetime64("2018-12-31"))
        return panel.dates[valid]
    shared_training_dates = np.intersect1d(training_dates(expanded), training_dates(original)) if alignment else None
    base = {"format": FORMAT_VERSION, "source": source_fingerprint(), "expanded_dataset": expanded_panel.metadata["dataset_fingerprint"],
            "last_input_date": str(expanded.dates[-1]), "readout_start": "2006-01-01",
            "eligibility": _hash_array(eligible), "symbols": list(expanded.symbols),
            "context": 60, "peer_seed": 42, "max_stocks": 32, "training_date_alignment": alignment}
    normalizer_path = cache / f"expanded_raw_normalizer_{_hash_json(base)}.json"
    if normalizer_path.exists():
        raw_normalizer = json.loads(normalizer_path.read_text())
    else:
        raw_fit_panel = copy.copy(expanded)
        train_start = "2006-01-01"
        if shared_training_dates is not None:
            raw_fit_panel.universe = expanded.universe.copy()
            train_rows = (expanded.dates >= np.datetime64("2006-01-01")) & (expanded.dates <= np.datetime64("2018-12-31"))
            raw_fit_panel.universe[train_rows & ~np.isin(expanded.dates, shared_training_dates)] = False
            train_start = alignment["start"]
        raw_normalizer = fit_normalizer(raw_fit_panel, cutoff="2018-12-31", start=train_start)
        _write_json(normalizer_path, raw_normalizer)
    raw_id = _hash_json({**base, "raw_normalizer": raw_normalizer, "layout": "latest_mean60_fraction_66"})
    raw = cached_raw_features(expanded, raw_normalizer, eligible, cache / f"raw_book_{raw_id}", raw_id, context=60)
    print(json.dumps({"stage": "raw_ready", "elapsed": round(time.perf_counter() - started)}), flush=True)
    fam = np.load(args.families, allow_pickle=False)
    t_index = np.searchsorted(fam["dates"].astype("datetime64[D]"), expanded.dates.astype("datetime64[D]"))
    macro = np.where(fam["macro_obs"][t_index], fam["macro"][t_index], np.nan).astype(np.float32)  # (T, M)
    M = macro.shape[-1]
    events = np.where(fam["events_obs"][t_index], fam["events"][t_index], np.nan).astype(np.float32)  # (T, K, 7)
    # winsorize event/macro features at training-period 1st/99th percentiles (fit on 2006-2018 common rows)
    train_rows = (expanded.dates >= np.datetime64("2006-01-01")) & (expanded.dates <= np.datetime64("2018-12-31"))
    def clip_fit(x2d):
        lo, hi = np.nanpercentile(x2d, [1, 99], axis=0)
        return lo, hi
    mlo, mhi = clip_fit(macro[train_rows])
    macro = np.clip(macro, mlo, mhi)
    ev_train = events[train_rows][common[train_rows]]
    elo, ehi = clip_fit(ev_train)
    events = np.clip(events, elo, ehi)
    def rank_by_date(x3):
        """Per-date normal scores over the scoring-cohort rows; missing stays missing."""
        from scipy.stats import norm
        out = np.full_like(x3, np.nan)
        for t in range(T):
            m = common[t]
            if m.sum() < 5:
                continue
            block = x3[t, m]
            res = np.full_like(block, np.nan)
            for j in range(block.shape[1]):
                col = block[:, j]; ok = np.isfinite(col)
                n = ok.sum()
                if n >= 5:
                    r = np.empty(n); r[np.argsort(col[ok], kind="stable")] = np.arange(n)
                    res[ok, j] = norm.ppf((r + 0.5) / n)
            out[t, m] = res
        return out.astype(np.float32)
    def sparse(name, width, fill):
        arr = np.memmap(out / f"{name}.f32", mode="w+", dtype=np.float32, shape=(T, K, width))
        for t in range(T):
            m = common[t]
            if m.any():
                arr[t, m] = fill(t, m)
        arr.flush()
        return arr
    fund = None
    if args.fundamentals:
        fz = np.load(args.fundamentals, allow_pickle=False)
        fund = np.where(fz["fundamentals_obs"][t_index], fz["fundamentals"][t_index], np.nan).astype(np.float32)  # (T, K, 7)
        flo, fhi = clip_fit(fund[train_rows][common[train_rows]])
        fund = np.clip(fund, flo, fhi)
    tfm = None
    if args.timesfm:
        tz = np.load(args.timesfm, allow_pickle=False)
        tfm = np.where(tz["timesfm_obs"][t_index], tz["timesfm"][t_index], np.nan).astype(np.float32)  # (T, K, 15)
        tlo, thi = clip_fit(tfm[train_rows][common[train_rows]])
        tfm = np.clip(tfm, tlo, thi)
    if args.rank_families:
        events = rank_by_date(events)
        if fund is not None:
            fund = rank_by_date(fund)
        if tfm is not None:
            tfm = rank_by_date(tfm)
    feats = {
        "raw": raw,
        "raw_plus_macro": sparse("raw_plus_macro", 66 + M, lambda t, m: np.concatenate([raw[t, m], np.broadcast_to(macro[t], (m.sum(), M))], -1)),
        "raw_plus_events": sparse("raw_plus_events", 66 + 7, lambda t, m: np.concatenate([raw[t, m], events[t, m]], -1)),
        "raw_plus_macro_events": sparse("raw_plus_macro_events", 66 + M + 7, lambda t, m: np.concatenate([raw[t, m], np.broadcast_to(macro[t], (m.sum(), M)), events[t, m]], -1)),
        "macro_only": sparse("macro_only", M, lambda t, m: np.broadcast_to(macro[t], (m.sum(), M))),
        "events_only": sparse("events_only", 7, lambda t, m: events[t, m]),
    }
    if fund is not None:
        nf = fund.shape[-1]
        feats["raw_plus_fundamentals"] = sparse("raw_plus_fundamentals", 66 + nf, lambda t, m: np.concatenate([raw[t, m], fund[t, m]], -1))
        feats["raw_plus_events_fundamentals"] = sparse("raw_plus_events_fundamentals", 66 + 7 + nf, lambda t, m: np.concatenate([raw[t, m], events[t, m], fund[t, m]], -1))
        feats["raw_plus_all"] = sparse("raw_plus_all", 66 + M + 7 + nf, lambda t, m: np.concatenate([raw[t, m], np.broadcast_to(macro[t], (m.sum(), M)), events[t, m], fund[t, m]], -1))
        feats["fundamentals_only"] = sparse("fundamentals_only", nf, lambda t, m: fund[t, m])
    if tfm is not None:
        nt = tfm.shape[-1]
        feats["raw_plus_timesfm"] = sparse("raw_plus_timesfm", 66 + nt, lambda t, m: np.concatenate([raw[t, m], tfm[t, m]], -1))
        feats["timesfm_only"] = sparse("timesfm_only", nt, lambda t, m: tfm[t, m])
        if fund is not None:
            nf = fund.shape[-1]
            feats["raw_plus_fundamentals_timesfm"] = sparse("raw_plus_fundamentals_timesfm", 66 + nf + nt, lambda t, m: np.concatenate([raw[t, m], fund[t, m], tfm[t, m]], -1))
            feats["raw_plus_everything"] = sparse("raw_plus_everything", 66 + M + 7 + nf + nt, lambda t, m: np.concatenate([raw[t, m], np.broadcast_to(macro[t], (m.sum(), M)), events[t, m], fund[t, m], tfm[t, m]], -1))
    if args.readouts:
        keep = set(args.readouts.split(","))
        for name in list(feats):
            if name not in keep:
                if name != "raw":
                    (out / f"{name}.f32").unlink(missing_ok=True)
                del feats[name]
        if "raw" not in feats:
            feats = {"raw": raw, **feats}
    labels, entry_dates, ends = make_forward_labels(expanded, 5)
    print(json.dumps({"stage": "readouts", "elapsed": round(time.perf_counter() - started)}), flush=True)
    evaluated = evaluate_arrays(feats, labels, expanded.dates, label_end_dates=ends, entry_dates=entry_dates, eligible=common,
                                symbols=expanded.symbols, years=years, horizon=5, purge=20, bootstrap_samples=500, seed=42)
    metrics = evaluated["report"]["metrics"]
    daily = {m: daily_ics(evaluated["predictions"], m) for m in feats}
    result = {"cohort": args.cohort, "horizon": 5, "years": years, "rank_families": bool(args.rank_families),
              "mean_daily_rank_ic": {m: metrics[m]["development"]["mean_daily_rank_ic"] for m in feats},
              "ci95": {m: metrics[m]["development"]["rank_ic_95ci_date_blocks"] for m in feats},
              "yearly": {m: {y: v["mean_rank_ic"] for y, v in metrics[m]["development"]["yearly"].items()} for m in feats},
              "paired_vs_raw": {m: paired(daily[m], daily["raw"]) for m in feats if m != "raw"},
              "paired_vs_raw_plus_fundamentals": ({m: paired(daily[m], daily["raw_plus_fundamentals"]) for m in feats if m != "raw_plus_fundamentals"}
                                                  if "raw_plus_fundamentals" in feats else None),
              "event_coverage_common_rows": float(np.isfinite(events[..., 0])[common].mean()),
              "fundamentals_coverage_common_rows": float(np.isfinite(fund[..., 4])[common].mean()) if fund is not None else None,
              "timesfm_coverage_common_rows": float(np.isfinite(tfm[..., 0])[common].mean()) if tfm is not None else None,
              "seconds": round(time.perf_counter() - started, 1)}
    (out / "results.json").write_text(json.dumps(result, indent=1))
    for name in feats:
        if name != "raw":
            (out / f"{name}.f32").unlink(missing_ok=True)
    print("FAMILIES " + json.dumps({"ic": result["mean_daily_rank_ic"], "paired_vs_raw": {k: (v["mean_difference"], v["ci95"]) for k, v in result["paired_vs_raw"].items()}, "yearly": result["yearly"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
