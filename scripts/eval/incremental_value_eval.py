"""Incremental value of an encoder on top of the raw indicators (common cohort, 5-day).

Same protocol as the expanded evaluator (shared raw normalizer, peer groups, annual expanding
ridge with purge, 2019-2021), but the readouts are:
  raw                 66 raw features                      (the bar)
  encoder             256 encoder coordinates              (as before)
  raw_plus_encoder    raw ‖ encoder (322)                  <- the question
  raw_plus_random     raw ‖ same-config random encoder     <- capacity/noise control
Reports mean daily rank IC for each and the paired per-date difference
(raw_plus_encoder - raw) and (raw_plus_random - raw) with moving-block intervals.

Usage:
  work/colab-test-env/bin/python scripts/eval/incremental_value_eval.py \
      --run-dir outputs/colab-tokenization/live-run/artifacts/T1_s42_seed42 --step 2500 \
      --cache-dir outputs/colab-tokenization-10k/local-eval/cache --out outputs/incremental-value
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import fit_normalizer, load_panel  # noqa: E402
from src.evaluation import date_block_interval, evaluate_arrays, make_forward_labels, rank_ic  # noqa: E402
from src.expanded_eval import (FORMAT_VERSION, _crop, _hash_array, _hash_json, _normalized_view,  # noqa: E402
                                       _write_json, cached_encoder_features, cached_raw_features,
                                       context_eligibility, peer_groups, shared_origin_mask)
from src.model import FutureJEPA, PilotConfig  # noqa: E402
from src.runner import source_fingerprint  # noqa: E402
from src.training import load_snapshot, seed_everything  # noqa: E402


def daily_ics(predictions, model):
    by_date = {}
    for r in predictions:
        if r["model"] == model and r["cohort"] == "development" and r["target"] is not None:
            by_date.setdefault(r["origin_date"], []).append(r)
    return {d: rank_ic([r["prediction"] for r in g], [r["target"] for r in g]) for d, g in sorted(by_date.items())}


def paired(a, b):
    common = [d for d in a if d in b and a[d] is not None and b[d] is not None]
    diff = [a[d] - b[d] for d in common]
    lo, hi = date_block_interval(diff, 2000, 20, 42) if len(diff) > 40 else (None, None)
    return {"mean_difference": float(np.mean(diff)) if diff else None, "ci95": [lo, hi], "dates": len(diff)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    run_dir, cache, out = Path(args.run_dir), Path(args.cache_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or f"{run_dir.name}@{args.step}"
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    snapshot = run_dir / f"encoder_step_{args.step:06d}.pt"
    started = time.perf_counter()
    expanded_panel = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
    original_panel = load_panel(ROOT / "work/fjepa-expanded-data/original_panel.npz")
    years = (2019, 2020, 2021)
    expanded, original = _crop(expanded_panel, max(years)), _crop(original_panel, max(years))
    cohorts = shared_origin_mask(expanded, original, context=60)
    eligible, common = cohorts["expanded"], cohorts["common"]
    encoder = load_snapshot(snapshot, device="cpu")
    source = source_fingerprint()
    feature_set = metadata["variant_spec"]["feature_set"]
    alignment = metadata.get("panel_metadata", {}).get("training_date_alignment")
    shared_training_dates = None
    if alignment:
        def training_dates(panel):
            valid = context_eligibility(panel, 60).sum(axis=1) >= 2
            valid &= panel.market_obs[:, 0]
            valid &= (panel.dates >= np.datetime64("2006-01-01")) & (panel.dates <= np.datetime64("2018-12-31"))
            return panel.dates[valid]
        shared_training_dates = np.intersect1d(training_dates(expanded), training_dates(original))
    max_stocks = int(metadata["inference_peer_partition_size"])
    groups, _ = peer_groups(expanded, eligible, peer_seed=42, max_stocks=max_stocks)
    expanded_fp = expanded_panel.metadata["dataset_fingerprint"]
    base = {"format": FORMAT_VERSION, "source": source, "expanded_dataset": expanded_fp,
            "last_input_date": str(expanded.dates[-1]), "readout_start": "2006-01-01",
            "eligibility": _hash_array(eligible), "symbols": list(expanded.symbols),
            "context": 60, "peer_seed": 42, "max_stocks": max_stocks, "training_date_alignment": alignment}
    view = _normalized_view(expanded, metadata["normalizer"], feature_set)
    model_identity = {**base, "normalizer": metadata["normalizer"], "feature_set": feature_set,
                      "model_config": metadata["model_config"]}
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    trained_id = _hash_json({**model_identity, "checkpoint_sha256": snapshot_hash})
    print(json.dumps({"stage": "encoding_trained", "label": label}), flush=True)
    trained, _ = cached_encoder_features(view, encoder, eligible, groups, cache / f"trained_{trained_id}", trained_id,
                                         device="cpu", batch_size=16, max_stocks=max_stocks)
    seed_everything(metadata["seed"])
    random_encoder = FutureJEPA(PilotConfig(**metadata["model_config"]))
    random_id = _hash_json({**model_identity, "random_seed": metadata["seed"]})
    print(json.dumps({"stage": "encoding_random"}), flush=True)
    random, _ = cached_encoder_features(view, random_encoder, eligible, groups, cache / f"random_{random_id}", random_id,
                                        device="cpu", batch_size=16, max_stocks=max_stocks)
    del view
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
    print(json.dumps({"stage": "raw_features"}), flush=True)
    raw = cached_raw_features(expanded, raw_normalizer, eligible, cache / f"raw_book_{raw_id}", raw_id, context=60)
    # Combined readouts: written only on common-cohort rows (sparse memmap files).
    T, K = eligible.shape
    combos = {}
    for name, extra in (("raw_plus_encoder", trained), ("raw_plus_random", random)):
        path = out / f"{label.replace('@', '_')}_{name}.f32"
        arr = np.memmap(path, mode="w+", dtype=np.float32, shape=(T, K, raw.shape[-1] + extra.shape[-1]))
        for t in range(T):
            m = common[t]
            if m.any():
                arr[t, m] = np.concatenate([raw[t, m], extra[t, m]], axis=-1)
        arr.flush()
        combos[name] = arr
    labels, entry_dates, ends = make_forward_labels(expanded, 5)
    print(json.dumps({"stage": "readouts", "elapsed": round(time.perf_counter() - started)}), flush=True)
    evaluated = evaluate_arrays({"raw": raw, "encoder": trained, "random": random, **combos}, labels, expanded.dates,
                                label_end_dates=ends, entry_dates=entry_dates, eligible=common,
                                symbols=expanded.symbols, years=years, horizon=5, purge=20,
                                bootstrap_samples=500, seed=42)
    metrics = evaluated["report"]["metrics"]
    daily = {m: daily_ics(evaluated["predictions"], m) for m in ("raw", "encoder", "random", "raw_plus_encoder", "raw_plus_random")}
    result = {"label": label, "snapshot": str(snapshot), "cohort": "common", "horizon": 5, "years": years,
              "mean_daily_rank_ic": {m: metrics[m]["development"]["mean_daily_rank_ic"] for m in daily},
              "ci95": {m: metrics[m]["development"]["rank_ic_95ci_date_blocks"] for m in daily},
              "paired": {"raw_plus_encoder_minus_raw": paired(daily["raw_plus_encoder"], daily["raw"]),
                         "raw_plus_random_minus_raw": paired(daily["raw_plus_random"], daily["raw"]),
                         "raw_plus_encoder_minus_raw_plus_random": paired(daily["raw_plus_encoder"], daily["raw_plus_random"]),
                         "encoder_minus_raw": paired(daily["encoder"], daily["raw"])},
              "seconds": round(time.perf_counter() - started, 1)}
    (out / f"{label.replace('@', '_')}.json").write_text(json.dumps(result, indent=1))
    for p in combos:
        (out / f"{label.replace('@', '_')}_{p}.f32").unlink(missing_ok=True)
    print("INCREMENTAL " + json.dumps({k: result[k] for k in ("label", "mean_daily_rank_ic", "paired", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
