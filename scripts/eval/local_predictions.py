"""Regenerate walk-forward out-of-sample predictions locally from the saved yearly encoders (teacher / random kinds).

The Colab job's per-row predictions were not downloaded; this reproduces them with the same code path
(walkforward_readout.window_view + expanded_eval.cached_encoder_features + walkforward_readout.walk_forward_ridge, same row
thinning) for the test years requested, so stacking / incremental-value tests can run on the Mac.

  work/colab-test-env/bin/python scripts/eval/local_predictions.py --seeds 42,43 --test-years 2022,2023,2024,2025 --device mps
Outputs: <out>/v2_s<seed>/features_year_<Y>.npz (float16 teacher/random rows), <out>/v2_s<seed>/predictions_<Y>.npz,
<out>/v2_s<seed>/readout_<Y>.json (mean daily IC per kind/horizon, to compare with the Colab readouts).
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel  # noqa: E402
from src.evaluation import context_eligibility, make_forward_labels  # noqa: E402
from src.expanded_eval import _normalized_view, cached_encoder_features, peer_groups  # noqa: E402
from src.model import FutureJEPA, PilotConfig  # noqa: E402
from src.training import seed_everything  # noqa: E402
from src.walkforward_data import normal_scores_by_date  # noqa: E402
from src.walkforward_readout import TeacherView, daily_ic, walk_forward_ridge, window_view  # noqa: E402


def log(**info):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **info}), flush=True)


def extract(panel, encoders, normalizer, year_mask, eligible, cache_dir, *, device, batch_size, context, signature):
    """Like walkforward_readout.extract_year_features but for the given {kind: encoder} only; cache kept on disk."""
    rows_t = np.flatnonzero(year_mask & eligible.any(axis=1))
    view, offset = window_view(panel, rows_t[0], rows_t[-1], context)
    normalized = _normalized_view(view, normalizer, "book", cutoff_limit=None)
    elig_view = np.zeros(normalized.universe.shape, bool); elig_view[rows_t - offset] = eligible[rows_t]
    groups, _ = peer_groups(normalized, elig_view, peer_seed=42, max_stocks=32)
    out = {}
    for kind, encoder in encoders.items():
        feats, _ = cached_encoder_features(normalized, encoder, elig_view, groups, Path(cache_dir) / kind, f"{signature}|{kind}",
                                           device=device, batch_size=batch_size, max_stocks=32, phase=f"features_{kind}")
        t_v, k_v = np.nonzero(elig_view)
        out[kind] = np.asarray(feats[t_v, k_v], dtype=np.float16); del feats
    out["t_idx"], out["k_idx"] = t_v + offset, k_v
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default=str(ROOT / "work/fjepa-assembled-data/panel_russell2000.npz"))
    p.add_argument("--artifacts", default=str(ROOT / "outputs/russell2000-walkforward/live-run/artifacts"))
    p.add_argument("--out", default=str(ROOT / "outputs/russell2000-walkforward/local-predictions"))
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--years", default=",".join(str(y) for y in range(2015, 2026)), help="encoder years (features)")
    p.add_argument("--test-years", default="2022,2023,2024,2025")
    p.add_argument("--kinds", default="teacher,random")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-start", default="2006-01-01")
    p.add_argument("--max-history-rows", type=int, default=500_000)
    p.add_argument("--max-readout-rows", type=int, default=600_000)
    p.add_argument("--horizons", default="5,20")
    p.add_argument("--keep-cache", action="store_true")
    args = p.parse_args(argv)
    years = [int(y) for y in args.years.split(",")]; test_years = [int(y) for y in args.test_years.split(",")]
    kinds = args.kinds.split(","); horizons = [int(h) for h in args.horizons.split(",")]
    started = time.perf_counter()
    raw = load_panel(args.panel)
    dates = raw.dates.astype("datetime64[D]"); year_of = dates.astype("datetime64[Y]").astype(int) + 1970
    context = 60
    eligible = context_eligibility(raw, context) & (dates >= np.datetime64(args.train_start))[:, None] & raw.market_obs[:, 0, None]
    labels = {h: make_forward_labels(raw, h) for h in horizons}
    rank_labels = {h: normal_scores_by_date(np.where(eligible & np.isfinite(labels[h][0]), labels[h][0], np.nan), eligible) for h in horizons}
    log(stage="panel_ready", rows=int(eligible.sum()), elapsed=round(time.perf_counter() - started))
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        run = f"v2_s{seed}"; run_dir = Path(args.artifacts) / f"{run}_seed{seed}"; out = out_root / run; out.mkdir(exist_ok=True)
        random_model = None
        for year in years:
            target = out / f"features_year_{year}.npz"
            if target.exists():
                continue
            t0 = time.perf_counter()
            payload = torch.load(run_dir / f"encoder_{year}.pt", map_location="cpu", weights_only=False)
            cfg = PilotConfig(**payload["model_config"]); model = FutureJEPA(cfg); model.load_state_dict(payload["model"]); model.eval()
            if random_model is None:
                seed_everything(seed); random_model = FutureJEPA(cfg).eval()
            encoders = {k: {"teacher": TeacherView(model), "online": model, "random": random_model}[k] for k in kinds}
            masks = {"year": year_of == year}
            if year == years[0]:
                history = (year_of < year) & (dates >= np.datetime64(args.train_start))
                rows = int(eligible[history].sum()); stride = max(1, int(np.ceil(rows / args.max_history_rows)))
                keep = np.zeros(len(dates), bool); idx = np.flatnonzero(history); keep[idx[::stride]] = True
                masks["history"] = keep
                log(stage="history_sampling", rows_available=rows, date_stride=stride, rows_kept=int(eligible[keep].sum()))
            parts = []
            for name, mask in masks.items():
                cache = out / "cache" / f"{year}_{name}"
                feats = extract(raw, encoders, payload["normalizer"], mask, eligible, cache, device=args.device, batch_size=args.batch_size,
                                context=context, signature=f"{run}|{year}|{name}")
                parts.append(feats)
                if not args.keep_cache:
                    shutil.rmtree(cache, ignore_errors=True)
            merged = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
            np.savez(target, **merged)
            log(stage="features", run=run, year=year, rows=int(len(merged["t_idx"])), seconds=round(time.perf_counter() - t0))
            del model, payload, merged, parts; gc.collect()
        for year in test_years:
            if (out / f"predictions_{year}.npz").exists():
                continue
            t0 = time.perf_counter()
            files = [out / f"features_year_{y}.npz" for y in years if y <= year]
            all_t = np.concatenate([np.load(f)["t_idx"] for f in files]); test_rows = year_of[all_t] == year
            n_train = int((~test_rows).sum()); train_dates = np.unique(all_t[~test_rows])
            stride = int(np.ceil(n_train / args.max_readout_rows)) if n_train > args.max_readout_rows else 1
            keep_dates = np.zeros(len(dates), bool); keep_dates[train_dates[::stride]] = True; keep_dates[year_of == year] = True
            cat = {}
            for f in files:
                z = np.load(f); keep = keep_dates[z["t_idx"]]
                for k in z.files:
                    cat.setdefault(k, []).append(z[k][keep])
            cat = {k: np.concatenate(v) for k, v in cat.items()}
            t_idx, k_idx = cat["t_idx"], cat["k_idx"]
            result = {"year": year, "run": run, "rows_total": int(len(t_idx)), "rows_year": int((year_of[t_idx] == year).sum()),
                      "readout_sampling": {"train_rows_available": n_train, "date_stride": stride, "rows_used": int(len(t_idx))}, "models": {}}
            preds = {"t_idx": t_idx, "k_idx": k_idx}
            for h in horizons:
                lab, _, ends = labels[h]
                y_raw = lab[t_idx, k_idx].astype(np.float64); y_rank = rank_labels[h][t_idx, k_idx].astype(np.float64)
                for kind in kinds:
                    pred_map, fold = walk_forward_ridge(cat[kind], y_rank, y_raw, t_idx, dates, ends, year, purge=max(20, h))
                    pred = np.full(len(t_idx), np.nan)
                    for i, v in pred_map.items():
                        pred[i] = v
                    ic = daily_ic(pred, y_raw, t_idx, dates)
                    result["models"].setdefault(f"{kind}_ridge", {})[f"h{h}"] = {"fold": fold, "mean_ic": float(np.mean(list(ic.values()))) if ic else None, "daily_ic": ic}
                    preds[f"{kind}_ridge_h{h}"] = pred.astype(np.float32)
            np.savez_compressed(out / f"predictions_{year}.npz", **preds)
            (out / f"readout_{year}.json").write_text(json.dumps(result))
            log(stage="readout", run=run, year=year, mean_ic={m: {h: round(v["mean_ic"], 4) for h, v in hs.items()} for m, hs in result["models"].items()},
                seconds=round(time.perf_counter() - t0))
    log(stage="done", elapsed=round(time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
