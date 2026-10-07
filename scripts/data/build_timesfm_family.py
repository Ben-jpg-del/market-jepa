"""Build the TimesFM forecast family: frozen TimesFM 2.5 forecasts of each stock's own price history.

For every (date t, stock k) in the target mask, two univariate series ending at the close of t are
forecast 20 sessions ahead with frozen TimesFM 2.5 (200M, zero-shot):
  abs : log adjusted close
  rel : log adjusted close - log SPY adjusted close   (matches the SPY-relative target)
plus one SPY series per date. Features (15 channels, all in log-return units except spreads which
are quantile widths in the same units):
  tfm_abs_r5, tfm_abs_r10, tfm_abs_r20     point forecast at h minus last value
  tfm_abs_spread5, tfm_abs_spread20        q90 - q10 at h (forecast uncertainty)
  tfm_abs_gap20                            forecast r20 minus trailing 20-session return
  tfm_rel_*                                same six on the SPY-relative series
  tfm_spy_r5, tfm_spy_r20, tfm_spy_spread20
Point-in-time by construction: the input window ends at the close of t. Resumable: results are
written into a memmap with a per-date done flag; rerun with the same --out to continue.

Usage (local CPU, ~160 series/s on Apple silicon):
  work/colab-test-env/bin/python scripts/data/build_timesfm_family.py --mask outputs/timesfm-family/common_mask.npy \
      --out outputs/timesfm-family --context 256 --batch 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ("tfm_abs_r5", "tfm_abs_r10", "tfm_abs_r20", "tfm_abs_spread5", "tfm_abs_spread20", "tfm_abs_gap20",
            "tfm_rel_r5", "tfm_rel_r10", "tfm_rel_r20", "tfm_rel_spread5", "tfm_rel_spread20", "tfm_rel_gap20",
            "tfm_spy_r5", "tfm_spy_r20", "tfm_spy_spread20")
CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


def log(**info):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **info}), flush=True)


def load_model(context, horizon, batch):
    import torch
    import timesfm
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    model.compile(timesfm.ForecastConfig(max_context=context, max_horizon=max(32, horizon), normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=False, fix_quantile_crossing=True, per_core_batch_size=batch))
    return model


def summarize(point, quant, last, trailing20):
    """point (n, H), quant (n, H, 10) [q0.., q10=idx1 .. q90=idx9], last (n,), trailing20 (n,) -> (n, 6)."""
    r5 = point[:, 4] - last
    r10 = point[:, 9] - last
    r20 = point[:, 19] - last
    s5 = quant[:, 4, 9] - quant[:, 4, 1]
    s20 = quant[:, 19, 9] - quant[:, 19, 1]
    return np.stack([r5, r10, r20, s5, s20, r20 - trailing20], axis=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=str(ROOT / "work/fjepa-expanded-data/panel_raw.npz"))
    parser.add_argument("--mask", required=True, help="bool [T, K] .npy of rows to forecast (e.g. the common cohort)")
    parser.add_argument("--out", default=str(ROOT / "outputs/timesfm-family"))
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--min-context", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--limit-dates", type=int, default=None, help="smoke test: only the first N eligible dates")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.panel, allow_pickle=True)
    dates = z["dates"].astype("datetime64[D]"); symbols = list(z["symbols"])
    close = np.asarray(z["close"], dtype=np.float64); spy = np.asarray(z["benchmark_close"], dtype=np.float64)
    T, K = close.shape
    mask = np.load(args.mask)
    assert mask.shape == (T, K), mask.shape
    logp = np.where(close > 0, np.log(np.where(close > 0, close, 1.0)), np.nan)
    logspy = np.where(spy > 0, np.log(np.where(spy > 0, spy, 1.0)), np.nan)
    rel = logp - logspy[:, None]
    in_range = (dates >= np.datetime64(args.start)) & (dates <= np.datetime64(args.end))
    todo_dates = [t for t in range(T) if in_range[t] and mask[t].any()]
    if args.limit_dates:
        todo_dates = todo_dates[:args.limit_dates]
    config = {"checkpoint": CHECKPOINT, "context": args.context, "min_context": args.min_context, "horizon": args.horizon,
              "features": FEATURES, "start": args.start, "end": args.end, "panel": Path(args.panel).name,
              "mask": Path(args.mask).name, "normalize_inputs": True, "quantile_head": True}
    feat_path, done_path = out / "timesfm_features.f32", out / "done_dates.npy"
    mode = "r+" if feat_path.exists() else "w+"
    feats = np.memmap(feat_path, mode=mode, dtype=np.float32, shape=(T, K, len(FEATURES)))
    if mode == "w+":
        feats[:] = np.nan; feats.flush()
    done = np.load(done_path) if done_path.exists() else np.zeros(T, dtype=bool)
    (out / "config.json").write_text(json.dumps(config, indent=1))

    if not args.finalize_only:
        pending = [t for t in todo_dates if not done[t]]
        log(stage="start", dates_total=len(todo_dates), dates_pending=len(pending), rows=int(sum(mask[t].sum() for t in pending)))
        if pending:
            model = load_model(args.context, args.horizon, args.batch)
            started = time.perf_counter(); n_series = 0
            spy_cache = {}
            for i, t in enumerate(pending):
                ks = np.flatnonzero(mask[t])
                lo = max(0, t + 1 - args.context)
                series, owners, last_abs, last_rel, tr_abs, tr_rel = [], [], [], [], [], []
                for k in ks:
                    w_abs = logp[lo:t + 1, k]; w_rel = rel[lo:t + 1, k]
                    ok = np.isfinite(w_abs)
                    if ok.sum() < args.min_context or not np.isfinite(w_abs[-1]) or not np.isfinite(w_rel[-1]):
                        continue
                    first = np.argmax(ok)  # strip leading missing; interior gaps are linearly interpolated by TimesFM
                    series.append(w_abs[first:]); series.append(w_rel[first:]); owners.append(k)
                    last_abs.append(w_abs[-1]); last_rel.append(w_rel[-1])
                    t20 = t - 20
                    tr_abs.append(w_abs[-1] - logp[t20, k] if t20 >= 0 and np.isfinite(logp[t20, k]) else np.nan)
                    tr_rel.append(w_rel[-1] - rel[t20, k] if t20 >= 0 and np.isfinite(rel[t20, k]) else np.nan)
                # SPY series for this date
                w_spy = logspy[lo:t + 1]; spy_ok = np.isfinite(w_spy).sum() >= args.min_context and np.isfinite(w_spy[-1])
                if spy_ok:
                    series.append(w_spy[np.argmax(np.isfinite(w_spy)):])
                if not owners:
                    done[t] = True; continue
                point, quant = model.forecast(horizon=args.horizon, inputs=[np.asarray(s, dtype=np.float32) for s in series])
                n = len(owners)
                p_abs, q_abs = point[0:2 * n:2], quant[0:2 * n:2]
                p_rel, q_rel = point[1:2 * n:2], quant[1:2 * n:2]
                f_abs = summarize(p_abs, q_abs, np.array(last_abs), np.array(tr_abs))
                f_rel = summarize(p_rel, q_rel, np.array(last_rel), np.array(tr_rel))
                if spy_ok:
                    ps, qs = point[2 * n], quant[2 * n]
                    f_spy = np.array([ps[4] - w_spy[-1], ps[19] - w_spy[-1], qs[19, 9] - qs[19, 1]], dtype=np.float32)
                else:
                    f_spy = np.full(3, np.nan, dtype=np.float32)
                block = np.concatenate([f_abs, f_rel, np.broadcast_to(f_spy, (n, 3))], axis=1).astype(np.float32)
                feats[t, np.array(owners)] = block
                done[t] = True
                n_series += len(series)
                if (i + 1) % 25 == 0 or i + 1 == len(pending):
                    feats.flush(); np.save(done_path, done)
                    el = time.perf_counter() - started
                    log(stage="progress", dates_done=i + 1, dates_pending=len(pending) - i - 1, series=n_series,
                        series_per_s=round(n_series / el, 1), eta_min=round((len(pending) - i - 1) * (el / (i + 1)) / 60, 1))
            feats.flush(); np.save(done_path, done)
    # finalize: compressed archive with observation mask
    arr = np.array(feats)
    obs = np.isfinite(arr) & mask[..., None]
    arr = np.where(obs, arr, 0.0).astype(np.float32)
    np.savez_compressed(out / "timesfm_channels.npz", dates=dates, symbols=np.array(symbols), timesfm=arr, timesfm_obs=obs,
                        timesfm_names=np.array(FEATURES))
    yrs = dates.astype("datetime64[Y]").astype(int) + 1970
    sel = (yrs >= 2006) & (yrs <= 2021)
    cover = float(obs[sel][..., 0][mask[sel]].mean()) if mask[sel].any() else 0.0
    ranges = {}
    for i, n in enumerate(FEATURES):
        v = arr[sel][..., i][obs[sel][..., i]]
        ranges[n] = {"median": float(np.median(v)), "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99))} if len(v) else None
    summary = {"config": config, "dates_done": int(done[in_range].sum()), "dates_target": len(todo_dates),
               "coverage_of_mask_rows_2006_2021": cover, "value_ranges_2006_2021": ranges}
    (out / "timesfm_channels_summary.json").write_text(json.dumps(summary, indent=1))
    log(stage="finalized", coverage=round(cover, 4), dates_done=summary["dates_done"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
