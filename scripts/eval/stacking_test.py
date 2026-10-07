"""Incremental-value (stacking) test: does the encoder prediction add IC on top of the raw-feature LightGBM?

Inputs (all out of sample, fit on years before each test year):
  gate predictions  outputs/raw-feature-gate-pred/<tag>_h<h>/predictions.npz   (ridge_v2raw, lgbm_v2raw per eligible row)
  walk-forward predictions    outputs/russell2000-walkforward/local-predictions/v2_s<seed>/predictions_<Y>.npz (teacher_ridge_h<h>, random_ridge_h<h>)
Per date, every prediction is rank-normalised (normal scores); combined = z(base) + w * z(encoder).
w is chosen on the previous test year (expanding, out of sample) from a grid; fixed-w rows are reported too.
Scored: mean daily rank IC 2022-2025, paired per-date difference combined - base with a 20-day moving-block CI,
and the mean per-date rank correlation between the encoder and base predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from src.data import load_panel  # noqa: E402
from src.evaluation import date_block_interval, make_forward_labels, rank_ic  # noqa: E402
from supervised_bar import normal_scores_by_group  # noqa: E402

GRID = (0.25, 0.5, 1.0, 2.0)


def daily(pred, y, t_idx, dates):
    out = {}
    good = np.isfinite(pred) & np.isfinite(y)
    for t in np.unique(t_idx[good]):
        sel = good & (t_idx == t); ic = rank_ic(pred[sel], y[sel])
        if ic is not None:
            out[str(dates[t])] = float(ic)
    return out


def paired(a, b, lo, hi):
    keys = sorted(k for k in a if k in b and lo <= int(k[:4]) <= hi); d = [a[k] - b[k] for k in keys]
    if len(d) < 25:
        return None
    ci = date_block_interval(d)
    return {"mean": float(np.mean(d)), "ci95": [float(ci[0]), float(ci[1])], "dates": len(d)}


def window(d, lo, hi):
    v = [x for k, x in d.items() if lo <= int(k[:4]) <= hi]
    return float(np.mean(v)) if v else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default=str(ROOT / "work/fjepa-assembled-data/panel_russell2000.npz"))
    p.add_argument("--gate", default=str(ROOT / "outputs/raw-feature-gate-pred")); p.add_argument("--gate-tag", default="russell2000")
    p.add_argument("--predictions", default=str(ROOT / "outputs/russell2000-walkforward/local-predictions"))
    p.add_argument("--seeds", default="42,43"); p.add_argument("--horizons", default="5,20")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="first year is only used to pick w for the next")
    p.add_argument("--window", default="2022,2025")
    p.add_argument("--out", default=str(ROOT / "outputs/russell2000-walkforward/stack_test.json"))
    args = p.parse_args(argv)
    lo, hi = [int(x) for x in args.window.split(",")]; years = [int(y) for y in args.years.split(",")]
    panel = load_panel(args.panel); dates = panel.dates.astype("datetime64[D]"); year_of = dates.astype("datetime64[Y]").astype(int) + 1970
    K = len(panel.symbols)
    report = {"window": [lo, hi], "grid": GRID, "results": {}}
    for h in [int(x) for x in args.horizons.split(",")]:
        g = np.load(Path(args.gate) / f"{args.gate_tag}_h{h}" / "predictions.npz")
        g_key = g["t_idx"].astype(np.int64) * K + g["k_idx"]; g_pos = {int(k): i for i, k in enumerate(g_key)}
        labels = make_forward_labels(panel, h)[0]
        for seed in [int(s) for s in args.seeds.split(",")]:
            rows = []  # one record per test-year row: t, k, y, base preds, encoder preds
            for y in years:
                z = np.load(Path(args.predictions) / f"v2_s{seed}" / f"predictions_{y}.npz")
                sel = year_of[z["t_idx"]] == y
                t, k = z["t_idx"][sel], z["k_idx"][sel]
                pos = np.array([g_pos.get(int(a * K + b), -1) for a, b in zip(t, k)])
                ok = pos >= 0
                rec = {"t": t[ok], "k": k[ok], "y": labels[t[ok], k[ok]], "lgbm": g["lgbm_v2raw"][pos[ok]], "raw_ridge": g["ridge_v2raw"][pos[ok]],
                       "teacher": z[f"teacher_ridge_h{h}"][sel][ok], "random": z[f"random_ridge_h{h}"][sel][ok], "year": y, "unmatched": int((~ok).sum())}
                rows.append(rec)
            t_idx = np.concatenate([r["t"] for r in rows]); y_raw = np.concatenate([r["y"] for r in rows]).astype(np.float64)
            cols = {n: np.concatenate([r[n] for r in rows]).astype(np.float64) for n in ("lgbm", "raw_ridge", "teacher", "random")}
            zs = {n: normal_scores_by_group(v[:, None], t_idx)[:, 0].astype(np.float64) for n, v in cols.items()}
            yr = year_of[t_idx]
            res = {"unmatched_rows": int(sum(r["unmatched"] for r in rows)), "rows": int(len(t_idx)), "ic": {}, "paired_vs_base": {}, "chosen_w": {}, "corr": {}}
            singles = {n: daily(zs[n], y_raw, t_idx, dates) for n in zs}
            for n, d in singles.items():
                res["ic"][n] = window(d, lo, hi)
            for enc in ("teacher", "random"):
                for base in ("lgbm", "raw_ridge"):
                    # fixed grid
                    for w in GRID:
                        d = daily(zs[base] + w * zs[enc], y_raw, t_idx, dates)
                        res["ic"][f"{base}+{w}*{enc}"] = window(d, lo, hi)
                        res["paired_vs_base"][f"{base}+{w}*{enc}"] = paired(d, singles[base], lo, hi)
                    # w chosen on the previous year (out of sample)
                    comb = np.full(len(t_idx), np.nan); chosen = {}
                    for y in years[1:]:
                        prev = yr == y - 1; best, best_ic = None, -np.inf
                        for w in GRID:
                            d = daily((zs[base] + w * zs[enc])[prev], y_raw[prev], t_idx[prev], dates)
                            ic = window(d, y - 1, y - 1) if d else -np.inf
                            if ic is not None and ic > best_ic:
                                best, best_ic = w, ic
                        chosen[str(y)] = best; sel = yr == y
                        comb[sel] = zs[base][sel] + best * zs[enc][sel]
                    d = daily(comb, y_raw, t_idx, dates)
                    res["ic"][f"{base}+w*{enc} (w from prior year)"] = window(d, lo, hi)
                    res["paired_vs_base"][f"{base}+w*{enc} (w from prior year)"] = paired(d, singles[base], lo, hi)
                    res["chosen_w"][f"{base}+{enc}"] = chosen
                    # per-date rank correlation between encoder and base predictions
                    cors = [rank_ic(zs[enc][t_idx == t], zs[base][t_idx == t]) for t in np.unique(t_idx[(yr >= lo) & (yr <= hi)])]
                    res["corr"][f"{enc}~{base}"] = float(np.nanmean([c for c in cors if c is not None]))
            report["results"][f"h{h}|s{seed}"] = res
            print(f"== h{h} seed {seed}: rows {res['rows']} (unmatched {res['unmatched_rows']}); IC lgbm {res['ic']['lgbm']:+.4f} raw_ridge {res['ic']['raw_ridge']:+.4f} "
                  f"teacher {res['ic']['teacher']:+.4f} random {res['ic']['random']:+.4f}; corr teacher~lgbm {res['corr']['teacher~lgbm']:.2f} random~lgbm {res['corr']['random~lgbm']:.2f}")
            for key, pv in res["paired_vs_base"].items():
                if pv:
                    print(f"   {key:36s} IC {res['ic'][key]:+.4f}  vs base {pv['mean']:+.4f} [{pv['ci95'][0]:+.4f}, {pv['ci95'][1]:+.4f}]")
            print("   chosen w:", res["chosen_w"])
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
