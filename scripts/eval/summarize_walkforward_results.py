"""Summarize a walk-forward run: per-seed and seed-mean IC by window, paired differences vs random and vs the raw bar.

  work/colab-test-env/bin/python scripts/eval/summarize_walkforward_results.py --artifacts outputs/russell2000-walkforward/live-run/artifacts \
      --gate outputs/raw-feature-gate --gate-tag russell2000 --out outputs/russell2000-walkforward

Reads evaluations/<run>/year_<Y>/readout.json (daily IC per model x horizon) and, if present, the Phase-0 gate
daily_ic.json files (ridge|v2raw, lgbm|v2raw) for the paired comparison against the raw bar.
Writes <out>/summary.json and <out>/RESULTS_TABLES.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation import date_block_interval  # noqa: E402

WINDOWS = {"2015-18": (2015, 2018), "2019-21": (2019, 2021), "2022-25": (2022, 2025)}


def window_mean(daily, lo, hi):
    v = [x for k, x in daily.items() if lo <= int(k[:4]) <= hi]
    return float(np.mean(v)) if v else None


def paired(a, b, lo, hi):
    keys = [k for k in a if k in b and lo <= int(k[:4]) <= hi]
    d = [a[k] - b[k] for k in keys]
    if len(d) < 25:
        return None
    ci = date_block_interval(d)
    return {"mean": float(np.mean(d)), "ci95": [float(ci[0]), float(ci[1])], "dates": len(d)}


def fmt(x):
    return "—" if x is None else f"{x:+.4f}"


def fmtp(p):
    return "—" if p is None else f"{p['mean']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--gate", default=None)
    parser.add_argument("--gate-tag", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    arts = Path(args.artifacts); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for run_dir in sorted((arts / "evaluations").glob("v2_s*")):
        daily = {}
        for rd in sorted(run_dir.glob("year_*/readout.json")):
            r = json.loads(rd.read_text())
            for m, hs in r["models"].items():
                for h, v in hs.items():
                    daily.setdefault(m, {}).setdefault(h, {}).update(v["daily_ic"])
        if daily:
            runs[run_dir.name] = daily
    if not runs:
        raise SystemExit("no readouts found")
    gate = {}
    if args.gate and args.gate_tag:
        for h in ("5", "20"):
            p = Path(args.gate) / f"{args.gate_tag}_h{h}" / "daily_ic.json"
            if p.exists():
                for k, d in json.loads(p.read_text()).items():
                    gate.setdefault(k, {})[f"h{h}"] = d
    horizons = sorted({h for d in runs.values() for hs in d.values() for h in hs}, key=lambda s: int(s[1:]))
    models = ["teacher_ridge", "online_ridge", "aux_direct", "random_ridge"]
    summary = {"runs": {}, "seed_mean": {}, "gate": {}, "paired": {}}
    for run, daily in runs.items():
        summary["runs"][run] = {m: {h: {w: window_mean(daily[m][h], lo, hi) for w, (lo, hi) in WINDOWS.items()} for h in daily.get(m, {})} for m in models if m in daily}
    for m in models:
        for h in horizons:
            series = [runs[r][m][h] for r in runs if m in runs[r] and h in runs[r][m]]
            if not series:
                continue
            keys = set.intersection(*(set(s) for s in series))
            mean_daily = {k: float(np.mean([s[k] for s in series])) for k in sorted(keys)}  # date order: the block bootstrap needs it
            summary["seed_mean"].setdefault(m, {})[h] = {w: window_mean(mean_daily, lo, hi) for w, (lo, hi) in WINDOWS.items()}
            summary["seed_mean"][m][h]["_daily"] = mean_daily
    for k, hs in gate.items():
        summary["gate"][k] = {h: {w: window_mean(d, lo, hi) for w, (lo, hi) in WINDOWS.items()} for h, d in hs.items()}
    for m in ("teacher_ridge", "online_ridge", "aux_direct"):
        for h in horizons:
            if m not in summary["seed_mean"] or h not in summary["seed_mean"][m]:
                continue
            a = summary["seed_mean"][m][h]["_daily"]
            comps = {"random_ridge": summary["seed_mean"].get("random_ridge", {}).get(h, {}).get("_daily")}
            for gk, hs in gate.items():
                comps[gk] = hs.get(h)
            for name, b in comps.items():
                if b:
                    summary["paired"][f"{m}-{name}|{h}"] = {w: paired(a, b, lo, hi) for w, (lo, hi) in WINDOWS.items()}
    for m in summary["seed_mean"]:
        for h in summary["seed_mean"][m]:
            summary["seed_mean"][m][h].pop("_daily", None)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# walk-forward results", "", f"Runs: {', '.join(runs)}. Mean daily rank IC of the SPY-relative forward return; walk-forward encoder retraining each year; ridge readouts fit on earlier years' features from their own year's model.", ""]
    for h in horizons:
        lines += [f"## Horizon {h[1:]} days", "", "| model | " + " | ".join(WINDOWS) + " |", "|---|" + "---|" * len(WINDOWS)]
        for m in models:
            if m in summary["seed_mean"] and h in summary["seed_mean"][m]:
                lines.append(f"| {m} (seed mean) | " + " | ".join(fmt(summary["seed_mean"][m][h][w]) for w in WINDOWS) + " |")
        for gk in summary["gate"]:
            if h in summary["gate"][gk]:
                lines.append(f"| {gk} (gate) | " + " | ".join(fmt(summary["gate"][gk][h][w]) for w in WINDOWS) + " |")
        lines += ["", "Paired per-date differences (seed mean; moving-block 95% CI):", "", "| comparison | " + " | ".join(WINDOWS) + " |", "|---|" + "---|" * len(WINDOWS)]
        for key, ws in summary["paired"].items():
            if key.endswith("|" + h):
                lines.append(f"| {key[:-len(h) - 1]} | " + " | ".join(fmtp(ws[w]) for w in WINDOWS) + " |")
        lines.append("")
        lines += ["Per seed:", ""]
        for run, d in summary["runs"].items():
            for m in models:
                if m in d and h in d[m]:
                    lines.append(f"- {run} {m}: " + ", ".join(f"{w} {fmt(d[m][h][w])}" for w in WINDOWS))
        lines.append("")
    (out / "RESULTS_TABLES.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
