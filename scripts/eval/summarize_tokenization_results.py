"""Summarize a run-table experiment (tokenization / target arms) from downloaded reports.

Reads <artifacts>/expanded_job_state.json and <artifacts>/evaluations/<run>_step_<n>/expanded_job_report.json,
groups runs by arm (run name prefix before `_s<seed>`), and prints/writes:
  * per arm × checkpoint: mean IC over seeds (trained / random / raw), per-seed values,
    and the 2,500 − 500 change per seed (primary: 5-day, common cohort);
  * projection-norm ratios 2,500/500 per channel from training_metrics.json when present;
  * the pre-registered rule outcomes from PROTOCOL.md.
Usage: python scripts/eval/summarize_tokenization_results.py [--artifacts outputs/colab-tokenization/live-run/artifacts]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAST = {"r1", "overnight_gap", "log_high_low", "relative_volume"}


def arm_and_seed(run_name: str) -> tuple[str, int | None]:
    m = re.match(r"^(.*)_s(\d+)$", run_name)
    return (m.group(1), int(m.group(2))) if m else (run_name, None)


def load_reports(artifacts: Path):
    rows = []
    for report_path in sorted(artifacts.glob("evaluations/*/expanded_job_report.json")):
        label = report_path.parent.name
        m = re.match(r"^(.*)_step_(\d+)$", label)
        if not m:
            continue
        run_name, step = m.group(1), int(m.group(2))
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            continue
        arm, seed = arm_and_seed(run_name)
        for horizon, cohorts in report.get("results", {}).items():
            for cohort, block in cohorts.items():
                metrics = (block or {}).get("metrics") or {}
                entry = {"run": run_name, "arm": arm, "seed": seed, "step": step, "horizon": int(horizon), "cohort": cohort}
                for model in ("trained_encoder", "random_encoder", "raw_features"):
                    dev = (metrics.get(model) or {}).get("development") or {}
                    entry[model] = dev.get("mean_daily_rank_ic")
                    ci = dev.get("rank_ic_95ci_date_blocks")
                    entry[model + "_ci"] = ci if isinstance(ci, list) else None
                rows.append(entry)
    return rows


def projection_ratios(artifacts: Path):
    """Per run: {channel: norm@2500 / norm@500} from the saved training history."""
    out = {}
    for metrics_path in sorted(artifacts.glob("*_seed*/training_metrics.json")):
        try:
            history = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        by_step = {row["step"]: row.get("proj_norm") for row in history if row.get("proj_norm")}
        if 500 in by_step and 2500 in by_step:
            ratios = {}
            for key, late in by_step[2500].items():
                early = by_step[500].get(key)
                if key.startswith("proj_norm/stock/") and not key.endswith("_total_values") and early:
                    ratios[key.split("/")[-1]] = late / early
            out[metrics_path.parent.name] = ratios
    return out


def fmt(x):
    return "   n/a " if x is None else f"{x:+.5f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=str(ROOT / "outputs" / "colab-tokenization" / "live-run" / "artifacts"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--cohort", default="common")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    artifacts = Path(args.artifacts)
    rows = load_reports(artifacts)
    if not rows:
        print("no evaluation reports found under", artifacts)
        return 1
    primary = [r for r in rows if r["horizon"] == args.horizon and r["cohort"] == args.cohort]
    by_arm_step = defaultdict(list)
    for r in primary:
        by_arm_step[(r["arm"], r["step"])].append(r)
    arms = sorted({a for a, _ in by_arm_step})
    steps = sorted({s for _, s in by_arm_step})
    print(f"\n== mean daily rank IC, h{args.horizon}, {args.cohort} cohort (mean over seeds; per-seed in brackets) ==")
    print(f"{'arm':6s} {'step':>5s} {'n':>2s} {'trained':>9s} {'random':>9s} {'raw':>9s}   per-seed trained")
    summary = {}
    for arm in arms:
        for step in steps:
            group = by_arm_step.get((arm, step), [])
            if not group:
                continue
            def mean(key):
                vals = [g[key] for g in group if g[key] is not None]
                return sum(vals) / len(vals) if vals else None
            per_seed = ", ".join(f"s{g['seed']}={g['trained_encoder']:+.4f}" for g in sorted(group, key=lambda g: g['seed'] or 0) if g['trained_encoder'] is not None)
            print(f"{arm:6s} {step:5d} {len(group):2d} {fmt(mean('trained_encoder')):>9s} {fmt(mean('random_encoder')):>9s} {fmt(mean('raw_features')):>9s}   [{per_seed}]")
            summary[f"{arm}@{step}"] = {"n": len(group), "trained_mean": mean("trained_encoder"), "random_mean": mean("random_encoder"),
                                        "raw_mean": mean("raw_features"), "per_seed": {g["seed"]: g["trained_encoder"] for g in group}}
    print("\n== change 2,500 - 500 per seed (trained) ==")
    deltas = {}
    for arm in arms:
        early = {g["seed"]: g["trained_encoder"] for g in by_arm_step.get((arm, 500), [])}
        late = {g["seed"]: g["trained_encoder"] for g in by_arm_step.get((arm, 2500), [])}
        common = [s for s in early if s in late and early[s] is not None and late[s] is not None]
        if not common:
            continue
        d = {s: late[s] - early[s] for s in common}
        deltas[arm] = d
        mean_d = sum(d.values()) / len(d)
        declines = sum(1 for v in d.values() if v < 0)
        print(f"{arm:6s} mean {mean_d:+.5f}  declines {declines}/{len(d)}  " + ", ".join(f"s{s}={v:+.4f}" for s, v in sorted(d.items())))
    ratios = projection_ratios(artifacts)
    if ratios:
        print("\n== projection-norm ratio 2,500/500, mean over seeds per arm (non-target channels marked *) ==")
        by_arm = defaultdict(list)
        for run_dir, r in ratios.items():
            by_arm[arm_and_seed(run_dir.rsplit("_seed", 1)[0])[0]].append(r)
        channels = sorted({c for r in ratios.values() for c in r})
        header = "channel".ljust(30) + "".join(f"{a:>8s}" for a in arms)
        print(header)
        for c in channels:
            tag = " " if c in FAST else "*"
            line = f"{tag}{c:29s}"
            for a in arms:
                vals = [r[c] for r in by_arm.get(a, []) if c in r]
                line += f"{(sum(vals)/len(vals)):8.2f}" if vals else "     n/a"
            print(line)
    # pre-registered rules
    print("\n== pre-registered rules (PROTOCOL.md) ==")
    verdict = {}
    for arm, d in deltas.items():
        n = len(d)
        fixed = sum(1 for v in d.values() if v >= -0.003) >= max(2, (n + 1) // 2) if n else None
        verdict[arm] = {"fixes_deterioration": fixed, "declines": sum(1 for v in d.values() if v < 0), "seeds": n}
        retained = None
        arm_ratios = [r for run_dir, r in ratios.items() if arm_and_seed(run_dir.rsplit('_seed', 1)[0])[0] == arm]
        if arm_ratios:
            worst = min(min(v for c, v in r.items() if c not in FAST) for r in arm_ratios if any(c not in FAST for c in r))
            retained = worst >= 0.95
            verdict[arm]["worst_nontarget_ratio"] = worst
        verdict[arm]["retains_channels"] = retained
        print(f"{arm:6s} fixes_deterioration={fixed} retains_channels={retained} ({verdict[arm].get('worst_nontarget_ratio', 'n/a')})")
    out_path = Path(args.out) if args.out else artifacts.parent / "summary.json"
    out_path.write_text(json.dumps({"primary": {"horizon": args.horizon, "cohort": args.cohort}, "summary": summary,
                                    "deltas": deltas, "projection_ratios": ratios, "verdict": verdict, "rows": rows}, indent=1))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
