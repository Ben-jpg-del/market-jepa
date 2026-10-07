"""Evaluate encoder exports locally as the watcher downloads them (common cohort, 5-day).

Polls the artifacts folder; whenever a run directory has encoder_step_<N>.pt for a requested
step and no local result yet, runs the standard evaluator on this Mac and appends the IC to
<out>/summary.json (one JSON line per result on stdout, prefixed LOCAL_IC). Exits when every
expected (run, step) has been evaluated, or on --once.

Usage:
  work/colab-test-env/bin/python scripts/eval/local_eval_poller.py --artifacts outputs/colab-tokenization-10k/live-run/artifacts \
      --out outputs/colab-tokenization-10k/local-eval --steps 2500 5000 7500 10000 --expect-runs 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel  # noqa: E402
from src.expanded_eval import evaluate_expanded_run  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[2500, 10000])
    parser.add_argument("--expect-runs", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    artifacts, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    results = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    expanded = original = None
    while True:
        pending = []
        for run_dir in sorted(p for p in artifacts.glob("*_seed*") if (p / "run_metadata.json").exists()):
            m = re.match(r"^(\w+?)_s(\d+)_seed\d+$", run_dir.name)
            if not m:
                continue
            arm, seed = m.group(1), int(m.group(2))
            for step in args.steps:
                key = f"{arm}_s{seed}@{step}"
                if key in results:
                    continue
                snapshot = run_dir / f"encoder_step_{step:06d}.pt"
                if snapshot.exists():
                    pending.append((key, arm, seed, step, run_dir, snapshot))
        for key, arm, seed, step, run_dir, snapshot in pending:
            if expanded is None:
                expanded = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
                original = load_panel(ROOT / "work/fjepa-expanded-data/original_panel.npz")
            started = time.perf_counter()
            try:
                report = evaluate_expanded_run(expanded, original, run_dir, snapshot=snapshot,
                                               output_dir=out / f"{run_dir.name}_step_{step:06d}", device="cpu",
                                               years=(2019, 2020, 2021), primary_horizon=5, horizons=(5,), peer_seed=42,
                                               batch_size=16, bootstrap_samples=500, cache_dir=out / "cache",
                                               cohort_names=("common",))
            except Exception as exc:  # keep polling; report the failure
                print(json.dumps({"key": key, "error": type(exc).__name__, "message": str(exc)[:300]}), flush=True)
                continue
            metrics = report["results"]["5"]["common"]["metrics"]
            row = {"arm": arm, "seed": seed, "step": step, "seconds": round(time.perf_counter() - started, 1)}
            for model in ("trained_encoder", "random_encoder", "raw_features"):
                row[model] = round(metrics[model]["development"]["mean_daily_rank_ic"], 5)
            ci = metrics["trained_encoder"]["development"].get("rank_ic_95ci_date_blocks")
            row["trained_ci"] = [round(v, 5) for v in ci] if ci else None
            results[key] = row
            summary_path.write_text(json.dumps(results, indent=1))
            print("LOCAL_IC " + json.dumps(row), flush=True)
        done = sum(1 for k in results)
        if done >= args.expect_runs * len(args.steps) or args.once:
            print(json.dumps({"poller_done": True, "results": done}), flush=True)
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
