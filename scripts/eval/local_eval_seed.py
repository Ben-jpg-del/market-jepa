"""Evaluate downloaded encoder exports locally (CPU) with the standard expanded evaluator.

Common cohort, 5-day readout by default. Writes under outputs/colab-tokenization/local-eval/
and prints one JSON line per finished (arm, step) with trained / random / raw IC.

Usage:
  work/colab-test-env/bin/python scripts/eval/local_eval_seed.py --seed 42 --steps 500 2500
"""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--artifacts", default=str(ROOT / "outputs" / "colab-tokenization" / "live-run" / "artifacts"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "colab-tokenization" / "local-eval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 2500])
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--cohorts", default="common")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--threads", type=int, default=max(1, (torch.get_num_threads() or 4)))
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    artifacts, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expanded = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
    original = load_panel(ROOT / "work/fjepa-expanded-data/original_panel.npz")
    cohorts = tuple(c for c in args.cohorts.split(",") if c)
    run_dirs = sorted(p for p in artifacts.glob(f"*_s{args.seed}_seed{args.seed}") if (p / "run_metadata.json").exists())
    if args.arms:
        run_dirs = [p for p in run_dirs if p.name.split(f"_s{args.seed}")[0] in args.arms]
    results = {}
    for step in args.steps:
        for run_dir in run_dirs:
            arm = run_dir.name.split(f"_s{args.seed}")[0]
            snapshot = run_dir / f"encoder_step_{step:06d}.pt"
            if not snapshot.exists():
                print(json.dumps({"arm": arm, "step": step, "skipped": "no snapshot"}), flush=True)
                continue
            started = time.perf_counter()
            report = evaluate_expanded_run(expanded, original, run_dir, snapshot=snapshot,
                                           output_dir=out / f"{run_dir.name}_step_{step:06d}", device="cpu",
                                           years=(2019, 2020, 2021), primary_horizon=args.horizon,
                                           horizons=(args.horizon,), peer_seed=42, batch_size=16,
                                           bootstrap_samples=500, cache_dir=out / "cache", cohort_names=cohorts)
            row = {"arm": arm, "seed": args.seed, "step": step, "seconds": round(time.perf_counter() - started, 1)}
            for cohort in cohorts:
                metrics = report["results"][str(args.horizon)][cohort]["metrics"]
                for model in ("trained_encoder", "random_encoder", "raw_features"):
                    dev = metrics[model]["development"]
                    row[f"{cohort}/{model}"] = round(dev["mean_daily_rank_ic"], 5)
                ci = metrics["trained_encoder"]["development"].get("rank_ic_95ci_date_blocks")
                row[f"{cohort}/trained_ci"] = [round(v, 5) for v in ci] if ci else None
            results[f"{arm}@{step}"] = row
            print("LOCAL_IC " + json.dumps(row), flush=True)
            (out / f"summary_seed{args.seed}.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
