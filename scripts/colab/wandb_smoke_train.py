"""End-to-end W&B smoke test: a tiny synthetic F-JEPA run that logs to the f-jepa project.

Runs locally (CPU) or on a Colab GPU. Uses src.data.synthetic_panel, so no
market data or FMP key is involved. The run is tagged ``smoke`` so it can be filtered
out of research views. Exit code 0 only if the run went online and finished.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    HERE = Path(__file__).resolve()
except NameError:  # executed inside a notebook kernel
    HERE = Path.cwd() / "wandb_smoke_train.py"
for candidate in (HERE.parents[2], HERE.parents[1], HERE.parent, Path("/content/fjepa_expanded")):
    if (candidate / "src").is_dir():
        sys.path.insert(0, str(candidate))
        break

from src.data import synthetic_panel  # noqa: E402
from src.model import PilotConfig  # noqa: E402
from src.runner import run_variant  # noqa: E402
from src.wandb_logging import WandbTrainingLogger, deterministic_run_id, resolve_wandb_key  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--variant", default="F")
    parser.add_argument("--mode", default="auto", choices=("auto", "online", "offline", "disabled"))
    parser.add_argument("--group", default="smoke")
    parser.add_argument("--out", default=None)
    parser.add_argument("--require-online", action="store_true")
    args = parser.parse_args(argv)

    key_source = "env" if os.environ.get("WANDB_API_KEY") else ("colab_secrets" if resolve_wandb_key() else "none")
    print(json.dumps({"stage": "key", "source": key_source}), flush=True)
    if args.require_online and key_source == "none":
        print(json.dumps({"error": "no WANDB_API_KEY available"}), flush=True)
        return 3

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="fjepa-wandb-smoke-"))
    panel = synthetic_panel(n_dates=400, n_stocks=8, seed=7)
    host = os.uname().nodename
    logger = WandbTrainingLogger(
        run_name=f"smoke | {args.variant} | {host} | {time.strftime('%Y%m%d-%H%M%S')}",
        run_id=deterministic_run_id(group=args.group, host=host, t=time.time()),
        group=args.group, job_type="smoke", tags=["smoke", f"variant_{args.variant}", "synthetic"],
        config={"steps": args.steps, "device": args.device, "synthetic": True}, mode=args.mode, log_every=5,
        directory=out)
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    result = run_variant(panel, args.variant, out / f"{args.variant}_smoke", seed=1, screen_steps=args.steps,
                         total_steps=args.steps, batch_dates=2, stocks=4, device=device,
                         model_config=PilotConfig(width=32, predictor_width=16, temporal_layers=1,
                                                  sigreg_directions=16, sigreg_points=9),
                         accumulation_steps=1, log_every=5, logger=logger)
    fake_report = {"optimizer_step": args.steps, "results": {5: {"synthetic": {"metrics": {
        "trained_encoder": {"development": {"mean_daily_rank_ic": 0.0, "valid_ic_dates": 0}}}}}}}
    logger.log_evaluation(fake_report, checkpoint_step=args.steps)
    info = logger.describe()
    logger.finish()
    summary = {"stage": "done", "optimizer_step": result["optimizer_step"], "device": result["device"],
               "precision": result["precision"], "elapsed_seconds": round(result["elapsed_seconds"], 2),
               "wandb": info, "run_dir": str(out)}
    print("WANDB_SMOKE " + json.dumps(summary), flush=True)
    if args.require_online and (info["mode"] != "online" or not info["url"] or info["failures"]):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
