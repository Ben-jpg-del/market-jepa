"""Held-out pretext (JEPA) loss for full training states: does the pretext task itself
generalize, separately from whether it transfers to return ranking?

For each full checkpoint given (must contain the predictor and teacher), sample historical-masking
batches with the variant's own sampler from (a) the training period 2010-2018 and (b) the
held-out period 2019-2021, and report mean prediction loss and SIGReg over N batches.
Default: the replayed F recipe states from the prevention test (step 500 and 2,500), which
reproduce the F0/expanded_F control.

Usage: work/colab-test-env/bin/python scripts/eval/heldout_pretext_loss.py --batches 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import DateBatchSampler, load_panel, normalize_panel  # noqa: E402
from src.model import FutureJEPA, PilotConfig  # noqa: E402
from src.runner import VariantSampler  # noqa: E402
from src.training import batch_to_device  # noqa: E402

OUT = ROOT / "outputs" / "colab-tokenization" / "heldout-pretext-loss"
DEFAULT = {
    "F_step500": ROOT / "outputs/colab-expanded/prevention-test/replay_F_seed42/step_000500.pt",
    "F_step2500_control": ROOT / "outputs/colab-expanded/prevention-test/control_F_seed42/step_002500.pt",
}


def sampler_for(panel, variant, start, end, seed):
    s = VariantSampler(panel, variant, batch_dates=16, stocks=32, seed=seed)
    # Replace the origin range while keeping the variant's masking logic and RNG.
    s.base = DateBatchSampler(panel, batch_dates=16, stocks_per_date=32, context=60, horizon=s.spec["horizon"],
                              seed=seed, start=start, end=end, feature_set=s.spec["feature_set"])
    return s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=None, help="label=path; repeatable")
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = dict(DEFAULT) if not args.checkpoint else {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in args.checkpoint}
    torch.set_num_threads(6)
    panel = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
    results = {}
    for label, path in checkpoints.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["metadata"]
        model = FutureJEPA(PilotConfig(**ck["model_config"]))
        model.load_state_dict(ck["model"])
        model.eval()
        normalized = normalize_panel(panel, meta["normalizer"])
        variant = meta["variant"]
        results[label] = {"step": int(ck["optimizer_step"]), "variant": variant, "periods": {}}
        for period, (start, end) in {"train_2010_2018": ("2010-01-01", "2018-12-31"),
                                     "heldout_2019_2021": ("2019-01-01", "2021-12-31")}.items():
            sampler = sampler_for(normalized, variant, start, end, args.seed)
            pred, reg, tot = [], [], []
            with torch.no_grad():
                for _ in range(args.batches):
                    batch = batch_to_device(sampler(), torch.device("cpu"))
                    out = model.loss(batch)
                    pred.append(float(out["prediction_loss"])); reg.append(float(out["sigreg_loss"])); tot.append(float(out["loss"]))
            results[label]["periods"][period] = {"prediction_loss": float(np.mean(pred)), "prediction_loss_se": float(np.std(pred) / np.sqrt(len(pred))),
                                                  "sigreg_loss": float(np.mean(reg)), "total_loss": float(np.mean(tot)), "batches": args.batches}
            print(json.dumps({"checkpoint": label, "step": results[label]["step"], "period": period, **results[label]["periods"][period]}), flush=True)
    lines = ["| checkpoint | step | train pred. loss | held-out pred. loss | train SIGReg | held-out SIGReg |", "|---|---:|---:|---:|---:|---:|"]
    for label, r in results.items():
        a, b = r["periods"]["train_2010_2018"], r["periods"]["heldout_2019_2021"]
        lines.append(f"| {label} | {r['step']} | {a['prediction_loss']:.4f} ± {a['prediction_loss_se']:.4f} | {b['prediction_loss']:.4f} ± {b['prediction_loss_se']:.4f} | {a['sigreg_loss']:.3f} | {b['sigreg_loss']:.3f} |")
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "report.md").write_text("# Held-out pretext loss\n\nBatches: 16 dates x 32 stocks, historical masking as in training; teacher targets from the checkpoint's own teacher.\n\n" + text + "\n")
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
