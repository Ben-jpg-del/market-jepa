"""Build a assembled panel (15 stock / 11 market channels) from a base panel and its family files.

  work/colab-test-env/bin/python scripts/data/build_assembled_panel.py --base work/fjepa-cloud-data/panel_raw.npz \
      --families work/fjepa-families/family_channels_orig2025.npz \
      --fundamentals work/fjepa-families/fundamentals_channels_orig2025.npz \
      --out work/fjepa-assembled-data/panel_largecap426.npz [--min-coverage 0.9]

Asserts that origin eligibility is identical to the base panel's, checks per-channel yearly coverage,
and writes <out> plus <out stem>_coverage.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel, save_panel  # noqa: E402
from src.evaluation import context_eligibility  # noqa: E402
from src.panel_assembly import assemble_panel, check_panel_coverage  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--fundamentals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--first-year", type=int, default=2006)
    parser.add_argument("--min-coverage-overrides", default="eps_sue=0.5,revenue_surprise=0.5",
                        help="per-channel floors, name=value pairs; the two surprise channels are sparse by construction")
    args = parser.parse_args(argv)
    base = load_panel(args.base)
    families = np.load(args.families, allow_pickle=False)
    fundamentals = np.load(args.fundamentals, allow_pickle=False)
    panel = assemble_panel(base, families, fundamentals)
    same = bool(np.array_equal(context_eligibility(panel, 60), context_eligibility(base, 60)))
    if not same:
        raise SystemExit("eligibility differs from the base panel")
    overrides = {k: float(v) for k, v in (x.split("=") for x in args.min_coverage_overrides.split(",") if x)}
    coverage = check_panel_coverage(panel, min_coverage=args.min_coverage, first_year=args.first_year, overrides=overrides)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    save_panel(panel, out)
    check = load_panel(out)
    years = check.dates.astype("datetime64[Y]").astype(int) + 1970
    elig = context_eligibility(check, 60)
    summary = {"path": str(out), "sha256": hashlib.file_digest(out.open("rb"), "sha256").hexdigest(),
               "stock_channels": check.stock.shape[-1], "market_channels": check.market.shape[-1], "dates": len(check.dates),
               "symbols": len(check.symbols), "first_date": str(check.dates[0]), "last_date": str(check.dates[-1]),
               "dataset_fingerprint": check.metadata["dataset_fingerprint"], "base_dataset_fingerprint": check.metadata["base_dataset_fingerprint"],
               "eligibility_identical": same, "origin_observed_carried": check.origin_observed is not None,
               "eligible_per_day_by_year": {int(y): float(elig[years == y].sum(1).mean()) for y in range(args.first_year, int(years.max()) + 1) if (years == y).any()},
               "min_yearly_coverage": {n: min(v.values()) for n, v in {**coverage["stock"], **coverage["market"]}.items()},
               "coverage_thresholds": {"default": args.min_coverage, **overrides},
               "coverage": coverage}
    Path(str(out.with_suffix("")) + "_coverage.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("coverage", "min_yearly_coverage")}, indent=1))
    print("min yearly coverage:", {n: round(c, 3) for n, c in summary["min_yearly_coverage"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
