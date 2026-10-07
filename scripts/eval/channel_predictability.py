"""Channel predictability matrix: how well is each stock channel's latest value predicted by
the other channels' visible 60-day history (the arithmetic-shortcut map for masking design).

Rows: the fixed diagnostic panel (every 5th session 2010-2021, up to 32 common-cohort stocks per
date). Predictors: the full 60-day history of a source channel (60 numbers) plus a bias, ridge
regression, fit on 2010-2017, R² reported on 2019-2021 (out of sample in time).
Outputs a 14x14 pairwise matrix (target x source), an "all other stock channels" column, an
"all others + market" column, and proposed shortcut groups (pairwise R² >= threshold).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from src.data import load_panel  # noqa: E402
from src.expanded_eval import _normalized_view  # noqa: E402
from src.features import MARKET_FEATURES, STOCK_FEATURES  # noqa: E402
from diagnose_channel_recoverability import fixed_sample  # noqa: E402

OUT = ROOT / "outputs" / "colab-tokenization" / "channel-predictability"


def r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
    original = load_panel(ROOT / "work/fjepa-expanded-data/original_panel.npz")
    rows, _ = fixed_sample(panel, original)
    del original
    meta = json.loads((ROOT / "outputs/colab-tokenization/live-run/artifacts/F0_s42_seed42/run_metadata.json").read_text())
    view = _normalized_view(panel, meta["normalizer"], "book")
    n, C, M = len(rows), len(STOCK_FEATURES), len(MARKET_FEATURES)
    hist = np.zeros((n, 60, C), np.float32)
    hobs = np.zeros((n, 60, C), bool)
    mhist = np.zeros((n, 60, M), np.float32)
    for j, (t, k) in enumerate(rows):
        w = slice(t - 59, t + 1)
        hist[j], hobs[j], mhist[j] = view.stock[w, k], view.stock_obs[w, k], view.market[w]
    hist = np.where(hobs, hist, 0.0)
    dates = panel.dates[rows[:, 0]]
    yr = dates.astype("datetime64[Y]").astype(int) + 1970
    fit, test = (yr >= 2010) & (yr <= 2017), (yr >= 2019) & (yr <= 2021)
    target = hist[:, -1, :].copy()
    target_ok = hobs[:, -1, :]
    pair = np.full((C, C), np.nan)
    all_stock = np.full(C, np.nan)
    all_plus_market = np.full(C, np.nan)
    def score(X, c):
        ok = target_ok[:, c]
        f, s = fit & ok, test & ok
        mu, sd = X[f].mean(0), X[f].std(0) + 1e-8
        Xs = (X - mu) / sd
        model = Ridge(alpha=args.alpha).fit(Xs[f], target[f, c])
        return float(r2(target[s, c], model.predict(Xs[s])))
    for c in range(C):
        for src in range(C):
            if src == c:
                continue
            pair[c, src] = score(hist[:, :, src], c)
        others = [s for s in range(C) if s != c]
        all_stock[c] = score(hist[:, :, others].reshape(n, -1), c)
        all_plus_market[c] = score(np.concatenate([hist[:, :, others].reshape(n, -1), mhist.reshape(n, -1)], 1), c)
    # also: target predicted from its OWN history excluding the last day (persistence)
    persist = np.array([score(hist[:, :-1, c], c) for c in range(C)])
    names = list(STOCK_FEATURES)
    lines = [f"Out-of-sample R² (2019-2021) of each channel's latest value; fit 2010-2017 on {int(fit.sum())} rows, test {int(test.sum())} rows", ""]
    head = f"{'target':30s}" + "".join(f"{nm[:7]:>8s}" for nm in names) + f"{'ALLstk':>8s}{'ALL+mkt':>8s}{'own-1':>8s}"
    lines.append(head)
    for c, nm in enumerate(names):
        row = f"{nm:30s}" + "".join(f"{pair[c, s]:8.2f}" if s != c else f"{'  .':>8s}" for s in range(C))
        row += f"{all_stock[c]:8.2f}{all_plus_market[c]:8.2f}{persist[c]:8.2f}"
        lines.append(row)
    groups = {}
    for c, nm in enumerate(names):
        partners = [names[s] for s in range(C) if s != c and (pair[c, s] >= args.threshold or pair[s, c] >= args.threshold)]
        groups[nm] = partners
    lines += ["", f"Shortcut partners (pairwise R² >= {args.threshold} in either direction):"]
    for nm, ps in groups.items():
        lines.append(f"  {nm:30s} <- {', '.join(ps) if ps else '(none)'}")
    text = "\n".join(lines)
    print(text)
    (OUT / "report.md").write_text("# Channel predictability\n\n```\n" + text + "\n```\n")
    json.dump({"names": names, "pairwise_r2": pair.tolist(), "all_stock_r2": all_stock.tolist(),
               "all_plus_market_r2": all_plus_market.tolist(), "own_history_r2": persist.tolist(),
               "threshold": args.threshold, "shortcut_partners": groups,
               "protocol": "ridge alpha=1 on standardized 60-day source history; fixed diagnostic panel; fit 2010-2017; R2 on 2019-2021"},
              open(OUT / "results.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
