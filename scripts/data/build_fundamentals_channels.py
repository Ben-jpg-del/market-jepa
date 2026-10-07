"""Build the point-in-time fundamentals family aligned to the expanded panel.

Inputs: work/fjepa-expanded-data/panel_raw.npz (dates, symbols, adjusted close)
        work/fjepa-families/statements_raw.json (quarterly income / balance / cash-flow per symbol)
Output: work/fjepa-families/fundamentals_channels.npz  fundamentals [T, K, 7] + obs, names, summary JSON

Timing: a quarter's statements become usable at the close of the first session strictly after the
SEC acceptance date (acceptedDate; filingDate as fallback; period end + 90 days if both missing).
Values are the feed's current version of each period (restatements possible); we use items that are
rarely restated. FMP restates share counts for splits (checked on AAPL 2014/2020), so shares x
adjusted close is a consistent market capitalisation.

Channels (all cross-sectional, price-linked ones update daily with the adjusted close):
  log_book_to_market   log(total stockholders' equity / market cap); equity <= 0 -> missing
  gross_profitability  trailing-4Q gross profit / total assets
  asset_growth         log(total assets / total assets four quarters earlier)
  accruals             (trailing-4Q net income - trailing-4Q operating cash flow) / total assets
  log_market_cap       log(shares outstanding x adjusted close)
  earnings_yield       trailing-4Q net income / market cap
  dividend_yield       -trailing-4Q common dividends paid / market cap
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel  # noqa: E402
from src.features import FUNDAMENTAL_FEATURES  # noqa: E402

FAM = ROOT / "work" / "fjepa-families"


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def quarters_for(entry):
    """Merge the three statements by period end; return sorted list of dicts with availability date."""
    merged = {}
    for kind in ("income-statement", "balance-sheet-statement", "cash-flow-statement"):
        for r in entry.get(kind) or []:
            d = str(r.get("date") or "")[:10]
            if len(d) != 10:
                continue
            q = merged.setdefault(d, {"date": np.datetime64(d)})
            acc = str(r.get("acceptedDate") or r.get("filingDate") or "")[:10]
            if len(acc) == 10:
                q["available"] = max(q.get("available", np.datetime64("1900-01-01")), np.datetime64(acc))
            for key in ("revenue", "grossProfit", "netIncome", "weightedAverageShsOut", "totalAssets",
                        "totalStockholdersEquity", "operatingCashFlow", "commonDividendsPaid", "netDividendsPaid"):
                if key in r and key not in q:
                    q[key] = _num(r[key])
    out = []
    for d in sorted(merged):
        q = merged[d]
        if "available" not in q:
            q["available"] = q["date"] + np.timedelta64(90, "D")
        q["available"] = max(q["available"], q["date"])  # never before the period end
        out.append(q)
    return out


def ttm(values, i, dates):
    """Sum of the four quarters ending at i if they span <= 400 days and are all present."""
    if i < 3:
        return np.nan
    span = (dates[i] - dates[i - 3]).astype("timedelta64[D]").astype(int)
    if span > 400:
        return np.nan
    window = values[i - 3:i + 1]
    return float(np.sum(window)) if np.all(np.isfinite(window)) else np.nan


def build(panel, statements):
    dates = panel.dates.astype("datetime64[D]")
    T, K = len(dates), len(panel.symbols)
    close = np.where(panel.close > 0, panel.close, np.nan).astype(float)
    out = np.full((T, K, len(FUNDAMENTAL_FEATURES)), np.nan, np.float32)
    covered = 0
    for k, sym in enumerate(panel.symbols):
        entry = statements.get(sym)
        if not entry:
            continue
        qs = quarters_for(entry)
        if len(qs) < 4:
            continue
        covered += 1
        qd = np.array([q["date"] for q in qs])
        gp = np.array([q.get("grossProfit", np.nan) for q in qs]); ni = np.array([q.get("netIncome", np.nan) for q in qs])
        ocf = np.array([q.get("operatingCashFlow", np.nan) for q in qs])
        div = np.array([q.get("commonDividendsPaid", q.get("netDividendsPaid", np.nan)) for q in qs])
        assets = np.array([q.get("totalAssets", np.nan) for q in qs]); eq = np.array([q.get("totalStockholdersEquity", np.nan) for q in qs])
        shares = np.array([q.get("weightedAverageShsOut", np.nan) for q in qs])
        # per-quarter static values (known once the quarter is available)
        static = np.full((len(qs), 4), np.nan)  # gross_profitability, asset_growth, accruals, plus dividend ttm
        ni_ttm = np.full(len(qs), np.nan); div_ttm = np.full(len(qs), np.nan)
        for i in range(len(qs)):
            a = assets[i]
            if np.isfinite(a) and a > 0:
                g = ttm(gp, i, qd); n = ttm(ni, i, qd); o = ttm(ocf, i, qd)
                static[i, 0] = g / a if np.isfinite(g) else np.nan
                static[i, 2] = (n - o) / a if np.isfinite(n) and np.isfinite(o) else np.nan
                if i >= 4 and np.isfinite(assets[i - 4]) and assets[i - 4] > 0 and (qd[i] - qd[i - 4]).astype("timedelta64[D]").astype(int) <= 420:
                    static[i, 1] = np.log(a / assets[i - 4])
            ni_ttm[i] = ttm(ni, i, qd); div_ttm[i] = ttm(div, i, qd)
        # availability index: first session strictly after the acceptance date
        avail = np.array([q["available"] for q in qs])
        avail_idx = np.searchsorted(dates, avail, side="right")
        # latest available quarter per session (quarters may arrive out of order after restated availability)
        j = -1
        latest = np.full(T, -1)
        order = np.argsort(avail_idx, kind="stable")
        ptr = 0
        for t in range(T):
            while ptr < len(order) and avail_idx[order[ptr]] <= t:
                cand = order[ptr]
                if j < 0 or qd[cand] >= qd[j]:
                    j = cand
                ptr += 1
            latest[t] = j
        for t in range(T):
            i = latest[t]
            if i < 0 or not np.isfinite(close[t, k]):
                continue
            mcap = shares[i] * close[t, k] if np.isfinite(shares[i]) and shares[i] > 0 else np.nan
            btm = np.log(eq[i] / mcap) if np.isfinite(mcap) and np.isfinite(eq[i]) and eq[i] > 0 else np.nan
            out[t, k] = (btm, static[i, 0], static[i, 1], static[i, 2],
                         np.log(mcap) if np.isfinite(mcap) else np.nan,
                         ni_ttm[i] / mcap if np.isfinite(mcap) and np.isfinite(ni_ttm[i]) else np.nan,
                         -div_ttm[i] / mcap if np.isfinite(mcap) and np.isfinite(div_ttm[i]) else np.nan)
    obs = np.isfinite(out)
    out = np.where(obs, out, 0.0).astype(np.float32)
    return out, obs, covered


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(ROOT / "work/fjepa-expanded-data/panel_raw.npz"))
    parser.add_argument("--suffix", default="", help="output name suffix, e.g. _2025")
    parser.add_argument("--families-dir", default=str(FAM), help="folder holding the raw FMP pulls and receiving the outputs")
    args = parser.parse_args(argv)
    fam = Path(args.families_dir)
    fam.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.panel)
    statements = json.loads((fam / "statements_raw.json").read_text())
    fund, obs, covered = build(panel, statements)
    years = panel.dates.astype("datetime64[Y]").astype(int) + 1970
    universe = panel.universe
    summary = {"fundamental_features": FUNDAMENTAL_FEATURES, "symbols_covered": covered, "symbols": len(panel.symbols),
               "coverage_by_year_over_universe": {int(y): float((obs[years == y][..., 4] & universe[years == y]).sum() / max(1, universe[years == y].sum())) for y in sorted(set(years))},
               "channel_coverage_over_universe": {n: float(obs[..., i][universe].mean()) for i, n in enumerate(FUNDAMENTAL_FEATURES)}}
    sel = (years >= 2006) & (years <= 2021)
    summary["value_ranges_2006_2021"] = {}
    for i, n in enumerate(FUNDAMENTAL_FEATURES):
        v = fund[sel][..., i][obs[sel][..., i] & universe[sel]]
        summary["value_ranges_2006_2021"][n] = {"median": float(np.median(v)), "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99))} if len(v) else None
    np.savez_compressed(fam / f"fundamentals_channels{args.suffix}.npz", dates=panel.dates, symbols=np.array(panel.symbols),
                        fundamentals=fund, fundamentals_obs=obs, fundamental_names=np.array(FUNDAMENTAL_FEATURES))
    (fam / f"fundamentals_channels_summary{args.suffix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
