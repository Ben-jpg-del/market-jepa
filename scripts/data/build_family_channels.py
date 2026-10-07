"""Build the macro and earnings-event channel arrays aligned to the expanded panel.

Inputs: work/fjepa-expanded-data/panel_raw.npz (dates, symbols, close, benchmark_close),
        work/fjepa-families/macro_raw.json, work/fjepa-families/earnings_raw.json
Output: work/fjepa-families/family_channels.npz
        macro   [T, 12]  + macro_obs      market-level, known at the close of date t
        events  [T, K, 7] + events_obs    stock-level, known from the session after an announcement
        names, coverage summary (also written as family_channels_summary.json)

Timing rules: all macro series are market prices/yields as of the close (no release lag).
Earnings rows carry an announcement date without a time of day, so anything derived from an
announcement on date d becomes available at the close of the next session (d+1). The "next
report" estimate is last announcement + 91 calendar days, never the actual future date.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel  # noqa: E402
from src.features import EVENT_FEATURES, MACRO_FEATURES  # noqa: E402

FAM = ROOT / "work" / "fjepa-families"


def series_on_dates(dates, rows, field, key="date", max_ffill=5):
    """Map a dated list of rows onto panel dates (forward-fill up to max_ffill sessions)."""
    values = {}
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        try:
            values[np.datetime64(str(r[key])[:10])] = float(v)
        except (TypeError, ValueError):
            continue
    if not values:
        return np.full(len(dates), np.nan)
    src_dates = np.array(sorted(values))
    src_vals = np.array([values[d] for d in src_dates])
    idx = np.searchsorted(src_dates, dates, side="right") - 1
    out = np.full(len(dates), np.nan)
    ok = idx >= 0
    gap = np.full(len(dates), 10**6)
    gap[ok] = (dates[ok] - src_dates[idx[ok]]).astype("timedelta64[D]").astype(int)
    take = ok & (gap <= max_ffill + 2)  # calendar days; weekends included
    out[take] = src_vals[idx[take]]
    return out


def lag_diff(x, n):
    out = np.full_like(x, np.nan)
    out[n:] = x[n:] - x[:-n]
    return out


def build_macro(panel, raw):
    dates = panel.dates.astype("datetime64[D]")
    # The treasury-rates endpoint returns at most ~63 rows per call; the fetch script pulls monthly
    # windows. The 10y change uses ^TNX (CBOE 10y yield index, yield x10); the curve and bill
    # features use the treasury table (2y, 10y, 3m).
    price = {name: np.log(series_on_dates(dates, blob["rows"], "adjClose")) for name, blob in raw["prices"].items()}
    y10 = series_on_dates(dates, raw["prices"]["tnx"]["rows"], "adjClose") / 10.0
    spy = np.log(panel.benchmark_close.astype(float))
    tr = raw["treasury"]  # monthly-window refetch: complete daily series
    y2 = series_on_dates(dates, tr, "year2"); y10_tr = series_on_dates(dates, tr, "year10"); m3 = series_on_dates(dates, tr, "month3")
    curve = y10_tr - y2
    cols = [lag_diff(y10, 20), lag_diff(y10, 60), lag_diff(price["duration"], 20), lag_diff(price["usdjpy"], 20),
            lag_diff(price["ig_credit"], 20) - lag_diff(price["duration"], 20),
            price["vix"], lag_diff(price["vix"], 20), lag_diff(price["oil"], 20), lag_diff(price["dollar"], 20),
            lag_diff(price["gold"], 20), lag_diff(price["small_caps"], 20) - lag_diff(spy, 20), lag_diff(price["spx"], 60),
            curve, lag_diff(curve, 20), lag_diff(m3, 60)]
    macro = np.stack(cols, axis=1).astype(np.float32)
    obs = np.isfinite(macro)
    macro = np.where(obs, macro, 0.0).astype(np.float32)
    return macro, obs


def build_events(panel, earnings):
    dates = panel.dates.astype("datetime64[D]")
    T, K = len(dates), len(panel.symbols)
    close = np.log(np.where(panel.close > 0, panel.close, np.nan))
    spy = np.log(panel.benchmark_close.astype(float))
    rel = close - spy[:, None]  # log price relative to SPY
    ev = np.full((T, K, len(EVENT_FEATURES)), np.nan, np.float32)
    covered = 0
    for k, sym in enumerate(panel.symbols):
        rows = earnings.get(sym)
        if not rows:
            continue
        ann = []
        for r in rows:
            if r.get("date") and r.get("epsActual") is not None:
                d = np.datetime64(str(r["date"])[:10])
                if d <= dates[-1]:
                    ann.append((d, r.get("epsActual"), r.get("epsEstimated"), r.get("revenueActual"), r.get("revenueEstimated")))
        if not ann:
            continue
        ann.sort(key=lambda a: a[0])
        # keep one row per announcement date (the feed occasionally duplicates a period)
        ann = [a for i, a in enumerate(ann) if i == 0 or a[0] != ann[i - 1][0]]
        covered += 1
        ann_dates = np.array([a[0] for a in ann])
        # index of the announcement session (first panel date >= d) and the "known" session (next one)
        ann_idx = np.searchsorted(dates, ann_dates, side="left")
        surprises = []
        sue_hist = []
        for j, (d, eps_a, eps_e, rev_a, rev_e) in enumerate(ann):
            s = (eps_a - eps_e) if (eps_e is not None) else np.nan
            if np.isfinite(s) and len(sue_hist) >= 4:
                sd = np.std(sue_hist[-8:]) + 1e-6
                sue = float(np.clip(s / sd, -5, 5))
            else:
                sue = np.nan
            if np.isfinite(s):
                sue_hist.append(s)
            rsur = float(np.clip((rev_a - rev_e) / abs(rev_e), -0.2, 0.2)) if (rev_a is not None and rev_e is not None and rev_e != 0) else np.nan
            surprises.append((sue, rsur))
        # walk forward over panel dates
        j = -1  # index of last announcement whose "known" session <= t
        for t in range(T):
            while j + 1 < len(ann) and ann_idx[j + 1] + 1 <= t:
                j += 1
            if j < 0:
                continue
            d_idx = ann_idx[j]
            since = int((dates[t] - ann_dates[j]).astype("timedelta64[D]").astype(int))
            expected_next = ann_dates[j] + np.timedelta64(91, "D")
            to_next = int((expected_next - dates[t]).astype("timedelta64[D]").astype(int))
            sue, rsur = surprises[j]
            # announcement abnormal return: rel[d+1] - rel[d-1] (2-session window around d)
            a0, a1 = d_idx - 1, min(d_idx + 1, T - 1)
            abn = rel[a1, k] - rel[a0, k] if a0 >= 0 else np.nan
            drift = rel[t, k] - rel[a1, k] if t > a1 else 0.0
            ev[t, k] = (np.log1p(min(max(since, 0), 365)), np.log1p(max(to_next, 0)), sue, rsur, abn, drift, 1.0 if 0 <= to_next <= 14 else 0.0)
    obs = np.isfinite(ev)
    ev = np.where(obs, ev, 0.0).astype(np.float32)
    return ev, obs, covered


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
    macro_raw = json.loads((fam / "macro_raw.json").read_text())
    earnings = json.loads((fam / "earnings_raw.json").read_text())
    macro, macro_obs = build_macro(panel, macro_raw)
    events, events_obs, covered = build_events(panel, earnings)
    years = panel.dates.astype("datetime64[Y]").astype(int) + 1970
    summary = {"macro_features": MACRO_FEATURES, "event_features": EVENT_FEATURES,
               "macro_coverage_by_year": {int(y): float(macro_obs[years == y].mean()) for y in sorted(set(years))},
               "events_symbols_covered": covered, "symbols": len(panel.symbols),
               "events_coverage_by_year_over_universe": {int(y): float((events_obs[years == y][..., 0] & panel.universe[years == y]).sum() / max(1, panel.universe[years == y].sum())) for y in sorted(set(years))},
               "event_channel_coverage": {n: float(events_obs[..., i][panel.universe].mean()) for i, n in enumerate(EVENT_FEATURES)}}
    np.savez_compressed(fam / f"family_channels{args.suffix}.npz", dates=panel.dates, symbols=np.array(panel.symbols), macro=macro, macro_obs=macro_obs,
                        events=events, events_obs=events_obs, macro_names=np.array(MACRO_FEATURES), event_names=np.array(EVENT_FEATURES))
    (fam / f"family_channels_summary{args.suffix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
