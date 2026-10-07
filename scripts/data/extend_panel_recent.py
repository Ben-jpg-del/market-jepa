"""Extend the expanded (IWB-proxy, 1,439-symbol) panel from 2021-12-31 to 2025-12-31.

Run from a terminal (it needs the FMP key, read from ~/.config/fmp/api_key or FMP_API_KEY):
  work/colab-test-env/bin/python scripts/data/extend_panel_recent.py

Steps (all cached / restartable under work/fmp-live-cache and work/iwb-history-cache):
  1. Monthly IWB holdings 2022-01 .. 2025-12 from iShares (no key), appended to the existing roster
     bundle -> work/fjepa-expanded-data/iwb_rosters_2025.json; membership_from_snapshots as before.
  2. Adjusted + raw price histories 2003-01-01 .. 2025-12-31 for the 1,439 retained symbols, SPY and
     the reference ETFs (about half are already cached to 2025; the rest are fetched, rate-limited).
  3. Price-anchor validation of the new snapshots (raw close within 5 % of the archived holdings
     price), monthly_universe(max_stocks=1000) and build_panel on the SAME 1,439 symbols, so cohort
     definitions stay comparable -> work/fjepa-expanded-data/panel_raw_2025.npz + coverage_2025.json.
The 2003-2021 part of the new panel is recomputed from the same inputs and should match the old
panel's features; a check against panel_raw.npz is written into coverage_2025.json.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import DataAccessError, FMPClient, build_panel, fetch_price_history, load_panel, monthly_universe, save_panel  # noqa: E402
from src.expanded_data import (IWBHistoryClient, fetch_monthly_history, load_roster_bundle,  # noqa: E402
                                       membership_from_snapshots, save_roster_bundle)
from src.features import REFERENCE_ETFS  # noqa: E402

OUT = ROOT / "work/fjepa-expanded-data"
CACHE = ROOT / "work/fmp-live-cache"
START, END = "2003-01-01", "2025-12-31"
SYMBOL_MAP = {"BRKB": "BRK-B", "BF/B": "BF-B", "BF.B": "BF-B"}


def log(**info):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **info}), flush=True)


def cached_request(endpoint, params):
    identity = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return (CACHE / (hashlib.sha256(identity.encode()).hexdigest() + ".json")).exists()


def main() -> int:
    old = load_panel(OUT / "panel_raw.npz")
    coverage = json.loads((OUT / "coverage.json").read_text())
    symbols = list(coverage["retained_symbols"])
    assert symbols == list(old.symbols), "retained symbols must match the existing panel"

    # ---- calendar from SPY (cached to 2025-12-31) ---------------------------------------------
    client = FMPClient(CACHE)
    spy = fetch_price_history(client, "SPY", START, END, adjusted=True)
    dates = spy.index.to_numpy(dtype="datetime64[D]")
    assert np.array_equal(dates[: len(old.dates)], old.dates.astype("datetime64[D]")), "calendar prefix mismatch"
    log(stage="calendar", first=str(dates[0]), last=str(dates[-1]), sessions=len(dates), new_sessions=int(len(dates) - len(old.dates)))

    # ---- rosters ----------------------------------------------------------------------------------
    snapshots, roster_report = load_roster_bundle(OUT / "iwb_rosters.json")
    have = {s.as_of_date for s in snapshots}
    iwb = IWBHistoryClient(ROOT / "work/iwb-history-cache", symbol_map=SYMBOL_MAP)
    new_snaps, new_report = fetch_monthly_history(iwb, dates, start="2022-01-01", end=END, allow_internal_gaps=True)
    added = [s for s in new_snaps if s.as_of_date not in have]
    rights = {"LIBERTY VENTURES RTS", "LIBERTY MEDIA RIGHTS EXP JUNE COR", "T MOBILE US SUBSCRIPTION RIGHTS IN"}
    for snapshot in added:  # same identified-rights exclusion as the 2021 build
        for holding in snapshot.holdings:
            if holding["name"] in rights:
                holding["symbol"] = None
    snapshots = sorted([*snapshots, *added], key=lambda s: s.as_of_date)
    roster_report = {**roster_report, "extension_2022_2025": {k: v for k, v in new_report.items() if k != "limitations"},
                     "extended_snapshots": len(added), "last_snapshot": snapshots[-1].as_of_date}
    save_roster_bundle(snapshots, roster_report, OUT / "iwb_rosters_2025.json")
    all_symbols, membership_all, timing = membership_from_snapshots(dates, snapshots)
    position = {sym: i for i, sym in enumerate(all_symbols)}
    membership = np.zeros((len(dates), len(symbols)), dtype=bool)
    for j, sym in enumerate(symbols):
        if sym in position:
            membership[:, j] = membership_all[:, position[sym]]
    joined_after_2021 = sorted(set(all_symbols) - set(symbols))
    log(stage="rosters", snapshots=len(snapshots), added=len(added), last=snapshots[-1].as_of_date,
        roster_symbols=len(all_symbols), retained=len(symbols), new_members_not_in_retained=len(joined_after_2021))

    # ---- prices -------------------------------------------------------------------------------------
    gate = threading.Lock(); next_request = [0.0]; local = threading.local()

    class LimitedClient(FMPClient):
        def get(self, endpoint, params=None, refresh=False):
            if refresh or not cached_request(endpoint, dict(params or {})):
                with gate:
                    wait = next_request[0] - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    next_request[0] = time.monotonic() + .25
            return super().get(endpoint, params, refresh)

    def fetch(symbol):
        if not hasattr(local, "client"):
            local.client = LimitedClient(CACHE)
        frames, errors = [], {}
        for adjusted in (True, False):
            try:
                frames.append(fetch_price_history(local.client, symbol, START, END, adjusted))
            except DataAccessError as exc:
                errors["adjusted" if adjusted else "raw"] = exc.as_dict()
                if adjusted:
                    break
        if not frames:
            return symbol, None, errors
        return symbol, frames[0].join(frames[1], how="left") if len(frames) == 2 else frames[0], errors

    names = list(dict.fromkeys([*symbols, "SPY", *REFERENCE_ETFS]))
    prices, failures, done = {}, {}, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for symbol, frame, errors in pool.map(fetch, names):
            done += 1
            if frame is not None:
                prices[symbol] = frame
            if errors:
                failures[symbol] = errors
            if done % 100 == 0 or done == len(names):
                log(stage="prices", downloaded=done, total=len(names), usable=len(prices), failures=len(failures))
    if any(s not in prices for s in ["SPY", *REFERENCE_ETFS]):
        raise RuntimeError("Missing required reference history")
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "RawOpen", "RawClose", "RawVolume"],
                         index=pd.DatetimeIndex([], name="date"), dtype=float)
    for s in symbols:
        prices.setdefault(s, empty.copy())

    # ---- price-anchor validation for the new snapshots ---------------------------------------------
    lookup = {s: i for i, s in enumerate(symbols)}
    chosen = np.asarray(timing["snapshot_date_by_origin"])
    counts = {"matched": 0, "mismatch": 0, "missing_raw_close": 0, "missing_source_price_or_non_usd": 0, "unknown_symbol": 0}
    for snapshot in added:
        origin_rows = chosen == snapshot.as_of_date
        day = pd.Timestamp(snapshot.as_of_date)
        for holding in snapshot.holdings:
            symbol = holding["symbol"]
            if symbol is None:
                continue
            if symbol not in lookup:
                counts["unknown_symbol"] += 1
                continue
            price = holding.get("source_price"); currency = holding.get("market_currency")
            if currency in ("-", None, "") and holding.get("currency") == "USD" and holding.get("source_fx_rate") == "1.00":
                currency = "USD"
            raw = prices[symbol].get("RawClose", pd.Series(dtype=float)).get(day, np.nan)
            if price is None or not np.isfinite(price) or price <= 0 or currency != "USD":
                reason = "missing_source_price_or_non_usd"
            elif not np.isfinite(raw):
                reason = "missing_raw_close"
            elif abs(raw / price - 1) > .05:
                reason = "mismatch"
            else:
                reason = "matched"
            counts[reason] += 1
            if reason != "matched":
                membership[origin_rows, lookup[symbol]] = False
    log(stage="anchor_validation_new_snapshots", **counts)
    # symbols that joined IWB after 2021 are not in the 1,439 list: the universe stays the retained
    # symbols (survivorship-safe forward: names that left keep dropping out when prices end).

    calendar = pd.DatetimeIndex(dates)

    def matrix(column):
        return np.column_stack([prices[s].reindex(calendar)[column].to_numpy(float) if column in prices[s] else np.full(len(dates), np.nan) for s in symbols])
    raw_close, raw_volume = matrix("RawClose"), matrix("RawVolume")
    selected = monthly_universe(dates, membership, raw_close, raw_volume, max_stocks=1000)
    retained = {s: prices[s] for s in [*symbols, "SPY", *REFERENCE_ETFS]}
    panel = build_panel(retained, symbols, membership=membership, universe_mode="historical_candidate", max_stocks=1000)
    # comparability check against the existing panel on the overlapping window
    n_old = len(old.dates)
    same_close = np.allclose(np.nan_to_num(panel.close[:n_old]), np.nan_to_num(old.close), rtol=1e-6, atol=1e-6)
    same_stock = np.allclose(np.nan_to_num(panel.stock[:n_old]), np.nan_to_num(old.stock), rtol=1e-5, atol=1e-5)
    universe_overlap = float((panel.universe[:n_old] == old.universe).mean())
    parent = panel.metadata["dataset_fingerprint"]
    panel.metadata.update(universe_label="Historical IWB monthly holdings proxy, extended 2022-2025 on the 2021 retained symbols",
                          exact_russell_index_membership=False, extension={"from": "2022-01-01", "to": END, "added_snapshots": len(added),
                          "anchor_counts": counts, "overlap_close_equal": same_close, "overlap_stock_equal": same_stock,
                          "overlap_universe_agreement": universe_overlap},
                          dataset_fingerprint=hashlib.sha256((parent + "extended-2025").encode()).hexdigest(), point_in_time_verified=False)
    out_path = OUT / "panel_raw_2025.npz"
    save_panel(panel, out_path)
    report = {"start": START, "end": END, "symbols": len(symbols), "sessions": len(dates), "snapshots": len(snapshots),
              "added_snapshots": len(added), "new_members_not_in_retained": joined_after_2021, "anchor_counts": counts, "download_failures": failures,
              "overlap_close_equal": same_close, "overlap_stock_equal": same_stock, "overlap_universe_agreement": universe_overlap,
              "universe_count_by_year": {int(y): float(panel.universe[(dates.astype('datetime64[Y]').astype(int) + 1970) == y].sum(1).mean()) for y in range(2019, 2026)},
              "panel_path": str(out_path), "panel_sha256": hashlib.file_digest(out_path.open("rb"), "sha256").hexdigest()}
    (OUT / "coverage_2025.json").write_text(json.dumps(report, indent=2))
    log(stage="done", **{k: v for k, v in report.items() if k not in ("download_failures",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
