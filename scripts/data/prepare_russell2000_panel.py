"""Build the historical Russell 2000 (iShares IWM holdings) proxy panel, 2003-01-01 .. 2025-12-31.

Same pipeline as the IWB expanded panel (scripts/data/prepare_expanded_panel.py), parameterized for IWM:
  1. Monthly IWM holdings from the iShares archive (no key), 2003 .. 2025, into work/iwm-history-cache/
     and work/fjepa-russell2000-data/iwm_rosters.json. Rights / warrants / units are excluded by name.
  2. Adjusted + raw FMP price histories for every roster symbol plus SPY and the reference ETFs
     (key from FMP_API_KEY or ~/.config/fmp/api_key; cached, rate-limited, restartable).
  3. Price-anchor validation (raw close within 5 % of the archived holdings price, USD only).
  4. monthly_universe(max_stocks=1000) by trailing 60-session dollar volume, build_panel on the symbols
     ever selected, metadata + coverage.json, like the expanded panel.

Run from a terminal:  work/colab-test-env/bin/python scripts/data/prepare_russell2000_panel.py [--roster-only]
`--roster-only` does step 1 only (no key needed) and prints the candidate count.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import DataAccessError, FMPClient, build_panel, fetch_price_history, load_panel, monthly_universe, save_panel  # noqa: E402
from src.expanded_data import (IWM, HoldingsUnavailable, IWBHistoryClient, fetch_monthly_history,  # noqa: E402
                                       load_roster_bundle, membership_from_snapshots, save_roster_bundle)
from src.cloud_data import _source_fingerprint  # noqa: E402
from src.features import REFERENCE_ETFS, rolling  # noqa: E402

OUTPUT = ROOT / "work/fjepa-russell2000-data"
CACHE = ROOT / "work/fmp-live-cache"
ROSTER_CACHE = ROOT / "work/iwm-history-cache"
START, END = "2003-01-01", "2025-12-31"
MAX_STOCKS = 1000
SYMBOL_MAP = {"BRKB": "BRK-B", "BF/B": "BF-B", "BF.B": "BF-B"}
NON_COMMON = re.compile(r"\b(RTS|RIGHTS?|WARRANTS?|WTS?|CVR|UNITS?|SUBSCRIPTION|ESCROW)\b", re.I)
PARSER_SHA256 = hashlib.sha256((ROOT / "src/expanded_data.py").read_bytes()).hexdigest()
ROSTER_POLICY_VERSION = 2


def log(**info):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **info}), flush=True)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp"); temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


def cached_request(endpoint, params):
    identity = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return (CACHE / (hashlib.sha256(identity.encode()).hexdigest() + ".json")).exists()


ARTIFACT = re.compile(r"(\.\d+$|\.$|\.XX\d*$|^\d)")  # iShares placeholders for delisted/renamed lines: AAN.2, ATLS., AWGI.XX1, 0557910D


def dotted_to_hyphen(symbol):
    """iShares writes class shares as MOG.A; FMP uses MOG-A. Placeholder tickers become None (unsupported)."""
    if not symbol:
        return symbol
    if ARTIFACT.search(symbol):
        return None
    return re.sub(r"^([A-Z]+)\.([A-Z])$", r"\1-\2", symbol)


def rosters(dates):
    roster = OUTPUT / "iwm_rosters.json"
    if roster.exists():
        snapshots, report = load_roster_bundle(roster)
        if report.get("parser_sha256") == PARSER_SHA256 and report.get("roster_policy_version") == ROSTER_POLICY_VERSION:
            return snapshots, report
    log(stage="roster_fetch", fund="IWM", cache=str(ROSTER_CACHE))
    client = IWBHistoryClient(ROSTER_CACHE, symbol_map=SYMBOL_MAP, spec=IWM)
    progress = lambda info: log(stage="roster_month", **info) if info.get("error") or info["date"].endswith("-12-31") else None
    try:
        snapshots, report = fetch_monthly_history(client, dates, start=START, end=END, allow_internal_gaps=True, progress=progress)
    except HoldingsUnavailable as exc:
        if hasattr(exc, "coverage_report"):
            write_json(OUTPUT / "roster_failure.json", exc.coverage_report)
        raise
    excluded, placeholders = [], []
    for snapshot in snapshots:
        for holding in snapshot.holdings:
            if holding["symbol"] is None:
                continue
            if NON_COMMON.search(holding["name"] or ""):
                excluded.append({"date": snapshot.as_of_date, "symbol": holding["symbol"], "name": holding["name"]})
                holding["symbol"] = None
            else:
                cleaned = dotted_to_hyphen(holding["symbol"])
                if cleaned is None:
                    placeholders.append({"date": snapshot.as_of_date, "symbol": holding["symbol"], "name": holding["name"]})
                holding["symbol"] = cleaned
    report["union_symbols"] = sorted(set().union(*(set(s.symbols) for s in snapshots)))  # after cleanup
    report.update(excluded_placeholder_tickers=placeholders,validated_format_aliases=SYMBOL_MAP, parser_sha256=PARSER_SHA256, roster_policy_version=ROSTER_POLICY_VERSION,
                  excluded_non_common_rows=excluded, dotted_class_shares_hyphenated=True)
    save_roster_bundle(snapshots, report, roster)
    return snapshots, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roster-only", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=MAX_STOCKS)
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar_panel = load_panel(ROOT / "work/fjepa-cloud-data/panel_raw.npz")
    dates = calendar_panel.dates[calendar_panel.dates <= np.datetime64(END)]
    snapshots, roster_report = rosters(dates)
    symbols, membership, timing = membership_from_snapshots(dates, snapshots)
    log(stage="rosters_ready", snapshots=len(snapshots), first=roster_report["first_snapshot"], last=roster_report["last_snapshot"],
        candidate_symbols=len(symbols), missing_months=len(roster_report["missing_months"]), excluded_non_common=len(roster_report["excluded_non_common_rows"]))
    report = {"status": "rosters_ready", "fund": "IWM", "universe": roster_report, "timing": timing, "start": START, "end": END,
              "max_stocks": args.max_stocks, "candidate_count": len(symbols),
              "limitations": roster_report["limitations"] + ["Terminal delisting returns unavailable; no investable alpha claim."]}
    write_json(OUTPUT / "coverage.json", report)
    if args.roster_only:
        return 0

    # ---- prices (needs the FMP key) ----------------------------------------------------------
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

    FMPClient(CACHE)  # fail early if the key is missing

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
    prices, failures, done, last = {}, {}, 0, 0.0
    report["status"] = "downloading"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for symbol, frame, errors in pool.map(fetch, names):
            done += 1
            if frame is not None:
                prices[symbol] = frame
            if errors:
                failures[symbol] = errors
            if time.monotonic() - last >= 30 or done == len(names):
                last = time.monotonic()
                report["downloads"] = {"completed": done, "requested": len(names), "adjusted_histories": len(prices), "failures": failures}
                write_json(OUTPUT / "coverage.json", report)
                log(stage="prices", downloaded=done, total=len(names), usable=len(prices), failures=len(failures))
    if any(s not in prices for s in ["SPY", *REFERENCE_ETFS]):
        raise RuntimeError("Missing required reference history")
    if not np.array_equal(prices["SPY"].index.to_numpy(dtype="datetime64[D]"), dates):
        raise RuntimeError("Benchmark calendar mismatch")
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "RawOpen", "RawClose", "RawVolume"],
                         index=pd.DatetimeIndex([], name="date"), dtype=float)
    for s in symbols:
        prices.setdefault(s, empty.copy())

    # ---- price-anchor validation ---------------------------------------------------------------
    lookup = {s: i for i, s in enumerate(symbols)}
    chosen = np.asarray(timing["snapshot_date_by_origin"])
    counts = {"matched": 0, "mismatch": 0, "missing_raw_close": 0, "missing_source_price_or_non_usd": 0}
    anchor_records = []
    for snapshot in snapshots:
        origin_rows = chosen == snapshot.as_of_date
        day = pd.Timestamp(snapshot.as_of_date)
        for holding in snapshot.holdings:
            symbol = holding["symbol"]
            if symbol is None or symbol not in lookup:
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
                anchor_records.append({"as_of_date": snapshot.as_of_date, "symbol": symbol, "source_name": holding["name"],
                                       "source_price": price, "fmp_raw_close": float(raw) if np.isfinite(raw) else None, "reason": reason})
    report["price_anchor_validation"] = {"relative_tolerance": .05, "counts": counts,
        "note": "Membership requires a contemporaneous raw close within 5% of the archived USD holdings price.",
        "exceptions_file": str(OUTPUT / "price_anchor_exclusions.json")}
    write_json(OUTPUT / "price_anchor_exclusions.json", anchor_records)
    log(stage="price_anchor_validation", **counts)

    # ---- selection and build -------------------------------------------------------------------
    calendar = pd.DatetimeIndex(dates)

    def matrix(column):
        return np.column_stack([prices[s].reindex(calendar)[column].to_numpy(float) if column in prices[s] else np.full(len(dates), np.nan) for s in symbols])
    raw_close, raw_volume = matrix("RawClose"), matrix("RawVolume")
    selected = monthly_universe(dates, membership, raw_close, raw_volume, max_stocks=args.max_stocks)
    count = selected.sum(axis=1)
    train = (dates >= np.datetime64(timing["first_covered_date"])) & (dates <= np.datetime64("2018-12-31"))
    report["selection"] = {"train_dates": int(train.sum()), "minimum_train_count": int(count[train].min()), "median_train_count": float(np.median(count[train])),
                           "maximum_count": int(count.max()), "mean_count_by_year": {int(y): float(count[(dates.astype('datetime64[Y]').astype(int) + 1970) == y].mean()) for y in range(2006, 2026)}}
    if np.median(count[train]) < 256:
        raise RuntimeError("Russell 2000 data coverage too small to support the intended comparison")
    keep = selected.any(axis=0); kept = [s for s, k in zip(symbols, keep) if k]
    report["retained_symbols"] = kept; report["retained_symbol_count"] = len(kept)
    report["never_selected_symbols"] = [s for s, k in zip(symbols, keep) if not k]
    report["stock_coverage"] = {s: {"adjusted_rows": int(prices[s].get("Close", pd.Series(dtype=float)).notna().sum()),
                                    "raw_rows": int(prices[s].get("RawClose", pd.Series(dtype=float)).notna().sum()),
                                    "member_dates": int(membership[:, i].sum()), "selected_dates": int(selected[:, i].sum())} for i, s in enumerate(symbols)}
    report["status"] = "computing_features"; write_json(OUTPUT / "coverage.json", report)
    log(stage="computing_features", retained=len(kept), selection=report["selection"])
    source = _source_fingerprint(prices, symbols, membership)
    retained = {s: prices[s] for s in [*kept, "SPY", *REFERENCE_ETFS]}
    panel = build_panel(retained, kept, membership=membership[:, keep], universe_mode="historical_candidate", max_stocks=args.max_stocks)
    if not np.array_equal(panel.universe, selected[:, keep]):
        raise RuntimeError("Selection changed after pruning")
    roster_hash = hashlib.sha256((OUTPUT / "iwm_rosters.json").read_bytes()).hexdigest()
    parent = panel.metadata["dataset_fingerprint"]
    panel.metadata.update(universe_label="Historical IWM monthly holdings proxy (Russell 2000)", exact_russell_index_membership=False,
                          source_fingerprint=source, roster_sha256=roster_hash, first_membership_date=timing["first_covered_date"],
                          dataset_fingerprint=hashlib.sha256((parent + source + roster_hash).encode()).hexdigest(), point_in_time_verified=False)
    panel.metadata["limitations"] += report["limitations"]
    panel_path = OUTPUT / "panel_raw.npz"; save_panel(panel, panel_path)
    report.update(status="ready_research_only", dataset_fingerprint=panel.metadata["dataset_fingerprint"], source_fingerprint=source,
                  roster_sha256=roster_hash, stock_shape=list(panel.stock.shape), market_shape=list(panel.market.shape), panel_path=str(panel_path),
                  panel_sha256=hashlib.file_digest(panel_path.open("rb"), "sha256").hexdigest())
    complete = rolling(panel.stock_obs[:, :, :4].all(axis=-1).astype(float), 60, "sum") == 60
    eligibility = panel.universe & complete
    years = panel.dates.astype("datetime64[Y]").astype(int) + 1970
    report["eligible_by_year"] = {int(y): float(eligibility[years == y].sum(1).mean()) for y in range(2006, 2026)}
    write_json(OUTPUT / "coverage.json", report)
    log(stage="done", status=report["status"], panel_bytes=panel_path.stat().st_size, retained=len(kept), eligible_by_year=report["eligible_by_year"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
