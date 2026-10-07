"""Fetch the raw inputs for new information families from FMP (cached, resumable).

  --what macro     treasury curve 2003-2025 (monthly windows) and daily prices for macro proxies
  --what earnings  `earnings` for every expanded-universe symbol
  --what statements quarterly income/balance/cash-flow statements for every symbol

Everything goes through FMPClient (header auth, per-request JSON cache under work/fmp-live-cache),
so re-runs are free and interrupted runs resume. Consolidated outputs land in
work/fjepa-families/{macro_raw.json, earnings_raw.json, statements_raw.json} plus a coverage log.
The key is read from FMP_API_KEY and never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import DataAccessError, FMPClient  # noqa: E402

OUT = ROOT / "work" / "fjepa-families"
MACRO_PROXIES = {"vix": "^VIX", "ig_credit": "LQD", "duration": "TLT", "gold_etf": "GLD", "small_caps": "IWM",
                 "oil": "CLUSD", "gold": "GCUSD", "dollar": "DX-Y.NYB", "eurusd": "EURUSD", "usdjpy": "USDJPY",
                 "spx": "^GSPC", "rut": "^RUT", "tnx": "^TNX"}


def symbols(source=None):
    """Symbols from a panel .npz (its `symbols` array) or a coverage.json (`retained_symbols`); default = expanded panel."""
    import numpy as np
    source = Path(source) if source else ROOT / "work/fjepa-expanded-data/coverage.json"
    if source.suffix == ".npz":
        return [str(s) for s in np.load(source, allow_pickle=True)["symbols"]]
    cov = json.loads(source.read_text())
    syms = cov.get("retained_symbols") or cov.get("symbols") or cov.get("kept_symbols")
    if not syms:
        syms = list(np.load(source.parent / "panel_raw.npz", allow_pickle=True)["symbols"])
    return [str(s) for s in syms]


def log(**info):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **info}), flush=True)


def fetch_macro(client):
    out = {"treasury": [], "prices": {}}
    import calendar
    seen = set()
    for year in range(2003, 2026):  # the endpoint returns at most ~63 rows per call, so use monthly windows
        for month in range(1, 13):
            start, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
            try:
                for row in client.get("treasury-rates", {"from": start, "to": end}):
                    if row.get("date") not in seen:
                        seen.add(row["date"]); out["treasury"].append(row)
            except DataAccessError as exc:
                log(stage="treasury", window=start, error=exc.as_dict())
    log(stage="treasury_done", rows=len(out["treasury"]))
    for name, sym in MACRO_PROXIES.items():
        rows = []
        for year in range(2003, 2026):
            try:
                rows.extend(client.get("historical-price-eod/dividend-adjusted", {"symbol": sym, "from": f"{year}-01-01", "to": f"{year}-12-31"}))
            except DataAccessError as exc:
                log(stage="price", symbol=sym, year=year, error=exc.as_dict())
                break
        out["prices"][name] = {"symbol": sym, "rows": rows}
        log(stage="price_done", name=name, symbol=sym, rows=len(rows), first=min((r["date"] for r in rows), default=None))
    (OUT / "macro_raw.json").write_text(json.dumps(out))


def fetch_earnings(client, syms):
    out, missing = {}, []
    for i, s in enumerate(syms):
        try:
            rows = client.get("earnings", {"symbol": s, "limit": 400})
        except DataAccessError as exc:
            rows = []
            log(stage="earnings_error", symbol=s, error=exc.as_dict())
            time.sleep(1.0)
        if rows:
            out[s] = rows
        else:
            missing.append(s)
        if (i + 1) % 100 == 0:
            log(stage="earnings_progress", done=i + 1, with_data=len(out), missing=len(missing))
        time.sleep(0.2)
    (OUT / "earnings_raw.json").write_text(json.dumps(out))
    (OUT / "earnings_coverage.json").write_text(json.dumps({"symbols": len(syms), "with_data": len(out), "missing": missing}, indent=1))
    log(stage="earnings_done", with_data=len(out), missing=len(missing))


def fetch_statements(client, syms):
    out, missing = {}, {}
    for i, s in enumerate(syms):
        entry = {}
        for kind in ("income-statement", "balance-sheet-statement", "cash-flow-statement"):
            try:
                entry[kind] = client.get(kind, {"symbol": s, "period": "quarter", "limit": 400})
            except DataAccessError as exc:
                entry[kind] = []
                missing.setdefault(s, []).append({kind: exc.as_dict()})
                time.sleep(1.0)
            time.sleep(0.15)
        if any(entry.values()):
            out[s] = entry
        else:
            missing.setdefault(s, []).append("empty")
        if (i + 1) % 100 == 0:
            log(stage="statements_progress", done=i + 1, with_data=len(out), missing=len(missing))
    (OUT / "statements_raw.json").write_text(json.dumps(out))
    (OUT / "statements_coverage.json").write_text(json.dumps({"symbols": len(syms), "with_data": len(out), "missing": missing}, indent=1))
    log(stage="statements_done", with_data=len(out), missing=len(missing))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=("macro", "earnings", "statements"), required=True)
    parser.add_argument("--symbols-from", default=None, help="panel .npz or coverage.json giving the symbol list (default: expanded panel)")
    parser.add_argument("--out", default=None, help="output folder (default: work/fjepa-families)")
    args = parser.parse_args(argv)
    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    client = FMPClient(ROOT / "work" / "fmp-live-cache")
    syms = symbols(args.symbols_from)
    log(stage="start", what=args.what, symbols=len(syms))
    if args.what == "macro":
        fetch_macro(client)
    elif args.what == "earnings":
        fetch_earnings(client, syms)
    else:
        fetch_statements(client, syms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
