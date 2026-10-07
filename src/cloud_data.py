"""Explicit, restartable cloud data preparation; no network on import.

Run ``python -m src.cloud_data --output-dir /content/fjepa/data``.
Credentials come from FMP_API_KEY or Colab Secrets. No automatic paid API
upgrade occurs. Full historical-roster download may require thousands of
requests; FMP's existing plan quotas apply. Cached successful requests resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import (FMPClient,DataAccessError,fetch_price_history,preflight,download_dataset,
    reconstruct_membership,monthly_universe,build_panel,save_panel)
from .features import REFERENCE_ETFS


# Explicit diagnostic fallback, never represented as historical constituents.
DIAGNOSTIC_SYMBOLS = (
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","AVGO","ORCL",
    "JPM","BAC","GS","V","MA","BRK-B","JNJ","UNH",
    "LLY","PFE","MRK","ABBV","XOM","CVX","COP","CAT",
    "HON","GE","PG","KO","PEP","WMT","COST","HD",
)


class PreparationError(RuntimeError):
    """A safe error whose details contain no provider response or credentials."""


def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,sort_keys=True))
    temp.replace(path)


def price_coverage(frame,start,end):
    """Detect truncated entitlement responses before a large universe download."""
    selected=frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    expected=len(pd.bdate_range(start,end))
    usable=selected[["Open","High","Low","Close","Volume"]].notna().all(axis=1)
    first=str(selected.index.min().date()) if len(selected) else None
    last=str(selected.index.max().date()) if len(selected) else None
    start_ok=bool(len(selected) and selected.index.min()<=pd.Timestamp(start)+pd.Timedelta(days=7))
    end_ok=bool(len(selected) and selected.index.max()>=pd.Timestamp(end)-pd.Timedelta(days=7))
    rows_ok=bool(len(selected)>=.90*expected)
    values_ok=bool(len(selected) and usable.mean()>=.99)
    train_ok=bool((selected.index<=pd.Timestamp("2018-12-31")).any())
    return {"ok":bool(start_ok and end_ok and rows_ok and values_ok and train_ok),
        "first_date":first,"last_date":last,"rows":len(selected),
        "expected_weekdays":expected,"usable_rows":int(usable.sum()),
        "start_covered":start_ok,"end_covered":end_ok,"row_count_sufficient":rows_ok,
        "pre2019_history_present":train_ok,"requested_start":start,"requested_end":end}


def _symbol(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9.\-^]{1,20}",value):
        raise PreparationError("Membership contains an unsupported symbol; review schema/aliases before proceeding")
    return value


def historical_candidate(client,dates,current_asof):
    """Strictly reconstruct a candidate universe; never certify completeness."""
    current=client.get("sp500-constituent",refresh=True)
    events=client.get("historical-sp500-constituent",refresh=True)
    if not current or not events:
        raise PreparationError("Historical/current constituent endpoint returned no records")
    if any(not isinstance(row,dict) or "symbol" not in row for row in current):
        raise PreparationError("Current constituent schema must contain a symbol in every row")
    current_symbols=sorted({_symbol(row["symbol"]) for row in current})
    if len(current_symbols)<400:
        raise PreparationError("Current constituent response is too small; possible truncation or wrong schema")
    cleaned=[];seen=set();all_symbols=set(current_symbols)
    for row in events:
        if not isinstance(row,dict) or not {"date","symbol","removedTicker"}.issubset(row):
            raise PreparationError("Historical constituent schema must contain date, symbol and removedTicker")
        try:
            day=pd.Timestamp(row["date"])
        except (ValueError,TypeError):
            raise PreparationError("Historical constituent event date is invalid") from None
        if pd.isna(day) or day.tzinfo is not None:
            raise PreparationError("Historical constituent event date is invalid")
        date=day.strftime("%Y-%m-%d")
        added=_symbol(row["symbol"]) if row["symbol"] else ""
        removed=_symbol(row["removedTicker"]) if row["removedTicker"] else ""
        if not added and not removed:
            raise PreparationError("Historical constituent event contains neither addition nor removal")
        identity=(date,added,removed)
        if identity in seen: continue
        seen.add(identity)
        cleaned.append({"date":date,"symbol":added,"removedTicker":removed})
        all_symbols.update(s for s in (added,removed) if s)
    if min(row["date"] for row in cleaned)>str(dates[0]):
        raise PreparationError("Historical constituent response does not reach requested warmup start; completeness cannot be assumed")
    names=sorted(all_symbols)
    try:
        membership=reconstruct_membership(dates,names,current_symbols,cleaned,current_asof)
    except ValueError:
        raise PreparationError("Historical constituent reconstruction failed; review dates and schema") from None
    # Symbols absent for the entire requested period do not need price requests.
    used=membership.any(axis=0)
    names=[s for s,keep in zip(names,used) if keep]
    membership=membership[:,used]
    report={"mode":"historical_candidate","current_snapshot_asof":current_asof,
        "current_constituents":len(current_symbols),"historical_events":len(cleaned),
        "earliest_event":min(row["date"] for row in cleaned),"candidate_symbols":names,
        "point_in_time_verified":False,
        "membership_fingerprint":hashlib.sha256(json.dumps({"current":current_symbols,"events":cleaned,
            "asof":current_asof},sort_keys=True).encode()).hexdigest(),
        "unresolved_checks":["Completeness/effective time of changes", "Ticker alias and identifier continuity",
                             "Delisted histories and terminal delisting returns"]}
    return names,membership,report


def _source_fingerprint(prices,symbols,membership):
    digest=hashlib.sha256(json.dumps(symbols).encode())
    digest.update(np.ascontiguousarray(membership).tobytes())
    for symbol in sorted(prices):
        frame=prices[symbol].sort_index().sort_index(axis=1)
        digest.update(symbol.encode())
        digest.update(json.dumps(list(frame.columns)).encode())
        digest.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    return digest.hexdigest()


def prepare_cloud_data(output_dir,cache_dir=None,allow_diagnostic=False,start="2003-01-01",
                       end="2025-12-31",max_stocks=128,client=None,current_asof=None,
                       progress=print):
    """Save raw feature panel and accounting report, or a safe failure report.

    Missing adjusted histories remain listed in the report and are represented
    as unavailable columns during full-universe selection. Missing raw data is
    never replaced by adjusted dollar volume. No result claims PIT validation.
    """
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    report_path=output/"coverage.json"
    report={"status":"preflight","point_in_time_verified":False,"start":start,"end":end,
        "allow_diagnostic":bool(allow_diagnostic),"max_stocks":max_stocks}
    client=client or FMPClient(cache_dir or output/"fmp-cache")
    asof=current_asof or datetime.now(timezone.utc).date().isoformat()
    try:
        if pd.Timestamp(end)>pd.Timestamp(asof):
            raise PreparationError("Requested end is later than current snapshot date")
        report["preflight"]=preflight(client)
        if not report["preflight"]["ok"]:
            raise PreparationError("FMP adjusted-price entitlement/schema preflight failed")
        progress("Checking full SPY history before downloading stock histories.")
        spy=fetch_price_history(client,"SPY",start,end,adjusted=True)
        report["benchmark_coverage"]=price_coverage(spy,start,end)
        _write_json(report_path,report)
        if not report["benchmark_coverage"]["ok"]:
            raise PreparationError("SPY history is truncated or incomplete; broad download stopped before consuming stock requests")
        dates=spy.loc[pd.Timestamp(start):pd.Timestamp(end)].index.to_numpy(dtype="datetime64[D]")
        try:
            symbols,membership,universe_report=historical_candidate(client,dates,asof)
            mode="historical_candidate"
            if not report["preflight"]["checks"]["raw_prices"]["ok"]:
                raise PreparationError("Raw-price access is required for historical liquidity selection")
        except (DataAccessError,PreparationError) as exc:
            safe_reason=exc.as_dict() if isinstance(exc,DataAccessError) else str(exc)
            report["historical_candidate_failure"]=safe_reason
            if not allow_diagnostic:
                raise PreparationError("Historical-universe preparation unavailable; diagnostic fallback requires --allow-diagnostic") from None
            mode="diagnostic_current"
            symbols=list(DIAGNOSTIC_SYMBOLS)
            membership=np.ones((len(dates),len(symbols)),bool)
            universe_report={"mode":mode,"candidate_symbols":symbols,"point_in_time_verified":False,
                "reason":"Explicitly authorized 32-stock diagnostic fallback; survivor-biased historical sample"}
        report["universe"]=universe_report
        report["status"]="downloading"
        report["initial_price_endpoint_requests"]=2*(len(symbols)+1+len(REFERENCE_ETFS))
        report["pagination_policy"]="Only responses capped at 5,000 rows request earlier history; cached responses are reused. Additional backfill requests may apply."
        _write_json(report_path,report)
        progress(f"Preparing {len(symbols)} candidate stocks plus SPY and nine reference ETFs. Successful responses are cached.")
        prices,download_report=download_dataset(client,symbols,start,end)
        report["downloads"]=download_report
        reference_missing=[s for s in ("SPY",*REFERENCE_ETFS) if s not in prices]
        if reference_missing:
            report["missing_references"]=reference_missing
            raise PreparationError("A required reference history is unavailable; training panel not created")
        # Explicitly account for unavailable histories without pretending they
        # were not members of the original candidate universe.
        empty=pd.DataFrame(columns=["Open","High","Low","Close","Volume","RawOpen","RawClose","RawVolume"],
                           index=pd.DatetimeIndex([],name="date"),dtype=float)
        for symbol in symbols:
            if symbol not in prices: prices[symbol]=empty.copy()
        # Restrict provider rows to the requested benchmark calendar.
        for symbol in prices:
            prices[symbol]=prices[symbol].loc[pd.Timestamp(start):pd.Timestamp(end)]
        calendar=pd.DatetimeIndex(dates)
        def raw_matrix(column):
            return np.column_stack([prices[s].reindex(calendar)[column].to_numpy(float)
                if column in prices[s] else np.full(len(dates),np.nan) for s in symbols])
        if mode=="historical_candidate":
            raw_close,raw_volume=raw_matrix("RawClose"),raw_matrix("RawVolume")
            selected=monthly_universe(dates,membership,raw_close,raw_volume,max_stocks)
        else:
            # Diagnostic selection may still use observed raw liquidity when
            # available; build_panel applies the same policy after pruning.
            raw_close,raw_volume=raw_matrix("RawClose"),raw_matrix("RawVolume")
            if np.isfinite(raw_close).any() and np.isfinite(raw_volume).any():
                selected=monthly_universe(dates,membership,raw_close,raw_volume,max_stocks)
            else:
                selected=membership.copy();selected[:,max_stocks:]=False
        count=selected.sum(axis=1)
        train=(dates>=np.datetime64("2006-01-01")) & (dates<=np.datetime64("2018-12-31"))
        report["selection"]={"minimum_train_count":int(count[train].min()) if train.any() else 0,
            "median_train_count":float(np.median(count[train])) if train.any() else 0,
            "maximum_count":int(count.max()),"train_dates":int(train.sum()),
            "train_dates_under_32":int((count[train]<32).sum()),
            "train_dates_under_2":int((count[train]<2).sum())}
        if not train.any() or np.max(count[train],initial=0)<2:
            raise PreparationError("Insufficient pre-2019 stock history after selection; training cannot proceed")
        # Selection is evaluated over the full reconstructed candidate roster
        # first. Columns never selected have no contribution to JANUS ranks.
        keep=selected.any(axis=0)
        kept_symbols=[s for s,included in zip(symbols,keep) if included]
        report["never_selected_symbols"]=[s for s,included in zip(symbols,keep) if not included]
        report["retained_symbols"]=kept_symbols
        report["original_candidate_count"]=len(symbols)
        report["retained_symbol_count"]=len(kept_symbols)
        source_fingerprint=_source_fingerprint(prices,symbols,membership)
        report["source_fingerprint"]=source_fingerprint
        # Per-stock coverage remains explicit even for columns removed above.
        report["stock_coverage"]={s:{"adjusted_rows":int(prices[s].get("Close",pd.Series(dtype=float)).notna().sum()),
            "raw_rows":int(prices[s].get("RawClose",pd.Series(dtype=float)).notna().sum()),
            "member_dates":int(membership[:,i].sum()),"selected_dates":int(selected[:,i].sum()),
            "first_price_date":str(prices[s].index.min().date()) if len(prices[s]) else None,
            "last_price_date":str(prices[s].index.max().date()) if len(prices[s]) else None}
            for i,s in enumerate(symbols)}
        _write_json(report_path,report)
        progress(f"Computing causal features for {len(kept_symbols)} stocks selected at least once.")
        retained={s:prices[s] for s in [*kept_symbols,"SPY",*REFERENCE_ETFS]}
        panel=build_panel(retained,kept_symbols,membership=membership[:,keep],universe_mode=mode,max_stocks=max_stocks)
        if not np.array_equal(panel.universe,selected[:,keep]):
            raise PreparationError("Pruned-universe selection changed unexpectedly; panel rejected")
        parent_fingerprint=panel.metadata["dataset_fingerprint"]
        panel.metadata["source_fingerprint"]=source_fingerprint
        panel.metadata["unpruned_source_selection_accounted"]=True
        panel.metadata["dataset_fingerprint"]=hashlib.sha256((parent_fingerprint+source_fingerprint).encode()).hexdigest()
        panel.metadata["coverage_report"]=str(report_path)
        panel.metadata["membership_snapshot_asof"]=asof if mode=="historical_candidate" else None
        panel.metadata["point_in_time_verified"]=False
        panel.metadata["limitations"].append("Candidate membership and missing/delisted histories are accounted for but not independently validated; no alpha claim.")
        path=output/"panel_raw.npz";save_panel(panel,path)
        report.update({"status":"ready_research_only","panel_path":str(path),
            "dataset_fingerprint":panel.metadata["dataset_fingerprint"],
            "stock_shape":list(panel.stock.shape),"market_shape":list(panel.market.shape),
            "point_in_time_verified":False})
        _write_json(report_path,report)
        progress("Feature panel saved. Historical data limitations remain in coverage.json.")
        return panel,report
    except (PreparationError,DataAccessError) as exc:
        report["status"]="blocked"
        report["error"]=exc.as_dict() if isinstance(exc,DataAccessError) else str(exc)
        _write_json(report_path,report)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-diagnostic",action="store_true")
    parser.add_argument("--start",default="2003-01-01")
    parser.add_argument("--end",default="2025-12-31")
    parser.add_argument("--max-stocks",type=int,default=128)
    args=parser.parse_args()
    try:
        _,report=prepare_cloud_data(**vars(args))
    except (PreparationError,DataAccessError) as exc:
        print("Data preparation stopped:",str(exc))
        raise SystemExit(2) from None
    print(json.dumps({k:report[k] for k in ("status","panel_path","stock_shape","point_in_time_verified")},indent=2))


if __name__=="__main__":
    main()
