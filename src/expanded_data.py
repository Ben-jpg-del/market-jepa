"""Historically dated IWB holdings proxy for a broader US stock universe.

IWB tracks the Russell 1000, but fund holdings are NOT exact index membership.
The official Swiss iShares route preserves historical CSV downloads even when
the redesigned US route ignores asOfDate. Every response date is verified.
No current list is projected backwards and no credentials are required.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,dataclass,field
from datetime import datetime,timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests

IWB_PAGE="https://www.ishares.com/ch/professionals/en/products/239707/ishares-russell-1000-etf"
IWB_CSV=IWB_PAGE+"/1495092304805.ajax"


@dataclass(frozen=True)
class FundSpec:
    """Which iShares fund archive to read. The Swiss professionals route serves dated CSVs for every fund."""
    ticker: str
    page_url: str
    ajax_id: str
    fund_title: str
    index_name: str
    file_name: str
    min_equities: int
    max_equities: int

    @property
    def csv_url(self):
        return f"{self.page_url}/{self.ajax_id}"


IWB=FundSpec(ticker="IWB",page_url=IWB_PAGE,ajax_id="1495092304805.ajax",fund_title="iShares Russell 1000 ETF",
             index_name="Russell 1000",file_name="IWB_holdings",min_equities=700,max_equities=1400)
# Verified 2026-10-06: the same ajax id serves IWM history (2010-01-29 -> 2,006 equity rows, 2024-01-31 -> 1,963).
IWM=FundSpec(ticker="IWM",page_url="https://www.ishares.com/ch/professionals/en/products/239710/ishares-russell-2000-etf",
             ajax_id="1495092304805.ajax",fund_title="iShares Russell 2000 ETF",index_name="Russell 2000",
             file_name="IWM_holdings",min_equities=1500,max_equities=2300)


class HoldingsUnavailable(RuntimeError):
    def __init__(self,code,date,fund="IWB"):
        self.code,self.date=code,str(date)
        super().__init__(f"{fund} holdings {code}: {date}")


@dataclass
class HoldingsSnapshot:
    as_of_date: str
    requested_date: str
    source_url: str
    content_sha256: str
    retrieved_at: str
    holdings: list[dict]
    unsupported_tickers: list[str]
    excluded_non_equity_rows: int
    duplicate_resolution: list[dict] = field(default_factory=list)

    @property
    def symbols(self):
        return sorted({row["symbol"] for row in self.holdings if row["symbol"] is not None})


def _number(value):
    return float(value.replace(",","").replace("’","").replace("'","").replace("\u00a0","").strip())


def parse_iwb_csv(content,requested_date,symbol_map=None,retrieved_at=None,min_equities=None,max_equities=None,spec=IWB):
    """Parse and validate provenance before exposing any historical symbols.

    ``spec`` selects the fund (title check, plausible equity-count bounds, source URL);
    explicit ``min_equities``/``max_equities`` override the spec's bounds (tests use tiny files).
    """
    if min_equities is None:min_equities=spec.min_equities
    if max_equities is None:max_equities=spec.max_equities
    body=content.decode("utf-8-sig") if isinstance(content,bytes) else content.lstrip("\ufeff")
    day=pd.Timestamp(requested_date).strftime("%Y-%m-%d")
    if "<html" in body[:500].lower() or "<!doctype" in body[:500].lower():
        raise HoldingsUnavailable("html_instead_of_csv",day,spec.ticker)
    rows=list(csv.reader(io.StringIO(body)))
    if not rows or spec.fund_title not in rows[0][0]:
        raise HoldingsUnavailable("wrong_fund",day,spec.ticker)
    date_rows=[row for row in rows[:15] if row and row[0].strip()=="Fund Holdings as of"]
    if len(date_rows)!=1 or len(date_rows[0])<2 or date_rows[0][1].strip() in ("","-"):
        raise HoldingsUnavailable("historical_date_unavailable",day,spec.ticker)
    try:actual=pd.to_datetime(date_rows[0][1].replace("Sept","Sep"),dayfirst=True).strftime("%Y-%m-%d")
    except (ValueError,TypeError):raise HoldingsUnavailable("invalid_as_of_header",day,spec.ticker) from None
    if actual!=day:raise HoldingsUnavailable("as_of_date_mismatch",day,spec.ticker)
    indices=[i for i,row in enumerate(rows) if row and row[0].strip()=="Ticker" and "Asset Class" in row]
    if len(indices)!=1:raise HoldingsUnavailable("holdings_header_missing",day,spec.ticker)
    header=rows[indices[0]]
    required={"Ticker","Name","Asset Class","Market Value","Weight (%)"}
    if not required.issubset(header):raise HoldingsUnavailable("holdings_schema_incomplete",day,spec.ticker)
    holdings=[];unsupported=[];excluded=0;mapping=symbol_map or {};source_rows=[]
    for row in rows[indices[0]+1:]:
        if not row or not any(cell.strip() for cell in row):continue
        if len(row)!=len(header):
            # Legal footers contain one long CSV field; malformed data rows fail.
            if len(row)==1:continue
            raise HoldingsUnavailable("malformed_holdings_row",day,spec.ticker)
        item=dict(zip(header,row))
        if item["Asset Class"]!="Equity":excluded+=1;continue
        ticker=item["Ticker"].strip()
        try:market_value=_number(item["Market Value"]);weight=_number(item["Weight (%)"])
        except ValueError:raise HoldingsUnavailable("invalid_holding_number",day,spec.ticker) from None
        if not np.isfinite(market_value) or not np.isfinite(weight):
            raise HoldingsUnavailable("invalid_holding_number",day,spec.ticker)
        if market_value<=0:excluded+=1;continue
        symbol=mapping.get(ticker,ticker)
        if not isinstance(symbol,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]{0,19}",symbol):
            symbol=None;unsupported.append(ticker)
        try:source_price=_number(item.get("Price",""))
        except ValueError:source_price=None
        if source_price is not None and not np.isfinite(source_price):source_price=None
        holdings.append({"source_ticker":ticker,"symbol":symbol,"name":item["Name"],
            "sector":item.get("Sector"),"market_value":market_value,"weight_percent":weight,
            "source_price":source_price,"currency":item.get("Currency"),
            "market_currency":item.get("Market Currency"),"source_fx_rate":item.get("FX Rate"),
            "location":item.get("Location"),"exchange":item.get("Exchange"),
            "cusip":item.get("CUSIP"),"isin":item.get("ISIN")})
        source_rows.append(item)
    if not min_equities<=len(holdings)<=max_equities:
        raise HoldingsUnavailable("implausible_equity_count",day,spec.ticker)
    # Historical files can retrospectively relabel two different securities with
    # one modern ticker/name. Same-name MRK rows in 2006, for example, have very
    # different prices. Never join those solely because their labels agree.
    groups={}
    for index,holding in enumerate(holdings):
        if holding["symbol"] is not None:groups.setdefault(holding["symbol"],[]).append(index)
    resolutions=[];remove=set()
    for symbol,indices in groups.items():
        if len(indices)<2:continue
        originals=[source_rows[i] for i in indices]
        identity_fields=("Ticker","Name","Exchange","Currency","Market Currency","CUSIP","ISIN")
        identity_same=all(tuple(row.get(k) for k in identity_fields)==
                          tuple(originals[0].get(k) for k in identity_fields) for row in originals)
        try:prices=[_number(row.get("Price","")) for row in originals]
        except ValueError:prices=[]
        price_same=bool(prices) and all(np.isfinite(p) and p>0 and p==prices[0] for p in prices)
        safe=identity_same and price_same
        resolution={"symbol":symbol,"resolution":"aggregated_identical_source_identity_and_price" if safe
                    else "excluded_ambiguous_symbol_for_snapshot", "source_row_count":len(indices),
                    "original_source_rows":originals}
        resolutions.append(resolution)
        if safe:
            holdings[indices[0]]["market_value"]=sum(holdings[i]["market_value"] for i in indices)
            holdings[indices[0]]["weight_percent"]=sum(holdings[i]["weight_percent"] for i in indices)
            holdings[indices[0]]["aggregated_source_row_count"]=len(indices)
            remove.update(indices[1:])
        else:
            for i in indices:
                holdings[i]["symbol"]=None
                holdings[i]["exclusion_reason"]="ambiguous_duplicate_source_identity"
    holdings=[row for i,row in enumerate(holdings) if i not in remove]
    payload=content if isinstance(content,bytes) else content.encode("utf-8")
    return HoldingsSnapshot(actual,day,spec.csv_url,hashlib.sha256(payload).hexdigest(),
        retrieved_at or datetime.now(timezone.utc).isoformat(),holdings,sorted(set(unsupported)),excluded,resolutions)


class IWBHistoryClient:
    def __init__(self,cache_dir,session=None,symbol_map=None,request_spacing=.3,spec=IWB):
        self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True)
        self.session=session
        self.symbol_map=symbol_map or {}
        self.request_spacing=request_spacing
        self.spec=spec

    def fetch(self,date):
        day=pd.Timestamp(date).strftime("%Y-%m-%d")
        fund=self.spec.ticker
        path=self.cache/f"{fund}_{day}.csv"
        if path.exists():payload=path.read_bytes()
        else:
            params={"fileType":"csv","fileName":self.spec.file_name,"dataType":"fund",
                    "asOfDate":pd.Timestamp(day).strftime("%Y%m%d")}
            for attempt in range(3):
                try:
                    get=self.session.get if self.session is not None else requests.get
                    response=get(self.spec.csv_url,params=params,timeout=(10,45))
                except requests.RequestException:
                    if attempt==2:raise HoldingsUnavailable("network_error",day,fund) from None
                    time.sleep(2**attempt);continue
                if response.status_code in (429,500,502,503,504) and attempt<2:
                    time.sleep(2**attempt);continue
                if response.status_code!=200:raise HoldingsUnavailable(f"http_{response.status_code}",day,fund)
                payload=response.content
                if len(payload)>5_000_000:raise HoldingsUnavailable("response_too_large",day,fund)
                # Cache even an explicitly unavailable date; never treat it as a snapshot.
                temp=path.with_suffix(".tmp");temp.write_bytes(payload);temp.replace(path)
                time.sleep(self.request_spacing)
                break
            else:raise HoldingsUnavailable("retries_exhausted",day,fund)
        return parse_iwb_csv(payload,day,self.symbol_map,spec=self.spec)


def monthly_request_dates(trading_dates,start="2006-01-01",end="2021-12-31"):
    dates=pd.DatetimeIndex(np.asarray(trading_dates,dtype="datetime64[D]"))
    dates=dates[(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))]
    if not len(dates):raise ValueError("No trading dates in requested interval")
    if not dates.is_monotonic_increasing or dates.has_duplicates:raise ValueError("Trading dates must be ordered and unique")
    return dates.to_series().groupby(dates.to_period("M")).max().dt.strftime("%Y-%m-%d").tolist()


def fetch_monthly_history(client,trading_dates,start="2006-01-01",end="2021-12-31",
                          workers=3,allow_internal_gaps=False,progress=None):
    """Fetch month ends; allow missing prefix, fail on unexplained later gaps."""
    requested=monthly_request_dates(trading_dates,start,end)
    def fetch(day):
        try:return day,client.fetch(day),None
        except HoldingsUnavailable as exc:return day,None,exc.code
    with ThreadPoolExecutor(max_workers=max(1,min(workers,4))) as pool:
        results=list(pool.map(fetch,requested))
    snapshots=[];missing=[]
    for day,snapshot,error in results:
        if snapshot is not None:snapshots.append(snapshot)
        else:missing.append({"date":day,"reason":error,"position":"prefix" if not snapshots else "internal_or_tail"})
        if progress:progress({"date":day,"equities":len(snapshot.holdings) if snapshot else 0,"error":error})
    if not snapshots:raise HoldingsUnavailable("no_valid_historical_snapshots",start)
    internal=[x for x in missing if x["position"]!="prefix"]
    spec=getattr(client,"spec",IWB)
    report={"source":f"official_iShares_{spec.ticker}_monthly_holdings_proxy","fund":spec.ticker,"exact_russell_index_membership":False,
        "first_snapshot":snapshots[0].as_of_date,"last_snapshot":snapshots[-1].as_of_date,
        "requested_months":len(requested),"valid_months":len(snapshots),"missing_months":missing,
        "union_symbols":sorted(set().union(*(set(s.symbols) for s in snapshots))),
        "minimum_equities":min(len(s.holdings) for s in snapshots),"maximum_equities":max(len(s.holdings) for s in snapshots),
        "unsupported_tickers":sorted(set().union(*(set(s.unsupported_tickers) for s in snapshots))),
        "duplicate_symbol_resolutions":[{"date":s.as_of_date,"symbol":r["symbol"],
            "resolution":r["resolution"],"source_row_count":r["source_row_count"]}
            for s in snapshots for r in s.duplicate_resolution],
        "source_headers_verified":True,"publication_timestamps_verified":False,"point_in_time_verified":False,
        "limitations":[f"ETF holdings proxy, not exact {spec.index_name} index membership.",
            "Historical publication timestamps, revisions, ticker identity/renames and delisted prices require further validation.",
            "No universe assigned before the first dated snapshot; no use of today's list for earlier dates."],
        "internal_gaps_allowed":allow_internal_gaps}
    if internal and not allow_internal_gaps:
        error=HoldingsUnavailable("missing_internal_months",internal[0]["date"])
        error.coverage_report=report
        raise error
    return snapshots,report


def membership_from_snapshots(trading_dates,snapshots,publication_lag_sessions=1,max_staleness_days=45):
    """Past-only monthly membership with explicit assumed publication lag."""
    if publication_lag_sessions<1:raise ValueError("At least one subsequent trading session is required")
    dates=np.asarray(trading_dates,dtype="datetime64[D]")
    if dates.ndim!=1 or not len(dates) or (np.diff(dates).astype(int)<=0).any():
        raise ValueError("Trading dates must be strictly increasing")
    ordered=sorted(snapshots,key=lambda s:s.as_of_date)
    if len({s.as_of_date for s in ordered})!=len(ordered):raise ValueError("Duplicate snapshot dates")
    symbols=sorted(set().union(*(set(s.symbols) for s in ordered))) if ordered else []
    lookup={s:i for i,s in enumerate(symbols)}
    membership=np.zeros((len(dates),len(symbols)),bool)
    chosen=np.full(len(dates),np.datetime64("NaT","D"),dtype="datetime64[D]")
    for i,snapshot in enumerate(ordered):
        asof=np.datetime64(snapshot.as_of_date,"D")
        begin=np.searchsorted(dates,asof,side="right")+publication_lag_sessions-1
        finish=len(dates)
        if i+1<len(ordered):
            finish=np.searchsorted(dates,np.datetime64(ordered[i+1].as_of_date),side="right")+publication_lag_sessions-1
        finish=min(finish,len(dates))
        indices=np.arange(min(begin,len(dates)),finish)
        indices=indices[(dates[indices]-asof).astype(int)<=max_staleness_days]
        columns=[lookup[s] for s in snapshot.symbols]
        membership[np.ix_(indices,columns)]=True
        chosen[indices]=asof
    report={"publication_lag_sessions_assumed":publication_lag_sessions,"max_staleness_days":max_staleness_days,
        "uncovered_dates":int(np.isnat(chosen).sum()),"first_covered_date":str(dates[~np.isnat(chosen)][0]) if (~np.isnat(chosen)).any() else None,
        "snapshot_date_by_origin":chosen.astype(str).tolist(),"point_in_time_verified":False,
        "timing_note":"Previous month-end holdings are first used on a subsequent trading session; actual historical release timestamps are unverified."}
    return symbols,membership,report


def save_roster_bundle(snapshots,report,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={"schema_version":1,"report":report,"snapshots":[asdict(s) for s in snapshots]}
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(payload,indent=2));temp.replace(path)


def load_roster_bundle(path):
    value=json.loads(Path(path).read_text())
    if value.get("schema_version")!=1:raise ValueError("Unsupported roster schema")
    return [HoldingsSnapshot(**s) for s in value["snapshots"]],value["report"]
