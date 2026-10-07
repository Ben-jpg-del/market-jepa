"""Safe FMP access, causal panels, normalization, and resumable date sampling.

Live downloads are explicit function calls. Importing this module makes no
network requests. No key is stored in cache files, exception text or reports.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import requests

from .features import (REFERENCE_ETFS, STOCK_FEATURES, MARKET_FEATURES,
                       FEATURE_NOTES, compute_features, rolling)


class DataAccessError(RuntimeError):
    def __init__(self, code, endpoint, status=None):
        self.code,self.endpoint,self.status = code,endpoint,status
        super().__init__(f"FMP {code}; endpoint={endpoint}; status={status}")

    def as_dict(self):
        return {"ok":False,"code":self.code,"endpoint":self.endpoint,"status":self.status}


KEY_FILE = Path.home() / ".config" / "fmp" / "api_key"  # owner-only file; never logged or printed


def get_api_key():
    key = os.environ.get("FMP_API_KEY","")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("FMP_API_KEY")
        except Exception:
            pass
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise DataAccessError("missing_secret","credentials")
    return key


class FMPClient:
    """Bounded requests, redacted failures, content caches and header auth."""
    BASE = "https://financialmodelingprep.com/stable/"

    def __init__(self,cache_dir,api_key=None,session=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self._key = api_key if api_key is not None else get_api_key()
        self._session = session or requests.Session()

    def __repr__(self):
        return "FMPClient(credentials=<redacted>)"

    def get(self,endpoint,params=None,refresh=False):
        if not re.fullmatch(r"[a-z0-9/-]+",endpoint):
            raise ValueError("Invalid endpoint")
        params = dict(params or {})
        if any("key" in str(k).lower() or "token" in str(k).lower() for k in params):
            raise ValueError("Credentials must be passed through the private header")
        identity = json.dumps({"endpoint":endpoint,"params":params},sort_keys=True)
        path = self.cache_dir/(hashlib.sha256(identity.encode()).hexdigest()+".json")
        if path.exists() and not refresh:
            return json.loads(path.read_text())["data"]
        for attempt in range(3):
            try:
                response = self._session.get(self.BASE+endpoint,params=params,
                    headers={"apikey":self._key},timeout=(10,45))
                if response.status_code in (429,500,502,503,504) and attempt<2:
                    time.sleep(2**attempt)
                    continue
                if response.status_code!=200:
                    raise DataAccessError("http_error",endpoint,response.status_code)
                payload = response.json()
            except DataAccessError:
                raise
            except (requests.RequestException,ValueError):
                # Do not propagate provider body, request URL, headers or exception.
                raise DataAccessError("network_or_json_error",endpoint) from None
            if not isinstance(payload,list):
                raise DataAccessError("unexpected_schema",endpoint)
            # Provider may reflect a credential in an error row; don't cache errors.
            if payload and not isinstance(payload[0],dict):
                raise DataAccessError("unexpected_schema",endpoint)
            if any(any(str(k).lower() in ("error","error message","apikey") for k in row) for row in payload):
                raise DataAccessError("provider_error",endpoint)
            if self._key and self._key in json.dumps(payload):
                raise DataAccessError("unsafe_response",endpoint)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps({"request":json.loads(identity),"data":payload}))
            temp.replace(path)
            return payload
        raise DataAccessError("retries_exhausted",endpoint)


def _prices_frame(rows,adjusted=True):
    if not rows:
        raise DataAccessError("empty_history","prices")
    frame = pd.DataFrame(rows)
    if adjusted:
        columns = {"adjOpen":"Open","adjHigh":"High","adjLow":"Low","adjClose":"Close","volume":"Volume"}
    else:
        # FMP's non-split-adjusted endpoint uses the misleading adj* names in
        # its response too. Adjustment meaning comes from the endpoint, not
        # the field prefix. Retain support for unprefixed response revisions.
        if {"adjOpen","adjClose"}.issubset(frame.columns):
            columns = {"adjOpen":"RawOpen","adjClose":"RawClose","volume":"RawVolume"}
        else:
            columns = {"open":"RawOpen","close":"RawClose","volume":"RawVolume"}
    if not {"date",*columns}.issubset(frame.columns):
        raise DataAccessError("price_schema_missing_fields","prices")
    frame = frame[["date",*columns]].rename(columns=columns)
    frame["date"] = pd.to_datetime(frame["date"],errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise DataAccessError("invalid_or_duplicate_dates","prices")
    frame = frame.set_index("date").sort_index()
    for col in columns.values():
        frame[col] = pd.to_numeric(frame[col],errors="coerce")
        frame.loc[frame[col]<=0,col] = np.nan
    return frame


def fetch_price_history(client,symbol,start,end,adjusted=True):
    """Backfill FMP's 5,000-row cap without forcing requests for short histories.

    The first full-span response is cache-compatible with earlier downloads.
    Only a full 5,000-row response whose oldest date is later than start causes
    another request, ending one calendar day before that oldest observation.
    A short response may represent an IPO; coverage validation belongs to the
    caller rather than inventing observations before listing.
    """
    if not re.fullmatch(r"[A-Za-z0-9.\-^]{1,20}",symbol):
        raise ValueError("Invalid symbol")
    first_day,last_day=pd.Timestamp(start),pd.Timestamp(end)
    if first_day>last_day:
        raise ValueError("History start must not follow end")
    endpoint="historical-price-eod/"+("dividend-adjusted" if adjusted else "non-split-adjusted")
    request_end=last_day
    chunks=[]
    previous_oldest=None
    for _ in range(32):
        payload=client.get(endpoint,{"symbol":symbol,"from":first_day.strftime("%Y-%m-%d"),
                                     "to":request_end.strftime("%Y-%m-%d")})
        if not payload:
            if not chunks: raise DataAccessError("empty_history",endpoint)
            break
        chunk=_prices_frame(payload,adjusted=adjusted)
        oldest=chunk.index.min()
        if previous_oldest is not None and oldest>=previous_oldest:
            raise DataAccessError("backfill_no_progress",endpoint)
        if chunk.index.max()>request_end or oldest<first_day:
            raise DataAccessError("history_outside_requested_dates",endpoint)
        chunks.append(chunk)
        if len(payload)<5000 or oldest<=first_day:
            break
        previous_oldest=oldest
        request_end=oldest-pd.Timedelta(days=1)
        if request_end<first_day: break
    else:
        raise DataAccessError("backfill_page_limit",endpoint)
    combined=pd.concat(chunks).sort_index()
    # Adjacent page requests do not overlap; keep one identical-date record if
    # a provider includes an inclusive boundary in a future response revision.
    combined=combined.loc[~combined.index.duplicated(keep="first")]
    return combined.loc[first_day:last_day]


def preflight(client):
    """Small entitlement/schema check. A success is not a PIT certification."""
    result = {"ok":True,"checks":{},"point_in_time_verified":False}
    checks = {
        "adjusted_prices":("historical-price-eod/dividend-adjusted",{"symbol":"SPY","from":"2024-01-02","to":"2024-01-12"}),
        "raw_prices":("historical-price-eod/non-split-adjusted",{"symbol":"SPY","from":"2024-01-02","to":"2024-01-12"}),
        "historical_membership":("historical-sp500-constituent",{}),
    }
    for name,(endpoint,params) in checks.items():
        try:
            rows = client.get(endpoint,params)
            if name.endswith("prices"):
                _prices_frame(rows,adjusted=name=="adjusted_prices")
            elif not rows or not {"date","symbol","removedTicker"}.issubset(rows[0]):
                raise DataAccessError("membership_schema_needs_review",endpoint)
            result["checks"][name] = {"ok":True,"rows":len(rows)}
        except DataAccessError as exc:
            result["checks"][name] = exc.as_dict()
            if name=="adjusted_prices":
                result["ok"] = False
    return result


def download_dataset(client,symbols,start="2003-01-01",end="2025-12-31"):
    """Explicit download of requested stocks + SPY/reference ETFs; cached/restartable.

    Raw endpoint failure retains a diagnostic panel but disables liquidity/PIT
    claims. No endpoint substitution can silently turn adjusted data into raw.
    """
    names = list(dict.fromkeys([*symbols,"SPY",*REFERENCE_ETFS]))
    prices,failures,raw_failures = {},{},{}
    for symbol in names:
        if not re.fullmatch(r"[A-Za-z0-9.\-^]{1,20}",symbol):
            raise ValueError("Invalid symbol")
        try:
            frame = fetch_price_history(client,symbol,start,end,adjusted=True)
        except DataAccessError as exc:
            failures[symbol] = exc.as_dict()
            continue
        try:
            raw = fetch_price_history(client,symbol,start,end,adjusted=False)
            frame = frame.join(raw,how="left")
        except DataAccessError as exc:
            raw_failures[symbol] = exc.as_dict()
        prices[symbol] = frame
    report = {"requested_symbols":names,"downloaded_symbols":sorted(prices),
        "adjusted_failures":failures,"raw_failures":raw_failures,"start":start,"end":end,
        "point_in_time_verified":False,"purpose":"diagnostic_current_universe",
        "limitations":["Historical membership, corporate actions, delisted histories and delisting returns require independent validation.",
                        "Volume supplied by dividend-adjusted endpoint is used as delivered; split-volume convention requires validation."]}
    return prices,report


def reconstruct_membership(dates,symbols,current_symbols,events,current_asof):
    """Reverse dated constituent changes from an explicitly dated current roster.

    Events are FMP rows with date, symbol (addition), removedTicker. This
    reconstructs a candidate roster only; feed completeness is not certified.
    Events on a date are applied for that day's membership. Symbol aliases and
    effective intraday timing must be validated before an investment backtest.
    """
    dates = np.asarray(dates,dtype="datetime64[D]")
    if dates[-1] > np.datetime64(current_asof):
        raise ValueError("Membership snapshot is older than requested dates")
    normalized = []
    for row in events:
        if not {"date","symbol","removedTicker"}.issubset(row):
            raise ValueError("Historical membership schema needs review")
        day = np.datetime64(row["date"],"D")
        if day > np.datetime64(current_asof):
            raise ValueError("Event occurs after current membership snapshot")
        normalized.append((day,row.get("symbol") or "",row.get("removedTicker") or ""))
    normalized.sort(key=lambda x:x[0],reverse=True)
    active = set(current_symbols)
    result = np.zeros((len(dates),len(symbols)),bool)
    cursor=0
    for t in range(len(dates)-1,-1,-1):
        while cursor<len(normalized) and normalized[cursor][0]>dates[t]:
            _,added,removed=normalized[cursor]
            if added: active.discard(added)
            if removed: active.add(removed)
            cursor+=1
        result[t] = [s in active for s in symbols]
    return result


@dataclass
class MarketPanel:
    dates: np.ndarray
    symbols: list[str]
    stock: np.ndarray
    market: np.ndarray
    stock_obs: np.ndarray
    market_obs: np.ndarray
    open: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    volume_baseline20: np.ndarray
    benchmark_open: np.ndarray
    benchmark_close: np.ndarray
    universe: np.ndarray
    metadata: dict
    origin_observed: np.ndarray | None = None  # [T,K] carried eligibility mask (assembled panels); None = derive from the 4 fast channels


def fast_observed(panel):
    """Per-day 'fast channels observed' mask that defines origin eligibility.

    v1 panels: the first four stock channels (r1, overnight_gap, log_high_low, relative_volume) all observed.
    assembled panels drop two of those channels but carry the identical mask as ``origin_observed``.
    """
    carried = getattr(panel, "origin_observed", None)
    if carried is not None:
        return np.asarray(carried, dtype=bool)
    return np.asarray(panel.stock_obs[..., :4]).all(axis=-1)


def monthly_universe(dates,membership,raw_close,raw_volume,max_stocks=128):
    """At first session each month select using preceding 60 raw-dollar sessions."""
    liquidity = rolling(raw_close*raw_volume,60,shift=1)
    out = np.zeros_like(membership,bool)
    months = np.asarray(dates,dtype="datetime64[M]")
    chosen = np.array([],dtype=int)
    for t in range(len(dates)):
        if t==0 or months[t]!=months[t-1]:
            valid = np.flatnonzero(membership[t] & np.isfinite(liquidity[t]))
            chosen = valid[np.argsort(-liquidity[t,valid],kind="stable")[:max_stocks]]
        out[t,chosen] = membership[t,chosen]
    return out


def build_panel(prices,symbols,membership=None,universe_mode="diagnostic_current",max_stocks=128):
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise ValueError("At least one stock is required")
    missing = set([*symbols,"SPY",*REFERENCE_ETFS])-prices.keys()
    if missing:
        raise ValueError("Missing required price histories: "+", ".join(sorted(missing)))
    if universe_mode not in ("diagnostic_current","historical_candidate"):
        raise ValueError("PIT verification is not automatic; use diagnostic_current or historical_candidate")
    dates = prices["SPY"].index.sort_values()
    def matrix(column,names):
        return np.column_stack([prices[s].reindex(dates)[column].to_numpy(float)
            if column in prices[s] else np.full(len(dates),np.nan) for s in names])
    open_,high,low,close,volume = [matrix(c,symbols) for c in ("Open","High","Low","Close","Volume")]
    ref = matrix("Close",REFERENCE_ETFS)
    spy_open = matrix("Open",["SPY"])[:,0]
    spy_close = matrix("Close",["SPY"])[:,0]
    d = dates.to_numpy(dtype="datetime64[D]")
    if membership is None:
        if universe_mode=="historical_candidate":
            raise ValueError("Historical mode requires an aligned membership array")
        membership = np.ones(close.shape,bool)
    membership = np.asarray(membership,dtype=bool)
    if membership.shape!=close.shape:
        raise ValueError("Membership shape must equal [date,stock]")
    raw_close,raw_volume = matrix("RawClose",symbols),matrix("RawVolume",symbols)
    raw_available = np.isfinite(raw_close).any() and np.isfinite(raw_volume).any()
    if raw_available:
        universe = monthly_universe(d,membership,raw_close,raw_volume,max_stocks)
        selection = "monthly preceding-60-session raw-dollar-volume"
    elif universe_mode=="historical_candidate":
        raise ValueError("Historical selection requires raw as-traded close and volume")
    else:
        # Fixed explicit diagnostic symbols, not fake adjusted-dollar liquidity.
        universe = membership.copy()
        universe[:,max_stocks:] = False
        selection = "fixed explicit diagnostic symbol list; raw liquidity unavailable"
    stock,market,base = compute_features(open_,high,low,close,volume,spy_close,ref,universe)
    stock_obs,market_obs = np.isfinite(stock),np.isfinite(market)
    digest = hashlib.sha256()
    # Include every feature source, selection source, resulting feature and mask.
    # In particular, a reference-ETF correction must invalidate old snapshots
    # even if all traded stocks' close prices remained unchanged.
    fingerprint_arrays = {"dates":d.astype("int64"),"open":open_,"high":high,"low":low,
        "close":close,"volume":volume,"reference_close":ref,"benchmark_open":spy_open,
        "benchmark_close":spy_close,"raw_close":raw_close,"raw_volume":raw_volume,
        "membership":membership,"universe":universe,"stock":stock,"market":market,
        "stock_obs":stock_obs,"market_obs":market_obs,"volume_baseline20":rolling(volume,20)}
    for name,arr in fingerprint_arrays.items():
        digest.update(json.dumps({"name":name,"dtype":str(arr.dtype),"shape":arr.shape},sort_keys=True).encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    digest.update(json.dumps({"symbols":symbols,"stock_features":STOCK_FEATURES,
        "market_features":MARKET_FEATURES,"feature_notes":FEATURE_NOTES,
        "universe_mode":universe_mode,"max_stocks":max_stocks},sort_keys=True).encode())
    meta = {"schema_version":1,"feature_version":"book-pilot-1",
        "stock_features":list(STOCK_FEATURES),"market_features":list(MARKET_FEATURES),
        "feature_notes":FEATURE_NOTES,"dataset_fingerprint":digest.hexdigest(),
        "universe_mode":universe_mode,"point_in_time_verified":False,
        "universe_selection":selection,"normalizer":None,
        "limitations":["Research diagnostic only until historical membership, delisted prices/returns, corporate actions and adjustment conventions are validated.",
                       "Book features including stock JANUS use the explicit research conventions in feature_notes."]}
    return MarketPanel(d,symbols,stock.astype(np.float32),market.astype(np.float32),stock_obs,market_obs,
        open_,close,volume,rolling(volume,20),spy_open,spy_close,universe,meta)


def fit_normalizer(panel,cutoff="2018-12-31",start="2006-01-01"):
    train = (panel.dates>=np.datetime64(start)) & (panel.dates<=np.datetime64(cutoff))
    if not train.any():
        raise ValueError("No observations in normalizer training period")
    def fit(values,observed,eligibility):
        medians,scales,counts = [],[],[]
        for j in range(values.shape[-1]):
            valid = observed[...,j] & eligibility
            v = values[...,j][valid]
            if not len(v):
                medians.append(0.);scales.append(1.);counts.append(0)
                continue
            q = np.quantile(v,[.25,.5,.75])
            scale = float(q[2]-q[0])
            if scale <= 1e-6:  # degenerate IQR (binary/sparse channel): normal-consistent std scale
                scale = float(max(1.349*np.std(v),1e-6))
            medians.append(float(q[1]));scales.append(scale);counts.append(len(v))
        return {"median":medians,"iqr":scales,"counts":counts}
    return {"stock":fit(panel.stock[train],panel.stock_obs[train],panel.universe[train]),
            "market":fit(panel.market[train],panel.market_obs[train],np.ones(train.sum(),bool)),
            "train_start":start,"train_cutoff":cutoff,"clip":10.,"dataset_fingerprint":panel.metadata["dataset_fingerprint"]}


def normalize_panel(panel,normalizer):
    def apply(x,obs,spec):
        z = np.clip((x-np.array(spec["median"]))/np.array(spec["iqr"]),-normalizer["clip"],normalizer["clip"])
        return np.where(obs,z,0.).astype(np.float32)
    return replace(panel,stock=apply(panel.stock,panel.stock_obs,normalizer["stock"]),
        market=apply(panel.market,panel.market_obs,normalizer["market"]),
        metadata={**panel.metadata,"normalizer":normalizer})


def save_panel(panel,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays = {k:v for k,v in vars(panel).items() if isinstance(v,np.ndarray)}
    with path.open("wb") as f:
        np.savez_compressed(f,**arrays,symbols=np.array(panel.symbols),metadata=np.array(json.dumps(panel.metadata)))


def load_panel(path):
    with np.load(path,allow_pickle=False) as z:
        data = {k:z[k] for k in z.files if k not in ("symbols","metadata")}
        return MarketPanel(**data,symbols=z["symbols"].tolist(),metadata=json.loads(str(z["metadata"])))


class DateBatchSampler:
    """Origin selection never inspects future asset observations or labels."""
    def __init__(self,panel,batch_dates=16,stocks_per_date=32,context=60,horizon=5,
                 seed=42,start="2006-01-01",end="2018-12-31",feature_set="book",min_stocks=2):
        if horizon not in (1,5,20): raise ValueError("Pilot horizons are 1, 5 or 20 (20 only through the walk-forward sampler)")
        if feature_set not in ("basic","book"): raise ValueError("Unknown feature set")
        if context%5: raise ValueError("Context must contain complete five-day patches")
        if not panel.metadata.get("normalizer"): raise ValueError("Normalize panel before sampling")
        self.panel,self.batch_dates,self.stocks_per_date=panel,batch_dates,stocks_per_date
        self.context,self.horizon,self.feature_set=context,horizon,feature_set
        self.start,self.end,self.min_stocks=start,end,min_stocks
        self.normalizer_hash=hashlib.sha256(json.dumps(panel.metadata["normalizer"],sort_keys=True).encode()).hexdigest()
        self.rng=np.random.default_rng(seed)
        # Eligibility uses only full fast-input context and membership at origin.
        fast = fast_observed(panel)
        complete = rolling(fast.astype(float),context,"sum")==context
        self.eligibility = panel.universe & complete
        times = np.arange(len(panel.dates))
        valid = (times>=context-1) & (times+horizon<len(times))
        valid &= (panel.dates>=np.datetime64(start)) & (panel.dates<=np.datetime64(end))
        # Future labels must stay within encoder-training cutoff (date only).
        cutoff_index=np.searchsorted(panel.dates,np.datetime64(end),side="right")-1
        valid &= times+horizon<=cutoff_index
        valid &= self.eligibility.sum(axis=1)>=min_stocks
        valid &= panel.market_obs[:,0]
        self.origins=np.flatnonzero(valid)
        if not len(self.origins): raise ValueError("No valid origins for sampler period/context")

    def state_dict(self):
        return {"rng":self.rng.bit_generator.state,"context":self.context,"horizon":self.horizon,
                "feature_set":self.feature_set,"batch_dates":self.batch_dates,
                "stocks_per_date":self.stocks_per_date,"start":self.start,"end":self.end,
                "min_stocks":self.min_stocks,"normalizer_hash":self.normalizer_hash,
                "dataset_fingerprint":self.panel.metadata["dataset_fingerprint"]}

    def load_state_dict(self,state):
        for key in ("context","horizon","feature_set","batch_dates","stocks_per_date",
                    "start","end","min_stocks","normalizer_hash"):
            if state.get(key)!=getattr(self,key): raise ValueError("Sampler configuration changed: "+key)
        if state["dataset_fingerprint"]!=self.panel.metadata["dataset_fingerprint"]:
            raise ValueError("Sampler dataset changed")
        self.rng.bit_generator.state=state["rng"]

    def sample(self):
        if self.horizon>5: raise ValueError("The base sampler builds a five-day future buffer; use walkforward_data.WalkForwardSampler for longer horizons")
        p=self.panel;B=self.batch_dates;K=self.stocks_per_date;C=self.context
        origins=self.rng.choice(self.origins,B,replace=True)
        Cs,Cm=p.stock.shape[-1],p.market.shape[-1]
        batch={"stock":np.zeros((B,K,C,Cs),np.float32),"stock_obs":np.zeros((B,K,C,Cs),bool),
               "market":np.zeros((B,C,Cm),np.float32),"market_obs":np.zeros((B,C,Cm),bool),
               "eligibility":np.zeros((B,K),bool),"future_stock":np.zeros((B,K,5,Cs),np.float32),
               "future_stock_obs":np.zeros((B,K,5,Cs),bool),"future_market":np.zeros((B,5,Cm),np.float32),
               "future_market_obs":np.zeros((B,5,Cm),bool),"target_valid":np.zeros((B,K),bool),
               "origins":origins,"symbol_indices":np.full((B,K),-1,np.int64)}
        norm=p.metadata["normalizer"]["stock"]
        for b,t in enumerate(origins):
            candidates=np.flatnonzero(self.eligibility[t])
            chosen=self.rng.choice(candidates,min(K,len(candidates)),replace=False)
            n=len(chosen);window=slice(t-C+1,t+1);fut=slice(t+1,t+self.horizon+1);pad=5-self.horizon
            batch["symbol_indices"][b,:n]=chosen
            batch["stock"][b,:n]=p.stock[window][:,chosen].transpose(1,0,2)
            batch["stock_obs"][b,:n]=p.stock_obs[window][:,chosen].transpose(1,0,2)
            batch["market"][b]=p.market[window];batch["market_obs"][b]=p.market_obs[window]
            batch["eligibility"][b,:n]=True
            future=p.stock[fut][:,chosen,:4].transpose(1,0,2).copy()
            observed=p.stock_obs[fut][:,chosen,:4].transpose(1,0,2).copy()
            # Target volume is relative to a frozen baseline known at origin.
            baseline=p.volume_baseline20[t,chosen]
            with np.errstate(divide="ignore",invalid="ignore"):
                vol=np.log(p.volume[fut][:,chosen].T/baseline[:,None])
            observed[:,:,3]=np.isfinite(vol)
            future[:,:,3]=np.where(observed[:,:,3],np.clip((vol-norm["median"][3])/norm["iqr"][3],-10,10),0)
            batch["future_stock"][b,:n,pad:,:4]=future
            batch["future_stock_obs"][b,:n,pad:,:4]=observed
            batch["future_market"][b,pad:,0]=p.market[fut,0]
            batch["future_market_obs"][b,pad:,0]=p.market_obs[fut,0]
            batch["target_valid"][b,:n]=observed.all(axis=(1,2)) & p.market_obs[fut,0].all()
        if self.feature_set=="basic":
            batch["stock"][...,9:]=0;batch["stock_obs"][...,9:]=False
            batch["market"][...,2:]=0;batch["market_obs"][...,2:]=False
        return batch

    __call__=sample


def synthetic_panel(n_dates=800,n_stocks=8,seed=42,start="2016-01-01"):
    """Deterministic realistic-shaped test fixture; never a financial result."""
    rng=np.random.default_rng(seed)
    dates=pd.bdate_range(start,periods=n_dates)
    symbols=[f"S{i:03d}" for i in range(n_stocks)]
    common=rng.normal(.0002,.009,n_dates)
    prices={}
    for symbol in [*symbols,"SPY",*REFERENCE_ETFS]:
        ret=common+rng.normal(0,.007,n_dates)
        close=100*np.exp(np.cumsum(ret))
        open_=close*np.exp(rng.normal(0,.003,n_dates))
        high=np.maximum(open_,close)*np.exp(rng.uniform(.001,.015,n_dates))
        low=np.minimum(open_,close)*np.exp(-rng.uniform(.001,.015,n_dates))
        volume=rng.lognormal(14,.3,n_dates)
        prices[symbol]=pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Volume":volume,
                                   "RawOpen":open_,"RawClose":close,"RawVolume":volume},index=dates)
    panel=build_panel(prices,symbols)
    panel.metadata["synthetic"]=True
    panel.metadata["purpose"]="software smoke test only"
    return panel
