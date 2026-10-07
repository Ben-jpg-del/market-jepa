"""Causal feature construction for the separate future-JEPA research pilot.

Book references: Timothy Masters, photographed pp. 184–189, 269–282,
300–316 and the author's Mult/JANUS.CPP. This is a research implementation,
not a claim of predictive value or bitwise agreement with all author code.
See FEATURE_NOTES for intentional departures and precise window conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata

REFERENCE_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
STOCK_FEATURES = ("r1", "overnight_gap", "log_high_low", "relative_volume",
    "r5", "r20", "r60", "log_rv20", "log_rv20_rv60", "reactivity_10_4",
    "deviation_60_ema3", "purified_momentum20_outer252", "janus_ref_rm_fractile63",
    "janus_ref_rm_fractile_delta5")
MARKET_FEATURES = ("spy_r1", "spy_log_rv20", "turbulence252", "coherence60",
    "coherence_delta5", "absorption252_top2_ema126", "absorption_shift15_252",
    "janus_reference_rss_delta5")
# Economic families used by the family-token encoder and family-mask objective.
# Order and membership are part of the model contract; do not reorder.
# Per-feature grouping (one token stream per indicator) for tokenization="feature".
STOCK_FEATURE_GROUPS = tuple((name, (i,)) for i, name in enumerate(STOCK_FEATURES))
# Shortcut partners for feature masking: when feature X is the prediction target, these
# channels are hidden with it because they reconstruct X arithmetically from the same window.
# Filled from outputs/colab-tokenization/channel-predictability (pairwise R2 >= 0.8, either
# direction) on 2026-09-19; see FEATURE_NOTES["shortcut_partners"].
FEATURE_SHORTCUT_PARTNERS = {
    "r1": ("r5", "r20", "r60"), "r5": ("r1", "r20", "r60"), "r20": ("r1", "r5", "r60"), "r60": ("r1", "r5", "r20"),
    "log_rv20": ("log_rv20_rv60",), "log_rv20_rv60": ("log_rv20",),
    "janus_ref_rm_fractile63": ("janus_ref_rm_fractile_delta5",), "janus_ref_rm_fractile_delta5": ("janus_ref_rm_fractile63",),
}
# New information families (2026-09-19). Macro channels join the MARKET stream; event channels
# join the STOCK stream as a fifth family. Built by scripts/data/build_family_channels.py.
MACRO_FEATURES = ("rate10y_chg20", "rate10y_chg60", "duration_20", "usdjpy_20", "credit_ig_vs_duration_20",
                  "vix_log", "vix_chg20", "oil_20", "dollar_20", "gold_20", "small_minus_large_20", "spx_60",
                  "curve_2s10s", "curve_chg20", "bill3m_chg60")
EVENT_FEATURES = ("days_since_report_log", "days_to_expected_report_log", "eps_sue", "revenue_surprise",
                  "announcement_abnormal_return", "post_announcement_drift", "report_expected_within_10")
STOCK_FAMILIES = (
    ("fast", (0, 1, 2, 3)),          # r1, overnight_gap, log_high_low, relative_volume
    ("trend", (4, 5, 6)),            # r5, r20, r60
    ("risk", (7, 8)),                # log_rv20, log_rv20_rv60
    ("book", (9, 10, 11, 12, 13)),   # reactivity, deviation, purified momentum, JANUS fractile + delta
)
FUNDAMENTAL_FEATURES = ("log_book_to_market", "gross_profitability", "asset_growth", "accruals",
                        "log_market_cap", "earnings_yield", "dividend_yield")
STOCK_FAMILIES_EXTENDED = STOCK_FAMILIES + (("events", tuple(range(14, 14 + len(EVENT_FEATURES)))),)
STOCK_FAMILIES_FULL = STOCK_FAMILIES_EXTENDED + (
    ("fundamentals", tuple(range(14 + len(EVENT_FEATURES), 14 + len(EVENT_FEATURES) + len(FUNDAMENTAL_FEATURES)))),)


# ---- assembled layout (2026-10-06): the stable-sign core from outputs/indicator-ic, four contiguous families ----
STOCK_FEATURES_ASSEMBLED = ("r1", "overnight_gap", "purified_momentum20_outer252", "r5", "janus_ref_rm_fractile_delta5",   # reversal
                     "log_rv20", "log_rv20_rv60", "janus_ref_rm_fractile63",                                        # risk
                     "eps_sue", "revenue_surprise", "post_announcement_drift", "days_since_report_log",              # earnings
                     "log_market_cap", "gross_profitability", "log_book_to_market")                                  # fundamentals
MARKET_FEATURES_ASSEMBLED = ("spy_r1", "spy_log_rv20", "turbulence252", "coherence60", "absorption252_top2_ema126", "absorption_shift15_252",
                      "vix_log", "rate10y_chg60", "curve_2s10s", "credit_ig_vs_duration_20", "small_minus_large_20")
STOCK_FAMILIES_ASSEMBLED = (("reversal", (0, 1, 2, 3, 4)), ("risk", (5, 6, 7)), ("earnings", (8, 9, 10, 11)), ("fundamentals", (12, 13, 14)))
# where each assembled channels comes from: (source array, name in that source)
ASSEMBLED_CHANNEL_SOURCES = {**{n: ("stock", n) for n in STOCK_FEATURES_ASSEMBLED if n in STOCK_FEATURES},
                      **{n: ("events", n) for n in STOCK_FEATURES_ASSEMBLED if n in EVENT_FEATURES},
                      **{n: ("fundamentals", n) for n in STOCK_FEATURES_ASSEMBLED if n in FUNDAMENTAL_FEATURES},
                      **{n: ("market", n) for n in MARKET_FEATURES_ASSEMBLED if n in MARKET_FEATURES},
                      **{n: ("macro", n) for n in MARKET_FEATURES_ASSEMBLED if n in MACRO_FEATURES}}
ELIGIBILITY_CHANNELS = STOCK_FEATURES[:4]  # fast channels whose 60-session observation defines origin eligibility


def stock_feature_names(count: int, layout: str = "v1") -> tuple[str, ...]:
    if layout == "v2":
        if count != len(STOCK_FEATURES_ASSEMBLED):
            raise ValueError(f"assembled layout has {len(STOCK_FEATURES_ASSEMBLED)} stock channels, not {count}")
        return STOCK_FEATURES_ASSEMBLED
    names = STOCK_FEATURES + EVENT_FEATURES + FUNDAMENTAL_FEATURES
    if count > len(names):
        return names + tuple(f"stock_c{i}" for i in range(len(names), count))
    return names[:count]


def market_feature_names(count: int, layout: str = "v1") -> tuple[str, ...]:
    if layout == "v2":
        if count != len(MARKET_FEATURES_ASSEMBLED):
            raise ValueError(f"assembled layout has {len(MARKET_FEATURES_ASSEMBLED)} market channels, not {count}")
        return MARKET_FEATURES_ASSEMBLED
    names = MARKET_FEATURES + MACRO_FEATURES
    if count > len(names):
        return names + tuple(f"market_c{i}" for i in range(len(names), count))
    return names[:count]


def stock_families_for(count: int, layout: str = "v1"):
    """Token families covering exactly `count` stock channels (v1: 14 base, 21 + events, 28 + fundamentals; 15)."""
    if layout == "v2":
        if count != len(STOCK_FEATURES_ASSEMBLED):
            raise ValueError(f"assembled layout has {len(STOCK_FEATURES_ASSEMBLED)} stock channels, not {count}")
        return STOCK_FAMILIES_ASSEMBLED
    if count == len(STOCK_FEATURES):
        return STOCK_FAMILIES
    if count == len(STOCK_FEATURES) + len(EVENT_FEATURES):
        return STOCK_FAMILIES_EXTENDED
    if count == len(STOCK_FEATURES) + len(EVENT_FEATURES) + len(FUNDAMENTAL_FEATURES):
        return STOCK_FAMILIES_FULL
    raise ValueError(f"No family layout for {count} stock channels")
FEATURE_NOTES = {
    "timing": "All context features are known after origin close. No forward fill. Missing and warmup values retain false observation masks.",
    "returns": "Natural-log returns; relative volume is log(volume / preceding 20-session mean). RV uses population standard deviation of 20 or 60 daily log returns.",
    "reactivity": "Book LB=10, smoothing multiplier=4; LB+1 range, first positive-volume initialization corrected. 100*Phi(0.6*PM*range*SV/(SR^2*V))-50.",
    "deviation": "Current-inclusive 60 log-price regression, residual/RMS, 100*Phi(z/60^(1/6))-50; EMA span 3.",
    "purification": "20-session log momentum regressed over current-inclusive 252 cases on SPY log-price Legendre trend20, acceleration20, and mean absolute log changes over 60 prices. Polynomial bases use their actual inner lengths (a correction to the printed constructor). Returns residual/(RMS+1e-6).",
    "shortcut_partners": "Pairwise out-of-sample R2 >= 0.8 of a channel's latest value from another channel's 60-day history (outputs/colab-tokenization/channel-predictability, 2026-09-19): return horizons r1/r5/r20/r60 mutually 0.97-1.00; log_rv20 <-> log_rv20_rv60 0.87; JANUS fractile <-> its delta 1.00. Under feature_mask the target is hidden together with these partners.",
    "turbulence": "252 prior returns excluding current, covariance shrunk 10% toward its diagonal, log1p(Mahalanobis squared distance / 9). Deliberately does not use the book's disputed F calibration.",
    "dependence": "Coherence uses 60 returns; absorption uses 252 returns, top 2 of 9 eigenvalues, separately smoothed numerator/denominator span126. These parameters count RETURNS, whereas author C++ lookback counts prices (one fewer return). Absorption shift is (mean15-mean252)/population_std252, code sign convention.",
    "janus": "Reference-basket extension: exact RS/RSS/DOM/RM mechanics on nine fixed ETFs; each stock scored against reference median returns and reference RSS-expansion DOM gate, then ranked among origin-eligible stocks. This is NOT the author's original joint stock-basket JANUS. Average ranks resolve ties. Stock RM requires 126 prior valid returns; RSS uses tail fraction .1. Five-session change uses values/ranks actually computed at each historical date.",
}


def rolling(x, window, operation="mean", shift=0):
    frame = pd.DataFrame(x).shift(shift).rolling(window, min_periods=window)
    out = frame.std(ddof=0) if operation == "std" else getattr(frame, operation)()
    out = out.to_numpy()
    return out[:, 0] if np.asarray(x).ndim == 1 else out


def lag(x, n):
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[n:] = x[:-n]
    return out


def log_return(close, n=1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(close / lag(close, n))


def _ema_valid(x, span):
    """Restart after gaps; never smooth through missing observations."""
    out = np.full_like(x, np.nan)
    state = np.nan
    alpha = 2 / (span + 1)
    for t, value in enumerate(x):
        if not np.isfinite(value):
            state = np.nan
        else:
            state = value if not np.isfinite(state) else alpha * value + (1-alpha)*state
            out[t] = state
    return out


def reactivity(high, low, close, volume, lookback=10, multiplier=4):
    out = np.full_like(close, np.nan)
    alpha = 2 / (lookback * multiplier + 1)
    for k in range(close.shape[1]):
        start = None
        sr = sv = np.nan
        for t in range(len(close)):
            valid = np.isfinite([high[t,k], low[t,k], close[t,k], volume[t,k]]).all() and volume[t,k] > 0
            if not valid:
                start = None
                continue
            if start is None:
                start, sr, sv = t, high[t,k]-low[t,k], volume[t,k]
            else:
                lo = max(start, t-lookback)
                width = high[lo:t+1,k].max()-low[lo:t+1,k].min()
                sr = alpha*width+(1-alpha)*sr
                sv = alpha*volume[t,k]+(1-alpha)*sv
            if t-start >= lookback and sr > 0:
                raw = (close[t,k]-close[t-lookback,k])*width*sv/(sr*sr*volume[t,k])
                out[t,k] = 100*ndtr(.6*raw)-50
    return out


def deviation(close, benchmark, window=60):
    y, x = np.log(close), np.log(benchmark)[:,None]
    mx, my = rolling(x,window), rolling(y,window)
    vx = np.maximum(rolling(x*x,window)-mx*mx, 0)
    vy = np.maximum(rolling(y*y,window)-my*my, 0)
    cov = rolling(x*y,window)-mx*my
    beta = np.divide(cov,vx,out=np.zeros_like(cov),where=vx>1e-18)
    residual = y-my-beta*(x-mx)
    rms = np.sqrt(np.maximum(vy-beta*cov,0))
    z = np.divide(residual,rms,out=np.zeros_like(residual),where=rms>1e-12)
    out = 100*ndtr(z/window**(1/6))-50
    out[~np.isfinite(my) | ~np.broadcast_to(np.isfinite(mx),my.shape)] = np.nan
    return np.column_stack([_ema_valid(out[:,k],3) for k in range(out.shape[1])])


def purification(momentum, benchmark, outer=252):
    """Rolling regression with shared controls and strict complete windows."""
    lx = np.log(benchmark)
    trend = np.full(len(lx),np.nan)
    acceleration = trend.copy()
    x = np.linspace(-1,1,20)
    p1 = x-x.mean()
    p2 = .5*(3*x*x-1)
    p2 -= p2.mean()
    for t in range(19,len(lx)):
        window = lx[t-19:t+1]
        if np.isfinite(window).all():
            trend[t] = window@p1
            acceleration[t] = window@p2
    # 60 prices contain 59 changes, matching printed close-only volatility.
    volatility = rolling(np.abs(log_return(benchmark)),59)
    controls = np.column_stack((trend,acceleration,volatility))
    mx = rolling(controls,outer)
    my = rolling(momentum,outer)
    xy = np.stack([rolling(controls[:,i,None]*momentum,outer) for i in range(3)],axis=1)
    yy = rolling(momentum*momentum,outer)
    xx = np.empty((len(lx),3,3))
    for i in range(3):
        for j in range(3):
            xx[:,i,j] = rolling(controls[:,i]*controls[:,j],outer)-mx[:,i]*mx[:,j]
    out = np.full_like(momentum,np.nan)
    for t in range(outer-1,len(lx)):
        good = np.isfinite(my[t])
        if not good.any() or not np.isfinite(xx[t]).all():
            continue
        # Local control scaling preserves the fitted space and conditions solve.
        scales = np.sqrt(np.maximum(np.diag(xx[t]),1e-30))
        corr = xx[t]/scales[:,None]/scales[None,:]
        cov = xy[t][:,good]-mx[t,:,None]*my[t,good]
        beta = np.linalg.pinv(corr,rcond=1e-7)@(cov/scales[:,None])
        residual = momentum[t,good]-my[t,good] - ((controls[t]-mx[t])/scales)@beta
        variance = np.maximum(yy[t,good]-my[t,good]**2-np.sum(beta*cov/scales[:,None],axis=0),0)
        out[t,good] = residual/(np.sqrt(variance)+1e-6)
    return out


def janus_score(returns, index, limit):
    """Author RS/RM offensive-minus-defensive calculation for one window."""
    median = np.median(index)
    offensive = index >= median
    io = (index[offensive]-median).sum()+1e-30
    idef = (index[~offensive]-median).sum()-1e-30
    mo = (returns[offensive]-median).sum(axis=0)
    md = (returns[~offensive]-median).sum(axis=0)
    return np.clip(70.710678*(mo/io-md/idef),-limit,limit)


def janus_reference(ref_returns, stock_returns, eligible, lookback=63):
    """Port core reference mechanics, then documented reference-basket extension."""
    n = len(ref_returns)
    rss = np.full(n,np.nan)
    ref_index = np.median(ref_returns,axis=1)
    for t in range(lookback,n):
        r = ref_returns[t-lookback+1:t+1]
        if np.isfinite(r).all():
            scores = np.sort(janus_score(r,ref_index[t-lookback+1:t+1],200))
            tails = max(1,int(.1*(r.shape[1]+1)))
            rss[t] = np.mean(scores[-tails:]-scores[:tails])
    changes = rss-lag(rss,1)
    gate = changes > 0
    dom_ref = np.where(gate[:,None],ref_returns,0.)
    dom_stock = np.where(gate[:,None],stock_returns,0.)
    # Unknown gate must stay unknown, not become a zero return.
    dom_ref[~np.isfinite(changes)] = np.nan
    dom_stock[~np.isfinite(changes)] = np.nan
    dom_stock[~np.isfinite(stock_returns)] = np.nan
    dom_index = np.median(dom_ref,axis=1)
    fractile = np.full_like(stock_returns,np.nan)
    for t in range(2*lookback,n):
        start = t-lookback+1
        idx = dom_index[start:t+1]
        valid = eligible[t] & np.isfinite(stock_returns[t-2*lookback+1:t+1]).all(axis=0)
        if valid.sum()<2 or not np.isfinite(idx).all():
            continue
        scores = janus_score(dom_stock[start:t+1,valid],idx,300)
        fractile[t,valid] = (rankdata(scores,method="average")-1)/(len(scores)-1)
    return fractile, rss


def market_dependence(returns):
    n,m = returns.shape
    turbulence,coherence,absorption = [np.full(n,np.nan) for _ in range(3)]
    numer = denom = np.nan
    alpha = 2/127
    weights = np.linspace(1,-1,m)
    for t in range(n):
        if t>=252 and np.isfinite(returns[t-252:t+1]).all():
            hist = returns[t-252:t]
            cov = np.cov(hist,rowvar=False,ddof=0)
            cov = .9*cov+.1*np.diag(np.diag(cov))
            diff = returns[t]-hist.mean(axis=0)
            d2 = diff@np.linalg.pinv(cov,rcond=1e-8)@diff
            turbulence[t] = np.log1p(max(d2,0)/m)
        if t>=59 and np.isfinite(returns[t-59:t+1]).all():
            corr = np.corrcoef(returns[t-59:t+1],rowvar=False)
            if np.isfinite(corr).all():
                eig = np.maximum(np.linalg.eigvalsh(corr)[::-1],0)
                coherence[t] = 200*(weights@eig/eig.sum()-.5)
        if t>=251 and np.isfinite(returns[t-251:t+1]).all():
            eig = np.maximum(np.linalg.eigvalsh(np.cov(returns[t-251:t+1],rowvar=False,ddof=0)),0)
            top,total = eig[-2:].sum(),eig.sum()
            numer = top if not np.isfinite(numer) else alpha*top+(1-alpha)*numer
            denom = total if not np.isfinite(denom) else alpha*total+(1-alpha)*denom
            absorption[t] = 100*numer/(denom+1e-30)
        else:
            numer = denom = np.nan
    std = rolling(absorption,252,"std")
    shift = (rolling(absorption,15)-rolling(absorption,252))/np.where(std>1e-12,std,np.nan)
    return turbulence,coherence,absorption,shift


def compute_features(open_,high,low,close,volume,benchmark_close,reference_close,eligible):
    r1 = log_return(close)
    rv20,rv60 = rolling(r1,20,"std"),rolling(r1,60,"std")
    volume_baseline = rolling(volume,20,shift=1)
    with np.errstate(divide="ignore",invalid="ignore"):
        basic = [r1,np.log(open_/lag(close,1)),np.log(high/low),np.log(volume/volume_baseline),
                 log_return(close,5),log_return(close,20),log_return(close,60),np.log(rv20),np.log(rv20/rv60)]
    janus,rss = janus_reference(log_return(reference_close),r1,eligible)
    stock = np.stack(basic+[reactivity(high,low,close,volume),deviation(close,benchmark_close),
        purification(basic[5],benchmark_close),janus,janus-lag(janus,5)],axis=-1)
    turbulence,coherence,absorption,shift = market_dependence(log_return(reference_close))
    spy_r = log_return(benchmark_close)
    with np.errstate(divide="ignore",invalid="ignore"):
        market = np.column_stack((spy_r,np.log(rolling(spy_r,20,"std")),turbulence,coherence,
            coherence-lag(coherence,5),absorption,shift,rss-lag(rss,5)))
    stock[~np.isfinite(stock)] = np.nan
    market[~np.isfinite(market)] = np.nan
    return stock,market,volume_baseline
