"""Walk-forward sampler: context window + future windows per horizon + auxiliary rank targets, all inside a cutoff.

WalkForwardSampler(panel, horizons=(5, 20), ..., end=cutoff) samples origins t with t + max(h) <= cutoff
index (the base class rule), and for every origin returns, besides the usual context batch:
  future_stock      [B, K, H, C, Cs]   the window ending at t + h for each horizon (same length as the context)
  future_stock_obs  [B, K, H, C, Cs]
  future_market     [B, H, C, Cm], future_market_obs
  future_valid      [B, K, H]          stock eligible at t and SPY observed at t + h
  future_end_index  [B, H]             t + h (for leakage tests)
  aux_target        [B, K, H]          per-date normal score of the h-day SPY-relative forward return
  aux_valid         [B, K, H]          label exists and its end date is within the cutoff
Nothing after the cutoff is read: future windows end at t + h <= cutoff and labels end at or before it.
"""
from __future__ import annotations

import numpy as np

from .data import DateBatchSampler
from .evaluation import make_forward_labels


def normal_scores_by_date(values, mask, min_n=5):
    """Per-date normal scores (rank transform) of values[T, K] over mask[T, K]; NaN elsewhere."""
    from scipy.stats import norm
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for t in range(values.shape[0]):
        m = mask[t] & np.isfinite(values[t])
        n = int(m.sum())
        if n < min_n:
            continue
        r = np.empty(n); r[np.argsort(values[t][m], kind="stable")] = np.arange(n)
        out[t, m] = norm.ppf((r + 0.5) / n)
    return out


def forward_realised_vol(panel, horizon):
    """[T, K] std of the close-to-close log returns over sessions t+1..t+h (NaN unless every close is observed),
    and the end date of that window per origin."""
    close = np.asarray(panel.close, dtype=np.float64)
    dates = np.asarray(panel.dates, dtype="datetime64[D]")
    T = len(dates)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.full_like(close, np.nan); logret[1:] = np.log(close[1:] / close[:-1])
    logret[~np.isfinite(logret)] = np.nan
    vol = np.full(close.shape, np.nan, np.float32); ends = np.full(T, np.datetime64("NaT", "D"), dtype="datetime64[D]")
    if T > horizon:
        windows = np.lib.stride_tricks.sliding_window_view(logret[1:], horizon, axis=0)   # [T-h, K, h]: rows t -> returns t+1..t+h
        vol[:T - horizon] = np.nanstd(windows, axis=-1, ddof=1).astype(np.float32) if horizon > 1 else np.abs(windows[..., 0])
        vol[:T - horizon][np.isnan(windows).any(-1)] = np.nan
        ends[:T - horizon] = dates[horizon:]
    return vol, ends


def _winsorise_by_date(values, ok, lo=0.01, hi=0.99):
    out = np.where(ok, values, np.nan).astype(np.float32)
    for t in range(out.shape[0]):
        row = out[t]; good = np.isfinite(row)
        if good.sum() >= 20:
            a, b = np.nanquantile(row, [lo, hi]); out[t] = np.clip(row, a, b)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def aux_targets(panel, horizons, cutoff, eligibility, kind="return_rank"):
    """Targets [T, K, H], validity [T, K, H] and raw SPY-relative returns [T, K, H] (per-date winsorised 1%/99%,
    for the sharpe loss); rows whose label window ends after `cutoff` are invalid (never read).
    kind="return_rank": per-date normal score of the relative return; "vol_rank": of the forward realised volatility."""
    if kind not in {"return_rank", "vol_rank"}:
        raise ValueError("aux target kind must be return_rank or vol_rank")
    cutoff = np.datetime64(cutoff, "D")
    T, K = eligibility.shape
    target = np.zeros((T, K, len(horizons)), np.float32)
    valid = np.zeros((T, K, len(horizons)), bool)
    raw = np.zeros((T, K, len(horizons)), np.float32)
    for hi, h in enumerate(horizons):
        labels, _, ends = make_forward_labels(panel, h)
        ok = eligibility & np.isfinite(labels) & (~np.isnat(ends) & (ends <= cutoff))[:, None]
        raw[..., hi] = _winsorise_by_date(labels, ok)
        if kind == "vol_rank":
            vol, vol_ends = forward_realised_vol(panel, h)
            ok = eligibility & np.isfinite(vol) & (~np.isnat(vol_ends) & (vol_ends <= cutoff))[:, None]
            source = vol
        else:
            source = labels
        scores = normal_scores_by_date(np.where(ok, source, np.nan), ok)
        good = np.isfinite(scores)
        target[..., hi] = np.where(good, scores, 0.0)
        valid[..., hi] = good
    return target, valid, raw


class WalkForwardSampler(DateBatchSampler):
    def __init__(self, panel, horizons=(5, 20), batch_dates=16, stocks_per_date=32, context=60, seed=42,
                 start="2006-01-01", end="2018-12-31", min_stocks=2, aux_target="return_rank", include_future=True):
        self.horizons = tuple(int(h) for h in horizons)
        self.aux_kind = aux_target; self.include_future = bool(include_future)
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        super().__init__(panel, batch_dates=batch_dates, stocks_per_date=stocks_per_date, context=context,
                         horizon=max(self.horizons), seed=seed, start=start, end=end, feature_set="book", min_stocks=min_stocks)
        self.aux_target, self.aux_valid, self.aux_raw = aux_targets(panel, self.horizons, end, self.eligibility, kind=aux_target)

    def state_dict(self):
        return {**super().state_dict(), "horizons": list(self.horizons)}

    def load_state_dict(self, state):
        if tuple(state.get("horizons", ())) != self.horizons:
            raise ValueError("Sampler configuration changed: horizons")
        super().load_state_dict({k: v for k, v in state.items() if k != "horizons"})

    def sample(self):
        p = self.panel; B = self.batch_dates; K = self.stocks_per_date; C = self.context; H = len(self.horizons)
        origins = self.rng.choice(self.origins, B, replace=True)
        Cs, Cm = p.stock.shape[-1], p.market.shape[-1]
        batch = {"stock": np.zeros((B, K, C, Cs), np.float32), "stock_obs": np.zeros((B, K, C, Cs), bool),
                 "market": np.zeros((B, C, Cm), np.float32), "market_obs": np.zeros((B, C, Cm), bool),
                 "eligibility": np.zeros((B, K), bool),
                 "future_stock": np.zeros((B, K, H, C, Cs), np.float32), "future_stock_obs": np.zeros((B, K, H, C, Cs), bool),
                 "future_market": np.zeros((B, H, C, Cm), np.float32), "future_market_obs": np.zeros((B, H, C, Cm), bool),
                 "future_valid": np.zeros((B, K, H), bool), "future_end_index": np.zeros((B, H), np.int64),
                 "aux_target": np.zeros((B, K, H), np.float32), "aux_valid": np.zeros((B, K, H), bool),
                 "aux_raw": np.zeros((B, K, H), np.float32),
                 "origins": origins, "symbol_indices": np.full((B, K), -1, np.int64)}
        for b, t in enumerate(origins):
            candidates = np.flatnonzero(self.eligibility[t])
            chosen = self.rng.choice(candidates, min(K, len(candidates)), replace=False)
            n = len(chosen); window = slice(t - C + 1, t + 1)
            batch["symbol_indices"][b, :n] = chosen
            batch["stock"][b, :n] = p.stock[window][:, chosen].transpose(1, 0, 2)
            batch["stock_obs"][b, :n] = p.stock_obs[window][:, chosen].transpose(1, 0, 2)
            batch["market"][b] = p.market[window]; batch["market_obs"][b] = p.market_obs[window]
            batch["eligibility"][b, :n] = True
            for hi, h in enumerate(self.horizons if self.include_future else ()):
                fw = slice(t + h - C + 1, t + h + 1)
                batch["future_end_index"][b, hi] = t + h
                batch["future_stock"][b, :n, hi] = p.stock[fw][:, chosen].transpose(1, 0, 2)
                batch["future_stock_obs"][b, :n, hi] = p.stock_obs[fw][:, chosen].transpose(1, 0, 2)
                batch["future_market"][b, hi] = p.market[fw]; batch["future_market_obs"][b, hi] = p.market_obs[fw]
                batch["future_valid"][b, :n, hi] = bool(p.market_obs[t + h, 0])
            batch["aux_target"][b, :n] = self.aux_target[t, chosen]
            batch["aux_valid"][b, :n] = self.aux_valid[t, chosen]
            batch["aux_raw"][b, :n] = self.aux_raw[t, chosen]
        return batch

    __call__ = sample
