"""Assemble a assembled panel: a named subset of base, event, fundamentals and macro channels.

The assembled layout (features.STOCK_FEATURES_ASSEMBLED / MARKET_FEATURES_ASSEMBLED) drops two of the four "fast" channels
that define origin eligibility in v1 panels. To keep eligibility bit-identical, the assembled panel
carries ``origin_observed[T, K] = base.stock_obs[..., :4].all(-1)`` and every eligibility rule reads it
through ``data.fast_observed``.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .data import MarketPanel
from .features import (ELIGIBILITY_CHANNELS, MARKET_FEATURES_ASSEMBLED, STOCK_FAMILIES_ASSEMBLED, STOCK_FEATURES_ASSEMBLED,
                       ASSEMBLED_CHANNEL_SOURCES)


def _aligned(name, dates, symbols, arr):
    if not np.array_equal(np.asarray(arr["dates"]).astype("datetime64[D]"), dates):
        raise ValueError(f"{name}: dates do not match the base panel")
    if list(arr["symbols"]) != list(symbols):
        raise ValueError(f"{name}: symbols do not match the base panel")


def _source_tables(base, families, fundamentals):
    """name -> (values, observed) for every channel the assembled layout can draw from."""
    dates = np.asarray(base.dates).astype("datetime64[D]")
    tables = {}
    stock_names = list(base.metadata["stock_features"]); market_names = list(base.metadata["market_features"])
    for i, n in enumerate(stock_names):
        tables[("stock", n)] = (base.stock[..., i], base.stock_obs[..., i])
    for i, n in enumerate(market_names):
        tables[("market", n)] = (base.market[..., i], base.market_obs[..., i])
    if families is not None:
        _aligned("families", dates, base.symbols, families)
        for i, n in enumerate(families["event_names"]):
            tables[("events", str(n))] = (families["events"][..., i], families["events_obs"][..., i])
        for i, n in enumerate(families["macro_names"]):
            tables[("macro", str(n))] = (families["macro"][..., i], families["macro_obs"][..., i])
    if fundamentals is not None:
        _aligned("fundamentals", dates, base.symbols, fundamentals)
        for i, n in enumerate(fundamentals["fundamental_names"]):
            tables[("fundamentals", str(n))] = (fundamentals["fundamentals"][..., i], fundamentals["fundamentals_obs"][..., i])
    return tables


def assemble_panel(base: MarketPanel, families=None, fundamentals=None,
                      stock_names=STOCK_FEATURES_ASSEMBLED, market_names=MARKET_FEATURES_ASSEMBLED, families_index=STOCK_FAMILIES_ASSEMBLED) -> MarketPanel:
    """Gather the assembled channels in layout order; carry eligibility; write metadata and a new fingerprint."""
    tables = _source_tables(base, families, fundamentals)
    base_stock_names = list(base.metadata["stock_features"])
    for ch in ELIGIBILITY_CHANNELS:
        if ch not in base_stock_names:
            raise ValueError(f"base panel lacks eligibility channel {ch}")
    elig_idx = [base_stock_names.index(ch) for ch in ELIGIBILITY_CHANNELS]
    origin_observed = np.asarray(base.stock_obs[..., elig_idx]).all(axis=-1)

    def gather(names, kind):
        values, observed = [], []
        for n in names:
            key = ASSEMBLED_CHANNEL_SOURCES.get(n)
            if key is None or key not in tables:
                raise KeyError(f"{kind} channel {n!r} has no source (need {key})")
            v, o = tables[key]
            if kind == "stock" and v.ndim != 2 or kind == "market" and v.ndim != 1:
                raise ValueError(f"{n}: wrong rank for a {kind} channel")
            values.append(np.where(o, v, 0.0).astype(np.float32)); observed.append(np.asarray(o, dtype=bool))
        return np.stack(values, axis=-1), np.stack(observed, axis=-1)

    stock, stock_obs = gather(stock_names, "stock")
    market, market_obs = gather(market_names, "market")
    digest = hashlib.sha256()
    digest.update(json.dumps({"base": base.metadata["dataset_fingerprint"], "layout": "v2", "stock": list(stock_names),
                              "market": list(market_names)}, sort_keys=True).encode())
    for arr in (stock, stock_obs, market, market_obs, origin_observed):
        digest.update(hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest().encode())
    families_map = {name: [stock_names[i] for i in idx] for name, idx in families_index}
    meta = {**base.metadata,
            "feature_layout": "v2", "stock_features": list(stock_names), "market_features": list(market_names),
            "families": {"stock": {name: list(idx) for name, idx in families_index}, "stock_by_name": families_map},
            "eligibility": {"carried_from": base.metadata["dataset_fingerprint"], "channels": list(ELIGIBILITY_CHANNELS),
                            "rule": "universe & all eligibility channels observed on each of the 60 context sessions & spy_r1 observed"},
            "base_dataset_fingerprint": base.metadata["dataset_fingerprint"], "dataset_fingerprint": digest.hexdigest(),
            "normalizer": None}
    return MarketPanel(dates=base.dates, symbols=list(base.symbols), stock=stock, market=market, stock_obs=stock_obs, market_obs=market_obs,
                       open=base.open, close=base.close, volume=base.volume, volume_baseline20=base.volume_baseline20,
                       benchmark_open=base.benchmark_open, benchmark_close=base.benchmark_close, universe=base.universe,
                       metadata=meta, origin_observed=origin_observed)


def check_panel_coverage(panel: MarketPanel, min_coverage=0.9, first_year=2006, last_year=None, overrides=None) -> dict:
    """Yearly observed fraction per channel over the universe (stock) / all dates (market); raise below min_coverage.

    ``overrides`` maps channel name -> its own threshold (sparse-by-construction channels such as eps_sue,
    which needs four prior surprises, are allowed a lower floor)."""
    overrides = overrides or {}
    dates = np.asarray(panel.dates).astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    last_year = last_year or int(years.max())
    report = {"stock": {}, "market": {}, "failures": []}
    for i, n in enumerate(panel.metadata["stock_features"]):
        per_year = {}
        for y in range(first_year, last_year + 1):
            sel = years == y
            if not sel.any():
                continue
            u = panel.universe[sel]
            per_year[y] = float(panel.stock_obs[sel][..., i][u].mean()) if u.any() else float("nan")
        report["stock"][n] = per_year
        bad = [y for y, c in per_year.items() if np.isfinite(c) and c < overrides.get(n, min_coverage)]
        if bad:
            report["failures"].append({"channel": n, "years": bad, "threshold": overrides.get(n, min_coverage)})
    for i, n in enumerate(panel.metadata["market_features"]):
        per_year = {y: float(panel.market_obs[years == y][:, i].mean()) for y in range(first_year, last_year + 1) if (years == y).any()}
        report["market"][n] = per_year
        bad = [y for y, c in per_year.items() if c < overrides.get(n, min_coverage)]
        if bad:
            report["failures"].append({"channel": n, "years": bad, "threshold": overrides.get(n, min_coverage)})
    if report["failures"]:
        raise ValueError("assembled panel coverage below threshold: " + json.dumps(report["failures"]))
    return report
