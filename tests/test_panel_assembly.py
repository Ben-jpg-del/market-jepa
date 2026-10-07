import json
import numpy as np
import pytest

from src.data import MarketPanel, fast_observed, load_panel, save_panel
from src.evaluation import context_eligibility
from src.features import (ELIGIBILITY_CHANNELS, EVENT_FEATURES, FUNDAMENTAL_FEATURES, MACRO_FEATURES, MARKET_FEATURES,
                                  MARKET_FEATURES_ASSEMBLED, STOCK_FAMILIES_ASSEMBLED, STOCK_FEATURES, STOCK_FEATURES_ASSEMBLED)
from src.panel_assembly import assemble_panel, check_panel_coverage


def base_panel(T=400, K=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.arange(np.datetime64("2015-01-01"), np.datetime64("2015-01-01") + np.timedelta64(T, "D")).astype("datetime64[D]")
    stock = rng.normal(size=(T, K, len(STOCK_FEATURES))).astype(np.float32)
    stock_obs = rng.random((T, K, len(STOCK_FEATURES))) > 0.05
    stock_obs[:, 0, 2] = False  # log_high_low missing for stock 0: stock 0 is never eligible
    market = rng.normal(size=(T, len(MARKET_FEATURES))).astype(np.float32)
    market_obs = np.ones((T, len(MARKET_FEATURES)), bool)
    close = np.exp(np.cumsum(rng.normal(0, .01, (T, K)), 0)) * 50
    return MarketPanel(dates=dates, symbols=[f"S{i}" for i in range(K)], stock=stock, market=market, stock_obs=stock_obs,
                       market_obs=market_obs, open=close, close=close, volume=np.ones((T, K)), volume_baseline20=np.ones((T, K)),
                       benchmark_open=close[:, 0], benchmark_close=close[:, 0], universe=np.ones((T, K), bool),
                       metadata={"dataset_fingerprint": "base-fp", "stock_features": list(STOCK_FEATURES),
                                 "market_features": list(MARKET_FEATURES), "feature_notes": {}, "normalizer": None})


def family_arrays(panel, seed=1):
    rng = np.random.default_rng(seed)
    T, K = panel.universe.shape
    fam = {"dates": panel.dates, "symbols": np.array(panel.symbols),
           "events": rng.normal(size=(T, K, len(EVENT_FEATURES))).astype(np.float32), "events_obs": rng.random((T, K, len(EVENT_FEATURES))) > 0.1,
           "macro": rng.normal(size=(T, len(MACRO_FEATURES))).astype(np.float32), "macro_obs": np.ones((T, len(MACRO_FEATURES)), bool),
           "event_names": np.array(EVENT_FEATURES), "macro_names": np.array(MACRO_FEATURES)}
    fund = {"dates": panel.dates, "symbols": np.array(panel.symbols),
            "fundamentals": rng.normal(size=(T, K, len(FUNDAMENTAL_FEATURES))).astype(np.float32),
            "fundamentals_obs": rng.random((T, K, len(FUNDAMENTAL_FEATURES))) > 0.1, "fundamental_names": np.array(FUNDAMENTAL_FEATURES)}
    return fam, fund


def test_assembly_order_sources_and_carried_eligibility(tmp_path):
    base = base_panel(); fam, fund = family_arrays(base)
    assembled = assemble_panel(base, fam, fund)
    assert assembled.metadata["stock_features"] == list(STOCK_FEATURES_ASSEMBLED) and assembled.metadata["market_features"] == list(MARKET_FEATURES_ASSEMBLED)
    assert assembled.stock.shape[-1] == 15 and assembled.market.shape[-1] == 11 and assembled.metadata["feature_layout"] == "v2"
    # channel values come from the right source
    i_r1 = STOCK_FEATURES.index("r1"); assert np.array_equal(assembled.stock[..., 0], np.where(base.stock_obs[..., i_r1], base.stock[..., i_r1], 0))
    i_sue = list(EVENT_FEATURES).index("eps_sue"); j = STOCK_FEATURES_ASSEMBLED.index("eps_sue")
    assert np.array_equal(assembled.stock_obs[..., j], fam["events_obs"][..., i_sue])
    i_vix = list(MACRO_FEATURES).index("vix_log"); jm = MARKET_FEATURES_ASSEMBLED.index("vix_log")
    assert np.array_equal(assembled.market[:, jm], fam["macro"][:, i_vix])
    # eligibility identical although log_high_low / relative_volume are not assembled channels
    assert assembled.origin_observed is not None and np.array_equal(fast_observed(assembled), base.stock_obs[..., :4].all(-1))
    assert np.array_equal(context_eligibility(assembled, 60), context_eligibility(base, 60))
    assert not context_eligibility(assembled, 60)[:, 0].any()
    # survives a save/load round trip
    save_panel(assembled, tmp_path / "assembled.npz"); back = load_panel(tmp_path / "assembled.npz")
    assert back.origin_observed is not None and np.array_equal(back.origin_observed, assembled.origin_observed)
    assert back.metadata["families"]["stock"]["reversal"] == list(STOCK_FAMILIES_ASSEMBLED[0][1])
    assert back.metadata["eligibility"]["channels"] == list(ELIGIBILITY_CHANNELS)


def test_fingerprint_depends_on_every_source_array():
    base = base_panel(); fam, fund = family_arrays(base)
    fp = assemble_panel(base, fam, fund).metadata["dataset_fingerprint"]
    fund2 = dict(fund); fund2["fundamentals"] = fund["fundamentals"].copy(); fund2["fundamentals"][5, 2, 0] += 1.0
    assert assemble_panel(base, fam, fund2).metadata["dataset_fingerprint"] != fp
    fam2 = dict(fam); fam2["macro_obs"] = fam["macro_obs"].copy(); fam2["macro_obs"][100:, list(MACRO_FEATURES).index("vix_log")] = False
    assert assemble_panel(base, fam2, fund).metadata["dataset_fingerprint"] != fp
    fam3 = dict(fam); fam3["macro_obs"] = fam["macro_obs"].copy(); fam3["macro_obs"][100:, list(MACRO_FEATURES).index("oil_20")] = False
    assert assemble_panel(base, fam3, fund).metadata["dataset_fingerprint"] == fp  # unused source channels do not matter


def test_coverage_check_raises_on_a_gap_and_reports_per_year():
    base = base_panel(T=800); fam, fund = family_arrays(base)
    ok = check_panel_coverage(assemble_panel(base, fam, fund), min_coverage=0.5, first_year=2015)
    assert set(ok["stock"]) == set(STOCK_FEATURES_ASSEMBLED) and set(ok["market"]) == set(MARKET_FEATURES_ASSEMBLED) and not ok["failures"]
    fam2 = dict(fam); fam2["macro_obs"] = fam["macro_obs"].copy(); fam2["macro_obs"][400:, :] = False
    with pytest.raises(ValueError, match="coverage below threshold"):
        check_panel_coverage(assemble_panel(base, fam2, fund), min_coverage=0.9, first_year=2015)


def test_misaligned_family_file_is_rejected():
    base = base_panel(); fam, fund = family_arrays(base)
    fam2 = dict(fam); fam2["symbols"] = np.array(list(reversed(base.symbols)))
    with pytest.raises(ValueError, match="symbols"):
        assemble_panel(base, fam2, fund)
