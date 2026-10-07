"""Integration checks for matched variants and resumable research runs."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import MarketPanel, fit_normalizer, normalize_panel
from src.model import PilotConfig
from src.runner import (VARIANTS, PILOT_VARIANTS, VariantSampler, budget_from_profiles,
                                evaluate_variant, profile_variant, run_variant)
from src.training import load_snapshot


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def panel():
    rng = np.random.default_rng(19)
    t, k = 180, 4
    dates = pd.bdate_range("2018-01-01", periods=t).to_numpy(dtype="datetime64[D]")
    stock = rng.normal(size=(t, k, 14)).astype(np.float32)
    market = rng.normal(size=(t, 8)).astype(np.float32)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (t, k)), axis=0))
    volume = np.ones((t, k)) * 1000
    p = MarketPanel(dates, [f"S{i}" for i in range(k)], stock, market,
                    np.ones_like(stock, bool), np.ones_like(market, bool),
                    close * 0.999, close, volume, volume.copy(),
                    np.ones(t)*100, np.ones(t)*101, np.ones((t, k), bool),
                    {"dataset_fingerprint": "synthetic-runner-fixture", "normalizer": None,
                     "synthetic": True, "point_in_time_verified": False})
    return normalize_panel(p, fit_normalizer(p))


def model_config():
    return PilotConfig(width=16, predictor_width=8, temporal_layers=1,
                       predictor_layers=1, sigreg_directions=8, sigreg_points=9)


def test_historical_control_hides_only_selected_latest_blocks_and_keeps_teacher_complete():
    sampler = VariantSampler(panel(), "A", batch_dates=2, stocks=4)
    b = sampler()
    assert b["future_stock"].shape == (2, 4, 60, 14)
    assert b["prediction_valid"].sum(axis=1).tolist() == [1, 1]
    for row in range(2):
        selected = b["prediction_valid"][row]
        assert not b["stock_obs"][row, selected, -10:].any()
        assert np.all(b["stock_obs"][row, selected, :-10, :9])
        assert np.all(b["stock_obs"][row, ~selected, :, :9])
        assert np.all(b["future_stock_obs"][row, :, :, :4])
        assert not b["future_stock_obs"][row, :, :, 4:].any()
    assert b["future_market_obs"][..., 0].all()
    assert not b["future_market_obs"][..., 1:].any()
    assert np.array_equal(b["target_valid"], b["eligibility"])


def test_variant_masks_and_one_day_padding_match_experiment_definitions():
    p = panel()
    for name in VARIANTS:
        b = VariantSampler(p, name, batch_dates=1, stocks=3)()
        if name in ("A", "B"):
            assert not b["stock_obs"][..., 9:].any()
            assert not b["market_obs"][..., 2:].any()
        elif name in ("F", "F1", "T2"):
            for row in range(len(b["stock"])):
                chosen = b["prediction_valid"][row]
                assert b["stock_obs"][row, ~chosen, :, 9:].all()
                assert b["stock_obs"][row, chosen, :-10, 9:].all()
                assert not b["stock_obs"][row, chosen, -10:].any()
            assert b["market_obs"][..., 2:].all()
        else:
            assert b["stock_obs"][..., 9:].all()
            assert b["market_obs"][..., 2:].all()
        if name == "E":
            assert not b["future_stock_obs"][..., :4, :].any()
            assert b["future_stock_obs"][..., 4, :4].all()
            assert not b["future_market_obs"][..., :4, :].any()


@pytest.mark.parametrize("variant", ["A", "F"])
def test_historical_sampler_restores_both_data_and_mask_rng(variant):
    sampler = VariantSampler(panel(), variant, batch_dates=2, stocks=4)
    sampler()
    state = copy.deepcopy(sampler.state_dict())
    expected = sampler()
    sampler.load_state_dict(state)
    actual = sampler()
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])


def test_run_screen_then_continue_matches_uninterrupted_and_exports_provenance(tmp_path):
    p = panel()
    kwargs = dict(seed=42, screen_steps=1, total_steps=3, batch_dates=1, stocks=3,
                  device="cpu", model_config=model_config(), accumulation_steps=1, log_every=0)
    full = run_variant(p, "C", tmp_path/"full", stop_at=3, **kwargs)
    partial = run_variant(p, "C", tmp_path/"split", **kwargs)
    assert partial["optimizer_step"] == 1
    resumed = run_variant(p, "C", tmp_path/"split", resume=True, stop_at=3, **kwargs)
    one, two = load_snapshot(full["encoder_export"]), load_snapshot(resumed["encoder_export"])
    for key, value in one.online.state_dict().items():
        torch.testing.assert_close(value, two.online.state_dict()[key], rtol=0, atol=0)
    assert two.snapshot_metadata["optimizer_step"] == 3
    assert two.snapshot_metadata["normalizer"] == p.metadata["normalizer"]
    assert two.snapshot_metadata["code"]["sha256"]
    assert two.snapshot_metadata["sampler_config"]["stocks"] == 3
    assert (tmp_path/"split"/"training_metrics.json").exists()
    with pytest.raises(FileExistsError):
        run_variant(p, "C", tmp_path/"split", **kwargs)


def test_evaluation_enforces_dataset_and_feature_matching(tmp_path, monkeypatch):
    p = panel()
    run_variant(p, "B", tmp_path, total_steps=1, screen_steps=1, batch_dates=1,
                stocks=3, device="cpu", model_config=model_config(), accumulation_steps=1, log_every=0)
    called = {}
    def fake_evaluate(data, encoder, **kwargs):
        called.update(kwargs)
        return {"metrics": {}, "tested": True}
    monkeypatch.setattr("src.runner.evaluate_encoder", fake_evaluate)
    report = evaluate_variant(p, tmp_path, device="cpu")
    assert report["variant"] == "B"
    assert called["feature_set"] == "basic"
    assert called["normalized"] is True
    assert called["years"] == (2019, 2020, 2021)
    bad = copy.copy(p)
    bad.metadata = {**p.metadata, "dataset_fingerprint": "other"}
    with pytest.raises(ValueError, match="differs"):
        evaluate_variant(bad, tmp_path, device="cpu")


def test_budget_accounts_for_all_screens_continuation_and_replication():
    profiles = {name: {"mean_seconds_per_update": 1.0} for name in PILOT_VARIANTS}
    full = budget_from_profiles(profiles)
    assert full["screen_steps"] == 500 and full["total_steps"] == 2500
    assert full["total_optimizer_updates"] == 7000
    assert full["estimated_training_hours"] == 7000/3600
    reduced = budget_from_profiles(profiles, budget_hours=1)
    assert reduced["screen_steps"] < 500 and reduced["total_steps"] < 2500
    assert reduced["estimated_training_hours"] <= 0.8
    assert reduced["seed_43_total_steps"] == reduced["total_steps"]
    assert reduced["selected_variant_continuation_steps"] == reduced["total_steps"] - reduced["screen_steps"]
    with pytest.raises(ValueError, match="all five"):
        budget_from_profiles({"C": profiles["C"]})
    with pytest.raises(ValueError, match="all five"):
        budget_from_profiles({**profiles, "F": {"mean_seconds_per_update": 1.0}})


def test_book_historical_control_keeps_book_context_and_ignores_all_future_buffers(monkeypatch):
    p = panel()
    sampler = VariantSampler(p, "F", batch_dates=2, stocks=4, seed=42)
    source = sampler.base.sample()
    unchanged = copy.deepcopy(source)
    # Poison every prospective target supplied by the generic date sampler.
    # Historical F must build targets exclusively from the recorded past.
    source["future_stock"][:] = np.nan
    source["future_stock_obs"][:] = False
    source["future_market"][:] = np.nan
    source["future_market_obs"][:] = False
    source["target_valid"][:] = False
    monkeypatch.setattr(sampler.base, "sample", lambda: source)
    b = sampler.sample()
    assert b["future_stock"].shape == unchanged["stock"].shape
    np.testing.assert_array_equal(b["future_stock"][..., :4], unchanged["stock"][..., :4])
    np.testing.assert_array_equal(b["future_stock_obs"][..., :4], unchanged["stock_obs"][..., :4])
    assert (b["future_stock"][..., 4:] == 0).all()
    assert not b["future_stock_obs"][..., 4:].any()
    np.testing.assert_array_equal(b["future_market"][..., 0], unchanged["market"][..., 0])
    assert (b["future_market"][..., 1:] == 0).all()
    assert not b["future_market_obs"][..., 1:].any()
    np.testing.assert_array_equal(b["target_valid"], unchanged["eligibility"])
    np.testing.assert_array_equal(b["market"], unchanged["market"])
    assert b["market_obs"][..., 2:].all()
    for row in range(2):
        chosen = b["prediction_valid"][row]
        assert chosen.sum() == 1
        np.testing.assert_array_equal(b["stock"][row, ~chosen], unchanged["stock"][row, ~chosen])
        np.testing.assert_array_equal(b["stock"][row, chosen, :-10], unchanged["stock"][row, chosen, :-10])
        assert b["stock_obs"][row, chosen, :-10, 9:].all()
        assert (b["stock"][row, chosen, -10:] == 0).all()
        assert not b["stock_obs"][row, chosen, -10:].any()


def test_profile_variant_is_explicit_synthetic_timing_only():
    result = profile_variant(panel(), "C", batch_dates=1, stocks=3, device="cpu",
                             profile_steps=1, warmup_steps=1, model_config=model_config(), accumulation_steps=1)
    assert result["variant"] == "C" and result["synthetic"]
    assert result["mean_seconds_per_update"] > 0
