from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation import (
    RidgeReadout,
    context_eligibility,
    date_block_interval,
    evaluate_arrays,
    extract_encoder_features,
    make_forward_labels,
    portfolio_costs,
    purged_training_dates,
    rank_ic,
)


def business_dates(start, end):
    days = np.arange(np.datetime64(start), np.datetime64(end), dtype="datetime64[D]")
    return days[np.is_busday(days)]


def test_forward_labels_are_next_open_and_fifth_close():
    dates = business_dates("2020-01-01", "2020-02-01")
    n = len(dates)
    opened = np.arange(n, dtype=float)[:, None] + 100
    closed = opened + 10
    panel = SimpleNamespace(dates=dates, open=opened, close=closed,
                            benchmark_open=np.full(n, 100.), benchmark_close=np.full(n, 102.))
    y, entry, end = make_forward_labels(panel, horizon=5)
    assert y[0, 0] == pytest.approx(115 / 101 - 1.02)
    assert entry[0] == dates[1]
    assert end[0] == dates[5]
    assert np.isnan(y[-5:]).all()
    assert np.isnat(end[-5:]).all()


def test_purge_checks_trading_rows_and_actual_label_ends():
    dates = business_dates("2018-10-01", "2019-02-01")
    ends = np.r_[dates[5:], np.full(5, np.datetime64("NaT"))]
    first = np.searchsorted(dates, np.datetime64("2019-01-01"))
    ends[0] = np.datetime64("2019-01-02")  # anomalously late label must be excluded
    mask = purged_training_dates(dates, ends, "2019-01-01", purge=20)
    assert not mask[0]
    assert np.flatnonzero(mask)[-1] == first - 21
    assert (ends[mask] < np.datetime64("2019-01-01")).all()


def test_standardization_fit_never_changes_when_predicting_extreme_future():
    x = np.asarray([[0., np.nan, 2.], [1., np.nan, 2.], [2., np.nan, 2.], [3., np.nan, 2.]])
    model = RidgeReadout(1).fit(x, [0, 1, 2, 3])
    before = model.mean_.copy(), model.scale_.copy(), model.coef_.copy()
    prediction = model.predict([[1e8, np.nan, 2.]])
    assert np.isfinite(prediction).all()
    assert model.mean_[0] == 1.5
    for a, b in zip(before, (model.mean_, model.scale_, model.coef_)):
        np.testing.assert_array_equal(a, b)


def test_constant_predictions_do_not_invent_ic_or_trades():
    assert rank_ic([1, 1, 1], [1, 2, 3]) is None
    assert rank_ic([1, 2], [1, 2]) is None
    assert rank_ic([1, 2, 2, 4], [2, 3, 3, 5]) == pytest.approx(1)
    x = np.zeros((10, 3))
    model = RidgeReadout().fit(x, np.arange(10))
    np.testing.assert_allclose(model.predict(x), 4.5)
    assert rank_ic(model.predict(x), np.arange(10)) is None


def _portfolio_rows(missing=False):
    rows = []
    for t in range(11):
        for k in range(10):
            rows.append({"origin_index": t, "origin_date": f"date{t}",
                         "entry_date": f"entry{t}", "exit_date": f"exit{t + 5}",
                         "symbol": str(k), "prediction": float(k),
                         "target": None if missing and t == 0 and k == 9 else k / 1000})
    return rows


def test_costs_charge_entry_and_exit_with_no_overlapping_cycles():
    result = portfolio_costs(_portfolio_rows(), horizon=5)
    assert [r["origin_index"] for r in result["trades"]] == [0, 5, 10]
    assert result["cycles"] == 3
    scenarios = {r["cost_bps_per_side"]: r for r in result["scenarios"]}
    assert scenarios[0]["total_turnover"] == 6
    assert scenarios[0]["mean_completed_cycle_net_return"] - scenarios[10]["mean_completed_cycle_net_return"] == pytest.approx(.002)


def test_missing_future_price_does_not_silently_change_portfolio():
    result = portfolio_costs(_portfolio_rows(missing=True))
    assert result["missing_selected_return_cycles"] == 1
    assert not result["portfolio_return_coverage_complete"]
    assert all(r["annualized_net_return"] is None and r["net_sharpe"] is None for r in result["scenarios"])


def test_date_block_ci_is_deterministic_and_handles_no_variation():
    assert date_block_interval([None] * 100) is None
    assert date_block_interval([.2] * 80, samples=50) == pytest.approx([.2, .2])
    x = np.sin(np.arange(100))
    assert date_block_interval(x, samples=50, seed=7) == date_block_interval(x, samples=50, seed=7)


def test_expanding_probe_keeps_held_out_targets_out_of_earlier_predictions():
    dates = business_dates("2016-01-01", "2021-01-01")
    rng = np.random.default_rng(9)
    x = rng.normal(size=(len(dates), 6, 2))
    y = x[..., 0] * .01 + rng.normal(0, .001, x.shape[:2])
    ends = np.r_[dates[5:], np.full(5, np.datetime64("NaT"))]
    y[-5:] = np.nan
    first = evaluate_arrays({"x": x}, y, dates, label_end_dates=ends,
                            years=(2019,), bootstrap_samples=0)
    y2 = y.copy()
    y2[dates >= np.datetime64("2019-01-01")] *= -100
    second = evaluate_arrays({"x": x}, y2, dates, label_end_dates=ends,
                             years=(2019,), bootstrap_samples=0)
    np.testing.assert_allclose([r["prediction"] for r in first["predictions"]],
                               [r["prediction"] for r in second["predictions"]])
    assert all(r["origin_date"].startswith("2019") for r in first["predictions"])
    fold = first["report"]["folds"][0]
    assert fold["latest_training_label_end"] < "2019-01-01"
    assert first["report"]["metrics"]["x"]["development"]["mean_daily_rank_ic"] > .8
    with pytest.raises(ValueError, match="20 trading rows"):
        evaluate_arrays({"x": x}, y, dates, label_end_dates=ends, purge=5)


def test_missing_labels_preserved_in_origin_prediction_rows():
    dates = business_dates("2018-01-01", "2020-01-01")
    n = len(dates)
    y = np.ones((n, 5))
    ends = np.r_[dates[5:], np.full(5, np.datetime64("NaT"))]
    t = np.searchsorted(dates, np.datetime64("2019-06-03"))
    y[t, 4] = np.nan
    result = evaluate_arrays({}, y, dates, label_end_dates=ends,
                             fixed_predictions={"fixed": np.tile(np.arange(5), (n, 1))},
                             years=(2019,), bootstrap_samples=0)
    target_row = [r for r in result["predictions"] if r["origin_index"] == t and r["symbol"] == "4"][0]
    assert target_row["target"] is None


def test_encoder_context_eligibility_does_not_depend_on_future_quotes(tmp_path):
    torch = pytest.importorskip("torch")
    n, k = 65, 7
    dates = business_dates("2019-01-01", "2020-01-01")[:n]
    rng = np.random.default_rng(12)
    panel = SimpleNamespace(dates=dates, symbols=[f"S{i}" for i in range(k)],
                            stock=rng.normal(size=(n, k, 14)).astype("float32"),
                            market=rng.normal(size=(n, 8)).astype("float32"),
                            stock_obs=np.ones((n, k, 14), bool), market_obs=np.ones((n, 8), bool),
                            universe=np.ones((n, k), bool),
                            open=np.ones((n, k)), close=np.ones((n, k)))
    class Dummy(torch.nn.Module):
        def encode(self, stock, market, stock_obs, market_obs, valid):
            return {"stock_embeddings": stock.mean(2)[..., :3],
                    "market_embedding": market.mean(1)[..., :2], "valid_stock_mask": valid}
    eligible = context_eligibility(panel)
    a, av, peers, _ = extract_encoder_features(panel, Dummy(), max_stocks=3, batch_size=2)
    panel.open[-1, 0] = np.nan
    panel.close[-1, 0] = np.nan
    b, bv, peers2, _ = extract_encoder_features(panel, Dummy(), max_stocks=3, batch_size=4,
                                               output_path=tmp_path / "features.f32")
    assert isinstance(b, np.memmap)
    assert (tmp_path / "features.f32").stat().st_size == n * k * 5 * 4
    np.testing.assert_equal(a, b)
    np.testing.assert_array_equal(av, bv)
    np.testing.assert_array_equal(eligible, context_eligibility(panel))
    assert peers == peers2
    assert a.shape == (n, k, 5)
    assert all(len(p["symbols"]) <= 3 for p in peers)


def _dense_ridge_statistics(x, y, row_mask=None, **unused):
    """Independent reference preserving the former all-in-memory equations."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if row_mask is not None:
        mask &= row_mask
    x, y = x[mask], y[mask]
    finite = np.isfinite(x)
    counts = finite.sum(0)
    mean = np.divide(np.where(finite, x, 0).sum(0), counts,
                     out=np.zeros(x.shape[1]), where=counts > 0)
    centered = np.where(finite, x - mean, 0)
    var = np.mean(centered ** 2, axis=0)
    scale = np.where(var > 1e-12, np.sqrt(var), 1.)
    z = centered / scale
    return {"mean": mean, "scale": scale, "intercept": float(y.mean()),
            "gram": z.T @ z, "cross": z.T @ (y - y.mean()), "examples": len(y)}


def test_streamed_ridge_matches_dense_equations_with_missing_and_constant_inputs(tmp_path):
    rng = np.random.default_rng(77)
    x = rng.normal(size=(101, 7, 11)).astype("float32")
    x[..., 1] = np.nan
    x[..., 2] = 3.0
    x[::3, 0, 4] = np.nan
    y = rng.normal(size=x.shape[:2])
    y[::5, 2] = np.nan
    mask = rng.random(y.shape) > .15
    disk = np.memmap(tmp_path / "ridge.f32", mode="w+", dtype="float32", shape=x.shape)
    disk[:] = x
    streamed = RidgeReadout.statistics(disk, y, mask, chunk_size=31)
    dense = _dense_ridge_statistics(x, y, mask)
    for key in ("mean", "scale", "intercept", "gram", "cross", "examples"):
        np.testing.assert_allclose(streamed[key], dense[key], rtol=1e-11, atol=1e-11)
    for alpha in (.1, 1., 10., 100.):
        actual = RidgeReadout(alpha).fit_statistics(streamed)
        expected = RidgeReadout(alpha).fit_statistics(dense)
        np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(actual.predict_masked(disk, mask, chunk_size=23)[mask],
                                   expected.predict(x[mask]), rtol=1e-10, atol=1e-11)


def test_chronological_metrics_match_dense_reference_and_reuse_statistics(monkeypatch):
    dates = business_dates("2016-01-01", "2021-01-01")
    rng = np.random.default_rng(26)
    x = rng.normal(size=(len(dates), 6, 7)).astype("float32")
    y = .01 * x[..., 0] + rng.normal(0, .001, x.shape[:2])
    x[::2, 1, 4] = np.nan
    x[..., 6] = 1
    ends = np.r_[dates[5:], np.full(5, np.datetime64("NaT", "D"))]
    calls = []
    original = RidgeReadout.statistics
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(RidgeReadout, "statistics", staticmethod(counted))
    streamed = evaluate_arrays({"x": x}, y, dates, label_end_dates=ends,
                               years=(2019, 2020), bootstrap_samples=0)
    assert len(calls) == 3  # 2018 inner, 2019 outer/next-inner, 2020 outer
    monkeypatch.setattr(RidgeReadout, "statistics", staticmethod(_dense_ridge_statistics))
    dense = evaluate_arrays({"x": x}, y, dates, label_end_dates=ends,
                            years=(2019, 2020), bootstrap_samples=0)
    assert [f["alpha"] for f in streamed["report"]["folds"]] == [f["alpha"] for f in dense["report"]["folds"]]
    np.testing.assert_allclose([p["prediction"] for p in streamed["predictions"]],
                               [p["prediction"] for p in dense["predictions"]], rtol=1e-9, atol=1e-12)
    assert streamed["report"]["metrics"] == dense["report"]["metrics"]
