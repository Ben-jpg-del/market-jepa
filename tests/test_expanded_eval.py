import csv
import gzip
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import MarketPanel, fit_normalizer, normalize_panel
from src.evaluation import context_eligibility, extract_encoder_features, raw_feature_baseline
from src.expanded_eval import (cached_encoder_features, cached_raw_features,
    evaluate_expanded_run, peer_groups, shared_origin_mask, _normalized_view, _basic_raw_view,
    _save_evaluation)
from src.model import FutureJEPA, PilotConfig
from src.runner import VARIANTS, source_fingerprint
from src.training import export_encoder, seed_everything


def panel_fixture(start="2016-01-01", end="2020-01-01", stocks=5):
    days = np.arange(np.datetime64(start), np.datetime64(end), dtype="datetime64[D]")
    dates = days[np.is_busday(days)]
    rng = np.random.default_rng(817)
    stock = rng.normal(0, .1, (len(dates), stocks, 14)).astype("float32")
    market = rng.normal(0, .1, (len(dates), 8)).astype("float32")
    close = 100 * np.exp(np.cumsum(stock[..., 0] * .1, axis=0))
    opened = close * np.exp(stock[..., 1] * .01)
    benchmark = 100 * np.exp(np.cumsum(market[:, 0] * .05))
    return MarketPanel(dates=dates, symbols=[f"S{k}" for k in range(stocks)], stock=stock, market=market,
        stock_obs=np.ones_like(stock, bool), market_obs=np.ones_like(market, bool),
        open=opened, close=close, volume=np.full(close.shape, 1e6), volume_baseline20=np.full(close.shape, 1e6),
        benchmark_open=benchmark * 1.001, benchmark_close=benchmark,
        universe=np.ones(close.shape, bool), metadata={"dataset_fingerprint": "expanded-raw", "normalizer": None})


def narrower(panel, indices=(2, 0, 1)):
    from dataclasses import replace
    names = ("stock", "stock_obs", "open", "close", "volume", "volume_baseline20", "universe")
    return replace(panel, **{name: getattr(panel, name)[:, indices].copy() for name in names},
                   symbols=[panel.symbols[k] for k in indices],
                   metadata={"dataset_fingerprint": "original-raw", "normalizer": None})


def test_common_mask_matches_dates_symbols_and_origin_inputs_only():
    wide = panel_fixture()
    narrow = narrower(wide)
    t = 700
    narrow.universe[t, 0] = False  # S2, which is index 2 in expanded panel
    masks = shared_origin_mask(wide, narrow)
    assert masks["common"][t].tolist() == [True, True, False, False, False]
    assert masks["expanded"][t].all()
    before = masks["common"].copy()
    wide.close[t + 1, :] = np.nan
    wide.open[t + 1, :] = np.nan
    np.testing.assert_array_equal(shared_origin_mask(wide, narrow)["common"], before)
    assert np.all(masks["common"] <= masks["expanded"])


class SmallEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(width=3, context_days=60)
        self.calls = 0

    def encode(self, stock, market, stock_obs, market_obs, valid):
        self.calls += 1
        return {"stock_embeddings": stock.mean(2)[..., :3],
                "market_embedding": market.mean(1)[..., :3], "valid_stock_mask": valid}


def test_persisted_encoder_cursor_resumes_without_repeating_committed_batches(tmp_path):
    panel = panel_fixture("2019-01-01", "2019-04-15", stocks=5)
    eligible = context_eligibility(panel)
    groups, _ = peer_groups(panel, eligible, max_stocks=3)
    model = SmallEncoder()
    class Deadline(Exception):
        pass
    def stop(info):
        if info.get("completed_groups", 0) >= 2:
            raise Deadline()
    with pytest.raises(Deadline):
        cached_encoder_features(panel, model, eligible, groups, tmp_path / "cache", "same",
                                batch_size=2, max_stocks=3, checkpoint_batches=1, on_progress=stop)
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["cursor"] == 2
    before = model.calls
    actual, timing = cached_encoder_features(panel, model, eligible, groups, tmp_path / "cache", "same",
                                             batch_size=2, max_stocks=3, checkpoint_batches=1)
    assert timing["resumed_at_group"] == 2
    assert model.calls - before == (len(groups) - 2 + 1) // 2
    expected, expected_valid, _, _ = extract_encoder_features(panel, SmallEncoder(), max_stocks=3, batch_size=2)
    np.testing.assert_allclose(actual[eligible], expected[eligible], rtol=0, atol=0)
    assert isinstance(actual, np.memmap)
    before = model.calls
    _, cached = cached_encoder_features(panel, model, eligible, groups, tmp_path / "cache", "same", max_stocks=3)
    assert cached["cache_hit"] and model.calls == before


def test_raw_cache_matches_dense_baseline_and_basic_mask(tmp_path):
    panel = panel_fixture("2018-01-01", "2018-07-01", stocks=5)
    panel.stock_obs[::7, 1, 10] = False
    normalizer = fit_normalizer(panel)
    normalized = normalize_panel(panel, normalizer)
    eligible = context_eligibility(panel)
    actual = cached_raw_features(panel, normalizer, eligible, tmp_path / "raw", "raw")
    expected = raw_feature_baseline(normalized)
    np.testing.assert_allclose(actual[eligible], expected[eligible], rtol=1e-5, atol=1e-6)
    basic = _basic_raw_view(actual, tmp_path / "basic", "basic", eligible)
    basic_expected = raw_feature_baseline(_normalized_view(panel, normalizer, "basic"))
    np.testing.assert_allclose(basic[eligible], basic_expected[eligible], rtol=1e-5, atol=1e-6)
    assert np.isnan(basic[eligible][:, 9:14]).all()
    assert (basic[eligible][:, 37:42] == 0).all()


def save_run(tmp_path, panel, variant="E", alignment=None):
    run = tmp_path / ("run_" + panel.metadata["dataset_fingerprint"] + "_" + variant)
    run.mkdir(parents=True, exist_ok=True)
    config = PilotConfig(width=8, temporal_layers=1, cross_stock_layers=1, heads=2,
                         predictor_width=8, predictor_layers=1, sigreg_directions=8)
    normalizer = fit_normalizer(panel)
    metadata = {"variant": variant, "variant_spec": VARIANTS[variant],
                "dataset_fingerprint": panel.metadata["dataset_fingerprint"],
                "normalizer": normalizer, "model_config": asdict(config), "code": source_fingerprint(),
                "seed": 42, "encoder_train_cutoff": "2018-12-31",
                "inference_peer_partition_size": 32, "panel_metadata": panel.metadata}
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    seed_everything(5)
    export_encoder(FutureJEPA(config), run / "encoder_step_000002.pt", {**metadata, "optimizer_step": 2})
    return run


def prediction_targets(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["origin_date"], row["symbol"]): row["target"]
            for row in rows if row["model"] == "trained_encoder"}, rows


def test_predictions_gzip_roundtrip_preserves_columns_and_missing_targets(tmp_path):
    row = {"model": "trained_encoder", "origin_index": 12, "origin_date": "2019-01-02",
           "entry_date": "2019-01-03", "exit_date": "2019-01-09", "symbol": "BRK.B",
           "prediction": .0125, "target": None, "cohort": "development"}
    report = _save_evaluation({"report": {"test": True}, "predictions": [row]}, tmp_path)
    path = tmp_path / "predictions.csv.gz"
    assert report["artifacts"]["predictions"] == str(path)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = list(reader)
        assert reader.fieldnames == list(row)
    assert actual == [{k: "" if v is None else str(v) for k, v in row.items()}]
    assert not (tmp_path / "predictions.csv.gz.tmp").exists()
    assert not (tmp_path / "predictions.csv").exists()
    saved = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert saved["artifacts"]["predictions"] == str(path)


def test_horizon_one_and_common_full_cohorts_share_targets_and_embeddings(tmp_path):
    torch.set_num_threads(1)
    wide = panel_fixture()
    narrow = narrower(wide)
    run = save_run(tmp_path, wide, "E")
    progress = []
    report = evaluate_expanded_run(wide, narrow, run, years=(2019,), device="cpu",
                                  batch_size=32, bootstrap_samples=0, on_progress=progress.append)
    assert report["primary_horizon"] == 1
    assert set(report["results"]) == {"1"}
    assert not report["historical_evaluation_consulted"]
    common = report["results"]["1"]["common"]
    full = report["results"]["1"]["expanded"]
    assert common["protocol"]["primary_horizon_trading_days"] == 1
    common_targets, common_rows = prediction_targets(common["artifacts"]["predictions"])
    full_targets, full_rows = prediction_targets(full["artifacts"]["predictions"])
    assert len(common_targets) < len(full_targets)
    assert all(full_targets[key] == target for key, target in common_targets.items())
    for rows in (common_rows, full_rows):
        sets = [{(r["origin_date"], r["symbol"]) for r in rows if r["model"] == name}
                for name in ("trained_encoder", "random_encoder", "raw_features")]
        assert sets[0] == sets[1] == sets[2]
    assert report["inference"]["reused_between_common_and_expanded"]
    again = evaluate_expanded_run(wide, narrow, run, years=(2019,), device="cpu",
                                 batch_size=32, bootstrap_samples=0)
    assert again["evaluation_cache_hit"]
    assert again["results"] == report["results"]
    assert not list((tmp_path / "expanded_evaluation_cache").glob("trained_*"))
    assert list((tmp_path / "expanded_evaluation_cache").glob("random_*"))
    assert list((tmp_path / "expanded_evaluation_cache").glob("raw_book_*"))
    with pytest.raises(ValueError, match="development"):
        evaluate_expanded_run(wide, narrow, run, years=(2022,), device="cpu")
    with pytest.raises(ValueError, match="Primary readout"):
        evaluate_expanded_run(wide, narrow, run, primary_horizon=5, device="cpu")


def test_narrow_and_wide_training_share_deployment_peers_and_cohort_fingerprints(tmp_path):
    torch.set_num_threads(1)
    wide = panel_fixture()
    narrow = narrower(wide)
    wide_run, narrow_run = save_run(tmp_path, wide, "B"), save_run(tmp_path, narrow, "B")
    reports = [evaluate_expanded_run(wide, narrow, run, years=(2019,), device="cpu",
                                    batch_size=32, bootstrap_samples=0)
               for run in (wide_run, narrow_run)]
    assert reports[0]["training_universe"] == "expanded"
    assert reports[1]["training_universe"] == "original"
    assert reports[0]["peer_partitions"] == reports[1]["peer_partitions"]
    for cohort in ("common", "expanded"):
        a, b = [r["results"]["5"][cohort] for r in reports]
        assert a["cohort"] == b["cohort"]
        assert a["metrics"]["raw_features"] == b["metrics"]["raw_features"]
