"""Small local tests for expansion controls; no APIs or cloud workloads."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

import src.expanded_job as expanded
import src.runner as runner
from src.data import MarketPanel, fit_normalizer, normalize_panel, save_panel
from src.model import PilotConfig


@pytest.fixture(autouse=True)
def isolated_tests(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    # Independent evaluator source may be authored concurrently; source identity
    # logic is already covered by runner tests, so freeze this fixture's identity.
    identity = lambda: {"sha256": "expanded-test-source", "files": {}}
    monkeypatch.setattr(runner, "source_fingerprint", identity)
    monkeypatch.setattr(expanded, "source_fingerprint", identity)
    yield
    torch.set_num_threads(previous)


def make_panel(k=3, name="original", delay=0):
    rng = np.random.default_rng(9)
    t = 130
    dates = pd.bdate_range("2018-01-01", periods=t).to_numpy(dtype="datetime64[D]")
    stock, market = rng.normal(size=(t,k,14)).astype("float32"), rng.normal(size=(t,8)).astype("float32")
    price, volume = np.full((t,k),100.), np.full((t,k),1000.)
    universe = np.ones((t,k),bool)
    universe[:delay] = False
    return MarketPanel(dates, [f"S{i}" for i in range(k)], stock, market,
                       np.ones_like(stock,bool), np.ones_like(market,bool),
                       price,price,volume,volume,np.full(t,100.),np.full(t,100.),universe,
                       {"dataset_fingerprint":name, "normalizer":None, "synthetic":True,
                        "point_in_time_verified":False})


def panel_paths(tmp_path):
    original, broad = tmp_path/"original.npz", tmp_path/"expanded.npz"
    save_panel(make_panel(), original)
    save_panel(make_panel(k=4,name="expanded",delay=75), broad)
    return original, broad


def config():
    return PilotConfig(width=16,predictor_width=8,temporal_layers=1,predictor_layers=1,
                       sigreg_directions=8,sigreg_points=9)


def options():
    return dict(hours=1,device="cpu",total_steps=2,intermediate_steps=1,batch_dates=1,
                stocks=3,accumulation_steps=1,evaluation_batch_size=1,bootstrap_samples=2,
                intermediate_min_remaining_seconds=0,model_config=config(),
                allow_synthetic=True,use_signals=False)


def fake_evaluate(expanded_panel, original_panel, run_dir, **kwargs):
    assert expanded_panel.metadata["normalizer"] is None
    assert original_panel.metadata["normalizer"] is None
    assert kwargs["years"] == (2019,2020,2021)
    spec = expanded.RUNS[run_dir.name.removesuffix("_seed42")]
    horizon = spec["horizon"]
    assert kwargs["primary_horizon"] == horizon
    assert kwargs["horizons"] == tuple(spec["readout_horizons"])
    kwargs["on_progress"]({"phase":"fake_readout", "fraction":1.0})
    return {"primary_horizon":horizon, "results":{str(h):{
        cohort:{"metrics":{"trained_encoder":{"development":{"mean_daily_rank_ic":0.0}}}}
        for cohort in ("common","expanded")} for h in spec["readout_horizons"]}}


def test_training_calendar_alignment_is_causal_and_preserves_raw_panels():
    old, broad = make_panel(), make_panel(4,"expanded",75)
    original_universe = old.universe.copy()
    aligned, report = expanded.align_training_panels({"original":old,"expanded":broad})
    assert report["start"] == str(broad.dates[75])
    assert report["count"] == len(broad.dates)-75
    assert not aligned["original"].universe[:75].any()
    np.testing.assert_array_equal(old.universe, original_universe)
    for name in ("original","expanded"):
        p = aligned[name]
        assert p.metadata["dataset_fingerprint"] != p.metadata["source_dataset_fingerprint"]
        assert p.metadata["training_date_alignment"] == report
        aligned[name] = expanded.normalize_training_panel(p, train_start=report["start"])
        assert aligned[name].metadata["normalizer"]["train_start"] == report["start"]
    a = runner.VariantSampler(aligned["original"],"B",batch_dates=1,stocks=3)
    b = runner.VariantSampler(aligned["expanded"],"B",batch_dates=1,stocks=3)
    np.testing.assert_array_equal(aligned["original"].dates[a.base.origins], aligned["expanded"].dates[b.base.origins])
    e = runner.VariantSampler(aligned["expanded"],"E",batch_dates=1,stocks=3)
    assert len(e.base.origins) > len(b.base.origins)  # horizon-specific end availability remains separate


def test_chunked_normalization_matches_reference_without_large_temporary_arrays():
    p = make_panel()
    p.stock_obs[50,0,7] = False
    p.stock[50,0,7] = np.nan
    expected = normalize_panel(p, fit_normalizer(p))
    actual = expanded.normalize_training_panel(p, chunk_dates=7)
    np.testing.assert_allclose(actual.stock, expected.stock, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual.market, expected.market, atol=2e-6, rtol=2e-6)
    assert actual.stock[50,0,7] == 0 and np.isnan(p.stock[50,0,7])


def test_all_arms_train_fully_and_e_uses_one_day_readout_then_resume_skips_work(tmp_path,monkeypatch):
    original,broad = panel_paths(tmp_path)
    calls=[]
    def evaluate(*args,**kwargs):
        calls.append((args[2].name,kwargs["snapshot"].name))
        return fake_evaluate(*args,**kwargs)
    monkeypatch.setattr(expanded,"evaluate_expanded_run",evaluate)
    state=expanded.run_expanded_job(original,broad,tmp_path/"run",**options())
    assert state["status"] == "completed"
    assert state["planned_optimizer_updates"] == len(expanded.RUNS) * options()["total_steps"]
    assert len(state["training"]) == len(expanded.RUNS) * 2
    assert len(state["evaluations"]) == len(expanded.RUNS) + len(expanded.INTERMEDIATE_EVALUATION_RUNS)
    assert list(state["training"])[0] == "expanded_C_step_000001"
    for name in expanded.RUNS:
        assert state["training"][f"{name}_step_000002"]["optimizer_step"] == 2
        metadata=json.loads((tmp_path/"run"/f"{name}_seed42"/"run_metadata.json").read_text())
        assert metadata["train_config"]["steps"] == 2
        assert metadata["seed"] == 42
        assert metadata["panel_metadata"]["training_date_alignment"] == state["training_date_alignment"]
    assert all(snapshot == "encoder_step_000002.pt" for _,snapshot in calls[:len(expanded.RUNS)])
    assert state["factorial_controls"] == {"basic_historical":"expanded_A", "book_historical":"expanded_F",
                                           "basic_future":"expanded_B", "book_future":"expanded_C"}
    for name in ("expanded_A", "expanded_F"):
        assert state["evaluations"][f"{name}_step_000002"]["readout_horizons"] == [5, 1]
    assert not state["model_selection_performed"] and not state["historical_evaluation_consulted"]
    monkeypatch.setattr(expanded,"run_variant",lambda *args,**kwargs: pytest.fail("retrained completed run"))
    monkeypatch.setattr(expanded,"evaluate_expanded_run",lambda *args,**kwargs: pytest.fail("reevaluated completed run"))
    resumed=expanded.run_expanded_job(original,broad,tmp_path/"run",**options())
    assert resumed["status"] == "completed" and len(resumed["attempts"]) == 2


def test_partial_c_snapshot_pauses_all_comparisons_and_resumes_same_budget(tmp_path,monkeypatch):
    original,broad = panel_paths(tmp_path)
    monkeypatch.setattr(expanded,"evaluate_expanded_run",fake_evaluate)
    original_run=expanded.run_variant
    def stop(*args,**kwargs):
        kwargs["on_step"] = lambda row: False
        return original_run(*args,**kwargs)
    monkeypatch.setattr(expanded,"run_variant",stop)
    opts={**options(),"intermediate_steps":2,"total_steps":3}
    paused=expanded.run_expanded_job(original,broad,tmp_path/"run",**opts)
    assert paused["status"] == "paused_deadline"
    assert len(paused["training"]) == 1 and not paused["evaluations"]
    assert paused["training"]["expanded_C_step_000002"]["optimizer_step"] == 1
    monkeypatch.setattr(expanded,"run_variant",original_run)
    complete=expanded.run_expanded_job(original,broad,tmp_path/"run",**opts)
    assert complete["status"] == "completed"
    assert all(complete["training"][f"{name}_step_000003"]["optimizer_step"] == 3 for name in expanded.RUNS)


def test_optional_intermediate_eval_defers_without_mislabeling_final_controls(tmp_path,monkeypatch):
    original,broad = panel_paths(tmp_path)
    monkeypatch.setattr(expanded,"evaluate_expanded_run",fake_evaluate)
    opts={**options(),"intermediate_min_remaining_seconds":10000}
    state=expanded.run_expanded_job(original,broad,tmp_path/"run",**opts)
    assert state["status"] == "completed_with_deferred_intermediate"
    assert len(state["deferred_evaluations"]) == len(expanded.INTERMEDIATE_EVALUATION_RUNS)
    assert all(state["evaluations"][f"{name}_step_000002"]["status"] == "completed" for name in expanded.RUNS)


def test_real_aligned_checkpoints_evaluate_with_raw_foreign_panel_for_h5_and_h1(tmp_path,monkeypatch):
    """Actual training/export/evaluator integration, including derived identity."""
    from dataclasses import replace
    import src.expanded_eval as evaluator
    monkeypatch.setattr(evaluator, "source_fingerprint", runner.source_fingerprint)
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2016-01-01","2019-12-31").to_numpy(dtype="datetime64[D]")
    t,k = len(dates),5
    stock,market = rng.normal(size=(t,k,14)).astype("float32"),rng.normal(size=(t,8)).astype("float32")
    closed = 100*np.exp(np.cumsum(rng.normal(0,.01,size=(t,k)),axis=0))
    opened = closed*np.exp(rng.normal(0,.002,size=(t,k)))
    volume = np.full((t,k),1000.)
    benchmark = 100*np.exp(np.cumsum(rng.normal(0,.005,size=t)))
    wide = MarketPanel(dates,[f"S{i}" for i in range(k)],stock,market,
                       np.ones_like(stock,bool),np.ones_like(market,bool),opened,closed,volume,volume,
                       benchmark*.999,benchmark,np.ones((t,k),bool),
                       {"dataset_fingerprint":"actual-integration-expanded","normalizer":None,"synthetic":True})
    narrow = replace(wide,symbols=wide.symbols[:3],stock=wide.stock[:,:3],stock_obs=wide.stock_obs[:,:3],
                     open=wide.open[:,:3],close=wide.close[:,:3],volume=wide.volume[:,:3],
                     volume_baseline20=wide.volume_baseline20[:,:3],universe=wide.universe[:,:3].copy(),
                     metadata={**wide.metadata,"dataset_fingerprint":"actual-integration-original"})
    wide.universe[:90] = False
    aligned,alignment = expanded.align_training_panels({"original":narrow,"expanded":wide})
    encoder_passes = []
    original_cached_encoder_features = evaluator.cached_encoder_features
    def counted_features(*args, **kwargs):
        encoder_passes.append(1)
        return original_cached_encoder_features(*args, **kwargs)
    monkeypatch.setattr(evaluator, "cached_encoder_features", counted_features)
    for dataset,variant,horizon in (("original","B",5),("expanded","E",1),("expanded","F",5)):
        prepared = expanded.normalize_training_panel(aligned[dataset],train_start=alignment["start"])
        run_dir = tmp_path/f"{dataset}_{variant}"
        trained = runner.run_variant(prepared,variant,run_dir,screen_steps=1,total_steps=1,
                                     batch_dates=1,stocks=3,device="cpu",model_config=config(),
                                     accumulation_steps=1,log_every=0)
        readout_horizons = (5, 1) if variant == "F" else (horizon,)
        passes_before = len(encoder_passes)
        report = evaluator.evaluate_expanded_run(wide,narrow,run_dir,snapshot=trained["encoder_export"],
            device="cpu",years=(2019,),horizons=readout_horizons,batch_size=32,bootstrap_samples=0)
        assert len(encoder_passes) - passes_before == 2  # Trained + random, independent of readout count.
        assert report["primary_horizon"] == horizon
        assert report["training_universe"] == dataset
        assert report["training_dataset_fingerprint"] == prepared.metadata["dataset_fingerprint"]
        assert report["training_source_dataset_fingerprint"] == aligned[dataset].metadata["source_dataset_fingerprint"]
        assert report["training_date_alignment"] == alignment
        assert not report["historical_evaluation_consulted"]
        if variant == "F":
            assert report["pretraining"]["objective"] == "historical_masked_block"
            assert report["pretraining"]["future_horizon"] is None
            assert report["pretraining"]["historical_hidden_days"] == 10
            assert report["primary_readout_horizon"] == 5
            assert report["prespecified_secondary_readout_horizons"] == [1]
            assert report["readout_horizons"] == [5, 1]
            assert report["inference"]["reused_between_readout_horizons"]
        for h in readout_horizons:
            for cohort in ("common","expanded"):
                metrics = report["results"][str(h)][cohort]["metrics"]
                assert metrics["trained_encoder"]["development"]["valid_ic_dates"] > 0
                assert metrics["raw_features"]["development"]["valid_ic_dates"] > 0


def test_custom_run_table_with_seeds_overrides_extra_steps_and_cohorts(tmp_path, monkeypatch):
    original, broad = panel_paths(tmp_path)
    runs = {
        "F0_s42": {"panel": "expanded", "variant": "F", "horizon": 5, "readout_horizons": [5], "seed": 42,
                   "extra_evaluation_steps": [2]},
        "T1_s43": {"panel": "expanded", "variant": "T1", "horizon": 5, "readout_horizons": [5], "seed": 43,
                   "model_overrides": {"family_width": 4}},
    }
    seen = []
    def evaluate(expanded_panel, original_panel, run_dir, **kwargs):
        seen.append((run_dir.name, kwargs["snapshot"].name, kwargs["cohort_names"]))
        return {"primary_horizon": 5, "optimizer_step": 3, "results": {"5": {
            c: {"metrics": {"trained_encoder": {"development": {"mean_daily_rank_ic": 0.0}}}} for c in kwargs["cohort_names"]}}}
    monkeypatch.setattr(expanded, "evaluate_expanded_run", evaluate)
    opts = {**options(), "total_steps": 3, "intermediate_steps": 1}
    state = expanded.run_expanded_job(original, broad, tmp_path / "run", runs=runs,
                                      final_cohorts=("common", "expanded"), intermediate_cohorts=("common",), **opts)
    assert state["status"] == "completed", state.get("error")
    assert (tmp_path / "run" / "F0_s42_seed42" / "run_metadata.json").exists()
    meta = json.loads((tmp_path / "run" / "T1_s43_seed43" / "run_metadata.json").read_text())
    assert meta["seed"] == 43 and meta["model_config"]["tokenization"] == "family" and meta["model_config"]["family_width"] == 4
    assert state["planned_optimizer_updates"] == 6
    names = [(r, s) for r, s, _ in seen]
    assert ("F0_s42_seed42", "encoder_step_000003.pt") in names and ("T1_s43_seed43", "encoder_step_000003.pt") in names
    assert ("F0_s42_seed42", "encoder_step_000001.pt") in names and ("F0_s42_seed42", "encoder_step_000002.pt") in names
    cohorts = {(r, s): c for r, s, c in seen}
    assert cohorts[("F0_s42_seed42", "encoder_step_000003.pt")] == ("common", "expanded")
    assert cohorts[("F0_s42_seed42", "encoder_step_000002.pt")] == ("common",)
    assert (tmp_path / "run" / "F0_s42_seed42" / "encoder_step_000002.pt").exists()
