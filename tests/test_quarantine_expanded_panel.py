"""The quarantine is a retrospective exclusion, never an identity remapping."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.data import MarketPanel, load_panel, save_panel
from src.features import STOCK_FEATURES, janus_reference

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/data/quarantine_expanded_panel.py"
spec = importlib.util.spec_from_file_location("quarantine_expanded_panel", SCRIPT)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def fixture_panel(n=360, k=12):
    rng = np.random.default_rng(817)
    ref = rng.normal(0,.014,(n,9))
    returns = rng.normal(0,.02,(n,k))
    returns[:,1] = returns[:,0]  # Tied per-security scores remain tied.
    returns[150:153,3] = np.nan
    universe = rng.random((n,k))>.15
    universe[200] = False
    universe[200,:2] = True  # After removing S1, JANUS must be unavailable.
    fractiles,_ = janus_reference(ref,returns,universe)
    stock = rng.normal(size=(n,k,14)).astype(np.float32)
    stock[...,12] = fractiles
    stock[...,13] = np.nan
    stock[5:,:,13] = fractiles[5:]-fractiles[:-5]
    market = rng.normal(size=(n,8)).astype(np.float32)
    close = np.full((n,k),100.,np.float64)
    panel = MarketPanel(dates=np.arange(np.datetime64("2017-01-01"),np.datetime64("2017-01-01")+np.timedelta64(n,"D")),
        symbols=[f"S{i}" for i in range(k)],stock=stock,stock_obs=np.isfinite(stock),
        market=market,market_obs=np.isfinite(market),open=close.copy(),close=close,
        volume=close.copy()*1000,volume_baseline20=close.copy()*1000,
        benchmark_open=np.full(n,100.),benchmark_close=np.full(n,100.),universe=universe,
        metadata={"dataset_fingerprint":"fixture-source","normalizer":None,
                  "stock_features":list(STOCK_FEATURES),"limitations":[],"feature_notes":{}})
    return panel,ref,returns


def write_input(folder,panel,name="panel"):
    panel_path=folder/(name+".npz")
    coverage_path=folder/(name+"_coverage.json")
    save_panel(panel,panel_path)
    coverage={"status":"ready_research_only","panel_sha256":q.file_sha256(panel_path),
        "stock_coverage":{s:{"raw_rows":len(panel.dates),"adjusted_rows":len(panel.dates),
                              "member_dates":100,"selected_dates":100} for s in panel.symbols}}
    coverage_path.write_text(json.dumps(coverage))
    return panel_path,coverage_path


def audit_file(folder):
    path=folder/"audit.json"
    path.write_text(json.dumps({"priority_identity_conflicts":{s:{"reason":"fixture conflict"} for s in ("S1","S4","S7")}}))
    return path


def test_reranking_matches_full_janus_recomputation_with_ties_missing_and_small_cohorts():
    panel,ref,returns=fixture_panel()
    keep=[k for k,s in enumerate(panel.symbols) if s not in {"S1","S4","S7"}]
    fractiles,delta=q.rerank_janus(panel.stock[:,keep,12],panel.stock_obs[:,keep,12],panel.universe[:,keep])
    expected,_=janus_reference(ref,returns[:,keep],panel.universe[:,keep])
    expected_delta=np.full_like(expected,np.nan)
    expected_delta[5:]=expected[5:]-expected[:-5]
    np.testing.assert_allclose(fractiles,expected,atol=1e-7,equal_nan=True)
    np.testing.assert_allclose(delta,expected_delta,atol=1e-7,equal_nan=True)
    assert np.isnan(fractiles[200]).all()
    assert np.isfinite(expected).sum()>500


def test_transform_all_axes_preserves_source_and_recomputes_availability(tmp_path):
    panel,ref,returns=fixture_panel()
    original=copy.deepcopy(panel)
    policy=q.load_policy(audit_file(tmp_path),3,end_date="2017-10-31")
    result=q.quarantine_panel(panel,policy)
    assert len(result.symbols)==9
    assert str(result.dates[-1])=="2017-10-31"
    for name in q.STOCK_AXIS_FIELDS:
        assert getattr(result,name).shape[:2]==(304,9)
    np.testing.assert_array_equal(result.stock_obs,np.isfinite(result.stock))
    np.testing.assert_array_equal(result.market,panel.market[:304])
    np.testing.assert_array_equal(panel.stock,original.stock)
    np.testing.assert_array_equal(panel.universe,original.universe)
    assert result.metadata["dataset_fingerprint"]!="fixture-source"
    assert result.metadata["identity_quarantine"]["point_in_time_verified"] is False
    panel.metadata["normalizer"]={"mean":0}
    with pytest.raises(ValueError,match="raw panel"):
        q.quarantine_panel(panel,policy)


def test_files_backups_sha_idempotence_and_original_copy_availability(tmp_path):
    panel,ref,returns=fixture_panel()
    source,coverage=write_input(tmp_path,panel)
    old_sha=q.file_sha256(source)
    audit=audit_file(tmp_path)
    result=q.quarantine_files(source,coverage,audit,expected_count=3)
    assert result["remaining_symbols"]==9
    report=json.loads(coverage.read_text())
    assert report["panel_sha256"]==q.file_sha256(source)
    assert q.file_sha256(tmp_path/"panel.pre_identity_quarantine.npz")==old_sha
    assert report["selection"]["total_selected_stock_dates"]==int(load_panel(source).universe.sum())
    assert set(report["stock_coverage"])==set(load_panel(source).symbols)
    assert not any("raw_rows" in v for v in report["stock_coverage"].values())
    assert report["samplers"]["5"]["origins"]>0
    repeated=q.quarantine_files(source,coverage,audit,expected_count=3)
    assert repeated["idempotent_existing_quarantine"]
    assert repeated["panel_sha256"]==result["panel_sha256"]

    # Original retains its selected pool but adopts expanded per-origin masks.
    original=copy.deepcopy(panel)
    original.universe[:]=True
    original.stock[...,12],_=janus_reference(ref,returns,original.universe)
    original.stock_obs=np.isfinite(original.stock)
    old,old_coverage=write_input(tmp_path,original,"original_archive")
    hashes=q.file_sha256(old),q.file_sha256(old_coverage)
    destination=tmp_path/"original_filtered.npz"
    destination_coverage=tmp_path/"original_filtered.json"
    q.quarantine_files(old,old_coverage,audit,output_panel_path=destination,
        output_coverage_path=destination_coverage,expected_count=3,origin_availability_path=source)
    assert hashes==(q.file_sha256(old),q.file_sha256(old_coverage))
    filtered,expanded=load_panel(destination),load_panel(source)
    np.testing.assert_array_equal(filtered.universe,expanded.universe)
    np.testing.assert_allclose(filtered.stock[...,12],expanded.stock[...,12],equal_nan=True,atol=1e-7)
    np.testing.assert_allclose(filtered.stock[...,13],expanded.stock[...,13],equal_nan=True,atol=1e-7)
    assert "expanded origin availability" in filtered.metadata["universe_label"]
    assert filtered.metadata["dataset_fingerprint"]!=expanded.metadata["dataset_fingerprint"]
    again=q.quarantine_files(old,old_coverage,audit,output_panel_path=destination,
        output_coverage_path=destination_coverage,expected_count=3,origin_availability_path=source)
    assert again["idempotent_existing_quarantine"]


def test_incomplete_preparation_rejected_without_mutating_panel(tmp_path):
    panel,_,_=fixture_panel()
    path,coverage=write_input(tmp_path,panel)
    digest=q.file_sha256(path)
    coverage.write_text(json.dumps({"status":"computing_features"}))
    with pytest.raises(RuntimeError,match="not complete"):
        q.quarantine_files(path,coverage,audit_file(tmp_path),expected_count=3)
    assert q.file_sha256(path)==digest
