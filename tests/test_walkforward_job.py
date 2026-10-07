import json

import numpy as np
import pytest
import torch

from src.data import save_panel, synthetic_panel
from src.model import PilotConfig
from src.panel_assembly import assemble_panel
from src.walkforward_job import run_walkforward_job
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent))
from test_panel_assembly import family_arrays  # noqa: E402


def tiny_config():
    return PilotConfig(stock_features=15, market_features=11, width=32, family_width=8, heads=4, predictor_width=16, predictor_layers=1,
                       temporal_layers=1, tokenization="family", objective="future_family", feature_layout="v2", horizons=(5, 20),
                       context_days=20, sigreg_directions=16, sigreg_points=9)


@pytest.fixture(scope="module")
def panel_path(tmp_path_factory):
    base = synthetic_panel(n_dates=1500, n_stocks=8, seed=21, start="2013-01-01")
    fam, fund = family_arrays(base, seed=22)
    assembled = assemble_panel(base, fam, fund)
    assembled.metadata["synthetic"] = True
    path = tmp_path_factory.mktemp("panel") / "panel_russell2000.npz"
    save_panel(assembled, path)
    return path


def job(panel_path, out, **kw):
    args = dict(hours=1.0, device="cpu", runs={"v2_s7": {"seed": 7}}, first_year=2016, last_year=2017, train_start="2013-06-01",
                initial_steps=3, continuation_steps=2, batch_dates=2, stocks=3, accumulation_steps=1, evaluation_batch_size=4,
                allow_synthetic=True, use_signals=False, model_config=tiny_config(), wandb_mode="disabled")
    args.update(kw)
    return run_walkforward_job(panel_path, out, **args)


def test_job_runs_to_completion_with_per_year_cutoffs_and_outputs(tmp_path, panel_path):
    state = job(panel_path, tmp_path / "run")
    assert state["status"] == "completed", state.get("error")
    run = tmp_path / "run" / "v2_s7_seed7"
    for year, steps in ((2016, 3), (2017, 2)):
        meta = json.loads((run / f"year_{year}" / "run_metadata.json").read_text())
        assert meta["cutoff"] == f"{year - 1}-12-31" and meta["steps"] == steps
        export = torch.load(run / f"encoder_{year}.pt", map_location="cpu", weights_only=False)
        assert export["normalizer"]["train_cutoff"] == f"{year - 1}-12-31" and export["artifact_type"] == "walkforward_encoder"
        readout = json.loads((tmp_path / "run" / "evaluations" / "v2_s7" / f"year_{year}" / "readout.json").read_text())
        assert set(readout["models"]) == {"teacher_ridge", "online_ridge", "random_ridge", "aux_direct"}
        assert all(set(v) == {"h5", "h20"} for v in readout["models"].values())
        feats = np.load(tmp_path / "run" / "evaluations" / "v2_s7" / f"year_{year}" / "features.npz")
        assert feats["online"].shape[1] == 64 and feats["aux"].shape[1] == 2 and len(feats["t_idx"]) == readout["rows_year"]
    meta2017 = json.loads((run / "year_2017" / "run_metadata.json").read_text())
    assert meta2017["warm_start_sha256"] is not None
    assert (tmp_path / "run" / "evaluations" / "v2_s7" / "year_2016" / "features_history.npz").exists()
    summary = json.loads((tmp_path / "run" / "evaluations" / "v2_s7" / "summary.json").read_text())
    assert "teacher_ridge" in summary["mean_daily_rank_ic"] and "yearly" in summary["mean_daily_rank_ic"]["teacher_ridge"]["h5"]
    state_file = json.loads((tmp_path / "run" / "expanded_job_state.json").read_text())
    assert state_file["job"] == "walkforward" and state_file["status"] == "completed"


def test_second_invocation_skips_completed_stages(tmp_path, panel_path):
    job(panel_path, tmp_path / "run")
    run = tmp_path / "run" / "v2_s7_seed7"
    before = {p.name: p.stat().st_mtime for p in run.glob("encoder_*.pt")}
    state = job(panel_path, tmp_path / "run")
    assert state["status"] == "completed"
    assert {p.name: p.stat().st_mtime for p in run.glob("encoder_*.pt")} == before
    assert len(state["attempts"]) == 2 and state["attempts"][-1]["elapsed_seconds"] < 5


def test_state_mismatch_is_refused(tmp_path, panel_path):
    job(panel_path, tmp_path / "run")
    with pytest.raises(ValueError, match="does not match"):
        job(panel_path, tmp_path / "run", continuation_steps=5)


def test_interrupted_training_resumes_to_identical_weights(tmp_path, panel_path):
    full = job(panel_path, tmp_path / "full", initial_steps=4, continuation_steps=2)
    assert full["status"] == "completed"
    # interrupt the first year after 2 steps via a tiny deadline; the job pauses, then a resume completes it
    calls = {"n": 0}
    import src.walkforward_job as wj
    original = wj.train_steps

    def stopping(model, sampler, tc, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(model, sampler, tc, **{**kw, "stop_at": 2})
        return original(model, sampler, tc, **kw)
    wj.train_steps = stopping
    try:
        paused = job(panel_path, tmp_path / "part", initial_steps=4, continuation_steps=2)
        assert paused["status"] == "paused_deadline"
        resumed = job(panel_path, tmp_path / "part", initial_steps=4, continuation_steps=2)
    finally:
        wj.train_steps = original
    assert resumed["status"] == "completed"
    for year in (2016, 2017):
        a = torch.load(tmp_path / "full" / "v2_s7_seed7" / f"encoder_{year}.pt", map_location="cpu", weights_only=False)["model"]
        b = torch.load(tmp_path / "part" / "v2_s7_seed7" / f"encoder_{year}.pt", map_location="cpu", weights_only=False)["model"]
        for name in a:
            torch.testing.assert_close(a[name], b[name], msg=f"{year}:{name}")
