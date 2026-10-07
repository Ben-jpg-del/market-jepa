"""Tiny local orchestration tests: no live APIs and no GPU workloads."""
from contextlib import contextmanager
import copy
import json
import signal
import time

import numpy as np
import pandas as pd
import pytest
import torch

import src.cloud_job as job
from src.data import MarketPanel, save_panel
from src.model import PilotConfig


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def panel_file(tmp_path):
    rng = np.random.default_rng(4)
    t, k = 120, 3
    dates = pd.bdate_range("2018-01-01", periods=t).to_numpy(dtype="datetime64[D]")
    stock, market = rng.normal(size=(t,k,14)).astype("float32"), rng.normal(size=(t,8)).astype("float32")
    price, volume = np.full((t,k),100.), np.full((t,k),1000.)
    panel = MarketPanel(dates, ["A","B","C"], stock, market,
                        np.ones_like(stock,bool), np.ones_like(market,bool),
                        price,price,volume,volume,np.full(t,100.),np.full(t,100.),
                        np.ones((t,k),bool), {"dataset_fingerprint":"cloud-test", "normalizer":None,
                                              "synthetic":True, "point_in_time_verified":False})
    path = tmp_path/"panel.npz"
    save_panel(panel,path)
    return path


def small_config():
    return PilotConfig(width=16,predictor_width=8,temporal_layers=1,predictor_layers=1,
                       sigreg_directions=8,sigreg_points=9)


def fake_profile(panel,name,**kwargs):
    return {"variant":name,"mean_seconds_per_update":0.01,"profile_updates":kwargs["profile_steps"]}


def report(name):
    score={"A":.01,"B":.02,"C":.04,"D":.01,"E":.01}[name]
    return {"metrics":{"trained_encoder":{"development":{
        "mean_daily_rank_ic":score,"rank_ic_95ci_date_blocks":[score-.005,score+.005]}}}}


def fake_eval(panel,run_dir,**kwargs):
    assert kwargs["years"]==(2019,2020,2021)
    return report(run_dir.name.split("_")[1])


def options():
    return dict(hours=1,device="cpu",batch_dates=1,stocks=3,accumulation_steps=1,
                profile_steps=1,warmup_steps=0,screen_steps=1,total_steps=2,
                bootstrap_samples=2,evaluation_batch_size=1,model_config=small_config(),
                allow_synthetic=True,use_signals=False)


def test_small_job_runs_matched_stages_then_resumes_without_retraining(tmp_path,monkeypatch):
    panel=panel_file(tmp_path)
    monkeypatch.setattr(job,"profile_variant",fake_profile)
    monkeypatch.setattr(job,"evaluate_variant",fake_eval)
    result=job.run_job(panel,tmp_path/"run",**options())
    assert result["status"]=="completed"
    assert result["selection"]["variant"]=="C"
    assert len(result["training"])==7
    assert result["training"]["final_C_seed43"]["optimizer_step"]==2
    assert len(result["development"])==5 and len(result["final_development"])==2
    assert not result["historical_evaluation_consulted"]
    assert len(result["available_encoder_exports"])==7
    def forbidden(*args,**kwargs):
        raise AssertionError("Completed stage was unnecessarily repeated")
    monkeypatch.setattr(job,"profile_variant",forbidden)
    monkeypatch.setattr(job,"run_variant",forbidden)
    monkeypatch.setattr(job,"evaluate_variant",forbidden)
    resumed=job.run_job(panel,tmp_path/"run",**options())
    assert resumed["status"]=="completed"
    assert len(resumed["attempts"])==2


def test_partial_first_snapshot_pauses_without_selecting_then_resumes(tmp_path,monkeypatch):
    panel=panel_file(tmp_path)
    monkeypatch.setattr(job,"profile_variant",fake_profile)
    monkeypatch.setattr(job,"evaluate_variant",fake_eval)
    original=job.run_variant
    def stop_after_first(*args,**kwargs):
        kwargs["on_step"]=lambda row: False
        return original(*args,**kwargs)
    monkeypatch.setattr(job,"run_variant",stop_after_first)
    settings={**options(),"screen_steps":2,"total_steps":3}
    paused=job.run_job(panel,tmp_path/"run",**settings)
    assert paused["status"]=="paused_deadline"
    assert paused["training"]["screen_C_seed42"]["optimizer_step"]==1
    assert paused["selection"] is None and not paused["development"]
    assert len(paused["available_encoder_exports"])==1
    monkeypatch.setattr(job,"run_variant",original)
    completed=job.run_job(panel,tmp_path/"run",**settings)
    assert completed["status"]=="completed"
    assert completed["training"]["screen_C_seed42"]["optimizer_step"]==2
    assert completed["training"]["final_C_seed43"]["optimizer_step"]==3


def test_selection_matches_one_se_preference_and_requires_all_controls():
    reports={name:report(name) for name in job.VARIANTS}
    reports["B"]["metrics"]["trained_encoder"]["development"]["mean_daily_rank_ic"]=.039
    selection=job.choose_future_variant(reports,{"screen_steps":500,"total_steps":2500},"test")
    assert selection["variant"]=="B" and selection["highest_score_variant"]=="C"
    with pytest.raises(ValueError,match="All five"):
        job.choose_future_variant({"C":report("C")},{"screen_steps":500,"total_steps":2500},"test")


@pytest.mark.skipif(not hasattr(signal,"setitimer"),reason="POSIX alarm unavailable")
def test_deadline_interrupts_a_long_stage_and_restores_signal_handler():
    previous=signal.getsignal(signal.SIGALRM)
    with pytest.raises(job.JobDeadline):
        with job.Deadline(.000003).enforce():
            time.sleep(.05)
    assert signal.getsignal(signal.SIGALRM)==previous
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
