import json

import numpy as np
import pytest
import torch

from src.model import FutureJEPA, PilotConfig
from src.wandb_logging import (WandbTrainingLogger, channel_projection_norms,
                                       deterministic_run_id, evaluation_payload)


class FakeRun:
    def __init__(self):
        self.logged, self.summary, self.artifacts, self.finished = [], {}, [], False
        self.url = "https://example.invalid/run"

    def log(self, payload):
        self.logged.append(dict(payload))

    def define_metric(self, *args, **kwargs):
        return None

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True


class FakeSDK:
    def __init__(self, fail=False):
        self.runs, self.fail, self.init_kwargs = [], fail, []

    def login(self, **kwargs):
        return True

    def Settings(self, **kwargs):
        return kwargs

    def Artifact(self, name, type):
        return {"name": name, "type": type, "files": []}

    def init(self, **kwargs):
        if self.fail:
            raise RuntimeError("network down")
        self.init_kwargs.append(kwargs)
        run = FakeRun()
        self.runs.append(run)
        return run


def test_disabled_without_key_is_noop(monkeypatch):
    import src.wandb_logging as module
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(module, "_netrc_key", lambda: None)
    sdk = FakeSDK()
    logger = WandbTrainingLogger(run_name="x", mode="auto", sdk=sdk)
    assert logger.mode == "offline" and logger.enabled
    calls = []
    wrapped = logger.wrap(lambda row: calls.append(row["step"]) or False)
    assert wrapped({"step": 1, "loss": 1.0}) is False
    assert calls == [1] and sdk.init_kwargs[0]["mode"] == "offline"
    logger.finish()
    disabled = WandbTrainingLogger(run_name="x", mode="disabled", sdk=FakeSDK())
    assert not disabled.enabled and disabled.url is None
    disabled.on_step({"step": 1, "loss": 1.0})
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert not WandbTrainingLogger(run_name="x", mode="auto", sdk=FakeSDK()).enabled


def test_offline_logging_flattens_rows_and_norms(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    sdk = FakeSDK()
    model = FutureJEPA(PilotConfig())
    logger = WandbTrainingLogger(run_name="arm", run_id="fixed", group="g", tags=["t"],
                                 config={"seed": 42, "cfg": PilotConfig()}, mode="offline", log_every=2, sdk=sdk)
    logger.attach(model)
    row = {"step": 2, "loss": 0.5, "prediction_loss": 0.4, "sigreg_loss": 2.0, "lr": 1e-4, "ema": 0.99,
           "gradient_norm": 1.5, "seconds": 0.1,
           "encoder": {"global": {"effective_rank": 10.0}, "within_date": {"participation_rank": 5.0},
                       "valid_embeddings": 64}}
    logger.on_step(row)
    assert sdk.init_kwargs[0]["id"] == "fixed" and sdk.init_kwargs[0]["mode"] == "offline"
    assert sdk.init_kwargs[0]["config"]["cfg"]["width"] == 128
    payload = sdk.runs[0].logged[0]
    assert payload["train/optimizer_step"] == 2 and payload["train/loss"] == 0.5
    assert payload["encoder/global/effective_rank"] == 10.0
    assert payload["encoder/valid_embeddings"] == 64
    assert "proj_norm/stock/r1" in payload and "proj_norm/market/spy_r1" in payload
    logger.on_step({"step": 3, "loss": 0.4})
    assert "proj_norm/stock/r1" not in sdk.runs[0].logged[1]
    logger.on_step({"step": 4, "loss": 0.3, "proj_norm": {"proj_norm/stock/r1": 1.25}})
    assert sdk.runs[0].logged[2]["proj_norm/stock/r1"] == 1.25
    logger.log_training_result({"optimizer_step": 3, "precision": "fp32", "device": "cpu"})
    assert sdk.runs[0].summary["training/precision"] == "fp32"
    logger.finish()
    assert sdk.runs[0].finished


def test_sdk_failure_never_raises(monkeypatch):
    logger = WandbTrainingLogger(run_name="arm", mode="offline", sdk=FakeSDK(fail=True))
    for step in range(1, 8):
        logger.on_step({"step": step, "loss": 1.0})
    assert not logger.enabled and logger.failures >= 5


def test_channel_norms_cover_every_channel():
    model = FutureJEPA(PilotConfig())
    norms = channel_projection_norms(model)
    stock = [k for k in norms if k.startswith("proj_norm/stock/") and not k.endswith("_total_values")]
    market = [k for k in norms if k.startswith("proj_norm/market/") and not k.endswith("_total_values")]
    assert len(stock) == 14 and len(market) == 8
    assert all(v > 0 for v in norms.values())
    # zeroing one channel's columns must zero exactly that channel's norm
    with torch.no_grad():
        w = model.online.stock_projection.weight
        for d in range(5):
            w[:, d * 28 + 4] = 0  # channel index 4 = r5
    norms = channel_projection_norms(model)
    assert norms["proj_norm/stock/r5"] == 0 and norms["proj_norm/stock/r1"] > 0


def test_evaluation_payload_matches_report_schema():
    report = {"optimizer_step": 500, "results": {5: {"common": {"metrics": {
        "trained_encoder": {"development": {"mean_daily_rank_ic": 0.02, "rank_ic_95ci_date_blocks": [0.0, 0.04],
                                            "valid_ic_dates": 752, "prediction_rows": 91051,
                                            "yearly": {"2019": {"mean_rank_ic": 0.01}}}},
        "random_encoder": {"development": {"mean_daily_rank_ic": 0.025}},
        "raw_features": {"development": {"mean_daily_rank_ic": 0.035, "rank_ic_95ci_date_blocks": None}}}}}}}
    payload = evaluation_payload(report)
    assert payload["eval/checkpoint_step"] == 500
    assert payload["eval/h5/common/trained_encoder/mean_daily_rank_ic"] == 0.02
    assert payload["eval/h5/common/trained_encoder/rank_ic_ci_high"] == 0.04
    assert payload["eval/h5/common/trained_encoder/2019/mean_rank_ic"] == 0.01
    assert payload["eval/h5/common/delta_vs_raw_features"] == pytest.approx(-0.015)
    assert "eval/h5/common/raw_features/rank_ic_ci_low" not in payload
    json.dumps(payload)  # must be serializable


def test_run_id_is_stable_and_identity_sensitive():
    a = deterministic_run_id(group="g", arm="expanded_F", seed=42)
    assert a == deterministic_run_id(group="g", arm="expanded_F", seed=42)
    assert a != deterministic_run_id(group="g", arm="expanded_F", seed=43)
    assert a.startswith("fjepa-") and len(a) == 30
