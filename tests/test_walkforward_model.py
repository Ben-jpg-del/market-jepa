import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from src.model import FutureJEPA, PilotConfig
from src.training import TrainConfig, train_steps
from src.walkforward_data import WalkForwardSampler
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent))
from test_walkforward_data import panel_assembly  # noqa: E402


def config(**kw):
    base = dict(stock_features=15, market_features=11, width=32, family_width=8, heads=4, predictor_width=16, predictor_layers=1,
                temporal_layers=1, tokenization="family", objective="future_family", feature_layout="v2", horizons=(5, 20), context_days=20,
                sigreg_directions=16, sigreg_points=9)
    base.update(kw)
    return PilotConfig(**base)


def fake_batch(B=2, K=3, C=20, H=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"stock": r(B, K, C, 15), "stock_obs": torch.ones(B, K, C, 15, dtype=torch.bool), "market": r(B, C, 11),
            "market_obs": torch.ones(B, C, 11, dtype=torch.bool), "eligibility": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
            "future_stock": r(B, K, H, C, 15), "future_stock_obs": torch.ones(B, K, H, C, 15, dtype=torch.bool),
            "future_market": r(B, H, C, 11), "future_market_obs": torch.ones(B, H, C, 11, dtype=torch.bool),
            "future_valid": torch.ones(B, K, H, dtype=torch.bool), "aux_target": r(B, K, H), "aux_valid": torch.ones(B, K, H, dtype=torch.bool)}


def test_config_validation_and_json_round_trip():
    cfg = config(horizons=[5, 20])
    assert cfg.horizons == [5, 20] and cfg.slot_width == 8
    assert PilotConfig(**json.loads(json.dumps(asdict(cfg)))) == cfg
    with pytest.raises(ValueError, match="width == family_width"):
        config(width=40)
    with pytest.raises(ValueError, match="family tokenization"):
        config(tokenization="bundled", width=32)
    with pytest.raises(ValueError, match="assembled layout"):
        config(stock_features=14)


def test_loss_terms_and_horizon_conditioning():
    model = FutureJEPA(config()); batch = fake_batch()
    out = model.loss(batch)
    for key in ("loss", "prediction_loss", "sigreg_loss", "aux_loss", "prediction_loss_h5", "prediction_loss_h20"):
        assert key in out and out[key].ndim == 0 and torch.isfinite(out[key])
    ctx = model.online(batch["stock"], batch["market"], batch["stock_obs"], batch["market_obs"], batch["eligibility"])
    p5 = model.predictor(ctx, query=model.family_query[0] + model.horizon_embedding[0])
    p20 = model.predictor(ctx, query=model.family_query[0] + model.horizon_embedding[1])
    assert p5.shape == (2, 3, 32) and not torch.allclose(p5, p20)
    # an all-invalid future makes the batch unusable rather than silently zero
    bad = dict(batch); bad["future_valid"] = torch.zeros_like(batch["future_valid"])
    with pytest.raises(ValueError, match="no valid future"):
        model.loss(bad)


def test_projection_head_takes_sigreg_without_touching_the_readout_embedding():
    torch.manual_seed(0)
    model = FutureJEPA(config()); batch = fake_batch()
    first = model.loss(batch)
    with torch.no_grad():
        model.projection_head.weight.mul_(3.0)
    second = model.loss(batch)
    torch.testing.assert_close(first["context_embeddings"], second["context_embeddings"])
    assert not torch.isclose(first["sigreg_loss"], second["sigreg_loss"])
    assert torch.isclose(first["prediction_loss"], second["prediction_loss"])


def test_teacher_covers_online_only_and_heads_train():
    model = FutureJEPA(config())
    teacher = {n for n, _ in model.teacher.named_parameters()}; online = {n for n, _ in model.online.named_parameters()}
    assert teacher == online and all(not p.requires_grad for p in model.teacher.parameters())
    assert model.aux_head.weight.requires_grad and model.horizon_embedding.requires_grad and model.projection_head.weight.requires_grad
    out = model.loss(fake_batch()); out["loss"].backward()
    assert model.aux_head.weight.grad is not None and model.horizon_embedding.grad is not None and model.projection_head.weight.grad is not None


def test_train_steps_logs_new_keys_and_warm_start_resume_matches(tmp_path):
    panel = panel_assembly(n_dates=600, n_stocks=6, cutoff="2017-12-31")
    cfg = config()
    tc = TrainConfig(steps=4, accumulation_steps=1, lr=1e-3, warmup_fraction=0.25, device="cpu", precision="fp32", seed=1, checkpoint_every=2, log_every=1)
    def sampler(seed=3):
        return WalkForwardSampler(panel, horizons=(5, 20), batch_dates=2, stocks_per_date=3, context=20, seed=seed, start=str(panel.dates[0]), end="2017-12-31")
    torch.manual_seed(1); model = FutureJEPA(cfg)
    result = train_steps(model, sampler(), tc, checkpoint_dir=tmp_path / "a", metadata={"year": 2018})
    row = result["history"][-1]
    assert {"aux_loss", "prediction_loss_h5", "prediction_loss_h20"} <= set(row) and result["optimizer_step"] == 4
    # warm start: a continuation from the final weights with a fresh schedule
    cont = TrainConfig(steps=3, accumulation_steps=1, lr=5e-4, warmup_fraction=0.34, ema_start=0.995, device="cpu", precision="fp32", seed=2, checkpoint_every=1, log_every=1)
    torch.manual_seed(2); m1 = FutureJEPA(cfg); m1.load_state_dict(copy.deepcopy(model.state_dict()))
    full = train_steps(m1, sampler(5), cont, checkpoint_dir=tmp_path / "b", metadata={"year": 2019})
    torch.manual_seed(2); m2 = FutureJEPA(cfg); m2.load_state_dict(copy.deepcopy(model.state_dict()))
    train_steps(m2, sampler(5), cont, checkpoint_dir=tmp_path / "c", stop_at=1, metadata={"year": 2019})
    resumed = train_steps(m2, sampler(5), cont, checkpoint_dir=tmp_path / "c", resume_from=tmp_path / "c" / "step_000001.pt", metadata={"year": 2019})
    assert resumed["optimizer_step"] == full["optimizer_step"] == 3
    for (n, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        torch.testing.assert_close(a, b, msg=n)
