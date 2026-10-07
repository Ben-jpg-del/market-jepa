"""Behavioral checks for leakage, missing entities, EMA, and exact resume."""
import copy

import numpy as np
import pytest
import torch

from src.model import FutureJEPA, PilotConfig, encoder_diagnostics, sigreg
from src.training import TrainConfig, export_encoder, load_snapshot, profile_train, train_steps
from src.training import _device_precision
from src.preflight import native_bf16_supported


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    options = dict(width=16, predictor_width=8, temporal_layers=1,
                   cross_stock_layers=1, predictor_layers=1, context_days=20,
                   sigreg_directions=16, sigreg_points=9)
    options.update(kwargs)
    return PilotConfig(**options)


class Sampler:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        stock = self.rng.normal(size=(2, 3, 20, 14)).astype("float32")
        market = self.rng.normal(size=(2, 20, 8)).astype("float32")
        fs = self.rng.normal(size=(2, 3, 5, 14)).astype("float32")
        fm = self.rng.normal(size=(2, 5, 8)).astype("float32")
        fso, fmo = np.zeros(fs.shape, bool), np.zeros(fm.shape, bool)
        fso[..., :4], fmo[..., 0] = True, True
        return {"stock": stock, "market": market,
                "stock_obs": np.ones(stock.shape, bool),
                "market_obs": np.ones(market.shape, bool),
                "eligibility": np.ones((2, 3), bool),
                "future_stock": fs, "future_market": fm,
                "future_stock_obs": fso, "future_market_obs": fmo,
                "target_valid": np.ones((2, 3), bool)}

    def state_dict(self):
        return copy.deepcopy(self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state)


def batch():
    return {k: torch.from_numpy(v) for k, v in Sampler()().items()}


def encode(model, b):
    return model.encode(b["stock"], b["market"], b["stock_obs"],
                        b["market_obs"], b["eligibility"])


def test_future_perturbation_does_not_change_context_representation():
    model, b = FutureJEPA(config()).eval(), batch()
    before = encode(model, b)["stock_embeddings"].detach()
    torch.manual_seed(31)
    first_loss = model.loss(b)["prediction_loss"].item()
    b["future_stock"] = b["future_stock"] * 20 - 100
    b["future_market"] = b["future_market"] * 20 + 100
    after = encode(model, b)["stock_embeddings"].detach()
    torch.manual_seed(31)
    second_loss = model.loss(b)["prediction_loss"].item()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert first_loss != second_loss


def test_teacher_covers_every_encoder_parameter_and_is_frozen_deterministic():
    model, b = FutureJEPA(config(dropout=0.3)).train(), batch()
    assert not model.teacher.training
    assert dict(model.online.named_parameters()).keys() == dict(model.teacher.named_parameters()).keys()
    with torch.no_grad():
        for parameter in model.online.parameters():
            parameter.add_(1)
    before = {name: parameter.clone() for name, parameter in model.teacher.named_parameters()}
    model.update_teacher(0.9)
    for name, parameter in model.teacher.named_parameters():
        torch.testing.assert_close(parameter, before[name] + 0.1)
        assert not parameter.requires_grad
    args = (b["future_stock"], b["future_market"], b["future_stock_obs"],
            b["future_market_obs"], b["target_valid"])
    with torch.no_grad():
        one, two = model.teacher(*args), model.teacher(*args)
    torch.testing.assert_close(one["stock_embeddings"], two["stock_embeddings"], rtol=0, atol=0)
    model.loss(b)["loss"].backward()
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert model.online.stock_projection.weight.grad.abs().sum() > 0


def test_ineligible_stock_and_unavailable_values_cannot_change_valid_stocks():
    model, b = FutureJEPA(config()).eval(), batch()
    b["eligibility"][:, 1] = False
    before = encode(model, b)
    b["stock"][:, 1] = float("nan")
    after = encode(model, b)
    torch.testing.assert_close(before["stock_embeddings"], after["stock_embeddings"], rtol=0, atol=0)
    torch.testing.assert_close(before["market_embedding"], after["market_embedding"], rtol=0, atol=0)
    assert not after["valid_stock_mask"][:, 1].any()
    b["stock_obs"][:, 0, :5] = False
    before = encode(model, b)
    b["stock"][:, 0, :5] = float("nan")
    after = encode(model, b)
    torch.testing.assert_close(before["stock_embeddings"], after["stock_embeddings"], rtol=0, atol=0)


def test_sigreg_retains_scale_and_offset_sensitivity():
    torch.manual_seed(12)
    x = torch.randn(1000, 16)
    torch.manual_seed(15)
    normal = sigreg(x, directions=128)
    torch.manual_seed(15)
    scaled = sigreg(x * 3, directions=128)
    torch.manual_seed(15)
    shifted = sigreg(x + 3, directions=128)
    assert scaled > normal * 10
    assert shifted > normal * 10


def test_resume_matches_uninterrupted_training_including_sampler_and_ema(tmp_path):
    torch.manual_seed(3)
    original = FutureJEPA(config())
    full, resumed = copy.deepcopy(original), copy.deepcopy(original)
    cfg = TrainConfig(steps=4, accumulation_steps=2, device="cpu", precision="fp32",
                      checkpoint_every=0, log_every=0)
    full_result = train_steps(full, Sampler(), cfg)
    partial = train_steps(resumed, Sampler(), cfg, checkpoint_dir=tmp_path, stop_at=2)
    resumed_result = train_steps(resumed, Sampler(seed=500), cfg, resume_from=partial["checkpoint"])
    assert full_result["optimizer_step"] == resumed_result["optimizer_step"] == 4
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    for a, b in zip(full_result["history"], resumed_result["history"]):
        assert a["loss"] == b["loss"]
        assert a["lr"] == b["lr"]
        assert a["ema"] == b["ema"]


def test_profiler_does_not_mutate_model_sampler_or_rng():
    model, sampler = FutureJEPA(config()), Sampler()
    original = copy.deepcopy(model.state_dict())
    rng, sampler_state = torch.get_rng_state().clone(), sampler.state_dict()
    cfg = TrainConfig(device="cpu", precision="fp32", accumulation_steps=1)
    result = profile_train(model, sampler, cfg, steps=2, warmup_steps=1)
    assert result["median_seconds_per_update"] > 0
    for name, value in original.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert sampler.state_dict() == sampler_state
    assert torch.equal(torch.get_rng_state(), rng)


def test_within_date_collapse_visible_despite_global_variance():
    market = torch.randn(8, 1, 16)
    same_for_all_stocks = market.expand(-1, 4, -1)
    stats = encoder_diagnostics(same_for_all_stocks, torch.ones(8, 4, dtype=torch.bool))
    assert stats["global"]["mean_std"] > 0
    assert stats["within_date"]["mean_std"] == 0
    assert stats["within_date"]["effective_rank"] == 0


def test_compact_export_preserves_encoder_output_and_metadata_without_predictor(tmp_path):
    model, b = FutureJEPA(config()).eval(), batch()
    model.update_teacher(0.9)
    path = tmp_path / "encoder.pt"
    metadata = {"dataset_fingerprint": "example", "normalizer": {"median": [0.0], "scale": [1.0]}}
    export_encoder(model, path, metadata)
    saved = torch.load(path, weights_only=False)
    assert "predictor" not in saved and "model" not in saved and "optimizer" not in saved
    assert set(saved["online"]) == set(model.online.state_dict())
    restored = load_snapshot(path)
    torch.testing.assert_close(encode(model, b)["stock_embeddings"],
                               encode(restored, b)["stock_embeddings"], rtol=0, atol=0)
    assert restored.snapshot_metadata == metadata
    with pytest.raises(ValueError, match="Compact encoder"):
        train_steps(restored, Sampler(), TrainConfig(steps=1, device="cpu"))
    teacher_restored = load_snapshot(path, teacher=True)
    expected = model.teacher(b["stock"], b["market"], b["stock_obs"], b["market_obs"], b["eligibility"])
    torch.testing.assert_close(expected["stock_embeddings"],
                               encode(teacher_restored, b)["stock_embeddings"], rtol=0, atol=0)


def test_callback_false_stops_at_successful_update_and_saves_resumable_checkpoint(tmp_path):
    model = FutureJEPA(config())
    cfg = TrainConfig(steps=4, device="cpu", accumulation_steps=1,
                      checkpoint_every=100, log_every=0)
    stopped = train_steps(model, Sampler(), cfg, checkpoint_dir=tmp_path,
                          on_step=lambda row: False if row["step"] == 2 else None)
    assert stopped["optimizer_step"] == 2 and stopped["stopped_early"]
    assert stopped["checkpoint"].endswith("step_000002.pt")
    resumed = train_steps(model, Sampler(), cfg, resume_from=stopped["checkpoint"])
    assert resumed["optimizer_step"] == 4 and not resumed["stopped_early"]


def test_t4_uses_fp16_even_when_torch_reports_emulated_bf16_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    assert not native_bf16_supported(torch)
    _, precision, dtype = _device_precision(TrainConfig(device="cuda"))
    assert precision == "fp16" and dtype == torch.float16
    with pytest.raises(ValueError, match="native BF16"):
        _device_precision(TrainConfig(device="cuda", precision="bf16"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    assert native_bf16_supported(torch)
    _, precision, dtype = _device_precision(TrainConfig(device="cuda"))
    assert precision == "bf16" and dtype == torch.bfloat16
