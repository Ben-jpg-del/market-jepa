"""Family-token encoder, family-mask objective, and the F1/T1/T2 experiment variants."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.features import STOCK_FAMILIES, STOCK_FEATURES
from src.model import FutureJEPA, PilotConfig
from src.runner import VARIANTS, VariantSampler, _variant, run_variant
from src.training import load_snapshot
from src.wandb_logging import channel_projection_norms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pilot_runner import model_config, panel  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def family_config(**kwargs):
    options = dict(width=16, family_width=4, predictor_width=8, temporal_layers=1, cross_stock_layers=1,
                   predictor_layers=1, context_days=20, sigreg_directions=16, sigreg_points=9,
                   tokenization="family")
    options.update(kwargs)
    return PilotConfig(**options)


def context_batch(b=2, k=3, t=20, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {"stock": torch.randn(b, k, t, 14, generator=g), "stock_obs": torch.ones(b, k, t, 14, dtype=torch.bool),
            "market": torch.randn(b, t, 8, generator=g), "market_obs": torch.ones(b, t, 8, dtype=torch.bool),
            "eligibility": torch.ones(b, k, dtype=torch.bool)}


def test_families_partition_all_stock_features():
    covered = sorted(i for _, idx in STOCK_FAMILIES for i in idx)
    assert covered == list(range(len(STOCK_FEATURES)))


def test_config_validation():
    with pytest.raises(ValueError):
        PilotConfig(objective="family_mask")  # needs family tokens
    assert PilotConfig(tokenization="family").slot_width * len(STOCK_FAMILIES) == 128
    ext = PilotConfig(tokenization="family", stock_features=21, market_features=20)
    assert len(ext.token_groups) == 5 and ext.slot_width == 128 // 5
    with pytest.raises(ValueError):
        PilotConfig(tokenization="family", stock_features=17)  # no family layout


def test_family_encoder_isolates_hidden_family_and_reports_family_summaries():
    torch.manual_seed(0)
    model = FutureJEPA(family_config()).eval()
    b = context_batch()
    full = model.encode(b["stock"], b["market"], b["stock_obs"], b["market_obs"], b["eligibility"])
    assert full["stock_embeddings"].shape == (2, 3, 16)
    assert full["family_summaries"].shape == (2, 3, 4, 16) and full["family_valid"].all()
    # Hide the trend family for stock 0 of row 0: its trend summary must vanish and the
    # values of that hidden family must no longer influence any output.
    obs = b["stock_obs"].clone()
    trend = list(dict(STOCK_FAMILIES)["trend"])
    obs[0, 0, :, trend] = False
    masked = model.encode(b["stock"], b["market"], obs, b["market_obs"], b["eligibility"])
    assert not masked["family_valid"][0, 0, 1] and masked["family_valid"][0, 0, 0]
    perturbed = b["stock"].clone()
    perturbed[0, 0, :, trend] += 5.0
    again = model.encode(perturbed, b["market"], obs, b["market_obs"], b["eligibility"])
    torch.testing.assert_close(masked["stock_embeddings"], again["stock_embeddings"])
    # Other families still carry signal: perturbing visible fast channels changes the output.
    perturbed = b["stock"].clone()
    perturbed[0, 0, :, 0] += 5.0
    changed = model.encode(perturbed, b["market"], obs, b["market_obs"], b["eligibility"])
    assert not torch.allclose(masked["stock_embeddings"][0, 0], changed["stock_embeddings"][0, 0])


def test_family_mask_loss_trains_and_targets_are_teacher_summaries():
    torch.manual_seed(1)
    model = FutureJEPA(family_config(objective="family_mask"))
    b = context_batch()
    out = model.loss(b)
    assert torch.isfinite(out["loss"]) and out["prediction_loss"] > 0
    out["loss"].backward()
    grads = [p.grad for p in model.online.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert model.family_query.grad is not None
    with pytest.raises(ValueError):
        model.loss(b, context_mask_probability=0.5)


def test_family_mask_needs_two_observed_families():
    torch.manual_seed(2)
    model = FutureJEPA(family_config(objective="family_mask"))
    b = context_batch()
    b["stock_obs"][..., 4:] = False  # only the fast family observed: nothing left as context
    with pytest.raises(ValueError):
        model.loss(b)
    b["stock_obs"][..., 4:7] = True  # fast + trend observed (basic-feature-like)
    out = model.loss(b)
    assert torch.isfinite(out["loss"])


def test_variant_specs_carry_defaults_and_new_fields():
    for name in ("A", "B", "C", "D", "E", "F"):
        spec = _variant(name)[1]
        assert spec["target_channels"] == "fast" and spec["tokenization"] == "bundled" and spec["objective"] == "block"
    assert _variant("F1")[1]["target_channels"] == "all" and _variant("F1")[1]["historical"]
    assert _variant("T1")[1]["objective"] == "family_mask" and _variant("T1")[1]["tokenization"] == "family"
    assert _variant("T2")[1]["historical"] and _variant("T2")[1]["tokenization"] == "family"
    assert set(VARIANTS) >= {"F1", "T1", "T2"}


def test_f1_target_exposes_all_channels_while_f_exposes_fast_only():
    p = panel()
    for variant, expect_all in (("F", False), ("F1", True)):
        sampler = VariantSampler(p, variant, batch_dates=2, stocks=4)
        batch = sampler()
        slow_observed = batch["future_stock_obs"][..., 4:].any()
        market_slow = batch["future_market_obs"][..., 1:].any()
        assert bool(slow_observed) is expect_all and bool(market_slow) is expect_all
        # Student context still hides the last ten sessions for chosen stocks.
        chosen = batch["prediction_valid"]
        assert chosen.any() and not batch["stock_obs"][chosen][:, -10:].any()


@pytest.mark.parametrize("variant", ["F1", "T1", "T2"])
def test_new_variants_train_export_and_reload(tmp_path, variant):
    p = panel()
    kwargs = dict(seed=42, screen_steps=1, total_steps=2, batch_dates=1, stocks=3, device="cpu",
                  model_config=model_config(), accumulation_steps=1, log_every=1)
    if variant.startswith("T"):
        kwargs["model_config"] = PilotConfig(**{**kwargs["model_config"].__dict__, "family_width": kwargs["model_config"].width // 4})
    result = run_variant(p, variant, tmp_path / variant, stop_at=2, **kwargs)
    assert result["optimizer_step"] == 2
    snapshot = load_snapshot(result["encoder_export"])
    meta = snapshot.snapshot_metadata
    assert meta["variant"] == variant
    assert meta["model_config"]["tokenization"] == ("family" if variant.startswith("T") else "bundled")
    assert meta["model_config"]["objective"] == ("family_mask" if variant == "T1" else "block")
    assert meta["target_channels"] == ("all" if variant == "F1" else "fast")
    history = json.loads((tmp_path / variant / "training_metrics.json").read_text())
    assert "proj_norm" in history[0] and f"proj_norm/stock/{STOCK_FEATURES[9]}" in history[0]["proj_norm"]
    # A same-config random encoder (what the evaluator builds) matches the export's shapes.
    random_model = FutureJEPA(PilotConfig(**meta["model_config"]))
    assert {k: v.shape for k, v in random_model.online.state_dict().items()} == \
           {k: v.shape for k, v in snapshot.online.state_dict().items()}


def test_channel_norms_cover_every_channel_for_family_encoder():
    norms = channel_projection_norms(FutureJEPA(PilotConfig(tokenization="family")))
    stock = [k for k in norms if k.startswith("proj_norm/stock/") and not k.endswith("_total_values")]
    assert sorted(stock) == sorted(f"proj_norm/stock/{n}" for n in STOCK_FEATURES)
    assert "proj_norm/market/spy_r1" in norms


def test_feature_tokens_config_slots_and_hide_sets():
    from src.features import FEATURE_SHORTCUT_PARTNERS
    cfg = PilotConfig(tokenization="feature", objective="feature_mask")
    assert cfg.slot_width == 128 // 14 and len(cfg.token_groups) == 14
    with pytest.raises(ValueError):
        PilotConfig(objective="feature_mask")  # needs feature tokens
    model = FutureJEPA(cfg)
    assert model.online.slot_padding == 128 - 14 * (128 // 14)
    names = list(STOCK_FEATURES)
    for f, name in enumerate(names):
        hidden = {names[i] for i in torch.nonzero(model.hide_channels[f]).flatten().tolist()}
        assert name in hidden and model.family_channels[f].sum() == 1
        for partner in FEATURE_SHORTCUT_PARTNERS.get(name, ()):
            assert partner in hidden, (name, partner)
    b = context_batch(t=20)
    out = model.loss(b)
    assert torch.isfinite(out["loss"])
    enc = model.encode(b["stock"], b["market"], b["stock_obs"], b["market_obs"], b["eligibility"])
    assert enc["stock_embeddings"].shape == (2, 3, 128) and enc["family_summaries"].shape[2] == 14
    # padded slots are exactly zero before cross-stock attention; after it the output is dense,
    # but hiding a feature must still zero its own slot in the pre-attention token: check via valid mask
    obs = b["stock_obs"].clone(); obs[0, 0, :, 4] = False  # hide r5 for one stock
    enc2 = model.encode(b["stock"], b["market"], obs, b["market_obs"], b["eligibility"])
    assert not enc2["family_valid"][0, 0, 4] and enc2["family_valid"][0, 0, 0]


def test_feature_mask_variant_trains_and_reloads(tmp_path):
    p = panel()
    cfg = PilotConfig(width=56, family_width=14, predictor_width=8, temporal_layers=1, cross_stock_layers=1,
                      predictor_layers=1, heads=4, sigreg_directions=8, sigreg_points=9)
    result = run_variant(p, "T3", tmp_path / "T3", seed=42, screen_steps=1, total_steps=2, stop_at=2, batch_dates=1,
                         stocks=3, device="cpu", model_config=cfg, accumulation_steps=1, log_every=1)
    snap = load_snapshot(result["encoder_export"])
    assert snap.snapshot_metadata["model_config"]["tokenization"] == "feature"
    assert snap.snapshot_metadata["model_config"]["objective"] == "feature_mask"
    random_model = FutureJEPA(PilotConfig(**snap.snapshot_metadata["model_config"]))
    assert {k: v.shape for k, v in random_model.online.state_dict().items()} == {k: v.shape for k, v in snap.online.state_dict().items()}


def test_extended_channels_family_encoder_and_family_mask():
    cfg = PilotConfig(tokenization="family", objective="family_mask", stock_features=21, market_features=20,
                      width=40, predictor_width=8, temporal_layers=1, cross_stock_layers=1, predictor_layers=1,
                      context_days=20, sigreg_directions=8, sigreg_points=9)
    model = FutureJEPA(cfg)
    assert model.online.slot_padding == 40 - 5 * (40 // 5)
    g = torch.Generator().manual_seed(5)
    b = {"stock": torch.randn(2, 3, 20, 21, generator=g), "stock_obs": torch.ones(2, 3, 20, 21, dtype=torch.bool),
         "market": torch.randn(2, 20, 20, generator=g), "market_obs": torch.ones(2, 20, 20, dtype=torch.bool),
         "eligibility": torch.ones(2, 3, dtype=torch.bool)}
    out = model.loss(b)
    assert torch.isfinite(out["loss"])
    enc = model.encode(b["stock"], b["market"], b["stock_obs"], b["market_obs"], b["eligibility"])
    assert enc["stock_embeddings"].shape == (2, 3, 40) and enc["family_summaries"].shape[2] == 5
    norms = channel_projection_norms(model)
    assert "proj_norm/stock/eps_sue" in norms and "proj_norm/market/vix_log" in norms
