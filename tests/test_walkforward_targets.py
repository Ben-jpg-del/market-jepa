import numpy as np
import torch

from src.model import FutureJEPA, PilotConfig
from src.walkforward_data import WalkForwardSampler, aux_targets, forward_realised_vol
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent))
from test_walkforward_data import panel_assembly  # noqa: E402
from test_walkforward_model import config, fake_batch  # noqa: E402


def test_supervised_config_skips_the_prediction_term_and_future_windows():
    cfg = config(prediction_weight=0.0, aux_weight=1.0, sigreg_weight=0.0)
    model = FutureJEPA(cfg); batch = fake_batch()
    batch = {k: v for k, v in batch.items()}
    batch["future_stock"] = torch.full_like(batch["future_stock"], float("nan"))  # must never be read
    out = model.loss(batch)
    assert torch.isfinite(out["loss"]) and float(out["prediction_loss"]) == 0.0
    torch.testing.assert_close(out["loss"], out["aux_loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.online.parameters())


def test_config_validation_for_new_knobs():
    for bad in ({"prediction_weight": 0.0, "aux_weight": 0.0}, {"aux_target": "nope"}, {"aux_loss": "nope"},
                {"aux_loss": "sharpe", "aux_target": "vol_rank"}):
        try:
            config(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_sharpe_loss_prefers_scores_aligned_with_returns():
    cfg = config(prediction_weight=0.0, aux_weight=1.0, sigreg_weight=0.0, aux_loss="sharpe")
    model = FutureJEPA(cfg)
    B, K, H = 6, 10, 2
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(B, K, H, generator=g); mask = torch.ones(B, K, H, dtype=torch.bool)
    good = model._sharpe_loss(raw.clone(), raw, mask)             # scores = returns
    bad = model._sharpe_loss(-raw.clone(), raw, mask)             # scores = -returns
    noise = model._sharpe_loss(torch.randn(B, K, H, generator=g), raw, mask)
    assert good < noise < bad and torch.isfinite(good)
    batch = fake_batch(); batch["aux_raw"] = torch.randn(2, 3, 2, generator=g); out = model.loss(batch)
    assert torch.isfinite(out["loss"])


def test_vol_rank_targets_are_per_date_scores_and_never_cross_the_cutoff():
    panel = panel_assembly(n_dates=400, n_stocks=8, seed=7)
    elig = np.ones((len(panel.dates), 8), bool)
    vol, ends = forward_realised_vol(panel, 5)
    assert vol.shape == (400, 8) and np.isnan(vol[-5:]).all() and np.isfinite(vol[:300]).mean() > 0.9
    cutoff = str(panel.dates[300])
    target, valid, raw = aux_targets(panel, (5, 20), cutoff, elig, kind="vol_rank")
    assert valid.shape == (400, 8, 2) and not valid[301:].any() and valid[:250].mean() > 0.9
    # per-date normal scores: mean ~0 across stocks on a valid date
    t = 100
    assert abs(target[t, valid[t, :, 0], 0].mean()) < 0.3
    # perturbing closes after the cutoff changes no valid target
    panel2 = panel_assembly(n_dates=400, n_stocks=8, seed=7)
    panel2.close = panel2.close.copy(); panel2.close[302:] *= np.exp(np.random.default_rng(1).normal(0, 0.5, panel2.close[302:].shape))
    target2, valid2, _ = aux_targets(panel2, (5, 20), cutoff, elig, kind="vol_rank")
    assert np.array_equal(valid, valid2) and np.allclose(target[valid], target2[valid2])


def test_sampler_without_future_leaves_future_arrays_empty_and_carries_raw():
    panel = panel_assembly(n_dates=400, n_stocks=8, seed=3)
    s = WalkForwardSampler(panel, horizons=(5, 20), batch_dates=2, stocks_per_date=4, context=20, start=str(panel.dates[0]),
                           end=str(panel.dates[300]), include_future=False)
    b = s.sample()
    assert not b["future_valid"].any() and not b["future_stock"].any()
    assert b["aux_raw"].shape == b["aux_target"].shape and np.isfinite(b["aux_raw"]).all()
