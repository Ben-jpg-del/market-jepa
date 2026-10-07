import numpy as np
import pytest

from src.data import fit_normalizer, normalize_panel, synthetic_panel
from src.evaluation import context_eligibility, make_forward_labels
from src.walkforward_data import WalkForwardSampler, aux_targets, normal_scores_by_date
from src.panel_assembly import assemble_panel
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent))
from test_panel_assembly import family_arrays  # noqa: E402


def panel_assembly(n_dates=700, n_stocks=8, seed=3, cutoff="2017-12-31"):
    base = synthetic_panel(n_dates=n_dates, n_stocks=n_stocks, seed=seed)
    base.metadata.setdefault("stock_features", list(base.metadata["stock_features"]))
    fam, fund = family_arrays(base, seed=seed + 1)
    raw = assemble_panel(base, fam, fund)
    return normalize_panel(raw, fit_normalizer(raw, cutoff=cutoff, start=str(raw.dates[0])))


def test_batch_shapes_and_future_windows_never_cross_the_cutoff():
    cutoff = "2017-12-31"
    panel = panel_assembly(cutoff=cutoff)
    sampler = WalkForwardSampler(panel, horizons=(5, 20), batch_dates=4, stocks_per_date=3, context=20, seed=1,
                                 start=str(panel.dates[0]), end=cutoff)
    batch = sampler.sample()
    T = len(panel.dates); cutoff_index = int(np.searchsorted(panel.dates, np.datetime64(cutoff), side="right") - 1)
    assert batch["stock"].shape == (4, 3, 20, 15) and batch["market"].shape == (4, 20, 11)
    assert batch["future_stock"].shape == (4, 3, 2, 20, 15) and batch["future_market"].shape == (4, 2, 20, 11)
    assert batch["future_valid"].shape == (4, 3, 2) and batch["aux_target"].shape == (4, 3, 2)
    assert (batch["future_end_index"] <= cutoff_index).all()
    assert (batch["future_end_index"][:, 1] - batch["origins"] == 20).all() and (batch["future_end_index"][:, 0] - batch["origins"] == 5).all()
    assert (sampler.origins + 20 <= cutoff_index).all()
    # the window ending at t+h is exactly the panel slice (no other data enters)
    b = 0; t = int(batch["origins"][b]); k = int(batch["symbol_indices"][b, 0])
    np.testing.assert_array_equal(batch["future_stock"][b, 0, 1], panel.stock[t + 20 - 19:t + 21, k])


def test_perturbing_post_cutoff_data_changes_no_batch_array():
    cutoff = "2017-12-31"
    panel = panel_assembly(cutoff=cutoff)
    cutoff_index = int(np.searchsorted(panel.dates, np.datetime64(cutoff), side="right") - 1)
    def draw(p):
        s = WalkForwardSampler(p, horizons=(5, 20), batch_dates=6, stocks_per_date=3, context=20, seed=7, start=str(p.dates[0]), end=cutoff)
        return [s.sample() for _ in range(3)]
    before = draw(panel)
    import copy
    poisoned = copy.copy(panel)
    poisoned.stock = panel.stock.copy(); poisoned.market = panel.market.copy(); poisoned.close = panel.close.copy(); poisoned.open = panel.open.copy()
    poisoned.stock[cutoff_index + 1:] = 99.0; poisoned.market[cutoff_index + 1:] = 99.0
    poisoned.close[cutoff_index + 1:] *= 3.0; poisoned.open[cutoff_index + 1:] *= 3.0
    after = draw(poisoned)
    for x, y in zip(before, after):
        for key in x:
            np.testing.assert_array_equal(x[key], y[key], err_msg=key)


def test_aux_targets_are_per_date_normal_scores_within_the_cutoff():
    panel = panel_assembly(cutoff="2017-12-31")
    elig = context_eligibility(panel, 20)
    target, valid, raw = aux_targets(panel, (5, 20), "2017-12-31", elig)
    labels, _, ends = make_forward_labels(panel, 20)
    cutoff = np.datetime64("2017-12-31")
    assert not valid[(ends > cutoff) | np.isnat(ends), :, 1].any()
    t = int(np.flatnonzero(valid[:, :, 1].sum(1) >= 5)[0])
    m = valid[t, :, 1]
    assert abs(target[t, m, 1].mean()) < 1e-6 and np.argsort(np.argsort(target[t, m, 1])).tolist() == np.argsort(np.argsort(labels[t, m])).tolist()
    scores = normal_scores_by_date(np.array([[3.0, 1.0, 2.0, np.nan, 5.0, 4.0]]), np.array([[True] * 6]), min_n=5)
    assert np.isnan(scores[0, 3]) and scores[0, 1] < scores[0, 2] < scores[0, 0] < scores[0, 5] < scores[0, 4]


def test_state_round_trip_and_horizon_check():
    panel = panel_assembly()
    a = WalkForwardSampler(panel, horizons=(5, 20), batch_dates=2, stocks_per_date=3, context=20, seed=5, start=str(panel.dates[0]), end="2017-12-31")
    a.sample(); state = a.state_dict()
    b = WalkForwardSampler(panel, horizons=(5, 20), batch_dates=2, stocks_per_date=3, context=20, seed=99, start=str(panel.dates[0]), end="2017-12-31")
    b.load_state_dict(state)
    x, y = a.sample(), b.sample()
    np.testing.assert_array_equal(x["origins"], y["origins"]); np.testing.assert_array_equal(x["aux_target"], y["aux_target"])
    c = WalkForwardSampler(panel, horizons=(5,), batch_dates=2, stocks_per_date=3, context=20, seed=5, start=str(panel.dates[0]), end="2017-12-31")
    with pytest.raises(ValueError, match="horizons"):
        c.load_state_dict(state)


def test_origin_observed_eligibility_equals_base_rule():
    base = synthetic_panel(n_dates=500, n_stocks=6, seed=11)
    fam, fund = family_arrays(base, seed=2)
    assembled = assemble_panel(base, fam, fund)
    assert np.array_equal(context_eligibility(assembled, 60), context_eligibility(base, 60))
