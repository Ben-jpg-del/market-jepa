import numpy as np
import torch

from src.evaluation import make_forward_labels
from src.model import FutureJEPA
from src.walkforward_data import normal_scores_by_date
from src.walkforward_readout import daily_ic, paired_block_difference, walk_forward_ridge, window_means, window_view
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parent))
from test_walkforward_data import panel_assembly  # noqa: E402
from test_walkforward_model import config, fake_batch  # noqa: E402


def rows(panel, horizon=5):
    labels, _, ends = make_forward_labels(panel, horizon)
    elig = np.isfinite(labels)
    rank = normal_scores_by_date(np.where(elig, labels, np.nan), elig)
    t_idx, k_idx = np.nonzero(elig)
    return labels[t_idx, k_idx], rank[t_idx, k_idx], t_idx, k_idx, ends


def test_ridge_never_uses_rows_whose_labels_end_at_or_after_the_boundary():
    panel = panel_assembly(n_dates=900, n_stocks=8, seed=4)
    y_raw, y_rank, t_idx, k_idx, ends = rows(panel)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(t_idx), 6)).astype(np.float32)
    year = int(np.unique(panel.dates.astype("datetime64[Y]").astype(int) + 1970)[-1])
    pred_a, rec_a = walk_forward_ridge(X, y_rank, y_raw, t_idx, panel.dates, ends, year, purge=20)
    assert rec_a["status"] == "ok" and pred_a
    # poison every row that ends on or after the boundary (minus the purge): the fit must not change
    boundary = np.datetime64(f"{year}-01-01")
    year_of_row = panel.dates[t_idx].astype("datetime64[Y]").astype(int) + 1970
    # pre-test rows whose labels end inside the purge zone (or after the boundary) must be excluded from the fit
    late = (ends[t_idx] >= boundary - np.timedelta64(20, "D")) & (year_of_row < year)
    assert late.any()
    X2 = X.copy(); X2[late] = 1e6
    y2 = y_rank.copy(); y2[late] = 1e3
    pred_b, _ = walk_forward_ridge(X2, y2, y_raw, t_idx, panel.dates, ends, year, purge=20)
    assert set(pred_a) == set(pred_b) and all(abs(pred_a[i] - pred_b[i]) < 1e-9 for i in pred_a)


def test_aux_direct_equals_head_applied_to_online_embedding():
    model = FutureJEPA(config()).eval(); batch = fake_batch()
    with torch.no_grad():
        ctx = model.online(batch["stock"], batch["market"], batch["stock_obs"], batch["market_obs"], batch["eligibility"])
        direct = model.aux_head(ctx["stock_embeddings"])
        manual = ctx["stock_embeddings"] @ model.aux_head.weight.T + model.aux_head.bias
    torch.testing.assert_close(direct, manual)


def test_window_view_offsets_and_ic_helpers():
    panel = panel_assembly(n_dates=400, n_stocks=6, seed=5)
    view, offset = window_view(panel, 100, 150, context=20)
    assert offset == 81 and len(view.dates) == 70 and np.array_equal(view.dates, panel.dates[81:151])
    t = np.repeat(np.arange(10), 6); y = np.random.default_rng(1).normal(size=60); p = y + 0.1 * np.random.default_rng(2).normal(size=60)
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-11")).astype("datetime64[D]")
    ic = daily_ic(p, y, t, dates)
    assert len(ic) == 10 and all(v > 0.5 for v in ic.values())
    means = window_means(ic, {"w": (2020, 2020), "none": (2021, 2021)})
    assert means["w"] > 0.5 and means["none"] is None
    assert paired_block_difference(ic, ic, 2020, 2020) is None  # fewer than 25 dates
