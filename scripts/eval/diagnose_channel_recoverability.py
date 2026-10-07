"""Per-channel recoverability for any set of encoder exports on the fixed diagnostic panel.

Same fixed sample as the earlier diagnostics (every fifth session 2010-2021, up to 32
common-cohort stocks per date in the seed-42 hash order, expanded peer groups). For each
encoder export given (and a same-config random encoder per distinct model config), encode the
rows, then fit a ridge readout from the frozen 128-d stock block to each channel's latest
observed value (fit 2010-2015, alpha on 2016-2017, refit 2010-2017, score 2019-2021 by mean
within-date Spearman). Reports the value at 500 vs 2,500 per arm.

Usage:
  work/colab-test-env/bin/python scripts/eval/diagnose_channel_recoverability.py \
      --artifacts outputs/colab-tokenization/live-run/artifacts --seed 42 --steps 500 2500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel  # noqa: E402
from src.expanded_eval import _normalized_view, shared_origin_mask  # noqa: E402
from src.features import STOCK_FEATURES  # noqa: E402
from src.model import FutureJEPA, PilotConfig  # noqa: E402
from src.training import seed_everything  # noqa: E402

FAST = {"r1", "overnight_gap", "log_high_low", "relative_volume"}
ALPHAS = (0.1, 1.0, 10.0, 100.0)
RNG = np.random.default_rng(42)


def log(msg):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg}), flush=True)


def fixed_sample(panel, original):
    eligible = shared_origin_mask(panel, original, context=60)
    date_ids = np.flatnonzero((panel.dates >= np.datetime64("2010-01-01")) &
                              (panel.dates <= np.datetime64("2021-12-31")))[::5]
    order = sorted(range(len(panel.symbols)), key=lambda k: hashlib.sha256(f"42:{panel.symbols[k]}".encode()).digest())
    rows, groups = [], []
    for t in date_ids:
        chosen = [k for k in order if eligible["common"][t, k]][:32]
        row_id = {k: len(rows) + j for j, k in enumerate(chosen)}
        rows.extend((int(t), int(k)) for k in chosen)
        peers = [k for k in order if eligible["expanded"][t, k]]
        for start in range(0, len(peers), 32):
            ids = peers[start:start + 32]
            selected = [(j, row_id[k]) for j, k in enumerate(ids) if k in row_id]
            if selected:
                groups.append((int(t), ids, selected))
    return np.array(rows, dtype=np.int64), groups


def encode(model, view, rows, groups, batch_size=16):
    features = np.full((len(rows), 256), np.nan, np.float32)
    with torch.inference_mode():
        for first in range(0, len(groups), batch_size):
            items = groups[first:first + batch_size]
            b = len(items)
            sx = np.zeros((b, 32, 60, 14), np.float32)
            mx = np.zeros((b, 60, 8), np.float32)
            so, mo, sv = np.zeros_like(sx, bool), np.zeros_like(mx, bool), np.zeros((b, 32), bool)
            for j, (t, ids, _) in enumerate(items):
                window = slice(t - 59, t + 1)
                sx[j, :len(ids)] = view.stock[window][:, ids].transpose(1, 0, 2)
                so[j, :len(ids)] = view.stock_obs[window][:, ids].transpose(1, 0, 2)
                mx[j], mo[j], sv[j, :len(ids)] = view.market[window], view.market_obs[window], True
            out = model.encode(*(torch.as_tensor(a) for a in (sx, mx, so, mo, sv)))
            sz, mz = out["stock_embeddings"].numpy(), out["market_embedding"].numpy()
            for j, (_, _, selected) in enumerate(items):
                for slot, row in selected:
                    features[row] = np.concatenate([sz[j, slot], mz[j]])
    assert np.isfinite(features).all()
    return features


def within_date_spearman(dates, pred, target):
    order = np.argsort(dates, kind="stable")
    dates, pred, target = dates[order], pred[order], target[order]
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    out_d, out_r = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        p, t = pred[a:b], target[a:b]
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 3 or np.ptp(t[ok]) == 0 or np.ptp(p[ok]) == 0:
            continue
        out_d.append(dates[a])
        out_r.append(spearmanr(p[ok], t[ok]).statistic)
    return np.array(out_d), np.array(out_r)


def block_ci(diff, block=4, n=2000):
    m = len(diff)
    if m < block + 1:
        return float("nan"), float("nan")
    starts = RNG.integers(0, m - block + 1, size=(n, int(np.ceil(m / block))))
    idx = (starts[..., None] + np.arange(block)).reshape(n, -1)[:, :m]
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def recoverability(features, dates, latest, years):
    x = features[:, :128].astype(np.float64)
    fit, val, refit, test = (years >= 2010) & (years <= 2015), (years >= 2016) & (years <= 2017), (years >= 2010) & (years <= 2017), (years >= 2019) & (years <= 2021)
    xm, xs = x[refit].mean(0), x[refit].std(0) + 1e-8
    x = (x - xm) / xs
    out = {}
    for j, name in enumerate(STOCK_FEATURES):
        y = latest[:, j]
        ok = np.isfinite(y)
        mu, sd = y[ok & refit].mean(), y[ok & refit].std() + 1e-12
        ys = (y - mu) / sd
        best, best_score = None, -np.inf
        for a in ALPHAS:
            model = Ridge(alpha=a).fit(x[fit & ok], ys[fit & ok])
            score = np.corrcoef(model.predict(x[val & ok]), ys[val & ok])[0, 1]
            if score > best_score:
                best, best_score = a, score
        model = Ridge(alpha=best).fit(x[refit & ok], ys[refit & ok])
        pred = model.predict(x)
        t = test & ok
        d, r = within_date_spearman(dates[t], pred[t], ys[t])
        out[name] = {"spearman": float(np.nanmean(r)), "dates": d, "per_date": r, "alpha": best}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=str(ROOT / "outputs" / "colab-tokenization" / "live-run" / "artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 2500])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    artifacts = Path(args.artifacts)
    out = Path(args.out) if args.out else artifacts.parent.parent / "channel-recoverability"
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(p for p in artifacts.glob(f"*_s{args.seed}_seed{args.seed}") if (p / "run_metadata.json").exists())
    if not run_dirs:
        print("no run directories for seed", args.seed)
        return 1
    log("loading panels")
    panel = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
    original = load_panel(ROOT / "work/fjepa-expanded-data/original_panel.npz")
    rows, groups = fixed_sample(panel, original)
    del original
    dates = panel.dates[rows[:, 0]].astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    log(f"fixed sample: {len(rows)} rows, {len(set(dates.tolist()))} dates, {len(groups)} groups")
    views, results, encodings = {}, {}, {}
    for run_dir in run_dirs:
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        arm = run_dir.name.split(f"_s{args.seed}")[0]
        norm_key = hashlib.sha256(json.dumps(meta["normalizer"], sort_keys=True).encode()).hexdigest()
        if norm_key not in views:
            views[norm_key] = _normalized_view(panel, meta["normalizer"], meta["variant_spec"]["feature_set"])
        view = views[norm_key]
        latest = np.where(view.stock_obs[rows[:, 0], rows[:, 1]], view.stock[rows[:, 0], rows[:, 1]], np.nan).astype(np.float64)
        config = PilotConfig(**meta["model_config"])
        targets = {}
        for step in args.steps:
            path = run_dir / f"encoder_step_{step:06d}.pt"
            if not path.exists():
                log(f"{arm}: missing {path.name}")
                continue
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = FutureJEPA(config)
            model.online.load_state_dict(ck["online"], strict=True)
            model.eval().requires_grad_(False)
            started = time.perf_counter()
            feats = encode(model, view, rows, groups)
            log(f"{arm} step {step}: encoded in {time.perf_counter()-started:.1f}s")
            targets[step] = recoverability(feats, dates, latest, years)
            encodings[f"{arm}_{step}"] = feats
        seed_everything(args.seed)
        random_model = FutureJEPA(config).eval().requires_grad_(False)
        feats = encode(random_model, view, rows, groups)
        targets["random"] = recoverability(feats, dates, latest, years)
        results[arm] = targets
    # ---- report --------------------------------------------------------------
    steps = [s for s in args.steps]
    lines = [f"Per-channel recoverability (within-date Spearman, 2019-2021), seed {args.seed}", ""]
    header = f"{'channel':30s}" + "".join(f"{arm:>10s}{'':>9s}" for arm in results) + "   (each arm: @{} -> @{}, then random)".format(*steps[:2])
    lines.append(header)
    table = {}
    for name in STOCK_FEATURES:
        tag = " " if name in FAST else "*"
        line = f"{tag}{name:29s}"
        for arm, t in results.items():
            a = t.get(steps[0], {}).get(name, {}).get("spearman")
            b = t.get(steps[-1], {}).get(name, {}).get("spearman")
            r = t["random"][name]["spearman"]
            line += f" {a if a is None else round(a,3)!s:>6s}->{b if b is None else round(b,3)!s:<6s} r{r:.2f}"
            table.setdefault(name, {})[arm] = {"early": a, "late": b, "random": r}
        lines.append(line)
    lines.append("")
    lines.append("Paired change (late - early) on non-target channels, mean over channels [95% block CI of the per-channel mean]:")
    verdict = {}
    for arm, t in results.items():
        if steps[0] not in t or steps[-1] not in t:
            continue
        deltas, cis = [], []
        for name in STOCK_FEATURES:
            if name in FAST:
                continue
            e, l = t[steps[0]][name], t[steps[-1]][name]
            common, ie, il = np.intersect1d(e["dates"], l["dates"], return_indices=True)
            diff = l["per_date"][il] - e["per_date"][ie]
            deltas.append(float(diff.mean()))
            cis.append(block_ci(diff))
        worst = min(deltas)
        lines.append(f"  {arm:4s} mean {np.mean(deltas):+.3f}  worst channel {worst:+.3f}  channels with CI entirely < 0: {sum(1 for lo, hi in cis if hi < 0)}/{len(cis)}")
        verdict[arm] = {"mean_nontarget_delta": float(np.mean(deltas)), "worst_nontarget_delta": worst,
                        "channels_significantly_worse": sum(1 for lo, hi in cis if hi < 0)}
    report = "\n".join(lines)
    print("\n" + report)
    (out / f"report_seed{args.seed}.md").write_text("# Channel recoverability, seed {}\n\n```\n{}\n```\n".format(args.seed, report))
    json.dump({"seed": args.seed, "steps": steps, "table": table, "verdict": verdict,
               "protocol": "fixed panel every 5th session 2010-2021, 32 common stocks/date, ridge fit 2010-15, alpha 2016-17, refit 2010-17, scored 2019-21"},
              open(out / f"results_seed{args.seed}.json", "w"), indent=1)
    np.savez_compressed(out / f"encodings_seed{args.seed}.npz", rows=rows, dates=dates, **encodings)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
