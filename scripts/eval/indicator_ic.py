"""Standalone daily cross-sectional rank IC of every input indicator, by period and horizon.

Original 426-symbol panel (2003-2025), its own origin-eligible universe (128 names/day), SPY-relative
forward return from the next open to the h-th close. Rank IC is invariant to monotonic transforms, so
raw channel values are used directly. Families (fundamentals, events) from the *_orig2025 builds.
Output: outputs/indicator-ic/indicator_ic.json and a printed table.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel
from src.evaluation import context_eligibility, make_forward_labels

def daily_rank_ic(x, y, mask):
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        m = mask[t] & np.isfinite(x[t]) & np.isfinite(y[t])
        if m.sum() < 20: continue
        a = np.argsort(np.argsort(x[t][m])).astype(float); b = np.argsort(np.argsort(y[t][m])).astype(float)
        a -= a.mean(); b -= b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
        if d > 0: out[t] = (a*b).sum()/d
    return out

panel = load_panel(ROOT / "work/fjepa-cloud-data/panel_raw.npz")
dates = panel.dates.astype("datetime64[D]"); years = dates.astype("datetime64[Y]").astype(int) + 1970
elig = context_eligibility(panel, 60) & (dates >= np.datetime64("2006-01-01"))[:, None]
names = list(panel.metadata["stock_features"])
chans = {n: np.where(panel.stock_obs[..., i], panel.stock[..., i], np.nan) for i, n in enumerate(names)}
fz = np.load(ROOT / "work/fjepa-families/fundamentals_channels_orig2025.npz")
for i, n in enumerate(fz["fundamental_names"]): chans[str(n)] = np.where(fz["fundamentals_obs"][..., i], fz["fundamentals"][..., i], np.nan)
ez = np.load(ROOT / "work/fjepa-families/family_channels_orig2025.npz")
for i, n in enumerate(ez["event_names"]): chans[str(n)] = np.where(ez["events_obs"][..., i], ez["events"][..., i], np.nan)
periods = {"2006-14": (2006, 2014), "2015-18": (2015, 2018), "2019-21": (2019, 2021), "2022-25": (2022, 2025)}
res = {}
for h in (5, 20):
    y, _, _ = make_forward_labels(panel, h)
    for n, x in chans.items():
        d = daily_rank_ic(x, y, elig)
        row = {}
        for p, (a, b) in periods.items():
            s = (years >= a) & (years <= b) & np.isfinite(d)
            row[p] = float(np.nanmean(d[s])) if s.any() else None
        row["2006-25"] = float(np.nanmean(d[(years >= 2006) & np.isfinite(d)]))
        res.setdefault(n, {})[f"h{h}"] = row
# redundancy among the 14 stock indicators: average cross-sectional rank correlation, effective number
sel = np.flatnonzero((years >= 2006) & (years <= 2018))[::5]
C = np.zeros((len(names), len(names))); cnt = 0
for t in sel:
    m = elig[t]
    X = np.stack([chans[n][t][m] for n in names], 1)
    ok = np.isfinite(X).all(1)
    if ok.sum() < 30: continue
    R = np.argsort(np.argsort(X[ok], 0), 0).astype(float)
    C += np.corrcoef(R.T); cnt += 1
C /= cnt
ev = np.clip(np.linalg.eigvalsh(C), 0, None); eff = float(ev.sum()**2 / (ev**2).sum())
out = {"panel": "original 426-symbol (128/day)", "periods": periods, "ic": res,
       "stock_indicator_corr": {"names": names, "matrix": C.round(3).tolist()}, "effective_independent_stock_indicators": eff}
(ROOT / "outputs/indicator-ic/indicator_ic.json").write_text(json.dumps(out, indent=1))
print("effective independent stock indicators (participation ratio):", round(eff, 1))
for h in ("h5", "h20"):
    print(f"\n{h}: {'indicator':32s} " + " ".join(f"{p:>8s}" for p in list(periods) + ["2006-25"]))
    for n in res:
        r = res[n][h]; print(f"    {n:32s} " + " ".join(f"{(r[p] if r[p] is not None else float('nan')):+8.4f}" for p in list(periods) + ["2006-25"]))
