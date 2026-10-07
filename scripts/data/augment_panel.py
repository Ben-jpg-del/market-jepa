"""Create the extended expanded panel: base stock channels + event family, base market channels + macro family.

Reads work/fjepa-expanded-data/panel_raw.npz and work/fjepa-families/family_channels.npz, checks
date/symbol alignment, and writes work/fjepa-expanded-data/panel_raw_ext.npz with updated metadata
(channel names, family notes, new dataset_fingerprint derived from the base fingerprint and the
family arrays). The base panel is untouched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_panel, save_panel  # noqa: E402
from src.features import EVENT_FEATURES, FUNDAMENTAL_FEATURES, MACRO_FEATURES  # noqa: E402


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-fundamentals", action="store_true",
                        help="also append the 7 fundamentals channels (work/fjepa-families/fundamentals_channels.npz) -> panel_raw_full.npz")
    parser.add_argument("--macro-channels", type=int, default=None,
                        help="use only the first N macro channels (12 reproduces the 2026-09-19 Colab run panels byte-for-byte)")
    parser.add_argument("--suffix", default="", help="output name suffix, e.g. _m15")
    args = parser.parse_args(argv)
    panel = load_panel(ROOT / "work/fjepa-expanded-data/panel_raw.npz")
    fam = np.load(ROOT / "work/fjepa-families/family_channels.npz", allow_pickle=False)
    assert np.array_equal(fam["dates"].astype("datetime64[D]"), panel.dates.astype("datetime64[D]")), "date mismatch"
    assert list(fam["symbols"]) == list(panel.symbols), "symbol mismatch"
    assert tuple(fam["event_names"]) == EVENT_FEATURES and tuple(fam["macro_names"]) == MACRO_FEATURES
    n_macro = args.macro_channels or len(MACRO_FEATURES)
    macro_names = list(MACRO_FEATURES[:n_macro])
    macro_arr = np.ascontiguousarray(fam["macro"][:, :n_macro]); macro_obs_arr = np.ascontiguousarray(fam["macro_obs"][:, :n_macro])
    stock = np.concatenate([panel.stock, fam["events"].astype(np.float32)], axis=-1)
    stock_obs = np.concatenate([panel.stock_obs, fam["events_obs"]], axis=-1)
    stock_names = list(panel.metadata["stock_features"]) + list(EVENT_FEATURES)
    family_index = {"events": list(range(panel.stock.shape[-1], stock.shape[-1]))}
    extra_digest, extra_notes = [], {}
    if args.with_fundamentals:
        fund = np.load(ROOT / "work/fjepa-families/fundamentals_channels.npz", allow_pickle=False)
        assert np.array_equal(fund["dates"].astype("datetime64[D]"), panel.dates.astype("datetime64[D]")) and list(fund["symbols"]) == list(panel.symbols)
        assert tuple(fund["fundamental_names"]) == FUNDAMENTAL_FEATURES
        first = stock.shape[-1]
        stock = np.concatenate([stock, fund["fundamentals"].astype(np.float32)], axis=-1)
        stock_obs = np.concatenate([stock_obs, fund["fundamentals_obs"]], axis=-1)
        stock_names += list(FUNDAMENTAL_FEATURES)
        family_index["fundamentals"] = list(range(first, stock.shape[-1]))
        extra_digest = [fund["fundamentals"], fund["fundamentals_obs"]]
        extra_notes["fundamentals"] = ("Point-in-time fundamentals from FMP quarterly statements, usable from the first session after the SEC acceptance date. "
                                       "Trailing-4Q flows; market cap = restated shares x adjusted close. See scripts/data/build_fundamentals_channels.py.")
    market = np.concatenate([panel.market, macro_arr.astype(np.float32)], axis=-1)
    market_obs = np.concatenate([panel.market_obs, macro_obs_arr], axis=-1)
    digest = hashlib.sha256()
    digest.update(panel.metadata["dataset_fingerprint"].encode())
    for arr in [fam["events"], fam["events_obs"], macro_arr, macro_obs_arr] + extra_digest:
        arr = np.ascontiguousarray(arr)
        digest.update(hashlib.sha256(arr.tobytes()).hexdigest().encode())
    meta = {**panel.metadata,
            "stock_features": stock_names,
            "market_features": list(panel.metadata["market_features"]) + macro_names,
            "feature_notes": {**panel.metadata.get("feature_notes", {}),
                              "events": "Earnings-event family from FMP `earnings` (announcement date, EPS/revenue actual vs estimate). Known from the session after the announcement date. SUE = surprise / std of last 8 surprises (>=4 required), clipped ±5. Expected next report = last + 91 calendar days (no future dates used). Days since capped at 365; revenue surprise clipped ±20%. Abnormal return = SPY-relative log return over the 2-session window around the announcement; drift = SPY-relative log return since that window.",
                              **extra_notes, "macro": "Market-priced macro state as of the close: 10y change (20d), 2s10s slope and its 20d change, 3m bill 60d change, LQD-minus-TLT 20d return, log VIX and its 20d change, 20d log returns of oil (CLUSD), dollar (DX-Y.NYB), gold (GCUSD), IWM-minus-SPY, and 60d SPX return. No release-lagged statistics are used."},
            "base_dataset_fingerprint": panel.metadata["dataset_fingerprint"],
            "dataset_fingerprint": digest.hexdigest(),
            "families": {"stock": family_index,
                         "market": {"macro": list(range(panel.market.shape[-1], market.shape[-1]))}}}
    ext = replace(panel, stock=stock, stock_obs=stock_obs, market=market, market_obs=market_obs, metadata=meta)
    out = ROOT / "work/fjepa-expanded-data" / (("panel_raw_full" if args.with_fundamentals else "panel_raw_ext") + args.suffix + ".npz")
    save_panel(ext, out)
    check = load_panel(out)
    print(json.dumps({"path": str(out), "stock_channels": check.stock.shape[-1], "market_channels": check.market.shape[-1],
                      "dates": len(check.dates), "symbols": len(check.symbols), "fingerprint": meta["dataset_fingerprint"][:16],
                      "size_mb": round(out.stat().st_size / 1e6, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
