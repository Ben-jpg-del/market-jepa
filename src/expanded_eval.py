"""Matched-cohort evaluation of models trained on different stock universes.

All models see the same expanded deployment inputs and peer partitions. The
common cohort is a scoring/readout mask, not a second encoder pass with different
peers. Each encoder retains its own training-only normalizer. This explicitly
tests transfer into the expanded deployment context, not only training-set size.
"""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .data import fit_normalizer
from .evaluation import (context_eligibility, evaluate_arrays, make_forward_labels,
                         _json_safe)
from .model import FutureJEPA, PilotConfig
from .runner import source_fingerprint
from .training import load_snapshot, seed_everything


FORMAT_VERSION = 1


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False))
    temporary.replace(path)


def _hash_json(value):
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True).encode()).hexdigest()


def _hash_array(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(callback, phase, **details):
    if callback is not None:
        callback({"phase": phase, **details})


def _crop(panel, last_year):
    result = copy.copy(panel)
    count = int(np.searchsorted(np.asarray(panel.dates, dtype="datetime64[D]"),
                                np.datetime64(f"{last_year}-12-31"), side="right"))
    for name, value in vars(panel).items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(panel.dates):
            setattr(result, name, value[:count])
    return result


def shared_origin_mask(expanded_panel, original_panel, *, context=60, start="2006-01-01"):
    """Align original origin-known eligibility by exact date and symbol.

    No future target/price availability is used. The expanded mask supplies the
    deployment peer pool; the common mask only controls readout fitting/scoring.
    """
    for panel in (expanded_panel, original_panel):
        dates = np.asarray(panel.dates, dtype="datetime64[D]")
        if np.any(np.isnat(dates)) or np.any(dates[1:] <= dates[:-1]):
            raise ValueError("Panel dates must be sorted, unique, and nonmissing")
        if len(set(panel.symbols)) != len(panel.symbols):
            raise ValueError("Panel symbols must be unique")
    expanded = context_eligibility(expanded_panel, context)
    expanded &= (expanded_panel.dates >= np.datetime64(start))[:, None]
    expanded &= expanded_panel.market_obs[:, 0, None]
    original = context_eligibility(original_panel, context)
    original &= original_panel.market_obs[:, 0, None]
    common = np.zeros(expanded.shape, dtype=bool)
    date_lookup = {str(day): i for i, day in enumerate(original_panel.dates)}
    symbol_lookup = {name: k for k, name in enumerate(original_panel.symbols)}
    destination, source = [], []
    for k, symbol in enumerate(expanded_panel.symbols):
        if symbol in symbol_lookup:
            destination.append(k)
            source.append(symbol_lookup[symbol])
    if not destination:
        raise ValueError("Original and expanded panels have no common symbols")
    for t, day in enumerate(expanded_panel.dates):
        other_t = date_lookup.get(str(day))
        if other_t is not None:
            common[t, destination] = original[other_t, source] & expanded[t, destination]
    if not common.any():
        raise ValueError("Original and expanded panels have no common eligible origins")
    return {"common": common, "expanded": expanded}


def _normalized_view(panel, normalizer, feature_set, cutoff_limit="2018-12-31"):
    """Apply the frozen training transform in bounded date chunks.

    The foreign dataset fingerprint is deliberately retained in the normalizer;
    cross-universe provenance is explicit rather than falsely relabelling it.
    ``cutoff_limit=None`` disables the 2018 guard (walk-forward normalizers end at Y-1).
    """
    if cutoff_limit is not None and np.datetime64(normalizer["train_cutoff"]) > np.datetime64(cutoff_limit):
        raise ValueError("Encoder normalizer includes post-2018 data")
    if feature_set not in ("basic", "book"):
        raise ValueError("feature_set must be basic or book")
    out = copy.copy(panel)
    for field, observed_field in (("stock", "stock_obs"), ("market", "market_obs")):
        values, observed = getattr(panel, field), getattr(panel, observed_field)
        spec = normalizer[field]
        median, iqr = np.asarray(spec["median"]), np.asarray(spec["iqr"])
        if (median.shape != (values.shape[-1],) or iqr.shape != median.shape
                or not np.isfinite(median).all() or not np.isfinite(iqr).all() or np.any(iqr <= 0)):
            raise ValueError("Normalizer channels do not match deployment features")
        transformed = np.empty(values.shape, dtype=np.float32)
        for first in range(0, len(values), 32):
            sl = slice(first, first + 32)
            z = np.clip((values[sl] - median) / iqr, -normalizer["clip"], normalizer["clip"])
            transformed[sl] = np.where(observed[sl], z, 0)
        setattr(out, field, transformed)
        setattr(out, observed_field, observed.copy() if feature_set == "basic" else observed)
    if feature_set == "basic":
        out.stock[..., 9:], out.stock_obs[..., 9:] = 0, False
        out.market[..., 2:], out.market_obs[..., 2:] = 0, False
    out.metadata = {**panel.metadata, "normalizer": normalizer}
    return out


def peer_groups(panel, eligible, *, peer_seed=42, max_stocks=32):
    if max_stocks < 2:
        raise ValueError("Peer partitions require at least two stock slots")
    order = sorted(range(len(panel.symbols)),
                   key=lambda k: hashlib.sha256(f"{peer_seed}:{panel.symbols[k]}".encode()).digest())
    groups, records = [], []
    for t in range(len(panel.dates)):
        selected = [k for k in order if eligible[t, k]]
        for start in range(0, len(selected), max_stocks):
            indices = selected[start:start + max_stocks]
            groups.append((t, indices))
            records.append({"date": str(panel.dates[t]), "partition": start // max_stocks,
                            "symbols": [panel.symbols[k] for k in indices]})
    return groups, records


def _open_cache(directory, signature, shape):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    path, valid_path = directory / "features.f32", directory / "valid.bool"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature or manifest["shape"] != list(shape):
            raise ValueError("Cached embedding identity does not match this evaluation")
        expected = int(np.prod(shape)) * 4
        if not path.exists() or path.stat().st_size != expected or not valid_path.exists() or valid_path.stat().st_size != int(np.prod(shape[:-1])):
            raise ValueError("Cached feature files are missing or truncated")
        features = np.memmap(path, mode="r+", dtype=np.float32, shape=shape)
        valid = np.memmap(valid_path, mode="r+", dtype=np.bool_, shape=shape[:-1])
    else:
        required = int(np.prod(shape)) * 4 + int(np.prod(shape[:-1]))
        if shutil.disk_usage(directory).free < required * 1.1 + 64 * 1024**2:
            raise RuntimeError(f"Insufficient local scratch disk: this cache needs {required / 1024**3:.1f} GiB")
        # New files are zero-filled by the filesystem. Rows are never read until
        # their independently persisted valid mask is true; no dense NaN fill.
        features = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
        valid = np.memmap(valid_path, mode="w+", dtype=np.bool_, shape=shape[:-1])
        manifest = {"signature": signature, "shape": list(shape), "cursor": 0, "complete": False}
        _write_json(manifest_path, manifest)
    return features, valid, manifest, manifest_path


def cached_encoder_features(panel, model, eligible, groups, cache_dir, signature, *,
                            device="cpu", batch_size=16, max_stocks=32,
                            on_progress=None, phase="encoder", checkpoint_batches=10):
    """Resume exact deterministic peer groups after a persisted batch cursor."""
    if batch_size < 1 or checkpoint_batches < 1:
        raise ValueError("Batch and checkpoint sizes must be positive")
    width, context = model.config.width, model.config.context_days
    shape = (*eligible.shape, 2 * width)
    features, valid, manifest, manifest_path = _open_cache(cache_dir, signature, shape)
    if manifest["complete"]:
        if not np.array_equal(valid, eligible):
            raise ValueError("Completed cache has a different eligible stock-date mask")
        _progress(on_progress, phase, status="cached", groups=len(groups))
        return features, {"cache_hit": True, "groups": len(groups), "seconds": 0.0}
    start = int(manifest["cursor"])
    if not 0 <= start <= len(groups):
        raise ValueError("Embedding cache cursor is invalid")
    started = time.perf_counter()
    model.to(device).eval()
    with torch.inference_mode():
        for offset in range(start, len(groups), batch_size):
            items = groups[offset:offset + batch_size]
            b = len(items)
            sx = np.zeros((b, max_stocks, context, panel.stock.shape[-1]), np.float32)
            mx = np.zeros((b, context, panel.market.shape[-1]), np.float32)
            so, mo = np.zeros_like(sx, bool), np.zeros_like(mx, bool)
            sv = np.zeros((b, max_stocks), bool)
            for j, (t, indices) in enumerate(items):
                if t < context - 1:
                    raise ValueError("Eligible origin lacks the configured history")
                window = slice(t - context + 1, t + 1)
                count = len(indices)
                sx[j, :count] = panel.stock[window][:, indices].transpose(1, 0, 2)
                so[j, :count] = panel.stock_obs[window][:, indices].transpose(1, 0, 2)
                mx[j], mo[j], sv[j, :count] = panel.market[window], panel.market_obs[window], True
            result = model.encode(*(torch.as_tensor(value, device=device) for value in (sx, mx, so, mo, sv)))
            stock_z = result["stock_embeddings"].detach().float().cpu().numpy()
            market_z = result["market_embedding"].detach().float().cpu().numpy()
            actual_valid = result["valid_stock_mask"].detach().cpu().numpy()
            if not np.array_equal(actual_valid, sv):
                raise ValueError("Encoder changed the shared eligibility mask; comparisons would be unmatched")
            for j, (t, indices) in enumerate(items):
                count = len(indices)
                encoded = np.concatenate([stock_z[j, :count],
                                          np.broadcast_to(market_z[j], (count, width))], axis=-1)
                if not np.isfinite(encoded).all():
                    raise FloatingPointError("Nonfinite encoder output; no stock silently removed from the comparison")
                features[t, indices], valid[t, indices] = encoded, True
            cursor = offset + len(items)
            should_checkpoint = ((offset - start) // batch_size + 1) % checkpoint_batches == 0 or cursor == len(groups)
            if should_checkpoint:
                features.flush()
                valid.flush()
                manifest.update(cursor=cursor, complete=cursor == len(groups))
                _write_json(manifest_path, manifest)
                # A deadline raised here retains the just-completed groups.
                _progress(on_progress, phase, status="encoding", completed_groups=cursor, total_groups=len(groups))
    if not np.array_equal(valid, eligible):
        raise ValueError("Encoder did not produce every shared eligible origin")
    return features, {"cache_hit": False, "resumed_at_group": start, "groups": len(groups),
                      "seconds": time.perf_counter() - started}


def cached_raw_features(panel, normalizer, eligible, cache_dir, signature, *, on_progress=None, context=60):
    """66-channel latest/mean/availability baseline with O(K*channels) state.

    Rolling sums use float64 and output float32. All models share this baseline
    normalization, fitted to the expanded training universe through 2018 only.
    """
    shape = (*eligible.shape, 3 * (panel.stock.shape[-1] + panel.market.shape[-1]))
    features, valid, manifest, manifest_path = _open_cache(cache_dir, signature, shape)
    if manifest["complete"]:
        if not np.array_equal(valid, eligible):
            raise ValueError("Raw baseline cache has the wrong cohort")
        return features
    view = _normalized_view(panel, normalizer, "book")
    # Recreate the small rolling state in the same order on resume, without
    # recomputing or overwriting already committed feature rows.
    stock_sum = np.zeros(view.stock.shape[1:], dtype=np.float64)
    stock_count = np.zeros(view.stock.shape[1:], dtype=np.int64)
    market_sum = np.zeros(view.market.shape[1:], dtype=np.float64)
    market_count = np.zeros(view.market.shape[1:], dtype=np.int64)
    cursor = int(manifest["cursor"])
    for t in range(len(view.dates)):
        stock_sum += np.where(view.stock_obs[t], view.stock[t], 0)
        stock_count += view.stock_obs[t]
        market_sum += np.where(view.market_obs[t], view.market[t], 0)
        market_count += view.market_obs[t]
        if t >= context:
            stock_sum -= np.where(view.stock_obs[t - context], view.stock[t - context], 0)
            stock_count -= view.stock_obs[t - context]
            market_sum -= np.where(view.market_obs[t - context], view.market[t - context], 0)
            market_count -= view.market_obs[t - context]
        if t < cursor:
            continue
        chosen = np.flatnonzero(eligible[t])
        if len(chosen):
            sc = stock_count[chosen]
            sm = np.divide(stock_sum[chosen], sc, out=np.full(sc.shape, np.nan), where=sc > 0)
            mm = np.divide(market_sum, market_count, out=np.full(market_count.shape, np.nan), where=market_count > 0)
            sf = np.concatenate([np.where(view.stock_obs[t, chosen], view.stock[t, chosen], np.nan), sm, sc / context], axis=-1)
            mf = np.concatenate([np.where(view.market_obs[t], view.market[t], np.nan), mm, market_count / context])
            features[t, chosen] = np.concatenate([sf, np.broadcast_to(mf, (len(chosen), len(mf)))], axis=-1)
            valid[t, chosen] = True
        if (t + 1) % 100 == 0 or t + 1 == len(view.dates):
            features.flush()
            valid.flush()
            manifest.update(cursor=t + 1, complete=t + 1 == len(view.dates))
            _write_json(manifest_path, manifest)
            _progress(on_progress, "raw_features", completed_dates=t + 1, total_dates=len(view.dates))
    if not np.array_equal(valid, eligible):
        raise ValueError("Raw baseline did not cover the shared eligibility mask")
    return features


def _basic_raw_view(full, directory, signature, eligible):
    # The first nine stock and first two market channels retain exactly the
    # same rolling statistics. Mask book-only columns without another encoder or
    # rolling-feature pass. Slice layout follows raw_feature_baseline().
    result, valid, manifest, manifest_path = _open_cache(directory, signature, full.shape)
    if manifest["complete"]:
        return result
    for first in range(int(manifest["cursor"]), len(full), 32):
        stop = min(first + 32, len(full))
        result[first:stop] = full[first:stop]
        for lo, hi in ((9, 14), (23, 28), (44, 50), (52, 58)):
            result[first:stop, :, lo:hi] = np.nan
        result[first:stop, :, 37:42] = 0
        result[first:stop, :, 60:66] = 0
        valid[first:stop] = eligible[first:stop]
        result.flush()
        valid.flush()
        manifest.update(cursor=stop, complete=stop == len(full))
        _write_json(manifest_path, manifest)
    return result


def _save_evaluation(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    predictions = directory / "predictions.csv.gz"
    temporary = predictions.with_suffix(".gz.tmp")
    fields = ["model", "origin_index", "origin_date", "entry_date", "exit_date", "symbol", "prediction", "target", "cohort"]
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["predictions"])
    temporary.replace(predictions)
    report = result["report"]
    report["artifacts"] = {"predictions": str(predictions)}
    _write_json(directory / "evaluation_report.json", report)
    return report


def evaluate_expanded_run(expanded_panel, original_panel, run_dir, *, snapshot=None,
                          output_dir=None, device="auto", years=(2019, 2020, 2021),
                          primary_horizon=None, horizons=None, peer_seed=42, batch_size=16,
                          bootstrap_samples=500, on_progress=None, cache_dir=None,
                          keep_trained_cache=False, cohort_names=("common", "expanded")):
    """Evaluate common and full cohorts from one cached embedding pass/model.

    Results are report['results'][str(horizon)]['common'|'expanded']['metrics'].
    A/F/B/C default to five-day readouts; E defaults to its one-day target. A/F
    pretraining predicts historical representations, not a five-day future. Historical
    2022–2025 input and labels are rejected from this development workflow.
    """
    years = tuple(years)
    if not years or min(years) < 2019 or max(years) > 2021:
        raise ValueError("This expansion workflow evaluates development 2019–2021 only")
    if expanded_panel.metadata.get("normalizer"):
        raise ValueError("Pass the raw expanded panel; each model must use its own frozen training normalizer")
    directory = Path(run_dir)
    metadata = json.loads((directory / "run_metadata.json").read_text())
    source = source_fingerprint()
    if metadata["code"] != source:
        raise ValueError("Source differs from training; use the saved experiment source")
    original_fp, expanded_fp = original_panel.metadata["dataset_fingerprint"], expanded_panel.metadata["dataset_fingerprint"]
    training_source_fp = metadata.get("panel_metadata", {}).get("source_dataset_fingerprint", metadata["dataset_fingerprint"])
    if training_source_fp not in (original_fp, expanded_fp):
        raise ValueError("Model was trained on neither supplied research panel")
    if metadata["normalizer"]["dataset_fingerprint"] != metadata["dataset_fingerprint"]:
        raise ValueError("Saved normalizer does not belong to the derived training dataset")
    if np.datetime64(metadata["encoder_train_cutoff"]) > np.datetime64("2018-12-31"):
        raise ValueError("Encoder was trained after the permitted cutoff")
    feature_set = metadata["variant_spec"]["feature_set"]
    planned_readout = metadata.get("primary_readout_horizon", metadata["variant_spec"]["horizon"])
    primary_horizon = planned_readout if primary_horizon is None else int(primary_horizon)
    if primary_horizon != planned_readout:
        raise ValueError("Primary readout must match the planned readout horizon; use horizons for explicitly secondary transfer")
    horizons = (primary_horizon,) if horizons is None else tuple(dict.fromkeys(int(h) for h in horizons))
    if primary_horizon not in horizons or any(h not in (1, 5, 20) for h in horizons):
        raise ValueError("Include the primary horizon; supported readouts are 1, 5, and 20 sessions")
    if snapshot is None:
        candidates = sorted(directory.glob("encoder_step_*.pt"))
        if not candidates:
            raise FileNotFoundError("No saved encoder snapshot")
        snapshot = candidates[-1]
    snapshot = Path(snapshot)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    encoder = load_snapshot(snapshot, device=device)
    for field in ("dataset_fingerprint", "normalizer", "variant", "model_config", "code", "seed"):
        if encoder.snapshot_metadata.get(field) != metadata[field]:
            raise ValueError(f"Snapshot {field} differs from its run metadata")
    snapshot_hash = _file_hash(snapshot)
    result_root = Path(output_dir) if output_dir is not None else directory / f"expanded_evaluation_step_{encoder.snapshot_metadata['optimizer_step']:06d}"
    result_root.mkdir(parents=True, exist_ok=True)
    cohort_names = tuple(dict.fromkeys(cohort_names))
    if not cohort_names or any(name not in ("common", "expanded") for name in cohort_names):
        raise ValueError("cohort_names must be a nonempty subset of common/expanded")
    identity_fields = {"format": FORMAT_VERSION, "source": source,
        "snapshot": snapshot_hash, "original_dataset": original_fp, "expanded_dataset": expanded_fp,
        "years": years, "horizons": horizons, "primary_horizon": primary_horizon,
        "peer_seed": peer_seed, "bootstrap_samples": bootstrap_samples}
    if cohort_names != ("common", "expanded"):
        identity_fields["cohorts"] = cohort_names
    request_identity = _hash_json(identity_fields)
    final_report_path = result_root / "comparison_report.json"
    if final_report_path.exists():
        previous = json.loads(final_report_path.read_text())
        if previous.get("request_identity") != request_identity:
            raise ValueError("Output folder belongs to another comparison; choose another output_dir")
        if previous.get("status") == "complete":
            for horizon in horizons:
                for cohort in cohort_names:
                    prediction_path = previous["results"][str(horizon)][cohort]["artifacts"]["predictions"]
                    if not Path(prediction_path).exists():
                        raise ValueError("Completed comparison is missing its predictions artifact")
            _progress(on_progress, "completed_comparison", status="cached")
            return {**previous, "evaluation_cache_hit": True}
    context = encoder.config.context_days
    expanded = _crop(expanded_panel, max(years))
    original = _crop(original_panel, max(years))
    cohorts = shared_origin_mask(expanded, original, context=context)
    eligible = cohorts["expanded"]
    cohorts = {name: cohorts[name] for name in cohort_names}
    alignment = metadata.get("panel_metadata", {}).get("training_date_alignment")
    shared_training_dates = None
    if alignment:
        def training_dates(panel):
            valid = context_eligibility(panel, context).sum(axis=1) >= 2
            valid &= panel.market_obs[:, 0]
            valid &= (panel.dates >= np.datetime64("2006-01-01")) & (panel.dates <= np.datetime64("2018-12-31"))
            return panel.dates[valid]
        shared_training_dates = np.intersect1d(training_dates(expanded), training_dates(original))
        digest = hashlib.sha256(shared_training_dates.astype("int64").tobytes()).hexdigest()
        if digest != alignment["shared_origin_date_sha256"] or len(shared_training_dates) != alignment["count"]:
            raise ValueError("Saved matched training calendar differs from the raw source panels")
    max_stocks = int(metadata["inference_peer_partition_size"])
    groups, peers = peer_groups(expanded, eligible, peer_seed=peer_seed, max_stocks=max_stocks)
    cache = Path(cache_dir) if cache_dir is not None else directory.parent / "expanded_evaluation_cache"
    cache.mkdir(parents=True, exist_ok=True)
    base = {"format": FORMAT_VERSION, "source": source, "expanded_dataset": expanded_fp,
            "last_input_date": str(expanded.dates[-1]), "readout_start": "2006-01-01",
            "eligibility": _hash_array(eligible), "symbols": list(expanded.symbols),
            "context": context, "peer_seed": peer_seed, "max_stocks": max_stocks,
            "training_date_alignment": alignment}
    peer_identity = _hash_json(base)
    peer_path = cache / f"peer_partitions_{peer_identity}.json"
    if not peer_path.exists():
        _write_json(peer_path, peers)
    view = _normalized_view(expanded, metadata["normalizer"], feature_set)
    model_identity = {**base, "normalizer": metadata["normalizer"], "feature_set": feature_set,
                      "model_config": metadata["model_config"]}
    trained_id = _hash_json({**model_identity, "checkpoint_sha256": snapshot_hash})
    trained_cache_dir = cache / f"trained_{trained_id}"
    trained, trained_timing = cached_encoder_features(
        view, encoder, eligible, groups, trained_cache_dir, trained_id,
        device=device, batch_size=batch_size, max_stocks=max_stocks,
        on_progress=on_progress, phase="trained_encoder")
    encoder.to("cpu")
    seed_everything(metadata["seed"])
    random_encoder = FutureJEPA(PilotConfig(**metadata["model_config"]))
    random_id = _hash_json({**model_identity, "random_seed": metadata["seed"]})
    random, random_timing = cached_encoder_features(
        view, random_encoder, eligible, groups, cache / f"random_{random_id}", random_id,
        device=device, batch_size=batch_size, max_stocks=max_stocks,
        on_progress=on_progress, phase="random_encoder")
    random_encoder.to("cpu")
    del view
    # A common canonical raw baseline removes model-specific normalization as
    # a source of raw-baseline differences between narrow/wide trained models.
    normalizer_path = cache / f"expanded_raw_normalizer_{_hash_json(base)}.json"
    if normalizer_path.exists():
        raw_normalizer = json.loads(normalizer_path.read_text())
    else:
        _progress(on_progress, "raw_normalizer", status="fitting_through_2018")
        raw_fit_panel = copy.copy(expanded)
        train_start = "2006-01-01"
        if shared_training_dates is not None:
            raw_fit_panel.universe = expanded.universe.copy()
            train_rows = (expanded.dates >= np.datetime64("2006-01-01")) & (expanded.dates <= np.datetime64("2018-12-31"))
            raw_fit_panel.universe[train_rows & ~np.isin(expanded.dates, shared_training_dates)] = False
            train_start = alignment["start"]
        raw_normalizer = fit_normalizer(raw_fit_panel, cutoff="2018-12-31", start=train_start)
        _write_json(normalizer_path, raw_normalizer)
    raw_id = _hash_json({**base, "raw_normalizer": raw_normalizer, "layout": "latest_mean60_fraction_66"})
    raw_full = cached_raw_features(expanded, raw_normalizer, eligible, cache / f"raw_book_{raw_id}", raw_id,
                                   on_progress=on_progress, context=context)
    raw = (_basic_raw_view(raw_full, cache / f"raw_basic_{raw_id}", raw_id + "_basic", eligible)
           if feature_set == "basic" else raw_full)
    features = {"trained_encoder": trained, "random_encoder": random, "raw_features": raw}
    evaluation_identity = _hash_json({"trained": trained_id, "random": random_id, "raw": raw_id,
                                     "feature_set": feature_set, "years": years,
                                     "common_mask": _hash_array(cohorts["common"]), "horizons": horizons,
                                     "bootstrap_samples": bootstrap_samples, "bootstrap_seed": peer_seed})
    manifest_path = result_root / "comparison_identity.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != evaluation_identity:
        raise ValueError("Output folder belongs to a different comparison; choose another output_dir")
    _write_json(manifest_path, {"identity": evaluation_identity})
    report = {"schema_version": 1, "status": "in_progress", "request_identity": request_identity,
              "evaluation_cache_hit": False, "variant": metadata["variant"],
              "training_universe": "original" if training_source_fp == original_fp else "expanded",
              "training_dataset_fingerprint": metadata["dataset_fingerprint"],
              "training_source_dataset_fingerprint": training_source_fp,
              "training_date_alignment": alignment,
              "encoder_snapshot": str(snapshot), "optimizer_step": encoder.snapshot_metadata["optimizer_step"],
              "primary_horizon": primary_horizon, "development_years": list(years), "results": {},
              "pretraining": {"objective": "historical_masked_block" if metadata["variant_spec"]["historical"] else "future_block",
                              "future_horizon": None if metadata["variant_spec"]["historical"] else metadata["variant_spec"]["horizon"],
                              "historical_context_days": context if metadata["variant_spec"]["historical"] else None,
                              "historical_hidden_days": 10 if metadata["variant_spec"]["historical"] else None,
                              "feature_set": feature_set},
              "primary_readout_horizon": primary_horizon,
              "readout_horizons": list(horizons),
              "prespecified_secondary_readout_horizons": [1] if metadata["variant_spec"]["historical"] and 1 in horizons else [],
              "original_dataset_fingerprint": original_fp, "expanded_dataset_fingerprint": expanded_fp,
              "peer_partitions": str(peer_path), "peer_seed": peer_seed, "peer_partition_size": max_stocks,
              "inference": {"trained": trained_timing, "random": random_timing,
                            "reused_between_common_and_expanded": True,
                            "reused_between_readout_horizons": True, "storage": "persistent local memmaps"},
              "normalization": {"encoder": "each model's own frozen training normalizer",
                                "random": "same normalizer and feature mask as its trained encoder",
                                "raw": "shared expanded-universe normalizer fitted through 2018-12-31"},
              "historical_evaluation_consulted": False, "point_in_time_verified": False,
              "limitations": [
                  "All models are transferred to identical expanded deployment inputs and peers; common cohort masks do not restore original-only peer context.",
                  "Expanded JANUS fractiles differ from original-universe fractiles. Training sample count, normalized input distribution, and deployment context are coupled.",
                  "Readout fitting and scoring use the same origin mask within each cohort for trained, random, and raw models; future missingness never determines encoder eligibility.",
                  "Raw-feature normalization is shared across models; trained/random transforms retain their own training-universe statistics.",
                  "Membership, ticker continuity, delisting outcomes and adjustment conventions remain unverified; no alpha claim.",
                  "Sector, size and liquidity subgroup labels are not fabricated; those subgroup analyses require point-in-time metadata.",
              ]}
    # Return scores for exact common stock-date origins and the full deployment
    # set, without repeating trained/random inference for either mask/horizon.
    for horizon in horizons:
        labels, entry_dates, ends = make_forward_labels(expanded, horizon)
        report["results"][str(horizon)] = {}
        for cohort_name, mask in cohorts.items():
            target_dir = result_root / f"h{horizon}_{cohort_name}"
            report_path = target_dir / "evaluation_report.json"
            _progress(on_progress, "readout", cohort=cohort_name, horizon=horizon,
                      status="cached" if report_path.exists() else "fitting")
            if report_path.exists():
                cohort_report = json.loads(report_path.read_text())
            else:
                evaluated = evaluate_arrays(features, labels, expanded.dates,
                    label_end_dates=ends, entry_dates=entry_dates, eligible=mask,
                    symbols=expanded.symbols, years=years, horizon=horizon, purge=20,
                    bootstrap_samples=bootstrap_samples, seed=peer_seed)
                evaluated["report"]["cohort"] = {
                    "name": cohort_name, "origin_mask_sha256": _hash_array(mask),
                    "available_target_mask_sha256": _hash_array(mask & np.isfinite(labels)),
                    "eligible_stock_dates_all_years": int(mask.sum()),
                    "development_stock_dates": int((mask & np.isin(expanded.dates.astype("datetime64[Y]").astype(int) + 1970, years)[:, None]).sum()),
                    "label_source": "same expanded-panel next-open to horizon-close stock-minus-SPY returns",
                    "peer_context": "full expanded eligible set partitioned deterministically into groups of 32",
                }
                cohort_report = _save_evaluation(evaluated, target_dir)
                del evaluated  # Do not retain millions of prediction dictionaries between cohorts.
            report["results"][str(horizon)][cohort_name] = cohort_report
            _write_json(result_root / "comparison_report.json", report)
            _progress(on_progress, "readout", cohort=cohort_name, horizon=horizon, status="complete")
    report["evaluation_dir"] = str(result_root)
    report["status"] = "complete"
    _write_json(final_report_path, report)
    if not keep_trained_cache:
        # Only our generated checkpoint-specific cache is deleted, after all
        # reports/CSV files are durable. Model checkpoints and shared baseline
        # caches remain. Incomplete evaluations never reach this cleanup.
        del features["trained_encoder"]
        trained.flush()
        del trained
        try:
            shutil.rmtree(trained_cache_dir)
            report["inference"]["trained_cache"] = "reclaimed after complete reports; checkpoint retained"
        except OSError:
            report["inference"]["trained_cache"] = "retained because cleanup was unavailable"
    else:
        report["inference"]["trained_cache"] = "retained by request"
    _write_json(final_report_path, report)
    return report
