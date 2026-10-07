"""Matched pilot experiments, resumable execution, and frozen evaluation.

These helpers never download data or start jobs on import. A run consumes an
explicit MarketPanel and writes only inside its explicitly selected run folder.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .data import DateBatchSampler, fit_normalizer, normalize_panel
from .evaluation import evaluate_encoder
from .model import FutureJEPA, PilotConfig
from .training import (TrainConfig, export_encoder, load_snapshot, profile_train,
                       seed_everything, train_steps)


VARIANTS = {
    "A": {"description": "Historical masked-block control, basic features", "feature_set": "basic", "horizon": 5, "historical": True, "context_mask_probability": 0.0},
    "B": {"description": "Future five-day block, basic features", "feature_set": "basic", "horizon": 5, "historical": False, "context_mask_probability": 0.0},
    "C": {"description": "Future five-day block, book features", "feature_set": "book", "horizon": 5, "historical": False, "context_mask_probability": 0.0},
    "D": {"description": "Future five-day block, book features, context masking", "feature_set": "book", "horizon": 5, "historical": False, "context_mask_probability": 0.25},
    "E": {"description": "Future one-day block, book features", "feature_set": "book", "horizon": 1, "historical": False, "context_mask_probability": 0.0},
    "F": {"description": "Historical masked-block control, book features", "feature_set": "book", "horizon": 5, "historical": True, "context_mask_probability": 0.0},
    # 2026-09-18 tokenization / target experiment (see outputs/colab-expanded/channel-diagnostic/report.md)
    "F1": {"description": "Historical masked block, book features, ALL channels in the teacher target", "feature_set": "book", "horizon": 5, "historical": True, "context_mask_probability": 0.0, "target_channels": "all"},
    "T1": {"description": "Family tokens; hide one indicator family, predict its teacher summary (T-JEPA analogue)", "feature_set": "book", "horizon": 5, "historical": False, "context_mask_probability": 0.0, "tokenization": "family", "objective": "family_mask"},
    "T2": {"description": "Family tokens with the original historical masked-block, fast-only target", "feature_set": "book", "horizon": 5, "historical": True, "context_mask_probability": 0.0, "tokenization": "family"},
    "T3": {"description": "Per-feature tokens; hide one indicator plus its arithmetic shortcut partners, predict its teacher summary", "feature_set": "book", "horizon": 5, "historical": False, "context_mask_probability": 0.0, "tokenization": "feature", "objective": "feature_mask"},
}
VARIANT_DEFAULTS = {"target_channels": "fast", "tokenization": "bundled", "objective": "block"}

# Preserve the completed pilot's five-arm protocol. New factorial experiments
# explicitly select F instead of silently changing old notebook/job loops.
PILOT_VARIANTS = {name: VARIANTS[name] for name in ("A", "B", "C", "D", "E")}


def _variant(name):
    name = str(name).upper()
    if name not in VARIANTS:
        raise ValueError("variant must be one of " + ", ".join(VARIANTS))
    return name, {**VARIANT_DEFAULTS, **VARIANTS[name]}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (Path, np.datetime64)):
        return str(value)
    return value


def _write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_fingerprint():
    """Record every pilot source file so snapshots identify their implementation."""
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(Path(__file__).parent.glob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def _normalized(panel, normalizer=None):
    fingerprint = panel.metadata.get("dataset_fingerprint")
    if not fingerprint:
        raise ValueError("Panel needs a dataset_fingerprint")
    existing = panel.metadata.get("normalizer")
    if normalizer is not None and normalizer["dataset_fingerprint"] != fingerprint:
        raise ValueError("Saved normalizer belongs to a different dataset")
    if existing:
        if existing["dataset_fingerprint"] != fingerprint:
            raise ValueError("Panel normalization belongs to a different dataset")
        if normalizer is not None and existing != normalizer:
            raise ValueError("Panel normalization differs from the encoder snapshot")
        normalizer = existing
    else:
        normalizer = normalizer or fit_normalizer(panel)
        panel = normalize_panel(panel, normalizer)
    if np.datetime64(normalizer["train_cutoff"]) > np.datetime64("2018-12-31"):
        raise ValueError("Normalizer must be fitted only through 2018-12-31")
    return panel


class VariantSampler:
    """Date sampler plus a historical-masking control with resumable mask RNG."""
    def __init__(self, panel, variant, batch_dates=16, stocks=32, seed=42, context=60):
        self.variant, self.spec = _variant(variant)
        if self.spec["historical"] and context < 15:
            raise ValueError("Historical control needs at least three patches")
        self.settings = {"variant": self.variant, "batch_dates": int(batch_dates),
                         "stocks": int(stocks), "seed": int(seed), "context": int(context),
                         "start": "2006-01-01", "end": "2018-12-31"}
        self.base = DateBatchSampler(panel, batch_dates=batch_dates, stocks_per_date=stocks,
                                     context=context, horizon=self.spec["horizon"], seed=seed,
                                     feature_set=self.spec["feature_set"])
        self.rng = np.random.default_rng(seed + 100003)

    def sample(self):
        batch = self.base.sample()
        if not self.spec["historical"]:
            return batch
        # Copy targets before erasing the last two context patches. Historical
        # A/F use fast-only historical teacher channels; F retains book features
        # everywhere in the student context except the selected hidden blocks.
        # Base-sampler future buffers and target masks are completely replaced.
        if self.spec.get("target_channels", "fast") == "all":
            # F1: the teacher sees every channel of the historical window, so the
            # objective rewards retaining slow/book channels, not only fast ones.
            target_stock, target_obs = batch["stock"].copy(), batch["stock_obs"].copy()
            target_market, target_market_obs = batch["market"].copy(), batch["market_obs"].copy()
        else:
            target_stock = np.zeros_like(batch["stock"])
            target_obs = np.zeros_like(batch["stock_obs"])
            target_stock[..., :4], target_obs[..., :4] = batch["stock"][..., :4], batch["stock_obs"][..., :4]
            target_market, target_market_obs = np.zeros_like(batch["market"]), np.zeros_like(batch["market_obs"])
            target_market[..., 0], target_market_obs[..., 0] = batch["market"][..., 0], batch["market_obs"][..., 0]
        chosen_mask = np.zeros_like(batch["eligibility"])
        for row in range(len(chosen_mask)):
            eligible = np.flatnonzero(batch["eligibility"][row])
            if not len(eligible):
                continue
            chosen = self.rng.choice(eligible, max(1, len(eligible)//4), replace=False)
            chosen_mask[row, chosen] = True
            batch["stock"][row, chosen, -10:] = 0
            batch["stock_obs"][row, chosen, -10:] = False
        batch.update(future_stock=target_stock, future_stock_obs=target_obs,
                     future_market=target_market, future_market_obs=target_market_obs,
                     target_valid=batch["eligibility"].copy(), prediction_valid=chosen_mask)
        return batch

    __call__ = sample

    def state_dict(self):
        return {"settings": dict(self.settings), "base": self.base.state_dict(),
                "mask_rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state["settings"] != self.settings:
            raise ValueError("Variant sampler configuration changed")
        self.base.load_state_dict(state["base"])
        self.rng.bit_generator.state = copy.deepcopy(state["mask_rng"])


def _metadata(panel, name, spec, model_config, train_config, sampler, peer_seed):
    return _json_safe({
        "experiment_schema": 1, "variant": name, "variant_spec": spec,
        "seed": train_config.seed, "dataset_fingerprint": panel.metadata["dataset_fingerprint"],
        "panel_metadata": panel.metadata, "normalizer": panel.metadata["normalizer"],
        "encoder_train_start": "2006-01-01", "encoder_train_cutoff": "2018-12-31",
        "model_config": asdict(model_config), "train_config": asdict(train_config),
        "sampler_config": sampler.settings, "code": source_fingerprint(),
        "stock_features": panel.metadata.get("stock_features"),
        "market_features": panel.metadata.get("market_features"),
        "feature_notes": panel.metadata.get("feature_notes"),
        "inference_peer_seed": int(peer_seed), "inference_peer_partition_size": sampler.settings["stocks"],
        "representation_selection": "online encoder; teacher export retained for explicit later comparison",
        "prediction_target": (
            "teacher summary of one hidden indicator family, same historical window" if spec.get("objective") == "family_mask"
            else "new future fast observations only" if not spec["historical"]
            else "historical observations of ALL channels, final two patches hidden on selected stocks" if spec.get("target_channels") == "all"
            else "historical fast observations, final two patches hidden on selected stocks"),
        "pretraining_objective": ("family_mask" if spec.get("objective") == "family_mask"
                                  else "historical_masked_block" if spec["historical"] else "future_block"),
        "tokenization": spec.get("tokenization", "bundled"),
        "target_channels": spec.get("target_channels", "fast"),
        "pretraining_future_horizon": None if spec["historical"] else spec["horizon"],
        "primary_readout_horizon": spec["horizon"],
        "origin_calendar_lookahead_days": spec["horizon"],
        "point_in_time_verified": bool(panel.metadata.get("point_in_time_verified", False)),
    })


def run_variant(panel, variant, run_dir, seed=42, screen_steps=500, total_steps=2500,
                batch_dates=16, stocks=32, device="auto", resume=False, stop_at=None,
                model_config=None, accumulation_steps=2, log_every=50, on_step=None,
                peer_seed=42, logger=None):
    """Screen to screen_steps; continue with resume=True, stop_at=total_steps.

    The planned total_steps stays fixed, so the resumed optimizer, EMA and LR
    schedules match uninterrupted training. No automatic variant selection occurs.
    ``logger`` is an optional ``WandbTrainingLogger``; it never affects training.
    """
    name, spec = _variant(variant)
    panel = _normalized(panel)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(directory.glob("step_*.pt"))
    if checkpoints and not resume:
        raise FileExistsError("Run folder already contains checkpoints; use resume=True or a new folder")
    if resume and not checkpoints:
        raise FileNotFoundError("No training checkpoint exists in this run folder")
    if not 1 <= screen_steps <= total_steps:
        raise ValueError("Require 1 <= screen_steps <= total_steps")
    model_config = model_config or PilotConfig()
    # The variant decides tokenization/objective; the recorded model_config carries it
    # so evaluation rebuilds an identical random-encoder baseline.
    model_config = replace(model_config, tokenization=spec["tokenization"], objective=spec["objective"])
    config = TrainConfig(steps=total_steps, accumulation_steps=accumulation_steps,
                         seed=seed, device=device, checkpoint_every=min(100, screen_steps),
                         log_every=log_every, context_mask_probability=spec["context_mask_probability"])
    sampler = VariantSampler(panel, name, batch_dates, stocks, seed, model_config.context_days)
    metadata = _metadata(panel, name, spec, model_config, config, sampler, peer_seed)
    if resume:
        previous_metadata = json.loads((directory / "run_metadata.json").read_text())
        if previous_metadata != metadata:
            raise ValueError("Run metadata changed (data, normalization, code, seed, or configuration)")
    else:
        _write_json(metadata, directory / "run_metadata.json")
    seed_everything(seed)
    model = FutureJEPA(model_config)
    stop_at = screen_steps if stop_at is None else stop_at
    if stop_at < 1 or stop_at > total_steps:
        raise ValueError("stop_at must be within the planned training schedule")
    if logger is not None:
        logger.attach(model)
        on_step = logger.wrap(on_step)
    result = train_steps(model, sampler, config, checkpoint_dir=directory,
                         resume_from=checkpoints[-1] if resume else None,
                         stop_at=stop_at, on_step=on_step, metadata=metadata)
    if logger is not None:
        logger.log_training_result(result)
    checkpoint = result["checkpoint"] or (str(checkpoints[-1]) if checkpoints else None)
    export_metadata = {**metadata, "optimizer_step": result["optimizer_step"],
                       "training_checkpoint": Path(checkpoint).name if checkpoint else None}
    export_path = directory / f"encoder_step_{result['optimizer_step']:06d}.pt"
    export_encoder(model, export_path, export_metadata)
    result = {**result, "checkpoint": checkpoint, "encoder_export": str(export_path),
              "metadata": metadata, "run_dir": str(directory)}
    _write_json({key: value for key, value in result.items() if key != "history"}, directory / "run_result.json")
    _write_json(result["history"], directory / "training_metrics.json")
    return result


def profile_variant(panel, variant="C", *, batch_dates=16, stocks=32, seed=42,
                    device="auto", profile_steps=10, warmup_steps=2,
                    model_config=None, accumulation_steps=2):
    """Disposable timing; output is an estimate, not a promised completion time."""
    name, spec = _variant(variant)
    panel = _normalized(panel)
    model_config = model_config or PilotConfig()
    sampler = VariantSampler(panel, name, batch_dates, stocks, seed, model_config.context_days)
    seed_everything(seed)
    model = FutureJEPA(model_config)
    config = TrainConfig(device=device, accumulation_steps=accumulation_steps, seed=seed,
                         context_mask_probability=spec["context_mask_probability"])
    result = profile_train(model, sampler, config, steps=profile_steps, warmup_steps=warmup_steps)
    return {**result, "variant": name, "batch_dates": batch_dates, "stocks": stocks,
            "model_config": asdict(model_config), "synthetic": bool(panel.metadata.get("synthetic", False))}


def budget_from_profiles(profiles, budget_hours=4, screen_steps=500, total_steps=2500,
                         reserve_fraction=0.20):
    """Reduce matched budgets before training: five screens + continuation + seed.

    Uniform update time gives 4*S + 2*T total updates. Distinct variant timings
    use sum(rate)*S + max(future_rate)*(2*T-S), allowing an unknown winning
    future variant. The reserve covers unmeasured overhead/evaluation only
    approximately; data downloads and feature preparation are excluded.
    """
    if budget_hours <= 0 or not 0 <= reserve_fraction < 1 or not 1 <= screen_steps <= total_steps:
        raise ValueError("Invalid time budget, reserve, or step counts")
    if isinstance(profiles, (list, tuple)):
        profiles = {item["variant"]: item for item in profiles}
    if set(profiles) != set(PILOT_VARIANTS):
        raise ValueError("Profile all five variants A-E before allocating matched budgets")
    rates = {name: float(profiles[name]["mean_seconds_per_update"]) for name in PILOT_VARIANTS}
    if any(not np.isfinite(rate) or rate <= 0 for rate in rates.values()):
        raise ValueError("All profiles need finite positive update durations")
    future_rate = max(rates[name] for name in ("B", "C", "D", "E"))
    estimate = lambda s, t: sum(rates.values()) * s + future_rate * (2*t - s)
    available = budget_hours * 3600 * (1 - reserve_fraction)
    requested = estimate(screen_steps, total_steps)
    scale = min(1.0, available/requested)
    s, t = max(1, math.floor(screen_steps*scale)), max(1, math.floor(total_steps*scale))
    t = max(s, t)
    if estimate(s, t) > available:
        raise ValueError("Budget is too small even for one update per variant")
    return {"screen_steps": s, "total_steps": t, "seed_43_total_steps": t,
            "selected_variant_continuation_steps": t-s,
            "total_optimizer_updates": 4*s+2*t, "estimated_training_hours": estimate(s,t)/3600,
            "requested_training_hours": requested/3600, "budget_hours": budget_hours,
            "reserve_fraction": reserve_fraction, "reduced_before_training": scale < 1,
            "seconds_per_update": rates,
            "limitations": "Timing estimate excludes downloads/feature preparation; runtime speed and evaluation overhead can vary."}


def evaluate_variant(panel, run_dir, *, device="auto", years=(2019, 2020, 2021),
                     horizon=5, batch_size=16, bootstrap_samples=500, snapshot=None):
    """Evaluate an explicit saved run with matching features and normalization."""
    directory = Path(run_dir)
    metadata = json.loads((directory / "run_metadata.json").read_text())
    if metadata["dataset_fingerprint"] != panel.metadata.get("dataset_fingerprint"):
        raise ValueError("Evaluation panel differs from training dataset")
    if metadata["code"] != source_fingerprint():
        raise ValueError("Pilot source changed since training; evaluate with the recorded source version")
    panel = _normalized(panel, metadata["normalizer"])
    if snapshot is None:
        available = sorted(directory.glob("encoder_step_*.pt"))
        if not available:
            raise FileNotFoundError("No compact encoder snapshot found")
        snapshot = available[-1]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_snapshot(snapshot, device=device)
    snapshot_meta = encoder.snapshot_metadata
    for field in ("dataset_fingerprint", "normalizer", "variant", "model_config", "code", "seed"):
        if snapshot_meta.get(field) != metadata[field]:
            raise ValueError(f"Snapshot {field} does not match the run")
    seed_everything(metadata["seed"])
    random_encoder = FutureJEPA(PilotConfig(**metadata["model_config"]))
    name, spec = _variant(metadata["variant"])
    years = tuple(years)
    if not years:
        raise ValueError("At least one evaluation year is required")
    output = directory / f"evaluation_step_{snapshot_meta['optimizer_step']:06d}_h{horizon}_{min(years)}_{max(years)}"
    report = evaluate_encoder(panel, encoder, random_encoder=random_encoder, device=device,
                              output_dir=output, horizon=horizon,
                              max_stocks=metadata["inference_peer_partition_size"],
                              batch_size=batch_size, seed=metadata["inference_peer_seed"],
                              bootstrap_samples=bootstrap_samples, normalized=True, years=years,
                              encoder_cutoff=metadata["encoder_train_cutoff"],
                              feature_set=spec["feature_set"])
    return {**report, "variant": name, "encoder_snapshot": str(snapshot), "evaluation_dir": str(output)}
