"""Single-GPU Colab training with successful-update schedules and full resume."""
from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import FutureJEPA, PilotConfig, encoder_diagnostics
from .preflight import native_bf16_supported
from .wandb_logging import channel_projection_norms


@dataclass
class TrainConfig:
    steps: int = 2500
    accumulation_steps: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    clip_grad_norm: float = 1.0
    ema_start: float = 0.99
    ema_end: float = 0.9999
    device: str = "auto"
    precision: str = "auto"  # auto -> CUDA BF16 if supported, otherwise FP16
    seed: int = 42
    checkpoint_every: int = 100
    log_every: int = 50
    context_mask_probability: float = 0.0

    def __post_init__(self):
        if self.steps < 1 or self.accumulation_steps < 1:
            raise ValueError("steps and accumulation_steps must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        if not 0 <= self.context_mask_probability <= 1:
            raise ValueError("context_mask_probability must be in [0,1]")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")


def seed_everything(seed):
    """Call before constructing a model when comparing seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _device_precision(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; connect a Colab GPU runtime")
    precision = config.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and native_bf16_supported(torch, device) else "fp16" if device.type == "cuda" else "fp32"
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("This pilot supports mixed precision on CUDA only")
    if precision == "bf16" and not native_bf16_supported(torch, device):
        raise ValueError("This GPU does not support native BF16; use FP16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    return device, precision, dtype


def batch_to_device(batch, device):
    result = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            value = torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            # non-blocking only on CUDA: on MPS the async copy can read the numpy buffer after it is freed (garbage inputs)
            value = value.to(device=device, non_blocking=(device.type == "cuda"))
            if value.is_floating_point():
                value = value.float()
        result[key] = value
    return result


def _sample(sampler):
    return sampler() if callable(sampler) else sampler.sample()


def _lr_for_update(update, config):
    warmup = max(1, int(config.steps * config.warmup_fraction)) if config.warmup_fraction else 0
    if warmup and update < warmup:
        return config.lr * (update + 1) / warmup
    progress = (update - warmup) / max(1, config.steps - warmup - 1)
    return config.lr * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


def _ema_for_update(update, config):
    progress = update / max(config.steps - 1, 1)
    return config.ema_end - (config.ema_end - config.ema_start) * (1 + math.cos(math.pi * progress)) / 2


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _same_experiment(saved, current):
    """Checkpoint metadata must match except for the source fingerprint: a bug fix between sessions must not
    strand a half-trained year (the resumed state is still exact; the exported encoder records the new code)."""
    strip = lambda m: {k: v for k, v in (m or {}).items() if k != "code_sha256"}
    return strip(saved) == strip(current)


def _checkpoint(model, optimizer, scaler, config, sampler, step, history, metadata):
    sampler_state = sampler.state_dict() if hasattr(sampler, "state_dict") else None
    return {"format_version": 1, "model": model.state_dict(),
            "model_config": asdict(model.config), "train_config": asdict(config),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "optimizer_step": step, "scheduler": {"successful_updates": step},
            "rng": _rng_state(), "sampler": sampler_state,
            "exact_sampler_resume": sampler_state is not None,
            "history": history, "metadata": metadata or {}}


def train_steps(model, sampler, config=None, checkpoint_dir=None, resume_from=None,
                stop_at=None, on_step=None, metadata=None):
    """Train to config.steps (or stop_at) successful optimizer updates.

    The sampler must return a batch dict and should expose state_dict and
    load_state_dict. stop_at allows a 500-update screen using the same planned
    schedule as a 2500-update continuation. Do not change steps on exact resume.
    Checkpoints are trusted local pickle files; never load an untrusted file.
    """
    config = config or TrainConfig()
    if getattr(model, "inference_only", False):
        raise ValueError("Compact encoder exports cannot resume training; use the full training checkpoint")
    device, precision, dtype = _device_precision(config)
    model.to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    step, history = 0, []
    if resume_from:
        saved = torch.load(resume_from, map_location=device, weights_only=False)
        if saved["model_config"] != asdict(model.config):
            raise ValueError("Checkpoint model configuration does not match")
        schedule_fields = ("steps", "accumulation_steps", "lr", "weight_decay", "warmup_fraction",
                           "ema_start", "ema_end", "context_mask_probability", "clip_grad_norm", "seed")
        for field in schedule_fields:
            if saved["train_config"][field] != getattr(config, field):
                raise ValueError(f"Exact resume requires unchanged {field}; start a separately named run")
        if metadata is not None and _same_experiment(saved.get("metadata"), metadata):
            pass  # keep the caller's metadata (it names the code that finishes the run)
        elif metadata is not None:
            raise ValueError("Checkpoint metadata does not match dataset/experiment metadata")
        else:
            metadata = saved.get("metadata", {})
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, history = int(saved["optimizer_step"]), list(saved["history"])
        if saved["sampler"] is not None:
            if not hasattr(sampler, "load_state_dict"):
                raise ValueError("Sampler cannot restore its saved state")
            sampler.load_state_dict(saved["sampler"])
        _restore_rng(saved["rng"])
    else:
        seed_everything(config.seed)
    end = config.steps if stop_at is None else min(stop_at, config.steps)
    if end < step:
        raise ValueError("stop_at precedes the checkpoint step")
    started = time.perf_counter()
    consecutive_skips = 0
    last_checkpoint = None
    stopped_early = False
    while step < end:
        optimizer.zero_grad(set_to_none=True)
        lr = _lr_for_update(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = None
        update_started = time.perf_counter()
        for _ in range(config.accumulation_steps):
            batch = batch_to_device(_sample(sampler), device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
                result = model.loss(batch, config.context_mask_probability)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"Non-finite loss before optimizer update {step+1}")
            scaler.scale(result["loss"] / config.accumulation_steps).backward()
            if totals is None:  # every scalar loss term the objective reports (the walk-forward objective adds aux and per-horizon terms)
                totals = {key: 0.0 for key, value in result.items()
                          if key in ("loss", "prediction_loss", "sigreg_loss") or (key.endswith("_loss") or key.startswith("prediction_loss_")) and torch.is_tensor(value) and value.ndim == 0}
            for key in totals:
                totals[key] += result[key].detach().float().item() / config.accumulation_steps
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.clip_grad_norm)
        old_scale = scaler.get_scale()
        if not scaler.is_enabled() and not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradients before optimizer update {step+1}")
        scaler.step(optimizer)
        scaler.update()
        successful = not scaler.is_enabled() or scaler.get_scale() >= old_scale
        if not successful:
            consecutive_skips += 1
            if consecutive_skips >= 30:
                raise FloatingPointError("30 consecutive mixed-precision overflows; lower the batch size or use FP32")
            continue
        consecutive_skips = 0
        momentum = _ema_for_update(step, config)
        model.update_teacher(momentum)
        step += 1
        row = {"step": step, **totals, "lr": lr, "ema": momentum,
               "gradient_norm": float(gradient_norm),
               "seconds": time.perf_counter() - update_started}
        if config.log_every and (step == 1 or step % config.log_every == 0 or step == end):
            row["encoder"] = encoder_diagnostics(result["context_embeddings"], result["valid_stock_mask"])
            row["proj_norm"] = channel_projection_norms(model)
            print(json.dumps(row), flush=True)
        history.append(row)
        stop_requested = on_step is not None and on_step(row) is False
        if checkpoint_dir and (stop_requested or step == end or (config.checkpoint_every and step % config.checkpoint_every == 0)):
            path = Path(checkpoint_dir) / f"step_{step:06d}.pt"
            _atomic_save(_checkpoint(model, optimizer, scaler, config, sampler, step, history, metadata), path)
            last_checkpoint = str(path)
        if stop_requested:
            stopped_early = step < end
            break
    return {"optimizer_step": step, "history": history, "checkpoint": last_checkpoint,
            "elapsed_seconds": time.perf_counter()-started, "precision": precision,
            "device": str(device), "exact_sampler_resume": hasattr(sampler, "state_dict"),
            "stopped_early": stopped_early, "requested_stop_at": end}


def profile_train(model, sampler, config=None, steps=10, warmup_steps=2):
    """Measure a disposable model copy; restore sampler and all caller RNGs."""
    config = copy.deepcopy(config or TrainConfig())
    config.steps = steps + warmup_steps
    config.checkpoint_every = 0
    config.log_every = 0
    rng = _rng_state()
    sampler_state = copy.deepcopy(sampler.state_dict()) if hasattr(sampler, "state_dict") else None
    temporary_sampler = sampler if sampler_state is not None else copy.deepcopy(sampler)
    device, _, _ = _device_precision(config)
    disposable = copy.deepcopy(model)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = train_steps(disposable, temporary_sampler, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = [row["seconds"] for row in result["history"][warmup_steps:]]
        return {"median_seconds_per_update": float(np.median(times)),
                "mean_seconds_per_update": float(np.mean(times)),
                "profile_updates": steps, "warmup_updates": warmup_steps,
                "peak_allocated_gb": torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else None,
                "precision": result["precision"], "device": str(device),
                "accumulation_steps": config.accumulation_steps}
    finally:
        if sampler_state is not None:
            sampler.load_state_dict(sampler_state)
        _restore_rng(rng)
        del disposable
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_snapshot(path, device="cpu", teacher=False):
    """Load an encoder/training snapshot created by this module (trusted files)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FutureJEPA(PilotConfig(**checkpoint["model_config"]))
    if checkpoint.get("artifact_type") == "encoder":
        model.online.load_state_dict(checkpoint["online"])
        model.teacher.load_state_dict(checkpoint["teacher"])
        # The predictor is intentionally untrained in a compact export. Only
        # encode() is supported; training resumes require the full checkpoint.
        model.inference_only = True
    else:
        model.load_state_dict(checkpoint["model"])
        model.inference_only = False
    model.snapshot_metadata = checkpoint.get("metadata", {})
    if teacher:
        model.online = copy.deepcopy(model.teacher)
    return model.to(device).eval()


def export_encoder(model, path, metadata=None):
    """Portable online+teacher snapshot, with config and experiment provenance."""
    _atomic_save({"format_version": 1, "artifact_type": "encoder",
                  "model_config": asdict(model.config),
                  "online": model.online.state_dict(), "teacher": model.teacher.state_dict(),
                  "metadata": metadata or {}}, path)
    return str(path)
