"""Optional Weights & Biases logging for pilot training and evaluation.

Design rules:
- Never required. Without an API key the logger writes an offline run directory
  (synced later from a logged-in machine); with ``mode="disabled"`` it is a no-op.
  A W&B failure never raises into the training loop.
- The key is read from WANDB_API_KEY or Colab Secrets. It is never printed,
  written to disk by this module, or placed in run config.
- One W&B run per (experiment group, arm, seed). Training stages that resume
  the same arm (500 -> 2,500) and later evaluations reuse the run through a
  deterministic run id, so curves and readouts stay on one page.
- Step axis is the optimizer update (`train/optimizer_step`), not W&B's
  internal counter, so resumed uploads never collide.
- Logged content: loss components, learning rate, EMA, gradient norm, encoder
  rank diagnostics, per-channel input-projection norms, and evaluation scalar
  summaries (rank IC, intervals, coverage, yearly IC). No predictions, no
  trades, no raw data.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .features import MARKET_FEATURES, STOCK_FEATURES, market_feature_names, stock_feature_names

TRAIN_FIELDS = ("loss", "prediction_loss", "sigreg_loss", "aux_loss", "prediction_loss_h5", "prediction_loss_h20", "lr", "ema", "gradient_norm", "seconds")
DIAGNOSTIC_FIELDS = ("effective_rank", "participation_rank", "entropy_rank", "top_eigenvalue_share",
                     "mean_std", "rank_ceiling")
EVAL_MODELS = ("trained_encoder", "random_encoder", "raw_features")
EVAL_COHORTS = ("common", "expanded")
DEFAULT_ENTITY = os.environ.get("FJEPA_WANDB_ENTITY", "bzhou1018-uc-berkeley-electrical-engineering-computer-sc")
DEFAULT_PROJECT = os.environ.get("FJEPA_WANDB_PROJECT", "f-jepa")


def _netrc_key() -> str | None:
    """API key saved by ``wandb login`` in ~/.netrc or ~/_netrc, if any."""
    import netrc
    for name in (".netrc", "_netrc"):
        path = Path.home() / name
        if not path.is_file():
            continue
        try:
            entry = netrc.netrc(str(path)).authenticators("api.wandb.ai")
        except Exception:
            continue
        if entry and entry[2]:
            return entry[2]
    return None


def resolve_wandb_key() -> str | None:
    """WANDB_API_KEY from the environment, else Colab Secrets, else a saved login."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:  # pragma: no cover - only available inside Colab
        from google.colab import userdata  # type: ignore
        key = userdata.get("WANDB_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return _netrc_key()


def deterministic_run_id(**identity: Any) -> str:
    """Stable W&B run id from experiment identity (group, arm, seed, data)."""
    blob = json.dumps(identity, sort_keys=True, default=str).encode()
    return "fjepa-" + hashlib.sha256(blob).hexdigest()[:24]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def channel_projection_norms(model, patch_days: int | None = None) -> dict[str, float]:
    """Frobenius norm of the input-projection columns that read each channel's values.

    Layout per patch: for each day, all channel values then all observation flags.
    Works for the pilot Encoder (``online.stock_projection`` / ``market_projection``).
    """
    encoder = getattr(model, "online", model)
    config = getattr(model, "config", None) or getattr(encoder, "config", None)
    days = patch_days or getattr(config, "patch_days", 5)
    out: dict[str, float] = {}
    projections = getattr(encoder, "family_projections", None)
    if projections is not None:  # family-token encoder: one projection per family
        total = 0.0
        for idx, layer in zip(encoder.family_index, projections):
            weight = layer.weight.detach().float().cpu().numpy()
            per_day = 2 * len(idx)
            if weight.shape[1] != days * per_day:
                continue
            names = stock_feature_names(getattr(config, "stock_features", len(STOCK_FEATURES)))
            for j, channel in enumerate(idx):
                cols = [d * per_day + j for d in range(days)]
                norm = float(np.linalg.norm(weight[:, cols]))
                label = names[channel] if channel < len(names) else f"c{channel}"
                out[f"proj_norm/stock/{label}"] = norm
                total += norm ** 2
        out["proj_norm/stock/_total_values"] = float(np.sqrt(total))
    n_stock = getattr(config, "stock_features", len(STOCK_FEATURES))
    n_market = getattr(config, "market_features", len(MARKET_FEATURES))
    for scope, attr, names, count in (
        ("stock", "stock_projection", stock_feature_names(n_stock, getattr(config, 'feature_layout', 'v1')), n_stock),
        ("market", "market_projection", market_feature_names(n_market, getattr(config, 'feature_layout', 'v1')), n_market),
    ):
        if scope == "stock" and projections is not None:
            continue
        layer = getattr(encoder, attr, None)
        if layer is None:
            continue
        weight = layer.weight.detach().float().cpu().numpy()
        per_day = 2 * count
        if weight.shape[1] != days * per_day:
            continue
        labels = list(names) if len(names) == count else [f"c{j}" for j in range(count)]
        total = 0.0
        for j, label in enumerate(labels):
            cols = [d * per_day + j for d in range(days)]
            norm = float(np.linalg.norm(weight[:, cols]))
            out[f"proj_norm/{scope}/{label}"] = norm
            total += norm ** 2
        out[f"proj_norm/{scope}/_total_values"] = float(np.sqrt(total))
    return out


def evaluation_payload(report: Mapping, checkpoint_step: int | None = None) -> dict[str, float | int | None]:
    """Flatten an expanded-job or pilot evaluation report to scalar summaries."""
    step = checkpoint_step if checkpoint_step is not None else report.get("optimizer_step")
    payload: dict[str, Any] = {}
    if step is not None:
        payload["eval/checkpoint_step"] = int(step)
    results = report.get("results", {})
    for horizon_key, cohorts in results.items():
        horizon = str(horizon_key)
        if not isinstance(cohorts, Mapping):
            continue
        for cohort, block in cohorts.items():
            metrics = block.get("metrics") if isinstance(block, Mapping) else None
            if not isinstance(metrics, Mapping):
                continue
            prefix = f"eval/h{horizon}/{cohort}/"
            for model in EVAL_MODELS:
                data = metrics.get(model, {})
                data = data.get("development", data) if isinstance(data, Mapping) else {}
                if not isinstance(data, Mapping):
                    continue
                base = prefix + model + "/"
                for field in ("mean_daily_rank_ic", "valid_ic_dates", "prediction_rows"):
                    if field in data:
                        payload[base + field] = _number(data[field])
                ci = data.get("rank_ic_95ci_date_blocks")
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    payload[base + "rank_ic_ci_low"], payload[base + "rank_ic_ci_high"] = map(_number, ci)
                for year, ydata in (data.get("yearly") or {}).items():
                    if isinstance(ydata, Mapping) and "mean_rank_ic" in ydata:
                        payload[f"{base}{year}/mean_rank_ic"] = _number(ydata["mean_rank_ic"])
            trained = payload.get(prefix + "trained_encoder/mean_daily_rank_ic")
            for baseline in ("random_encoder", "raw_features"):
                other = payload.get(f"{prefix}{baseline}/mean_daily_rank_ic")
                if trained is not None and other is not None:
                    payload[f"{prefix}delta_vs_{baseline}"] = trained - other
    return payload


class WandbTrainingLogger:
    """Attach to ``train_steps`` via ``on_step``; safe no-op when disabled."""

    def __init__(self, *, run_name: str, project: str = DEFAULT_PROJECT, entity: str | None = DEFAULT_ENTITY,
                 run_id: str | None = None, group: str | None = None, tags: tuple[str, ...] | list[str] = (),
                 config: Mapping | None = None, job_type: str = "train", mode: str = "auto",
                 log_every: int = 50, notes: str | None = None, sdk=None, directory: str | Path | None = None):
        if mode not in {"auto", "online", "offline", "disabled"}:
            raise ValueError("mode must be auto, online, offline, or disabled")
        self.run_name, self.project, self.entity = run_name, project, entity
        self.run_id, self.group, self.tags = run_id, group, list(tags)
        self.config = _json_safe(config or {})
        self.job_type, self.notes, self.log_every = job_type, notes, max(1, int(log_every))
        self.directory = str(directory) if directory else None
        self._sdk = sdk
        self._run = None
        self._model = None
        self.failures = 0
        self.disabled_reason: str | None = None
        env_mode = os.environ.get("WANDB_MODE")
        key_present = resolve_wandb_key() is not None or env_mode == "offline"
        if mode == "disabled":
            self.mode = "disabled"
            self.disabled_reason = "requested"
        elif mode == "auto" and env_mode in {"disabled", "offline"}:
            self.mode = env_mode
            self.disabled_reason = "WANDB_MODE environment" if env_mode == "disabled" else None
        elif mode == "auto":
            # No key on the machine (the normal case on a CLI-created Colab runtime):
            # write an offline run next to the checkpoints and sync it later from a
            # logged-in machine with `wandb sync` / scripts/colab/sync_offline_wandb_runs.py.
            self.mode = "online" if key_present else "offline"
        else:
            self.mode = mode

    # -- lifecycle ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode != "disabled" and self.failures < 5

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    def _guard(self, action: Callable[[], Any], what: str):
        if not self.enabled:
            return None
        try:
            return action()
        except Exception as exc:  # never let logging break training
            self.failures += 1
            print(json.dumps({"wandb_warning": what, "error": type(exc).__name__,
                              "message": str(exc)[:300], "failures": self.failures}), flush=True)
            if self.failures >= 5:
                self.disabled_reason = f"disabled after {self.failures} failures"
            return None

    def start(self):
        if self._run is not None or not self.enabled:
            return self
        def _init():
            sdk = self._sdk
            if sdk is None:
                import wandb as sdk  # type: ignore
                self._sdk = sdk
            if self.mode == "online":
                key = resolve_wandb_key()
                if key:
                    sdk.login(key=key, relogin=False, verify=False)
            settings = None
            try:
                settings = sdk.Settings(start_method="thread", silent=True, _disable_stats=False)
            except Exception:
                settings = None
            run = sdk.init(project=self.project, entity=self.entity, name=self.run_name, id=self.run_id,
                           group=self.group, tags=self.tags or None, config=self.config, job_type=self.job_type,
                           notes=self.notes, mode=self.mode, resume="allow", reinit=True, dir=self.directory,
                           settings=settings)
            try:
                run.define_metric("train/optimizer_step")
                run.define_metric("train/*", step_metric="train/optimizer_step")
                run.define_metric("encoder/*", step_metric="train/optimizer_step")
                run.define_metric("proj_norm/*", step_metric="train/optimizer_step")
                run.define_metric("eval/checkpoint_step")
                run.define_metric("eval/*", step_metric="eval/checkpoint_step")
            except Exception:
                pass
            self._run = run
            return run
        self._guard(_init, "init")
        return self

    def attach(self, model) -> "WandbTrainingLogger":
        self._model = model
        return self

    def finish(self):
        if self._run is None:
            return
        run, self._run = self._run, None
        self._guard(run.finish, "finish")

    # -- logging -----------------------------------------------------------
    def log(self, payload: Mapping[str, Any]):
        if not payload:
            return
        self.start()
        if self._run is None:
            return
        clean = {k: v for k, v in _json_safe(payload).items() if v is not None}
        self._guard(lambda: self._run.log(clean), "log")

    def on_step(self, row: Mapping[str, Any]):
        step = int(row["step"])
        payload: dict[str, Any] = {"train/optimizer_step": step}
        for field in TRAIN_FIELDS:
            if field in row:
                payload["train/" + field] = _number(row[field])
        encoder = row.get("encoder")
        if isinstance(encoder, Mapping):
            for scope in ("global", "within_date"):
                block = encoder.get(scope, {})
                for field in DIAGNOSTIC_FIELDS:
                    if field in block:
                        payload[f"encoder/{scope}/{field}"] = _number(block[field])
            for field in ("valid_embeddings", "within_date_count"):
                if field in encoder:
                    payload["encoder/" + field] = _number(encoder[field])
        recorded = row.get("proj_norm")
        if isinstance(recorded, Mapping):
            payload.update({k: _number(v) for k, v in recorded.items()})
        elif self._model is not None and (step == 1 or step % self.log_every == 0):
            payload.update(channel_projection_norms(self._model))
        self.log(payload)

    def wrap(self, on_step: Callable[[Mapping], Any] | None):
        """Compose with an existing ``on_step``; the wrapped callback's return wins."""
        def combined(row):
            self.on_step(row)
            return on_step(row) if on_step is not None else None
        return combined

    def log_training_result(self, result: Mapping[str, Any]):
        summary = {f"training/{k}": _json_safe(result[k]) for k in
                   ("optimizer_step", "elapsed_seconds", "precision", "device", "stopped_early", "requested_stop_at")
                   if k in result}
        self.start()
        if self._run is None:
            return
        def _apply():
            for k, v in summary.items():
                self._run.summary[k] = v
        self._guard(_apply, "summary")

    def log_evaluation(self, report: Mapping, checkpoint_step: int | None = None):
        payload = evaluation_payload(report, checkpoint_step)
        self.log(payload)
        return payload

    def log_file(self, path: str | Path, name: str, artifact_type: str = "result"):
        self.start()
        if self._run is None:
            return
        def _upload():
            artifact = self._sdk.Artifact(name, type=artifact_type)
            artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        self._guard(_upload, "artifact")

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "enabled": self.enabled, "disabled_reason": self.disabled_reason,
                "project": self.project, "entity": self.entity, "run_id": self.run_id,
                "run_name": self.run_name, "group": self.group, "url": self.url, "failures": self.failures,
                "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
