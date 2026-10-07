"""walk-forward job: yearly encoder retraining with warm starts, per-year features and readouts.

For each run (seed) and each test year Y in [first_year, last_year]:
  train(Y)    normalizer and sampler restricted to train_start .. (Y-1)-12-31; future windows and auxiliary
              labels end at or before that cutoff; first year trains from scratch (initial_steps), later years
              continue from encoder_{Y-1}.pt (continuation_steps, fresh short warmup); export encoder_{Y}.pt
  features(Y) teacher / online / random-init embeddings and the aux-head prediction for the eligible rows
              of year Y (plus, for the first year, the history rows train_start .. Y-1 from the same model)
  readout(Y)  annual expanding ridge on rows of years < Y (each year's rows from that year's model) scoring
              year Y; the aux head is scored directly; daily rank IC per model and horizon
State lives in <output>/expanded_job_state.json (same name and status vocabulary as expanded_job so the
watcher and supervisor work unchanged). Every stage is idempotent and resumable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .cloud_job import Deadline, JobDeadline
from .data import load_panel
from .evaluation import context_eligibility, make_forward_labels
from .expanded_job import normalize_training_panel
from .model import FutureJEPA, PilotConfig
from .runner import _write_json, source_fingerprint
from .training import TrainConfig, _atomic_save, seed_everything, train_steps
from .walkforward_data import WalkForwardSampler, normal_scores_by_date
from .walkforward_readout import daily_ic, extract_year_features, paired_block_difference, walk_forward_ridge, window_means
from .wandb_logging import DEFAULT_ENTITY, DEFAULT_PROJECT, WandbTrainingLogger, deterministic_run_id

DEFAULT_RUNS = {"v2_s42": {"seed": 42}, "v2_s43": {"seed": 43}}
WINDOWS = {"holdout_2015_2018": (2015, 2018), "dev_2019_2021": (2019, 2021), "recent_2022_2025": (2022, 2025)}
FEATURE_KINDS = ("teacher", "online", "random")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def default_model_config(panel):
    return PilotConfig(stock_features=panel.stock.shape[-1], market_features=panel.market.shape[-1], width=128, family_width=32,
                       tokenization="family", objective="future_family", feature_layout=panel.metadata.get("feature_layout", "v1"),
                       horizons=(5, 20))


def run_walkforward_job(panel_path, output, hours=6.0, device="cuda", *, runs=None, first_year=2015, last_year=2025,
                        train_start="2006-01-01", initial_steps=2500, continuation_steps=1000, continuation_warmup=0.10,
                        continuation_ema_start=0.995, batch_dates=16, stocks=32, accumulation_steps=2, evaluation_batch_size=16,
                        context=None, max_history_rows=500_000, max_readout_rows=600_000, allow_synthetic=False, use_signals=True, model_config=None,
                        wandb_mode="auto", wandb_project=DEFAULT_PROJECT, wandb_entity=DEFAULT_ENTITY, wandb_group=None, wandb_sdk=None):
    deadline = Deadline(hours)
    runs = dict(runs) if runs else dict(DEFAULT_RUNS)
    years = list(range(int(first_year), int(last_year) + 1))
    if not years or initial_steps < 1 or continuation_steps < 1:
        raise ValueError("Invalid year range or step counts")
    wandb_group = wandb_group or f"walkforward-{datetime.now(timezone.utc):%Y%m%d}"
    if allow_synthetic and wandb_mode == "auto":
        wandb_mode = "disabled"
    directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "expanded_job_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 2, "job": "walkforward", "created_at": _utc(), "training": {}, "evaluations": {}, "attempts": [],
        "status": "initializing", "historical_evaluation_consulted": True, "model_selection_performed": False}
    raw = load_panel(panel_path)
    if raw.metadata.get("synthetic") and not allow_synthetic:
        raise ValueError("Synthetic data rejected outside an explicitly enabled software test")
    model_config = model_config or default_model_config(raw)
    settings = json.loads(json.dumps({
        "job": "walkforward", "runs": runs, "years": years, "train_start": train_start, "initial_steps": initial_steps,
        "continuation_steps": continuation_steps, "continuation_warmup": continuation_warmup, "continuation_ema_start": continuation_ema_start,
        "batch_dates": batch_dates, "stocks": stocks, "accumulation_steps": accumulation_steps, "evaluation_batch_size": evaluation_batch_size,
        "device": device, "model_config": asdict(model_config), "allow_synthetic": allow_synthetic, "max_history_rows": max_history_rows, "max_readout_rows": max_readout_rows}))
    code = source_fingerprint()
    fingerprint = raw.metadata["dataset_fingerprint"]
    if state["training"] or state["evaluations"]:
        saved = state.get("settings") or {}
        changed = {k: {"saved": saved[k], "now": settings[k]} for k in saved if k in settings and saved[k] != settings[k]}
        added = sorted(set(settings) - set(saved))
        if state.get("dataset_fingerprint") != fingerprint or changed:
            raise ValueError(f"Job state does not match this panel or these settings ({sorted(changed)}); use a fresh output directory")
        if added:  # new settings with defaults (added by a code fix mid-run) are recorded, not refused
            state.setdefault("settings_added", []).append({"at": _utc(), "keys": added, "values": {k: settings[k] for k in added}})
        if state.get("code", {}).get("sha256") != code:  # a code fix mid-run is recorded, not refused
            state.setdefault("code_changes", []).append({"at": _utc(), "from": state.get("code", {}).get("sha256"), "to": code,
                                                         "completed_training": sorted(k for k, v in state["training"].items() if v.get("status") == "completed")})
    state.update(dataset_fingerprint=fingerprint, settings=settings, code={"sha256": code},
                 panel_metadata={"symbols": len(raw.symbols), "dates": len(raw.dates), "feature_layout": raw.metadata.get("feature_layout"),
                                 "stock_channels": raw.stock.shape[-1], "market_channels": raw.market.shape[-1]})
    attempt = {"started_at": _utc(), "hours": hours, "status": "running"}
    state["attempts"].append(attempt); state["status"] = "running"; state.pop("error", None); state.pop("reason", None)
    context_days = context or model_config.context_days
    dates = raw.dates.astype("datetime64[D]")
    year_of = dates.astype("datetime64[Y]").astype(int) + 1970
    eligible = context_eligibility(raw, context_days) & (dates >= np.datetime64(train_start))[:, None] & raw.market_obs[:, 0, None]
    labels = {h: make_forward_labels(raw, h) for h in model_config.horizons}
    rank_labels = {h: normal_scores_by_date(np.where(eligible & np.isfinite(labels[h][0]), labels[h][0], np.nan), eligible) for h in model_config.horizons}
    random_models = {}

    def persist():
        state["updated_at"] = _utc(); state["attempt_elapsed_seconds"] = time.monotonic() - deadline.started
        state["attempt_remaining_seconds"] = deadline.remaining; _write_json(state, state_path)

    def stage(name):
        deadline.check(reserve=2); state["stage"] = name; persist()
        print(json.dumps({"stage": name, "remaining_seconds": round(deadline.remaining, 1)}), flush=True)

    def run_cfg(run_name):
        overrides = runs[run_name].get("model_overrides") or {}
        return replace(model_config, **overrides) if overrides else model_config

    def run_dir(run_name, seed):
        return directory / f"{run_name}_seed{seed}"

    def export_path(run_name, seed, year):
        return run_dir(run_name, seed) / f"encoder_{year}.pt"

    def logger_for(run_name, seed, year, job_type="train"):
        identity = {"group": wandb_group, "arm": run_name, "seed": seed, "year": year, "panel": fingerprint}
        logger = WandbTrainingLogger(run_name=f"{run_name} | year {year}", run_id=deterministic_run_id(**identity), project=wandb_project,
                                     entity=wandb_entity, group=wandb_group, job_type=job_type,
                                     tags=[run_name, f"seed{seed}", f"year{year}", "walkforward_v2"],
                                     config={"arm": run_name, "seed": seed, "year": year, "settings": settings, "code_sha256": code},
                                     mode=wandb_mode, sdk=wandb_sdk, directory=str(directory))
        state.setdefault("wandb", {})[f"{run_name}_year_{year}"] = logger.describe()
        return logger

    # ------------------------------------------------------------------ training --------------------------------------------------
    def train(run_name, seed, year):
        label = f"{run_name}_year_{year}"
        export = export_path(run_name, seed, year)
        if state["training"].get(label, {}).get("status") == "completed" and export.exists():
            return
        stage("train_" + label)
        cfg = run_cfg(run_name)
        cutoff = f"{year - 1}-12-31"
        year_dir = run_dir(run_name, seed) / f"year_{year}"; year_dir.mkdir(parents=True, exist_ok=True)
        first = year == years[0]
        steps = initial_steps if first else continuation_steps
        previous = None if first else export_path(run_name, seed, year - 1)
        if previous is not None and not previous.exists():
            raise RuntimeError(f"Missing warm start {previous}")
        panel_y = normalize_training_panel(raw, train_start=train_start, cutoff=cutoff)
        normalizer = panel_y.metadata["normalizer"]
        normalizer_hash = hashlib.sha256(json.dumps(normalizer, sort_keys=True).encode()).hexdigest()
        metadata = json.loads(json.dumps({"job": "walkforward_v2", "run": run_name, "seed": seed, "year": year, "cutoff": cutoff,
                                          "train_start": train_start, "dataset_fingerprint": fingerprint, "normalizer_sha256": normalizer_hash,
                                          "model_config": asdict(cfg), "steps": steps, "warm_start_sha256": _sha256(previous) if previous else None,
                                          "code_sha256": code}))
        checkpoints = sorted(year_dir.glob("step_*.pt"))
        final = year_dir / f"step_{steps:06d}.pt"
        seed_everything(seed * 1000 + year)
        model = FutureJEPA(cfg)
        if export.exists():
            record = {"status": "completed", "optimizer_step": steps, "encoder_export": str(export), "recovered_completed_stage": True}
        elif final.exists():
            payload = torch.load(final, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            _atomic_save({"format_version": 2, "artifact_type": "walkforward_encoder", "model_config": asdict(cfg), "model": model.state_dict(),
                          "normalizer": normalizer, "metadata": {**metadata, "optimizer_step": steps}}, export)
            record = {"status": "completed", "optimizer_step": steps, "encoder_export": str(export), "recovered_from_checkpoint": True}
        else:
            if previous is not None and not checkpoints:
                model.load_state_dict(torch.load(previous, map_location="cpu", weights_only=False)["model"])
            sampler = WalkForwardSampler(panel_y, horizons=cfg.horizons, batch_dates=batch_dates, stocks_per_date=stocks, context=context_days,
                                         seed=seed * 1000 + year, start=train_start, end=cutoff,
                                         aux_target=cfg.aux_target, include_future=cfg.prediction_weight > 0)
            tc = TrainConfig(steps=steps, accumulation_steps=accumulation_steps, device=device, seed=seed * 1000 + year, checkpoint_every=100,
                             log_every=50, warmup_fraction=0.05 if first else continuation_warmup,
                             ema_start=0.99 if first else continuation_ema_start)

            def on_step(row):
                state["current_training"] = {"run": run_name, "year": year, "step": row["step"], "target": steps, "loss": row["loss"], "run_dir": str(year_dir)}
                if row["step"] == 1 or row["step"] % 10 == 0:
                    persist()
                return deadline.remaining > max(15.0, 2 * row["seconds"])
            logger = logger_for(run_name, seed, year)
            try:
                logger.start(); logger.attach(model)
                result = train_steps(model, sampler, tc, checkpoint_dir=year_dir, resume_from=checkpoints[-1] if checkpoints else None,
                                     on_step=logger.wrap(on_step), metadata=metadata)
                logger.log_training_result(result)
            finally:
                state["wandb"][label] = logger.describe(); logger.finish()
            _write_json({"optimizer_step": result["optimizer_step"], "elapsed_seconds": result["elapsed_seconds"], "stopped_early": result["stopped_early"],
                         "requested_stop_at": result["requested_stop_at"], "precision": result["precision"]}, year_dir / "run_result.json")
            _write_json(result["history"], year_dir / "training_metrics.json")
            if result["optimizer_step"] == steps:
                _atomic_save({"format_version": 2, "artifact_type": "walkforward_encoder", "model_config": asdict(cfg), "model": model.state_dict(),
                              "normalizer": normalizer, "metadata": {**metadata, "optimizer_step": steps}}, export)
            record = {"status": "completed" if result["optimizer_step"] == steps else "partial", "optimizer_step": result["optimizer_step"],
                      "encoder_export": str(export) if export.exists() else None, "elapsed_seconds": result["elapsed_seconds"]}
        _write_json(metadata, year_dir / "run_metadata.json")
        state["training"][label] = record; state["current_training"] = None; persist()
        del model; gc.collect()
        if record["status"] != "completed":
            raise JobDeadline("A training year stopped early; it resumes from its last checkpoint")

    # ------------------------------------------------------------------ features --------------------------------------------------
    def load_export(run_name, seed, year):
        payload = torch.load(export_path(run_name, seed, year), map_location="cpu", weights_only=False)
        cfg = PilotConfig(**payload["model_config"]); model = FutureJEPA(cfg); model.load_state_dict(payload["model"]); model.eval()
        return model, payload["normalizer"]

    def random_model(run_name, seed):
        if run_name not in random_models:
            seed_everything(seed); random_models[run_name] = FutureJEPA(run_cfg(run_name)).eval()
        return random_models[run_name]

    def eval_dir(run_name, year):
        return directory / "evaluations" / run_name / f"year_{year}"

    def features(run_name, seed, year):
        label = f"{run_name}_features_{year}"
        out = eval_dir(run_name, year); out.mkdir(parents=True, exist_ok=True)
        export = export_path(run_name, seed, year)
        signature = f"{_sha256(export)}|{_sha256(panel_path)}|{code}|{context_days}"
        manifest = out / "features_manifest.json"
        if (state["evaluations"].get(label, {}).get("status") == "completed" and manifest.exists()
                and json.loads(manifest.read_text()).get("signature") == signature and (out / "features.npz").exists()):
            return
        last = f"{run_name}_readout_{years[-1]}"
        if state["evaluations"].get(last, {}).get("status") == "completed" and (eval_dir(run_name, years[-1]) / "readout.json").exists():
            return  # every readout of this run exists (features are only their inputs); do not rebuild them after a resume
        stage("features_" + label)
        started = time.perf_counter()
        model, normalizer = load_export(run_name, seed, year)
        masks = {"features.npz": year_of == year}
        if year == years[0]:
            history = (year_of < year) & (dates >= np.datetime64(train_start))
            # keep whole cross-sections but thin the dates so the ridge history stays bounded (and extraction cheap)
            rows = int(eligible[history].sum())
            stride = max(1, int(np.ceil(rows / max_history_rows)))
            keep = np.zeros(len(dates), bool); idx = np.flatnonzero(history); keep[idx[::stride]] = True
            masks["features_history.npz"] = keep
            state.setdefault("history_sampling", {})[run_name] = {"rows_available": rows, "date_stride": stride, "rows_kept": int(eligible[keep].sum())}
        def progress(info):
            state["evaluation_progress"] = {"label": label, **info}
            if info.get("completed_groups", 0) % 200 == 0:
                persist()
            deadline.check(reserve=30)
        for name, mask in masks.items():
            feats = extract_year_features(raw, model, random_model(run_name, seed), normalizer, mask, eligible, out / "cache" / name.split(".")[0],
                                          device=device, batch_size=evaluation_batch_size, context=context_days, signature=signature + "|" + name,
                                          on_progress=progress)
            np.savez_compressed(out / name, **feats)
        _write_json({"signature": signature, "year": year, "files": list(masks)}, manifest)
        state["evaluations"][label] = {"status": "completed", "elapsed_seconds": time.perf_counter() - started, "rows": int(masks["features.npz"].sum())}
        state["evaluation_progress"] = None; persist()
        del model; gc.collect()

    # ------------------------------------------------------------------ readout ---------------------------------------------------
    def readout(run_name, seed, year):
        label = f"{run_name}_readout_{year}"
        out = eval_dir(run_name, year)
        if state["evaluations"].get(label, {}).get("status") == "completed" and (out / "readout.json").exists():
            return
        stage("readout_" + label)
        started = time.perf_counter()
        files = []
        for y in years:
            if y > year:
                break
            d = eval_dir(run_name, y)
            files += [d / name for name in (["features_history.npz", "features.npz"] if y == years[0] else ["features.npz"])]
        # Pass 1 (row indices only): bound the ridge's training rows by thinning the pre-test dates with a uniform
        # stride over the distinct dates present (the history file is already date-thinned, so a modulo on t_idx
        # would compound the two strides). Pass 2 loads each file and drops rows before concatenation, so the
        # 12 GB Colab host never holds every year three times over (that OOM-killed the 2021 readout).
        all_t = np.concatenate([np.load(f)["t_idx"] for f in files])
        test_rows = year_of[all_t] == year
        n_train = int((~test_rows).sum())
        train_dates = np.unique(all_t[~test_rows])
        stride = int(np.ceil(n_train / max_readout_rows)) if n_train > max_readout_rows else 1
        keep_dates = np.zeros(len(dates), bool); keep_dates[train_dates[::stride]] = True; keep_dates[year_of == year] = True
        del all_t, test_rows
        cat = {}
        for f in files:
            z = np.load(f)
            keep = keep_dates[z["t_idx"]]
            if keep.any():
                for k in z.files:
                    cat.setdefault(k, []).append(z[k][keep])
            del z
        cat = {k: np.concatenate(v) for k, v in cat.items()}
        gc.collect()
        readout_sampling = {"train_rows_available": n_train, "date_stride": stride, "rows_used": int(len(cat["t_idx"]))}
        t_idx, k_idx = cat["t_idx"], cat["k_idx"]
        result = {"year": year, "run": run_name, "seed": seed, "rows_total": int(len(t_idx)), "rows_year": int((year_of[t_idx] == year).sum()),
                  "readout_sampling": readout_sampling, "models": {}}
        preds_out = {"t_idx": t_idx, "k_idx": k_idx}
        for hi, h in enumerate(model_config.horizons):
            lab, _, ends = labels[h]
            y_raw = lab[t_idx, k_idx].astype(np.float64); y_rank = rank_labels[h][t_idx, k_idx].astype(np.float64)
            test = year_of[t_idx] == year
            for kind in FEATURE_KINDS:
                pred_map, fold = walk_forward_ridge(cat[kind], y_rank, y_raw, t_idx, dates, ends, year, purge=max(20, h))
                pred = np.full(len(t_idx), np.nan)
                for i, v in pred_map.items():
                    pred[i] = v
                ic = daily_ic(pred, y_raw, t_idx, dates)
                result["models"].setdefault(f"{kind}_ridge", {})[f"h{h}"] = {"fold": fold, "daily_ic": ic, "mean_ic": float(np.mean(list(ic.values()))) if ic else None}
                preds_out[f"{kind}_ridge_h{h}"] = pred.astype(np.float32)
            aux = np.where(test, cat["aux"][:, hi], np.nan).astype(np.float64)
            ic = daily_ic(aux, y_raw, t_idx, dates)
            result["models"].setdefault("aux_direct", {})[f"h{h}"] = {"daily_ic": ic, "mean_ic": float(np.mean(list(ic.values()))) if ic else None}
            preds_out[f"aux_direct_h{h}"] = aux.astype(np.float32)
        np.savez_compressed(out / "predictions.npz", **preds_out)
        _write_json(result, out / "readout.json")
        state["evaluations"][label] = {"status": "completed", "elapsed_seconds": time.perf_counter() - started,
                                       "mean_ic": {m: {h: v["mean_ic"] for h, v in hs.items()} for m, hs in result["models"].items()}}
        persist()

    def summary(run_name, seed):
        daily = {}
        for y in years:
            path = eval_dir(run_name, y) / "readout.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            for m, hs in r["models"].items():
                for h, v in hs.items():
                    daily.setdefault(m, {}).setdefault(h, {}).update(v["daily_ic"])
        out = {"run": run_name, "seed": seed, "years": years, "windows": WINDOWS,
               "mean_daily_rank_ic": {m: {h: {**window_means(d, WINDOWS), "yearly": {str(y): window_means(d, {"y": (y, y)})["y"] for y in years}} for h, d in hs.items()} for m, hs in daily.items()},
               "paired_vs_random": {}}
        for m in ("teacher_ridge", "online_ridge", "aux_direct"):
            for h in daily.get(m, {}):
                if h in daily.get("random_ridge", {}):
                    out["paired_vs_random"][f"{m}_minus_random_ridge|{h}"] = {w: paired_block_difference(daily[m][h], daily["random_ridge"][h], lo, hi) for w, (lo, hi) in WINDOWS.items()}
        _write_json(out, directory / "evaluations" / run_name / "summary.json")
        state.setdefault("summaries", {})[run_name] = {m: {h: {w: v[w] for w in WINDOWS} for h, v in hs.items()} for m, hs in out["mean_daily_rank_ic"].items()}
        persist()

    raise_after = None
    try:
        with deadline.enforce(use_signals):
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; connect the selected cloud GPU")
            stage("load_and_validate_panel")
            if raw.origin_observed is None and raw.metadata.get("feature_layout") == "v2":
                raise ValueError("assembled panel lacks the carried eligibility mask")
            for run_name, spec in runs.items():
                seed = int(spec.get("seed", 42))
                for year in years:
                    train(run_name, seed, year)
                    features(run_name, seed, year)
                    readout(run_name, seed, year)
                summary(run_name, seed)
            state["status"] = attempt["status"] = "completed"; state["stage"] = "completed_walkforward"
    except JobDeadline as exc:
        state["status"] = attempt["status"] = "paused_deadline"; state["reason"] = str(exc)
    except KeyboardInterrupt:
        state["status"] = attempt["status"] = "paused_interrupt"; state["reason"] = "Interrupted; resume with the same panel, settings and output directory"
    except Exception as exc:
        state["status"] = attempt["status"] = "failed"; state["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        raise_after = exc
    finally:
        attempt["finished_at"] = _utc(); attempt["elapsed_seconds"] = time.monotonic() - deadline.started
        state["available_encoder_exports"] = [str(p) for p in sorted(directory.glob("*_seed*/encoder_*.pt"))]
        state["latest_checkpoints"] = {f"{folder.parent.name}/{folder.name}": str(sorted(folder.glob("step_*.pt"))[-1])
                                       for folder in directory.glob("*_seed*/year_*") if list(folder.glob("step_*.pt"))}
        persist()
        print(json.dumps({"status": state["status"], "stage": state.get("stage"), "state_file": str(state_path),
                          "encoder_exports": len(state["available_encoder_exports"])}), flush=True)
    if raise_after is not None and allow_synthetic:
        raise raise_after
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs-json", default=None)
    parser.add_argument("--first-year", type=int, default=2015)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--train-start", default="2006-01-01")
    parser.add_argument("--initial-steps", type=int, default=2500)
    parser.add_argument("--continuation-steps", type=int, default=1000)
    parser.add_argument("--continuation-warmup", type=float, default=0.10)
    parser.add_argument("--batch-dates", type=int, default=16)
    parser.add_argument("--stocks", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--max-history-rows", type=int, default=500_000, help="thin the pre-first-year history (by date stride) to at most this many rows")
    parser.add_argument("--max-readout-rows", type=int, default=600_000, help="thin the ridge's training rows (by date stride) to at most this many rows per readout")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--wandb-mode", default="auto", choices=("auto", "online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args(argv)
    runs = json.loads(Path(args.runs_json).read_text()) if args.runs_json else None
    state = run_walkforward_job(args.panel, args.output, hours=args.hours, device=args.device, runs=runs, first_year=args.first_year,
                                last_year=args.last_year, train_start=args.train_start, initial_steps=args.initial_steps,
                                continuation_steps=args.continuation_steps, continuation_warmup=args.continuation_warmup,
                                batch_dates=args.batch_dates, stocks=args.stocks, accumulation_steps=args.accumulation_steps,
                                evaluation_batch_size=args.evaluation_batch_size, max_history_rows=args.max_history_rows, max_readout_rows=args.max_readout_rows, allow_synthetic=args.allow_synthetic,
                                wandb_mode=args.wandb_mode, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity, wandb_group=args.wandb_group)
    return 0 if state["status"].startswith("completed") else 2 if state["status"].startswith("paused") else 1


if __name__ == "__main__":
    raise SystemExit(main())
