"""Matched, restartable original-versus-expanded universe experiment.

All seven encoders start afresh from the same source and seed. Expanded A/F
(basic/book historical masking) and B/C (basic/book future prediction) form the
matched 2x2 control. Original B/C test universe expansion; expanded E/book
predicts one day and receives a one-day readout. Historical A/F receive primary
five-day and prespecified secondary one-day readouts from the same embeddings.
No winner is
selected and no 2022-2025 results are consulted. Existing pilot files are reused
without changing the archived experiment implementation.
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
from .data import fast_observed, fit_normalizer, load_panel
from .model import PilotConfig
from .runner import _write_json, run_variant, source_fingerprint
from .training import export_encoder, load_snapshot
from .wandb_logging import DEFAULT_ENTITY, DEFAULT_PROJECT, WandbTrainingLogger, deterministic_run_id


RUNS = {
    "expanded_C": {"panel": "expanded", "variant": "C", "horizon": 5, "readout_horizons": [5]},
    "expanded_F": {"panel": "expanded", "variant": "F", "horizon": 5, "readout_horizons": [5, 1]},
    "expanded_A": {"panel": "expanded", "variant": "A", "horizon": 5, "readout_horizons": [5, 1]},
    "expanded_B": {"panel": "expanded", "variant": "B", "horizon": 5, "readout_horizons": [5]},
    "original_B": {"panel": "original", "variant": "B", "horizon": 5, "readout_horizons": [5]},
    "original_C": {"panel": "original", "variant": "C", "horizon": 5, "readout_horizons": [5]},
    "expanded_E": {"panel": "expanded", "variant": "E", "horizon": 1, "readout_horizons": [1]},
}

INTERMEDIATE_EVALUATION_RUNS = tuple(name for name, spec in RUNS.items() if spec["variant"] != "E")


def evaluate_expanded_run(*args, **kwargs):
    # Keep orchestration importable while the independent evaluator is packaged.
    from .expanded_eval import evaluate_expanded_run as evaluate
    return evaluate(*args, **kwargs)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def normalize_training_panel(panel, chunk_dates=128, train_start="2006-01-01", cutoff="2018-12-31"):
    """Same training-only robust scale, with bounded temporary array memory.

    The existing normalizer is fitted unchanged. Applying it in float32 chunks
    avoids a full expanded-panel float64 intermediate on a 12.7 GB Colab host.
    """
    normalizer = panel.metadata.get("normalizer")
    fingerprint = panel.metadata["dataset_fingerprint"]
    if normalizer:
        if normalizer["dataset_fingerprint"] != fingerprint:
            raise ValueError("Panel's normalizer fingerprint does not match")
        if np.datetime64(normalizer["train_cutoff"]) > np.datetime64(cutoff):
            raise ValueError("Normalization extends beyond the encoder training cutoff")
        return panel
    normalizer = fit_normalizer(panel, cutoff=cutoff, start=train_start)
    def apply(values, observed, spec):
        out = np.empty(values.shape, dtype=np.float32)
        median, scale = np.asarray(spec["median"], dtype=np.float32), np.asarray(spec["iqr"], dtype=np.float32)
        for start in range(0, len(values), chunk_dates):
            stop = min(start+chunk_dates, len(values))
            target = out[start:stop]
            np.subtract(values[start:stop], median, out=target)
            np.divide(target, scale, out=target)
            np.clip(target, -normalizer["clip"], normalizer["clip"], out=target)
            target[~observed[start:stop]] = 0
        return out
    return replace(panel, stock=apply(panel.stock, panel.stock_obs, normalizer["stock"]),
                   market=apply(panel.market, panel.market_obs, normalizer["market"]),
                   metadata={**panel.metadata, "normalizer": normalizer})


def _input_eligible_training_dates(panel, context):
    """Eligibility at the origin only; no future prices or target masks are read."""
    fast = fast_observed(panel)
    count = np.zeros(fast.shape[1], dtype=np.int32)
    result = []
    for t in range(len(fast)):
        count += fast[t]
        if t >= context:
            count -= fast[t-context]
        if t < context-1:
            continue
        date = panel.dates[t]
        if date < np.datetime64("2006-01-01") or date > np.datetime64("2018-12-31"):
            continue
        if panel.market_obs[t, 0] and np.count_nonzero(panel.universe[t] & (count == context)) >= 2:
            result.append(date)
    return np.asarray(result, dtype="datetime64[D]")


def align_training_panels(panels, context=60):
    """Use the same origin-eligible training calendar for the two data universes.

    Horizon-specific target calendar/observation validity remains in the sampler.
    The B/C five-day arms therefore share origins; E uses the same input calendar
    with its separate one-day target rule. Raw evaluation panels stay unchanged.
    """
    if any(panel.metadata.get("normalizer") for panel in panels.values()):
        raise ValueError("Date alignment requires raw panels so normalization can be refitted on matched training dates")
    shared = np.intersect1d(_input_eligible_training_dates(panels["original"], context),
                            _input_eligible_training_dates(panels["expanded"], context))
    if not len(shared):
        raise ValueError("The panels have no shared origin-eligible training dates")
    digest = hashlib.sha256(shared.astype("int64").tobytes()).hexdigest()
    alignment = {"version": 1, "shared_origin_date_sha256": digest, "count": len(shared),
                 "start": str(shared[0]), "end": str(shared[-1]), "context_days": context,
                 "eligibility": f"origin-known universe, {context}-day fast-feature availability, market return availability, at least two stocks",
                 "target_validity": "separate horizon-specific calendar and observation checks; future price availability does not select origins"}
    aligned = {}
    for name, panel in panels.items():
        universe = panel.universe.copy()
        train = (panel.dates >= np.datetime64("2006-01-01")) & (panel.dates <= np.datetime64("2018-12-31"))
        universe[train & ~np.isin(panel.dates, shared)] = False
        raw_fingerprint = panel.metadata["dataset_fingerprint"]
        identity = {"source_dataset_fingerprint": raw_fingerprint, "training_date_alignment": alignment}
        derived = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        aligned[name] = replace(panel, universe=universe, metadata={**panel.metadata, **identity,
            "dataset_fingerprint": derived, "normalizer": None,
            "training_dataset_derivation": "matched origin-eligible calendar; original raw feature history retained"})
    return aligned, alignment


def run_expanded_job(original_panel_path, expanded_panel_path, output, hours=4,
                     device="cuda", *, total_steps=2500, intermediate_steps=500,
                     seed=42, batch_dates=16, stocks=32, accumulation_steps=2,
                     evaluation_batch_size=16, bootstrap_samples=500,
                     evaluate_intermediate=True, intermediate_min_remaining_seconds=600,
                     model_config=None, allow_synthetic=False, use_signals=True,
                     wandb_mode="auto", wandb_project=DEFAULT_PROJECT, wandb_entity=DEFAULT_ENTITY,
                     wandb_group=None, wandb_sdk=None, runs=None,
                     final_cohorts=("common", "expanded"), intermediate_cohorts=("common", "expanded")):
    """Run seven matched experiments and two evaluation cohorts, or resume.

    Expanded C's 500-update snapshot is saved first. Every run then gets its
    500 snapshot and full 2500 updates on the originally planned schedule. Final
    matched evaluations take precedence over optional 500-update A/B/C/F readouts.
    A resumed invocation adds its explicitly supplied time budget; completed
    stages and cached evaluator work are reused. Step budgets are never reduced.

    W&B: one run per arm/seed (deterministic id from group, arm, seed, panels),
    reused across the 500/2,500 stages and their evaluations. ``wandb_mode``
    "auto" logs online with a key, otherwise offline for later sync.

    ``runs`` overrides the default seven-arm table. Each entry may add
    ``seed`` (per-run training seed), ``model_overrides`` (PilotConfig fields),
    and ``extra_evaluation_steps`` (checkpoints evaluated on intermediate cohorts).
    """
    deadline = Deadline(hours)
    runs = dict(runs) if runs else dict(RUNS)
    for name, spec in runs.items():
        missing = {"panel", "variant", "horizon", "readout_horizons"} - set(spec)
        if missing:
            raise ValueError(f"run {name} lacks {sorted(missing)}")
    intermediate_runs = tuple(name for name, spec in runs.items() if spec["variant"] != "E")
    final_cohorts, intermediate_cohorts = tuple(final_cohorts), tuple(intermediate_cohorts)
    wandb_group = wandb_group or f"expanded-{datetime.now(timezone.utc):%Y%m%d}"
    if allow_synthetic and wandb_mode == "auto":
        wandb_mode = "disabled"  # software tests never log to the research project by default
    if not 1 <= intermediate_steps <= total_steps:
        raise ValueError("Require 1 <= intermediate_steps <= total_steps")
    if batch_dates < 1 or stocks < 2 or accumulation_steps < 1:
        raise ValueError("Invalid training batch settings")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "expanded_job_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "created_at": _utc(), "training": {}, "evaluations": {},
        "attempts": [], "status": "initializing", "historical_evaluation_consulted": False,
        "model_selection_performed": False}
    model_config = model_config or PilotConfig()
    settings = {"total_steps": total_steps, "intermediate_steps": intermediate_steps, "seed": seed,
                "runs": runs, "final_cohorts": final_cohorts, "intermediate_cohorts": intermediate_cohorts,
                "batch_dates": batch_dates, "stocks": stocks, "accumulation_steps": accumulation_steps,
                "evaluation_batch_size": evaluation_batch_size, "bootstrap_samples": bootstrap_samples,
                "device": device, "model_config": asdict(model_config), "allow_synthetic": allow_synthetic,
                "evaluate_intermediate": evaluate_intermediate,
                "intermediate_min_remaining_seconds": intermediate_min_remaining_seconds}
    settings = json.loads(json.dumps(settings))  # JSON-normalized so resume comparisons match the saved state
    paths = {"original": str(Path(original_panel_path)), "expanded": str(Path(expanded_panel_path))}
    attempt = {"started_at": _utc(), "hours": hours, "status": "running"}
    state["attempts"].append(attempt)
    state["status"] = "running"
    state.pop("error", None)
    state.pop("reason", None)
    training_panels, raw_panels = {}, {}

    def run_seed(run_name):
        return int(runs[run_name].get("seed", seed))

    def run_model_config(run_name):
        overrides = runs[run_name].get("model_overrides") or {}
        return replace(model_config, **overrides) if overrides else model_config

    def wandb_logger(run_name, job_type="train"):
        spec = runs[run_name]
        this_seed = run_seed(run_name)
        identity = {"group": wandb_group, "arm": run_name, "seed": this_seed,
                    "panels": state.get("dataset_fingerprints") or paths}
        logger = WandbTrainingLogger(
            run_name=f"{run_name} | seed {this_seed}", run_id=deterministic_run_id(**identity),
            project=wandb_project, entity=wandb_entity, group=wandb_group, job_type=job_type,
            tags=[run_name, f"variant_{spec['variant']}", spec["panel"], f"seed{this_seed}", "expanded_job"],
            config={"arm": run_name, "spec": spec, "seed": this_seed, "settings": settings,
                    "dataset_fingerprints": state.get("dataset_fingerprints"),
                    "code_sha256": state.get("code", {}).get("sha256") if isinstance(state.get("code"), dict) else None},
            mode=wandb_mode, sdk=wandb_sdk, directory=str(directory))
        state.setdefault("wandb", {})[run_name] = logger.describe()
        return logger

    def persist():
        state["updated_at"] = _utc()
        state["attempt_elapsed_seconds"] = time.monotonic()-deadline.started
        state["attempt_remaining_seconds"] = deadline.remaining
        _write_json(state, state_path)

    def stage(name):
        deadline.check(reserve=2)
        state["stage"] = name
        persist()
        print(json.dumps({"stage": name, "remaining_seconds": round(deadline.remaining, 1)}), flush=True)

    def train(run_name, target):
        label = f"{run_name}_step_{target:06d}"
        spec = runs[run_name]
        this_seed = run_seed(run_name)
        run_dir = directory / f"{run_name}_seed{this_seed}"
        snapshot = run_dir / f"encoder_step_{target:06d}.pt"
        if state["training"].get(label, {}).get("status") == "completed" and snapshot.exists():
            return
        stage("train_"+label)
        checkpoint_at_target = run_dir / f"step_{target:06d}.pt"
        if not snapshot.exists() and checkpoint_at_target.exists():
            recovered = load_snapshot(checkpoint_at_target)
            metadata = {**recovered.snapshot_metadata, "optimizer_step": target,
                        "training_checkpoint": checkpoint_at_target.name}
            export_encoder(recovered, snapshot, metadata)
            del recovered
        if snapshot.exists():
            record = {"status": "completed", "optimizer_step": target, "encoder_export": str(snapshot),
                      "recovered_completed_stage": True}
        else:
            def on_step(row):
                state["current_training"] = {"run": run_name, "step": row["step"], "target": target,
                                             "loss": row["loss"], "run_dir": str(run_dir)}
                if row["step"] == 1 or row["step"] % 10 == 0:
                    persist()
                return deadline.remaining > max(15.0, 2*row["seconds"])
            logger = wandb_logger(run_name)
            try:
                result = run_variant(training_panels[spec["panel"]], spec["variant"], run_dir,
                                     seed=this_seed, screen_steps=intermediate_steps, total_steps=total_steps,
                                     batch_dates=batch_dates, stocks=stocks, device=device,
                                     resume=any(run_dir.glob("step_*.pt")), stop_at=target,
                                     model_config=run_model_config(run_name), accumulation_steps=accumulation_steps,
                                     on_step=on_step, peer_seed=42, logger=logger)
            finally:
                state["wandb"][run_name] = logger.describe()
                logger.finish()
            record = {key: result[key] for key in ("optimizer_step", "checkpoint", "encoder_export",
                                                  "elapsed_seconds", "stopped_early", "requested_stop_at")}
            record["status"] = "completed" if result["optimizer_step"] == target else "partial"
            record["wandb_url"] = state["wandb"][run_name].get("url")
        state["training"][label] = record
        state["current_training"] = None
        persist()
        if record["status"] != "completed":
            raise JobDeadline("A matched training stage stopped early; all planned update counts remain unchanged")

    def evaluate(run_name, target, optional=False):
        label = f"{run_name}_step_{target:06d}"
        record = state["evaluations"].get(label, {})
        if record.get("status") == "completed":
            return True
        if optional:
            durations = [item["elapsed_seconds"] for item in state["evaluations"].values()
                         if item.get("status") == "completed" and item.get("elapsed_seconds", 0) > 0]
            estimated = max(durations) * 1.15 if durations else intermediate_min_remaining_seconds
            required = max(intermediate_min_remaining_seconds, estimated)
            if deadline.remaining <= required:
                state["evaluations"][label] = {"status": "deferred_budget", "optional_intermediate": True,
                                               "estimated_required_seconds": required}
                persist()
                return False
        stage("evaluate_"+label)
        spec = runs[run_name]
        run_dir = directory / f"{run_name}_seed{run_seed(run_name)}"
        snapshot = run_dir / f"encoder_step_{target:06d}.pt"
        full_checkpoint = run_dir / f"step_{target:06d}.pt"
        if not snapshot.exists() and full_checkpoint.exists():
            recovered = load_snapshot(full_checkpoint)
            export_encoder(recovered, snapshot, {**recovered.snapshot_metadata, "optimizer_step": target,
                                                 "training_checkpoint": full_checkpoint.name})
            del recovered
        cohorts_for_stage = final_cohorts if target == total_steps else intermediate_cohorts
        output_dir = directory / "evaluations" / label
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "expanded_job_report.json"
        started = time.monotonic()
        report = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except json.JSONDecodeError:
                pass
        if report is None:
            last_progress_write, last_phase = 0.0, None
            def on_progress(progress):
                nonlocal last_progress_write, last_phase
                state["evaluation_progress"] = {"run": run_name, "step": target, **progress}
                deadline.check(reserve=5)
                now, phase = time.monotonic(), progress.get("phase")
                if now-last_progress_write >= 15 or phase != last_phase:
                    persist()
                    last_progress_write, last_phase = now, phase
            report = evaluate_expanded_run(raw_panels["expanded"], raw_panels["original"], run_dir,
                snapshot=snapshot, output_dir=output_dir, device=device, years=(2019, 2020, 2021),
                primary_horizon=spec["horizon"], horizons=tuple(spec["readout_horizons"]),
                peer_seed=42, batch_size=evaluation_batch_size,
                bootstrap_samples=bootstrap_samples, on_progress=on_progress,
                cohort_names=cohorts_for_stage)
            _write_json(report, report_path)
        logger = wandb_logger(run_name, job_type="evaluate")
        try:
            logger.log_evaluation(report, checkpoint_step=target)
        finally:
            state["wandb"][run_name] = logger.describe()
            logger.finish()
        state["evaluations"][label] = {"status": "completed", "report": str(report_path),
                                      "elapsed_seconds": time.monotonic()-started,
                                      "horizon": spec["horizon"],
                                      "readout_horizons": spec["readout_horizons"],
                                      "cohorts": list(cohorts_for_stage),
                                      "optional_intermediate": optional}
        state["evaluation_progress"] = None
        persist()
        return True

    try:
        with deadline.enforce(use_signals):
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; connect the selected cloud GPU")
            stage("load_and_validate_panels")
            fingerprints, panel_metadata = {}, {}
            for name, path in paths.items():
                panel = load_panel(path)
                if panel.metadata.get("synthetic") and not allow_synthetic:
                    raise ValueError("Synthetic data rejected outside an explicitly enabled software test")
                fingerprints[name] = panel.metadata["dataset_fingerprint"]
                panel_metadata[name] = {"symbols": len(panel.symbols), "dates": len(panel.dates),
                    "maximum_selected_stocks": int(np.asarray(panel.universe).sum(axis=1).max()),
                    "point_in_time_verified": bool(panel.metadata.get("point_in_time_verified", False)),
                    "synthetic": bool(panel.metadata.get("synthetic", False))}
                training_panels[name] = panel
                del panel
            if fingerprints["original"] == fingerprints["expanded"]:
                raise ValueError("Original and expanded panels have the same fingerprint")
            code = source_fingerprint()
            if "dataset_fingerprints" in state:
                if state["dataset_fingerprints"] != fingerprints or state["settings"] != settings or state["code"] != code:
                    raise ValueError("Existing experiment has different panels, settings, or source; use the recorded setup or a new output directory")
            all_training_present = all(
                state["training"].get(f"{name}_step_{target:06d}", {}).get("status") == "completed"
                and (directory/f"{name}_seed{run_seed(name)}"/f"encoder_step_{target:06d}.pt").exists()
                for name in runs for target in {intermediate_steps, total_steps})
            if all_training_present:
                # A resumed evaluation needs raw arrays only; do not allocate
                # normalized copies or recompute the already recorded calendar.
                raw_panels, training_panels = training_panels, {}
                alignment = state["training_date_alignment"]
                training_fingerprints = state["training_dataset_fingerprints"]
            else:
                training_panels, alignment = align_training_panels(training_panels, model_config.context_days)
                for name in training_panels:
                    training_panels[name] = normalize_training_panel(training_panels[name], train_start=alignment["start"])
                training_fingerprints = {name: panel.metadata["dataset_fingerprint"] for name, panel in training_panels.items()}
            state.update(dataset_fingerprints=fingerprints, panel_paths=paths, panel_metadata=panel_metadata,
                         settings=settings, code=code, planned_optimizer_updates=len(runs)*total_steps,
                         training_date_alignment=alignment,
                         training_dataset_fingerprints=training_fingerprints,
                         comparison="Expanded 2x2: basic/book by historical masking/future five-day prediction; original B/C universe controls; expanded E one-day readout",
                         factorial_controls={"basic_historical": "expanded_A", "book_historical": "expanded_F",
                                             "basic_future": "expanded_B", "book_future": "expanded_C"},
                         evaluation_context="Identical expanded deployment peers for all models; common original-origin and full expanded cohorts")
            persist()
            # Expanded book C first, followed by each matched intermediate point.
            for run_name in runs:
                train(run_name, intermediate_steps)
            for run_name in runs:
                train(run_name, total_steps)
            # Release training-normalized arrays before loading raw evaluation
            # inputs. Each model's own saved normalizer is applied by evaluator.
            training_panels.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stage("load_raw_evaluation_panels")
            if not raw_panels:
                raw_panels = {name: load_panel(path) for name, path in paths.items()}
            for run_name in runs:
                evaluate(run_name, total_steps)
            if evaluate_intermediate and intermediate_steps < total_steps:
                for run_name in intermediate_runs:
                    evaluate(run_name, intermediate_steps, optional=True)
                for run_name in runs:
                    for extra in runs[run_name].get("extra_evaluation_steps") or ():
                        if extra not in (intermediate_steps, total_steps):
                            evaluate(run_name, int(extra), optional=True)
            deferred = [name for name, record in state["evaluations"].items() if record["status"] != "completed"]
            state["status"] = attempt["status"] = "completed_with_deferred_intermediate" if deferred else "completed"
            state["stage"] = "completed_matched_training_and_final_development"
            state["deferred_evaluations"] = deferred
            state["remaining_research"] = ["Independent seed replication", "Frozen-choice historical evaluation", "Verified delisting-return coverage"]
    except JobDeadline as exc:
        state["status"] = attempt["status"] = "paused_deadline"
        state["reason"] = str(exc)
    except KeyboardInterrupt:
        state["status"] = attempt["status"] = "paused_interrupt"
        state["reason"] = "Interrupted; resume with the same panels, source, settings, and output directory"
    except Exception as exc:
        state["status"] = attempt["status"] = "failed"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        attempt["finished_at"] = _utc()
        attempt["elapsed_seconds"] = time.monotonic()-deadline.started
        state["available_encoder_exports"] = [str(path) for path in sorted(directory.glob("*_seed*/encoder_step_*.pt"))]
        state["latest_checkpoints"] = {folder.name: str(sorted(folder.glob("step_*.pt"))[-1])
            for folder in directory.glob("*_seed*") if list(folder.glob("step_*.pt"))}
        persist()
        print(json.dumps({"status": state["status"], "stage": state.get("stage"),
                          "state_file": str(state_path), "encoder_exports": len(state["available_encoder_exports"])}), flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-panel", required=True)
    parser.add_argument("--expanded-panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument("--intermediate-steps", type=int, default=500)
    parser.add_argument("--batch-dates", type=int, default=16)
    parser.add_argument("--stocks", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--skip-intermediate-evaluation", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true", help="Software tests only")
    parser.add_argument("--wandb-mode", default="auto", choices=("auto", "online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-group", default=None, help="Experiment group; default expanded-YYYYMMDD")
    parser.add_argument("--runs-json", default=None, help="JSON file with a run table replacing the default seven arms")
    parser.add_argument("--final-cohorts", default="common,expanded")
    parser.add_argument("--intermediate-cohorts", default="common,expanded")
    args = parser.parse_args(argv)
    runs = json.loads(Path(args.runs_json).read_text()) if args.runs_json else None
    state = run_expanded_job(args.original_panel, args.expanded_panel, args.output,
        hours=args.hours, device=args.device, total_steps=args.total_steps,
        intermediate_steps=args.intermediate_steps, batch_dates=args.batch_dates,
        stocks=args.stocks, accumulation_steps=args.accumulation_steps,
        evaluation_batch_size=args.evaluation_batch_size, bootstrap_samples=args.bootstrap_samples,
        evaluate_intermediate=not args.skip_intermediate_evaluation, allow_synthetic=args.allow_synthetic,
        wandb_mode=args.wandb_mode, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group, runs=runs,
        final_cohorts=tuple(x for x in args.final_cohorts.split(",") if x),
        intermediate_cohorts=tuple(x for x in args.intermediate_cohorts.split(",") if x))
    return 0 if state["status"].startswith("completed") else 2 if state["status"].startswith("paused") else 1


if __name__ == "__main__":
    raise SystemExit(main())
