"""Bounded, restartable execution of the already-defined Colab pilot.

Only explicit CLI invocation/run_job starts work. This module reads an existing
panel, never contacts a data API, and never stores credentials. Stop/restart the
same output directory to resume completed stages and full training checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import load_panel
from .model import PilotConfig
from .runner import (PILOT_VARIANTS as VARIANTS, _normalized, _write_json, budget_from_profiles, evaluate_variant,
                     profile_variant, run_variant, source_fingerprint)


class JobDeadline(RuntimeError):
    pass


def _utc():
    return datetime.now(timezone.utc).isoformat()


class Deadline:
    def __init__(self, hours):
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError("hours must be finite and positive")
        self.started = time.monotonic()
        self.expires = self.started + hours*3600

    @property
    def remaining(self):
        return max(0.0, self.expires-time.monotonic())

    def check(self, reserve=0):
        if self.remaining <= reserve:
            raise JobDeadline("Wall-clock budget reached; resume from the saved job state")

    @contextmanager
    def enforce(self, enabled=True):
        """POSIX deadline interrupts long evaluation stages as well as training.

        Python signal delivery may wait for a running native operation. This is
        a process deadline, not a guarantee about provider billing or shutdown.
        """
        if not enabled:
            yield
            return
        if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Hard job deadline requires a POSIX main-thread process")
        previous_alarm = signal.getsignal(signal.SIGALRM)
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        def stop(signum, frame):
            reason = "Runtime termination requested" if signum == signal.SIGTERM else "Wall-clock deadline reached"
            raise JobDeadline(reason)
        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, self.remaining)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
            signal.signal(signal.SIGTERM, previous_term)
            if previous_timer[0] > 0:
                elapsed = time.monotonic()-self.started
                signal.setitimer(signal.ITIMER_REAL, max(0.001, previous_timer[0]-elapsed), previous_timer[1])


def choose_future_variant(reports, budget, fingerprint):
    """Same development-only one-SE approximation as the interactive notebook."""
    if any(name not in reports for name in VARIANTS):
        raise ValueError("All five matched development evaluations are required")
    scores = {name: reports[name]["metrics"]["trained_encoder"]["development"]["mean_daily_rank_ic"]
              for name in ("B", "C", "D", "E")}
    if any(score is None or not np.isfinite(score) for score in scores.values()):
        raise ValueError("A future variant has no finite development IC; inspect coverage/representations")
    best = max(scores, key=scores.get)
    interval = reports[best]["metrics"]["trained_encoder"]["development"]["rank_ic_95ci_date_blocks"]
    margin = (interval[1]-interval[0])/3.92 if interval is not None else 0.0
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Invalid date-block uncertainty interval")
    selected = next(name for name in ("B", "C", "D", "E") if scores[name] >= scores[best]-margin)
    return {"variant": selected, "development_scores": scores, "highest_score_variant": best,
            "approximate_one_se_margin": margin,
            "selection_rule": "development IC; approximate one-SE preference B, C, D, E",
            "screen_steps": budget["screen_steps"], "total_steps": budget["total_steps"],
            "dataset_fingerprint": fingerprint, "historical_evaluation_consulted": False,
            "point_in_time_verified": False,
            "interpretation": "Engineering continuation choice; this heuristic is not a paired significance test or evidence of alpha."}


def run_job(panel_path, output, hours=4, device="cuda", *, batch_dates=16, stocks=32,
            accumulation_steps=2, profile_steps=100, warmup_steps=5,
            screen_steps=500, total_steps=2500, bootstrap_samples=500,
            evaluation_batch_size=16, reserve_fraction=0.25,
            model_config=None, allow_synthetic=False, use_signals=True):
    """Run or resume A-E, selected continuation, and seed 43 within one budget.

    C is screened first so the book-based future encoder is saved promptly.
    Selection waits for every matched screen/evaluation. 2022-2025 is untouched
    by this job. Each resumed invocation receives its explicit wall-clock budget;
    a previously frozen matched step plan is never silently reduced afterward.
    """
    deadline = Deadline(hours)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "job_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "created_at": _utc(), "profiles": {}, "training": {},
        "development": {}, "final_development": {}, "budget": None,
        "selection": None, "attempts": [], "status": "initializing"}
    prior_state = state_path.exists()
    model_config = model_config or PilotConfig()
    settings = {"batch_dates": batch_dates, "stocks": stocks, "accumulation_steps": accumulation_steps,
                "profile_steps": profile_steps, "warmup_steps": warmup_steps,
                "requested_screen_steps": screen_steps, "requested_total_steps": total_steps,
                "bootstrap_samples": bootstrap_samples, "evaluation_batch_size": evaluation_batch_size,
                "reserve_fraction": reserve_fraction, "device": device,
                "model_config": asdict(model_config), "allow_synthetic": allow_synthetic}
    attempt = {"started_at": _utc(), "hours": hours, "status": "running"}
    state["attempts"].append(attempt)
    state["status"] = "running"
    state.pop("error", None)
    state.pop("reason", None)

    def persist():
        state["updated_at"] = _utc()
        state["attempt_elapsed_seconds"] = time.monotonic()-deadline.started
        state["attempt_remaining_seconds"] = deadline.remaining
        _write_json(state, state_path)

    def stage(label):
        deadline.check(reserve=2)
        state["stage"] = label
        persist()
        print(json.dumps({"stage": label, "remaining_seconds": round(deadline.remaining, 1)}), flush=True)

    def train(name, seed, target, label):
        stage(label)
        run_dir = directory / f"variant_{name}_seed{seed}"
        snapshot = run_dir / f"encoder_step_{target:06d}.pt"
        if snapshot.exists():
            record = {"status": "completed", "optimizer_step": target,
                      "encoder_export": str(snapshot), "resumed_completed_stage": True}
        else:
            def on_step(row):
                state["current_training"] = {"variant": name, "seed": seed,
                                             "step": row["step"], "target": target,
                                             "loss": row["loss"], "run_dir": str(run_dir)}
                if row["step"] == 1 or row["step"] % 10 == 0:
                    persist()
                # Leave time to write a full checkpoint and compact export.
                return deadline.remaining > max(15.0, 2*row["seconds"])
            result = run_variant(panel, name, run_dir, seed=seed,
                                 screen_steps=state["budget"]["screen_steps"],
                                 total_steps=state["budget"]["total_steps"],
                                 batch_dates=batch_dates, stocks=stocks, device=device,
                                 resume=any(run_dir.glob("step_*.pt")), stop_at=target,
                                 model_config=model_config, accumulation_steps=accumulation_steps,
                                 on_step=on_step, peer_seed=42)
            record = {key: result[key] for key in ("optimizer_step", "checkpoint", "encoder_export",
                                                   "elapsed_seconds", "stopped_early", "requested_stop_at")}
            record["status"] = "completed" if result["optimizer_step"] == target else "partial"
        state["training"][label] = record
        state["current_training"] = None
        persist()
        if record["status"] != "completed":
            raise JobDeadline("Training stopped cleanly before its matched update target")
        return snapshot

    def evaluate(name, seed, target, label):
        stage(label)
        run_dir = directory / f"variant_{name}_seed{seed}"
        snapshot = run_dir / f"encoder_step_{target:06d}.pt"
        report_path = run_dir / f"evaluation_step_{target:06d}_h5_2019_2021" / "evaluation_report.json"
        report = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except json.JSONDecodeError:
                # A hard deadline can interrupt the evaluator's final report
                # write. Recompute that incomplete stage on the next invocation.
                pass
        if report is None:
            report = evaluate_variant(panel, run_dir, device=device, years=(2019, 2020, 2021),
                                      horizon=5, batch_size=evaluation_batch_size,
                                      bootstrap_samples=bootstrap_samples, snapshot=snapshot)
        deadline.check()
        return report, str(report_path)

    try:
        with deadline.enforce(use_signals):
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; this cloud job requires the selected GPU runtime")
            if profile_steps < 1 or warmup_steps < 0 or batch_dates < 1 or stocks < 2:
                raise ValueError("Invalid profiling or batch settings")
            panel = load_panel(panel_path)
            if panel.metadata.get("synthetic") and not allow_synthetic:
                raise ValueError("Synthetic panel rejected; use real data or explicitly enable a software test")
            fingerprint = panel.metadata["dataset_fingerprint"]
            code = source_fingerprint()
            if prior_state and "dataset_fingerprint" in state:
                if state.get("dataset_fingerprint") != fingerprint or state.get("settings") != settings or state.get("code") != code:
                    raise ValueError("Existing job has different data, source, or settings; use its original setup or a new output folder")
            state.update(dataset_fingerprint=fingerprint, settings=settings, code=code,
                         panel_path=str(Path(panel_path)), synthetic=bool(panel.metadata.get("synthetic", False)),
                         point_in_time_verified=bool(panel.metadata.get("point_in_time_verified", False)),
                         historical_evaluation_consulted=False)
            persist()
            panel = _normalized(panel)
            for name in ("C", "A", "B", "D", "E"):
                if name not in state["profiles"]:
                    stage(f"profile_{name}")
                    state["profiles"][name] = profile_variant(
                        panel, name, batch_dates=batch_dates, stocks=stocks, device=device,
                        profile_steps=profile_steps, warmup_steps=warmup_steps,
                        model_config=model_config, accumulation_steps=accumulation_steps)
                    persist()
            if state["budget"] is None:
                stage("freeze_matched_update_budget")
                state["budget"] = budget_from_profiles(state["profiles"],
                    budget_hours=deadline.remaining/3600, screen_steps=screen_steps,
                    total_steps=total_steps, reserve_fraction=reserve_fraction)
                state["budget"]["profiling_already_consumed_seconds"] = time.monotonic()-deadline.started
                _write_json(state["budget"], directory / "matched_budget.json")
                persist()
            s, t = state["budget"]["screen_steps"], state["budget"]["total_steps"]
            for name in ("C", "A", "B", "D", "E"):
                label = f"screen_{name}_seed42"
                if state["training"].get(label, {}).get("status") != "completed":
                    train(name, 42, s, label)
            reports_path = directory / "development_reports.json"
            reports = json.loads(reports_path.read_text()) if reports_path.exists() else {}
            for name in ("A", "B", "C", "D", "E"):
                if name not in reports:
                    report, path = evaluate(name, 42, s, f"development_{name}_seed42")
                    reports[name] = report
                    state["development"][name] = {"status": "completed", "report": path}
                    _write_json(reports, reports_path)
                    persist()
            if state["selection"] is None:
                stage("freeze_development_selection")
                state["selection"] = choose_future_variant(reports, state["budget"], fingerprint)
                _write_json(state["selection"], directory / "frozen_selection.json")
                persist()
            chosen = state["selection"]["variant"]
            for seed in (42, 43):
                label = f"final_{chosen}_seed{seed}"
                if state["training"].get(label, {}).get("status") != "completed":
                    train(chosen, seed, t, label)
            final_path = directory / "final_development_reports.json"
            final_reports = json.loads(final_path.read_text()) if final_path.exists() else {}
            for seed in (42, 43):
                if str(seed) not in final_reports:
                    report, path = evaluate(chosen, seed, t, f"final_development_{chosen}_seed{seed}")
                    final_reports[str(seed)] = report
                    state["final_development"][str(seed)] = {"status": "completed", "report": path}
                    _write_json(final_reports, final_path)
                    persist()
            state["status"] = attempt["status"] = "completed"
            state["stage"] = "completed_training_and_development"
            state["remaining_research"] = ["Separate frozen-choice historical evaluation", "Indicator role ablations", "Verified point-in-time universe"]
    except JobDeadline as exc:
        state["status"] = attempt["status"] = "paused_deadline"
        state["reason"] = str(exc)
    except KeyboardInterrupt:
        state["status"] = attempt["status"] = "paused_interrupt"
        state["reason"] = "Interrupted; completed stages and latest atomic training checkpoints remain resumable"
    except Exception as exc:
        state["status"] = attempt["status"] = "failed"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        attempt["finished_at"] = _utc()
        attempt["elapsed_seconds"] = time.monotonic()-deadline.started
        state["available_encoder_exports"] = [str(path) for path in sorted(directory.glob("variant_*/encoder_step_*.pt"))]
        state["latest_checkpoints"] = {
            folder.name: str(sorted(folder.glob("step_*.pt"))[-1])
            for folder in directory.glob("variant_*") if list(folder.glob("step_*.pt"))}
        persist()
        print(json.dumps({"status": state["status"], "stage": state.get("stage"),
                          "state_file": str(state_path), "encoder_exports": len(state["available_encoder_exports"])}), flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-dates", type=int, default=16)
    parser.add_argument("--stocks", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--screen-steps", type=int, default=500)
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--reserve-fraction", type=float, default=0.25)
    parser.add_argument("--allow-synthetic", action="store_true", help="Software tests only; never financial evidence")
    args = parser.parse_args(argv)
    state = run_job(args.panel, args.output, args.hours, args.device,
                    batch_dates=args.batch_dates, stocks=args.stocks,
                    accumulation_steps=args.accumulation_steps,
                    profile_steps=args.profile_steps, warmup_steps=args.warmup_steps,
                    screen_steps=args.screen_steps,
                    total_steps=args.total_steps, bootstrap_samples=args.bootstrap_samples,
                    evaluation_batch_size=args.evaluation_batch_size,
                    reserve_fraction=args.reserve_fraction,
                    allow_synthetic=args.allow_synthetic)
    return 0 if state["status"] == "completed" else 2 if state["status"].startswith("paused") else 1


if __name__ == "__main__":
    raise SystemExit(main())
