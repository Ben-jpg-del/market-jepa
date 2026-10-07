"""Keep the tokenization experiment running across Colab session terminations.

Every two minutes: read the watcher-downloaded job state. If it is terminal, exit. Otherwise
check the current session; if Colab has dropped it (or the job process is gone), wait for the
old watcher to finish its last downloads, then call resume_tokenization_experiment.py with a
new session name. Stops after --max-resumes to bound compute spend.

Usage: nohup work/colab-test-env/bin/python scripts/colab/supervise_tokenization_experiment.py > outputs/colab-tokenization/supervisor.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "colab"))
import launch_tokenization_experiment as L  # noqa: E402

STATE = L.OUT / "live-run" / "artifacts" / "expanded_job_state.json"
TERMINAL = {"completed", "completed_with_deferred_intermediate", "failed", "paused_deadline", "paused_interrupt"}


def now():
    return datetime.now(timezone.utc)


def log(**info):
    print(json.dumps({"at": now().isoformat(timespec="seconds"), **info}), flush=True)


def session_alive(session):
    proc = subprocess.run([str(L.CLI), "status", "-s", session], text=True, capture_output=True, timeout=120)
    text = proc.stdout + proc.stderr
    if "not found" in text.lower():
        return False
    return proc.returncode == 0


def job_process_alive(session):
    try:
            out, _, code = L.exec_code(session, "import json,os\nl=json.load(open('"+L.REMOTE_RUN+"/launch.json'))\ntry:\n    os.kill(l['pid'],0); print('PID_ALIVE 1')\nexcept ProcessLookupError:\n    print('PID_ALIVE 0')", timeout=60)
    except Exception as exc:  # a hung/failed exec must never kill the supervisor
        log(event="probe_error", error=type(exc).__name__)
        return None
    if "PID_ALIVE 1" in out:
        return True
    if "PID_ALIVE 0" in out:
        return False
    return None  # unknown (exec failed)


def current_session():
    launch = json.loads((L.OUT / "launch.json").read_text()) if (L.OUT / "launch.json").exists() else {}
    if "session" in launch:
        return launch["session"]
    return json.loads((L.OUT / "launch_log.json").read_text())["session"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-session", default=None, help="default: the session in <out>/launch_log.json")
    parser.add_argument("--max-resumes", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--hours", type=float, default=6.0)
    args = parser.parse_args(argv)
    if args.base_session is None:
        args.base_session = json.loads((L.OUT / "launch_log.json").read_text())["session"]
    resumes = 0
    session = current_session()
    log(event="supervisor_started", session=session)
    stale_checks = 0
    while True:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        status = state.get("status")
        if status in TERMINAL:
            log(event="job_terminal", status=status, stage=state.get("stage"))
            return 0
        try:
            alive = session_alive(session)
        except Exception as exc:
            log(event="status_error", error=type(exc).__name__)
            time.sleep(args.poll_seconds)
            continue
        need_resume = not alive
        if alive:
            pid_alive = job_process_alive(session)
            if pid_alive is False:
                stale_checks += 1
                log(event="job_process_missing", checks=stale_checks)
                need_resume = stale_checks >= 2
            else:
                stale_checks = 0
        if need_resume:
            if resumes >= args.max_resumes:
                log(event="max_resumes_reached", resumes=resumes)
                return 2
            resumes += 1
            new_session = f"{args.base_session}-r{resumes}"
            log(event="resuming", reason="session_gone" if not alive else "job_process_missing", new_session=new_session)
            if alive:
                subprocess.run([str(L.CLI), "stop", "-s", session], capture_output=True, text=True)
            # give the old watcher a moment to exit / finish, then kill it so two watchers never overlap
            procs = json.loads((L.OUT / "local_processes.json").read_text()) if (L.OUT / "local_processes.json").exists() else {}
            if procs.get("watcher_pid"):
                subprocess.run(["kill", str(procs["watcher_pid"])], capture_output=True)
            result = subprocess.run([str(L.PY_TEST), str(ROOT / "scripts" / "colab" / "resume_tokenization_experiment.py"),
                                     "--session", new_session, "--hours", str(args.hours)], text=True, capture_output=True)
            tail = [line for line in result.stdout.splitlines() if line.startswith('{"stage"')][-3:]
            log(event="resume_finished", code=result.returncode, tail=tail, stderr=result.stderr[-400:] if result.returncode else "")
            if result.returncode == 0:
                session = new_session
                stale_checks = 0
            else:
                time.sleep(300)  # back off before trying again
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
