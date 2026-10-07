"""Resume the tokenization experiment on a fresh Colab T4 from locally downloaded artifacts.

Builds a restore archive (job state, run table, per-run metadata/results/histories, every
encoder export, the latest full checkpoint of runs not yet at 2,500, finished evaluation
JSON), provisions a new session, installs the source bundle, re-transfers the two panels,
extracts the archive into the job directory and relaunches ``expanded_job`` with the same
settings so it continues from the saved state. Starts a new local watcher for the session.

Usage: work/colab-test-env/bin/python scripts/colab/resume_tokenization_experiment.py --session fjepa-tok-20260918-r1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "colab"))
import launch_tokenization_experiment as L  # noqa: E402

ARTIFACTS = L.OUT / "live-run" / "artifacts"
RECOVERY = L.OUT / "recovery"

RESTORE = r'''
from pathlib import Path
import hashlib, json, tarfile
out = Path('REMOTE_RUN_LITERAL'); out.mkdir(exist_ok=True)
with tarfile.open('/content/restore.tar.gz', 'r:gz') as archive:
    for member in archive.getmembers():
        assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    archive.extractall(out, filter='data')
manifest = json.loads(Path('/content/restore_manifest.json').read_text())
bad = [n for n, d in manifest.items() if hashlib.sha256((out / n).read_bytes()).hexdigest() != d]
assert not bad, bad[:5]
state = json.loads((out / 'expanded_job_state.json').read_text())
print('TOK_RESTORED ' + json.dumps({'files': len(manifest), 'completed_training': sum(1 for v in state['training'].values() if v.get('status') == 'completed'),
      'evaluations': len(state.get('evaluations', {})), 'status': state.get('status'), 'stage': state.get('stage')}))
'''


def build_restore_archive():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    state = json.loads((ARTIFACTS / "expanded_job_state.json").read_text())
    files = []
    for name in ("expanded_job_state.json", "runs.json", "runtime_preflight.json", "validation.log"):
        if (ARTIFACTS / name).exists():
            files.append(ARTIFACTS / name)
    if state.get("job") == "walkforward":
        for run_dir in sorted(p for p in ARTIFACTS.glob("*_seed*") if p.is_dir()):
            files.extend(sorted(run_dir.glob("encoder_*.pt")))
            for year_dir in sorted(p for p in run_dir.glob("year_*") if p.is_dir()):
                for name in ("run_metadata.json", "run_result.json", "training_metrics.json"):
                    if (year_dir / name).exists():
                        files.append(year_dir / name)
                year = year_dir.name.split("_")[-1]
                fulls = sorted(year_dir.glob("step_*.pt"))
                if fulls and not (run_dir / f"encoder_{year}.pt").exists():
                    files.append(fulls[-1])  # exact-resume state for the unfinished year
        files.extend(sorted(p for p in ARTIFACTS.glob("evaluations/**/*.json") if "cache" not in p.parts))  # features npz / caches are regenerated
        return _pack(files, state)
    total_steps = state["settings"]["total_steps"]
    for run_dir in sorted(p for p in ARTIFACTS.glob("*_seed*") if p.is_dir()):
        for name in ("run_metadata.json", "run_result.json", "training_metrics.json"):
            if (run_dir / name).exists():
                files.append(run_dir / name)
        files.extend(sorted(run_dir.glob("encoder_step_*.pt")))
        finished = (run_dir / f"encoder_step_{total_steps:06d}.pt").exists()
        fulls = sorted(run_dir.glob("step_*.pt"))
        if fulls and not finished:
            files.append(fulls[-1])  # exact-resume state for the unfinished run
    files.extend(sorted(ARTIFACTS.glob("evaluations/**/*.json")))
    files.extend(sorted(ARTIFACTS.glob("evaluations/**/*.csv.gz")))
    return _pack(files, state)


def _pack(files, state):
    manifest, archive_path = {}, RECOVERY / "restore.tar.gz"
    # A restored state must not look terminal: the new watcher polls it before the job rewrites the status,
    # and it would stop the runtime on "failed"/"completed". Ship it as "paused_resume" instead.
    neutral = None
    if state.get("status") in ("failed", "completed", "completed_with_deferred_intermediate"):
        neutral = RECOVERY / "expanded_job_state.json"
        patched = {**state, "status": "paused_resume", "status_before_resume": state.get("status")}
        neutral.write_text(json.dumps(patched, indent=1))
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for path in files:
            rel = path.relative_to(ARTIFACTS).as_posix()
            source = neutral if (neutral is not None and rel == "expanded_job_state.json") else path
            archive.add(source, arcname=rel, recursive=False)
            manifest[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
    (RECOVERY / "restore_manifest.json").write_text(json.dumps(manifest, indent=1))
    return archive_path, manifest, state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--group", default=None, help="must match the original W&B group; default read from launch_log.json")
    args = parser.parse_args(argv)
    launch_log = json.loads((L.OUT / "launch_log.json").read_text())
    group = args.group or launch_log["group"]
    log = {"session": args.session, "group": group, "steps": []}

    def stage(name, **info):
        log["steps"].append({"stage": name, "at": datetime.now(timezone.utc).isoformat(), **info})
        (RECOVERY / f"resume_log_{args.session}.json").write_text(json.dumps(log, indent=2))
        print(json.dumps({"stage": name, **info}), flush=True)

    archive_path, manifest, state = build_restore_archive()
    stage("restore_archive_ready", files=len(manifest), mb=round(archive_path.stat().st_size / 1e6, 1),
          completed_training=sum(1 for v in state["training"].values() if v.get("status") == "completed"),
          evaluations=len(state.get("evaluations", {})))
    bundles = sorted(L.OUT.glob("fjepa-source-tok-*.zip"))
    bundle = bundles[-1]
    L.colab("new", "-s", args.session, "--gpu", args.gpu)
    stage("allocated")
    try:
        L.colab("upload", "-s", args.session, str(bundle), "/content/fjepa-source.zip")
        L.colab("upload", "-s", args.session, str(L.OUT / "source_manifest.json"), "/content/source_manifest.json")
        out, err, _ = L.exec_code(args.session, L.render(L.PREPARE, args.hours, group), timeout=1200)
        ready = [line for line in out.splitlines() if line.startswith("TOK_RUNTIME_READY")]
        if not ready:
            stage("prepare_failed", stdout=out[-1200:], stderr=err[-1200:])
            raise SystemExit(3)
        stage("runtime_ready", **json.loads(ready[-1][len("TOK_RUNTIME_READY "):]))
        L.transfer_panels(args.session)
        stage("panels_transferred")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "colab" / "upload_colab_file.py"), "--cli", str(L.CLI),
                        "--session", args.session, "--source", str(archive_path), "--remote", "/content/restore.tar.gz",
                        "--manifest", str(RECOVERY / "restore_transfer.json")], check=True)
        L.colab("upload", "-s", args.session, str(RECOVERY / "restore_manifest.json"), "/content/restore_manifest.json")
        out, err, _ = L.exec_code(args.session, RESTORE.replace("REMOTE_RUN_LITERAL", L.REMOTE_RUN), timeout=600)
        restored = [line for line in out.splitlines() if line.startswith("TOK_RESTORED")]
        if not restored:
            stage("restore_failed", stdout=out[-1200:], stderr=err[-1200:])
            raise SystemExit(5)
        stage("restored", **json.loads(restored[-1][len("TOK_RESTORED "):]))
        out, err, _ = L.exec_code(args.session, L.render(L.LAUNCH, args.hours, group), timeout=120)
        launched = [line for line in out.splitlines() if line.startswith("TOK_LAUNCH")]
        if not launched:
            stage("launch_failed", stdout=out[-1200:], stderr=err[-1200:])
            raise SystemExit(4)
        launch = json.loads(launched[-1][len("TOK_LAUNCH "):])
        (L.OUT / "launch.json").write_text(json.dumps({**launch, "session": args.session}, indent=2))
        stage("job_resumed", started_at=launch["started_at"], pid=launch["pid"])
    except BaseException:
        if not any(s["stage"] == "job_resumed" for s in log["steps"]):
            subprocess.run([str(L.CLI), "stop", "-s", args.session], capture_output=True, text=True)
            stage("runtime_released_after_failure")
        raise
    live = L.OUT / "live-run"
    watcher = subprocess.Popen(
        ["caffeinate", "-i", "-s", str(L.PY_TEST), str(ROOT / "scripts" / "colab" / "watch_colab_run.py"),
         "--session", args.session, "--remote-output", L.REMOTE_RUN, "--output", str(live),
         "--job-start", launch["started_at"], "--hours", str(args.hours), "--grace-minutes", "25",
         "--poll-seconds", "120", "--state-file", "expanded_job_state.json"],
        stdout=(L.OUT / "watcher.stdout.log").open("ab"), stderr=subprocess.STDOUT, start_new_session=True)
    procs = json.loads((L.OUT / "local_processes.json").read_text()) if (L.OUT / "local_processes.json").exists() else {}
    procs.update({"watcher_pid": watcher.pid, "session": args.session, "resumed": datetime.now(timezone.utc).isoformat()})
    (L.OUT / "local_processes.json").write_text(json.dumps(procs, indent=2))
    stage("local_watcher_started", watcher_pid=watcher.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
