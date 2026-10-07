"""Launch the tokenization / target experiment on a Colab T4 through the CLI.

Arms (all expanded universe, book indicators, 2,500 updates, 500-update export):
  F0  historical masked block, fast-only target            (control = expanded_F)
  F1  historical masked block, ALL channels in the target
  T1  family tokens, hide-one-family objective              (T-JEPA analogue)
  T2  family tokens, original fast-only historical target
Seeds 42/43/44. Final checkpoints evaluated on common+expanded cohorts; 500-update
checkpoints on the common cohort; 1,000/1,500 for seed 42 on the common cohort.

Steps: build bundle -> allocate T4 -> install + tests on the VM -> transfer both
panels (verified chunks) -> start the job detached on the VM -> start the local
download watcher (which also releases the runtime) and the offline W&B sync loop.
No secrets are sent to the VM; W&B runs are written offline and synced from here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "work" / "colab-client-env" / "bin" / "colab"
PY_TEST = ROOT / "work" / "colab-test-env" / "bin" / "python"
PY_WANDB = ROOT / "work" / "wandb-monitor-env" / "bin" / "python"
OUT = ROOT / "outputs" / os.environ.get("FJEPA_TOK_OUT", "colab-tokenization")
DATA = ROOT / "work" / "fjepa-expanded-data"  # legacy alias; DATA_DIR is the configurable one
REMOTE_DATA = os.environ.get("FJEPA_TOK_REMOTE_DATA", "/content/fjepa-expanded-data")
EXPANDED_PANEL = os.environ.get("FJEPA_TOK_EXPANDED_PANEL", "panel_raw.npz")  # file under work/fjepa-expanded-data
MODEL_OVERRIDES = json.loads(os.environ.get("FJEPA_TOK_MODEL_OVERRIDES", "{}"))  # PilotConfig fields applied to every run
# Job kind: "expanded" (two-panel src.expanded_job, the default) or "walkforward" (single-panel src.walkforward_job)
JOB_KIND = os.environ.get("FJEPA_TOK_JOB_KIND", "expanded")
JOB_MODULE = os.environ.get("FJEPA_TOK_JOB_MODULE", "src.expanded_job" if JOB_KIND == "expanded" else "src.walkforward_job")
JOB_ARGS = json.loads(os.environ.get("FJEPA_TOK_JOB_ARGS", "[]"))  # extra CLI args appended to the job command
DATA_DIR = ROOT / os.environ.get("FJEPA_TOK_DATA", "work/fjepa-expanded-data")
# panels to upload: local file names under DATA_DIR; for the expanded kind the remote names are fixed
PANELS = [x for x in os.environ.get("FJEPA_TOK_PANELS", "").split(",") if x] or (["original_panel.npz", EXPANDED_PANEL] if JOB_KIND == "expanded" else [])
ARM_VARIANTS = {"F0": "F", "F1": "F1", "T1": "T1", "T2": "T2", "T3": "T3"}
ARMS = {a: ARM_VARIANTS[a] for a in os.environ.get("FJEPA_TOK_ARMS", "F0,F1,T1,T2").split(",")}
SEEDS = tuple(int(x) for x in os.environ.get("FJEPA_TOK_SEEDS", "42,43,44").split(","))
TOTAL_STEPS = int(os.environ.get("FJEPA_TOK_TOTAL_STEPS", "2500"))
INTERMEDIATE_STEPS = int(os.environ.get("FJEPA_TOK_INTERMEDIATE_STEPS", "500"))
EXTRA_STEPS = [int(x) for x in os.environ.get("FJEPA_TOK_EXTRA_STEPS", "1000,1500").split(",") if x]
REMOTE_RUN = os.environ.get("FJEPA_TOK_REMOTE_RUN", "/content/fjepa-tok-run")

PREPARE = r'''
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile
root = Path('/content/fjepa_expanded'); root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/fjepa-source.zip') as archive:
    for name in archive.namelist():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    archive.extractall(root)
manifest = json.loads(Path('/content/source_manifest.json').read_text())
bad = [n for n, d in manifest.items() if hashlib.sha256((root / n).read_bytes()).hexdigest() != d]
assert not bad, bad
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '-r', str(root / 'src/requirements.txt')], check=True)
env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_MODE='disabled')
tests = sorted(str(p.relative_to(root)) for p in (root / 'tests').glob('test_*.py'))
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests], cwd=root, env=env, text=True, capture_output=True)
Path('REMOTE_RUN_LITERAL').mkdir(exist_ok=True); Path('REMOTE_DATA_LITERAL').mkdir(exist_ok=True)
Path('REMOTE_RUN_LITERAL/validation.log').write_text(result.stdout + result.stderr)
assert result.returncode == 0, result.stdout[-1500:]
import torch
info = {'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch': torch.__version__,
        'python': sys.version.split()[0], 'disk_free_gb': round(shutil.disk_usage('/content').free / 1024**3, 1),
        'tests': result.stdout.strip().splitlines()[-1]}
assert info['gpu'], 'No GPU on this runtime'
assert info['disk_free_gb'] > 60, 'Insufficient scratch disk for evaluation caches'
Path('REMOTE_RUN_LITERAL/runtime_preflight.json').write_text(json.dumps(info, indent=2))
print('TOK_RUNTIME_READY ' + json.dumps(info))
'''

LAUNCH = r'''
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys, threading
root = Path('/content/fjepa_expanded'); out = Path('REMOTE_RUN_LITERAL'); out.mkdir(exist_ok=True)
assert not (out / 'launch.json').exists(), 'A launch is already recorded'
for name in PANELS_LITERAL:
    assert (Path('REMOTE_DATA_LITERAL') / name).is_file(), name
command = [sys.executable] + COMMAND_LITERAL
env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
with (out / 'training.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
launch = {'pid': process.pid, 'started_at': datetime.now(timezone.utc).isoformat(), 'training_budget_hours': HOURS_LITERAL,
          'remote_release_backstop_hours': BACKSTOP_LITERAL, 'command': command, 'status': 'process_started'}
(out / 'launch.json').write_text(json.dumps(launch, indent=2))
def release_backstop():
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as exc:
        (out / 'release_backstop_error.json').write_text(json.dumps({'type': type(exc).__name__}))
timer = threading.Timer(BACKSTOP_LITERAL * 3600, release_backstop); timer.daemon = True; timer.start()
print('TOK_LAUNCH ' + json.dumps(launch))
'''


def remote_panels():
    """remote file name per local panel; the expanded job expects fixed names"""
    if JOB_KIND == "expanded":
        return {PANELS[0]: "original_panel.npz", PANELS[1]: "expanded_panel.npz"}
    return {name: name for name in PANELS}


def job_command(hours, group):
    out = REMOTE_RUN
    if JOB_KIND == "expanded":
        cmd = ["-u", "-m", JOB_MODULE, "--original-panel", f"{REMOTE_DATA}/original_panel.npz", "--expanded-panel", f"{REMOTE_DATA}/expanded_panel.npz",
               "--output", out, "--hours", str(hours), "--device", "cuda", "--total-steps", str(TOTAL_STEPS), "--intermediate-steps", str(INTERMEDIATE_STEPS),
               "--runs-json", f"{out}/runs.json", "--final-cohorts", "common,expanded", "--intermediate-cohorts", "common",
               "--wandb-mode", "auto", "--wandb-group", group]
    else:
        panel = remote_panels()[PANELS[0]]
        cmd = ["-u", "-m", JOB_MODULE, "--panel", f"{REMOTE_DATA}/{panel}", "--output", out, "--hours", str(hours), "--device", "cuda",
               "--runs-json", f"{out}/runs.json", "--initial-steps", str(TOTAL_STEPS), "--wandb-mode", "auto", "--wandb-group", group]
    return cmd + [str(a) for a in JOB_ARGS]


def render(template, hours, group):
    return (template.replace("HOURS_LITERAL", repr(hours)).replace("BACKSTOP_LITERAL", repr(hours + 0.5))
            .replace("GROUP_LITERAL", group).replace("TOTAL_LITERAL", str(TOTAL_STEPS))
            .replace("INTERMEDIATE_LITERAL", str(INTERMEDIATE_STEPS)).replace("REMOTE_RUN_LITERAL", REMOTE_RUN)
            .replace("REMOTE_DATA_LITERAL", REMOTE_DATA).replace("PANELS_LITERAL", json.dumps(list(remote_panels().values())))
            .replace("COMMAND_LITERAL", json.dumps(job_command(hours, group))))


def colab(*args, timeout=600):
    proc = subprocess.run([str(CLI), *args], text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"colab {args[0]} failed (exit {proc.returncode}): {proc.stderr.strip()[-600:]}")
    return proc.stdout


def exec_code(session, code, timeout=900):
    proc = subprocess.run([str(CLI), "exec", "-s", session, "--timeout", str(timeout)], input=code,
                          text=True, capture_output=True, timeout=timeout + 60)
    return proc.stdout, proc.stderr, proc.returncode


RUNS_JSON = os.environ.get("FJEPA_TOK_RUNS_JSON", "")  # optional: a file holding the complete run table (per-run model_overrides)


def run_table():
    runs = {}
    if RUNS_JSON:
        return json.loads((ROOT / RUNS_JSON).read_text())
    if JOB_KIND == "walkforward":
        for seed in SEEDS:
            spec = {"seed": seed}
            if MODEL_OVERRIDES:
                spec["model_overrides"] = dict(MODEL_OVERRIDES)
            runs[f"v2_s{seed}"] = spec
        return runs
    for seed in SEEDS:
        for arm, variant in ARMS.items():
            spec = {"panel": "expanded", "variant": variant, "horizon": 5, "readout_horizons": [5], "seed": seed,
                    "arm": arm}
            if seed == SEEDS[0] and EXTRA_STEPS:
                spec["extra_evaluation_steps"] = EXTRA_STEPS
            if MODEL_OVERRIDES:
                spec["model_overrides"] = dict(MODEL_OVERRIDES)
            runs[f"{arm}_s{seed}"] = spec
    return runs


def panel_sha256(local_name):
    import hashlib
    digest = hashlib.sha256()
    with (DATA_DIR / local_name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expanded_panel_sha256():  # kept for older callers
    return panel_sha256(EXPANDED_PANEL)


def transfer_panels(session):
    """Upload every configured panel and verify its digest; returns {local_name: sha256}."""
    got = {}
    for local, remote in remote_panels().items():
        label = Path(local).stem
        digest = transfer(session, DATA_DIR / local, f"{REMOTE_DATA}/{remote}", label)
        assert digest == panel_sha256(local), f"panel digest mismatch for {local}"
        got[local] = digest
    return got


def transfer(session, source, remote, label):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "colab" / "upload_colab_file.py"), "--cli", str(CLI),
                    "--session", session, "--source", str(source), "--remote", remote,
                    "--manifest", str(OUT / f"{label}_transfer.json")], check=True)
    transferred = json.loads((OUT / f"{label}_transfer.json").read_text())
    return transferred["sha256"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=f"fjepa-tok-{datetime.now(timezone.utc):%Y%m%d}")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--hours", type=float, default=6.5, help="job wall-clock budget on the VM")
    parser.add_argument("--group", default=f"tokenization-{datetime.now(timezone.utc):%Y%m%d}")
    parser.add_argument("--skip-transfer", action="store_true", help="panels already on the runtime")
    parser.add_argument("--reuse-session", action="store_true", help="attach to an existing session instead of allocating a new runtime")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs.json").write_text(json.dumps(run_table(), indent=2))
    log = {"session": args.session, "group": args.group, "hours": args.hours, "steps": [],
           "arms": ARMS, "seeds": SEEDS, "total_steps": TOTAL_STEPS, "intermediate_steps": INTERMEDIATE_STEPS,
           "extra_steps": EXTRA_STEPS, "remote_run": REMOTE_RUN, "out": str(OUT), "remote_data": REMOTE_DATA,
           "expanded_panel": EXPANDED_PANEL, "model_overrides": MODEL_OVERRIDES, "job_kind": JOB_KIND, "job_module": JOB_MODULE,
           "job_args": JOB_ARGS, "data_dir": str(DATA_DIR), "panels": PANELS}

    def stage(name, **info):
        log["steps"].append({"stage": name, "at": datetime.now(timezone.utc).isoformat(), **info})
        (OUT / "launch_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps({"stage": name, **info}), flush=True)

    bundle_name = f"fjepa-source-tok-{datetime.now(timezone.utc):%Y%m%d}"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "colab" / "build_source_bundle.py"), "--out", str(OUT),
                    "--name", bundle_name], check=True)
    stage("bundle_built", zip=f"{bundle_name}.zip")

    if args.reuse_session:
        stage("reusing_session", session=args.session)
    else:
        colab("new", "-s", args.session, "--gpu", args.gpu)
        stage("allocated")
    try:
        colab("upload", "-s", args.session, str(OUT / f"{bundle_name}.zip"), "/content/fjepa-source.zip")
        colab("upload", "-s", args.session, str(OUT / "source_manifest.json"), "/content/source_manifest.json")
        out, err, code = exec_code(args.session, render(PREPARE, args.hours, args.group), timeout=1200)
        ready = [line for line in out.splitlines() if line.startswith("TOK_RUNTIME_READY")]
        if not ready:
            stage("prepare_failed", stdout=out[-1500:], stderr=err[-1500:])
            raise SystemExit(3)
        runtime = json.loads(ready[-1][len("TOK_RUNTIME_READY "):])
        stage("runtime_ready", **runtime)

        if not args.skip_transfer:
            transfer_panels(args.session)
            stage("panels_transferred")

        colab("upload", "-s", args.session, str(OUT / "runs.json"), f"{REMOTE_RUN}/runs.json")
        out, err, code = exec_code(args.session, render(LAUNCH, args.hours, args.group), timeout=120)
        launched = [line for line in out.splitlines() if line.startswith("TOK_LAUNCH")]
        if not launched:
            stage("launch_failed", stdout=out[-1200:], stderr=err[-1200:])
            raise SystemExit(4)
        launch = json.loads(launched[-1][len("TOK_LAUNCH "):])
        (OUT / "launch.json").write_text(json.dumps(launch, indent=2))
        stage("job_started", started_at=launch["started_at"], pid=launch["pid"])
    except BaseException:
        # Anything failing before the job is running must not leave a billable VM behind.
        if not (OUT / "launch.json").exists():
            subprocess.run([str(CLI), "stop", "-s", args.session], capture_output=True, text=True)
            stage("runtime_released_after_failure")
        raise

    # Local watcher: downloads artifacts as they appear and releases the runtime at the end.
    live = OUT / "live-run"
    live.mkdir(exist_ok=True)
    watcher = subprocess.Popen(
        ["caffeinate", "-i", "-s", str(PY_TEST), str(ROOT / "scripts" / "colab" / "watch_colab_run.py"),
         "--session", args.session, "--remote-output", REMOTE_RUN, "--output", str(live),
         "--job-start", launch["started_at"], "--hours", str(args.hours), "--grace-minutes", "25",
         "--poll-seconds", "120", "--state-file", "expanded_job_state.json"],
        stdout=(OUT / "watcher.stdout.log").open("ab"), stderr=subprocess.STDOUT, start_new_session=True)
    syncer = subprocess.Popen(
        ["caffeinate", "-i", str(PY_WANDB), str(ROOT / "scripts" / "colab" / "sync_offline_wandb_runs.py"),
         "--root", str(live), "--watch", "180"],
        stdout=(OUT / "wandb-sync.stdout.log").open("ab"), stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / "local_processes.json").write_text(json.dumps({"watcher_pid": watcher.pid, "wandb_sync_pid": syncer.pid,
                                                          "started": datetime.now(timezone.utc).isoformat()}, indent=2))
    stage("local_watchers_started", watcher_pid=watcher.pid, wandb_sync_pid=syncer.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
