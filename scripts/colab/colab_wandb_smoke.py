"""Colab end-to-end smoke: allocate a GPU, install the pilot source, run a tiny training
job that logs to W&B using the Colab Secret, download the result, release the VM.

Usage: python scripts/colab/colab_wandb_smoke.py [--gpu T4|none] [--steps 60] [--keep]
Always stops the session unless --keep is passed, even on failure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "work" / "colab-client-env" / "bin" / "colab"
BUNDLE_DIR = ROOT / "outputs" / "colab-smoke"
SESSION_DEFAULT = "fjepa-wandb-smoke"

PREPARE = r'''
from pathlib import Path
import hashlib, json, subprocess, sys, zipfile
root = Path('/content/fjepa_expanded'); root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/fjepa-source.zip') as archive:
    for name in archive.namelist():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    archive.extractall(root)
manifest = json.loads(Path('/content/source_manifest.json').read_text())
bad = [n for n, d in manifest.items() if hashlib.sha256((root / n).read_bytes()).hexdigest() != d]
assert not bad, bad
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '-r', str(root / 'src/requirements.txt')], check=True)
import torch, wandb
try:
    from google.colab import userdata
    secret = {'WANDB_API_KEY': bool(userdata.get('WANDB_API_KEY')), 'FMP_API_KEY': bool(userdata.get('FMP_API_KEY'))}
except Exception as exc:
    secret = {'error': type(exc).__name__}
info = {'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch': torch.__version__, 'wandb': wandb.__version__, 'python': sys.version.split()[0], 'secrets_visible': secret}
print('SMOKE_READY ' + json.dumps(info))
'''

RUN = '''
import runpy, sys, subprocess
sys.argv = ['wandb_smoke_train.py', '--device', '{device}', '--steps', '{steps}', '--mode', 'auto',
            '--out', '/content/fjepa-wandb-smoke']
try:
    runpy.run_path('/content/fjepa_expanded/scripts/colab/wandb_smoke_train.py', run_name='__main__')
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
subprocess.run(['tar', 'czf', '/content/wandb-offline.tar.gz', '-C', '/content/fjepa-wandb-smoke', 'wandb'], check=True)
print('OFFLINE_BUNDLE_READY')
'''


def colab(*args, timeout=600, check=True):
    proc = subprocess.run([str(CLI), *args], text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode:
        raise RuntimeError(f"colab {' '.join(args[:2])} failed (exit {proc.returncode}): {proc.stderr.strip()[-800:]}")
    return proc


def exec_script(session, code, timeout=900):
    return subprocess.run([str(CLI), "exec", "-s", session, "--timeout", str(timeout)], input=code,
                          text=True, capture_output=True, timeout=timeout + 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--session", default=SESSION_DEFAULT)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--phase", default="all", choices=("all", "prepare", "run"),
                        help="prepare: allocate+install and print the browser URL (session kept); run: run on an existing prepared session")
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        args.keep = True
    zips = sorted(BUNDLE_DIR.glob("fjepa-source-*.zip"))
    if not zips:
        print("No bundle; run scripts/colab/build_source_bundle.py --out outputs/colab-smoke --name fjepa-source-<date>")
        return 2
    bundle, manifest = zips[-1], BUNDLE_DIR / "source_manifest.json"
    log = {"session": args.session, "bundle": bundle.name, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.time()
    try:
        if args.phase in ("all", "prepare"):
            new_args = ["new", "-s", args.session] + ([] if args.gpu.lower() == "none" else ["--gpu", args.gpu])
            proc = colab(*new_args, timeout=600)
            log["new"] = (proc.stdout + proc.stderr).strip()[-500:]
            print(json.dumps({"stage": "allocated", "seconds": round(time.time() - started)}), flush=True)
            for local, remote in ((bundle, "/content/fjepa-source.zip"), (manifest, "/content/source_manifest.json")):
                colab("upload", "-s", args.session, str(local), remote)
            print(json.dumps({"stage": "uploaded"}), flush=True)
            prep = exec_script(args.session, PREPARE, timeout=900)
            ready = [line for line in prep.stdout.splitlines() if line.startswith("SMOKE_READY")]
            log["prepare_stdout_tail"] = prep.stdout[-1500:]
            log["prepare_stderr_tail"] = prep.stderr[-1500:]
            if not ready:
                print(json.dumps({"stage": "prepare_failed", "stderr": prep.stderr[-1200:], "stdout": prep.stdout[-800:]}), flush=True)
                return 3
            info = json.loads(ready[-1][len("SMOKE_READY "):])
            log["runtime"] = info
            print(json.dumps({"stage": "ready", **info}), flush=True)
            (BUNDLE_DIR / "session_runtime.json").write_text(json.dumps(info, indent=2))
            if args.phase == "prepare":
                url = colab("url", "-s", args.session, timeout=120).stdout.strip()
                log["url"] = url
                print(json.dumps({"stage": "attach_browser", "session": args.session, "url": url}), flush=True)
                return 0
        else:
            info = json.loads((BUNDLE_DIR / "session_runtime.json").read_text())
        device = "cuda" if info.get("cuda") else "cpu"
        exec_script(args.session, "from pathlib import Path; Path('/content/fjepa_expanded/scripts').mkdir(exist_ok=True); print('ok')", timeout=60)
        colab("upload", "-s", args.session, str(ROOT / "scripts" / "colab" / "wandb_smoke_train.py"),
              "/content/fjepa_expanded/scripts/colab/wandb_smoke_train.py")
        run = exec_script(args.session, RUN.format(device=device, steps=args.steps), timeout=900)
        result = [line for line in run.stdout.splitlines() if line.startswith("WANDB_SMOKE")]
        log["run_stdout_tail"] = run.stdout[-2000:]
        log["run_stderr_tail"] = run.stderr[-2000:]
        if not result:
            print(json.dumps({"stage": "run_failed", "stderr": run.stderr[-1500:], "stdout": run.stdout[-1000:]}), flush=True)
            return 4
        summary = json.loads(result[-1][len("WANDB_SMOKE "):])
        log["result"] = summary
        print(json.dumps({"stage": "smoke_done", "device": summary["device"], "precision": summary["precision"],
                          "mode": summary["wandb"]["mode"], "run_id": summary["wandb"]["run_id"]}), flush=True)
        if summary["wandb"]["mode"] == "offline":
            local = BUNDLE_DIR / f"offline-smoke-colab-{time.strftime('%Y%m%d-%H%M%S')}"
            local.mkdir(parents=True, exist_ok=True)
            colab("download", "-s", args.session, "/content/wandb-offline.tar.gz", str(local / "wandb-offline.tar.gz"))
            subprocess.run(["tar", "xzf", str(local / "wandb-offline.tar.gz"), "-C", str(local)], check=True)
            sync = subprocess.run([str(ROOT / "work" / "wandb-monitor-env" / "bin" / "python"),
                                   str(ROOT / "scripts" / "colab" / "sync_offline_wandb_runs.py"), "--root", str(local)],
                                  text=True, capture_output=True)
            log["sync_stdout"] = sync.stdout[-1500:]
            print(json.dumps({"stage": "synced_from_mac", "ok": sync.returncode == 0,
                              "tail": sync.stdout.strip().splitlines()[-2:] if sync.stdout.strip() else sync.stderr[-300:]}), flush=True)
            return 0 if sync.returncode == 0 and '"ok": true' in sync.stdout else 6
        return 0 if summary["wandb"]["url"] else 5
    finally:
        if not args.keep and not (args.phase == "run" and "result" not in log):
            stop = colab("stop", "-s", args.session, timeout=300, check=False)
            log["stopped"] = stop.returncode == 0
            log["stop_output"] = (stop.stdout + stop.stderr).strip()[-300:]
            print(json.dumps({"stage": "stopped", "ok": log["stopped"]}), flush=True)
        log["elapsed_seconds"] = round(time.time() - started)
        (BUNDLE_DIR / "colab_smoke_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
