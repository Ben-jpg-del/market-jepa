"""Sync offline W&B run directories (written on a Colab runtime without a key) from this Mac.

Finds every ``wandb/offline-run-*`` folder under the given roots (default: outputs/), skips
ones already synced, and runs ``wandb sync`` with the login saved on this machine. Safe to
re-run; it is what the Codex-era uploader did, but for the richer per-step logs the new
training logger writes. Pair it with the artifact watcher so runs sync as checkpoints land.

Usage:
  work/wandb-monitor-env/bin/python scripts/colab/sync_offline_wandb_runs.py [--root outputs/colab-expanded] [--watch 120]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "outputs" / "wandb-local-runs" / "offline_sync_state.json"
ENTITY = "bzhou1018-uc-berkeley-electrical-engineering-computer-sc"
PROJECT = "f-jepa"


def find_runs(roots):
    for root in roots:
        yield from sorted(Path(root).rglob("wandb/offline-run-*"))


def run_is_complete(path: Path) -> bool:
    # wandb writes .wandb data file continuously; a finished run has run-*.wandb closed and
    # files/wandb-summary.json present. Incomplete runs can still be synced (partial upload)
    # and re-synced later, which wandb handles idempotently by run id.
    return any(path.glob("run-*.wandb"))


def sync_once(roots, entity, project, state, wandb_bin):
    synced = 0
    for run_dir in find_runs(roots):
        if not run_is_complete(run_dir):
            continue
        key = str(run_dir)
        mtime = run_dir.stat().st_mtime
        if state.get(key, {}).get("mtime") == mtime and state[key].get("ok"):
            continue
        proc = subprocess.run([str(wandb_bin), "sync", "--entity", entity, "--project", project, str(run_dir)],
                              text=True, capture_output=True)
        ok = proc.returncode == 0
        state[key] = {"mtime": mtime, "ok": ok, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "tail": (proc.stdout + proc.stderr).strip()[-300:]}
        print(json.dumps({"run": run_dir.name, "ok": ok, "tail": state[key]["tail"][-160:]}), flush=True)
        synced += ok
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))
    return synced


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--watch", type=int, default=0, help="seconds between passes; 0 = single pass")
    args = parser.parse_args(argv)
    roots = args.root or [str(ROOT / "outputs")]
    wandb_bin = Path(sys.executable).with_name("wandb")
    if not wandb_bin.exists():
        print("run with work/wandb-monitor-env/bin/python (wandb CLI not next to this interpreter)")
        return 2
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    while True:
        n = sync_once(roots, args.entity, args.project, state, wandb_bin)
        print(json.dumps({"pass_complete": True, "synced": n}), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
