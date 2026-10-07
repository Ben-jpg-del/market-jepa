#!/bin/bash
# Orderly shutdown once the twelve finals are downloaded:
# stop the supervisor (so it cannot resume again), give the watcher one last pass,
# release the Colab session, stop the watcher, run a final W&B sync, stop the sync loop.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/colab-tokenization"
CLI="$ROOT/work/colab-client-env/bin/colab"
cd "$ROOT"
SESSION=$(python3 -c "import json;print(json.load(open('$OUT/launch.json')).get('session',''))")
echo "session: $SESSION"
pkill -f supervise_tokenization_experiment.py && echo "supervisor stopped"
# wait for the final report to be on disk locally (watcher polls every 2 min)
for i in $(seq 1 12); do
  if [ -f "$OUT/live-run/artifacts/evaluations/T2_s44_step_002500/expanded_job_report.json" ]; then echo "final report downloaded"; break; fi
  sleep 20
done
"$CLI" stop -s "$SESSION" 2>&1 | tail -1
pkill -f watch_colab_run.py && echo "watcher stopped"
"$ROOT/work/wandb-monitor-env/bin/python" scripts/colab/sync_offline_wandb_runs.py --root "$OUT/live-run" 2>&1 | tail -1
pkill -f sync_offline_wandb_runs.py && echo "wandb sync loop stopped"
pkill -f "caffeinate -i" 2>/dev/null
"$CLI" sessions 2>&1 | tail -3
echo "local processes remaining:"; pgrep -fl "watch_colab_run|sync_offline|supervise_tokenization|local_eval_seed" || echo "none"
