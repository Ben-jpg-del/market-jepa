#!/bin/bash
# Local (Mac GPU, MPS) walk-forward arms after the walk-forward verdict: supervised encoder (pretext off, aux head at full weight),
# width-256 supervised, hybrid (pretext + full aux), Sharpe loss, volatility-rank target; then the 426 large-cap assembled panel.
# Same job code as the Colab run (src.walkforward_job); resumable: rerun this script and finished stages are skipped.
#   nohup scripts/local/run_walkforward_arms_local.sh > outputs/arms-russell2000/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=work/colab-test-env/bin/python
run_job() {  # out panel
  local out=$1 panel=$2
  echo "== $(date -u +%FT%TZ) start $out"
  caffeinate -i $PY -m src.walkforward_job --panel "$panel" --output "$out" --device mps --hours 48 \
      --runs-json "$out/runs.json" --wandb-mode offline --wandb-group "arms-$(basename $out)" >> "$out/job.log" 2>&1
  echo "== $(date -u +%FT%TZ) end $out exit $?"
}

export PYTHONPATH="."
run_job outputs/arms-russell2000 work/fjepa-assembled-data/panel_russell2000.npz
# the 426 large-cap panel (outputs/arms-orig426) runs on Colab in parallel (outputs/largecap426_env.sh); run it here only as a fallback:
#   run_job outputs/arms-orig426 work/fjepa-assembled-data/panel_largecap426.npz
echo "ARMS_DONE $(date -u +%FT%TZ)"
