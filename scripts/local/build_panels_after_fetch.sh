#!/bin/bash
# Everything after Ben's FMP fetches for the Russell 2000 walk-forward run (no key needed here).
#   1. family channels + fundamentals for the Russell 2000 panel (work/fjepa-families-russell2000/)
#   2. the 426 panel's families rebuilt with the refreshed macro series (copied into work/fjepa-families/)
#   3. assembled panels for both universes, coverage-checked
#   4. Phase-0 gate on both (informational)
# Usage: scripts/local/build_panels_after_fetch.sh 2>&1 | tee outputs/russell2000-walkforward/after_fetch.log
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=work/colab-test-env/bin/python
R2K=work/fjepa-russell2000-data; FAM_R2K=work/fjepa-families-russell2000; FAM=work/fjepa-families; ASM=work/fjepa-assembled-data
for f in $R2K/panel_raw.npz $FAM_R2K/macro_raw.json $FAM_R2K/earnings_raw.json $FAM_R2K/statements_raw.json; do
  [ -f "$f" ] || { echo "MISSING $f (run the fetch step first)"; exit 1; }
done
echo "== 1. Russell 2000 families"
$PY scripts/data/build_family_channels.py --panel $R2K/panel_raw.npz --families-dir $FAM_R2K --suffix _r2k > $FAM_R2K/build_family.log
$PY scripts/data/build_fundamentals_channels.py --panel $R2K/panel_raw.npz --families-dir $FAM_R2K --suffix _r2k > $FAM_R2K/build_fundamentals.log
echo "== 2. 426 panel families with the refreshed macro (2003-2025)"
cp $FAM/macro_raw.json $FAM/macro_raw.pre_2025_refetch.json 2>/dev/null || true
cp $FAM_R2K/macro_raw.json $FAM/macro_raw.json
$PY scripts/data/build_family_channels.py --panel work/fjepa-cloud-data/panel_raw.npz --families-dir $FAM --suffix _orig2025 > $FAM/build_family_orig2025.log
echo "== 3. assembled panels"
mkdir -p $ASM
$PY scripts/data/build_assembled_panel.py --base $R2K/panel_raw.npz --families $FAM_R2K/family_channels_r2k.npz --fundamentals $FAM_R2K/fundamentals_channels_r2k.npz --out $ASM/panel_russell2000.npz --min-coverage 0.85
$PY scripts/data/build_assembled_panel.py --base work/fjepa-cloud-data/panel_raw.npz --families $FAM/family_channels_orig2025.npz --fundamentals $FAM/fundamentals_channels_orig2025.npz --out $ASM/panel_largecap426.npz --min-coverage 0.85
echo "== 4. Phase-0 gate (informational)"
mkdir -p outputs/raw-feature-gate
$PY scripts/eval/raw_feature_gate.py --panel $ASM/panel_russell2000.npz --tag russell2000 | tail -1
$PY scripts/eval/raw_feature_gate.py --panel $ASM/panel_largecap426.npz --tag orig426 | tail -1
echo "AFTER_FETCH_DONE"
