# Colab job: JEPA (seeds 42/43) and supervised arm on the 426 large-cap assembled panel. Usage: source outputs/largecap426_env.sh
export FJEPA_TOK_OUT=arms-orig426
export FJEPA_TOK_JOB_KIND=walkforward
export FJEPA_TOK_DATA=work/fjepa-assembled-data
export FJEPA_TOK_PANELS=panel_largecap426.npz
export FJEPA_TOK_RUNS_JSON=outputs/arms-orig426/runs.json
export FJEPA_TOK_SEEDS=42,43
export FJEPA_TOK_TOTAL_STEPS=2500
export FJEPA_TOK_JOB_ARGS='["--continuation-steps","1000","--first-year","2015","--last-year","2025"]'
export FJEPA_TOK_REMOTE_RUN=/content/fjepa-largecap-run
export FJEPA_TOK_REMOTE_DATA=/content/fjepa-largecap-data
