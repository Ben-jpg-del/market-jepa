# Environment for the walk-forward run on the Russell 2000 panel. `source outputs/russell2000_env.sh` before
# scripts/launch_tokenization_experiment.py / supervise_tokenization_experiment.py / resume_tokenization_experiment.py.
export FJEPA_TOK_OUT=russell2000-walkforward
export FJEPA_TOK_JOB_KIND=walkforward
export FJEPA_TOK_DATA=work/fjepa-assembled-data
export FJEPA_TOK_PANELS=panel_russell2000.npz
export FJEPA_TOK_SEEDS=42,43
export FJEPA_TOK_TOTAL_STEPS=2500          # initial (from-scratch) steps for the first year
export FJEPA_TOK_JOB_ARGS='["--continuation-steps","1000","--first-year","2015","--last-year","2025","--train-start","2006-01-01"]'
export FJEPA_TOK_REMOTE_RUN=/content/fjepa-walkforward-run
export FJEPA_TOK_REMOTE_DATA=/content/fjepa-assembled-data
export FJEPA_TOK_MODEL_OVERRIDES='{}'
