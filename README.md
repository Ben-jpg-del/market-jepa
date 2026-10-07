# market-jepa

Joint-embedding predictive architecture (JEPA) for cross-sectional stock panels, with a walk-forward
training and readout harness, point-in-time panel builders (iShares IWB / IWM rosters + FMP prices,
earnings, statements, macro), and the Colab / local run tooling. Code only: data, runs and results live
outside the repo (see `.gitignore`).

## Layout

```
src/      the package (encoder, objectives, samplers, readouts, jobs)
  model.py            PilotConfig, FutureJEPA: temporal + cross-stock transformer, EMA target encoder,
                      family tokens, objectives block / family_mask / feature_mask / future_family (current),
                      SIGReg (projection head), aux head (return-rank or vol-rank; MSE or Sharpe loss)
  training.py         train_steps (AdamW, warm-up, EMA, checkpoints, exact resume), export_encoder
  data.py             MarketPanel, normaliser, DateBatchSampler, FMP client, synthetic panel
  features.py         channel layouts (original 14-channel, current 26-channel), families, eligibility channels
  walkforward_data.py WalkForwardSampler (future windows ending at t+h ≤ cutoff), aux targets
  panel_assembly.py   assemble / check a 26-channel panel from a base panel + family files
  walkforward_readout.py per-year feature extraction, walk-forward ridge, daily IC, block bootstrap
  walkforward_job.py  yearly train → features → readout → summary, resumable (CLI)
  evaluation.py       labels (SPY-relative forward return), rank IC, ridge readout, purging, bootstrap
  expanded_data.py    iShares holdings history (FundSpec: IWB, IWM), parsing, symbol maps
  expanded_eval.py    cohorts, peer groups, cached encoder / raw features
  expanded_job.py     original cloud job (fixed-window pretraining, step-wise evaluations)
  cloud_data.py, cloud_job.py, runner.py, preflight.py, wandb_logging.py
tests/            pytest suite for the package and the Colab watcher (157 tests)
scripts/          command-line tools
  data/   prepare_expanded_panel.py (IWB), prepare_russell2000_panel.py (IWM), extend_panel_recent.py,
          quarantine_expanded_panel.py, fetch_fmp_families.py, fmp_endpoint_check.py,
          build_family_channels.py, build_fundamentals_channels.py, build_timesfm_family.py,
          augment_panel.py, build_assembled_panel.py
  eval/   supervised_bar.py, raw_feature_gate.py, indicator_ic.py, incremental_value_eval.py,
          incremental_value_families.py, local_predictions.py, stacking_test.py,
          summarize_walkforward_results.py, summarize_tokenization_results.py, summarize_local_seeds.py,
          local_eval_seed.py, local_eval_poller.py, heldout_pretext_loss.py,
          channel_predictability.py, diagnose_channel_recoverability.py
  colab/  build_source_bundle.py, launch_tokenization_experiment.py, resume_tokenization_experiment.py,
          supervise_tokenization_experiment.py, watch_colab_run.py, upload_colab_file.py,
          upload_colab_data.py, sync_offline_wandb_runs.py, colab_wandb_smoke.py, wandb_smoke_train.py,
          wrap_up_tokenization_experiment.sh
  local/  run_walkforward_arms_local.sh (Mac GPU walk-forward arms), build_panels_after_fetch.sh (panel → families → assembled panel → gate)
configs/          environment files for the Colab launcher (one per experiment)
requirements/     pinned requirement sets for the local test env, the Colab client env, the Colab runtime
```

## Conventions

- Data directories are expected under `work/` (ignored): `work/fjepa-cloud-data/`, `work/fjepa-expanded-data/`,
  `work/fjepa-russell2000-data/`, `work/fjepa-families*/`, `work/fjepa-assembled-data/`. Results go under `outputs/` (ignored).
- The FMP key is read from `FMP_API_KEY` or `~/.config/fmp/api_key`; the W&B key from `~/.netrc`. Nothing in the
  repo holds a secret.
- Every readout is a mean daily cross-sectional rank IC of the 5- or 20-day SPY-relative return, with per-date rank
  targets, a purge at year boundaries, and 20-day moving-block bootstrap intervals for paired comparisons.

## Run

```bash
python -m pytest tests -q
python -m src.walkforward_job --panel work/fjepa-assembled-data/panel_russell2000.npz --output outputs/<run> \
    --device mps --runs-json outputs/<run>/runs.json          # local; --device cuda on Colab
source configs/russell2000_env.sh && python scripts/colab/launch_tokenization_experiment.py --session <name> --gpu T4 --hours 6
```
