# src — the market-jepa package

Encoder, objectives, samplers, readouts and the resumable training jobs. Import as `src.<module>` from the repo root
(`sys.path` includes the root; the scripts insert it).

| module | contents |
|---|---|
| `model.py` | `PilotConfig`; `FutureJEPA`: temporal transformer over 5-day patches (one token stream per channel family), cross-stock transformer over the peer group, EMA target encoder, predictor with family + horizon queries, SIGReg on a throw-away projection head, aux head (return-rank or vol-rank target; MSE or Sharpe loss); objectives `block`, `family_mask`, `feature_mask`, `future_family` |
| `training.py` | `TrainConfig`, `train_steps` (AdamW, warm-up, EMA schedule, grad accumulation, mixed precision on CUDA, checkpoints every N steps, exact resume), `export_encoder` |
| `data.py` | `MarketPanel` (dates × symbols × channels + observed masks, prices, universe, eligibility), normaliser, `DateBatchSampler`, FMP client (`FMP_API_KEY` or `~/.config/fmp/api_key`), synthetic panel |
| `features.py` | channel definitions and families for the 14-channel and 26-channel layouts, eligibility channels, channel sources |
| `walkforward_data.py` | `WalkForwardSampler` (context window + future windows ending at t+h, all inside the training cutoff), aux targets (per-date normal scores of the SPY-relative return or of forward realised volatility), raw winsorised returns |
| `panel_assembly.py` | `assemble_panel` (base panel + event / macro / fundamentals channel files → 26-channel panel with carried eligibility and a new fingerprint), `check_panel_coverage` |
| `walkforward_readout.py` | per-year feature extraction (target encoder, context encoder, untrained control, aux head), `walk_forward_ridge` (alpha by inner-year IC, purge), `daily_ic`, window means, paired moving-block bootstrap |
| `walkforward_job.py` | the walk-forward job: for each test year, train (warm start from the previous year) → features → ridge readout; summary per run; resumable state file; CLI |
| `evaluation.py` | SPY-relative forward labels, rank IC, `RidgeReadout`, purged training dates, `date_block_interval`, context eligibility |
| `expanded_data.py` | iShares holdings history (`FundSpec`: IWB, IWM), CSV parsing, symbol maps, monthly universe |
| `expanded_eval.py` | cohorts, peer groups, cached encoder features (memmap), cached raw features, normalised views |
| `expanded_job.py` | the earlier fixed-window cloud job (pretrain once, evaluate exports at step milestones) |
| `cloud_data.py`, `cloud_job.py` | panel construction from FMP (adjusted + raw prices, anchor validation, coverage report); deadline-aware job helpers |
| `runner.py`, `preflight.py`, `wandb_logging.py` | run bookkeeping and source fingerprint; Colab runtime preflight; offline W&B logging of every scalar loss term |

Tests: `python -m pytest tests -q` from the repo root. Colab bundles are built from `src/` and `tests/` by
`scripts/colab/build_source_bundle.py`; `requirements.txt` here is the Colab runtime set.
