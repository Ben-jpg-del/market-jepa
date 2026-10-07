# F-JEPA Colab research pilot

This package implements a separate future-block experiment. It does not modify
the original F-JEPA model or claim an improvement in stock prediction.

## Start in Colab

1. Upload `notebooks/F_JEPA_Future_Block_Colab.ipynb` using Colab's Upload notebook
   button. It embeds this source snapshot; no unpublished Git branch is needed.
2. Select a GPU under Runtime → Change runtime type. Start with the setup cell.
3. Run the dependency and source cells, then the included tests.
4. Add `FMP_API_KEY` through Colab Secrets and enable notebook access. Never put
   the value in notebook source. The access check reports endpoint status only.
5. Choose the checkpoint destination. Mounting Drive requires Google's consent;
   local `/content` files are temporary. Data arrays stay on local runtime disk.
6. Run the synthetic smoke test, then enable the small diagnostic download and
   real-data profiling cells. Enable longer runs only after inspecting the
   estimated duration and memory use.

The separate `F_JEPA_Colab_Setup_Check.ipynb` only checks hardware, dependencies,
and a small GPU forward/backward operation. It does not access credentials.

## Experiment

All variants use the new temporal/cross-stock encoder backbone. A is a repaired
historical-block task control, not a reproduction of the old checkpoint.

| Variant | Inputs | Target | Additional context masking |
|---|---|---|---|
| A | Basic | Historical block representation | Last two patches of selected stocks |
| B | Basic | Next five sessions | None |
| C | Book feature pack | Next five sessions | None |
| D | Book feature pack | Next five sessions | Occasional older contiguous patches |
| E | Book feature pack | Next session | None |

The default encoder has 60 context days, five-day patches, width 128, three
temporal layers, and one cross-stock layer. Its predictor has width 64 and two
layers. A complete EMA teacher supplies future targets. Targets expose only new
fast channels, with slow book channels unavailable. One-day targets are padded
with explicit missing masks. The objective combines prediction with SIGReg
without standardizing its random projections.

Book stock channels: Reactivity, Deviation, purified momentum, JANUS relative
momentum fractile, and its five-session change. Book market channels: turbulence,
Coherence and its change, Absorption Ratio and Shift, and JANUS RSS change.
Exact definitions and intentional departures are in `features.py:FEATURE_NOTES`.
In particular, stock JANUS is a reference-basket extension, not a verbatim port
of the original joint-stock-basket algorithm. Return-count and smoothing
conventions are recorded. No alpha-decay resistance has been established.

## Evaluation and limitations

Encoder training and preprocessing stop in 2018. Development uses 2019–2021.
2022–2025 is explicitly historical evaluation, already examined in earlier work.
Annual expanding ridge readouts use purged chronological boundaries. The
primary label enters at the next session's open and exits at the fifth session's
close, subtracting SPY. Trained, random, raw-feature, momentum, and reversal
controls share the evaluation protocol. Confidence intervals resample blocks
of dates, not individual stocks. Costs are illustrative and omit borrow and
market impact.

The initial 32-stock list is a software/representation diagnostic. Historical
constituent reconstruction and monthly prior-dollar-volume selection are
provided in the data module, but constituent completeness, symbol changes,
delisted prices/returns and price/volume adjustment conventions still require
validation. `point_in_time_verified` remains false. No missing future prices
are silently treated as zero returns or used to select the origin universe.

The notebook measures actual runtime performance before matched experiments.
Its budget is an estimate, not a hard spending or wall-clock limit. Downloads,
CPU feature computation, evaluation and Colab scheduling can add substantial
time. CUDA operation, FMP account access, and real financial results cannot be
validated by the local CPU tests.

Full training checkpoints include the optimizer, scaler, schedule position,
sampler and RNG state. Resume using these files and the same source/config/data.
Compact exports contain online/teacher encoder weights and preprocessing
metadata; they are for inference and cannot resume predictor training.

Deferred: macro/news inputs, TimesFM fusion, and post-pilot indicator-role
ablation retraining. They are not silently included in this first experiment.

## Direct Colab access

For the bounded cloud run, `cloud_data.py` prepares a cached historical-candidate
panel, backfills the provider's 5,000-row history cap, and records missing data
in `coverage.json`. The FMP key stays on the local preparation machine; only
the feature panel and source snapshot need to be uploaded to Colab.

`cloud_job.py --panel PATH --output DIR --hours 4 --device cuda` profiles all
five variants, freezes a matched update budget, saves C first, evaluates the
screens on development years, and continues the chosen future variant with a
second seed. Its process deadline saves resumable state; provider shutdown is
handled separately by the local run watcher. A resumed invocation receives a
new explicit time budget. T4 precision is FP16. The watcher requires the local
computer to remain awake and connected for retrieval and runtime release.

The verified local client pairing is `google-colab-cli==0.6.0` with
`jupyter-kernel-client==0.15.0`; the unrestricted latest dependency is incompatible
with this CLI release. The client authentication state is never part of the
training bundle or notebook.

Google provides [Colab MCP](https://github.com/googlecolab/colab-mcp) for a local
agent/browser notebook connection and [Colab CLI](https://github.com/googlecolab/google-colab-cli)
for runtime management, execution and file retrieval. Neither is configured by
this notebook. Browser sign-in alone does not establish either connection.

Codex can register Google's MCP command using the standard STDIO setup:

```sh
codex mcp add colab -- uvx git+https://github.com/googlecolab/colab-mcp
```

Restart the MCP connection, open the browser connection tool, and approve the
Google-side notebook connection. Google's server requires dynamic tool-list
notifications; end-to-end compatibility must be checked after connection.
See [Codex MCP configuration](https://developers.openai.com/codex/mcp/).
