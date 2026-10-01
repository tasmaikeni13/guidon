# Phase 05 — Three-seed 125M hyperparameter sweep and scratch freeze

**Authorization:** the user requested this replacement on 2026-10-01. Rewriting
and registering this phase does not execute its training jobs. The separate Phase
06 confirmation remains 125,226,240 parameters, 2.5B main loss tokens per run,
and seeds 42, 43, 44.

**Method dependency:** GUIDON v0.2 retains the checked signal floor 0.01,
current-gradient reprojection, and the realized-weight certificate. **Entry:**
hash-verified gates 02/03/04. Validation/test remain sealed throughout the sweep.

## Exact sweep contract

Use the full registered GPT-2 decoder, **125,226,240 actual parameters**, and the
pinned FineWeb-Edu `sample-10BT` corpus/tokenizer from the scratch configuration.
The 600M pilot tape is a fixed deterministic subset of the same pinned FineWeb-Edu
token stream used to form the 2.5B scratch training tape; register the exact slice
and hashes before pilots. Do not run corpus-wide deduplication for this user-directed
data plan. Disclose that benchmark/source overlap was not removed by deduplication.
Every primary tuning run consumes **600,000,000 main training loss tokens in
exactly 2,500 optimizer updates**, across three pilot seeds **101, 102, 103**.
The independent confirmation seeds remain 42, 43, 44.

Each update has exactly **240,000 positive next-token labels**. Allocate a physical
128-by-2048 batch (262,144 positions), with 22,144 zero-loss padding positions.
Count labels from the actual masks; padding never contributes to gradients, NLL,
or consumed-token cursors. Packing resets attention at document boundaries and
does not borrow labels from another role or reuse targets. Each arm/seed receives
the same frozen pilot training tape and paired initialization.

The executable registration and arithmetic checks are:

```bash
uv run python -m guidon.sweep configs/sweep_125m_600m.json
uv run python -m guidon.sweep configs/sweep_125m_600m.json \
  --register artifacts/phase05/registered-v1
```

The second command publishes configs and the entire randomized run matrix; it
refuses to overwrite a registration. It does not launch training. After the
prerequisite gates pass and the user starts Phase 05, launch every registered row
through `guidon.train` using its pilot-only training capability.

## Fair search and the information control

Each arm gets **12 configurations × 3 seeds**: AdamW, GUIDON, and
AdamW-plus-guidance. This is **108 primary tuning jobs, 64.8B main loss tokens**,
plus all extra guidance and diagnostics. Register the wall-time envelope from
Phase 04 measurements before launching. Preserve every null, failed, interrupted,
and resumed job and its cost. No successive halving or score-based early stopping.

The pinned generation seed is 20261001, execution-order seed 20261002. Trial zero
retains the documented defaults. The remaining fixed balanced draws cover peak
learning rate 3e-4/6e-4/1.2e-3, beta2 0.95/0.99/0.999, weight decay 0.01/0.1,
global clipping norm 0.5/1/2, and final LR fraction 0.1/0.2. GUIDON additionally
varies radius 0.05/0.15/0.3. Both AdamW arms have independent final selections
within the same common search space and twelve-configuration budget.

The main search fixes probe interval 64, warmup 128 updates, and guide batch
32×2048 for all GUIDON/control trials. Updates 128, 192, …, 2496 yield **38 fresh
probes, 2,490,368 extra guide tokens per guided/control run**. Keeping this cadence
common preserves the identical-example information control after independent
hyperparameter selection. Pilot guidance is separate from confirmation guidance;
it may be shared across paired arms/trials/seeds but never recycled within a run.

At a probe, the information control combines the ordinary and identical guide
gradients **weighted by their actual positive loss-token counts**, then applies
the same global clipping rule and AdamW moments. GUIDON clips its ordinary training
gradient identically to train-only AdamW and uses the unclipped guide gradient for
its block coefficients. Record these distinct gradient/moment uses. Treat extra
examples and guide backward compute as optimization cost.

## Diagnostics and selection

Before jobs start, register separate mechanism diagnostics, their seeds/budgets,
and every run ID: radius zero; permuted/sign-reversed guide coefficients;
unconstrained weights; removed stale reprojection; and frozen block scales.
Register any interval 16/128 diagnostics with paired information controls and
matched probe compute. These are additional ablations, not an adaptive expansion
of the twelve-config primary search. Each primary search run still uses the exact
125M/600M/2,500-update contract. Report every additional diagnostic's full cost.

Use a separate development process on fixed development documents. Choose each
arm's hyperparameters by the arithmetic mean **final development NLL of all three
pilot seeds**; ties use the lowest trial ID. A config with an unresolved missing or
failed seed is ineligible, with the missing outcome and cost retained. Resume an
infrastructure interruption as the same run from consistent model/moment/RNG/token
and probe cursors. Never restart a healthy seed for an unfavorable score.

Log train/development/source-slice curves at registered token fractions; actual and
linear guide change, group coefficients, fallback/schedule faults, peak memory,
communication, guide compute, I/O/checkpoints, and complete wall time. Compare the
ranking at the 300M checkpoint with 600M before using shorter runs as a future
selection proxy. A checkpoint or document is not an independent training seed.
Sweep losses are development evidence and cannot establish independent superiority.

**Exit:** all registered tuning jobs and diagnostics are accounted for, training
is stable, baseline tuning is strong, the information/mechanism controls are
interpretable, and the measured 1.10× compute/wall-time envelope is viable. On
failure diagnose and repair the relevant earlier phase; retain results, propagate
invalidation, and preregister the replacement before running it. Never lower the
0.005-nat improvement or resource thresholds after seeing confirmation evidence.

**Outputs:** immutable search ledger and raw per-seed records, exact 108-row run
matrix, full tuning/failed-run cost table, development-only selection record,
ablations and learning curves, selected scratch configs for all three arms, and
a hash-addressed scratch confirmation bundle covering data, tasks, checkpoints,
failure policy and analysis. Set scratch `gates.frozen=true` only when this bundle
exists and gate 05 is directly verified. No sealed validation/test access here.

Continued-pretraining tuning is no longer an executed deliverable of Phase 05.
Phase 07 must separately register equal-budget continued pilots, tune on its own
development cohort, and freeze its complete protocol before opening continued
validation/test. Scratch final scores must not influence that selection. The 1B
continued budget, information control, seeds and success thresholds remain intact.
