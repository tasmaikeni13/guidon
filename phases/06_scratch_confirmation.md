# Phase 06 — Full 125M scratch competition

**Method dependency:** GUIDON v0.2 uses $\tau=\rho_t/\max(\|q\|_\infty,0.01)$ and exact-idempotent numerical reprojection; the old v0.1 plans were invalidated by Phase 01. Require the current hash-verified prerequisite gates.

**Entry:** gate 05 valid and frozen. This phase launches the requested experiment;
it has not been run as part of repository preparation.

Execute the six primary jobs and the three required information controls:

| Arm | Seeds | Main training loss tokens per run |
|---|---|---|
| AdamW | 42, 43, 44 | 2,500,000,000 |
| GUIDON | 42, 43, 44 | 2,500,000,000 |
| AdamW plus identical guidance examples | 42, 43, 44 | 2,500,000,000 plus registered guide tokens |

Use the actual 125,226,240-parameter decoder from the frozen 125M configuration,
trained from random initialization on the pinned, audited FineWeb-Edu tape.
Pair initial parameters and training tape/order within each seed. Randomize/interleave
arm execution order to limit pod/load drift. A different optimizer-specific LR
is permitted only if selected within the equally budgeted frozen pilot protocol.

With the proposed batch 256×2048, there are 4,768 full updates and one final masked
update with **194,816** loss tokens: 4,769 updates exactly. Probe updates 128, 192,
… below 4,769 use 73 fresh batches of 32×2048, totaling **4,784,128 additional guide
tokens**. These are separately reported and also given to the information control.
Recompute from the final frozen cadence if it changes in pilots. AdamW's primary
arm retains exactly the requested 2.5B budget.

Record every consumed loss token and stream/probe cursor. Compile/warm graphs without
silently advancing training or discarding updates; record compilation time separately.
Save checkpoints at registered token fractions and the exact final token budget.
The final checkpoint is selected by token count, not by sealed loss. The training
process cannot read final evaluation. Run the frozen evaluator on sealed validation
only after all primary/control runs in this regime complete.

Record complete learning curves, train and development diagnostics, final sealed
NLL/perplexity, wall time, accelerator hours, peak HBM, measured/estimated FLOPs with
method/error, probe overhead, checkpoint/I/O/compile time, and crashes. A step-only
timing cannot satisfy the wall-time requirement. Publish each seed and paired
differences, mean effect, and the registered interval; include control comparisons
and total tuning cost. No best-seed reporting or silent seed replacement.

**Exit:** all nine jobs are accounted for, tokens/checkpoints/data match the frozen
protocol, and raw results support the registered loss and 1.10x resource targets.
Three seeds can leave uncertainty inconclusive; write that outcome directly. If a
resource fault occurs, diagnose and resume the same run. If superiority fails,
preserve results, identify the causal failure, invalidate the relevant upstream
gates, rewrite dependent phases, and return to adaptive research. Any inspected
confirmation set becomes development for redesign; obtain a new sealed set before
another confirmation. Write gate 06 with pass/fail/inconclusive evidence.

**Artifacts:** nine immutable run manifests/logs/checkpoint references,
`results/scratch/raw.jsonl`, reproducible analysis/figures, timing traces,
`research/gates/06.json`, and claim/evidence records. These paths are future outputs,
not fabricated existing results.
