# Phase 07 — Continued pretraining and retention

**Method dependency:** GUIDON v0.2 uses $\tau=\rho_t/\max(\|q\|_\infty,0.01)$ and exact-idempotent numerical reprojection; the old v0.1 plans were invalidated by Phase 01. Require the current hash-verified prerequisite gates.

**Entry:** valid 05/06, continued protocol frozen before scratch confirmation.
Use **EleutherAI/pythia-160m-deduped**, pinned to the repository revision in the
configuration, and the higher-quality **FineMath-4+** subset. These choices test
specialization and retention at a model size feasible on v4-32, not an expectation
that selecting a dataset guarantees a win. Primary sources:
[Pythia](https://huggingface.co/EleutherAI/pythia-160m-deduped),
[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath).

Import the identical final base checkpoint for all arms, reset Adam moments and
guide state, retain its GPT-NeoX architecture/tokenizer, and verify source/JAX logits.
Seeds 42/43/44 control paired data/probe ordering and any registered stochasticity;
they do not create different base checkpoints. Report that initialization is shared.

Run AdamW and GUIDON for **1,000,000,000** main loss tokens each per seed, with a
90% FineMath / 10% independently reserved FineWeb-Edu retention mixture. Match the
guidance mixture and give the same examples to the separately tuned AdamW information
control: six primary and three control jobs. Use fresh guidance within each trajectory
and global split roles from Phase 03; previously sealed scratch sources cannot become
retention training. Report all added guide exposure and budget separately.

At the proposed batch size: 1,907 full updates, a final 182,784-token masked update,
1,908 total updates, 28 guide probes, and 1,835,008 extra guide tokens per guided/control
run. Verify these counts from the final frozen config. Select the exact final token
checkpoint. Preserve all RNG, data, and controller state across resumes.

Measure sealed math NLL and perplexity, independent general-language retention NLL,
registered math reasoning tasks, and unchanged-base scores. Separate improvement
from more math tokens, high-quality data alone, and loss of original capability.
Test the mechanism's retention hypothesis using group-scale/gradient diagnostics
and the ablations registered in Phase 05; do not retrofit their selection to results.

Audit benchmark overlap in the base and incremental corpora. A shared pretrained
checkpoint permits an incremental optimizer comparison, but cannot certify that
public benchmarks were unseen in base training. Contaminated/uncertain tasks are
reported separately from clean source-held evaluation and cannot prove leakage-free
real-world generalization.

**Exit:** all nine jobs complete or have preserved fault accounting; the registered
math improvement, retention ceiling (0.01 nats/token regression), and 1.10x resource
limits hold with honest uncertainty. Failure triggers diagnosis/research, updated
theory and dependent phases, and a new protocol with fresh sealed evaluation after
any redesign. Never restart a healthy bad-performing seed. **Outputs:** checkpoint
hashes/source mapping, raw per-seed results/learning curves, base-versus-adapted and
retention comparisons, contamination report, timing/memory records, and gate 07.
