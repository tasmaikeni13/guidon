# Literature and novelty ledger

Search date: 2026-09-30. Evidence labels below distinguish source-reported methods,
local replication, inference, and open speculation. Papers were retrieved from
arXiv/PMLR/NeurIPS; official model, dataset, TPU, and agent documentation was also read.
No cited optimizer's published learning results have been replicated here.

## Search ledger

| Query family / traversal | Primary hits | What it tested |
|---|---|---|
| Validation-guided optimizer, online hypergradient learning-rate adaptation | Baydin; MetaLR | Outer feedback through step sizes already exists |
| Validation gradient, robust example reweighting | Ren et al. | Clean feedback through training weights already exists |
| Generalization estimation, gradient alignment, language-model domain weighting | DoGE → DoReMi references | Alignment-guided LLM data allocation already exists |
| Conflict-averse / projected multiobjective gradients | CAGrad → MGDA/PCGrad discussion | Constrained objective protection is established |
| Layerwise/blockwise LLM learning rates and recent variants | LLR, LANTON discovery | Layerwise scales themselves are not novel |
| Training-neutral validation scales; training-gradient orthogonal validation; AdamW block projection | Above families; 2026 feasible-set projection discovery | Search for equivalent constraint/composition, not just a new name |
| Exact design facets: current training progress, disjoint Adam directions, projected guide coefficient, stale reprojection | No read source with all facets | Weak negative search evidence, not uniqueness proof |
| Adaptive data analysis and holdout reuse | Dwork et al. | Small scalar access and repeated testing can still adapt to a holdout |
| `GUIDON` optimizer / optimization algorithm | No matching optimizer name located | Branding collision screen only |
| FineWeb-Edu, FineMath, Pythia, v4-32, JAX TPU; official AGENTS.md | Official artifacts below | Data/model/hardware feasibility and agent instructions |

Coverage limits: finite general web and primary-repository retrieval, not an exhaustive
patent/thesis or citation-index search. Some arXiv HTML endpoints failed; the PDF
versions of closest papers were read instead. OpenReview blocked direct access to
the 2026 feasible-set projection paper and its public API returned HTTP 403; this
remaining comparison must be resolved before publication. Search results/snippets
alone are discovery evidence for that item, not full method evidence.

## Closest-work matrix and compact evidence cards

| Primary work | Operation / feedback | Regime / cost | Relationship and precise difference |
|---|---|---|---|
| [AdamW, Loshchilov & Hutter](https://arxiv.org/abs/1711.05101) | Decouples weight decay from adaptive moments | General model training; two moment buffers | Baseline. GUIDON retains moment/decay math and adds group coefficients |
| [Hypergradient descent, Baydin et al.](https://arxiv.org/abs/1703.04782) | Gradient dot products adapt a learning rate online | First-order update; historical direction state | Component precedent; no separately stated training-neutral block constraint in the read formulation |
| [Learning to Reweight, Ren et al.](https://proceedings.mlr.press/v80/ren18a.html) | Clean validation meta-objective determines training-example weights | Noisy/imbalanced image classification; online approximation | Strong feedback precedent. Reweights examples, not disjoint AdamW coordinate updates; source includes a clean-data baseline control |
| [DoGE, Fan et al.](https://arxiv.org/abs/2310.15393) | Target/source gradient alignment learns domain sampling weights | Proxy stage then base LLM training; first-order approximation | Closest LLM generalization precedent. Changes data allocation, not a protected block-update plane |
| [MetaLR, Chen et al.](https://arxiv.org/abs/2206.01408) | Online meta-learning adjusts layerwise learning rates using feedback | Medical transfer; provisional update and online LR adaptation | Closest coefficient-feedback precedent. Broad validation-guided layerwise-LR novelty would be false; GUIDON adds a current-progress equality and sparse stale reprojection |
| [CAGrad, Liu et al.](https://arxiv.org/abs/2110.14048) | Worst local task improvement constrained around average gradient | Multiple task gradients and a small dual optimization | Protection/projection precedent. GUIDON's plane acts on coordinate-block coefficients with one training objective; it does not inherit CAGrad convergence |
| [LLR, heavy-tail guided layerwise LRs](https://arxiv.org/abs/2605.22297) | Spectral layer statistics set bounded LR schedules | LLM pretraining; periodic spectral measurements | Recent schedule precedent. No validation-gradient feedback in its inspected Algorithm 1 |
| [Regularizing Optimizer Updates via Feasible-Set Projection](https://openreview.net/pdf?id=ZNNXOKeTPj) | Full methods inaccessible in this session | 2026 workshop item discovered | Novelty unresolved for this comparison; do not infer its full rule from a norm-control figure |

Source-reported evidence: AdamW gives the decoupled baseline; Baydin's equations
use a derivative through a prior step; Ren's method optimizes example weights on
clean feedback; DoGE's method section derives a first-order domain-weight update;
MetaLR's Algorithm 1 updates layer LRs online; CAGrad constrains a task-combination
direction; LLR's Algorithm 1 periodically sets scales from spectral statistics.
These methods operate on different data, models, and budgets; their headline metrics
cannot rank GUIDON or establish its TPU behavior.

Read scope: the closest papers' method definitions, operational update equations,
algorithm sections, assumptions, and pertinent evaluation/control descriptions.
Artifacts exist for these methods, but no independent reproduction of their empirical
claims is asserted. The locally independently checked result is **only** our shared
AdamW implementation's agreement with Optax, documented in verification.

Inference: a constrained combination of sparse layer feedback and current-training
neutrality may offer a different efficiency/transfer tradeoff. Open speculation:
that tradeoff beats a strong AdamW on LLM generalization. No source proves it.

## Frontier, contradiction, and opportunity tickets

Well-covered: gradient alignment, online metatuning, layerwise schedules, and
multiobjective projections. Unresolved in this design: whether sparse, low-dimensional
feedback has a measurable benefit after information/tuning matching and TPU reductions.
The apparent gap is a specific composition and operating constraint, not an empty field.

| Tension | Explanation to test | Decisive evidence |
|---|---|---|
| Improving guide loss versus independent generalization | Guide mismatch or adaptive overfit | New source-held evaluation and guide permutation |
| Preserved linear training progress versus increased actual loss | Curvature, noise, momentum, stale feedback | Manufactured Hessians and true loss changes |
| Tiny model-token overhead versus slow wall time | Optimizer memory traffic, collectives, recompilation | End-to-end pod profiler, not FLOP arithmetic alone |
| Published layerwise/validation methods versus a “new optimizer” | Renaming established components | Final-equation novelty search and facet comparison |

1. **Training-neutral block feedback (chosen, implemented).** Delta: project guide
   alignment coefficients off current blockwise training progress while retaining
   AdamW moments. Why test: reduces the feedback action space and protects a local
   training quantity. Cheapest test: analytic two-block case plus shuffled-guide and
   collinear controls. Kill: no transfer under equal-information AdamW or overhead
   exceeds the target. Novelty falsifier: prior work with the same coefficient plane,
   AdamW state, and sparse/current reprojection rule. Confidence in algebra high;
   performance unknown; novelty provisional.
2. **Sparse-feedback drift boundary.** Delta: identify when cached coefficients stop
   predicting guide/evaluation benefit despite current training neutrality. Cheapest
   test: controlled rotating quadratics with interval/drift/noise sweeps. Kill: no
   usable regime at the cadence needed for near-parity compute. Novelty falsifier:
   an equivalent existing delayed block-controller result. This may produce a useful
   negative result even if the chosen optimizer fails.

Verdict: apparently distinct from the **read** closest formulations, with the FSP
comparison and broader coverage unresolved. No absolute novelty claim. Phase 09
repeats the search on the final method and obtains the missing primary methods.

## Official artifact provenance

Pinned dataset/model revisions were obtained from Hugging Face's public repository
metadata, not by downloading training data: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath),
[Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m-deduped),
[GPT-2 tokenizer](https://huggingface.co/openai-community/gpt2).
Hardware/install references: [TPU v4 topology](https://docs.cloud.google.com/tpu/docs/v4),
[JAX installation](https://docs.jax.dev/en/latest/installation.html).
Evaluation tooling: [EleutherAI harness](https://github.com/EleutherAI/lm-evaluation-harness).
Holdout validity: [Dwork et al.](https://arxiv.org/abs/1506.02629).
Agent instructions use concise repo-specific commands and contextual pointers from
[official AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
and [current instruction-maintenance advice](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra).
