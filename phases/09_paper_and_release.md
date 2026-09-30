# Phase 09 — Paper and public research release

**Entry:** 01–08 verified, with complete evidence or explicitly reported scientific
limitations. No full paper is created during the initial repository preparation.

Repeat a primary-source novelty search on the *final* equations, composition,
prior optimizers, validation guidance, constrained projections, and layerwise LRs.
Read the closest papers/code, include overlap and negative results, and update the
literature/claim ledger. Formal identities based on standard projection mathematics
are supporting assurance, not a claim to invent projection or prove superiority.

Create `paper/main.tex`, `paper/references.bib`, tables/figures, and a reproducible
build. Write the full paper around the actual evidence: question, prior work, data
information boundary, final algorithm/equations, exact theorem assumptions, hardware,
matched tuning/data/compute, all seeds, ablations, both pretraining regimes, shifted
evaluation, failed branches, uncertainty, contamination limits, and total cost.
Include the Lean source and theorem map. Separate formal safety from empirical
generalization. If superiority is inconclusive or fails, write the result honestly;
never manufacture a positive paper to complete a checklist. Continuing redesign
requires upstream invalidation and new confirmation evidence.

Clean and organize the repository: remove generated dependency caches from tracking,
organize source/tests/configs/proofs/research/paper, keep raw artifacts linked by hash,
retire misleading examples, and preserve user-supplied `prompt.md` and `skills/`.
Add comments explaining nonobvious equations, reduction ownership, masks, and timing;
avoid restating every Python line. Apply PEP 8 formatting/lint and meaningful type,
correctness, resume, sharding, logit, data-boundary, and kernel tests. Rebuild Lean
without new axioms/placeholders and regenerate tables/figures from raw data.

Rewrite README and paper in direct human prose: lead with the observed result,
explain how to reproduce each run and analyze it, show the complete result table,
state experimental scope and limitations, and distinguish prototype from validated
TPU capabilities. Check equations/notation consistently across code/theory/paper.
Document dataset/model licenses and provenance; select the repository's release
license with the owner if it has not already been specified. Do not grant rights
to third-party data/checkpoints that their licenses do not permit.

From a clean clone, install pinned environments, run tests, build proofs/paper, and
reproduce a smoke training/evaluation plus all analysis artifacts. Audit required
125M/2.5B/42,43,44 jobs, continued jobs, information controls, failures, and costs.
No stale gate or placeholder table can pass release.

**Outputs:** paper PDF/sources, clear README/reproduction commands, environment and
artifact manifests, formatted/commented code, reproducibility/claim audit, changelog,
release tag, and `research/gates/09.json`. Commit/push the reviewed release when that
publishing action is within the user's execution request. Do not autonomously submit
a paper or contact others. If the audit fails, repair the responsible dependency,
rewrite affected phases and paper, and rerun its evidence before release signoff.
