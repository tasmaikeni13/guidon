# Phase 05 — Fair pilots and protocol freeze

**Method dependency:** GUIDON v0.2 uses $\tau=\rho_t/\max(\|q\|_\infty,0.01)$ and exact-idempotent numerical reprojection; the old v0.1 plans were invalidated by Phase 01. Require the current hash-verified prerequisite gates.

**Entry:** 02/03/04 valid. Pilot both scratch and continued regimes **before** opening
either final evaluation. Use development data and seeds 101/102; reserve 42/43/44
for confirmation. Do not substitute pilot results for the full requested runs.

Give each arm (AdamW, GUIDON, AdamW-plus-guidance) the same maximum 12 configurations
at 32M training tokens per configuration per pilot seed. Register trial generation,
all trial IDs, early failure rules, and total compute before execution. Use a
predefined space covering learning rate/schedule, beta2, weight decay, and clipping
for the strong baseline, plus radius and interval for GUIDON. Include the documented
defaults and the v0.2 signal floor; changing that floor after simulation invalidates 01/02 and their dependents. A successive-halving variant is permitted only if its resource allocation
and stopping rule are fixed and equally available. Report algorithm-specific search
dimensions and the entire search cost.

The information control computes the identical fresh guide batch gradient at each
scheduled probe and adds it to AdamW's ordinary gradient, weighted by the respective
loss-token counts. It receives 2.5B (or 1B) main tokens **plus** exactly the same guide
tokens as GUIDON. Give it independent, equal-budget hyperparameter tuning. Describe
its different gradient/moment use rather than calling it identical to train-only
AdamW. Freeze paired training/probe tapes and initialization for each seed.

Use discriminating reduced-scale ablations: radius zero; random/permuted and
sign-reversed guide coefficients; unconstrained weights with the same radius;
no stale reprojection; frozen blockwise scales; and interval/radius sweeps. Controls
are perturbations of GUIDON or uses of AdamW, not extra competitor optimizers.
Match extra probe compute when removing the signal. Check whether improvement is
explained by more examples, a layerwise schedule, or lucky tuning. Ablations that
violate neutrality are diagnostics, never the advertised method.

Measure learning curves, train/development/source-slice NLL, guide linear and actual
change, fallback and schedule failures, coefficient variance and active dimensions,
memory, probe cost, and complete throughput. Validate the proxy's ranking on at
least one intermediate token budget; if rankings are unreliable, use small pilots
for correctness and take selection uncertainty forward rather than overstating them.

**Exit:** stable training; explanatory controls support the intended mechanism;
no protocol leaks; viable 10% compute/speed budget; a strong tuned baseline; and a
plausible independent-loss benefit in both target regimes. If these fail, diagnose,
research, revise earlier mathematics/data/kernels, invalidate dependents, and rerun
fresh pilots. Do not lower the outcome threshold after observing it.

**Outputs:** all raw pilot runs, ablation figures, search ledger, tuning-cost table,
final scratch and continued configs, exact run matrices/order, evaluation tasks,
metrics/directions, effect/uncertainty analysis, contamination exclusions, failure
handling, and a signed/hash-addressed protocol bundle. Set `gates.frozen=true` only
when the bundle exists and all future choices are fixed. Write gate 05. No final
validation/test access before this frozen bundle.
