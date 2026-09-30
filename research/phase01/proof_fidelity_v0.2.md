# Phase 01 proof fidelity — version 0.2

The exact controller preserves the supplied training gradient's first-order
progress and improves a fresh guide linear model; those statements pass the Lean
and independent coefficient audits. Actual descent, current-guide benefit between
probes, independent generalization and TPU efficiency remain conditional or
empirical. The implementation needed a numerical revision before this gate.

The equation/theorem/dependency table is [claims.md](claims.md). Moments update from
the supplied training gradient only, with t+1 bias correction in both implementations.
Training and guide block coefficients use that same adaptive direction at the
pre-update parameters. C is normalized once per fresh scheduled probe; a is
renormalized and C reprojected every update. Count is completed updates; a probe at
zero-indexed t stores last_probe=t, then age=t−last_probe and rho_t=rho*max(0,1−age/K).
Decay is fixed and never scaled by guidance. Radius zero uses the identical AdamW
moment/decay path. The reference rejects missed/extra probes; JAX exposes
`schedule_ok` and the caller must abort. A compiled CPU check is not TPU profiling.

Accumulation and clipping are caller operations. Accumulate microbatch gradients,
then apply the same global clip in all arms, then compute moments and coefficients.
The protected gradient is that supplied clipped gradient. A common positive clip
rescales the training coefficient vector for a fixed u, so raw/clipped linear
neutrality transfers, but clipping also changes u through moments. Leaf/group
layouts must be disjoint and nonempty. Canonical tied weights occur once; repeated
NumPy memory and identical JAX leaf objects are rejected at initialization. Equal
but distinct arrays are valid coordinates; alias canonicalization and exclusion of
zero-coordinate groups remain future trainer obligations. Guide gradients can use
bfloat16 interfaces, but parameters/moments and coefficient reductions are FP32.

The preregistered initial audit used 7,168 actual JIT cases across groups 1/2/14/64,
scales 1e−30 through 1e30, exact/near/random collinearity and FP32/bfloat16 input
interfaces. It compared quantized inputs to float64 and an independent SVD
nullspace oracle. Tolerances were frozen before draws: relative neutrality≤1e−5,
weight-radius rounding≤2e−7, finite weights and nonnegative gain up to 1e−12
relative numerical allowance. Resolved parameter tolerance was 3e−6 absolute,
weight tolerance 2e−4. Fault injections (wrong sign, omitted projection, coordinate
clipping, excess radius, nonfinite weights, stale cached weights) were rejected.

The original full-radius normalization amplified collinearity error. In v1, the
bfloat16 resolved subset had 8.96% fallbacks and some weight discrepancies of 0.15.
The raw record and summary `math-v1.json` remain. We revised the positive scalar to
rho_t/max(norm_inf(q),0.01), not the projection denominator. This keeps neutrality,
fresh gain and radius bounds; new Lean proofs cover its exact domain. A second
projection removes numerical residue and is an exact-real identity.

An intermediate v2 retained 503 tiny-signal gain violations according to the
independent checker. Investigation showed XLA could simplify the realized
correction `(1+delta)−1` across a rounding boundary. An optimization barrier before
subtracting one and a conservative guide summation-error budget repaired that
certificate. The v2 record, protocol and failed outcome remain; its raw check used
reconstructed coefficients and the final verifier uses the emitted coefficients.
Therefore v2 is diagnostic evidence, not final signoff. The floor constant was
chosen from coefficient precision before the final fresh v3 draw stream, not from
any optimizer loss or sealed evaluation.

The final v3 audit retained all 7,168 records. The separate verifier re-read actual
emitted a,C,w and passed every coefficient invariant. In the resolved-signal
subset (norm_inf(q)>0.01, scales 1e−6..1e15), FP32 and bfloat16 each had 382 cases
and zero fallback. Maximum weight discrepancies were 8.79e−8 and 8.43e−8;
parameter discrepancies were below 3.5e−10. Maximum relative neutrality over that
subset was below 3.4e−7. Near-collinear fallbacks remain explained by the sign
certificate and quantization; they suppress weak signal and are not asserted to
be rare in all distributions. There were 1,024 finite extreme-input cases with
FP32 moment overflow; every one now sets `numerics_ok=False`, requiring run abort.
The valid arithmetic envelope is explicitly restricted; safe weights cannot
rescue an invalid moment state. The raw C=0 and a=0 branches, one-group restriction,
moment alignment and fixed-decay cases are checked separately by rational and
regression tests.

A CLI reproduction with a relative raw path exposed a provenance-path error after
writing the raw file. Its traceback and identical evidence remain as
`math-final.txt` and `artifacts/phase01/math-final.jsonl.gz`; the CLI now resolves
paths. This was an observation/recording fault, not a scientific restart. Gate
signoff uses the verified v3 raw file. Every raw file is immutable and excluded
from Git; tracked hashes and exact commands allow regeneration locally.

Exact attacks retained in `check_math.exact_witnesses`:

| Attack | Witness / implication |
|---|---|
| Hand two blocks | a=(1,1), c=(2,0), delta=(3/20,−3/20); neutral, raw gain 3/10 |
| One group, a≠0 / collinear | Nullspace or projected q is zero; no fresh correction |
| a=0 / c=0 | q=C when a=0; zero correction when C=0 |
| Adverse momentum | g=(1,1), u=(−1,−1), baseline training progress −2; neutrality preserves ascent |
| Opposite population gradient | Sample neutrality and fresh guide gain can oppose evaluation gradient −c |
| Large curvature | F=x1+x2+50*norm(x)^2, theta=0, eta=1; training rises to 401/4 despite first-order progress 2 |
| Decay cancellation | u=(1,1), lambda*theta=(−1,−1); full baseline vanishes but guided correction does not |
| Conflicting / stale guidance | Current c=−stored c turns fresh positive 3/10 into −3/10; P14 drift bound is tight |
| q→0 in old normalization | Any epsilon*(1,−1) got full radius; the new q-scale floor tends to zero |
| Skew zero-mean raw coefficient noise | s=1/2 with probability 2/3 and −1 with probability 1/3; E[s]=0, expected normalized correction=1/20 in first coordinate |
| Unconditional generalization | P10 train loss 1→0 while independent loss 1→4 |

The floor does not cure the normalization of raw c at c=0 or make adaptive
coefficient noise unbiased. Noise/variance/drift hypotheses and their unresolved
assumptions are listed in the claim table and measured in Phase 02. No proof says
more than the mechanism it is used to support. The 0.1 equations/config plans are
archived; phases 02–09 now depend on 0.2 and their old plans are invalidated. The
125M/2.5B/42,43,44 contract, loss threshold and resource ceilings are unchanged.
