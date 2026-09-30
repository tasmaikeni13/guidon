# Measurement and uncertainty budget — GUIDON 0.2

| Component | Experimental unit / evidence | Quantification | Remaining systematic uncertainty |
|---|---|---|---|
| Monte Carlo precision | One independent synthetic trajectory, paired arms; N=1000 per condition/stream | Sample SD/sqrt(N), pointwise 1.96-SE intervals; all rows retained | Generator misspecification does not shrink with N |
| Numerical implementation | Realized coefficients/weights; JIT FP32 and quantized bf16 versus float64/SVD | Phase 01 tolerance/certificate, overflow flags, mutation rejection; scalar reference replay in Phase 02 | Formal real algebra is not a floating-point theorem; future trainer reductions differ in scale |
| Guidance sampling | Fresh gradients plus a finite-source latent offset | Noise/covariance/batch/source-size factors; first-probe a,c error variance and normalized-correction bias | Hierarchical Gaussian source model does not establish web-data independence |
| Drift / moment dependence | Raw coefficient changes and rotating objectives | Time variance, age curves, lag and delay; fresh/stale group interaction | No learned upper bound on real Transformer coefficient drift |
| Independent evaluation sampling | Fresh spurious-feature held samples, separate RNG | Per-draw evaluation variation included in MC SE; bitwise training noninterference checked | 256 synthetic evaluation examples are a chosen apparatus, not LLM documents |
| Future training variability | Three paired LLM seeds 42/43/44 | Paired t interval with df=2; 240k design simulations, analytic normal coverage/power checks | Normality unresolved with n=3, especially shared continued initialization |
| Future document/source uncertainty | Conditional on each trained model | Paired group/document bootstrap secondary, nested within seed | Cannot create more trained-model replication or justify token pseudoreplication |
| Tuning and selection | Future equally budgeted Phase 05 pilots | Freeze configurations/task weights/checkpoint rule before final scores; retain all trials | Rankings at reduced token budgets can mislead; no tuning evidence yet |
| Multiple tasks / adaptive studies | Registered primary outcomes and secondary family | Frozen analysis specification, per-task effects, conditional Holm policy; exploratory screens labeled | Followups have their own fresh streams; finite prior searches are not exhaustive novelty |
| Cost / external validity | CPU NumPy/JAX apparatus only | Provenance reports elapsed time where retained, all failed checker/recording logs kept | No TPU wall-time/FLOP or LLM generalization evidence |

For the 0.005-nat effect, Gaussian paired SD 0.005 yields only 18.12% two-sided
power and a mean 0.02203-nat interval width at three pairs. Document counts may
reduce conditional evaluation error but not the independent seed component.
Heavy-tail/skew sensitivity shows that a nominal t interval is not automatically
95% covered. Report means, every seed difference and assumptions even when the
result is inconclusive; preserve the registered three-seed contract.

The discovery screen, confirmation screen, registered drift followup, all-arm
mismatch witness and secondary same-identity diagnostics have distinct labels.
No favorable synthetic condition is treated as confirmation of the LLM hypothesis.
The guide-stream offset is a stylized finite-source calibration effect, not an
independent evaluation set or a hidden extra tuning signal.
