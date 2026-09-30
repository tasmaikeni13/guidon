# Recording fault and driver equivalence

The first relative-path drift-discovery invocation completed all 18,000 raw draws,
then the provenance writer called `relative_to(ROOT)` on a relative path. That is
an infrastructure recording fault, the same kind identified in the math-audit CLI.
The raw trajectory data and traceback are preserved. No healthy trajectory was
restarted, no seed was changed and no loss was used for selection. Its missing
provenance is recovered from the registered order, per-draw seeds, complete raw
file and the exact source snapshot `0902877`; the independent verifier checks all
identities, counts and scalar-reference replays before this record can be used.

The driver now resolves the three CLI paths on entry and records its code commit.
This changes filesystem binding/provenance only. The loop, RNG construction,
condition generation, scientific stopping rule, simulator, optimizer, losses and
raw serialization remain byte-identical. `synthetic.py`, `reference.py`, the
registered primary protocol and analysis specification retain their original
hashes. Older immutable provenance hashes are checked against the retained
`0902877` source when the current driver differs; their algorithm/source snapshots
are not rewritten. Current scalar replay and invariant checks still apply.

This administrative repair does not change a load-bearing equation or scientific
protocol. It needs no Phase 01 invalidation or new scientific draw stream. The
followup itself already uses fresh separately registered discovery/confirmation
streams. Keep the recording-fault log and its raw file in the failure ledger.
