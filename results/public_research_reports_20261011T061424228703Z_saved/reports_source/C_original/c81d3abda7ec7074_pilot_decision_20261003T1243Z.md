# V7 validation pilot decision — 2026-10-03 20:43 Beijing

All entries: seed128,100epochs, official229 dev only. Metrics and labels independently verified.

| Pilot | Dev batch MSE | Dev MAE | Train maximum common rank | Dev maximum common rank |
|---|---:|---:|---:|---:|
| coupled_seed128_4060Ti | 0.663423 | 0.590352 | 0.00488281 | 0.00000000 |
| instant_seed128_P4 | 0.683362 | 0.581323 | 0.00039063 | 0.00000000 |
| memory_seed128_P4 | 0.703611 | 0.592691 | 0.00312500 | 0.00000000 |
| memory_seed128_4060Ti | 0.743371 | 0.610884 | 0.00000000 | 0.00000000 |

## Action

4060Ti memory has zero measured common rank in every train/dev epoch: guarded/unguarded are explicitly skipped on this hardware. Same-card memory MAE is0.020532 higher than coupled; one seed cannot establish a general effect.

P4 memory has sparse training support but zero dev support. Guarded/unguarded rho0.5 seed128100epoch pilots are queued behind independently verified P4 coupled completion. Queue status is waiting; this is not executed compression training.

## Relation hypothesis revision

Local motion overlap is not established semantic overlap and is not reliably present across trained trajectories. Preserve the current threshold. More probes alone cannot solve the V8 identifiability counterexample. A future estimator must use source-availability interventions through the full trained flow in a new version, then demonstrate task relevance on validation.

All four completed pilots used rho0 and zero compression control. None of their gains can be attributed to compression. Do not freeze a superiority claim before hardware control and mechanism checks.

## Storage

Protocol/history/selection/results/229 predictions/logs are local and audited. Complete weights remain on rented servers; final copy SHA proofs are tracked separately. No permanent full-backup claim.
