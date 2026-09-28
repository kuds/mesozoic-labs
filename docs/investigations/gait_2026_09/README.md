# Gait audit evidence (2026-09-28)

Evidence for [GAIT_AUDIT_2026_09.md](../GAIT_AUDIT_2026_09.md), frozen with it. `gait_probe.py` is the audit's hand-run
probe, imported by nothing; PR-G1 of the [gait plan](../../GAIT_QUALITY_PLAN_2026_09.md) ports it into the library.
`gait_audit_2026_09.csv` has one row per audited node and checkpoint: summary means, floor contact unless marked touch,
gate (whole-episode) speed, lowest-foot duty, highest-foot phantom support, quadruped pairs as fore/hind. `SHA256SUMS`
pins the audit's 153 JSON, trace, plot and contact-sheet files (64.7 MB, not in the repository). To regenerate one, run
the probe with `PYTHONPATH` at a `7ae0a19` checkout on the node's checkpoint pair and replay video (on Drive under
`mesozoic-labs/logs/`) with its JSON's `meta` (sha256s, seeds, episodes); means match, bytes do not. Helpers not kept
here made the recovery pushes, the July brachiosaurus (`e179198`), the statue baselines and the extra sheet views.
