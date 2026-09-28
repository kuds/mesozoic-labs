# Gait audit evidence (2026-09-28)

Evidence for [GAIT_AUDIT_2026_09.md](../GAIT_AUDIT_2026_09.md), frozen with it. `gait_probe.py` is the audit's hand-run
probe, imported by nothing; PR-G1 of the [gait plan](../../GAIT_QUALITY_PLAN_2026_09.md) ports it into the library.
`gait_audit_2026_09.csv` has one row per audited node and checkpoint: summary means, floor contact unless marked touch,
gate (whole-episode) speed, lowest-foot duty, highest-foot phantom support, quadruped pairs as fore/hind. The touchdown-event columns (alternation_index, simultaneous_fraction,
lead_swaps_per_stride, stride_hz) average only the episodes with enough touchdowns (`summary.<key>.n` in each JSON),
which on stance and statue rows can be a few of 30 or 40; a blank cell means none. `SHA256SUMS`
pins the audit's 153 JSON, trace, plot and contact-sheet files (64.7 MB, not in the repository). To regenerate one, run
the probe with `PYTHONPATH` at a `7ae0a19` checkout on the node's checkpoint pair and replay video (on Drive under
`mesozoic-labs/logs/`) with `--seed-scheme auto`, the CSV's `episodes`, and `--seed-base` set to the node's recorded `evaluation_seed` (3044
for trex `20260925_033501`, the default 3042 for every other node); run a `_replayNNNN` file with `--seed-scheme panel
--seed-base NNNN --episodes 1`. Means match, bytes do not. Helpers not kept
here made the recovery pushes, the July brachiosaurus (`e179198`), the statue baselines and the extra sheet views.
Not pinned: probe runs cited in the note and plan (the dibothrosuchus stance and its statue on the 40-episode panel,
`--seed-scheme panel --episodes 40`; the compsognathus, robot and velociraptor zero-action stances, `--zero-action`;
compsognathus locomotion on seeds 5000–5099; the trex seed-44 recovery realizations), and the hop-flight fractions of
plan §4.3 A, which a wrapper not kept here added to the probe's episode rows using plan §3.1's definition.
