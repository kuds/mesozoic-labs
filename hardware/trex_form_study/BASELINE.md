# Baseline and source reconciliation (inspected 2026-11-07)

This file records what exists, what matches, what was reproduced, and what remains
unavailable. It also gives the exact baseline the round-one study builds on.

## 1. Branches and pull requests (verified on GitHub and with `git fetch`)

| PR | State | Base → head | Head SHA |
|---|---|---|---|
| #635 | open | `main` ← `codex/mini-trex-forward-knee-example` | `398c02c13bdf82c907f1a75f34c30840ffba2d8e` |
| #648 | open, draft | #635 head ← `codex/mini-juvenile-hardware-fit` | `00589040f540a70660752a6c9830d99b4c4c1594` |
| #649 | open, draft | #648 head ← `codex/mini-trex-m1-skeleton` | `6e5fb9bc4418e653b6957ab382cc2d62e4263f8c` |
| #651 | open, draft | #649 head ← `codex/trex-60cm-skeleton-study` | `edd76a2828222174016712eaf6d38da8ea78b852` |
| #662 | open, draft | #651 head ← `codex/intermediate-rex-platform` | `e103f7ff2609fa1d7eb08507fb50ed9bb410b632` |
| #663 | open | `codex/intermediate-rex-platform` ← `codex/trex-r3-simulation-findings` | `b5c18bce1e576542a51532ffb442c238d83b0c9a` |
| #664 | open | `main` ← `codex/trex-body-shape-posture-study` | `b5c18bce1e576542a51532ffb442c238d83b0c9a` |

* **#663 and #664 point to the same commit.** #664 adds nothing beyond #663. Its title refers to a body-shape/posture study, but its description lists robot models, CI and documentation. Against `main` it shows 421 changed files (+272,945 / −74 lines, 36 commits), because the whole unmerged stack (#635 → #663) is under it. GitHub reports `mergeable_state: dirty` (conflicts with `main`).
* #663 adds only nine files under `hardware/intermediate_rex_r2/`: the findings, a one-page partner brief, two render PNGs and five JSON summaries. It adds no executable source.
* `main` is at `820dbdaffb2d4222d3540e8a6f47fbcbf6cdee6c`. It contains none of the stack's robot or intermediate embodiments, and it has moved ahead of the stack's merge base by 123 commits (recorded in #663's `baseline/source.json`).

## 2. The R2/R3 simulation handoff

| Reference | Status |
|---|---|
| Commit `f7d9857791e0b75d388118fd38fac2d2ce2c0794` (full R2/R3 handoff named by #663) | **Not retrievable.** `git fetch origin f7d98577…` fails with `remote error: upload-pack: not our ref`. It is on no branch of the repository. |
| Commit `eb0e984a6eebdcefc324268e03f9e6ba94c721f3` (#663's clean-checkout test) | **Not retrievable** (same error). |
| Short commit `85ae1705` (earlier `Intermediate-Rex-Handoff.zip`) | Does not resolve on the remote. |
| `examples/intermediate_rex_r2/…` (R3 model, gait bundle, follow trial) | Exists on no fetched branch. |
| `Intermediate-T-Rex-Simulation-Handoff.zip` | Referenced only as a private ChatGPT library link in the uploaded study; not accessible here. |

Without those sources, the following #663 results **cannot be reproduced or inspected**: the R2 and R3 models and their generators, the 75/90 cm R2 size variants, the trained R3 policies, observation normalization, gait-panel and follow-trial logs, videos, the 13/20 gait result and the person-following results. The committed JSON and PNG summaries are kept as historical evidence. This study does not reconstruct the missing models and does not present any approximation as R2 or R3.

What the #663 summaries do establish:
* R2 (60 cm): 10.282 kg. R3 (60 cm, 175 mm soles): 10.370 kg, 179.5 W, 18.4 min. The R2 75 cm and 90 cm variants: 11.13 and 12.13 kg, 13.8 and 9.8 min.
* The repository review shows the 75/90 cm R2 variants appear to keep the same RMD-X4 leg drives and the same 6S 3.3 Ah pack. This is inferred from the renders (same 61 mm motor cases at every size) and from runtime arithmetic (every runtime matches 54.945 Wh / P). If so, the earlier size comparison changed shells and links but not actuation or power. **Its conclusion that larger robots run shorter and need more lateral torque does not carry over to a resized design.**
* R3 needs about 9 N·m single-support hip roll against a 6 N·m continuous allowance. This study reproduces the same magnitude on the #662 nominal model with independent code (§4).

## 3. The uploaded posture study (round zero)

The project owner supplied `T_rex_body_study.zip` (SHA-256 `671b6b68…4fd95`). It is the only body-shape/posture study found. It covers nine 60 cm variants: three body families (A tapered, B deep ribcage, C open rib frame) at 0/10/20° chest-up, all built on the #662 nominal model and actuator schedule. It is preserved in `r0_posture_study_2026-11-07/` with full provenance and a reproduction record (`PROVENANCE.md`). In summary: inputs are byte-identical to #662, all 102 internal checksums pass, the nine models rebuild with ≤0.34 g shell differences, and the full pipeline reruns with identical pass/fail outcomes.

Its main findings, which motivated this round: every 60 cm variant still needs 9.5–9.9 N·m single-support hip roll (8.3–8.6 N·m in the best searched poses). The open-loop weight-transfer probe failed for all ten models. Covering shape barely changes these loads, because they are set by the leg geometry it kept.

## 4. Cross-check of the shared physics code against #662

`designlab.statics` was run on the #662 nominal MJCF (from the uploaded archive, byte-identical to #662). It reproduces:
* the double-support joint torques (hip 1.48, knee 3.30–3.33, ankle 1.17–1.19 N·m vs. 1.48 / 3.35 / 1.19 in the archive's table);
* single-support hip roll of 9.7–9.9 N·m at the joint limits (archive: 9.56–9.73).

The new tool therefore measures the same quantity as the earlier work.

## 5. Clean-checkout reproduction of the #662 platform

*Pending: filled from the reproduction audit (`REPRODUCTION.md`).*

## 6. Baseline used for new work

* **Branch:** `claude/mesozoic-trex-design-iteration-azm2fs`, created from `main` at `820dbdaffb2d4222d3540e8a6f47fbcbf6cdee6c`.
* **Why `main` and not the stack:** this round-one study is self-contained (`hardware/trex_form_study/`). It imports nothing from the unmerged stack and edits no shared file, so it does not conflict with #635–#664 and can be reviewed on its own. The #662 inputs it compares against are pinned by SHA (the copies inside `r0_posture_study_2026-11-07/source/pr662/` are byte-identical to #662 head `e103f7ff…`).
* **Training identity:** none was created. The round-one candidates are design models, not registered plants, so no training contract, digest or manifest changes. Registering a selected design as an embodiment (about 77 files of registry, configs and tests, per the repository review) is deferred until after design review.
* **Runtime:** Python 3.13, MuJoCo 3.10.0 (the repository's canonical version), numpy 2.5, scipy 1.18, Pillow 12.3. Rendering uses OSMesa.
