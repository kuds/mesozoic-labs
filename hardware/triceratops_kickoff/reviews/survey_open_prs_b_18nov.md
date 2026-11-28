## Review of open PRs #649, #651, #655, #662, #663, #664 (kuds/mesozoic-labs). Read on 18 Nov 2026; nothing posted, merged or closed.

**How I checked:** PR metadata and descriptions came from GitHub MCP `pull_request_read get`. I compared each branch with main using `git ls-tree`, `git merge-base`, `git diff --dirstat` and `git merge-tree`. Main is at 0748ac12, which is the merge of #667 on 14 Nov. I also read main's `hardware/trex_form_study/REPRODUCTION.md`, an audit of #662, #663 and #664 added in commit 40e00ce3. I did not read the full diffs, run any tests, or read review comments (#655 has one comment).

**How the PRs stack.** #635 (main) ← #648 ← #649 ← #651 ← #662 ← #663. #664 is the whole chain opened against main, at the same commit as #663 (b5c18bce). #655 stands alone. #635 and #648 are also still open but were not part of this review.

**On main or not.** None of the paths these PRs add are on main or on `claude/mesozoic-trex-stage2-refine-4jffre`:
- `hardware/mini_tyrannosaurus_rex/`
- `hardware/tyrannosaurus_rex_60cm/`
- `hardware/intermediate_rex_r2/`
- `docs/investigations/TREX_STANCE_POP_POSE_SATURATION_2026_10.md`
- `.github/workflows/metre-trex-concept.yml`

The form study lists #649, #651 and #662 by commit in its `BASELINE.md`, and its `REPRODUCTION.md` audits #662, #663 and #664. It does not take over their code.

### #649 Add mini T. rex M1 skeleton and hardware milestone workflow
- **Basics:** by kuds. Draft. `codex/mini-trex-m1-skeleton` → `codex/mini-juvenile-hardware-fit` (#648). Created and updated 28 Oct. 1 commit, 66 files, +3,796 / −29. GitHub shows it as mergeable ("clean") against its stacked base, not against main.
- **What it changes:**
  - A 10-actuator mini T. rex of about 754 g, registered as `tyrannosaurus_rex / robot`, with SB3 recipes.
  - The `hardware/mini_tyrannosaurus_rex/` folders for M1, M2 and M3, with rules for freezing each hardware revision.
  - Simulation evidence only: clearance and standing checks pass, and no learned-behaviour gate was evaluated.
- **Status:** not on main. Overtaken by the 75 cm track in #667 and the form study.
- **Recommendation:** close as superseded by #667 and the form-study PR (P48), and keep the branch. First, optionally lift the M1/M2/M3 milestone and revision-freeze layout from `hardware/mini_tyrannosaurus_rex/README.md` as a template for Triceratops hardware milestones.

### #651 Start a 60 cm T. rex skeleton and hardware study
- **Basics:** by kuds. Draft. `codex/trex-60cm-skeleton-study` → `codex/mini-trex-m1-skeleton` (#649). Created and updated 28 Oct. 1 commit, 19 files, +3,801. Mergeable against its stacked base.
- **What it changes:**
  - A 60 cm skeleton with 12 RMD-X4-P36 leg axes and a 9.21 kg mass allocation, including a Jetson-kit reservation.
  - Clearance, mass and standing/crouch checks in simulation. The model is not registered for training.
- **Status:** not on main. The owner parked the 60 cm size (P43), and the 75 cm track replaced it.
- **Recommendation:** close as superseded by the 75 cm track (#667 and the form-study PR), and keep the branch. Optionally first extract the check harness pattern for the Triceratops envelope work: the full-size motor envelopes, the mass ledger and the deliberate undersized-motor negative control in `hardware/tyrannosaurus_rex_60cm/m01_skeleton/README.md`.

### #655 Add a root-cause diagnosis of the T. rex stance pop, #489 and #491 (report only)
- **Basics:** by kuds. Ready for review (not a draft). `copilot/trex-stance-pop-pose-saturation` → main. Created 30 Oct, updated 31 Oct. 1 commit, 30 files, +6,577 / −1.
- **Conflicts:** GitHub shows it as having conflicts ("dirty"). My `git merge-tree` run found conflicts in 5 index files only:
  - `CHANGELOG.md`
  - `docs/README.md`
  - `docs/investigations/stance_2026_10/README.md`
  - `docs/investigations/stance_2026_10/SHA256SUMS`
  - `docs/investigations/stance_2026_10/SOURCES.txt`
- **What it changes:** docs only, with no code changes. It diagnoses the anatomical T. rex stance problems under reinforcement learning. Its main findings:
  - The reset keyframe is not a static equilibrium.
  - The head and tail "rails" are windup of the PD (position-control) targets, not counterweights.
  - PPO's initial exploration noise topples the standing pose.
  - The chatter is a 4–5-step limit cycle.
- **Status:** not on main and not superseded. I found no mention of it in the study's `BASELINE.md` or `LESSONS_LEARNED.md`.
- **Recommendation:** keep it. Resolve the 5 index conflicts, recheck SHA256SUMS, then merge as a dated record. If that isn't wanted, extract the note `docs/investigations/TREX_STANCE_POP_POSE_SATURATION_2026_10.md`. Its lessons on keyframe equilibrium, PD-target windup and exploration noise apply directly to training the Triceratops to stand.

### #662 Develop and evaluate an intermediate T. rex simulation platform
- **Basics:** by kuds. Draft. `codex/intermediate-rex-platform` → `codex/trex-60cm-skeleton-study` (#651). Created and updated 7 Nov. 3 commits, 104 files, +41,298 / −42. Mergeable against its stacked base.
- **What it changes:**
  - A 9.90 kg intermediate T. rex plant and a comparison of chassis layouts.
  - Gait trials: 2 of 20 met every criterion, and 18 missed the swing-clearance gate.
  - A camera, RGB-D and range person-follower that never cleared the pedestal.
- **Audit on main (`REPRODUCTION.md`):** the mechanics reproduce to 1e-11, but:
  - The committed engineering report is stale.
  - 11 contract tests fail on a clean checkout.
  - No policy bundle is committed, so no learned result can be reproduced.
- **Status:** not on main. The form study cites its mechanics as a baseline (`BASELINE.md` §4).
- **Recommendation:** keep parked as a draft. It is the reproducible mechanics baseline for when T. rex work resumes (end 2027 / early 2028). Do not merge. Before parking, extract the person-follow lessons (single selected onlooker, stop when identity is ambiguous, the lower camera mount and RGB-D tracking result of 443 of 450 frames), because they matter for a public Triceratops demo.

### #663 Compare intermediate T. rex sizes and report trained R3 walking and navigation results
- **Basics:** by kuds. Ready for review (not a draft). `codex/trex-r3-simulation-findings` → `codex/intermediate-rex-platform` (#662). Created and updated 7 Nov. 1 commit, 9 files, +2,711. Mergeable against its stacked base.
- **What it changes:** results files only, with no code. The R3 60 cm design is 10.37 kg with 18.4 minutes of walking and a nominal cost of $16,447 (the PR's own figure). Its trained actor passed 13 of 20 gait trials, against 2 of 20 with its learned correction disabled.
- **Audit on main:** nothing regenerates, and both commits it cites are missing from GitHub.
- **Status:** not on main. The form study holds the audit, and the 75 cm track replaced this work.
- **Recommendation:** close as superseded by #667 and the form-study PR. First extract `hardware/intermediate_rex_r2/FINDINGS.md` and the partner brief, either into a T. rex resume note or onto the archive data branch. Label their numbers as reported claims, since they are not reproducible.

### #664 Codex/trex body shape posture study
- **Basics:** by kuds. Ready for review (not a draft). `codex/trex-body-shape-posture-study` → main (820dbdaf). Created and updated 7 Nov. 36 commits, 421 files, +272,945 / −74.
- **Conflicts:** GitHub shows conflicts ("dirty"). The audit found 4 conflicting files: `configs/digest_snapshot.generated.txt` and three shared test files.
- **What it changes:** the title is misleading. It is the whole #635 → #663 stack against main: the 50, 60 and 100 cm examples, the mini and intermediate models and their registry entries, CI and pre-commit coverage, a 1 m T. rex concept workflow, and `CONTRIBUTING.md` changes. It is the same commit as #663.
- **Status:** not on main. It duplicates the stack PRs and is superseded by #667 and the form study.
- **Recommendation:** close as superseded: it duplicates #663 and the stack, and #667 plus the form-study PR replace it. Keep the branch for preservation. Do not merge: it is 273k lines, it conflicts, and it adds the 11 failing contract tests. The `CONTRIBUTING.md` section on shared code structure for robot concepts could be lifted on its own if the Triceratops follows the same layout.

### Summary table

| PR | Branch → base | Created | Size (files, +/−) | Draft | Mergeable | On main? | Recommendation |
|---|---|---|---|---|---|---|---|
| #649 | mini-trex-m1-skeleton → #648 | 28 Oct | 66, +3,796/−29 | yes | clean (on stack) | no | Close, superseded by #667 and form study; optionally lift M1/M2/M3 milestone template |
| #651 | trex-60cm-skeleton-study → #649 | 28 Oct | 19, +3,801 | yes | clean (on stack) | no | Close, superseded by 75 cm track (60 cm parked, P43); optionally lift check-harness pattern |
| #655 | copilot/trex-stance-pop… → main | 30 Oct | 30, +6,577/−1 | no | dirty (5 index files) | no | Keep: resolve conflicts and merge, or extract the note; lessons apply to Triceratops stance |
| #662 | intermediate-rex-platform → #651 | 7 Nov | 104, +41,298/−42 | yes | clean (on stack) | no (cited) | Keep parked for T. rex resume; extract person-follow lessons |
| #663 | trex-r3-simulation-findings → #662 | 7 Nov | 9, +2,711 | no | clean (on stack) | no (audited) | Close, superseded by #667 and form study; extract FINDINGS.md and partner brief first |
| #664 | trex-body-shape-posture-study → main | 7 Nov | 421, +272,945/−74 | no | dirty (4 files) | no | Close, duplicate of #663 and stack, superseded by #667 and form study; keep branch |

**Open points**
- None of the branches above should be deleted when the PRs close. The form study pins #649, #651 and #662 by commit in `BASELINE.md`.
- #635 and #648, the bottom of the same stack, are still open. They were outside this review and need the same decision.
- Not done: reading the full diffs, reading #655's comment, running anything, and checking any fork.