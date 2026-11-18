# Consolidation plan: from the paused T. rex to the Triceratops build

18 Nov 2026. For the owner first (the short list below), then for whoever does the work (sections 1 to 8). Nothing
here is adopted: every recommendation waits for the owner's answer.

It rests on four read-only surveys of the repository made on 18 Nov 2026 (PR #669; the open pull requests, in two
parts; the branches, local state and forks), on the checked parts list [PARTS_PREMIUM.md](PARTS_PREMIUM.md) in this
folder, and on the PLEM fit analysis [PLEM_FIT.md](PLEM_FIT.md), also in this folder. The surveys and checks are in
[reviews/](reviews/). Where the repository changed after the surveys, the current state is given and marked
*re-read 18 Nov*.

**Links.** Each repository link names its branch in its address and, where it helps, in the text beside it.
"The working branch" is the T. rex study's branch, `claude/mesozoic-trex-stage2-refine-4jffre`, at `ceadab71` when
this plan was written and at `a0d0041a` after it (*re-read 18 Nov*: that commit adds the review records in
`reviews/`; the surveys saw `c5274905`). This folder was committed on the local branch `wt/trike-kickoff`, branched
from `b1d5e98e`; the working branch was then merged into it, so it also holds `PLEM_FIT.md`, `reviews/` and the T. rex
`RESUME.md`. Links to the T. rex files and to `PLEM_FIT.md` name the working branch, where they are on GitHub today;
links within this folder are relative.

## In short, for the owner

1. **The lead opens the T. rex study pull request (P48) from the working branch's current tip** (`ceadab71` or
   later), not from `c5274905` or `b1d5e98e`. You review it and merge it with a merge commit. It merges cleanly into
   main today.
2. **Review PR #669 (the Triceratops design study) and merge it as a concept study** once an errata section is in
   (section 2). The PR holds only the built PDF, with no source, so the Triceratops session drafts the errata (text
   for its README and a list for the PDF's maintainer) on its own branch or as review text; you apply them and merge.
   The session does not push to #669's branch or merge any pull request unless you say so in the session. #669 is not
   a build specification: the parts list replaces its drive, transmission and power choices.
3. **Make five decisions by about 15 Dec 2026** (section 7, round 1): mass, size, compute, bus voltage, and the
   chassis route. Recommended: 20 kg, 75 cm, a Jetson Orin Nano Super with a Teensy, a 48 V (12S) bus, and PR #669's
   chassis with the parts list's changes. Long-lead parts (foot load cells at 21-23 weeks) then go on order in early
   January.
   - **Cost against your spend wish.** The class is $17k-$32k: the T. rex's 75 cm build ($17.2k self-built, $20.0k
     turnkey) up to its 100 cm flagship ($29k-$32k). The recommended T20 is $31.8k nominal counted the T. rex way,
     the top of the class, and $37.3k landed, above it; T15 is $28.3k the T. rex way and $32.8k landed. The first
     spend is the T20 one-foreleg bench set, $7.1k of parts ($8.9k landed).
   - **A price flag.** The drives are priced at MyActuator's own US store. The PLEM fit's reviewer found third-party
     US drive prices about twice that (RobotShop, 18 Nov 2026), so a reseller purchase would raise every total.
4. **By about 15 Jan 2027 (round 2), say what the T. rex milestone at end 2027 / early 2028 shows**, whether to plan a
   PLEM bench pilot, and the pull-request triage. If the milestone is a built T. rex, its simulation work has to
   restart by 1 Mar 2027 and its drives be ordered in Q2 2027 (section 5.3).
5. **Close eight stale T. rex pull requests as superseded, keeping their branches for now** (#585, #623, #635, #648,
   #649, #651, #663, #664). Merge #646 after a refresh. Decide #655. Keep #596 and #662 parked (section 1).
6. **Delete nothing that holds the T. rex**: not the two data branches, not the working branch, not the walking-pilot
   branches. Only you delete branches: six merged branches can go now, a few more once their PRs are settled (section 3).
7. **Start the Triceratops session** with [NEW_SESSION_PROMPT.md](NEW_SESSION_PROMPT.md): simulation first, renders
   for the mid-December museum contact, then a one-foreleg bench plan.
8. **Send WIM and MyActuator the PLEM questions in section 6** (the session drafts them; [PLEM_FIT.md](PLEM_FIT.md),
   section 5.4) before anything is bought or built around PLEM. A PLEM bench pilot runs only if their answers clear
   the blockers (brakeless drives unsupported, AGX Orin only, at most 8 axes per bus); its go or no-go is yours by
   end of February 2027 (decision 7).

## 1. Open pull requests and recent branches

### 1.1 Pull requests

Thirteen pull requests are open (*re-read 18 Nov*, GitHub). The P48 study PR is not opened yet. "Behind" counts
commits on main since the branch's merge base.

| PR | What it is | State | Recommendation | Extract first |
|---|---|---|---|---|
| (P48, to open) | The T. rex study: working branch to main. 683 commits ahead of main at `ceadab71` (684 at `a0d0041a`); 1,539 changed paths under `hardware/trex_form_study/`, and under `hardware/triceratops_kickoff/` 2 at `ceadab71`, 8 at `a0d0041a` (`PLEM_FIT.md` and the review records), before this folder's own files are merged in (*re-read 18 Nov*) | `git merge-tree` against main: clean (*re-read 18 Nov*) | **Open and merge** (P48), with a merge commit | none |
| #669 | Triceratops design study, `codex/triceratops-quadruped-design` (84a13e1e), 2 commits, 8 new files, +6,517 | Mergeable, no conflicts, no reviews, no CI run (the CI path filter leaves out `hardware/**`) | **Merge as a concept study** once an errata section is in (section 2) | none |
| #668 | Jetson Orin + Partial PLEM study | Closed unmerged on 16 Nov | Nothing to do: its branch is inside the working branch's history and reaches main with the P48 PR | none |
| #664 | "T. rex body shape posture study": in fact the whole #635 to #663 stack against main, 421 files, +272,945 | Conflicts in 4 files; same commit as #663 | **Close, superseded** by #667 and the study PR; keep the branch | Optional: the `CONTRIBUTING.md` section on shared code layout for robot concepts |
| #663 | Intermediate T. rex sizes and trained R3 results (results files only) | Stacked on #662; main's `REPRODUCTION.md` found nothing regenerates | **Close, superseded** by #667 and the study PR; keep the branch | `hardware/intermediate_rex_r2/FINDINGS.md` and the partner brief, labelled as reported, not reproducible |
| #662 | Intermediate T. rex simulation platform (draft) | Stacked on #651; cited by main's `BASELINE.md` | **Keep parked** as the mechanics baseline for the T. rex restart | The person-follow lessons (one selected visitor, stop when identity is unclear, the lower camera, RGB-D tracking 443 of 450 frames): they matter for a museum demo |
| #655 | Root cause of the T. rex stance pop (report only) | Conflicts in 5 index files | **Keep**: resolve the conflicts, recheck `SHA256SUMS`, merge as a dated record; or keep only its note | Its lessons (reset keyframe not in equilibrium, PD-target windup, exploration noise toppling the stance) apply to training the Triceratops to stand |
| #651 | 60 cm T. rex skeleton (draft) | Stacked on #649 | **Close, superseded** (60 cm parked, P43); keep the branch, which #662 uses as its base | Optional: the check harness (full-size motor envelopes, mass ledger, an undersized-motor negative control) |
| #649 | Mini T. rex M1 and milestone workflow (draft) | Stacked on #648 | **Close, superseded**; keep the branch | Optional: the M1/M2/M3 milestone and revision-freeze layout, as a template for Triceratops hardware milestones |
| #648 | Mini T. rex torso layouts (draft) | Stacked on #635 | **Close with #635**; keep the branch | Optional: one paragraph on leg folding for the T. rex restart notes |
| #646 | MicroDuck-inspired gait and sim-to-real design (docs, 1 file + 1 link) | Mergeable | **Merge** after refreshing its decision references (D-D17, D-D33, D-D44) and cross-linking main's `REFERENCE_MICRODUCK_2026_10.md` and `SIM_TO_REAL_PLAN.md` | none: species-neutral, useful for the quadruped's sim-to-real path |
| #635 | T. rex concepts (mini, 50 cm, 100 cm) and medium hardware, 204 files, +166,099 | Checks not green (GitHub's state) | **Close, superseded** by the form study; keep the branch | `examples/tyrannosaurus_rex_50cm/long_body/HARDWARE.md` (sourced part sizes, the packing method) and the 100 cm `hardware.md` and `bom.csv` (a RobStride price point), as a short doc on main |
| #623 | T. rex foot "r10" (draft) | Conflicts; main gave the name r10 to #643 (D-D44) | **Close, superseded** by #643 and the pause | Optional, for the T. rex restart: its evidence (push margins, noise fall counts, the CoP-bar calibration) |
| #596 | Walk-first gait checker, #585's fixes and a T. rex gait reward (draft) | Conflicts; paths renamed on main since | **Keep parked.** When Triceratops locomotion training starts, port the checker without the T. rex `gait-r1` part, then close it | `docs/GAIT_CERTIFICATION.md`, the `quadruped_walk` profile and its tests, the evidence tables |
| #585 | Gait reports and opt-in certification | Conflicts; both commits are inside #596 unchanged | **Close, superseded** by #596 | none |

**Closing the stack.** #635, #648, #649, #651, #662 and #663 are stacked, each on the one before. Close from the top
down (#664, #663, then #651, #649, #648, #635) and keep every branch, so that #662 keeps its base branch and the
commits pinned in main's [BASELINE.md](https://github.com/kuds/mesozoic-labs/blob/main/hardware/trex_form_study/BASELINE.md)
(main) stay reachable by name.

### 1.2 Recent branches (commits since 4 Nov, plus the T. rex data branches)

| Branch (tip) | State | Recommendation |
|---|---|---|
| `claude/mesozoic-trex-stage2-refine-4jffre` (ceadab71) | The working branch; not merged; holds C1 `cc437b6a` and all of `jetson-orin-plem-study` | **Keep** (P48). Source of the study PR |
| `claude/mesozoic-trex-stage2-bundle-s2-b1` (ecf95a81) | T. rex launch inputs; in no other branch | **Keep frozen.** Never merge, move, rename or delete (P50) |
| `claude/mesozoic-trex-stage2-archive-a1` (f71cba4b) | The 119 frozen T. rex arrays; **pushed** (*re-read 18 Nov*; the survey saw it unpushed) | **Keep.** Never merge (P49) |
| `codex/triceratops-quadruped-design` (84a13e1e) | PR #669 | Merge (section 2); the owner may delete it afterwards |
| `jetson-orin-plem-study` (6162b455) | PR #668, closed; an ancestor of the working branch | The owner may delete it once the study PR is merged |
| `claude/mesozoic-trex-design-iteration-azm2fs`, `claude/velociraptor-form-factor`, `claude/stance-cert-records`, `claude/stance-ltsb-main`, `claude/velociraptor-trex-lessons`, `claude/gait-ledger-page` | Merged (#667, #666, #665, #661, #660, #659) | The owner may delete them; nothing is lost |
| `codex/trex-body-shape-posture-study`, `codex/trex-r3-simulation-findings` (both b5c18bce), `codex/intermediate-rex-platform` (e103f7ff) | PRs #664, #663, #662 | Keep (section 1.1) |
| `claude/pilot-walk-r10-p4` (0e6a57bb), `claude/pilot-walk-r10-p3` (31d02e42) | Walking-pilot code, "pilot-only, never merged"; the only copies | **Keep** as archives for the T. rex restart |
| `claude/handoff-walking-pilot4` (f90304fa), `claude/handoff-walking-pilot3` (90e515c8) | Text-only handoffs, no shared history with main. Pilot 4's binaries were left in a session scratchpad, not in git, and may already be gone | **Keep** |
| Local only: `wt/wp-l`, `wt/wpc-cb`, `wt/wpc-ck`, `wt/wpc-ce` | 77 unpushed commits; all four tips are in the committed bundle `worktrees-2026-11-18.bundle` on the working branch (heads checked, *re-read 18 Nov*) | Nothing to do |
| Local only: `wt/archive-a1-full` (be474190) | An earlier build of the archive branch, never pushed; superseded by f71cba4b (RESUME.md, step 0) | Nothing to do |
| Local only: `wt/trike-kickoff` (this folder) | Three files | **The lead merges it into the working branch and pushes**, before the study PR is opened, so the kickoff folder reaches main with it |

The one fork, `yunerious/mesozoic-labs`, was last pushed on 8 Nov 2026 (metadata only), so it holds none of the
14-18 Nov work. Nothing to do.

## 2. The order of merges into main

Main has not moved since `0748ac12`, the merge of #667 on 14 Nov. No CI runs on any of these, because the CI path
filter leaves out `hardware/**`; each PR's own checks are the evidence. Merge each with a **merge commit**, not a
squash or a rebase, so that every commit the T. rex records pin (C1 `cc437b6a` among them) stays in main's history.

1. **The T. rex study PR (P48).**
   - Before opening (the lead): merge `wt/trike-kickoff` into the working branch and push; open the PR from the tip
     at that moment; say in its body that it also carries `hardware/triceratops_kickoff/` (the PLEM fit analysis,
     this plan, the parts list and the new-session prompt).
   - Before merging (the owner): `git merge-tree` against main is clean (it was at `ceadab71`); the data branches
     still read `ecf95a81` and `f71cba4b` (`git ls-remote`); RESUME.md's step 5 checks printed their expected lines
     on 18 Nov in a checkout of the pause's code (the bit-identity pair and `regress check`, 256 of 256).
   - Known and not a blocker: 8 added files exceed the pre-commit hook's 500 KB limit (the largest is
     `stage2/runs/2026-11-15-75cm/bar_tables.json`, 2.7 MB). Main already holds 36 such files from #667, and CI does
     not run that hook. Say so in the PR body.
2. **PR #669, Triceratops design study.**
   - The PR holds only the built PDF; its README points at no source, so the PDF cannot be edited here. Its fixes go
     in as an errata section in its README plus a list for the PDF's maintainer: the PDF's p.6 "Orin / MCU" label;
     the battery at 0.755 kg and the bay at least 165 mm; the 1:1 belt and the 20 kg case not sized (parts list,
     sections 1 and 2.7); the T. rex appendix (pp.7-9) superseded by P47-P51.
   - Who does what: the Triceratops session drafts the errata on its own branch, or as review text for the owner
     ([NEW_SESSION_PROMPT.md](NEW_SESSION_PROMPT.md), task 1); the owner applies them to #669's branch and merges. The
     session does not push to #669's branch or merge any pull request unless the owner says so in the session.
   - Then: `git merge-tree` against the new main (no path overlap with the study, so expected clean); run
     `models/check_models.py` with `mujoco==3.10.0` and keep its output, because nothing else runs it (the session's
     task 1; read the PR's files after `git fetch origin pull/669/head:pr-669` with `git show pr-669:<path>`); check
     that its links pinned to commit `42e6f507` still resolve (they did on 18 Nov).
3. **#646, MicroDuck transfer design (docs).** Before: refresh its decision references and cross-links (section 1.1);
   let CI run if `docs/**` triggers it.
4. **#655, only if you want the record on main.** Before: resolve the conflicts in `CHANGELOG.md`, `docs/README.md`,
   `docs/investigations/stance_2026_10/README.md`, `SHA256SUMS` and `SOURCES.txt`; recompute `SHA256SUMS`.
5. **Later, two small extraction PRs:** #596's checker (when Triceratops locomotion training starts), and a short
   hardware note from #635 and #663. Then close #596.

Not to be merged: #585, #623, #635, #648, #649, #651, #662, #663, #664.

## 3. Branches to keep or delete

**The lead deletes no branch. Only the owner deletes branches**, and nothing is deleted before the study PR is merged.

- **Never delete:**
  - the two T. rex data branches, `claude/mesozoic-trex-stage2-bundle-s2-b1` (the Colab notebook fetches it by
    name; P50) and `claude/mesozoic-trex-stage2-archive-a1` (P49);
  - the working branch `claude/mesozoic-trex-stage2-refine-4jffre` (P48: it stays);
  - the walking-pilot branches and their two handoff branches (the only copies);
  - the PR branches of #596, #655 and #662, and the stack under #662 (`codex/trex-60cm-skeleton-study`,
    `codex/mini-trex-m1-skeleton`, `codex/mini-juvenile-hardware-fit`, `codex/mini-trex-forward-knee-example`).
- **The owner may delete, with no content lost:**
  - now: the six merged branches in section 1.2;
  - after the study PR merges: `jetson-orin-plem-study`;
  - after #669 merges: `codex/triceratops-quadruped-design`;
  - after #646 merges: `codex/microduck-transfer-design`;
  - after #585 closes: `codex/gait-certification` (both its commits are in #596).
- **Keep for now, decide later:** the branches of #623, #663 and #664 (save their evidence first if you ever delete
  them).

## 4. What carries over from the T. rex to the Triceratops

All T. rex paths below are on the working branch, under `hardware/trex_form_study/`; the list follows
[RESUME.md, section 4](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/RESUME.md)
(working branch). They reach main with the study PR. Copy what you reuse into the Triceratops folders; do not edit the
T. rex files.

| What | Where (working branch) | Use on the Triceratops |
|---|---|---|
| **The T. rex RESUME.md** | [RESUME.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/RESUME.md) | How to restart the T. rex (section 5.3), what is kept where, and the carry-over list |
| **Colab notebook pipeline with Drive checkpoints** | [stage2/stage2_training.ipynb](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/stage2_training.ipynb); [designlab/stage2/nb.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/designlab/stage2/nb.py); [designlab/stage2/train2.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/designlab/stage2/train2.py) (`--resume`); [designlab/stage2/vecpool.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/designlab/stage2/vecpool.py); [stage2/tb/colab_steps.template.txt](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/tb/colab_steps.template.txt); [stage2/tb/requirements-colab.txt](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/tb/requirements-colab.txt) | Training from a fixed commit, checkpoints copied to Drive, resume after a killed runtime, a runtime-hour tally. It needs a Triceratops environment module and its own bundle. It was tested on a stand-in Drive, never on Colab itself, so the Triceratops's first Colab run is also its first real test |
| **Bundle tools** | [designlab/stage2/bundle.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/designlab/stage2/bundle.py); [tools/restore_inputs.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/tools/restore_inputs.py); [stage2/tb/validation/drive_route.json](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/tb/validation/drive_route.json); [worktrees-2026-11-18.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.md) | Large git-ignored inputs packed into checked tar parts; the data-branch route (P46), because the Drive connector carried only about 3.1 MB per file; git bundles for unpushed work |
| **Servo-delay model and bench sheet** | [designlab/stage1/speedfit.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/designlab/stage1/speedfit.py) (`servo_delay`); [designlab/stage2/servo.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/designlab/stage2/servo.py) with [SERVO_MODEL_DESIGN.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/design/servo_model/SERVO_MODEL_DESIGN.md); [stage2/jetson75/BENCH_SHEET.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/jetson75/BENCH_SHEET.md); [MAKER_QUESTIONS.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/design/servo_model/MAKER_QUESTIONS.md) | Delay, friction, gear play and voltage sag in simulation for any legged body. The bench sheet becomes the delay section of the foreleg bench; the maker questions apply as they are, because the drive family is the same |
| **Latency budget** | [stage2/jetson75/LATENCY_AND_BUS.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/jetson75/LATENCY_AND_BUS.md), `latency_bus.py`, `latency_bus.json` | Re-run for four buses of three drives. The surveys' arithmetic: 2 classic buses x 6 drives load 67-81 % at 500 Hz and do not fit 1 kHz; 4 x 3 load 67-81 % at 1 kHz |
| **Price pages** | [stage2/jetson75/BUDGET_75.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/jetson75/BUDGET_75.md), `budget_75.py`; [stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md); the catalog [designlab/data/actuators.json](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/designlab/data/actuators.json) and `parts.json`; page tools [tools/report_page.py](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/tools/report_page.py), `tools/shot_page.js`, `designlab/page_kit.py` | Catalog prices with source and date (P31); the parts list already uses them. Owner pages from Markdown reports |
| **Pre-registration and decision log** | [DECISION_LOG.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/DECISION_LOG.md); [STAGE2_75_PLAN.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/STAGE2_75_PLAN.md); [BUY_GATE_DRAFT.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/BUY_GATE_DRAFT.md); [ROUND_C_RULINGS_17NOV.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/stage2/design/ROUND_C_RULINGS_17NOV.md); [LESSONS_LEARNED.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/LESSONS_LEARNED.md) sections 3 to 5; the independent checks in `stage2/design/checkpoint/reviews/` | Rules fixed before runs, two seeds or controllers for a verdict, an independent check of every build and report, each owner answer logged with the options put. The Triceratops should start its own decision log; the staged purchase gates of `BUY_GATE_DRAFT.md` fit the foreleg-first plan |
| **Container tools** | [tools/README.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/tools/README.md) (`keepalive.sh`, `memguard.sh`, `lowprio.sh`, `setup_venv.sh`) | Long jobs in a cloud container that may restart |

Also useful, already on **main**: [RAPTOR_SCALING_AND_ALTERNATIVES.md](https://github.com/kuds/mesozoic-labs/blob/main/docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md)
section 4 (a 12S, three-CAN-bus custom quadruped with the knee drive kept near the body, and its gates);
[SIM_TO_REAL_PLAN.md](https://github.com/kuds/mesozoic-labs/blob/main/docs/hardware/SIM_TO_REAL_PLAN.md);
[REFERENCE_MICRODUCK_2026_10.md](https://github.com/kuds/mesozoic-labs/blob/main/docs/hardware/REFERENCE_MICRODUCK_2026_10.md);
the floor-truth gait recorder [environments/shared/gait/recorder.py](https://github.com/kuds/mesozoic-labs/blob/main/environments/shared/gait/recorder.py);
and two quadruped species environments, [brachiosaurus_altithorax](https://github.com/kuds/mesozoic-labs/tree/main/environments/brachiosaurus_altithorax)
and [dibothrosuchus_elaphros](https://github.com/kuds/mesozoic-labs/tree/main/environments/dibothrosuchus_elaphros) (all main).

**What does not carry over:** the T. rex bodies, walking styles and frozen arrays; the biped-specific panels; the
anatomical-simulation T. rex work in #596 (`gait-r1`) and #623.

## 5. The museum timeline, the Triceratops milestones, and when the T. rex resumes

### 5.1 The fixed dates (the owner's)

- **Mid-December 2026:** first contact with museums, with simulation renders. The owner named "the Field Museum or
  the Natural History Museum" (P51 records "Field Museum, a natural history museum"). Griffin MSI (Chicago's Museum
  of Science and Industry) is a further candidate the owner did not name.
- **Q3 2027:** live Triceratops demos.
- **End 2027 / early 2028:** the T. rex flagship, shown on the roadmap from the first contact on (P51). The owner
  accepts that leading with a Triceratops loses the direct T. rex tie-in with SUE, the Field Museum's T. rex specimen
  (the owner wrote "Sioux connection"), and still wants to build the T. rex; the owner updates the pitch deck.

### 5.2 Triceratops milestones (a proposal; the dates are planning estimates)

| # | Milestone | Target | Latest | Slack | Depends on |
|---|---|---|---|---|---|
| T0 | Study PR and PR #669 merged; round 1's five decisions taken (section 7). Round 2 follows by about 15 Jan 2027 | 4 Dec 2026 | 15 Dec 2026 | 11 days | Owner's review |
| T1 | **Museum renders:** MuJoCo stills and a short clip of the 75 cm robot standing with its real mass distribution and, if ready, crawling with a scripted gait; the anatomical adult; one roadmap line for the T. rex flagship. Every image labelled "simulation" | 8 Dec 2026 | 14 Dec 2026 | 6 days | Fallback: stance renders from PR #669's two models, which already stand |
| T2 | **Phase-1 order:** the single-foreleg bench set (about $7.1k of parts, $8.9k landed; parts list) plus long-lead items the bench will not change: foot load cells (21-23 weeks), main fuse (16-23 weeks), tether supply (about 12 weeks) | 8 Jan 2027 | 31 Jan 2027 | 3 weeks | T0; the owner's purchase approval |
| T3 | **Simulation:** walk at 0.30 m/s with 0-10 ms servo delay and noise on grippy and slippery floors; actuator sizing confirmed in simulation; a short trot | 31 Jan 2027 | 28 Feb 2027 | 4 weeks | T0 |
| T4 | **Foreleg bench:** torque and speed at 12S, heat over the show loop, command-to-torque delay, the stop chain's sit-then-cut; the PLEM pilot decision (PLEM_FIT suggests end of February) | 15 Mar 2027 | 15 Apr 2027 | 4 weeks | Drive delivery |
| T5 | **Four-leg gate and full-build order:** the remaining drives, CNC frame, MJF shells | 1 Apr 2027 | 1 May 2027 | 4 weeks | T3, T4 |
| T6 | Bare chassis walks at full mass under the overhead catch | 15 Jun 2027 | 15 Jul 2027 | 4 weeks | T5 deliveries |
| T7 | Skin fitted; 30-minute show loop; the museum's acceptance pack (risk assessment, insurance, barrier layout, charging rules) | 31 Jul 2027 | 31 Aug 2027 | 4 weeks | T6 |
| T8 | Live museum demo | Aug 2027 | 30 Sep 2027 | about 6 weeks from T7's target to the end of Q3 | T7 |

- **Slack does not add up.** A slip at one step uses the later steps' slack. The whole chain has about six weeks
  between the target demo readiness (mid-August) and the end of Q3.
- **The critical path is drive delivery.** Resellers listed the X4, X6 and X8 as backordered on 18 Nov 2026 (parts
  list, section 8, item 13). MyActuator's own store was not checked. Order the phase-1 drives as soon as the owner
  approves.
- **The foot cells are not on the walk's critical path.** Drive current gives a second contact estimate, so the bare
  chassis can walk (T6) before the cells arrive.

### 5.3 When the T. rex work must resume, and what the Triceratops de-risks for it

What the T. rex must show at end 2027 / early 2028 is the owner's call (section 7, decision 6). The owner said "I still
want to build the T-Rex", so the dates below assume a built T. rex at the milestone; the last column gives the later
dates if the milestone is simulation and renders only. This follows the parts check's F13 (parts list, section 7).

| T. rex work | Resume by (built T. rex) | If renders and simulation only | What the Triceratops work gives it |
|---|---|---|---|
| **Bench test A17** (each drive's command-to-motion delay, end to end on the bus) | During T4, Feb-Apr 2027, on the Triceratops foreleg bench: same X4-36-L and X6-60 drives. Add the T. rex bench sheet's rows to the foreleg bench plan | Same: it costs little on the shared rig | The measured delay replaces the assumed 10 ms by which every T. rex verdict was judged (P29, P32); the verdicts are then re-run at the bench value (P30) |
| **Simulation training from the preserved launch commit** (C1 `cc437b6a` with the data branch s2-b1: the day-one Colab smoke, then the 40M-step runs, at most 220 runtime-hours, P29) | Q1 2027, latest 1 Mar 2027, so that buy-gate stage 1 (a trained policy in harsher simulation; P40, P41) can be judged before the drives are ordered | Q3 2027, after the Triceratops demo | A first real Colab run of the shared pipeline (T3); a servo model calibrated on the bench |
| **T. rex drives** (75-M-H-10: 10 x X4 + 2 x X6) | Ordered in Q2 2027, once the foreleg bench confirms the drive family | Not needed | Measured torque, heat and speed of the same drives at 12S, including whether the X4 is capped at 83 rpm above 24 V |
| **CAD** (frame, drive mounts, shells at build detail) | Q2 2027, latest 1 Jul 2027, so the frame can be quoted when the drives arrive; legs-only prototype Q3-Q4 2027; walking under the catch Q4 2027 to Q1 2028 | Renders only | The Triceratops's frame, shell and supplier experience; the same materials |

- **What else the Triceratops de-risks:** the same compute and Teensy CAN firmware; the safety chain, power path,
  tether and charger; the bench rig, torque rig, catch gantry and thermal camera; the XM540 neck servo and the
  battery; and the museum relationship and acceptance process.
- **What it does not de-risk:** balance on two legs under delay (in the model the 75 cm T. rex stayed up in every run only at
  6 ms of delay or less); the pushes and quick turn it failed; its tail servo's load; its ankle drives. The X4's 83 rpm cap on 12S
  also needs a T. rex re-check (parts list, F1).
- **Staffing.** Resuming the T. rex in Q1 2027 runs two lines of work while the Triceratops is being built. A
  separate T. rex session, started from
  [RESUME.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/trex_form_study/RESUME.md)
  (working branch), keeps the two apart. Its restart section points back to this section for the dates once this
  folder is merged into the working branch; the Triceratops prompt carries the same resume line.
- **Keep the restart possible:** C1 and the data branch must not move (P50); the day-one sheet stays valid while they
  do not.

## 6. WIM's PLEM and W-RC on a quadruped: a question for the owner to raise with WIM

The deck pitched PLEM (a real-time control layer on a Jetson: a 1 kHz torque loop, sensor sync, thermal and regen
protection, a ROS 2 hardware layer) and W-RC (a hardware safety stop) for the biped. The fit analysis
[PLEM_FIT.md](https://github.com/kuds/mesozoic-labs/blob/claude/mesozoic-trex-stage2-refine-4jffre/hardware/triceratops_kickoff/PLEM_FIT.md)
(working branch; checked, sound after fixes) read PLEM's open repository at release v0.11.0. What still applies to a
slow museum quadruped:

| Pitched for the biped | On the Triceratops |
|---|---|
| 1 kHz torque loop | Matters less with three feet down, but PR #669 still tests 0-10 ms of delay, and its 1.5 Hz walk is a walk, not a crawl. PR #669's two classic CAN buses cannot carry 1 kHz; four buses or EtherCAT can. PLEM runs EtherCAT only, on the Jetson AGX Orin only (Orin Nano and NX "not supported"), at most 8 axes per bus, and its hardware evidence is two braked industrial servos |
| Protective stops | PLEM stops by disabling the drive and engaging its brake. These leg drives have no brakes, so the legs would go limp and the robot drop from about 400 mm. A quadruped, unlike a biped, can sit down under control, so the stop has to command a damped sit before power is cut |
| W-RC hardware stop | **Still applies in full.** It is not in PLEM's repository. The parts list designs a W-RC-style chain (a PL d wireless stop, a Pilz relay that sits the robot down and then opens the contactor, a watchdog) |
| Thermal and regen protection | Applies: held poses and long standing are the heat case; the 12S bus needs a regen clamp below the drives' 55 V |
| Sensor sync | Applies (IMU, foot cells, camera), but PLEM takes no sensor input |
| ROS 2 layer, telemetry, checked configuration | Useful tooling for either robot |
| Compute | The parts list recommends an Orin Nano Super; a PLEM path needs an AGX Orin ($3,499 list and +0.8 to 0.9 kg, per PLEM_FIT) |

PLEM_FIT recommends its option C: build the Triceratops on CAN for Q3 2027, buy the "-E" drives (CAN and EtherCAT on
the same drive) so EtherCAT stays open, and run a gated PLEM bench pilot on two, then three, drives off the critical
path, with a decision by the end of February 2027. For the T. rex flagship, PLEM is a candidate only if WIM clears the
same blockers, which are worse on a biped: any disable stop is a fall.

**Questions for the owner to send to WIM and MyActuator** (abridged from PLEM_FIT, section 5.4; the Triceratops
session drafts the messages, and the owner sends them). A bench pilot runs only after the answers clear the blockers;
its go or no-go is decision 7, by end of February 2027.

**To WIM:**

1. Would WIM reframe the pitch around a museum quadruped in Q3 2027, with the T. rex as the 2028 flagship case?
2. Has PLEM run brakeless drives such as MyActuator's "-E" series, and can its stop hold position or sit down damped
   instead of disabling?
3. Will one bus carry more than 8 axes, or have two buses run on one AGX Orin? When will the Orin Nano Super and NX be
   supported?
4. Is W-RC a product, and can it drive a sit-then-cut sequence rather than a plain power cut?
5. Is there a measured delay from a user's controller to the drive, and can a user's controller run inside the
   real-time loop?
6. What would a one-foreleg bench bring-up cost, or would WIM do it as a joint case study?

**To MyActuator:**

1. What are the "-E" drives' shortest EtherCAT cycle and sync modes, and their torque and position units in CiA 402?
2. What does a drive do when the EtherCAT watchdog expires or it is disabled (hold, damp or torque off), and is CAN or
   EtherCAT chosen by the wiring or by a setting?

## 7. Decisions for the owner

Asked with the question tool, at most four at a time, the recommended option first.

- **Round 1, by about 15 Dec 2026: decisions 1 to 5**, the five of "In short" step 3 and milestone T0, so that T2's
  order can go out in early January. Five questions take two calls: decisions 1, 2, 4 and 5, then 3.
- **Round 2, by about 15 Jan 2027: decisions 6, 7 and 8.** Decision 6 decides whether the T. rex simulation restarts
  by 1 Mar 2027 (section 5.3), so it cannot wait longer. Decision 7 says whether a PLEM pilot is planned; its go or
  no-go comes by end of February 2027, after WIM's and MyActuator's answers (section 6).

1. **Mass: 20 kg (T20, recommended) or 15 kg (T15).**
   - T20 passes the 0.30 m/s walk on every axis with room to spare and the 0.60 m/s trot at the 55 % front-load design
     point. Mass reserve 1.17 kg. Cost $37.3k nominal, landed ($22.9k low, $66.4k high); $31.8k measured the T. rex
     way; about $29.3k landed with the parts list's savings.
   - T15 keeps PR #669's drive set, with a 1.3:1 flexion linkage and 48 V. Mass reserve 0.39 kg, before the stop
     receiver's unpublished mass. Its roll drives run over their heat limit in a sustained trot, so trots stay short.
     Cost $32.8k nominal, landed; $28.3k the T. rex way.
   - Neither reaches the 0.40 m/s walk with X4 pitch drives: it needs 75-77 rpm at the pitch joint against the X4's
     83 rpm cap, 91-93 % of the cap, with no margin under load.
   - **Against the owner's spend wish.** The class is $17k-$32k: the T. rex's 75 cm build ($17.2k self-built, $20.0k
     turnkey) up to its 100 cm flagship ($29k-$32k). Counted the T. rex way, T20 ($31.8k) is at the top of the class
     and T15 ($28.3k) inside it; landed, T20's nominal ($37.3k) is above it and T15's ($32.8k) just above. The first
     spend is the T20 one-foreleg bench set: $7.1k of parts, $8.9k landed. The drive prices are MyActuator's own US
     store (catalog, 7 Nov 2026); the PLEM fit's reviewer found third-party US drive prices about twice that
     (RobotShop, 18 Nov 2026), which would raise every total.
2. **Size: 75 cm to the frill crown (recommended) or another size.** 75 cm is the only size with loads, parts and a
   MuJoCo model. The 60 cm companion is a packaging sketch, not costed. Another size means re-sizing from the start.
3. **Compute: Jetson Orin Nano Super 8 GB with a Teensy 4.1 on four CAN buses (recommended).** Alternatives: PR #669's
   Pi 5 + AI HAT+ (about $290 against $399 for the Jetson kit); a Jetson AGX Orin only if the PLEM pilot clears. The
   Jetson matches the T. rex's Jetson study and the PLEM direction. Compute is not the bottleneck; the bus is.
4. **Bus voltage: 48 V class, 12S (recommended), or 24 V, 6S.** The X6 and X8 are 48 V drives; at 6S their speed
   falls to about 77 and 69 rpm no-load, which rules out the hind 1.3:1 linkage. 12S needs a regen clamp, a precharge
   and 12 V regulators for the servos. It does not speed up the X4: above its 24 V rating it is capped at 83 rpm
   (reseller text quoting MyActuator; to confirm with the maker).
5. **Chassis: PR #669's design with the parts list's changes (recommended), or adapt a proven open quadruped's leg
   design.**
   - A: PR #669's chassis, consistent across its PDF, OpenSCAD and MuJoCo files, with a four-bar pushrod instead of
     the 12 mm belt and, for T20, larger front flexion and roll drives. The one-foreleg bench tests its riskiest part
     before four legs are bought.
   - B: the nearest proven open layout at this scale is the MIT Mini Cheetah leg, which main's hardware doc (section
     4) already uses as precedent: hip and knee drives at the body, the knee driven through the upper leg, with a
     belt resonance near 30 Hz documented in its thesis. The Mini Cheetah weighs about 9 kg, and Pupper V3 (3 kg) is
     far too small, so scaling either to 15-20 kg and a 1.5 m body is new design in its own right.
   - Why A: it already fits this body, these drives and the museum look. Take B's lessons into A (keep distal mass
     low; test one leg for impact, bandwidth and heat before freezing the transmission).
6. **What the T. rex milestone at end 2027 / early 2028 shows.** (a) A built T. rex walking under a catch (what the
   owner said; recommended, with section 5.3's Q1 2027 restart); (b) a legs-only rig plus a simulated walk; (c)
   simulation and renders only. Each later option frees Q1-Q2 2027 for the Triceratops.
7. **A PLEM bench pilot: yes, gated on WIM's and MyActuator's answers, decided by end of February 2027
   (recommended), or no.** With it, PLEM_FIT's option C: build the Triceratops on CAN for Q3 2027 with the "-E"
   drives, which keep EtherCAT open. The pilot runs only if the answers clear the blockers (brakeless drives
   unsupported, AGX Orin only, at most 8 axes per bus), and stays off the critical path
   ([PLEM_FIT.md](PLEM_FIT.md), sections 5.2 to 5.4).
8. **The pull-request triage of section 1** (close, merge, park), and the branch deletions of section 3.

## 8. What this plan rests on, and what was not done

- **Read on 18 Nov 2026:** the four surveys; the checked parts list; PLEM_FIT.md; RESUME.md and DECISION_LOG.md rows
  P29-P51 on the working branch; GitHub's open pull requests; `git ls-remote` and `git merge-tree` (the working branch
  at `ceadab71`, the archive branch at `f71cba4b`, the data branch at `ecf95a81`, main at `0748ac12`).
- **Not done:** no CI, test or simulation was run; no PR diff was read beyond the surveys; no vendor, museum or WIM was
  contacted; no lead time was confirmed with a vendor (the ones above are distributor figures found by web search on
  18 Nov 2026); no fork content was read. The milestone dates and the T. rex resume dates are planning estimates, not
  commitments.
- **Revised after the first critic round** ([reviews/critic_round1_18nov.md](reviews/critic_round1_18nov.md)), 18 Nov
  2026: PR #669's fixes as an errata the owner applies; the T. rex resume line in the prompt and a pointer to section
  5.3 in RESUME.md; the decision rounds dated; cost against the spend wish; SUE and both museums named; the prompt's
  pull-request list, PR fetch command, 0.40 m/s, belt and 18 mm figures, and render size limit corrected; the parts
  list's section 7 timing marked as replaced by section 5.3.
