# Clean-checkout reproduction audit: PR #662, #663 and #664

Run on 2026-11-07. The question was which committed results regenerate from a clean checkout of
each PR head, and which do not.

*Later environment (note added 14 Nov 2026):* this audit is of the main repository's PRs as they stood on 7 Nov. The
form study's own environment since Stage 1 (Python 3.13.18, MuJoCo 3.10.0, numpy 2.5.4, torch 2.15.0,
stable-baselines3 2.9.2, gymnasium 1.4.0 and the rest) is pinned in `tools/requirements-study.txt`; Stage 1's PPO
training and its R3/R4 evaluation rows used torch 2.9.0+cpu, stable-baselines3 2.9.0 and gymnasium 1.3.0
(`tools/requirements-stage1-ppo.txt`). How a fresh clone
restores the study's git-ignored inputs is in `tools/README.md`, and why tests run with `MUJOCO_GL=disable` is in
`LESSONS_LEARNED.md` §4.5.

## Short answer

* **#662 (e103f7ff):** every mechanical and engineering result regenerates. Numbers match to
  floating-point noise (relative difference ≤ 1e-11) and every pass/fail outcome is the same.
  Four exceptions:
  1. The committed `engineering_report.json` is stale against its own committed sources.
  2. The committed single-support MP4 does not regenerate byte-for-byte.
  3. 11 registry/contract tests fail on the clean checkout.
  4. No learned-policy result (gait or person-follow) can be reproduced, because no policy
     bundle, checkpoint or training input is committed.
* **#663 (b5c18bce):** nothing regenerates. It adds 9 result files and no code, and both commits
  it cites are missing from GitHub.
* **#664:** the same commit as #663. Merging it into `main` conflicts in 4 files.

## Setup

* Python 3.13.18, MuJoCo 3.10.0, numpy 2.5.4 on x86_64 Linux. The committed reports record numpy
  2.3.5, and `baseline.json` names an Apple M1 Pro host.
* Also installed: gymnasium 1.4.0, pytest 9.2.0, imageio 2.38.1, imageio-ffmpeg 0.8.1, and
  opencv-python-headless 5.0.0.93 (the autonomy tests need it). Torch and Stable-Baselines3 were
  not installed.
* Each PR head was checked out into a separate git worktree, and both worktrees were removed
  afterwards. Runtimes are wall-clock on 4 CPUs shared with a 3-worker sweep.

## 1. PR #662 commands

All commands run from the worktree root.

| Command | Exit / time | Committed files changed | Result |
|---|---|---|---|
| `python -m examples.tyrannosaurus_rex_60cm.platform.build` (and `--variant all`) | 0 / 0.12 s | none | All 3 `model.xml` + `components.json` and the registered `tyrannosaurus_rex_intermediate.xml` are byte-identical |
| `… platform.engineering --service-xml <scratch>/rex-service.xml` | 0 / 1.07 s | `engineering_report.json` | Committed report is stale (note A) |
| `… platform.tendon_bench` | 0 / 0.51 s | `tendon_bench.json` | 35 of 123 numbers differ; max relative 1.8e-12. All recorded hashes match |
| `… platform.proportions` | 0 / 21.3 s | `proportions.json`, `proportions.png` | JSON: 22 of 96 numbers differ, max relative 9e-14. PNG: rendered geometry is pixel-identical; 1.8 % of pixels differ, all in the four text rows (font rasterization) |
| `… platform.validate_platform --verify-sources` | **1** / 0.27 s | none | `Recorded renderer_sha256 differs for …/gait_video.py` (recorded fabf2933…, committed file c6ca4224…). The `source_sha256` block and the MP4 hash match |
| `… platform.validate_platform --video` | 0 / 59.5 s | `mechanical_report.json`, `nominal_single_support.mp4` | Report reproduces (note B), including "overall mechanical release gate: fail". The video does not regenerate byte-for-byte |
| `… platform.study_clearance` | 0 / 3.25 s | creates `clearance_study/components.json` and `m02_integrated/clearance_study_r02.json`, never committed | The regenerated r02 figures match `mechanical_checks.md`: 9.960 kg; 145 of 5,717 flagged vs 162 of 5,315; 20 cleared, 3 new; hard-floor hold 1.936 s at 5.57°; carpet 18.29° (fail) |
| `python -m examples.tyrannosaurus_rex_60cm.build` (M1) | 0 / 0.18 s | none | `model.xml`, `components.json`, `mass_budget.csv` identical |
| `python -m examples.tyrannosaurus_rex_60cm.validate` (M1) | 0 / 1.84 s | `m01_skeleton/validation.json` | All 248 numbers identical; only the numpy version string and the order of 28 `excluded_pairs` (same set) differ. "Clearance: pass"; "Full joint-box probe: fail {'combined_range': 10}" |
| `python -m examples.tyrannosaurus_rex_60cm.render` (M1) | 0 / 19.5 s | `render_manifest.json` (one hash) | `preview.png` and `hardware_layout.png` byte-identical |
| Archived initial static screen (validator + model inside `nominal_initial_sources.zip`) | 0 / 0.4 s | n/a | Matches `nominal_initial_screen.json` (max relative 5.4e-16). `robot_floor.py` and the compact/wide inputs are recorded but not archived |
| `… platform.evaluate_gait --reference {nominal,compact,wide} --panel development` | 0 / 7–15 s | none | Matches the mechanics README text; no committed data exists to compare against. Nominal 0/8 (all fail transition acceleration, 0.108–0.113 m/s); compact 4/8 (carpet passes, hard floor fails on acceleration); wide 0/8 (body contact at 4.1 s on hard floor, 10.9–11.3 s on carpet) |

**Note A, engineering report.**
* Numerically the regenerated report agrees: 59 of 3,041 numbers differ, max relative 1e-14.
* It has 3 extra keys (`variants.*.installed_size_checks`), and its `assumptions_sha256` and
  `validator_sha256` change. The committed report carries the hashes of the b4c8821 versions.
  e103f7f then edited `assumptions.json` and `engineering.py` without regenerating the report.
* The README's regulator and service-proxy figures (12.0 mm, 9.6 mm, more than 10.7 mm, two
  undersized frozen proxies) exist only in the regenerated output.
* The README headline figures all match: $11,966 / $15,621 / $20,636; 106 / 171 / 246 W;
  31.0 / 19.3 / 13.4 min.

**Note B, mechanical report and video.**
* Free-root trials agree to 3.3e-11 relative, and static checks to 3e-15.
* All trial statuses and boolean checks are identical, as are the headline holds (1.942 / 1.964 s)
  and the carpet tilt (18.28°).
* Clearance: the same 500 / 578 / 512 flagged pairs, in the same order with the same statuses.
  Some per-family sample counts shift by up to 18 of about 39,000, and the reported sample index
  moves for about 100 pairs (distance change ≤ 0.057 mm).
* Video: both files are 80 frames at 880×380 and 25 fps. Frame 40 shows the same pose, time and
  PASS label with different camera framing, so the bytes differ.

**Not reproducible from #662:**
* No policy bundles or checkpoints are committed. `evaluate_gait --bundle`, `gait_controls`,
  `fit_gait_*`, `follow_demo --policy` and `train_low_noise` therefore cannot reproduce any
  learned-policy claim.
* `hardware/tyrannosaurus_rex_60cm/platform/README.md` (linked from 3 documents) and
  `platform/gait/` (linked from 2) do not exist.

## 2. Tests on the clean #662 checkout

* **Documented intermediate tests:** 46 passed in 0.43 s. The files are `test_intermediate_mechanics`,
  `test_robot_floor`, `test_robot_clearance`, `test_trex_60cm_skeleton_study`,
  `test_robot_autonomy_transport`, `test_robot_autonomy` and
  `tyrannosaurus_rex/tests/test_intermediate_gait_protocol`.
* **8 registry/contract files:** 277 passed, 11 failed, with the worktree on `PYTHONPATH`. Without it,
  3 more fail (`test_refusals_come_before_the_run[…]`: `No module named 'environments'` in a
  subprocess). The 11 failures:
  * plant manifest is stale for `tyrannosaurus_rex/intermediate` (3 tests);
  * intermediate is missing from the digest-snapshot tables, and its golden probes fail (4 tests);
  * the species catalog is stale: it lists policy interface r1 where the manifest and
    `plant_versions.toml` say r3 (4 tests).
* **`python -m environments.shared.plant_contract --check`** exits 2 ("generated plant manifest is
  stale"). Exactly 3 values differ, all for intermediate: physics sha (797cbcfa… → b83d0d4f…),
  visual sha (14d063ae… → 0a604bb1…) and bundle sha. The MJCF source digest (06019e3c…) and all
  other plants match.
* **Not run (time):** most of the roughly 1,400 tests in the #662-touched files. A partial run
  reached 167 passed, 0 failed, all in `test_behavior_env`, before it was stopped. The one test
  that needs Stable-Baselines3 was skipped.

## 3. PR #663 (b5c18bce)

It adds 9 files under `hardware/intermediate_rex_r2/`:
* `FINDINGS.md` and `Partner-brief.pdf`;
* 5 JSON summaries (autonomy, gait panel, baseline, release checks);
* 2 PNGs.

There is no code. `examples/intermediate_rex_r2` exists in no remote ref and nowhere in history.
Every input the JSON files reference is absent: trial reports, episode records, the successor
`model.xml`, the policy bundle and the CI comparison.

The cited commits cannot be fetched:
```
fatal: remote error: upload-pack: not our ref f7d9857791e0b75d388118fd38fac2d2ce2c0794
fatal: remote error: upload-pack: not our ref eb0e984a6eebdcefc324268e03f9e6ba94c721f3
```
The GitHub API returns `422 No commit found for SHA` for both, and for `85ae1705` (the short SHA
in `baseline/source.json`). `FINDINGS.md` cites f7d9857, while `release-checks.json` records
checkout eb0e984a.

| FINDINGS claim | In committed data? | Regenerable? |
|---|---|---|
| Hip-window fill 48 → 84 %; only 60 cm completed the initial walk screen; sole spacing 214 → 175 mm cut sway 14.4 % | Only in `FINDINGS.md` | No (no models, ledgers or scripts) |
| Specimen table (crown, mass, runtime, cost for R2, mid, large, R3); $21,817 high estimate | Only in `FINDINGS.md` | No |
| `common-scale.png`, `redesign-at-60cm.png` | PNGs only | No generator |
| Gait 13/20 vs reference 2/20; 0.104 m/s; 16.4 mm; 5 acceleration and 2 tilt failures | `reconfirmation` block of `panel-summary.json`. Its `confirmation` block, and `index.json`, say 15/20 | No (episodes, policies, R3 model and evaluator absent) |
| Pedestal follow trials: raw 3/4 equal to reference; fitted corrections 2/4; no gallery contact or identity switch in 12 trials | `raw-policy/summary.json` | No (per-trial reports and `follow_trial` absent) |
| Governed: crossing cart 1/4, pedestal 0/4 (all excessive tilt) | `governed/index.json`. "Stopped, resumed" is not recorded there | No |
| R3 hip needs about 9 N·m against 6 N·m; some joint positions collide | Not recorded for R3 | No |
| 450-frame byte-identical video; 983 tests passed | `release-checks.json` | No (commit missing; module and inputs absent) |

## 4. PR #664

* Same commit as #663: `refs/pull/664/head` and `codex/trex-body-shape-posture-study` are both at
  b5c18bce.
* GitHub reports 36 commits and 421 files against `main` (820dbdaf), with `mergeable: false`.
* `git merge-tree --write-tree origin/main origin/codex/trex-body-shape-posture-study` exits 1,
  with content conflicts in 4 files:
  * `configs/digest_snapshot.generated.txt` (3 hunks);
  * `environments/shared/tests/test_plant_contract_manifest.py` (1 hunk: `main` has
    `VELOCIRAPTOR_MONGOLIENSIS: (69, 22)`; the branch adds `intermediate (57, 14)`,
    `robot (41, 10)` and `(70, 22)`);
  * `environments/shared/tests/test_species_catalog.py` (2 hunks);
  * `environments/shared/tests/test_species_integration.py` (1 hunk).
* The generated manifests, `species.generated.json`, `plant_versions.toml` and
  `species_manifest.toml` merge cleanly as text. They were not validated after the merge.

## What this means for this study

* The #662 mechanics are a sound numerical baseline. This study's statics reproduce its
  double-support torques and its single-support hip-roll figure; see `BASELINE.md` §4.
* The #662 *learned-control* results, and every #663 result, are claims without reproducible
  evidence. They are cited as reported, and nothing here depends on them.
* Resolving the 11 contract test failures and the four #664 conflicts belongs to the integration
  of #662–#664. It is outside this study, and this branch does not touch those files.
