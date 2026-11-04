# port/m4 -- the M4 judge (K1-K4) and the parent rule, ported to this machine

Folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/port/m4` (= `$M`).
Python: `work/venv/bin/python -I -B` (Python 3.13.18; torch 2.13.0+cpu, numpy 2.5.4, gymnasium 1.4.0, SB3 2.9.0,
mujoco 3.10.0). Every command and its UTC time is in `CMDLOG.txt`. Labels: [measured: source] / [inferred: basis].

## What is here

| path | what |
|---|---|
| `orig/main_laneA/` | the originals: `git show origin/main:docs/investigations/stance_2026_10/stance6/tools/laneA/tools/*.py.gz` (+ gunzipped), `SHA256SUMS.txt`, `runs/run_validations.sh`. judge 76da649c, kscore 3a2e8ea9, r10m_common 97431323, make_judge 7e85dfa9 = laneA/SHA256SUMS.txt [measured: sha256sum] |
| `orig/prereg/` | `origin/claude/handoff-phase-critic-prereg:tools/{pc,additions,orig,adapt_diffs}` and criteria.txt; every `tools/pc/*` file equals its `round/APPROVED` pin [measured: checks in CMDLOG]; `tools/orig/laneA/judge_r10ltsb.py` = main's (76da649c) |
| `ref/` | STANCE6's records from origin/main: `panels/stance6_r10LTSB_s{60,42,80}/<label>/m4/judge_<label>.json` (18), `kscore/kscore_stance6_r10LTSB_s{60,42,80}.json`, `stance6_APPROVED` (sha256 07246480..., the `approved_sha256` the STANCE6 records name) |
| `deps/` | main's published copies of the judge's/scorer's inputs (score_stance3.py, judge_stance5r.py, sigma_s70_s80.json, frozen_params.json, score_stance5r.py) |
| `deps_restored/` | `build_deps.py` output: those copies with the scratch-path prefixes restored (see DEPENDENCIES) + `BUILD_REPORT.json` |
| `copies/` | `build_copies.py` output: sealed measuring copies `copy_8737cd3d`, `copy_1dfe2327` (r10m_common.build_copy, git archive of the read-only handoff repository) + `BUILD_COPIES.json` |
| `make_ports.py` | derives `s6/` and `pc/` from the originals by exact counted replacements; writes `diffs/*.diff` (port vs original) |
| `s6/` | STANCE6 binding: `judge_r10ltsb.py`, `kscore_r10ltsb.py`, `r10m_common.py` |
| `pc/` | PC binding: `judge_r10ltsb.py`, `kscore_r10ltsb.py`, `r10m_common.py` (the PC round's adapted one), `parent_m4.py` (parent rule harness), `scorer/` (byte copies of the pinned PC scorer modules), `v4_actor_only.py` (V4 hook harness) |
| `checks/` | `derive_check.py`, `jcompare.py` and the outputs of the build/compare steps |
| `out/v1..v4`, `out/parent` | validation outputs |

(Results, the change list and the run commands follow below; they are filled in as the validations complete.)

## DEPENDENCIES (checked against the sha256 the originals and the STANCE6 records state)

The originals read four files from a scratch folder that does not exist here
(`/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-.../scratchpad`). main holds published copies whose bytes differ:
the publication shortened absolute scratch paths to `scratchpad` (and in frozen_params.json replaced two commit-trailer
lines by `"<redacted: ...>"`). `build_deps.py` restores exactly those strings [measured: checks/build_deps.out,
deps_restored/BUILD_REPORT.json]:

| file (origin/main:docs/investigations/stance_2026_10/...) | main's sha256 | restored sha256 | pinned by | equal to pin |
|---|---|---|---|---|
| stance3/judge/score_stance3.py (line 63 `SCRATCH`) | 2756e3a4 | 2c3879d3 | kscore SC3_SHA256; STANCE6 kscore records `score_stance3.sha256` | YES |
| stance5/replication/judge/judge_stance5r.py (lines 3, 114, 115) | 77f606e2 | 16d39f7d | make_judge SRC_SHA256; judge docstring | YES |
| stance5/replication/sigma_s70_s80.json (every `"scratchpad/` string) | 939285e3 | 89778b3f | judge CONTROL_SIGMA_SIDECAR_SHA256, kscore SIDECAR_SHA256; STANCE6 judge records `control_sigma_file_sha256` | YES |
| symmetry/frozen_params.json (every `"scratchpad/` string) | b5366a0e | 47c2788e | judge/kscore FROZEN_PARAMS_SHA256 58b2c8fb; STANCE6 records `frozen_params_sha256` | **NO** (the redacted commit_trailer cannot be restored) |

- judge_stance5r.py is not read at run time; with it restored, make_judge_r10ltsb.py's replacements derive
  judge_r10ltsb.py byte for byte (76da649c) [measured: checks/derive_check.out].
- frozen_params.json: the judge and scorer read only `criteria.H4` (control_sigma[seed].source / .vector,
  episodes_per_seed 20, spawn_range [3042, 3061], space "joint"). In all 18 STANCE6 judge records the recorded
  `noise_matched_control_joint.sigma_source`, `.sigma_vector` and `.seeds` equal the restored file's (seeds 60, 42) or
  the pinned sidecar's (seed 80) exactly; and the pinned sidecar's control_sigma 70 and 80 are the element-wise mean of
  the restored file's 42/52/60 vectors to 0.0 [measured: deps_restored/BUILD_REPORT.json frozen_check]. So the port reads
  the restored view under its own pin (FROZEN_PARAMS_FILE_SHA256 = 47c2788e) and keeps 58b2c8fb as the original's
  identity (the sidecar's `source_sha256` check). Consequence: every record the port writes says
  `frozen_params_sha256` 47c2788e... where STANCE6's say 58b2c8fb... (listed in V2).
- the K3 panel's copy inside the measuring copies (docs/.../sigma_s70_s80.json) is 939285e3 = K3_SIGMA_REPO_SHA256 in
  both copies [measured: sha256sum].
- The measuring copies: `copies/copy_8737cd3d` seal = STANCE6's measuring copy's seal (tree digest 331fe8fc..., 1406
  files, archive sha256 e7ae1e91...) and = the coordinator's work/copy_8737cd3d; `copies/copy_1dfe2327` (9417b765...,
  1410 files) = work/copy_1dfe2327 [measured: copies/BUILD_COPIES.json].

## CHANGES (port vs original; full unified diffs in diffs/; each replacement in make_ports.py)

s6/r10m_common.py (diffs/s6_r10m_common.py.diff), from orig/main_laneA/r10m_common.py:
1. `BR` -> the handoff worktree repository (read-only git: `cat-file`, `rev-list`, `diff` with --no-optional-locks):
   where approved_branch_head checks the HEAD. Binding (path).
2. `APPROVED` -> ref/stance6_APPROVED (main's copy of STANCE6's approval record, sha256 07246480 = the records'
   `approved_sha256`). Binding (path).

s6/judge_r10ltsb.py (diffs/s6_judge_r10ltsb.py.diff), from orig/main_laneA/judge_r10ltsb.py:
3. `FROZEN_PARAMS` -> deps_restored/symmetry/frozen_params.json. Binding (path).
4. + `FROZEN_PARAMS_FILE_SHA256 = 47c2788e...` and the file check in main() uses it (message names both). Binding: the
   original bytes are not available (DEPENDENCIES).
5. `CONTROL_SIGMA_SIDECAR` -> deps_restored/.../sigma_s70_s80.json (same pin 89778b3f). Binding (path).
   Unchanged and unused in full mode: COMMON_SIGMA_ZIP, TAIL_SEEDS_FILE (only --common-noise / --tail read them; those
   report-only modes are NOT available on this machine). VERSION and every other line unchanged.

s6/kscore_r10ltsb.py (diffs/s6_kscore_r10ltsb.py.diff), from orig/main_laneA/kscore_r10ltsb.py:
6. `SC3_PATH` -> deps_restored/stance3/judge/score_stance3.py (same pin 2c3879d3). Binding (path).
7. `FROZEN_PARAMS` -> restored file; + `FROZEN_PARAMS_FILE_SHA256`, used by frozen_with_sidecar's file check (the
   sidecar's source_sha256 is still checked against 58b2c8fb). Binding (as 4).
8. `SIDECAR` -> restored sidecar (same pin). Binding (path).
9. `VIEW_TMP` -> port/m4/tmp_views (the RA-2 symlink views; was an S2 caches path). Binding (path).
   Unused and unchanged: REPL_SCORE_FINAL, REPL_JUDGED (--reproduce-repl only; not available here).

pc/r10m_common.py (diffs/pc_r10m_common.py.diff), from orig/prereg/tools/pc/r10m_common.py (c070a4e2 = APPROVED pin):
10. `BR` -> the handoff worktree repository (the PC tools' WT); 11. `ROUND` -> handoff/round (<PCR>: APPROVED,
   criteria.txt of record), so `APPROVED` = handoff/round/APPROVED. Binding (paths). Its approved_branch_head (the PC
   rule: HEAD from APPROVED, descends from 8737cd3d, M1/M3 pinned files unchanged) is the PC round's, unchanged.

pc/judge_r10ltsb.py (diffs/pc_judge_r10ltsb.py.diff) = 3-5 plus the PC binding:
12. `BRANCH_NAME` -> "claude/screen-trex-r10-phasecritic" (recorded as branch_expected). Binding (branch).
13. + `--run-id` (required) and `pc_switch_binding()`: the arm from the RUN_ID (stance7_r10LTSBpc_s* PHASE,
    stance7_r10LTSBoff_s* OFF; RUN_ID must be r10m_common's for --seed), the recipe pin HP_SHA256 = r10m_common
    HP_SHA256 (c64211e8, switch ON) or HP_SHA256_ROUND (054ab483, switch OFF), each checked against round/APPROVED
    pins.switch_on / switch_off. Binding: the two arms train one HEAD under two recipes; the original pins one
    (054ab483) and would refuse every PHASE run (run_hyperparameters_sha256 c64211e8).
14. Ctx: the clone's recipe digest is regenerated with `[ppo] critic_phase = true` for the PHASE arm, exactly as the
    notebook's resolve cell writes it (copy_1dfe2327 notebooks/sb3_training.ipynb cell 9:
    `STAGE_CONFIGS[...].setdefault("ppo_kwargs", {})["critic_phase"] = True`); unchanged for OFF. Binding (as 13).
15. + `pc_checkpoint_binding()`: the checkpoint's `mesozoic_critic_phase` attribute (read from the zip's data member,
    nothing unpickled) must be present for PHASE, absent for OFF, else a hard failure (refused before rolling); the
    record gets `pc_binding` {run_id, arm, switch, pin, approved + sha256, r10m_common, checkpoint_critic_phase,
    hyperparameters_sha256_clone_with_switch}. Binding (the checkpoint is the arm's), report fields added.
16. + `import importlib.util`, `import re` (used by 13).

pc/kscore_r10ltsb.py (diffs/pc_kscore_r10ltsb.py.diff) = 6-9 plus:
17. + `pc_hp_pin()` and, in score_run, score_stance3's `HP_SHA256` rebound to the run's arm's pin (as 13), recorded in
    `hp_pin` and the `rebound` list. Binding: score_stance3.validity_issues refuses a record whose run
    hyperparameters_sha256 is not HP_SHA256 (score_stance3.py lines 291-293), which every PHASE record would be.
    HEADS is rebound to the HEAD of round/APPROVED (1dfe2327) through the PC r10m_common, as the original reads its own.

Not changed anywhere: every measurement, rail, statistic, seed, bar, the K1-K4 arithmetic (score_stance3.py is the
pinned original), the parent rule (score_r10round.py is the pinned original).

## VALIDATIONS (all four PASS)

V1 kscore re-score -- PASS [measured: out/v1/compare_s{60,42,80}.{txt,json}, out/v1/summary.txt].
`s6/kscore_r10ltsb.py --panels ref/panels stance6_r10LTSB_s<S> <S>` over STANCE6's 18 judge records reproduces
ref/kscore/kscore_stance6_r10LTSB_s{60,42,80}.json leaf for leaf (115 / 130 / 118 leaves): every K value, verdict, K1
failure mode, `missing`/`invalid`/`notes`, `head`, `score_stance3.sha256` (2c3879d3), the approval's sha256 and comment
lines. The only differences, 10 per run, are paths: `score_stance3.path`, `judged_root`, `panels_files.*` (6),
`head_basis.approval.approved`, `head_basis.approval.repo`. Reproduced verdicts: s60 HOLDS; s42 fails K1; s80 fails
K1 and K3 (joint falls 6.3M 5, final 2).

V2 s6 judge on OFF s80 final with copy_8737cd3d -- PASS [measured: out/v2/compare.{txt,json}, out/v2/judge.log].
`s6/judge_r10ltsb.py work/fetch/stance7_r10LTSBoff_s80 --arm LTSB --seed 80 --steps final --clone copies/copy_8737cd3d
--allow-commit-mismatch` (the relaxation is needed: the OFF run's stage_config.json records git_commit 1dfe2327, the
copy is 8737cd3d). Against ref/panels/stance6_r10LTSB_s80/final/m4/judge_final.json: 9784 leaves, 39 differ, none in
a panel, statue, noise, band or check value; every episode row of the panel, the statue and both noise tests is
bitwise equal (clean 40/40, LCB 0.92784, joint falls 2 [3051, 3061], own falls 0, K2a 13 under joint noise ...).
The 39, by cause:
- paths (24): argv[0], argv[8], argv[10], judge, clone, run_dir, stage_dir, model, vecnorm, frozen_params,
  control_sigma_file, k3_panel_sigma.file, branch_head_approval.approved, branch_head_approval.repo,
  checks.symmetry.clone_mirror_map_path, git.seal.repo, modules.* (8);
- times (5): panel.panel_seconds, noise_matched_own.panel_seconds, noise_matched_control_joint.panel_seconds,
  judge_seconds, git.seal.built_utc;
- seal / tool identity (2): git.seal.built_by, judge_sha256 (the port's file);
- HEAD (4): argv length (+ --allow-commit-mismatch), checks.relaxed.allow_commit_mismatch true,
  checks.stage_config.git_commit 1dfe2327, checks.stage_config.git_commit_match false;
- dependency pin (1): frozen_params_sha256 47c2788e (the restored view; DEPENDENCIES);
- interpreter (1): python 3.13.18 (3.13.17 then);
- input file identity (2): model_sha256 and panel.handoff.checkpoint_sha256 f002647b (the OFF run's own zip) vs
  55ca16f4 (STANCE6's s80 zip). Not path/time/seal/HEAD: the two zips are different files. vecnorm_sha256 is equal
  (47dab7fd). That the weights are equal was measured last round (task statement), not re-measured here; the bitwise
  equal rollouts are consistent with it [inferred].
Equal, among the checks: copy_tree_digest_after and git.tree_digest (331fe8fc = STANCE6's copy), git.head 8737cd3d,
control_sigma_file_sha256 89778b3f, k3_panel_sigma sha256s, task, plant, symmetry, seed, num_timesteps, n_envs,
hyperparameters_sha256 (054ab483 match true), hard_failures [].
The original scorer refuses this V2 record (relaxation + commit), as it should [measured: out/v2/kscore_s6_relaxed.json:
invalid "judged with allow_commit_mismatch", "git_commit 1dfe2327 is not the judged clone HEAD"].

V3 PC judge on the same checkpoint with copy_1dfe2327 -- PASS [measured: out/v3/compare_v2.{txt,json},
out/v3/compare_ref.{txt,json}, out/v3/judge.log].
`pc/judge_r10ltsb.py ... --clone copies/copy_1dfe2327 --run-id stance7_r10LTSBoff_s80`, no relaxation:
git_commit_match true, hyperparameters_sha256 054ab483 (run = clone = OFF pin), HEAD 1dfe2327 from round/APPROVED
(checked by the PC r10m_common rule in the handoff repository), checkpoint has no mesozoic_critic_phase (OFF), hard
failures []. Against V2: 48 leaves differ, all metadata (paths, times, seal/commit/tree digest of the other copy, HEAD
fields, branch_expected, branch_head_approval (the PC rule's fields), judge_sha256, argv, modules_all_under_clone 99 ->
100 (one more environments module loads from 1dfe2327 [inferred: the changed mirror_kl_ppo imports critic_phase]),
the new pc_binding block); panel, statue, both noise tests and every other check equal bitwise. Against STANCE6's
record: 60 metadata differences, no value difference.
Scored by pc/kscore_r10ltsb.py (out/v3/kscore_stance7_r10LTSBoff_s80.json): the final cell is VALID (invalid [], the
5 other J slots missing as not judged), and its K values equal STANCE6 s80 final's: K1 pass at final (40 clean, LCB,
components), K2a 0, K2b 0, K2c median 49.857776 mm, offset -2.880002 mm, K3 joint 2 / own 0, K4 0.

V4 PC judge smoke on PHASE s42 final -- PASS [measured: out/v4/judge.log, out/v4/hooks.json,
out/v4/panels/stance7_r10LTSBpc_s42/final/m4/judge_final_smoke5.json (sha256 d723a2c3...)].
`pc/v4_actor_only.py --hooks-out out/v4/hooks.json work/fetch/stance7_r10LTSBpc_s42 --arm LTSB --seed 42 --steps final
--clone copies/copy_1dfe2327 --run-id stance7_r10LTSBpc_s42 --smoke-episodes 5` (5 episodes per panel; the record is
marked smoke and no scorer accepts it; its numbers are not read here). Binding: arm PHASE, pin c64211e8 = run =
clone-with-switch digest, git_commit_match true, no relaxation, hard failures [], copy unchanged after.
Only the actor acts:
- from the record: `pc_binding.checkpoint_critic_phase` = {schema mesozoic.critic-phase/v1, features ramp/settled/
  discount, settle_steps 200, gamma 0.98, width 3, input "the critic's only ..."}: the checkpoint's critic reads the
  phase; the sidecar out/v4/hooks.json: both policies SB3 built (panel load, noise load) are
  environments.shared.critic_phase_ppo.CriticPhaseActorCriticPolicy with observation_space (63,), actor first layer 63
  inputs, critic first layer 66; module calls during the whole judge run: actor policy_net 15000 and action_net 15000
  (5 panel + 5 own-noise + 5 joint-noise episodes x 1000 steps; the statue rolls zero actions), critic
  mlp_extractor.value_net 0 and value_net 0; the actor's input width was 63 at every call.
- from code: the judge builds actions only through (a) stance_report._load_policy's predict (copy_1dfe2327
  environments/shared/reporting/stance_report.py lines 200-206: model.predict on the 63-value normalised observation)
  -> SB3 ActorCriticPolicy._predict -> get_distribution -> mlp_extractor.forward_actor (venv stable_baselines3/common/
  policies.py lines 709-718, 743-752), used by the panel (pc/judge_r10ltsb.py line 902); (b) noise_matched's explicit
  `pol.action_net(pol.mlp_extractor.forward_actor(f))` (pc/judge_r10ltsb.py line 1000); (c) the statue's zero action.
  forward_actor slices the first 63 columns (critic_phase_ppo.py lines 207-212); no path calls predict_values,
  forward or evaluate_actions; and forward_critic refuses any input that is not 66 wide (critic_phase_ppo.py lines
  214-220), while the judge never builds a 66-wide input, so a critic call would have raised.
The PC scorer on this smoke record (out/v4/kscore_pc_s42_smoke.json) lists only smoke-made issues (smoke; 5 not 40
panel / statue episodes; noise n and spawns); the original scorer (out/v4/kscore_s6_s42_smoke.json) adds the HEAD
(1dfe2327 not 8737cd3d) and recipe (c64211e8 not 054ab483) issues that the PC binding removes.

## THE PARENT RULE (pc/parent_m4.py)

A harness, not a rewrite: it checks the sha256 of pc/scorer/{score_r10round, r10round_common, ledger_r10round,
complete_r10round}.py against round/APPROVED's pins, imports score_r10round.py and calls its `parent_named(per_seed,
round_dir, line, panels_format)` unchanged, as score_pc.py line 444 does (line 4, "{label}/{kind}").
Inputs it reads (score_r10round.py lines 505-540): per run, classes.gate and classes.outcome (from the PC scorer's
per_seed rows, never from M4); per kept run, the first file in sorted order of
`<round_dir>/panels/<RUN_ID>/final/m4/*.json` that holds both noise keys, and from it only panel.n_clean,
noise_matched_own.falls, noise_matched_control_joint.falls. It does not check the record's validity (schema, smoke,
relaxations, HEAD, slot), so only a valid full-mode judge_final.json may sit in that folder (a smoke record such as
out/v4/.../judge_final_smoke5.json would be read).
Checked on STANCE6's three final records with SYNTHETIC classes (out/parent/): all POSITIVE -> named s60 (40 clean, 1
fall) over s80 (40, 2) and s42 (38, 0); s60's record absent -> "not named: an input is absent"; s60 gate FAIL -> s80.
These agree with the rule computed by hand from the records [measured: out/parent/*.json]; there is no recorded parent
reading to reproduce: STANCE6's reading was line 1 [measured: origin/main stance6_results/readouts/READOUTS.txt
line 7], for which parent_named returns {} (it names only on lines 3 and 4).

## RUN COMMANDS (one (run, label); one simulation process at a time)

    M=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/port/m4
    W=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work
    # judge (RUN_ID stance7_r10LTSB{pc,off}_s{60,42,80}, S its seed, LABEL one of 4194304 5242880 6291456 8388608
    # 10485760 final; the output folder must exist; the judge never overwrites)
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=$M/caches/mpl nice -n 10 $W/venv/bin/python -I -B \
      $M/pc/judge_r10ltsb.py $W/fetch/RUN_ID --arm LTSB --seed S --steps LABEL --clone $M/copies/copy_1dfe2327 \
      --run-id RUN_ID --out PANELS/RUN_ID/LABEL/m4/judge_LABEL.json
    # K1-K4 of one run (reads PANELS/RUN_ID/<label>/m4/judge_<label>.json)
    $W/venv/bin/python -I -B $M/pc/kscore_r10ltsb.py --panels PANELS RUN_ID S --json OUT.json
    # the parent rule (DIR/panels/RUN_ID/final/m4/; CLASSES = the PC scorer's per_seed rows for the PHASE runs)
    $W/venv/bin/python -I -B $M/pc/parent_m4.py --round-dir DIR --classes CLASSES.json --line 4 --json OUT.json

About 290 s per full judge at 1 thread here (V2 292.0 s, V3 287.4 s; panel about 190 s, each noise test about 50 s).
A checkpoint needs both stage1_<N>_steps.zip and stage1_vecnormalize_<N>_steps.pkl (final: stage1_final.zip and
stage1_final_vecnorm.pkl) and 01_stance/{stage_config, task_fingerprint, plant_identity}.json.

## NOT AVAILABLE / NOT COMPUTABLE HERE

- frozen_params.json's original bytes (58b2c8fb): main's copy has a redacted commit_trailer; the port reads a restored
  view pinned 47c2788e (every value read equals the STANCE6 records'). Records say frozen_params_sha256 47c2788e.
- the judge's report-only modes --common-noise and --tail (their scratch inputs COMMON_SIGMA_ZIP / TAIL_SEEDS_FILE are
  not on this machine; not part of K1-K4), and kscore --reproduce-repl (REPL's judged files absent).
- weight equality of the OFF s80 final zip with STANCE6's (STANCE6's zip is not here): stated by the task from last
  round, not re-measured.
- a reference output for the parent rule (STANCE6 read line 1; nothing was ever named).

## RULE SLIPS

- One shell command had a sed syntax error (`sed -n 80-130p`); it printed nothing and was rerun correctly (CMDLOG).
- One check command passed a nonexistent path (`$M/copy_dummy`) to sha256sum; it read and wrote nothing.
- Some read-only checks used the system python3 -I instead of the venv (no simulation, no checkpoint loaded).
None of: training, optimizer steps, code changes in the copies, writes outside port/m4, git writes, Drive writes,
touching the main checkout, more than one simulation at a time.
