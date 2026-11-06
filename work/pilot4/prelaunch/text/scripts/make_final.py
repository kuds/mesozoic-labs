"""Build the final pilot-4 pre-registration text from draft v3 (reproducible; nothing else is read for its wording).

Stage 1 (corrections.py): the tools' fix stage's 18 text corrections, every OLD line checked against v3 first, applied
bottom-up, C6 in its alternative (a) form (decision (35)).
Stage 2 (here): every further edit is an exact replacement whose OLD text must occur exactly once, a section swap whose
old section must equal v3's lines exactly, or an insertion after a unique anchor line. Then the checks: width <= 118,
ASCII only, no trailing space, no absolute path, and a line diff against v3 in which every v3 line that is not kept
lies inside a region this script is allowed to change. Any failed check writes nothing.
Output: <SP>/work/pilot4/frozen_candidate/criteria_pilot4.txt; a report in data/ and the diff against v3 in data/.
Paths are worked out from this file's location; none is spelled.
"""
import difflib
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corrections  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
P4 = os.path.dirname(os.path.dirname(LANE))
WORK = os.path.dirname(P4)
OUT_DIR = os.path.join(P4, "frozen_candidate")
OUT = os.path.join(OUT_DIR, "criteria_pilot4.txt")
P3_TEXT = os.path.join(WORK, "pilot3", "frozen_candidate2", "criteria_pilot3.txt")
P3_SHA256 = "853f49d6b7846e29e820de5e6572d36fc9f6b2c20b83c178e46f9a62eb291df4"
SUMS = os.path.join(P4, "tools", "SHA256SUMS.txt")
SUMS_SHA256 = "0f42f29f45092ad87004519783af5175bcb1b7a7c61dcccd4985b682554cc448"
WIDTH = 118


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Text:
    def __init__(self, lines):
        self.t = "\n".join(lines) + "\n"
        self.log = []

    def rep(self, label, old, new):
        n = self.t.count(old)
        if n != 1:
            raise SystemExit(f"{label}: OLD occurs {n} times, not once; nothing written:\n{old}")
        self.t = self.t.replace(old, new)
        self.log.append({"edit": label, "kind": "replace", "old_lines": old.count("\n") + 1,
                         "new_lines": new.count("\n") + 1})

    def insert_after(self, label, anchor_line, new):
        lines = self.t.split("\n")
        idx = [i for i, ln in enumerate(lines) if ln == anchor_line]
        if len(idx) != 1:
            raise SystemExit(f"{label}: anchor occurs {len(idx)} times, not once; nothing written:\n{anchor_line}")
        add = new.split("\n")
        lines[idx[0] + 1:idx[0] + 1] = add
        self.t = "\n".join(lines)
        self.log.append({"edit": label, "kind": "insert_after", "new_lines": len(add)})

    def swap(self, label, start_marker, end_marker, expected_old, new):
        a = self.t.index(start_marker) if start_marker else 0
        b = self.t.index(end_marker)
        if self.t.count(end_marker) != 1 or (start_marker and self.t.count(start_marker) != 1):
            raise SystemExit(f"{label}: markers not unique; nothing written")
        old = self.t[a:b]
        if old != expected_old:
            raise SystemExit(f"{label}: the section is not v3's; nothing written")
        self.t = self.t[:a] + new + self.t[b:]
        self.log.append({"edit": label, "kind": "section", "old_lines": old.count("\n"), "new_lines": new.count("\n")})


HEADER = """\
T. rex walking pilot 4: fixed criteria (FROZEN CANDIDATE, 2026-11-06: written before any pilot-4 training step, the
only training since draft v3 being the pre-registered 65,536-step CPU notebook smoke of decision (36); built from
draft v3 with the maintainer's decisions (31)-(36), the tools' fix stage's 18 text corrections and the pre-launch
checks' measurements, see "Changes from v3" at the end, then "Changes in v3" and "Changes in v2"; not yet approved
or frozen)

Status. A candidate for the freeze, modelled on walking pilot 3's approved criteria ("P3": SP/work/pilot3/
frozen_candidate2/criteria_pilot3.txt, sha256 853f49d6b7846e29e820de5e6572d36fc9f6b2c20b83c178e46f9a62eb291df4
[measured: sha256sum; the run log's line 1037 records the approval with that sha256, and line 1039, the durable
copy, names it again]); [P3 Lnnn] cites its line nnn. Every pilot-3 rule is kept unless this text gives the evidence
for a change. The maintainer took every item Q1-Q27 of draft v3 as proposed and decided (35) and (36) ("Decisions
taken"); no bar, seed, budget, cadence or outcome rule differs from v3's, and the one value that does is the
straightness tool's turned-twin tolerance, a check in its tests, 1e-5 -> 1e-4 deg (decision (35)). Every pre-launch
check that can run before the push is filled from measured evidence; open, in this order ("Pre-launch checks"): the
maintainer's approval of this text (its sha256 recorded with the approval; it covers the smoke RESUME's departure
from pilot 3's procedure named there) and of the fourteen local commits (Q18), the OK to push, the push and the
pushed head checked, the durable copy of the pinned tools, the Colab steps, and the launch approval. No seed is
launched before all of them [P3 L15-16].
  Branch pilot-walk-r10-p4: local, never merged, no pull request, not pushed; head
  0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8, the commit the runs must record ("Code") [measured: the setup lane's
  clean clone of SP/work/pilot4/tools/tree, SP/work/pilot4/prelaunch/setup/logs/]. The pilot-4 tools:
  SP/work/pilot4/tools/, pinned by its SHA256SUMS.txt, sha256
  0f42f29f45092ad87004519783af5175bcb1b7a7c61dcccd4985b682554cc448 (1152 entries; it replaces the tools' fix
  stage's c2a86878..., kept as SP/work/pilot4/prelaunch/tol/SHA256SUMS_before.txt), sha256sum -c 1152 OK [measured:
  sha256sum of the list; SP/work/pilot4/prelaunch/smoke/pin/sha256sums_check_end.txt].
SP = the session scratchpad, as in pilot 3's text; RL nnnn = line nnnn of SP/work/run_log.txt. [measured: source] =
read or computed from the named file; [measured as recorded: source] = a number an earlier stage measured and
recorded there, not re-run here; [inferred: basis] = reasoning; rules carry no label. Branch names are written
without their agent prefix [P3 L18-19]. Numbers the first draft measured itself ("this draft" in the text below)
are in SP/work/pilot4/build/draft/out/ (scripts in draft/scripts/; nothing trained: deterministic rollouts and
offline arithmetic on CPU). Numbers the text review measured are in SP/work/pilot4/build/review_text/out/ (cited as
review_text/out/<file>); the fix stage's runs are in SP/work/pilot4/build/fix/logs/ (cited as fix/logs/<file>).
Numbers the postfix verifications measured are in SP/work/pilot4/postfix/text/out/ and SP/work/pilot4/postfix/code/
logs/ (cited as postfix/text/out/<file> and postfix/code/logs/<file>); the postfix fix stage's runs are in
SP/work/pilot4/postfix/fix/logs/ (cited as postfix/fix/logs/<file>). The pilot-4 tools' own stages (build, reviews,
the tools' fix stage) are in SP/work/pilot4/tools/out/. The pre-launch lanes' measurements are in
SP/work/pilot4/prelaunch/setup/, tol/ (the tolerance stage) and smoke/ (cited as setup/<file>, tol/<file> and
smoke/<file>); the arithmetic this text did on their records is in SP/work/pilot4/prelaunch/text/
(scripts/extract_facts.py, data/facts_text.json; cited as text/<file>).

"""

DECISIONS = """\
Decisions taken (2026-11-06; items Q1-Q27 as draft v3 listed them, each with the maintainer's choice; the run log's
maintainer decisions keep their parenthesized numbers) [measured: RL 1093, 05:27 UTC, "MAINTAINER DECISIONS
(AskUserQuestion, on draft v3 Q1-Q27)"; RL 1096, 11:21 UTC, "MAINTAINER DECISIONS (AskUserQuestion)"]. The
alternatives v3 listed beside each proposal were not chosen and are not carried here (v3 keeps them).
  (31) Straightness rule, "Accept as proposed (Recommended)": Q1-Q3, Q9, Q19, Q20, Q22 and Q25.
  (32) Judging and outcome, "Accept as proposed (Recommended)": Q8, Q10, Q21, Q23 and Q27.
  (33) Running, "Accept as proposed (Recommended)": Q4-Q7, Q11 and Q12.
  (34) Code defaults and next step, "Accept, build tools next (Recommended)": Q13-Q18, Q24 and Q26; next the
       pilot-4 judge, scorer and straightness tool on CPU, reviewed; the QUICK_TEST smoke, the freeze, the push and
       the launch each wait for the maintainer.
  (35) The turned-twin tolerance, "Loosen to 1e-4 deg (Recommended)": the tools' fix stage's C6, alternative (a):
       pilot4_straight.TWIN_HEADING_TOL_DEG 1e-4, the scorer's constants and tool sha256 following, the three
       baselines re-run, the tools re-pinned (done by the tolerance stage, "Pre-launch checks"; the Part B test in
       the TOOL PLAN reads 1e-4).
  (36) "Smoke, then final text (Recommended)": the QUICK_TEST notebook smoke on CPU with the pinned tools (it trains
       a throwaway model; approved), the judge, score and resume checks on its output, then the 18 text corrections
       and the tolerance applied to make this text; the push and the Colab launch still wait.
  Q1. The STRAIGHT bars, (31): (a) |median| <= 5 deg and p95 <= 15 deg; (b) p95 D <= 11.5 deg, with Q20's guard;
     (c) median r <= 5 deg and p95 r <= 11.5 deg ("STRAIGHT").
  Q2. The weave measure, (31): D, the deviation of the 1-s mean heading from the commanded heading; R report only.
  Q3. The recovery design, (31): the mid-walk nudge after the prey pass, +-15 deg at state 650, the prey turned with
     the walker (Q19), read over states 900-1000; the turn at reset report only; episodes that end before the nudge
     are left out (the pass rule: Q19).
  Q4. The parent-file route, (33): Colab reads each seed's pair from Drive by path (pilot3_walk_s<seed>/
     03_locomotion/models/), the run records the sha256 it loaded and the judge refuses any other; no branch change.
  Q5. The Colab path, (33): the notebook's manual cell (MANUAL_NODE "locomotion", MANUAL_LOAD_MODE
     "initialize_next_stage", MANUAL_LOAD_PATH the 8M stem, MANUAL_VECNORM_PATH explicit, MANUAL_TIMESTEPS None),
     pilot 3's route.
  Q6. Whether s42 stays, (33): it stays (the scored set fixed before results; the limp is a report-only readout).
  Q7. The cadence, (33): eval_freq 65,536 and checkpoint_freq 250,000, as committed in the TOML; the judged
     checkpoints 0.5 / 1.0 / 1.5M (trajectory) and 2M (deciding).
  Q8. The judge env, (32): the pilot-4 env only; pilot 3's env report only.
  Q9. The recovery test's place, (31): part of STRAIGHT.
  Q10. The straightness look, (32): veto power (a "not straight" makes CURVED WALKS).
  Q11. The resume rule (D11), (33): RESUME from any intact periodic pair from 250,000 on.
  Q12. The collapse reference, (33): 1064.0 (floor 478.8), measured with the price on.
  Q13. The command table's switch_interval_s 4.0 and switch_jitter_s 0.0 (dataclass defaults, in the task digest),
      (34): kept as committed; the table is not rewritten.
  Q14. The behaviour recipes inherit nothing of pilot 4, (34): as built (recorded for completeness).
  Q15. The species catalog shows 2M on this pilot-only branch, (34): accepted.
  Q16. The four known failing tests, (34): unedited; on the branch's test runs deselected by id, or run and named by
      id with their cause (pilot 3's convention [P3 L44, L977]); the stale name of
      test_every_committed_stage_is_command_mode_none_in_phase_c (test_sb3_notebook_pins.py) stays, its docstring
      stating the pilot-4 exception.
  Q17. The [ppo] schedules, (34): production's schedule as committed (learning rate 5e-5 -> 1e-5 over 2M, entropy
      0.005 decaying toward 0.001 over 4M); no [ppo] change.
  Q18. The approvals, (34): the order as proposed; the approvals themselves are still open: this text once frozen
      (its sha256), the fourteen local commits (seven built, four review fixes, three postfix fixes), the OK to push
      and the launch ("Status", "Pre-launch checks").
  Q19. (c)'s prey and counting rule, (31): the prey turned with the walker at the nudge; only nudges after the
      episode's prey pass count; NOT READ below 30 counting episodes.
  Q20. The zig-zag guard, (31): p95 W0 <= 18.5 deg.
  Q21. The straightness look's scope, (32): every seed that is WALKS, video-agreed and passes (a) and (b).
  Q22. The episodes (a) and (b) read, (31): every full-length panel episode whatever its label.
  Q23. The pilot outcome, (32): pilot 3's 2-of-3 rule over the exact seed outcomes, with the count of seeds passing
      (a), (b) and the look printed beside it, report only.
  Q24. The entry-baseline hold, (34): hold unless every prepared parent reads WALKS in the pilot-4 env (each did:
      "Pre-launch checks").
  Q25. The movement condition and the read window, (31): a net horizontal pelvis displacement of at least 0.35 m
      between states 650 and 1000; the single read window m_950. The nudge mechanics are fixed in (c), not a choice.
  Q26. The training diagnostics, (34): no change; the judged readouts (STRAIGHT, the price per episode) carry the
      price and the heading error.
  Q27. The straightness look's criterion, (32): path and heading, so that a visible weave, zig-zag or crab off the
      commanded line reads "not straight".

"""

PRE_HEAD_OLD = ("Pre-launch checks (none trains a pilot-4 step; each to be filled from measured evidence, or written as "
                "not run)")
PRE_HEAD_NEW = """\
Pre-launch checks (none trains a pilot-4 step; each filled from measured evidence, or written as not run yet; the
lanes' folders are cited as setup/, tol/ and smoke/ ("Status"), each with a CMDLOG.txt of its commands and their exit
statuses)"""

PRE_1 = """\
    Done at the head but for the push (the setup lane, 2026-11-06: a clean --shared clone of the read-only
    SP/work/pilot4/tools/tree; the CPU venv of "Known limits"; nothing trained) [measured: setup/logs/, setup/data/]:
    - HEAD 0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8 on pilot-walk-r10-p4, 14 commits in 31d02e42..HEAD; git status
      --porcelain --ignored --untracked-files=all empty before and after every check (status_checks_start.txt,
      status_checks_end.txt).
    - notebook_cells.py: "all notebook code cells parse"; plant_contract --check: "Plant manifest is current";
      species_catalog --check: "Species catalog is current"; digest_snapshot --block-optional-backends --check:
      "649 lines, 0 errors", "Digest snapshot is current"; a digest written outside the tree is byte-identical to the
      committed golden (notebook_cells.txt, plant_contract_check.txt, species_catalog_check.txt,
      digest_check_0e6a57b.txt, digest_write_0e6a57b.txt).
    - Against 31d02e42's golden exactly 8 lines moved (8 removed, 8 added), the 8 listed under "Code"; the diff is
      byte-identical to the postfix code verification's at 7700f28 (setup/logs/
      golden_diff_31d02e42_vs_generated_0e6a57b.txt against postfix/code/logs/
      golden_diff_31d02e42_vs_generated_7700f28.txt) [measured: grep -c and cmp, text/CMDLOG.txt].
    - The build stage's 57 target files (the whole T. rex suite, 8 files, and 49 shared files; setup/data/
      targets_final.txt) under its guarded runner (build/code/scripts/guarded_pytest.py: "blocked 34 entry points")
      with its 219 deselected training-test ids (build/code/logs/deselect_final2.txt): 2713 passed, 4 failed, 4
      xfailed, 393 deselected, 0 guard hits (pt_p0.txt to pt_p2.txt), the model run's counts at 0e6a57b
      (postfix/fix/logs/pt_full_f0.txt to f2.txt). The 4 failures pin main's recipe digests, which the pilot-only
      locomotion.toml moves (expected fde660ac... and 49ef875a..., found fbc94bef... and 49fa0f8e...; "Code"):
        environments/shared/tests/test_critic_phase.py::
          test_the_tyrannosaurus_rex_recipe_digests_are_main_s[locomotion-PPO] and [locomotion-SAC],
          test_the_recipe_digest_records_a_warm_start_log_std_when_a_stage_declares_one;
        environments/shared/tests/test_curriculum_warm_start_log_std.py::
          TestSchema::test_the_recipe_digest_covers_it.
      The 4 xfails are the strict physics-r10 calibration targets of environments/tyrannosaurus_rex/tests/
      test_com_stance_quality.py::TestComStanceQuality: test_target_turn_from_spawn, test_target_still_from_3s,
      test_target_settled_by_2s, test_target_foot_slip_after_landing.
    - Not re-run here: ruff, mypy and CI's wheel and baseline steps (ruff and mypy passed at 0e6a57b, "Code").
    Not run yet: the pushed head equals the local head. The branch is not pushed (Q18); after the push git
    ls-remote must show the remote branch at 0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8."""

PRE_2 = """\
    Done at the head on all three pairs (the setup lane: the pinned nb_pilot_runner4.py 33fd52cc... through iso.py
    5da55763... with --stop-at-learn, in a second clean clone, on byte copies of the 8M pairs, their sha256 equal to
    the table under "Parent and entry" before and after; the notebook's own cells, Run all stopped by the guard at
    cell 20, then the manual cell 22; learn() stopped before any update, the training primitives guarded, 0 guard
    hits; nothing trained) [measured: setup/data/facts_head.json and compare_head_vs_7700f28.json; setup/runs/
    s<seed>/setup_check.json, judge_bound_keys.json and stage/]. Per seed 42 / 52 / 60:
    - Runner exit 0; learn() called once (PPO, total_timesteps 2,000,000, reset_num_timesteps true); every entry line
      present ("Quick test mode: False", "Timesteps: 2,000,000", "VecNormalize loaded from:
      .../stage2_vecnormalize_8000000_steps.pkl", the seed's own), no forbidden and no Warm start line; each pair
      accepted under initialize_next_stage.
    - Of 13 tensors only mlp_extractor.policy_net.0.weight and value_net.0.weight changed, 1,536 entries each
      (columns 60-62), now 0.0; columns 0-59 bit-equal; log_std bit-equal to the parent (mean -1.9381 / -1.9425 /
      -1.9591); Adam moments unchanged, step counts 742,878 / 744,057 / 742,882; the optimizer holds 1.002304e-05.
    - VecNormalize: obs_rms[0:60] bit-equal to the parent's; obs_rms[60:63] mean 0 and var 1 (the parent's var
      3.12e-12); ret_rms reset (mean 0, var 1, count 1e-4).
    - At learn(): ent_coef 0.005, clip 0.2, the learning-rate schedule 5e-5 -> 1e-5, n_envs 4, n_steps 4096, batch
      256, 10 epochs, gamma 0.995, gae_lambda 0.95, target_kl 0.05; on all 4 envs forward weight 2.0, the price 0.2 /
      0.5, command_mode heading, prey [0, 0]. In pilot 3's exact [env], over 1,000 steps from reset 3042, the start
      actions equal pilot 3's 8M walker's (action, mean and value delta 0.0, no nonzero command input).
    - stage_config.json: git_commit 0e6a57bb..., task_sha256 914a1cae... and hyperparameters_sha256 fbc94bef...,
      each equal to the tree's own derivation; load_mode initialize_next_stage, the seed's parent sha256,
      parent_task_sha256 782d95b6...; no critic_phase_dropped and no warm_start_log_std record. The lineage, read
      from the JSON data member of a zip SB3's save() wrote at learn(): initialize_next_stage, parent stage 2,
      zeroed_command_parameters.
    - Against the 7700f28 re-run (postfix/code/logs/setup_summary.json): of 146 leaves per seed only 3 differ, none
      a number (two module paths and git_commit).
    - The pinned judge's own check functions (pilot4_judge.py 648dd557...) accept each recorded stage_config.json,
      plant_identity.json and lineage at the pilot budget 2,000,000 (binding problems: none); across the seeds the
      judge-bound keys differ only in run.seed and run.parent_checkpoint_sha256."""

PRE_3 = """\
    Done, the notebook smoke (decision (36); the smoke lane, 2026-11-06; this local CPU machine, 4 CPUs, not Colab;
    the CPU venv of "Known limits"; MUJOCO_GL=disable, OMP_NUM_THREADS=1, nice -n 10; a fresh --shared clone at
    0e6a57b, clean before and after; read-only byte copies of s42's 8M pair; the pinned runner nb_pilot_runner4.py
    33fd52cc...; QUICK_TEST, RUN_ID pilot4_walk_s42 under ppo_quick_test/) [measured: smoke/nb/params_s42.json,
    nb/launch_s42/record.json and runner.stdout.log, out/smoke_evidence_launch_s42.json, out/timing.json,
    runs/launch/pilot4_walk_s42/]:
    - 12:27:59-12:34:52 UTC, exit 0; cells setup 3, config 6, resume 18, chain 20, manual 22, disconnect 32; Run all
      stopped at the guard in cell 20, then the manual cell 22 alone, then cell 32.
    - The cadence override, in memory, on stage 2: eval_freq and checkpoint_freq 65,536 / 250,000 -> 32,768 /
      32,768, eval_episodes 10 and max_checkpoints 0 unchanged; hyperparameters_sha256 fbc94bef... before and after;
      stage_config.json's curriculum block reads 32,768 (the proof it took).
    - The entry lines (stdout line numbers): 3 "Running locally; using this checkout. REPO_REF selects only the Colab
      checkout."; 15 "Quick test mode: True"; 16 "Behavior recipe: walk"; 17 "In-run K3 panel: off"; 18 "Critic
      phase: off"; 23 "Run directory: .../ppo_quick_test/pilot4_walk_s42 (new run: RUN_ID names no existing run)";
      85 "Node locomotion (stage 2): Locomotion (ppo)"; 87 "Timesteps: 50,000"; 89 "VecNormalize loaded from:
      .../pilot3_walk_s42/03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl"; 91 "Eval num_timesteps=32768,
      episode_reward=3039.26 +/- 33.09"; 96 the same at 65536, 3046.73 +/- 45.36. No "Warm start" line on stdout or
      stderr and no critic-phase drop line. The manual node's gate verdict PASS (reward_and_length/v1; recorded, not
      a criterion) and "Result bundle not rewritten for the manual node: ..." are printed.
    - learn() trained 65,536 steps in 170.1 s, 385.2 steps/s (SB3's time/fps 384), the in-training evaluations
      included.
    - stage_config.json equals the setup lane's 2,000,000-step entry on every key the judge binds but three:
      run.timesteps 50,000 (QUICK_TEST), curriculum eval_freq and checkpoint_freq 32,768 (the override);
      plant_identity.json physics r10 (6a47a0eb...), policy interface r15 (ffbf6631...), observation_dim 63.
    - The lineage: in the zip's JSON "data" member, key mesozoic_task_lineage (mode initialize_next_stage,
      parent_task_sha256 782d95b6..., child_task_sha256 914a1cae..., parent stage 2, zeroed_command_parameters
      [mlp_extractor.policy_net.0.weight, mlp_extractor.value_net.0.weight]), in every zip: stage2_32768_steps,
      stage2_65536_steps, stage2_final, best_model and robust_best_model.
    - TensorBoard events under the child's own 03_locomotion/tensorboard/PPO_1 (241 scalar tags); the ledger
      03_locomotion/eval_episodes.csv, 192 columns, 10 rows per evaluation, every column pilot4_habit.py reads.
      Report only: train/std 0.1467-0.1469; value_loss 0.118 -> 0.0003; explained variance -0.073 -> 0.059.
    - Not checkable locally, so read at the Colab launch: the "Repository ref: ...; commit: ..." line (only Colab's
      branch of cell 3 prints it; locally stage_config.json's git_commit and provenance.json's repository_commit
      0e6a57bb..., repository_dirty false, show the commit), the /tmp/tb_buffer branch, and real CUDA, cuDNN and
      driver values (null on CPU). The final replay was not rendered ("Video recording failed." under
      MUJOCO_GL=disable; neither EGL nor OSMesa works on this machine), so its reset seed is not checked here.
    Done, the RESUME test, with one departure from pilot 3's procedure for the maintainer to confirm [measured:
    smoke/scripts/runner_snapshot.py, snap/snapshot.json, nb/resume/record.json and runner.stdout.log,
    nb/resume_stage_config_before.json and _after.json, out/compare_resume.json, runs/resume/pilot4_walk_s42/]:
    pilot 3 resumed a second launch stopped after its 32,768 pair [P3 L1105-1115]; here the RESUME started from the
    smoke's own 32,768 pair, because decision (36) allows only the 65,536-step smoke and its RESUME: a wrapper that
    loads the pinned runner unchanged copied the run folder inside SB3's CheckpointCallback._on_step right after
    that pair was saved (where the runner's --stop-after-pair exits), a copy with no 65,536 or final pair whose
    32,768 pair is byte-equal to the launch's, and the RESUME ran on it under the same RUN_ID.
    - 12:35:28-12:38:56 UTC, exit 0, the notebook's RESUME cell (RESUME_STAGE 'locomotion'): "Newest intact periodic
      checkpoint: .../stage2_32768_steps.zip", "Checkpoint steps: 32,768 of 50,000", "remaining budget: 17,232",
      "Timesteps: 17,232", no Warm start line; it trained 32,768 -> 65,536 at 362 steps/s.
    - stage_config.json changed only in run.resume_load_path, run.resume_checkpoint_sha256 (the 32,768 zip's,
      4ab2de8b...), run.timesteps 50,000 -> 17,232 and run.duration_seconds; it kept load_mode
      initialize_next_stage, the parent sha256 41fc0b0c..., run.parent_task_sha256 782d95b6..., the seed, n_envs,
      hyperparameters_sha256, git_commit, task_sha256 and every curriculum and env key; no warm_start_log_std or
      critic_phase_dropped record appeared.
    - The resumed 65,536 zip (and its final, best and robust best) keeps the lineage record, equal to the launch's.
    - Report only: not bitwise the unbroken run (at 65,536 the policy, optimizer and sidecar differ; Eval 3056.29
      against 3046.73), as in pilot 3 [P3 L933-941].
    Done, the judge's validation steps on copies of the smoke's runs (the pinned pilot4_judge.py 648dd557... through
    iso.py 5da55763..., on a second clean clone at the head, clean before and after; each step with its expected
    exit) [measured: smoke/judge/status.txt, judge_summary.txt, logs/, copies/copies.json, OUT/, OUT_resume/]:
    - Accepted, exit 0, validation only (run budget 50,000; cadence and text anchors recorded as not checked):
      --check-only at 32,768 and at 65,536 (nothing written); judged the launch's 65,536 and 32,768 pairs; the final
      pair through --final-fallback --steps 50000 (the 50,000 pair missing, the final's num_timesteps 65,536 equal to
      the computed uninterrupted end of 50,000 in whole rollouts of 16,384); the resumed run's --check-only and its
      65,536 pair on schedule (resume step 32,768, budget 17,232 + 32,768 = 50,000, shaping inert).
    - Refused, exit 2, nothing written, 17 steps: a parent sha256 not s42's, in the run record and on the command
      line (s52's); a rewritten command table (yaw_gain 1.5 -> 1.0); a drifted env kwarg (forward_vel_weight 2.0 ->
      3.0); a dirty tree; a second judgement of 65,536; the pilot budget without --expect-budget (budget 50,000,
      cadence 32,768); a final zip rewritten to 49,152; git_commit 31d02e42 against the head; run.n_envs 2;
      run.warm_start_log_std -2.0; a run.critic_phase_dropped record; run.parent_task_sha256 914a1cae...; the
      lineage removed from the zip; a folder named pilot3_walk_s42; --final-fallback on the resumed run; a resume
      path naming stage2_final. The lane's own check that no refusal wrote an OUT folder was miswritten once and
      re-run correctly: none did.
    - Not reachable by a smoke: an on-schedule cell at the pilot budget; the judge's unit tests cover it (282 of 282
      at the tolerance stage, among them resumed copies that keep the zip's lineage)."""

PRE_4 = """\
    Done. The tool's baseline mode (the tolerance stage, the entry baseline below) reproduces the entry rows of the
    evidence table (resets 3042-3061) to the table's 0.1 deg in every column of every seed (yaw median and p95
    |yaw|; D p95, median and max; R median and p95; W0 p95), and the recovery table's (c) row is its own (c)
    [measured: text/scripts/extract_facts.py on tol/val_final/baseline_s42.json, _s52, _s60; text/data/
    facts_text.json]. The probe on the smoke's pilot-4 pairs (each judged cell above): pilot 3's labels WALK 40/40
    in every cell (WALKS), every episode 1,000 steps, "L=0.8766 m", physics r10 (6a47a0eb...), policy interface r15
    (ffbf6631...), env_kwargs_drift_vs_recorded {} [measured: smoke/judge/OUT/s42/32768, 65536 and 50000, and
    OUT_resume/s42/65536 (probe.json, binding.json); smoke/judge/logs/A3_judge_65536.log]. The readouts on the
    smoke's validation cell (65,536) [measured: smoke/readouts/]:
    - pilot4_straight.py (870bab61...) seed mode with --validation: 40/40 re-rolled episodes equal probe.json; (a)
      median -27.8, p95 |yaw| 34.3; (b) p95 D 28.1, p95 W0 33.6; (c) 40 counting, median r 24.3, p95 r 42.5; each
      FAIL, STRAIGHT FAIL, near s42's entry baseline below; states 0..650 equal 80/80, the state-651 observation
      beyond 1e-3 only in obs[60:63] on 40/40; psi_cmd 0.0 deg (the info check within 4.4e-16 rad). Refused without
      --validation (a validation-only cell).
    - pilot4_score.py (98fd9e4a...): as is, no smoke cell is scored ("validation-only judge (run budget 50000)");
      with --validation --primary 65536 and that record, WALKS by the rule, the video review pending, PILOT OUTCOME
      PENDING_REVIEW (possible CURVED WALKS or NOT WALKS), marked as validation.
    - pilot4_habit.py (36465e8c...): the ledgers 0 alarms over 2 evaluations; the seed mode with --validation NO
      HABIT (20 of 20 full-horizon trace episodes), refused without it. pilot4_noise.py (8905669b...) on both runs:
      exp(log_std) mean 0.1467 at 32,768 and 0.1469 at 65,536; the forward term 1941.078 and 1944.641 per evaluation
      against pilot 3's own 1943.495 at 8M.
    - Not run: pilot4_video.py (0c904fc4...) stopped at import under MUJOCO_GL=egl and osmesa (no libEGL and no
      libOSMesa on this machine), so the panel videos and the video tool's re-roll check wait for a machine that
      renders (pilot 3's smoke used a private Xvfb with glfw [P3 L1058-1059], which writes outside the lane's
      folders); pilot4_mirror.py, pilot4_mech.py and pilot3_vg.py were not run on the smoke."""

PRE_5 = """\
    Done (the tolerance stage, decision (35): the pinned pilot4_straight.py 870bab61... in baseline mode, each pair
    prepared in memory with an action and value delta of 0.0 against the command-blind parent, in the committed
    pilot-4 env at 0e6a57b, labelled by pilot 3's pinned pilot3_score.py 286ce04a...; every run exit 0; nothing
    trained) [measured: tol/val_final/baseline_s42.json, _s52, _s60; tol/val/status.txt; tol/logs/facts_tol.json]:
    - Labels: s42 WALK 40; s52 WALK 39 and FALL 1 (reset 3050, 90 steps); s60 WALK 40. WALKS on each seed: the
      launch is not held (Q24).
    - (a) median -25.3 / -47.8 / -28.9 deg, p95 |yaw| 31.7 / 54.6 / 47.6 (n 40 / 39 / 40); (b) p95 D 25.2 / 44.8 /
      49.0, p95 W0 30.0 / 49.4 / 55.3; (c) counting 40 / 38 / 40, median r 22.8 / 34.8 / 29.1, p95 r 40.2 / 55.4 /
      52.7, within 11.5 deg 18/40, 2/38, 3/40, within 5 deg 12/40, 0/38, 1/40, share removed 0: each FAIL, and
      STRAIGHT FAIL on all three, as expected.
    - The mechanics on every nudge: states 0..650 equal 80/80, 76/76, 80/80; the state-651 observation beyond 1e-3
      only in obs[60] and obs[62] on 40/40, 38/38, 40/40; a 0-deg nudge reproduces the panel episode 20/20 per seed;
      the turned twin within 2.73e-5 / 2.53e-6 / 3.71e-6 deg, none flagged at 1e-4; actions after state 651 within
      3.90e-6 / 1.03e-6 / 9.54e-7.
    - Every value of the six records equals the tools' fix stage's (SP/work/pilot4/tools/out/fix/val_final/) but the
      tool sha256, the constant, s42's flagged list (3 -> 0) and file paths [measured: tol/logs/
      cmp_tol_records.json]."""

PRE_6 = """\
    Done but for the durable copy (the tolerance stage's final run, 11:51-11:57 UTC, the files unchanged during it;
    the tests' tree tools/tree clean at 0e6a57b; nothing trains in the tests) [measured: tol/tests_final/
    summary.json and rcs.txt; tol/val_final/p3val_s42.json, _s52, _s60; tol/e2e/e2e_summary.json and
    e2e_fix_checks.json; tol/pin/sha256sums_check.txt and before_list_check.txt]:
    - 567 pytest cases in the nine test files, 0 failed, 0 skipped (judge 282, straight 74, habit 65, vg 34, runner
      33, noise 26, mirror 19, mech 19, video 15); test_pilot4_score.py 480/480 through iso.py with the tree (pure
      mode 428/428, 1 skipped).
    - On pilot 3's judged 8M cells: every re-rolled episode (40 per seed) equals probe.json, yaw_change_deg
      reproduced with a difference of 0.0; (a), (b) and (c) FAIL on all three.
    - The end-to-end on stand-in runs built by the real set-up path (learn() stopped): 65 of 65 and 18 of 18 checks.
    - Pinned by SP/work/pilot4/tools/SHA256SUMS.txt (1152 entries), sha256
      0f42f29f45092ad87004519783af5175bcb1b7a7c61dcccd4985b682554cc448; sha256sum -c from the tools folder 1152 OK at
      the tolerance stage and at the smoke's start and end (smoke/pin/sha256sums_check_start.txt and _end.txt); the
      tools' fix stage's list (c2a86878..., 605 entries) still verifies 601, the 4 edited files failing, by design.
      The tools: pilot4_judge.py 648dd557..., pilot4_score.py 98fd9e4a..., pilot4_straight.py 870bab61...,
      pilot4_video.py 0c904fc4..., pilot4_habit.py 36465e8c..., pilot4_noise.py 8905669b..., pilot4_mirror.py
      862b25b9..., pilot4_mech.py 5d9e21e7..., pilot3_vg.py a58d5f4f... and iso.py 5da55763... (byte copies of
      pilot 3's), and the smoke runner nb_pilot_runner4.py 33fd52cc....
    Not run yet: the durable copy outside the session scratchpad [P3 L1401-1403]: the pinned files, byte-identical,
    checked by sha256sum -c against SHA256SUMS.txt. The list names the tolerance stage's evidence as
    ../prelaunch/tol/... and test_pilot4_score.py reads ../prelaunch/tol/val_final by default, so the copy carries
    SP/work/pilot4/prelaunch/tol/ beside the tools folder (or the test is given PILOT4_STRAIGHT_VAL), and sha256sum
    -c runs from the tools folder."""

PRE_7 = """\
    Not run yet. The steps give, for each seed: the parent-file route of Q4 (MANUAL_LOAD_PATH the seed's own 8M stem
    on Drive, MANUAL_VECNORM_PATH its explicit sidecar, the recorded sha256 equal to the table under "Parent and
    entry"); the guard line typed as the chain-loop cell's first line (the smoke's: raise RuntimeError("Walking
    pilot 4: chain loop skipped on purpose. Now run the manual cell (section 7) by itself -- unless you are resuming:
    then stop here.") [measured: smoke/nb/params_s42.json]); REPO_REF at the pushed head; QUICK_TEST False; the
    launch lines and first-hour checks under "Run handling"; the RESUME and _b rules.
  Pending, in this order: the maintainer's approval of this text (its sha256; with it the smoke RESUME's departure
  above) and of the fourteen local commits; the OK to push, then the push with an explicit refspec to the
  same-named remote branch, never a bare git push [P3 L1397-1400], after which git ls-remote must show the remote
  branch at 0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8, and nothing more is pushed to it until every seed is judged
  ("Code"); the durable copy of the pinned tools; the Colab steps, reviewed against this text; the maintainer's
  launch approval."""

CHANGES = """\

Changes from v3 (against draft v3, SP/work/pilot4/postfix/fix/criteria_pilot4_draft_v3.txt, sha256 1c52644d...c19d;
v3 is kept unchanged; this text is made from it by SP/work/pilot4/prelaunch/text/scripts/make_final.py, which checks
every replaced text before replacing it)
  - The tools' fix stage's 18 text corrections (SP/work/pilot4/tools/out/fix/text_corrections.txt, sha256
    78f17142...): C1-C5, C7, C8 (inserted), C9a, C9b and C10-C16 with their NEW text verbatim, and C6 in its
    alternative (a) form, decision (35): the turned-twin tolerance 1e-4 deg, with the tolerance stage's measurement.
    Every OLD text matched v3; none changes a bar, a seed, a budget or an outcome rule.
  - C5, C9a, C9b, C12a and C12b cite the tolerance stage's records (SP/work/pilot4/prelaunch/tol/val_final/) in
    place of the tools' fix stage's (SP/work/pilot4/tools/out/fix/val_final/), which the re-pinned scorer refuses
    (their tool sha256 is the earlier tool's [measured: tol/logs/scorer_vs_fixstage_records.txt]); every number is
    the same [measured: tol/logs/cmp_tol_records.json; text/data/facts_text.json]; C9a's, C12a's and C5's passages
    are re-wrapped to keep the width.
  - Title and Status: a frozen candidate; the head 0e6a57b, the tools' SHA256SUMS.txt sha256, what stays open; the
    pre-launch lanes' folders; "this draft" names the first draft, as in v3.
  - "Maintainer decisions behind this pilot" points to (31)-(36).
  - "Remaining maintainer decisions" becomes "Decisions taken": (31)-(36) and each of Q1-Q27 with its choice and
    decision number; the alternatives not chosen are not carried (v3 keeps them).
  - Settled wording, no value changed: D7, the [curriculum] block (eval_freq and checkpoint_freq), the budget
    heading, the s42 line, the yaw note and the W0 note of the evidence table, the recovery table's (c) row and the
    baseline item of the pre-launch checks say decided where v3 said proposed.
  - Filled from the pre-launch lanes: "Parent and entry" (the setup check re-run at the head); "Run handling" (the
    smoke printed no Warm start or critic-phase drop line; where the archive keeps the lineage); the course line of
    "Report only" (which resets its numbers cover); every item of "Pre-launch checks" (done, with numbers and
    sources, or not run yet), and the pending list after them.
  - TOOL PLAN's heading records the pin."""


def build():
    v3_lines, lines, report, meta = corrections.stage1()
    if sha_file(P3_TEXT) != P3_SHA256:
        raise SystemExit("pilot 3's text sha256 differs: refusing")
    if sha_file(SUMS) != SUMS_SHA256:
        raise SystemExit("SHA256SUMS.txt sha256 differs: refusing")
    T = Text(lines)

    # header (v3 lines 1-22, untouched by the corrections)
    T.swap("header", None, "Where pilot 3 ended [measured as recorded: RL 1067, 1069, 1070, 1071]",
           "\n".join(v3_lines[0:22]) + "\n", HEADER)
    # decisions behind this pilot: a pointer to (31)-(36)
    T.rep("decisions pointer",
          "  and no outcome here reads that module's flags (its numbers appear only as report-only references).\n",
          "  and no outcome here reads that module's flags (its numbers appear only as report-only references).\n"
          "  (31)-(36), 2026-11-06, on draft v3 and before the freeze: \"Decisions taken\" below [measured: RL 1093,\n"
          "  1096].\n")
    # settled wording
    T.rep("D7",
          "  D7  Budget 2,000,000 (decision (25)); cadence PROPOSED: an evaluation every 65,536 steps, checkpoints every "
          "250,000.",
          "  D7  Budget 2,000,000 (decision (25)); cadence, decision (33): an evaluation every 65,536 steps, checkpoints\n"
          "      every 250,000.")
    T.rep("setup check at the head",
          "  as recorded: postfix/code/logs/setup_summary.json]. Commits 12-14 were not run through it (\"Pre-launch "
          "checks\").",
          "  as recorded: postfix/code/logs/setup_summary.json]. The setup lane re-ran it at the head 0e6a57b (commits\n"
          "  12-14 included) on all three pairs: of 146 leaves per seed only two module paths and git_commit differ\n"
          "  from the 7700f28 record (\"Pre-launch checks\").")
    T.rep("curriculum block",
          "    [curriculum] timesteps 2000000, eval_freq 65536 (PROPOSED), eval_episodes 10, checkpoint_freq 250000\n"
          "      (PROPOSED), max_checkpoints 0, initialize_from_own_checkpoint true, warmup_timesteps 0, ramp_start_value "
          "2.0\n"
          "      (ramp_timesteps 2,000,000 stays, inert), no warm_start_log_std; collapse_peak_floor_reference 1064.0,",
          "    [curriculum] timesteps 2000000, eval_freq 65536 (decision (33)), eval_episodes 10, checkpoint_freq 250000\n"
          "      (decision (33)), max_checkpoints 0, initialize_from_own_checkpoint true, warmup_timesteps 0,\n"
          "      ramp_start_value 2.0 (ramp_timesteps 2,000,000 stays, inert), no warm_start_log_std;\n"
          "      collapse_peak_floor_reference 1064.0,")
    T.rep("budget heading", "Budget and cadence (decision (25); the cadence PROPOSED)",
          "Budget and cadence (decision (25); the cadence decision (33))")
    T.rep("s42 stays", "  below (proposed: it stays).", "  below (decided: it stays, decision (33)).")
    T.rep("smoke entry lines",
          "  drop line [inferred: no warm_start_log_std, no phase input; the exact entry lines are read off the CPU "
          "smoke]; a",
          "  drop line [inferred: no warm_start_log_std, no phase input; measured: the CPU smoke printed neither]; a")
    T.rep("lineage place",
          "  model's lineage attribute (zeroed_command_parameters) is checked by the judge on the judged zip; where the\n"
          "  archive keeps it is read off the notebook smoke's pair before the freeze. Also in the first hour: "
          "provenance.json",
          "  model's lineage attribute (zeroed_command_parameters) is checked by the judge on the judged zip; the archive\n"
          "  keeps it in the zip's JSON \"data\" member, key mesozoic_task_lineage, in every zip the run writes [measured:\n"
          "  the notebook smoke, \"Pre-launch checks\"]. Also in the first hour: provenance.json")
    T.rep("course line",
          "    mean heading, and the end cross-track from the commanded line (entry: +4.7 / -1.5 / +2.3 deg and -0.21 / "
          "-3.92\n"
          "    / -2.81 m median [measured: review_text/out/review_calc.txt P5]).",
          "    mean heading, and the end cross-track from the commanded line (entry, resets 3042-3061: +4.7 / -1.5 / +2.3\n"
          "    deg and -0.21 / -3.92 / -2.81 m median [measured: review_text/out/review_calc.txt P5; the tolerance stage's\n"
          "    records give the same on those resets, and +4.8 / -1.4 / +2.3 deg and +0.39 / -4.06 / -2.97 m over the\n"
          "    whole panel's full-length episodes, the figures the straightness tool reports (TOOL PLAN), text/data/\n"
          "    facts_text.json]).")
    T.rep("yaw note", "which part (a) as proposed leaves out", "which part (a) as decided leaves out")
    T.rep("W0 note", "Against the proposed W0 bar (18.5)", "Against the decided W0 bar (18.5)")
    T.rep("recovery table (c) row",
          "    proposed (c), prey turned with the walker (the fixed mechanics):",
          "    (c) as decided, prey turned with the walker (the fixed mechanics):")
    # re-citations of the corrected passages (C5, C9a, C9b, C12a, C12b): the tolerance stage's records
    T.rep("C9a cite",
          "    removed 0, so (c) fails on all three [measured: pilot4_straight.py baseline mode on every nudge, fix stage,\n"
          "    SP/work/pilot4/tools/out/fix/val_final/baseline_s42.json, _s52, _s60; reproduced by an independent rollout "
          "within\n"
          "    6.1e-8 deg, SP/work/pilot4/tools/out/review_validity/data/cmp_s*.json]. With\n",
          "    removed 0, so (c) fails on all three [measured: pilot4_straight.py baseline mode on every nudge, tolerance\n"
          "    stage, SP/work/pilot4/prelaunch/tol/val_final/baseline_s42.json, _s52, _s60; reproduced by an independent\n"
          "    rollout within 6.1e-8 deg, SP/work/pilot4/tools/out/review_validity/data/cmp_s*.json]. With\n")
    T.rep("C12a cite",
          "    they read median r 21.6 / 34.9 / 28.2 deg (under the fixed mechanics 21.3 / 34.7 / 28.5 [measured: fix "
          "stage,\n"
          "    SP/work/pilot4/tools/out/fix/val_final/baseline_s42.json, _s52, _s60]), share removed -0.11 to +0.05 "
          "(medians per\n"
          "    seed and sign), and the nudge made no walker fall (118 nudged episodes, all counting under the pass rule; "
          "s52's\n"
          "    two left out are the panel fall at reset 3050, 90 steps) [measured: out/recov_rule.txt, "
          "out/recov_summary.txt]. A\n"
          "    turn at the reset\n",
          "    they read median r 21.6 / 34.9 / 28.2 deg (under the fixed mechanics 21.3 / 34.7 / 28.5 [measured: tolerance\n"
          "    stage, SP/work/pilot4/prelaunch/tol/val_final/baseline_s42.json, _s52, _s60]), share removed -0.11 to\n"
          "    +0.05 (medians per seed and sign), and the nudge made no walker fall (118 nudged episodes, all counting\n"
          "    under the pass rule; s52's two left out are the panel fall at reset 3050, 90 steps) [measured:\n"
          "    out/recov_rule.txt, out/recov_summary.txt]. A turn at the reset\n")
    T.rep("C12b cite",
          "    in place, median r 12.9 / 6.0 / 1.4, within 11.5 deg 17 / 30 / 40 of 40 [measured: fix stage,\n"
          "    SP/work/pilot4/tools/out/fix/val_final/baseline_s42.json, _s52, _s60]) by steering",
          "    in place, median r 12.9 / 6.0 / 1.4, within 11.5 deg 17 / 30 / 40 of 40 [measured: tolerance stage,\n"
          "    SP/work/pilot4/prelaunch/tol/val_final/baseline_s42.json, _s52, _s60]) by steering")
    T.rep("C5 cite",
          "      end cross-track, in the record, the printout and on the path plot (entry: +4.8 / -1.4 / +2.3 deg and +0.39 "
          "/\n"
          "      -4.06 / -2.97 m [measured: fix stage, SP/work/pilot4/tools/out/fix/val_final/baseline_s42.json, _s52, "
          "_s60]).\n",
          "      end cross-track, in the record, the printout and on the path plot (entry: +4.8 / -1.4 / +2.3 deg and +0.39 "
          "/\n"
          "      -4.06 / -2.97 m [measured: tolerance stage, SP/work/pilot4/prelaunch/tol/val_final/baseline_s42.json,\n"
          "      _s52, _s60]).\n")
    T.rep("C9b cite",
          "      on every nudge, fix stage, SP/work/pilot4/tools/out/fix/val_final/baseline_s42.json, _s52, _s60; the",
          "      on every nudge, tolerance stage, SP/work/pilot4/prelaunch/tol/val_final/baseline_s42.json, _s52, _s60; the")
    # TOOL PLAN heading
    T.rep("tool plan heading",
          "test pinned in a new SHA256SUMS.txt before launch; none on the branch)",
          "test pinned in a new SHA256SUMS.txt before launch, done: sha256 0f42f29f..., \"Pre-launch checks\"; none on the\n"
          "branch)")
    # decisions section (v3 lines 770-840, untouched by the corrections)
    T.swap("decisions taken", "Remaining maintainer decisions (items Q1-Q27",
           "Evidence for the bars (measured unless marked)", "\n".join(v3_lines[769:840]) + "\n", DECISIONS)
    # pre-launch checks: keep each item (the rule), fill it below
    T.rep("pre-launch heading", PRE_HEAD_OLD, PRE_HEAD_NEW)
    T.rep("pre-launch baseline item wording",
          "    draft's recov.py does) reproduces the entry rows and the recovery baseline at the end of this text (its "
          "proposed\n"
          "    (c) row was computed from recorded un-nudged rollouts, so the tool's run is its first direct measurement "
          "on all\n"
          "    40 episodes per seed); the probe\n",
          "    draft's recov.py does) reproduces the entry rows and the recovery baseline at the end of this text (its (c) "
          "row,\n"
          "    as decided, was computed from recorded un-nudged rollouts, so the tool's run is its first direct "
          "measurement on\n"
          "    all 40 episodes per seed); the probe\n")
    T.insert_after("pre-launch 1",
                   "    targets pass but for the known 4, each id named with its cause (the build stage's run is the "
                   "model).", PRE_1)
    T.insert_after("pre-launch 2",
                   "    move]); the exact stage_config.json and lineage key paths the judge binds are copied from it.",
                   PRE_2)
    T.insert_after("pre-launch 3", "    by code reading); the judge's unit tests include such a resumed copy.", PRE_3)
    T.insert_after("pre-launch 4", "    up to 0.59-0.76 [measured as recorded: the setup check]).", PRE_4)
    T.insert_after("pre-launch 5", "    The same run prints (a), (b) and (c) at entry (expected to fail).", PRE_5)
    T.insert_after("pre-launch 6",
                   "    pinned by sha256 before launch, with a durable copy outside the scratchpad [P3 L1401-1403].",
                   PRE_6)
    T.insert_after("pre-launch 7", "    parent-file route (Q4).", PRE_7)
    # changes from v3
    body = T.t.rstrip("\n")
    T.t = body + "\n" + CHANGES + "\n"
    T.log.append({"edit": "changes from v3", "kind": "append", "new_lines": CHANGES.count("\n")})
    return v3_lines, T, report, meta


def check(v3_lines, text, report):
    out = text.split("\n")
    if out[-1] == "":
        out = out[:-1]
    problems = []
    for i, ln in enumerate(out, 1):
        if len(ln) > WIDTH:
            problems.append(f"line {i} is {len(ln)} chars")
        if ln != ln.rstrip():
            problems.append(f"line {i} has trailing space")
        if any(ord(c) > 126 or (ord(c) < 32 and c != "\t") for c in ln) or "\t" in ln:
            problems.append(f"line {i} has a non-ASCII or control character")
        if "/tmp/" in ln.replace("/tmp/tb_buffer", "") or "/home/" in ln:  # the notebook's Colab buffer is no path here
            problems.append(f"line {i} spells an absolute path")
    # every v3 line not kept must lie in a region this build may change
    allowed = set(range(0, 22)) | set(range(769, 840))
    for r in report:
        a, b = r["lines"]
        allowed |= set(range(a - 1, b))
    edited_v3_lines = {  # v3 line numbers (1-based) of stage 2's edits outside the corrected ranges
        87, 167, 282, 283, 284, 302, 326, 512, 526, 527, 603, 604, 655, 674, 675, 676, 693, 859, 864,
    }
    allowed |= {n - 1 for n in edited_v3_lines}
    sm = difflib.SequenceMatcher(a=v3_lines, b=out, autojunk=False)
    removed = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(range(i1, i2))
    outside = [i + 1 for i in removed if i not in allowed]
    if outside:
        problems.append(f"v3 lines changed outside the allowed regions: {outside}")
    return out, problems, sorted(i + 1 for i in removed)


def main():
    v3_lines, T, report, meta = build()
    out, problems, removed = check(v3_lines, T.t, report)
    if problems:
        raise SystemExit("checks failed, nothing written:\n" + "\n".join(problems))
    os.makedirs(OUT_DIR, exist_ok=True)
    data = ("\n".join(out) + "\n").encode("ascii")
    with open(OUT, "wb") as f:
        f.write(data)
    diff = difflib.unified_diff(v3_lines, out, "criteria_pilot4_draft_v3.txt", "criteria_pilot4.txt", lineterm="",
                                n=1)
    with open(os.path.join(LANE, "data", "diff_v3_to_final.txt"), "w") as f:
        f.write("\n".join(diff) + "\n")
    rep = {"output_sha256": hashlib.sha256(data).hexdigest(), "output_lines": len(out),
           "max_width": max(len(ln) for ln in out), "stage1": meta, "corrections": report, "stage2": T.log,
           "v3_lines_not_kept": removed, "p3_sha256": P3_SHA256, "sha256sums_sha256": SUMS_SHA256}
    with open(os.path.join(LANE, "data", "make_final_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("output_sha256", "output_lines", "max_width")}))
    print("stage 2 edits:", len(T.log))


if __name__ == "__main__":
    main()
