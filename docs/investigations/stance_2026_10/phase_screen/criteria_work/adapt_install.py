"""Adapter lane: install <PS>/criteria.txt and <PS>/SCREEN.json from the drafts (WORK/criteria_v1.txt,
WORK/SCREEN_v1.json), applying the listed edits A1-A23 (each old text must occur exactly once in the draft) and filling
PINS from the files as they stand.  Writes only <PS>/criteria.txt and <PS>/SCREEN.json; prints the edit log.
    cd /tmp && PY -I -B adapt_install.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PS = Path("/home/user/work/phase_screen")
WORK = PS / "criteria_work"
DRAFT, DRAFT_SHA = WORK / "criteria_v1.txt", "d45c6df7c1e93db7cd5b635321e02f52445c4ac954fc99cfafb8ea6517012c67"
SDRAFT, SDRAFT_SHA = WORK / "SCREEN_v1.json", "5cb031a9951b58bb472f81d4ad286bbbf4550a52fff9933f784b13dc46692c7f"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert sha(DRAFT) == DRAFT_SHA and sha(SDRAFT) == SDRAFT_SHA, "a draft changed"
text = DRAFT.read_text(encoding="utf-8")
EDITS = []


def edit(tag, why, old, new):
    EDITS.append((tag, why, old, new))


edit("A1", "title: the installed version, not the draft",
     "STATE, CPU, SCRATCH.  CRITERIA v1 -- DRAFT FOR INDEPENDENT REVIEW.  NOT APPROVED.  NOTHING TRAINS UNTIL <PS>/APPROVED\n"
     "EXISTS.\n",
     "STATE, CPU, SCRATCH.  CRITERIA v1 AS INSTALLED BY THE SCORER ADAPTER -- FOR INDEPENDENT REVIEW.  NOT APPROVED.\n"
     "NOTHING TRAINS UNTIL <PS>/APPROVED EXISTS.\n")
edit("A2", "STATUS: the install, the draft's sha256 and where the edits are listed",
     "bytes and sha256 into RUNLOG before launch: P1 as sharpened by M1). It is not edited after the first optimizer step.\n",
     "bytes and sha256 into RUNLOG before launch: P1 as sharpened by M1). It is not edited after the first optimizer step.\n"
     "INSTALLED 2026-10-31 by the scorer adapter as <PS>/criteria.txt from the draft WORK/criteria_v1.txt (sha256\n"
     "d45c6df7c1e93db7cd5b635321e02f52445c4ac954fc99cfafb8ea6517012c67), with SCREEN_v1.json as <PS>/SCREEN.json (the\n"
     "queue unchanged). Every edit (A1-A26), each reconciling the draft with the harness as built, is listed with its\n"
     "old and new text and its reason in WORK/ADAPT_NOTES.txt (EDITS) and WORK/adapt_logs/install.out; none changes the\n"
     "question, the arms' contract, the primary, B_S, the gate, the test, a line's text or a secondary's definition; the\n"
     "built form of the manipulation check (A6-A9), SEC-TAIL2's source (A15) and the report-only phase swing (A17) are\n"
     "stated as built for the reviewers to see. PINS are filled from the files as they stand at the install.\n")
edit("A3", "the built names of the phase (the draft left them to the adapter)",
     "- THE PHASE, k and the features (the arm's definition; the adapter fills the built names):\n",
     "- THE PHASE, k and the features (the arm's definition; the built names: bs_common.PHASE_SLOTS (60, 61, 62),\n"
     "  PHASE_FEATURES (\"ramp\", \"settled\", \"discount\"), SETTLE_STEPS 200, PHASE_GAMMA 0.98, phase_features(k)):\n")
edit("A4", "the feature paragraph re-checked against the built file: unchanged, so the conditional instruction goes",
     "  Features (the builder's definition as read from <PS>/tools/bs_common.py lines 111-141 at 13:2x UTC; the adapter\n"
     "    re-checks it against the pinned file): ramp",
     "  Features (the builder's definition, <PS>/tools/bs_common.py lines 111-141; re-checked by the scorer adapter\n"
     "    against the file pinned below: unchanged): ramp")
edit("A4b", "the draft's instruction to the adapter resolved",
     "    (1, 1, 1) the settled one. [If the built feature set changes, the adapter replaces this paragraph with the built\n"
     "    definition and lists the change; the CONTRACT below does not change.]\n",
     "    (1, 1, 1) the settled one. [Re-checked by the adapter: the built feature set is this one, so the paragraph\n"
     "    stands; the CONTRACT below does not change.]\n")
edit("A5", "ARM_CHANGE cited by line, as pinned",
     "As built (bs_common.ARM_CHANGE[\"critic_phase\"], read at 13:2x UTC):",
     "As built (bs_common.ARM_CHANGE[\"critic_phase\"], lines\n      155-169, re-checked by the adapter):")
edit("A6", "C2 as CHECK checks it (bs_hooks _phase_problems: zero-init rollouts need zero phase columns and a zero "
     "response on every critic path); V1's record named",
     "        round-off of reordered sums, |dV| <= 1e-6, the precedent's own allclose atol 1e-6), measured by V1; on the\n"
     "        first rollout the buffer values change by exactly 0.0 when the phase is perturbed (zero weights times any\n"
     "        finite phase) [CHECK, rollout 1].\n",
     "        round-off of reordered sums, |dV| <= 1e-6, the precedent's own allclose atol 1e-6), measured by V1 (bitwise:\n"
     "        <PS>/validate/logs/v1_independent.out); on the rollout before the first update (rollout 1 of a\n"
     "        continuation) the critic's phase columns are exactly 0 and its output through predict_values, forward,\n"
     "        evaluate_actions and the mirrored observations changes by exactly 0.0 under the five phase perturbations of\n"
     "        THE MANIPULATION CHECK (zero weights times any finite phase) [CHECK, rollout 1].\n")
edit("A7", "C3: 175 evidence rollouts, not 176 (one per update row, read_bs.evidence_check; BS2's count); the built "
     "test (columns nonzero, min > 0 on each of the four value paths)",
     "      C3 (the critic learns the phase) from the second rollout on, the critic's output changes when the phase\n"
     "        features are perturbed (min over the checked rows > 0) [CHECK, rollouts 2-176]; after one optimizer step\n"
     "        the critic's phase weights have moved (V5, on a discarded copy).\n",
     "      C3 (the critic learns the phase) from the second rollout on (every rollout after at least one update), the\n"
     "        critic's phase columns are nonzero and its output changes when the phase features are perturbed, min > 0\n"
     "        on each of the four value paths [CHECK, rollouts 2-175]; after one optimizer step the critic's phase\n"
     "        weights have moved (V5, on a discarded copy).\n")
edit("A8", "C5 as built: the per-rollout recount covers the buffer rows (exact after VecNormalize's formula); the k = 1000 "
     "terminal observation is not in the buffer, so it is covered by the unit test on the real env, not by CHECK",
     "      C5 (the phase is the env's own count) the phase features stored with every buffer row equal phi(k) for k\n"
     "        recounted independently (from the env's _step_count read at each rollout start and the buffer's\n"
     "        episode_starts; to float32), and the bootstrapped time-limit terminal observation carries phi(1000) [CHECK,\n"
     "        every rollout; V2].\n",
     "      C5 (the phase is the env's own count) the phase features stored with every buffer row equal phi(k) for k\n"
     "        recounted independently (from the envs' _step_count read at each rollout start and the buffer's\n"
     "        episode_starts), after VecNormalize's own formula, exactly (max error 0.0), and the recount reaches the\n"
     "        envs' counters at the rollout's end [CHECK, every rollout; V2]; the bootstrapped time-limit terminal\n"
     "        observation carries phi(1000) [unit test on the real env, <PS>/tools/tests/test_bs_hooks.py lines 474 and\n"
     "        504-506; not re-checked per rollout: SB3 does not store it in the buffer, it bootstraps predict_values on\n"
     "        it (SB3 on_policy_algorithm.py lines 236-245) after VecNormalize normalises it (vec_normalize.py lines\n"
     "        203-204)].\n")
edit("A9", "THE MANIPULATION CHECK as built: 175 rollouts; the five perturbations and five actor paths of bs_hooks; "
     "the structural checks; run_meta's phase records; control's checks as built",
     "- CHECK (read_bs.py --check, run by the runner as each continuation exits) passes only if, on every one of the 176\n"
     "  rollouts (arm_evidence.jsonl: one line per rollout end, its summary equal to run_meta's), the evidence finds the\n"
     "  arm's signature and no other's:\n"
     "    critic_phase: C1 (the actor's mean and log-probabilities change by exactly 0.0 when the phase features are set\n"
     "      three ways (all 0; all 1; the row's own features with the ramp replaced by 0.5) on the buffer's first 256\n"
     "      rows, through forward, evaluate_actions, get_distribution and the mirrored observations); C2 on rollout 1\n"
     "      (the buffer values equal the critic's with the phase features zeroed, change exactly 0.0); C3 on rollouts\n"
     "      2-176 (the critic's output changes with the phase perturbation, min > 0); C4 (the critic's output changes\n"
     "      with the normalised observation 6, min > 0); C5 (the buffer's phase features equal the independent recount on\n"
     "      every row; every bootstrapped truncation used phi(1000));\n"
     "    control: the critic's output changes with observation 6 on every rollout (min > 0); classes stock; no phase\n"
     "      feature in the buffer or the observation space;\n",
     "- CHECK (read_bs.py --check, run by the runner as each continuation exits) passes only if, on every one of the 175\n"
     "  rollouts that end before an update (arm_evidence.jsonl: one line per rollout end at S + 8,192 k, k = 1 ... 175, its\n"
     "  summary equal to run_meta's; the 176th rollout, ending at E, has no update after it, as in BS2), bs_hooks.\n"
     "  evidence_problems finds the arm's signature and no other's (as built: bs_hooks.py lines 948-1047 with\n"
     "  PHASE_PERTURB lines 70-71; read_bs.py phase_meta_problems lines 442-476):\n"
     "    critic_phase: C1 (the actor's deterministic mean, its deterministic actions and their log-probabilities through\n"
     "      forward, the log-probabilities of the buffer's actions and the entropy through evaluate_actions, and the\n"
     "      mirrored observations' mean change by exactly 0.0 when the normalised phase slots 60-62 are set five ways\n"
     "      (all 0; all 1; +1; all 3; 1 - x) on the buffer's first 256 rows; the actor's phase columns unchanged and\n"
     "      their Adam moments exactly 0); C2 on the rollout before the first update (the critic's phase columns exactly\n"
     "      0 and its output through predict_values, forward, evaluate_actions and the mirrored observations changes by\n"
     "      exactly 0.0 under the same five perturbations); C3 on rollouts 2-175 (the critic's phase columns nonzero and\n"
     "      its output changes under the perturbations, min > 0 on each of the four paths); C4 (the critic's output\n"
     "      changes with the normalised observation 6, min > 0); C5 (the buffer's phase slots equal VecNormalize's\n"
     "      formula on the independent recount on every row, max error exactly 0.0; the recount consistent with the\n"
     "      envs' counters at the rollout's start and end); C6 (the statistics' slots frozen at mean 0 / var 1 on every\n"
     "      rollout); C7 in part (MirrorKLPPO's map sends the slots to themselves with sign +1, and its tensors once\n"
     "      built; every training env carries the phase class); and on run_meta: the load transform verified (the\n"
     "      parameters changed at the load, the Adam state not), the phase definition, the probe feeding the settled\n"
     "      bank's phase (k >= 200), exactly 1 rollout before the first update and 174 after;\n"
     "    control: the critic's output changes with observation 6 on every rollout (min > 0); classes stock; no phase\n"
     "      evidence on any rollout and no phase record in run_meta (the loaded state is the start pair's: digest_load =\n"
     "      digest_start); the actor responds to the command slots on every rollout (min > 0: no mask); that its slots\n"
     "      carry the zero command is shown by V2 (its environment side equals the phase arm's outside the slots);\n")
edit("A10", "the V2 pairing as measured: equal outside the phase slots (the phase arm's slots carry the phase by "
     "construction), values and advantages equal bitwise",
     "  classes before and after); and its environment side (observations, actions, rewards, episode starts) equals the\n"
     "  phase arm's on the first rollout bitwise, with the buffer values equal to the precision C2 states (the phase arm's\n"
     "  critic is the parent's at load).\n",
     "  classes before and after); and its environment side (observations outside the phase slots 60-62, which carry the\n"
     "  phase in the phase arm; actions; rewards; episode starts) equals the phase arm's bitwise on every rollout of V2\n"
     "  (31 of 31 at seeds 9401 and 9411; the whole-observation digest differs on 31 of 31, as it must), with the buffer\n"
     "  values and advantages equal bitwise (the phase arm's critic is the parent's at load, C2) [measured, <PS>/validate/\n"
     "  logs/v2_check.out lines 26-28 and 58-60].\n")
edit("A11", "the critic form the scorer refuses, as built (read_bs.loop_markers_one / full_read)",
     "- The scorer prints each COUNTED continuation's evidence summary and refuses (input error) markers whose critic form is\n"
     "  not the arm's, or whose loop row at E records a num_timesteps other than E.\n",
     "- The scorer prints each COUNTED continuation's evidence summary and refuses (input error) markers whose critic form is\n"
     "  not the arm's (as built: control: critic_form \"plain\" on every loop row, the effective critic = the raw one, no\n"
     "  phase record, the probe not phase-fed (probe_phase.installed false); critic_phase: \"phase+raw+settled\" on every\n"
     "  loop row, a phase record on each, the raw / effective / settled slopes on every state set, the probe phase-fed;\n"
     "  read_bs.py lines 1037, 1058-1079, 1217), or whose loop row at E records a num_timesteps other than E.\n")
edit("A12", "the queue's file installed",
     "- THE QUEUE (<PS>/SCREEN.json, pinned; draft WORK/SCREEN_v1.json):",
     "- THE QUEUE (<PS>/SCREEN.json, pinned; installed from WORK/SCREEN_v1.json, the queue unchanged):\n ")
edit("A13", "the scorer named; what it reads after all are decided",
     "the scorer\n  (the adapter's, this text's reading). Launch:",
     "score_bs.py (the\n  scorer, the adapter's: this text's reading; once all are decided it also reads each COUNTED run's\n"
     "  policy_probe.npz for SEC-TAIL2). Launch:")
edit("A14", "the launch floor as built (8 continuations, not BS2's 9); today's disk",
     "- Disk: free space on / was 2.0 GB at 12:57 UTC today [measured, df -h /, WORK/CMDLOG.txt]: BELOW the launch rule.\n",
     "- Disk: free space on / was 2.0 GB at 12:57 UTC today [measured, df -h /, WORK/CMDLOG.txt] and 1.8 GB at 13:57 UTC\n"
     "  [measured, df -h /, WORK/ADAPT_NOTES.txt]: BELOW the launch rule.\n")
edit("A14b", "the launch floor as built",
     "the runner's own launch floor (BS2: 2,082,471,936 bytes; as built,\n  bs_common.LAUNCH_DISK_BYTES) is met;",
     "the runner's own launch floor (as built, bs_common.LAUNCH_DISK_BYTES =\n"
     "  2,030,043,136 bytes: the 1.5 GiB floor + 8 x 50 MiB, lines 78-81; BS2's was 2,082,471,936 with 9) is met;\n ")
edit("A15", "SEC-TAIL2's source as built: the pinned reader exports only column 14, so the scorer reads the run's own "
     "probe file, bound by sha256 to check.json and the markers and by column 14 to the markers' values; the value's "
     "definition is unchanged",
     "    arithmetic mean of rollout_mu_mean[:, 13] (tail_2_pitch_act) over the primary's 8 probe rows (3,473,408 ...\n"
     "    3,530,752), from the markers (probe.rollout_mu_mean, all 15 columns, or a probe.rollout_mu13_mean column: the\n"
     "    reader must export it; the adapter names the key). DIRECTION: phase HIGHER.",
     "    arithmetic mean of rollout_mu_mean[:, 13] (tail_2_pitch_act, the code copy's MJCF actuator order) over the\n"
     "    primary's 8 probe rows (3,473,408 ... 3,530,752), exact rationals of the float32 values. SOURCE (as built: the\n"
     "    pinned reader exports only column 14, read_bs.py line 621): the scorer reads column 13 from the run's own\n"
     "    policy_probe.npz, the file the markers were read from and CHECK passed: its sha256 must equal check.json's\n"
     "    sources and the markers' probe.sha256, and its column 14 at the 8 rows must equal the markers'\n"
     "    rollout_mu14_mean exactly (an input error otherwise); the file is opened only once every entry is decided and\n"
     "    read (OUTCOME-BLINDNESS). DIRECTION: phase HIGHER.")
edit("A15b", "SEC-TAIL2 MISSING when its file is missing or changed (a file defect: PROVISIONAL, BS2's D7 rule)",
     "MISSING when the primary is (the same rows).",
     "MISSING when the primary is (the same rows), or\n"
     "    when the run's probe file is missing or changed since the read (a file defect: also a PROVISIONAL item, BS2's\n"
     "    D7 rule).")
edit("A16", "SEC-MECH's on-policy counter and the control's stock critic cited in the built reader",
     "(k counted in that collection from its\n    reset), for control the stock critic.",
     "(k counted in that collection from its\n"
     "    reset: the counter read_bs.onpolicy_states records, lines 922-995, fed at lines 1059-1062), for control the\n"
     "    stock critic (effective = raw, line 1079).\n   ")
edit("A17", "the report-only phase swing as the reader exports it: settled (1, 1, 1) minus slots zero (the reader does "
     "not feed phi(0), which differs only in the discount slot); V_mean over all the set's states",
     "    THE PHASE SWING (phase arm; control n/a): at every saved pair, the mean over the reader's on-policy states of\n"
     "      V(z, phi(1000)) - V(z, phi(0)), in normalised value units; 0 at S by construction (C2);\n",
     "    THE PHASE SWING (phase arm; control n/a): at every saved pair, the mean over the reader's on-policy states of\n"
     "      V(z, phi(1000)) - V(z, slots 0), in normalised value units: as built, the reader's \"settled\" V_mean (every\n"
     "      state fed (1, 1, 1) = phi(k >= 200) = phi(1000)) minus its \"raw\" V_mean (the slots zero: no phase input,\n"
     "      which differs from phi(0) = (0, 0, 0.0176) only in the discount slot; the reader does not feed phi(0)), each\n"
     "      the mean over all the set's states (read_bs.py lines 875-893, 1058-1062); beside it the own-phase swing\n"
     "      V(z, phi(k)) - V(z, slots 0); 0 at S by construction (C2);\n")
edit("A18", "the per-actuator rises' sources (the run's probe file as SEC-TAIL2; the parent's F2 probe)",
     "    every actuator's on-policy command rise (rollout_mu_mean[:, j] over the primary's rows minus the parent's rows\n"
     "      248-255, j = 0 ... 14) per continuation and arm medians;",
     "    every actuator's on-policy command rise (rollout_mu_mean[:, j] over the primary's rows minus the parent's rows\n"
     "      248-255, j = 0 ... 14; the run's probe file as SEC-TAIL2, the parent's F2 probe at FMAN's sha256) per\n"
     "      continuation and arm medians;")
edit("A19", "RR-1 ported (the draft's 'if ported' resolved)",
     "RR-1: a relaunch after ALL_DECIDED keeps\n"
     "  the first logs/atimes_at_ALL_DECIDED.tsv (as LB's runner does, if ported; otherwise the coordinator copies it aside\n"
     "  before any relaunch, as for BS2: RUNLOG line 789).",
     "RR-1: a relaunch after ALL_DECIDED keeps\n"
     "  the first logs/atimes_at_ALL_DECIDED.tsv (ported from LB: run_waves_bs.sh lines 5-6, 290-305 and 591; a later\n"
     "  snapshot goes to logs/atimes_at_ALL_DECIDED.relaunch<k>.tsv).")
edit("A20", "I1 ported (the draft's 'if ported' resolved)",
     "as LB's runner does (<LB>/NOTES.txt\n  lines 248-252), if ported.",
     "as LB's runner does (<LB>/NOTES.txt\n  lines 248-252): ported (run_waves_bs.sh lines 7, 228-232 and 427-430).")
edit("A21", "outcome-blindness: the scorer opens no probe file either while incomplete",
     "prints only which are decided and opens no markers and no\n  run_meta.",
     "prints only which are decided and opens no markers, no\n  run_meta and no probe file.")
edit("A22", "PROVISIONAL: the probe file item (A15b)",
     "a COUNTED continuation whose markers\n  lack the loop row at E. Cleared only",
     "a COUNTED continuation whose markers\n"
     "  lack the loop row at E; a COUNTED continuation whose probe file is missing or changed since the read (SEC-TAIL2).\n"
     "  Cleared only")
edit("A23", "VALIDATIONS: the adapter's dry runs named",
     "A synthetic dry run with loop rows at E exercises SEC-MARGIN and SEC-MECH.\n",
     "A synthetic dry run with loop rows at E exercises SEC-MARGIN and SEC-MECH.\n"
     "  [Done by the adapter: WORK/dry_parent and WORK/dry_line1, dry_line2, dry_line3, each with score/score.txt and\n"
     "  score.json; WORK/ADAPT_NOTES.txt DRY RUNS.]\n")
edit("A24", "RR-1 in HOW THE COORDINATOR PINS resolved",
     "RR-1: before any runner relaunch after ALL_DECIDED, the first\n"
     "  logs/atimes_at_ALL_DECIDED.tsv must survive (the runner keeps it if LB's fix is ported; otherwise copy it aside).",
     "RR-1: before any runner relaunch after ALL_DECIDED, the first\n"
     "  logs/atimes_at_ALL_DECIDED.tsv must survive (the runner keeps it: LB's fix is ported, run_waves_bs.sh lines\n"
     "  290-305).")
edit("A25", "END line",
     "END (criteria v1 draft; the designer's notes and every command: WORK/DESIGN_NOTES.txt, WORK/CMDLOG.txt).\n",
     "END (criteria v1 as installed by the scorer adapter; the designer's notes and every command: WORK/DESIGN_NOTES.txt,\n"
     "WORK/CMDLOG.txt; the adapter's: WORK/ADAPT_NOTES.txt).\n")

# ---- SCREEN.json first (it is pinned below)
sd = json.loads(SDRAFT.read_text())
sd["_note"] = sd["_note"].replace(
    "draft v1, to be installed as <PS>/SCREEN.json by the scorer adapter, the queue unchanged; the intervention name is "
    "the builder's arm name, critic_phase in the draft",
    "installed by the scorer adapter 2026-10-31 from criteria_work/SCREEN_v1.json, the queue unchanged; the intervention "
    "name is the builder's arm name, bs_common.INTERVENTIONS")
assert "installed by the scorer adapter" in sd["_note"]
txt = ("{\n" + f' "schema": {json.dumps(sd["schema"])},\n' + f' "_note": {json.dumps(sd["_note"])},\n'
       + f' "intervention": {json.dumps(sd["intervention"])},\n'
       + ' "queue": [' + ", ".join(json.dumps(q) for q in sd["queue"][:4]) + ",\n           "
       + ", ".join(json.dumps(q) for q in sd["queue"][4:]) + "]\n}\n")
assert json.loads(txt)["queue"] == json.loads(SDRAFT.read_text())["queue"]
(PS / "SCREEN.json").write_text(txt, encoding="utf-8")
# ---- PINS (filled from the files as they stand)
PIN_FILES = ("SCREEN.json", "tools/score_bs.py", "tools/tests/test_score_bs.py", "tools/bs_common.py",
             "tools/bs_hooks.py", "tools/run_bs.py", "tools/run_waves_bs.sh", "tools/read_bs.py")
pin_old = ("PINS (to be filled by the scorer adapter from the files as installed; RE-FILLED by the fixer from the files as fixed;\n"
           "re-reviewed (P2); any change to a pinned file changes its line)\n"
           "  This text's sha256 cannot be held in the text itself: it goes into <PS>/APPROVED (criteria_sha256) and RUNLOG.\n"
           "    SCREEN.json                          (to be filled)\n"
           "    tools/<scorer>                       (to be filled)\n"
           "    tools/tests/<scorer tests>           (to be filled)\n"
           "    tools/bs_common.py                   (to be filled)\n"
           "    tools/bs_hooks.py                    (to be filled)\n"
           "    tools/run_bs.py                      (to be filled)\n"
           "    tools/run_waves_bs.sh                (to be filled)\n"
           "    tools/read_bs.py                     (to be filled)\n"
           "    (and any new tool module the builder adds, e.g. a phase module: every file the runner, CHECK, the reader or the\n"
           "    scorer imports from <PS>/tools)\n"
           "  The code copy <BS>/code (read only, reused): commit 8737cd3d86431de1bdd0fcdb49a0d956a45e084d, 1,406 files, tree\n"
           "    digest 331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1 [to be re-measured]\n")
SEAL = sys.argv[1] if len(sys.argv) > 1 else "[not re-measured]"
pin_new = ("PINS (filled by the scorer adapter 2026-10-31 from the files as installed; RE-FILLED by the fixer from the files as\n"
           "fixed; re-reviewed (P2); any change to a pinned file changes its line)\n"
           "  This text's sha256 cannot be held in the text itself: it goes into <PS>/APPROVED (criteria_sha256) and RUNLOG.\n"
           + "".join(f"    {rel:<37}{sha(PS / rel)}\n" for rel in PIN_FILES) +
           "    (every file the runner, CHECK, the reader or the scorer imports from <PS>/tools: run_bs.py and read_bs.py\n"
           "    import bs_common.py and bs_hooks.py only (run_bs.py lines 71-72, read_bs.py lines 98-99), the scorer\n"
           "    bs_common.py and read_bs.py; tools/check_inputs.py and tools/verify_copy.py are the builder's one-off checks,\n"
           "    imported by none: not pinned, as in BS2; the other test files are in <PS>/SHA256SUMS.txt.)\n"
           "  The code copy <BS>/code (read only, reused): commit 8737cd3d86431de1bdd0fcdb49a0d956a45e084d, 1,406 files, tree\n"
           f"    digest 331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1 {SEAL}\n")
edit("A26", "PINS filled from the files as they stand; the code copy re-measured", pin_old, pin_new)

out = text
for tag, why, old, new in EDITS:
    n = out.count(old)
    assert n == 1, f"{tag}: the old text occurs {n} times"
    out = out.replace(old, new)
for tag, why, old, new in EDITS:
    print(f"=== {tag}: {why}")
    print("--- old")
    print(old.rstrip("\n"))
    print("+++ new")
    print(new.rstrip("\n"))
assert "(to be filled)" not in out and "if ported" not in out and "DRAFT FOR" not in out
(PS / "criteria.txt").write_text(out, encoding="utf-8")
print("WROTE", PS / "criteria.txt", sha(PS / "criteria.txt"), len(out.splitlines()), "lines")
print("WROTE", PS / "SCREEN.json", sha(PS / "SCREEN.json"))
