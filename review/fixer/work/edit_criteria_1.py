"""Fixer: criteria.txt edits, batch 1 (exact replacements; each old text must occur exactly once)."""
from pathlib import Path

p = Path("/home/user/work/phase_screen/criteria.txt")
s = p.read_text()
R = []
R.append(("""T. REX r10 PHASE SCREEN (PS): A CRITIC THAT ALSO SEES THE EPISODE PHASE AGAINST CONTROL, FROM STANCE6 s42's 2,097,152
STATE, CPU, SCRATCH.  CRITERIA v1 AS INSTALLED BY THE SCORER ADAPTER -- FOR INDEPENDENT REVIEW.  NOT APPROVED.""",
"""T. REX r10 PHASE SCREEN (PS): A CRITIC THAT ALSO SEES THE EPISODE PHASE AGAINST CONTROL, FROM STANCE6 s42's 2,097,152
STATE, CPU, SCRATCH.  CRITERIA v1 AS INSTALLED BY THE SCORER ADAPTER AND FIXED -- FOR RE-REVIEW (P2).  NOT APPROVED."""))
R.append(("""stated as built for the reviewers to see. PINS are filled from the files as they stand at the install.
Labels:""", """stated as built for the reviewers to see. PINS are filled from the files as they stand at the install.
FIXED 2026-10-31 by the fixer after the independent review (build check, design and process lenses): every finding,
its verdict and every edit are in <PS>/review/FIXES.txt; the reviewed text is <PS>/review/criteria_reviewed_v1.txt
(sha256 2dea6b205c26a83475742d5a654f4365cdc4c873997a9902187fb15216a2a629) and every reviewed file's sha256 is in
<PS>/review/reviewed_sha256.txt. Edits to this text: C6's float32 claim narrowed (build I1); the TensorBoard events
file's place (P-B1: WHERE AND HOW, OUTCOME-BLINDNESS); the end-of-run actor check (P-i3: THE MANIPULATION CHECK);
line 3's rarer-climbs clause (D1, with score_bs.LINE_TEXT[3]); line 2's KNOWN LIMITS sentence and a report-only
slope-at-E-against-S read (D2); the per-bucket V - G left out, with the reason (D3, KNOWN LIMITS); VALIDATIONS' k =
1000 source and the fixer's re-validation (D4); the guard's MISSING note (design I3); the swing's expected size
(design I4); the PINS check and what APPROVED pins (P-m2, P-i7). None changes the question, the arms' contract, the
primary, B_S, the gate, the test or a secondary's definition; line 3's text changes in one clause (D1). PINS are
RE-FILLED from the files as fixed.
Labels:"""))
R.append(("""        documented affine map (as built: (f - 0) / sqrt(1 + 1e-8), which is f in float32); never divided by a running""",
"""        documented affine map (as built: (f - 0) / sqrt(1 + 1e-8), which is exactly float32(f) for every k 0-1000 on
        the paths that round the phase to float32 first: the env's observation (every buffer row) and the reader
        [measured, <PS>/check_build/logs/c_arm_9421b.out C_env_path_slot_equals_float32_phase_all_k, c_reader.out];
        the probe's and V1's float64 feeding paths (bs_hooks.phase_probe_class._row, run_bs.phase_bank_check) round
        17 pre-settle k (9 ... 194) 1 ulp differently, at most 1e-7 relative, and every settled k exactly [measured,
        c_arm_9421b.out C_float64_path_ulp_diffs; <PS>/review/fixer/logs/verify_i1_i4.out]; no measured quantity
        depends on it (the probe's bank is all settled, k 200 / 401 / 600; V1's check runs at zero columns), so the
        code is left as validated: build check I1); never divided by a running"""))
R.append(("""  <PS>/score/score.{json,txt}; harness-owned by design: <PS>/tmp/<run>/ (each run's TMPDIR and MPLCONFIGDIR; a
  continuation's holds its live eval_stage1_*/evaluations.npz), <PS>/tmp/read_<run>/ and <PS>/reads/_cache/.""",
"""  <PS>/score/score.{json,txt}; harness-owned by design: <PS>/tmp/<run>/ (each run's TMPDIR and MPLCONFIGDIR; a
  continuation's holds its live eval_stage1_*/evaluations.npz and learn()'s TensorBoard events file,
  tmp/<run>/tb/PPO_0/events.out.tfevents.*), <PS>/tmp/read_<run>/ and <PS>/reads/_cache/.
  The TensorBoard file (fixer, process finding P-B1) [measured]: the start zip's saved data names tensorboard_log =
  /tmp/tb_buffer/_content_drive_..._s42_01_stance_tensorboard (the parent's Colab buffer); with use_tensorboard False
  the code copy's train_base.py (lines 344-370) leaves it out of the load's arguments, so SB3's load kept it and every
  run_bs.py process that called learn() wrote its events file (outcome-bearing scalars) there, outside <PS> (BS2's 8
  continuations, PS's V2 and the build checker's no-update run: <PS>/review/process/logs/tb_buffer_stat.txt).
  run_bs.py now sets the model's tensorboard_log to <PS>/tmp/<run>/tb before learn() (run_bs.redirect_tensorboard;
  refused outside <PS>/tmp; recorded in run_meta "tensorboard_log"; wrapper_problems refuses learn() without it): the
  logger is built as before (verbose 0, the tensorboard format only), only its folder moves. The older files under
  /tmp/tb_buffer are not this screen's and are not opened."""))
R.append(("""  And on run_meta / the probe: the arm's classes at the load and "in force" at the end; 175 probe rows exactly at""",
"""  And on run_meta / the probe: the arm's classes at the load and "in force" at the end (for critic_phase "in force"
  also needs, after the last update, which no per-rollout evidence follows, the actor's phase columns equal to those
  at the load and their Adam moments exactly 0: run_bs.end_phase_problems on bs_hooks.phase_state, run_meta
  "phase_state_end"; fixer, process finding P-i3); 175 probe rows exactly at"""))
R.append(("""     FROM GIVING THE CRITIC THE EPISODE PHASE (not excluded: a reduction of about half the rise or smaller (power about
     0.3-1.0 at half the control rise, by the between-continuation spread; about 0.5-0.85 at BS2's control spread;
     about 0.1-0.6 if the phase input makes climbs rarer rather than smaller), a critic that had not yet moved its
     weight from the tail angle to the phase, the phase after E, the rail)". When the as-counted p is <= 0.05 and the""",
"""     FROM GIVING THE CRITIC THE EPISODE PHASE (not excluded: a reduction of about half the rise or smaller (power about
     0.3-1.0 at half the control rise, by the between-continuation spread; about 0.5-0.85 at BS2's control spread;
     about 0.1-0.3 if the phase input makes climbs rarer rather than smaller (half as many climbs)), a critic that had
     not yet moved its weight from the tail angle to the phase, the phase after E, the rail)". When the as-counted p is
     <= 0.05 and the"""))
R.append(("""    because it stopped climbing" (KNOWN LIMITS); the report-only slopes at the earlier saved pairs and the phase swing
    are printed to help.""",
"""    because it stopped climbing" (KNOWN LIMITS); the report-only slopes at the earlier saved pairs and the phase swing
    are printed to help. SEC-MECH compares the arms at E, not each arm with S or with 0: whether the phase arm's
    critic stopped paying for the tail is read from the report-only slope at E against S (REPORT ONLY; fixer,
    design D2)."""))
R.append(("""    resampling; one migrating run among four: p 19/70) [computed, WORK power_ps.out G]: NOT SUPPORTED is weak
    evidence that nothing moved.""",
"""    resampling; one migrating run among four: p 19/70) [computed, WORK power_ps.out G]: NOT SUPPORTED is weak
    evidence that nothing moved. With any MISSING value the guard cannot warn (the fill puts its p at 36/70 or more,
    as for the primary), and its line then carries "(with any MISSING value the guard cannot warn: read NOT SUPPORTED
    as not assessed)" (fixer, design I3)."""))
R.append(("""      phase arm also with the phase held at phi(1000) for every state, and on OWN states (all settled,
      k 500-1000: a within-phase slope);""",
"""      phase arm also with the phase held at phi(1000) for every state, and on OWN states (all settled,
      k 500-1000: a within-phase slope);
    THE SLOPE AT E AGAINST S (fixer, design D2; both arms; score_bs.slope_end_vs_start): per continuation the
      effective critic's on-policy slope at E minus at S (the start pair, the same for every continuation: +0.02143
      per degree in V3), and per arm the count with slope(E) <= slope(S) and with slope(E) <= 0: what SEC-MECH, a
      comparison of the arms at E, cannot show (whether a critic stopped paying for the tail);"""))
R.append(("""      [inferred] a critic that has learned the gate should show a positive swing of the order of the settled-minus-
      pre-settle value gap at matched posture (0.17-0.23, WHY); a swing near 0 at E means the phase was not learned;""",
"""      [inferred] a critic that has learned the gate should show a positive swing of the order of 0.2-0.35: the WHY
      gap (0.17-0.23) is settled against steps 50-200, where the gated stream already enters V with weight
      0.98^(200-k), 0.31 on average, while the swing is settled against phase zero, weight 0.98^200 = 0.018, which
      the same gap puts at about 0.24-0.33 before the pop's own effect (<PS>/review/fixer/logs/verify_i1_i4.out);
      it is an off-distribution contrast (settled on-policy states fed a phase they never had), so its size is an
      extrapolation (fixer, design I4); a swing near 0 at E means the phase was not learned;"""))
R.append(("""  evidence on every rollout, C1-C6 for the phase arm, the control unchanged and stock, same-seed pairing, the
  phase recount and the k = 1000 bootstrap observed at least once (a truncation happens within 262,144 steps)), V3""",
"""  evidence on every rollout, C1-C6 for the phase arm, the control unchanged and stock, same-seed pairing, the
  phase recount), V3"""))
R.append(("""  <PS>/validate, discarded after the evidence is recorded: the critic's phase weights move, no actor parameter
  receives a phase gradient), the unit tests, and P3 (the cache key).""",
"""  <PS>/validate, discarded after the evidence is recorded: the critic's phase weights move, no actor parameter
  receives a phase gradient), the unit tests, and P3 (the cache key). The k = 1000 time-limit terminal observation
  carries phi(1000): unit test <PS>/tools/tests/test_bs_hooks.py lines 474 and 504-506, and the build checker's real
  rollout <PS>/check_build/logs/c_arm_9421b.out F_time_limit_terminal_obs_settled (8 truncations, each (1, 1, 1));
  V2's records hold no terminal observation (fixer, design D4)."""))
R.append(("""- What each line can and cannot say [inferred]:
    line 1 can say only that the climb was not reproduced under this build from this state; it says nothing about
      the phase input.
    line 2 can say that, from this state and in this window, giving the critic the phase (tail angle still seen)
      lowered the on-policy tail_3 command relative to a phase-blind critic; with SEC-MECH SUPPORTED and a positive
      phase swing, that the critic stopped paying for the tail as the phase account predicts. It cannot say that""",
"""- What each line can and cannot say [inferred]:
    line 1 can say only that the climb was not reproduced under this build from this state; it says nothing about
      the phase input.
    line 2 can say that, from this state and in this window, giving the critic the phase (tail angle still seen)
      lowered the on-policy tail_3 command relative to a phase-blind critic; with SEC-MECH SUPPORTED and a positive
      phase swing, that the phase arm's critic paid less for the tail than control's at E, consistent with the phase
      account (not separable from a tail that climbed less, below); whether it stopped paying for the tail is read
      only from the report-only slope at E against S (fixer, design D2). It cannot say that"""))
R.append(("""- SEC-MECH is partly confounded with the primary: a tail that climbs is a better clock (AUC 0.61 at P2 -> 0.95 at
  control_2511's E, WHY analysis.txt), so a phase arm that does not climb would show a smaller slope even without using
  the phase; the slopes at the earlier saved pairs (before the arms' commands diverge much) and the swing are printed,
  untested.""",
"""- SEC-MECH is partly confounded with the primary: a tail that climbs is a better clock (AUC 0.61 at P2 -> 0.95 at
  control_2511's E, WHY analysis.txt), so a phase arm that does not climb would show a smaller slope even without using
  the phase; the slopes at the earlier saved pairs (before the arms' commands diverge much) and the swing are printed,
  untested. And SEC-MECH can read SUPPORTED with a phase arm whose slope never moved from its S value (+0.0214): P
  about 0.48 at BS2's control SD, 0.71 at a phase-arm SD of 0.005 [computed, <PS>/review/design/logs/power_rd.out G;
  <PS>/review/fixer/logs/verify_d2.out], hence the report-only slope at E against S.
- The critic's error by episode step (#655 Open Question 1, lines 953-975; the PR-655 review's
  /home/user/work/pr655_review/xref/XREF.txt section 3.4 asks for V - G by bucket 0-50, 50-150, 150-200, 200-250,
  250-1000 as a manipulation check) is NOT measured by this screen (fixer, design D3, option (b)): the pinned harness
  has no tool for it (the reader's on-policy collection records each state's k, not the rewards and the episode
  continuations a discounted return-to-go needs), and building, validating and reviewing one now would delay the
  screen for a report-only read that decides nothing. What stands in for it is weaker: SEC-MECH (confounded, above),
  the report-only slope at E against S, and the phase swing (an off-distribution contrast). Each COUNTED
  continuation's final pair is kept, so per-arm V - G by those buckets (realised discounted return-to-go, gamma 0.98,
  SB3's truncation bootstrap; the phase arm's critic fed each state's own phase) can be measured afterwards by a
  separately written, reviewed and pinned read-only tool, as a report on this screen's outputs that never changes its
  reading."""))
R.append(("""  three quarters 0.25. If the phase input makes climbs rarer rather than smaller (two-state model), at f = 1 0.99 /
  0.82 / 0.58 for q = 1 / 0.9 / 0.75, at f = 0.5 0.20-0.31. A null therefore does not exclude a reduction of about half
  the rise at BS2's spread, nor any smaller one.""",
"""  three quarters 0.25. If the phase input makes climbs rarer rather than smaller (two-state model), at f = 1 0.99 /
  0.82 / 0.58 for q = 1 / 0.9 / 0.75, at f = 0.5 (half as many climbs) 0.20-0.31 for q = 0.75-1 and 0.11 at q = 0.5
  (line 3's "about 0.1-0.3"; WORK power_ps.out F). A null therefore does not exclude a reduction of about half the rise
  at BS2's spread, nor any smaller one."""))
R.append(("""  the reader's <PS>/tmp/read_<name>/; reads/<name>; logs/<name>.attempt<k>.log) except: run_meta.json's fields status,""",
"""  the reader's <PS>/tmp/read_<name>/; reads/<name>; logs/<name>.attempt<k>.log; the TensorBoard events file under
  <PS>/tmp/<name>/tb/, fixer P-B1) except: run_meta.json's fields status,"""))
R.append(("""  APPROVED / tool sha256, or changed since CHECK; a check.json or markers.json made by another read_bs.py; a ledger""",
"""  APPROVED / tool sha256, or changed since CHECK; any 64-hex PINS line of this text not equal to the named file's
  current sha256 or, for a pinned tool, to APPROVED's (score_bs.pins_items; fixer, process finding P-m2); a
  check.json or markers.json made by another read_bs.py; a ledger"""))
R.append(("""  2. sha256sum <PS>/criteria.txt <PS>/SCREEN.json and every pinned tool; check them against the PINS lines and the
     re-reviewed list; chmod 0444 <PS>/criteria.txt <PS>/SCREEN.json (BS2 process I4); write <PS>/APPROVED ONCE as a""",
"""  2. sha256sum <PS>/criteria.txt <PS>/SCREEN.json and every pinned tool; check them against the PINS lines and the
     re-reviewed list (the scorer also checks every PINS line itself: step 4(a); fixer, P-m2). APPROVED's
     tool_sha256 holds exactly the six files score_bs.PINNED_TOOLS names (bs_common.py, bs_hooks.py, run_bs.py,
     run_waves_bs.sh, read_bs.py and score_bs.py); tools/tests/test_score_bs.py is pinned by its PINS line (which the
     scorer checks against the file) and the re-reviewed list, not by APPROVED (fixer, P-i7). chmod 0444
     <PS>/criteria.txt <PS>/SCREEN.json (BS2 process I4); write <PS>/APPROVED ONCE as a"""))
R.append(("""  4. Before launch, all read only: (a) the scorer with --out <PS>/logs/prelaunch_score must exit 3 with STATUS
     incomplete (0 of 8 decided) and NO PROVISIONAL item; (b) run_bs.py --preflight-only for all 8 entries on the real""",
"""  4. Before launch, all read only: (a) the scorer with --out <PS>/logs/prelaunch_score must exit 3 with STATUS
     incomplete (0 of 8 decided) and NO PROVISIONAL item (this includes every PINS line against its file and, for the
     six pinned tools, against APPROVED: fixer, P-m2); (b) run_bs.py --preflight-only for all 8 entries on the real"""))
R.append(("""END (criteria v1 as installed by the scorer adapter; the designer's notes and every command: WORK/DESIGN_NOTES.txt,
WORK/CMDLOG.txt; the adapter's: WORK/ADAPT_NOTES.txt).""",
"""END (criteria v1 as installed by the scorer adapter and fixed by the fixer; the designer's notes and every command:
WORK/DESIGN_NOTES.txt, WORK/CMDLOG.txt; the adapter's: WORK/ADAPT_NOTES.txt; the fixer's: <PS>/review/FIXES.txt and
<PS>/review/fixer/notes.txt)."""))
for old, new in R:
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)
p.write_text(s)
print("applied", len(R))
