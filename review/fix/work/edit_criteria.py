"""The fixer's edit of <BS2>/criteria.txt (the verified review findings; <BS2>/review/FIXES.txt).  Exact old -> new
replacements, each old text present exactly once; PINS are re-filled separately (fill_pins.py) after the tool edits.
Run: PY -I -B edit_criteria.py <BS2>/criteria.txt"""
import sys

p = sys.argv[1]
src = open(p, encoding="utf-8").read()
E = []

# ---- STATUS: the fix
E.append(("""text or a secondary); PINS filled from the files as installed (to be re-filled after the fix, before the re-review: P2).
""", """text or a secondary); PINS filled from the files as installed (to be re-filled after the fix, before the re-review: P2).
FIXED by the fixer (2026-10-30 12:01-12:5x UTC, date -u) after the independent review (build check; design and process
  lenses): each finding verified against its sources, and the confirmed ones fixed, as listed in <BS2>/review/FIXES.txt
  (the reviewed text: <BS2>/review/criteria_reviewed_v1.txt, sha256 c6535c88..., in full in
  <BS2>/review/reviewed_sha256.txt). No fix changes the question, the arms, the primary, B_S, the gate, the test's statistic, alpha or a
  secondary's definition. Changed: wording and figures (PREDICTED OUTCOME, THE TEST's size, the multiplicity bound,
  KNOWN LIMITS, line 3's power clause 0.25-0.96 -> 0.25-1.00), citations, RUN HANDLING (STOPPED, REFUSED),
  OUTCOME-BLINDNESS (tmp/<name>, P4), the disk rule, the pinning steps, and three tool changes (run_waves_bs.sh: the
  scorer pin, the P4 atime re-arm and snapshot, the REFUSED note; score_bs.py 2.1: the qualified size and bound, line
  3's clause, a PROVISIONAL item for a COUNTED continuation without its final pair). To be RE-REVIEWED (P2) before the
  pin.
"""))

# ---- SHORT NAMES: <BS> in the tools; BS/code kept (BC-I5, PR-I2)
E.append(("""  <BS> = S2/r10round/bias_screen (the finished first screen, read only; its criteria v2, sha256 9c490cc6..., are the
         form followed here; its code copy <BS>/code is reused read only)
""", """  <BS> = S2/r10round/bias_screen (the finished first screen, read only; its criteria v2, sha256 9c490cc6..., are the
         form followed here; its code copy <BS>/code is reused read only and must not be removed or modified while
         BS2 may run or be read). In this text <BS> is always the finished first screen; in the BS2 tools' own
         docstrings and messages "<BS>" means the screen's root, <BS2> (bs_common.py lines 17-18).
"""))

# ---- PREDICTED OUTCOME: the ECS rises (D3)
E.append(("""  (the push at S is already present, t 2.8, and grows tenfold by 3,145,728 in the parent; the ECS's twelve s42
  continuations from 3,145,728 all rose, by 0.30-0.38 over 1.57M [measured, <BS>/scratch/criteria_work/
  ecs_spread.out]). Between REDUCED and NO REDUCTION DETECTED, REDUCED is somewhat more likely: the window is the
""", """  (the push at S is already present, t 2.8, and grows tenfold by 3,145,728 in the parent; the ECS's twelve s42
  continuations from 3,145,728 all rose, by 0.24-0.48 (group means 0.30 and 0.38) over 1.57M [measured,
  <BS>/scratch/criteria_work/ecs_spread.out lines 2-13: last 8 rows minus own first 8 rows]). Between REDUCED and NO
  REDUCTION DETECTED, REDUCED is somewhat more likely: the window is the
"""))

# ---- WHERE AND HOW: harness, outputs (BC-I2), disk (PR-M1, PR-I3)
E.append(("""  own sha256 and its modules' (P3)); score_bs.py (this text's reading, by the scorer adapter). Launch: the builder's
  NOTES, HOW TO LAUNCH.
- Outputs: <BS2>/runs/<arm>_<seed>/, <BS2>/reads/<arm>_<seed>/markers.{json,txt}, <BS2>/lost.json, <BS2>/logs/,
  <BS2>/score/score.{json,txt}.
- Disk: a fresh queue starts only with at least 2,082,471,936 bytes free (1.5 GiB + 9 x 50 MiB, the runner's launch
  floor, bs_common.LAUNCH_DISK_BYTES); before each start at least 1,610,612,736 bytes (1.5 GiB): below it the runner
  pauses (logs/PAUSED_disk), the running ones go on, and it resumes by itself. 2.8 GB was free at 10:45 UTC [measured,
  df -h /tmp]. If it pauses, the coordinator frees space inside <BS2> only (validation folders after copying their
  records; tmp/), never a continuation's file, and reads nothing of any outcome while doing so.
""", """  own sha256 and its modules' (P3)); score_bs.py (this text's reading, by the scorer adapter). The runner also refuses
  unless APPROVED pins tools/score_bs.py at its current sha256, and keeps the P4 atime record (OUTCOME-BLINDNESS).
  Launch: the builder's NOTES, HOW TO LAUNCH, after HOW THE COORDINATOR PINS step 4.
- Outputs: <BS2>/runs/<arm>_<seed>/, <BS2>/reads/<arm>_<seed>/markers.{json,txt}, <BS2>/lost.json, <BS2>/logs/,
  <BS2>/score/score.{json,txt}; and, harness-owned by design as in BS: <BS2>/tmp/<run>/ (each run's TMPDIR and
  MPLCONFIGDIR; a continuation's holds its live eval_stage1_*/evaluations.npz), <BS2>/tmp/read_<run>/ (the reader's)
  and <BS2>/reads/_cache/ (the reader's loop cache, P3).
- Disk: a fresh queue starts only with at least 2,082,471,936 bytes free (1.5 GiB + 9 x 50 MiB, the runner's launch
  floor, bs_common.LAUNCH_DISK_BYTES); before each start at least 1,610,612,736 bytes (1.5 GiB): below it the runner
  pauses (logs/PAUSED_disk), the running ones go on, and it resumes by itself. 2.8 GB was free at 10:45 UTC and
  2,853,748,736 bytes at 12:09 UTC [measured, df]. The coordinator launches only with at least 2.6 GiB free (df -h
  /tmp; about 40 MB per BS2 continuation plus the reads, above the agents' 2.2 GB working floor [inferred, process
  review I3]); below that it first removes <BS2>/validate/v1final, v2final and the fixer's v1fix (about 200 MB in all)
  after copying their records. If the runner pauses, the coordinator frees space inside <BS2> only: validation folders
  after copying their records, and the tmp/ folders of validation runs (tmp/save_loaded_only_*, tmp/no_update_*,
  tmp/read_<validation run>, tmp/read_parent_s42, tmp/parent_schedule); never tmp/<queue name> or tmp/read_<queue
  name> (a running continuation's TMPDIR holds its live evaluations.npz; removing it crashes the run at its next
  evaluation [inferred: SB3 EvalCallback makes the folder only at init and saves into it at every evaluation]), never
  a continuation's file, and it reads nothing of any outcome while doing so.
"""))

# ---- THE MANIPULATION CHECK: control untouched is shown indirectly (BC-I1)
E.append(("""- That the control arm is untouched is shown before launch by the builder's V2 (same seed, train() a no-op, from S).
""", """- That the control arm is untouched is shown before launch by the builder's V2 (same seed, train() a no-op, from S),
  indirectly as in BS: control's loaded state equals an independent load of the start pair, it installs nothing (its
  classes are the stock ones before and after), and the other arms reproduce its environment side on 31 of 31
  rollouts; no separate hook-free run exists (build check I1).
"""))

# ---- THE TEST: size only under exchangeability (D1)
E.append(("""- Minimum attainable p = 1/70 = 0.0143. Attainable p <= 0.05: 1/70, 2/70, 3/70 (0.0429); 4/70 = 0.0571 is not. Size
  under exchangeable continuous values: 3/70 = 0.0429 [computed, WORK power_bs2.out A].
""", """- Minimum attainable p = 1/70 = 0.0143. Attainable p <= 0.05: 1/70, 2/70, 3/70 (0.0429); 4/70 = 0.0571 is not. Size:
  3/70 = 0.0429 when the 8 primaries are exchangeable under no effect (continuous values) [computed, WORK
  power_bs2.out A]. If blinding changes only the between-continuation spread (BS: SD 0.137 control, 0.029 blind),
  P(REDUCED) with equal means is about 0.06 (0.057-0.064 for SD ratios 3-7), and up to about 0.09 given the gate at a
  control rise near 0.29 [computed, <BS2>/review/design/work/rd_power.out P2; reproduced by the fixer,
  <BS2>/review/fix/work/mc_check.out]. A studentized statistic does not change this with 4 + 4 (it ranks the 70
  relabelings identically: rd_power2.out). Read a REDUCED with each arm's printed SD.
"""))

# ---- What NOT INFORMATIVE would mean (D4)
E.append(("""    NOT INFORMATIVE: the median control continuation did not make a third of the parent's climb in 1.44M steps from
      the same state; with BS this would say the climb is not locked in even at 2,097,152 under fresh rollout seeds,
      and a screen of the critic would need a later start (3,145,728, where every ECS s42 continuation rose) or more
      seeds. No reading about the critic.
""", """    NOT INFORMATIVE: the median control continuation did not make a third of the parent's climb in 1.44M steps from
      the same state; with BS this would say the climb is not reproduced from 2,097,152 under this harness's fresh
      draws (new seeds, CPU, a fresh reset at S: the named departures above); it does not separate those causes. A
      screen of the critic would then need a later start (3,145,728, where every ECS s42 continuation rose) or more
      seeds. No reading about the critic.
"""))

# ---- line 3's power clause (D5; score_bs.LINE_TEXT[3] changed with it)
E.append(("""     0.25-0.96 at half the parent's rise, by the between-continuation spread; about 0.1-0.3 if blinding makes climbs
""", """     0.25-1.00 at half the parent's rise, by the between-continuation spread; about 0.1-0.3 if blinding makes climbs
"""))

# ---- multiplicity (D1)
E.append(("""- Multiplicity: three secondaries at 3/70 each; under the global null the chance that at least one reads SUPPORTED is
  at most 9/70 = 0.129 (Bonferroni bound). A secondary with p = 1/70 (0.0143 <= 0.05 / 3 = 0.0167) is printed as also
""", """- Multiplicity: three secondaries at 3/70 each; under the global null the chance that at least one reads SUPPORTED is
  at most 9/70 = 0.129 (Bonferroni bound) when each secondary's 8 values are exchangeable under no effect; about 0.18
  (3 x 0.06) if the arms' spreads differ (THE TEST). A secondary with p = 1/70 (0.0143 <= 0.05 / 3 = 0.0167) is
  printed as also
"""))

# ---- SEC-G citation (PR-I2)
E.append(("""    (3,670,016) [measured, EW ew_r10.jsonl; <BS>/reads/parent_s42/markers.txt], so a control arm that climbs like the
""", """    (3,670,016) [measured, EW ew_r10.jsonl lines 3, 1 and 4; <BS2>/validate/v3final/parent_s42/markers.txt lines
    17-18 for S and 3,145,728 (<BS>/reads/parent_s42/markers.txt has S only)], so a control arm that climbs like the
"""))

# ---- SEC-MARGIN citation (PR-I2)
E.append(("""    UTC); in the parent it fell +1.399 (S) -> +0.240 (3,145,728) -> -0.061 (3,670,016) [measured: <BS>/reads/
    parent_s42/markers.txt; 1 - G - c0 from EW ew_r10.jsonl]. It shares G with SEC-G: if SEC-G holds, SEC-MARGIN
""", """    UTC); in the parent it fell +1.399 (S) -> +0.240 (3,145,728) -> -0.061 (3,670,016) [measured: <BS2>/validate/
    v3final/parent_s42/markers.txt lines 17-18 for S and 3,145,728; 1 - G - c0 from EW ew_r10.jsonl line 4 for
    3,670,016]. It shares G with SEC-G: if SEC-G holds, SEC-MARGIN
"""))

# ---- a COUNTED continuation without its final pair (D7)
E.append(("""  A secondary's value is MISSING for a LOST, STOPPED or UNREADABLE continuation and when its marker is absent or not
  finite (cause printed). Secondaries need M-LOOP's final-pair row: a full-read failure confined to the loop markers
  is a tool defect (RUN HANDLING) and holds the reading, as BS.
""", """  A secondary's value is MISSING for a LOST, STOPPED or UNREADABLE continuation and when its marker is absent or not
  finite (cause printed). Secondaries need M-LOOP's final-pair row: a full-read failure confined to the loop markers
  is a tool defect (RUN HANDLING) and holds the reading, as BS. Likewise a COUNTED continuation whose markers lack the
  loop row at E (the final pair stage1_final.zip / stage1_final_vecnorm.pkl not read: read_bs.saved_pairs leaves out
  a missing pair silently and CHECK does not look at it) reads SEC-G and SEC-MARGIN MISSING, filled, and is a
  PROVISIONAL item (score_bs 2.1): a tool or file defect, which holds the reading until it is found.
"""))

# ---- STOPPED cannot occur (D6, PR-I4)
E.append(("""- STOPPED: CHECK passed, stop reason "collapse" (the run's own collapse stop, patience 10) at T < E; possible, if at
  all, only at the window's 10th evaluation, 3,407,872 [inferred, as BS]. MISSING, filled (primary and secondaries).
""", """- STOPPED: CHECK passed, stop reason "collapse" (the run's own collapse stop, patience 10) at T < E. Not expected: the
  stop cannot fire in (S, E] (collapse_min_evals 20, patience 10; the resume seeds the parent's 16 evaluations at or
  before S, so the 11 window evaluations give at most 8 patience observations < 10, and without the seeding none)
  [measured: F2 evaluations.npz.copy1 (16 at or before S, 11 in (S, E]), stage_config.json.copy1; inferred from
  <BS>/code environments/shared/curriculum/early_stopping.py lines 129-130, 188-193 and 254, and checkpoints.py lines
  387-429]. Kept as a guard: if it occurs, MISSING, filled (primary and secondaries).
"""))

# ---- REFUSED: the cause from a preflight re-run (PR-M3)
E.append(("""- REFUSED by preflight (exit 2, before any optimizer step): fixed and restarted under the same name; a pinned-file
  change needs the maintainer's word and a re-pin; unfixed, VOID.
""", """- REFUSED by preflight (exit 2, before any optimizer step): fixed and restarted under the same name; a pinned-file
  change needs the maintainer's word and a re-pin; unfixed, VOID. Its cause is found by re-running, read only,
      cd /tmp && nice -n 10 S2/venv/bin/python -I -B <BS2>/tools/run_bs.py --arm ARM --seed N --preflight-only
  (it prints the same problems and creates nothing; the runner's REFUSED note names this command), or, if the refusal
  left a folder, from its run_meta.json error and traceback; never from the attempt log.
"""))

# ---- OUTCOME-BLINDNESS: tmp/<name> (PR-M1), the refusal cause (PR-M3), P4 (PR-M2)
E.append(("""- Until every queue entry is decided (COUNTED, STOPPED or LOST) and none is VOID, nobody (no person and no agent; the
  pinned tools' own work, run_bs.py and CHECK, excepted) opens any file of a continuation (runs/<name> and every
  runs/<name>_crashed<k> / _void<k> / _refused<k>, reads/<name>, logs/<name>.attempt<k>.log) except: run_meta.json's
  fields status, exit, stop_reason, heartbeat, last_step, error and traceback (the last two only to find a crash's or a
  refusal's cause); check.json's pass / fail with its failures list and its identity and source fields, and
  logs/<name>.check.log. Everything else is outcome-bearing or may be (01_stance/progress.json, evaluations.npz,
  train_episodes.csv, eval_episodes.csv, diagnostics.npz, gate_progress.npz, policy_probe.npz, the checkpoints,
  arm_evidence.jsonl, run_meta's other fields, the attempt logs) and is not opened. The runner's own records
  (logs/run_waves_bs.log, decided.tsv, crash_ledger.tsv, PAUSED_disk, HALTED, proctable.log, run_waves_bs.exit) and
  <BS2>/lost.json may be read. read_bs.py refuses a continuation's markers until all are decided; score_bs.py, while
  any entry is undecided or unread, prints only which are decided and opens no markers and no run_meta.
""", """- Until every queue entry is decided (COUNTED, STOPPED or LOST) and none is VOID, nobody (no person and no agent; the
  pinned tools' own work, run_bs.py and CHECK, excepted) opens any file of a continuation (runs/<name> and every
  runs/<name>_crashed<k> / _void<k> / _refused<k>; its TMPDIR <BS2>/tmp/<name>/, which holds its live
  eval_stage1_*/evaluations.npz (the evaluation returns); the reader's <BS2>/tmp/read_<name>/; reads/<name>;
  logs/<name>.attempt<k>.log) except: run_meta.json's fields status, exit, stop_reason, heartbeat, last_step, error and
  traceback (the last two only to find a crash's or a refusal's cause; a preflight refusal that left no folder is
  diagnosed by re-running --preflight-only, RUN HANDLING, never from its attempt log); check.json's pass / fail with
  its failures list and its identity and source fields, and logs/<name>.check.log. Everything else is
  outcome-bearing or may be (01_stance/progress.json, evaluations.npz, train_episodes.csv, eval_episodes.csv,
  diagnostics.npz, gate_progress.npz, policy_probe.npz, the checkpoints, arm_evidence.jsonl, run_meta's other fields,
  the attempt logs, tmp/<name>/) and is not opened. The runner's own records (logs/run_waves_bs.log, decided.tsv,
  crash_ledger.tsv, PAUSED_disk, HALTED, proctable.log, run_waves_bs.exit, atimes_at_ALL_DECIDED.tsv) and
  <BS2>/lost.json may be read. read_bs.py refuses a continuation's markers until all are decided; score_bs.py, while
  any entry is undecided or unread, prints only which are decided and opens no markers and no run_meta.
- P4 (relatime; BS's process lesson, RUNLOG 09:13 UTC): right after the runner decides a continuation COUNTED or
  STOPPED it re-arms every regular file under runs/<name>/ and tmp/<name>/ and its logs/<name>.attempt<k>.log (atime =
  mtime - 1 ns; content, mtime and sha256 unchanged), so any later read of one moves its atime; at ALL_DECIDED, before
  it writes logs/ALL_DECIDED and before the full reads, it writes logs/atimes_at_ALL_DECIDED.tsv (path, atime, mtime,
  ctime and rearmed_unread for runs/**, logs/** and tmp/**; lstat only, which moves no atime). The process
  verification compares them: every such file of a COUNTED or STOPPED continuation must still read rearmed_unread 1
  there, except run_meta.json and check.json (whose allowed fields may have been read); any other 0 is a read after
  the decision and is reported. Reads before a decision (while the run runs, and CHECK's own) are not covered.
"""))

# ---- PROVISIONAL list: D7
E.append(("""  read_bs.py; a ledger changed since the read; M5 off its pins; the code copy (<BS>/code = bs_common.CODE, not under
  <BS2>) failing M1's check_copy. Cleared only by measuring again with the pinned files, never by a ruling.
""", """  read_bs.py; a ledger changed since the read; M5 off its pins; the code copy (<BS>/code = bs_common.CODE, not under
  <BS2>) failing M1's check_copy; a COUNTED continuation whose markers lack the loop row at E (THE SECONDARIES).
  Cleared only by measuring again with the pinned files, never by a ruling.
"""))

# ---- VALIDATIONS: the dry run with D7, and the fixer's re-runs
E.append(("""  8 if the parent's markers have no loop row at E, as read_bs.py --reference gives: the parent's F2 pairs in [S, E]
  are 2,097,152 and 3,145,728 only). A synthetic dry run with loop rows at E exercises SEC-G and SEC-MARGIN (WORK
  dryrun_root/, ADAPT_NOTES.txt).
""", """  8 if the parent's markers have no loop row at E, as read_bs.py --reference gives: the parent's F2 pairs in [S, E]
  are 2,097,152 and 3,145,728 only; with score_bs 2.1 each of the 8 is then also a PROVISIONAL item, "the final pair
  was not read"). A synthetic dry run with loop rows at E exercises SEC-G and SEC-MARGIN (WORK dryrun_root/,
  ADAPT_NOTES.txt).
- After the fixer's changes (run_waves_bs.sh, score_bs.py and their tests; run_bs.py, read_bs.py, bs_common.py and
  bs_hooks.py unchanged): all unit tests, ruff, the scorer's mutation tests and both dry runs re-run; V1 re-run
  (<BS2>/validate/v1fix) and V3 re-run on V2's outputs (<BS2>/validate/v3fix); V2 not re-run (run_bs.py and
  bs_hooks.py unchanged) [measured: <BS2>/review/FIXES.txt, <BS2>/review/fix/NOTES.txt].
"""))

# ---- KNOWN LIMITS: the gate and the test (D1, D2)
E.append(("""- The informativeness gate and the test read the same control values. Under no effect, P(REDUCED) is at most 3/70
  over the whole screen; given that the gate passed, it is about 0.08 if the control arm's true rise is 0.20 (the
  gate), 0.11-0.15 at 0.15, 0.15-0.26 at 0.10, 0.05-0.06 at 0.25 and at most 0.053 at 0.29 or more [computed, WORK
  power_bs2.out E]. A REDUCED read with a control median rise below about 0.29 is less secure than its p says.
""", """- The informativeness gate and the test read the same control values. Under no effect, with the 8 primaries
  exchangeable, P(REDUCED) is at most 3/70 over the whole screen; given that the gate passed, it is about 0.08 if the
  control arm's true rise is 0.20 (the gate), 0.11-0.15 at 0.15, 0.15-0.26 at 0.10, 0.05-0.06 at 0.25 and at most
  0.053 at 0.29 or more [computed, WORK power_bs2.out E]. If blinding changes only the spread, about 0.06 overall and
  up to about 0.09 given the gate (THE TEST). In the normal model this risk is confined to control rises below about
  0.29. In the two-state model, where each continuation climbs or stays, it is 0.06-0.10 given the gate when half or
  fewer continuations climb (q 0.5-0.25), and 0.08-0.14 when the control median rise is >= 0.29 [computed,
  <BS2>/review/design/work/rd_power.out P3; reproduced by the fixer, <BS2>/review/fix/work/mc_check.out]. A REDUCED in
  which only 2 of the 4 control primaries climbed (the printed values show it) is therefore less secure than its p
  says.
"""))

# ---- KNOWN LIMITS: secondaries (D1) and the P4 audit's reach (PR-M2)
E.append(("""- The three secondaries are tested at 0.05 each, unadjusted (up to 0.129 family-wise under the global null); SEC-G and
  SEC-MARGIN share G. A SUPPORTED secondary is a pre-registered observation, not a line.
""", """- The three secondaries are tested at 0.05 each, unadjusted (up to 0.129 family-wise under the global null with
  exchangeable values, about 0.18 if the arms' spreads differ); SEC-G and SEC-MARGIN share G. A SUPPORTED secondary is
  a pre-registered observation, not a line.
- Outcome-blindness is audited from atimes (P4): a read of a continuation's file before its decision (while it runs,
  or CHECK's own) is not visible, and the audit holds only on this relatime ext4 filesystem (/proc/mounts) [measured].
"""))

# ---- HOW THE COORDINATOR PINS step 4 (PR-M4, PR-I6, PR-I3)
E.append(("""  4. run_bs.py --preflight-only for one entry on the real <BS2> (expected exit 0); run_waves_bs.sh --plan; launch.
""", """  4. Before launch, all read only: (a) cd /tmp && S2/venv/bin/python -I -B <BS2>/tools/score_bs.py --out
     <BS2>/logs/prelaunch_score must exit 3 with STATUS incomplete (0 of 8 decided) and NO PROVISIONAL item (it
     checks APPROVED's criteria, SCREEN and code digest and all six tool pins, score_bs.py's included, this text's
     naming of the scorer, M5's pins and the code copy); (b) run_bs.py --preflight-only for all 8 entries on the real
     <BS2> (expected exit 0 each); (c) RUNLOG holds APPROVED's sha256 and every re-reviewed sha256 (grep; P1, P2); (d)
     df -h /tmp shows at least 2.6 GiB free (WHERE AND HOW); (e) run_waves_bs.sh --plan; then launch. The runner
     itself refuses (exit 2, nothing started) unless APPROVED pins tools/score_bs.py at its current sha256.
"""))

for old, new in E:
    assert src.count(old) == 1, old[:120]
    src = src.replace(old, new)
open(p, "w", encoding="utf-8").write(src)
print("ok", len(E))
