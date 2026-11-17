Report: rulings 14 and 15 (G9-ref), worktree /home/user/mesozoic-wt/wpc-cg, branch wt/wpc-cg

Both rulings are built and tested. The tests file gives "43 passed, 1 skipped". G9-ref (f) now passes on S1 and V0. One thing blocks ruling 14's procedure in practice: sets.freeze will not freeze a set again once it drops a frozen entry (lead point A).

Nothing was pushed. No record was written. Nothing ran closed-loop. The main checkout was not touched, and bar.py was not edited.

1. MERGES
- First merge: my head 4329171e was an ancestor of the working branch (6c62f30e), so it was a fast-forward (`git merge --ff-only`). No merge commit, no conflicts.
- Second merge: the working branch moved to 3d261a79 while I worked (r2lfit/CF, servo S1, the rules draft's sixth list, the handover). I merged it as 85d963b5 (`--no-ff`, message from a file with `-F`). No conflicts.
- r2lfit calls only `gates2.require_g9_ref`, which I did not change. Its gate tests on the merged tree: "3 passed, 32 deselected in 2.46s".

2. PER ITEM

Ruling 14 (`designlab/stage2/gates2.py`):
- New `still_available(v, ref)`. A failing item's reference counts as still available while either holds:
  - its entry is usable as the env and the bar read it (`r5lib.why_not` is None);
  - or the set's latest freeze lists it.
  A reference that cannot be named counts as still available.
- `part_a` changes:
  - it now records `failing_references`, which replaces `not_available`; each entry has item, reference, failing terms, `entry_available`, `in_latest_freeze` and `still_available`;
  - a new list, `still_available`, names those references;
  - `followed` says: "G9-ref fails while a failing item's reference is still available in the set (...): the lead marks them not available, freezes the set again and runs G9-ref again (the lead's ruling 14 of 17 Nov 2026)".
- `g9_ref` passes only when `still_available` is empty, so it exits 1 otherwise. `g9_ref_lines` prints each failing reference.
- `RULES["z2_ref"]` holds ruling 14 in its wording part and my reading after `MODULE_READING`.
- Tests:
  - `test_g9_ref_passes_only_when_every_part_does` (the fixer's test marked pending this ruling) now pins it, step by step:
    - a failing turn item gives exit 1, with `still_available == ["transitions/turn90_pi4_left.npz"]` named in `followed`;
    - entry marked not available, set not frozen again: still exit 1;
    - frozen again without the entry, old refcheck: (a) is not read, exit 1;
    - frozen again and the refcheck run again (the item "not run"): exit 0.
  - `test_g9_ref_part_a_on_a_doctored_reference_check` is extended: entry available but not in the freeze; marked but still in the freeze; neither; an unnamed reference.

Ruling 15 (`gates2.py`):
- New `walk_steps(seg, nj)`: per joint, the largest |change| between two consecutive frames of a walk's window, including the last frame back to the first (playback wraps).
- New `_walk_left`, and `joins(track, nj, joints)`. Per join it records `walk`, `walk_is`, and per joint `{jump_rad, walk_step_rad}`. It also records `over` (the joints whose jump is over their limit) and `ok` (jump <= step at every joint; a NaN fails).
- `JOIN_TOL_RAD` is removed. `part_f` passes the joint names and records `join_limit` in place of `join_tol_rad`.
- The rule text now reads: "[changed after a result (rule Z2's label: changed after G9's result): the lead's ruling 15 of 17 Nov 2026 ..., made after the numbers the joins gave on Stage 1's own arrays (up to 1.1e-4 rad) and on V0 (up to 5.3e-5 rad), seen on 17 Nov, where nothing is wrong, and before any look set was frozen and before any closed-loop episode on one; as first written: joint targets within 1e-6 rad]". The second label was added because the merged rules draft's Z2 Check looks for "changed after G9's result" in g9_ref.json (commit bd29fe88).
- Tests:
  - `test_the_join_rule_on_tables` is rewritten (at the limit passes, one ulp over fails, the wrap counts, joins that leave no walk, a track with no walk);
  - `test_g9_ref_part_f_on_the_s1_view` now passes (S1 walk→stop 3.77e-5 rad, brisk→brisk_to_nominal 1.11e-4 rad; both over 1e-6);
  - new `test_g9_ref_part_f_fails_a_doctored_join_whose_jump_exceeds_its_walks_own_step`: the S1 stop's first frame is moved at left_knee by 1.5 times the nominal walk's step there, so (f) fails and `over == ["left_knee"]`; at 0.5 times it passes;
  - `test_g9_ref_on_the_frozen_set_v0`: (f) True, largest jump 5.33e-5 rad (nominal→nominal_to_v0.35);
  - new `test_the_rule_text_holds_rulings_14_and_15_of_17_nov`.

For the lead:
- A. Ruling 14's procedure is blocked by sets.freeze (not my file). `sets.freeze` refuses a freeze that drops an entry an earlier freeze holds (sets.py line 2146, "a frozen entry is never dropped"), and `manifest_of` leaves out entries marked not available. It also needs a refcheck of the frozen files (line 2197). G9-ref runs only on a frozen set, so "mark not available, freeze again" is refused as the code stands. Options:
  - (i) `sets.freeze` accepts dropping an entry the set marks not available, with a label;
  - (ii) a new set version;
  - (iii) G9-ref reads the entry's flag only. Its record would then cite a freeze that still lists the reference.
  I built the fail-closed reading (entry usable OR in the latest freeze) and recommend (i).
- B. My reading of "the walk it leaves" for a join that leaves no walk (start, move or hold): the walk the track played last before the join; else the first walk after it; a track with no walk at all (the stand task) is held to 0.
  - On V0 only one such join is non-zero: start_stop's stand hold into the start, 2.64e-6 rad at left_ankle_pitch. It is held to v0.4 and passes.
  - Alternatives:
    - keep 1e-6 rad for such joins: (f) then fails on V0 at that join;
    - use the walk the join enters;
    - use the move's own frame-to-frame change: a hold has none, so this also fails.
  - I recommend my reading.
- C. The wrap (the window's last frame to its first) counts as a consecutive pair. Leaving it out is the alternative. On S1 and V0 it changes no outcome (largest jump/step ratio 0.0065).
- D. Texts that still hold the old rules (the lead's, not mine):
  - `STAGE2_TRACKB_PLAN.md` line 1394 still says "(joint targets within 1e-6 rad)";
  - lines 1430-1431 still say a failing reference is "treated as not available";
  - `ROUND_C_DESIGN.md` line 1186 still has the 1e-6 rad rule.
- E. Renamed record keys: (a) `not_available` became `failing_references`; (f) `join_tol_rad` became `join_limit`. grep finds no other reader.
- F. gates2 is in `train2.SOURCE_FILES`, so a session's source hashes move. No session exists yet.

3. OUTPUTS I SAW
- `tests/test_stage2_gates2.py` on the final tree: "43 passed, 1 skipped in 88.24s (0:01:28)". The skip is the STAGE2_SLOW r6_start test, which I did not run.
- `g9-ref --body 75-M-H-10 --set V0`, no `--record`, output in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/cg1415/g9ref_V0_head.txt`. First line: "G9-ref V0 75-M-H-10 (plain): (a) True, (b) False, (c) False, (f) True: FAIL; no closed-loop episode runs on this set and body; it goes to the lead". Exit 1.
  - **Part (f): True**; the builder's run of the old rule gave False.
  - It was a FAIL overall because of (b) and (c), as before.
  - `git check-ignore`: the path is outside the repository (exit 128). No `stage2/tb/gates/` was created.
- A scratch probe (`.../wave7/cg1415/f_probe.py`, no record):
  - S1 (f) ok, largest jump 1.1057e-4 rad;
  - V0 (f) ok, largest jump 5.334e-5 rad;
  - largest jump-to-step ratio: 0.0065 on S1, 0.0041 on V0.
- Mutations (in-process plugin, nothing on disk): 11 run, 11 caught. Results are in `.../wave7/cg1415/MUTATIONS.txt`.
- Record-key hashes after the work: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).
- `ruff check` on both files: "All checks passed!".

4. `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/gates2.py     | 152 +++++++++--
 .../trex_form_study/tests/test_stage2_gates2.py    | 303 ++++++++++++++++++---
 2 files changed, 392 insertions(+), 63 deletions(-)
```
`git status --porcelain`: empty. `run_trackb.sh` was not changed.

5. COMMITS, oldest first
- 7466f71f: rulings 14 and 15, with their tests;
- 85d963b5: merge of the working branch at 3d261a79;
- bd29fe88: rule Z2's label "changed after G9's result" added beside "changed after a result".

All three end with the two attribution lines and name no model elsewhere.

**Head: bd29fe8822c2e164bb743f359b3f2a5266690579.**

Not done: no gate record; no G9-ref on any look set (none is frozen); the slow r6_start test; the chain's dry run (the chain was not changed).
