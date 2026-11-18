# review:CK (roundC, wave 3): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

REVIEW OF WORK PACKAGE CK (designlab/stage2/lookread.py, tests/test_stage2_lookread.py), branch wt/wpc-ck at 01fccec
Simulation only. Everything below is kinematic playback, synthetic series or tests, except the one closed-loop episode on the view S1 that the package's own slow test runs (no record). I changed no tracked file and committed nothing.

VERDICT
Merge as it stands, but as the five content commits cherry-picked onto the working branch, not as a merge of the branch. I found no blocker and no major defect in the code. The minor points below can go in with the next change to the two files. Those that touch lookread.py should go in before code state B, because the file is hashed into every Track B row's sources.

FINDINGS, MOST SERIOUS FIRST

1. Blocker for a plain merge only (no code change): two commits lack the attribution lines.
- cc8eec5 ("Merge branch 'wt/wp-r' into wt/wpc-ck") and 3182a6e ("Merge branch 'wt/wp-b' into wt/wpc-ck") consist of the subject line alone. Both are the earlier builder's and are not in the working branch.
- The other six unmerged commits (6d6d38d, 11a23b4, 8fbd47e, 0ccabf2, 62751fc, 01fccec) end with exactly the two lines and name no AI model elsewhere.
- Fix: cherry-pick 6d6d38d, 11a23b4, 0ccabf2, 62751fc, 01fccec, as was done for CB and CE.
- Tried in a scratch clone (`git merge-tree --write-tree --merge-base=<c>^`, chained): all five apply in order without conflict on the working branch's tip fb421ba. The final tree is 011728eb, the same tree as `git merge-tree --write-tree <working> wt/wpc-ck` (exit 0). `git diff --stat <working> <result>` shows only lookread.py (+592) and test_stage2_lookread.py (+822).
- The five commits touch only those two files.

2. Minor (tests): two behaviours of the frozen API are not pinned.
I ran 56 single-fault copies of lookread.py (scratch copies loaded in place of the module) against the package's test file: 47 caught, 9 not caught. Two of the nine are equivalent: the two hip-roll anchors are one point on the midline on both bodies, and `_Played.q` already holds the last pose. The gaps that matter:
- (a) The bob as the mean of the own steps' ranges. A copy that returns the whole window's max less min passes all tests. With a 2 mm/s drift of the hip height the code reads 7.571 mm (7.971 mm without drift) where the window's range is 27.246 mm, so the code is right and no test shows it.
- (b) Too few steps alone. A copy with `too_few = own.lift_offs < 4` passes; no test has own steps 3 with 4 lift-offs. The code itself gives True.
- Two proposed tests are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_ck_wave3/test_extra.py`. They pass on the real module and fail on the two fault copies. Ignore the third test in that file: it is wrong, because the anchors coincide.
- (c) Gate helpers, lower priority:
  - `against_ladder`'s 0.02 rad toe-off limit never bites alone (the "limits bite" case is caught by the 10 % limit);
  - `against_ladder`'s tail wrap has no test;
  - `controls`' `ok` is not shown to need the cut, the exact ratio 1, or "exactly its term" (`term in failed` passes).

3. Minor: `window(b, rec)` gives a window to a walk episode that fell inside its last two updates (lookread.py:413-419).
- The bar reads no window after a fall (bar.py:1314-1317).
- Probe: `_built("bar_walk")`, a trace ending at 19.98 s with `rec.out = {"fell": True}` returns (9.98, 19.98).
- The builder names this for CW. It can be closed here within the frozen signature: `if (getattr(rec, "out", None) or {}).get("fell"): return None`. `metrics.Recorder.end` stores `out` (metrics.py:182).

4. Minor: the playback of just the window does not always read what a whole trace reads inside it.
- An event on the window's first physics step is counted in a trace and cannot be counted in the window's own playback (`readings.touchdowns`: "Step 0 is never one").
- On the 75 cm v040 rung's loads, over 600 window starts one physics step apart, the counts differ by one lift-off or touchdown at 5 (for example start step 172: [25, 25] against [25, 24]).
- On the 50 real items I played, the counts were equal every time.
- Effect: one event of six or more, the same for every episode of a condition.
- Fix, either: start `played` a little before `ta_ref` and read inside the window, or say so in the docstring (lines 25-27 say "read alike").

5. Minor: `series`' bounds check is off by one (lookread.py:192, `>` where `>=` is exact). Six poses every 10 steps with 50 loads are accepted, though pose 5 is step 50. A live recorder never produces this.

6. Minor: `RULES["bands"]` and `RULES["reference"]` (lookread.py:94-110) open with "STAGE2_TRACKB_PLAN rule U3." and "rule U5." and then hold sentences that are the module's readings, not the rule's words:
- a robot's value that cannot be computed fails its term;
- the bob has no threshold;
- the tail reads the phase alone;
- "or lacks one of its own values".
These strings go into records as rule text. Mark them as the module's reading.

7. Minor, for CW: `played` turns every `ValueError` of `bar._Played` into `LookError("not played: ...")` (lookread.py:370-373). No walk or turn item holds a turning gait, so the wrapper must treat any `LookError` from a playback as a definition error (U5), never as "this item has no look reading".

8. Nit: the skip reasons name a command that is a usage error as typed (`python tools/restore_inputs.py --group ...`, test file lines 401, 601, 640, 700, 802). The tool needs `restore` or `copy-from DIR`.

9. State, not a defect: the branch is 41 commits behind the working branch (it merged cd68f2c; the tip is fb421ba).
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` therefore lists 34 files today; 32 are the working branch's later changes.
- Against the merge base (`...HEAD`) only the two CK files differ, and `git diff cd68f2c 8fbd47e --stat` shows the builder's merge took the working branch's version of everything else.

THE BUILDER'S POINTS FOR THE LEAD: OPTIONS AND MY CHOICE

a. Attribution. Options: plain merge (brings the two commits), cherry-pick, squash. Choose cherry-pick (finding 1).

b. `on_command_at_stop`. Options: a function in lookread, or in lookbar (CW); bar.py cannot be edited. Choose CW's brief.
- It is a term of the companion record on the recorder's CoM rows and `t_c`, not a look measure. Both files are in the rows' sources either way.
- The brief must fix three things: which stride (I would take the walk's planned period on the simulated clock at the nominal rate), speed along the commanded heading averaged over the update rows in [t_c - T, t_c], and 0.9 x `b.target_speed`.
- I agree that rulings 5 and 8 otherwise ask nothing of lookread beyond "nudge and stop items have no window", which is built and tested.

c. Z2 (b)'s toe-off limit. Options: 0.02 rad (brief, design 3.7), or 0.02 rad and 10 % (rule as settled 15 Nov). Choose the rule, as built. All six rungs pass both; the worst is -5.1 %. The design's sentence is owed a correction.

d. A reference that lacks a value. Options: LookError, pass as "not read", fail the term. Choose LookError, as built. Passing would fail open, and U5 calls a faulty reference an error of the definition.

e. The tail's amplitude. Recommend a ruling before state B.
- Probe: a tail of 1e-4 rad noise against a 0.12 rad reference passes the tail term in 40 of 200 draws. A small tail with a steady phase would pass every time.
- Options: leave it (the rule's letter, as built); fail the term when the robot's fitted amplitude is under 0.02 rad while the reference's is read; a relative amplitude band.
- Choose the second: it reuses U3's own threshold and changes nothing for a robot that tracks. It needs a dated sentence in U3, one line in `compare` and a test.

f. The bob without a threshold. Confirmed on the three frozen 75 cm sets (F25F9, V0, F25F9cl; 50 built walk and turn items, kinematic).
- The reference's bob is 0.0000 mm, so `bob_ratio` is float noise (0.33 to 1.6; the term is True or False by chance).
- The reference's own head pitch is 1.98 to 2.03 deg RMS, so `head` fails.
- Options: leave it; a "not read" threshold for the bob (a U3 change); CW records only the numbers on a set outside the look sets.
- Choose the last. U4 already says that on any other set "the numbers are recorded", and no rule changes. The look sets' bobs are 4 to 10 mm.

g. Contact read literally. Options: as built; the bar's air-gap classes (needs more inputs than the frozen API has); a minimum air time.
- Choose as built for now. Probe: a standing robot with five 22 ms blips per foot reads 9 steps and `too_few_steps` False, and still fails bob, tail and toe-off.
- Ask CW to record, per episode, how many of the window's air runs are shorter than `b.air_gap_max_s`. Change U1 only on G9-run's evidence, labelled.

h. Near 0.02 rad. Nothing to decide in the code: U3 keeps the threshold and the records print the side (`compare`'s `reference` block does). `window_ref` fixes one pose grid per condition, so the side is the same in the gate and in every episode. Ask CG to list each item's reference toe-off and side in g9_ref.json.

i to m. I agree with each as built:
- i: an RMS about the mean; home does not enter.
- j: equals `foot_pitch_solved` on the ladder (test, 1e-9) and is heading-free. look.py's own form, atan2(-R20, R00), is not.
- k: sound, with finding 4.
- l: the brief says `readings`; the slow test shows equality with `Recorder.contact()`.
- m: U2's words win; design 3.5's "3 s of walking" is owed a correction.

CHECKED AND FOUND RIGHT
- Frozen API: `BANDS is motion.BANDS`, `MIN_STEPS`, `series` (extra `t0=0.0`), `reading`, `compare`, `playback`, `window`, `sources` are all present with the brief's behaviour. `bar` is imported only inside `played`.
- U1 against look.py:
  - hip midpoint from the `*_hip_roll` anchors as look.py:607 and :628;
  - head as `gait._nose_up`;
  - `_fit_phase` the same form;
  - contact, lift-offs and touchdowns by `readings.contact_flags` and `touchdowns`.
- U2: the walk window mirrors `bar._ran_to_end` and `_judge`; the turn window mirrors `readings.turn_reading`'s `window_reached`.
- U3 and U5: no slack; thresholds are strict `<`; a reference with too few steps raises; too few own steps fail three terms and still read the head.
- `window_ref` on real `BarBuilt`s equals RATE x `window` on a nominal-clock trace.
- Acceptance table: my own run through `lookread.playback` reproduces the builder's six rows digit for digit, and two calls return equal dicts.
- Non-finite poses raise nothing and fail all four terms.
- Record-key hashes are the registered pair in the worktree and on the cherry-picked tree: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- `lookread.sources()` gives `designlab/stage2/lookread.py: 199bf696...`.
- The module writes no file. The restored inputs and the builder's logs are ignored (`stage2/sets/.gitignore:1:*.npz`, `.gitignore:34:logs/`, `.gitignore:49:*.npz`). Nothing in the worktree is newer than my first scratch file.
- The builder's scratch scripts are what the report says: closed-loop on S1 only; the look sets played kinematically and read only.

EXACT TEST LINES (standing environment, through tools/lowprio.sh)
- Worktree, `pytest -p no:cacheprovider tests/test_stage2_lookread.py`: "53 passed, 1 skipped in 7.14s", exit 0.
- Worktree, the same with STAGE2_SLOW=1: "54 passed in 11.08s", exit 0.
- `ruff check --no-cache` on the two files: "All checks passed!".
- `git status --porcelain` in the worktree: empty, before and after.
- Scratch export of tree 011728eb (working tip plus the two files), no ignored inputs: "38 passed, 16 skipped in 2.91s"; every skip is clean.
- The same export after `tools/restore_inputs.py copy-from ... --group picks-models picks-arrays look x sets` ("all here with their listed sha256"), STAGE2_SLOW=1: "54 passed in 9.94s", exit 0.
- The earlier builder's two files as of 8fbd47e, in that export: "49 passed in 5.00s".
- Single faults: "caught 47, not caught 9".

NOT CHECKED
- No `-J` body.
- The unfrozen look sets in wt/wp-l: the builder's scan was not re-run, so "at least 6 steps and 7 lift-offs" and "controls ok" there are his.
- No 60 cm frozen set (none is in the worktree), so real bar items were played for 75-M-H-10 only.
- No closed-loop episode beyond the slow test's one, so nothing on stance chatter or on the fall-at-the-end case in a real episode.
- The builder's S1 numbers (23 steps, -0.0029 cycle, 1.007) beyond what the slow test asserts.
- Not the full suite and no other package's tests.
- Nothing on Colab itself.

Scratch files are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_ck_wave3/`: mutate.py, run_mut.py, mutate.out, table.py, items.py, items.out, items2.out, probes.py, test_extra.py, run_extra.py, and the scratch clone `clone/`.
