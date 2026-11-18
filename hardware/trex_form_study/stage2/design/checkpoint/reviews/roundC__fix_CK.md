# fix:CK (roundC, wave 3): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WORK PACKAGE CK (the look reading): FIXER'S REPORT
Worktree /home/user/mesozoic-wt/wpc-ck, branch wt/wpc-ck, head 9caa811. Simulation only: everything below is kinematic playback, synthetic series or tests, except the package's own slow test (one closed-loop episode on the view S1, no record). Nothing pushed, no history rewritten, nothing written in the main checkout (its status is empty, head fb421ba). Only the two CK files changed, plus one merge of the working branch.

0. THE MERGE
The working branch had moved (cd68f2c to fb421ba, 41 commits). Merged with `git merge --no-ff --no-commit` ("Automatic merge went well"), committed from a message file as eb477ca. After it, the diff against the working branch lists only the two CK files.

1. PER FINDING

Finding 1 (two merge commits without the attribution lines): for the lead, no code change.
- cc8eec5 and 3182a6e (the earlier builder's merges) still lack the two lines. I did not rewrite them.
- Recommendation: cherry-pick the nine content commits in this order: 6d6d38d, 11a23b4, 0ccabf2, 62751fc, 01fccec, 8b8cf5a, 2a64f58, 5b4e304, 9caa811.
- Checked in a scratch clone (`git merge-tree --write-tree --merge-base=<c>^`, chained on fb421ba): all nine apply without conflict, and the final tree 8973e592 is the branch head's tree. Each touches only the two CK files.
- Alternative: a plain merge, which brings the two commits in.

Finding 2 (test gaps): fixed, tests only, commit 8b8cf5a.
- (a) `test_the_bob_is_the_mean_of_the_own_steps_ranges_not_the_windows_range`.
- (b) `test_too_few_own_steps_fail_bob_tail_and_toe_off_and_are_no_error` now also takes 3 steps with 4 lift-offs.
- (c) `test_the_ladders_limits_each_bite_alone_and_the_tails_difference_is_wrapped` (no model): the 0.02 rad and the 10 % toe-off limits each fail where the other holds, and the tail is compared across the wrap. `test_the_controls_verdict_needs_each_of_its_parts`: a copy failing two terms or none, a cut not labelled or failing no term, an unchanged playback without the exact 1 or 0.
- The reviewer's third proposed test is not taken, as he said. I measured the two hip-roll anchors 0.0 m apart with the trunk rolled 7 degrees on 75-M-H-10, 60-N-T-10 and 75-M-H-10-X.
- Evidence: the reviewer's 56 single faults plus 11 of mine, run against the tests at the head: "caught 65, not caught 2". The two are the reviewer's equivalent ones (the hip anchors; `_Played.q` already holds the last pose). All seven of his nine that mattered are now caught.

Finding 3 (a window for a walk episode that fell in its last two updates): fixed, commit 2a64f58.
- `lookread.window` returns None for a walk item when `rec.out["fell"]` is set. A recorder without `out` (a kinematic playback's) did not fall. A turn item's window is unchanged: the bar reads it whenever the trace spans it, also after a later fall (bar.py, the `bar_turn` branch).
- Test: `test_a_walk_episode_that_fell_has_no_window`. It fails on the module before. The slow test pins `rec.out["fell"]` on a live recorder.

Finding 4 (a playback of just the window misses an event on its first physics step): fixed, commit 9caa811, by the reviewer's first option.
- `lookread.played(..., lead_s=0.0)` can begin whole poses earlier, never before the track's time 0. It returns `window` and `lead_s` as well.
- `lookread.playback` plays with `LEAD_S = 0.1` s and reads the window alone. Its poses are those of a playback that begins with the window. Its result holds `lead_s`.
- Tests: `test_a_playback_counts_a_lift_off_or_touchdown_on_the_windows_first_step` and the lead's arrays in `test_playback_through_the_bars_player_is_the_rungs_own_reading`. Both fail on the module before: without the lead the first test reads (steps, lift-offs, strides) (23, 24, 11) where the trace reads (23, 25, 12).
- Effect, kinematic, no record:
  - The six ladder rungs through `playback` reproduce the builder's table to the digits shown. Against the module before the fix, counts are equal and numbers within 5e-16, for the 10 s and the 3 s window.
  - The 10 walk and turn items of F25F9 (75-M-H-10) that build here: counts equal, numbers within 1e-15.
- See "for the lead" 1.

Finding 5 (`series`' bounds check one off): fixed, commit 2a64f58.
- `(n - 1) * every >= steps` is now refused: every pose needs its load. A live recorder always satisfies it (control.py records the load of step k before the update of step k; `steps = k + 1`), and so does `played`.
- Test: the bounds in `test_the_series_are_the_look_arrays_own_columns_and_leave_the_plant_alone` (5 poses with 40 loads, 6 with 50 and 1 with 0 refused; 5 with 41 taken). It fails on the module before.
- The package's own contact test used the refused case (201 poses, 2000 loads); it now takes 200.

Finding 6 (rule texts holding the module's readings as rule text): fixed, commit 5b4e304.
- Each of `RULES`' four texts opens with the plan's rule and holds, after one mark (`MODULE_READING`: "This module's own reading, which is not the rule's wording: "), the module's sentences. The four the reviewer named are behind the mark.
- Test: `test_the_rule_texts_mark_what_is_the_modules_own_reading`. It fails on the texts before.
- See "for the lead" 2.

Finding 7 (`played` turns every `ValueError` of `bar._Played` into "not played"): not a defect in CK. The brief asks for exactly this.
- No behaviour change. The docstrings of `played` and `playback` now say that for a bar walk or turn item it is a fault of the definition (U5), never "no look reading".
- For the lead: CW's brief should say so.

Finding 8 (skip reasons name a usage error): fixed, commit 8b8cf5a.
- The reasons now read `python tools/restore_inputs.py restore --group ...`, from one table `HINTS`.
- Test: `test_the_skip_reasons_name_commands_the_restore_tool_takes` asks the tool's own parser. With the old form put back it fails ("'--group' != 'restore'").

Finding 9 (branch behind): done by the merge eb477ca.

FOR THE LEAD
Mine, from this round:
1. `LEAD_S = 0.1` s is a number no rule gives. On clean support flags any lead of one physics step or more gives the same events; I took five poses, five times the debounce. Alternatives: no lead and a corrected docstring (the reviewer's second option), or another length. Commit 9caa811 is last and can be left out alone; the tree before it (5b4e304) passes its tests ("59 passed, 1 skipped").
2. I marked the texts for U1 (measures) and U2 (window) the same way, which the reviewer did not ask for. The part before each mark paraphrases the rule closely; it is not verbatim and keeps short glosses in brackets. If you want the plan's words verbatim there, say so before state B: the file is hashed into every Track B row.
3. `lookread.py`'s sha256 is now 3bc409e5...834b40. No record holds it yet.

The builder's points a to m: I decided none. The reviewer's choices stand as his recommendations; the code is as built on each. Still open for you:
- b: `on_command_at_stop` needs an owner (reviewer: CW's brief).
- c: design 3.7's toe-off sentence is owed a correction; the code holds the rule (0.02 rad and 10 %).
- e: the tail's amplitude is banded by nothing. The reviewer recommends a ruling before state B: fail the term when the robot's fitted amplitude is under 0.02 rad while the reference's is read. Not built.
- f: the bob has no "not read" threshold (reviewer: CW records only the numbers outside the look sets).
- g: contact is read literally (reviewer: as built; CW records the short air runs).
- h: CG lists each item's reference toe-off and its side of 0.02 rad.
- m: design 3.5's "3 s of walking" is owed a correction.

For CW and CG, what changed in the API:
- `played` takes `lead_s` and returns `window` and `lead_s`; `playback` returns `lead_s`; new constants `LEAD_S` and `MODULE_READING`.
- `window` reads the fall itself, so the wrapper need not check `out["fell"]` first.
- `series` refuses a pose past the last load.
- For gate (c) on what `playback` reads: `a = played(..., lead_s=lookread.LEAD_S)`, then `controls(..., t0=a["t0"], ta=a["window"][0], tb=a["window"][1])`.

Not done: no `-J` body; no 60 cm frozen set; no look set played (none is frozen); not the full suite and no other package's tests; the builder's scan of the unfrozen look sets not re-run.

2. EXACT TEST LINES (head 9caa811, standing environment, through tools/lowprio.sh, `--basetemp` under the scratch directory)
- `pytest -p no:cacheprovider tests/test_stage2_lookread.py`: "60 passed, 1 skipped in 7.42s", exit 0.
- The same with STAGE2_SLOW=1: "61 passed in 10.45s", exit 0.
- `ruff check --no-cache` on the two files: "All checks passed!".
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).
- A scratch export of the head without the ignored inputs: "43 passed, 18 skipped in 3.18s"; every skip names the restore command.
- At each commit of the round the fast tests passed: 8b8cf5a "57 passed, 1 skipped"; 2a64f58 "58 passed, 1 skipped"; 5b4e304 "59 passed, 1 skipped"; 9caa811 "60 passed, 1 skipped".
- Single faults at the head: "caught 65, not caught 2".
- My earlier runs in this round used pytest's default temporary directory (/tmp/pytest-of-root), as the builder's did; nothing was written in the worktree beyond the tracked files (checked: no untracked or ignored file is newer than the round's start).

3. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/lookread.py   |  649 ++++++++++++
 .../trex_form_study/tests/test_stage2_lookread.py  | 1104 ++++++++++++++++++++
 2 files changed, 1753 insertions(+)
`git status --porcelain`: empty.

4. COMMITS OF THIS ROUND, OLDEST FIRST
- eb477ca4fb8826d9498b334ef5d994a7e19a52c0 merge of the working branch (fb421ba)
- 8b8cf5a6e6364f819b28da54a3b3301869a53e28 tests only: the gaps the single faults showed, and the skip reasons' command (findings 2, 8)
- 2a64f5845edbf61d03360bbb946b66ec8a1ce282 a walk episode that fell has no window; every pose needs its load (findings 3, 5)
- 5b4e304131516cbbc7a7e34d4924dd9c8190665d the rule texts mark the module's own reading; the "not played" docstrings (findings 6, 7)
- 9caa811cb25d4f621a1e18f6f264af1e830e08f2 the reference's playback begins before the window (finding 4)
Head: 9caa811cb25d4f621a1e18f6f264af1e830e08f2.
All five end with exactly the two lines and name no AI model elsewhere. On the branch, two commits lack them: cc8eec59b5453ae28805cdce9cadf8605c7009d2 and 3182a6e8ce4478aef9c7fdbb04775b3dfadf76c8 (the earlier builder's merges of wt/wp-r and wt/wp-b). The other eleven unmerged commits have them.

Files changed: /home/user/mesozoic-wt/wpc-ck/hardware/trex_form_study/designlab/stage2/lookread.py and /home/user/mesozoic-wt/wpc-ck/hardware/trex_form_study/tests/test_stage2_lookread.py.
Scratch (not tracked), all in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/fix_ck/: mutate.py and mutate_head.out (the single faults), table.py (the six rungs), items.py (the F25F9 items), build_mod_states.py and build_test_states.py (the per-commit file states), clone/ (the cherry-pick check).
