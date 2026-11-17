# integrate:TB6 (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

REPORT: the sixth list of rulings written into the Track B rules draft (worktree /home/user/mesozoic-wt/prereg-b, branch wt/prereg-b)

1. MERGE OF THE WORKING BRANCH
- wt/prereg-b (f8067851) was an ancestor of claude/mesozoic-trex-stage2-refine-4jffre (fccebe42). I merged with `git merge --no-ff -F <file>`, giving merge commit 3f7d32de (parents f8067851 and fccebe42). There were no conflicts.
- After the merge, `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` was empty, and `cmp` of the worktree's STAGE2_TRACKB_PLAN.md against the main checkout's printed nothing. The two copies of the draft were identical.

2. PER ITEM (STAGE2_TRACKB_PLAN.md only; line numbers are those at ac86c4d5, 3,078 lines)
Every new wording carries "[settled 16 Nov 2026, before any Track B run]" and "ruling n of the sixth list". Each replaced sentence is quoted once.

- **Ruling 1 (R0L's baseline)**
  - M1 rule, l.952-955: after "each must end *read* or *fell*" it adds 'or, for a condition without a reference, be "not run"'. The mark names the added phrase.
  - M1 head tags: l.936-938.
  - M1 Check, l.1048-1050: r0l.json's episodes are read, fell or `not_run` (`monitor2.baseline_problem`).

- **Ruling 3 (other exits before any episode), worded against the merged monitor2.py and train2.py**
  - New M1 paragraph, l.984-987. It is the lead's sentence: the three exits are each a refusal as ruling 6 has it; no attempt, and the trainer halts.
  - M1 author's note, rewritten at l.1003-1042. It gives the code's kinds (`sources`, `checkpoint`, `other_run`, `run`, `row`) and `EXIT_REFUSAL` = 8. It quotes the old "a code of its own ... it gives no number". It names the notes `monitor_<n>.refused.json` / `r0l.refused.json`, the trainer's `refused`/`refusal` fields, and halt exit 5.
  - The row's `s2rows.RowError` is named in the note as a refusal under the lead's ruling 10 of 17 Nov.
  - The note's "No parameter is ruled for that resume." is completed: after the refusal of the attempt that --retry-check let run, the two never-closed attempts still stand, and the halt names the parameter (the closing paragraph of the list).
  - M1 Check, l.1068-1080: `monitor2.EXIT_REFUSAL`, `refused`/`refusal`, `baseline_attempts`, the note files, and `run.json` `halt` naming --retry-check (on Colab, `HALT.json` `remedy`). The old phrase is quoted.
  - Pointers in M4 (l.1144-1145) and M5 (l.1184-1185).
  - K5 reading, l.1735-1740, and its Check (the kill-around-a-refusal test).

- **Ruling 2 (Z2 (c))**
  - Z2 head, l.1348-1350.
  - Part (c), l.1361-1366: "five degraded playbacks", with the scaled tail. "four degraded playbacks ... flat feet)" is quoted.
  - New author's note, l.1372-1384: why the scaled tail fails by the amplitude limit alone. The reference tail amplitude on the ladder's rungs is 0.120-0.125 rad (from the follow-up's kinematic playbacks, no record). On V6noF5, part (c) has four playbacks.
  - Z2 Check, l.1414-1417.

- **Ruling 4 (L12, converse case)**
  - The author's reading is replaced by the lead's sentence, l.2163-2171; the old reading is quoted in the mark.
  - L12 head, l.2126-2129; L12 Check, l.2196-2200.
  - Brief mark, l.284; change list item 30, l.2427-2428.

- **Frame**
  - Status, l.50-56.
  - New "What had been run" bullet, l.168-198. It covers: the follow-ups; the CW, CN and CT3 packages (tests, episodes on S1, tiny R5/R6 jobs on S1, 49+8 check episodes on frozen V0, none a record); and wt/wp-l at 2237ba9c. On that branch all six look sets have `freezes` empty and are blocked; I read only `freezes` and `blocked.by`.
  - Brief, l.330-336.
  - Tags entry, l.435-440.
  - Change list opening, l.2287-2290.
  - "three" now reads "four amendments" in the Status, brief, Tags entry and change list opening.
  - Pointers: third amendment item 7 ("the last two amendments", l.2665-2666); fifth amendment items 1, 4, 5 and 6, its two code bullets and its two paragraphs with the lead.

- **New "Amendment of 16 Nov 2026: the sixth list"** (l.2976-3078)
  - The four rulings and the closing paragraph for the code.
  - What the code holds today:
    - lookread.py holds the tail band (34afe717, on the branch since 19:52 UTC 16 Nov).
    - monitor2.py and train2.py hold the refusal (0ad2072d, bf9bc434; ruling 3 via 695e8429, 9c348ca7, eee35062, merged at ff04355a).
    - nb.py writes HALT.json on exit 5.
    - Ruling 1: `baseline_problem`.
    - Ruling 2: `lookread.DEGRADED` holds only four controls.
  - "What else changed", "Seen while this was written in" and "Seen and left".

FOR THE LEAD
1. **M1's ruled sentences do not name `s2rows.RowError`** (the row refusing while it is built). Your ruling 10 of 17 Nov makes it a refusal under ruling 3, and the code does so. It is written only in M1's note.
   - (a) Add a ruled sentence in M1.
   - (b) Leave it in the note.
   - I recommend (a).
2. **"After a refusal under --retry-check".** The trainer names the parameter only when the refused attempt was the one --retry-check let run (`after_halt`). It does not name it after a fresh attempt's refusal in a --retry-check session. Written in as the trainer reads it.
   - (a) Confirm that reading.
   - (b) Name the parameter after every refusal in such a session.
   - I recommend (a): only then is the parameter needed.
3. **Owed changes in other modules** (not mine):
   - monitor2.py `_checkpoint` checks only the shape of `files`. A record without a `layout` (or with a non-object `fingerprint`) gives exit 1 and a counted attempt (roundC__check_CT3_2.md finding 1). Your ruling 12 of 17 Nov owes the fix; until then ruling 3's "wrong shape" is not fully held.
   - Ruling 2's fifth control is not in `lookread.DEGRADED`/`controls`. It goes to the gates package's brief.
4. **Seen, no decision needed:**
   - The scaled tail fails only where the reference's tail amplitude is under 0.2 rad. The ladder's rungs are at 0.120-0.125 rad.
   - All six look sets on wt/wp-l are blocked by `start` or `nominal_to_v0.4`, so L12's decision on the other sets is now due for every body.

3. EXACT OUTPUTS SEEN
- Hash line, run after the merge and again after the last commit: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).
- `tests/test_stage2_monitor2.py -k "refus or exits or baseline"` (through tools/lowprio.sh, standing environment; the six tests stub every episode): `6 passed, 20 deselected in 20.05s`.
- Not run: train2's tests (they train on stubs), the full suite, and any episode, fit, training run or dry run.
- Not read: the worktrees of wt/wpc-cg and wt/wp-l, beyond the two set.json keys via `git show`.
- Not done: no independent review of this writing-in.

4. DIFF AND STATUS
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
  ` hardware/trex_form_study/STAGE2_TRACKB_PLAN.md | 408 ++++++++++++++++++++-----`
  ` 1 file changed, 325 insertions(+), 83 deletions(-)`
- `git status --porcelain`: empty in the worktree and in the main checkout.
- Scratch files (edit specs, staging script, test log) are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/trackb_r6/, outside the repository.

5. COMMITS, OLDEST FIRST
All are on wt/prereg-b, not pushed, no amend, and every message ends with the two given lines.
- 3f7d32de: merge of the working branch
- d8eca747: rulings 1 and 3 and the closing paragraph (M1, M4, M5, K5)
- 14fe4d7c: ruling 2 (Z2)
- eb9ec6f8: ruling 4 (L12, the brief's mark, item 30)
- 0373c4ee: the frame (Status, "What had been run", brief, Tags, change list opening)
- ac86c4d5: the fourth amendment with what the code holds today, and the pointers to it

Branch head: ac86c4d5 (wt/prereg-b).
