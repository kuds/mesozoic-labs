# fix:TB6 (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

REPORT: TB6 fix round, applying the review of the sixth list's writing-in (worktree /home/user/mesozoic-wt/prereg-b, branch wt/prereg-b)

All seven defect findings are fixed in STAGE2_TRACKB_PLAN.md, the only file changed. I left finding 5 for the lead. The working branch moved twice during this work, so I merged it twice, and I re-read the code it brought in.

MERGES (both clean, made with `git merge --no-ff -F`)
- db469746 merges a5aac7a8. It brings in the follow-up before the acceptance run on the notebook, trainer and rows (CN, CT, CW), merged 05:16 UTC.
- ce0b8c75 merges 6c62f30e. It brings in the gates package (3a42df8a, 05:25 UTC) and the lead's rulings 13 to 20 of 17 Nov.
- Neither merge changed the draft on the working branch.
- 6c62f30e is still the working-branch tip, and it is an ancestor of HEAD.

What the merges changed in the code the draft describes:
- `monitor2._checkpoint` now refuses a record without a `layout` or `fingerprint` object (fd1f88ff, ruling 12 of 17 Nov).
- train2 now carries an allowance for moved sources (a24a9c6a): `sources_allowed`, `SOURCE_ALLOWANCE_RULE`.
- A halt for moved sources now names --allow-source-change.
- nb.py's HALT.json `remedy` can now hold `allow_source_change` (`nb.halt_remedy`). LEAD.json's `retry_check` is now passed on only in the session that resumes the halt whose remedy names it.
- gates2.py holds the fifth control of Z2 (c): `tail_tenth`, with `TAIL_SCALE` 0.1.
- So the review's wording for fix 1 ("which the code does not yet refuse") would now be false. I worded the fix against the merged code instead.

PER FINDING
1. **Fixed.**
   - M1's note now says the "wrong shape" covers the `files`, and also `layout` and `fingerprint` being objects. It cites ruling 12 of 17 Nov and says the code holds it since dfd2fbc2. The old phrase is quoted in a mark.
   - Carried to the fourth amendment:
     - ruling 3's line ("but for one wrong shape" is dropped, with a mark);
     - "What else changed" (l.3057);
     - the "Not yet held" bullet, rewritten to say the shape is now held, by fd1f88ff merged at dfd2fbc2.
2. **Fixed.** The draft now says 2237ba9c was wt/wp-l's head "when it was read, 04:23 UTC on 17 Nov". A mark notes that the head at 04:19 was a024cbe8, with the same stage2/sets/.
3. **Fixed.** M5's pointer now adds: "after the second, where the refused attempt was the one that parameter let run, again with it (M1's note, K5)", with a mark.
4. **Fixed.** L12's ruled sentence now uses the lead's words ("whose start fails"). After it comes a marked author's reading: "start" means `start` or `nominal_to_v0.4`. The old wording is quoted in the mark.
5. **For the lead. Nothing changed.** K9 has no pointer to M1 for a fingerprint failure at a monitor check, which is now a refusal and halts the run.
   - (a) Add "(at a check of the monitor: a refusal, M1)".
   - (b) Leave it.
   - I recommend (a).
6. **Fixed.** "the three paragraphs above" now reads "the paragraphs above", with a mark.
7. **Fixed.** The note now says the ruling names only a missing run.json. A run.json that cannot be read or is no JSON object is the code's wider kind `run`. The old phrase is quoted in a mark.

ALSO CHANGED, so the draft stays true against the merged code (each change marked as the fix round's)
- M1's note: a halt after a refusal for moved sources also names K4's --allow-source-change (ruling 12 of 17 Nov).
- The amendment's account of the code is now read at 6c62f30e:
  - the source allowance, and the halt that names --allow-source-change;
  - HALT.json `remedy` `allow_source_change`, and LEAD.json `retry_check` only for its own halt;
  - Ruling 2's bullet: gates2's `tail_tenth` (the old "No code of gate G9 is on the working branch" is quoted in a mark).
- "What had been run" now says what the review and this fix round ran, and records the two merges. It also says that, by the lead's rulings of 05:27 UTC, no look set was frozen and no closed-loop episode had been run on one.
- New paragraph in the amendment, "Seen in the fix round", for the lead (item 1 below).

FOR THE LEAD
1. **K4 against ruling 12 of 17 Nov.** The trainer now allows a session on sources that an earlier session given --allow-source-change ran on. K4 still reads "Changed sources (...): unless `--allow-source-change`". I wrote nothing into K4, because ruling 12 is not part of the sixth list.
   - (a) Write the allowance into K4, and the moved-sources halt into M1's Check, as a separate amendment.
   - (b) Leave it only in M1's note and the account of the code.
   - I recommend (a).
2. **Kind `run` is wider than ruling 3.** The code refuses an unreadable or non-object run.json, not only a missing one. The draft now marks this as the code's reading.
   - (a) Confirm it.
   - (b) Narrow it.
   - I recommend (a).
3. Items 1 and 2 of the first report stand, unchanged; the reviewer calls both the lead's choice. They are `s2rows.RowError` named only in M1's note, and the retry-check halt reading (`after_halt`).
4. The first report's item 3 ("owed in other modules") is no longer owed: both fixes are now merged (fd1f88ff and 3a42df8a).
5. Rulings 13 to 20 of 17 Nov are not written into the draft; they are not part of this item.

TESTS AND CHECKS (worktree, standard environment, tests through tools/lowprio.sh)
- Text check of the findings, `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/fix_tb6/check_review.py`:
  - On HEAD's draft: "10 passed, 0 failed".
  - On ac86c4d5's draft (before the fix): "1 passed, 9 failed". The one pass is the check that the merged monitor2 refuses a missing `layout`/`fingerprint`; that check fails against fccebe42's monitor2.
- `tests/test_stage2_monitor2.py`: "24 passed, 2 skipped in 23.55s".
  - Skip reasons: "lookbar is merged: the stand-in is no longer reached"; "Stage 1's arrays are not restored".
- `tests/test_stage2_train2.py -k "refuses_to_run_is_no_attempt or kill_around_a_refusal or refusal_without_its_note or passes_its_allow_source_change or allowance_for_moved_sources or halt_for_moved_sources"` (stub env, stub monitor): "6 passed, 50 deselected in 72.87s (0:01:12)".
- Hash line, after each merge and after the last commit: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- Not run: the full suite, the gates2, nb and lookbar tests, and any episode, fit, training run or dry run.
- Not read: the reports of the follow-up and gates packages, beyond the code the draft describes.

DIFF AND STATUS
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
  ` hardware/trex_form_study/STAGE2_TRACKB_PLAN.md | 495 ++++++++++++++++++++-----`
  ` 1 file changed, 405 insertions(+), 90 deletions(-)`
- `git status --porcelain` in the worktree: empty, and `--ignored` is empty too.
- The main checkout, which I did not touch, shows three untracked files that are not mine: `stage2/design/checkpoint/reviews/roundC__check_L_17nov.md`, `roundC__fix_CF_17nov.md`, `stage2/design/servo_model/S1_fix_17nov.md`.

COMMITS, OLDEST FIRST
Nothing pushed, nothing amended. Every message ends with the two given lines, and no model is named anywhere else.
- db469746: merge of the working branch at a5aac7a8
- ce0b8c75: merge of the working branch at 6c62f30e
- 63aedb16: M1's note (findings 1, 6, 7; the halt for moved sources)
- 47eeec34: M5's pointer (finding 3)
- 5a37a238: L12 in the ruling's words (finding 4)
- 31a1d298: "What had been run" (finding 2; the review's and the fix round's runs)
- 25ac4eab: the fourth amendment's account of the merged code (finding 1 carried; the paragraph for the lead)

Branch head: 25ac4eab (wt/prereg-b).

The check script, logs and patches are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/fix_tb6/ (outside the repository).
