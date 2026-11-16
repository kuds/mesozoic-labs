# Track B rules draft: the lead's fourth list of rulings (16 Nov 2026, afternoon)

Working paper. The third list (`TRACKB_RULINGS3_16NOV.md`) was written into the draft on `wt/prereg-b`, reviewed
independently and fixed (`checkpoint/reviews/trackb_rules3__{integrate,review,fix}.md`). The fixer handed fourteen
points up. They are ruled here. **No Track B run, fit or training run has been made and no look set is frozen.**
"The reviewer's text" is the replacement text in `checkpoint/reviews/trackb_rules3__review.md` for that finding.

From here on, a later review's points of wording alone are fixed by the fixer and listed; only a point that changes
what a rule decides comes back to the lead.

1. **Ruling 6, where the caps come from.** Each cap is 1.5 times the bar runner's dry-run total **on the frozen set
   itself**, 60 cm included, committed before that row's first panel record. No number is fixed now. The stand-in
   numbers written into Q5 go, with the reviewer's text.
2. **Ruling 6, its five undecided cases.** The reviewer's text: the units run in the order 10, 0, 4, 6 ms; a unit
   starts by the bar's own unit rule; one run directory and one `budget.json` per cap. Fall attribution is not in
   the dry-run total: it runs last and is cut first. With ruling 1 the 10 ms unit always fits its cap, so "never
   cut" holds.
3. **Ruling 7, K5's resume.** K5 keeps the attempts logged for a check on the checkpoint it resumes from, so that
   M1's count of closed attempts survives a resume. The reviewer's sentence.
4. **Ruling 7, three cases in M1 and M4.** The reviewer's three sentences: the 18 episodes of a check that counts
   as failed are *errors*; a halt after two attempts that were never closed; what "a record" of a check is.
5. **Ruling 3, L8's Check.** It says what is equal: on the shared strides the move's times and slow variations are
   the window's; its frames are read by the join gates. (The fixer's option b.)
6. **Ruling 3, L10's boundary and test.** A stance is flat when it "ends after that instant and no later than the
   end of the move's last ramp". The Check reads the lift-off `foot_pitch` and `rolling` as gait a's spec gives
   them on the move's own step times. The reviewer's wording for both.
7. **Ruling 4, the body's other look sets.** The set whose `start` or `nominal_to_v0.4` fails holds no freeze.
   Whether that body's other look sets are frozen is then the lead's decision, recorded with its date and reason,
   because a body's sets are meant to agree (L12). No rule freezes or blocks them by itself.
8. **Ruling 5.** "Printed beside" applies to stop items only: walks and turns have no reading 1 in the bar's own
   record. On any other set the companion's counts equal the bar's.
9. **Ruling 9, "the share".** 4.0 core-hours less everything the fit has spent.
10. **Ruling 8.** The reviewer's text: the owner's answer is the field `budget_set_on`, written when the queue is
    made and before its first job. Y11's parenthesis stays.
11. **Ruling 1, "the Colab queue".** The reviewer's gloss.
12. **Ruling 2, L11's "every move".** Both places gain "but the turn in place's start and stop (L13)".
13. **The 15 Nov sentences that say a ruling or design is "not in the repository".** One dated sentence corrects
    them where they stand; the old sentences are kept as quoted history.
14. **Three times for the second list** (the file's heading, the draft's sentence, the commit). One dated sentence
    states them as the files and git give them. No rule depends on them.

Passed to the look work package (not a ruling): its `moves_look.py` still builds L10 as first committed, and
`move.command_phase` is null for `start` and `nominal_to_v0.4`. It is brought to L10 as settled (rulings 3 and 6
here) in its next round.

**Ruling 13 of the third list (the reward's wording)** is integrated in the same pass, from
`stage2/design/CE_RULE_WORDING.md` as it stands on `wt/wpc-ce` after the env's small fix round; the integrator names
the commit it read.
