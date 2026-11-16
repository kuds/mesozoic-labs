# Track B rules draft: the lead's fifth list of rulings (16 Nov 2026, 16:28 UTC)

Working paper. The fourth list and the env's rule wording were written into the draft, reviewed independently and
fixed (`checkpoint/reviews/trackb_rules4__{integrate,review,fix}.md`). The draft is now on the working branch as
`STAGE2_TRACKB_PLAN.md`. The fixer handed four points up that change what a rule decides. They are ruled here and
are **not yet written into the draft**; they go in with the next round. No Track B run, fit or training run has
been made and no look set is frozen.

1. **M1, a record on a set with a "not run" condition.** A check's record holds, for each of the 18 episodes, an
   outcome or, for a condition without a reference, "not run". The rule, the author's note and the Check say so.
   Read to the letter as it stands, a body without a quick turn would fail every check.
2. **After the halt for two attempts that were never closed.** The lead has the fault repaired and resumes with a
   recorded parameter, which lets the check run once more per resume. A run that cannot be resumed is
   `unfinished: <reason>`.
3. **Q5, when S2-9 asks for the other delays after a row's cap is committed.** The three other delays get a cap of
   their own, with its own dry run, run directory and `budget.json`. The Check then holds per cap as written.
4. **L12, "that body's other look sets".** It takes in the sets of the body's `-X` head: L12 itself says a cut is
   taken for every look set of that body and of its `-X` head.

Confirmed as the fixer left them: rule E4's sentence on a plan given as a plan (the lead's ruling of 03:02 UTC);
the candidate's two caps are made again; K4 keeps the wording file's entry 8. Q5's run directories are named
`stage2/tb/bar/<run>/`, each with a `DRY_RUN.json` at its top.
