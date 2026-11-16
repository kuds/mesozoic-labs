# Track B rules draft: the lead's sixth list of rulings (16 Nov 2026, 19:52 UTC)

Working paper. The fifth list was written into the draft, reviewed independently and fixed
(`checkpoint/reviews/trackb_rules5__{integrate,review,fix}.md`); the draft on the working branch holds it. Four
points came back that change what a rule decides. They are ruled here and are **not yet written into the draft**.
No Track B run, fit or training run has been made and no look set is frozen.

1. **M1, R0L's baseline on a set with a condition that has no reference.** Each of the baseline's 18 episodes must
   end *read* or *fell* or, for a condition without a reference, be "not run". The monitor's code already reads it
   so.
2. **Z2 (c), a control for the tail's new limit.** A fifth degraded playback joins the controls: the tail scaled to
   a tenth about its mean. It must fail the tail term and no other. It goes to the gates package's brief.
3. **Other exits of the monitor before any episode** (a missing `run.json`; a checkpoint record that cannot be read
   or has the wrong shape; a fingerprint not reproduced at the row's load). Each is a refusal as ruling 6 of the
   fifth list has it: no attempt, and the trainer halts. Owed as a small change to the monitor.
4. **L12, the converse case.** When the set whose start fails is the `-X` head's own, the lead's decision also
   takes in the plain head's sets of that body. Confirmed as the integrator wrote it.

Also ruled, for the code: the monitor keeps its note files `monitor_<n>.refused.json` and `r0l.refused.json`; the
halt message after a refusal under `--retry-check` names that parameter.
