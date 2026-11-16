# The servo model for gate 1: where the design stands (16 Nov 2026)

Working papers. Nothing here is a rule or a result. Asked for by the owner (`DECISION_LOG.md` P39): a realistic
servo model in the training path (friction, gear play, torque and speed limits, voltage sag, delay; drawn at
random per run), for gate 1 of `BUY_GATE_DRAFT.md`.

**What exists.** Three readers' reports, gathered in parallel on 16 Nov, not reviewed:

| File | What it holds |
|---|---|
| `reader1_code.md` | What the simulation models today for each actuator, the delay and noise it draws, what enters a record's key, and where a richer model could attach |
| `reader2_hardware_facts.md` | Every fact the repository holds on the real actuators of the 75 cm build, the study's own assumptions, and what the bench sheet plans to measure |
| `reader3_outside_sources.md` | Makers' data and third-party measurements for the build's actuators, how others model such actuators for learned walking, and the ranges they draw from. 33 links, quoted as found and not checked a second time |

**The design note (as of 16 Nov, 16:00 UTC).** `SERVO_MODEL_DESIGN.md` is the note after two independent checks
and two revisions (834 lines). `REVIEW.md` is the first review of the first version, `FIX_REPORT.md` the first
revision's report, `RECHECK.md` the second independent check, `FINISH_REPORT.md` the second revision's report, and
`LEAD_RULINGS.md` the lead's rulings on its open choices. The second check's fixes and the rulings are in the note
and were not checked a third time. Its section 11 is the text for the owner; its appendix A holds five questions
for the maker of the leg drives, ready to send. Nothing in it is built.

**History.** The designer was first started at 03:05 UTC and stopped at 03:22 UTC before it had written the note,
by the lead's mistake (a watcher's time-out was read as the designer's result). The workflow was resumed at
13:13 UTC from the three readers' cached results.
