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

**What does not exist yet.** The design note and its independent review. The designer was started at 03:05 UTC
and was stopped at 03:22 UTC before it had written the note. That stop was the lead's mistake: a watcher that was
meant to report "the designer has finished" also ended, with the same exit code, when it timed out, and the lead
stopped the workflow without reading which of the two had happened. The designer's 17 minutes are lost. It was not
started again because the session was near the owner's usage line (about 87 % against 90 %).

**To finish it.** In the same session, resume the workflow run `wf_7aac8b25-fd4` from its saved script: the three
readers' results are cached, and the designer and the reviewer run. In a new session, give a designer these three
files and the brief in that script (the nine items the note must hold), then one independent reviewer.
