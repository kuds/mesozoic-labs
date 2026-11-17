<!-- Working paper, 16 Nov 2026: the fixer's report on applying that review. Nothing here is a rule. -->

## Fix round on the fifth list's writing-in (`STAGE2_TRACKB_PLAN.md`, worktree `/home/user/mesozoic-wt/prereg-b`, branch `wt/prereg-b`)

The working branch had not moved (`ffecc2c`, an ancestor of the branch): no merge was needed. Every finding was checked against the draft, the rulings and the code, and every one held. No rule of the lead's was changed. I ran no episode, fit, training run, dry run or test. I ran the review's pure-function tail probe once (synthetic numbers) and the hash line, which printed the registered pair `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. Line numbers are the file's at `f806785` (2,836 lines).

### Per finding

1. **"What had been run": the look branch moved. Fixed (bullet, lines 132-161).** `git log wt/wp-l` shows `33652cb` at 19:15:57 UTC; its `set.json` for `V6` on `75-M-H-10` has `freezes: []`, `blocked.by: "start"` and `not_available.start.reason: "no transition within the gates"`. The bullet now names the head the writing-in named (`de96fc0`), the later commit and what its record says, and what the review and this fix round ran. The old clause is quoted once. I did not call it "L12's case": I did not read whether the `start` still fails under every cut.
2. **M1's Check for a refusal. Fixed (M1 Check, lines 981-991; note, lines 955-960).** `monitor2.main` returns 1 for every `MonitorError`; `train2._rollback` moves a later check's entry to `discarded_after_resume`. The Check now reads: the entry logged before the monitor started, closed with the code `monitor2.py` is to name (no ruling gives the number), in `monitor[]` or in that session's `discarded_after_resume`. The note no longer says the entry always stays in `monitor[]`.
3. **Ruling 2's Checks and K5's reading. Fixed (M1 Check, lines 981-986; M1 note, line 944; K5, lines 1616-1620; K5 Check, lines 1625-1628).** M1's Check: every attempt made after two never-closed attempts is marked `after_halt`, lies in a session whose `retry_check` is true, and is that session's only one. K5 gains the case of the allowed attempt closed without a record: it counts toward M1's two; as the first of them the run halts again. This follows M1's count and ruling 2's "once more per resume", and `train2.py` lines 1256-1263 do it.
4. **Q5. Fixed, four points.**
   - "is made" now reads "is committed", old sentence quoted (lines 1405-1413).
   - `DRY_RUN.json` gets four key names, labelled the author's: `commands`, `rows`, `total_core_h`, `set_freeze` (lines 1439-1445).
   - The Check's last item is read from files: a record at 0, 4 or 6 ms beside a first `DRY_RUN.json` that names 10 ms rows alone; old item quoted (lines 1494-1511).
   - Change list item 15 says "the first panel record the cap covers" (line 2236).
5. **Ruling 1's Check. Fixed (M1 Check, lines 974-980).** It names where a condition without a reference is read: the bar's planner on the run's set (`bar.BarPlanner.missing`, from `refs.json`, `transitions.json` and `set.json` `not_available`; H3).
6. **Small points. Fixed.**
   - Line 133: "on a Stage 2 set" added (now line 134), and in the third amendment's opening (line 2728).
   - U3's reading: "from the same fit as the tail phase", old words quoted (line 1157).
   - E4's mark quotes the list's own words; it names one sentence where three stand (line 470). The same in the amendment (line 2755).
   - Tags entry: says which confirmations carry a mark; K4 carries none (lines 389-394).

The third amendment gained two paragraphs (lines 2809-2836): what this fix round changed, with the two replaced Check phrases quoted, and the points that stay with the lead. Its account of the code gained three facts I read in `train2.py` and `monitor2.py` (lines 2772-2779).

### Points of wording I settled
- The four key names of `DRY_RUN.json`. The lead may rename them before a builder writes the file.
- K5's sentence on the allowed attempt closed without a record (finding 3).
- Q5's reading turns on the commit of the first dry run, not its making.
- No number is given for the monitor's refusal exit code; the Check says the module "is to name" it.
- The amendment says the scaled tail, not the tail held still, is the playback that would fail by the new limit alone (a fact of the probe; the control itself is the lead's, point 2 below).

### For the lead (each changes what a rule decides; nothing written in but point 4, which the integrator wrote in)
1. **M1, R0L's baseline with a "not run" condition.** "Each must end *read* or *fell*" names no condition without a reference. (a) Add "or, for a condition without a reference, is "not run""; (b) leave. I recommend (a): `monitor2.baseline_problem` already takes it.
2. **Z2 (c), a control for the new tail limit.** (a) A fifth degraded playback; (b) leave it to `lookread`'s test. I recommend (a), with a tail scaled to a tenth about its mean. My run of the probe: the scaled tail (0.0125 rad) keeps its phase and passes today's term; a tail held at 0 already fails today by phase (difference +0.27 cycle).
3. **Exits of the monitor before any episode that ruling 6 does not list:** a missing `run.json`; a checkpoint record that cannot be read; a fingerprint not reproduced at the row's load (`s2load.CheckpointError`, which `monitor2.main` does not catch). (a) Each is a refusal: no attempt, halt; (b) each is a counted attempt. I recommend (a).
4. **L12, the converse case** (the failing set is the `-X` head's own). Written in as the author's reading: yes. To confirm. The look branch's new `V6` build is recorded as blocked by its `start`.

Seen and left: `monitor2.is_record` takes `not_run` on any episode, without reading whether the condition has a reference (a code point, now in the amendment). The code changes for rulings 5 and 6 are still owed.

### Commits (not pushed, no amend, no history rewritten)
- `87f580d` what had been run
- `a88e167` M1's and K5's Checks and notes
- `1875976` Q5's reading and Check
- `6363cc8` four small points of wording
- `f806785` the third amendment's account

Branch head: `f806785`. Commits on the branch not in the working branch: 12; none lacks the two required lines, and none names a model elsewhere.

`git status --porcelain`: empty, in the worktree and in the main checkout.

`git diff --stat claude/mesozoic-trex-stage2-refine-4jffre wt/prereg-b`:
```
 hardware/trex_form_study/STAGE2_TRACKB_PLAN.md | 644 +++++++++++++++++++------
 1 file changed, 490 insertions(+), 154 deletions(-)
```

Edit specs and helper scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trackb_r5_fix/`.
