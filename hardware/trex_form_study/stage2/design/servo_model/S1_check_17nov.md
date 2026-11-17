# check:S1 (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

**Verdict: merge as it stands.** I found no wrong number under any judged setting. Every formula I recomputed by hand matches the code. Undo is exact, bit for bit, after every setting. The table matches the design, the two hashes are the registered pair, and the merge is clean. The findings below are all low severity. They are best fixed in S1's planned fix round, before S2 relies on the layer. This is simulation only: everything I ran was plant-level physics in memory, with no record and no episode on a look set.

**Findings, most serious first**

1. **For the lead (medium). The gain-copy split has a side effect the builder did not mention (his item (a)).**
   - `servo.py:965-966` sets `p.kp`/`p.kv` at entry.
   - `control.Tracker` reads `p.kp, p.kv` into its ankle cap when it is built (`control.py:1240`).
   - On the judging side the tracker is built inside the episode (`panels` `TrackerRow.episode`), which is after entry. So a served R2 row's ankle cap takes the run's gains. The env's trackers are built before entry and keep the nominal ones.
   - My probe (`check_cap_copy.txt`): under J+ and W1 the ankle motors' kv copy is 11.28 before the first step, against 2.08 nominal. Under N and G5 it stays 2.08.
   - This contradicts design 2d's gain-copies table ("nominal on both sides") and the premise of ruling 21. No judged setting moves: N and D change only the tail's kv, and G5 leaves the ankle motors' gains alone.
   - Options:
     - Keep the split and amend 2d and ruling 21's note to say that served R2 rows' cap follows at J+ and W1.
     - Set the copies at the first step. That breaks the delay-queue rule at a mid-task start.
     - Set both copies and gains at entry. That breaks G8s's reset.
   - I agree the split is the only option that keeps both rules; the text needs amending.

2. **A defect (low). X6L's peak is written when the envelope binds, not at the first physics step.**
   - `ServoEnvelope.bind`/`restore` (`servo.py:747-748`, `753-761`) write `peak_run`. So under X6L the reset's forward pass sees ±31.5 N·m on the hip pitch.
   - `check_x6l_reset.txt`, at a start pose that loads the hip pitch:
     - Entered before the reset: forward-pass force −31.5 N·m. Entered after: −42 N·m.
     - First readings differ, and 0 of 20 per-step states are equal.
     - Under N the two orders agree on 20 of 20.
   - This contradicts HOOKS_RULE's own sentence ("the reset's forward pass, its first reading … are today's"). T4 (`test_stage2_servo.py:207`) encodes the current behaviour.
   - Impact is small: X6L is reported only and is not in T13's list.
   - Fix: write the nominal peak while the layer is not yet acting, and `peak_run` in `_start`. Then drop the X6L exception from T4's line 207 and add an entry-order test.

3. **A defect (low). Gaps in the tests.** I ran 36 single-line mutants of `servo.py` against the test file; results are in `mutants.txt`. These five broke behaviour and still passed all 52 tests:
   - **M08 and M36:** the force range ignores the sagged voltage, or scales the corner. T7's plant tests check the range only with `bus_r=0` (lines 489-503), and the V- test checks V alone (lines 505-526). Add a V- case at a fast state, checking the force range against `line_limits` at `lay.V`.
   - **M29:** the `mj_setConst` call removed. T10 checks only after exit (lines 649-656). Removing the call leaves `dof_M0`, `invweight0` and `acc0` stale. Add `env.setconst_changes(p.m) == []` inside the layer after the first step.
   - **M19:** the box servos' reading hold removed. No test covers it.
   - **M24:** the `inside_play_share` comparison inverted. No test covers it, although P41 reports this reading.
   - **M31:** the step-0 first-target rule removed. T8's box test (lines 549-564) cannot see it, because the neck's and tail's home ctrl is 0. Start from a nonzero command.
   - The current code is right on all five: my hand checks pass for the sag-to-line coupling, `setConst` inside the layer, the reading hold and the play share.
   - Every other mutant failed at least one test: the dead zone, the load term, the 10 ms filter, the root, the 0.92, P's speed, the lag, the hold period, every undo, the kv rule and copies, the offset sign, the ankle map, the ring, the panel tag, the zero rule, the panel's armature, the supply window, ankle friction, X6L's peak, the cap, the kv_scale base, the braking side, the sense steps and the floor count.

4. **A defect (low, fails open). An envelope swapped inside the layer is half-applied without a word.**
   - `_step` (`servo.py:1203-1216`) never checks `p.envelope is self._env`, and `exit` (1025) silently puts the old envelope back.
   - Probe (`check_edges.txt`): a 0.7-scaled `drive.Envelope` bound inside N leaves the X6 ranges at ±29.4 with no error.
   - Fix: an identity check per step that raises `ServoError`.

5. **A defect (low, fails open). The mass guard accepts any changed trunk.**
   - The guard at `servo.py:806-808` checks only "not as built". M+ was accepted with the trunk at ×0.9 (`check_mass.txt`).
   - Fix: check `body_mass[trunk]` against the nominal trunk mass × `params.mass`.

6. **For the lead (minor). Five cells where the design states no kind.** These are armature on the neck, kp_scale on the tail, kv_scale on the neck and tail, and latency_steps.
   - The code gives all five "the study's assumption".
   - For kp_scale tail and kv_scale neck and tail (`servo.py:207-215`) it also writes "nominal [A]" into the source text, which the design's table does not print.
   - The counts still give 1 + 4 + 6 + 28.

7. **For the lead (minor). Trap 8 wording.** The design says the layer "restores whatever step it found". The code refuses to enter over an existing step hook (`servo.py:947-948`). That is the stricter reading, and I would keep it.

8. **For the lead (S2, about kill and resume). `train2.SOURCE_FILES` does not list `servo.py`** (`train2.py:141-145`). Once `r5env` imports it, a resumed training run on a changed layer would not be flagged. Add `servo` (and `g8s`) in S2.

**The builder's items (a) to (e)**
- (a) See finding 1.
- (b) Defensible, but untested (M31).
- (c) Fine.
- (d) Open for the lead. The code follows "the rest as N", so X6L's slope is 6.23 where the X6-60's is 8.30.
- (e) Fine, apart from the X6L entry-order effect in finding 2.

**What I checked and found right**

*Formulas by hand* (`check_formulas.txt`):
- **Friction:** over 200 steps on F+ and D/3045, with τ̄ from my own recursion, the friction-loss arrays match to a relative gap of 4.5e-16. Damping is exact, and the ankle joints' friction and damping are 0.
- **Dead zone:** 12 leg drives at u/kp = ±0.2 to ±5 b, through the full layer step. Worst gap 3.6e-15 N·m, and the force is 0 inside the play.
- **Line with sag (V-):** P = 528.19 W gives V = 16.0697 V. The force ranges match exactly on step 1 and to 3.6e-15 on step 2 (one step of lag on P). The collapse case at 1905.6 W gives V0/2 = 10.5 V and is flagged.
- **Lag:** read back through the force, 1 − exp(−n·dt/T) holds to 3.4e-15 on all four classes.
- **Catalogue arithmetic:**
  - Corners 8.69174 and 8.01106; zeros 11.62389 and 9.21534.
  - c_cu 0.1852 and 0.1395.
  - Nominal corners 5.7596 and 4.1574; the 48 V slope 8.304.
  - Tail kv 2.7229 today and 3.5878 at 0.028 kg·m².
  - Box-servo gains under D, a training draw and J+ match by hand; the kv_scale multiplies the rule's kv and the copy keeps it.

*Exact undo* (`check_undo.txt`): all 15 named settings, D/3042 to D/3044 and train/1000003, with the caller's trunk mass where a setting has one. Each ran 300 zero-action steps, a reset inside, then 40 steps.
- After each, every MjModel array is unchanged, and so is the plant as built once the mass was undone.
- The gain copies are as before, and the plant and envelope objects are the same ones.
- The only differences were `plant.rng` (replaced by each reset) and the `envelope._qvel` view (live state). Neither comes from the layer.
- A run without the layer afterwards gave the same 300 per-step digests as before; `setconst_changes` is empty.

*The rest:*
- **T11:** the judge's peak, rated and no_load are nominal inside every setting; under X6L `peak_run` is 31.5 while `envelope.peak` stays 42.
- **Table** (`check_table.txt`): all 39 rows and 43 cells match my transcription of section 3 on today, nominal, low and high. All stated kinds match.
- **Table hash:** 342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57, which recomputes to the same value, with no numpy scalar in the hashed document.
- **Other checks:**
  - The draw follows the draw-order rule.
  - D holds armature and bus_r at nominal; P41's judged and reported lists are right.
  - `servo.py` writes no file, and nothing hashed globs it.
- **Cost** (`check_cost.txt`, plant-level, noisy; load average was about 2 to 3 when I started it):
  - Baseline 43.6 µs per step without the layer.
  - `today` +1.6, N +37.7, P+ +39.8, V- +19.6, D/3042 +37.2 µs per step.
  - The builder's probe shows all settings undone exactly and every digest equal afterwards; it writes only to the scratchpad.

**Exact outputs**
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- `tests/test_stage2_servo.py`: `52 passed in 2.20s`. Through the loader with an identical copy: `52 passed in 1.71s`.
- ruff on both files: `All checks passed!`. I removed the `.ruff_cache/` this created (it is ignored). Both checkouts are clean.

**What to bring in**
- Tip 6afd4f215c60fd0976d9842ae717b9747e8d99b6 on branch `wt/wpc-s1`. It is a fast-forward of `claude/mesozoic-trex-stage2-refine-4jffre`, whose tip fccebe42 is an ancestor.
- `git merge-tree --write-tree`: exit 0, tree ac15168dfbbcf36f70cd087a81d76c31e2355c13, no conflict.
- Only two files are added: `designlab/stage2/servo.py` and `tests/test_stage2_servo.py`.
- Both commits, b68648d2 and 6afd4f21, end with the two attribution lines and name no model elsewhere. No commit is missing them.

**What I did not check**
- The full test suite.
- The S2 to S4 items: T1 to T3 env parts, T13, and T14's delay queue.
- The builder's tracker-level stand probe; I re-checked at plant level with a zero action.
- The source strings word for word against the design, or the cited RES lines and maker pages. I compared numbers and kinds only.
- Timing on an idle machine.

My scripts and outputs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/check_s1/`:
- `check_formulas.py` and `.txt`
- `check_undo.py` and `.txt`
- `check_table.py` and `.txt`
- `check_edges.py` and `.txt`
- `check_cap_copy.py` and `.txt`
- `check_x6l_reset.py` and `.txt`
- `check_mass.py` and `.txt`
- `check_tail_kv.py` and `.txt`
- `check_cost.py` and `.txt`
- `check_setconst_inside.py`
- `mutants.txt`
- `mut/` (the loader and the mutants)

They are outside the repository.
