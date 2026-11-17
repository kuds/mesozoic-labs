# build:L (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

WP L (look sets) builder report. Worktree /home/user/mesozoic-wt/wp-l, branch wt/wp-l. Simulation only: everything below is prescribed motion (kinematic builds and Stage 0 screens). No physics episode of any kind was run, nothing was frozen and nothing was pushed.

**Status in brief.** No look set can be frozen as built. On every set and body, the `start` passes all its entry gates but goes over the knee's Stage 0 speed limit. On the 60 cm bodies `nominal_to_v0.4` is over that limit too. By A8/L12 each set is therefore stopped and goes to the lead. Anything that needs a frozen V6 or V6noF5 has to wait for a lead ruling and a rebuild of the moves: that is R5 on V6, and R2L on V6noF5 (Q1). The walks can be reused.

**0. What the earlier builder left.** Its commits on the branch are the look laws `designlab/stage2/moves_look.py` (L1–L8, L10 as settled, L13, J1–J3, the K-stride walk), the `LOOK_SETS` entries in `sets.py`, the tests, `V6/tail_phase.json`, and the full records of V6 and V6noF5 on 75-M-H-10. It also left an untracked, half-built V6/60-N-T-10: five of six walks, killed by the reboot and resumed here. Its code needed no change: no defect found, so `moves_look.py`, `sets.py` and the tests are as the earlier builder committed them.

**1. Merges.**
- `75d43036`: merge of the working branch at `c7902f2`, done first.
- `2237ba9c`: a second merge after the builds, at `fccebe42`.
- Neither had a conflict. The working branch touches no file of this package, and none of the modules a look set's records hash (sets, moves, moves_look, setscreen, screen, look, setio, s2io).

**2. Items**

**(a) Builds.** All sets were built in L14's order with `sets build --procs 1`, then `setscreen run --procs 1`, then `verify`. The six set/body pairs (logs `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpl/{build,screen,verify}_<set>_<body>.log`):

**V6 / 75-M-H-10** (earlier builder's records; checked here)
- Walks: all 6 within Stage 0. Windows of 9 to 13 strides; `nominal` and `v0.4` equal the ladder rungs.
- STOPPED by `start`: entry gates all pass; Stage 0 knee speed 1.1261 > 1.0 blocks the freeze.
- Available: `nominal_to_v0.3/.35/.4/.45` (knee 0.8775 / 0.9022 / 0.9188 / 0.9353); `stop2_v0.4` (join_command warm-up ZMP 3.8682 mm, limit 4.0); `stop1_v0.4`.
- Not available:
  - `nominal_to_v0.5`: zmp_error 4.158 mm (ramp 1) and 4.045 mm (ramp 2), limit 4.0 mm. Fallback `v0.45_to_v0.5` fails join_a (ZMP 2.280 mm) and zmp_error.
  - `turn90_0.35` and `turn90_0.5`, both at stride factor 1: join_command warm-up ZMP 5.5434 mm (left) and 5.6951 mm (right), limit 4.0.
  - `turn90_pi4` (factor 0.8): hip_yaw peak 1.1444 / 1.1455 and knee speed 1.0384 / 1.0386; the left turn also fails join_command at 4.3974 mm.
  - `turn90_1` (factor 0.6): hip_roll, hip_pitch and hip_yaw peaks (hip_yaw 1.6693 / 1.6710), knee speed 1.3413 / 1.3414, and heading_at_slowest_clock.

**V6noF5 / 75-M-H-10** (earlier builder's records)
- Walks all within Stage 0.
- STOPPED by `start`: knee speed 1.1261.
- Available: the four changes; both stops; `turn90_0.35` and `turn90_0.5` at stride factor 1 (thigh twist about 0.447 and 0.647; warm-up ZMP 3.5173 / 2.3852 mm).
- Not available: `nominal_to_v0.5` (zmp_error 4.312 / 4.173 mm); `turn90_pi4` and `turn90_1`, over leg drives as on V6.

**V6 / 60-N-T-10** (commit `2a5466ab`)
- `v0.5` dropped: knee speed 1.0163. The other walks are within Stage 0.
- `start`: knee speed 1.2340.
- `nominal_to_v0.4`: knee speed 1.0958 with one stride of ramp, 1.0678 with two.
- STOPPED; the builder built no other move.

**V6noF5 / 60-N-T-10** (`ba439717`)
- `v0.5` dropped: knee speed 1.0406.
- `start`: knee speed 1.2340.
- `nominal_to_v0.4`: knee speed 1.0964 (ramp 1), 1.0679 (ramp 2).
- STOPPED.

**V6 / 75-M-H-10-X** (`a9ea7f58`)
- All 6 walks within Stage 0.
- `nominal_to_v0.4` available: knee speed 0.9044.
- STOPPED by `start`: knee speed 1.1245.
- Neck yaw and jaw stay exactly at their home angles in every file (largest deviation 0.0).

**V6 / 60-N-T-10-X** (`a024cbe8`)
- All 6 walks within Stage 0, `v0.5` included (knee speed 0.9954). So the two heads of 60-N-T-10 differ in which rungs they hold.
- `start`: knee speed 1.2232.
- `nominal_to_v0.4`: knee speed 1.0609 (ramp 1), 1.0337 (ramp 2).
- STOPPED.

On every set, every move entry names its rules: `S2A-gates`, S2L-L1 to L8, L10, J1 to J3. `command_phase` holds the instant: −0.25 for a start, 0.0 for a change.

**(b) Why the start fails** (diagnosed on V6/75, where V6noF5 gives the same 1.1261, so F5 is not the cause).
- The peak is the left knee at 8.511 rad/s, at phase 0.0375, the first swing (u = 0.12). The limit is 8.6917 / 1.15 = 7.558 rad/s; the walk's own peak is 5.725 rad/s.
- The first swing lifts a flat foot (L10) from an almost unbent stand: the amplitude ramp is at about 0.06 at lift-off. It does so with V6's swing time of 0.356 s (F3) and lift peak 0.375 (F4).
- For comparison, V0's start (lift peak 0.5, swing 0.454 s) reads 0.5671.
- This is the rules as written, not a code fault.
- Diagnostic run (scratch only, not a record): with the lift peak mixed from the stand's 0.5 by the amplitude ramp at each lift-off, the knee peak falls to 6.398 rad/s (ratio 0.8465).
- I did not diagnose why `nominal_to_v0.4` is over the limit on the 60 cm bodies.

**For the lead (start):**
- (a) Leave it: no look set is freezable.
- (b) Rule that a start's lift law, and possibly its duty (L1/L4), mixes the stand's law (Stage 1's) by the amplitude ramp. That is a change after a result and needs the label.
- (c) Build the start differently.
- Recommendation: (b), then rebuild the moves of all six sets. A change's failure on the 60 cm bodies would still be open.

**(c) The V6 turns at 0.35 and 0.5 rad/s on 75-M-H-10 fail join_command.**
- The frames over the limit are all within 3 frames of planned events; away from events the largest is 0.723 mm (from `join_events`).
- This is L12's trigger. The cut needs the independent review of the join code on record first (`set.json` `look.join_review` is null), so the builder took no cut.
- For the lead: order that review, then decide on cut 3 for the body (all its look sets, the -X head included).
- Unverified observation on `moves.py` (not this package's file): `join_command` reads the ZMP "up to two frames before the command", but the nested central differences make frame k−2 depend on q at the command frame. The warm-up reading changes with turn rate (5.54 / 4.40 / 3.54 mm at stride factor 1.0 / 0.8 / 0.6), which fits that.

**(d) Turn in place (L13).**
- `setscreen.py` (not this package's) reads every entry of `refs.json` as a walk, so a set holding the turn in place can no longer be screened or verified. I therefore built it only on scratch copies of the two complete 75 cm sets (`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpl/inplace_root`). It is identical on V6 and V6noF5.
- Periodicity passes: CoM 0.0008 / 0.0005 mm, ZMP about 0, joints 2.2e-6 / 1.3e-6 rad.
- It is still not available:
  - the reference's thigh-twist use is 0.9114 / 0.9115, above 0.87;
  - its start is over knee speed (1.1071 / 1.1664);
  - its hip-yaw range margin is 0.0266 rad, under 0.05.
- Not built on the stopped 60 cm and -X sets.
- For the lead: `setscreen` must learn the periodic turn before the turn in place can sit in a set's records.

**(e) Stopped sets.** The builder stops a build once `start` or `nominal_to_v0.4` fails, so the four new sets hold no other moves. Their verify therefore lists 14 items "not built" plus "the set is stopped". The two 75 cm sets are complete because the earlier builder built them through.
For the lead: one more `sets build` call per set builds the rest, at about 45–60 min each, if you want their numbers before ruling. I recommend waiting for the start ruling first.

**(f) Minor points.**
- On the -X models, `moves.py:966` warns of a 0/0 at the held joints (neck yaw, jaw); the recorded value is 0.0.
- `turn90_pi4_right`'s reason says "also outside join_command". That text belongs to its rate group; its own warm-up reading of 3.6611 mm passes.
- `bar refcheck` was not run.

**3. Outputs**
- Tests: `243 passed, 3 skipped in 321.61s (0:05:21)`. The skips are the three STAGE2_SLOW full-pass tests.
- `moves legacy --body 60-N-T-10 --name start --name stop`: "legacy check: 2 of 2 equal".
- `regress check`: "stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)".
- The record-key hash pair is unchanged: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- Acceptance build of V6noF5/75-M-H-10 into an empty scratch root (`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpl/acc_root`):
  - it stopped at `start` (1.1261);
  - all 6 walks (both formats), the start's built file and `nominal_to_v0.4` hash equal to the committed records, and the screen verdicts are equal;
  - verify: "V6noF5 75-M-H-10: verify FAILED: 7 references, 1 not available, 15 failures".
- Verify on the worktree:
  - V6/75: "verify FAILED: 12 references, 10 not available, 1 failures"
  - V6noF5/75: "16 references, 6 not available, 1 failures"
  - V6/60: "5 references, 3 not available, 15 failures"
  - V6noF5/60: "5 references, 3 not available, 15 failures"
  - V6/75-X: "7 references, 1 not available, 15 failures"
  - V6/60-X: "6 references, 2 not available, 15 failures"

**4. Diff and status**
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`: 34 files changed, 123133 insertions(+), 40 deletions(-). Only package files:
  - `designlab/stage2/moves_look.py` +1274
  - `designlab/stage2/sets.py` 631
  - `tests/test_stage2_moves_look.py` +1117
  - `stage2/sets/V6/tail_phase.json` +854
  - for each of V6/{75-M-H-10, 60-N-T-10, 75-M-H-10-X, 60-N-T-10-X} and V6noF5/{75-M-H-10, 60-N-T-10}: `SCREEN.md`, `refs.json`, `screen.json`, `set.json`, `transitions.json`
- Tracked: the five records per set/body, plus `tail_phase.json`.
- Ignored by `stage2/sets/.gitignore`: `refs/`, `look/` and `transitions/` arrays (`*.npz`, line 1), `.lock` (line 3), `partial/` (line 4), `logs/` (line 6).
- Scratch outputs are outside the repo, under `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpl/`.
- `git status --porcelain`: empty.

**5. Commits this round** (oldest first): `75d43036`, `2a5466ab`, `ba439717`, `a9ea7f58`, `a024cbe8`, `2237ba9c`. Branch head: `2237ba9c`.
Commits on the branch but not on the working branch that lack attribution lines at the end: `7f96371b` (Merge wt/wp-m), `5436c8da` (Merge wt/wp-r), `7fef4d62` (Merge wt/wp-c). All three are from the earlier builder.
