# Work package L (later; starts after code state A): the look in moves, `moves_look.py`

You are building one module of a robotics simulation study (a robotic T. rex biped; everything is simulation).
You need no other context than this brief and the design it points into:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`
("the design"; read its sections 1 (D5, D6, D17, D18, D20), 2.2, 3.1-3.3, 3.9 in full). Maps of the code you port:
`.../scratchpad/maps/01_gait_look.md` (sections 1.4, 1.5, 3) and `.../maps/03_turning.md` section 7.

Background. The owner picked the look ladder's V6 (every look feature) as Stage 2's frozen reference. Periodic
look walks exist (`gait._generate_look`, `designlab/stage2/look.py`), but no start, speed change, stop or turn
carries a look feature: `transitions.gait_of` and `turning.generate` build today's walk only.
`designlab/stage2/moves.py` (merged, frozen with state A) builds all of those for today's laws and takes every
law from a `Laws` object; when `laws.solver == "look"` its `generate` loads `designlab.stage2.moves_look` by name
and calls it. You write that module, so that the V6 set (and `V6noF5`, V6 without F5's toe-off, for the tracker
fit R2L) gets its walks in Stage 1's reference format with feedforward, its starts, changes, two-touchdown stops
and turns, including periodic walking turns at 0.35-1.0 rad/s and the 0.5 rad/s turn in place.

## Files you own (and no others)

- `designlab/stage2/moves_look.py`, `tests/test_stage2_moves_look.py`
- in `designlab/stage2/sets.py`: only the two new entries of `SETS` (`V6`, `V6noF5`), their spec functions and the
  K-stride walk writer they need (`sets.py` is in no record key, so this changes no earlier record)

You do not edit `moves.py`, `setio.py`, `readings.py`, `bar.py`, any Stage 1 file or anything under
`stage2/look/`. If a `Laws` or `PlanCtx` signature turns out wrong, stop and ask the lead: a change to `moves.py`
needs `stage2/sets/legacy_check.json` made again and `sets verify` passing on every frozen set. The one Stage 1
edit the look needs (`control.Track.add_periodic` reading `window_strides`) is already in (design 2.4).

## Interface you implement (called by `moves.generate` when `laws.solver == "look"`)

```python
LOOK: moves.Laws                                           # name "look"; every law of design 3.9's table
def laws_for(a: moves.GaitX, b: moves.GaitX, flat_commanded: bool = False) -> moves.Laws   # flat_commanded: rule L10
def solve_frames(M, q_home, pl, com_t, geo, ctx) -> (qs, foot_err, com_err)   # as transitions.solve_frames, plus foot pitch,
                                                                              # contact offsets, trunk roll from the laws
def idyn(M, qs, qd, qdd, swing, ctx) -> (tau, cop_margin, normal, friction)   # look.look_inverse_dynamics' rolling split, per frame
def initial_state(pl, ref_a, geo, a_time_s) -> x0          # a K-stride window resampled on the move's phase grid (J1)
def compare(g, ref, side, q_home) -> dict                  # transitions.compare for a K-stride reference (J2)
def arrays_extra(g) -> dict                                # the look keys appended after transitions.arrays' keys
def periodic_turn(src, a: moves.GaitX, rate, strides=1) -> dict   # a periodic turn cut from a long plan's hold; gated for periodicity
```

## Rules to implement, each with its test (design 3.9; each is a pre-registered rule a reviewer checks on arrays)

L1 (F3 duty from `describe`), L2 (F9 tail amplitude and phase mixed with the ramp's weight, and in a turn's quicker
steps the `(T(p)/T_walk)^2` amplitude rule for tail and neck), L3 (F10 stabilised head and settle dips), L4 (F4
lift shape sampled at lift-off), L5 (F1/F2 bob between the plan's touchdowns; 12 correction passes), L6 (F6 trunk
roll), L7-L8 (F14 step times through `period_factor`; the variation envelope 1 on shared strides, fading to 0
across the next stride inward), L9 (F5 rolling contact per stance, about the yawed foot's own lateral axis) and
**L10, the fallback, per stance**: in a commanded move every stance that ends after the command is flat, the
swing that lifts at the command carries the walk's toe-off pitch to zero by touchdown, toe-off returns with the
first stance that begins after the move's last ramp. J1-J3 for K-stride windows, with the resampling through
`gait._Warp`.

Order of work, cheapest first, each step leaving a working module: (1) L1, L2, L3, L4 on starts, changes, the
stop and the turn; (2) L5, L6; (3) the periodic look walk in Stage 1 format (`tau`, `normal` from
`look.look_inverse_dynamics`; `window_strides`); (4) L10 (this completes `V6noF5` and a flat-footed V6);
(5) periodic turns cut from long plans; (6) L7-L8 and J1-J3; (7) L9. If time runs out, cut from the end, in the
design's stated order: L9 inside turns and stops (L10 stands), then inside starts and changes, then J1's
resampling ("a move leaves and enters a K-stride window only at its boundary"), then F14 inside periodic turns.
Say in every entry's `rules` which rule built it.

## Tests (`tests/test_stage2_moves_look.py`)

- With every look field at its default, `laws_for` gives arrays `array_equal` to `TODAY`'s for a start, a change,
  the stop and the turn (one pass), and `tests/test_stage2_moves.py` passes unchanged.
- Per rule, a check on the arrays: the duty of each swing equals `describe(H)["duty"]` mixed by the ramp (L1); the
  tail's fitted phase and amplitude at both ends of a change equal the two gaits' (L2); the settle dip follows
  each touchdown by `settle_lag` (L3); the lift's peak falls at `lift_peak` of each swing (L4); the pelvis trough
  follows each touchdown as in the periodic walk at both ends (L5); the trunk roll is toward the stance leg (L6);
  a move's first and last strides equal the K-stride window's resampled strides within 1.0 mm CoM, 2.0 mm ZMP
  (J1, J2) and the variation is zero between the fades (L8); with L10 the foot pitch is zero in every stance that
  ends after the command, and **the frames before and at the command equal the look walk's, foot pitch
  included**.
- Causality: `test_nothing_before_the_command_knows` and `test_null_command_reproduces_the_walk` of
  `tests/test_stage2_moves.py`, repeated with look laws.
- A periodic turn's consecutive hold strides map onto each other by the stride motion within 1.0 mm, 2.0 mm,
  1e-3 rad.
- A K-stride walk file loads on a `control.Track`, its `info["period"]` is the stride's, and a plan item counts
  windows.

## Acceptance

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_moves_look.py tests/test_stage2_moves.py tests/test_stage2_sets.py
$PY -m designlab.stage2.moves legacy --body 60-N-T-10 --name start --name stop       # still equal (you did not touch moves.py)
$PY -m designlab.stage2.regress check
bash tools/lowprio.sh $PY -m designlab.stage2.sets build --set V6noF5 --body 75-M-H-10 --procs 1 --root <scratch root>
bash tools/lowprio.sh $PY -m designlab.stage2.setscreen run --set V6noF5 --body 75-M-H-10 --root <scratch root>
$PY -m designlab.stage2.sets verify --set V6noF5 --body 75-M-H-10 --root <scratch root>
```
Done per step of the order above when its tests pass; report, for each built reference, the gates of design 3.3,
the Stage 0 verdict, the stride factor of each turn and the rule ids used. The V6 look's frozen walks (by hash,
`stage2/look/look_refs.json`) must be the walks of the V6 set: its look-format files must hash to the ladder's V6
rungs, as V0's do to the V0 rungs.

## Setup and commit

Worktree: the absolute path the lead gives you. `$PY tools/restore_inputs.py copy-from
/home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays look sets`. Environment as
in the design's section 4.7. One heavy process at a time through `tools/lowprio.sh` (a V6 walk takes about 5
minutes to generate). Commit on your own branch, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so.
