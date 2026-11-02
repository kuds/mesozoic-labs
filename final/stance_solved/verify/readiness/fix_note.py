import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
reps = [
 ("Stock-critic checkpoints (OFF/STANCE6) store `ActorCriticPolicy` [measured: `zip_classes.out`], so they need no cross-stage rule [inferred]. The best of them, s80, has a saturating neck habit [measured: SCORE line for OFF s80]. GQP's \"Children inherit gaits\" argues against using it.",
  "OFF s60's zip stores the stock `ActorCriticPolicy` [measured: `zip_classes.out`; only s60 read], so stock-critic finals would need no cross-stage rule [inferred]. The cleanest of them, OFF s80, has a saturating neck habit [measured: SCORE l.342], and GQP's \"Children inherit gaits\" argues against it."),
 ("I checked its numbers, quotes, code lines and plan text against MAIN (4c9d5e9b), BR (1dfe2327), RB (8737cd3d), RUNLOG, SCORE and the F2 copies. None of its conclusions is wrong.",
  "Its numbers, quotes, code lines and plan text check out against MAIN, BR, RB, RUNLOG, SCORE and the F2 copies; no conclusion is wrong."),
 ("- I re-read the PHASE action std independently: 0.0597, 0.0569 and 0.0585 at 12,001,280 steps, with `ent_coef` 0 [measured: `std_check.out`].\n", "- PHASE action std re-read: 0.0597, 0.0569, 0.0585 at 12,001,280 steps, `ent_coef` 0 [measured: `std_check.out`].\n"),
 ("- The pushes: \"The reward and the observation are deliberately untouched\" (shared `base_env.py` l.1262-1263) [measured].\n", ""),
 ("   - No hours are given [measured]. The planned route is therefore longer than the note lists.\n", "   - No hours are given. The planned route is longer than the note lists [inferred].\n"),
]
for a, b in reps:
    if a not in t:
        print("MISSING:", a[:60]); continue
    t = t.replace(a, b)
open(p, "w", encoding="utf-8").write(t)
