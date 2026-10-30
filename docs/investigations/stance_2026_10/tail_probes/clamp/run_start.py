"""run_clamp.py on the two START checkpoints of the ECS (start_s42 = stance6 s42 3,145,728; start_s80 = stance6 s80
4,194,304, as the policies lens named them: run_policies.py START).  Same arguments as run_clamp.py; only the
checkpoint lookup is extended.  Read only; no training.  Exploratory addition: whether the s42 lineage's 5 mm backward
CoM drift (start -> ends) survives with tail_3 held at 0 in both the start and the ends."""

import sys

import run_clamp as rc

START = {"start_s42": ("s42_A_1101", 3_145_728), "start_s80": ("s80_A_1301", 4_194_304)}
_orig = rc.checkpoint


def checkpoint(name):
    if name in START:
        run, steps = START[name]
        m = rc.RUNS / run / "01_stance" / "models"
        return m / f"stage1_{steps}_steps.zip", m / f"stage1_vecnormalize_{steps}_steps.pkl"
    return _orig(name)


rc.checkpoint = checkpoint
sys.exit(rc.main())
