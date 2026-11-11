"""Stage 2: the agility round, with character (``STAGE2_PLAN.md``).  Simulation only; nothing here changes a body or
adopts anything without the owner (P24).

Modules:

* ``look``: Track A's look ladder (plan section 4 A; ``stage2/research/research_look.md`` section 4.1): the reference
  walk variants V0-V6 per body and speed, built from ``gait.GaitSpec``'s look fields, with the per-frame quantities
  the Stage 0 screen and the kinematic renders read, the measured look metrics and the tail-phase check.
* ``screen``: the Stage 0 screen of every ladder rung, before any render, and the tail-phase check.
* ``clips``: kinematic clips of the rungs (plan 4 A.4; research 4.1 item 3): playback at 1x, covered and point-light,
  from the visitor's and the side view; labelled "kinematic reference, not a simulation".
* ``motion``: the rungs' machine metrics (research 4.1 item 5; ``stage2/look/metrics.json``), path plots and phase
  charts.

Run from hardware/trex_form_study:
    python -m designlab.stage2.look prepare --body 60-N-T-10 --out stage2/look
    python -m designlab.stage2.clips render --media <dir> --procs 3
    python -m designlab.stage2.motion run --media <dir>
"""
