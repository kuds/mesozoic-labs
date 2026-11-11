"""Stage 2: the agility round, with character (``STAGE2_PLAN.md``).  Simulation only; nothing here changes a body or
adopts anything without the owner (P24).

Modules:

* ``look``: Track A's look ladder (plan section 4 A; ``stage2/research/research_look.md`` section 4.1): the reference
  walk variants V0-V6 per body and speed, built from ``gait.GaitSpec``'s look fields, with the per-frame quantities
  the Stage 0 screen and the kinematic renders read, the measured look metrics and the tail-phase check.

Run from hardware/trex_form_study:
    python -m designlab.stage2.look prepare --body 60-N-T-10 --out stage2/look
"""
