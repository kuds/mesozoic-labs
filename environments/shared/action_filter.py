"""First-order low-pass filtering of commanded actions.

The filter is part of the PLANT INTERFACE, not the curriculum.  An enabled
cutoff changes what a policy's action *means* — the plant responds to the
filtered command, so a checkpoint trained with the filter is incompatible
with the unfiltered plant even though every tensor dimension matches
(docs/PLANT_CONTRACT.md, "action meaning" rule).  Species therefore enable
it through plant-level declarations only:

- SB3: ``action_filter_cutoff_hz`` class attribute on the species env
  (deliberately not an ``__init__`` kwarg, so stage TOMLs cannot set it);
- MJX: ``action_filter_cutoff_hz`` in ``register_species_mjx`` (listed in
  ``_PLANT_INTERFACE_CONFIG_FIELDS``, so stage TOMLs cannot override it).

The plant contract fingerprints this module and records the cutoff when a
species enables it, and asserts both backends agree.

Why it exists: both 2026-08 trex stage-1 runs converged on poses that are
not statically stable and survive only through 16.8–18.7 Hz command
chatter, while the balance task itself needs ~1.1–1.4 Hz of closed-loop
bandwidth (measured by the stance-gate filter probe).  Amplitude penalties
(smoothness/jerk) cannot price that strategy out — a saturated command is
perfectly smooth — so the filter removes the bandwidth itself: poses that
need fast stabilisation stop being reachable attractors during training.
See docs/investigations/TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md.

The discretization matches the eval-side probe filter
(``_low_pass_predict`` in environments/shared/reporting/stance_report.py):
a discrete RC filter with ``alpha = dt / (RC + dt)``, ``RC = 1/(2*pi*fc)``,
state seeded with the first post-reset action so episodes do not open with
a transient toward zero.

Both functions are fingerprinted by the plant contract: no f-strings
(digests.py forbids them inside fingerprinted callables), and any edit
here moves the policy-interface fingerprint of every species with the
filter enabled — bump their ``policy_interface_revision`` accordingly.
"""

from __future__ import annotations

import math
from typing import Any

# Arrays from either backend (numpy or jax.numpy); plain arithmetic only.
Array = Any


def low_pass_alpha(cutoff_hz: float, control_dt: float) -> float:
    """Return the per-control-step blend factor for a first-order low-pass.

    ``alpha = dt / (RC + dt)`` with ``RC = 1 / (2 * pi * cutoff_hz)`` — the
    standard discrete RC filter, identical to the stance-gate probe's.
    """
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be positive; use no filter instead of a zero cutoff")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    return control_dt / (rc + control_dt)


def apply_low_pass(previous: Array, current: Array, alpha: float) -> Array:
    """One filter update: blend ``current`` into the carried filter state.

    Backend-neutral (numpy and JAX arrays) and trace-safe: arithmetic only.
    Seeding — returning ``current`` unblended on the first post-reset step —
    is the caller's job, because the two backends detect the episode
    boundary differently (Python state vs a traced ``step_count``).
    """
    return previous + alpha * (current - previous)
