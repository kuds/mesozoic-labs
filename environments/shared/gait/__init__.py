"""Floor-truth gait measurement: which feet bear the animal's weight, read from the physics itself.

The touch sensors the envs observe can be fooled (a sole pressed on the
other foot reads as support, the MIN over substeps drops brief contacts), so
the ``stance_quality/v2`` gate judges stance on FLOOR TRUTH instead: the
normal force between each leg's geoms and the floor, decoded per physics
substep exactly as ``mj_contactForce`` reports it.  GAIT_QUALITY_PLAN_2026_09.md
§3.1 names this package; the stance subset landed first, with the
``stance_quality/v2`` gate kind (decision D-D23), and the locomotion metrics
and report the plan describes extend it rather than duplicate it.

Modules, bottom-up:

* :mod:`~environments.shared.gait.constants` -- the measurement constants,
  :data:`~environments.shared.gait.constants.MEASUREMENT_VERSION`, and the
  measurement manifest (and its sha256) a report is bound to; pure;
* :mod:`~environments.shared.gait.events` -- the down state and debounced
  touchdowns, windows in seconds; pure numpy;
* :mod:`~environments.shared.gait.morphology` -- feet, legs, support geoms,
  soles and body weight read off a live env, with each embodiment class's
  explicit ``FOOT_SUPPORT`` entry (``support_entry``), re-exported from
  :mod:`environments.shared.floor_contact.morphology`, and the measurement
  definition; imports mujoco;
* :mod:`~environments.shared.gait.recorder` -- ``SubstepContactRecorder``,
  attached through ``BaseDinoEnv._substep_probe_hook``, and the per-step
  ``EpisodeTrace`` it reduces an episode to; imports mujoco;
* :mod:`~environments.shared.gait.stance_metrics` --
  ``episode_stance_metrics`` and the ``StanceEpisodeMetrics`` row every v2
  threshold key reads; pure numpy;
* :mod:`~environments.shared.gait.pad_telemetry` -- report-only per-foot
  pad telemetry on the same trace (each box pad's signed fore-aft centre of
  pressure, the gate's per-foot statistic, the digits' load share) for the
  periodic evaluations and the stance replay CSV; pure numpy;
* :mod:`~environments.shared.gait.balance_telemetry` -- report-only
  whole-body balance and floor-force telemetry on a trace recorded with
  ``balance=True`` (the centre of mass and its extrapolation against the
  toe line, Hof's split of its acceleration, the achieved leg offsets, the
  actuators' force limits, the floor's shear and friction-cone use, the
  ankle torque's lever) for the periodic evaluations; pure numpy at module
  level.

Layering rules: no species env module imports this package (an import line
would change the env module's bytes, and with them its identity), and this
package imports neither ``curriculum`` nor ``reporting``, which build on it.
The contact reduction the recorder shares with the T. rex's opt-in CoP band
(scratch, the stance4 copband design) lives outside it, in
:mod:`environments.shared.floor_contact`, which imports nothing from here.
Only the pure constants are re-exported here, so a pure reader (the gate) can
import the package without mujoco; import the rest from its module.
"""

from __future__ import annotations

from .constants import (
    CONTACT_THRESHOLD_N,
    DEBOUNCE_S,
    DOWN_SUBSTEP_FRACTION,
    LOAD_WINDOW_S,
    MEASUREMENT_VERSION,
    SATURATION_ABS,
    SOLE_ROLLED_DEG,
    SPAWN_GRACE_S,
    SUPPORT_GEOM_DOWN_FRACTION,
)

__all__ = [
    "CONTACT_THRESHOLD_N",
    "DEBOUNCE_S",
    "DOWN_SUBSTEP_FRACTION",
    "LOAD_WINDOW_S",
    "MEASUREMENT_VERSION",
    "SATURATION_ABS",
    "SOLE_ROLLED_DEG",
    "SPAWN_GRACE_S",
    "SUPPORT_GEOM_DOWN_FRACTION",
]
