"""The floor-truth measurement constants, and the manifest that binds a report to them.

Every value below is part of the MEANING of every ``stance_quality/v2``
threshold key: ``min_all_feet_support = 0.98`` means "0.98 of the window's
steps had every leg's floor force above :data:`CONTACT_THRESHOLD_N` on at
least :data:`DOWN_SUBSTEP_FRACTION` of the substeps", and the same bar under
another threshold is another gate.  So the constants are versioned together:
changing any one of them after a stage adopts the kind, or changing an entry
of :data:`.morphology.SUPPORT_REGISTRY`, needs a new
:data:`MEASUREMENT_VERSION`, and the gate refuses a report measured under a
manifest other than the one it re-derives (the gate map's "measurement
block").  :func:`measurement_manifest` is that manifest, and
:func:`measurement_sha256` its digest.

Pure: no mujoco import, so the pure gate module
(``curriculum/stance_gate_v2.py``) and the reporting layer can read the
constants without building an env.  :func:`measurement_manifest` reads names
off a compiled model it is handed, and never imports mujoco to do so.

The values, and where each was measured (``61a3424``, mujoco 3.10.0, the
40-episode zero-action statue panels of all six species on seeds 3042-3081
and the trex ``20260914`` / ``20260920`` / ``20260930`` / ``20261001`` and
velociraptor ``20260922`` stance checkpoints of the 2026-10 stance-hack
audit):

* :data:`CONTACT_THRESHOLD_N` 0.1 N is the threshold every other contact
  reader already uses (``stance_diagnostics.derive_stance_info``,
  ``metrics.py`` and the MJX per-foot sensor read), so floor truth and touch
  disagree only where the physics does.
* :data:`DOWN_SUBSTEP_FRACTION` 0.5: a leg is down on a control step when
  its whole-limb floor force exceeds 0.1 N on at least half of the step's
  substeps -- the audit's floor-truth channel, which agreed with video
  wherever a frame could resolve the contact (GAIT_QUALITY_PLAN_2026_09.md
  §3.1).
* :data:`DEBOUNCE_S` 20 ms: runs shorter than this are flicker.  Defined in
  SECONDS and converted per species (:func:`.events.steps_for`): 2 control
  steps at dt 0.01 (trex, velociraptor, dibothrosuchus, brachiosaurus) but 1
  step -- the identity -- at dt 0.02 (compsognathus and its robot), which is
  the plan's "in seconds" rule rather than a 2-step rule that would mean
  40 ms on the 0.02 s species.
* :data:`SPAWN_GRACE_S` 0.10 s: the reset transient is a PLANT property,
  not a policy one.  Every statue but trex pops off its reset pose at 1.4-3.1
  body weights (velociraptor 1.67-2.42, brachiosaurus up to 3.05) and the
  dibothrosuchus, robot and compsognathus statues are fully airborne for up
  to 9, 3 and 1 substeps at spawn; on every statue that pop is over by
  t = 0.04 s (step 0-4).  The settle-window hop and impact checks therefore
  start at 0.1 s, where every statue is at or below 1.52 body weights with 0
  airborne substeps, and the trex hacks' hops -- which begin inside the
  grace -- still fail every episode after it.
* :data:`LOAD_WINDOW_S` 1.0 s: the sub-window of the windowed load share,
  which catches a temporary one-leg stance a whole-window mean averages away.
* :data:`SATURATION_ABS` 0.99: ``|a_i| >= 0.99`` is a saturated command, the
  threshold of ``stance_report.saturated_actuator_indices``.
* :data:`SOLE_ROLLED_DEG` 2.0 degrees: a sole tilted more than this BEYOND
  its keyframe tilt is "rolled" (brachiosaurus authors its ellipsoid feet at
  15.8-19.2 degrees, so the absolute tilt cannot be the measure).
* :data:`SUPPORT_GEOM_DOWN_FRACTION` 0.5: a registered support geom is
  loaded on a step when its floor force exceeds 0.1 N on at least half of the
  substeps, the per-geom twin of :data:`DOWN_SUBSTEP_FRACTION`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .morphology import Morphology

#: The version of the definitions in this package.  Bump it (and say why in
#: the CHANGELOG) whenever a constant, a registry entry or a metric
#: definition changes what an adopted ``stance_quality/v2`` key means.
MEASUREMENT_VERSION = "floor-truth/v1"

#: Schema of the dict :func:`measurement_manifest` returns.
MEASUREMENT_MANIFEST_SCHEMA = "mesozoic.floor-truth-measurement/v1"

CONTACT_THRESHOLD_N = 0.1
DOWN_SUBSTEP_FRACTION = 0.5
DEBOUNCE_S = 0.020
SPAWN_GRACE_S = 0.10
LOAD_WINDOW_S = 1.0
SATURATION_ABS = 0.99
SOLE_ROLLED_DEG = 2.0
SUPPORT_GEOM_DOWN_FRACTION = 0.5

# Decimal places the manifest keeps of the two model-derived reals it records.
# The body weight is ``body_subtreemass[root] * |g_z|``; the compiler derives
# the subtree mass, and the plant contract allows derived quantities a few
# ULPs of cross-architecture noise (``PORTABLE_FLOAT_SIGNIFICANT_DIGITS``).
# Rounding to a micronewton (and dt to a nanosecond) keeps that noise out of
# the digest without hiding any change a plant revision could make.
_BODY_WEIGHT_DECIMALS = 6
_DT_DECIMALS = 9


def measurement_constants() -> dict[str, Any]:
    """The version and every constant above, by lowercase name: the manifest's ``constants`` block.

    Model-free, so a pure reader (the gate, a backfill) can check that a
    recorded manifest was measured under the definitions this checkout
    implements without building an env.
    """
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "contact_threshold_n": CONTACT_THRESHOLD_N,
        "down_substep_fraction": DOWN_SUBSTEP_FRACTION,
        "debounce_s": DEBOUNCE_S,
        "spawn_grace_s": SPAWN_GRACE_S,
        "load_window_s": LOAD_WINDOW_S,
        "saturation_abs": SATURATION_ABS,
        "sole_rolled_deg": SOLE_ROLLED_DEG,
        "support_geom_down_fraction": SUPPORT_GEOM_DOWN_FRACTION,
    }


def measurement_manifest(morphology: "Morphology", model: Any) -> dict[str, Any]:
    """Everything that fixes what a floor-truth metric MEANS for one species' env.

    ``{schema, constants, species, frame_skip, dt, body_weight_n, registry,
    feet}``:

    * ``constants`` -- :func:`measurement_constants`;
    * ``registry`` -- the species' :class:`.morphology.FootRegistryEntry`
      templates as declared (``None`` when the species has no entry and the
      generic rule chose its geoms), so a registry edit moves the digest even
      where the resolved names happen not to change;
    * ``feet`` -- per foot, in ``_foot_sensor_groups`` order: the label, the
      touch sensors, the reference site, the support geoms and the sole geom
      by NAME, and the sole frame's normal / forward / lateral axes (``"-z"``,
      ``"+x"``, ...) that the sole roll, pitch, corner lift and CoP are
      expressed in;
    * ``frame_skip`` and ``dt`` -- the substep count every per-step reduction
      runs over, and the step the seconds-defined windows convert with;
    * ``body_weight_n`` -- the divisor of every ``*_bw`` metric.

    The keyframe's authored sole tilt (``home_sole_tilt_deg``) is left out:
    it is a plant fact the plant identity already binds, computed through
    trigonometry whose last bits are not portable.  ``model`` is the compiled
    model the morphology was built from, used only to read names.
    """

    def geom_name(geom_id: int) -> str | None:
        return None if geom_id < 0 else str(model.geom(int(geom_id)).name)

    def axis(index: int, sign: float) -> str:
        return f"{'-' if sign < 0 else '+'}{'xyz'[index]}"

    entry = morphology.registry_entry
    feet = []
    for foot in morphology.feet:
        has_sole = foot.sole_geom >= 0
        feet.append(
            {
                "label": foot.label,
                "sensors": list(foot.sensor_names),
                "reference_site": str(model.site(int(foot.primary_site)).name),
                "support_geoms": [geom_name(geom) for geom in foot.support_geoms],
                "sole_geom": geom_name(foot.sole_geom),
                "sole_normal": axis(foot.sole_normal_axis, foot.sole_normal_sign) if has_sole else None,
                "sole_forward": axis(foot.sole_forward_axis, foot.sole_forward_sign) if has_sole else None,
                "sole_lateral": axis(foot.sole_lateral_axis, foot.sole_lateral_sign) if has_sole else None,
            }
        )
    return {
        "schema": MEASUREMENT_MANIFEST_SCHEMA,
        "constants": measurement_constants(),
        "species": morphology.species,
        "frame_skip": int(morphology.frame_skip),
        "dt": round(float(morphology.dt), _DT_DECIMALS),
        "body_weight_n": round(float(morphology.body_weight_n), _BODY_WEIGHT_DECIMALS),
        "registry": None if entry is None else entry.as_manifest(),
        "feet": feet,
    }


#: The manifest keys code alone fixes -- the schema, the constants, the
#: species and its registry entry -- as opposed to the ones read off a
#: compiled model (``frame_skip``, ``dt``, ``body_weight_n``, ``feet``).  A
#: reader with no model compares these against
#: :func:`.morphology.measurement_definition`; the model-read ones are bound
#: through the plant and task identity instead.
MEASUREMENT_DEFINITION_KEYS: tuple[str, ...] = ("schema", "constants", "species", "registry")


def measurement_definition_sha256(manifest: Mapping[str, Any]) -> str:
    """The digest of *manifest*'s :data:`MEASUREMENT_DEFINITION_KEYS` alone (a key it lacks hashes as null).

    What the ``stance_quality/v2`` panel CSV stamps beside the whole
    manifest's :func:`measurement_sha256`, so publication -- which reads
    the CSV and no model -- can refuse rows measured under constants or a
    registry entry other than this checkout's, even when nobody bumped
    :data:`MEASUREMENT_VERSION` for the edit.
    """
    return measurement_sha256({key: manifest.get(key) for key in MEASUREMENT_DEFINITION_KEYS})


def measurement_sha256(manifest: dict[str, Any]) -> str:
    """The ``sha256:<hex>`` canonical-JSON digest of a :func:`measurement_manifest` dict.

    The repository's one canonical-JSON hasher
    (``result_bundle.hashing.canonical_json_sha256``: sorted keys, compact
    separators, UTF-8), so a manifest copied into a report JSON and read back
    hashes identically.  Imported at call time: the ``result_bundle`` package
    imports ``curriculum`` at its top, and the v2 gate in ``curriculum``
    imports this package, so a module-level import here would be a cycle.
    """
    from ..result_bundle.hashing import canonical_json_sha256

    return canonical_json_sha256(manifest)
