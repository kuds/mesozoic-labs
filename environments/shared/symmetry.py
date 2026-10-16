"""Sagittal mirror symmetry of a plant and of its policy interface.

An animal standing on a level floor is left-right symmetric: the mirror image of
any state is another valid state, and the mirror image of a good action in that
state is a good action in the mirrored one.  This module derives the maps that
express this for a MuJoCo plant, checks them, writes them to a per-model JSON
file and loads them back for training (``mirror_kl_ppo.MirrorKLPPO`` uses the
observation and action maps; nothing here imports torch or Stable-Baselines3).

The reflection is across the world xz-plane, ``S = diag(1, -1, 1)``.  A body pose
``(p, R)`` maps to ``(S p, S R S)``, which is again a proper rotation.  Under it:

* polar vectors (positions, linear velocity and acceleration, gravity, target
  directions) map by ``S``: the y component flips;
* pseudovectors (angular velocity, torque) map by ``-S = diag(-1, 1, -1)``;
* a quaternion ``(w, x, y, z)`` of ``R`` maps to ``(w, -x, y, -z)``;
* a hinge about body-local axis ``a`` maps to a hinge about ``-S a`` at the same
  angle, so on paired bodies whose frames are mirror images the partner's angle
  is ``+theta`` for a pitch axis and ``-theta`` for a roll or yaw axis.

Every map is a signed permutation, ``(M x)[i] = sign[i] * x[perm[i]]``, and an
involution (``M M = I``).  :func:`derive_model_mirror` builds the model-level
maps (bodies, geoms, sites, joints, actuators, sensors, ``qpos``, ``qvel``,
``ctrl``) from the names (``r_``/``l_`` and the other side tokens of
:data:`DEFAULT_PREFIX_RULES`) and the compiled geometry, and refuses anything it
cannot prove mirror-consistent: a pair whose positions, axes, ranges, masses,
inertias, contact or actuator parameters differ, a sensor type without a rule,
or a feature it does not mirror (tendons, equality constraints, plugins).  Link
frames that a CAD export rotates differently on the two sides are handled by a
per-body frame correction.  Of the six plants, the five anatomical ones derive
cleanly; the Compsognathus robot does not, because its MJCF is not mirror
symmetric as authored (one side's ankle chain sits 37 mm from the mirror of the
other's).

:func:`build_env_mirror` adds the policy interface: the observation map from the
model's observation layout (:data:`OBSERVATION_LAYOUTS`, keyed by the plant's
``observation_schema``; ``bipedal-target-heading-free/v1`` is the one layout
today, the T. rex and the anatomical Compsognathus) and the action map, the
actuator map, after checking that the env's action scaling commutes with it.
:func:`mirror_env_state` writes the mirror image of one env's full simulator
and episode state into another env (or the same one), which is how the tests
check equivariance by paired rollouts.

The map file, ``configs/<species>/<embodiment>/mirror_map.json``, is generated
(``python -m environments.shared.symmetry --write --species S --embodiment E``)
and checked (``--check``: every committed map must equal a fresh derivation,
byte for byte).  It records the plant's ``physics_sha256`` and
``policy_interface_sha256``, and :meth:`MirrorMap.load` refuses a plant whose
digests differ, so a plant or interface revision cannot silently reuse a stale
map, and a file that differs from a fresh derivation, so neither can a hand edit.

The training switch is the ``[ppo]`` table's ``symmetry`` key
(:data:`SYMMETRY_KEYS`, :func:`pop_symmetry_config`); absent, training is stock
PPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Hashable, Mapping, Sequence

import mujoco
import numpy as np

from .command_frame import COMMAND_COMPONENTS
from .model_id import ModelId
from .paths import REPOSITORY_ROOT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .plant_contract import PlantIdentity

S = np.diag([1.0, -1.0, 1.0])
POLAR = np.array([1.0, -1.0, 1.0])
PSEUDO = np.array([-1.0, 1.0, -1.0])
QUAT_SIGNS = np.array([1.0, -1.0, 1.0, -1.0])

#: Side tokens, tried in order: prefixes first, then suffixes.  r_/l_ (the dinosaurs), right_/left_ (the
#: Compsognathus robot), fr_/fl_ and rr_/rl_ (the quadrupeds' front and rear legs).
DEFAULT_PREFIX_RULES: tuple[tuple[str, str], ...] = (("r_", "l_"), ("right_", "left_"), ("fr_", "fl_"), ("rr_", "rl_"))
DEFAULT_SUFFIX_RULES: tuple[tuple[str, str], ...] = (("_r", "_l"), ("_right", "_left"))

#: The body-relative command (command_frame.COMMAND_COMPONENTS): forward speed keeps its sign, the lateral speed
#: and the yaw rate (the z component of a pseudovector) flip.
COMMAND_SIGNS = {"v_x_cmd": 1.0, "v_y_cmd": -1.0, "yaw_rate_cmd": -1.0}

HEADING_FREE_OBSERVATION_SCHEMA = "bipedal-target-heading-free/v1"
MIRROR_MAP_SCHEMA = "mesozoic.mirror-map/v1"
MIRROR_MAP_FILENAME = "mirror_map.json"
CONFIGS_ROOT = REPOSITORY_ROOT / "configs"

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_OBJ = mujoco.mjtObj


class SymmetryError(ValueError):
    """A model, layout, map file, plant binding or symmetry config failed a fail-closed check."""


# ---------------------------------------------------------------------------------------------------- helpers
def mirror_name(
    name: str,
    prefix_rules: Sequence[tuple[str, str]] = DEFAULT_PREFIX_RULES,
    suffix_rules: Sequence[tuple[str, str]] = DEFAULT_SUFFIX_RULES,
) -> str:
    """The mirror partner's name: the first matching side prefix (or suffix) swapped; unsided names map to themselves."""
    for a, b in prefix_rules:
        if name.startswith(a):
            return b + name[len(a) :]
        if name.startswith(b):
            return a + name[len(b) :]
    for a, b in suffix_rules:
        if name.endswith(a):
            return name[: -len(a)] + b
        if name.endswith(b):
            return name[: -len(b)] + a
    return name


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    q = np.asarray(quat, dtype=np.float64)
    mujoco.mju_quat2Mat(out, q / np.linalg.norm(q))
    return out.reshape(3, 3)


def _axis_label(axis: np.ndarray) -> str:
    """``pitch`` / ``roll`` / ``yaw`` for a unit axis along body y / x / z, else ``oblique``."""
    a = np.abs(np.asarray(axis, dtype=np.float64))
    k = int(np.argmax(a))
    return ("roll", "pitch", "yaw")[k] if a[k] >= 1.0 - 1e-9 else "oblique"


def _name(model: mujoco.MjModel, objtype: Any, index: int) -> str:
    return mujoco.mj_id2name(model, objtype, int(index)) or ""


def _is_signed_perm(matrix: np.ndarray, atol: float = 1e-9) -> bool:
    a = np.abs(np.asarray(matrix))
    return bool(
        np.all((a < atol) | (np.abs(a - 1.0) < atol))
        and np.allclose(a.sum(0), 1.0, atol=atol)
        and np.allclose(a.sum(1), 1.0, atol=atol)
    )


def _require(what: str, error: float, tol: float) -> None:
    if not error <= tol:
        raise SymmetryError(f"not mirror-symmetric: {what} (error {error:.3e} > {tol:.1e})")


def _sign_mapped_range(lo_hi: np.ndarray, sign: float) -> np.ndarray:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return np.array([lo, hi]) if sign > 0 else np.array([-hi, -lo])


@dataclass(frozen=True)
class SignedPermutation:
    """``(M x)[..., i] = sign[i] * x[..., perm[i]]``, validated to be an involution.

    ``names`` and ``rules`` (optional, one per index) say what each index is and why it maps as it does.
    """

    perm: np.ndarray
    sign: np.ndarray
    names: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        perm = np.asarray(self.perm, dtype=np.int64)
        sign = np.asarray(self.sign, dtype=np.float64)
        n = int(perm.shape[0]) if perm.ndim == 1 else -1
        if n < 0 or sign.shape != (n,):
            raise SymmetryError(
                f"a signed permutation needs one index and one sign per entry: {perm.shape}, {sign.shape}"
            )
        if sorted(perm.tolist()) != list(range(n)):
            raise SymmetryError("not a permutation of 0..n-1")
        if not np.all((sign == 1.0) | (sign == -1.0)):
            raise SymmetryError("every sign must be +1 or -1")
        if not np.array_equal(perm[perm], np.arange(n)) or not np.all(sign * sign[perm] == 1.0):
            raise SymmetryError("the signed permutation is not an involution (M M != I)")
        for label, values in (("names", self.names), ("rules", self.rules)):
            if values and len(values) != n:
                raise SymmetryError(f"{len(values)} {label} for {n} entries")
        object.__setattr__(self, "perm", perm)
        object.__setattr__(self, "sign", sign)
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "rules", tuple(self.rules))

    @property
    def dim(self) -> int:
        return int(self.perm.shape[0])

    def apply(self, x: Any) -> np.ndarray:
        """The map on the last axis; float32 stays float32 (a sign flip is exact in either precision)."""
        x = np.asarray(x)
        sign = self.sign.astype(x.dtype) if np.issubdtype(x.dtype, np.floating) else self.sign
        return np.asarray(x[..., self.perm] * sign)

    def entries(self) -> list[dict[str, Any]]:
        """One JSON row per index: ``index``, ``name``, ``mirror_index``, ``sign`` and ``rule``."""
        return [
            {
                "index": i,
                "name": self.names[i] if self.names else "",
                "mirror_index": int(self.perm[i]),
                "sign": int(self.sign[i]),
                "rule": self.rules[i] if self.rules else "",
            }
            for i in range(self.dim)
        ]

    @classmethod
    def from_entries(cls, entries: Any, what: str) -> SignedPermutation:
        """The map a list of :meth:`entries` rows describes (validated)."""
        if not isinstance(entries, list) or not all(isinstance(row, Mapping) for row in entries):
            raise SymmetryError(f"{what}: expected a list of rows")
        rows = sorted(entries, key=lambda row: int(row["index"]))
        if [int(row["index"]) for row in rows] != list(range(len(rows))):
            raise SymmetryError(f"{what}: the indices are not 0..{len(rows) - 1}")
        try:
            return cls(
                perm=np.array([int(row["mirror_index"]) for row in rows], dtype=np.int64),
                sign=np.array([float(row["sign"]) for row in rows]),
                names=tuple(str(row.get("name", "")) for row in rows),
                rules=tuple(str(row.get("rule", "")) for row in rows),
            )
        except SymmetryError as exc:
            raise SymmetryError(f"{what}: {exc}") from exc


# ---------------------------------------------------------------------------------------------------- model maps
@dataclass(frozen=True)
class ModelMirror:
    """Every model-level mirror map of one compiled model (:func:`derive_model_mirror`).

    ``body_C`` / ``site_C``: per body (site), the proper rotation ``C`` with ``R[mirror b] = S R[b] S C_b`` in the
    world; the identity when the pair's local frames are pure mirror images, as on every dinosaur plant.
    """

    body_map: np.ndarray
    geom_map: np.ndarray
    site_map: np.ndarray
    joint_map: np.ndarray
    joint_sign: np.ndarray
    joint_rule: tuple[str, ...]
    actuator_map: np.ndarray
    actuator_sign: np.ndarray
    actuator_rule: tuple[str, ...]
    sensor_map: np.ndarray
    sensor_signs: tuple[np.ndarray, ...]
    sensor_comp_perm: tuple[np.ndarray, ...]
    sensor_rule: tuple[str, ...]
    body_C: np.ndarray
    site_C: np.ndarray
    mocap_map: np.ndarray
    qpos: SignedPermutation
    qvel: SignedPermutation
    ctrl: SignedPermutation


def _pair_objects(
    model: mujoco.MjModel,
    objtype: Any,
    out: np.ndarray,
    rules: tuple[Sequence[tuple[str, str]], Sequence[tuple[str, str]]],
    consistent: Callable[[int, int], bool],
    candidates: Callable[[int], list[int]],
    label: str,
    on_pair: Callable[[int, int], None] | None = None,
) -> None:
    """Fill the involution *out* (-1 = unassigned) for one object kind.

    The name rule's partner is taken when it is mirror-consistent; otherwise the unique mirror-consistent candidate
    (a pairing by geometry, e.g. two geoms named by index).  No consistent partner, or more than one, raises.
    """
    names = [_name(model, objtype, i) for i in range(len(out))]
    index = {name: i for i, name in enumerate(names) if name}
    for i in range(len(out)):
        if out[i] >= 0:
            continue
        partner = index.get(mirror_name(names[i], *rules)) if names[i] else None
        if partner is not None and (out[partner] < 0 or out[partner] == i) and consistent(i, partner):
            choice = partner
        else:
            found = [c for c in candidates(i) if (out[c] < 0 or out[c] == i) and consistent(i, c)]
            if len(found) != 1:
                hint = f"the name rule gave {names[partner]!r}" if partner is not None else "no name-rule partner"
                raise SymmetryError(
                    f"{label} {names[i] or i!r}: {len(found)} mirror-consistent partners ({hint}); "
                    "the model is not mirror-symmetric here"
                )
            choice = found[0]
        out[i] = choice
        out[choice] = i
        if on_pair is not None:
            on_pair(i, choice)


def _geom_frame_error(model: mujoco.MjModel, g: int, p: int, correction: np.ndarray) -> float:
    """Frame mismatch of geom *p* against the mirror of geom *g*, up to the shape's own symmetry."""
    rg, rp = _quat_to_mat(model.geom_quat[g]), _quat_to_mat(model.geom_quat[p])
    expected = correction.T @ S @ rg @ S
    kind = int(model.geom_type[g])
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return 0.0
    if kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        return float(min(np.max(np.abs(rp[:, 2] - expected[:, 2])), np.max(np.abs(rp[:, 2] + expected[:, 2]))))
    # Boxes, ellipsoids and planes are symmetric under each local axis flip; a mirrored mesh's principal frame
    # (MuJoCo re-centres meshes on their inertia) is the mirror of the source's up to axis signs too.
    return float(
        max(min(np.max(np.abs(rp[:, k] - expected[:, k])), np.max(np.abs(rp[:, k] + expected[:, k]))) for k in range(3))
    )


def derive_model_mirror(  # noqa: C901 - one derivation, object kind by object kind
    model: mujoco.MjModel,
    *,
    prefix_rules: Sequence[tuple[str, str]] = DEFAULT_PREFIX_RULES,
    suffix_rules: Sequence[tuple[str, str]] = DEFAULT_SUFFIX_RULES,
    tol: float = 1e-9,
) -> ModelMirror:
    """Derive and verify every model-level mirror map; raises :class:`SymmetryError` on any inconsistency.

    *tol* bounds the compiled float parameters (positions, axes, ranges, inertias); the MJCFs are authored in
    decimals and mirror-authored pairs agree to about 1e-15.  Masses, inertias, contact and actuator parameters of a
    pair must be equal: a plant that is symmetric in structure but not in mass properties has no exact map.
    """
    rules = (prefix_rules, suffix_rules)
    m = model
    if m.ntendon or m.neq or m.npair or m.nplugin or m.na:
        raise SymmetryError(
            "tendons, equality constraints, explicit contact pairs, plugins and actuator activations are not "
            f"mirrored here (ntendon={m.ntendon} neq={m.neq} npair={m.npair} nplugin={m.nplugin} na={m.na})"
        )
    _require("opt.gravity", float(np.max(np.abs(m.opt.gravity * POLAR - m.opt.gravity))), tol)
    _require("opt.wind", float(np.max(np.abs(m.opt.wind * POLAR - m.opt.wind))), tol)

    # ---- bodies (parents precede children, so a parent's pair is known first)
    eye = np.eye(3)
    body_map = np.full(m.nbody, -1, dtype=np.int64)
    body_map[0] = 0
    body_C = np.zeros((m.nbody, 3, 3))
    body_C[0] = eye
    local_rot = [_quat_to_mat(m.body_quat[b]) for b in range(m.nbody)]

    def inertia(b: int) -> np.ndarray:
        r = _quat_to_mat(m.body_iquat[b])
        return np.asarray(r @ np.diag(m.body_inertia[b]) @ r.T)

    def correction(b: int, p: int) -> np.ndarray:
        return np.asarray(S @ local_rot[b].T @ S @ body_C[m.body_parentid[b]] @ local_rot[p])

    def body_errors(b: int, p: int) -> dict[str, float]:
        c_parent = body_C[m.body_parentid[b]]
        c = correction(b, p)
        ib = inertia(b)
        return {
            "position": float(np.max(np.abs(m.body_pos[p] - c_parent.T @ S @ m.body_pos[b]))),
            "mass": abs(float(m.body_mass[p] - m.body_mass[b])) / max(1.0, float(m.body_mass[b])),
            "inertial frame": float(np.max(np.abs(m.body_ipos[p] - c.T @ S @ m.body_ipos[b]))),
            "inertia": float(np.max(np.abs(inertia(p) - c.T @ S @ ib @ S @ c))) / max(1e-12, float(np.max(np.abs(ib)))),
        }

    # Inertial properties are compiled (MuJoCo diagonalises each body's inertia), so they agree less tightly than
    # the authored positions do; the bounds still demand a mirror-authored pair.
    body_tolerances = {"position": tol, "mass": 1e-9, "inertial frame": 1e-7, "inertia": 1e-6}

    def body_ok(b: int, p: int) -> bool:
        if b == 0 or p == 0:
            return b == p
        if m.body_parentid[p] != body_map[m.body_parentid[b]]:
            return False
        return all(error <= body_tolerances[key] for key, error in body_errors(b, p).items())

    def body_pair(b: int, p: int) -> None:
        body_C[b] = correction(b, p)
        body_C[p] = correction(p, b)

    _pair_objects(
        m,
        _OBJ.mjOBJ_BODY,
        body_map,
        rules,
        body_ok,
        lambda b: [c for c in range(1, m.nbody) if m.body_parentid[c] == body_map[m.body_parentid[b]]],
        "body",
        on_pair=body_pair,
    )
    for b in range(1, m.nbody):
        c = body_C[b]
        _require(
            f"frame correction of body {_name(m, _OBJ.mjOBJ_BODY, b)!r} is a proper rotation paired with its partner's",
            max(
                float(np.max(np.abs(c.T @ c - eye))),
                abs(float(np.linalg.det(c)) - 1.0),
                float(np.max(np.abs(body_C[body_map[b]] - S @ c.T @ S))),
            ),
            1e-9,
        )
    mocap_map = np.full(m.nmocap, -1, dtype=np.int64)
    for b in range(m.nbody):
        if m.body_mocapid[b] >= 0:
            mocap_map[m.body_mocapid[b]] = m.body_mocapid[body_map[b]]

    def to_partner(b: int) -> np.ndarray:
        """Body *b*'s local polar coordinates expressed in its partner's local frame."""
        return np.asarray(body_C[b].T @ S)

    # ---- geoms
    geom_map = np.full(m.ngeom, -1, dtype=np.int64)
    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)

    def geom_ok(g: int, p: int) -> bool:
        b = int(m.geom_bodyid[g])
        return bool(
            m.geom_bodyid[p] == body_map[b]
            and m.geom_type[p] == m.geom_type[g]
            and np.max(np.abs(m.geom_size[p] - m.geom_size[g])) <= tol
            and np.max(np.abs(m.geom_pos[p] - to_partner(b) @ m.geom_pos[g])) <= tol
            and _geom_frame_error(m, g, p, body_C[b]) <= 1e-7
            and (int(m.geom_type[g]) != mesh or m.mesh_vertnum[m.geom_dataid[g]] == m.mesh_vertnum[m.geom_dataid[p]])
        )

    _pair_objects(
        m,
        _OBJ.mjOBJ_GEOM,
        geom_map,
        rules,
        geom_ok,
        lambda g: [h for h in range(m.ngeom) if m.geom_bodyid[h] == body_map[m.geom_bodyid[g]]],
        "geom",
    )
    for g in range(m.ngeom):
        p = int(geom_map[g])
        same = (
            m.geom_contype[p] == m.geom_contype[g]
            and m.geom_conaffinity[p] == m.geom_conaffinity[g]
            and m.geom_condim[p] == m.geom_condim[g]
            and m.geom_priority[p] == m.geom_priority[g]
            and np.array_equal(m.geom_friction[p], m.geom_friction[g])
            and np.array_equal(m.geom_solref[p], m.geom_solref[g])
            and np.array_equal(m.geom_solimp[p], m.geom_solimp[g])
            and m.geom_margin[p] == m.geom_margin[g]
            and m.geom_gap[p] == m.geom_gap[g]
            and m.geom_solmix[p] == m.geom_solmix[g]
        )
        _require(
            f"geom {_name(m, _OBJ.mjOBJ_GEOM, g)!r} contact and solver parameters equal its partner's",
            float(not same),
            0.0,
        )
    # Contact exclusions (signature (body1 << 16) + body2) must map onto exclusions.
    excluded = {frozenset(((int(s) >> 16), (int(s) & 0xFFFF))) for s in m.exclude_signature}
    mirrored = {frozenset(int(body_map[b]) for b in pair) for pair in excluded}
    _require("contact exclusions map onto contact exclusions", float(len(excluded ^ mirrored)), 0.0)

    # ---- sites
    site_map = np.full(m.nsite, -1, dtype=np.int64)
    site_C = np.zeros((m.nsite, 3, 3))

    def site_correction(s: int, p: int) -> np.ndarray:
        return np.asarray(
            S @ _quat_to_mat(m.site_quat[s]).T @ S @ body_C[m.site_bodyid[s]] @ _quat_to_mat(m.site_quat[p])
        )

    def site_ok(s: int, p: int) -> bool:
        c = site_correction(s, p)
        size = np.abs(c).T @ m.site_size[s] if _is_signed_perm(c) else m.site_size[s]
        return bool(
            m.site_bodyid[p] == body_map[m.site_bodyid[s]]
            and m.site_type[p] == m.site_type[s]
            and np.max(np.abs(m.site_pos[p] - to_partner(int(m.site_bodyid[s])) @ m.site_pos[s])) <= tol
            and np.max(np.abs(m.site_size[p] - size)) <= tol
        )

    def site_pair(s: int, p: int) -> None:
        site_C[s] = site_correction(s, p)
        site_C[p] = site_correction(p, s)

    _pair_objects(
        m,
        _OBJ.mjOBJ_SITE,
        site_map,
        rules,
        site_ok,
        lambda s: [h for h in range(m.nsite) if m.site_bodyid[h] == body_map[m.site_bodyid[s]]],
        "site",
        on_pair=site_pair,
    )

    # ---- joints, qpos and qvel
    joint_map = np.full(m.njnt, -1, dtype=np.int64)

    def expected_axis(j: int) -> np.ndarray:
        local = to_partner(int(m.jnt_bodyid[j]))
        # A hinge axis is a pseudovector (-C^T S a), a slide axis a polar one (C^T S a).
        return np.asarray(-local @ m.jnt_axis[j] if int(m.jnt_type[j]) == _HINGE else local @ m.jnt_axis[j])

    def joint_ok(j: int, p: int) -> bool:
        if m.jnt_bodyid[p] != body_map[m.jnt_bodyid[j]] or m.jnt_type[p] != m.jnt_type[j]:
            return False
        if np.max(np.abs(m.jnt_pos[p] - to_partner(int(m.jnt_bodyid[j])) @ m.jnt_pos[j])) > tol:
            return False
        if int(m.jnt_type[j]) in (_HINGE, _SLIDE):
            e = expected_axis(j)
            return bool(min(np.max(np.abs(m.jnt_axis[p] - e)), np.max(np.abs(m.jnt_axis[p] + e))) <= tol)
        return True

    _pair_objects(
        m,
        _OBJ.mjOBJ_JOINT,
        joint_map,
        rules,
        joint_ok,
        lambda j: [h for h in range(m.njnt) if m.jnt_bodyid[h] == body_map[m.jnt_bodyid[j]]],
        "joint",
    )
    joint_sign = np.ones(m.njnt)
    joint_rule = [""] * m.njnt
    qpos_perm, qpos_sign = np.arange(m.nq), np.ones(m.nq)
    qpos_names, qpos_rules = [""] * m.nq, [""] * m.nq
    dof_perm, dof_sign = np.arange(m.nv), np.ones(m.nv)
    dof_names, dof_rules = [""] * m.nv, [""] * m.nv
    for j in range(m.njnt):
        p = int(joint_map[j])
        jname, pname = _name(m, _OBJ.mjOBJ_JOINT, j), _name(m, _OBJ.mjOBJ_JOINT, p)
        kind = int(m.jnt_type[j])
        qa, qp = int(m.jnt_qposadr[j]), int(m.jnt_qposadr[p])
        da, dp = int(m.jnt_dofadr[j]), int(m.jnt_dofadr[p])
        pure = bool(np.max(np.abs(body_C[m.jnt_bodyid[j]] - eye)) <= 1e-9)
        pair_word = "self" if p == j else f"pair {jname}<->{pname}"
        if kind in (_FREE, _BALL) and not pure:
            raise SymmetryError(f"joint {jname!r}: a free or ball joint needs pure mirror frames (C = I)")
        if kind == _FREE:
            if p != j or body_map[m.jnt_bodyid[j]] != m.jnt_bodyid[j]:
                raise SymmetryError(f"free joint {jname!r} must sit on a self-mirrored body")
            for k, (component, component_sign, why) in enumerate(
                (
                    ("x", 1, "root position (world, polar)"),
                    ("y", -1, "root position (world, polar): y flips"),
                    ("z", 1, "root position (world, polar)"),
                    ("qw", 1, "root quaternion: S R S keeps w"),
                    ("qx", -1, "root quaternion: S R S flips x"),
                    ("qy", 1, "root quaternion: S R S keeps y"),
                    ("qz", -1, "root quaternion: S R S flips z"),
                )
            ):
                qpos_sign[qa + k] = component_sign
                qpos_names[qa + k], qpos_rules[qa + k] = f"{jname}.{component}", why
            for k, (component, component_sign, why) in enumerate(
                (
                    ("vx", 1, "root linear velocity (world, polar)"),
                    ("vy", -1, "root linear velocity (world, polar): y flips"),
                    ("vz", 1, "root linear velocity (world, polar)"),
                    ("wx", -1, "root angular velocity (body, pseudovector): x flips"),
                    ("wy", 1, "root angular velocity (body, pseudovector): y keeps"),
                    ("wz", -1, "root angular velocity (body, pseudovector): z flips"),
                )
            ):
                dof_sign[da + k] = component_sign
                dof_names[da + k], dof_rules[da + k] = f"{jname}.{component}", why
            joint_rule[j] = (
                "free root: position polar, quaternion (w,-x,y,-z), velocity world polar, angular body pseudo"
            )
            _require(
                f"free joint {jname!r} reference pose is mirror-invariant",
                float(np.max(np.abs(m.qpos0[qa : qa + 7] * np.r_[POLAR, QUAT_SIGNS] - m.qpos0[qa : qa + 7]))),
                tol,
            )
            continue
        if kind == _BALL:
            for k in range(4):
                qpos_perm[qa + k], qpos_sign[qa + k] = qp + k, QUAT_SIGNS[k]
                qpos_names[qa + k] = f"{jname}.q{'wxyz'[k]}"
                qpos_rules[qa + k] = f"ball quaternion, {pair_word}: (w,-x,y,-z)"
            for k in range(3):
                dof_perm[da + k], dof_sign[da + k] = dp + k, PSEUDO[k]
                dof_names[da + k] = f"{jname}.w{'xyz'[k]}"
                dof_rules[da + k] = f"ball angular velocity, {pair_word}: pseudovector (-,+,-)"
            joint_rule[j] = f"ball, {pair_word}: quaternion (w,-x,y,-z), angular velocity pseudo"
            continue
        e = expected_axis(j)
        if np.max(np.abs(m.jnt_axis[p] - e)) <= tol:
            sign = 1.0
        elif np.max(np.abs(m.jnt_axis[p] + e)) <= tol:
            sign = -1.0
        else:  # pragma: no cover - joint_ok already refused this pairing
            raise SymmetryError(f"joint {jname!r}/{pname!r}: the axis is not the mirror of its partner's")
        joint_sign[j] = sign
        axis = _axis_label(m.jnt_axis[j]) if kind == _HINGE else "slide"
        joint_rule[j] = (
            f"{'hinge' if kind == _HINGE else 'slide'}, {pair_word}, {axis} axis: "
            f"the mirrored axis is {'+' if sign > 0 else '-'}the partner's -> sign {int(sign):+d}"
        )
        qpos_perm[qa], qpos_sign[qa], qpos_names[qa], qpos_rules[qa] = qp, sign, jname, joint_rule[j]
        dof_perm[da], dof_sign[da], dof_names[da], dof_rules[da] = dp, sign, jname, joint_rule[j]
        _require(
            f"joint {jname!r} reference qpos0 is the partner's, sign-mapped",
            abs(float(m.qpos0[qp] - sign * m.qpos0[qa])),
            tol,
        )
        _require(
            f"joint {jname!r} spring reference is the partner's, sign-mapped",
            abs(float(m.qpos_spring[qp] - sign * m.qpos_spring[qa])),
            tol,
        )
        if m.jnt_limited[j] or m.jnt_limited[p]:
            _require(
                f"joint {jname!r} range is the partner's, sign-mapped",
                max(
                    float(m.jnt_limited[j] != m.jnt_limited[p]),
                    float(np.max(np.abs(m.jnt_range[p] - _sign_mapped_range(m.jnt_range[j], sign)))),
                ),
                tol,
            )
        _require(
            f"joint {jname!r} stiffness, damping, armature, frictionloss, solref, solimp and margin equal the partner's",
            max(
                abs(float(m.jnt_stiffness[p] - m.jnt_stiffness[j])),
                abs(float(m.dof_damping[dp] - m.dof_damping[da])),
                abs(float(m.dof_armature[dp] - m.dof_armature[da])),
                abs(float(m.dof_frictionloss[dp] - m.dof_frictionloss[da])),
                float(np.max(np.abs(m.jnt_solref[p] - m.jnt_solref[j]))),
                float(np.max(np.abs(m.jnt_solimp[p] - m.jnt_solimp[j]))),
                abs(float(m.jnt_margin[p] - m.jnt_margin[j])),
            ),
            0.0,
        )

    # ---- actuators (joint transmissions only)
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise SymmetryError(f"actuator {_name(m, _OBJ.mjOBJ_ACTUATOR, a)!r}: only joint transmissions are mirrored")
    actuator_map = np.full(m.nu, -1, dtype=np.int64)

    def actuator_ok(a: int, p: int) -> bool:
        return bool(
            m.actuator_trnid[p, 0] == joint_map[m.actuator_trnid[a, 0]]
            and abs(abs(float(m.actuator_gear[p, 0])) - abs(float(m.actuator_gear[a, 0]))) <= tol
        )

    _pair_objects(m, _OBJ.mjOBJ_ACTUATOR, actuator_map, rules, actuator_ok, lambda a: list(range(m.nu)), "actuator")
    actuator_sign = np.ones(m.nu)
    actuator_rule = [""] * m.nu
    for a in range(m.nu):
        p = int(actuator_map[a])
        aname, pname = _name(m, _OBJ.mjOBJ_ACTUATOR, a), _name(m, _OBJ.mjOBJ_ACTUATOR, p)
        ja = int(m.actuator_trnid[a, 0])
        ga, gp = float(m.actuator_gear[a, 0]), float(m.actuator_gear[p, 0])
        sign = float(joint_sign[ja] * np.sign(ga) * np.sign(gp))
        actuator_sign[a] = sign
        actuator_rule[a] = (
            f"{'self' if p == a else f'pair {aname}<->{pname}'}: drives {_name(m, _OBJ.mjOBJ_JOINT, ja)} "
            f"(joint sign {int(joint_sign[ja]):+d}) -> sign {int(sign):+d}"
        )
        _require(f"actuator {aname!r} has a scalar gear", float(np.max(np.abs(m.actuator_gear[a, 1:]))), tol)
        _require(
            f"actuator {aname!r} control range is the partner's, sign-mapped",
            max(
                float(m.actuator_ctrllimited[p] != m.actuator_ctrllimited[a]),
                float(np.max(np.abs(m.actuator_ctrlrange[p] - _sign_mapped_range(m.actuator_ctrlrange[a], sign)))),
            ),
            tol,
        )
        _require(
            f"actuator {aname!r} force range is the partner's, sign-mapped",
            max(
                float(m.actuator_forcelimited[p] != m.actuator_forcelimited[a]),
                float(np.max(np.abs(m.actuator_forcerange[p] - _sign_mapped_range(m.actuator_forcerange[a], sign)))),
            ),
            tol,
        )
        _require(
            f"actuator {aname!r} dynamics, gain and bias equal the partner's",
            max(
                float(m.actuator_dyntype[p] != m.actuator_dyntype[a])
                + float(m.actuator_gaintype[p] != m.actuator_gaintype[a])
                + float(m.actuator_biastype[p] != m.actuator_biastype[a]),
                float(np.max(np.abs(m.actuator_gainprm[p] - m.actuator_gainprm[a]))),
                float(np.max(np.abs(m.actuator_biasprm[p] - m.actuator_biasprm[a]))),
                float(np.max(np.abs(m.actuator_dynprm[p] - m.actuator_dynprm[a]))),
            ),
            0.0,
        )
        if sign < 0:
            # force = gain(l, v) ctrl + bias(l, v) must be odd in (ctrl, l, v): no length or velocity gain terms and
            # no constant bias.
            _require(
                f"sign -1 actuator {aname!r} is odd (no constant bias, no length or velocity gain)",
                max(float(np.max(np.abs(m.actuator_gainprm[a, 1:3]))), abs(float(m.actuator_biasprm[a, 0]))),
                0.0,
            )

    # ---- sensors
    st = mujoco.mjtSensor
    object_maps = {
        int(_OBJ.mjOBJ_SITE): site_map,
        int(_OBJ.mjOBJ_BODY): body_map,
        int(_OBJ.mjOBJ_XBODY): body_map,
        int(_OBJ.mjOBJ_GEOM): geom_map,
        int(_OBJ.mjOBJ_JOINT): joint_map,
        int(_OBJ.mjOBJ_ACTUATOR): actuator_map,
    }
    sensor_map = np.full(m.nsensor, -1, dtype=np.int64)

    def sensor_ok(s: int, p: int) -> bool:
        kind = int(m.sensor_objtype[s])
        obj = object_maps[kind][m.sensor_objid[s]] if kind in object_maps else m.sensor_objid[s]
        return bool(
            m.sensor_type[p] == m.sensor_type[s]
            and m.sensor_objtype[p] == m.sensor_objtype[s]
            and m.sensor_dim[p] == m.sensor_dim[s]
            and m.sensor_reftype[p] == m.sensor_reftype[s]
            and m.sensor_objid[p] == obj
        )

    _pair_objects(m, _OBJ.mjOBJ_SENSOR, sensor_map, rules, sensor_ok, lambda s: list(range(m.nsensor)), "sensor")
    by_type: dict[int, tuple[str, str]] = {}
    for type_names, rule_kind, rule_why in (
        (("mjSENS_TOUCH", "mjSENS_RANGEFINDER", "mjSENS_CLOCK"), "scalar", "scalar, mirror-invariant"),
        (("mjSENS_ACCELEROMETER", "mjSENS_VELOCIMETER", "mjSENS_FORCE"), "site_polar", "polar, in its site frame"),
        (("mjSENS_GYRO", "mjSENS_TORQUE"), "site_pseudo", "pseudovector, in its site frame"),
        (
            ("mjSENS_FRAMEPOS", "mjSENS_FRAMELINVEL", "mjSENS_FRAMELINACC", "mjSENS_SUBTREECOM"),
            "world_polar",
            "polar, world frame",
        ),
        (
            ("mjSENS_FRAMEANGVEL", "mjSENS_FRAMEANGACC", "mjSENS_SUBTREEANGMOM"),
            "world_pseudo",
            "pseudovector, world frame",
        ),
        (("mjSENS_FRAMEXAXIS", "mjSENS_FRAMEZAXIS"), "polar_axis", "frame x or z axis: polar"),
        (("mjSENS_FRAMEYAXIS",), "pseudo_axis", "frame y axis: pseudo"),
        (("mjSENS_FRAMEQUAT", "mjSENS_BALLQUAT"), "quat", "quaternion (w,-x,y,-z)"),
    ):
        for type_name in type_names:
            if hasattr(st, type_name):
                by_type[int(getattr(st, type_name))] = (rule_kind, rule_why)
    joint_scalar = {
        int(getattr(st, n)) for n in ("mjSENS_JOINTPOS", "mjSENS_JOINTVEL", "mjSENS_JOINTACTFRC") if hasattr(st, n)
    }
    actuator_scalar = {
        int(getattr(st, n))
        for n in ("mjSENS_ACTUATORPOS", "mjSENS_ACTUATORVEL", "mjSENS_ACTUATORFRC")
        if hasattr(st, n)
    }

    def frame_correction(s: int) -> np.ndarray | None:
        objtype, obj = int(m.sensor_objtype[s]), int(m.sensor_objid[s])
        if objtype == int(_OBJ.mjOBJ_SITE):
            return np.asarray(site_C[obj])
        if objtype == int(_OBJ.mjOBJ_XBODY):
            return np.asarray(body_C[obj])
        return None

    sensor_signs: list[np.ndarray] = []
    sensor_comp_perm: list[np.ndarray] = []
    sensor_rule: list[str] = []
    for s in range(m.nsensor):
        p = int(sensor_map[s])
        stype, dim = int(m.sensor_type[s]), int(m.sensor_dim[s])
        sname = _name(m, _OBJ.mjOBJ_SENSOR, s)
        if int(m.sensor_reftype[s]) not in (0, int(_OBJ.mjOBJ_UNKNOWN)):
            raise SymmetryError(f"sensor {sname!r}: a sensor with a reference frame is not mirrored here")
        components = np.arange(dim)
        if stype in by_type:
            rule, why = by_type[stype]
            if rule == "scalar":
                signs = np.ones(dim)
            elif rule in ("site_polar", "site_pseudo"):
                c = frame_correction(p)  # the partner's correction maps its frame onto this sensor's
                if c is None:
                    raise SymmetryError(f"sensor {sname!r}: a site-frame sensor on a non-site object")
                t = c.T @ S * (1.0 if rule == "site_polar" else -1.0)
                if not _is_signed_perm(t):
                    raise SymmetryError(f"sensor {sname!r}: its site frame correction is not a signed permutation")
                components = np.argmax(np.abs(t), axis=1)
                signs = np.sign(t[np.arange(3), components])
            elif rule == "world_polar":
                signs = POLAR.copy()
            elif rule == "world_pseudo":
                signs = PSEUDO.copy()
            else:
                c = frame_correction(s)
                if c is not None and not np.allclose(c, eye, atol=1e-9):
                    raise SymmetryError(f"sensor {sname!r}: a frame sensor on a frame with a correction C != I")
                signs = {"polar_axis": POLAR, "pseudo_axis": PSEUDO, "quat": QUAT_SIGNS}[rule].copy()
        elif stype in joint_scalar:
            signs = np.array([joint_sign[m.sensor_objid[s]]])
            why = f"joint scalar, joint sign {int(signs[0]):+d}"
        elif stype in actuator_scalar:
            signs = np.array([actuator_sign[m.sensor_objid[s]]])
            why = f"actuator scalar, actuator sign {int(signs[0]):+d}"
        else:
            raise SymmetryError(f"sensor {sname!r}: type {stype} has no mirror rule here")
        if len(signs) != dim:
            raise SymmetryError(f"sensor {sname!r}: dimension {dim} against a rule of width {len(signs)}")
        sensor_signs.append(np.asarray(signs, dtype=np.float64))
        sensor_comp_perm.append(np.asarray(components, dtype=np.int64))
        sensor_rule.append(f"{'self' if p == s else f'pair {sname}<->{_name(m, _OBJ.mjOBJ_SENSOR, p)}'}: {why}")

    ctrl_names = tuple(_name(m, _OBJ.mjOBJ_ACTUATOR, a) for a in range(m.nu))
    return ModelMirror(
        body_map=body_map,
        geom_map=geom_map,
        site_map=site_map,
        joint_map=joint_map,
        joint_sign=joint_sign,
        joint_rule=tuple(joint_rule),
        actuator_map=actuator_map,
        actuator_sign=actuator_sign,
        actuator_rule=tuple(actuator_rule),
        sensor_map=sensor_map,
        sensor_signs=tuple(sensor_signs),
        sensor_comp_perm=tuple(sensor_comp_perm),
        sensor_rule=tuple(sensor_rule),
        body_C=body_C,
        site_C=site_C,
        mocap_map=mocap_map,
        qpos=SignedPermutation(qpos_perm, qpos_sign, tuple(qpos_names), tuple(qpos_rules)),
        qvel=SignedPermutation(dof_perm, dof_sign, tuple(dof_names), tuple(dof_rules)),
        ctrl=SignedPermutation(actuator_map.copy(), actuator_sign.copy(), ctrl_names, tuple(actuator_rule)),
    )


# ---------------------------------------------------------------------------------------------------- observation
def _sensor_at(model: mujoco.MjModel, address: int, sensor_type: Any = None) -> int:
    hits = [s for s in range(model.nsensor) if int(model.sensor_adr[s]) == int(address)]
    if len(hits) != 1:
        raise SymmetryError(f"no unique sensor at sensordata address {address}")
    if sensor_type is not None and int(model.sensor_type[hits[0]]) != int(sensor_type):
        raise SymmetryError(f"the sensor at sensordata address {address} is not of type {sensor_type}")
    return hits[0]


def heading_free_observation_keys(env: Any) -> tuple[list[tuple[Any, ...]], list[str]]:
    """The semantic key and the name of every index of a ``bipedal-target-heading-free/v1`` observation.

    The layout is the plant contract's (``policy_layer._heading_free_observation_segments``): ``qpos[7:]``,
    ``qvel[6:]``, the root's projected gravity, the gyro, the heading-frame root velocity (forward, left, up), the
    accelerometer, one summed touch channel per foot (``env._foot_sensor_groups``), the target direction in root
    coordinates, the target distance and the body-relative command.
    """
    m = env.model
    free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == _FREE]
    if len(free) != 1 or int(m.jnt_qposadr[free[0]]) != 0 or int(m.jnt_dofadr[free[0]]) != 0:
        raise SymmetryError("the heading-free layout needs one free root joint at qpos[0:7] / qvel[0:6]")
    joints_q = sorted((j for j in range(m.njnt) if j != free[0]), key=lambda j: int(m.jnt_qposadr[j]))
    joints_v = sorted((j for j in range(m.njnt) if j != free[0]), key=lambda j: int(m.jnt_dofadr[j]))
    if [int(m.jnt_qposadr[j]) for j in joints_q] != list(range(7, m.nq)) or [
        int(m.jnt_dofadr[j]) for j in joints_v
    ] != list(range(6, m.nv)):
        raise SymmetryError("qpos[7:] and qvel[6:] must hold one scalar per non-root joint")
    keys: list[tuple[Any, ...]] = []
    names: list[str] = []

    def add(key: tuple[Any, ...], name: str) -> None:
        keys.append(key)
        names.append(name)

    for j in joints_q:
        add(("qpos", _name(m, _OBJ.mjOBJ_JOINT, j)), f"qpos.{_name(m, _OBJ.mjOBJ_JOINT, j)}")
    for j in joints_v:
        add(("qvel", _name(m, _OBJ.mjOBJ_JOINT, j)), f"qvel.{_name(m, _OBJ.mjOBJ_JOINT, j)}")
    for c, axis in enumerate("xyz"):
        add(("vec", "projected_gravity", c, "polar"), f"projected_gravity.{axis}")
    gyro = _name(m, _OBJ.mjOBJ_SENSOR, _sensor_at(m, env._sensor_gyro_start, mujoco.mjtSensor.mjSENS_GYRO))
    for c, axis in enumerate("xyz"):
        add(("sensor", gyro, c), f"{gyro}.{axis}")
    for c, axis in enumerate(("forward", "left", "up")):
        add(("vec", "heading_linear_velocity", c, "polar"), f"heading_linear_velocity.{axis}")
    accel = _name(m, _OBJ.mjOBJ_SENSOR, _sensor_at(m, env._sensor_accel_start, mujoco.mjtSensor.mjSENS_ACCELEROMETER))
    for c, axis in enumerate("xyz"):
        add(("sensor", accel, c), f"{accel}.{axis}")
    for group in env._foot_sensor_groups:
        sensors = tuple(_name(m, _OBJ.mjOBJ_SENSOR, _sensor_at(m, address)) for address in group)
        add(("sensor_sum", sensors), "foot_contact." + "+".join(sensors))
    for c, axis in enumerate("xyz"):
        add(("vec", "target_direction", c, "polar"), f"target_direction.{axis}")
    add(("scalar", "target_distance"), "target_distance")
    for component in COMMAND_COMPONENTS:
        add(("command", component), f"command.{component}")
    if len(keys) != int(np.prod(env.observation_space.shape)):
        raise SymmetryError(f"the layout has {len(keys)} entries, the observation {env.observation_space.shape}")
    return keys, names


#: The observation layout of each ``observation_schema`` (configs/plant_versions.toml) that has one.
OBSERVATION_LAYOUTS: dict[str, Callable[[Any], tuple[list[tuple[Any, ...]], list[str]]]] = {
    HEADING_FREE_OBSERVATION_SCHEMA: heading_free_observation_keys,
}


def _mirror_key(mm: ModelMirror, model: mujoco.MjModel, key: tuple[Any, ...]) -> tuple[tuple[Any, ...], float, str]:
    """The mirror of one observation key, its sign and the rule that gives them."""
    kind = key[0]
    if kind in ("qpos", "qvel"):
        j = mujoco.mj_name2id(model, _OBJ.mjOBJ_JOINT, key[1])
        if j < 0 or int(model.jnt_type[j]) not in (_HINGE, _SLIDE):
            raise SymmetryError(f"{key}: only scalar joints are supported in a joint segment")
        partner = _name(model, _OBJ.mjOBJ_JOINT, int(mm.joint_map[j]))
        return (kind, partner), float(mm.joint_sign[j]), f"{kind} of {mm.joint_rule[j]}"
    if kind == "vec":
        _, segment, component, _vector = key
        sign = float(POLAR[component])
        return (
            key,
            sign,
            f"{segment}[{component}], a polar vector in a self-mirrored frame (y flips): sign {int(sign):+d}",
        )
    if kind == "sensor":
        _, sensor, component = key
        s = mujoco.mj_name2id(model, _OBJ.mjOBJ_SENSOR, sensor)
        partner = _name(model, _OBJ.mjOBJ_SENSOR, int(mm.sensor_map[s]))
        source = int(mm.sensor_comp_perm[s][component])
        return (
            ("sensor", partner, source),
            float(mm.sensor_signs[s][component]),
            f"{sensor}[{component}] {mm.sensor_rule[s]}",
        )
    if kind == "sensor_sum":
        partners = []
        for sensor in key[1]:
            s = mujoco.mj_name2id(model, _OBJ.mjOBJ_SENSOR, sensor)
            if int(model.sensor_type[s]) != int(mujoco.mjtSensor.mjSENS_TOUCH):
                raise SymmetryError(f"{key}: a summed group must hold touch sensors")
            partners.append(_name(model, _OBJ.mjOBJ_SENSOR, int(mm.sensor_map[s])))
        side = "self" if tuple(partners) == tuple(key[1]) else "swapped"
        return ("sensor_sum", tuple(partners)), 1.0, f"a foot's summed touch ({side}): scalar"
    if kind == "scalar":
        return key, 1.0, f"{key[1]}: a mirror-invariant scalar"
    if kind == "command":
        return key, COMMAND_SIGNS[key[1]], f"body-relative command {key[1]}: sign {int(COMMAND_SIGNS[key[1]]):+d}"
    raise SymmetryError(f"unknown observation key kind {kind!r}")


def derive_observation_map(
    mm: ModelMirror, model: mujoco.MjModel, keys: Sequence[tuple[Any, ...]], names: Sequence[str]
) -> SignedPermutation:
    """The observation map ``M_o`` of a layout of semantic keys (one per observation index)."""
    index: dict[Hashable, int] = {}
    for i, key in enumerate(keys):
        if key in index:
            raise SymmetryError(f"duplicate observation key {key}")
        index[key] = i
    perm = np.zeros(len(keys), dtype=np.int64)
    sign = np.zeros(len(keys))
    rules = []
    for i, key in enumerate(keys):
        mirrored, sign[i], why = _mirror_key(mm, model, key)
        if mirrored not in index:
            raise SymmetryError(f"observation {names[i]}: its mirror {mirrored} is not in the observation")
        perm[i] = index[mirrored]
        rules.append(why)
    return SignedPermutation(perm, sign, tuple(names), tuple(rules))


# ---------------------------------------------------------------------------------------------------- env level
@dataclass(frozen=True)
class EnvMirror:
    """A plant's model maps and its policy-interface maps (``obs`` = ``M_o``, ``action`` = ``M_a``)."""

    model: ModelMirror
    obs: SignedPermutation
    action: SignedPermutation
    foot_perm: np.ndarray
    height_check_perm: np.ndarray


def build_env_mirror(env: Any, observation_schema: str) -> EnvMirror:
    """Derive *env*'s maps and check that its action mapping commutes with the actuator map."""
    model = env.model
    mm = derive_model_mirror(model)
    if observation_schema not in OBSERVATION_LAYOUTS:
        raise SymmetryError(
            f"no mirror layout for observation schema {observation_schema!r} (have {sorted(OBSERVATION_LAYOUTS)})"
        )
    keys, names = OBSERVATION_LAYOUTS[observation_schema](env)
    obs = derive_observation_map(mm, model, keys, names)
    if tuple(env.action_space.shape) != (model.nu,):
        raise SymmetryError(f"the action is not one entry per actuator: {env.action_space.shape} vs nu {model.nu}")
    action = SignedPermutation(
        mm.actuator_map.copy(),
        mm.actuator_sign.copy(),
        tuple(f"action.{name}" for name in mm.ctrl.names),
        mm.actuator_rule,
    )
    # The env maps a normalised action to ctrl (residuals around the home control on the dinosaurs): the map is
    # only an action map if that scaling commutes with it, scale(M_a a) == M_ctrl scale(a).
    rng = np.random.default_rng(0)
    probes = [
        np.zeros(model.nu),
        np.ones(model.nu),
        -np.ones(model.nu),
        np.full(model.nu, 0.5),
        np.full(model.nu, -0.5),
    ]
    probes += list(rng.uniform(-1.2, 1.2, size=(64, model.nu)))
    worst = max(
        float(np.max(np.abs(env._scale_action(action.apply(p)) - mm.ctrl.apply(env._scale_action(p))))) for p in probes
    )
    _require("the env's action scaling commutes with the actuator map", worst, 1e-9)
    groups = [tuple(int(a) for a in group) for group in getattr(env, "_foot_sensor_groups", ())]
    foot_perm = np.arange(len(groups))
    for i, group in enumerate(groups):
        mirrored = tuple(int(model.sensor_adr[mm.sensor_map[_sensor_at(model, a)]]) for a in group)
        if mirrored not in groups:
            raise SymmetryError(f"foot group {group} mirrors to {mirrored}, which is not a declared group")
        foot_perm[i] = groups.index(mirrored)
    checks = list(getattr(env, "_substep_height_checks", ()))
    height_perm = np.arange(len(checks))
    for i, (kind, element) in enumerate(checks):
        partner = int(mm.site_map[element]) if kind == "site" else int(mm.body_map[element])
        if (kind, partner) not in checks:
            raise SymmetryError(f"height check {(kind, element)} mirrors outside the declared checks")
        height_perm[i] = checks.index((kind, partner))
    return EnvMirror(mm, obs, action, foot_perm, height_perm)


def mirror_physics_state(mm: ModelMirror, model: mujoco.MjModel, src: Any, dst: Any) -> None:
    """Write the mirror of *src*'s primary physics state into *dst* (``MjData``; may be the same).  No forward pass.

    Time, ``qpos``, ``qvel``, ``ctrl``, the warm start, the applied forces and the mocap bodies (the target).
    """
    values = {
        "qpos": mm.qpos.apply(np.asarray(src.qpos, dtype=np.float64)),
        "qvel": mm.qvel.apply(np.asarray(src.qvel, dtype=np.float64)),
        "ctrl": mm.ctrl.apply(np.asarray(src.ctrl, dtype=np.float64)),
        "qacc_warmstart": mm.qvel.apply(np.asarray(src.qacc_warmstart, dtype=np.float64)),
        "qfrc_applied": mm.qvel.apply(np.asarray(src.qfrc_applied, dtype=np.float64)),
    }
    xfrc = np.asarray(src.xfrc_applied, dtype=np.float64)[mm.body_map].copy()
    xfrc[:, 0:3] *= POLAR
    xfrc[:, 3:6] *= PSEUDO
    mocap_pos = np.asarray(src.mocap_pos)[mm.mocap_map] * POLAR
    mocap_quat = np.asarray(src.mocap_quat)[mm.mocap_map] * QUAT_SIGNS
    dst.time = float(src.time)
    for name, value in values.items():
        getattr(dst, name)[:] = value
    dst.xfrc_applied[:] = xfrc
    if model.nmocap:
        dst.mocap_pos[:] = mocap_pos
        dst.mocap_quat[:] = mocap_quat


def _mirror_env_attributes(src: Any, em: EnvMirror) -> dict[str, Any]:
    """The mirrored value of every episode-state attribute of a ``BaseDinoEnv`` that *src* has."""
    have = vars(src)
    if have.get("direction_controller") is not None or have.get("_command_state") is not None:
        raise SymmetryError("a live command mode's controller state is not mirrored here")
    if have.get("_quad_touchdown_sequence"):
        raise SymmetryError("quadruped touchdown labels are not mirrored here")
    out: dict[str, Any] = {}

    def action(x: Any) -> Any:
        return None if x is None else em.action.apply(np.asarray(x, dtype=np.float64)).astype(np.asarray(x).dtype)

    def xy(x: Any) -> Any:
        return None if x is None else (np.asarray(x) * np.asarray([1, -1], dtype=np.asarray(x).dtype)).copy()

    if "_command" in have:
        command = np.asarray(src._command)
        signs = np.asarray([COMMAND_SIGNS[c] for c in COMMAND_COMPONENTS], dtype=command.dtype)
        out["_command"] = (command * signs).astype(command.dtype)
    for name in ("_prev_action", "_prev_prev_action", "_policy_action", "_action_filter_state"):
        if name in have:
            out[name] = action(have[name])
    for name in ("_initial_prey_dir_2d", "_initial_pos_2d", "_prev_pos_2d"):
        if name in have:
            out[name] = xy(have[name])
    if "_prev_r_in_contact" in have:
        out["_prev_r_in_contact"], out["_prev_l_in_contact"] = src._prev_l_in_contact, src._prev_r_in_contact
    if "_touchdown_sequence" in have:
        out["_touchdown_sequence"] = [{"R": "L", "L": "R"}[x] for x in src._touchdown_sequence]
    if "_prev_diag_a_in_contact" in have:
        out["_prev_diag_a_in_contact"] = src._prev_diag_b_in_contact
        out["_prev_diag_b_in_contact"] = src._prev_diag_a_in_contact
    if "_substep_min_foot_forces" in have:
        forces = have["_substep_min_foot_forces"]
        out["_substep_min_foot_forces"] = None if forces is None else np.asarray(forces)[em.foot_perm].copy()
    if "_substep_foot_forces" in have:
        block = have["_substep_foot_forces"]
        out["_substep_foot_forces"] = None if block is None else np.asarray(block)[:, em.foot_perm].copy()
    if "_substep_min_heights" in have:
        heights = have["_substep_min_heights"]
        out["_substep_min_heights"] = None if heights is None else np.asarray(heights)[em.height_check_perm].copy()
    if "_substep_floor_hit_geom" in have:
        geom = have["_substep_floor_hit_geom"]
        out["_substep_floor_hit_geom"] = None if geom is None else int(em.model.geom_map[geom])
    if have.get("_push_schedule_directions") is not None:
        out["_push_schedule_directions"] = xy(have["_push_schedule_directions"])
    for name in (
        "_push_schedule_starts",
        "_step_count",
        "_policy_action_step",
        "_substep_contact_step",
        "_distance_traveled",
        "_prev_prey_distance",
        "_stance_width_target",
    ):
        if name in have:
            value = have[name]
            out[name] = value.copy() if isinstance(value, np.ndarray) else value
    return out


def mirror_env_state(src: Any, dst: Any, em: EnvMirror, *, forward: bool = True) -> None:
    """Write the mirror image of *src*'s simulator and episode state into *dst* (``dst is src`` mirrors in place).

    With *forward*, ``mj_forward`` recomputes every derived quantity (positions, sensors, contacts) at the mirrored
    state.  Mid-episode the source's own derived quantities are those of its last substep's pre-integration state,
    so mirror a source after an ``mj_forward`` of its own when comparing observations at that instant; every later
    step is consistent either way.
    """
    if dst.model.nq != src.model.nq or dst.model.nu != src.model.nu:
        raise SymmetryError("the source and destination envs must share the model")
    attributes = _mirror_env_attributes(src, em)
    mirror_physics_state(em.model, src.model, src.data, dst.data)
    for name, value in attributes.items():
        setattr(dst, name, value)
    if forward:
        mujoco.mj_forward(dst.model, dst.data)


# ---------------------------------------------------------------------------------------------------- the map file
def mirror_map_path(model_id: ModelId, configs_root: Path = CONFIGS_ROOT) -> Path:
    """``configs/<species>/<embodiment>/mirror_map.json``."""
    return model_id.path(configs_root) / MIRROR_MAP_FILENAME


def mirror_map_record(model_id: ModelId) -> dict[str, Any]:
    """The map file's content for *model_id*, derived from its current plant (the env built with its defaults)."""
    from .plant_contract import current_plant_identity
    from .plant_contract.versions import load_plant_versions
    from .species_registry import get_species_config

    identity = current_plant_identity(model_id)
    schema = load_plant_versions()[1][model_id].observation_schema
    env = get_species_config(model_id).env_class()
    try:
        em = build_env_mirror(env, schema)
    finally:
        env.close()
    return {
        "schema": MIRROR_MAP_SCHEMA,
        **model_id.record(),
        "physics_sha256": identity.physics_sha256,
        "policy_interface_sha256": identity.policy_interface_sha256,
        "observation_schema": schema,
        "reflection": "world xz-plane, S = diag(1, -1, 1): a body pose (p, R) maps to (S p, S R S)",
        "convention": "(M x)[index] = sign * x[mirror_index]; each map is a signed permutation and an involution",
        "generated_by": "python -m environments.shared.symmetry --write --species <species> --embodiment <embodiment>",
        "observation": em.obs.entries(),
        "action": em.action.entries(),
    }


def render_mirror_map(record: Mapping[str, Any]) -> str:
    """The file text of a map record: one JSON object, each map row on a line of its own, a trailing newline."""
    lines = []
    for key, value in record.items():
        if isinstance(value, list):
            rows = ",\n".join("  " + json.dumps(row, ensure_ascii=False) for row in value)
            lines.append(f" {json.dumps(key)}: [\n{rows}\n ]")
        else:
            lines.append(f" {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def write_mirror_map(model_id: ModelId, configs_root: Path = CONFIGS_ROOT) -> Path:
    path = mirror_map_path(model_id, configs_root)
    path.write_text(render_mirror_map(mirror_map_record(model_id)), encoding="utf-8")
    return path


def committed_mirror_maps(configs_root: Path = CONFIGS_ROOT) -> list[ModelId]:
    """The manifest models that have a committed map file."""
    from .species_names import model_ids

    return [model_id for model_id in model_ids() if mirror_map_path(model_id, configs_root).is_file()]


def stale_mirror_maps(configs_root: Path = CONFIGS_ROOT) -> list[str]:
    """One line per committed map that differs from a fresh derivation (empty when every map is current)."""
    problems = []
    for model_id in committed_mirror_maps(configs_root):
        path = mirror_map_path(model_id, configs_root)
        try:
            expected = render_mirror_map(mirror_map_record(model_id))
        except SymmetryError as exc:
            problems.append(f"{path}: the model no longer derives a mirror map: {exc}")
            continue
        if path.read_text(encoding="utf-8") != expected:
            problems.append(
                f"{path} is stale; regenerate it with python -m environments.shared.symmetry --write "
                f"--species {model_id.species} --embodiment {model_id.embodiment}"
            )
    return problems


@dataclass(frozen=True)
class MirrorMap:
    """The observation and action maps of one plant, loaded from its committed file and bound to it (fail closed)."""

    model_id: ModelId
    obs: SignedPermutation
    action: SignedPermutation
    sha256: str
    path: str

    @classmethod
    def load(
        cls,
        model_id: ModelId,
        plant_identity: "PlantIdentity | Mapping[str, Any] | None",
        *,
        configs_root: Path = CONFIGS_ROOT,
        verify: bool = True,
    ) -> MirrorMap:
        """Load *model_id*'s map and refuse it unless it names this model and this plant.

        The file's ``physics_sha256`` and ``policy_interface_sha256`` must equal *plant_identity*'s (a
        ``PlantIdentity`` or its ``to_dict()``), and its widths the plant's observation and action widths.  With
        *verify* (the default; about a second) the file must also equal a fresh derivation from the current plant,
        byte for byte, so a hand-edited map is refused too.
        """
        if plant_identity is None:
            raise SymmetryError("a mirror map is loaded against the plant identity it is bound to (fail closed)")
        identity = dict(plant_identity) if isinstance(plant_identity, Mapping) else plant_identity.to_dict()
        path = mirror_map_path(model_id, configs_root)
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise SymmetryError(f"cannot read the mirror map of {model_id} at {path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema") != MIRROR_MAP_SCHEMA:
            raise SymmetryError(f"{path}: not a {MIRROR_MAP_SCHEMA} file")
        try:
            names_model = model_id.matches(data)
        except ValueError as exc:
            raise SymmetryError(f"{path} does not name its model: {exc}") from exc
        if not names_model or not model_id.matches(identity):
            raise SymmetryError(f"{path} and the plant identity must both name {model_id}")
        problems = [
            f"{key}: map {data.get(key)!r}, plant {identity.get(key)!r}"
            for key in ("physics_sha256", "policy_interface_sha256")
            if data.get(key) != identity.get(key)
        ]
        if problems:
            raise SymmetryError(
                f"{path} is bound to another plant (fail closed): {'; '.join(problems)}. Regenerate it with "
                f"python -m environments.shared.symmetry --write --species {model_id.species} "
                f"--embodiment {model_id.embodiment} after checking that the plant is still mirror-symmetric"
            )
        obs = SignedPermutation.from_entries(data.get("observation"), f"{path} observation")
        action = SignedPermutation.from_entries(data.get("action"), f"{path} action")
        if (obs.dim, action.dim) != (identity.get("observation_dim"), identity.get("action_dim")):
            raise SymmetryError(
                f"{path}: widths {obs.dim}/{action.dim} differ from the plant's "
                f"{identity.get('observation_dim')}/{identity.get('action_dim')}"
            )
        if verify and raw.decode("utf-8") != render_mirror_map(mirror_map_record(model_id)):
            raise SymmetryError(
                f"{path} differs from a fresh derivation from the current plant (fail closed); check it with "
                "python -m environments.shared.symmetry --check"
            )
        return cls(model_id, obs, action, hashlib.sha256(raw).hexdigest(), str(path))


# ---------------------------------------------------------------------------------------------------- training switch
#: The ``[ppo]`` keys of mirror-symmetric training.  ``symmetry = "mirror_kl"`` selects MirrorKLPPO; the others are
#: optional and default to the values validated on the T. rex stance (SymmetryConfig).  Absent (or
#: ``symmetry = "none"``), training is stock PPO.
SYMMETRY_KEYS = ("symmetry", "symmetry_kappa", "symmetry_sigma_floor", "symmetry_mirror_critic")
SYMMETRY_METHODS = ("none", "mirror_kl")
#: The SB3 checkpoint attribute (JSON) that records how a checkpoint was trained.
CHECKPOINT_ATTRIBUTE = "mesozoic_symmetry"
CHECKPOINT_SCHEMA = "mesozoic.mirror-kl/v1"


@dataclass(frozen=True)
class SymmetryConfig:
    """MirrorKLPPO's settings.  The defaults are the T. rex stance screen's (arm L, 2026-10): kappa 0.3, a sigma
    floor of 0.01 in the mirror loss and the mirrored critic on."""

    kappa: float = 0.3
    sigma_floor: float = 0.01
    mirror_critic: bool = True
    method: str = "mirror_kl"

    def __post_init__(self) -> None:
        if self.method != "mirror_kl":
            raise SymmetryError(f"symmetry method {self.method!r}: the one method is 'mirror_kl'")
        for name in ("kappa", "sigma_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SymmetryError(f"[ppo] symmetry_{name} must be a finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if self.kappa < 0.0 or self.sigma_floor <= 0.0:
            raise SymmetryError(
                f"[ppo] needs symmetry_kappa >= 0 and symmetry_sigma_floor > 0, got {self.kappa} / {self.sigma_floor}"
            )
        if not isinstance(self.mirror_critic, bool):
            raise SymmetryError(f"[ppo] symmetry_mirror_critic must be true or false, got {self.mirror_critic!r}")

    def checkpoint_record(self, mirror_map: MirrorMap) -> dict[str, Any]:
        """The ``mesozoic_symmetry`` record a checkpoint trained with these settings carries."""
        return {
            "schema": CHECKPOINT_SCHEMA,
            "method": self.method,
            **mirror_map.model_id.record(),
            "mirror_map_sha256": f"sha256:{mirror_map.sha256}",
            "kappa": self.kappa,
            "sigma_floor": self.sigma_floor,
            "mirror_critic": self.mirror_critic,
        }


def pop_symmetry_config(ppo_kwargs: dict[str, Any]) -> SymmetryConfig | None:
    """Remove the :data:`SYMMETRY_KEYS` from *ppo_kwargs* (in place): the config, or ``None`` for stock PPO."""
    present = {key: ppo_kwargs.pop(key) for key in SYMMETRY_KEYS if key in ppo_kwargs}
    if not present:
        return None
    method = present.get("symmetry")
    if method not in SYMMETRY_METHODS:
        raise SymmetryError(
            f"[ppo] symmetry must be one of {SYMMETRY_METHODS}, got {method!r}"
            + ("" if "symmetry" in present else f" (the table sets {sorted(present)} without it)")
        )
    if method == "none":
        return None
    settings = {
        name: present[f"symmetry_{name}"]
        for name in ("kappa", "sigma_floor", "mirror_critic")
        if f"symmetry_{name}" in present
    }
    return SymmetryConfig(**settings)


def symmetry_mismatch(recorded: Any, wanted: Mapping[str, Any] | None) -> str | None:
    """Why a checkpoint's recorded ``mesozoic_symmetry`` and the stage's differ (``None`` when they agree)."""
    if recorded is None and wanted is None:
        return None
    if wanted is None:
        return f"the checkpoint was trained mirror-symmetrically ({recorded!r}) and the stage declares no symmetry"
    if recorded is None:
        return f"the stage declares mirror-symmetric training and the checkpoint records none ({CHECKPOINT_ATTRIBUTE})"
    if not isinstance(recorded, Mapping) or dict(recorded) != dict(wanted):
        return f"the checkpoint records {recorded!r}, the stage {dict(wanted)!r}"
    return None


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write or verify the committed mirror maps (configs/*/*/mirror_map.json)"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="every committed map must equal a fresh derivation")
    action.add_argument("--write", action="store_true", help="derive and write one model's map")
    parser.add_argument("--species")
    parser.add_argument("--embodiment")
    args = parser.parse_args(argv)
    if args.write:
        if not args.species or not args.embodiment:
            parser.error("--write needs --species and --embodiment")
        path = write_mirror_map(ModelId(args.species, args.embodiment))
        print(f"Wrote {path.relative_to(REPOSITORY_ROOT)}")
        return 0
    problems = stale_mirror_maps()
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"Mirror maps are current ({', '.join(str(m) for m in committed_mirror_maps()) or 'none committed'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
