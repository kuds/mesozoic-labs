"""Shared code for the candidate checks of 28 Nov 2026 (``design/PREREG_CANDIDATES_2026_11_28.md``, experiments E1-E5).
Code only: importing this module or running its tests runs no experiment. Every figure it produces is DESK (geometry,
arithmetic, the build's thermal model) or SIMULATION (MuJoCo runs of the build's env); nothing is measured.

What is here, by experiment:

- **E1 clash reach.** ``StepLogger`` (a per-physics-step hook: the 16 leg joint angles every 10 ms, and from 2 s on each
  leg drive's mean squared torque split into motoring and braking samples, its drive-side mean square and a histogram
  of its speed); ``ClashGeometry`` (the clearance of clash pairs #9, the front and hind legs of one side, and #10, the
  two hind legs, at any logged pose, with ``sim/check_packaging60.py``'s parts, probe rods and distance calls);
  ``candidate_stops`` (the packaging record's candidate stops, ``check_packaging60.LIMITS``, mirrored to both sides),
  ``apply_stop``, ``evaluate_stop``, ``coupled_limit`` and ``e1_verdict``.
- **E2 transmission and speed cap.** ``drive_sq_from_split`` and ``drive_sq_bounds`` (the reduction's desk mapping of
  joint torque to drive torque), ``reduced_profile``, ``heat_rules`` (C1 and C2 through the form-factor study's
  ``heat``, i.e. ``rl/telemetry/thermal.py`` at 35 C, method A), ``speed_check`` (99th percentile and peak of the drive
  speed from ``StepLogger`` histograms) and ``e2a_speed_check`` (the same on the reduced drives only);
  ``transmission_stiffness`` (rod, toothed belt, cable, from the dated sources in ``SOURCES``), ``write_variant`` (the
  adopted model with each reduced drive as a rotor coupled to its joint through a fixed-tendon spring, or reduced and
  rigid, written to a scratch folder), ``write_timestep_copy`` (the adopted model with only its option timestep
  changed: the baselines B025 and B0125), ``e2b_model`` (every E2(b) variant's model by name), ``SpeedCap`` (the 83 rpm
  drive speed cap as a per-step torque envelope), ``e2b_compare``, ``e2b_verdict`` and ``e2b_judged_runs`` (which
  step and baselines each variant is judged at, section 3 item 4 of the pre-registration).
- **E3 per-policy heat.** ``policy_heat`` (each drive's demo-loop RMS over continuous and hottest winding at 35 C, the
  deciding drive and its mirror, the even split) and ``demo_make`` (a ``rl/demo_course.py`` walker built through the
  robustness sweep's loaders, for the policies with no stored demo course).
- **E4 forced air.** ``forced_air_c1`` (C1 with the fitted thermal resistances scaled by a sourced forced/passive
  ratio; used only if a source gives one, and only for a winding-to-ambient or total resistance pair).
- **E5 retuned trot.** ``PitchedTrot`` (``walk/walk60.py``'s ``TrotController`` with a body-pitch setpoint: its attitude
  reading offset and its foot targets shifted; at 0 deg exactly the frozen controller), ``e5_settings`` (the stage-1
  candidate settings, fixed here before any run), ``trot_judged_episode``, ``trot_push_episode`` and ``e5_success``.
- Shared: ``CandidateEnv`` (the sweep's ``PointEnv`` with extra per-step hooks, reset hooks and, for a series variant,
  the joint-to-rotor sync at reset), ``policy_env_config`` (refuses a model whose own timestep is not the step asked
  for), ``policy_episode`` (one E1 / E2(b) episode through ``sweep.run_episode`` with the 10 ms delay at any physics
  step), ``verify_checkpoint`` and ``load_policy`` (only through the sweep's loaders: ``gait_pipe1.load_obs_stats``, a
  restricted unpickler, and ``build_policy``, ``torch.load(weights_only=True)``), ``provenance`` (the commit and the
  sha256 of this file, the runner script and the build tools, for every result file), ``plan`` (the frozen settings,
  written to ``results/candidates_2026-11-28/plan.json``).

Usage (from hardware/triceratops_build):
  python sim/candidate_checks.py plan [--out sim/results/candidates_2026-11-28/plan.json]
The plan command verifies the policy checkpoints' sha256 against their stored sweeps and writes the frozen settings;
it runs no simulation.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
for _p in (HERE, BUILD / "rl", BUILD / "rl" / "analysis", BUILD / "rl" / "analytics", BUILD / "rl" / "robustness",
           BUILD / "walk", BUILD / "design"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATE = "2026-11-28"
MODEL = HERE / "generated" / "trike60_C_closed_capsule_fore128.xml"   # the adopted 60 cm model (read only)
RESULTS = HERE / "results" / "candidates_2026-11-28"
SCRATCH = Path(os.environ.get("TRIKE_SCRATCH",
                              "/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad"))
CAND_SCRATCH = SCRATCH / "candidates"
POLICY_DIRS = {**{f"fore128-s{i}": SCRATCH / "d22_runs" / f"fore128-s{i}" / "checkpoints" for i in range(4)},
               **{f"fore160-s{i}": SCRATCH / "d22_runs" / f"fore160-s{i}" / "checkpoints" for i in range(4)},
               "run3-s0": SCRATCH / "run3_s0" / "checkpoints", "run3-s1": SCRATCH / "run3_s1" / "checkpoints",
               "run2b": SCRATCH / "run2b_ckpt"}
SWEEP_RESULTS = BUILD / "rl" / "robustness" / "results"
DEMO_DIR = BUILD / "rl" / "analysis" / "results" / "d22" / "demo"
PACKAGING_JSON = HERE / "results" / "packaging60_2026-11-27" / "packaging60.json"

CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
CHAIN = ("roll", "pitch", "flexion", "ankle")
LOG_JOINTS = tuple(f"{c}_{a}" for c in CORNERS for a in CHAIN)          # 16 joints: the clash geometry's
LEG_ACTUATORS = tuple(f"{c}_{a}" for c in CORNERS for a in ("roll", "pitch", "flexion"))
LOG_EVERY_S = 0.010
WINDOW_S = 2.0                 # the build's measuring window (rl/demo_course.py WINDOW_S, rl/gait_metrics.py)
CLEAR_MM = 5.0                 # E1: a stop keeps clearance >= 5 mm ...
CLIP_SHARE = 0.01              # ... and clips at most 1 % of samples
CAP_RPM = 83.0                 # the X4's rated-speed cap above 24 V (DECISION_LOG.md E1); E2 applies it to every
PEAK_FACTOR = 1.1              # reduced drive, the X6 too (stricter than the X6's own 108 rpm allowance)
ROOM_C = 35.0                  # D48
C1_LIMIT_C, C2_LIMIT = 90.0, 0.80
SCORED_TC_S = 480.0            # the build's headline thermal time constant (8 min); 4 and 15 min reported
REDUCTIONS = (1.0, 1.1, 1.2, 1.25, 1.3)
PLACEMENTS = {"pitch": ("pitch",), "knees": ("flexion",), "both": ("pitch", "flexion")}
SIM_R = 1.25                   # E2(b): the reduction simulated, at hip pitch and the knees together
ETA_OLD = {"roll": 1.0, "pitch": 1.0, "flexion": 0.90}   # sim/data/drives.json transmission_efficiency (1:1 linkages)
THERMAL_EQ_NM = {"X4-36-L": 7.5, "X6-60": 12.0}         # drive-side continuous ratings (sim/data/drives.json)
OUT_SUFFIX = "__out"           # a series variant's leg (output) joint; the rotor takes the joint's own name
SERIES_DT = 0.00025            # E2(b): physics step of the variants (the rod's stiffness needs it; section E2)
FALLBACK_DT = 0.000125         # E2(b): the step a run with an "unstable_simulation" episode is rerun at (item 4)
UNSTABLE = "unstable_simulation"   # rl/trike_env.py's termination for a non-finite state or a huge acceleration
DAMPING_RATIO = 0.05           # E2(b): the transmission's structural damping, against the rotor (an assumption)
DELAY_MS = 10.0                # the measuring delay (the sweep's nominal 10 ms), at every physics step

# ------------------------------------------------------------------------------------------------- seeds, conditions
E1_BLOCKS = {"A": (700000, 700001, 700002, 700003), "B": (710000, 710001, 710002, 710003)}
POLICY_CONDITIONS = {  # name: (command m/s, sideways push dv m/s; + left, - right; at 8 s for 0.1 s, the sweep's)
    "walk_0.25": (0.25, 0.0), "walk_0.15": (0.15, 0.0), "stand": (0.0, 0.0),
    "push_left_0.5": (0.25, 0.5), "push_right_0.5": (0.25, -0.5),
    "push_left_1.0": (0.25, 1.0), "push_right_1.0": (0.25, -1.0)}
E1_POLICIES = tuple(f"fore128-s{i}" for i in range(4))
E1_TROT_SEEDS = (101, 102, 103, 104)
TROT_PUSHES = {"push_back_0.5": ("fore_aft", -0.5), "push_right_0.5": ("sideways", -0.5)}
E2B_CONDITIONS = ("walk_0.25", "push_left_0.5", "push_right_0.5")
E5_SCREEN_SEEDS = (101, 102, 103, 104, 105)
E5_TUNING_SEEDS = tuple(range(101, 111))
E5_HELDOUT = {"301-310": tuple(range(301, 311)), "401-410": tuple(range(401, 411))}
E5_PUSHES = {f"push_{d}_{v}": (ax, -v) for d, ax in (("back", "fore_aft"), ("right", "sideways"))
             for v in (0.3, 0.4, 0.5)}
E5_SCREEN_CONDITIONS = ("J1", "J3", "push_back_0.5", "push_right_0.5")
E5_COMMAND = 0.25
E5_MAX_SETTINGS = 60

# ------------------------------------------------------------------------------------------------------------- sources
SOURCES = {
    "steel_E_pa": {"value": 200e9, "units": "Pa",
                   "source": "steel, E 200 GPa: matweb AISI 1018 datasheet (web-search summary, page not opened), as "
                             "the packaging record cites it (design/PACKAGING_PUSHROD_60CM.md, Sources)",
                   "read": "2026-11-27"},
    "m8_stress_area_m2": {"value": 36.6e-6, "units": "m^2",
                          "source": "M8 tensile stress area 36.6 mm^2: engineeringtoolbox.com / roymech.org (web-search "
                                    "summary), as the packaging record cites it; the class 8.8 rolled M8 stud there "
                                    "passes every rod check", "read": "2026-11-27"},
    "rod_min_transmission_deg": {"value": 43.0, "units": "deg",
                                 "source": "the 1:1 parallelogram's minimum transmission angle at -47 deg flexion "
                                           "(design/pushrod60.py, design/PACKAGING_PUSHROD_60CM.md section 2.1; desk)",
                                 "read": "2026-11-27"},
    "crank_drive_m": {"value": 0.028, "units": "m",
                      "source": "the build's 28.00 mm cranks (design/PACKAGING_PUSHROD_60CM.md section 2.1; desk)",
                      "read": "2026-11-27"},
    "gt2_3m_modulus_lbf_per_in": {"value": 30000.0, "units": "lbf per inch of belt width per unit strain",
                                  "source": "Gates 'Synchronous Belt Data Summary: Belt Tensile Properties' (dated 3/07, "
                                            "neoprene/fiberglass construction; hosted at fab.cba.mit.edu, "
                                            "BeltTensileProperties.pdf): 3M PowerGrip GT2 row. Web-search summaries, "
                                            "page not opened (host outside this session's network); the column "
                                            "reading is inferred: the same sheet's 2M GT2 row was summarised as an "
                                            "average tensile modulus of 18,000 lb/in/in on a unit-strain basis, and "
                                            "the 3M row's 2,200 lb / 9,786 N is its total ultimate strength",
                                  "read": "2026-11-28"},
    "wire_7x19_E_pa": {"value": 97.1e9, "units": "Pa",
                       "source": "Ronstan 7x19 316 stainless wire rope, modulus 97.1 kN/mm^2 (14.1 x 10^6 psi): "
                                 "theboatwarehouse.com.au Ronstan listing and ronstantensilearch.com (web-search "
                                 "summaries, pages not opened)", "read": "2026-11-28"},
    "wire_7x19_3mm": {"value": {"diameter_m": 0.003, "metallic_area_m2": 4.6e-6, "breaking_n": 5300.0},
                      "units": "m, m^2, N",
                      "source": "Ronstan RON-WR6719-03M, 3.0 mm 7x19 316: metallic area 4.6 mm^2, breaking load 5.3 "
                                "kN (theboatwarehouse.com.au listing; web-search summary; another supplier lists "
                                "4.69 kN; Ronstan's ratings exclude the typical 10 % swaging loss)",
                      "read": "2026-11-28"},
    "cable_bend_ratio": {"value": 18.0, "units": "minimum bend radius / cable radius",
                         "source": "about 18:1 for steel cable near 3 mm diameter (a web-search summary of cable-driven "
                                   "robot-arm papers on arxiv.org, 2507.15693 and 2502.12963; the paper giving it was "
                                   "not identified with certainty; used only to choose the pulley radius)",
                         "read": "2026-11-28"},
    "eta_belt": {"value": 0.98, "units": "-",
                 "source": "synchronous (toothed) belts about 98 % efficient (98-99 % in Gates' energy-savings "
                           "material for PowerGrip GT2; 98 % in a US DOE tip sheet citing Gates; vendor-sponsored; "
                           "web-search summary)", "read": "2026-11-28"},
    "eta_cable": {"value": 0.96, "units": "-",
                  "source": "98-99 % per passive pulley at pulley-to-wire diameter ratio 15 (earlier work cited in "
                            "arxiv.org 2509.11506, 'Design and Development of a Remotely Wire-Driven Walking Robot'; "
                            "web-search summary): two pulleys at the lower figure, 0.98^2 = 0.9604, taken as 0.96",
                  "read": "2026-11-28"},
    "eta_rod": {"value": 0.95, "units": "-",
                "source": "no dated source found for a rod-end linkage (28 Nov 2026), so the lead's rule's default "
                          "(E2: 'a stated transmission efficiency from a dated source, 0.95 if none'); judged",
                "read": "2026-11-28"},
    "eta_rod_build": {"value": 0.90, "units": "-",
                      "source": "the build's own 1:1 linkage efficiency, sim/data/drives.json transmission_efficiency "
                                "(added in dc2e405d, 18 Nov 2026; no source field): an undated, unsourced build "
                                "figure, so reported beside the judged 0.95 only (supporting, not judged)",
                      "read": "2026-11-28"},
    "eta_default": {"value": 0.95, "units": "-", "source": "the lead's rule (E2): 0.95 when no dated source",
                    "read": "2026-11-28"},
    "belt_installation_tension": {"value": None, "units": "N per span",
                                  "source": "no sourced installation tension for a 15 mm 3M GT2 belt was found (28 Nov "
                                            "2026); both spans taut needs a pretension of at least half the effective "
                                            "tension (belt_pretension), so a one-span belt (half the stiffness) is run "
                                            "as a supporting variant (E2(b) belt_1span)",
                                  "read": "2026-11-28"},
}
# Transmission geometry (desk choices, fixed before any run). Spans: the hind and fore upper links' pitch-to-knee
# lengths in the adopted model (fore128.xml:186, :250); the hip pitch drive placed 100 mm from the pitch axis in the
# body (an assumption: nothing is drawn).
SPANS_M = {"pitch": 0.100, "fore_flexion": float(math.hypot(0.0946259392, 0.0861970511)),
           "hind_flexion": float(math.hypot(0.120121, 0.14462))}
BELT = {"profile": "3M PowerGrip GT2", "pitch_m": 0.003, "width_m": 0.015, "teeth_drive": 56}
TRANSMISSIONS = {
    "rod": {"eta": SOURCES["eta_rod"]["value"], "label": "M8 class 8.8 rolled stud and rod ends, cranks 28 mm (drive) "
                                                         "and r x 28 mm (joint)"},
    "belt": {"eta": SOURCES["eta_belt"]["value"], "label": "3M GT2 toothed belt, 15 mm wide, 56-tooth drive pulley "
                                                           "and a 56 r-tooth joint pulley, both spans pretensioned"},
    "cable": {"eta": SOURCES["eta_cable"]["value"], "label": "3.0 mm 7x19 316 stainless wire rope loop, pulleys 28 mm "
                                                             "(drive) and r x 28 mm (joint), both strands pretensioned"},
}
# Supporting (not judged) efficiencies, reported beside the judged ones: the rod at the build's own 0.90. In E2(a) it
# adds rows (its r = 1.0 knee row is today's linkage exactly, the control row); in E2(b) the judged rod run's torques
# are re-mapped to the drive at 0.90 (``relog_drive_sq``) with the run's own (0.95) force limits.
SUPPORTING_ETA = {"rod": SOURCES["eta_rod_build"]["value"]}
DEFAULT_STRANDS = {"rod": 1, "belt": 2, "cable": 2}


def _r(x, n=6):
    return None if x is None else float(f"{float(x):.{n}g}")


def axis_of(joint: str) -> str:
    for a in ("roll", "pitch", "flexion", "ankle"):
        if joint.endswith("_" + a):
            return a
    raise ValueError(joint)


def drive_of(joint: str) -> str:
    return "X6-60" if (axis_of(joint) == "flexion" and "_fore_" in joint) else "X4-36-L"


def span_of(joint: str) -> float:
    a = axis_of(joint)
    if a == "pitch":
        return SPANS_M["pitch"]
    if a == "flexion":
        return SPANS_M["fore_flexion"] if "_fore_" in joint else SPANS_M["hind_flexion"]
    raise ValueError(f"{joint}: only hip pitch and knee joints are reduced")


def reduced_joints(placement: str) -> tuple:
    axes = PLACEMENTS[placement]
    return tuple(j for j in LEG_ACTUATORS if axis_of(j) in axes)


# --------------------------------------------------------------------------------------------------- lazy imports
def _sweep():
    import sweep as S  # rl/robustness/sweep.py (imports analysis/gait_pipe1.py and rl/trike_env.py)
    return S


def _cp():
    import check_packaging60 as CP
    return CP


def _ff():
    import form_factor_study as FF
    return FF


def _walk60():
    import walk60
    return walk60


def _f128():
    import fore128_walk60 as F
    return F


def _sg():
    import fore128_slow_gait as SG
    return SG


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ================================================================================================= policies (loaders)
def stored_sweep(run: str) -> dict:
    return json.loads(gzip.decompress((SWEEP_RESULTS / run / f"sweep_{run}.json.gz").read_bytes()).decode())


def verify_checkpoint(run: str, ckpt_dir: Path | None = None) -> dict:
    """The final checkpoint pair of ``run`` (``sweep.find_checkpoint``) and its run record, after checking both files'
    sha256 against the ones its stored sweep records (``rl/robustness/results/<run>/sweep_<run>.json.gz``). Raises
    ValueError on any mismatch; nothing in the files is loaded here."""
    S = _sweep()
    cdir = Path(ckpt_dir) if ckpt_dir is not None else POLICY_DIRS[run]
    zp, pk = S.find_checkpoint(cdir)
    ck = stored_sweep(run)["checkpoint"]
    got = {"zip_sha256": sha256_file(zp), "vecnormalize_sha256": sha256_file(pk)}
    if zp.name != ck["zip_name"] or pk.name != ck["vecnormalize_name"]:
        raise ValueError(f"{run}: checkpoint names {zp.name}, {pk.name} differ from the stored sweep's")
    for k, v in got.items():
        if v != ck[k]:
            raise ValueError(f"{run}: {k} {v} differs from the stored sweep's {ck[k]}")
    rec_path = cdir / "run_record.json"
    if not rec_path.exists():
        rec_path = cdir.parent / "run_record.json"
    return {"run": run, "zip": zp, "pkl": pk, "record": json.loads(rec_path.read_text()), **got, "match": True}


def run_env_config(run: str) -> dict:
    return dict(verify_checkpoint(run)["record"]["settings"]["env_config"])


def load_policy(run: str, env):
    """(obs statistics, policy) through the sweep's loaders only, after ``verify_checkpoint``."""
    S = _sweep()
    v = verify_checkpoint(run)
    st = v["record"]["settings"]
    stats = S.G.load_obs_stats(Path(v["pkl"]))
    assert stats[0].shape == env.observation_space.shape, (stats[0].shape, env.observation_space.shape)
    policy = S.G.build_policy(env, Path(v["zip"]), st["net"], st["log_std_init"])
    return stats, policy


def policy_act(stats, policy):
    """obs -> deterministic action, exactly as sweep.run_episode and steer_eval.load_policy compute it."""
    import torch
    mean, var, clip, eps = stats

    def act(obs):
        o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        with torch.no_grad():
            a, _, _ = policy(torch.as_tensor(o[None]), deterministic=True)
        return a.numpy()[0]
    return act


# ============================================================================================================ the env
def candidate_env_class():
    """The robustness sweep's ``PointEnv`` (``rl/trike_env.py``'s env, a sweep point applied at reset) with:
    ``extra_hooks`` called after the caller's ``substep_hook`` every physics step; ``reset_hooks`` called at the end
    of every reset (``hook.on_reset(env)``); ``sync_pairs`` [(rotor qpos address, leg joint qpos address)] copied
    rotor -> leg joint at reset, after the reset noise and before the root is placed again and the observation is
    taken, so a series variant starts with its springs unloaded (the rotor carries the reset noise the leg joint
    carries in the adopted model)."""
    S = _sweep()
    if hasattr(S, "_CandidateEnv"):
        return S._CandidateEnv
    import mujoco

    class CandidateEnv(S.PointEnv):
        def __init__(self, config=None, **kw):
            self.extra_hooks, self.reset_hooks, self.sync_pairs, self._cand_ready = [], [], [], False
            super().__init__(config, **kw)
            self._cand_ready = True

        @property
        def substep_hook(self):
            user = self.__dict__.get("_user_hook")
            extra = self.__dict__.get("extra_hooks") or []
            if user is None and not extra:
                return None

            def hook(e):
                if user is not None:
                    user(e)
                for h in extra:
                    h(e)
            return hook

        @substep_hook.setter
        def substep_hook(self, value):
            self.__dict__["_user_hook"] = value

        def _reset_episode_state(self, command, delay):
            super()._reset_episode_state(command, delay)
            if self.__dict__.get("_cand_ready") and self.sync_pairs:
                m, d = self.model, self.data
                for rq, oq in self.sync_pairs:
                    d.qpos[oq] = d.qpos[rq]
                mujoco.mj_kinematics(m, d)
                d.qpos[2] += self.cfg.reset_sole_clearance - self._lowest_sole_z()
                mujoco.mj_forward(m, d)

        def reset(self, *, seed=None, options=None):
            out = super().reset(seed=seed, options=options)
            for h in self.reset_hooks:
                h.on_reset(self)
            return out

    S._CandidateEnv = CandidateEnv
    return CandidateEnv


def model_timestep(model_path: Path | str) -> float:
    import mujoco
    return float(mujoco.MjModel.from_xml_path(str(model_path)).opt.timestep)


def policy_env_config(ec: dict, model_path: Path | str | None = None, physics_dt: float = 0.001):
    """The sweep's measuring config for a run (``sweep.build_env``: the run record's config on the flat floor,
    randomisation off, observation noise as recorded), on ``model_path`` (default the run's own model in
    sim/generated), with the 50 Hz control kept at ``physics_dt`` (frame_skip x physics_dt = the record's). The env
    takes its physics step from the model's own ``<option timestep>``, so a model whose timestep is not
    ``physics_dt`` is refused (ValueError): it would run at its own step with a frame_skip meant for another, i.e. a
    different control rate. Use ``write_timestep_copy`` or ``write_variant`` for a model at another step."""
    S = _sweep()
    G = S.G
    path = Path(model_path) if model_path is not None else G.GEN / Path(ec["model_path"]).name
    own = model_timestep(path)
    if abs(own - physics_dt) > 1e-12:
        raise ValueError(f"{path.name}: its timestep is {own:g} s, not the physics step {physics_dt:g} s asked for")
    cfg = G.flat_floor(S.TrikeEnvConfig.from_dict({**ec, "model_path": str(path)})).replace(domain_randomization=False)
    control_dt = cfg.frame_skip * 0.001      # every run record here is at the build's 1 ms step
    fs = int(round(control_dt / physics_dt))
    if abs(fs * physics_dt - control_dt) > 1e-12:
        raise ValueError(f"physics step {physics_dt} does not divide the control step {control_dt}")
    return cfg.replace(frame_skip=fs)


def delay_steps_for(physics_dt: float, delay_ms: float = DELAY_MS) -> int:
    """The env's delay is counted in physics steps: 10 ms at ``physics_dt``."""
    return int(round(delay_ms * 1e-3 / physics_dt))


def make_env(cfg, sync_pairs=None):
    """A ``CandidateEnv`` on ``cfg``; a series variant's rotor / leg joint pairs are found in its model when
    ``sync_pairs`` is None."""
    env = candidate_env_class()(cfg)
    if sync_pairs is None:
        sync_pairs = sync_pairs_for(env.model)
    if sync_pairs:
        env.sync_pairs = list(sync_pairs)
    return env


def trot_sweep_config(model_path: Path | str):
    """The trot's sweep plant (``fore128_walk60.env_config`` then ``sweep.build_env``: walk60.plant_config on the
    model, flat floor, randomisation off) for any model path, including a scaled model in a scratch folder."""
    S, walk60 = _sweep(), _walk60()
    ec = walk60.plant_config(model_path=str(model_path)).to_dict()
    return S.G.flat_floor(S.TrikeEnvConfig.from_dict(ec)).replace(domain_randomization=False)


def policy_point(condition: str, physics_dt: float) -> dict:
    """The sweep point of a POLICY_CONDITIONS condition (command, sideways push dv) with the 10 ms servo delay at
    ``physics_dt``. The sweep's point counts its "delay_ms" in physics steps (it is passed straight to the env's
    ``delay_steps``, which is in physics steps, = ms only at 1 ms), so it is set to ``delay_steps_for(physics_dt)``:
    10 at 1 ms, 40 at 0.25 ms, 80 at 0.125 ms."""
    cmd, dv = POLICY_CONDITIONS[condition]
    return _sweep().make_point(command=cmd, push_dv=dv, delay_ms=delay_steps_for(physics_dt))


def policy_episode(env, seed: int, condition: str, policy=None, stats=None, controller=None, hooks=()) -> dict:
    """One E1 / E2(b) episode through ``sweep.run_episode`` (the sweep's own record and fall rule) at
    ``policy_point(condition, env.physics_dt)``, with ``hooks`` (e.g. a ``StepLogger``, a ``SpeedCap``) appended to the
    env's per-step hooks, and to its reset hooks if they have ``on_reset``, for this episode only. ``policy`` and
    ``stats`` (``load_policy``) go to the sweep's own policy path; ``controller`` (obs -> action) instead, for tests.
    Returns the sweep's record with the delay actually used (``delay_steps``, ``delay_s``), the physics and control
    steps, and ``unstable`` (termination "unstable_simulation"). Raises if the delay is not 10 ms."""
    S = _sweep()
    p = policy_point(condition, env.physics_dt)
    added_step, added_reset = [], []
    for h in hooks:
        env.extra_hooks.append(h)
        added_step.append(h)
        if hasattr(h, "on_reset"):
            env.reset_hooks.append(h)
            added_reset.append(h)
    try:
        rec = S.run_episode(env, policy, stats, seed, p, controller=controller)
    finally:
        for h in added_step:
            env.extra_hooks.remove(h)
        for h in added_reset:
            env.reset_hooks.remove(h)
    delay_s = env.delay_steps * env.physics_dt
    if env.delay_steps != p["delay_ms"] or abs(delay_s - DELAY_MS * 1e-3) > 1e-12:
        raise RuntimeError(f"delay {env.delay_steps} steps x {env.physics_dt:g} s is not {DELAY_MS:g} ms")
    rec.update(condition=condition, delay_steps=int(env.delay_steps), delay_s=float(delay_s),
               physics_dt=float(env.physics_dt), control_dt=float(env.dt), unstable=rec["termination"] == UNSTABLE)
    return rec


# ===================================================================================================== E1: the logger
class StepLogger:
    """A per-physics-step hook (``CandidateEnv.extra_hooks`` and ``reset_hooks``). Every ``every_s`` (10 ms): the angles
    of ``joint_names`` (rad; for a series variant, the leg joints' output names). From ``window_s`` (2 s, as
    ``rl/demo_course.py``) on, for each of the 12 leg drives, with tau and omega the actuator's force and velocity
    (joint-side units) and drive-side tau_d = tau / (r eta) when tau x omega >= 0 (motoring), tau x eta / r otherwise
    (``rl/demo_course.py``'s rule with a reduction r): the mean of tau^2 over motoring and over braking samples
    (``sq_motor``, ``sq_brake``: each sum over the window's sample count), the mean of tau_d^2 (``drive_sq``), and a
    histogram of the drive speed |omega| x r (rpm, ``rpm_bin`` bins, the last bin open) with its exact peak."""

    def __init__(self, env, joint_names=LOG_JOINTS, r=None, eta=None, every_s=LOG_EVERY_S, window_s=WINDOW_S,
                 rpm_bin=0.1, rpm_max=300.0):
        m = env.model
        self.names = list(joint_names)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.names], dtype=np.int64)
        self.idx = np.asarray(env._leg_act, dtype=np.int64)
        self.acts = [m.actuator(int(i)).name for i in self.idx]
        self.r = np.ones(12) if r is None else np.array([float(r.get(a, 1.0)) for a in self.acts])
        self.eta = (np.array([ETA_OLD[axis_of(a)] for a in self.acts]) if eta is None
                    else np.array([float(eta.get(a, ETA_OLD[axis_of(a)])) for a in self.acts]))
        dt = float(m.opt.timestep)
        self.every = max(1, int(round(every_s / dt)))
        self.window = float(window_s)
        self.rpm_bin, self.nbins = float(rpm_bin), int(round(rpm_max / rpm_bin)) + 1
        self.reset()

    def reset(self):
        self.n = 0
        self.t, self.q = [], []
        self.nw = 0
        self.sm, self.sb, self.sd = np.zeros(12), np.zeros(12), np.zeros(12)
        self.hist = np.zeros((12, self.nbins), dtype=np.int64)
        self.peak = np.zeros(12)

    def on_reset(self, env):
        self.reset()

    def __call__(self, env):
        d = env.data
        self.n += 1
        if self.n % self.every == 0:
            self.t.append(float(d.time))
            self.q.append(d.qpos[self.qadr].astype(np.float32))
        if d.time >= self.window:
            tau = d.actuator_force[self.idx]
            om = d.actuator_velocity[self.idx]
            mot = tau * om >= 0
            t2 = tau * tau
            self.sm += np.where(mot, t2, 0.0)
            self.sb += np.where(mot, 0.0, t2)
            td = np.where(mot, tau / (self.r * self.eta), tau * self.eta / self.r)
            self.sd += td * td
            rpm = np.abs(om) * self.r * (60.0 / (2.0 * math.pi))
            self.peak = np.maximum(self.peak, rpm)
            b = np.minimum((rpm / self.rpm_bin).astype(np.int64), self.nbins - 1)
            self.hist[np.arange(12), b] += 1
            self.nw += 1

    def result(self) -> dict:
        """The log. Poses with a non-finite angle (an episode ending "unstable_simulation" can log some before the
        env stops it) are dropped from ``t`` and ``q`` and counted in ``q_nonfinite_dropped``."""
        n = max(self.nw, 1)
        q = np.asarray(self.q, dtype=np.float32).reshape(-1, len(self.names))
        t = np.asarray(self.t)
        ok = np.isfinite(q).all(axis=1)
        return {"joints": self.names, "actuators": self.acts, "t": t[ok], "q": q[ok],
                "q_nonfinite_dropped": int((~ok).sum()), "window_samples": self.nw,
                "sq_motor": self.sm / n, "sq_brake": self.sb / n, "drive_sq": self.sd / n, "rpm_hist": self.hist,
                "rpm_bin": self.rpm_bin, "rpm_peak": self.peak.copy(), "r": self.r.copy(), "eta": self.eta.copy()}


# ======================================================================================================== E1: geometry
CLASH_PAIRS = {"9L": ("left_fore", "left_hind"), "9R": ("right_fore", "right_hind"), "10": ("left_hind", "right_hind")}
CLASH_OF = {"9L": "9", "9R": "9", "10": "10"}


class ClashGeometry:
    """Clearance of clash pairs #9 (front and hind leg of each side: "9L", "9R") and #10 (the two hind legs, "10") at a
    pose of the 16 leg joints, with ``sim/check_packaging60.py``'s geometry: the adopted model with a probe copy of
    each leg's pushrod placed where the parallelogram puts it (``with_probe_rods``, ``linkage``, ``place_rod``; the
    model's fixed rod left out, as ``leg_grid`` does), every solid part of each leg's four bodies
    (``solid_geoms``) except the schematic limb covers (``LEAST_KINDS`` "solid parts (no covers)", the record's
    figure; the full capsule soles are in, as "foot shape"), signed distances by ``mj_geomDistance`` capped at 50 mm
    (an exact 0 re-measured 0.06 deg away by ``measure``). Each pair of parts is first bounded by containing capsules
    (``bound_capsule``, ``seg_dist``); a pair bounded at 50 mm or more is not measured (its distance is at least
    that); a capsule pair's bound is its exact distance. Also the least distance with the soles left out ("solid
    parts, soles aside")."""

    def __init__(self, model_path: Path | str = MODEL):
        import mujoco
        CP = _cp()
        self.mj, self.CP = mujoco, CP
        m = self.m = CP.with_probe_rods(Path(model_path))
        self.d = mujoco.MjData(m)
        self.lk = {c: CP.linkage(m, c) for c in CORNERS}
        solid = set(CP.solid_geoms(m))
        self.parts = {}
        for c in CORNERS:
            bodies = CP.leg_body_ids(m, c)
            gs = [g for g in sorted(solid) if m.geom_bodyid[g] in bodies and m.geom(g).name != f"{c}_pushrod"
                  and "limb_cover" not in m.geom(g).name]
            gs.append(m.geom(f"probe_rod_{c}").id)
            self.parts[c] = gs
        allg = sorted({g for gs in self.parts.values() for g in gs})
        self.gidx = {g: i for i, g in enumerate(allg)}
        self.allg = np.array(allg, dtype=np.int64)
        hr = [CP.bound_capsule(m, g) for g in allg]
        self.half = np.array([h for h, _ in hr])
        self.rad = np.array([r for _, r in hr])
        G = mujoco.mjtGeom
        self.capsule = np.array([m.geom_type[g] in (G.mjGEOM_CAPSULE, G.mjGEOM_SPHERE) for g in allg])
        self.pairs = {}
        for key, (a, b) in CLASH_PAIRS.items():
            ga = np.array([self.gidx[g] for g in self.parts[a] for _ in self.parts[b]])
            gb = np.array([self.gidx[g] for _ in self.parts[a] for g in self.parts[b]])
            sole = np.array([m.geom(int(allg[i])).name.endswith("_sole") or m.geom(int(allg[j])).name.endswith("_sole")
                             for i, j in zip(ga, gb)])
            self.pairs[key] = (ga, gb, sole)
        self.jq = np.array([m.jnt_qposadr[m.joint(n).id] for n in LOG_JOINTS], dtype=np.int64)
        self.joints = [m.joint(n) for n in LOG_JOINTS]
        self.cap = CP.CAP_M
        self._ft = np.zeros(6)

    def pose(self, q) -> None:
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        d.qpos[self.jq] = np.asarray(q, dtype=np.float64)
        self.mj.mj_kinematics(m, d)
        for c in CORNERS:
            self.CP.place_rod(m, d, c, self.lk[c])

    def clearance(self, q) -> dict:
        """{key: (least m, least m with the soles aside, (part a, part b) of the least)} at joint angles ``q`` (rad,
        LOG_JOINTS order)."""
        self.pose(q)
        m, d = self.m, self.d
        X = d.geom_xpos[self.allg]
        R = d.geom_xmat[self.allg].reshape(-1, 3, 3)
        hw = np.einsum("nij,nj->ni", R, self.half)
        p0, p1 = X - hw, X + hw
        out = {}
        for key, (ga, gb, sole) in self.pairs.items():
            bound = self.CP.seg_dist(p0[ga], p1[ga], p0[gb], p1[gb]) - self.rad[ga] - self.rad[gb]
            vals = np.full(len(ga), self.cap)
            near = bound < self.cap
            for k in np.flatnonzero(near):
                i, j = int(ga[k]), int(gb[k])
                if self.capsule[i] and self.capsule[j]:
                    x = float(bound[k])
                else:
                    g, h = int(self.allg[i]), int(self.allg[j])
                    x = float(self.mj.mj_geomDistance(m, d, g, h, self.cap, self._ft))
                    if x == 0.0:
                        x = self.CP.measure(m, d, g, h, self.joints, [round(math.degrees(v), 2) for v in q])
                vals[k] = min(x, self.cap)
            k = int(np.argmin(vals))
            ks = np.flatnonzero(~sole)
            k2 = int(ks[np.argmin(vals[ks])]) if len(ks) else None
            out[key] = (float(vals[k]), float(vals[k2]) if k2 is not None else self.cap,
                        (m.geom(int(self.allg[ga[k]])).name, m.geom(int(self.allg[gb[k]])).name))
        return out

    def batch(self, Q, keys=tuple(CLASH_PAIRS)) -> dict:
        """Least clearance (m) of each key, and with the soles aside, at every row of ``Q`` (N x 16, rad)."""
        Q = np.asarray(Q, dtype=np.float64).reshape(-1, len(LOG_JOINTS))
        out = {k: np.zeros(len(Q)) for k in keys}
        out.update({f"{k}_soles_aside": np.zeros(len(Q)) for k in keys})
        for n, q in enumerate(Q):
            c = self.clearance(q)
            for k in keys:
                out[k][n], out[f"{k}_soles_aside"][n] = c[k][0], c[k][1]
        return out


def deg_pose(named: dict) -> np.ndarray:
    """A 16-joint pose (rad, LOG_JOINTS order) from {joint name: deg}; joints not named are at 0."""
    q = np.zeros(len(LOG_JOINTS))
    for n, v in named.items():
        q[LOG_JOINTS.index(n)] = math.radians(v)
    return q


# =========================================================================================================== E1: stops
def _mirror(c: str) -> str:
    return c.replace("left_", "right_") if c.startswith("left_") else c.replace("right_", "left_")


def candidate_stops() -> list:
    """Every candidate stop in the packaging record (``check_packaging60.LIMITS``, which the record's items 9 and 10
    list), applied to both sides (the record's left side stands for both), with the record's least distance over the
    narrowed ranges (``packaging60.json`` ``leg_limits_against_each_other``, "solid parts (no covers)"; desk)."""
    CP = _cp()
    stored = json.loads(PACKAGING_JSON.read_text()).get("leg_limits_against_each_other", {})
    out = []
    for label, (pair, cases) in CP.LIMITS.items():
        clash = "10" if set(pair) == {"left_hind", "right_hind"} else "9"
        for i, case in enumerate(cases):
            limits = {}
            for legs, joints in zip(case[::2], case[1::2]):
                for c in legs:
                    for cc in sorted({c, _mirror(c)}):       # sorted: the plan's bytes do not depend on hashing
                        for jn, (lo, hi) in joints.items():
                            limits[f"{cc}_{jn}"] = (lo, hi)
            rec = None
            rows = stored.get(label)
            if rows and i < len(rows):
                rec = rows[i]["least"]["solid parts (no covers)"]["min_mm"]
            desc = ", ".join(f"{j} {'>=' if lo is not None else '<='} {lo if lo is not None else hi:g} deg"
                             for j, (lo, hi) in sorted(limits.items()))
            out.append({"id": f"{clash}:{label}:{i}", "clash": clash, "family": label, "limits_deg": limits,
                        "label": desc, "record_least_mm_full_range": rec})
    return out


def apply_stop(Q, limits_deg: dict):
    """``Q`` (N x 16, rad) with each limited joint clipped to its stop; and which rows changed."""
    Q = np.asarray(Q, dtype=np.float64)
    Qc = Q.copy()
    for jn, (lo, hi) in limits_deg.items():
        k = LOG_JOINTS.index(jn)
        if lo is not None:
            Qc[:, k] = np.maximum(Qc[:, k], math.radians(lo))
        if hi is not None:
            Qc[:, k] = np.minimum(Qc[:, k], math.radians(hi))
    changed = np.any(Qc != Q, axis=1)
    return Qc, changed


def clash_least(clear: dict, clash: str) -> np.ndarray:
    """The least clearance over the keys of one clash (#9: both sides) per sample."""
    keys = [k for k, c in CLASH_OF.items() if c == clash]
    return np.min(np.vstack([np.asarray(clear[k]) for k in keys]), axis=0)


def evaluate_stop(geom: ClashGeometry, Q, base_clear: dict, stop: dict) -> dict:
    """E1 for one stop on one controller's samples: the share of samples it clips (changes), the least clearance of the
    clash it addresses over every sample with the stop applied (unchanged samples keep their logged clearance), and
    ACCEPTABLE (least >= 5 mm and share <= 1 %)."""
    Qc, changed = apply_stop(Q, stop["limits_deg"])
    least = clash_least(base_clear, stop["clash"]).copy()
    if changed.any():
        keys = [k for k, c in CLASH_OF.items() if c == stop["clash"]]
        new = geom.batch(Qc[changed], keys=tuple(keys))
        least[changed] = np.min(np.vstack([new[k] for k in keys]), axis=0)
    share = float(changed.mean()) if len(changed) else 0.0
    lm = 1000.0 * float(least.min()) if len(least) else None
    return {"stop": stop["id"], "clash": stop["clash"], "samples": int(len(changed)), "clipped": int(changed.sum()),
            "clipped_share": share, "least_mm_after": lm,
            "acceptable": bool(lm is not None and lm >= CLEAR_MM and share <= CLIP_SHARE)}


def coupled_limit(base_clear: dict, clash: str) -> dict:
    """The record's coupled software limit drawn on the overlapping poses, taken at a 5 mm margin: it changes exactly
    the samples whose clearance is under 5 mm and leaves them at 5 mm (an idealised limit, exact on the model's
    shapes), so it is ACCEPTABLE when those samples are at most 1 %."""
    least = clash_least(base_clear, clash)
    share = float((least < CLEAR_MM / 1000.0).mean()) if len(least) else 0.0
    return {"stop": f"{clash}:coupled software limit at 5 mm", "clash": clash, "samples": int(len(least)),
            "clipped": int((least < CLEAR_MM / 1000.0).sum()), "clipped_share": share,
            "least_mm_after": max(CLEAR_MM, 1000.0 * float(least.min())) if len(least) else None,
            "acceptable": share <= CLIP_SHARE}


def e1_verdict(per_controller: dict) -> dict:
    """``per_controller``: {name: {"least_mm": {"9": m, "10": m}, "stops": {stop id: evaluate_stop result}}} for the five
    controllers. #9 / #10 REACHED if at least two controllers have a logged pose with clearance < 0; a stop ACCEPTABLE
    if it is acceptable on every controller."""
    out = {"reached": {}, "acceptable_stops": {}}
    for clash in ("9", "10"):
        n = [c for c, v in per_controller.items() if v["least_mm"][clash] < 0.0]
        out["reached"][clash] = {"controllers_overlapping": n, "REACHED": len(n) >= 2}
    stops = sorted({s for v in per_controller.values() for s in v["stops"]})
    for s in stops:
        rows = {c: v["stops"].get(s) for c, v in per_controller.items()}
        out["acceptable_stops"][s] = {"ACCEPTABLE": all(r is not None and r["acceptable"] for r in rows.values()),
                                      "worst_clipped_share": max(r["clipped_share"] for r in rows.values() if r),
                                      "least_mm_after": min(r["least_mm_after"] for r in rows.values() if r)}
    return out


# ================================================================================================ E2(a): desk mapping
def drive_sq_from_split(sq_motor, sq_brake, r: float, eta: float):
    """Drive-side mean squared torque from the joint-side motoring and braking means (StepLogger ``sq_motor``,
    ``sq_brake``): tau_d = tau / (r eta) motoring, tau eta / r braking."""
    return np.asarray(sq_motor) / (r * eta) ** 2 + np.asarray(sq_brake) * (eta / r) ** 2


def drive_sq_bounds(stored_sq, eta_old, r: float, eta: float):
    """When the motoring / braking split is not known: the new drive-side mean square lies between the stored one x
    the smaller and x the larger of the two factors, (eta_old / (r eta))^2 (motoring) and (eta / (r eta_old))^2
    (braking). Returns (low, high)."""
    s = np.asarray(stored_sq, dtype=float)
    eo = np.asarray(eta_old, dtype=float)
    fm, fb = (eo / (r * eta)) ** 2, (eta / (r * eo)) ** 2
    return s * np.minimum(fm, fb), s * np.maximum(fm, fb)


def reduced_profile(stored: dict, joints: list, placement: str | None, r: float, eta: float, split: dict | None = None,
                    tol: float = 1e-6) -> dict:
    """The demo loop's walking (0.25 m/s) and standing segments' per-joint drive-side mean squares with a reduction r
    (efficiency eta) at ``placement``'s joints (the others unchanged). ``stored``: {"walk_0.25": sq, "stand": sq} as
    the demo course stores ``tau_drive_sq_mean``. ``split``: {segment: (sq_motor, sq_brake)} from StepLogger on the
    same episodes; used ("exact") when it reproduces the stored figure within ``tol`` (relative) on every joint, else
    the bounds' high value is used ("bound") with the low one beside it."""
    red = set(reduced_joints(placement)) if placement else set()
    eo = np.array([ETA_OLD[axis_of(j)] for j in joints])
    out = {"method": "exact", "reduced": sorted(red), "r": r, "eta": eta}
    for seg in ("walk_0.25", "stand"):
        s = np.asarray(stored[seg], dtype=float)
        lo, hi = drive_sq_bounds(s, eo, r, eta)
        mask = np.array([j in red for j in joints])
        new_hi, new_lo = np.where(mask, hi, s), np.where(mask, lo, s)
        exact = None
        if split is not None and seg in split:
            sm, sb = (np.asarray(x, dtype=float) for x in split[seg])
            repro = sm / eo ** 2 + sb * eo ** 2
            rel = np.abs(repro - s) / np.maximum(np.abs(s), 1e-12)
            out[f"{seg}_reproduction_rel_max"] = float(rel.max())
            if rel.max() <= tol:
                exact = np.where(mask, drive_sq_from_split(sm, sb, r, eta), s)
        if exact is None:
            out["method"] = "bound"
        out[seg] = {"sq": exact if exact is not None else new_hi, "sq_low": new_lo, "sq_high": new_hi}
    if out["method"] == "bound":     # one method for both segments
        for seg in ("walk_0.25", "stand"):
            out[seg]["sq"] = out[seg]["sq_high"]
    return out


def heat_rules(demo: dict, walk_sq, stand_sq, minutes: float | None = None, room_c: float = ROOM_C) -> dict:
    """C1 (the hottest winding over the demo, 35 C, method A, 8 min; 4 and 15 min beside) and C2 (the worst drive's
    loop RMS over its continuous rating; loop = 90 s walking + 30 s standing) on per-joint drive-side mean squares, by
    the form-factor study's own ``heat`` (rl/telemetry/thermal.py through rl/analysis/thermal_ambient.py) and its
    loop rule, today's drives (D0)."""
    from telemetry.thermal import tc_key
    FF = _ff()
    joints = demo["segments"]["stand"]["joints"]
    d0 = FF.DRIVE_SETS["D0"]
    eq = np.array([FF.drives_of(d0)[FF.axis_of(j)]["eq"] for j in joints])
    w, s = np.asarray(walk_sq, dtype=float), np.asarray(stand_sq, dtype=float)
    loop = (FF.WALK_S * w + FF.STAND_S * s) / (FF.WALK_S + FF.STAND_S)
    ratio = np.sqrt(loop) / eq
    h = FF.heat(demo, w, s, d0, float(minutes if minutes is not None else demo["minutes"]), room_c=room_c)
    k = tc_key(SCORED_TC_S)
    c1 = h["by_tc"][k]["peak_c"]
    c2 = float(ratio.max())
    return {"C1_c": float(c1), "C1_joint": h["by_tc"][k]["joint"], "C1_by_tc_c": {kk: v["peak_c"] for kk, v in
                                                                                  h["by_tc"].items()},
            "C1_pass": bool(c1 <= C1_LIMIT_C), "C2": c2, "C2_joint": joints[int(np.argmax(ratio))],
            "C2_pass": bool(c2 <= C2_LIMIT), "loop_ratio": dict(zip(joints, ratio.tolist())),
            "peak_c_8min": h["per_joint_peak_c_8min"]}


def e2a_specs() -> list:
    """(name, transmission, eta, judged) of E2(a)'s row families: the three transmissions at their judged
    efficiencies, then the supporting ones (``SUPPORTING_ETA``: the rod at the build's 0.90, named "rod_eta0.9")."""
    out = [(tr, tr, float(spec["eta"]), True) for tr, spec in TRANSMISSIONS.items()]
    out += [(f"{tr}_eta{eta:g}", tr, float(eta), False) for tr, eta in SUPPORTING_ETA.items()]
    return out


def e2a_grid(demo: dict, segments: dict, split: dict | None = None, room_c: float = ROOM_C) -> list:
    """E2(a) on one demo course: every transmission (its eta), reduction and placement; C1 and C2 per row
    (``reduced_profile`` then ``heat_rules``). ``segments``: the demo's stored drive-side mean squares
    (``stored_segments``). Rows with ``judged`` False (the rod at 0.90) are supporting only."""
    joints = demo["segments"]["stand"]["joints"]
    rows = []
    for name, tr, eta, judged in e2a_specs():
        for r in REDUCTIONS:
            for pl in PLACEMENTS:
                p = reduced_profile(segments, joints, pl, r, eta, split)
                h = heat_rules(demo, p["walk_0.25"]["sq"], p["stand"]["sq"], demo["minutes"], room_c)
                row = {"transmission": name, "kind": tr, "eta": eta, "judged": judged, "r": r, "placement": pl,
                       "method": p["method"],
                       **{k: h[k] for k in ("C1_c", "C1_joint", "C1_by_tc_c", "C1_pass", "C2", "C2_joint",
                                            "C2_pass")}}
                if p["method"] == "bound":
                    lo = heat_rules(demo, p["walk_0.25"]["sq_low"], p["stand"]["sq_low"], demo["minutes"], room_c)
                    row.update(C1_c_low=lo["C1_c"], C2_low=lo["C2"])
                rows.append(row)
    return rows


def stored_demo(run: str) -> dict:
    return json.loads((DEMO_DIR / run / "demo_course.json").read_text())


def stored_segments(demo: dict) -> dict:
    return {seg: np.asarray(demo["segments"][seg]["tau_drive_sq_mean"], dtype=float) for seg in ("walk_0.25", "stand")}


# ================================================================================================= E2(a): speed check
def weighted_percentile_from_hist(hists: list, weights: list, rpm_bin: float, q: float = 99.0) -> np.ndarray:
    """Per drive, the q-th percentile of |speed| (rpm) of a mixture of histograms (each normalised, then weighted:
    the demo loop's 90 s walking and 30 s standing as 0.75 and 0.25), at the upper edge of the bin it falls in."""
    mix = None
    for h, w in zip(hists, weights):
        h = np.asarray(h, dtype=float)
        tot = h.sum(axis=1, keepdims=True)
        part = w * h / np.maximum(tot, 1.0)
        mix = part if mix is None else mix + part
    cdf = np.cumsum(mix, axis=1) / np.maximum(mix.sum(axis=1, keepdims=True), 1e-300)
    k = np.argmax(cdf >= q / 100.0 - 1e-12, axis=1)
    return (k + 1) * rpm_bin


def speed_check(hists: list, weights: list, peaks: list, rpm_bin: float, r: np.ndarray,
                cap_rpm: float = CAP_RPM) -> dict:
    """E2's speed rule on joint-speed histograms (r = 1 logs) scaled by each drive's reduction r: the drive's
    99th-percentile speed <= cap and its peak <= 1.1 x cap. Percentiles and peaks scale exactly with r."""
    p99 = weighted_percentile_from_hist(hists, weights, rpm_bin) * np.asarray(r)
    pk = np.max(np.vstack(peaks), axis=0) * np.asarray(r)
    ok99, okpk = p99 <= cap_rpm, pk <= PEAK_FACTOR * cap_rpm
    return {"p99_rpm": p99.tolist(), "peak_rpm": pk.tolist(), "p99_pass": ok99.tolist(), "peak_pass": okpk.tolist(),
            "pass": bool(ok99.all() and okpk.all())}


def e2a_speed_check(hists: list, weights: list, peaks: list, rpm_bin: float, actuators: list, placement: str,
                    r: float, cap_rpm: float = CAP_RPM) -> dict:
    """E2(a)'s speed check for one (r, placement): ``speed_check`` on the reduced drives only (``reduced_joints``:
    the four hip pitch drives, the four knees, or both), each at r; the unreduced drives (roll, and the other axis)
    are left out, as the rule tests "the drive" of a reduction. ``hists`` / ``peaks``: StepLogger's 12-drive
    ``rpm_hist`` / ``rpm_peak`` (r = 1 logs) in ``actuators`` order."""
    red = reduced_joints(placement)
    idx = [list(actuators).index(j) for j in red]
    out = speed_check([np.asarray(h)[idx] for h in hists], weights, [np.asarray(p)[idx] for p in peaks], rpm_bin,
                      np.full(len(idx), float(r)), cap_rpm)
    return {"placement": placement, "r": r, "drives": list(red), **out}


# =============================================================================================== E2(b): transmissions
def transmission_stiffness(kind: str, joint: str, r: float = SIM_R, strands: int | None = None) -> dict:
    """Joint-side torsional stiffness (N m/rad) of a reduction r at ``joint`` (hip pitch or knee), from SOURCES:
    - rod: one M8 class 8.8 rod of the joint's span, k_axial = E A / L, at the joint crank r x 28 mm and the minimum
      transmission angle 43 deg: k = k_axial (r 0.028)^2 sin^2(43 deg). The rod ends', pins' and bearings' compliance
      is not included (no source), so this is an upper bound;
    - belt: 3M GT2, 15 mm wide, EA = 30,000 lbf/in x 15 mm per unit strain; both spans of the joint's span length in
      tension (pretensioned): k = 2 EA / L x R_j^2, R_j the joint pulley's pitch radius (56 r teeth x 3 mm / 2 pi).
      Tooth deflection is not included, so this too is an upper bound. ``strands`` 1: one span only (the slack span
      unloaded, if the pretension is under half the effective tension: ``belt_pretension``), half the stiffness;
    - cable: 3.0 mm 7x19 316 wire rope, EA = 97.1 GPa x 4.6 mm^2; a pretensioned loop of two strands over the span:
      k = 2 EA / L x (r 0.028)^2. Constructional stretch and pretension loss are not modelled."""
    L = span_of(joint)
    if strands is not None and kind == "rod" and strands != 1:
        raise ValueError("a rod is one member")
    if kind == "rod":
        EA = SOURCES["steel_E_pa"]["value"] * SOURCES["m8_stress_area_m2"]["value"]
        Rj = r * SOURCES["crank_drive_m"]["value"]
        s2 = math.sin(math.radians(SOURCES["rod_min_transmission_deg"]["value"])) ** 2
        k = EA / L * Rj ** 2 * s2
        formula = "E A / L x R_j^2 x sin^2(43 deg)"
        strands = 1
    elif kind == "belt":
        per_mm = SOURCES["gt2_3m_modulus_lbf_per_in"]["value"] * 4.4482216152605 / 25.4   # N per mm width
        EA = per_mm * BELT["width_m"] * 1000.0
        teeth = int(round(BELT["teeth_drive"] * r))
        Rj = teeth * BELT["pitch_m"] / (2.0 * math.pi)
        strands = DEFAULT_STRANDS["belt"] if strands is None else int(strands)
        k = strands * EA / L * Rj ** 2
        formula = f"{strands} EA / L x R_j^2"
    elif kind == "cable":
        w = SOURCES["wire_7x19_3mm"]["value"]
        EA = SOURCES["wire_7x19_E_pa"]["value"] * w["metallic_area_m2"]
        Rj = r * SOURCES["crank_drive_m"]["value"]
        strands = DEFAULT_STRANDS["cable"] if strands is None else int(strands)
        k = strands * EA / L * Rj ** 2
        formula = f"{strands} EA / L x R_j^2"
    else:
        raise ValueError(kind)
    if strands not in (1, 2):
        raise ValueError(f"strands {strands}")
    return {"kind": kind, "joint": joint, "r": r, "span_m": L, "EA_n": EA, "strands": strands, "R_joint_m": Rj,
            "k_nm_per_rad": k, "formula": formula}


def write_timestep_copy(out_dir: Path, physics_dt: float, src: Path = MODEL, name: str | None = None) -> tuple:
    """Write a copy of the adopted model that differs from it only in its ``<option timestep>`` (the judged baselines
    B025 and, under item 4's fallback, B0125): a text edit of that one attribute, every other byte kept (force limits,
    armatures, gains, contacts). Checked: exactly one ``<option`` element with one timestep attribute, and the copy
    compiles with ``opt.timestep == physics_dt``. Returns (path, record)."""
    import re
    import mujoco
    text = Path(src).read_text()
    opts = re.findall(r"<option\b[^>]*>", text)
    if len(opts) != 1:
        raise ValueError(f"{src}: {len(opts)} <option> elements")
    pat = re.compile(r'(<option\b[^>]*?\btimestep=")([^"]*)(")')
    if len(pat.findall(text)) != 1:
        raise ValueError(f"{src}: the <option> element has no single timestep attribute")
    old = pat.search(text).group(2)
    out_text = pat.sub(lambda mt: f"{mt.group(1)}{physics_dt:.9g}{mt.group(3)}", text, count=1)
    m = mujoco.MjModel.from_xml_string(out_text)
    if abs(float(m.opt.timestep) - physics_dt) > 1e-15:
        raise ValueError(f"the copy's timestep is {m.opt.timestep}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = name or f"baseline_dt{physics_dt:g}"
    path = out_dir / f"trike60_fore128_cand_{tag}.xml"
    path.write_text(out_text)
    rec = {"transmission": None, "r": 1.0, "placement": None, "eta": None, "physics_dt": physics_dt,
           "builder": "write_timestep_copy", "timestep_before": old, "timestep_after": f"{physics_dt:.9g}",
           "path": str(path), "source": str(Path(src)), "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "xml_sha256": hashlib.sha256(out_text.encode()).hexdigest(), "nq": int(m.nq), "nv": int(m.nv),
           "mass_kg": float(m.body_mass.sum())}
    return path, rec


def belt_pretension(joint: str, r: float = SIM_R, eta: float | None = None) -> dict:
    """DESK: the pretension per span the belt needs for both spans to stay taut (half the effective tension) at the
    belt variant's joint-side force limit (today's limit / today's eta x eta x r), at the joint pulley's pitch
    radius. Below it the slack span unloads and the stiffness halves (``transmission_stiffness(strands=1)``)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    eta = float(eta if eta is not None else TRANSMISSIONS["belt"]["eta"])
    lim0 = float(m.actuator(joint).forcerange[1])
    lim = lim0 / ETA_OLD[axis_of(joint)] * eta * r
    Rj = transmission_stiffness("belt", joint, r)["R_joint_m"]
    T = lim / Rj
    return {"joint": joint, "drive": drive_of(joint), "limit_joint_side_nm": lim, "R_joint_m": Rj,
            "effective_tension_n": T, "pretension_per_span_needed_n": T / 2.0}


def write_variant(out_dir: Path, *, transmission: str | None, r: float = SIM_R, placement: str = "both",
                  eta: float | None = None, physics_dt: float = SERIES_DT, src: Path = MODEL,
                  damping_ratio: float = DAMPING_RATIO, name: str | None = None, strands: int | None = None) -> tuple:
    """Write a copy of the adopted model with ``placement``'s drives reduced by r (r = 1 with no transmission is
    refused: it would still rescale the force limits by eta / today's eta, so it is not the adopted plant; a baseline
    at another step is ``write_timestep_copy``):
    - transmission None ("rigid"): each reduced joint's armature (the drive's rotor inertia, sim/data/drives.json,
      reflected to the drive output) x r^2;
    - "rod" / "belt" / "cable" ("series"): each reduced drive becomes a rotor, a hinge in joint-side units
      (drive angle / r) named as the joint so the env's actuator and observation follow it (the drive's encoder),
      on a world body of its own (no geometry that collides), carrying the armature x r^2; the leg joint is renamed
      ``<name>__out`` (armature 0, its range, damping and the rest kept) and coupled to the rotor by a fixed tendon
      L = q_rotor - q_out with the transmission's stiffness (``transmission_stiffness``) and damping 2 zeta sqrt(k J),
      J the rotor's armature, zeta ``damping_ratio``.
    In both, the reduced actuator's force limit (joint side) becomes r x eta x 0.70 x the drive's peak (today's
    0.70 x peak x 0.90 at the flexion linkages, 0.70 x peak at hip pitch); kp and kv are kept, so the joint-side servo
    stiffness is the adopted one; ctrlrange is kept. The option timestep is set to ``physics_dt``. Keyframes must be
    empty (the adopted model's are). Returns (path, record)."""
    import mujoco
    if transmission is not None and transmission not in TRANSMISSIONS:
        raise ValueError(transmission)
    if transmission is None and abs(r - 1.0) < 1e-12:
        raise ValueError("no transmission and r = 1 is the adopted model: use write_timestep_copy for a baseline")
    if strands is not None and transmission is None:
        raise ValueError("strands needs a transmission")
    eta = float(eta if eta is not None else (TRANSMISSIONS[transmission]["eta"] if transmission
                                             else SOURCES["eta_default"]["value"]))
    text = Path(src).read_text()
    root = ET.fromstring(text)
    span_tag = (f"_{strands}span" if transmission and strands is not None and strands != DEFAULT_STRANDS[transmission]
                else "")
    tag = name or f"{transmission or 'rigid'}{span_tag}_r{r:g}_{placement}_dt{physics_dt:g}"
    root.set("model", f"{root.get('model')}_cand_{tag}")
    opt = root.find("option")
    opt.set("timestep", f"{physics_dt:.9g}")
    for key in root.iter("key"):
        if any(key.get(a) for a in ("qpos", "qvel", "act", "mpos", "mquat")):
            raise NotImplementedError("a keyframe with a state: extend it for the rotors first")
    joints = {j.get("name"): j for j in root.iter("joint")}
    acts = {a.get("name"): a for a in root.find("actuator")}
    wb = root.find("worldbody")
    tendon = root.find("tendon")
    rec = {"transmission": transmission or "rigid", "r": r, "placement": placement, "eta": eta,
           "physics_dt": physics_dt, "damping_ratio": damping_ratio if transmission else None, "builder": "write_variant",
           "strands": (DEFAULT_STRANDS[transmission] if strands is None else int(strands)) if transmission else None,
           "joints": {}}
    for jn in reduced_joints(placement):
        jel, a = joints[jn], acts[jn]
        arm = float(jel.get("armature"))
        lim0 = float(a.get("forcerange").split()[1])
        lim = lim0 / ETA_OLD[axis_of(jn)] * eta * r
        a.set("forcerange", f"{-lim:.6f} {lim:.6f}")
        row = {"drive": drive_of(jn), "armature_drive_kgm2": arm, "limit_joint_side_nm_before": lim0,
               "limit_joint_side_nm": lim}
        if transmission is None:
            jel.set("armature", f"{arm * r * r:.9g}")
            row["armature_joint_kgm2"] = arm * r * r
        else:
            if tendon is None:
                tendon = ET.SubElement(root, "tendon")
            st = transmission_stiffness(transmission, jn, r, strands)
            J = arm * r * r
            c = 2.0 * damping_ratio * math.sqrt(st["k_nm_per_rad"] * J)
            jel.set("name", jn + OUT_SUFFIX)
            jel.set("armature", "0")
            b = ET.SubElement(wb, "body", {"name": f"cand_rotor_{jn}", "pos": "0 0 -1"})
            ET.SubElement(b, "joint", {"name": jn, "type": "hinge", "axis": "0 0 1", "limited": "false",
                                       "damping": "0", "armature": f"{J:.9g}", "stiffness": "0"})
            ET.SubElement(b, "geom", {"name": f"cand_rotor_geom_{jn}", "type": "sphere", "size": "0.001",
                                      "mass": "0.0001", "contype": "0", "conaffinity": "0", "group": "5"})
            f = ET.SubElement(tendon, "fixed", {"name": f"cand_spring_{jn}", "stiffness": f"{st['k_nm_per_rad']:.9g}",
                                                "damping": f"{c:.9g}", "springlength": "0"})
            ET.SubElement(f, "joint", {"joint": jn, "coef": "1"})
            ET.SubElement(f, "joint", {"joint": jn + OUT_SUFFIX, "coef": "-1"})
            row.update(stiffness=st, rotor_armature_kgm2=J, damping_nms_per_rad=c,
                       spring_hz_on_rotor=math.sqrt(st["k_nm_per_rad"] / J) / (2 * math.pi))
        rec["joints"][jn] = row
    out_text = ET.tostring(root, encoding="unicode") + "\n"
    m = mujoco.MjModel.from_xml_string(out_text)      # it must compile
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trike60_fore128_cand_{tag}.xml"
    path.write_text(out_text)
    rec.update(path=str(path), source=str(Path(src)), source_sha256=hashlib.sha256(text.encode()).hexdigest(),
               xml_sha256=hashlib.sha256(out_text.encode()).hexdigest(), nq=int(m.nq), nv=int(m.nv),
               mass_kg=float(m.body_mass.sum()))
    return path, rec


def sync_pairs_for(model) -> list:
    """[(rotor qpos address, leg joint qpos address)] of a series variant (empty for any other model)."""
    out = []
    for j in range(model.njnt):
        n = model.joint(j).name
        if n.endswith(OUT_SUFFIX):
            rot = model.joint(n[: -len(OUT_SUFFIX)]).id
            out.append((int(model.jnt_qposadr[rot]), int(model.jnt_qposadr[j])))
    return out


def output_joint_names(model) -> list:
    """LOG_JOINTS as the model names them (a series variant's leg joints carry OUT_SUFFIX)."""
    names = {model.joint(j).name for j in range(model.njnt)}
    return [n + OUT_SUFFIX if (n + OUT_SUFFIX) in names else n for n in LOG_JOINTS]


class SpeedCap:
    """The 83 rpm drive speed cap, as a firmware speed limit: from 0.95 to 1.0 of the cap the drive's torque in the
    direction it is turning falls linearly to zero, and at or above the cap it can only brake (the torque against the
    motion is unchanged). Applied after every physics step to the force limits of the next step; the drive speed is
    r x the actuator joint's speed (qvel, current). ``on_reset`` takes the episode's force limits."""

    def __init__(self, env, r_by_actuator: dict, cap_rpm: float = CAP_RPM, soft: float = 0.95):
        m = env.model
        self.names = list(r_by_actuator)
        self.idx = np.array([m.actuator(n).id for n in self.names], dtype=np.int64)
        self.dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in self.idx], dtype=np.int64)
        self.r = np.array([float(r_by_actuator[n]) for n in self.names])
        self.cap = cap_rpm * 2.0 * math.pi / 60.0
        self.soft = float(soft)
        self.base = m.actuator_forcerange[self.idx].copy()
        self.steps = 0
        self.at_cap = np.zeros(len(self.idx), dtype=np.int64)

    def on_reset(self, env):
        self.base = env.model.actuator_forcerange[self.idx].copy()
        self.steps = 0
        self.at_cap[:] = 0

    def envelope(self, v: np.ndarray) -> np.ndarray:
        wd = np.abs(v) * self.r
        return np.clip((self.cap - wd) / (self.cap * (1.0 - self.soft)), 0.0, 1.0)

    def __call__(self, env):
        m, d = env.model, env.data
        v = d.qvel[self.dof]
        f = self.envelope(v)
        lo, hi = self.base[:, 0].copy(), self.base[:, 1].copy()
        pos, neg = v > 0, v < 0
        hi[pos] *= f[pos]
        lo[neg] *= f[neg]
        m.actuator_forcerange[self.idx, 0] = lo
        m.actuator_forcerange[self.idx, 1] = hi
        self.steps += 1
        self.at_cap += (np.abs(v) * self.r >= self.cap)


def _v(builder, transmission, r, dt, cap, role, strands=None):
    return {"builder": builder, "transmission": transmission, "r": r, "physics_dt": dt, "cap": cap, "role": role,
            "strands": strands}


E2B_VARIANTS = {   # name: settings. role "baseline": judged baselines; "judged": the three transmissions
    "B1": _v("adopted", None, 1.0, 0.001, False, "baseline"),                 # the adopted model as trained (1 ms)
    "B025": _v("write_timestep_copy", None, 1.0, SERIES_DT, False, "baseline"),   # the same, only the step changed
    "R": _v("write_variant", None, SIM_R, SERIES_DT, True, "supporting"),     # reduced, rigid, capped, eta 0.95
    "rod": _v("write_variant", "rod", SIM_R, SERIES_DT, True, "judged"),
    "belt": _v("write_variant", "belt", SIM_R, SERIES_DT, True, "judged"),
    "cable": _v("write_variant", "cable", SIM_R, SERIES_DT, True, "judged"),
    "belt_1span": _v("write_variant", "belt", SIM_R, SERIES_DT, True, "supporting", strands=1),   # half stiffness
    "B0125": _v("write_timestep_copy", None, 1.0, FALLBACK_DT, False, "fallback baseline"),   # item 4 only
}
E2B_RUNS_AT_START = ("B1", "B025", "R", "rod", "belt", "cable", "belt_1span")


def e2b_model(variant: str, out_dir: Path, physics_dt: float | None = None) -> tuple:
    """(path, record) of an E2(b) variant's model: B1 the adopted model itself (not copied); B025 / B0125
    ``write_timestep_copy`` (only the step changed); the others ``write_variant`` with the variant's transmission, r,
    strands and step. ``physics_dt`` overrides the step (item 4's 0.125 ms rerun of a variant)."""
    s = E2B_VARIANTS[variant]
    dt = float(physics_dt if physics_dt is not None else s["physics_dt"])
    if s["builder"] == "adopted":
        if abs(dt - 0.001) > 1e-12:
            raise ValueError("B1 is the adopted model at its own 1 ms step")
        return MODEL, {"variant": variant, "builder": "adopted", "path": str(MODEL), "physics_dt": 0.001,
                       "xml_sha256": sha256_file(MODEL)}
    if s["builder"] == "write_timestep_copy":
        path, rec = write_timestep_copy(out_dir, dt, name=f"{variant}_dt{dt:g}")
    else:
        path, rec = write_variant(out_dir, transmission=s["transmission"], r=s["r"], physics_dt=dt,
                                  strands=s["strands"], name=f"{variant}_r{s['r']:g}_dt{dt:g}")
    rec["variant"] = variant
    return path, rec


def variant_r_eta(variant: str) -> tuple:
    """(r by actuator, eta by actuator) for the drive-side heat of an E2(b) variant (roll unreduced)."""
    s = E2B_VARIANTS[variant]
    tr, r = s["transmission"], s["r"]
    if r == 1.0:
        return {}, {}
    eta = TRANSMISSIONS[tr]["eta"] if tr else SOURCES["eta_default"]["value"]
    js = reduced_joints("both")
    return {j: r for j in js}, {j: eta for j in js}


def relog_drive_sq(res: dict, eta_by_actuator: dict) -> np.ndarray:
    """A StepLogger result's drive-side mean squares recomputed with other efficiencies on the named actuators (same
    r, same samples): sq_motor / (r eta)^2 + sq_brake (eta / r)^2. E2(b)'s supporting rod reading at 0.90."""
    eta = np.array([float(eta_by_actuator.get(a, e)) for a, e in zip(res["actuators"], res["eta"])])
    return drive_sq_from_split(res["sq_motor"], res["sq_brake"], np.asarray(res["r"], dtype=float), eta)


def e2b_judged_runs(unstable_at_start: set) -> dict:
    """Section 3 item 4 (fixed before any run): which step and baselines each non-baseline variant is judged at,
    given the runs of ``E2B_RUNS_AT_START`` that had at least one episode ending "unstable_simulation". A variant with
    one (or every variant, if B025 had one) is rerun at 0.125 ms on the same seeds and judged against B1 and B0125
    (the adopted model at 0.125 ms); its 0.25 ms run is reported beside it (supporting). Otherwise 0.25 ms against B1
    and B025. B1 (1 ms, the plant as trained) is never rerun."""
    bad = set(unstable_at_start)
    unknown = bad - set(E2B_RUNS_AT_START)
    if unknown:
        raise ValueError(f"not a start run: {sorted(unknown)}")
    out = {}
    for v, s in E2B_VARIANTS.items():
        if "baseline" in s["role"]:
            continue
        fall = "B025" in bad or v in bad
        out[v] = {"role": s["role"], "physics_dt": FALLBACK_DT if fall else SERIES_DT,
                  "baselines": ("B1", "B0125") if fall else ("B1", "B025"), "rerun_at_fallback": fall}
    out["_B0125_needed"] = any(v["rerun_at_fallback"] for v in out.values())
    return out


def drive_ratios(drive_sq: np.ndarray, actuators: list) -> np.ndarray:
    """Drive-side RMS over the drive's own continuous rating (X4 7.5, X6 12 N m)."""
    eq = np.array([THERMAL_EQ_NM[drive_of(a)] for a in actuators])
    return np.sqrt(np.asarray(drive_sq, dtype=float)) / eq


def is_unstable(row: dict) -> bool:
    return row.get("termination", "") == UNSTABLE


def e2b_compare(variant: dict, baseline: dict) -> dict:
    """One policy, one condition: ``variant`` and ``baseline`` are {seed: {"fell": bool, "termination": str, "speed":
    m/s, "drive_sq": (12), "samples": int}} on the same seeds. No new falls (no seed falls in the variant that did not
    in the baseline); mean speed within 0.02 m/s and worst-drive RMS / continuous (pooled over the seeds that stayed
    up in both, weighted by samples) not worse by more than 0.03.
    An episode ending "unstable_simulation" (in either run) is a harness failure, not a fall: its seed is left out of
    the falls and of the pairing, listed in ``unstable``, and the comparison is not judged here (``pass`` None,
    ``harness_fault`` True); section 3 item 4 says which run is judged instead (``e2b_judged_runs``)."""
    seeds = sorted(set(variant) & set(baseline))
    unstable = [s for s in seeds if is_unstable(variant[s]) or is_unstable(baseline[s])]
    ok = [s for s in seeds if s not in unstable]
    new_falls = [s for s in ok if variant[s]["fell"] and not baseline[s]["fell"]]
    up = [s for s in ok if not variant[s]["fell"] and not baseline[s]["fell"]]

    def pooled(rows):
        n = sum(rows[s]["samples"] for s in up)
        return sum(np.asarray(rows[s]["drive_sq"]) * rows[s]["samples"] for s in up) / max(n, 1)

    out = {"seeds": seeds, "new_falls": new_falls, "paired_up": up, "unstable": unstable,
           "unstable_variant": [s for s in seeds if is_unstable(variant[s])],
           "unstable_baseline": [s for s in seeds if is_unstable(baseline[s])], "harness_fault": bool(unstable)}
    passed = False
    if up:
        acts = variant[up[0]]["actuators"]
        rv, rb = drive_ratios(pooled(variant), acts), drive_ratios(pooled(baseline), acts)
        sv, sb = float(np.mean([variant[s]["speed"] for s in up])), float(np.mean([baseline[s]["speed"] for s in up]))
        out.update(speed_variant=sv, speed_baseline=sb, worst_ratio_variant=float(rv.max()),
                   worst_ratio_baseline=float(rb.max()), worst_joint_variant=acts[int(np.argmax(rv))])
        passed = bool(not new_falls and abs(sv - sb) <= 0.02 and rv.max() - rb.max() <= 0.03)
    out["pass_on_stable_seeds"] = passed
    out["pass"] = None if unstable else passed
    return out


def e2b_verdict(per_policy: dict) -> dict:
    """{policy: {baseline name: {condition: e2b_compare result}}}: COMPATIBLE WITH TODAY'S POLICIES if at least 3 of
    the 4 policies pass every condition against every judged baseline. A policy fails if any of its comparisons
    fails; it is not judged (None) if none fails and one is not judged (a harness fault). The verdict is True with at
    least 3 passing policies, False once 3 can no longer pass (2 or more fail), and None ("NOT JUDGED") otherwise."""
    ok = {}
    for p, v in per_policy.items():
        res = [r["pass"] for b in v.values() for r in b.values()]
        ok[p] = False if any(x is False for x in res) else (None if any(x is None for x in res) else True)
    n_true = sum(x is True for x in ok.values())
    n_false = sum(x is False for x in ok.values())
    verdict = True if n_true >= 3 else (False if n_false >= len(ok) - 2 else None)
    return {"per_policy": ok, "COMPATIBLE": verdict, "not_judged": [p for p, x in ok.items() if x is None]}


# ================================================================================================= E3: per-policy heat
def policy_heat(demo: dict, room_c: float = ROOM_C) -> dict:
    """Each drive's demo-loop RMS over its continuous rating and hottest winding (35 C, method A, 8 min), the
    deciding drive and its mirror, the even left-right split (desk arithmetic) and its C1, by the form-factor study's
    functions on the policy's own demo course at 60 cm (k = 1). MEETS THE HEAT RULES AT 60 CM: C1 <= 90 C and
    C2 <= 0.80 on this demo course."""
    FF = _ff()
    d0 = FF.DRIVE_SETS["D0"]
    prof = FF.config_profile(demo, FF.S60, 1.0, d0)
    hr = heat_rules(demo, prof["walk_sq"], prof["stand_sq"], demo["minutes"], room_c)
    dm = FF.deciding_and_mirror(prof)
    ev = FF.even_split(prof)
    hev = heat_rules(demo, ev["walk_sq"], ev["stand_sq"], demo["minutes"], room_c)
    return {**hr, "deciding": dm, "even_split_C2": float(np.max(ev["loop_ratio"])), "even_split_C1_c": hev["C1_c"],
            "MEETS": bool(hr["C1_pass"] and hr["C2_pass"])}


def demo_make(run: str):
    """A ``rl/demo_course.py`` walker for ``run`` built as its ``policy_factory`` builds one (the run's config on the
    flat floor, randomisation off, noise on, 10 ms delay), with the policy through ``load_policy`` (the sweep's
    loaders, after the sha256 check). Returns (make, label) for ``demo_course.run``."""
    import trike_env as te
    ec = run_env_config(run)
    cfg = te.flat_floor(te.TrikeEnvConfig.from_dict({**ec, "model_path": str(te.MODEL_DIR /
                                                                             Path(ec["model_path"]).name)}))
    cfg = cfg.replace(domain_randomization=False, observation_noise=True, delay_steps_fixed=10)
    probe = te.TrikeEnv(cfg)
    act = policy_act(*load_policy(run, probe))

    def make(command):
        return te.TrikeEnv(cfg), act
    return make, f"policy {run} (sweep loaders)"


# ===================================================================================================== E4: forced air
FORCED_AIR_PATHS = ("winding_to_ambient", "total")


def forced_air_c1(demo: dict, ratio: float, *, heat_path: str, room_c: float = ROOM_C) -> dict:
    """C1 with every drive's fitted thermal resistance x ``ratio`` (R_forced / R_passive from a source for a comparable
    drive) and its heat capacity kept (so each time constant x ratio); method A, 35 C. Only to be called with a
    sourced ratio (E4). The build's model is one node, the winding against the ambient air
    (``rl/telemetry/thermal.py``), and this scales that whole resistance, so the sourced pair must be a
    winding-to-ambient (or total) resistance: ``heat_path`` must say so ("winding_to_ambient" or "total"). A
    housing-to-air or heat-sink pair is refused (ValueError): the winding-to-housing part does not change with the
    airflow, so scaling the whole R_th by it would overstate the fan."""
    if heat_path not in FORCED_AIR_PATHS:
        raise ValueError(f"heat path {heat_path!r}: only a winding-to-ambient or total R_th pair scales the build's "
                         f"one-node model")
    from telemetry import thermal as TH
    FF = _ff()
    prof = FF.config_profile(demo, FF.S60, 1.0, FF.DRIVE_SETS["D0"])
    therm, per = FF.thermal_inputs(demo, prof["walk_sq"], prof["stand_sq"], FF.DRIVE_SETS["D0"], demo["minutes"])
    out = {}
    with FF._persistent_fits():      # each drive's R_th and C fitted once per time constant, as the study does
        for tc in TH.TIME_CONSTANTS_S:
            best = (None, -1e9)
            for jn, d in therm.items():
                node = d.calibrate(tc, room_c=room_c)
                node = TH.ThermalNode(d, node.r_th_k_per_w * ratio, node.c_j_per_k, node.step_s, room_c=room_c)
                peak = max(T for _, T in node.run(per[jn]))
                if peak > best[1]:
                    best = (jn, peak)
            out[TH.tc_key(tc)] = {"joint": best[0], "peak_c": best[1]}
    return {"ratio": ratio, "heat_path": heat_path, "by_tc": out, "C1_c": out[TH.tc_key(SCORED_TC_S)]["peak_c"],
            "C1_pass": out[TH.tc_key(SCORED_TC_S)]["peak_c"] <= C1_LIMIT_C}


# ====================================================================================================== E5: the trot
E5_PARAMS = ("period_s", "duty", "body_pitch_deg", "lift_m", "max_reach_m")
E5_BOX_60 = {"period_s": (0.55, 0.85), "duty": (0.50, 0.65), "body_pitch_deg": (-2.0, 3.0), "lift_m": (0.020, 0.050),
             "max_reach_m": (0.060, 0.100)}
E5_STEP_60 = {"period_s": 0.05, "duty": 0.03, "body_pitch_deg": 0.5, "lift_m": 0.005, "max_reach_m": 0.010}
E5_ROUND = {"period_s": 3, "duty": 3, "body_pitch_deg": 2, "lift_m": 4, "max_reach_m": 4}
E5_SOBOL_SEED = 20261128
E5_SOBOL_N = 32
S50 = 0.50 / 0.60


def e5_box(size_m: float) -> dict:
    """The search box at a size: lengths x s, times x sqrt(s) (the Froude rule of build_scaled_trike), shares and the
    pitch angle kept."""
    s = size_m / 0.60
    f = {"period_s": math.sqrt(s), "duty": 1.0, "body_pitch_deg": 1.0, "lift_m": s, "max_reach_m": s}
    return {k: (lo * f[k], hi * f[k]) for k, (lo, hi) in E5_BOX_60.items()}, {k: E5_STEP_60[k] * f[k] for k in f}


def e5_default(size_m: float) -> dict:
    """The frozen controller's values at the size (WalkParams() at 60 cm; Froude-scaled at 50 cm, as
    build_scaled_trike.scaled_walk_params), body pitch 0."""
    walk60 = _walk60()
    P = walk60.WalkParams()
    s = size_m / 0.60
    return {"period_s": P.period_s * math.sqrt(s), "duty": P.duty, "body_pitch_deg": 0.0, "lift_m": P.lift_m * s,
            "max_reach_m": P.max_reach_m * s}


def e5_settings(size_m: float) -> list:
    """Stage 1's 33 settings at a size, fixed before any run: the default, then 32 points of a scrambled Sobol
    sequence (scipy.stats.qmc, seed 20261128) over the box, rounded."""
    from scipy.stats import qmc
    box, _ = e5_box(size_m)
    u = qmc.Sobol(d=len(E5_PARAMS), scramble=True, seed=E5_SOBOL_SEED).random(E5_SOBOL_N)
    rows = [{k: round(v, E5_ROUND[k]) for k, v in e5_default(size_m).items()}]
    for x in u:
        rows.append({k: round(box[k][0] + float(x[i]) * (box[k][1] - box[k][0]), E5_ROUND[k])
                     for i, k in enumerate(E5_PARAMS)})
    for i, r in enumerate(rows):
        r["id"] = f"H{size_m:.2f}-s1-{i:02d}"
    return rows


def e5_neighbours(best: dict, size_m: float, start: int) -> list:
    """Stage 3: one parameter at a time, +- one step from ``best`` (inside the box), at most 10 settings."""
    box, step = e5_box(size_m)
    out = []
    for k in E5_PARAMS:
        for sgn in (-1, 1):
            v = round(best[k] + sgn * step[k], E5_ROUND[k])
            if box[k][0] - 1e-9 <= v <= box[k][1] + 1e-9:
                out.append({**{kk: best[kk] for kk in E5_PARAMS}, k: v})
    for i, r in enumerate(out[:10]):
        r["id"] = f"H{size_m:.2f}-s3-{start + i:02d}"
    return out[:10]


def walk_params(setting: dict, size_m: float):
    """WalkParams at the size (frozen defaults, Froude-scaled at 50 cm) with the setting's period, duty, lift and
    reach; the body pitch is ``PitchedTrot``'s."""
    walk60 = _walk60()
    if abs(size_m - 0.60) < 1e-9:
        P = walk60.WalkParams()
    else:
        import build_scaled_trike as BS
        P = BS.scaled_walk_params(size_m / 0.60)
    return P.replace(period_s=float(setting["period_s"]), duty=float(setting["duty"]), lift_m=float(setting["lift_m"]),
                     max_reach_m=float(setting["max_reach_m"]))


class PitchedTrot:
    """``walk60.TrotController`` with a body-pitch setpoint theta (deg, + nose down, walk60's sign): the gravity reading
    it is given is turned by -theta about the body's y axis, so its attitude feedback holds the chassis at theta
    instead of level, and every foot target (chassis frame) is moved up by x tan(theta), x its fore-aft position, so
    the feet stay on level ground at that pitch. No other change; at theta = 0 its actions are the frozen
    controller's, bit for bit."""

    def __init__(self, env, P, pitch_deg: float = 0.0):
        walk60 = _walk60()
        self.ctrl = walk60.TrotController(env, P)
        self.theta = math.radians(float(pitch_deg))
        self.env = env
        i = 0
        self.gslot = None
        for nm, w in env.obs_layout:
            if nm == "gravity_body":
                self.gslot = slice(i, i + w)
            i += w
        if self.theta != 0.0:
            orig = self.ctrl.kin.ik
            tt = math.tan(self.theta)

            def ik(c, target, guess):
                t = np.asarray(target, dtype=np.float64).copy()
                t[2] += t[0] * tt
                return orig(c, t, guess)
            self.ctrl.kin.ik = ik

    def rotate_gravity(self, g) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        g = np.asarray(g, dtype=np.float64)
        return np.array([g[0] * c + g[2] * s, g[1], g[2] * c - g[0] * s])

    def act(self, obs):
        if self.theta != 0.0:
            obs = np.array(obs, copy=True)
            obs[self.gslot] = self.rotate_gravity(obs[self.gslot]).astype(obs.dtype)
        return self.ctrl.act(obs)


@contextlib.contextmanager
def _sweep_push_patches(axis: str):
    """For one trot push episode through ``sweep.run_episode``: the push along ``axis`` ("sideways", the sweep's own;
    "fore_aft": the chassis's forward axis at onset, as fore128_walk60 does) and the episode's 1 ms log kept (as
    fore128_slow_gait does), both undone after."""
    S, F = _sweep(), _f128()
    stash = {}
    orig_rec, orig_lat = S.episode_record, S.lateral_axis

    def rec(arr, p, seed, pusher):
        stash["arr"] = arr
        return F.record_with_speed(arr, p, seed, pusher)
    S.episode_record = rec
    S.lateral_axis = F.forward_axis if axis == "fore_aft" else orig_lat
    try:
        yield stash
    finally:
        S.episode_record, S.lateral_axis = orig_rec, orig_lat


def trot_judged_episode(model_path, cond: str, seed: int, P, pitch_deg: float = 0.0, extra_hooks=(),
                        command: float = E5_COMMAND) -> dict:
    """One PREREG_WALK60 condition (``judge_walk60.JUDGED`` J1-J4: randomisation and noise on, friction fixed, impratio)
    as ``fore128_walk60.judged_episode`` runs it, with ``PitchedTrot``; gait_metrics' measures and the slow-gait
    windows (``fore128_slow_gait.windows``)."""
    import gait_metrics
    import judge_walk60 as JW
    walk60, SG = _walk60(), _sg()
    _, plant, imp = JW.JUDGED[cond]
    env = make_env(walk60.plant_config(**{**plant, "model_path": str(model_path)}))
    if imp is not None:
        env.model.opt.impratio = float(imp)
    for h in extra_hooks:
        env.extra_hooks.append(h)
        if hasattr(h, "on_reset"):
            env.reset_hooks.append(h)
    ctrl = PitchedTrot(env, P, pitch_deg)
    log = gait_metrics.record_episode(env, ctrl.act, seed=seed, command=command,
                                      delay_steps=delay_steps_for(env.physics_dt))      # 10 ms (10 steps at 1 ms)
    ms = gait_metrics.measures(log, command)
    hk = [n for n in log["names"] if "hind_flexion" in n]
    return {"condition": cond, "seed": seed, "fell": bool(log["fell"]), "termination": log["termination"],
            "forward_speed_mps": float(ms["forward_speed_mps"]),
            "hind_knee_rms_over_rating_2_20": max(ms["drives"][n]["rms_over_continuous"] for n in hk),
            **SG.windows(log)}


def trot_push_episode(model_path, axis: str, dv: float, seed: int, P, pitch_deg: float = 0.0, extra_hooks=(),
                      command: float = E5_COMMAND) -> dict:
    """One push episode of the trot on the sweep's nominal plant (``trot_sweep_config``; noise on, 10 ms delay, push at
    8 s for 0.1 s, impulse = mass x dv) through ``sweep.run_episode``; its record and the slow-gait windows."""
    S, SG = _sweep(), _sg()
    env = make_env(trot_sweep_config(model_path))
    for h in extra_hooks:
        env.extra_hooks.append(h)
        if hasattr(h, "on_reset"):
            env.reset_hooks.append(h)
    ctrl = PitchedTrot(env, P, pitch_deg)
    with _sweep_push_patches(axis) as stash:
        rec = S.run_episode(env, None, None, seed, S.make_point(command=command, push_dv=dv,
                                                                delay_ms=delay_steps_for(env.physics_dt)),
                            controller=ctrl.act)
    w = SG.windows(stash["arr"])
    hk = [j for j, n in enumerate(stash["arr"]["names"]) if "hind_flexion" in n]
    a = stash["arr"]
    win = a["t"] >= WINDOW_S
    hk_2_20 = (float((np.sqrt((a["tau"][win][:, hk] ** 2).mean(0)) / SG.HIND_KNEE_RATING_NM).max())
               if win.sum() >= 100 else None)
    return {"axis": axis, "dv": dv, "seed": seed, "fell": bool(rec["fell"]), "termination": rec["termination"],
            "forward_speed_mps": rec.get("forward_speed_mps"), "hind_knee_rms_over_rating_2_20": hk_2_20, **w}


def is_slow_gait(ep: dict) -> bool:
    """``fore128_slow_gait.py``'s "slow gait" (its "slow and hot" count, the rule its report names the slow gait):
    over 10-20 s the speed under 0.225 m/s and the larger hind knee at or over 1.0 of its rating. A fall counts as
    a slow-gait episode here (E5)."""
    if ep["fell"] or ep.get("speed_10_20_mps") is None:
        return True
    return ep["speed_10_20_mps"] < 0.225 and ep["hind_knee_rms_over_rating_10_20"] >= 1.0


def worst_hind_knee(ep: dict) -> float:
    """The larger hind knee's RMS over its rating, the worse of the 2-20 s and the 10-20 s windows."""
    v = [x for x in (ep.get("hind_knee_rms_over_rating_2_20"), ep.get("hind_knee_rms_over_rating_10_20"))
         if x is not None]
    return max(v) if v else math.inf


def e5_success(by_condition: dict) -> dict:
    """E5's rule on one seed set: {condition: [episodes]} over J1-J4 and the six pushes. No falls in J1-J4, J3 mean
    speed (2-20 s) >= 0.225 m/s, slow-gait episodes <= 1 of 10 in every condition, worst hind knee <= 1.0 in every
    episode of every condition."""
    falls_j = {c: sum(e["fell"] for e in by_condition[c]) for c in ("J1", "J2", "J3", "J4")}
    j3 = float(np.mean([e["forward_speed_mps"] for e in by_condition["J3"]]))
    slow = {c: sum(is_slow_gait(e) for e in eps) for c, eps in by_condition.items()}
    hk = {c: max(worst_hind_knee(e) for e in eps) for c, eps in by_condition.items()}
    checks = {"no_falls_J1_J4": all(v == 0 for v in falls_j.values()), "J3_speed": j3 >= 0.225,
              "slow_gait_at_most_1_per_condition": all(v <= 1 for v in slow.values()),
              "hind_knee_at_most_1": all(v <= 1.0 for v in hk.values())}
    return {"falls_J": falls_j, "J3_speed_mps": j3, "slow_gait": slow, "worst_hind_knee": hk, "checks": checks,
            "SUCCEEDS": all(checks.values())}


def e5_screen_score(by_condition: dict) -> tuple:
    """Stage 1's ranking (lower is better): falls in J1 and J3, slow-gait episodes over the screen, the worst hind knee,
    minus the J3 speed."""
    falls = sum(e["fell"] for c in ("J1", "J3") for e in by_condition.get(c, []))
    slow = sum(is_slow_gait(e) for eps in by_condition.values() for e in eps)
    hk = max(worst_hind_knee(e) for eps in by_condition.values() for e in eps)
    j3 = float(np.mean([e["forward_speed_mps"] for e in by_condition["J3"]]))
    return (falls, slow, round(hk, 6), -round(j3, 6))


def e5_choose(results: dict) -> str:
    """The setting judged on the held-out seeds, from {setting id: e5_success on seeds 101-110}: of those that succeed,
    the lowest worst hind knee, then fewer slow-gait episodes, then the higher J3 speed; if none succeeds, the fewest
    failed checks first."""
    def key(i):
        r = results[i]
        return (sum(not v for v in r["checks"].values()), max(r["worst_hind_knee"].values()),
                sum(r["slow_gait"].values()), -r["J3_speed_mps"])
    return min(results, key=key)


# ===================================================================================================== provenance
BUILD_TOOLS = ("rl/trike_env.py", "walk/walk60.py", "rl/robustness/sweep.py", "sim/check_packaging60.py",
               "design/form_factor_study.py", "rl/telemetry/thermal.py", "rl/analysis/gait_pipe1.py",
               "rl/analysis/thermal_ambient.py", "rl/demo_course.py", "rl/gait_metrics.py", "walk/fore128_walk60.py",
               "walk/fore128_slow_gait.py", "walk/judge_walk60.py", "sim/build_scaled_trike.py", "sim/data/drives.json",
               "sim/generated/trike60_C_closed_capsule_fore128.xml")


def provenance(runner: Path | str | None = None, extra: dict | None = None) -> dict:
    """What every result file records (section 6): the commit (``git rev-parse HEAD``, read only), the sha256 of this
    file, of the runner script that wrote the result (default: the running ``__main__`` file, if any) and of the
    build tools it imports (``BUILD_TOOLS``), plus ``extra`` {label: path} (e.g. the variant models), and the
    Python, MuJoCo and NumPy versions."""
    import platform
    import subprocess
    import mujoco
    try:
        commit = subprocess.run(["git", "-C", str(BUILD), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001 - recorded as unknown, not hidden
        commit = None
    if runner is None:
        runner = getattr(sys.modules.get("__main__"), "__file__", None)
    files = {"sim/candidate_checks.py": HERE / "candidate_checks.py"}
    files.update({t: BUILD / t for t in BUILD_TOOLS})
    if runner is not None:
        files["runner:" + str(Path(runner).resolve())] = Path(runner)
    for k, v in (extra or {}).items():
        files[k] = Path(v)
    return {"commit": commit, "sha256": {k: (sha256_file(p) if Path(p).exists() else None) for k, p in files.items()},
            "runner": str(Path(runner).resolve()) if runner is not None else None,
            "python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__}


# =========================================================================================================== the plan
def plan(verify: bool = True) -> dict:
    """The frozen settings of E1-E5 (written before any experiment)."""
    out = {"date": DATE, "written_before_any_experiment": True, "prereg": "design/PREREG_CANDIDATES_2026_11_28.md",
           "model": str(MODEL.relative_to(BUILD)), "model_sha256": sha256_file(MODEL), "sources": SOURCES,
           "fixes_before_any_run": "independent check of 28 Nov 2026, findings F1-F6 (prereg section 0.1)"}
    if verify:
        rows = {}
        for run in POLICY_DIRS:
            try:
                v = verify_checkpoint(run)
                rows[run] = {"zip": v["zip"].name, "zip_sha256": v["zip_sha256"],
                             "vecnormalize_sha256": v["vecnormalize_sha256"], "match": True,
                             "model": Path(v["record"]["settings"]["env_config"]["model_path"]).name}
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                rows[run] = {"match": False, "error": repr(exc)}
        out["checkpoints"] = rows
    out["E1"] = {"policies": list(E1_POLICIES), "policy_conditions": POLICY_CONDITIONS, "seed_blocks": E1_BLOCKS,
                 "trot": {"conditions": ["J1", "J2", "J3", "J4"], "pushes": TROT_PUSHES, "seeds": E1_TROT_SEEDS},
                 "log_every_s": LOG_EVERY_S, "joints": list(LOG_JOINTS), "clash_pairs": CLASH_PAIRS,
                 "stops": candidate_stops(), "coupled_limit_margin_mm": CLEAR_MM, "clear_mm": CLEAR_MM,
                 "clip_share": CLIP_SHARE}
    jj = ("left_fore_pitch", "left_hind_pitch", "left_fore_flexion", "left_hind_flexion")
    stiff = {kind: {j: transmission_stiffness(kind, j) for j in jj} for kind in TRANSMISSIONS}
    stiff["belt_1span"] = {j: transmission_stiffness("belt", j, SIM_R, strands=1) for j in jj}
    out["E2"] = {"reductions": REDUCTIONS, "placements": {k: list(reduced_joints(k)) for k in PLACEMENTS},
                 "eta_old": ETA_OLD, "transmissions": TRANSMISSIONS, "supporting_eta": SUPPORTING_ETA,
                 "e2a_row_families": [list(s) for s in e2a_specs()], "stiffness_joint_side": stiff,
                 "belt_pretension": {j: belt_pretension(j) for j in jj},
                 "spans_m": SPANS_M, "belt": BELT, "sim_r": SIM_R, "series_physics_dt_s": SERIES_DT,
                 "fallback_physics_dt_s": FALLBACK_DT,
                 "delay_steps": {f"{dt:g}": delay_steps_for(dt) for dt in (0.001, SERIES_DT, FALLBACK_DT)},
                 "damping_ratio": DAMPING_RATIO, "variants": E2B_VARIANTS, "runs_at_start": list(E2B_RUNS_AT_START),
                 "unstable_termination": UNSTABLE, "conditions": list(E2B_CONDITIONS), "seed_blocks": E1_BLOCKS,
                 "cap_rpm": CAP_RPM, "peak_factor": PEAK_FACTOR, "thermal_eq_nm": THERMAL_EQ_NM}
    out["E4"] = {"heat_paths_accepted": list(FORCED_AIR_PATHS)}
    out["E3"] = {"policies": list(POLICY_DIRS), "stored_demos": sorted(p.name for p in DEMO_DIR.iterdir()),
                 "new_demo_seeds": [700000, 710000], "room_c": ROOM_C, "tc_s": SCORED_TC_S}
    out["E5"] = {"params": list(E5_PARAMS), "command_mps": E5_COMMAND, "screen_seeds": E5_SCREEN_SEEDS,
                 "tuning_seeds": E5_TUNING_SEEDS, "heldout": E5_HELDOUT, "pushes": E5_PUSHES,
                 "screen_conditions": E5_SCREEN_CONDITIONS, "max_settings": E5_MAX_SETTINGS,
                 "sizes": {f"{s:.2f}": {"box": e5_box(s)[0], "step": e5_box(s)[1], "default": e5_default(s),
                                        "stage1": e5_settings(s)} for s in (0.60, 0.50)},
                 "model_50cm": "sim/build_scaled_trike.py build_xml(0.50, 'D0', 'adopted', layout='L1')"}
    out["provenance"] = provenance(runner=HERE / "candidate_checks.py")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default=str(RESULTS / "plan.json"))
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        res = plan()
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                       + "\n")
        bad = [r for r, v in res["checkpoints"].items() if not v["match"]]
        print(f"wrote {out}; checkpoints verified {len(res['checkpoints']) - len(bad)} of {len(res['checkpoints'])}"
              + (f"; MISMATCH {bad}" if bad else ""))
        return 1 if bad else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
