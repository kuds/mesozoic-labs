"""Stage 2 study: NVIDIA Jetson Orin Nano Super + Partial PLEM architecture, walking speed ceilings,
and behavior impact on ``60-N-T-10`` (60 cm) and ``75-M-H-10`` (75 cm).

Simulation and verification only; nothing is adopted into the Round 2 build records (DECISION_LOG P24),
and none of ``round2/`` or the hash-locked ``designlab/stage1/`` modules are modified (``STAGE1B_PLAN.md``
decision 5).  Writes ``stage2/jetson_plem/jetson_plem_study.json``; ``stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md``
reports it.

Inputs, each checked before use:

* ``60-N-T-10`` and ``75-M-H-10`` rebuilt from their Round 2 records via ``refs.source``, verifying each
  compiled MJCF's sha256 against both the record's ``xml_sha256`` and ``refs.SHA256_PREFIX``, and reproducing
  the committed mass, single-support holds, double-support static torques, and nominal walk RMS requirements
  within 1e-9 relative;
* catalog hardware parts (``designlab/data/parts.json``): ``rpi5_8gb_ai_hat_plus_13tops``,
  ``nvidia_orin_nano_super_devkit``, ``pololu_d36v50f12`` (12 V / 4.5 A, 50 V max in),
  ``pololu_d42v110f12`` (12 V / 9 A, 60 V max in), ``pcan_usb_fd``, ``lipo_6s_3300``, and ``lipo_6s_5000``;
* Round 2 sensor blind-zone and stopping-deceleration screens (``*_sensors.json`` and ``*_envelope.json``);
* Stage 1 closed-loop MuJoCo plants (``plant.build``) and tuned controllers (``stage1/runs/2026-11-08`` and
  ``stage1/runs/2026-11-10-speed``).

Run from ``hardware/trex_form_study`` with ``PYTHONPATH=.``:

    MUJOCO_GL=disable OMP_NUM_THREADS=1 python stage2/jetson_plem/jetson_plem_study.py [--procs 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np

RUN = Path(__file__).resolve().parent
STUDY = RUN.parents[1]  # hardware/trex_form_study
OUT = RUN / "jetson_plem_study.json"

PAIRS = (
    ("60-N-T-10", "narrow_carpet_60"),
    ("75-M-H-10", "feet_75"),
)

VOLTAGES = (25.2, 24.0, 22.2, 21.0)
PDB_SHORT_LEN_M = 0.094  # 94 mm shortened PDB option for 60-N-T-10 tail-base abdomen bay (vs 110 mm default)

STUDY_RULES = dict(
    baseline_reproduction=(
        "refs.source rebuilds each body from its Round 2 record and checks xml_sha256 against the record and "
        "refs.SHA256_PREFIX; sizing.requirements (with refine._init settings) reproduces the record's mass, "
        "single-support holds, double-support static torques, and nominal walk RMS requirements within 1e-9 relative."
    ),
    jetson_packaging_and_power=(
        "Replaces selection['compute'] with catalog part nvidia_orin_nano_super_devkit (103 x 90.5 x 34.77 mm, "
        "0.174 kg) inside pipeline.build_candidate and checks clearance._same_body_overlaps at home, "
        "metrics.measure exterior shell dimensions, and 6S->12V buck regulation (pololu_d36v50f12 for 15 W mode, "
        "pololu_d42v110f12 for 25 W Super mode, eta=0.93). Partial PLEM replaces the 4.0 W Teensy MCU bridge "
        "with PREEMPT_RT + isolated Jetson CPU 5 + USB-CAN FD (pcan_usb_fd, 0.5 W) + minimal pelvis IMU/E-stop "
        "watchdog (1.5 W)."
    ),
    speed_ceilings=(
        "Evaluates Stage 0 Standard V0 (sizing.NOMINAL / sizing.BRISK / envelope.FAST + Froude scaling) against "
        "F25-style hard-floor tuned gait + F9 counter-phased tail (duty 0.55 at >=0.36 m/s, lift 0.081 H, crouch "
        "0.07 H at 60 cm / 0.05 H at 75 cm, tail_amp -0.125 rad, stride_rel anchored at GAIT_TUNE.md 0.713 / 0.863) "
        "across 6S voltages (25.2, 24.0, 22.2, 21.0 V), screen.passive_loads (A1 thermal ratios), single-support "
        "stance friction mu_req, and mid-swing contact-pad clearance (>= 0.04 H)."
    ),
    latency_and_stopping=(
        "Combines Round 2 chest VL53L5CX ToF floor horizon (15 Hz frame rate, 2-frame worst-case / 1-frame mean "
        "debounce) and skull OAK-D near-floor blind zone with transitions.py integer stride-boundary commitment "
        "and 1-foot braking deceleration across Pi 5 + Hailo (175-200 ms host latency) vs Jetson TensorRT + "
        "Partial PLEM (45-60 ms host latency), and probes closed-loop survival at 0.400 m/s under "
        "speedfit.servo_delay (0, 2, 6, 10 ms)."
    ),
)


def _setup() -> None:
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    import designlab  # noqa: F401


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _rel_diff(a: float, b: float) -> float:
    return 0.0 if a == b else abs(a - b) / max(abs(a), abs(b))


def verify_baseline_reproduction(src: dict, case: str) -> dict:
    """Check that the rebuilt Round 2 model, static holds, and nominal walk match the committed record."""
    from designlab import gait, metrics, refine, sizing

    refine._init(case)
    gait.set_accuracy("r2")
    M, q0, design, H = src["M"], src["q_home"], src["design"], src["H"]
    rec = src["record"]
    meas = metrics.measure(M, design, q0)
    req = sizing.requirements(M, q0, H, src["layout"])

    diffs = []
    diffs.append(_rel_diff(float(M.mass()), float(rec["metrics"]["mass"])))
    for g_name, r_g in req["groups"].items():
        for field in ("hold", "double_support", "rms"):
            diffs.append(_rel_diff(float(r_g[field]), float(rec["margins"][g_name][field])))

    max_rd = float(max(diffs))
    return dict(
        xml_matches_record=bool(src["model"]["matches_record"]),
        xml_matches_stage1_plan=bool(src["model"]["matches_plan"]),
        n_checks=len(diffs),
        max_rel_diff=max_rd,
        all_match=bool(src["model"]["matches_record"] and src["model"]["matches_plan"] and max_rd <= 1e-9),
        baseline_exterior_mm=dict(
            crown_mm=round(1000.0 * float(meas["crown"]), 2),
            torso_width_mm=round(1000.0 * float(meas["torso_width"]), 2),
            hardware_width_mm=round(1000.0 * float(meas["hardware_width"]), 2),
            thigh_width_mm=round(1000.0 * float(meas["thigh_width"]), 2),
            max_width_mm=round(1000.0 * float(meas["max_width"]), 2),
            com_xyz_mm=[round(1000.0 * float(x), 2) for x in meas["com"]],
        ),
        baseline_mass_kg=round(float(M.mass()), 6),
        baseline_drives_w=round(float(rec["power"]["drives_w"]), 3),
        baseline_total_w=round(float(rec["power"]["total_w"]), 3),
        baseline_runtime_min=round(float(rec["runtime"]["minutes"]), 2),
    )


def _build_with_compute(src: dict, case: str, compute_id: str, pdb_len_m: float | None = None) -> dict:
    """Build candidate with modified compute part (and optional shortened 24V PDB length) in a temp dir."""
    from designlab import build as dl_build, clearance, metrics, pipeline, refine

    refine._init(case)
    rec = src["record"]
    cand = src["design"].cand
    sel = dict(rec["selection"])
    sel["compute"] = compute_id

    orig_place = dl_build.Design._place_trunk_hardware

    def _patched_place(self_d):
        orig_place(self_d)
        if pdb_len_m is not None and self_d.cand.battery_pos == "tailbase" and self_d.sel["bus"] <= 24:
            B = self_d.sk.B
            piv = self_d.sk.pivot[0]
            comp_ = dl_build.catalog.parts()[self_d.sel["compute"]]
            tray_x = comp_.dims[0] + 0.017
            pdims = (pdb_len_m, 0.070, 0.050)
            px = min(0.22 * B + 0.5 * pdims[0], 0.60 * B - 0.5 * tray_x - 0.5 * pdims[0] - 0.010)
            stp = self_d.at(px)
            ppos = self_d._chest(np.array([px, 0.0, stp["ventral"] + 0.5 * pdims[2] + 0.014 + 0.01 * B]), px > piv)
            pR = dl_build.ry(self_d.sk.bend) if px > piv else np.eye(3)
            for idx, it in enumerate(self_d.bodies["trunk"].items):
                if it.name == "power_distribution":
                    self_d.bodies["trunk"].items[idx] = replace(it, pos=ppos, rot=pR, size=pdims)
                    break

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        try:
            if pdb_len_m is not None:
                dl_build.Design._place_trunk_hardware = _patched_place
            res = pipeline.build_candidate(cand, td_path, selection=sel, B=float(rec["B"]), write=True)
        finally:
            dl_build.Design._place_trunk_hardware = orig_place

        xml_p = td_path / f"{cand.id}.xml"
        M_j, q_j, d_j = res["model"], res["q_home"], res["design"]
        meas = metrics.measure(M_j, d_j, q_j)
        S_cl = clearance._Screen(str(xml_p))
        S_cl.set_q(q_j)
        sb = clearance._same_body_overlaps(S_cl)

        trunk_items = {it.name: it for it in d_j.bodies["trunk"].items}
        widen_mm = max(
            [1000.0 * float(w[3]) for w in d_j.geometry_checks.get("trunk_widened_for_hardware", [])],
            default=0.0,
        )
        overlaps = [
            dict(geoms=ov["geoms"], gap_mm=round(float(ov["min_mm"]), 2))
            for ov in sb.get("overlaps", [])
        ]
        g_pdb = S_cl.gid("hw_power_distribution")
        hw_comp = [
            int(g)
            for g in S_cl.screened
            if S_cl.kind[int(g)] == "hardware"
            and not any(t in S_cl.gname[int(g)] for t in clearance.STRUCTURE_TOKENS)
        ]
        pdb_pairs = [
            dict(geoms=[S_cl.gname[g_pdb], S_cl.gname[g2]], gap_m=S_cl.dist(g_pdb, g2, 0.20))
            for g2 in hw_comp
            if g_pdb >= 0
            and g2 != g_pdb
            and S_cl.geom_grp[g2] == S_cl.geom_grp[g_pdb]
            and not clearance._encloses(S_cl, g_pdb, g2)
            and not clearance._encloses(S_cl, g2, g_pdb)
        ]
        min_pdb_pair = min(pdb_pairs, key=lambda p: p["gap_m"]) if pdb_pairs else None

        return dict(
            compute_part=compute_id,
            pdb_length_mm=round(1000.0 * (pdb_len_m if pdb_len_m is not None else 0.110), 1),
            mass_kg=round(float(M_j.mass()), 4),
            added_mass_over_pi5_g=round(1000.0 * float(M_j.mass() - src["M"].mass()), 1),
            exterior_mm=dict(
                crown_mm=round(1000.0 * float(meas["crown"]), 2),
                torso_width_mm=round(1000.0 * float(meas["torso_width"]), 2),
                hardware_width_mm=round(1000.0 * float(meas["hardware_width"]), 2),
                thigh_width_mm=round(1000.0 * float(meas["thigh_width"]), 2),
                max_width_mm=round(1000.0 * float(meas["max_width"]), 2),
                com_xyz_mm=[round(1000.0 * float(x), 2) for x in meas["com"]],
            ),
            max_chest_profile_flare_mm=round(widen_mm, 2),
            compute_pos_trunk_mm=[round(1000.0 * float(x), 2) for x in trunk_items["compute"].pos],
            pdb_pos_trunk_mm=[round(1000.0 * float(x), 2) for x in trunk_items["power_distribution"].pos],
            same_body_overlap_count=len(overlaps),
            same_body_overlaps=overlaps,
            min_pdb_gap_mm=round(1000.0 * float(min_pdb_pair["gap_m"]), 2) if min_pdb_pair else None,
            min_pdb_pair=min_pdb_pair["geoms"] if min_pdb_pair else None,
        )


def evaluate_power_and_regulators(src: dict) -> dict:
    """Verify 6S->12V buck regulators and compute power/runtime across Pi 5 and Jetson + Partial PLEM stacks."""
    from designlab import catalog, power

    rec = src["record"]
    drives_w = float(rec["power"]["drives_w"])
    sensors_w = float(rec["power"]["aux"]["loads"]["sensors"])

    reg_15w = catalog.part("pololu_d36v50f12")
    reg_25w = catalog.part("pololu_d42v110f12")
    can_fd = catalog.part("pcan_usb_fd")
    orin = catalog.part("nvidia_orin_nano_super_devkit")
    pi5 = catalog.part("rpi5_8gb_ai_hat_plus_13tops")
    pack_3300 = catalog.part("lipo_6s_3300")
    pack_5000 = catalog.part("lipo_6s_5000")

    eta_12v = 0.93
    eta_5v = power.ETA_DCDC

    stacks = {
        "pi5_hailo_teensy_r2": dict(
            description="Round 2 baseline: Pi 5 + AI HAT+ (9.5 W) + Teensy 4.1 MCU bridge (4.0 W)",
            aux_input_w=(9.5 + 4.0 + sensors_w) / eta_5v,
            added_chest_mass_g=0.0,
            regulator_id=None,
        ),
        "jetson_15w_teensy": dict(
            description="Jetson Orin Nano Super @ 15 W mode (+2.5 W NVMe/fan) + Teensy 4.1 bridge (4.0 W)",
            aux_input_w=(17.5 / eta_12v) + ((4.0 + sensors_w) / eta_5v),
            added_chest_mass_g=round(1000.0 * (orin.mass - pi5.mass + reg_15w.mass), 1),
            regulator_id=reg_15w.id,
        ),
        "jetson_15w_partial_plem": dict(
            description="Jetson Orin Nano Super @ 15 W mode (+2.5 W NVMe/fan) + Partial PLEM RT CPU 5 + USB-CAN FD (2.0 W)",
            aux_input_w=(17.5 / eta_12v) + ((2.0 + sensors_w) / eta_5v),
            added_chest_mass_g=round(1000.0 * (orin.mass - pi5.mass + reg_15w.mass - 0.064), 1),
            regulator_id=reg_15w.id,
        ),
        "jetson_25w_partial_plem": dict(
            description="Jetson Orin Nano Super @ 25 W Super mode (+3.5 W NVMe/fan) + Partial PLEM RT CPU 5 + USB-CAN FD (2.0 W)",
            aux_input_w=(28.5 / eta_12v) + ((2.0 + sensors_w) / eta_5v),
            added_chest_mass_g=round(1000.0 * (orin.mass - pi5.mass + reg_25w.mass - 0.064), 1),
            regulator_id=reg_25w.id,
        ),
    }

    out_stacks = {}
    for k, st in stacks.items():
        tot_w = drives_w + st["aux_input_w"]
        rt_3300 = power.runtime(pack_3300, tot_w)
        rt_5000 = power.runtime(pack_5000, tot_w)
        out_stacks[k] = dict(
            description=st["description"],
            regulator_id=st["regulator_id"],
            drives_w=round(drives_w, 2),
            aux_input_w=round(st["aux_input_w"], 2),
            total_walking_w=round(tot_w, 2),
            net_mass_delta_g=st["added_chest_mass_g"],
            runtime_6s_3300_min=round(float(rt_3300["minutes"]), 2),
            runtime_6s_5000_min=round(float(rt_5000["minutes"]), 2),
            meets_20min_spec_on_3300=bool(rt_3300["minutes"] >= 20.0),
        )

    return dict(
        regulators=dict(
            mode_15w=dict(
                id=reg_15w.id,
                max_vin_v=50.0,
                vout_v=12.0,
                max_power_w=54.0,
                mass_g=round(1000.0 * reg_15w.mass, 1),
                price_usd=reg_15w.price,
                clears_6s_full_25_2v=True,
                load_fraction_at_18w_peak=round(18.0 / 54.0, 3),
            ),
            mode_25w=dict(
                id=reg_25w.id,
                max_vin_v=60.0,
                vout_v=12.0,
                max_power_w=108.0,
                mass_g=round(1000.0 * reg_25w.mass, 1),
                price_usd=reg_25w.price,
                clears_6s_full_25_2v=True,
                load_fraction_at_28_5w_peak=round(28.5 / 108.0, 3),
            ),
            usb_can_fd=dict(
                id=can_fd.id,
                mass_g=round(1000.0 * can_fd.mass, 1),
                power_typ_w=can_fd.power_typ,
                price_usd=can_fd.price,
            ),
        ),
        stacks=out_stacks,
    )


def _std_spec(H: float, v: float) -> object:
    from designlab import envelope, gait, sizing
    from designlab.stage2 import look

    nom = sizing.NOMINAL.describe(H)["speed"]
    brisk = sizing.BRISK.describe(H)["speed"]
    fast = envelope.FAST.describe(H)["speed"]
    fr = v * v / (gait.G * H)
    if abs(v - nom) < 1.5e-3:
        return sizing.NOMINAL
    if v < brisk - 1e-3:
        return look.speed_spec(H, v, lo=sizing.NOMINAL, hi=sizing.BRISK)
    if abs(v - brisk) <= 1.5e-3:
        return sizing.BRISK
    if v < fast - 1e-3:
        return look.speed_spec(H, v, lo=sizing.BRISK, hi=envelope.FAST)
    if abs(v - fast) <= 1.5e-3:
        return envelope.FAST
    s_rel = min(0.96, 0.80 * ((fr / 0.05) ** 0.26))
    return replace(envelope.FAST, froude=fr, stride_rel=s_rel)


def _tuned_f9_spec(body: str, H: float, v: float) -> object:
    from designlab import gait

    fr = v * v / (gait.G * H)
    crouch = 0.07 if body.startswith("60") else 0.05
    duty = 0.58 if v < 0.36 else 0.55
    base_sr = 0.713 if body.startswith("60") else 0.863
    sr_cap = 0.90 if body.startswith("60") else 0.94
    sr = round(min(sr_cap, max(0.62, base_sr * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    return gait.GaitSpec(
        froude=fr,
        duty=duty,
        lift_frac=0.081,
        head_amp=0.05,
        tail_amp=-0.125,
        sway_gain=1.0,
        stride_rel=sr,
        crouch=crouch,
        zmp_iters=6,
        zmp_gain=0.5,
        n=160,
    )


def _eval_speed_point(src: dict, spec, usable_map: dict) -> dict:
    from designlab import carpet, clearance, envelope, gait, sizing, statics, turning
    from designlab.stage1 import drive
    from designlab.stage2 import look, screen

    gait.set_accuracy("r2")
    M, q0, design, H = src["M"], src["q_home"], src["design"], src["H"]
    tr = gait.generate(M, q0, H, spec)
    idr = gait.inverse_dynamics(M, tr)
    ms = sizing._motor_space(M.names, idr["tau"], idr["omega"], src["layout"])
    ratios_24 = envelope._ratios(design, src["record"]["selection"]["bus"], ms)
    wg, wv = max(ratios_24.items(), key=lambda kv: kv[1]["worst"])

    v_map = {}
    for vb in VOLTAGES:
        raws = {}
        for g_n, (_tg, wg_arr) in ms.items():
            act = design.act(g_n)
            wpk = float(np.max(np.abs(wg_arr)))
            wlim = act.speed_at(vb)
            wnl = drive.no_load_speed_at(act, vb if act.box is None else None)
            raws[g_n] = (wpk / wlim, sizing.SPEED_MARGIN * wpk / wlim, wpk / wnl)
        top_g = max(raws.items(), key=lambda kv: kv[1][0])
        v_map[f"{vb:.1f}V"] = dict(
            worst_group=top_g[0],
            raw_speed_ratio=round(top_g[1][0], 4),
            planning_speed_ratio=round(top_g[1][1], 4),
            over_no_load=round(top_g[1][2], 4),
            knee_planning_ratio=round(raws["knee"][1], 4),
        )

    passive = screen.passive_loads(M, design, idr["tau"], idr["omega"])
    idr_turn = turning.inverse_dynamics(M, look.turning_traj(tr, M, q0), turning.MU, split="statics", ds_friction=False)
    ss = [k for k, p in enumerate(tr["phases"]) if len(p) == 1]
    mu_req = float(np.nanmax([idr_turn["feet"][tr["phases"][k][0]]["mu_req"][k] for k in ss]))

    prof = carpet.swing_profiles(M, tr)
    pad_mm = 1000.0 * min(
        float(np.min(p_sd["z_pad"][(p_sd["s"] >= carpet.MID_SWING[0]) & (p_sd["s"] <= carpet.MID_SWING[1])]))
        for p_sd in prof.values()
    )
    target_mm = 1000.0 * 0.04 * H

    q = np.asarray(tr["q"], float)
    min_j_margin = float("inf")
    for jn, urec in usable_map.items():
        if jn not in M.names:
            continue
        adr = int(M.m.jnt_qposadr[statics.jid(M.m, jn)])
        vals = q[:, adr]
        lo, hi = urec["usable"]
        rest = urec.get("rests_on_stop")
        if lo is not None and rest != "min":
            min_j_margin = min(min_j_margin, float(vals.min() - lo))
        if hi is not None and rest != "max":
            min_j_margin = min(min_j_margin, float(hi - vals.max()))

    gates_ok = bool(tr.get("ik_ok", True) and pad_mm >= target_mm - 0.05 and min_j_margin >= clearance.RANGE_MARGIN_RAD - 1e-6)
    return dict(
        speed_ms=round(float(tr["info"]["speed"]), 4),
        froude=round(float(spec.froude), 5),
        duty=round(float(spec.duty), 3),
        stride_rel=round(float(spec.stride_rel), 4),
        crouch=round(float(spec.crouch), 3),
        tail_amp=round(float(spec.tail_amp), 4),
        gates_ok=gates_ok,
        mid_swing_pad_mm=round(pad_mm, 2),
        min_joint_margin_rad=round(min_j_margin, 4),
        worst_group_24v=wg,
        worst_planning_ratio_24v=round(float(wv["worst"]), 4),
        voltage_speed=v_map,
        tail_cont_ratio_total=round(float(passive["tail"]["cont_ratio_total"]), 4),
        neck_cont_ratio_total=round(float(passive["neck"]["cont_ratio_total"]), 4),
        mu_req_max=round(mu_req, 4),
    )


def evaluate_speed_and_latency(src: dict, case: str) -> dict:
    """Evaluate milestone Stage 0 speeds, closed-loop 0.40 m/s latency probe, and ToF stopping bounds."""
    from designlab.stage1 import control, env as s1_env, plant, refs, speedfit, transitions, tune

    body = src["body"]
    H = src["H"]
    cl_doc = json.loads((STUDY / "round2" / "refine" / case / "review" / "screening" / f"{body}_clearance.json").read_text())
    sdoc = json.loads((STUDY / "round2" / "refine" / case / "review" / "screening" / f"{body}_sensors.json").read_text())
    edoc = json.loads((STUDY / "round2" / "refine" / case / "review" / "screening" / f"{body}_envelope.json").read_text())
    usable_map = cl_doc["summary"]["joint_range_margins"]

    milestones = (
        [0.218, 0.315, 0.400, 0.446, 0.520, 0.580]
        if body.startswith("60")
        else [0.243, 0.351, 0.400, 0.496, 0.560, 0.620]
    )
    std_rows = [_eval_speed_point(src, _std_spec(H, v), usable_map) for v in milestones]
    tuned_rows = [_eval_speed_point(src, _tuned_f9_spec(body, H, v), usable_map) for v in milestones]

    p24 = plant.build(body, "hard")
    s1b_path = STUDY / "stage1" / "runs" / "2026-11-10-speed" / body / "speedg5.json"
    if s1b_path.exists():
        sel = json.loads(s1b_path.read_text())["seeds"]["42"]["selected_gains"]
        gains = np.array([float(sel[n]) for n in control.GAIN_NAMES])
        ctrl_label = "R2s"
    else:
        s1_path = STUDY / "stage1" / "runs" / "2026-11-08" / body / "tune.json"
        gains, _ = s1_env.tuned_gains(s1_path.parent, body, s1_path)
        ctrl_label = "R2"

    sp_40 = _tuned_f9_spec(body, H, 0.400)
    tr_40 = __import__("designlab.gait", fromlist=["generate"]).generate(src["M"], src["q_home"], H, sp_40)
    idr_40 = __import__("designlab.gait", fromlist=["inverse_dynamics"]).inverse_dynamics(src["M"], tr_40)
    ref_40 = refs.Reference("tuned_040", refs.arrays(src, dict(traj=tr_40, idr=idr_40)))

    g_stand = transitions.Gait("stand")
    i40 = sp_40.describe(H)
    g_40 = transitions.Gait(
        name="tuned_040",
        period=float(i40["period"]),
        duty=float(sp_40.duty),
        lift=float(sp_40.lift_frac * H),
        speed=float(i40["speed"]),
        yaw_rate=0.0,
        head_amp=float(sp_40.head_amp),
        tail_amp=float(sp_40.tail_amp),
        crouch=float(sp_40.crouch * H),
        zmp_iters=2,
        zmp_gain=0.5,
        split="statics",
    )
    geo = transitions.Geometry.of(src["M"], src["q_home"], g_40.crouch)
    pl = transitions.plan("start", g_stand, g_40, geo)
    x0 = transitions.initial_state(pl, None, geo)
    zref = pl.zmp.copy()
    for it in range(3):
        com_t = transitions.com_path(pl, zref, x0, geo.omega)
        qs, ferr, cerr = transitions.solve_frames(src["M"], src["q_home"], pl, com_t, geo)
        qd, qdd = transitions.derivatives(src["M"].m, qs, pl.t)
        zmb = transitions.multibody_zmp(src["M"], qs, qd, qdd)
        if it < 2:
            zref = zref - 0.5 * np.nan_to_num(zmb - pl.zmp)
    o = pl.out
    tau, cop_m, norm_f, fric_f = transitions.idyn(src["M"], qs[o], qd[o], qdd[o], pl.swing[o], "statics")
    feet = np.zeros((pl.n_out, 2, 3))
    com_arr = np.zeros((pl.n_out, 3))
    for i, k in enumerate(range(o.start, o.stop)):
        src["M"].forward(qs[k])
        feet[i] = [src["M"].foot_pose(s)[0] for s in ("left", "right")]
        com_arr[i] = src["M"].com()
    tr_start = transitions.Transition(
        "start",
        transitions.arrays(
            dict(
                name="start",
                plan=pl,
                geo=geo,
                a=g_stand,
                b=g_40,
                q=qs[o],
                qd=qd[o],
                qdd=qdd[o],
                tau=tau,
                com_ref=com_t[o],
                zmp_ref=pl.zmp[o],
                zmp_mb=zmb[o],
                zref=zref[o],
                cop_margin=cop_m,
                normal=norm_f,
                friction=fric_f,
                feet=feet,
                com=com_arr,
                foot_error=ferr[o],
                com_error=cerr[o],
                zmp_error_history_m=[0.0],
                zmp_iters=2,
                zmp_gain=0.5,
                split="statics",
                x0=x0,
                seconds=0.0,
                ik_ok=True,
                q_home=np.asarray(src["q_home"], float).copy(),
                joints=list(src["M"].names),
            )
        ),
    )
    omega = control.omega0(p24, ref_40.q_home, crouch_frac=sp_40.crouch)
    track = control.Track(p24, omega)
    s0_frame = control._joint_stack(p24, ref_40.q[:1], ref_40.qd[:1], ref_40.tau[:1], ref_40.qdd[:1])[0]
    track.add_transition(tr_start, next_frame=s0_frame, next_dt=ref_40.step)
    track.add_periodic(ref_40, strides=5)

    latency_probe = {}
    for d_steps in (0, 1, 3, 5):  # 0, 2, 6, 10 ms
        n_ok = 0
        for s_i in range(3):
            tk = control.Tracker(p24, track, "R2", gains=gains, rate=float(tune.COMMAND_RATE))
            with speedfit.servo_delay(p24, d_steps):
                out_ep = control.run(
                    tk,
                    track.t_end / float(tune.COMMAND_RATE),
                    start="stance",
                    seed=tune.derived_seed(4100 + s_i, 0, 0),
                    noise_seed=tune.derived_seed(4100 + s_i, 0, 1),
                    joint_noise=tune.START_JOINT_NOISE_RAD,
                    root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S,
                )
            if not out_ep["fell"]:
                n_ok += 1
        latency_probe[f"{2 * d_steps}ms"] = dict(delay_ms=2 * d_steps, survived=n_ok, total=3, rate=round(n_ok / 3, 3))

    cam_near_m = float(sdoc["camera"]["poses"]["home"]["centre_column"]["visible"]["ahead_of_toes_m"])
    tof_in_m, tof_out_m = (float(x) for x in sdoc["summary"]["tof"]["tof_l"]["floor_ahead_of_toes_m"])
    a_1foot = float(edoc["stopping"]["max_one_foot_decel_ms2"])
    standoff_m = 0.10
    usable_tof_m = tof_out_m - standoff_m

    def safe_stop_speed(t_tof_plus_host: float, commit_strides: float) -> float:
        best = 0.0
        for v in np.linspace(0.20, 0.90, 141):
            fr = v * v / (9.81 * H)
            sr = min(0.92, max(0.55, 0.80 * ((max(fr, 0.012) / 0.05) ** 0.18)))
            l_str = sr * H
            d_stop = (commit_strides + 0.75) * l_str + (v * v) / (2.0 * min(a_1foot, 1.50))
            if v * t_tof_plus_host + d_stop <= usable_tof_m:
                best = float(round(v, 3))
        return best

    return dict(
        stage0_standard=std_rows,
        stage0_tuned_f9=tuned_rows,
        closed_loop_latency_probe_040ms=dict(controller=ctrl_label, results=latency_probe),
        stopping_horizon=dict(
            camera_near_floor_ahead_of_toes_m=round(cam_near_m, 4),
            tof_floor_window_ahead_of_toes_m=[round(tof_in_m, 4), round(tof_out_m, 4)],
            blind_gap_tof_to_camera_m=round(cam_near_m - tof_out_m, 4),
            standoff_m=standoff_m,
            usable_tof_stopping_budget_m=round(usable_tof_m, 4),
            max_one_foot_decel_ms2=round(a_1foot, 4),
            max_tof_safe_speed_pi5_worst_ms=safe_stop_speed(0.133 + 0.200, 1.00),
            max_tof_safe_speed_jetson_plem_worst_ms=safe_stop_speed(0.133 + 0.060, 1.00),
            max_tof_safe_speed_pi5_mean_ms=safe_stop_speed(0.067 + 0.175, 0.50),
            max_tof_safe_speed_jetson_plem_mean_ms=safe_stop_speed(0.067 + 0.045, 0.50),
        ),
    )


def _body_worker(pair: tuple[str, str]) -> tuple[str, dict]:
    _setup()
    from designlab.stage1 import refs

    t0 = time.perf_counter()
    body, case = pair
    src = refs.source(body)
    base_rep = verify_baseline_reproduction(src, case)

    pkg_dropin = _build_with_compute(src, case, "nvidia_orin_nano_super_devkit", pdb_len_m=None)
    pkg_short_pdb = (
        _build_with_compute(src, case, "nvidia_orin_nano_super_devkit", pdb_len_m=PDB_SHORT_LEN_M)
        if body.startswith("60")
        else None
    )
    pwr = evaluate_power_and_regulators(src)
    spd = evaluate_speed_and_latency(src, case)

    return body, dict(
        body=body,
        refine_case=case,
        H_m=round(float(src["H"]), 5),
        B_m=round(float(src["record"]["B"]), 5),
        wall_s=round(time.perf_counter() - t0, 2),
        baseline_reproduction=base_rep,
        jetson_packaging=dict(
            drop_in_devkit=pkg_dropin,
            shortened_94mm_pdb_fix=pkg_short_pdb,
        ),
        power_and_regulators=pwr,
        speed_and_latency=spd,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t0 = time.perf_counter()
    _setup()
    if args.procs > 1:
        ctx = get_context("spawn")
        with ctx.Pool(min(args.procs, len(PAIRS))) as pool:
            items = pool.map(_body_worker, PAIRS)
    else:
        items = [_body_worker(p) for p in PAIRS]

    doc = dict(
        study="stage2_jetson_orin_plem_and_speed",
        script="stage2/jetson_plem/jetson_plem_study.py",
        script_sha256=sha256(Path(__file__)),
        wall_s=round(time.perf_counter() - t0, 2),
        rules=STUDY_RULES,
        bodies={k: _plain(v) for k, v in items},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {args.out} in {doc['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
