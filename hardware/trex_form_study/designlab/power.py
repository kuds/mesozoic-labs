"""Electrical power, runtime and battery selection.

Per QDD leg drive (output torque tau, speed w, output-side Kt per A rms, per-phase R):
    I = |tau| / Kt ;  P_cu = 3 I^2 R ;  P_mech = tau w
    P_in = (max(P_mech, 0) / eta_gear + P_cu) / eta_drive + P_idle
Negative mechanical power is not credited (no regeneration), matching PR #662's assumption.
Box servos (brushed/coreless, 12 V): P_in = V_servo |tau| / Kt + P_idle_servo.
Auxiliaries (compute, sensors, MCU) are summed from the parts and passed through the DC-DC
efficiency.  Runtime = usable energy / mean electrical power over the nominal reference walk.

Constants are PR #662's planning values unless stated; they are assumptions, not measurements.
"""

from __future__ import annotations

import numpy as np

from . import catalog
from .sizing import SINGLE_GROUPS

ETA_GEAR = 0.75
ETA_DRIVE = 0.92
IDLE_QDD_W = 1.4  # per axis (PR #662)
IDLE_SERVO_W = 0.5
SERVO_V = 12.0
ETA_DCDC = 0.88
USABLE = 0.75
USABLE_RANGE = (0.65, 0.85)
TARGET_MIN = 20.0
MIN_MIN = 15.0
FALLBACK_KT_PER_RATED = 1.0 / 6.0  # if Kt unknown: assume rated torque at 6 A rms (flagged)
FALLBACK_R = 0.25

COMPUTE_BUDGET = {  # W at the compute rail, before the DC-DC, by selected compute part
    # Round one: PR #662's Jetson allowance, kept unchanged so round-one results reproduce (A13).
    "nvidia_orin_nano_super_devkit": {"jetson_module": 25.0, "carrier_storage_cooling": 6.0},
    # Round two (P14): the compute survey's typical board-level draw for the Pi 5 + AI HAT+ (Hailo-8L)
    # set, plus an allowance for the USB SSD the survey left out (assumption A14).
    "rpi5_8gb_ai_hat_plus_13tops": {"pi5_ai_hat_plus": 8.0, "usb_ssd": 1.5},
}
AUX_FIXED = {  # W (PR #662 auxiliary budget, kept as the baseline)
    "mcu_comms": 4.0,
}


def drive_power(act: catalog.Actuator, tau: np.ndarray, w: np.ndarray) -> dict:
    """Mean electrical input of one drive over a trajectory, with its parts."""
    tau = np.asarray(tau, float)
    w = np.asarray(w, float)
    flags = []
    if act.box is not None:
        kt = act.kt or 1.5
        p = SERVO_V * np.abs(tau) / kt + IDLE_SERVO_W
        return dict(mean=float(p.mean()), peak=float(p.max()), copper=0.0, mech=float(np.mean(np.clip(tau * w, 0, None))),
                    flags=flags)
    kt = act.kt
    if kt is None:
        kt = act.rated_torque * FALLBACK_KT_PER_RATED * 6.0 / 6.0
        flags.append("Kt estimated")
    R = act.phase_resistance
    if R is None:
        R = FALLBACK_R
        flags.append("R estimated")
    current = np.abs(tau) / kt
    p_cu = 3.0 * current**2 * R
    p_mech = tau * w
    p_in = (np.clip(p_mech, 0, None) / ETA_GEAR + p_cu) / ETA_DRIVE + IDLE_QDD_W
    return dict(mean=float(p_in.mean()), peak=float(p_in.max()), copper=float(p_cu.mean()),
                mech=float(np.mean(np.clip(p_mech, 0, None))), current_rms=float(np.sqrt(np.mean(current**2))),
                current_peak=float(current.max()), flags=flags)


def aux_power(design) -> dict:
    sensors = 0.0
    for b in design.bodies.values():
        for it in b.items:
            if it.category == "sensors" and it.power_nom:
                sensors += it.power_nom
    loads = dict(COMPUTE_BUDGET[design.sel["compute"]], **AUX_FIXED, sensors=sensors)
    total_load = sum(loads.values())
    return dict(loads=loads, load_w=total_load, input_w=total_load / ETA_DCDC)


def walking_power(design, req, motor_series: dict) -> dict:
    """Mean electrical power for the nominal reference walk.  ``motor_series[group]`` = (tau, w) with
    both sides stacked; each group has two drives."""
    per = {}
    total = 0.0
    peak = 0.0
    flags = []
    for g, (tau, w) in motor_series.items():
        act = design.act(g)
        n_drives = 1 if g in SINGLE_GROUPS else 2  # the neck, the tail and the expressive head have one drive
        dp = drive_power(act, tau, w)
        per[g] = dict(actuator=act.id, drives=n_drives, mean_w=dp["mean"] * n_drives, copper_w=dp["copper"] * n_drives,
                      mech_w=dp["mech"] * n_drives, flags=dp["flags"])
        total += dp["mean"] * n_drives
        peak += dp["peak"] * n_drives
        flags += [f"{g}: {f}" for f in dp["flags"]]
    aux = aux_power(design)
    return dict(drives=per, drives_w=total, aux=aux, total_w=total + aux["input_w"], peak_drives_w=peak, flags=flags)


def runtime(pack: catalog.Part, total_w: float) -> dict:
    e = pack.capacity_wh
    mins = {f"{u:.2f}": 60.0 * e * u / total_w for u in (USABLE_RANGE[0], USABLE, USABLE_RANGE[1])}
    return dict(pack=pack.id, energy_wh=e, minutes=60.0 * e * USABLE / total_w, minutes_range=mins)


def select_battery(bus: float, total_w: float, peak_w: float) -> dict:
    """Lightest pack on the bus that gives >= 20 min at 75 % usable and can supply the peak."""
    packs = [p for p in catalog.parts().values() if p.category == "battery" and abs(p.voltage - (22.2 if bus <= 24 else 44.4)) < 2.0]
    ok = []
    for p in packs:
        mins = 60.0 * p.capacity_wh * USABLE / total_w
        cur_ok = True
        imax = getattr(p, "max_current", None)
        if imax:
            cur_ok = imax * p.voltage >= peak_w
        if mins >= TARGET_MIN and cur_ok:
            ok.append((p.mass, p.id, mins))
    if ok:
        ok.sort()
        return dict(pack=ok[0][1], minutes=ok[0][2], meets_target=True)
    best = max(packs, key=lambda p: p.capacity_wh)
    return dict(pack=best.id, minutes=60.0 * best.capacity_wh * USABLE / total_w, meets_target=False)
