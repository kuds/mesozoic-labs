"""Drive winding temperature over a long demo: a first-order thermal model per drive (robot-agnostic; decision D21).

ESTIMATES, not measurements. Each drive is one thermal node, its winding, against the ambient air:

    C dT/dt = P_cu(t, T) - (T - T_amb) / R_th,     P_cu = 3 R(T) (tau_motor / Kt)^2,   R(T) = R_0 (1 + alpha (T - T_0))

- ``tau_motor`` is the torque the motor makes at the drive's output (the joint torque, or the joint torque over a
  linkage's efficiency); ``Kt`` (N m per A rms, output side) and ``R_0`` (per-phase, Y-equivalent, at ``T_0``) are the
  caller's catalog figures, as the gait report's copper loss uses them.
- **Calibration** from the maker's temperature-rise figure: a drive holding its ``thermal_eq_nm`` (the maker's
  thermal-equilibrium torque, not its rated torque) reaches ``ref_temp_c`` (100 C) after ``ref_minutes`` (90 min)
  from ``ambient_c`` (25 C) in open air. With the time constant ``tau = R_th C`` assumed, that fixes ``R_th`` (solved
  with the temperature-dependent resistance), and ``C = tau / R_th``. Each time constant gets its own ``R_th``, so
  the steady state at ``thermal_eq_nm`` differs between them (``steady_at_thermal_eq_c``): the longer the time
  constant, the further above 100 C, since 90 min is then further from steady.
- **The time constant** is ``R_th C`` at the reference resistance; the winding's resistance rises as it warms, so
  the response under load is somewhat slower than the label. It is not published as such, so it is a range. The
  maker's overload figures fix it roughly; fitted with this model, calibrated at the 83 rpm equilibrium torque the
  drive table uses (23 Nov 2026; two independent fits agree within 0.1 min):
  - X4-36-L at 10.5 N m reaches 100 C in 5.4-12.4 min: 6.5-15.3 min; at 21 N m it lasts 44 s: 4.6-4.7 min
    (``hardware/trex_form_study/stage2/design/servo_model/reader2_hardware_facts.md``, the maker's figures);
  - X6-50-L at 20 N m reaches 100 C in 4 min 9 s: 8.1 min (``hardware/trex_form_study/designlab/data/actuators.json``).
  Calibrated at the X4-36-L's 0 rpm equilibrium torque (8.5 N m) instead, the X4 fits are 4.4-10.0 and 3.5 min.
  Hence the default range 4, 8 and 15 min (``TIME_CONSTANTS_S``); every result is reported for each. Over a long
  demo the result barely depends on it.
- **The room** (``room_c``, added 27 Nov 2026 for decision D48: heat rules in degrees assume a 35 C room). ``R_th``
  and ``C`` belong to the drive, so they are fitted once, at the rating test's room (``ambient_c``, 25 C; the
  maker's X4-36-L and X6-50-L tables start at about 26-27 C), and a warmer room only raises the air the node sheds
  heat to: ``calibrate(tc, room_c=35.0)``. Left out (None), the room is ``ambient_c`` and nothing changes. Setting
  ``ambient_c`` itself to the room instead refits the drive as if the maker's test had started in that room: that
  needs a lower ``R_th`` (a drive that cools better in a hotter room), so it is only an optimistic bound
  (``rl/analysis/thermal_ambient.py``).
- Not modelled: the housing as a second node, the drive electronics' own losses, iron losses, airflow inside a
  closed shell (worse than open air), and the controller's own thermal limits. Calibrate against a bench
  temperature-rise test when one exists (the bench plan).

The inputs are per-drive loss histories, a piecewise-constant mean squared torque per segment of a demo, which
``ThermalNode.run`` integrates in 1 s steps (exponential within a step, the resistance updated each step: within
about 0.01 K of the exact solution over 90 min).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

ALPHA_CU = 0.00393  # 1/K, copper's resistance temperature coefficient near 20 C
RATING_AMBIENT_C = 25.0  # the room of the maker's temperature-rise test, as the calibration takes it
TIME_CONSTANTS_S = (240.0, 480.0, 900.0)  # 4, 8 and 15 min: the maker-data fits above


def tc_key(time_constant_s: float) -> str:
    """The result key of a time constant: exact, so two constants never share one."""
    return f"tau_{float(time_constant_s):g}s"


@dataclass(frozen=True)
class DriveThermal:
    name: str
    kt_nm_per_a: float  # output side, per A rms
    r_phase_ohm: float  # per phase (Y-equivalent), at r_ref_c
    thermal_eq_nm: float  # the maker's temperature-rise torque at the output
    r_ref_c: float = 25.0
    ambient_c: float = RATING_AMBIENT_C  # the rating test's room, which the calibration fits (not the demo's room)
    ref_temp_c: float = 100.0
    ref_minutes: float = 90.0
    alpha: float = ALPHA_CU

    def copper_loss(self, tau_motor_rms: float, temp_c: float) -> float:
        """Copper loss (W) at an RMS motor torque (N m, output side) and winding temperature."""
        r = self.r_phase_ohm * (1.0 + self.alpha * (temp_c - self.r_ref_c))
        return 3.0 * r * (tau_motor_rms / self.kt_nm_per_a) ** 2

    def calibrate(self, time_constant_s: float, room_c: float | None = None) -> "ThermalNode":
        """R_th and C such that holding thermal_eq_nm reaches ref_temp_c after ref_minutes from ``ambient_c`` (the
        rating test's room). ``room_c`` is the room the returned node runs in (None: ``ambient_c``); it does not enter
        the fit, since R_th and C belong to the drive."""
        t_end = self.ref_minutes * 60.0
        target = self.ref_temp_c

        def end_temp(r_th: float) -> float:
            node = ThermalNode(self, r_th, time_constant_s / r_th)
            return node.run([(t_end, self.thermal_eq_nm ** 2)], start_c=self.ambient_c)[-1][1]

        lo, hi = 1e-3, 1e3  # K/W; the end temperature rises with R_th
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if end_temp(mid) < target:
                lo = mid
            else:
                hi = mid
        r_th = math.sqrt(lo * hi)
        return ThermalNode(self, r_th, time_constant_s / r_th, room_c=room_c)


@dataclass(frozen=True)
class ThermalNode:
    drive: DriveThermal
    r_th_k_per_w: float
    c_j_per_k: float
    step_s: float = field(default=1.0)
    room_c: float | None = None  # the air the node sheds heat to; None: the drive's ambient_c (the calibration's)

    @property
    def time_constant_s(self) -> float:
        return self.r_th_k_per_w * self.c_j_per_k

    @property
    def air_c(self) -> float:
        """The room this node runs in (C)."""
        return self.drive.ambient_c if self.room_c is None else float(self.room_c)

    def run(self, segments, start_c: float | None = None):
        """Integrate over ``segments``: (duration s, mean squared motor torque in N^2 m^2) each. Returns [(t, T)] at
        every step_s and segment end. The squared torque, not the torque, carries a segment's copper loss: a gait's
        torque swings within a step, and its mean square is what heats."""
        a = self.air_c
        temp = a if start_c is None else float(start_c)
        t, out = 0.0, [(0.0, temp)]
        for duration, tau_sq in segments:
            left = float(duration)
            while left > 1e-12:
                h = min(self.step_s, left)
                # the resistance at the step's start temperature: losses vary slowly against step_s << tau
                p = self.drive.copper_loss(math.sqrt(max(0.0, tau_sq)), temp)
                ss = a + p * self.r_th_k_per_w
                temp = ss + (temp - ss) * math.exp(-h / self.time_constant_s)
                t += h
                left -= h
                out.append((t, temp))
        return out

    def steady_state_c(self, tau_sq: float) -> float | None:
        """The steady winding temperature at a constant mean squared torque in this node's room, or None if the
        resistance's rise makes it run away (no steady state)."""
        d, a = self.drive, self.air_c
        k = 3.0 * d.r_phase_ohm * tau_sq / d.kt_nm_per_a ** 2 * self.r_th_k_per_w  # K per (1 + alpha dT) unit
        denom = 1.0 - k * d.alpha
        if denom <= 0:
            return None
        return a + k * (1.0 + d.alpha * (a - d.r_ref_c)) / denom


def demo_temperatures(drives: dict[str, DriveThermal], segments: dict[str, list[tuple[float, float]]],
                      time_constants_s=TIME_CONSTANTS_S, limit_c: float = 100.0,
                      cooldown_to_c: float = 40.0, room_c: float | None = None) -> dict:
    """Per drive and time constant (keys ``tc_key``), starting at the room's temperature (``room_c``; None: each
    drive's ``ambient_c``, as before 27 Nov 2026), with each drive calibrated at its ``ambient_c``:
    - ``peak_c``, ``end_c``: over the demo's segments; ``minutes_to_limit``: when it first reaches ``limit_c`` (None
      if never); ``cooldown_minutes``: from the end to ``cooldown_to_c`` at zero load (None if not within six time
      constants, as always in a room at or above ``cooldown_to_c``);
    - ``continuous_c``: repeating the demo without a break, the middle of the settled temperature cycle (the steady
      state at the demo's time-weighted mean squared torque; segments short against the time constant); the cycle's
      peaks are higher by the swing within a loop. None if it runs away (no steady state);
    - ``steady_at_thermal_eq_c``: the calibration's own steady state at ``thermal_eq_nm``, in the calibration's room
      (``ambient_c``) whatever ``room_c`` is.
    ``segments[name]``: that drive's (duration s, mean squared motor torque) list."""
    keys = [tc_key(tc) for tc in time_constants_s]
    if len(set(keys)) != len(keys):
        raise ValueError("time constants must differ")
    out = {}
    for name, d in drives.items():
        seg = segments[name]
        total = sum(s for s, _ in seg)
        mean_sq = sum(s * q for s, q in seg) / total if total > 0 else 0.0
        per = {}
        for key, tc in zip(keys, time_constants_s):
            node = d.calibrate(tc, room_c=room_c)
            trace = node.run(seg)
            peak = max(T for _, T in trace)
            hit = next((t for t, T in trace if T >= limit_c), None)
            end = trace[-1][1]
            cool = None
            if end > cooldown_to_c:
                c = node.run([(6 * tc, 0.0)], start_c=end)
                cool = next((t for t, T in c if T <= cooldown_to_c), None)
            per[key] = {"time_constant_s": float(tc), "r_th_k_per_w": node.r_th_k_per_w, "c_j_per_k": node.c_j_per_k,
                        "peak_c": peak, "end_c": end, "minutes_to_limit": None if hit is None else hit / 60.0,
                        "cooldown_minutes": None if cool is None else cool / 60.0,
                        "continuous_c": node.steady_state_c(mean_sq),
                        "steady_at_thermal_eq_c": replace(node, room_c=None).steady_state_c(d.thermal_eq_nm ** 2)}
        out[name] = per
    return out
