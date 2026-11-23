"""Drive winding temperature over a long demo: a first-order thermal model per drive (robot-agnostic; decision D21).

ESTIMATES, not measurements. Each drive is one thermal node, its winding, against the ambient air:

    C dT/dt = P_cu(t, T) - (T - T_amb) / R_th,     P_cu = 3 R(T) (tau_motor / Kt)^2,   R(T) = R_0 (1 + alpha (T - T_0))

- ``tau_motor`` is the torque the motor makes at the drive's output (the joint torque, or the joint torque over a
  linkage's efficiency); ``Kt`` (N m per A rms, output side) and ``R_0`` (per-phase, Y-equivalent, at ``T_0``) are the
  caller's catalog figures, as the gait report's copper loss uses them.
- **Calibration** from the maker's temperature-rise figure: a drive holding its ``thermal_eq_nm`` reaches
  ``ref_temp_c`` (100 C) after ``ref_minutes`` (90 min) from ``ambient_c`` (25 C) in open air. With the time constant
  ``tau = R_th C`` assumed, that fixes ``R_th`` (solved with the temperature-dependent resistance), and ``C = tau /
  R_th``.
- **The time constant is not published** for these drives, so it is an assumption: ``time_constants_s`` gives a
  range (default 10, 20 and 40 min), and every result is reported for each. A longer time constant means slower
  heating and a slower cool-down, with the same steady state.
- Not modelled: the housing as a second node, the drive electronics' own losses, iron losses, airflow inside a
  museum shell (worse than open air), and the controller's own thermal limits. Calibrate against a bench
  temperature-rise test when one exists (the bench plan).

The inputs are per-drive loss histories, a piecewise-constant mean copper loss per segment of a demo, which
``simulate`` integrates exactly per segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ALPHA_CU = 0.00393  # 1/K, copper's resistance temperature coefficient near 20 C


@dataclass(frozen=True)
class DriveThermal:
    name: str
    kt_nm_per_a: float  # output side, per A rms
    r_phase_ohm: float  # per phase (Y-equivalent), at r_ref_c
    thermal_eq_nm: float  # the maker's temperature-rise torque at the output
    r_ref_c: float = 25.0
    ambient_c: float = 25.0
    ref_temp_c: float = 100.0
    ref_minutes: float = 90.0
    alpha: float = ALPHA_CU

    def copper_loss(self, tau_motor_rms: float, temp_c: float) -> float:
        """Copper loss (W) at an RMS motor torque (N m, output side) and winding temperature."""
        r = self.r_phase_ohm * (1.0 + self.alpha * (temp_c - self.r_ref_c))
        return 3.0 * r * (tau_motor_rms / self.kt_nm_per_a) ** 2

    def calibrate(self, time_constant_s: float) -> "ThermalNode":
        """R_th and C such that holding thermal_eq_nm reaches ref_temp_c after ref_minutes from ambient."""
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
        return ThermalNode(self, r_th, time_constant_s / r_th)


@dataclass(frozen=True)
class ThermalNode:
    drive: DriveThermal
    r_th_k_per_w: float
    c_j_per_k: float
    step_s: float = field(default=1.0)

    @property
    def time_constant_s(self) -> float:
        return self.r_th_k_per_w * self.c_j_per_k

    def run(self, segments, start_c: float | None = None):
        """Integrate over ``segments``: (duration s, mean squared motor torque in N^2 m^2) each. Returns [(t, T)] at
        every step_s and segment end. The squared torque, not the torque, carries a segment's copper loss: a gait's
        torque swings within a step, and its mean square is what heats."""
        temp = self.drive.ambient_c if start_c is None else float(start_c)
        t, out = 0.0, [(0.0, temp)]
        a = self.drive.ambient_c
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
        """The steady winding temperature at a constant mean squared torque, or None if the resistance's rise
        makes it run away (no steady state)."""
        d = self.drive
        k = 3.0 * d.r_phase_ohm * tau_sq / d.kt_nm_per_a ** 2 * self.r_th_k_per_w  # K per (1 + alpha dT) unit
        denom = 1.0 - k * d.alpha
        if denom <= 0:
            return None
        return d.ambient_c + k * (1.0 + d.alpha * (d.ambient_c - d.r_ref_c)) / denom


def demo_temperatures(drives: dict[str, DriveThermal], segments: dict[str, list[tuple[float, float]]],
                      time_constants_s=(600.0, 1200.0, 2400.0), limit_c: float = 100.0,
                      cooldown_to_c: float = 40.0) -> dict:
    """Per drive and time constant: the peak temperature over the demo's segments, the time it first reaches
    ``limit_c`` (None if never), the end temperature, and the cool-down time from the end to ``cooldown_to_c`` at
    zero load. ``segments[name]``: that drive's (duration s, mean squared motor torque) list."""
    out = {}
    for name, d in drives.items():
        per = {}
        for tc in time_constants_s:
            node = d.calibrate(tc)
            trace = node.run(segments[name])
            peak = max(T for _, T in trace)
            hit = next((t for t, T in trace if T >= limit_c), None)
            end = trace[-1][1]
            cool = None
            if end > cooldown_to_c:
                c = node.run([(6 * tc, 0.0)], start_c=end)
                cool = next((t for t, T in c if T <= cooldown_to_c), None)
            per[f"{tc / 60:.0f}min"] = {"r_th_k_per_w": node.r_th_k_per_w, "c_j_per_k": node.c_j_per_k,
                                        "peak_c": peak, "end_c": end, "minutes_to_limit": None if hit is None
                                        else hit / 60.0, "cooldown_minutes": None if cool is None else cool / 60.0}
        out[name] = per
    return out
