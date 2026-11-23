"""Battery and bus over a demo: runtime, currents, and braking energy (robot-agnostic; decision D21). ESTIMATES.

Inputs per demo segment, measured in simulation by the caller: the mean electrical draw of the drives (mechanical
power plus copper loss, on a shared bus where one drive's braking feeds another's motoring, so the net draw each
physics step is clipped at zero), the peak of that net draw, its mean square (for the RMS current), and the braking
surplus (the net power flowing back to the bus when the drives together brake): its peak, the share of time with
one, and the largest energy of one braking event. The pack is the caller's: capacity (Wh), the share usable, nominal
and low-charge voltages, and a constant "hotel" load (computers, sensors, servos at rest).

- Runtime: usable energy / (mean drive draw over the demo's mix + hotel load), and for each hotel load of an optional
  range.
- Pack current: mean, RMS and peak, at the nominal and at the low-charge voltage (the peak at low charge sizes the
  fuse, the wiring's short-term rating and the battery's discharge rating; the RMS current the wiring's heating).
  ``pack_currents`` gives the same for any power, such as a segment outside the demo's loop.
- Braking: the drives' peak surplus, and that peak less the hotel load, which takes braking power first. What
  neither the hotel load nor the other drives absorb raises the bus voltage, unless the pack can take charge current;
  where it cannot (ideal diodes in front of the packs, a full pack, or a BMS that blocks charging), a brake resistor
  or the drives' own dissipation must absorb it. The largest energy of one braking event (J) sizes that resistor.

Not modelled, so the runtime is optimistic and the currents low by these: servos and actuators the caller leaves out
of the drive draw; driver, converter, iron and gearbox losses (only copper loss is in the draw); the winding's
resistance rising as it warms (the caller's copper loss is usually at the catalog temperature; a winding at 50 C
loses about 10 % more than at 25 C); the pack's internal resistance, temperature and ageing.
"""

from __future__ import annotations

import math


def pack_currents(power_w: float, *, v_nominal: float, v_low: float) -> dict:
    """Pack current (A) for a power drawn from the pack (W), at the nominal and the low-charge voltage."""
    return {"nominal_v": power_w / v_nominal, "low_v": power_w / v_low}


def demo_battery(segments: list[dict], *, pack_wh: float, usable_share: float, v_nominal: float, v_low: float,
                 hotel_w: float, hotel_range_w: tuple[float, ...] = ()) -> dict:
    """``segments``: dicts with ``seconds`` (its share of the demo, > 0), ``mean_draw_w`` (drives, net, >= 0),
    ``peak_draw_w``, ``peak_surplus_w`` (>= 0), ``surplus_share`` (0-1), and optionally ``mean_sq_draw_w2`` (the
    mean of the squared net draw, for the RMS current) and ``brake_event_j_max``. Returns the demo's mean draw,
    runtime, currents and braking figures."""
    if not segments or any(s["seconds"] <= 0 for s in segments):
        raise ValueError("the demo needs segments, each of positive duration")
    total = sum(s["seconds"] for s in segments)
    mean_drives = sum(s["mean_draw_w"] * s["seconds"] for s in segments) / total
    mean_total = mean_drives + hotel_w
    peak = max(s["peak_draw_w"] for s in segments) + hotel_w
    usable_wh = pack_wh * usable_share
    out = {
        "mean_drive_draw_w": mean_drives, "hotel_w": hotel_w, "mean_total_w": mean_total,
        "runtime_minutes": 60.0 * usable_wh / mean_total, "usable_wh": usable_wh,
        "runtime_minutes_by_hotel_w": {f"{h:g}": 60.0 * usable_wh / (mean_drives + h) for h in hotel_range_w},
        "current_mean_a": pack_currents(mean_total, v_nominal=v_nominal, v_low=v_low),
        "current_peak_a": pack_currents(peak, v_nominal=v_nominal, v_low=v_low),
        "braking_peak_surplus_w": max(s["peak_surplus_w"] for s in segments),
        "braking_surplus_share": sum(s["surplus_share"] * s["seconds"] for s in segments) / total,
        "pack": {"wh": pack_wh, "usable_share": usable_share, "v_nominal": v_nominal, "v_low": v_low},
    }
    out["braking_peak_beyond_hotel_w"] = max(0.0, out["braking_peak_surplus_w"] - hotel_w)
    if all("mean_sq_draw_w2" in s for s in segments):
        # E[(p + h)^2] = E[p^2] + 2 h E[p] + h^2, time-weighted over the segments
        ms = sum((s["mean_sq_draw_w2"] + 2 * hotel_w * s["mean_draw_w"] + hotel_w ** 2) * s["seconds"]
                 for s in segments) / total
        out["current_rms_a"] = pack_currents(math.sqrt(ms), v_nominal=v_nominal, v_low=v_low)
    if all("brake_event_j_max" in s for s in segments):
        out["braking_event_j_max"] = max(s["brake_event_j_max"] for s in segments)
    return out
