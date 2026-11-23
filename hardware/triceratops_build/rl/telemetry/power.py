"""Battery and bus over a demo: runtime, currents, and braking energy (robot-agnostic; decision D21). ESTIMATES.

Inputs per demo segment, measured in simulation by the caller: the mean electrical draw of the drives (mechanical
power plus copper loss, on a shared bus where one drive's braking feeds another's motoring, so the net draw each
physics step is clipped at zero), the peak of that net draw, and the braking surplus (the net power flowing back to
the bus when the drives together brake). The pack is the caller's: capacity (Wh), the share usable, nominal and
low-charge voltages, and a constant "hotel" load (computers, sensors, servos at rest).

- Runtime: usable energy / (mean drive draw over the demo's mix + hotel load).
- Pack current: mean and peak, at the nominal and at the low-charge voltage (the peak at low charge sizes the fuse,
  wiring and the battery's discharge rating).
- Braking: the peak surplus power, and the share of time with a surplus. A pack that cannot take charge current (a
  full pack, or a BMS that blocks charging) leaves that power to a brake resistor or the drives' own dissipation.
Not modelled: converter and drive-electronics losses (other than the copper loss), the pack's internal resistance,
temperature and ageing.
"""

from __future__ import annotations


def demo_battery(segments: list[dict], *, pack_wh: float, usable_share: float, v_nominal: float, v_low: float,
                 hotel_w: float) -> dict:
    """``segments``: dicts with ``seconds`` (its share of the demo), ``mean_draw_w`` (drives, net, >= 0),
    ``peak_draw_w``, ``peak_surplus_w`` (>= 0), ``surplus_share`` (0-1). Returns the demo's mean draw, runtime and
    currents."""
    total = sum(s["seconds"] for s in segments)
    if total <= 0:
        raise ValueError("the demo needs a positive duration")
    mean_drives = sum(s["mean_draw_w"] * s["seconds"] for s in segments) / total
    mean_total = mean_drives + hotel_w
    peak = max(s["peak_draw_w"] for s in segments) + hotel_w
    usable_wh = pack_wh * usable_share
    return {
        "mean_drive_draw_w": mean_drives, "hotel_w": hotel_w, "mean_total_w": mean_total,
        "runtime_minutes": 60.0 * usable_wh / mean_total, "usable_wh": usable_wh,
        "current_mean_a": {"nominal_v": mean_total / v_nominal, "low_v": mean_total / v_low},
        "current_peak_a": {"nominal_v": peak / v_nominal, "low_v": peak / v_low},
        "braking_peak_surplus_w": max(s["peak_surplus_w"] for s in segments),
        "braking_surplus_share": sum(s["surplus_share"] * s["seconds"] for s in segments) / total,
        "pack": {"wh": pack_wh, "usable_share": usable_share, "v_nominal": v_nominal, "v_low": v_low},
    }
