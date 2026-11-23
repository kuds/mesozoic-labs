"""Direction-following measures of one episode, from its per-step ``info`` (robot-agnostic; decisions D20, D21).

The input is the list of step ``info`` dicts an env returns under a steering command mode, using the repository's
shared direction-command field names (``environments/shared/direction_commands.py``): ``command_v_x``,
``command_yaw_rate``, ``desired_heading``, ``desired_speed``, ``actual_heading``, ``heading_active``,
``command_event_id``, ``tracking_error_v``, ``tracking_error_yaw``, plus the env's own speed, turn-rate and time
keys, named by the caller (the 60 cm build's defaults: ``forward_speed``, ``lateral_speed``, ``yaw_rate``,
``time_s``). Optional: ``turn_in_place`` (a stop that turns on the spot), ``termination``.

The tolerances are the caller's: the shared module's defaults (0.2 m/s, 0.15 rad/s, 15 deg) suit 1 m/s animals; a
0.25 m/s robot passes tighter ones. Nothing here judges: the measures are reported, and a pre-registration says
which, if any, decide.

Measures:
- ``tracking``: mean and 95th percentile of the speed and turn-rate tracking errors over the steps after the first
  ``skip_s`` (the start-up), and the share of those steps within both tolerances (the shared "command in
  tolerance" test);
- ``events``: one row per command event (a new request) lasting at least ``settle_s``: the heading change it asked
  for, the heading error ``settle_s`` after it started, and the time until the heading came within tolerance and
  stayed there for ``hold_s`` (None if it never did within the event);
- ``turning``: while the robot was asked to turn at more than ``turn_threshold`` rad/s while walking, the median turn
  radius (|forward speed| / |turn rate|) and the median ratio of achieved to commanded turn rate;
- ``turn_in_place``: while turning on the spot, the median translational speed (drift) and the median ratio of
  achieved to commanded turn rate;
- ``stops``: while asked to stand (no heading, no turn), the median speed and turn rate.
"""

from __future__ import annotations

import math

import numpy as np


def _wrap(a):
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def _stats(x) -> dict:
    x = np.asarray([v for v in x if v is not None and math.isfinite(v)], dtype=np.float64)
    if not x.size:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}


def episode_measures(infos: list[dict], *, velocity_tolerance: float = 0.2, yaw_rate_tolerance: float = 0.15,
                     heading_tolerance: float = math.pi / 12.0, settle_s: float = 3.0, hold_s: float = 0.5,
                     skip_s: float = 1.0, turn_threshold: float = 0.1, speed_key: str = "forward_speed",
                     lateral_key: str = "lateral_speed", yaw_rate_key: str = "yaw_rate",
                     time_key: str = "time_s") -> dict:
    if not infos:
        return {"steps": 0}
    g = lambda k, default=np.nan: np.array([i.get(k, default) for i in infos], dtype=np.float64)  # noqa: E731
    t, v, vl, yr = g(time_key), g(speed_key), g(lateral_key, 0.0), g(yaw_rate_key)
    cv, cy = g("command_v_x"), g("command_yaw_rate")
    ev, ey = g("tracking_error_v"), g("tracking_error_yaw")
    active = np.array([bool(i.get("heading_active")) for i in infos])
    tip = np.array([bool(i.get("turn_in_place")) for i in infos])
    desired, actual = g("desired_heading"), g("actual_heading")
    event = np.array([int(i.get("command_event_id", 0)) for i in infos])
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    w = t >= skip_s
    ok = (ev <= velocity_tolerance) & (ey <= yaw_rate_tolerance)
    out: dict = {"steps": len(infos), "dt_s": dt, "fell": bool(infos[-1].get("termination")),
                 "termination": infos[-1].get("termination") or "",
                 "tolerances": {"velocity_mps": velocity_tolerance, "yaw_rate_radps": yaw_rate_tolerance,
                                "heading_rad": heading_tolerance},
                 "tracking": {"speed_error_mps": _stats(ev[w]), "yaw_rate_error_radps": _stats(ey[w]),
                              "share_in_tolerance": float(ok[w].mean()) if w.any() else None}}
    # events: each new request
    rows = []
    starts = np.flatnonzero(np.r_[True, event[1:] != event[:-1]])
    ends = np.r_[starts[1:], len(infos)]
    prev_desired = None
    for s, e in zip(starts, ends):
        duration = (e - s) * dt
        change = None if prev_desired is None else float(_wrap(desired[s] - prev_desired))
        prev_desired = desired[s]
        if duration < settle_s or not active[s:e].any():
            continue
        err = np.abs(_wrap(desired[s:e] - actual[s:e]))
        k = s + int(round(settle_s / dt)) - 1 if dt else e - 1
        need = max(1, int(round(hold_s / dt))) if dt else 1
        inside = err <= heading_tolerance
        settle = None
        for j in range(len(inside) - need + 1):
            if inside[j:j + need].all():
                settle = j * dt
                break
        rows.append({"event_id": int(event[s]), "start_s": float(t[s]), "duration_s": duration,
                     "requested_change_rad": change, "requested_speed_mps": float(infos[s].get("desired_speed", 0.0)),
                     "turn_in_place": bool(tip[s:e].any()),
                     "heading_error_at_settle_rad": float(err[min(k, e - 1) - s]), "settle_time_s": settle})
    out["events"] = rows
    big = [r for r in rows if r["requested_change_rad"] is not None and abs(r["requested_change_rad"]) >= math.radians(20)]
    out["events_summary"] = {
        "n": len(rows), "n_turns_over_20deg": len(big),
        "heading_error_at_settle_deg": _stats([math.degrees(r["heading_error_at_settle_rad"]) for r in rows]),
        "settled_share_of_turns": (sum(r["settle_time_s"] is not None for r in big) / len(big)) if big else None,
        "settle_time_s_of_turns": _stats([r["settle_time_s"] for r in big])}
    # turning while walking; turning on the spot; stops
    walk_turn = w & ~tip & (np.abs(cy) > turn_threshold) & (cv > 0.05)
    with np.errstate(invalid="ignore", divide="ignore"):
        radius = np.abs(v) / np.abs(yr)
        ratio = yr / cy
    out["turning"] = {"steps": int(walk_turn.sum()), "radius_m": _stats(radius[walk_turn & (np.abs(yr) > 1e-3)]),
                      "achieved_over_commanded": _stats(ratio[walk_turn])}
    on_spot = w & tip & (np.abs(cy) > turn_threshold)
    out["turn_in_place"] = {"steps": int(on_spot.sum()), "drift_speed_mps": _stats(np.hypot(v, vl)[on_spot]),
                            "achieved_over_commanded": _stats(ratio[on_spot])}
    stand = w & ~active & (np.abs(cy) <= 1e-9) & (np.abs(cv) <= 1e-9)
    out["stops"] = {"steps": int(stand.sum()), "speed_mps": _stats(np.hypot(v, vl)[stand]),
                    "yaw_rate_radps": _stats(np.abs(yr)[stand])}
    return out


def summarise(episodes: list[dict]) -> dict:
    """Across episodes: falls, the mean of the per-episode means and shares, pooled event figures."""
    def mean_of(path):
        vals = []
        for e in episodes:
            x = e
            for p in path:
                x = (x or {}).get(p)
            if isinstance(x, (int, float)) and x is not None and math.isfinite(x):
                vals.append(float(x))
        return float(np.mean(vals)) if vals else None

    rows = [r for e in episodes for r in e.get("events", [])]
    big = [r for r in rows if r["requested_change_rad"] is not None and abs(r["requested_change_rad"]) >= math.radians(20)]
    return {"episodes": len(episodes), "falls": int(sum(bool(e.get("fell")) for e in episodes)),
            "speed_error_mps_mean": mean_of(("tracking", "speed_error_mps", "mean")),
            "yaw_rate_error_radps_mean": mean_of(("tracking", "yaw_rate_error_radps", "mean")),
            "share_in_tolerance_mean": mean_of(("tracking", "share_in_tolerance")),
            "heading_error_at_settle_deg_median": _stats([math.degrees(r["heading_error_at_settle_rad"])
                                                          for r in rows])["median"],
            "turns_over_20deg": len(big),
            "turns_settled_share": (sum(r["settle_time_s"] is not None for r in big) / len(big)) if big else None,
            "turn_radius_m_median": mean_of(("turning", "radius_m", "median")),
            "turn_in_place_drift_mps_median": mean_of(("turn_in_place", "drift_speed_mps", "median")),
            "turn_in_place_rate_ratio_median": mean_of(("turn_in_place", "achieved_over_commanded", "median"))}
