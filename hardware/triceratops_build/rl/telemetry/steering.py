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

Measures (revised after the review of 23 Nov):
- ``tracking``: speed and turn-rate tracking judged on means over ``average_s`` (default 0.5 s, about one stride of
  a slow trot), after the first ``skip_s``: the per-step values mostly read the gait's own sway. The mean and 95th
  percentile of those window errors, and the share of windows within both tolerances. The per-step errors are kept
  under ``tracking_per_step``;
- ``events``: one row per command event (a new request) with a heading to follow, while walking or turning on the
  spot. A row is a TURN when the heading error at the request's start is at least ``turn_threshold_deg`` (default 20);
  its settle time is the time until the error came within tolerance and stayed there for ``hold_s``. An event that
  ends in a fall counts, as failed (``fell_during``), however short; other events shorter than ``settle_s`` are
  skipped;
- ``turning``: while the robot was asked to turn while walking, over ``average_s`` windows: the median turn radius
  (mean forward speed / |mean turn rate|) over the windows that turned the commanded way at ``min_turn_ratio`` or
  more of the commanded rate while moving at ``min_turn_speed`` or more (None if none did: a robot that stands has
  no radius), how many windows qualified, and the achieved-over-commanded turn-rate ratios of all turn windows;
- ``turn_in_place``: per turn on the spot, the net displacement over its duration (m/s; the sway of the steps is
  not drift), when the step info carries the base position (``position_keys``), and the median ratio of achieved to
  commanded turn rate;
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
                     skip_s: float = 1.0, average_s: float = 0.5, turn_threshold_deg: float = 20.0,
                     turn_rate_threshold: float = 0.1, min_turn_ratio: float = 0.5, min_turn_speed: float = 0.05,
                     speed_key: str = "forward_speed",
                     lateral_key: str = "lateral_speed", yaw_rate_key: str = "yaw_rate", time_key: str = "time_s",
                     position_keys: tuple[str, str] = ("base_x", "base_y")) -> dict:
    if not infos:
        return {"steps": 0}
    g = lambda k, default=np.nan: np.array([i.get(k, default) for i in infos], dtype=np.float64)  # noqa: E731
    t, v, vl, yr = g(time_key), g(speed_key), g(lateral_key, 0.0), g(yaw_rate_key)
    cv, cy = g("command_v_x"), g("command_yaw_rate")
    ev, ey = g("tracking_error_v"), g("tracking_error_yaw")
    active = np.array([bool(i.get("heading_active")) for i in infos])
    tip = np.array([bool(i.get("turn_in_place")) for i in infos])
    follow = active | tip  # a heading to follow: walking with one, or turning on the spot
    desired, actual = g("desired_heading"), g("actual_heading")
    px, py = g(position_keys[0]), g(position_keys[1])
    event = np.array([int(i.get("command_event_id", 0)) for i in infos])
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    fell = bool(infos[-1].get("termination"))
    out: dict = {"steps": len(infos), "dt_s": dt, "fell": fell, "termination": infos[-1].get("termination") or "",
                 "tolerances": {"velocity_mps": velocity_tolerance, "yaw_rate_radps": yaw_rate_tolerance,
                                "heading_rad": heading_tolerance}}
    # tracking over windows of average_s
    w = t >= skip_s
    k = max(1, int(round(average_s / dt))) if dt else 1
    idx = np.flatnonzero(w)
    wins = [idx[i:i + k] for i in range(0, len(idx) - k + 1, k)]
    e_v = [math.hypot(v[s].mean() - cv[s].mean(), vl[s].mean()) for s in wins]
    e_y = [abs(yr[s].mean() - cy[s].mean()) for s in wins]
    ok = [a <= velocity_tolerance and b <= yaw_rate_tolerance for a, b in zip(e_v, e_y)]
    out["tracking"] = {"average_s": k * dt, "windows": len(wins), "speed_error_mps": _stats(e_v),
                       "yaw_rate_error_radps": _stats(e_y), "share_in_tolerance": float(np.mean(ok)) if ok else None}
    out["tracking_per_step"] = {"speed_error_mps": _stats(ev[w]), "yaw_rate_error_radps": _stats(ey[w])}
    # events
    rows = []
    starts = np.flatnonzero(np.r_[True, event[1:] != event[:-1]])
    ends = np.r_[starts[1:], len(infos)]
    prev_desired = None
    for s, e in zip(starts, ends):
        duration = (e - s) * dt
        change = None if prev_desired is None else float(_wrap(desired[s] - prev_desired))
        prev_desired = desired[s]
        last = e == len(infos)
        fell_during = fell and last
        if not follow[s:e].any() or (duration < settle_s and not fell_during):
            continue
        err = np.abs(_wrap(desired[s:e] - actual[s:e]))
        need = max(1, int(round(hold_s / dt))) if dt else 1
        inside = err <= heading_tolerance
        settle = None
        if not fell_during:
            for j in range(len(inside) - need + 1):
                if inside[j:j + need].all():
                    settle = j * dt
                    break
        at = min(int(round(settle_s / dt)) - 1 if dt else len(err) - 1, len(err) - 1)
        rows.append({"event_id": int(event[s]), "start_s": float(t[s]), "duration_s": duration,
                     "requested_change_rad": change, "initial_error_rad": float(err[0]),
                     "is_turn": bool(err[0] >= math.radians(turn_threshold_deg)),
                     "requested_speed_mps": float(infos[s].get("desired_speed", 0.0)),
                     "turn_in_place": bool(tip[s:e].any()), "fell_during": fell_during,
                     "heading_error_at_settle_rad": None if fell_during else float(err[at]),
                     "settle_time_s": settle})
    out["events"] = rows
    turns = [r for r in rows if r["is_turn"]]
    out["events_summary"] = {
        "n": len(rows), "n_turns": len(turns), "turns_failed_by_fall": sum(r["fell_during"] for r in turns),
        "heading_error_at_settle_deg": _stats([math.degrees(r["heading_error_at_settle_rad"]) for r in rows
                                               if r["heading_error_at_settle_rad"] is not None]),
        "settled_share_of_turns": (sum(r["settle_time_s"] is not None for r in turns) / len(turns)) if turns else None,
        "settle_time_s_of_turns": _stats([r["settle_time_s"] for r in turns])}
    # turning while walking: per window, the radius of a window that really turned and moved
    walk_turn = w & ~tip & (np.abs(cy) > turn_rate_threshold) & (cv > 0.05)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = yr / cy
    radii, win_ratio = [], []
    for s in wins:
        if not walk_turn[s].all():
            continue
        r = float(yr[s].mean() / cy[s].mean())
        win_ratio.append(r)
        if r >= min_turn_ratio and v[s].mean() >= min_turn_speed:
            radii.append(float(v[s].mean() / abs(yr[s].mean())))
    out["turning"] = {"steps": int(walk_turn.sum()), "windows": len(win_ratio), "windows_turning": len(radii),
                      "radius_m_median": float(np.median(radii)) if radii else None,
                      "achieved_over_commanded": _stats(win_ratio)}
    # turning on the spot: net displacement per turn
    spins = []
    tip_w = w & tip
    edges = np.flatnonzero(np.diff(np.r_[0, tip_w.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 2 and np.isfinite(px[a]) and np.isfinite(px[b - 1]):
            spins.append(math.hypot(px[b - 1] - px[a], py[b - 1] - py[a]) / ((b - a) * dt))
    on_spot = tip_w & (np.abs(cy) > turn_rate_threshold)
    out["turn_in_place"] = {"steps": int(on_spot.sum()), "turns": len(spins),
                            "net_drift_mps": _stats(spins) if spins else None,
                            "achieved_over_commanded": _stats(ratio[on_spot])}
    stand = w & ~follow & (np.abs(cy) <= 1e-9) & (np.abs(cv) <= 1e-9)
    out["stops"] = {"steps": int(stand.sum()), "speed_mps": _stats(np.hypot(v, vl)[stand]),
                    "yaw_rate_radps": _stats(np.abs(yr)[stand])}
    return out


def summarise(episodes: list[dict]) -> dict:
    """Across episodes: falls, the mean of the per-episode window tracking, pooled turn figures."""
    def mean_of(path):
        vals = []
        for e in episodes:
            x = e
            for p in path:
                x = (x or {}).get(p) if isinstance(x, dict) else None
            if isinstance(x, (int, float)) and math.isfinite(x):
                vals.append(float(x))
        return float(np.mean(vals)) if vals else None

    rows = [r for e in episodes for r in e.get("events", [])]
    turns = [r for r in rows if r["is_turn"]]
    return {"episodes": len(episodes), "falls": int(sum(bool(e.get("fell")) for e in episodes)),
            "speed_error_mps_mean": mean_of(("tracking", "speed_error_mps", "mean")),
            "yaw_rate_error_radps_mean": mean_of(("tracking", "yaw_rate_error_radps", "mean")),
            "share_in_tolerance_mean": mean_of(("tracking", "share_in_tolerance")),
            "heading_error_at_settle_deg_median": _stats([math.degrees(r["heading_error_at_settle_rad"]) for r in rows
                                                          if r["heading_error_at_settle_rad"] is not None])["median"],
            "turns": len(turns), "turns_failed_by_fall": sum(r["fell_during"] for r in turns),
            "turns_settled_share": (sum(r["settle_time_s"] is not None for r in turns) / len(turns)) if turns else None,
            "turn_radius_m_median": mean_of(("turning", "radius_m_median")),
            "turn_in_place_net_drift_mps_median": mean_of(("turn_in_place", "net_drift_mps", "median")),
            "turn_in_place_rate_ratio_median": mean_of(("turn_in_place", "achieved_over_commanded", "median"))}
