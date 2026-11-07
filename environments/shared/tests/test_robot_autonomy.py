"""Safety/identity invariants for the onboard command boundary, without robot state."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("cv2")
from environments.shared.autonomy import CameraGeometry, OnboardFollower, RGBDFrame
from environments.shared.autonomy.controller import steer
from environments.shared.autonomy.perception import Detection


def person(left, red, box):
    feature = np.zeros(200)
    feature[[20 + red, 120 + red]] = 0.5
    return Detection(box, 2.8, left, feature, 0.94)


BLUE = person(0.30, 0, (25, 8, 45, 75))
BROWN = person(-0.50, 1, (55, 12, 75, 78))


def update(f, det, t):
    f.detector.detect = Mock(return_value=det)
    frame = RGBDFrame(np.zeros((90, 140, 3), np.uint8), np.full((90, 140), 4.0), (0.695, 0.708), t)
    f.heartbeat(t + 0.04)
    return f.update(frame, t + 0.04)


def selected():
    f = OnboardFollower()
    update(f, [BLUE, BROWN], 0)
    update(f, [BLUE, BROWN], 0.1)
    tracks = f.snapshot()["tracks"]
    assert len(tracks) == 2
    # A one-off external selection of the SECOND ID.
    f.select(tracks[1]["id"], 0.14)
    f.start(0.14)
    return f, tracks[1]["id"]


def test_occlusion_requires_three_consecutive_frames_and_original_signature():
    f, chosen = selected()
    assert update(f, [BROWN, BLUE], 0.2).yaw_rps < 0
    assert update(f, [BLUE], 0.3).state == "occluded"
    for t in (0.4, 0.5):
        c = update(f, [BLUE, BROWN], t)
        assert c.state == "confirming" and c.forward_mps == 0
    assert update(f, [BROWN, BLUE], 0.6).yaw_rps < 0
    assert f.snapshot()["selected"] == chosen
    for t in np.arange(0.7, 1.6, 0.1):
        c = update(f, [BLUE], float(t))
        assert c.forward_mps == c.yaw_rps == 0
    assert f.snapshot()["state"] == "lost_reselect"
    update(f, [BROWN, BLUE], 1.7)
    assert f.snapshot()["selected"] is None
    with pytest.raises(ValueError):
        f.start(1.74)


def test_identical_clothing_crossing_latches_reselection():
    f, _ = selected()
    assert update(f, [BROWN, person(-0.41, 1, (53, 12, 73, 78))], 0.2).state == "ambiguous_reselect"
    assert not f.snapshot()["armed"]


def test_watchdogs_stop_without_a_new_image_and_do_not_auto_restart():
    f, _ = selected()
    update(f, [BROWN, BLUE], 0.2)
    assert f.watchdog(0.56).state == "sensor_stale"
    f.heartbeat(0.6)
    assert not f.snapshot()["armed"]
    f, _ = selected()
    assert f.watchdog(1.0).state == "operator_disconnected"
    assert f.snapshot()["selected"] is None


def test_front_depth_barrier_and_near_blind_zone_stop():
    geom = CameraGeometry()
    h, w = 100, 160
    # Reproject only measured pixels on a vertical surface. Geometry is calculated
    # from range by the production path; labels and obstacle coordinates are absent.
    pts = geom.points(np.full((h, w), 0.84))
    assert steer(pts, (0.68, 0.68), BLUE, (h, w)).state == "blocked"
    clear = geom.points(np.full((h, w), 4.0))
    for tof in ((0.29, 0.68), (0.68, np.nan), (0.68,), (1.45, 0.68)):
        assert steer(clear, tof, BLUE, (h, w)).state == "near_range_stop"
    assert steer(geom.points(np.full((h, w), np.nan)), (0.68, 0.68), BLUE, (h, w)).state == "depth_unknown"
    close = person(0, 1, (10, 10, 30, 80))
    close.forward = 1.85
    assert steer(clear, (0.68, 0.68), close, (h, w)).forward_mps == 0


def test_reordered_sensor_packets_disarm_and_clear_owner():
    f, _ = selected()
    update(f, [BROWN], 0.2)
    assert update(f, [BROWN], 0.15).state == "sensor_stale"
    assert not f.snapshot()["armed"]


def test_local_operator_wire_protocol_and_spectator_heartbeat(monkeypatch):
    """Exercise actual HTTP handler over memory sockets, including JPEG payload."""
    import io
    import json
    from uuid import uuid4

    from environments.shared.autonomy import web

    class Socket:
        def __init__(self, raw):
            self.input = io.BytesIO(raw)
            self.output = bytearray()

        def makefile(self, *args, **kwargs):
            return self.input

        def sendall(self, data):
            self.output.extend(data)

    f, _ = selected()
    f.stop()
    monkeypatch.setattr(web.ThreadingHTTPServer, "server_bind", lambda self: None)
    monkeypatch.setattr(web.ThreadingHTTPServer, "server_activate", lambda self: None)
    clock = [0.14]
    ui = web.OperatorServer(f, port=0, clock=lambda: clock[0])
    owner, spectator = str(uuid4()), str(uuid4())
    ui.publish(np.full((90, 140, 3), 72, np.uint8))

    def request(path, data=None, host="localhost"):
        body = json.dumps(data or {}).encode()
        verb = "POST" if data else "GET"
        socket = Socket(
            f"{verb} {path} HTTP/1.0\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        ui.server.RequestHandlerClass(socket, ("127.0.0.1", 1), ui.server)
        head, body = bytes(socket.output).split(b"\r\n\r\n", 1)
        return int(head.split()[1]), body

    try:
        assert b"T. rex follow control" in request("/")[1]
        data = json.loads(request("/view")[1])
        assert data["image"] and len(data["tracks"]) == 2
        assert request("/view", host="foreign.invalid")[0] == 403
        assert request("/action", {"op": "select", "id": 2, "session": owner})[0] == 200
        assert request("/action", {"op": "start", "session": owner})[0] == 200
        assert request("/action", {"op": "target", "x": 1, "session": owner})[0] == 400
        # The operator closes; another browser's heartbeat cannot renew the lease.
        clock[0] = 1.0
        assert request("/action", {"op": "heartbeat", "session": spectator})[0] == 200
        assert f.watchdog(clock[0]).state == "operator_disconnected"
        assert request("/action", {"op": "start", "session": spectator})[0] == 400
    finally:
        ui.close()
