"""Round C scratch experiment C3 (not a record): a mid-task start without editing control.py.  A Tracker subclass
starts at a reference time inside a periodic segment: Plant.set_state with the file's full qpos and qvel at that
frame, the estimator reset from that known state (Tracker.reset's own "frame" branch, reached by lending the track
another q0/qd0 for the call), tau_ref set after, and speedfit.servo_delay's queue refilled with the commands the
reference gave over the k steps before.  Zero action (R5's base) on 75-M-H-10, where the open-loop walk stays up.
Usage: exp_c3_midstart.py [body] [walk] [floor]"""
import os
import sys

os.environ["MUJOCO_GL"] = "disable"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from designlab.stage1 import control, plant, speedfit, tune  # noqa: E402
from designlab.stage1.refs import Reference  # noqa: E402

body = sys.argv[1] if len(sys.argv) > 1 else "75-M-H-10"
walk = sys.argv[2] if len(sys.argv) > 2 else "nominal"
floor = sys.argv[3] if len(sys.argv) > 3 else "hard"
run_dir = tune.RUN / body
RATE = tune.COMMAND_RATE


class R5Tracker(control.Tracker):
    def reset_at(self, t_ref, qpos, qvel, noise_seed, queue=None):
        tr = self.track
        q0, qd0 = tr.q0, tr.qd0
        tr.q0, tr.qd0 = np.asarray(qpos, float), np.asarray(qvel, float)
        try:
            super().reset("frame", noise_seed=noise_seed)  # control.py:1292-1296, 1329-1337
        finally:
            tr.q0, tr.qd0 = q0, qd0
        self.tau_ref = float(t_ref)
        if queue is not None:  # the k commands before t_ref, oldest first (the reference's own, at the commanded clock)
            p, k = self.p, queue.maxlen - 1
            real = type(p).command.__get__(p)
            fill = []
            for j in range(k, 0, -1):
                x = tr.joint(max(t_ref - j * self.dt * self.rate, 0.0))
                nj = self.nj
                qd_c = self.rate * x[nj: 2 * nj]
                ff = x[2 * nj: 3 * nj] + self.damp * qd_c + self.fric * np.tanh(qd_c / control.FRICTION_SMOOTH)
                if self.extra is not None:
                    ff = ff + self.extra @ (self.rate * self.rate * x[3 * nj:])
                real(x[:nj], qd_c, ff)
                fill.append(p.d.ctrl.copy())
            queue.clear()
            queue.extend(fill)
            p.d.ctrl[:] = fill[0]
            import mujoco
            mujoco.mj_forward(p.m, p.d)


p = plant.build(body, floor)
ref = Reference.load(run_dir / "refs" / f"{walk}.npz")
omega = control.omega0(p, Reference.load(run_dir / "refs" / "nominal.npz").q_home)
track = control.Track(p, omega).add_periodic(ref, 12)
print(f"{body} {walk} on {floor}: period {ref.period:.3f} s; root z of frame 0 {ref.q[0][2]:.4f}, settled stance "
      f"{p.q_settled[2]:.4f}, z_floor {p.z_floor:.4f}")
seconds = 8.0
for k in (0, 5):
    for frame in (0, 20, 50, 80, 110, 140):
        tk = R5Tracker(p, track, "R2", gains=None, terms=(), ankle_cap=(), rate=RATE)
        t_ref = 2 * ref.period + frame * ref.step  # two strides in, then the frame
        with speedfit.servo_delay(p, k) as queue:
            tk.reset_at(t_ref, ref.q[frame], ref.qd[frame] * RATE, noise_seed=1, queue=queue)
            n = int(round(seconds / tk.dt))
            fell = None
            for i in range(n):
                tk.step()
                if p.fall_cause() is not None:
                    fell = (i + 1) * tk.dt
                    break
        yaw_err = tk.yaw_buf[-1] if tk.yaw_buf else float("nan")
        print(f"  k={k} frame {frame:3d} (swing {int(track.info(t_ref)[1]['swing'])}): "
              f"{'fell at %.2f s' % fell if fell else 'up %.0f s' % seconds}; tilt max now "
              f"{np.degrees(p.tilt_from_home()):.1f} deg", flush=True)
