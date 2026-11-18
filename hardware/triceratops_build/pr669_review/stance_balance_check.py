"""Extra checks on PR #669 robot.xml (review, 18 Nov 2026): settled stance torques,
foot loads, front share, CoM, and the lateral CoM shift needed for each 3-foot triangle."""
import sys, mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
mujoco.mj_step(m, d, nstep=3000)  # 3 s at 1 ms, zero command, as check_models.py
mujoco.mj_forward(m, d)
W = -m.opt.gravity[2]*m.body_mass.sum()
corners = ("left_fore","right_fore","left_hind","right_hind")

tid = np.flatnonzero(m.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)
loads = d.sensordata[m.sensor_adr[tid]]
names = [m.body(m.site_bodyid[m.sensor_objid[t]]).name for t in tid]
print("mass %.3f kg, weight %.1f N" % (m.body_mass.sum(), W))
for n, l in zip(names, loads): print("  %-16s %6.1f N  %5.1f %%" % (n, l, 100*l/loads.sum()))
front = sum(l for n,l in zip(names,loads) if "fore" in n)
print("front share %.1f %% (touch sum %.1f N = %.3f W)" % (100*front/loads.sum(), loads.sum(), loads.sum()/W))
com = d.subtree_com[1]; print("CoM", np.round(com,4))
print("settled actuator torques (N·m):")
for a in range(m.nu): print("  %-28s %7.2f  (limit %.1f)" % (m.actuator(a).name, d.actuator_force[a], m.actuator_forcerange[a,1]))
feet = {c: d.xpos[m.body(f"{c}_foot").id][:2] for c in corners}
print("foot xy:", {c: np.round(v,3).tolist() for c,v in feet.items()})
def side(p, a, b):  # signed distance of p from line a->b (positive = left of a->b)
    t = b-a; return (t[0]*(p[1]-a[1]) - t[1]*(p[0]-a[0]))/np.linalg.norm(t)
c2 = com[:2]
for lift in corners:
    sup = [feet[c] for c in corners if c != lift]
    # margin = min signed distance to the triangle's edges, oriented inward
    tri = np.array(sup); ctr = tri.mean(0); margins=[]
    for i in range(3):
        a, b = tri[i], tri[(i+1)%3]; s = np.sign(side(ctr, a, b)); margins.append(s*side(c2, a, b))
    # lateral shift needed: smallest |dy| that makes min margin >= 0
    need = None
    for dy in np.arange(0, 0.1, 0.0001):
        for sgn in (+1, -1):
            p = c2 + np.array([0, sgn*dy]); mm=[]
            for i in range(3):
                a, b = tri[i], tri[(i+1)%3]; s = np.sign(side(ctr, a, b)); mm.append(s*side(p, a, b))
            if min(mm) >= 0 and need is None: need = sgn*dy
        if need is not None: break
    print("  lift %-10s: margin at current CoM %+6.1f mm; lateral shift to reach the triangle edge %+5.1f mm" % (lift, 1000*min(margins), 1000*need))
