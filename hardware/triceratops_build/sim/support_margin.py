"""Static support margin of the centre of mass when one foot lifts, on three bases: point feet at the ankle pins;
the sole width at the pin (the ankle is a pitch hinge, so a sole is rigid in roll only); the full sole rectangles.
Positive = inside the support polygon. Run after 3 s of stance at zero command. Usage: python support_margin.py MODEL.xml ..."""
import sys, mujoco, numpy as np
def cr(u, v): return u[0]*v[1] - u[1]*v[0]
def hull(pts):
    pts = sorted(map(tuple, pts))
    def half(p):
        h = []
        for q in p:
            while len(h) >= 2 and cr(np.subtract(h[-1], h[-2]), np.subtract(q, h[-2])) <= 0: h.pop()
            h.append(q)
        return h
    lo, up = half(pts), half(pts[::-1]); return np.array(lo[:-1] + up[:-1])
def margin(p, poly):  # signed distance to the polygon boundary, positive inside (CCW hull)
    m = 1e9
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]; t = b - a
        m = min(m, cr(t, p - a) / np.linalg.norm(t))
    return m
C = ("left_fore", "right_fore", "left_hind", "right_hind")
for path in sys.argv[1:]:
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m); mujoco.mj_step(m, d, nstep=3000); mujoco.mj_forward(m, d)
    com = d.subtree_com[1][:2]
    out = []
    for lift in C:
        pins = [d.xpos[m.body(f"{c}_foot").id][:2] for c in C if c != lift]
        corners = []
        for c in C:
            if c == lift: continue
            g = m.geom(f"{c}_sole").id; R = d.geom_xmat[g].reshape(3, 3); h = m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    corners.append((d.geom_xpos[g] + R @ np.array([sx * h[0], sy * h[1], 0]))[:2])
        seg = []
        for c in C:
            if c == lift: continue
            g = m.geom(f"{c}_sole").id; R = d.geom_xmat[g].reshape(3, 3); h = m.geom_size[g]; pin = d.xpos[m.body(f"{c}_foot").id]
            for sy in (-1, 1):
                seg.append((pin + R @ np.array([0, sy * h[1], 0]))[:2])
        out.append(f"{lift}: pins {1000*margin(com, hull(pins)):+6.1f}, sole width at pin {1000*margin(com, hull(seg)):+6.1f}, full soles {1000*margin(com, hull(corners)):+6.1f} mm")
    print(path.split('/')[-1], "| CoM", (1000*com).round(1), "|", "; ".join(out))
