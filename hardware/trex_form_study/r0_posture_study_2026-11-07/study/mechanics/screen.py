"""Reproduce the comparative mechanical screen using MuJoCo 3.10, NumPy and SciPy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull

SIDES = ("left", "right")
MM = 1000.0
T = mj.mjtGeom
BASE = Path(__file__).resolve().parent


def model_key(path):
    if path.parent.name.lower()=="pr662":
        return "PR662"
    if path.stem=="model" and path.parent.name.lower() in ("successor","r3"):
        return "R3"
    return path.stem


def fresh(m):
    d = mj.MjData(m)
    if m.nkey:
        mj.mj_resetDataKeyframe(m, d, 0)
    mj.mj_forward(m, d)
    return d


def mesh_vertices(m, g):
    mesh = m.geom_dataid[g]
    start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    return m.mesh_vert[start:start + count].astype(float)


def bounds(m, d, g):
    """World axis-aligned extrema using exact support functions for these geom types."""
    p, r, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    t = m.geom_type[g]
    if t == T.mjGEOM_MESH:
        pts = mesh_vertices(m, g) @ r.T + p
        return np.array([pts.min(axis=0), pts.max(axis=0)])
    if t == T.mjGEOM_BOX:
        e = abs(r) @ s
    elif t == T.mjGEOM_CYLINDER:
        e = s[0] * np.sqrt(np.maximum(0, 1 - r[:, 2] ** 2)) + s[1] * abs(r[:, 2])
    elif t == T.mjGEOM_CAPSULE:
        e = s[0] + s[1] * abs(r[:, 2])
    elif t == T.mjGEOM_ELLIPSOID:
        e = np.linalg.norm(r * s, axis=1)
    elif t == T.mjGEOM_SPHERE:
        e = np.full(3, s[0])
    else:
        raise ValueError(f"Unhandled geom {m.geom(g).name} {t}")
    return np.array([p - e, p + e])


def aggregate_bounds(m, d, ids):
    spans = np.array([bounds(m, d, i) for i in ids])
    return np.array([spans[:, 0].min(axis=0), spans[:, 1].max(axis=0)])


def physical(m, g):
    # R2/R3 use group 5 for installation reservation; 3D physical hardware and skin remain.
    name = m.geom(g).name or ""
    return m.geom_bodyid[g] != 0 and m.geom_group[g] < 5 and not name.startswith(("outline_","patch_"))


def descendants(m, body):
    result = {body}
    for i in range(body + 1, m.nbody):
        if m.body_parentid[i] in result:
            result.add(i)
    return result


def unit_forces(m):
    return -m.opt.gravity * m.body_mass.sum()


def sole_corners(m,d,side):
    g = m.geom(side + "_sole").id
    x, y, z = m.geom_size[g]
    verts = np.array([[a * x, b * y, -z] for a in (-1,1) for b in (-1,1)])
    return verts @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g]


def support_margin(corners, xy):
    # Euclidean signed distance to closest support-polygon plane in the horizontal plane.
    hull = ConvexHull(corners[:, :2])
    return float(np.min(-(hull.equations[:, :2] @ xy + hull.equations[:, 2])))


def set_pose(m,d,commands,ground_side=None):
    d.qpos[:] = m.key_qpos[0] if m.nkey else m.qpos0
    d.qvel[:] = 0
    d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]] = commands
    mj.mj_forward(m,d)
    if ground_side:
        d.qpos[2] -= sole_corners(m,d,ground_side)[:,2].min()
        mj.mj_forward(m,d)


def home_command(m):
    q = m.key_qpos[0] if m.nkey else m.qpos0
    return q[m.jnt_qposadr[m.actuator_trnid[:,0]]].copy()


def static_load(m,d,sides):
    com = d.subtree_com[0]
    force = unit_forces(m)
    effort = d.qfrc_bias.copy()
    centers = [sole_corners(m,d,s).mean(axis=0) for s in sides]
    if len(sides) == 1:
        weights = [1.]
        points = [np.array([*com[:2], centers[0][2]])]
    else:
        # Choose a pair of CoPs at COM.x on sole centerlines. Vertical forces are
        # allocated from actual COM.y. Their resultant passes through the COM.
        left, right = centers
        weight = (com[1]-right[1])/(left[1]-right[1])
        weights = [weight, 1-weight]
        points = [np.array([com[0], c[1], c[2]]) for c in centers]
    margins = []
    for side, point, weight in zip(sides,points,weights):
        sole = m.geom(side+"_sole").id
        jac = np.zeros((3,m.nv))
        mj.mj_jac(m,d,jac,None,point,m.geom_bodyid[sole])
        effort -= jac.T @ (force * weight)
        margins.append(support_margin(sole_corners(m,d,side),point[:2]))
    joints = m.actuator_trnid[:,0]
    torques = {m.joint(j).name:float(effort[m.jnt_dofadr[j]]) for j in joints}
    if np.linalg.norm(effort[:6]) > 1e-6:
        raise ValueError("Unbalanced root wrench")
    return dict(joint_Nm=torques, reaction_share=weights, cop_margin_mm=np.array(margins)*MM,
                root_residual_force_N=np.linalg.norm(effort[:3]),root_residual_moment_Nm=np.linalg.norm(effort[3:6]))


def single_support(m,d):
    home = home_command(m)
    ids = {m.joint(j).name:i for i,j in enumerate(m.actuator_trnid[:,0])}
    limits = m.jnt_range[m.actuator_trnid[:,0]]
    results = {}
    for side, swing, sign in [("left","right",1),("right","left",-1)]:
        candidates = []
        for tilt in np.linspace(0,.45,91):
            q = home.copy()
            for s in SIDES:
                q[ids[s+"_hip_roll"]] += -sign*tilt
                q[ids[s+"_ankle_roll"]] += sign*tilt
            for joint,angle in [("hip_pitch",-.25),("knee",.5),("ankle_pitch",-.25)]:
                q[ids[swing+"_"+joint]] += angle
            if np.any(q < limits[:,0]-1e-9) or np.any(q > limits[:,1]+1e-9):
                continue
            set_pose(m,d,q,side)
            row = static_load(m,d,[side])
            row["hip_roll_offset_rad"] = -sign*tilt
            row["swing_sole_clearance_mm"] = float(sole_corners(m,d,swing)[:,2].min()*MM)
            row["support_sole_corner_height_range_mm"] = np.ptp(sole_corners(m,d,side)[:,2])*MM
            candidates.append(row)
        if not candidates:
            results[side]=None
            continue
        best=max(candidates,key=lambda x:x["cop_margin_mm"][0])
        feasible=[x for x in candidates if x["cop_margin_mm"][0]>=3 and x["swing_sole_clearance_mm"]>=8 and x["support_sole_corner_height_range_mm"]<.5]
        best["pose_count_margin_ge3_clearance_ge8"]=len(feasible)
        if feasible:
            alternate=min(feasible,key=lambda x:abs(x["joint_Nm"][side+"_hip_roll"]))
            best["lowest_feasible_hip_roll_Nm"]=abs(alternate["joint_Nm"][side+"_hip_roll"])
            best["lowest_feasible_tilt_rad"]=alternate["hip_roll_offset_rad"]
        results[side]=best
    return results


def ray(m,d,origin,direction):
    direction = np.asarray(direction)
    direction /= np.linalg.norm(direction)
    group = np.uint8([1,1,1,0,1,0])
    gid = np.array([-1],dtype=np.int32)
    dist = mj.mj_ray(m,d,origin,direction,group,True,-1,gid)
    if dist < 0:
        return None, None
    return m.geom(int(gid[0])).name, float(dist)


def optics(m,d):
    idx = m.camera("front_rgbd").id
    origin = d.cam_xpos[idx]
    r = d.cam_xmat[idx].reshape(3,3)
    vf = np.tan(np.radians(m.cam_fovy[idx]/2))
    # 640x400 crop specified in the platform installation. Uniformly sample the image plane.
    hf = vf*640/400
    hits = {}
    for u in np.linspace(-1,1,17):
        for v in np.linspace(-1,1,11):
            name,dist = ray(m,d,origin,r@np.array([u*hf,v*vf,-1]))
            if name is not None and m.geom_bodyid[m.geom(name).id] != 0:
                hits[name] = hits.get(name,0)+1
    low = r@np.array([0,-vf,-1])
    floor_x = None if low[2]>=0 else -origin[2]*low[0]/low[2]
    center = ray(m,d,origin,-r[:,2])
    tofs = {}
    for s in SIDES:
        site = d.site("tof_"+s)
        axis = site.xmat.reshape(3,3)[:,2]
        hit,meters = ray(m,d,site.xpos,axis)
        tofs[s] = dict(hit=hit,range_m=meters,
                       floor_forward_m=None if axis[2]>=0 else -site.xpos[2]*axis[0]/axis[2])
    return dict(camera_center_hit=center, self_occluded_rays=sum(hits.values()), grid_rays=187,
                ray_hits=hits, lowest_center_column_floor_forward_m=floor_x,tof=tofs)


def appendage_loads(m,d):
    out={}
    for body,jname in (("head","neck_pitch"),("tail","tail_yaw")):
        joints=m.joint(jname)
        root=m.body(body).id
        bodies=sorted(descendants(m,root))
        axis=d.xaxis[joints.id]
        pivot=d.xanchor[joints.id]
        # Each body's principal-axis inertia tensor is already mass-weighted in MuJoCo.
        torque=np.zeros(3); inertia=0.
        for b in bodies:
            cross=d.xipos[b]-pivot
            rotation=d.ximat[b].reshape(3,3)
            axis_body=rotation.T@axis
            inertia+=np.dot(axis_body**2,m.body_inertia[b])+m.body_mass[b]*np.linalg.norm(np.cross(cross,axis))**2
            torque+=np.cross(cross,m.opt.gravity*m.body_mass[b])
        out[jname]=dict(moving_mass_kg=m.body_mass[bodies].sum(),gravity_axis_Nm=np.dot(torque,axis),
                        gravity_off_axis_bending_Nm=np.linalg.norm(torque-axis*np.dot(torque,axis)),
                        reflected_link_inertia_kgm2=inertia, gravity_moment_xyz_Nm=torque)
    return out


def key_metrics(m,d):
    ids = [i for i in range(m.ngeom) if physical(m,i)]
    bb = aggregate_bounds(m,d,ids)
    head = [i for i in ids if m.geom_bodyid[i] in descendants(m,m.body("head").id)]
    headbb = aggregate_bounds(m,d,head)
    corners = np.concatenate([sole_corners(m,d,s) for s in SIDES])
    z0=corners[:,2].min()
    return dict(dimensions_mm=(bb[1]-bb[0])*MM,
        crown_mm=(headbb[1,2]-z0)*MM, hip_mm=(d.xanchor[m.joint("left_hip_pitch").id,2]-z0)*MM,
        physical_bounds_world_mm=bb*MM, lowest_sole_mm=z0*MM, mass_kg=m.body_mass.sum(),
        com_mm=d.subtree_com[0]*MM, double_support_hull_margin_mm=support_margin(corners,d.subtree_com[0,:2])*MM,
        neutral_sole_corner_spread_mm=np.ptp(corners[:,2])*MM, home_angles_rad=dict(zip([m.joint(j).name for j in m.actuator_trnid[:,0]], home_command(m))))


def convert(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(str(type(x)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("models",nargs="+",type=Path)
    parser.add_argument("--out",type=Path,default=BASE/"metrics.json")
    args=parser.parse_args()
    rows={}
    for path in args.models:
        m=mj.MjModel.from_xml_path(str(path.resolve()));d=fresh(m)
        name=model_key(path)
        row=key_metrics(m,d)
        row["xml_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        row["neutral_load"]=static_load(m,d,SIDES)
        row["appendage_loads"]=appendage_loads(m,d)
        row["single_support"]=single_support(m,d)
        rows[name]=row
        d=fresh(m)
        row["optics"]=optics(m,d)
        print(name, row["mass_kg"], row["com_mm"], flush=True)
    args.out.write_text(json.dumps(rows,default=convert,indent=2)+"\n")

if __name__=="__main__":
    main()
