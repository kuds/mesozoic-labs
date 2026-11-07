"""Sample real shell surfaces against moving hardware and opposing panels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.spatial import cKDTree

from motion import targets
from screen import BASE, SIDES, T, bounds, convert, descendants, fresh, home_command, mesh_vertices, optics, set_pose, model_key


def sampled_mesh(m,g,spacing=.005):
    mid=m.geom_dataid[g]
    start,num=m.mesh_faceadr[mid],m.mesh_facenum[mid]
    tri=mesh_vertices(m,g)[m.mesh_face[start:start+num]]
    long=np.max(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2),axis=1)
    subdivisions=np.maximum(1,np.ceil(long/spacing)).astype(int)
    out=[]
    for n in np.unique(subdivisions):
        uv=np.array([(i/n,j/n) for i in range(n+1) for j in range(n+1-i)])
        a,b,c=tri[subdivisions==n].transpose(1,0,2)
        out.append((a[:,None,:]+uv[None,:,0,None]*(b-a)[:,None,:]+uv[None,:,1,None]*(c-a)[:,None,:]).reshape(-1,3))
    return np.unique(np.round(np.concatenate(out),8),axis=0)


def mesh_triangles(m,g):
    mid=m.geom_dataid[g]
    start,num=m.mesh_faceadr[mid],m.mesh_facenum[mid]
    return mesh_vertices(m,g)[m.mesh_face[start:start+num]]


def shells_cross(tri_a,tri_b):
    """Detect transverse edge/face intersections on actual triangles. Coplanar
    tangencies are left unresolved. Returns a boolean, never a penetration depth."""
    amin,amax=tri_a.min(axis=1),tri_a.max(axis=1)
    bmin,bmax=tri_b.min(axis=1),tri_b.max(axis=1)
    broad=np.all(amax[:,None]>=bmin[None,:]-1e-10,axis=2)&np.all(bmax[None,:]>=amin[:,None]-1e-10,axis=2)
    ia,ib=np.where(broad)
    if not len(ia):return False
    for a,b in [(tri_a[ia],tri_b[ib]),(tri_b[ib],tri_a[ia])]:
        center=b[:,0];u=b[:,1]-center;v=b[:,2]-center
        for i in range(3):
            begin=a[:,i]; edge=a[:,(i+1)%3]-begin
            cross=np.cross(edge,v);det=np.einsum('ij,ij->i',u,cross)
            inv=np.divide(1.,det,out=np.zeros_like(det),where=abs(det)>1e-15)
            s=begin-center
            aa=inv*np.einsum('ij,ij->i',s,cross)
            k=np.cross(s,u)
            bb=inv*np.einsum('ij,ij->i',edge,k)
            alpha=inv*np.einsum('ij,ij->i',v,k)
            hit=(abs(det)>1e-15)&(aa>=0)&(bb>=0)&(aa+bb<=1)&(alpha>1e-8)&(alpha<1-1e-8)
            if np.any(hit):return True
    return False


def signed_distance(m,g,p):
    s=m.geom_size[g]
    if m.geom_type[g]==T.mjGEOM_BOX:
        q=abs(p)-s
    elif m.geom_type[g]==T.mjGEOM_CYLINDER:
        q=np.stack((np.linalg.norm(p[:,:2],axis=1)-s[0],abs(p[:,2])-s[1]),axis=1)
    elif m.geom_type[g]==T.mjGEOM_CAPSULE:
        p=p.copy()
        p[:,2]-=np.clip(p[:,2],-s[1],s[1])
        return np.linalg.norm(p,axis=1)-s[0]
    elif m.geom_type[g]==T.mjGEOM_SPHERE:
        return np.linalg.norm(p,axis=1)-s[0]
    else:
        raise ValueError(m.geom(g).name)
    return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)


def sample_cases(m):
    home=home_command(m)
    idx={m.joint(j).name:i for i,j in enumerate(m.actuator_trnid[:,0])}
    limits=m.jnt_range[m.actuator_trnid[:,0]]
    yield "home",home
    for n in range(16):
        yield f"transfer_{n:02}",targets(m,"transfer",.5+n/8)
    for n in range(9):
        q=home.copy()
        q[idx["neck_pitch"]]=np.linspace(*limits[idx["neck_pitch"]],9)[n]
        q[idx["tail_yaw"]]=np.linspace(*limits[idx["tail_yaw"]],9)[n]
        yield f"articulation_{n:02}",q
    rng=np.random.default_rng(735)
    for i in range(18):
        yield f"joint_box_{i:02}",rng.uniform(*limits.T)


def analyze(path):
    m=mj.MjModel.from_xml_path(str(path.resolve()));d=fresh(m)
    covers=[i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("cover_") and m.geom_type[i]==T.mjGEOM_MESH]
    if not covers:
        return {"scope":"No study cover_ meshes in source model"}
    # Only modeled solid motor housings, bearing blocks, structural links and feet.
    # Same-body fit, service and cable volumes are audited separately.
    parts=[i for i in range(m.ngeom) if m.geom_bodyid[i] and m.geom_type[i] in (T.mjGEOM_BOX,T.mjGEOM_CYLINDER,T.mjGEOM_CAPSULE,T.mjGEOM_SPHERE) and
           any(s in m.geom(i).name for s in ("_case","_rail_","_sole","_yoke","_spar","_bearing","_fork","_shaft","_motor","_arm_","_metatarsal")) and
           "reserve" not in m.geom(i).name]
    local={g:sampled_mesh(m,g) for g in covers}
    triangles={g:mesh_triangles(m,g) for g in covers}
    trees={g:cKDTree(pts) for g,pts in local.items()}
    pairs=[(a,b) for a in covers for b in parts if m.body_weldid[m.geom_bodyid[a]]!=m.body_weldid[m.geom_bodyid[b]]]
    shellpairs=[(a,b) for i,a in enumerate(covers) for b in covers[i+1:] if m.body_weldid[m.geom_bodyid[a]]!=m.body_weldid[m.geom_bodyid[b]]]
    # Include the two leg chains' opposing hardware regardless of whether a shell is present.
    # No two left/right parts in this set are intentional mating pairs.
    legsets=[[g for g in parts if (m.geom(g).name or "").startswith(side+"_")] for side in SIDES]
    cross_leg_pairs=[(a,b) for a in legsets[0] for b in legsets[1]]
    tail=[g for g in range(m.ngeom) if m.geom_bodyid[g] in descendants(m,m.body("tail").id) and m.geom_group[g]<5]
    records={}
    groups={"home":[],"transfer":[],"articulation":[],"joint_box":[]}
    for case,q in sample_cases(m):
        set_pose(m,d,q)
        # The combined leg box is an unloaded kinematic sample; do not infer foot contact.
        R=d.geom_xmat.reshape(-1,3,3)
        centers=d.geom_xpos
        transformed={a:xyz@R[a].T+centers[a] for a,xyz in local.items()}
        boxes={g:(np.array([transformed[g].min(axis=0),transformed[g].max(axis=0)]) if g in transformed else bounds(m,d,g)) for g in set(parts+covers)}
        witnesses=[]
        for a,b in pairs+shellpairs:
            lo,hi=boxes[b]
            aa,ab=boxes[a]
            if np.any(ab<lo-.015) or np.any(aa>hi+.015):
                continue
            pts=transformed[a]
            pts=pts[np.all((pts>=lo-.015)&(pts<=hi+.015),axis=1)]
            if not len(pts):continue
            pts=(pts-centers[b])@R[b]
            if b in trees:
                distance=float(trees[b].query(pts,distance_upper_bound=.015)[0].min())
                kind="shell_to_shell_unsigned"
            else:
                distance=float(signed_distance(m,b,pts).min())
                kind="shell_to_hardware_signed"
            if distance<.015:
                rec=dict(a=m.geom(a).name,b=m.geom(b).name,mm=distance*1000,kind=kind)
                if b in trees and distance<.003:
                    rec['transverse_triangle_intersection']=shells_cross(triangles[a]@R[a].T+centers[a],triangles[b]@R[b].T+centers[b])
                witnesses.append(rec)
        opposing=[]
        for a,b in cross_leg_pairs:
            lo,hi=boxes[b];aa,ab=boxes[a]
            if np.any(ab<lo-.006) or np.any(aa>hi+.006):continue
            distance=mj.mj_geomDistance(m,d,a,b,.006,None)
            if distance<.003:
                opposing.append(dict(a=m.geom(a).name,b=m.geom(b).name,mm=1000*distance))
        group=case.split("_")[0] if not case.startswith("joint_box") else "joint_box"
        row=dict(case=case,angles_rad=q,worst_signed_mm=min([w["mm"] for w in witnesses if "hardware" in w["kind"]],default=None),
                 worst_shell_pair_mm=min([w["mm"] for w in witnesses if "shell_unsigned" in w["kind"]],default=None),
                 tail_ground_mm=min(bounds(m,d,g)[0,2] for g in tail)*1000,
                 opposing_leg_witnesses_under_3mm=sorted(opposing,key=lambda x:x["mm"]),
                 witnesses_under_3mm=sorted([w for w in witnesses if w["mm"]<3], key=lambda x:x["mm"]))
        if group in ("articulation","home"):
            row["optics"]=optics(m,d)
        records[case]=row
        groups[group].append(row)
    summary={}
    for key, seq in groups.items():
        data=[(x["case"],w) for x in seq for w in x["witnesses_under_3mm"]]
        summary[key]=dict(poses=len(seq), poses_with_measured_gap_under_3mm=sum(bool(x["witnesses_under_3mm"]) for x in seq),
                          poses_confirmed_panel_to_panel_crossing=sum(any(w.get("transverse_triangle_intersection",False) for w in x["witnesses_under_3mm"]) for x in seq),
                          poses_opposing_leg_gap_under_3mm=sum(bool(x["opposing_leg_witnesses_under_3mm"]) for x in seq),
                          poses_with_confirmed_penetration=sum(any(w["mm"]<0 for w in x["witnesses_under_3mm"]) for x in seq),
                          worst_witness=min(data,key=lambda x:x[1]["mm"]) if data else None,
                          min_tail_ground_mm=min(x["tail_ground_mm"] for x in seq))
    return dict(xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mesh_sample_spacing_m=.005,cover_count=len(covers),primitive_pairs=len(pairs),shell_pairs=len(shellpairs),opposing_leg_pairs=len(cross_leg_pairs),summary=summary,cases=records)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("models",nargs="+",type=Path)
    p.add_argument("--out",type=Path,default=BASE/"clearance.json");a=p.parse_args()
    rows={}
    for path in a.models:
        rows[model_key(path)]=analyze(path)
        print(path.stem,rows[model_key(path)].get("summary",{}).get("home","source"),flush=True)
    a.out.write_text(json.dumps(rows,default=convert,indent=2)+"\n")

if __name__=="__main__":
    main()
