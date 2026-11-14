"""Installed-geometry screening in the flat-foot review pose; metres internally.

Run with MuJoCo 3.10 and NumPy. Uses actual rendered mesh triangles for open
shells; filling their convex hulls would turn a hollow panel into a solid.
"""
import argparse
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

import mujoco as mj
import numpy as np


BOX = mj.mjtGeom.mjGEOM_BOX
MESH = mj.mjtGeom.mjGEOM_MESH
CYLINDER = mj.mjtGeom.mjGEOM_CYLINDER
CAPSULE = mj.mjtGeom.mjGEOM_CAPSULE
SPHERE = mj.mjtGeom.mjGEOM_SPHERE
ELLIPSOID = mj.mjtGeom.mjGEOM_ELLIPSOID
CORNERS = np.array(list(product((-1., 1.), repeat=3)))
BOX_FACES = np.array([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]
])
# Dimensions follow installed XML geom axes, not supplier XYZ labels. Millimetres.
MIN_SIZES = {
    'battery_pack': (143, 44, 45), 'jetson_kit': (103.2, 90.7, 35.86),
    'teensy_41': (60.96, 18.38, 15), 'logic_converter': (17.78, 12.70, 3.556),
    'compute_converter': (25.4, 25.4, 9.525),
    'regulated_12v_ttl_case': (43.18, 31.75, 9.017),
    'oak_d_lite': (17.5, 91, 28), 'tof_left_carrier': (2, 18, 13),
    'tof_right_carrier': (2, 18, 13), 'neck_motor_case': (46.5, 34, 28.5),
    'tail_motor_case': (46.5, 28.5, 34)
}
OWNERS = {
    'battery_pack':'battery_reserve', 'jetson_kit':'compute_reserve',
    'teensy_41':'control_reserve','can_left':'control_reserve',
    'can_right':'control_reserve','pelvis_imu':'control_reserve',
    'logic_converter':'power_reserve', 'compute_converter':'power_reserve',
    'protection_distribution':'power_reserve',
    'regulated_12v_ttl_case':'regulated_12v_ttl_reserve',
    'drive_disconnect_case':'drive_disconnect_reserve', 'oak_d_lite':'rgbd_reserve',
    'tof_left_carrier':'tof_left_reserve', 'tof_right_carrier':'tof_right_reserve'
}


def tri_box(tris, half):
    """Exact triangle vs closed aligned box SAT, including crossing faces."""
    mask = np.all(tris.min(1) <= half + 1e-11, axis=1) & np.all(tris.max(1) >= -half - 1e-11, axis=1)
    ts = tris[mask]
    if not len(ts):
        return False
    edges = np.roll(ts, -1, axis=1) - ts
    axes = [np.cross(edges[:, 0], edges[:, 1])]
    axes.extend(np.cross(edges[:, i], unit) for unit in np.eye(3) for i in range(3))
    alive = np.ones(len(ts), dtype=bool)
    for axis in axes:
        proj = np.einsum('nvi,ni->nv', ts, axis)
        radius = np.abs(axis) @ half
        alive &= (proj.min(1) <= radius + 1e-12) & (proj.max(1) >= -radius - 1e-12)
        if not alive.any():
            return False
    return True


def tri_cylinder(tris, radius, half_height):
    """Exact triangle/capped-cylinder intersection by clipping to the axial slab."""
    bound=np.array([radius,radius,half_height])
    viable=np.all(tris.min(1)<=bound,axis=1) & np.all(tris.max(1)>=-bound,axis=1)
    for tri in tris[viable]:
        poly=list(tri)
        for sign in (-1,1):
            clipped=[]
            for a,b in zip(poly,poly[1:]+poly[:1]):
                da,db=half_height-sign*a[2],half_height-sign*b[2]
                if da>=0:
                    clipped.append(a)
                if (da<0) != (db<0):
                    clipped.append(a+da/(da-db)*(b-a))
            poly=clipped
            if not poly:
                break
        if not poly:
            continue
        xy=np.array(poly)[:,:2]
        nxt=np.roll(xy,-1,axis=0)
        edges=nxt-xy
        den=np.sum(edges*edges,axis=1)
        t=np.divide(-np.sum(xy*edges,axis=1),den,out=np.zeros_like(den),where=den>1e-18)
        if np.min(np.sum((xy+np.clip(t,0,1)[:,None]*edges)**2,axis=1))<=radius**2:
            return True
        signed=xy[:,0]*nxt[:,1]-xy[:,1]*nxt[:,0]
        if abs(np.sum(signed))>1e-16 and (np.all(signed>=-1e-16) or np.all(signed<=1e-16)):
            return True
    return False


def prism(r, h, sides=96):
    """Circumscribed cylinder; its radius overestimate is <0.017 mm at 30 mm."""
    a = np.arange(sides)*2*np.pi/sides
    circle = np.stack([np.cos(a), np.sin(a)], axis=1)*r/np.cos(np.pi/sides)
    low = np.column_stack([circle, np.full(sides,-h)])
    high = np.column_stack([circle, np.full(sides,h)])
    low2,high2 = np.roll(low,1,axis=0),np.roll(high,1,axis=0)
    return np.concatenate([np.stack([low,high,high2],axis=1), np.stack([low,high2,low2],axis=1),
                           np.stack([low,low2,np.tile([0,0,-h],(sides,1))],axis=1),
                           np.stack([high,high2,np.tile([0,0,h],(sides,1))],axis=1)])


def segment_box_sq(a, b, half):
    """Squared distance of segment to AABB by exact 1-D piecewise minimization."""
    v = b - a
    ts = [0.,1.]
    for i in range(3):
        if abs(v[i]) > 1e-12:
            ts += [float((f-a[i])/v[i]) for f in (-half[i],half[i]) if 0 < (f-a[i])/v[i] < 1]
    ts = sorted(ts)
    samples = list(ts)
    for t1,t2 in zip(ts,ts[1:]):
        p=a+v*(t1+t2)*.5
        active=np.abs(p)>half
        if not active.any():
            return 0.
        u=v[active]
        if u@u > 1e-18:
            t=np.dot(u, np.sign(p[active])*half[active]-a[active])/(u@u)
            samples.append(np.clip(t,t1,t2))
    diff=np.maximum(np.abs(a+np.array(samples)[:,None]*v)-half,0.)
    return np.min(np.sum(diff*diff,axis=1))


class Installed:
    def __init__(self, path):
        self.m = m = mj.MjModel.from_xml_path(str(path))
        self.d = d = mj.MjData(m)
        if m.nkey:
            mj.mj_resetDataKeyframe(m,d,0)
        mj.mj_forward(m,d)
        self.names={m.geom(i).name:i for i in range(m.ngeom)}
        self.mesh = {}
        self.prim = {}
        for name,i in self.names.items():
            if m.geom_type[i] == MESH:
                j=m.geom_dataid[i]; vs=m.mesh_vertadr[j]; fs=m.mesh_faceadr[j]
                v=m.mesh_vert[vs:vs+m.mesh_vertnum[j]].astype(float)
                f=m.mesh_face[fs:fs+m.mesh_facenum[j]]
                p,r,_=self.pose(name)
                self.mesh[name]=(v@r.T+p)[f]

    def pose(self,name):
        i=self.names[name]
        return self.d.geom_xpos[i],self.d.geom_xmat[i].reshape(3,3),self.m.geom_size[i].copy()

    def true_half(self,name):
        half=self.pose(name)[2]
        if name in MIN_SIZES:
            half=np.maximum(half,np.array(MIN_SIZES[name])*.0005)
        return half

    def hits(self, name, center, rot, half):
        """Does solid name intersect the oriented box? Nonconvex panels: real triangles."""
        i=self.names[name]
        typ=self.m.geom_type[i]
        p,r,s=self.pose(name)
        if name in self.mesh:
            return tri_box((self.mesh[name]-center)@rot,half)
        offset=(p-center)@rot
        axes=rot.T@r
        if typ in (CAPSULE,SPHERE):
            end=axes[:,2]*s[1] if typ==CAPSULE else np.zeros(3)
            return segment_box_sq(offset-end,offset+end,half) <= s[0]**2
        if typ == BOX:
            s=self.true_half(name)
            if (np.abs(r.T@(center-p)) <= s).all():
                return True
            return tri_box((CORNERS*s@axes.T+offset)[BOX_FACES],half)
        if typ==CYLINDER:
            loc = r.T@(center-p)
            if np.linalg.norm(loc[:2]) <= s[0] and abs(loc[2])<=s[1]:
                return True
            if name not in self.prim:
                self.prim[name]=prism(s[0],s[1])
            return tri_box(self.prim[name]@axes.T+offset,half)
        if typ == ELLIPSOID:
            # Conservatively enclose rarely near the battery (eyes, cosmetic detail).
            if (np.abs(r.T@(center-p)) <= s).all():
                return True
            return tri_box((CORNERS*s@axes.T+offset)[BOX_FACES],half)
        raise ValueError((name,typ))

    def ray(self, name, origin, direction):
        i=self.names[name]
        if name in self.mesh:
            return mj.mj_rayMesh(self.m,self.d,i,origin,direction)
        return mj.mju_rayGeom(self.d.geom_xpos[i],self.d.geom_xmat[i],self.m.geom_size[i],origin,direction,self.m.geom_type[i])


def screen(path):
    q=Installed(path); names=q.names
    valid=[n for n in names if n and n != 'floor' and not n.startswith(('outline_','guide_','proxy_','collision_','col_','patch_'))]
    covers=[n for n in valid if n.startswith('cover_') and n in q.mesh]
    actual=[n for n in valid if n in OWNERS or n in MIN_SIZES or
            (n.endswith('_case') and q.m.geom_type[names[n]] in (BOX,CYLINDER))]
    reserves=[n for n in valid if 'reserve' in n and q.m.geom_type[names[n]]==BOX]
    hardware=[]; pairs=[]
    for n in actual+reserves:
        p,r,s=q.pose(n)
        typ=q.m.geom_type[names[n]]
        is_box=typ==BOX
        if is_box:
            s=q.true_half(n)
        else:
            # Bounding box here is a conservative screen of radial motor clearances.
            s=np.array([s[0],s[0],s[1]])
        for cover in covers:
            if is_box:
                exact=q.hits(cover,p,r,s)
                guard=False if exact else q.hits(cover,p,r,s+.002)
            else:
                tris=(q.mesh[cover]-p)@r
                exact=tri_cylinder(tris,s[0],s[2])
                guard=False if exact else tri_cylinder(tris,s[0]+.002,s[2]+.002)
            if exact or guard:
                pairs.append({'hardware':n,'cover':cover,'type':'case' if n in actual else 'reserve',
                              'test':'intersection' if exact else 'within 2 mm axis guard'})
        size=q.pose(n)[2]
        d={'name':n,'world_center_mm':np.round(p*1000,3).tolist(),
           'full_size_xml_mm':np.round((size if is_box else np.array([size[0],size[0],size[1]]))*2000,3).tolist(),
           'world_local_xyz_rotation':np.round(r,6).tolist()}
        if n in MIN_SIZES:
            d['checked_full_size_mm']=np.round(s*2000,3).tolist()
        if n in OWNERS:
            bp,br,bs=q.pose(OWNERS[n])
            corners=(CORNERS*s@r.T+p-bp)@br
            d['reserve_face_margin_mm']=np.round((bs-np.max(abs(corners),axis=0))*1000,3).tolist()
        hardware.append(d)
    package_pairs=[]; structural=[]
    for n in actual:
        if q.m.geom_type[names[n]]!=BOX:
            continue
        p,r,_=q.pose(n); half=q.true_half(n)
        for other in actual:
            if other==n or (q.m.geom_type[names[other]]==BOX and n>=other):
                continue
            if q.hits(other,p,r,half):
                package_pairs.append([n,other])
        # Direct contact with one's intended mount is possible. These are inspection
        # leads for the hardware designer; keep every intersecting structural member.
        for other in valid:
            if other in actual or other in reserves or other in covers:
                continue
            if any(tag in other for tag in ('rail','hanger','spine','cross','yoke','strut','fork','frame','bearing')):
                if q.hits(other,p,r,half):
                    structural.append([n,other])
    reserve_structural=[]
    for n in reserves:
        p,r,half=q.pose(n)
        for obj in valid:
            if obj.startswith('frame_') and q.hits(obj,p,r,half):
                reserve_structural.append([n,obj])
    p,r,_=q.pose('battery_pack')
    half=q.true_half('battery_pack')+.006
    obstacles=[n for n in valid if n not in ('battery_pack','battery_reserve')]
    strokes=[('left',1,1,.24),('right',1,-1,.24),('down',2,-1,.22),('up',2,1,.24),('aft',0,-1,.34)]
    routes={}
    base_rot=q.d.xmat[q.m.body('pelvis').id].reshape(3,3)
    for label,axis,sign,stroke in strokes:
        direction=base_rot[:,axis]*sign
        sweep_half=half+abs(r.T@direction)*stroke/2
        center=p+direction*stroke/2
        blockers=[n for n in obstacles if q.hits(n,center,r,sweep_half+.002)]
        routes[label]=dict(stroke_mm=stroke*1000, exact_axis_union=bool(np.max(np.abs(r.T@direction))>1-1e-6),direction_world=np.round(direction,6).tolist(),
                           blockers=blockers,remaining_after_removing_all_intersecting_covers=[n for n in blockers if n not in covers])
    p,r,s=q.pose('jetson_kit')
    physical=[n for n in valid if n!='jetson_kit' and n not in reserves and q.m.geom_group[names[n]]!=3]
    # Test one direct lift and four alternative plug orientations. Port datums on
    # the source kit box are undefined; these midpoint probes are trial layouts.
    kit_half=q.true_half('jetson_kit')
    lift_extent=kit_half+np.array([0,0,.025])+.002
    lift_center=p+r[:,2]*.025
    others=[n for n in obstacles if n not in ('compute_reserve','jetson_kit')]
    others+=['battery_pack','battery_reserve']
    lift_hits=[n for n in others if q.hits(n,lift_center,r,lift_extent)]
    plugs={}
    for axis,axis_label in [(0,'x'),(1,'y')]:
        for sg,tag in [(-1,'minus'),(1,'plus')]:
            probe=np.full(3,.006); probe[axis]=.015; probe[2]=.007
            ctr=p+r[:,axis]*sg*(kit_half[axis]+probe[axis]+.0002)
            plugs[f'{tag}_{axis_label}']=[n for n in others if q.hits(n,ctr,r,probe+.002)]
    kit_routes={}
    for axis,label,sign,stroke in [(2,'top',1,.05),(2,'top_full',1,.25),(1,'left',1,.20),(1,'right',-1,.20)]:
        vec=r[:,axis]*sign; ext=kit_half+.002; ext[axis]+=stroke/2
        ctr=p+vec*stroke/2
        ob=[n for n in others if q.hits(n,ctr,r,ext)]
        kit_routes[label]=dict(motion_mm=stroke*1000,blockers=ob,
                               fixed_after_cover_removal=[n for n in ob if n not in covers])
    obstruction=Counter(); firsts=[]
    # 40 mm square fan-face test assumption. 10x10 rays from max-height installed kit.
    for u,v in product(np.linspace(-.018,.018,10),repeat=2):
        origin=p+r@np.array([u,v,q.true_half('jetson_kit')[2]+.0002])
        direction=np.ascontiguousarray(r[:,2])
        hits=[]
        for n in physical:
            d=q.ray(n,origin,direction)
            if 0<=d<=.30:
                hits.append((d,n))
        if hits:
            d,n=min(hits)
            obstruction[n]+=1; firsts.append(d)
    lateral_air={}
    for axis,label,sign in [(1,'port',1),(1,'starboard',-1),(0,'aft',-1),(0,'forward',1)]:
        direction=np.ascontiguousarray(r[:,axis]*sign); other=1-axis; blocked=Counter()
        # Test throughflow space above the kit, separately from the top light path.
        for u,h in product(np.linspace(-.018,.018,10),np.linspace(.003,.015,5)):
            local=np.zeros(3);local[other]=u;local[2]=kit_half[2]+h
            origin=p+r@local
            hits=[]
            for n in physical:
                rd=q.ray(n,origin,direction)
                if 0<=rd<=.30:
                    hits.append((rd,n))
            if hits:
                blocked[min(hits)[1]]+=1
        lateral_air[label]={'open_to_300mm':50-sum(blocked.values()),'rays':50,'first_blockers':dict(blocked)}
    shafts={}
    for prefix,joint_name in [('neck','neck_pitch'),('tail','tail_yaw')]:
        joint=q.m.joint(joint_name).id
        pivot=q.d.xanchor[joint]; direction=q.d.xaxis[joint]
        coords=[]; radial=[]; tilt=[]
        for sign in [-1,1]:
            ctr,rot_b,_=q.pose(f'{prefix}_bearing_{sign}')
            delta=ctr-pivot
            coords.append(delta@direction)
            radial.append(np.linalg.norm(delta-(delta@direction)*direction))
            tilt.append(np.degrees(np.arccos(np.clip(abs(rot_b[:,2]@direction),0.,1.))))
        sp,sr,ss=q.pose(prefix+'_output_shaft')
        shafts[prefix]=dict(bearing_center_spacing_mm=round(abs(coords[1]-coords[0])*1000,3),
            max_axis_offset_mm=round(float(max(radial))*1000,3),max_bearing_axis_error_deg=round(float(max(tilt)),5),
            shaft_axis_error_deg=round(float(np.degrees(np.arccos(np.clip(abs(sr[:,2]@direction),0.,1.)))),5),
            model_shaft_diameter_mm=round(float(2000*ss[0]),3))
    return {
        'model':path.stem,'xml_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mass_kg':float(q.m.body_mass.sum()),'covers_tested':covers,
        'components':hardware,'shaft_geometry':shafts,'shell_contacts':pairs,'case_pairs_intersecting':package_pairs,
        'case_to_structure_inspection':structural,'reserves_to_candidate_frames':reserve_structural,'battery_routes':routes,
        'compute_service':{'upward_lift_50mm_blockers':lift_hits,
                           'blockers_after_removing_intersecting_covers':[n for n in lift_hits if n not in covers],
                           'hypothetical_mid_edge_30mm_plug_corridor_blockers':plugs,'straight_routes':kit_routes},
        'cooling':{'total_vertical_rays':100,'rays_open_to_300mm':100-sum(obstruction.values()),
                   'nearest_obstruction_mm':round(min(firsts)*1000,2) if firsts else None,
                   'blocking_geoms':dict(obstruction),'lateral_plenum_rays':lateral_air}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('models',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'screen.json')
    args=p.parse_args()
    results={f.stem:screen(f) for f in args.models}
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    for name,r in results.items():
        b=r['battery_routes']['left']; c=r['cooling']
        print(name,len(r['covers_tested']),len(r['shell_contacts']),b['blockers'],c['rays_open_to_300mm'])

if __name__ == '__main__':
    main()
