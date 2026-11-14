"""Physical transverse cuts through the installed model, in its design pose."""
from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir() + "/meso-design-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import mujoco as mj
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
PALETTE = {"Cover": "#2d7165", "Drive": "#a76831", "Payload": "#366c92", "Structure": "#66665f"}
STATIONS = [-0.055, 0.035, 0.120]

def surface(model, i):
    kind = mj.mjtGeom(model.geom_type[i]).name[7:].lower()
    x, y, z = model.geom_size[i]
    if kind == "mesh":
        mesh = model.geom_dataid[i]
        v0, f0 = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        vs = model.mesh_vert[v0:v0+model.mesh_vertnum[mesh]]
        fs = model.mesh_face[f0:f0+model.mesh_facenum[mesh]]
        return np.asarray(vs, dtype=float), fs
    if kind == "box":
        tri = trimesh.creation.box(extents=[2*x, 2*y, 2*z])
    elif kind in ("ellipsoid", "sphere"):
        tri = trimesh.creation.icosphere(subdivisions=3)
        tri.vertices *= [x, y, z] if kind == "ellipsoid" else x
    elif kind == "cylinder":
        tri = trimesh.creation.cylinder(radius=x, height=2*y, sections=48)
    elif kind == "capsule":
        tri = trimesh.creation.capsule(radius=x, height=2*y, count=[12,24])
        tri.vertices -= (tri.bounds[1] + tri.bounds[0])/2
    else:
        return None
    return tri.vertices, tri.faces

def classify(name, material):
    label = (name + " " + material).lower()
    if any(s in label for s in ["_case", "_face_", "servo", "gearbox", "_shaft", "bearing", "actuator"]):
        return "Drive"
    if any(s in label for s in ["jetson", "battery", "bms", "dc_dc", "power", "breaker", "compute", "carrier", "fan", "fuse", "control_board", "pdb", "regulator", "electronics"]):
        return "Payload"
    if any(s in label for s in ["cover", "shield", "skin", "panel", "scute", "fairing", "shell", "flank", "belly", "dorsal", "abdominal", "ribcage", "haunch", "cheek", "shroud", "mantle", "apron"]):
        return "Cover"
    return "Structure"

def cut(verts, faces, x):
    tri = verts[faces]
    d = tri[:,:,0] - (x + 3e-9)
    spans = (np.min(d, axis=1) < 0) & (np.max(d, axis=1) > 0)
    tri, d = tri[spans], d[spans]
    if not len(tri):
        return np.empty((0,2,2))
    edges = np.roll(tri,-1,axis=1) - tri
    d2 = np.roll(d,-1,axis=1)
    r,c = np.nonzero((d<0) != (d2<0))
    points = tri[r,c] + edges[r,c] * (d[r,c]/(d[r,c]-d2[r,c]))[:,None]
    return points[:,1:].reshape(-1,2,2)

def input_model(path):
    m = mj.MjModel.from_xml_path(str(path.resolve()))
    d = mj.MjData(m)
    if m.nkey:
        mj.mj_resetDataKeyframe(m,d,0)
    mj.mj_forward(m,d)
    # Joint origin rather than the moving motor case defines the station datum.
    j = next(i for i in range(m.njnt) if "left_hip_pitch" in m.joint(i).name)
    hip = d.xanchor[j].copy()
    hip[1] = 0.0  # transverse spans use robot centerline
    geoms = []
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == 0 or m.geom_group[i] == 5 or m.geom_rgba[i,3] < 0.02:
            continue
        name = m.geom(i).name or ""
        mat = m.material(m.geom_matid[i]).name if m.geom_matid[i] >= 0 else ""
        if any(k in name for k in ("reserve", "clearance_envelope", "collision_proxy")):
            continue
        mesh = surface(m,i)
        if mesh is None:
            continue
        pts, faces = mesh
        pts = (pts @ d.geom_xmat[i].reshape(3,3).T + d.geom_xpos[i] - hip)*1000
        geoms.append((name, classify(name,mat), pts, faces))
    return geoms

def diagram(models, out):
    fig, axes = plt.subplots(len(models), len(STATIONS), figsize=(10.7, 2.30*len(models)+1.65), squeeze=False)
    fig.patch.set_facecolor("#fbfaf6")
    fig.subplots_adjust(left=.16,right=.97,top=.845,bottom=.085,hspace=.13,wspace=.07)
    fig.text(.04,.965,"Transverse cuts through the body",fontsize=24,fontweight="bold",color="#193937",va="top")
    fig.text(.04,.919,"Fixed vertical planes, measured from the hip pitch axis. All panels share the same physical scale.",fontsize=10,color="#4e5b59",va="top")
    all_rows = []
    for row,(label,path) in enumerate(models):
        geo = input_model(path)
        for col, station in enumerate(STATIONS):
            ax = axes[row,col]
            ax.set_facecolor("#f1f1e9")
            ax.set(xlim=(-208,208),ylim=(-94,215),aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d9dcd3")
            ax.axvline(0,lw=.5,ls="--",c="#b8bfb6")
            ax.axhline(0,lw=.5,ls="--",c="#b8bfb6")
            parts = []
            for name,cat,points,faces in geo:
                if points[:,0].min() > station*1000 or points[:,0].max() < station*1000:
                    continue
                lines = cut(points,faces,station*1000)
                if len(lines):
                    ax.add_collection(LineCollection(lines,colors=PALETTE[cat],linewidths=1.22 if cat=="Cover" else .86))
                    pts = lines.reshape((-1,2))
                    parts.append({"name":name,"role":cat,"y_min_mm":float(pts[:,0].min()),"y_max_mm":float(pts[:,0].max()),"z_min_from_hip_mm":float(pts[:,1].min()),"z_max_from_hip_mm":float(pts[:,1].max())})
            all_rows.append({"model":label,"x_from_hip_mm":round(station*1000),"parts":parts})
            if col == 0:
                ax.text(-.07,.5,label,transform=ax.transAxes,fontsize=12,fontweight="bold",ha="right",va="center",color="#244643")
            if row==0:
                ax.set_title(f"{'Aft' if station<0 else 'Forward'}  {abs(station*1000):.0f} mm",loc="left",fontsize=12,color="#244643",pad=6)
            if row==len(models)-1:
                ax.plot([-183,-133],[-79,-79],c="#374a48",lw=2)
                ax.text(-122,-79,"50 mm",fontsize=7,va="center",color="#374a48")
    legend=[Line2D([0],[0],color=v,label=k,lw=2.2) for k,v in PALETTE.items()]
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.56,.014),ncol=4,frameon=False)
    fig.savefig(out/"transverse_sections.png",dpi=180)
    plt.close(fig)
    (out/"sections.json").write_text(json.dumps(all_rows,indent=2)+"\n")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",action="append",required=True,metavar="LABEL=XML")
    p.add_argument("--out",type=Path,default=ROOT)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    diagram([(s.split("=",1)[0],Path(s.split("=",1)[1])) for s in args.model],args.out)

if __name__=="__main__":
    main()
