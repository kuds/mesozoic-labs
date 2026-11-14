"""Measure 2-D silhouette proportions directly from each compiled candidate.

Usage: python proportions.py --out proportions.json A10.xml B10.xml C10.xml
Projected sheet area is the union of painted pixels at 0.5 mm resolution.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'visuals'))
from scene import Scene

SHELL_PREFIX = ('cover_iliac', 'cover_saddle', 'cover_scapula', 'cover_shoulder',
                'cover_rib_', 'cover_sacral')
GRID_MM = 0.5

def projected_area(panels, hip_x):
    triangles = np.concatenate(panels)[:,:, [0,2]] * 1000
    lo = np.floor(triangles.min(axis=(0,1)) / GRID_MM)
    pixels = np.rint(triangles / GRID_MM - lo).astype(int)
    width, height = pixels.max(axis=(0,1)) + 1
    mask = Image.new('1', (int(width),int(height)))
    draw = ImageDraw.Draw(mask)
    for tri in pixels:
        draw.polygon([tuple(x) for x in tri],fill=1)
    covered = np.asarray(mask)
    split = round(hip_x*1000/GRID_MM-lo[0])
    return float(covered.sum()*GRID_MM**2 / 100), float(covered[:,split:].sum() / covered.sum())

def measure(path):
    scene = Scene(path)
    xyz = np.concatenate([p[0].reshape(-1,3) for p in scene.parts]).astype(float)
    muzzle, tip = xyz[:,0].max(), xyz[:,0].min()
    length = muzzle - tip
    selected = [p[0] for p in scene.parts if p[4].startswith(SHELL_PREFIX)]
    if not selected:
        raise ValueError(f'{path} has no candidate torso covers')
    head = np.vstack([p[0].reshape(-1,3) for p in scene.parts if
                      scene.model.geom_bodyid[scene.model.geom(p[4]).id] == scene.model.body('head').id]).astype(float)
    torso = np.concatenate(selected).reshape(-1,3).astype(float)
    head_x = np.ptp(head[:,0])
    hx = scene.hip[0]
    hinge = scene.data.xanchor[scene.model.joint('tail_yaw').id][0]
    area, ahead = projected_area(selected,hx)
    return dict(model_file=path.name, model_sha256=scene.sha,
                skull_x_mm=round(float(head_x)*1000,1),
                skull_percent_of_full_length=round(100*head_x/length,1),
                skull_width_mm=round(1000*np.ptp(head[:,1]),1),
                skull_depth_mm=round(1000*np.ptp(head[:,2]),1),
                trunk_cover_width_mm=round(1000*np.ptp(torso[:,1]),1),
                hip_to_tailtip_percent=round(100*(hx-tip)/length,1),
                tailhinge_to_tailtip_percent=round(100*(hinge-tip)/length,1),
                trunk_cover_side_projected_cm2=round(area,1),
                trunk_cover_projected_percent_ahead_of_hip=round(100*ahead,1))

def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--out',type=Path,required=True)
    args.add_argument('models',nargs='+',type=Path)
    a=args.parse_args()
    rows=[measure(p) for p in a.models]
    a.out.write_text(json.dumps(rows,indent=2,default=float)+'\n')

if __name__=='__main__':
    main()
