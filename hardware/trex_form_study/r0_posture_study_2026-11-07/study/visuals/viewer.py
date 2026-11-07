"""Package neutral-pose physical surfaces in a self-contained, offline WebGL viewer."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

from scene import Scene

HERE=Path(__file__).resolve().parent


def packed(scene):
    # Millimetre coordinates quantized to 0.05 mm. Per-corner normal and color
    # are included in deduplication, retaining hard edges and material changes.
    rows=[]
    for points,normals,color,shell,_ in scene.parts:
        p=(points.reshape(-1,3)*20000).round().astype('<i2')
        n=(np.clip(normals.reshape(-1,3),-1,1)*127).round().astype('i1')
        c=np.repeat(color,3,axis=0)
        rgb=(np.clip(c,0,1)*255).round().astype('uint8')
        raw=np.zeros((len(p),16),np.uint8)
        raw[:,:6]=p.view(np.uint8).reshape(-1,6)
        raw[:,6:9]=n.view(np.uint8)
        raw[:,9:12]=rgb
        raw[:,12]=int(shell)
        rows.append(raw)
    data=np.vstack(rows)
    unique,idx=np.unique(data.view('V16').reshape(-1),return_inverse=True)
    idx=idx.astype('<u4')
    contents=unique.tobytes()+idx.tobytes()
    return dict(verts=len(unique),indices=len(idx),
                encoded=base64.b64encode(gzip.compress(contents,compresslevel=5,mtime=0)).decode('ascii'),
                mass=round(scene.mass,3), crown=round(scene.crown*1000),sha=scene.sha)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=HERE.parent/'model/generated')
    base=p.add_mutually_exclusive_group(required=True)
    base.add_argument('--pr662',type=Path)
    base.add_argument('--r3',type=Path)
    p.add_argument('--out',type=Path,default=HERE/'Inspect-3D.html')
    args=p.parse_args()
    files={'P662':args.pr662} if args.pr662 else {'R3':args.r3}
    files.update({f+a:args.models/(f+a+'.xml') for f in 'ABC' for a in ['00','10','20']})
    models={k:packed(Scene(path)) for k,path in files.items()}
    template=HERE/'viewer_template.html'
    args.out.write_text(template.read_text().replace('__MODEL_ARCHIVE__',json.dumps(models,separators=(',',':'))))
    print(f'{len(models)} models; {args.out.stat().st_size:,} bytes')


if __name__=='__main__':
    main()
