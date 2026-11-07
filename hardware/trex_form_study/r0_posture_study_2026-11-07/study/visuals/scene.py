"""Orthographic views of the physical, visible surfaces in a compiled MuJoCo model.

Uses a small CPU z-buffer. All geometry and colors come from the supplied MJCF.
The display suppresses manufacturing reserves and coincident collision proxies.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def unit(v):
    a = np.asarray(v, dtype=float)
    return a / np.maximum(1e-12, np.linalg.norm(a, axis=-1, keepdims=True))


# Rows map world xyz onto camera right, up and depth. Top retains nose-right
# orientation. Front looks into the nose: anatomical left is on image right.
def camera(view):
    if view == 'top':
        return np.eye(3)
    eye = unit({'side': [0,-1,0], 'front':[1,0,0],
                'threequarter':[.61,-1,.41]}[view])
    right = unit(np.cross([0,0,1], eye))
    return np.array([right, np.cross(eye,right),eye])


@lru_cache(maxsize=256)
def primitive(kind, size_tuple):
    size = np.array(size_tuple)
    box, cyl, sphere, ellip, cap = (int(getattr(mujoco.mjtGeom, 'mjGEOM_'+s))
                                     for s in ['BOX','CYLINDER','SPHERE','ELLIPSOID','CAPSULE'])
    if kind == box:
        verts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*size
        quads = np.array([[0,3,2,1],[4,5,6,7],[0,1,5,4],
                          [1,2,6,5],[2,3,7,6],[3,0,4,7]])
        faces = np.concatenate((quads[:,[0,1,2]], quads[:,[0,2,3]]))
        poly = verts[faces]
        norm = unit(np.cross(poly[:,1]-poly[:,0], poly[:,2]-poly[:,0]))
        return poly, np.repeat(norm[:,None,:],3,axis=1)
    n = 24
    phi = np.arange(n)*2*np.pi/n
    ring = np.stack([np.cos(phi),np.sin(phi),np.zeros(n)],axis=-1)
    if kind == cyl:
        verts = np.vstack([ring*size[0]-[0,0,size[1]],ring*size[0]+[0,0,size[1]]])
        normals = np.vstack([ring,ring])
        j = np.arange(n)
        kk=(j+1)%n
        faces = np.vstack([np.c_[j,kk,j+n],np.c_[kk,kk+n,j+n]])
        p,norm=verts[faces],normals[faces]
        for sign in (-1,1):
            disc=ring*size[0]+[0,0,size[1]*sign]
            tris=np.stack([np.repeat([[0,0,size[1]*sign]],n,axis=0),disc[j],disc[kk]],axis=1)
            p=np.concatenate([p,tris])
            norm=np.concatenate([norm,np.broadcast_to([0,0,sign],tris.shape)])
        return p,norm
    if kind not in (cap,ellip,sphere):
        raise ValueError(f'Unknown physical primitive type {kind}')
    lat = np.linspace(-np.pi/2,np.pi/2,13)
    if kind==cap:
        lat = np.insert(lat,6,0.)
    dirs = np.stack([np.cos(lat)[:,None]*np.cos(phi),
                     np.cos(lat)[:,None]*np.sin(phi),
                     np.broadcast_to(np.sin(lat)[:,None],(len(lat),n))],axis=-1)
    radii = size if kind==ellip else np.full(3,size[0])
    verts=dirs*radii
    if kind==cap:
        verts[:7,:,2]-=size[1]
        verts[7:,:,2]+=size[1]
    j=(np.arange(len(lat)-1)[:,None]*n+np.arange(n)).ravel()
    kk=((j//n)*n+(j+1)%n)
    faces=np.vstack([np.c_[j,kk,j+n],np.c_[kk,kk+n,j+n]])
    return verts.reshape(-1,3)[faces],unit(dirs/radii).reshape(-1,3)[faces]


class Scene:
    def __init__(self,path,cutaway_names=()):
        self.path=Path(path)
        self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.model=m=mujoco.MjModel.from_xml_path(str(path))
        self.data=d=mujoco.MjData(m)
        # Candidate qpos0 is the author's planted pose. Match it exactly.
        mujoco.mj_forward(m,d)
        self.parts=[]
        self.head_surfaces=[]
        head_id=m.body('head').id
        for g in range(m.ngeom):
            name=m.geom(g).name
            mat=m.geom_matid[g]
            rgba=m.mat_rgba[mat] if mat>=0 else m.geom_rgba[g]
            if (m.geom_group[g] not in (1,2,4) or rgba[3] < 0.5
                    or name.startswith(('outline_', 'patch_'))):
                continue
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                k=m.geom_dataid[g]
                a=m.mesh_vertadr[k]; f=m.mesh_faceadr[k]; n=m.mesh_normaladr[k]
                count=m.mesh_facenum[k]
                p=m.mesh_vert[a+ m.mesh_face[f:f+count]]
                normals=m.mesh_normal[n+ m.mesh_facenormal[f:f+count]]
            else:
                p,normals=primitive(int(m.geom_type[g]),tuple(m.geom_size[g]))
            rotation=d.geom_xmat[g].reshape(3,3)
            surface=p@rotation.T+d.geom_xpos[g]
            parent=m.geom_bodyid[g]
            while parent and parent != head_id:
                parent=m.body_parentid[parent]
            if parent == head_id:
                self.head_surfaces.append(surface)
            shell = name.startswith(('cover_', 'frame_tooth_')) or name in cutaway_names
            self.parts.append((surface.astype(np.float32), (normals@rotation.T).astype(np.float32),
                               np.broadcast_to(rgba[:3],(len(p),3)),shell,name))
        if not self.parts:
            raise ValueError('No visible model geometry')
        self.com=d.subtree_com[m.body('pelvis').id].copy()
        self.mass=float(m.body_mass.sum())
        self.hip=d.xanchor[m.joint('left_hip_pitch').id].copy()
        self.crown=max(p[...,2].max() for p in self.head_surfaces)
        self.feet=[d.geom_xpos[m.geom(s+'_sole').id].copy() for s in ['left','right']]

    def max_size(self):
        v=np.concatenate([p[0].reshape(-1,3) for p in self.parts])
        return v.min(axis=0),v.max(axis=0)

    @lru_cache(maxsize=2)
    def triangles(self,cutaway):
        chosen=[p for p in self.parts if not(cutaway and p[3])]
        return tuple(np.concatenate([p[i] for p in chosen]) for i in range(3))


@lru_cache
def native():
    src=Path(__file__).with_name('raster.c')
    digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    base=Path(tempfile.gettempdir())/('rex-raster-'+digest+'-'+str(os.getuid()))
    base.mkdir(mode=0o700,exist_ok=True)
    lib=base/'raster.so'
    if not lib.exists():
        dest=lib.with_suffix(f'.{os.getpid()}.so')
        mode='-dynamiclib' if sys.platform=='darwin' else '-shared'
        subprocess.run(['cc','-O3',mode,'-fPIC',str(src),'-o',str(dest),'-lm'],check=True)
        os.replace(dest,lib)
    function=ctypes.CDLL(str(lib)).raster
    function.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*4
    function.restype=None
    return function


def projected(point,view,scale,box_center,world_center):
    delta=(np.asarray(point)-world_center)@camera(view)[:2].T
    return np.asarray(box_center)+delta*np.array([scale,-scale])


def render(scene,view,size,scale,world_center,cutaway=False,silhouette=False,shadow=True):
    w,h=size
    aa=2
    w*=aa;h*=aa;scale*=aa
    pts,normals,colors=scene.triangles(cutaway)
    basis=camera(view)
    v=(pts-world_center)@basis.T
    v[...,:2]=v[...,:2]*[scale,-scale]+[w/2,h/2]
    v=np.ascontiguousarray(v,np.float32)
    if silhouette:
        rgb=np.broadcast_to(np.array([19,23,21]),normals.shape)
    else:
        # Treat mesh interiors and back-facing thin edges as two-sided surfaces.
        n=normals*np.where(np.sum(normals*basis[2],axis=-1,keepdims=True)<0,-1.,1.)
        key=unit([.30,-.72,1.0]); fill=unit([-.5,.35,.7])
        diffuse=.30+.65*np.maximum(np.sum(n*key,axis=-1),0)+.10*np.maximum(np.sum(n*fill,axis=-1),0)
        spec=.045*np.maximum(np.sum(n*unit(key+basis[2]),axis=-1),0)**28
        linear=np.clip(colors[:,None,:]*diffuse[:,:,None]+spec[:,:,None],0,1)
        rgb=255*np.where(linear<.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
    rgb=np.ascontiguousarray(rgb,np.float32)
    depth=np.full((h,w),-np.inf,np.float32)
    out=np.zeros((h,w,4),np.uint8)
    native()(len(v),w,h,v.ctypes.data,rgb.ctypes.data,depth.ctypes.data,out.ctypes.data)
    if not silhouette:
        # Near edges receive a small contact tint if a closer surface borders them.
        occlusion=np.zeros((h,w),np.float32)
        clean=np.where(np.isfinite(depth),depth,-100)
        for dy,dx in [(-4,0),(4,0),(0,4),(0,-4),(-7,-7),(5,5)]:
            neighbor=np.roll(clean,(dy,dx),axis=(0,1))-clean
            occlusion+=((neighbor>.0025)&(neighbor<.075))/6
        out[:,:,:3]=(out[:,:,:3]*(1-.12*occlusion[:,:,None])).astype('uint8')
    result=Image.fromarray(out)
    if shadow and not silhouette and view=='threequarter':
        soft=Image.new('RGBA',(w,h))
        draw=ImageDraw.Draw(soft)
        for foot in scene.feet:
            phi=np.linspace(0,2*np.pi,60)
            points=foot+np.c_[.105*np.cos(phi),.062*np.sin(phi),np.full_like(phi,-foot[2])]
            screen=projected(points,view,scale,(w/2,h/2),world_center)
            draw.polygon([tuple(p) for p in screen],fill=(55,63,51,39))
        soft=soft.filter(ImageFilter.GaussianBlur(float(max(1,scale*.009))))
        soft.alpha_composite(result)
        result=soft
    return result.resize(size,Image.Resampling.LANCZOS)
