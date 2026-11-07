"""Indexed, open-ended, hollow sheet meshes and surface-integrated mass properties (metres, kg)."""
import math
import numpy as np

def number(values):
    return ' '.join(f'{float(x):.8g}' for x in np.asarray(values).ravel())

def ascending_material(stations, subdivisions=3):
    """Shape-preserving cubic Hermite interpolation of half-width, roof, floor."""
    from scipy.interpolate import PchipInterpolator
    p = np.asarray(stations, float)
    x = np.concatenate([np.linspace(a,b,subdivisions,endpoint=False) for a,b in zip(p[:-1,0],p[1:,0])]+[p[-1:,0]])
    return np.column_stack((x,PchipInterpolator(p[:,0],p[:,1:],axis=0)(x)))

def loft_grid(stations, lo, hi, steps=18, exponent=.84, twist=0):
    rings = ascending_material(stations)
    grid = []
    for x,width,bottom,top in rings:
        mid = (top+bottom)/2
        depth = (top-bottom)/2
        angles = np.linspace(np.radians(lo),np.radians(hi),steps+1)
        c, s = np.cos(angles),np.sin(angles)
        yy = width*np.sign(c)*abs(c)**exponent
        zz = mid+depth*np.sign(s)*abs(s)**exponent
        grid.append(np.column_stack((x+twist*s,yy,zz)))
    return np.asarray(grid)

def panel(grid, thickness=.0015):
    """Return inner/outer and trimmed edge walls. Conservative area model uses middle surface."""
    n,k,_=grid.shape
    dx,da=np.gradient(grid,axis=(0,1))
    norm=np.cross(da,dx)
    norm/=np.maximum(np.linalg.norm(norm,axis=2,keepdims=True),1e-12)
    verts=np.concatenate([(grid+norm*thickness/2).reshape(-1,3),(grid-norm*thickness/2).reshape(-1,3)])
    outer=[]
    for i in range(n-1):
        for j in range(k-1):
            a=i*k+j; b=a+k
            outer += [[a,a+1,b],[a+1,b+1,b]]
    outer=np.asarray(outer)
    faces=outer.tolist()+((outer[:,::-1]+n*k).tolist())
    border=[*range(k),*range(2*k-1,n*k,k),*range(n*k-2,(n-1)*k-1,-1),*range((n-2)*k,0,-k)]
    for a,b in zip(border,border[1:]+border[:1]):
        faces += [[b,a,a+n*k],[b,a+n*k,b+n*k]]
    return verts,np.asarray(faces),grid.reshape(-1,3)[outer]

def shell_mass(triangles, thickness, density):
    a,b,c=triangles.transpose(1,0,2)
    masses=np.linalg.norm(np.cross(b-a,c-a),axis=1)*(.5*density*thickness)
    mass=masses.sum()
    sums=a+b+c
    centroid=(sums*masses[:,None]).sum(axis=0)/(3*mass)
    # Exact uniform-lamina second moment, summed about the local origin.
    sec=(np.einsum('ij,ik,i->jk',sums,sums,masses)+
         np.einsum('nij,nik,n->jk',triangles,triangles,masses))/12
    sec-=mass*np.outer(centroid,centroid)
    return mass,centroid,np.trace(sec)*np.eye(3)-sec

def inertia_box(mass,half):
    x,y,z=half
    return mass/3*np.diag([y*y+z*z,x*x+z*z,x*x+y*y])

def combined(parts):
    mass=sum(q[0] for q in parts)
    pos=sum(q[0]*q[1] for q in parts)/mass
    inertia=np.zeros((3,3))
    for m,p,i in parts:
        d=p-pos
        inertia+=i+m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    return mass,pos,inertia

def ry(chest_up):
    angle=np.radians(chest_up)
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,-s],[0,1,0],[s,0,c]])

def clip_opening(tris,planes):
    """Subtract a convex solid, expressed as outward oriented halfspaces n·p <= h."""
    def split(poly,n,h):
        inside=[]; outside=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            da=np.dot(n,a)-h; db=np.dot(n,b)-h
            (inside if da<=1e-12 else outside).append(a)
            if (da<-1e-12 and db>1e-12) or (da>1e-12 and db<-1e-12):
                v=a+(b-a)*(da/(da-db));inside.append(v);outside.append(v)
        return inside,outside
    normals,offsets=planes
    signed=tris@normals.T-offsets
    # No surface can enter this volume if all its vertices lie beyond a separating plane.
    can_hit=~(signed.min(axis=1)>0).any(axis=1)
    if not can_hit.any(): return tris
    out=list(tris[~can_hit]); inside=(signed.max(axis=1)<=0).all(axis=1)
    for tr in tris[can_hit&~inside]:
        poly=list(tr)
        for n,h in zip(normals,offsets):
            poly,outer=split(poly,n,h)
            if len(outer)>=3:
                out.extend([[outer[0],outer[i],outer[i+1]] for i in range(1,len(outer)-1)])
            if len(poly)<3: break
    if not out:return np.empty((0,3,3))
    out=np.array(out)
    areas=np.linalg.norm(np.cross(out[:,1]-out[:,0],out[:,2]-out[:,0]),axis=1)
    return out[areas>2e-11]

def thicken_triangles(tris,thickness):
    """Thicken clipped midsurface facets and close cut edges. Merge matching corners."""
    flat=tris.reshape(-1,3)
    _,i,rev=np.unique(np.round(flat,8),axis=0,return_index=True,return_inverse=True)
    nodes=flat[i]; f=rev.reshape(-1,3); n=len(nodes)
    nn=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);normals=np.zeros_like(nodes)
    for j in range(3):np.add.at(normals,f[:,j],nn)
    normals/=np.maximum(1e-12,np.linalg.norm(normals,axis=1)[:,None])
    verts=np.vstack([nodes+normals*thickness*.5,nodes-normals*thickness*.5])
    edges=f[:,[0,1,1,2,2,0]].reshape(-1,2)
    _,first,counts=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
    a,b=edges[first[counts==1]].T
    fs=np.vstack([f,f[:,::-1]+n,np.column_stack((b,a,a+n)),np.column_stack((b,a+n,b+n))])
    return verts,fs
