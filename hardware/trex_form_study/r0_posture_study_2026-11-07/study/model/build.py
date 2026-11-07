"""Build independent, free-base design candidates from a verified baseline XML and matching component ledger."""
from pathlib import Path
import argparse, hashlib, json, math, xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import maximum_filter
from scipy.spatial.transform import Rotation
from mesh import number, panel, loft_grid, shell_mass, combined, inertia_box, ry, clip_opening, thicken_triangles
from shapes import FAMILIES

HERE=Path(__file__).resolve().parent
MM=.001
DENSITY=1010.  # design mass density, unfilled powder-bed PA12, kg/m3
def v(s):
    return np.asarray([float(x) for x in s.split()])
def json_value(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.floating): return float(x)
    raise TypeError(str(x))
def bounds_span(rows,a,b):
    p=np.array(rows,float)
    xs=np.r_[a,p[(p[:,0]>a)&(p[:,0]<b),0],b]
    return np.column_stack([xs,PchipInterpolator(p[:,0],p[:,1:],axis=0)(xs)])
def lookup(root,tag,name):
    return root.find(f".//{tag}[@name='{name}']")
def set_material(xml,name,rgb):
    obj=lookup(xml,'material',name)
    if obj is None: obj=ET.SubElement(xml.find('asset'),'material',name=name)
    obj.set('rgba',number([*rgb,1]))
def set_frame(obj,rot,center,origin):
    """Transform a geometry or site in its own body coordinate system."""
    if 'fromto' in obj.attrib:
        pts=v(obj.get('fromto')).reshape(-1,3)
        obj.set('fromto',number((pts-origin)@rot.T+center))
        return
    obj.set('pos',number(rot@(v(obj.get('pos','0 0 0'))-origin)+center))
    if 'xyaxes' in obj.attrib:
        obj.set('xyaxes',number(v(obj.get('xyaxes')).reshape(2,3)@rot.T))
    elif 'zaxis' in obj.attrib:
        obj.set('zaxis',number(rot@v(obj.get('zaxis'))))
    else:
        old=np.eye(3)
        if 'euler' in obj.attrib:
            old=Rotation.from_euler('XYZ',v(obj.attrib.pop('euler'))).as_matrix()
        if 'quat' in obj.attrib:
            old=Rotation.from_quat(v(obj.attrib.pop('quat'))[[1,2,3,0]]).as_matrix()
        obj.set('quat',number(Rotation.from_matrix(rot@old).as_quat()[[3,0,1,2]]))
def rods(body,name,points,radius,material='frame'):
    for i,(a,b) in enumerate(zip(points,points[1:])):
        ET.SubElement(body,'geom',name=f'{name}_{i}',type='capsule',fromto=number([a,b]),size=number([radius]),material=material,group='2')

class Designer:
    def __init__(self,xml,ledger,family,angle):
        self.xml=xml
        self.asset=xml.find('asset')
        self.f=family
        self.cfg=FAMILIES[family]
        self.angle=angle
        self.parts=defaultdict(list)
        self.covers=[]
        self.pending=[]
        self.reliefs=[]
        self.layout=[]
        self.ledger=ledger
        self.landmarks={}
        self.bodies={b.get('name'):b for b in xml.iter('body')}
        self.thick=self.cfg['thickness']*MM
        self.chest_center=np.array([-.010,0,.085])
        self.chest_rot=ry(angle)
    def chest(self,xyz):
        p=np.asarray(xyz)
        return (p-self.chest_center)@self.chest_rot.T+self.chest_center
    def warp(self,grid):
        # Pelvis remains stationary. A short fixed adapter joins it to the pitched chest.
        x=grid[:,:,0]
        blend=np.clip((x+.041)/.090,0,1)
        blend=blend*blend*(3-2*blend)
        return grid+blend[:,:,None]*(self.chest(grid)-grid)
    def mass(self,body,name,m,pos,I,origin):
        self.parts[body].append((m,np.array(pos),np.array(I)))
        self.layout.append(dict(name=name,body=body,mass_kg=m,pos_m=pos,tensor_kg_m2=I,origin=origin))
    def sheet(self,body,name,grid,material='cover_main',thickness=None,fasteners=2):
        self.pending.append((body,name,grid,material,thickness or self.thick,fasteners))
    def emit_sheet(self,body,name,grid,material,thickness,fasteners):
        vs,fs,tris=panel(grid,thickness)
        original_tris=tris
        for tool,planes in self.keepouts[body]:
            if tool.startswith('vent_') and not name.endswith('_roof'):continue
            prior=tris
            tris=clip_opening(tris,planes)
            if not len(tris): return
            if tris is not prior: self.reliefs.append((name,tool))
        clipped=tris is not original_tris
        if clipped: vs,fs=thicken_triangles(tris,thickness)
        asset=f'surface_{name}'
        ET.SubElement(self.asset,'mesh',name=asset,vertex=number(vs),face=' '.join(map(str,fs.ravel())))
        ET.SubElement(self.bodies[body],'geom',name=f'cover_{name}',type='mesh',mesh=asset,material=material,group='1',contype='0',conaffinity='0')
        mass,p,I=shell_mass(tris,thickness,DENSITY)
        self.mass(body,name,mass,p,I,'integrated_PA12')
        # Captive M2.5 screw + insert + mount allowance 2.5 g each, at opposed panel edges.
        for i,anchor in enumerate([grid[max(0,round(len(grid)*.12)),grid.shape[1]//2],grid[min(len(grid)-1,round(len(grid)*.85)),grid.shape[1]//2]][:fasteners]):
            centers=tris.mean(axis=1)
            pt=centers[np.argmin(np.linalg.norm(centers-anchor,axis=1))]
            q=.0025
            self.mass(body,f'{name}_mount_{i}',q,pt,inertia_box(q,[.004,.004,.003]),'mount_allowance')
        self.covers.append(dict(name=f'cover_{name}',body=body,area_m2=mass/(DENSITY*thickness),wall_mm=thickness*1000,
                                material='PA12',density_kg_m3=DENSITY,mass_kg=mass,fastener_mass_kg=.0025*fasteners,centroid_m=p,inertia_kg_m2=I))
        # Local convex contact patches follow the trimmed surface and project a bounded chord inward.
        # Sampled clearance tests can also use the exact explicit exterior surfaces.
        nx,na=grid.shape[:2]
        length=np.linalg.norm(np.diff(grid[:,na//2],axis=0),axis=1).sum()
        arc=np.linalg.norm(np.diff(grid[nx//2],axis=0),axis=1).sum()
        xs=np.unique(np.round(np.linspace(0,nx-1,max(2,math.ceil(length/.095)+1))).astype(int))
        js=np.unique(np.round(np.linspace(0,na-1,max(2,math.ceil(arc/.067)+1))).astype(int))
        for i,(x0,x1) in enumerate(zip(xs,xs[1:])):
            for j,(j0,j1) in enumerate(zip(js,js[1:])):
                # Outer extent from all original nodes; the convex hull does not bridge an entire hollow cavity.
                ids=np.array([r*na+c for r in range(x0,x1+1) for c in range(j0,j1+1)])
                if clipped:
                    # Keep contact facets in small cells, including polished reveal edges.
                    pts_grid=grid[x0:x1+1,j0:j1+1].reshape(-1,3)
                    mn=pts_grid.min(0)-.002
                    mx=pts_grid.max(0)+.002
                    pick=(tris.mean(1)>=mn).all(1)&(tris.mean(1)<=mx).all(1)
                    if not np.any(pick):continue
                    points,_=thicken_triangles(tris[pick],thickness)
                else:
                    points=vs[np.r_[ids,ids+nx*na]]
                token=f'patch_{name}_{i}_{j}'
                ET.SubElement(self.asset,'mesh',name=token,vertex=number(points))
                ET.SubElement(self.bodies[body],'geom',name=token,type='mesh',mesh=token,group='3',rgba='0 0 0 0',contype='2',conaffinity='1',margin='0')
    def torso(self):
        cfg=self.cfg
        rows=cfg['torso']
        for lbl,a,b,lo,hi in cfg['islands']:
            sub=bounds_span(rows,a,b)*MM
            # Dorsal bridge with two separate flank/lateral access panels; seams follow section changes.
            angles=[(lo,44,'port'),(45,135,'roof'),(136,hi,'starboard')] if lo<40 and hi>140 else [(lo,hi,'roof')]
            for start,end,side in angles:
                g=loft_grid(sub,start,end,steps=max(5,int((end-start)/9)),exponent=.87)
                self.sheet('pelvis',f'{lbl}_{side}',self.warp(g), 'cover_high' if side=='roof' else 'cover_main')
        if self.f=='C':
            # Swept ribs wrap the exposed electronics bay; the space between ribs remains physically open.
            for i,(x,dx) in enumerate([(-26,18),(17,25),(61,28),(106,31),(150,26)]):
                sub=bounds_span(rows,x-4.2,x+4.2)*MM
                for source,side in [((-70,48),'port'),((132,250),'starboard')]:
                    rib=loft_grid(sub,*source,steps=15)
                    rib[:,:,0]+=MM*dx*(1-np.sin(np.radians(np.linspace(*source,16))))[None,:]*.6
                    tip=np.maximum(0,-np.sin(np.radians(np.linspace(*source,16))))
                    rib[:,:,1]*=(1-.30*tip**2)[None,:]
                    rib[:,:,2]+=MM*(8+7*(i==4))*tip[None,:]**2
                    self.sheet('pelvis',f'rib_{i}_{side}',self.warp(rib),'cover_bone',thickness=.002,fasteners=1)
        # The caudal fairing ends just behind a fixed, removable pelvic collar. The 9 mm axial
        # gap at centerline opens for tail yaw. Dynamic checks measure the outer corner sweep.
        tx=cfg['tail_pivot']
        x1=cfg['islands'][0][1]-1
        rear=bounds_span(rows,x1+1,x1+1.01)[0]
        dw=cfg['tail_width'];dd=cfg['tail_depth']
        collar=np.array([[tx-64,dw*.80+6,89-dd*.36,89+dd*.27+4],
                         [tx-51,dw*.89+7,89-dd*.40,89+dd*.38+5],
                         [tx-32,dw+9,89-dd*.47,89+dd*.45+7],
                         [tx-9,dw+11,89-dd*.53,89+dd*.50+6],
                         [x1,rear[1]+2,rear[2]-1,rear[3]+3]])*MM
        for lo,hi,side in [(-45,89,'port'),(91,225,'starboard')]:
            grid=loft_grid(collar,lo,hi,21)
            self.sheet('pelvis',f'caudal_collar_{side}',grid,'cover_main')
        # Fixed undertray carries the tail servo service cover and meets the collar from below.
        tx=cfg['tail_pivot']
        skid=np.array([[tx-48,21,23,43],[tx-28,36,8,54],[tx+4,44,9,54],
                       [tx+33,43,19,62],[tx+55,35,38,67]])*MM
        self.sheet('pelvis','caudal_drive_undertray',loft_grid(skid,204,335,18),'cover_bone',thickness=.0014)
        # The original back rails cantilevered beyond the pelvis. These bent rails remain
        # beneath the changing shoulder breadth and receive the removable panel mounts.
        for sign in (-1,1):
            rr=np.array([[-188,sign*34,106],[-139,sign*54,118],[-64,sign*60,130],[0,sign*64,132],
                         [60,sign*58,129],[119,sign*64,119],[166,sign*61,107]],float)*MM
            points=self.warp(rr[:,None,:])[:,0]
            rods(self.bodies['pelvis'],f'frame_dorsal_{sign}',points,.0047)
            for x in ((-124,-64) if self.f=='A' else ()):
                rods(self.bodies['pelvis'],f'frame_tray_support_{sign}_{x}',
                     [np.array([x,sign*28,45])*MM,np.array([x,sign*44,117])*MM],.0033)
            if self.f!='A':
                pts=[[-135+sign*17.5,-76,23.3],[-135+sign*17.5,80,23.3]]
            else:
                pts=[[-203,sign*19,42],[-66,sign*19,42]]
            rods(self.bodies['pelvis'],f'frame_battery_slide_{sign}',np.array(pts)*MM,.0027)
        for x,w,z in [(-155,48,115),(-48,60,131),(132,62,74)]:
            pp=np.array([[x,-w,z],[x,w,z]],float)*MM
            rods(self.bodies['pelvis'],f'frame_cross_{x}',self.warp(pp[:,None,:])[:,0],.0035)
        for x in (65,165):
            self.landmarks[f'rail_datum_{x}']=self.chest(np.array([x*MM,0,.085]))
    def appendages(self):
        cfg=self.cfg
        # Lower cervical origin follows the front shoulder, upper mount rises in world coordinates.
        base=self.chest(np.array([.164,0,.105]))
        pivot=base+MM*np.array([cfg['neck_run'],0,cfg['neck_rise']])
        self.newhead=pivot
        # Torso-facing neck end extends within the open anterior rim.
        for i,(a,b,lo,hi) in enumerate([(0,.32,-51,231),(.335,1-35/cfg['neck_run'],-78,258)]):
            s=cfg['neck_run']; t=cfg['neck_rise']
            rr=[(-10,48,-32,47),(.22*s,44,-26,.23*t+39),(.46*s,34,.47*t-23,.56*t+32),
                (.71*s,33,.82*t-19,.86*t+32),(s,41,t-22,t+30)]
            sub=bounds_span(rr,-10 if i==0 else a*s,b*s)*MM
            self.sheet('pelvis',f'cervical_{i}',loft_grid(sub,lo,hi,25)+base,'cover_main')
        # A fixed nuchal shingle bridges the bearing from above; side and underside remain open.
        s=cfg['neck_run']; t=cfg['neck_rise']
        nape=np.array([[s*.29,34,t*.40+3,t*.49+34],[s*.48,32,t*.63+8,t*.74+38],
                       [s-33,34,t+19,t+43],[s-30,32,t+21,t+44]])*MM
        self.sheet('pelvis','nuchal_bridge',loft_grid(nape,42,138,12)+base,'cover_high',thickness=.0014)
        for sign in (-1,1):
            pts=[base+[0,sign*.060,.003],base+MM*np.array([cfg['neck_run']*.44,sign*25,cfg['neck_rise']*.53]),pivot+[0,sign*.032,0]]
            rods(self.bodies['pelvis'],f'frame_neck_{sign}',pts,.0045)
        h=cfg['head_depth']; w=cfg['head_width']; L=cfg['head_len']
        head_rows=np.array([(-19,w*.54,-.09*h,.33*h),(4,w*.82,-.12*h,.53*h),(.20*L,w,-.13*h,.60*h),
                           (.39*L,.91*w,-.10*h,.55*h),(.55*L,.72*w,-.16*h,.43*h),(.79*L,.64*w,-.16*h,.36*h),
                           (.96*L,.54*w,-.16*h,.34*h),(L,.34*w,-.09*h,.23*h)])
        # Continuous skull roof narrows from brow to blunt snout. Separate cheek columns leave actual orbital apertures.
        rhead=ry(cfg['head_pitch'])
        def head_panel(name,data,lo,hi,mat='cover_main'):
            self.sheet('head',name,loft_grid(data*MM,lo,hi,steps=max(5,int((hi-lo)/8)),exponent=.73)@rhead.T,mat)
        head_panel('cranial_table',head_rows,29,151,'cover_high')
        for tag,a,b in [('occiput',-18,.18*L),('lacrimal',.31*L,.39*L),('maxilla',.47*L,L)]:
            sub=bounds_span(head_rows,a,b)
            head_panel(f'{tag}_port',sub,-80,28)
            head_panel(f'{tag}_starboard',sub,152,260)
        # The rear orbit is triangulated and has a dark recessed eye inset. These are
        # passive printed inserts and are separate from the RGB-D imager.
        for sign in (-1,1):
            for label,xc,delta,ww in [('postorbital',.205,-.082,.96),('maxillary',.43,.082,.78)]:
                # A narrow triangular cheek strut connects the sides of the fenestrae.
                pts=MM*np.array([[L*(xc-delta*.5),sign*ww*w,h*.40],
                    [L*(xc+delta*.5),sign*(ww-.12)*w,-.07*h]])
                rods(self.bodies['head'],f'frame_{label}_{sign}',pts@rhead.T,.0037,'cover_main')
            eye=MM*np.array([.235*L,sign*(w*.84),h*.27])
            ET.SubElement(self.bodies['head'],'geom',name=f'frame_orbit_inset_{sign}',type='ellipsoid',
               pos=number(rhead@eye),size=number([.013,.006,.010]),material='motor',group='2')
            self.mass('head',f'orbit_insert_{sign}',.0035,rhead@eye,inertia_box(.0035,[.013,.006,.01]),'painted_PA12_insert')
            brow=MM*np.array([[.12*L,sign*.87*w,.36*h],[.22*L,sign*.97*w,.48*h],[.31*L,sign*.92*w,.44*h]])
            rods(self.bodies['head'],f'frame_orbital_brow_{sign}',brow@rhead.T,.004,'cover_main')
        # Blunt, continuous anterior cap; the nose slopes very slightly backward at the chin.
        theta=np.radians(np.linspace(-82,263,28))
        ring=[]
        for factor in (1,.70,.3,.02):
            ring.append(np.column_stack((np.full_like(theta,L+2*(1-factor)),.34*w*factor*np.cos(theta),
                         .07*h+.16*h*factor*np.sin(theta))))
        self.sheet('head','nasal_cap',(np.array(ring)*MM)@rhead.T,'cover_main',thickness=.0012,fasteners=0)
        # Both lateral cheek arches are structural design details, tied to the skull mount allowance.
        for sign in (-1,1):
            p=MM*np.array([[.085*L,sign*w*.70,-4],[.18*L,sign*w*.86,-.14*h],[.38*L,sign*w*.79,-.12*h],[.52*L,sign*w*.6,-.10*h]])
            rods(self.bodies['head'],f'frame_jugal_{sign}',p@rhead.T,.004,'rim')
        jawscale=1.0 if self.f=='B' else .88
        jaw=np.array([(-.03*L,.46*w,-.26*h,-.04*h),(.07*L,.72*w,-.47*h,.03*h),
                     (.20*L,.81*w,-.56*h,-.04*h),(.32*L,.84*w,-.55*h,-.12*h),(.46*L,.76*w,-.48*h,-.18*h),
                     (.66*L,.63*w,-.40*h,-.23*h),(.91*L,.52*w,-.34*h,-.21*h),(.992*L,.26*w,-.26*h,-.17*h)])
        jaw[:,2]*=jawscale
        # Skull-side cheek flanges give the rear cranium a load path under the open orbit.
        for sign in (-1,1):
            cheek=np.array([[.07*L,.79*w,-.22*h,.11*h],[.17*L,.89*w,-.22*h,.15*h],
                        [.28*L,.87*w,-.18*h,.07*h],[.38*L,.82*w,-.16*h,.09*h],[.51*L,.72*w,-.13*h,.11*h]])
            grid=[]
            for x,width,low,high in cheek:
                z=np.linspace(low,high,5)
                grid.append(np.column_stack([np.full(5,x),sign*(width-2*(np.linspace(-1,1,5)**2)),z]))
            self.sheet('head',f'cheek_flange_{sign}',(np.array(grid)*MM)@rhead.T,'cover_main',thickness=.0013)

        # Jaw opens superiorly so local cavity and fixed bite line remain visible in section.
        head_panel('mandible',jaw,158,382,'cover_bone')
        for sign in (-1,1):
            for j,(u,depth) in enumerate([(.52,7),(.61,8),(.71,7),(.80,6),(.89,5)]):
                # Rounded short tooth guards. Dimensional mass allowance is included explicitly.
                _,ww,bot,top=bounds_span(head_rows,u*L,u*L+.02)[0]
                p0=MM*np.array([u*L,sign*.74*ww,bot+3]); p1=p0+MM*np.array([-2,0,-depth])
                rods(self.bodies['head'],f'frame_tooth_{sign}_{j}',[rhead@p0,rhead@p1],.0016,'jaw')
                mass=DENSITY*np.pi*.0016**2*np.linalg.norm(p1-p0)
                self.mass('head',f'tooth_{sign}_{j}',mass,rhead@((p0+p1)/2),inertia_box(mass,[.002,.002,.004]),'PA12_guard')
        self.headcrown=rhead@(MM*np.array([.2*L,0,.60*h]))
        self.snout=rhead@(MM*np.array([L,0,.10*h]))
        # Tail yaw axis vertical; independent ventral curve is molded into the spar and shrouds.
        dl=cfg['tail_len']; dw=cfg['tail_width']; dd=cfg['tail_depth']; drop=cfg['tail_drop']
        rr=[]
        for u,wid,depth in [(1,.020,.025),(.84,.095,.10),(.60,.22,.24),(.36,.44,.47),(.16,.69,.68),(.035,.89,.90),(0,1,1)]:
            z=-drop*u-cfg['tail_curve']*math.sin(math.pi*u)
            rr.append([-dl*u,max(1.4,wid*dw),z-depth*dd/2,z+depth*dd/2])
        for i,(a,b) in enumerate([(-dl,-.61*dl),(-.607*dl,-.235*dl),(-.232*dl,0)]):
            sub=bounds_span(rr,a,b)*MM
            for start,end,side in [(-84,90,'port'),(91,264,'starboard')]:
                g=loft_grid(sub,start,end,18)
                g[:,:,0]-=.46*abs(g[:,:,1])*np.clip(1+g[:,:,0]/.115,0,1)**2
                self.sheet('tail',f'caudal_{i}_{side}',g,'cover_main',thickness=.0012,fasteners=1)
        mid=np.array([(r[0],0,(r[2]+r[3])/2) for r in rr])*MM
        rods(self.bodies['tail'],'frame_tail_spar',mid,.0038)
        self.tailtip=mid[0]
        self.tail_spar_points=mid
    def thighs(self):
        for side,sgn in [('left',1),('right',-1)]:
            body=self.bodies[side+'_thigh']
            end=v(body.find("body").get('pos'))
            grid=[]
            for t,w in [(0.055,14),(.17,.84*self.cfg['thigh']),(.37,self.cfg['thigh']),(.60,.75*self.cfg['thigh']),(.80,10)]:
                center=end*t
                theta=np.radians(np.linspace(-76,76,14))
                # Medial rails remain serviceable; a single lateral shield visually tapers into the knee.
                xyz=np.column_stack((w*MM*np.sin(theta),sgn*(.011+.015*np.cos(theta)),.006*np.sin(theta)))+center
                grid.append(xyz)
            self.sheet(side+'_thigh',side+'_femoral_blade',np.asarray(grid),'cover_main',thickness=.0015)
    def prepare(self):
        """Identify the metalwork and purchased parts first; replace named coverings explicitly."""
        self.removed=[]
        self.deleted_terms=('side_panel','rib_mount','dorsal_panel','ventral_panel')
        pelvis=self.bodies['pelvis']; head=self.bodies['head']; tail=self.bodies['tail']
        self.oldhead=v(head.get('pos')); self.oldtail=v(tail.get('pos'))
        # Baseline shell names and material-based cleanup are recorded so no historical inertia is silently reused.
        for body in self.bodies.values():
            for g in list(body.findall('geom')):
                name=g.get('name','')
                if (body is head and name!='neck_output_shaft') or (body is tail and name!='tail_output_shaft') or (
                    any(k in name for k in self.deleted_terms) or
                    ((g.get('type')=='mesh' or 'mesh' in g.attrib) and body is not head and body is not tail) or
                    (g.get('material') in ['skin','jaw'] and 'arm' not in name) or
                    name.startswith(('spine_', 'spine_cross', 'rear_battery_hanger', 'rear_battery_rail', 'pelvis_web', 'neck_rail','neck_fork','tail_root','tail_fork','camera_brace','sensor_strut'))):
                    body.remove(g); self.removed.append(name)
        # The crest of each torso station falls above the retained hip-yaw package.
        set_material(self.xml,'cover_main',self.cfg['color']); set_material(self.xml,'cover_high',self.cfg['accents'])
        set_material(self.xml,'cover_bone',(.65,.62,.49))
        self.torso(); self.appendages(); self.thighs()
        for item in ['mesh','texture']:
            used={g.get(item) for g in self.xml.iter('geom')}
            if item=='mesh':
                for a in list(self.asset.findall(item)):
                    if a.get('name') not in used: self.asset.remove(a)
        # Relative installed part motions are rigid per assembly. The pelvis/leg axes and source stance geometry stay fixed.
        I=np.eye(3)
        self.newtail=np.array([self.cfg['tail_pivot']*MM,0,.089])
        head.set('pos',number(self.newhead)); tail.set('pos',number(self.newtail))
        for body,site,pos in [(head,'head_crown',self.headcrown),(head,'snout',self.snout),(tail,'tail_tip',self.tailtip)]:
            marker=body.find(f"site[@name='{site}']")
            if marker is not None: marker.set('pos',number(pos))
        # Scan original joint-to-joint positions, then join the new servo fork to the nearest carried frame station.
        for sign in [-1,1]:
            h=self.newhead
            rods(pelvis,f'frame_head_bearing_yoke_{sign}',[h+[-.053,sign*.044,-.037],h+[0,sign*.043,0]],.005)
            t=self.newtail
            rods(pelvis,f'frame_tail_bearing_yoke_{sign}',[t+[.011,sign*.042,.035],t+[0,0,sign*.030]],.005)
        # If the purchased board bounding boxes are smaller than supplier values, expand the visuals in-place.
        self.box_changes={}
        for name,minimum in {'jetson_kit':[103.2,90.7,35.86],'logic_converter':[17.78,12.70,3.556],
                             'regulated_12v_ttl_case':[43.18,31.75,9.017]}.items():
            g=lookup(self.xml,'geom',name)
            if g is not None and g.get('type')=='box':
                old=v(g.get('size')); revised=np.maximum(old,np.array(minimum)*.0005)
                if np.any(revised>old+1e-8):
                    self.box_changes[name]=dict(prior_full_mm=old*2000,revised_full_mm=revised*2000)
                    g.set('size',number(revised))
        self.placements={}
        self.configure_hardware()
        for g in list(pelvis):
            n=g.get('name','')
            for key,terms in self.transform_lists.items():
                if any(n.startswith(prefix) for prefix in terms):
                    origin,center,rot=self.placements[key]
                    set_frame(g,rot,center,origin)
                    break
        for sign,side in [(-1,'right'),(1,'left')]:
            cam=lookup(self.xml,'geom','oak_d_lite')
            tof=lookup(self.xml,'geom','tof_'+side+'_carrier')
            if cam is not None:
                rods(pelvis,'frame_camera_brace_'+side,
                     [self.chest(np.array([.131,sign*.066,.09])),v(cam.get('pos'))+[-.006,sign*.040,0]],.003)
            if tof is not None:
                rods(pelvis,'frame_sensor_strut_'+side,[self.chest(np.array([.151,sign*.069,.065])),v(tof.get('pos'))],.0035)
        self.fit_caudal_collar()
        self.keepouts=self.make_keepouts()
        for entry in self.pending: self.emit_sheet(*entry)
        self.account_components()
    def fit_caudal_collar(self):
        # Form the fixed sleeve around the swept tail, including both sides of its yaw axis.
        # Index swept radii by x and transverse angle. Each bin takes the furthest surface sample,
        # grows into neighbouring bins, and receives a further 6.5 mm radial service clearance.
        bins=np.zeros((151,180))
        center=self.newtail
        for body,name,grid,material,t,fasteners in self.pending:
            if body!='tail':continue
            _,_,tri=panel(grid,t)
            cloud=[]
            for a in np.linspace(0,1,9):
                for b in np.linspace(0,1-a,max(2,round((1-a)*8)+1)):
                    cloud.append(tri[:,0]+a*(tri[:,1]-tri[:,0])+b*(tri[:,2]-tri[:,0]))
            points=np.vstack(cloud)
            for angle in np.linspace(-.38,.38,45):
                c=np.cos(angle);s=np.sin(angle)
                x=points[:,0]*c-points[:,1]*s
                y=points[:,0]*s+points[:,1]*c;z=points[:,2]
                selected=x>-.112
                xi=np.clip(np.floor((x[selected]+.2)/.002).astype(int),0,150)
                phi=(np.floor(np.degrees(np.arctan2(z[selected],y[selected]))/2).astype(int)+180)%180
                np.maximum.at(bins,(xi,phi),np.hypot(y[selected],z[selected]))
        bins=maximum_filter(np.pad(bins,((0,0),(4,4)),mode='wrap'),size=(5,7),mode='nearest')[:,4:-4]
        for body,name,grid,material,t,fasteners in self.pending:
            if not name.startswith('caudal_collar_'):continue
            local=grid-center
            x,y,z=local.transpose(2,0,1)
            ix=np.clip(np.floor((x+.2)/.002).astype(int),0,150)
            ph=(np.floor(np.degrees(np.arctan2(z,y))/2).astype(int)+180)%180
            radius=np.maximum(np.hypot(y,z),1e-9)
            scale=np.maximum(1,(bins[ix,ph]+.0065)/radius)
            grid[:,:,1]=center[1]+y*scale
            grid[:,:,2]=center[2]+z*scale
    def make_keepouts(self):
        import mujoco as mj
        model=mj.MjModel.from_xml_string(ET.tostring(self.xml,encoding='unicode'))
        d=mj.MjData(model);mj.mj_forward(model,d)
        keep={}
        for body in set(e[0] for e in self.pending):
            poses=[(0.,0.)]
            if body=='head': poses += [(k,0.) for k in [-.25,-.13,.14,.28]]
            if body=='tail': poses += [(0.,k) for k in [-.38,-.19,.19,.38]]
            tools=[]
            for pitch,yaw in poses:
              d.qpos[model.joint('neck_pitch').qposadr[0]]=pitch
              d.qpos[model.joint('tail_yaw').qposadr[0]]=yaw
              mj.mj_forward(model,d)
              ix=model.body(body).id; r=d.xmat[ix].reshape(3,3); o=d.xpos[ix]
              for i in range(model.ngeom):
                n=model.geom(i).name or ''
                if not (n.endswith('_case') or n in ['compute_reserve','battery_pack','oak_d_lite',
                        'drive_disconnect_reserve','tail_motor_cable_reserve','neck_motor_cable_reserve',
                        'protection_distribution','power_reserve','regulated_12v_ttl_reserve']
                        or n.startswith(('neck_bearing','tail_bearing')) or n.endswith('output_shaft')
                        or 'hip_yoke' in n or n.startswith('frame_head_bearing_yoke') or n.startswith('frame_tail_bearing_yoke')):
                    continue
                typ=model.geom_type[i];q=d.geom_xpos[i]; R=d.geom_xmat[i].reshape(3,3)
                margin=.004
                if typ==mj.mjtGeom.mjGEOM_BOX:
                    norm=np.vstack([np.eye(3),-np.eye(3)])
                    h=np.tile(model.geom_size[i]+margin,2)
                elif typ in (mj.mjtGeom.mjGEOM_CYLINDER,mj.mjtGeom.mjGEOM_CAPSULE):
                    nn=16; angles=np.arange(nn)*2*np.pi/nn
                    norm=np.vstack([np.c_[np.cos(angles),np.sin(angles),np.zeros(nn)],[0,0,-1],[0,0,1]])
                    hh=model.geom_size[i,1]+margin+(model.geom_size[i,0] if typ==mj.mjtGeom.mjGEOM_CAPSULE else 0)
                    h=np.r_[np.full(nn,model.geom_size[i,0]+margin),hh,hh]
                else:continue
                nbody=norm@R.T@r; pbody=r.T@(q-o)
                tools.append((n,(nbody,h+nbody@pbody)))
            if body=='pelvis':
                # Five open, longitudinal grille slots directly above the carrier fan.
                for vi,y in enumerate([-.030,-.015,0,.015,.030]):
                    nr=np.vstack([np.eye(3),-np.eye(3)])@self.chest_rot.T
                    center=self.chest(np.array([.106,y,.177]))
                    tools.append((f'vent_{vi}',(nr,np.tile([.035,.0035,.040],2)+nr@center)))
            keep[body]=tools
        return keep
    def configure_hardware(self):
        pel=self.bodies['pelvis']
        def loc(n,default):
            g=lookup(self.xml,'geom',n)
            return v(g.get('pos')) if g is not None else np.array(default,dtype=float)
        I=np.eye(3)
        battery=loc('battery_pack',[-.17,0,.04]); target=np.array([-.137 if self.f!='B' else -.145,0,.068])
        compute=loc('jetson_kit',[.105,0,.11]); cputgt=self.chest(np.array([.107,0,.103]))
        self.placements.update(
            battery=(battery,target,I) if self.f=='A' else (battery,np.array([-.135,0,.057]),Rotation.from_euler('z',90,degrees=True).as_matrix()),compute=(compute,cputgt,self.chest_rot),
            disconnect=(loc('drive_disconnect_case',[-.225,-.060,.065]),np.array([-.100,-.063,0.]),I),
            neck=(self.oldhead,self.newhead,I),tail=(self.oldtail,self.newtail,I),
            arms=(self.chest_center,self.chest_center,self.chest_rot),
            camera=(loc('oak_d_lite',[.21,0,.004]),np.array([.210,0,-.008]),I))
        # Range mount stays below the sternum, clear of the fixed arms; preserve calibrated axes.
        for side,sgn in [('left',1),('right',-1)]:
            p=loc(f'tof_{side}_carrier',[.205,sgn*.065,-.1])
            self.placements['tof_'+side]=(p,np.array([.201,sgn*.059,-.051]),I)
        self.transform_lists={
            'battery':('battery_reserve','battery_pack','battery_tray','rear_battery_rail'),
            'disconnect':('drive_disconnect','estop_disconnect'),
            'compute':('compute_reserve','jetson_kit','jetson_fan','jetson_sink'),
            'neck':('neck_motor_case','neck_motor_cable','neck_bearing'),
            'tail':('tail_motor_case','tail_motor_cable','tail_bearing'),
            'arms':('fixed_arm','arm_claw'), 'camera':('oak_d_lite','rgbd_reserve','front_rgbd'),
            'tof_left':('tof_left',),'tof_right':('tof_right',)}
    def redrawn_frame(self,budget):
        # Carry the frame procurement mass budget; use the new tube centerlines and real tray
        # locations for mass properties. All tubing dimensions are provisional build assumptions.
        tubemass=0.
        for g in self.bodies['pelvis'].findall('geom'):
            name=g.get('name','')
            if not name.startswith(('frame_dorsal_','frame_cross_','frame_battery_slide','frame_tray_support')):continue
            a,b=v(g.get('fromto')).reshape(2,3);ro=v(g.get('size'))[0];ri=max(0.,ro-.0012)
            L=np.linalg.norm(b-a);u=(b-a)/L;uu=np.outer(u,u)
            m=2700*np.pi*(ro**2-ri**2)*L
            I=m*((L**2/12)*(np.eye(3)-uu)+(ro**2+ri**2)/4*(np.eye(3)+uu))
            self.mass('pelvis',name,m,(a+b)/2,I,'6061_tube_1.2mm_wall');tubemass+=m
        residual=budget-tubemass
        if residual<=0:raise ValueError('Redrawn rails exceed source frame allowance')
        for key,fraction,half in [('battery',.40,[.078,.029,.004]),('compute',.35,[.057,.051,.004]),('power',.25,[.069,.037,.007])]:
            if key=='power':
                p=np.array([-.02,0,-.024]);r=np.eye(3)
            else:
                _,p,r=self.placements[key]
                p=p+r@np.array([0,0,-.028 if key=='battery' else -.023])
            m=residual*fraction
            self.mass('pelvis',f'{key}_tray_and_frame_fittings',m,p,r@inertia_box(m,half)@r.T,'remaining_source_frame_budget')
    def account_components(self):
        self.replaced_masses=[]
        for c in self.ledger['components']:
            b=c['body']; n=c['name']; m=c['mass_kg']; pos=np.array(c['pos_m']); half=np.array(c['inertia_box_halfsize_m'])
            # New thin-surface integration supersedes every earlier geometric covering allocation.
            if (c['category']=='covers_brackets' or n=='skull_and_jaw_cover' or n=='tail_spar_shell' or
                any(w in n for w in ('shroud','fairing','skin','shell','torso_panel','thigh_cover','shin_cover'))):
                self.replaced_masses.append((n,m));continue
            if n=='pelvis_spine_trays':
                self.redrawn_frame(m)
                continue
            rot=np.eye(3); key=None
            if b=='pelvis':
                if n=='battery': key='battery'
                elif n=='compute': key='compute'
                elif n=='drive_disconnect': key='disconnect'
                elif n.startswith('neck_'): key='neck'
                elif n.startswith('tail_'): key='tail'
                elif n.startswith('arm_'): key='arms'
                elif n.startswith('oak_'): key='camera'
                elif n.startswith('tof_'): key='tof_left' if 'left' in n else 'tof_right'
            if key:
                orig,dest,rot=self.placements[key]
                pos=rot@(pos-orig)+dest
            self.mass(b,n,m,pos,rot@inertia_box(m,half)@rot.T,'retained_source_allocation')
        # Carbon spar replacement, OD 8 mm, bore 6 mm, 1550 kg/m3, plus two 7 g end fittings.
        a=math.pi*(.004**2-.003**2)
        pts=self.tail_spar_points
        for i,(p,q) in enumerate(zip(pts,pts[1:])):
            d=q-p; L=np.linalg.norm(d); axis=d/L; mass=1550*a*L
            I=mass*(L**2/12*(np.eye(3)-np.outer(axis,axis))+(.004**2+.003**2)/4*(np.eye(3)+np.outer(axis,axis)))
            self.mass('tail',f'carbon_tube_{i}',mass,(p+q)/2,I,'tube_model')
        for i,pt in enumerate([pts[-1],pts[2]]):
            self.mass('tail',f'spar_fitting_{i}',.007,pt,inertia_box(.007,[.008,.007,.007]),'fitting_allowance')
        for b,parts in self.parts.items():
            m,p,I=combined(parts)
            el=self.bodies[b].find('inertial')
            if el is None: el=ET.SubElement(self.bodies[b],'inertial')
            el.attrib.clear(); el.set('mass',number([m])); el.set('pos',number(p)); el.set('fullinertia',number(I[(0,1,2,0,0,1),(0,1,2,1,2,2)]))
        self.xml.set('model',f'rex_form_study_{self.f}{self.angle:02d}')
    def report(self):
        return dict(id=f'{self.f}{self.angle:02}',rev='2',family=self.cfg['name'],chest_angle_deg=self.angle,
                    parameters_mm=self.cfg, pelvis_landmarks_m=self.landmarks,head_pitch_deg=self.cfg['head_pitch'],
                    head_pivot_pelvis_m=self.newhead,tail_pivot_pelvis_m=self.newtail,
                    fitted_box_updates=self.box_changes,positions={k:dict(from_m=o,to_m=p,rotation=r) for k,(o,p,r) in self.placements.items()},
                    reliefs=self.reliefs,removed_geoms=self.removed,replaced_allocations=self.replaced_masses,covers=self.covers,components=self.layout,
                    installed_mass_kg=sum(c['mass_kg'] for c in self.layout),shell_kg=sum(c['mass_kg'] for c in self.covers),
                    fastener_kg=sum(c['fastener_mass_kg'] for c in self.covers))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True,help='Source MJCF XML')
    parser.add_argument('--ledger',type=Path,required=True,help='Matching source component ledger')
    parser.add_argument('--out',type=Path,default=HERE/'generated')
    parser.add_argument('--families',default='ABC'); parser.add_argument('--angles',type=int,nargs='+',default=[0,10,20])
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    ledger=json.loads(args.ledger.read_text())
    for f in args.families:
        for angle in args.angles:
            root=ET.parse(args.source).getroot(); design=Designer(root,ledger,f,angle)
            design.prepare(); data=design.report()
            tag=data['id']; ET.indent(root)
            out=args.out/(tag+'.xml'); ET.ElementTree(root).write(out,encoding='utf-8',xml_declaration=True)
            data['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
            data['ledger_sha256']=hashlib.sha256(args.ledger.read_bytes()).hexdigest()
            data['xml_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
            (args.out/(tag+'.json')).write_text(json.dumps(data,indent=1,default=json_value))
            print(tag, data['installed_mass_kg'], data['shell_kg'])
if __name__=='__main__':
    main()
