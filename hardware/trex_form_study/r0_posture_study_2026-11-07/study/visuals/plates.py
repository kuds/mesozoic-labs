"""Build the geometry review plates from the supplied R2, R3 and candidate MJCFs."""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scene import Scene, camera, projected, render

PAPER='#f8f6f0'; INK='#182e29'; MUTED='#4e655d'; RULE='#cad1c7'; ACCENT='#a24326'
PATH=Path(__file__).resolve().parent
NAMES={'P662':'PR #662 nominal platform','R2':'R2 comparison model','R3':'R3 walking platform',
       'A':'Tapered flank','B':'Deep ribcage','C':'Open rib frame'}
FAMILIES='ABC'
VIEWS=['side','threequarter','top','front']
VIEW_NAMES={'side':'Side elevation','threequarter':'Front three-quarter','top':'Dorsal view','front':'Front elevation'}


@lru_cache
def face(size,kind='regular'):
    fonts={'regular':['Arial.ttf','DejaVuSans.ttf'],
           'bold':['Arial Bold.ttf','DejaVuSans-Bold.ttf'],
           'serif':['Georgia.ttf','DejaVuSerif.ttf'],
           'mono':['Courier New Bold.ttf','DejaVuSansMono.ttf']}[kind]
    for f in fonts:
        for folder in ['/System/Library/Fonts/Supplemental','/usr/share/fonts/truetype/dejavu','']:
            try: return ImageFont.truetype(str(Path(folder)/f),size)
            except OSError: pass
    return ImageFont.load_default(size=size)


def label(draw,xy,string,sz=26,ink=INK,kind='regular',anchor=None):
    draw.text(xy,string,fill=ink,font=face(sz,kind),anchor=anchor)


def page(size,title,subtitle):
    pg=Image.new('RGB',size,PAPER)
    d=ImageDraw.Draw(pg)
    label(d,(44,28),'MESOZOIC LABS    /    INTERMEDIATE TYRANNOSAURUS',19,MUTED,'bold')
    label(d,(42,59),title,53,INK,'serif')
    label(d,(44,128),subtitle,24,MUTED)
    d.line((44,173,size[0]-44,173),fill=RULE,width=2)
    return pg


def footer(pg,lines):
    d=ImageDraw.Draw(pg); w,h=pg.size
    y=h-len(lines)*31-32
    d.line((42,y-13,w-42,y-13),fill=RULE,width=2)
    for line in lines:
        label(d,(44,y),line,22,MUTED); y+=31


def scale_bar(d,x,y,scale):
    length=round(.10*scale)
    d.line((x,y,x+length,y),fill=INK,width=3)
    for xx in (x,x+length): d.line((xx,y-5,xx,y+5),fill=INK,width=2)
    label(d,(x+length/2,y+11),'100 mm',18,MUTED,anchor='ma')


def section_name(code):
    return NAMES[code if code in ('R2','R3','P662') else code[0]]


def station(code):
    return 'published pose' if code in ('R2','R3','P662') else str(int(code[1:]))+'° chest up'


def spec(scene):
    lo,hi=scene.max_size()
    return dict(height=scene.crown*1000,hip=scene.hip[2]*1000,
                length=(hi[0]-lo[0])*1000,width=(hi[1]-lo[1])*1000,
                mass=scene.mass,com=scene.com*1000)



class Plates:
    def __init__(self,models,out):
        self.scenes=models
        self.out=out
        self.out.mkdir(parents=True,exist_ok=True)
        self.bounds={}
        for view in VIEWS:
            lo,hi=[],[]
            for scene in self.scenes.values():
                verts=scene.triangles(False)[0].reshape(-1,3)@camera(view)[:2].T
                lo.append(verts.min(axis=0));hi.append(verts.max(axis=0))
            self.bounds[view]=(np.min(lo,axis=0),np.max(hi,axis=0))

    def scale(self,width,height,views):
        return min(min((width-38)/(self.bounds[v][1][0]-self.bounds[v][0][0]),
                       (height-32)/(self.bounds[v][1][1]-self.bounds[v][0][1])) for v in views)

    def center(self,view):
        lo,hi=self.bounds[view]
        return ((lo+hi)/2)@camera(view)[:2]

    def paste(self,pg,scene,view,box,scale,*,cut=False,sil=False,center=None,mark=False,ground=False):
        x,y,w,h=box
        ctr=self.center(view) if center is None else center
        im=render(scene,view,(w,h),scale,ctr,cutaway=cut,silhouette=sil)
        d=ImageDraw.Draw(pg)
        if ground and view in ('front','side'):
            floor_y=round(projected([0,0,0],view,scale,(x+w/2,y+h/2),ctr)[1])
            d.line((x+9,floor_y+1,x+w-9,floor_y+1),fill=RULE,width=2)
        pg.paste(im,(x,y),im)
        if mark:
            uv=projected(scene.com,view,scale,(x+w/2,y+h/2),ctr)
            self.target(d,uv)
        return lambda point: projected(point,view,scale,(x+w/2,y+h/2),ctr)

    @staticmethod
    def target(draw,point):
        u,v=point
        r=8
        draw.ellipse((u-r,v-r,u+r,v+r),fill=PAPER,outline=ACCENT,width=3)
        draw.line((u-r-5,v,u+r+5,v),fill=ACCENT,width=2)
        draw.line((u,v-r-5,u,v+r+5),fill=ACCENT,width=2)

    def save(self,pg,name):
        pg.save(self.out/name,optimize=True)

    def four_views(self,code):
        scene=self.scenes[code]; s=spec(scene)
        w,h=2240,1620
        title=(f'{code}   {section_name(code)}' if not code in ('R2','R3','P662') else section_name(code))
        pg=page((w,h),title,
                f'{station(code)[0].upper()+station(code)[1:]}   ·   Orthographic exterior   ·   Installed hardware visible through openings')
        d=ImageDraw.Draw(pg)
        bw,bh=1065,539
        sc=self.scale(bw,bh,VIEWS)
        for index,view in enumerate(VIEWS):
            x=44+(index%2)*1090; y=204+(index//2)*616
            label(d,(x,y),VIEW_NAMES[view],26,INK,'bold')
            self.paste(pg,scene,view,(x,y+38,bw,bh),sc,mark=view in ['front','side'],ground=True)
        x,y=164,1426
        scale_bar(d,46,y+12,sc)
        label(d,(x,y),f'Crown {s["height"]:.0f} mm    •    Hip {s["hip"]:.0f} mm    •    Length {s["length"]:.0f} mm    •    Width {s["width"]:.0f} mm    •    Installed mass {s["mass"]:.3f} kg',30,INK)
        com=', '.join(f'{c:.1f}' for c in s['com'])
        footer(pg,[f'Crosshair = whole-model center of mass. World (x, y, z) = ({com}) mm.   +x forward, +y anatomical left, z = 0 at the floor.',
                   f'Model SHA-256 {scene.sha}   •   All four views share one physical scale.'])
        self.save(pg,f'{code}_four_views.png')

    def family_board(self):
        baseline='R3' if 'R3' in self.scenes else 'P662'
        keys=[baseline,'A10','B10','C10']
        pg=page((2240,1520),'Three ways to build the torso',
                f'Front three-quarter at a common physical scale   ·   {section_name(baseline)} in geometric home   ·   All candidate chests at 10°')
        d=ImageDraw.Draw(pg);bw,bh=1066,496;sc=self.scale(bw,bh,['threequarter'])
        for i,code in enumerate(keys):
            scene=self.scenes[code]; s=spec(scene)
            x=47+1090*(i%2);y=203+604*(i//2)
            label(d,(x,y),section_name(code),36,INK,'serif')
            label(d,(x+bw,y+7),code,27,ACCENT,'bold','ra')
            self.paste(pg,scene,'threequarter',(x,y+45,bw,bh),sc)
            label(d,(x,y+552),f'{s["length"]:.0f} mm long    /    {s["height"]:.0f} mm crown    /    {s["mass"]:.3f} kg',23,MUTED)
        scale_bar(d,49,1438,sc)
        label(d,(175,1420),'Same light, orientation and model coordinates. Individual four-view plates identify the center of mass and model revision.',23,MUTED)
        self.save(pg,'family_exteriors.png')

    def silhouettes(self,source):
        keys=[k for k in ['R2','R3','P662','A10','B10','C10'] if k in self.scenes]
        if 'R2' not in keys:
            keys.insert(0,'R2_print')
        src=Image.open(source).convert('RGB')
        rgba=np.zeros((278,510,4),np.uint8)
        roi=np.asarray(src.crop((727,933,1237,1211)),dtype=float)
        # Invert the antialiasing of the solid silhouette against its native
        # background using the red channel, then blacken it consistently.
        rgba[:,:,:3]=(19,23,21)
        rgba[:,:,3]=np.rint(255*np.clip((245-roi[:,:,0])/(245-22),0,1))
        printed=Image.fromarray(rgba)
        native_scale=273/.5998
        bw,bh=940,365
        common=self.scale(bw,bh,['side']); based=list(self.scenes.values())
        target=max(spec(s)['height'] for s in based)*.001
        pg=page((2240,290+(bh+83)*len(keys)),'Silhouettes at two scales',
                'Each row uses the same model and pose in both columns. Left preserves size. Right scales the figure uniformly to the same crown height.')
        d=ImageDraw.Draw(pg)
        label(d,(265,193),'Common physical scale',28,INK,'bold')
        label(d,(1260,193),'Equal crown height',28,INK,'bold')
        for i,code in enumerate(keys):
            scene=self.scenes.get(code)
            s=spec(scene) if scene else {'height':599.8}
            y=238+i*(bh+83)
            x=42
            label(d,(x,y+22),'R2' if code=='R2_print' else code,32,ACCENT,'bold')
            short={'R2_print':'PR plate','P662':'Nominal','R2':'Comparison','R3':'Walking','A':'Tapered','B':'Deep','C':'Open'}
            label(d,(x,y+74),short.get(code,short.get(code[0])),24)
            if code[0] in FAMILIES:
                label(d,(x,y+107),'10°',24,MUTED)
            label(d,(x,y+156),f'{s["height"]:.0f} mm',22)
            label(d,(x,y+188),'crown',19,MUTED)
            for j in range(2):
                mult=target/(s['height']*.001) if j else 1
                local=common*mult
                # Normalize about the floor, keeping hip x=0 aligned in both columns.
                ctr=self.center('side').copy()
                if j: ctr/=mult
                if scene:
                    self.paste(pg,scene,'side',(234+j*1000,y,bw,bh+10),local,sil=True,center=ctr,ground=True)
                else:
                    zoom=local/native_scale
                    obj=printed.resize((round(printed.width*zoom),round(printed.height*zoom)),Image.Resampling.LANCZOS)
                    xleft=234+j*1000
                    fy=round(projected([0,0,0],'side',local,(xleft+bw/2,y+(bh+10)/2),ctr)[1])
                    d.line((xleft+9,fy+1,xleft+bw-9,fy+1),fill=RULE,width=2)
                    pg.paste(obj,(round(xleft+bw/2-obj.width/2),round(fy-(1209-933)*zoom)),obj)
                if j: label(d,(2070,y+bh+13),f'× {mult:.3f}',20,MUTED,anchor='ra')
            if i<len(keys)-1: d.line((44,y+bh+58,2184,y+bh+58),fill=RULE,width=2)
        footer(pg,['R2 is the published #663 silhouette calibrated to 599.8 mm. Other rows use compiled physical geometry. The floor is the common alignment datum.'])
        self.save(pg,'silhouettes.png')

    def postures(self):
        w,h=2260,1850
        pg=page((w,h),'Chest inclination and planted stance',
                'The feet stay on the floor. The neck, skull and tail have independent mounting angles. Crosshairs mark the whole-model center of mass.')
        d=ImageDraw.Draw(pg);bw,bh=692,400;sc=self.scale(bw,bh,['side'])
        for j,code in enumerate(['00','10','20']):
            x=44+735*j
            label(d,(x,200),f'{int(code)}° chest up',34,INK,'serif')
        for i,f in enumerate(FAMILIES):
            for j,ang in enumerate(['00','10','20']):
                code=f+ang; scene=self.scenes[code]; s=spec(scene)
                x=44+735*j;y=268+i*495
                label(d,(x,y),f'{code}   /   {section_name(code)}',22,ACCENT,'bold')
                self.paste(pg,scene,'side',(x,y+30,bw,bh),sc,mark=True,ground=True)
                label(d,(x,y+bh+38),
                      f'Crown {s["height"]:.0f}  ·  Hip {s["hip"]:.0f} mm  ·  {s["mass"]:.3f} kg',23,MUTED)
        footer(pg,['All nine poses use a common orthographic scale and camera origin. Torso angle is measured along the defined chest bracket in the sagittal plane.'])
        self.save(pg,'posture_matrix.png')

    def pr_comparison(self,source):
        """Compare the published R2 side view directly with the three candidate sides."""
        src=Image.open(source).convert('RGB')
        if src.size!=(1328,1270):
            raise ValueError('Expected the native 1328 × 1270 redesign-at-60cm.png from PR #663')
        # R2 occupies the upper-right side panel. All renderer text is outside
        # this crop. Set alpha from the flat background, preserving every
        # model pixel, including narrow open spaces between mechanical parts.
        crop=np.asarray(src.crop((710,215,1255,515))).astype(float)
        bg=np.asarray(src.getpixel((1300,300)),dtype=float)
        diff=np.max(np.abs(crop-bg),axis=-1)
        rgba=np.zeros(crop.shape[:2]+(4,),np.uint8)
        rgba[:,:,:3]=np.rint(crop).astype(np.uint8)
        rgba[:,:,3]=np.round(255*np.clip(diff/8,0,1)).astype(np.uint8)
        original=Image.fromarray(rgba)
        # 273 pixels span the exterior crown to the sole in the published
        # native panel. The report gives R2 crown = 599.8 mm.
        native_scale=273/.5998
        yfloor_original=503-215
        pg=page((2260,1550),'From the published R2 to three alternatives',
                'Side elevation at common physical scale   ·   Feet aligned on each row   ·   All three candidate chests at 10°')
        d=ImageDraw.Draw(pg);bw,bh=1060,500
        scale=self.scale(bw,bh,['side'])
        for i,code in enumerate(['R2','A10','B10','C10']):
            x=48+1100*(i%2);y=212+614*(i//2)
            label(d,(x,y),section_name(code),36,INK,'serif')
            label(d,(x+bw,y+4),code,26,ACCENT,'bold','ra')
            # A common height datum fixes every floor at the same row offset.
            floor=y+bh+20
            d.line((x+8,floor,x+bw-8,floor),fill=RULE,width=2)
            if code=='R2':
                zoom=scale/native_scale
                obj=original.resize((round(original.width*zoom),round(original.height*zoom)),Image.Resampling.LANCZOS)
                origin=(round(x+bw/2-obj.width/2),round(floor-yfloor_original*zoom))
                pg.paste(obj,origin,obj)
                caption='Original PR #663 image   /   599.8 mm crown   /   10.282 kg'
            else:
                scene=self.scenes[code]
                lo,hi=scene.max_size()
                ctr=np.array([(lo[0]+hi[0])/2,0,(floor-(y+40+bh/2))/scale])
                self.paste(pg,scene,'side',(x,y+40,bw,bh),scale,center=ctr)
                caption=f'{scene.crown*1000:.0f} mm crown   /   {(hi[0]-lo[0])*1000:.0f} mm long   /   {scene.mass:.3f} kg'
            label(d,(x,y+bh+50),caption,23,MUTED)
        scale_bar(d,49,1460,scale)
        label(d,(170,1443),'R2 is cropped from the PR plate and calibrated to its published crown height. Candidate images use compiled MJCF coordinates.',23,MUTED)
        self.save(pg,'pr663_side_comparison.png')

    def published_baselines(self,source,r3_image):
        original=Image.open(source).convert('RGB')
        r2=original.crop((731,554,1223,862))
        r3=Image.open(r3_image).convert('RGB')
        if r3.size!=(904,584): raise ValueError('Unexpected R3 partner-brief image')
        pg=page((2160,1230),'Two revisions in PR #663',
                'The redesign comparison shows R2. The partner brief shows the R3 walking platform. Both images below are taken from the published artifacts.')
        d=ImageDraw.Draw(pg)
        items=[('R2','Redesigned body',r2,'10.282 kg installed   ·   599.8 mm crown',
                'Source  /  redesign-at-60cm.png, right three-quarter view.',
                'Compact shape, 214 mm sole spacing.'),
               ('R3','Walking platform',r3,'10.370 kg installed   ·   599.8 mm crown',
                'Source  /  Partner-brief.pdf, embedded three-quarter view.',
                'Fixed arms remounted on the rails; soles narrowed to 175 mm.')]
        for j,(code,title,im,metrics,src,note) in enumerate(items):
            x=42+j*1056; label(d,(x,204),code,30,ACCENT,'bold')
            label(d,(x,246),title,38,INK,'serif')
            im=im.copy()
            ratio=min(988/im.width,625/im.height)
            im=im.resize((round(im.width*ratio),round(im.height*ratio)),Image.Resampling.LANCZOS)
            pg.paste(im,(int(x+(1000-im.width)/2),int(305+(625-im.height)/2)))
            label(d,(x,961),metrics,28,INK,'bold')
            label(d,(x,1013),src,23,MUTED)
            label(d,(x,1052),note,23,MUTED)
        footer(pg,['Each source retains its native camera. Images have been fitted independently for inspection and cannot be used for dimensional measurement.',
                   'Model SHA-256 prefixes: R2 857d189b8a0d   /   R3 87e36ef3cffc. Exact comparisons to newly generated shapes use dedicated physical-scale plates.'])
        self.save(pg,'published_baselines.png')

    def cutaways(self):
        pg=page((2240,2050),'What the coverings conceal',
                'Candidate chests at 10°   ·   Same physical scale and view in both columns   ·   Cooling, cable and removal reserve boxes hidden')
        d=ImageDraw.Draw(pg); bw,bh=1050,470;sc=self.scale(bw,bh,['threequarter'])
        label(d,(44,200),'Assembled exterior',30,INK,'bold')
        label(d,(1140,200),'Coverings removed',30,INK,'bold')
        for i,code in enumerate(['A10','B10','C10']):
            scene=self.scenes[code];y=267+i*550
            label(d,(48,y),f'{code}   /   {section_name(code)}',24,ACCENT,'bold')
            self.paste(pg,scene,'threequarter',(48,y+38,bw,bh),sc)
            self.paste(pg,scene,'threequarter',(1140,y+38,bw,bh),sc,cut=True)
            d.line((46,y+521,2190,y+521),fill=RULE,width=2)
        footer(pg,['Cutaways retain the modeled internal structure, drives, components and brackets. A removed face exposes its attachment geometry where modeled.',
                   'Use the component layout and service-path tests for access decisions. A visible component can still be obstructed along its extraction direction.'])
        self.save(pg,'hardware_cutaways.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=PATH.parent/'model'/'generated')
    baseline=parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument('--r3',type=Path)
    baseline.add_argument('--pr662',type=Path)
    parser.add_argument('--r2',type=Path)
    parser.add_argument('--pr-image',type=Path,default=PATH.parent.parent/'source/pr663/visual/redesign-at-60cm.png')
    parser.add_argument('--out',type=Path,default=PATH/'plates')
    args=parser.parse_args()
    paths={'R3':args.r3} if args.r3 else {'P662':args.pr662}
    if args.r2: paths={'R2':args.r2,**paths}
    paths.update({f+a:args.models/(f+a+'.xml') for f in FAMILIES for a in ['00','10','20']})
    scenes={k:Scene(v) for k,v in paths.items()}
    plates=Plates(scenes,args.out)
    for k in ['A10','B10','C10',*[c for c in scenes if c not in ('A10','B10','C10')]]:
        plates.four_views(k)
        print('Rendered',k,flush=True)
    plates.family_board()
    plates.pr_comparison(args.pr_image)
    plates.published_baselines(args.pr_image,args.pr_image.with_name('published-r3-partner-view.png'))
    plates.silhouettes(args.pr_image)
    plates.postures()
    plates.cutaways()
    (args.out/'render_manifest.json').write_text(json.dumps(
        {k:{'sha256':s.sha,'camera_pose':'qpos0','physical_mass_kg':s.mass,
            'extent_mm':(1000*(s.max_size()[1]-s.max_size()[0])).tolist(),
            'com_mm':(1000*s.com).tolist()} for k,s in scenes.items()},indent=2)+'\n')


if __name__=='__main__':
    main()
