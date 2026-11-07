"""Annotate the published R2 sheet. Run: python audit_r2.py /path/to/redesign-at-60cm.png

Requires Pillow. The overlay uses source pixels and calibrated projected lengths.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
args=parser.parse_args()
original=Image.open(args.source).convert('RGB')
if original.size!=(1328,1270):
    raise ValueError("Expected the original 1328 × 1270 PR #663 comparison sheet.")
out=Path(__file__).resolve().parent
paper=(245,243,237)
ink=(22,46,50)
muted=(65,91,92)
rust=(165,77,36)
green=(45,109,87)
line=(175,187,178)
sf=2
page=Image.new('RGB',(1490*sf,1030*sf),paper)
draw=ImageDraw.Draw(page)
def font(size,bold):
    options=(['/System/Library/Fonts/Supplemental/Arial Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','DejaVuSans-Bold.ttf'] if bold else ['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','DejaVuSans.ttf'])
    for opt in options:
        try:
            return ImageFont.truetype(opt,size*sf)
        except OSError:
            continue
    return ImageFont.load_default(size=size*sf)
def tx(x,y,t,size=18,color=ink,bold=False,anchor='lt'):
    draw.text((x*sf,y*sf),t,font=font(size,bold),fill=color,anchor=anchor)
def ln(points,color,width=1):
    draw.line([(round(x*sf),round(y*sf)) for x,y in points],fill=color,width=width*sf,joint='curve')
def circle(x,y,r,fill):
    draw.ellipse(((x-r)*sf,(y-r)*sf,(x+r)*sf,(y+r)*sf),fill)
def add_crop(crop,x,y,scale):
    img=original.crop(crop)
    pix=(round(img.width*scale*sf),round(img.height*scale*sf))
    page.paste(img.resize(pix,Image.Resampling.LANCZOS),(round(x*sf),round(y*sf)))
    return lambda px,py:(x+(px-crop[0])*scale,y+(py-crop[1])*scale)
def callout(n,xy,offset):
    dot=(xy[0]+offset[0],xy[1]+offset[1])
    ln([xy,dot],rust,2)
    circle(*xy,3,rust)
    circle(*dot,12,rust)
    tx(*dot,str(n),13,(255,255,255),True,'mm')
def br(x0,x1,y,t,color):
    ln([(x0,y-5),(x0,y+5)],color,1)
    ln([(x1,y-5),(x1,y+5)],color,1)
    ln([(x0,y),(x1,y)],color,1)
    tx((x0+x1)/2,y+9,t,16,color,True,'mt')
tx(52,37,'MESOZOIC LABS  /  PR #663',15,green,True)
tx(52,71,'What gives the R2 body its shape?',44,ink,True)
tx(54,133,'Published R2 exterior and geometric-home pose. The three images are crops of the original comparison sheet.',20,muted)
ln([(52,170),(1438,170)],line)
tx(54,193,'Side',23,ink,True)
tx(814,193,'Three-quarter',23,ink,True)
left=add_crop((710,214,1249,516),52,227,1.27)
right=add_crop((755,566,1213,855),821,222,1.26)
for n,p,o in [(1,(975,337),(-43,38)),(2,(1027,250),(-10,-22)),(3,(1105,275),(14,-30)),(4,(1030,345),(-31,30))]:
    callout(n,left(*p),o)
for n,p,o in [(5,(1033,584),(-13,-23)),(6,(1082,816),(33,21))]:
    callout(n,right(*p),o)
ln([(52,622),(1438,622)],line)
tx(54,646,'Two different tail measurements',23,ink,True)
# Small silhouette cropped just above feet; annotated with a rightward ruler.
small=add_crop((718,928,1243,1075),51,690,1.12)
x0=small(730,1029)[0]; xs=small(920,998)[0]; xh=small(1033,1051)[0]; xe=small(1233,969)[0]
for xx,yy in [(x0,small(730,1029)[1]),(xs,small(920,994)[1]),(xh,small(1033,1050)[1])]:
    for y in range(round(yy)+4,856,8):
        ln([(xx,y),(xx,y+3)],line)
br(x0,xs,858,'~38% to rear shell seam',rust)
br(x0,xh,902,'~60% to hip axle',green)
tx(53,955,'Pixel estimates relative to the 1,104 mm projected length.',16,muted)
tx(53,978,'The seam and axle each differ from the biological tail root.',16,muted)
notes=[('Rear lobe','The lower shell drops behind the hip.'),('Dome','The back crests close to the hip axis.'),('Neck junction','A deep V separates lid and neck.'),('Hip joint','A rectilinear drive projects into the hip opening.'),('Trunk width','The broad lid visually ties the panels together.'),('Foot','Toe accents end on a rectangular pad.')]
for i,(a,b) in enumerate(notes):
    y=654+55*i
    circle(837,y+8,12,rust);tx(837,y+8,str(i+1),13,(255,255,255),True,'mm')
    tx(862,y-3,a,18,ink,True);tx(862,y+20,b,16,muted)
page.resize((1490,1030),Image.Resampling.LANCZOS).save(out/'r2_visual_audit.png')
