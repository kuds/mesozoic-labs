"""Build the landscape review book. Uses reportlab and PIL; no office suite or internet."""
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import argparse
import json
import reportlab
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
W,H=900,650
PAPER="#FCFBF7"; INK="#173533"; GREEN="#31665A"; STROKE="#CBCFC4"; MUTED="#475F58"; CLAY="#96542F"
FONTS=Path(reportlab.__file__).parent/"fonts"
for name,file in [("Vera","Vera.ttf"),("VeraB","VeraBd.ttf"),("VeraI","VeraIt.ttf")]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))
def C(value):return colors.HexColor(value)

class Book:
    def __init__(self,path):
        self.c=Canvas(str(path),pagesize=(W,H),pageCompression=1)
        self.c.setTitle("Intermediate T. rex | Body and posture study")
        self.c.setAuthor("Mesozoic Labs design study")
        self.page=0
    def label(self,x,y,s,size=9,color=MUTED,font="VeraB"):
        self.c.setFillColor(C(color));self.c.setFont(font,size);self.c.drawString(x,H-y-size,s)
    def rule(self,x,y,end):
        self.c.setStrokeColor(C(STROKE));self.c.setLineWidth(.65);self.c.line(x,H-y,end,H-y)
    def new(self,section,title=None):
        self.page+=1
        c=self.c
        c.setFillColor(C(PAPER));c.rect(0,0,W,H,fill=1,stroke=0)
        self.label(34,16,"MESOZOIC LABS     /     INTERMEDIATE TYRANNOSAURUS REX",8)
        c.setFillColor(C(MUTED));c.setFont("Vera",8);c.drawRightString(W-34,H-24,section)
        self.rule(34,34,W-34)
        if title:self.label(34,45,title,28,INK,"Times-Roman")
        self.rule(34,H-27,W-34);self.label(34,H-20,"Body shape and posture study   •   7 November 2026",7.4)
        c.setFont("Vera",7.5);c.drawRightString(W-34,11,str(self.page))
    def text(self,x,y,w,s,size=12,color=INK,font="Vera",leading=None):
        p=Paragraph(s,ParagraphStyle("p",fontName=font,fontSize=size,leading=leading or size*1.40,textColor=C(color),alignment=TA_LEFT,spaceAfter=0,allowWidows=0,allowOrphans=0))
        _,ht=p.wrap(w,10000);p.drawOn(self.c,x,H-y-ht);return y+ht
    def block(self,x,y,w,title,s,size=11.5):
        y=self.text(x,y,w,escape(title),size+1,GREEN,"VeraB",size*1.28+1)+5
        return self.text(x,y,w,s,size)+14
    def image(self,path,x,y,w,h,crop=None):
        with Image.open(path) as im:
            if crop:im=im.crop(crop)
            im=im.convert("RGB");iw,ih=im.size
            scale=min(w/iw,h/ih);sw,sh=iw*scale,ih*scale
            # Convert once to JPEG at high quality to keep the review small.
            buf=BytesIO();im.save(buf,format="JPEG",quality=95,subsampling=0);buf.seek(0)
            self.c.drawImage(ImageReader(buf),x+(w-sw)/2,H-y-(h-sh)/2-sh,sw,sh)
    def table(self,x,y,width,columns,rows,col_widths=None,font=10.2):
        fractions=col_widths or [1/len(columns)]*len(columns)
        widths=[width*q for q in fractions]
        def line(row,head=False,index=0):
            nonlocal y
            textfont="VeraB" if head else "Vera"
            lead=font*1.33
            style=ParagraphStyle("cell",fontName=textfont,fontSize=font,leading=lead,textColor=C(PAPER if head else INK))
            ps=[Paragraph(str(v),style) for v in row]
            hs=[p.wrap(w-12,10000)[1] for p,w in zip(ps,widths)]
            rh=max(hs)+12
            self.c.setFillColor(C(GREEN if head else ("#EEF1EA" if index%2==0 else PAPER)))
            self.c.rect(x,H-y-rh,width,rh,fill=1,stroke=0)
            left=x
            for p,w,ph in zip(ps,widths,hs):
                p.drawOn(self.c,left+6,H-y-(rh-ph)/2-ph);left+=w
            y+=rh
        line(columns,True)
        for i,r in enumerate(rows):line(r,False,i)
        return y
    def finish(self):self.c.showPage()
    def save(self):self.c.save()

def draw(book, config):
    for page in config:
        book.new(page["section"],page.get("title"))
        for item in page["items"]:
            typ=item[0]
            if typ=="text":
                _,x,y,w,txt,*args=item
                end=book.text(x,y,w,txt,*args)
            elif typ=="label":
                book.label(*item[1:]);end=0
            elif typ=="blocks":
                _,x,y,w,size,blocks=item
                end=y
                for title,s in blocks:end=book.block(x,end,w,title,s,size)
            elif typ=="image":
                _,path,*args=item
                book.image(ROOT/path,*args);end=0
            elif typ=="table":
                end=book.table(*item[1:])
            elif typ=="rule":
                book.rule(*item[1:]);end=0
            if end>H-34:raise ValueError(f"Content below footer on p{book.page}: {typ} at {end:.1f}")
        book.finish()
    book.save()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=ROOT/"T_rex_body_study.pdf")
    args=ap.parse_args()
    book=Book(args.output)
    draw(book,json.loads((HERE/"pages.json").read_text()))
