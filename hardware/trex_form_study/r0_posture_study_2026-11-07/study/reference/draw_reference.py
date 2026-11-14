"""Original qualitative landmark diagram. Run from any directory with Python 3.

Creates reference.svg. Rasterize with Inkscape or the sharp Node package:
  inkscape reference.svg --export-filename=reference.png --export-width=1600
  node -e "require('sharp')('reference.svg').png().toFile('reference.png')"

"""
from pathlib import Path
from html import escape

out = Path(__file__).resolve().parent
ink, muted, teal, gold = '#16312e', '#52655f', '#28695c', '#b95832'
p = []
def path(d, fill='none', stroke=ink, width=2, extra=''):
    p.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" {extra}/>')
def text(x,y,s,size=18,color=ink,weight=400,anchor='start',space='0'):
    p.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" font-weight="{weight}" letter-spacing="{space}" text-anchor="{anchor}">{escape(s)}</text>')
def ring(x,y,r,fill,stroke=ink,sw=1.5):
    p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def leader(points,x,y):
    path(points,stroke=gold,width=1.65); ring(x,y,3.7,gold,stroke=gold,sw=1)

p.append('<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1030" viewBox="0 0 1600 1030"><rect width="1600" height="1030" fill="#f6f5ed"/>')
text(62,45,'MESOZOIC LABS  /  BODY STUDY',14,teal,700,space='2.4')
text(62,99,'Landmarks that make the silhouette read',43,ink,700)
text(64,136,'Original interpretive schematic. The outlines and joint positions are qualitative. The ratios below come from specimen reconstructions.',19,muted)
path('M62 161H1538',stroke='#aab8ab',width=1.2)
# Main diagram is authored in the facing-right direction and is independent of the robot model.
# Far leg, to give context to the weight-bearing near leg.
path('M799 348 C805 377 791 395 786 410 L810 450 Q817 459 807 468 L710 531 L725 565 L767 569 Q777 570 778 575 L721 575 Q715 575 710 569 L683 531 Q680 525 690 516 L761 453 L720 421 Q701 391 713 359 Z','#e3e7dc','#a7b7ab',1.6)
# Upper soft-tissue guide extending from tip through tail root, back, neck and skull.
path('M80 415 C163 416 237 391 332 362 C450 326 564 300 678 281 C734 271 798 272 858 279 C916 280 978 280 1024 289 C1059 296 1095 299 1118 284 C1144 268 1152 238 1175 227 C1195 218 1214 222 1233 236 L1247 264 C1217 274 1211 287 1195 303 C1173 325 1141 336 1112 345 C1087 355 1075 372 1054 388 C1027 407 993 424 946 435 C902 445 871 433 848 412 C832 398 802 389 784 377 C749 358 705 344 676 340 C572 328 457 347 356 371 C252 396 159 422 80 415Z','#dbe5d8','#50776a',2.6)
# Femoral volume, merge with pelvic flank against a gently rising belly.
path('M747 315 C766 298 812 306 831 337 C844 355 849 379 867 399 C879 416 881 432 874 445 C863 463 843 471 826 460 C809 445 803 424 785 416 C762 403 737 387 724 357 C718 339 723 325 747 315Z','#becfbd','#537b6b',2.1)
path('M865 443 C864 457 850 477 830 492 L785 531 L783 540 L769 541 Q765 535 771 525 L808 469 L826 446 Z','#cedbcc','#537b6b',1.9)
path('M776 530 C779 532 786 535 793 544 L811 578 L863 581 Q874 583 876 588 L815 588 L792 588 Q784 588 783 581 L766 543Z','#cad7c9','#537b6b',1.9)
path('M811 577 L834 572 L863 574 M812 584 L839 590 L860 591',stroke='#537b6b',width=1.6)
# Baseline drawn only as a witness at the feet.
path('M650 593H923',stroke='#91a69a',width=1.4)
# Soft-tissue over skull, visually distinct posterior head and snout, and a separate jaw.
path('M1219 251 Q1222 232 1231 224 L1244 221 L1259 216 L1287 221 L1300 232 L1350 239 L1380 244 Q1403 246 1405 254 L1400 269 L1389 275 L1338 280 L1297 288 L1263 292 L1237 281Z','#c1d2c1','#3f6e60',2.4)
path('M1236 273 Q1250 287 1291 283 L1338 277 L1397 269 L1396 277 Q1376 293 1329 298 L1287 306 L1266 309 Q1249 311 1238 296Z','#c9d7c8','#3f6e60',2)
path('M1278 232 Q1281 225 1289 229 Q1290 236 1281 238Z',fill=teal,stroke=teal,width=1)
path('M1376 251 Q1380 247 1385 250',stroke=teal,width=2)
# Axial line and tapered segmented caudal landmark marks.
path('M90 415 C200 410 292 377 370 356 C485 326 596 304 708 291 C768 284 824 284 900 291 C961 294 1036 293 1070 302 C1104 309 1138 297 1155 270 C1167 250 1178 239 1190 236 C1207 231 1216 236 1233 244',stroke=teal,width=4)
for n,(x,y) in enumerate([(194,401),(245,388),(297,373),(348,359),(397,345),(447,331),(497,318),(546,307),(595,298),(640,292),(681,286)]):
    h=3+n*0.67
    path(f'M{x} {y+6}l-3 {h}',stroke='#6c9280',width=1.5)
# Original costal landmarks, indication of gastral basket and scapular line.
for x0,y0,x1,y1 in [(827,291,859,401),(858,292,894,421),(893,293,928,429),(928,294,958,424),(961,296,988,414),(993,298,1015,402),(1023,300,1038,388),(1048,303,1057,373)]:
    path(f'M{x0} {y0} C{x0-2} {y0+35} {x1-15} {y1-15} {x1} {y1}',stroke='#7c9c88',width=2.4)
for i in range(8):
    x=875+i*18; y=409 + (12 if 1<=i<=4 else 4) - (max(i-4,0)*5)
    path(f'M{x-7} {y-5}l13 10 8 -9',stroke='#90aa97',width=1.5)
path('M1050 351 Q1027 336 996 326',stroke=teal,width=3)
# Ilium, hip cup and kinematic landmarks. Fixed forearm and two short fingers.
path('M714 309 Q757 287 816 299 L837 309 L806 313 L784 324 L760 317 L716 320Z','#c5d5c3',teal,2.4)
path('M770 331 Q797 368 851 435 L777 535 L802 581 L855 585',stroke=teal,width=3.7)
for x,y in [(770,331),(851,435),(777,535),(802,581)]:
    ring(x,y,5.3,'#f4f5eb',teal,2.2)
path('M1053 351 Q1063 363 1049 380 L1057 395 L1076 405 M1058 393 L1073 395 L1081 398',stroke=teal,width=3)
ring(1052,351,5.2,'#f4f5eb',teal,2.2)
# Callouts set outside body.
text(95,207,'Tail root and taper',22,ink,700);text(95,233,'Carry depth back from the pelvis.',17,muted)
leader('M262 245V270L549 327',549,327)
text(622,207,'Pelvis and thigh',22,ink,700);text(622,233,'The leg enters the flank.',17,muted)
leader('M752 245V265L773 319',773,319)
text(1080,199,'Curved neck',22,ink,700)
leader('M1140 204V231L1151 271',1151,271)
text(1310,332,'Deep skull',22,ink,700)
leader('M1304 323L1272 303',1272,303)
text(935,492,'Ribcage and gastralia',22,ink,700);text(935,518,'Keep depth under the ribs.',17,muted);text(935,540,'Recover height near the hip.',17,muted)
leader('M958 467V452L972 418',972,418)
text(420,504,'Knee forward, hock raised',22,ink,700)
text(420,530,'Show three leg segments clearly.',17,muted)
leader('M617 545H714L767 536',767,536)
path('M62 624H1538',stroke='#aab8ab',width=1.2)
# Lower-left plan of skull and separate sections
text(63,669,'Skull in plan',27,ink,700)
text(63,701,'A broad rear skull gives way to a narrower snout.',17,muted)
path('M98 769 Q101 745 118 740 L157 741 Q183 743 203 757 L230 761 L279 761 Q297 761 300 776 Q298 794 280 795 L229 794 L201 799 Q180 811 156 812 L120 814 Q104 811 98 790Z','#d5e2d4',teal,2.4)
path('M103 777H298',stroke='#7e9a87',width=1.5,extra='stroke-dasharray="5 5"')
text(97,845,'REAR SKULL',12,teal,700,space='1.2');text(263,845,'SNOUT',12,teal,700,space='1.2')
path('M543 654V894',stroke='#ccd4c8',width=1)
text(578,669,'Three trunk sections',27,ink,700)
text(578,701,'Independent width, depth and lower contour.',17,muted)
for cx,top,bot,w,label in [(641,730,813,43,'CHEST'),(771,746,803,33,'ABDOMEN'),(910,737,795,40,'PELVIS')]:
    cy=(top+bot)/2
    path(f'M{cx} {top}C{cx+w*.75} {top-1} {cx+w} {top+12} {cx+w} {cy} C{cx+w} {bot-8} {cx+17} {bot} {cx} {bot} C{cx-17} {bot} {cx-w} {bot-8} {cx-w} {cy} C{cx-w} {top+12} {cx-w*.75} {top-1} {cx} {top}Z','#e4eadd',teal,2)
    path(f'M{cx} {top-4} V{bot+4}',stroke='#99aa98',width=1,extra='stroke-dasharray="4 4"')
    text(cx,845,label,12,teal,700,anchor='middle',space='1.1')
path('M1011 654V894',stroke='#ccd4c8',width=1)
text(1046,669,'Two adult reconstructions',27,ink,700)
text(1046,701,'Fraction of full skeletal length¹',17,muted)
text(1390,742,'SUE',15,teal,700,'end');text(1517,742,'STAN',15,teal,700,'end')
path('M1046 755H1518',stroke='#ccd4c8',width=1)
for y,label,a,b in [(784,'Shoulder–hip spacing','22.5%','21.1%'),(818,'Free tail length','51.2%','59.4%'),(852,'Skeletal leg length','26.9%','26.0%')]:
    text(1046,y,label,17);text(1390,y,a,18,ink,700,'end');text(1517,y,b,18,ink,700,'end')
path('M62 895H1538',stroke='#aab8ab',width=1.2)
text(64,927,'¹ Hutchinson et al. (2011), table 3. Tail begins at the first free caudal vertebra. Leg lengths are the published values divided by full skeletal length; robot hip height uses a different measurement.',15,muted)
text(64,956,'Sources: Field Museum, SUE remount (2018); Hutchinson et al., PLOS ONE 6:e26037, figs. 2–4, 7; Hurum & Sabath, Acta Palaeontol. Pol. 48 (2003), 161–190.',15,muted)
text(64,984,'The section outlines and posture are design interpretations. No museum photographs, fossil meshes or published illustrations are reproduced.',15,muted)
p.append('</svg>')
(out/'reference.svg').write_text('\n'.join(p)+'\n')
