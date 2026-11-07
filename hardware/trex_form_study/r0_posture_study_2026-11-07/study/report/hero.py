from pathlib import Path
import sys
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'visuals'))
from scene import Scene, render, camera
scene=Scene(HERE.parent/'model/generated/B10.xml')
pts=scene.triangles(False)[0].reshape(-1,3)@camera('threequarter')[:2].T
lo,hi=pts.min(0),pts.max(0)
size=(1400,1040)
scale=min((np.array(size)-24)/(hi-lo))
center=((hi+lo)/2)@camera('threequarter')[:2]
fg=render(scene,'threequarter',size,scale,center)
paper=Image.new('RGBA',size,'#fcfbf7')
paper.alpha_composite(fg);paper.convert('RGB').save(HERE/'hero.png')
