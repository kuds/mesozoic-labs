"""Digitise element extents from a published figure with a scale bar.

Thresholds the (dark-on-white) photograph, labels connected components,
finds the thin vertical scale bar, and reports each component's bounding box
and vertical extent (in cm via the scale bar).  Approximate: photographs
carry perspective and the vertical extent is only the proximodistal length
when the element is photographed with its long axis vertical.

Usage: python -I lit_digitise_fig.py <png> <scale_cm> <out_json> [threshold]
"""

import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

png, scale_cm, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
thr = int(sys.argv[4]) if len(sys.argv) > 4 else 200

im = np.asarray(Image.open(png).convert("L")).astype(np.int32)
mask = im < thr
mask = ndimage.binary_opening(mask, iterations=1)
mask = ndimage.binary_fill_holes(mask)
lab, n = ndimage.label(mask)
objs = ndimage.find_objects(lab)
comps = []
for i, sl in enumerate(objs, start=1):
    h = sl[0].stop - sl[0].start
    w = sl[1].stop - sl[1].start
    area = int((lab[sl] == i).sum())
    if area < 400:
        continue
    comps.append({"id": i, "y0": sl[0].start, "y1": sl[0].stop, "x0": sl[1].start, "x1": sl[1].stop,
                  "h_px": h, "w_px": w, "area": area, "fill": area / float(h * w)})
# scale bar: tall, very thin, nearly fully filled
bars = [c for c in comps if c["h_px"] > 8 * c["w_px"] and c["w_px"] < 25 and c["fill"] > 0.6]
bars.sort(key=lambda c: -c["h_px"])
res = {"png": png, "image_shape": list(im.shape), "threshold": thr, "n_components": len(comps)}
if bars:
    bar = bars[0]
    px_per_cm = bar["h_px"] / scale_cm
    res["scale_bar"] = bar
    res["px_per_cm"] = px_per_cm
    for c in comps:
        c["h_cm"] = c["h_px"] / px_per_cm
        c["w_cm"] = c["w_px"] / px_per_cm
res["components"] = sorted([c for c in comps if c not in bars[:1]], key=lambda c: (c["x0"] // 300, c["y0"]))
json.dump(res, open(out, "w"), indent=1)
print("px_per_cm", res.get("px_per_cm"), "bar", res.get("scale_bar"))
for c in res["components"]:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()})
