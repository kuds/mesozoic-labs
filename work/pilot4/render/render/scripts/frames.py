"""Extract PNG frames from an mp4 with the venv's mediapy (decoding) and PIL (PNG writing); render lane, scratch.

    python -I -B frames.py VIDEO.mp4 OUT_DIR TAG FRAME[,FRAME...] [--sheet COLS]

Writes OUT_DIR/TAG_f<NNNN>.png for each named frame index (0-based, decoded order) and, with --sheet, one contact
sheet OUT_DIR/TAG_sheet.png (the named frames at half size, COLS per row). Prints each frame's index, time at the
file's fps, shape and mean RGB. Imports no repository code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mediapy
import numpy as np
from PIL import Image


def main() -> int:
    video, out_dir, tag, idx = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
    cols = int(sys.argv[sys.argv.index("--sheet") + 1]) if "--sheet" in sys.argv else 0
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = mediapy.read_video(str(video))
    fps = float(frames.metadata.fps) if getattr(frames, "metadata", None) is not None else float("nan")
    print(f"{video.name}: {len(frames)} decoded frames, {frames.shape[2]}x{frames.shape[1]}, fps {fps:g}")
    picked = []
    for i in idx:
        f = np.asarray(frames[i])
        Image.fromarray(f).save(out_dir / f"{tag}_f{i:04d}.png")
        picked.append(f)
        print(f"  frame {i} (t = {i / fps:.2f} s in the file): mean RGB {f.reshape(-1, 3).mean(axis=0).round(1).tolist()}")
    if cols:
        h, w = picked[0].shape[0] // 2, picked[0].shape[1] // 2
        rows = (len(picked) + cols - 1) // cols
        sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for k, f in enumerate(picked):
            small = np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR))
            r, c = divmod(k, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = small
        Image.fromarray(sheet).save(out_dir / f"{tag}_sheet.png")
        print(f"  sheet {out_dir / (tag + '_sheet.png')} ({rows}x{cols})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
