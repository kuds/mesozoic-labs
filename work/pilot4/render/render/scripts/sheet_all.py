"""One thumbnail per video at frame FRAME, tiled into one PNG (render lane; scratch; venv's mediapy and PIL only).

    python -I -B sheet_all.py OUT.png FRAME COLS VIDEO.mp4 [VIDEO.mp4 ...]

Streams each file with mediapy.VideoReader up to FRAME (nothing else held), shrinks the frame to a quarter of its
width and height, and tiles them in the order given, COLS per row. Prints each file's thumbnail position.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mediapy
import numpy as np
from PIL import Image


def frame_at(path: str, index: int) -> np.ndarray:
    with mediapy.VideoReader(path) as reader:
        for i, frame in enumerate(reader):
            if i == index:
                return np.asarray(frame)
    raise SystemExit(f"{path} has no frame {index}")


def main() -> int:
    out, index, cols, videos = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    thumbs = []
    for v in videos:
        f = frame_at(v, index)
        thumbs.append(np.asarray(Image.fromarray(f).resize((f.shape[1] // 4, f.shape[0] // 4), Image.BILINEAR)))
    h, w = thumbs[0].shape[:2]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, t in enumerate(thumbs):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
        print(f"row {r} col {c}: {Path(videos[k]).name}")
    Image.fromarray(sheet).save(out)
    print(f"wrote {out} ({rows}x{cols}, frame {index})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
