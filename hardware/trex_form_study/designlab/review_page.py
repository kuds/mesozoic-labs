"""Build the round-one review page (HTML + web-sized images) from the generated data and plates.

Run from hardware/trex_form_study after stage_a, round1 and report:
    python -m designlab.review_page --round1 round1 --out <dir>
Every number on the page is read from round1/review_data.json (written by designlab.report) or the
review-set screening files; every image is a plate rendered from the compiled models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from . import page_kit as K

HEIGHTS = (60, 75, 100, 120)
FAMILY = {"T": "Tapered", "S": "Substantial", "H": "Hybrid"}
WIDTH = {"N": "Narrow", "M": "Balanced", "W": "Broad"}
PART_NAMES = {
    "myactuator_rmd_x4_p36_36_e_l": "MyActuator X4-36-L", "myactuator_rmd_x4_p36_36_e": "MyActuator X4-36",
    "myactuator_rmd_x4_p12_10_e": "MyActuator X4-10", "myactuator_rmd_x6_p20_60_e": "MyActuator X6-60",
    "myactuator_rmd_x8_p20_120_e": "MyActuator X8-120", "myactuator_rmd_x8_p20_150_e": "MyActuator X8-150",
    "myactuator_rmd_x10_p20_200_e": "MyActuator X10-200", "myactuator_rmd_x12_p20_320_e": "MyActuator X12-320",
    "cubemars_ak45_36_v3": "CubeMars AK45-36", "cubemars_ak45_10_v3": "CubeMars AK45-10",
    "cubemars_ak60_39_v3": "CubeMars AK60-39", "cubemars_ak70_9_v3": "CubeMars AK70-9", "cubemars_ak80_64": "CubeMars AK80-64",
    "cubemars_ak80_9_v3": "CubeMars AK80-9", "cubemars_ak80_8": "CubeMars AK80-8", "cubemars_ak10_9_v3": "CubeMars AK10-9",
    "cubemars_ak60_6_v3": "CubeMars AK60-6", "cubemars_ak70_10": "CubeMars AK70-10",
    "robstride_00": "RobStride 00", "robstride_01": "RobStride 01", "robstride_02": "RobStride 02",
    "robstride_03": "RobStride 03", "robstride_04": "RobStride 04", "robstride_05": "RobStride 05", "robstride_06": "RobStride 06",
    "feetech_sts3215_c018": "Feetech STS3215", "feetech_sts3250": "Feetech STS3250",
    "robotis_xm540_w270_t": "ROBOTIS XM540-W270", "robotis_xm540_w150_t": "ROBOTIS XM540-W150",
    "robotis_xh540_w150_t": "ROBOTIS XH540-W150", "robotis_xm430_w350_t": "ROBOTIS XM430-W350",
    "robotis_xm430_w210_t": "ROBOTIS XM430-W210",
}
BATTERY_NAMES = {"lipo_6s_3300": "6S LiPo 3.3 Ah", "lipo_6s_5000": "6S LiPo 5 Ah", "lipo_6s_10000": "6S LiPo 10 Ah",
                 "lipo_12s_5000": "12S LiPo 5 Ah", "lipo_2x6s_3300_series": "2 × 6S LiPo 3.3 Ah (12S)",
                 "liion_6s2p_p45b": "6S2P Li-ion P45B", "liion_6s3p_50s": "6S3P Li-ion 50S",
                 "liion_12s2p_p45b": "12S2P Li-ion P45B", "liion_12s3p_50s": "12S3P Li-ion 50S"}


def part(pid: str | None) -> str:
    return PART_NAMES.get(pid or "", pid or "–")


def web_image(src: Path, dst: Path, width: int = 1800) -> tuple[int, int]:
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "WEBP", quality=84, method=6)
    return im.size


class Ctx:
    """Data access with small formatting helpers."""

    def __init__(self, round1: Path, out: Path):
        self.r1 = round1
        self.out = out
        self.data = json.loads((round1 / "review_data.json").read_text())
        self.C = self.data["candidates"]
        self.R = self.data.get("review", {})
        self.stats = self.data["stats"]
        self.plates = round1 / "review" / "plates"
        self.screen = round1 / "review" / "screening"
        self.images = []
        self.plate_no = 0

    def c(self, cid):
        return self.C[cid]

    def img(self, name: str, alt: str, title: str, caption: str, src_dir: Path | None = None) -> str:
        src = (src_dir or self.plates) / name
        if not src.exists():
            return f'<p class="note">Missing plate: <code>{K.esc(name)}</code></p>'
        self.plate_no += 1
        web = Path("img") / (Path(name).stem + ".webp")
        web_image(src, self.out / web)
        self.images.append(str(web))
        return K.figure(str(web), alt, f"Plate {self.plate_no} · {title}", caption)


def kg(v):
    return f"{v:.1f} kg"


def cm(v):
    return f"{100 * v:.0f} cm"


def usd(v):
    return f"${v / 1000:.1f}k"


def build(round1: Path, out: Path, sections) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(round1, out)
    body = "".join(fn(ctx) for fn in sections)
    html = K.page("T. rex Design Review", body)
    (out / "index.html").write_text(html)
    (out / "files.json").write_text(json.dumps(ctx.images, indent=1))
    return out / "index.html"


def main():
    from .review_content import SECTIONS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round1", default="round1")
    ap.add_argument("--out", default="round1/review_page")
    a = ap.parse_args()
    p = build(Path(a.round1), Path(a.out), SECTIONS)
    print(p)


if __name__ == "__main__":
    main()
