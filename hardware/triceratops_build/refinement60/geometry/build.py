"""Rebuild the 60 cm refinement from its self-contained open-rib reference model.

Unchanged mesh and texture assets resolve to the reference model directory.
Changed meshes and both immediately usable MJCFs are written to the output.
"""

import argparse
import filecmp
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import relieve
import repack
import shoulder_trim

MODELS = Path(__file__).resolve().parents[1] / "models"


def build(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Choose an output distinct from the reference model directory.")
    with TemporaryDirectory() as temporary:
        first = Path(temporary) / "motor-mount"
        sculpt = Path(temporary) / "final"
        repack.build(source, first)
        relieve.build(first, sculpt)
        shoulder_trim.trim(sculpt, sculpt)
        output.mkdir(parents=True, exist_ok=True)
        hero = ET.parse(sculpt / "triceratops.xml")
        for item in hero.findall(".//asset/*[@file]"):
            name = item.attrib["file"]
            original = source / name
            revised = sculpt / name
            if original.is_file() and filecmp.cmp(original, revised, shallow=False):
                item.set("file", Path(os.path.relpath(original, output)).as_posix())
            else:
                shutil.copyfile(revised, output / name)
        shutil.copyfile(sculpt / "physics.xml", output / "physics.xml")
        hero.write(output / "triceratops.xml", encoding="unicode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=MODELS / "reference")
    parser.add_argument("--output", type=Path, default=MODELS / "refined")
    args = parser.parse_args()
    build(args.reference, args.output)
