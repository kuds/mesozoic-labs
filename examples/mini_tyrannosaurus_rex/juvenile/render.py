"""Render the compiled candidate geometry, including its physical packing view."""

from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from environments.shared.robot_rendering import font, render
from examples.mini_tyrannosaurus_rex import engineering

HERE = Path(__file__).resolve().parent


def render_sheet(directory: Path = HERE) -> Path:
    model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
    data = engineering.home_data(model)
    sheet = Image.new("RGB", (1800, 1500), (245, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((55, 32), "MINI / JUVENILE THEROPOD", font=font(36), fill=(38, 55, 43))
    draw.text(
        (55, 87),
        "Actual MuJoCo geometry · full-size motors · explicit hardware reservations",
        font=font(21),
        fill=(83, 97, 88),
    )
    panels = [
        ((1, -1.6, 0.65), (1, 2), "01 / Assembled", None),
        ((0, -1, 0.03), (1, 2), "02 / Side profile", None),
        ((1, 0, 0.08), (1, 2), "03 / Front", None),
        ((1, -1.6, 0.65), (2, 4), "04 / Shell removed: raw parts and mechanism", None),
    ]
    for index, (direction, groups, title, hidden) in enumerate(panels):
        x, y = index % 2 * 900, 145 + index // 2 * 610
        panel = render(model, data, direction=direction, groups=groups, hidden_pattern=hidden, width=890, height=550)
        sheet.paste(panel, (x, y + 28))
        draw.text((x + 45, y), title, font=font(20), fill=(60, 76, 66))
    draw.text(
        (55, 1390),
        f"{model.body_mass.sum() * 1000:.0f} g allocated · 14 XL330-size motors · fixed tail · separate jaw and nod/yaw joints",
        font=font(21),
        fill=(60, 76, 66),
    )
    draw.text(
        (55, 1432),
        "Engineering candidate: read validation_summary.json for tested scope and remaining fabrication work.",
        font=font(19),
        fill=(83, 97, 88),
    )
    output = directory / "design.png"
    sheet.save(output)
    return output


if __name__ == "__main__":
    print(render_sheet())
