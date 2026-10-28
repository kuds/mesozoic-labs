"""Render the committed skeleton's exposed hardware without inventing shell CAD."""

import mujoco
from PIL import Image, ImageDraw

from environments.shared.robot_rendering import font, render
from examples.mini_tyrannosaurus_rex.skeleton.build import OUTPUT, ROOT


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    sheet = Image.new("RGB", (1500, 650), "#f5f7f4")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 20), "Mini T-Rex | M1 skeleton | hw_r01", font=font(30), fill="#24322c")
    draw.text(
        (30, 62),
        "Actual MuJoCo geometry: torso A, full hardware envelopes, forward-folding legs",
        font=font(20),
        fill="#526359",
    )
    for i, (label, direction) in enumerate(
        (("Side", (0, -1, 0.12)), ("Front", (1, 0, 0.1)), ("Three-quarter", (1, -1, 0.55)))
    ):
        sheet.paste(render(model, data, direction=direction, groups=(2, 4), width=500, height=470), (500 * i, 100))
        draw.text((500 * i + 30, 105), label, font=font(22), fill="#24322c")
    draw.text(
        (30, 585),
        "754.1 g allocated mass includes fixed future loads; markers and installation reserves hidden in these views.",
        font=font(19),
        fill="#526359",
    )
    draw.text(
        (30, 615),
        "Mounting CAD, ballast fixtures and moving cable routes remain to be designed. Tail has no actuator.",
        font=font(19),
        fill="#526359",
    )
    sheet.save(ROOT / "hardware/mini_tyrannosaurus_rex/m01_skeleton/preview.png")


if __name__ == "__main__":
    main()
