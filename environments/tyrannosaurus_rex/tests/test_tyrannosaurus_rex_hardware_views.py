"""Keep installed hardware visible and published candidate images source-bound."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from environments.shared.paths import REPOSITORY_ROOT
from examples.tyrannosaurus_rex_50cm import demo, hardware_views, redesign

HERE = Path(redesign.__file__).parent
CANDIDATE = HERE / "long_body"


def test_service_copy_exposes_every_assembly_part_and_preserves_physical_model():
    model, _ = demo.load_model(path=CANDIDATE / "model.xml")
    fields = ("geom_group", "geom_rgba", "geom_matid", "geom_pos", "geom_size", "body_mass", "body_inertia")
    before = {name: getattr(model, name).copy() for name in fields}
    service = hardware_views.service_model(model)
    report = hardware_views.hardware_visibility(service)
    assert report["passed"]
    assert set(report["parts"]) == set(hardware_views.required_hardware())
    assert all(row["visible_in"] for row in report["parts"].values())
    assert all(service.geom(name).group == 4 for name in hardware_views.required_hardware())
    for name in (
        "torso_front_shell",
        "torso_bridge_shell",
        "torso_rear_shell",
        "left_torso_cover",
        "right_torso_cover",
        "left_torso_vent_0",
        "right_torso_vent_0",
        "pelvis_structure_envelope",
        "head_shell",
    ):
        assert service.geom(name).group == 5, name
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(model, name), expected)
        if name != "geom_group":
            np.testing.assert_array_equal(getattr(service, name), expected)
    json.dumps(report, allow_nan=False)


def test_visibility_detects_shell_occlusion_and_writer_refuses_hidden_hardware(tmp_path, monkeypatch):
    model, _ = demo.load_model(path=CANDIDATE / "model.xml")
    service = hardware_views.service_model(model)
    service.geom("torso_rear_shell").group = 1
    report = hardware_views.hardware_visibility(service)
    assert not report["passed"]
    assert not report["parts"]["hardware_battery"]["visible_in"]
    monkeypatch.setattr(hardware_views, "service_model", lambda supplied: service)
    output = tmp_path / "hidden.png"
    with pytest.raises(ValueError, match="hardware_battery"):
        hardware_views.write_service_view(
            model, output=output, source_paths=(CANDIDATE / "model.xml",), source_root=REPOSITORY_ROOT
        )
    assert not output.exists()


def test_original_medium_model_cannot_be_mistaken_for_candidate_hardware():
    model, _ = demo.load_model()
    before = model.geom_group.copy()
    with pytest.raises(ValueError, match="separate long-body candidate"):
        hardware_views.service_model(model)
    np.testing.assert_array_equal(model.geom_group, before)


def test_service_png_is_a_real_cpu_render_with_source_and_visibility_metadata(tmp_path):
    model, _ = demo.load_model(path=CANDIDATE / "model.xml")
    original = model.geom_group.copy()
    output = tmp_path / "hardware.png"
    report = hardware_views.write_service_view(
        model, output=output, source_paths=(CANDIDATE / "model.xml",), source_root=REPOSITORY_ROOT
    )
    assert report["passed"]
    with Image.open(output) as picture:
        assert picture.width > picture.height
        assert np.asarray(picture).std() > 10
        assert json.loads(picture.info["hardware_visibility"]) == report
        hashes = json.loads(picture.info["source_sha256"])
    assert (CANDIDATE / "model.xml").relative_to(REPOSITORY_ROOT).as_posix() in hashes
    assert hardware_views.MANIFEST_PATH.relative_to(REPOSITORY_ROOT).as_posix() in hashes
    for name, expected in hashes.items():
        assert hashlib.sha256((REPOSITORY_ROOT / name).read_bytes()).hexdigest() == expected
    np.testing.assert_array_equal(model.geom_group, original)


def test_image_stamp_changes_when_an_input_changes(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"dimension_m":0.1}')
    first = hardware_views.source_stamp((source,), Path("/"))
    source.write_text('{"dimension_m":0.2}')
    second = hardware_views.source_stamp((source,), Path("/"))
    key = source.relative_to("/").as_posix()
    assert first["source_sha256"][key] != second["source_sha256"][key]


@pytest.mark.parametrize("filename", ["comparison.png", "hardware_layout.png", "crouch_to_stand.gif"])
def test_committed_visual_evidence_matches_current_sources_and_recorded_poses(filename):
    path = CANDIDATE / filename
    assert path.stat().st_size < 500 * 1024
    with Image.open(path) as image:
        metadata = (
            json.loads(image.info["comment"])
            if filename.endswith(".gif")
            else {
                name: json.loads(value)
                for name, value in image.info.items()
                if name in ("source_sha256", "hardware_visibility")
            }
        )
    hashes = metadata["source_sha256"]
    assert hashes
    required = (
        CANDIDATE / "model.xml",
        CANDIDATE / "components.json",
        hardware_views.MANIFEST_PATH,
        Path(hardware_views.__file__),
        HERE / "redesign_review.py",
    )
    assert {path.relative_to(REPOSITORY_ROOT).as_posix() for path in required} <= hashes.keys()
    for name, expected in hashes.items():
        assert hashlib.sha256((REPOSITORY_ROOT / name).read_bytes()).hexdigest() == expected, (
            f"Regenerate {filename}: {name}"
        )
    if filename == "hardware_layout.png":
        assert metadata["hardware_visibility"]["passed"]
        assert set(metadata["hardware_visibility"]["parts"]) == set(hardware_views.required_hardware())
    elif filename.endswith(".gif"):
        report = json.loads((CANDIDATE / "comparison.json").read_text())
        poses = report["cases"]["long_body"]["crouch_to_stand"][-1]["pose_samples"]
        encoded = json.dumps(poses, sort_keys=True, separators=(",", ":")).encode()
        assert metadata["pose_samples_sha256"] == hashlib.sha256(encoded).hexdigest()
