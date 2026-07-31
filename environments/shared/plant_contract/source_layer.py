"""Source-closure layer: the exact MJCF and referenced asset bytes.

Walks ``include``/mesh/texture references out of the MJCF so that any byte
change in the model or its assets changes the source digest."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from . import constants
from .errors import PlantContractError


def _source_dependencies(model_path: Path) -> list[Path]:
    """Return the recursive MJCF/include/asset dependency closure."""
    pending = [model_path.resolve()]
    visited: set[Path] = set()
    while pending:
        xml_path = pending.pop()
        if xml_path in visited:
            continue
        try:
            xml_path.relative_to(constants.REPOSITORY_ROOT)
        except ValueError as exc:
            raise PlantContractError(f"MJCF dependency is outside the repository: {xml_path}") from exc
        visited.add(xml_path)
        try:
            root = ET.parse(xml_path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise PlantContractError(f"cannot parse MJCF dependency {xml_path}: {exc}") from exc

        compiler = root.find("compiler")
        asset_dir = xml_path.parent
        mesh_dir = asset_dir
        texture_dir = asset_dir
        if compiler is not None:
            if compiler.get("assetdir"):
                asset_dir = (xml_path.parent / str(compiler.get("assetdir"))).resolve()
            mesh_dir = (
                (xml_path.parent / str(compiler.get("meshdir"))).resolve() if compiler.get("meshdir") else asset_dir
            )
            texture_dir = (
                (xml_path.parent / str(compiler.get("texturedir"))).resolve()
                if compiler.get("texturedir")
                else asset_dir
            )

        for element in root.iter():
            file_name = element.get("file")
            if not file_name:
                continue
            if element.tag == "include":
                dependency = (xml_path.parent / file_name).resolve()
                pending.append(dependency)
                continue
            base_dir = (
                mesh_dir if element.tag in {"mesh", "skin"} else texture_dir if element.tag == "texture" else asset_dir
            )
            dependency = (base_dir / file_name).resolve()
            try:
                dependency.relative_to(constants.REPOSITORY_ROOT)
            except ValueError as exc:
                raise PlantContractError(f"MJCF asset is outside the repository: {dependency}") from exc
            if not dependency.is_file():
                raise PlantContractError(f"MJCF asset does not exist: {dependency}")
            visited.add(dependency)
    return sorted(visited, key=lambda path: path.relative_to(constants.REPOSITORY_ROOT).as_posix())


def _source_payload(model_path: Path) -> dict[str, Any]:
    dependencies = []
    for path in _source_dependencies(model_path):
        relative = path.relative_to(constants.REPOSITORY_ROOT).as_posix()
        dependencies.append(
            {
                "logical_path": relative,
                "kind": "mjcf" if path.suffix.lower() == ".xml" else "asset",
                "content_sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            }
        )
    return {"root": model_path.relative_to(constants.REPOSITORY_ROOT).as_posix(), "dependencies": dependencies}
