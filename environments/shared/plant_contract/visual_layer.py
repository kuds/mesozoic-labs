"""Visual layer: render geometry, materials, textures, cameras, and lights.

Kept separate so a render-only improvement invalidates recorded videos
without invalidating a policy trained against the same physics."""

from __future__ import annotations

from typing import Any

import mujoco

from .digests import _array_digest
from .identity import PlantVersion
from .introspection import _fields, _names


def _visual_options(model: mujoco.MjModel) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name in ("global_", "quality", "headlight", "map", "scale", "rgba"):
        group = getattr(model.vis, group_name)
        result[group_name] = {
            name: getattr(group, name)
            for name in dir(group)
            if not name.startswith("_") and not callable(getattr(group, name))
        }
    return result


def _visual_payload(model: mujoco.MjModel, version: PlantVersion) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "geoms": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
            **_fields(
                model,
                (
                    "geom_bodyid",
                    "geom_type",
                    "geom_dataid",
                    "geom_pos",
                    "geom_quat",
                    "geom_size",
                    "geom_group",
                    "geom_matid",
                    "geom_rgba",
                ),
            ),
        },
        "sites": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            **_fields(
                model,
                (
                    "site_bodyid",
                    "site_type",
                    "site_pos",
                    "site_quat",
                    "site_size",
                    "site_group",
                    "site_matid",
                    "site_rgba",
                ),
            ),
        },
        "materials": _fields(
            model,
            (
                "mat_texid",
                "mat_texrepeat",
                "mat_texuniform",
                "mat_emission",
                "mat_specular",
                "mat_shininess",
                "mat_reflectance",
                "mat_metallic",
                "mat_roughness",
                "mat_rgba",
            ),
        ),
        "cameras": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
            **_fields(
                model,
                (
                    "cam_bodyid",
                    "cam_targetbodyid",
                    "cam_mode",
                    "cam_pos",
                    "cam_quat",
                    "cam_fovy",
                    "cam_ipd",
                    "cam_intrinsic",
                    "cam_projection",
                    "cam_resolution",
                    "cam_sensorsize",
                    "cam_output",
                    "cam_user",
                ),
            ),
        },
        "lights": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_LIGHT, model.nlight),
            **_fields(
                model,
                (
                    "light_type",
                    "light_bodyid",
                    "light_targetbodyid",
                    "light_mode",
                    "light_pos",
                    "light_dir",
                    "light_active",
                    "light_castshadow",
                    "light_bulbradius",
                    "light_attenuation",
                    "light_cutoff",
                    "light_exponent",
                    "light_intensity",
                    "light_range",
                    "light_texid",
                    "light_ambient",
                    "light_diffuse",
                    "light_specular",
                ),
            ),
        },
        "visual_options": _visual_options(model),
    }
    if model.ntex:
        payload["textures"] = {
            **_fields(
                model,
                ("tex_type", "tex_height", "tex_width", "tex_nchannel", "tex_adr", "tex_colorspace"),
            ),
            "data": _array_digest("mesozoic.texture-data/v1", model.tex_data),
        }
    if model.nhfield:
        payload["heightfields"] = {
            **_fields(model, ("hfield_adr", "hfield_nrow", "hfield_ncol", "hfield_size")),
            "data": _array_digest("mesozoic.visual-heightfield-data/v1", model.hfield_data),
        }
    if model.nmesh:
        payload["meshes"] = {
            **_fields(
                model,
                ("mesh_vertadr", "mesh_vertnum", "mesh_faceadr", "mesh_facenum", "mesh_scale", "mesh_pos", "mesh_quat"),
            ),
            "vertices": _array_digest("mesozoic.visual-mesh-vertices/v1", model.mesh_vert),
            "faces": _array_digest("mesozoic.visual-mesh-faces/v1", model.mesh_face),
            "normals": _array_digest("mesozoic.visual-mesh-normals/v1", model.mesh_normal),
            "texcoords": _array_digest("mesozoic.visual-mesh-texcoords/v1", model.mesh_texcoord),
        }
    if model.nskin:
        payload["skins"] = {
            **_fields(
                model,
                (
                    "skin_matid",
                    "skin_group",
                    "skin_rgba",
                    "skin_inflate",
                    "skin_vertadr",
                    "skin_vertnum",
                    "skin_faceadr",
                    "skin_facenum",
                ),
            ),
            "vertices": _array_digest("mesozoic.skin-vertices/v1", model.skin_vert),
            "faces": _array_digest("mesozoic.skin-faces/v1", model.skin_face),
        }
    return payload
