"""Skeletal template and family shape tables, in units of the hip height ``H``.

The template is the study's engineering adaptation of published *T. rex* proportions
(``data/anatomy.json`` holds the cited measurements and the derived ratios; ``REFERENCE.md``
explains each adaptation).  Values here are the ones the generator uses; every departure from
the anatomical ratio is named in ``ADAPTATIONS`` so reviewers can see what is reference,
what is engineering adaptation and what is artistic interpretation.

Coordinates: x forward along the torso axis, z up, origin at the hip-pitch axis midpoint, in
the *unbent* (level) trunk frame.  Lengths are fractions of H.
"""

from __future__ import annotations

import copy

import numpy as np

# ---------------------------------------------------------------------------------------------
# Skeleton landmarks (level design pose)
# ---------------------------------------------------------------------------------------------
SKELETON = dict(
    # Acetabulum -> glenoid (shoulder) distance.  Gleno-acetabular distance is 0.211-0.225 of body
    # length (Hutchinson et al. 2011, table 3: Stan, CM 9380, SUE) and standing hip height ~0.24 of
    # body length (derived, data/research/anatomy_2026-11-07.json), i.e. ~0.88 H.
    glenoid_x=0.88,
    glenoid_z=-0.02,
    # Lumbar bend pivot (chest rotates about this point for chest-up postures).
    lumbar_x=0.30,
    lumbar_z=0.10,
    # Neck base (cervico-dorsal junction, top of shoulders) in the chest frame.
    neck_base_x=0.92,
    neck_base_z=0.07,
    # Neck: two straight segments approximating the S-curve (length, angle above torso axis).
    neck_segments=((0.17, 26.0), (0.15, 5.0)),
    # Skull long-axis length (occipital condyle to snout tip, covering included) and shape.
    skull_len=0.47,
    skull_pitch_deg=-4.0,  # world pitch of the skull axis in the nominal pose (nose slightly down)
    # Tail root (first free caudal analogue) in the pelvis frame and tail length.
    tail_root_x=-0.34,  # with the battery in the belly
    tail_root_x_tailbase=-0.50,  # with the battery in a fixed tail base behind the hip
    tail_root_z=0.06,
    tail_len=1.55,  # anatomical first-free-caudal tail is ~2.0 H; see ADAPTATIONS
    tail_pitch_deg=2.0,  # world pitch of the tail root axis (slightly raised)
    tail_curve_deg=4.0,  # additional upward bend over the distal half
    # Fixed arms.
    arm_x=0.80,
    arm_z=-0.30,
    arm_len=0.22,
)

# Leg: segment lengths as fractions of the total chain length, design angles from vertical
# (positive = distal end forward), from SUE limb ratios femur:tibia:metatarsus ~ 1.32:1.14:0.67.
LEG = dict(
    # femur : tibia : metatarsal III = 1 : 0.90 : 0.53 (adult ratios, Persons & Currie 2016 table 1;
    # Persons et al. 2020 table 1; derived in data/research/anatomy_2026-11-07.json)
    femur_frac=0.411,
    tibia_frac=0.370,
    meta_frac=0.219,
    femur_deg=22.0,  # knee forward of hip
    tibia_deg=-28.0,  # hock behind knee
    meta_deg=15.0,  # MTP forward of hock (fixed hock bend = tibia/meta angle difference)
    mtp_height=0.050,  # ankle (MTP) axis above floor, fraction of H
    toe_len=0.20,  # digit III, MTP to tip
    toe_side_scale=0.82,
    toe_splay_deg=33.0,  # tracks: total II-IV divarication 51-80 deg, typical ~70 (McCrea et al. 2014)
    heel_len=0.06,  # pad behind the MTP axis
)

ADAPTATIONS = [
    ("tail_len", "Anatomical tail measured from the first free caudal is about 2.3 H (51-59 % of body "
     "length, Hutchinson et al. 2011 table 3; hip height ~0.24 of body length).  The study default is "
     "1.55 H to limit turning sweep, floor footprint and tail inertia; the 'anat' proportion variant "
     "restores 2.3 H for comparison."),
    ("legs", "The tibia and metatarsus are fused into one shank with a fixed hock bend; the actuated ankle "
     "(pitch and roll) sits at the metatarsophalangeal joint so the toes carry the load while the hock "
     "stays raised as in the animal."),
    ("hip_width", "Hip-joint spacing follows actuator envelopes, not the fossil pelvis; the width variants "
     "explore it explicitly."),
    ("neck", "A single powered neck-pitch joint at the cervico-dorsal junction moves a rigid S-shaped neck "
     "and skull; the nominal neck angle keeps the skull near level in every posture."),
    ("arms", "Arms are fixed, mounted low at the front of the chest (SUE 2018 remount places them low and "
     "forward)."),
]

# ---------------------------------------------------------------------------------------------
# Family shape tables
# ---------------------------------------------------------------------------------------------
# Trunk stations along the torso axis (x/H).  Rows: dorsal height above axis, ventral height (negative
# = below), half-width, upper superellipse exponent, lower exponent.  The trunk loft is interpolated
# with a monotone cubic.  Regions: tail root junction, ilium, hip, pubic boot, abdomen, rear ribs,
# mid ribs, pectoral, neck base.
TRUNK_X = np.array([-0.58, -0.40, -0.28, -0.12, 0.02, 0.16, 0.30, 0.44, 0.58, 0.72, 0.84, 0.95])

FAMILY_SHAPE = {
    # Tapered, lightly covered: lean trunk, belly rises behind the ribs, small pubic keel, slimmer
    # thigh fairings, partial coverings (open ventral abdomen, exposed lower legs).
    "T": dict(
        dorsal=[0.150, 0.190, 0.250, 0.272, 0.272, 0.258, 0.243, 0.232, 0.225, 0.218, 0.200, 0.150],
        ventral=[-0.095, -0.110, -0.130, -0.150, -0.210, -0.270, -0.225, -0.240, -0.285, -0.300, -0.255, -0.155],
        half_width=[0.078, 0.090, 0.108, 0.115, 0.112, 0.090, 0.122, 0.158, 0.180, 0.176, 0.152, 0.100],
        wide=[0.50, 0.55, 0.62, 0.74, 0.78, 0.62, 0.48, 0.42, 0.40, 0.42, 0.48, 0.52],
        n_up=[2.0] * 12,
        n_lo=[2.0, 2.0, 1.8, 1.5, 1.4, 1.4, 1.9, 2.2, 2.3, 2.3, 2.2, 2.1],
        thigh_scale=0.88,
        tail_root_scale=0.92,
        skull=dict(depth=0.40, width_rear=0.44, width_snout=0.20, brow=0.04),
        cover_regions=("skull", "jaw", "neck", "chest_dorsal", "chest_lateral", "abdomen_upper", "pelvis_dorsal",
                       "pelvis_lateral", "pelvis_ventral", "thigh", "hip_cap", "tail"),
        shell_mm=1.6,
    ),
    # Substantial, anatomically balanced: deep ribcage with gastral basket, belly line rising behind the
    # ribs to a distinct pubic boot keel (side-view waist), waist in plan view between ribcage and thighs,
    # slight dorsal ridge rather than a rounded back, robust thigh fairings and tail root.
    "S": dict(
        dorsal=[0.165, 0.205, 0.270, 0.292, 0.292, 0.276, 0.256, 0.242, 0.232, 0.224, 0.206, 0.160],
        ventral=[-0.110, -0.130, -0.155, -0.185, -0.260, -0.345, -0.262, -0.285, -0.345, -0.365, -0.300, -0.180],
        half_width=[0.088, 0.102, 0.124, 0.136, 0.135, 0.106, 0.152, 0.196, 0.220, 0.212, 0.180, 0.120],
        wide=[0.50, 0.55, 0.62, 0.74, 0.78, 0.62, 0.45, 0.38, 0.36, 0.38, 0.45, 0.50],
        n_up=[2.2, 2.2, 2.0, 1.9, 1.8, 1.8, 1.8, 1.8, 1.8, 1.9, 2.0, 2.2],
        n_lo=[2.2, 2.2, 1.9, 1.5, 1.4, 1.35, 1.9, 2.4, 2.6, 2.6, 2.4, 2.2],
        thigh_scale=1.12,
        tail_root_scale=1.05,
        skull=dict(depth=0.46, width_rear=0.50, width_snout=0.22, brow=0.05),
        cover_regions=("skull", "jaw", "neck", "chest_dorsal", "chest_lateral", "chest_ventral",
                       "abdomen", "pelvis_dorsal", "pelvis_lateral", "pelvis_ventral", "pubic_boot", "thigh", "hip_cap", "shank",
                       "tail"),
        shell_mm=1.8,
    ),
    # Hybrid: covered skull, neck, pectoral "plastron", pelvis and thighs; exposed spine beam and rib
    # hoops over the mid-trunk showing the battery and compute bays; segmented exposed distal tail.
    "H": dict(
        dorsal=[0.158, 0.198, 0.260, 0.282, 0.282, 0.266, 0.250, 0.237, 0.228, 0.221, 0.203, 0.155],
        ventral=[-0.102, -0.120, -0.142, -0.168, -0.235, -0.315, -0.248, -0.265, -0.320, -0.338, -0.280, -0.170],
        half_width=[0.083, 0.096, 0.116, 0.126, 0.124, 0.100, 0.138, 0.177, 0.200, 0.194, 0.165, 0.110],
        wide=[0.50, 0.55, 0.62, 0.74, 0.78, 0.62, 0.46, 0.40, 0.38, 0.40, 0.46, 0.51],
        n_up=[2.1, 2.1, 1.95, 1.85, 1.8, 1.8, 1.8, 1.8, 1.8, 1.9, 2.0, 2.1],
        n_lo=[2.1, 2.1, 1.85, 1.5, 1.4, 1.35, 1.9, 2.3, 2.5, 2.5, 2.3, 2.1],
        thigh_scale=1.0,
        tail_root_scale=1.0,
        skull=dict(depth=0.43, width_rear=0.47, width_snout=0.21, brow=0.045),
        cover_regions=("skull", "jaw", "neck", "chest_front", "pelvis_dorsal", "pelvis_lateral",
                       "pelvis_ventral", "pubic_boot", "thigh", "hip_cap", "tail_root", "tail_segments", "rib_hoops"),
        shell_mm=1.6,
    ),
}


#: Generator revision 2 (round two).  Round-one shapes stay under the plain family letter so round-one
#: models rebuild byte-identical; ``family_key`` picks the shape set for a candidate's revision
#: (``Candidate.overrides`` entry ``("rev", 2)``).  Each change is recorded in DECISION_LOG.md.
def family_key(family: str, rev: int = 1) -> str:
    return family if rev < 2 else f"{family}@2"


def _derived(base: str, **changes) -> dict:
    d = copy.deepcopy(FAMILY_SHAPE[base])
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


# F8: the stereo camera must sit inside the skull cover.  A slightly broader Tapered skull (it was the only
# one with no fitting position at 60 cm) and a flatter skull roof (exponent 2.6 -> 3.2, a broad flat roof
# between the brows, as in the animal) let it sit about 2-4 mm higher.
FAMILY_SHAPE["T@2"] = _derived("T", skull=dict(width_rear=0.46, roof_n=3.2))
FAMILY_SHAPE["S@2"] = _derived("S", skull=dict(roof_n=3.2))
FAMILY_SHAPE["H@2"] = _derived("H", skull=dict(roof_n=3.2))


def leg_chain_length(h: float, leg_scale: float = 1.0) -> float:
    """Total femur+shank chain length giving hip height ``h`` in the design pose."""
    a = np.radians([LEG["femur_deg"], LEG["tibia_deg"], LEG["meta_deg"]])
    fr = np.array([LEG["femur_frac"], LEG["tibia_frac"], LEG["meta_frac"]])
    vertical = float(np.sum(fr * np.cos(a)))
    return (h * (1.0 - LEG["mtp_height"])) / vertical * leg_scale
