# Kept for stage2/research/research_look.md, which used it (10 Nov 2026): arithmetic only, no simulation.
"""Hand estimates for the look research note (arithmetic only; no simulation).

Inputs are read from the round-two models' inertial blocks (copied below) and the candidate records' drive margins.
Passive joint terms are the MJCF defaults on every joint (damping 0.2 N m s/rad, frictionloss 0.15 N m), which the
Stage 1 plant applies and Stage 0's inverse dynamics leaves out.
"""
import math

G = 9.81
DAMP, FRIC = 0.2, 0.15

bodies = {
    "60-N-T-10": dict(
        H=0.4047, mass=10.81,
        head=dict(m=0.531, com=(0.119, 0.036), Iyy=0.001789, Izz=0.001568, x_world=0.363),
        tail=dict(m=0.2925, com=(-0.2085, 0.0099), Izz=0.005309, Iyy=0.005402),
        trunk=dict(m=4.70, com=(0.044, 0.011), Iyy=0.1168),
        neck=dict(name="STS3250", sus=0.819, peak=1.437, speed=4.71, hold=0.617),
        tailservo=dict(name="STS3215", sus=0.491, peak=0.862, speed=2.83),
        jaw_r=0.10, yaw_atlas_r=0.08, jaw_len=0.188,
        gait=dict(nominal=(1.020, 0.60), brisk=(0.796, 0.58), fr05=(0.727, 0.56)),
        stride=dict(nominal=0.2226, fr05=0.324), leg=0.385,
    ),
    "75-M-H-10": dict(
        H=0.5019, mass=14.85,
        head=dict(m=0.792, com=(0.150, 0.044), Iyy=0.004856, Izz=0.004031, x_world=0.4505),
        tail=dict(m=0.438, com=(-0.244, 0.0071), Izz=0.011962, Iyy=0.012193),
        trunk=dict(m=6.16, com=(0.093, 0.011), Iyy=0.2350),
        neck=dict(name="XM540-W270", sus=1.770, peak=3.106, speed=1.885, hold=1.160),
        tailservo=dict(name="STS3215", sus=0.491, peak=0.862, speed=2.83),
        jaw_r=0.12, yaw_atlas_r=0.10, jaw_len=0.233,
        gait=dict(nominal=(1.136, 0.60), brisk=(0.887, 0.58), fr05=(0.809, 0.56)),
        stride=dict(nominal=0.276, fr05=0.401), leg=0.477,
    ),
}

ADD = 0.075  # kg per added servo with horn and bracket (STS3215 0.055 kg + ~0.02 kg)


def minjerk(theta, T):
    """peak angular velocity and acceleration of a minimum-jerk move"""
    return 1.875 * theta / T, 5.7735 * theta / T**2


def sin_peak(A, f):
    w = 2 * math.pi * f
    return A * w, A * w * w


for name, b in bodies.items():
    print("=" * 20, name)
    hd, tl, nk, ts = b["head"], b["tail"], b["neck"], b["tailservo"]
    x, z = hd["com"]
    I_pitch = hd["Iyy"] + hd["m"] * (x * x + z * z)
    I_yaw_base = hd["Izz"] + hd["m"] * x * x + ADD * b["jaw_r"] ** 2  # yaw at neck base carries head + jaw servo
    # skull-only yaw at the atlas: assume 75 % of head mass, CoM 0.06 / 0.075 m ahead of the atlas, box inertia
    ms = 0.75 * hd["m"]
    rs = 0.06 if name.startswith("60") else 0.075
    Ls = 0.206 if name.startswith("60") else 0.254
    Ws = 0.118 if name.startswith("60") else 0.148
    I_yaw_atlas = ms * (rs * rs + (Ls * Ls + Ws * Ws) / 12) + ADD * (b["jaw_r"] - b["yaw_atlas_r"]) ** 2
    print(f"head: I_pitch {I_pitch:.4f}  I_yaw(base) {I_yaw_base:.4f}  I_yaw(atlas, est) {I_yaw_atlas:.4f}")
    # neck pitch hold with added parts
    hold = nk["hold"]
    jaw_add = ADD * G * b["jaw_r"]
    atlas_add = ADD * G * b["yaw_atlas_r"]
    print(f"neck pitch hold {hold:.3f} ({hold/nk['sus']:.2f} of A1 {nk['sus']}); +jaw servo {jaw_add:.3f} -> "
          f"{(hold+jaw_add)/nk['sus']:.2f}; +jaw +atlas yaw {(hold+jaw_add+atlas_add)/nk['sus']:.2f}")
    # hip pitch hold and CoM shift for the two placements
    xb = hd["x_world"]
    a_hip = ADD * G * xb + ADD * G * (xb + b["jaw_r"])
    b_hip = ADD * G * (xb + b["yaw_atlas_r"]) + ADD * G * (xb + b["jaw_r"])
    a_com = (ADD * xb + ADD * (xb + b["jaw_r"])) / (b["mass"] + 2 * ADD) * 1000
    print(f"hip-pitch hold +{a_hip:.2f} N m (yaw at base) / +{b_hip:.2f} (yaw at atlas); CoM +{a_com:.1f} mm")
    # gaze shifts (neck yaw)
    for th, T in ((0.6, 0.5), (0.6, 0.8), (0.3, 0.3)):
        w, a = minjerk(th, T)
        tin = I_yaw_base * a
        print(f"gaze {th} rad in {T} s: w {w:.2f} rad/s, inertial {tin:.3f} N m, +damping {DAMP*w:.3f}, "
              f"+friction {FRIC} -> model peak {tin+DAMP*w+FRIC:.3f} (STS3215 peak 0.862, A1 0.491); "
              f"atlas inertial {I_yaw_atlas*a:.3f}")
    for A, f in ((0.1, 3.0), (0.05, 2.5)):
        w, a = sin_peak(A, f)
        print(f"head shake +-{A} rad at {f} Hz: inertial {I_yaw_base*a:.3f}, damping {DAMP*w:.3f}, "
              f"model peak ~{I_yaw_base*a+FRIC:.3f}-{math.hypot(I_yaw_base*a, DAMP*w)+FRIC:.3f}")
    # roar head raise (neck pitch) 0.3 rad in 0.4 s
    w, a = minjerk(0.3, 0.4)
    print(f"roar raise 0.3 rad/0.4 s: w {w:.2f} (limit {nk['speed']}), inertial {I_pitch*a:.3f} + hold <= {hold+jaw_add:.3f}")
    # neck stabilisation vs nod at step frequency
    for gname in ("nominal", "fr05"):
        T, d = b["gait"][gname]
        fs = 2 / T
        for A, lab in ((0.03, "stabilise trunk pitch +-0.03"), (0.05, "ref nod 0.05"), (0.15, "spec nod 0.15")):
            w, a = sin_peak(A, fs)
            print(f"  {gname} step {fs:.2f} Hz {lab}: inertial {I_pitch*a:.3f}, damping {DAMP*w:.3f}")
    # jaw
    mj = 0.15 * hd["m"]
    Ij = mj * b["jaw_len"] ** 2 / 3 + 0.0
    rj = 0.45 * b["jaw_len"]
    print(f"jaw est m {mj:.3f} kg, I {Ij:.5f}, gravity {mj*G*rj:.3f} N m")
    for th, T in ((0.6, 0.3), (0.4, 0.15)):
        w, a = minjerk(th, T)
        print(f"  jaw {th} rad in {T} s: w {w:.2f} rad/s (STS3215 planning 2.83), inertial {Ij*a:.3f} N m")
    # tail
    tx, tz = tl["com"]
    It = tl["Izz"] + tl["m"] * (tx * tx) + 0.000226
    print(f"tail I_yaw {It:.4f}")
    for gname in ("nominal", "brisk", "fr05"):
        T, d = b["gait"][gname]
        f = 1 / T
        for A in (0.12, 0.2):
            w, a = sin_peak(A, f)
            inert, damp = It * a, DAMP * w
            # rms of inertial (in quadrature) + (friction*sign(v) + damping*v)
            rms_pas = math.sqrt(FRIC**2 + 2 * FRIC * DAMP * (2 * w / math.pi) + DAMP**2 * w * w / 2)
            rms = math.sqrt(rms_pas**2 + inert**2 / 2)
            print(f"  {gname} stride {f:.2f} Hz A {A}: inertial pk {inert:.3f}, damping pk {damp:.3f}, "
                  f"rms(model) ~{rms:.3f} ({rms/ts['sus']:.2f} of A1), passive-only rms {rms_pas:.3f}")
    for A, f in ((0.25, 0.2), (0.15, 0.1)):
        w, a = sin_peak(A, f)
        rms_pas = math.sqrt(FRIC**2 + 2 * FRIC * DAMP * (2 * w / math.pi) + DAMP**2 * w * w / 2)
        print(f"  idle sway A {A} at {f} Hz: inertial {It*a:.4f}, passive rms {rms_pas:.3f}")
    # tail angular momentum vs legs (rough)
    T, d = b["gait"]["nominal"]
    Lt = It * 0.12 * 2 * math.pi / T
    print(f"tail yaw momentum amplitude at 0.12 rad nominal: {Lt:.4f} kg m^2/s")
    # trunk pitch oscillation cost
    trunkI = b["trunk"]["Iyy"] + b["trunk"]["m"] * sum(c * c for c in b["trunk"]["com"])
    headI = hd["Iyy"] + hd["m"] * ((xb + x) ** 2 + 0.1 ** 2)
    tailI = tl["Iyy"] + tl["m"] * ((0.2425 + abs(tx)) ** 2 + 0.03 ** 2)
    Iup = trunkI + headI + tailI
    for gname in ("nominal", "fr05"):
        T, d = b["gait"][gname]
        ws = 2 * math.pi * 2 / T
        print(f"upper-body pitch I about hips {Iup:.3f}; +-1 deg trunk pitch at step freq ({gname}) -> "
              f"{Iup*math.radians(1)*ws*ws:.2f} N m")
    # compass-gait bob: leg shortening a constant-height walk forces at mid-stance
    for gname in ("nominal", "fr05"):
        T, d = b["gait"][gname]
        half = d * b["stride"][gname] / 2
        dL = half * half / (2 * b["leg"])
        print(f"constant-height {gname}: stance half {half:.4f} m, mid-stance shortening {dL*1000:.1f} mm = {dL/b['H']:.4f} H")
    # constant swing time rule
    Tsw = b["gait"]["fr05"][0] * (1 - b["gait"]["fr05"][1])
    for gname in ("nominal", "brisk"):
        T, d = b["gait"][gname]
        print(f"constant swing {Tsw:.3f} s -> {gname} duty {1 - Tsw / T:.3f} (now {d}); swing speed x{T*(1-d)/Tsw:.2f}")

# T. rex dynamic similarity
v, H, fstep = 1.28, 3.1, 0.66
print("T. rex Fr", v * v / (G * H), "rel stride", 2 * 1.94 / H)
for name, b in bodies.items():
    print(name, "T. rex-similar step freq", fstep * math.sqrt(H / b["H"]), "Hz; robot fr05 cadence",
          2 / b["gait"]["fr05"][0], "nominal", 2 / b["gait"]["nominal"][0])
