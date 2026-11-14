"""Anatomical stations. All distances are millimetres in a level chest bracket.

Torso rows: longitudinal station, half breadth, ventral height, dorsal height.
x=0 is the hip-yaw centerline; heights are measured from pelvis body origin.
Named spans deliberately leave windows for service or flexing and have changing section.
"""
FAMILIES = {
    'A': dict(name='Tapered flank', 
        torso=[(-195,57,38,140),(-150,62,27,145),(-89,85,36,161),(-25,89,45,165),
               (20,78,41,167),(75,83,7,166),(120,91,-12,156),(162,77,-2,141),(197,39,50,121)],
        islands=[('iliac',-190,-24, -57,229),('saddle',-23,63,5,175),
                 ('scapula',64,194,-76,248)],
        neck_run=84,neck_rise=58, head_len=229,head_width=65,head_depth=100,
        tail_len=384, tail_width=55,tail_depth=97,tail_drop=33,tail_curve=10,
        tail_pivot=-225,head_pitch=0,thigh=33,thickness=1.45,
        color=(.23,.39,.38), accents=(.42,.55,.49)),
    'B': dict(name='Deep ribcage',
        torso=[(-215,70,34,145),(-158,74,16,156),(-102,94,19,172),(-41,100,25,178),
               (13,90,27,180),(63,103,-17,185),(112,113,-43,181),(154,109,-39,164),
               (187,76,-7,144),(204,45,48,125)],
        islands=[('iliac',-209,-40,-58,232),('saddle',-39,49,-15,195),
                 ('scapula',50,199,-82,251)],
        neck_run=72,neck_rise=67,head_len=252,head_width=76,head_depth=120,
        tail_len=429,tail_width=69,tail_depth=109,tail_drop=41,tail_curve=20,
        tail_pivot=-232,head_pitch=-3,thigh=41,thickness=1.6,
        color=(.29,.39,.34),accents=(.52,.56,.43)),
    'C': dict(name='Open rib frame',
        torso=[(-200,61,41,140),(-144,65,32,156),(-82,91,42,173),(-23,90,46,177),
               (24,77,45,175),(79,91,4,167),(124,99,-21,151),(167,83,-11,136),(199,45,45,123)],
        islands=[('iliac',-196,-49,-15,193),('saddle',-48,72,42,138),
                 ('shoulder',73,194,20,160)],
        neck_run=97,neck_rise=63,head_len=241,head_width=69,head_depth=105,
        tail_len=442,tail_width=60,tail_depth=98,tail_drop=17,tail_curve=19,
        tail_pivot=-225,head_pitch=2,thigh=29,thickness=1.5,
        color=(.23,.34,.37),accents=(.62,.53,.34))
}
