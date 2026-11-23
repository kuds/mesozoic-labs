// Triceratops standing layout. Units: mm. +X points forward, +Y left, +Z up.
// Editable packaging concept. Open in OpenSCAD, select a view and press F5.
// Curved solids and link centerlines reserve volume. They do not define
// load-bearing joints, tolerances, cable paths, production shell thickness,
// mass properties or a locomotion model.

height_mm = 750;                  // [600,750]
view_mode = "assembled";         // [assembled,skeleton,packaging,shell]
show_drive_envelopes = true;
show_floor = false;
quality = 32;                    // [16,32,64]

// Dimensions at the 750 mm design point. Scalable geometry is authored below
// in these coordinates and uniformly transformed once at the top level.
stance_length = 650;                 // shoulder to hip and front to rear pad centers
track = 350;                     // left to right pad centers
girdle_z = 400;
upper_len = 235;
lower_len = 230;
ankle_z = 45;
pad_xy = [110,80];
front_overhang = 365;            // measured forward from shoulder station
rear_overhang = 485;             // measured backward from hip station
frill_top = 750;
scale_factor = height_mm / 750;
front_x = stance_length/2;
rear_x = -stance_length/2;
nose_x = front_x + front_overhang;
tail_x = rear_x - rear_overhang;
$fn = quality;

// Two X4 roll/pitch modules at each girdle and one X6 flexion module
// mounted high on each upper link. Equal pulleys turn its elbow/stifle.
// Catalog dimensions exclude plugs, shaft extensions and cable bends.
// X4: https://www.myactuator.com/x4-36-l-details
// X6: https://myactuator.cn/uploadpic/20250516142200rpyh.pdf
x4_d_mm = 60;
x4_length_mm = 64;
x6_d_mm = 80;
x6_length_mm = 67.5;
flexion_motor_offset_mm = 90; // shaft offset along upper link, both sizes
pulley_radius_mm = 20;        // 1:1 planning geometry; tooth profile TBD
belt_width_mm = 12;           // installed space allowance
compute_bay_mm = [120,110,65];
pack_bay_mm = [160,90,65];

// Circle intersection for a crouched standing pose, with pad under girdle.
// The front elbow folds aft and hind stifle folds forward.
d = girdle_z-ankle_z;
knee_drop = (upper_len*upper_len-lower_len*lower_len+d*d)/(2*d);
knee_reach = sqrt(upper_len*upper_len-knee_drop*knee_drop);
function girdle(front,side) = [front ? front_x : rear_x, side*track/2, girdle_z];
function foot(front,side) = [front ? front_x : rear_x, side*track/2, ankle_z];
function bend(front,side) = girdle(front,side) +
                            [(front ? -1 : 1)*knee_reach,0,-knee_drop];
assert(height_mm == 600 || height_mm == 750, "Select a 600 or 750 mm study.");
assert(d < upper_len+lower_len && d > abs(upper_len-lower_len));
assert(abs(norm(bend(true,1)-girdle(true,1))-upper_len) < 0.001);
assert(abs(norm(bend(true,1)-foot(true,1))-lower_len) < 0.001);
echo(crown_mm=height_mm, length_mm=(nose_x-tail_x)*scale_factor,
     track_mm=track*scale_factor, stance_length_mm=stance_length*scale_factor,
     upper_link_mm=upper_len*scale_factor, lower_link_mm=lower_len*scale_factor);

// Scaled structure, dark skin frames, pale bone and flexible horn envelopes.
frame = [0.19,0.29,0.32];
link = [0.44,0.57,0.57];
skin = [0.24,0.49,0.40];
crest = [0.33,0.56,0.43];
horn = [0.87,0.78,0.56];
pad = [0.16,0.19,0.19];

module rounded_link(a,b,r1,r2) {
    hull() {
        translate(a) sphere(r1);
        translate(b) sphere(r2);
    }
}
module oval(p,radii) {
    translate(p) scale(radii) sphere(1);
}
module scaled_structure() {
    color(frame) {
        for (side=[-1,1])
            rounded_link([rear_x,side*104,385],[front_x,side*104,385],12,12);
        for (station=[front_x,rear_x])
            rounded_link([station,-track/2,400],[station,track/2,400],18,18);
        // Lower torso tray and its suspension.
        translate([-80,0,325]) cube([390,145,10],center=true);
        for (x=[-255,95], s=[-1,1])
            rounded_link([x,s*68,325],[x,s*104,385],6,6);
        rounded_link([front_x-28,0,398],[405,0,500],22,20);
    }
    for (front=[false,true], s=[-1,1]) {
        a=girdle(front,s); b=bend(front,s); c=foot(front,s);
        color(link) {
            rounded_link(a,b,20,16);
            rounded_link(b,c,16,11);
        }
        color(frame) rounded_link(c,[c[0],c[1],18],10,14);
        color(pad)
            translate([c[0],c[1],9]) cube([pad_xy[0],pad_xy[1],18],center=true);
    }
}
module frill() {
    // Draw in transverse/vertical coordinates, lean crest aft. The top vertex
    // is exactly frill_top. Open windows keep the display outline readable.
    base_z = 466;
    h = frill_top-base_z;
    color(crest)
    multmatrix([[0,-0.38,1,431],[1,0,0,0],[0,1,0,base_z],[0,0,0,1]])
    linear_extrude(height=12,center=true)
    difference() {
        polygon([[-78,0],[-130,47],[-174,135],[-168,189],[-147,214],
                 [-130,246],[-102,253],[-77,273],[-40,266],[0,h],
                 [40,266],[77,273],[102,253],[130,246],[147,214],
                 [168,189],[174,135],[130,47],[78,0]]);
        for (s=[-1,1])
            translate([s*75,147]) rotate(-s*22) scale([43,79]) circle(1);
    }
}
module head_shell() {
    color(skin) difference() {
        hull() {
            oval([435,0,505],[65,89,75]);
            oval([575,0,476],[49,51,39]);
        }
        hull() {
            oval([436,0,504],[61,85,71]);
            oval([574,0,476],[44,46,34]);
        }
        // Open underside for visualization.
        translate([500,0,425]) cube([300,230,45],center=true);
    }
    frill();
    // Soft rounded tips terminate at the prescribed front extent.
    color(horn) {
        for (s=[-1,1])
            rounded_link([470,s*59,545],[nose_x-7,s*80,590],19,7);
        rounded_link([567,0,500],[605,0,552],14,6);
    }
}
module body_outline() {
    // Open ribs expose torso packaging in the assembled view.
    for (x=[-250,-70,130,255])
        color(skin)
        translate([x,0,395])
        rotate([0,90,0])
        linear_extrude(height=9,center=true)
        difference() {
            scale([113,140]) circle(1);
            scale([103,130]) circle(1);
            translate([75,0]) square([80,250],center=true);
        }
    // A tapered tail extends from the rear frame to the nominal aft extent.
    color(skin) hull() {
        oval([rear_x+28,0,401],[63,72,57]);
        oval([-540,0,362],[45,38,31]);
    }
    color(skin) hull() {
        oval([-540,0,362],[45,38,31]);
        translate([tail_x+5,0,310]) sphere(5);
    }
}
module drive(p, diameter, length, roll=false) {
    translate(p) rotate(roll ? [0,90,0] : [90,0,0])
        cylinder(d=diameter,h=length,center=true);
}
module knee_transmission(a,b,side) {
    // All inputs are installed physical coordinates. The X6 moves with the
    // upper link. Only the passive output pulley and axle sit at its distal end.
    here = a + (b-a)/norm(b-a)*flexion_motor_offset_mm;
    lane = [0,-side*(x6_length_mm/2 + 2 + belt_width_mm/2),0];
    color([0.32,0.37,0.42]) drive(here,x6_d_mm,x6_length_mm);
    color([0.51,0.56,0.58]) {
        drive(here+lane, 2*pulley_radius_mm, belt_width_mm);
        drive(b+lane, 2*pulley_radius_mm, belt_width_mm);
        drive(b+lane/2, 16, abs(lane[1])+8);
    }
    // Two equal-radius cylinders form an open belt path. Tooth pitch, wrap,
    // tensioner, guard and driven-shaft sizing require a load model.
    color([0.24,0.20,0.15]) difference() {
        hull() for (p=[here+lane,b+lane])
            drive(p,2*pulley_radius_mm+4,belt_width_mm);
        hull() for (p=[here+lane,b+lane])
            drive(p,2*pulley_radius_mm,belt_width_mm+1);
    }
}
module fixed_parts() {
    // Central compute and aft battery sit above the tray at both sizes.
    color([0.23,0.41,0.62])
        translate([-18*scale_factor,0,330*scale_factor + compute_bay_mm[2]/2])
        cube(compute_bay_mm,center=true);
    color([0.76,0.45,0.20])
        translate([-200*scale_factor,0,330*scale_factor + pack_bay_mm[2]/2])
        cube(pack_bay_mm,center=true);
    if (show_drive_envelopes)
        for (front=[false,true], s=[-1,1]) {
            a = girdle(front,s)*scale_factor;
            b = bend(front,s)*scale_factor;
            color([0.31,0.34,0.37]) {
                drive(a + [0,-s*80*scale_factor,0],x4_d_mm,x4_length_mm,true);
                drive(a,x4_d_mm,x4_length_mm);
            }
            knee_transmission(a,b,s);
        }
}
if (view_mode=="assembled" || view_mode=="skeleton")
    scale([scale_factor,scale_factor,scale_factor]) scaled_structure();
if (view_mode=="assembled" || view_mode=="shell")
    scale([scale_factor,scale_factor,scale_factor]) {
        body_outline();
        head_shell();
    }
if (view_mode=="assembled" || view_mode=="skeleton" || view_mode=="packaging")
    fixed_parts();
if (show_floor)
    %translate([(nose_x+tail_x)*scale_factor/2,0,-2])
        cube([(nose_x-tail_x+60)*scale_factor,(track+140)*scale_factor,4],center=true);
