"""Summarize hardware review outputs from screen.py."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def table_row(cols):
    return '| ' + ' | '.join(str(x) for x in cols)+' |'

def write_report():
    data=json.loads((HERE/'screen.json').read_text())
    baseline=next(iter(data))
    ref={q['name']:q for q in data[baseline]['components']}
    with (HERE/'installed_layout.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['model','assembly_or_reserve','center_x_mm','center_y_mm','center_z_mm','delta_x_from_source_mm',
            'delta_y_from_source_mm','delta_z_from_source_mm','body_local_full_size_mm','rotation_world_from_geom_xyz'])
        for name,r in data.items():
            for q in r['components']:
                n=q['name']; ctr=q['world_center_mm']; base=ref.get(n,{}).get('world_center_mm')
                dx=[round(v-a,3) for v,a in zip(ctr,base)] if base else ['', '', '']
                w.writerow([name,n,*ctr,*dx,' × '.join(str(x) for x in q['full_size_xml_mm']),json.dumps(q['world_local_xyz_rotation'])])
    lines=['# Installation and service screen','',
        'The review uses real skin triangles and vendor-sized cases, including hardware at its upper declared dimensional tolerance. The companion ledger lists the center, dimensions and rotation of each installed case and each reservation in the flat-foot review pose. Coordinate changes relative to the executable source are in millimetres. The source row in the data is named `model`. Consult the package source record for its revision.','',
        '![A10 and C10 left-battery service projections](service_paths.png)','','## Standing geometry','',
        table_row(['Model','Case / skin hits¹','Reserve / skin hits¹','Fixed blockers on left battery stroke²','Fixed blockers on top compute lift²','Upward fan rays open','Shortest clear height³']),
        table_row(['---','---:','---:','---:','---:','---:','---:'])]
    for name,r in data.items():
        c=r['cooling']; b=r['battery_routes']['left']; h=r['shell_contacts']
        size=lambda kind:sum(x['type']==kind and x['test']=='intersection' for x in h)
        lines.append(table_row([name if name!=baseline else 'Source',size('case') if r['covers_tested'] else 'n/a',
            size('reserve') if r['covers_tested'] else 'n/a',len(b['remaining_after_removing_all_intersecting_covers']),
            len(r['compute_service']['straight_routes']['top']['fixed_after_cover_removal']),
            f"{c['rays_open_to_300mm']}/100",f"{c['nearest_obstruction_mm']:.1f} mm" if c['nearest_obstruction_mm'] is not None else '>300 mm']))
    lines += ['','¹ Count of intersecting hardware–skin pairs at zero added margin. The JSON also retains near hits inside a 2 mm axis-wise guard. A reservation intersecting a cover means its allocated installation or access space has been consumed. A solid intersection is a physical fit failure. Original-source skin pairs are omitted from this column because their thin-wall geometry is unspecified.','',
        '² Beyond the installed pack plus its 6 mm all-around sleeve, the guard adds 2 mm on all faces. On the straight 240 mm robot-left stroke the entire swept volume is tested continuously. Crossing the far-side skin at the initial guarded pose also counts. The pack case is tolerance-max, and the plug and balance lead are assumed restrained. For the processor, raising the tolerance-max kit by 50 mm is a partial service lift. A full 250 mm top pull, two 200 mm side pulls and four hypothetical 30 mm midpoint plug-approach boxes are reported in `screen.json`. Placement of actual USB, barrel and Ethernet ports and the board-removal sequence need a detailed layout. A removal route may require a removable bridge or tray.','',
        '³ Smallest first-hit distance from a 40 × 40 mm assumed central fan face, ten-by-ten grid. The lateral grid uses 50 rays per direction 3–15 mm above the kit. A clear sightline shows a geometrically open path. Airflow, static pressure and cable blockage are unmeasured.','',
        '## Assembly evidence','']
    for name in data:
        if name==baseline or not name.endswith('10'):
            continue
        r=data[name]; b=r['battery_routes']['left']; lift=r['compute_service']['straight_routes']['top']
        left_skins=[n.removeprefix('cover_').replace('_',' ') for n in b['blockers'] if n in r['covers_tested']]
        fixed=b['remaining_after_removing_all_intersecting_covers']
        ob=r['case_to_structure_inspection']
        issue=[' / '.join(pair) for pair in ob if not pair[0].startswith(('tof_','oak_'))]
        lines.append(f"**{name}**. Left battery withdrawal crosses {len(left_skins)} skin pieces ({', '.join(left_skins) if left_skins else 'none'}) and {len(fixed)} fixed items. The 50 mm and 250 mm processor lifts cross {len(lift['fixed_after_cover_removal'])} and {len(r['compute_service']['straight_routes']['top_full']['fixed_after_cover_removal'])} fixed items respectively after skin removal. Remaining static structural contacts for non-optical boxes: {', '.join(issue) if issue else 'none in the checked frame members'}.")
        lines.append('')
    lines += ['The JSON records case–case intersections, contacts with frame members and parent-reserve containment margins separately. The two OAK braces and ToF struts terminate at their optical case/board by design. Sensor carrier brackets, board standoffs and strain relief require detailed load paths. A mount represented by a strut penetrating a supplier envelope needs a real mounting interface.','',
              '## Supported joints and manufacturing','']
    for name in data:
        if name=='C10':
            r=data[name]['shaft_geometry']
            lines.append(f"The C10 neck has an {r['neck']['model_shaft_diameter_mm']:.0f} mm shaft between bearing centers {r['neck']['bearing_center_spacing_mm']:.0f} mm apart. The tail has an {r['tail']['model_shaft_diameter_mm']:.0f} mm shaft with a {r['tail']['bearing_center_spacing_mm']:.0f} mm span. Both shaft and bearing axes match their modeled hinge within {max(x['max_axis_offset_mm'] for x in r.values()):.3f} mm radial offset. These checks establish coaxial model datums. The bearing types, shaft stress, hub fits and retention remain to be specified.")
    lines += ['','Specify real motor and bearing part numbers/revisions, face offsets, service tool sides and cable paths in a 1:1 mock-up. Fit a tolerance-max battery, dummy EC5 harness, working latch and removable flank; measure extraction force and tool clearance with the frame supported. Gauge kit-to-member clearance using the actual cooler, plugs, M.2 standoff and hose/duct or grille. Run the closed-cover thermal test under design duty. Verify each selector, fuse, hard stop and disconnect through the service opening.','',
              '## Reproduction','',
              'Use the reference model and the nine generated candidate XMLs. From the study package directory, run `python hardware/screen.py /path/to/reference/model.xml model/generated/*.xml`, then `python hardware/report.py`. Use `python hardware/service_figure.py` for the plan-view plate. Python with MuJoCo 3.10 and NumPy is sufficient for screening. The plate additionally uses SciPy and Matplotlib. Results bind each row to the SHA-256 of the XML that produced it.','',
              'The solver checks actual mesh triangles against oriented tolerance boxes via the separating-axis theorem. Capped motor cylinders use triangle clipping to their axial interval followed by a circular test. Battery and kit strokes against capsules use exact segment-to-box distance. Other cylinder obstacles use a 96-sided circumscribed prism. For the 30 mm case radius its maximum radial overestimate is below 0.017 mm. Ellipsoid obstacles use enclosing boxes. Left/right battery strokes are parallel to a pack-local axis for the distributed fixtures. The JSON labels any other non-axis-aligned stroke that needed a conservative enclosing box. Dynamic leg/sensor checks are in the mechanics appendix.','',
              'See [part sizes, sources, cooling target and validation sequence](component_basis.md), [complete numeric results](screen.json), and [compiled component layout and deltas](installed_layout.csv).']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    write_report()
