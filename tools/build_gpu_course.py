"""Create real box stairs and ramp lanes; preserve existing A1 assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import math


def build(source,out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'assets',out/'assets')
    root=ET.parse(source/'a1_flat.xml').getroot()
    # Warp MULTICCD box interactions require zero margin on both geoms.
    for geom in root.iter('geom'):
        geom.set('margin','0')
    world=root.find('worldbody')
    world.find("geom[@name='floor']").set('rgba','.26 .32 .38 1')
    def box(name,x,y,z,sx,sy,sz,color,euler=None):
        attrs=dict(name=name,type='box',pos=f'{x} {y} {z}',size=f'{sx} {sy} {sz}',
                   contype='1',conaffinity='0',condim='3',friction='1.0',margin='0',rgba=color)
        if euler:attrs['euler']=euler
        ET.SubElement(world,'geom',**attrs)
    # Six 3.5cm true risers, 40cm treads, 80cm summit and reverse descent.
    rise=.035
    for i in range(6):
        h=(i+1)*rise
        box(f'up_{i}',.2+.4*i,-3,h/2,.2,1.2,h/2,'.34 .57 .74 1')
        box(f'down_{i}',3.4+.4*i,-3,(6-i)*rise/2,.2,1.2,(6-i)*rise/2,'.34 .57 .74 1')
    box('stairs_summit',2.8,-3,.105,.4,1.2,.105,'.48 .72 .82 1')
    # Thin rotated boxes give an actual planar 6.84 degree incline.
    angle=math.atan(.12);thickness=.05
    length=math.sqrt(2**2+.24**2)/2
    box('ramp_up',1,3,.12-thickness*math.cos(angle),length,1.2,thickness,'.40 .64 .44 1',f'0 {-angle} 0')
    box('ramp_top',2.6,3,.12,.6,1.2,.12,'.53 .75 .49 1')
    box('ramp_down',4.2,3,.12-thickness*math.cos(angle),length,1.2,thickness,'.40 .64 .44 1',f'0 {angle} 0')
    # Self collision is disabled: any foot contact is with the environment.
    # Selecting only floor would miss box/ramp contacts.
    for sensor in root.findall('sensor/contact'):
        sensor.attrib.pop('geom2',None)
    key=root.find("keyframe/key[@name='home']")
    q=key.get('qpos').split();q[2]='.35';key.set('qpos',' '.join(q))
    ET.indent(root)
    ET.ElementTree(root).write(out/'a1_course.xml',encoding='unicode')
    manifest=dict(stairs_riser_m=rise,stairs_tread_m=.4,stairs_peak_m=.21,
                  slope_degrees=math.degrees(angle),slope_peak_m=.24,
                  lanes_y=[-3,0,3],lane_half_width=1.2,
                  source='Locally constructed MuJoCo primitives; distance-based terrain progression inspired by legged_gym, not copied geometry.',
                  xml_sha256=hashlib.sha256((out/'a1_course.xml').read_bytes()).hexdigest())
    (out/'course_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.source,a.out)
