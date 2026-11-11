"""Evaluation-only truth: world AABBs of task objects from LIBERO asset collision geoms + layout poses."""
import json, sys, xml.etree.ElementTree as ET
import numpy as np
L='/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/assets'
M='/workspace/gpt_field/output/foundation_harness_preparation_001/inputs/layout_manifest.json'
def qmat(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def boxes(xml):
    root=ET.parse(xml).getroot(); pts=[]
    for g in root.iter('geom'):
        if g.get('contype')=='0' and g.get('conaffinity')=='0': continue
        t=g.get('type','sphere')
        pos=np.array(g.get('pos','0 0 0').split(),float); quat=np.array(g.get('quat','1 0 0 0').split(),float)
        if t=='box':
            s=np.array(g.get('size').split(),float)
            c=np.array([[a,b,cc] for a in (-s[0],s[0]) for b in (-s[1],s[1]) for cc in (-s[2],s[2])])
            pts.append(c@qmat(quat).T+pos)
        elif t=='mesh':
            mesh=g.get('mesh'); asset=[m for m in root.iter('mesh') if m.get('name')==mesh][0]
            f=asset.get('file'); scale=np.array(asset.get('scale','1 1 1').split(),float)
            import os
            p=os.path.join(os.path.dirname(xml),f)
            if p.endswith('.obj'):
                v=np.array([[float(a) for a in l.split()[1:4]] for l in open(p) if l.startswith('v ')])
                pts.append((v*scale)@qmat(quat).T+pos)
    return np.concatenate(pts)
d=json.load(open(M)); poses=d['custom_after_passive_hold_object_poses']
assets={'white_yellow_mug_1':f'{L}/turbosquid_objects/white_yellow_mug/white_yellow_mug.xml',
        'porcelain_mug_1':f'{L}/turbosquid_objects/porcelain_mug/porcelain_mug.xml',
        'red_coffee_mug_1':f'{L}/turbosquid_objects/red_coffee_mug/red_coffee_mug.xml',
        'plate_1':f'{L}/stable_scanned_objects/plate/plate.xml','plate_2':f'{L}/stable_scanned_objects/plate/plate.xml'}
out={}
for k,xml in assets.items():
    try:
        local=boxes(xml)
    except Exception as e:
        print(k,'ERR',e); continue
    q=poses[k]['qpos']; R=qmat(q[3:]); t=np.array(q[:3])
    w=local@R.T+t
    out[k]={'lower':w.min(0).round(4).tolist(),'upper':w.max(0).round(4).tolist(),'size':(w.max(0)-w.min(0)).round(4).tolist()}
    print(k,out[k])
c=d['can']['collision_bounds']; out['can']={'lower':c['lower_xyz_m'],'upper':c['upper_xyz_m'],'size':c['size_xyz_m']}
out['table_top_z']=d['can']['table_height_m']
json.dump(out,open(sys.argv[1],'w'),indent=1)
