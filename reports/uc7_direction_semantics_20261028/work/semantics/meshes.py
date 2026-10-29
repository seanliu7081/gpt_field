"""Object footprints from LIBERO XML + meshes (no simulator). Read-only on LIBERO."""
import struct, os, re, sys, json, copy
import numpy as np
import xml.etree.ElementTree as ET
A = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/assets"
def qmat(q):
    w,x,y,z = np.array(q, float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def read_msh(p):
    b = open(p,'rb').read(); nv,nn,nt,nf = struct.unpack('4i', b[:16])
    return np.frombuffer(b[16:16+12*nv], dtype='<f4').reshape(nv,3).astype(float)
def read_stl(p):
    b=open(p,'rb').read()
    if b[:5]==b'solid' and b'facet' in b[:300]:
        V=[[float(t) for t in l.split()[1:4]] for l in b.decode('latin1').splitlines() if l.strip().startswith('vertex')]
        return np.array(V)
    n=struct.unpack('<I', b[80:84])[0]
    a=np.frombuffer(b[84:84+50*n], dtype=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')]))
    return a['v'].reshape(-1,3).astype(float)
def read_obj(p):
    V=[]
    for line in open(p, encoding='latin1'):
        if line.startswith('v '): V.append([float(t) for t in line.split()[1:4]])
    return np.array(V)
def fv(s, n=3, d=None):
    return np.array([float(t) for t in s.split()]) if s else d
def object_points(xml_path, use="visual"):
    """Return points (body frame of root) of visual meshes (group 1) or collision boxes (group 0)."""
    root = ET.parse(xml_path).getroot()
    base = os.path.dirname(xml_path)
    meshes = {}
    for m in root.iter('mesh'):
        meshes[m.get('name')] = (os.path.join(base, m.get('file')), fv(m.get('scale'), d=np.ones(3)))
    pts = []
    def walk(body, T):
        R, t = T
        bp = fv(body.get('pos'), d=np.zeros(3)); bq = fv(body.get('quat'), d=np.array([1,0,0,0.]))
        Rb = R @ qmat(bq); tb = t + R @ bp
        for g in body.findall('geom'):
            grp = g.get('group', '0'); typ = g.get('type', 'sphere')
            gp = fv(g.get('pos'), d=np.zeros(3)); gq = fv(g.get('quat'), d=np.array([1,0,0,0.]))
            Rg = Rb @ qmat(gq); tg = tb + Rb @ gp
            if use == "visual" and typ == "mesh" and grp == "1":
                f, sc = meshes[g.get('mesh')]
                V = read_msh(f) if f.endswith('.msh') else (read_stl(f) if f.lower().endswith('.stl') else read_obj(f))
                pts.append((Rg @ (V*sc).T).T + tg)
            if use == "collision" and typ == "box" and grp == "0":
                s = fv(g.get('size'))
                c = np.array([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])*s
                pts.append((Rg @ c.T).T + tg)
        for b in body.findall('body'):
            walk(b, (Rb, tb))
    wb = root.find('worldbody')
    # robosuite MujocoXMLObject._get_object_subtree() makes ./body/body[@name='object'] the root body ("main");
    # its own pos/quat are overridden by the free joint qpos (objects) or by body_pos/body_quat (LIBERO fixtures).
    obj = wb.find("./body/body[@name='object']")
    obj = copy.deepcopy(obj); obj.attrib.pop('pos', None); obj.attrib.pop('quat', None)
    walk(obj, (np.eye(3), np.zeros(3)))
    return np.vstack(pts) if pts else np.zeros((0,3))
def yawq(deg):
    a = np.radians(deg); return np.array([np.cos(a/2),0,0,np.sin(a/2)])
OBJ = {
 # name: (xml, effective init quat wxyz, why)
 "plate": (f"{A}/stable_scanned_objects/plate/plate.xml", yawq(90), "GoogleScanned rotation (pi/2,pi/2) axis x; xyzw [s,0,0,c] read as wxyz -> yaw +90"),
 "white_bowl": (f"{A}/stable_scanned_objects/white_bowl/white_bowl.xml", yawq(90), "GoogleScanned, same quirk"),
 "akita_black_bowl": (f"{A}/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml", yawq(90), "GoogleScanned, same quirk"),
 "chocolate_pudding": (f"{A}/stable_hope_objects/chocolate_pudding/chocolate_pudding.xml", yawq(180), "HOPE rotation (0,0) axis x; xyzw [0,0,0,1] read as wxyz -> yaw 180"),
 "porcelain_mug": (f"{A}/turbosquid_objects/porcelain_mug/porcelain_mug.xml", yawq(-90), "Turbosquid rotation (-pi/2,-pi/2) axis x -> wxyz (-.707,0,0,.707) = yaw -90"),
 "white_yellow_mug": (f"{A}/turbosquid_objects/white_yellow_mug/white_yellow_mug.xml", yawq(-90), "same"),
 "red_coffee_mug": (f"{A}/turbosquid_objects/red_coffee_mug/red_coffee_mug.xml", yawq(-90), "same"),
 "desk_caddy": (f"{A}/turbosquid_objects/desk_caddy/desk_caddy.xml", yawq(180), "fixture, region yaw (pi,pi), proper wxyz"),
 "flat_stove": (f"{A}/articulated_objects/flat_stove.xml", yawq(0), "fixture, default yaw [0,0]"),
}
if __name__ == "__main__":
    out = {}
    for name, (xml, q, why) in OBJ.items():
        R = qmat(q)
        res = {"xml": xml, "init_quat_wxyz": np.round(q,4).tolist(), "why": why}
        for use in ("visual", "collision"):
            P = object_points(xml, use)
            if len(P) == 0: continue
            W = (R @ P.T).T
            mn, mx = W.min(0), W.max(0)
            r = np.max(np.linalg.norm(W[:,:2], axis=1))
            res[use] = {"world_aabb_min": np.round(mn,4).tolist(), "world_aabb_max": np.round(mx,4).tolist(),
                        "half_xy": np.round((mx[:2]-mn[:2])/2,4).tolist(),
                        "centre_xy_offset": np.round((mx[:2]+mn[:2])/2,4).tolist(),
                        "max_xy_radius_from_origin": round(float(r),4), "n": int(len(P))}
        out[name] = res
        print(name, json.dumps({k:v for k,v in res.items() if k!='xml'}, indent=None))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "footprints.json"), "w"), indent=1)
