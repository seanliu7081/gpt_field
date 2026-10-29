"""Approximate xy extents of LIBERO objects from collision geoms (boxes) or *_col.obj meshes with XML scale."""
import re, sys, os, numpy as np, xml.etree.ElementTree as ET
A = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/assets"
def quat_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def obj_vertices(path):
    v = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                v.append([float(t) for t in line.split()[1:4]])
    return np.array(v)
def extent(xml_path):
    root = ET.parse(xml_path).getroot()
    meshes = {m.get('name'): m for m in root.iter('mesh')}
    pts = []
    for g in root.iter('geom'):
        if g.get('group') not in (None, '0') and g.get('contype') == '0':
            continue
        pos = np.array([float(t) for t in (g.get('pos') or '0 0 0').split()])
        q = [float(t) for t in (g.get('quat') or '1 0 0 0').split()]
        R = quat_mat(q)
        if g.get('type') == 'box':
            s = np.array([float(t) for t in g.get('size').split()])
            corners = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
            pts.append(corners @ R.T + pos)
        elif g.get('type') == 'mesh' and g.get('mesh') in meshes:
            m = meshes[g.get('mesh')]
            f = m.get('file')
            sc = np.array([float(t) for t in (m.get('scale') or '1 1 1').split()])
            p = os.path.join(os.path.dirname(xml_path), f)
            if p.endswith('.obj') and os.path.exists(p):
                v = obj_vertices(p) * sc
                pts.append(v @ R.T + pos)
    if not pts:
        return None
    P = np.vstack(pts)
    return P.min(0), P.max(0)
for rel in sys.argv[1:]:
    p = os.path.join(A, rel)
    e = extent(p)
    if e is None:
        print(rel, 'no geoms parsed'); continue
    lo, hi = e
    print(f"{rel}: x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] z[{lo[2]:+.3f},{hi[2]:+.3f}]  half-xy=({(hi[0]-lo[0])/2:.3f},{(hi[1]-lo[1])/2:.3f}) radius_xy={np.max(np.hypot(*np.array([lo[:2],hi[:2]]).T)):.3f}")
