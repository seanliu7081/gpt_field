import re, glob, os, numpy as np
P = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/envs/problems"
def qm(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
for f in sorted(glob.glob(P + "/*.py")):
    txt = open(f).read()
    for m in re.finditer(r'camera_name="(\w+)",\s*pos=\[([^\]]*)\],\s*quat=\[([^\]]*)\]', txt, re.S):
        name = m.group(1)
        if 'agentview' not in name: continue
        pos = [float(v) for v in m.group(2).replace('\n', ' ').split(',') if v.strip()]
        quat = [float(v) for v in m.group(3).replace('\n', ' ').split(',') if v.strip()]
        R = qm(quat)
        right = R[:2, 0] / np.linalg.norm(R[:2, 0])
        away_view = -R[:2, 2]; nv = np.linalg.norm(away_view)
        up = R[:2, 1] / np.linalg.norm(R[:2, 1])
        print(f"{os.path.basename(f):45s} {name:20s} pos={np.round(pos,3)} right={np.round(right,3)} away(-Z)={np.round(away_view/nv,3)} |-Zxy|={nv:.3f} imgup_xy={np.round(up,3)}")
