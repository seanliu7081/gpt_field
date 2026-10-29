import re, glob, os, numpy as np
P = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/envs/problems"
def R_wxyz(q):
    w,x,y,z = np.array(q, float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
for f in sorted(glob.glob(P+"/*.py")):
    src = open(f).read()
    for m in re.finditer(r'set_camera\(\s*camera_name="(\w+)",\s*pos=\[([^\]]*)\],\s*quat=\[([^\]]*)\]', src, re.S):
        name = m.group(1)
        pos = [float(v) for v in m.group(2).replace("\n"," ").split(",") if v.strip()]
        quat = [float(v) for v in m.group(3).replace("\n"," ").split(",") if v.strip()]
        line = src[:m.start()].count("\n")+1
        R = R_wxyz(quat)
        right, up, view = R[:,0], R[:,1], -R[:,2]
        pitch = np.degrees(np.arcsin(-view[2]))
        print(f"{os.path.basename(f)}:{line} {name:20s} pos={np.round(pos,3)} quat={np.round(quat,4)}")
        print(f"    image-right R[:,0]={np.round(right,4)}  image-up R[:,1]={np.round(up,4)}  view -R[:,2]={np.round(view,4)}  down-pitch={pitch:.1f}deg")
