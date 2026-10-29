import numpy as np
# wrist calibration at step 0 (relayout trial request 000001 geometry.cameras.wrist)
pos = np.array([0.0013740237971336597, -0.007887677621773618, 0.790080885011893])
R = np.array([[-0.0015285862551420852, 0.998478396441735, 0.055123091923458385],
              [-0.9999986804637666, -0.0014959443950389546, -0.0006334201543179445],
              [-0.0005499952595571429, -0.055123987423882935, 0.9984793655933535]])
f, c = 333.6256954473487, 255.5
def project(p):
    q = R.T @ (np.asarray(p, float) - pos)
    return np.array([c + f * q[0] / (-q[2]), c - f * q[1] / (-q[2])])
def cell_of(uv):
    u, v = uv
    if not (0 <= u < 512 and 0 <= v < 512): return None
    return int(v // 64) * 8 + int(u // 64) + 1
print("wrist image-right world:", np.round(R[:, 0], 3), " image-up world:", np.round(R[:, 1], 3))
for name, p in [("pudding (-0.16,-0.14)", (-0.16, -0.14, 0.45)), ("white mug (-0.12,0.18)", (-0.12, 0.18, 0.48)),
                ("plate (0.149,0.006)", (0.149, 0.006, 0.44)), ("red mug (0.06,-0.20)", (0.06, -0.20, 0.48)),
                ("LIBERO right region centre (0.15,0.10)", (0.15, 0.10, 0.43)), ("old +x region (0.26,0.0)", (0.26, 0.0, 0.43))]:
    uv = project(p); print(f"{name:40s} -> wrist uv=({uv[0]:6.1f},{uv[1]:6.1f}) cell {cell_of(uv)}")
print("planner's own wrist pointing (relayout plan): pudding w=55, white mug body w=49, plate w=12, region E w=29")

def cast(u, v, z):
    X = (u - c) / f; Y = -(v - c) / f
    d = R @ np.array([X, Y, -1.0]); t = (z - pos[2]) / d[2]; return pos + t * d
def cell_uv(cell):
    r, k = divmod(cell - 1, 8); return (k + 0.5) * 64, (r + 0.5) * 64
for cell in (29, 11, 12, 13, 20, 21):
    p = cast(*cell_uv(cell), 0.425)
    print(f"wrist cell {cell} -> world ({p[0]:+.3f},{p[1]:+.3f}) on z=0.425")
print("external cell 55 world (0.099,0.185) appears in wrist cell", cell_of(project((0.099, 0.185, 0.425))))
print("if the wrist image were turned 180deg (k=2), wrist cell 29 would show the area that is now at cell", 65 - 29)
p = cast(*cell_uv(65 - 29), 0.425); print(f"  -> world ({p[0]:+.3f},{p[1]:+.3f})")
