import numpy as np, json
# agentview calibration from R/trial/requests/000001 geometry.cameras.external
pos = np.array([0.6065773716836134, 0.0, 0.96])
R = np.array([[-1.7233905013069872e-06,-0.5287697435529835,0.8487653140297038],
              [0.9999999999985034,-7.823149652530503e-07,1.5430955956352577e-06],
              [-1.5194045527300304e-07,0.848765314031093,0.5287697435535403]])
f, c = 618.0386719675123, 255.5
def ray(u, v):
    X = (u - c) / f; Y = -(v - c) / f
    return R @ np.array([X, Y, -1.0])
def cast(u, v, z):
    d = ray(u, v); t = (z - pos[2]) / d[2]; return pos + t * d
def project(p):
    q = R.T @ (np.asarray(p, float) - pos)
    return np.array([c + f * q[0] / (-q[2]), c - f * q[1] / (-q[2])])
def cell_uv(cell):
    r, k = divmod(cell - 1, 8); return (k + 0.5) * 64, (r + 0.5) * 64
def cell_of(uv):
    u, v = uv
    if not (0 <= u < 512 and 0 <= v < 512): return None
    return int(v // 64) * 8 + int(u // 64) + 1
print("camera +X (image right) world:", np.round(R[:, 0], 4), " view dir:", np.round(-R[:, 2], 4))
for z in (0.425, 0.44):
    print(f"\n-- plane z={z}")
    for cell in (35, 38, 43, 53, 54, 55, 60, 61, 62, 63, 64):
        u, v = cell_uv(cell); p = cast(u, v, z)
        corners = np.array([cast(u + du, v + dv, z)[:2] for du in (-32, 32) for dv in (-32, 32)])
        span = corners.max(0) - corners.min(0)
        print(f"cell {cell:2d} uv=({u:.0f},{v:.0f}) -> world ({p[0]:+.3f},{p[1]:+.3f})  cell footprint dx={span[0]:.3f} dy={span[1]:.3f}")
zt = 0.425
plate = np.array([0.149, 0.006]); rp = 0.069
print("\nplate centre ->", np.round(project([*plate, zt]), 1), "cell", cell_of(project([*plate, zt])))
for name, pt in [("plate +y edge", [plate[0], plate[1] + rp]), ("plate -y edge", [plate[0], plate[1] - rp]),
                 ("plate +x edge (near cam)", [plate[0] + rp, plate[1]]), ("plate -x edge", [plate[0] - rp, plate[1]]),
                 ("old-rule region start x=0.223", [0.223, 0.006]), ("x=0.30,y=0", [0.30, 0.0]), ("x=0.35,y=0", [0.35, 0.0]),
                 ("release (0.29,-0.02)", [0.29, -0.02]), ("final pudding relayout (0.276,-0.020)", [0.276, -0.020]),
                 ("t6_007 success (0.145,0.146)", [0.145, 0.146]), ("t6_009 final (0.187,0.170)", [0.187, 0.170]),
                 ("t6_006 old-rule region centre (0.234,-0.009)", [0.234, -0.009])]:
    uv = project([*pt, zt]); print(f"{name:45s} -> uv=({uv[0]:6.1f},{uv[1]:6.1f}) cell {cell_of(uv)}")
print("\nLIBERO plate_right_region box corners (x 0.10..0.20, y 0.05..0.15):")
for x in (0.10, 0.20):
    for y in (0.05, 0.15):
        uv = project([x, y, zt]); print(f"  ({x:.2f},{y:.2f}) -> uv=({uv[0]:6.1f},{uv[1]:6.1f}) cell {cell_of(uv)}")
print("\nbottom image edge (v=511) on plane: x at u=255.5 ->", np.round(cast(255.5, 511, zt), 3))
print("image bottom-left/right corners on plane:", np.round(cast(0, 511, zt), 3), np.round(cast(511, 511, zt), 3))
# P0.5 style sign tests for recorded plans
plans = {"relayout": (55, 61), "t6_006": (43, 60), "t6_007": (54, 53), "t6_001..005 (no region)": (55, 53), "t6_009": (0, 53), "t6_011": (0, 60)}
print("\nP0.5 sign test: region cell vs anchor cell and vs true plate centre (z=0.425)")
for k, (e, a) in plans.items():
    if e == 0:
        print(f"  {k:24s} region x=0 -> no pointed cell; check cannot run"); continue
    pe = cast(*cell_uv(e), zt)[:2]; pa = cast(*cell_uv(a), zt)[:2]
    print(f"  {k:24s} E{e}->({pe[0]:+.3f},{pe[1]:+.3f}) C{a}->({pa[0]:+.3f},{pa[1]:+.3f})  d=+y: (E-C).y={pe[1]-pa[1]:+.3f}  (E-plate).y={pe[1]-plate[1]:+.3f} | old +x: (E-C).x={pe[0]-pa[0]:+.3f} (E-plate).x={pe[0]-plate[0]:+.3f}; edge test y>{plate[1]+rp:.3f}? {pe[1] > plate[1]+rp}")
