import numpy as np, json, subprocess
np.set_printoptions(precision=4, suppress=True)
RX, RY = (0.10, 0.20), (0.05, 0.15)       # plate_right_region, world (living-room offset 0)
RP_VIS, RP_COL = 0.0688, 0.0732             # plate radius: visual max radius / collision-box corner radius
PUD = {"init yaw 180 (long axis along x)": (0.0417, 0.0247), "rotated 90 (long axis along y)": (0.0247, 0.0417)}
def overlaps(cx, cy, hx, hy, px, py, r):
    dx = np.clip(px - cx, -hx, hx); dy = np.clip(py - cy, -hy, hy)
    return np.hypot(px - (cx + dx), py - (cy + dy)) < r
def band(px, py, hx, hy, r, n=401):
    xs = np.linspace(RX[0], RX[1], n)[1:-1]; ys = np.linspace(RY[0], RY[1], n)[1:-1]
    X, Y = np.meshgrid(xs, ys)
    ov = overlaps(X, Y, hx, hy, px, py, r)
    frac = ov.mean()
    # minimal y without overlap at x = plate x (worst case) and best x within region
    col = ~overlaps(np.full_like(ys, np.clip(px, *RX)), ys, hx, hy, px, py, r)
    ymin_at_px = ys[col][0] if col.any() else None
    ymin_any = Y[~ov].min() if (~ov).any() else None
    return frac, ymin_at_px, ymin_any
out = {}
plate_ep0 = (0.149, 0.0057)
for label, (hx, hy) in PUD.items():
    for rname, r in (("visual", RP_VIS), ("collision", RP_COL)):
        frac, y1, y2 = band(*plate_ep0, hx, hy, r)
        out[f"ep0 plate {plate_ep0} | pudding {label} | plate r {rname} {r}"] = dict(
            region_fraction_overlapping_plate=round(float(frac), 3),
            min_centre_y_no_overlap_at_plate_x=round(float(y1), 4), usable_band_y=[round(float(y1),4), RY[1]],
            band_width_m=round(RY[1]-float(y1), 4), min_centre_y_no_overlap_anywhere=round(float(y2),4))
for k, v in out.items(): print(k, "\n   ", v)
# t6 runs
for name, (x, y) in {"t6_007 (success, pudding still held)": (0.1446, 0.1454), "t6_009 (failure)": (0.1869, 0.1704), "uc7 relayout final": (0.276, -0.020)}.items():
    mx = min(x - RX[0], RX[1] - x); my = min(y - RY[0], RY[1] - y)
    gap = (y - 0.0247) - (plate_ep0[1] + RP_VIS)
    print(f"{name}: origin=({x},{y}) inside={RX[0]<x<RX[1] and RY[0]<y<RY[1]} margin_x={mx*1000:.1f}mm margin_y={my*1000:.1f}mm "
          f"pudding near edge - plate rim (y, yaw180) = {gap*1000:.1f}mm  centre dist to plate {np.hypot(x-plate_ep0[0], y-plate_ep0[1])*1000:.1f}mm")
# adjacency placements
for g in (0.0, 0.01, 0.02):
    for label, (hx, hy) in PUD.items():
        yc = plate_ep0[1] + RP_VIS + g + hy
        print(f"adjacent gap {g*100:.0f}cm, {label}: centre y={yc:.4f} -> margin to y_max {1000*(RY[1]-yc):.1f}mm, inside={RY[0]<yc<RY[1]}")
json.dump(out, open("/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics/t6_band.json","w"), indent=1)
