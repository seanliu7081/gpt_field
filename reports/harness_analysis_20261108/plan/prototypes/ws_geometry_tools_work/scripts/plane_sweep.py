"""Offline DA3-free support-height estimate by a horizontal plane sweep (no sim, no model, no GPU).

For one saved executive request (both upright RGB images + synchronized calibration), every
candidate height h maps wrist pixels -> world plane z=h -> external pixels. Score = median of
block-wise normalized cross-correlation between wrist intensities and remapped external
intensities. The argmax is the support height; the peak half-width gives an uncertainty.
No masks are used (robot/objects are outliers absorbed by the block median), so this is a
lower bound on what a masked implementation would achieve.

usage: plane_sweep.py RUN_DIR REQUEST_ORDINAL TRUTH_TABLE_Z
"""
import base64, io, json, math, sys, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def load(run, ordinal):
    reqs = sorted((Path(run) / 'trial' / 'requests').glob('*.request.json'))
    d = json.load(open(reqs[ordinal - 1]))
    fields, images = None, []
    for c in d['payload']['input'][-1]['content']:
        if c.get('type') == 'input_text' and c['text'].startswith('{'):
            fields = json.loads(c['text'])
        elif c.get('type') == 'input_image':
            raw = base64.b64decode(c['image_url'].split(',', 1)[1])
            images.append(np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')))
    return fields, images  # images: [external, wrist]


def rays(cam, uu, vv):
    R = np.asarray(cam['rotation_world_from_camera'], float)
    fx, fy = cam['focal_length_px']; cx, cy = cam['principal_point_px']
    d_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu)], -1)
    d = d_cam @ R.T
    return np.asarray(cam['position_world_m'], float), d


def project(cam, X):
    R = np.asarray(cam['rotation_world_from_camera'], float)
    local = (X - np.asarray(cam['position_world_m'], float)) @ R
    depth = -local[..., 2]
    fx, fy = cam['focal_length_px']; cx, cy = cam['principal_point_px']
    with np.errstate(divide='ignore', invalid='ignore'):
        u = cx + fx * local[..., 0] / depth
        v = cy - fy * local[..., 1] / depth
    return u, v, depth


def block_ncc(a, b, valid, block=16):
    h, w = a.shape
    scores = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            m = valid[y:y + block, x:x + block]
            if m.mean() < 0.9:
                continue
            pa = a[y:y + block, x:x + block][m]; pb = b[y:y + block, x:x + block][m]
            sa, sb = pa.std(), pb.std()
            if sa < 2.0 or sb < 2.0:   # textureless block: no evidence either way
                continue
            scores.append(float(((pa - pa.mean()) * (pb - pb.mean())).mean() / (sa * sb)))
    return (float(np.median(scores)), len(scores)) if scores else (float('nan'), 0)


def sweep(fields, ext_rgb, wri_rgb, heights):
    cams = fields['geometry']['cameras']
    ext = cv2.cvtColor(ext_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    wri = cv2.cvtColor(wri_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = wri.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    o, d = rays(cams['wrist'], uu, vv)
    # Fingertips occupy the wrist image's lower edge region; exclude the bottom 20 % (robot self-view).
    keep = vv < 0.8 * H
    out = []
    for h in heights:
        s = (h - o[2]) / d[..., 2]
        X = o + s[..., None] * d
        u, v, depth = project(cams['external'], X)
        valid = keep & (s > 0) & (depth > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        mapped = cv2.remap(ext, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        score, blocks = block_ncc(wri, mapped, valid)
        out.append((h, score, blocks))
    return out


def main():
    run, ordinal, truth = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    fields, (ext_rgb, wri_rgb) = load(run, ordinal)
    wz = fields['geometry']['cameras']['wrist']['position_world_m'][2]
    t0 = time.perf_counter()
    coarse = sweep(fields, ext_rgb, wri_rgb, np.arange(wz - 0.80, wz - 0.05, 0.01))
    best = max((r for r in coarse if r[2] > 0 and math.isfinite(r[1])), key=lambda r: r[1])
    fine = sweep(fields, ext_rgb, wri_rgb, np.arange(best[0] - 0.012, best[0] + 0.012, 0.001))
    t1 = time.perf_counter()
    fb = max((r for r in fine if r[2] > 0 and math.isfinite(r[1])), key=lambda r: r[1])
    # Half-width of the fine peak at (peak - 0.1) as a crude 1-sigma-like spread.
    above = [r[0] for r in fine if math.isfinite(r[1]) and r[1] >= fb[1] - 0.1]
    spread = (max(above) - min(above)) / 2 if above else float('nan')
    second = sorted((r for r in coarse if abs(r[0] - best[0]) > 0.05 and math.isfinite(r[1])),
                    key=lambda r: -r[1])[:1]
    print(f"{Path(run).name} req{ordinal} wrist_z={wz:.3f} truth={truth:.4f} est={fb[0]:.4f} "
          f"err={1000 * (fb[0] - truth):+.1f} mm peak_ncc={fb[1]:.3f} blocks={fb[2]} half_width~{1000 * spread:.1f} mm "
          f"next_peak_ncc={second[0][1] if second else float('nan'):.3f} time={t1 - t0:.2f}s")


if __name__ == '__main__':
    main()
