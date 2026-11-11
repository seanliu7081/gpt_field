import sys, json
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
import numpy as np
from PIL import Image
from v_common import *
exec(open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/v_can_raycast.py").read().split("sil = {}")[0])
sil = {}
for n, (K, E, _) in cams.items(): sil[n], _, _ = cast(K, E)
req = json.load(open(requests(run)[0])); ext, wr = decode_images(req)
out = []
for img, n, box in ((ext, "external", (30, 250, 150, 380)), (wr, "wrist", (390, 200, 512, 330))):
    im = img.copy().astype(float)
    m = masks[n]; s = sil[n]
    edge_m = m & ~np.roll(m, 1, 0) | m & ~np.roll(m, 1, 1)
    edge_s = s & ~np.roll(s, 1, 0) | s & ~np.roll(s, 1, 1)
    im[m] = im[m] * 0.6 + np.array([0, 120, 255]) * 0.4
    im[edge_s] = [255, 0, 0]
    x0, y0, x1, y1 = box
    crop = Image.fromarray(im[y0:y1, x0:x1].astype(np.uint8)).resize(((x1 - x0) * 4, (y1 - y0) * 4), Image.NEAREST)
    out.append(crop)
W = sum(c.width for c in out) + 20; H = max(c.height for c in out)
canvas = Image.new("RGB", (W, H), (255, 255, 255)); x = 0
for c in out: canvas.paste(c, (x, 0)); x += c.width + 20
canvas.save("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/t04_f0_can_masks_vs_truth.png")
print("saved")
