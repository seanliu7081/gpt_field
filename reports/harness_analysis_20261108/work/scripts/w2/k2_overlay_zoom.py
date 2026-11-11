import sys
import numpy as np, cv2
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from k2_scene_audit import replica
step = int(sys.argv[1]) if len(sys.argv) > 1 else 13
rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
prov = replica(CandidatePlanner(kin), scene, state, q)
cam = {x.name: x for x in scene.frame.cameras}
v = 'external'; m = prov['view'] == v
r, c, keep = prov['row'], prov['col'], prov['keep']
o = cam[v].rgb.copy().astype(float)
def paint(sel, color):
    o[r[sel], c[sel]] = 0.35*o[r[sel], c[sel]] + 0.65*np.array(color)
paint(m & ~keep, [255, 0, 0])                                   # self-removed
paint(m & keep & prov['in_sil'] & (prov['owners'] == ''), [0, 120, 255])   # kept inside robot silhouette (predicted foreground)
paint(m & keep & (prov['points'][:, 2] > .6) & ~prov['in_sil'], [255, 0, 255])  # kept floating > 0.6 m, outside silhouette
paint(m & keep & prov['strong'], [0, 255, 0])                   # strong
o = o.astype(np.uint8)
crop = lambda a: cv2.resize(a[0:220, 120:400], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
img = np.concatenate([crop(cam[v].rgb), crop(o)], 1)
cv2.imwrite(str(W2 / f'external_zoom_self_filter_step{step}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
full = np.concatenate([cam[v].rgb, o], 1)
cv2.imwrite(str(W2 / f'external_full_self_filter_step{step}.png'), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
print('ok')
