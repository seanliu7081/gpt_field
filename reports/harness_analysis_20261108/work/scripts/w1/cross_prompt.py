import json, glob, sys, collections
import numpy as np
for run in sys.argv[1:]:
    fv = run + '/trial/foundation_vision/'
    dup = collections.Counter(); total = 0; frames = 0; masks_per_view = collections.Counter()
    for cj in sorted(glob.glob(fv + 'frame_*_sam_candidates.json')):
        c = json.load(open(cj)); mk = np.load(cj.replace('.json', '.npz')); frames += 1
        cands = c['candidates']; total += len(cands)
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if a['view'] != b['view'] or a['target_key'] == b['target_key']: continue
                ma, mb = mk[a['array_key']], mk[b['array_key']]
                iou = (ma & mb).sum() / max(1, (ma | mb).sum())
                if iou > 0.8: dup[(a['target_key'], b['target_key'], a['view'])] += 1
        # distinct instances per view (union by IoU>0.8)
        for view in ('external', 'wrist'):
            vm = [mk[x['array_key']] for x in cands if x['view'] == view]
            uniq = []
            for m in vm:
                if not any((m & u).sum() / max(1, (m | u).sum()) > 0.8 for u in uniq): uniq.append(m)
            masks_per_view[view] += len(uniq)
    print(run.split('/')[-1][:40], 'frames', frames, 'candidates', total, 'distinct instances (ext, wrist) summed over frames', dict(masks_per_view))
    print('   cross-prompt duplicate masks (IoU>0.8):', dict(dup))
