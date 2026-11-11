import json, sys, glob
from pathlib import Path
from collections import Counter

def analyze(run):
    fv = Path(run) / 'trial' / 'foundation_vision'
    summ = json.load(open(fv / 'summary.json'))
    texts = {t['key']: t['text'] for t in summ['targets']}
    print('RUN', run)
    print('targets:', texts)
    tot = Counter(); seg_t = []; dep_t = []; geo_t = []; tot_t = []
    agree = []
    for snap in sorted(fv.glob('frame_*_snapshot.json')):
        d = json.load(open(snap))
        t = d['timing_s']; seg_t.append(t['segmentation']); dep_t.append(t['depth']); geo_t.append(t['geometry']); tot_t.append(t['total'])
        g = d['geometry']
        views = g['views']
        sup = {k: round(v['crossview_supported_points'] / max(1, v['valid_depth_pixels']), 3) for k, v in views.items()}
        con = {k: round(v['contradicted_points'] / max(1, v['valid_depth_pixels']), 3) for k, v in views.items()}
        line = []
        for o in d['objects']:
            k = o['key']; og = g['object_geometry'].get(k, {})
            cc = o['candidate_counts']
            ev = og.get('crossview_mask_evidence', [])
            fr = [round(e['mask_agreement_fraction'], 2) if e['mask_agreement_fraction'] is not None else None for e in ev]
            if fr: agree.append((k, d['frame_id'], fr))
            line.append(f"{k}:{cc.get('external',0)}/{cc.get('wrist',0)}:{o['status'][:12]}:{og.get('association_status','-')[:10]}:{fr}")
        print(f"frame {d['frame_id']:4d} total={t['total']:.2f}s seg={t['segmentation']:.2f} depth={t['depth']:.2f} sup={sup} con={con} | " + ' '.join(line))
    import statistics as S
    print('timing sum total', round(sum(tot_t), 2), 'mean seg', round(S.mean(seg_t), 2), 'mean depth', round(S.mean(dep_t), 2), 'mean geo', round(S.mean(geo_t), 2))
    print()

for run in sys.argv[1:]:
    analyze(run)
