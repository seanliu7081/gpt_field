import json, sys, glob, os, collections
run = sys.argv[1]
fv = os.path.join(run, 'trial/foundation_vision')
snaps = sorted(glob.glob(os.path.join(fv, 'frame_*_snapshot.json')))
summ = json.load(open(os.path.join(fv, 'summary.json')))
print('targets:', [(t['key'], t['text']) for t in summ['targets']])
cause = collections.Counter(); bykey = collections.defaultdict(collections.Counter)
timing = collections.defaultdict(list)
pairs = []
for s in snaps:
    d = json.load(open(s))
    cand = json.load(open(s.replace('_snapshot.json', '_sam_candidates.json')))
    fid = d['frame_id']
    for k in ('segmentation','depth','geometry','total'):
        timing[k].append(d['timing_s'][k])
    timing['age'].append(d['age_s'])
    views = d['geometry']['views']
    vq = {n: (v['valid_depth_pixels'], v['crossview_supported_points'], v['contradicted_points'], v['occluded_in_other_views_points']) for n, v in views.items()}
    line = [f'frame {fid:4d} status={d["status"]} age={d["age_s"]:.2f} seg={d["timing_s"]["segmentation"]:.2f} dep={d["timing_s"]["depth"]:.2f} geo={d["timing_s"]["geometry"]:.2f}']
    line.append('   depth quality ' + json.dumps({n: dict(valid=a, supp=round(b/a,4), contra=round(c/a,4), occl=round(e/a,4)) for n,(a,b,c,e) in vq.items()}))
    for o in d['objects']:
        g = o['geometry']; cc = o['candidate_counts']
        sc = {v: sorted([round(c['score'],3) for c in cand['candidates'] if c['target_key']==o['key'] and c['view']==v], reverse=True) for v in cc}
        ne, nw = cc.get('external',0), cc.get('wrist',0)
        if ne > 1 or nw > 1: c = 'multiple_candidate_ambiguity'
        elif ne == 0 and nw == 0: c = 'no_candidate_in_either_view'
        elif ne == 0 or nw == 0: c = 'candidate_missing_in_one_view'
        elif g.get('association_status') == 'rejected_mixed_instance': c = 'crossview_mask_association_rejected'
        elif g.get('status') != 'supported_surface_hypothesis': c = 'unique_both_views_but_unsupported'
        else: c = 'usable'
        cause[c] += 1; bykey[o['key']][c] += 1
        ev = [(p['source_view'][0], p['comparable_points'], p['mask_matches'], None if p['mask_agreement_fraction'] is None else round(p['mask_agreement_fraction'],3), p['occluded_points'], p['contradicted_points']) for p in g.get('crossview_mask_evidence', [])]
        if ev: pairs.append((fid, o['key'], ev, sc))
        line.append(f'   {o["key"]:8s} {o["status"]:20s} cand={cc} scores={sc} geo={g.get("status")}/{g.get("association_status")} hyp={g.get("surface_hypothesis_count")} supp={g.get("supported_point_count")} ev={ev}')
    print('\n'.join(line))
print('CAUSES', dict(cause))
for k, v in bykey.items(): print('  ', k, dict(v))
import statistics as st
for k, v in timing.items(): print('timing', k, 'n', len(v), 'min %.3f med %.3f mean %.3f max %.3f sum %.2f' % (min(v), st.median(v), st.mean(v), max(v), sum(v)))
