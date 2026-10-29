import json, sys, collections, statistics
runs=sys.argv[1:]
tail=collections.defaultdict(list)   # (skill,profile) -> steps spent with err<10mm before arrival
total=collections.defaultdict(list)
for r in runs:
    d=f'/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_{r}'
    cur=None; seq=[]
    with open(f'{d}/trace.jsonl','rb') as f:
        for line in f:
            if not line.startswith(b'{"event": "feedback"'): continue
            fb=json.loads(line)['feedback']; pf=fb.get('primitive_feedback') or {}
            if pf.get('primitive')!='execute_fragment': continue
            fr=pf['fragment']
            for s in fr['segments']:
                pass
            idx=fr['current_segment_index']; seg=fr['segments'][idx]
            key=(id_ := (fb['step']-fr['executed_steps']), idx)  # fragment start step + segment index
            err=seg.get('position_error_m')
            seq.append((key, seg['skill'], seg['request']['profile'], err, seg['status'], seg.get('reason')))
    # group by key
    groups=collections.OrderedDict()
    for key, skill, prof, err, status, reason in seq:
        groups.setdefault(key, []).append((skill,prof,err,status,reason))
    for key, rows in groups.items():
        skill,prof,_,_,_=rows[-1]
        final_reason=rows[-1][4]
        if final_reason!='pose_arrived': continue
        errs=[e for (_,_,e,_,_) in rows if e is not None]
        n_tail=sum(1 for e in errs if e<0.010)
        tail[(skill,prof)].append(n_tail); total[(skill,prof)].append(len(errs))
for k in tail:
    print(k, 'arrived segs', len(tail[k]), 'median steps total', statistics.median(total[k]), 'median steps within 10mm', statistics.median(tail[k]), 'sum within10mm', sum(tail[k]), 'sum total', sum(total[k]))
