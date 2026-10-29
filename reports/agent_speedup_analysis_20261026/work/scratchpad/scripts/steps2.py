import json, math, collections
R='/workspace/gpt_field/runs'
E='/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/evaluations'
names=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t7_001','t8_001','t9_001','t6_009','t6_011']
paths={n:R+'/agent_speedup_v2_user_check_libero_10_'+n for n in names}
paths['A_trial_002']=E+'/trial_002/trial'
def norm(v): return math.sqrt(sum(x*x for x in v))
V_TRANSIT=0.0051  # measured steady-state mm/step at transit cap (see ratio.py)
out={}
for n,d in paths.items():
    recs=[]
    with open(d+'/episode.events.jsonl') as f:
        for line in f:
            e=json.loads(line)
            if e.get('event')!='executed': continue
            fb=e.get('feedback') or {}
            fr=((fb.get('primitive_feedback') or {}).get('fragment')) or {}
            segs=fr.get('segments') or []
            cur=[s for s in segs if s.get('index')==fr.get('current_segment_index')]
            cur=cur[0] if cur else {}
            req=cur.get('request') or {}
            md=fb.get('measured_translation_delta_m') or e.get('measured_delta_xyz_m') or [0,0,0]
            recs.append(dict(call=e.get('api_request_log'), seg=fr.get('current_segment_index'), skill=cur.get('skill') or ('direct' if 'local_motion' in (e.get('execution_kind') or '') else None),
                             profile=req.get('profile'), d=norm(md), status=cur.get('status'), reason=cur.get('reason'), gripper=req.get('gripper')))
    # group by (call, seg) -> segment records; final status/reason from last step
    segs=collections.OrderedDict()
    for r in recs:
        k=(r['call'],r['seg'])
        segs.setdefault(k,[]).append(r)
    seg_info=[]
    for k,v in segs.items():
        last=v[-1]
        seg_info.append(dict(skill=v[0]['skill'],profile=v[0]['profile'],steps=len(v),path=sum(x['d'] for x in v),reason=last['reason'],gripper=v[0]['gripper']))
    mot=[s for s in seg_info if s['skill'] in ('move_to_pose','move_relative','approach')]
    grip=[s for s in seg_info if s['skill']=='set_gripper']
    stall=[s for s in mot if s['reason']=='segment_stall']
    budget=[s for s in seg_info if s['reason']=='segment_budget_exhausted']
    contact=[s for s in mot if s['profile']=='contact']; transit=[s for s in mot if s['profile']=='transit']
    # tiny-progress stretches
    runs=[]; cur=0
    for r in recs:
        if r['skill'] in ('move_to_pose','move_relative','approach') and r['d']<0.001: cur+=1
        else:
            if cur: runs.append(cur)
            cur=0
    if cur: runs.append(cur)
    long_runs=[x for x in runs if x>=10]
    total=len(recs)
    path_all=sum(r['d'] for r in recs if r['skill'] in ('move_to_pose','move_relative','approach','direct'))
    grip_steps=sum(s['steps'] for s in grip)
    n_close=sum(1 for s in grip if s['gripper']=='close'); n_open=sum(1 for s in grip if s['gripper']=='open')
    grip_min=12*n_close+8*n_open
    lb=path_all/V_TRANSIT+grip_min
    lb2=path_all/V_TRANSIT+grip_steps
    o=dict(steps=total, motion=sum(s['steps'] for s in mot), contact_steps=sum(s['steps'] for s in contact), transit_steps=sum(s['steps'] for s in transit),
           contact_path=sum(s['path'] for s in contact), transit_path=sum(s['path'] for s in transit), path=path_all,
           grip_steps=grip_steps, n_close=n_close, n_open=n_open, grip_min=grip_min,
           stall_segments=len(stall), stall_steps=sum(s['steps'] for s in stall), budget_segments=len(budget),
           tiny_runs_ge10=len(long_runs), tiny_steps_in_runs_ge10=sum(long_runs), max_tiny_run=max(runs) if runs else 0,
           lb_steps=lb, lb_steps_actualgrip=lb2, motion_segments=len(mot))
    out[n]=o
    print(f"{n:12s} steps={total:5d} motion={o['motion']:4d} contact={o['contact_steps']:4d}({100*o['contact_steps']/max(o['motion'],1):.0f}% of motion; path {o['contact_path']:.2f} m = {100*o['contact_path']/max(o['contact_path']+o['transit_path'],1e-9):.0f}%) transit={o['transit_steps']:4d} (path {o['transit_path']:.2f} m) "
          f"grip={grip_steps:3d} (close {n_close}, open {n_open}, min {grip_min}) stallSeg={o['stall_segments']} stallSteps={o['stall_steps']} budgetSeg={o['budget_segments']} tinyRuns>=10={o['tiny_runs_ge10']} ({o['tiny_steps_in_runs_ge10']} steps, max {o['max_tiny_run']}) "
          f"LB={lb:.0f} ({100*lb/total:.0f}% of actual)")
json.dump(out, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/steps2.json','w'), indent=1)
keys=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t7_001','t8_001','t9_001']
def s(k): return sum(out[n][k] for n in keys)
print('uc7 raw-success totals: steps',s('steps'),'motion',s('motion'),'contact',s('contact_steps'),'transit',s('transit_steps'),'grip',s('grip_steps'),'stall_steps',s('stall_steps'),'contact_path',round(s('contact_path'),2),'transit_path',round(s('transit_path'),2),'LB',round(s('lb_steps')),'tiny>=10 steps',s('tiny_steps_in_runs_ge10'))
print('contact share of motion steps %.1f%%, of path %.1f%%'%(100*s('contact_steps')/s('motion'),100*s('contact_path')/(s('contact_path')+s('transit_path'))))
print('LB/actual %.1f%%'%(100*s('lb_steps')/s('steps')))
