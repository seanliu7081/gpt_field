import json, os, sys, glob, collections, math
R='/workspace/gpt_field/runs'
E='/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/evaluations'
OUT='/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/steps_metrics.json'
names=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001','t6_007','t3_002']
paths={n:R+'/agent_speedup_v2_user_check_libero_10_'+n for n in names}
paths['A_trial_002']=E+'/trial_002/trial'
paths['B_trial_003']=E+'/trial_003/trial'
def norm(v): return math.sqrt(sum(x*x for x in v))
res={}
for n,d in paths.items():
    steps=[]
    frag_groups=collections.OrderedDict()
    with open(d+'/episode.events.jsonl') as f:
        for line in f:
            e=json.loads(line)
            if e.get('event')!='executed': continue
            fb=e.get('feedback') or {}
            pf=(fb.get('primitive_feedback') or {})
            fr=pf.get('fragment') if isinstance(pf,dict) else None
            md=e.get('measured_delta_xyz_m') or fb.get('measured_translation_delta_m')
            st=e.get('state') or []
            rec={'step':e.get('step'),'kind':e.get('execution_kind'),'delta':norm(md) if md else None,
                 'gap':fb.get('finger_gap_mm'),'grip_cmd':fb.get('gripper_command'),'xyz':st[:3] if st else None,
                 'call':e.get('api_request_log'),'wall':e.get('wall_elapsed_s')}
            if fr:
                seg_i=fr.get('current_segment_index'); segs=fr.get('segments') or []
                cur=[s for s in segs if s.get('index')==seg_i]
                cur=cur[0] if cur else (segs[-1] if segs else {})
                req=cur.get('request') or {}
                rec.update(skill=fr.get('current_skill'), profile=req.get('profile'), pos_err=fr.get('position_error_m'),
                           ori_err=fr.get('orientation_error_rad'), seg_index=seg_i, frag_status=fr.get('status'), frag_reason=fr.get('reason'))
                frag_groups[rec['call']]=fr
            else:
                rec.update(skill=None, profile=None)
            steps.append(rec)
    # summarize
    c_kind=collections.Counter(s['kind'] for s in steps)
    c_skill=collections.Counter((s.get('skill'),s.get('profile')) for s in steps)
    # motion steps
    mot=[s for s in steps if s.get('skill') in ('move_to_pose','move_relative','approach')]
    tiny=[s for s in mot if s['delta'] is not None and s['delta']<0.001]
    tail=[s for s in mot if s.get('pos_err') is not None and s['pos_err']<0.01]
    trans=[s for s in mot if s.get('profile')=='transit']
    cont=[s for s in mot if s.get('profile')=='contact']
    def pct(xs,q):
        xs=sorted(xs); 
        if not xs: return None
        k=(len(xs)-1)*q; f=int(k); c=min(f+1,len(xs)-1); return xs[f]+(xs[c]-xs[f])*(k-f)
    td=[s['delta'] for s in trans if s['delta'] is not None]
    cd=[s['delta'] for s in cont if s['delta'] is not None]
    path=sum(s['delta'] for s in steps if s['delta'] is not None)
    path_mot=sum(s['delta'] for s in mot if s['delta'] is not None)
    path_trans=sum(td); path_cont=sum(cd)
    # fragment end reasons and segment reasons
    reasons=collections.Counter(); seg_reasons=collections.Counter(); seg_actions=collections.defaultdict(int); n_segs_req=0; n_segs_exec=0
    per_call=[]
    for call,fr in frag_groups.items():
        reasons[fr.get('reason')]+=1
        segs=fr.get('segments') or []
        for s in segs:
            seg_reasons[(s.get('skill'),(s.get('request') or {}).get('profile'),s.get('reason'))]+=1
        per_call.append({'call':os.path.basename(call or '')[:6],'reason':fr.get('reason'),'executed':fr.get('executed_steps'),'budget':fr.get('budget_steps'),
                         'segs':[(s.get('skill'),(s.get('request') or {}).get('profile'),s.get('actions'),(s.get('request') or {}).get('max_steps'),s.get('reason')) for s in segs],
                         'pending':len(fr.get('pending_segments') or [])})
    # gripper set steps
    grip_steps=[s for s in steps if s.get('skill')=='set_gripper']
    wait_steps=[s for s in steps if s.get('skill')=='wait']
    # direct local motion
    direct=[s for s in steps if s['kind'] and 'local_motion' in s['kind']]
    res[n]=dict(n_steps=len(steps), kinds=dict(c_kind), skill_profile={f'{k[0]}|{k[1]}':v for k,v in c_skill.items()},
        motion_steps=len(mot), transit_steps=len(trans), contact_steps=len(cont), grip_steps=len(grip_steps), wait_steps=len(wait_steps), direct_steps=len(direct),
        tiny_lt1mm=len(tiny), tail_err_lt1cm=len(tail),
        transit_delta_p50=pct(td,.5), transit_delta_p90=pct(td,.9), transit_delta_max=max(td) if td else None,
        contact_delta_p50=pct(cd,.5), contact_delta_p90=pct(cd,.9),
        path_m=path, path_motion_m=path_mot, path_transit_m=path_trans, path_contact_m=path_cont,
        frag_reasons=dict(reasons), seg_reasons={f'{k[0]}|{k[1]}|{k[2]}':v for k,v in seg_reasons.items()}, per_call=per_call)
json.dump(res, open(OUT,'w'), indent=1)
for n,r in res.items():
    print(f"{n:12s} steps={r['n_steps']:5d} motion={r['motion_steps']:4d} transit={r['transit_steps']:4d} contact={r['contact_steps']:4d} grip={r['grip_steps']:3d} wait={r['wait_steps']:3d} direct={r['direct_steps']:3d} tiny<1mm={r['tiny_lt1mm']:4d} tail<1cm={r['tail_err_lt1cm']:4d} "
          f"tr_d50={(r['transit_delta_p50'] or 0)*1000:.1f}mm tr_d90={(r['transit_delta_p90'] or 0)*1000:.1f} tr_max={(r['transit_delta_max'] or 0)*1000:.1f} ct_d50={(r['contact_delta_p50'] or 0)*1000:.1f} path={r['path_m']:.2f} tr_path={r['path_transit_m']:.2f} ct_path={r['path_contact_m']:.2f}")
    print('     frag_reasons', r['frag_reasons'])
