import json, glob, os, collections
R='/workspace/gpt_field/runs'
E='/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/evaluations'
M=json.load(open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/runs_metrics.json'))
names=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001','t6_007','t3_002']
paths={n:R+'/agent_speedup_v2_user_check_libero_10_'+n for n in names}
paths['today_A_trial_002']=E+'/trial_002/trial'; paths['today_B_trial_003']=E+'/trial_003/trial'
out={}
for n,d in paths.items():
    types=collections.Counter(); nseg=[]; skills=collections.Counter(); profiles=collections.Counter(); macros=collections.Counter(); obs_after=0; maxsteps=[]
    decisions=[]
    for f in sorted(glob.glob(d+'/requests/*.response.json')):
        r=json.load(open(f))
        txt=None
        for o in r.get('output',[]):
            if o.get('type')=='message':
                for c in o.get('content',[]):
                    if c.get('type')=='output_text': txt=c['text']
        if txt is None: types['<no_text>']+=1; continue
        try: dd=json.loads(txt)
        except Exception: types['<unparsable>']+=1; continue
        t=dd.get('type'); types[t]+=1
        ex=dd.get('execution') or {}
        segs=ex.get('segments') or []
        if t=='execute':
            nseg.append(len(segs))
            for s in segs:
                skills[s.get('skill')]+=1; profiles[(s.get('skill'),s.get('profile'))]+=1; maxsteps.append(s.get('max_steps'))
                if s.get('observation_after'): obs_after+=1
        if t=='macro':
            macros[json.dumps(dd.get('macro'))[:120]]+=1
        decisions.append(dict(file=os.path.basename(f)[:6], type=t, nseg=len(segs), latency=(r.get('completed_at') or 0)-(r.get('created_at') or 0), plan=(dd.get('plan') or '')[:200], belief=(dd.get('belief') or '')[:300], why=(dd.get('why') or '')[:200], segs=[(s.get('skill'),s.get('profile'),s.get('max_steps'),s.get('xyz_m'),s.get('gripper'),s.get('observation_after')) for s in segs], motion=dd.get('motion'), macro=dd.get('macro')))
    ac=M[n].get('agent_counts') or {}
    out[n]=dict(types=dict(types), mean_segments=(sum(nseg)/len(nseg) if nseg else None), skills=dict(skills), profiles={f'{a}|{b}':v for (a,b),v in profiles.items()}, macros=dict(macros), obs_after=obs_after,
                agent_counts={k:ac.get(k) for k in ['fragment_actions','direct_actions','native_macro_actions','fragment_grants','optional_macro_grants','direct_segments','hold_actions','local_retries','preemptions','agent_decisions','rejections']},
                control_steps=M[n].get('control_steps'), exec_kind=M[n].get('exec_kind_executed'), steps=M[n]['steps'], calls=M[n]['api_calls'], decisions=decisions)
    o=out[n]
    print(f"{n:17s} steps={o['steps']:5d} calls={o['calls']:3d} steps/call={o['steps']/o['calls']:6.1f} types={o['types']} meanseg={o['mean_segments']}")
    print('     counts',o['agent_counts'])
    print('     ctrl_steps',o['control_steps'],' exec_kind',o['exec_kind'])
    print('     skills',o['skills'],' macros',o['macros'], 'obs_after',o['obs_after'])
json.dump(out, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/mix.json','w'), indent=1, default=str)
