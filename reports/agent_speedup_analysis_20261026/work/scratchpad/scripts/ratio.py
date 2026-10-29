import json, math, collections, sys
R='/workspace/gpt_field/runs'
def norm(v): return math.sqrt(sum(x*x for x in v))
for n in ['t1_001','t5_001','t8_001','t6_009']:
    d=R+'/agent_speedup_v2_user_check_libero_10_'+n
    bins=collections.defaultdict(list)
    prev=None
    with open(d+'/episode.events.jsonl') as f:
        for line in f:
            e=json.loads(line)
            if e.get('event')!='executed': continue
            fb=e.get('feedback') or {}
            fr=((fb.get('primitive_feedback') or {}).get('fragment')) or {}
            segs=fr.get('segments') or []
            cur=[s for s in segs if s.get('index')==fr.get('current_segment_index')]
            if not cur: continue
            cur=cur[0]; prof=(cur.get('request') or {}).get('profile'); sk=cur.get('skill')
            if sk not in ('move_to_pose','move_relative','approach'): continue
            act=fb.get('normalized_action') or e.get('action')
            cmd=norm([a*0.05 for a in act[:3]])
            md=norm(fb.get('measured_translation_delta_m') or [0,0,0])
            cap=0.02 if prof=='transit' else 0.005
            at_cap = cmd>=cap*0.98
            # consecutive steps in same segment at cap (steady state): use actions count>=5
            ss = cur.get('actions',0)>=6
            bins[(prof,at_cap,ss)].append((cmd,md))
    for k,v in sorted(bins.items(), key=lambda x:str(x[0])):
        cm=sum(a for a,b in v)/len(v); mm=sum(b for a,b in v)/len(v)
        print(n,k,len(v),'cmd_mean_mm=%.2f'%(cm*1000),'meas_mean_mm=%.2f'%(mm*1000),'ratio=%.3f'%(mm/cm if cm else 0))
