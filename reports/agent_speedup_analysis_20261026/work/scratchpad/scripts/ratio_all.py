import json, math, collections
R='/workspace/gpt_field/runs'
def norm(v): return math.sqrt(sum(x*x for x in v))
succ=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t7_001','t8_001','t9_001']
agg=collections.defaultdict(lambda:[0,0.0,0.0])
tail=0; motion=0
for n in succ:
    d=R+'/agent_speedup_v2_user_check_libero_10_'+n
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
            at_cap=cmd>=cap*0.98
            motion+=1
            if not at_cap: tail+=1
            k=(prof,at_cap, cur.get('actions',0)>=6)
            agg[k][0]+=1; agg[k][1]+=cmd; agg[k][2]+=md
for k,(c,cm,mm) in sorted(agg.items(), key=lambda x:str(x[0])):
    print(k,c,'cmd %.2f mm'%(1000*cm/c),'meas %.2f mm'%(1000*mm/c),'ratio %.3f'%(mm/cm if cm else 0))
print('motion steps',motion,'below-cap (proportional tail) steps',tail,'= %.1f%%'%(100*tail/motion))
