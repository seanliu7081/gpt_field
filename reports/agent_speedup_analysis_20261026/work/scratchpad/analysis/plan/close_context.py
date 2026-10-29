import json, sys
ROOT='/workspace/gpt_field/runs'
for r in sys.argv[1:]:
    d=f'{ROOT}/agent_speedup_v2_user_check_libero_10_{r}'
    h=json.load(open(f'{d}/user_check_plan_history.json'))
    print('=====',r)
    for i,row in enumerate(h['model_decisions']):
        a=row.get('action') or {}
        segs=(a.get('execution') or {}).get('segments') or []
        sk=[(s['skill'],s['gripper'],s['profile'],None if s['xyz_m'] is None else [round(x,3) for x in s['xyz_m']]) for s in segs]
        b=a.get('belief','')
        print(f"#{i+1} step={row.get('step')} type={a.get('type')} segs={sk}")
        print('   belief:',b[:420].replace('\n',' '))
