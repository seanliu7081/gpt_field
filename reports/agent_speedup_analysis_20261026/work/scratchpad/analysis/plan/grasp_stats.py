import json, sys, re, os, glob
ROOT='/workspace/gpt_field/runs'
RUNS=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001']
KW=['slip','dropp','fell','fallen','tipp','toppl','missed','empty','not hold','no object','lost','regrasp','re-grasp','retry','again','elbow','joint limit','reach','stretch','table','too low','too high','collid','stuck','blocked']
out={}
for r in RUNS:
    d=f'{ROOT}/agent_speedup_v2_user_check_libero_10_{r}'
    h=json.load(open(f'{d}/user_check_plan_history.json'))
    dec=h['model_decisions']
    closes=opens=0; close_dec=0; kwc={k:0 for k in KW}
    types={}
    for row in dec:
        a=row.get('action') or {}
        t=a.get('type'); types[t]=types.get(t,0)+1
        ex=a.get('execution') or {}
        segs=ex.get('segments') or []
        c=[s for s in segs if s.get('skill')=='set_gripper' and s.get('gripper')=='close']
        o=[s for s in segs if s.get('skill')=='set_gripper' and s.get('gripper')=='open']
        mo=a.get('motion') or {}
        if mo and mo.get('gripper')=='close': c.append(mo)
        if mo and mo.get('gripper')=='open': o.append(mo)
        closes+=len(c); opens+=len(o); close_dec+=bool(c)
        text=(a.get('belief','')+' '+a.get('plan','')+' '+a.get('why','')).lower()
        for k in KW:
            if k in text: kwc[k]+=1
    s=json.load(open(f'{d}/summary.json'))
    out[r]={'raw':s.get('task_success'),'calls':len(dec),'types':types,'close_cmds':closes,'open_cmds':opens,'kw':{k:v for k,v in kwc.items() if v}}
for r,v in out.items():
    print(r, v['raw'], 'calls',v['calls'],'close',v['close_cmds'],'open',v['open_cmds'], v['kw'])
json.dump(out,open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/grasp_stats.json','w'),indent=1)
