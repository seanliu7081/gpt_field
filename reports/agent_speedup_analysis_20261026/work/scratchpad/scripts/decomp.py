import json
M=json.load(open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/runs_metrics.json'))
uc7s=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t7_001','t8_001','t9_001']
rows=[]
def row(n):
    m=M[n]; tc=m['tc']
    model=tc['model_pipeline_s']; pol=tc['policy_compute_s']; obs=tc['observe_transition_s']; sim=tc['simulator_step_s']; vid=tc['video_render_s']; rs=tc['run_state_s']
    ctrl=m['control_elapsed_s']; el=m['elapsed_s']; prep=m['prep_s']; a2c=m.get('approval_to_claim_s') or 0
    resid=ctrl-(model+pol+obs+sim+vid+rs)
    outside=el-ctrl
    service=prep+a2c+el
    return dict(n=n,steps=m['steps'],calls=m['api_calls'],prep=prep,model=model,pol=pol,obs=obs,sim=sim,vid=vid,rs=rs,resid=resid,outside=outside,ctrl=ctrl,el=el,service=service,
                prep_model=m['prep_model_s'], startup=m['startup_s'], cl2n=m.get('claim_to_normalized_s'))
print('run      steps calls  prep  model   sim   vid    rs   obs   pol resid  outs  ctrl   elap  serv  serv-300  noModel noIO  noIO_modelBudget')
acc={}
for n in uc7s+['t6_009','t6_011','today_A_trial_002','today_B_trial_003','t6_007','t3_002','t2_001']:
    r=row(n)
    noModel=r['service']-r['model']
    noIO=r['service']-r['vid']-r['rs']
    budget_model=300-(noIO-r['model'])
    print(f"{n:17s} {r['steps']:5d} {r['calls']:4d} {r['prep']:5.1f} {r['model']:6.1f} {r['sim']:5.1f} {r['vid']:5.1f} {r['rs']:5.1f} {r['obs']:5.1f} {r['pol']:5.1f} {r['resid']:5.1f} {r['outside']:5.1f} {r['ctrl']:6.1f} {r['el']:6.1f} {r['service']:6.1f} {r['service']-300:7.1f} {noModel:7.1f} {noIO:6.1f} {budget_model:7.1f}")
    if n in uc7s:
        for k,v in r.items():
            if k!='n': acc.setdefault(k,[]).append(v)
print()
import statistics as st
print('MEAN over',len(uc7s),'uc7 raw successes:')
for k,v in acc.items():
    print(f"  {k:10s} mean={sum(v)/len(v):8.2f} median={st.median(v):8.2f} min={min(v):8.2f} max={max(v):8.2f}")
ms={k:sum(v)/len(v) for k,v in acc.items()}
print('share of service: ', {k: round(100*ms[k]/ms['service'],1) for k in ['prep','model','sim','vid','rs','obs','pol','resid','outside']})
print('mean service - model =', ms['service']-ms['model'])
print('mean service - vid - rs =', ms['service']-ms['vid']-ms['rs'])
print('mean service - vid - rs - model =', ms['service']-ms['vid']-ms['rs']-ms['model'])
per_step={k: ms[k]/ms['steps'] for k in ['sim','vid','rs','obs']}
print('per-step', per_step, 'sum', sum(per_step.values()))
print('per-call model', ms['model']/ms['calls'])
