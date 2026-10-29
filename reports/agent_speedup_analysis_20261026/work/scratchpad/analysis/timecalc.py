import json, datetime as dt
OFF = -646548.212  # UTC seconds-of-day = monotonic + OFF (calibrated on trial_001/trial_004 evidence mtimes)
def hms(sod):
    h=int(sod//3600); m=int((sod%3600)//60); s=sod%60
    return f"{h:02d}:{m:02d}:{s:05.2f}"
def utc(mono): return hms(mono+OFF)
def sod(s):  # 'HH:MM:SS.ss'
    h,m,x=s.split(':'); return int(h)*3600+int(m)*60+float(x)
R1='/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/evaluations'
print('== Campaign 1')
tot_req=0
for t in ['trial_001','trial_002','trial_003','trial_004','trial_005']:
    d=json.load(open(f'{R1}/{t}/service_clock.json'))
    ev=d['events']
    st=[e for e in ev if e['event']=='model_request_start']; en=[e for e in ev if e['event']=='model_request_end']
    req=sum(b['monotonic_s']-a['monotonic_s'] for a,b in zip(st,en))
    tot_req+=req
    marks={e['event']:e['monotonic_s'] for e in ev if e['event'] in ('reviewed_execution_start','policy_and_evaluator_finished','approval_complete','automatic_review_start','human_review_wait_start','human_review_wait_end','preparation_complete')}
    print(t,'start',utc(d['started_monotonic']),'end',utc(d['finished_monotonic']),'service',round(d['service_elapsed_s'],1),'human_wait',round(d['human_wait_s'],1),'n_req',len(st),'req_s',round(req,1))
    for k,v in marks.items(): print('    ',k,utc(v))
print('campaign1 total model-request seconds (sum, overlapping arms counted separately)',round(tot_req,1))
c1s=sod('05:51:23.70'); c1e=sod('07:48:50.80')
print('campaign1 wall',round(c1e-c1s,1),'s =',round((c1e-c1s)/3600,3),'h')
# live robot execution window (overlapping pair)
d2=json.load(open(f'{R1}/trial_002/service_clock.json'));d3=json.load(open(f'{R1}/trial_003/service_clock.json'))
x2=[e for e in d2['events'] if e['event']=='reviewed_execution_start'][0]['monotonic_s']; f2=d2['finished_monotonic']
x3=[e for e in d3['events'] if e['event']=='reviewed_execution_start'][0]['monotonic_s']; f3=d3['finished_monotonic']
lo=min(x2,x3); hi=max(f2,f3)
print('live exec window',utc(lo),'->',utc(hi),round(hi-lo,1),'s; per-arm exec',round(f2-x2,1),round(f3-x3,1))

def intervals(path):
    d=json.load(open(path)); ev=d['events']
    st=[e['monotonic_s'] for e in ev if e['event']=='model_request_start']; en=[e['monotonic_s'] for e in ev if e['event']=='model_request_end']
    return list(zip(st,en))
def union(iv):
    iv=sorted(iv); tot=0; cur=None
    for a,b in iv:
        if cur is None: cur=[a,b]
        elif a<=cur[1]: cur[1]=max(cur[1],b)
        else: tot+=cur[1]-cur[0]; cur=[a,b]
    if cur: tot+=cur[1]-cur[0]
    return tot
iv=[]
for t in ['trial_002','trial_003','trial_004','trial_005']:
    iv+=intervals(f'{R1}/{t}/service_clock.json')
print('campaign1 model-request wall-clock union',round(union(iv),1))
iv23=intervals(f'{R1}/trial_002/service_clock.json')+intervals(f'{R1}/trial_003/service_clock.json')
print(' of which trial_002/003 union',round(union(iv23),1))
# exec-window only requests union
x_iv=[(a,b) for a,b in iv23 if a>=lo]
print(' exec-window request union',round(union(x_iv),1),'of exec window',round(hi-lo,1))
print('== Campaign 3')
R3='/workspace/gpt_field/runs/v3_completion_baseline_20261026T133633Z/attempts'
for a in ['attempt_001','attempt_002','attempt_003']:
    d=json.load(open(f'{R3}/{a}/service_clock.json'))
    cs=d.get('control_started_monotonic'); cf=d.get('control_finished_monotonic')
    print(a,'start',utc(d['started_monotonic']),'end',utc(d['finished_monotonic']),'service',round(d['service_elapsed_s'],1),'control', (round(cf-cs,1) if cs else None), 'control', (utc(cs) if cs else None),'->',(utc(cf) if cf else None))
c3s=sod('13:36:33.0'); c3e=sod('14:51:30.0')
print('campaign3 wall',round(c3e-c3s,1))
c2s=sod('10:14:38.15'); c2e=sod('11:03:05.51')
print('campaign2 wall',round(c2e-c2s,1))
