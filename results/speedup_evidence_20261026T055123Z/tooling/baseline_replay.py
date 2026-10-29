"""Real frozen v7 policy/client replay with saved RGB/state and complete responses.

No model, simulator, truth, renderer or success scoring. Missing per-step joints
and calibration are recovered only at the 13 public model-request boundaries;
other steps use the stored four-field observation and no invented sensors.
"""
import argparse,copy,hashlib,importlib.util,json,pathlib,sys
import numpy as np
HERE=pathlib.Path(__file__).resolve().parents[1];BASE=HERE/'baseline'
sys.path.insert(0,str(BASE/'src'))
from gpt_field.agent_speedup_v2_user_check_client import PlanLock
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client
from gpt_field.agent_speedup_v2_user_check_agent import AgentFirstV2Policy
from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.observation import PolicyObservation
from gpt_field.agent_speedup_v3_performance import LiveSnapshotGate
spec=importlib.util.spec_from_file_location('baseline_replay_runner',BASE/'scripts/evaluate_agent_speedup_v2_user_check.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
class ReplayTransport:
    offline_mock=True
    def __init__(self,requests):self.requests=requests;self.payloads=[];self.historical_equal=[]
    def __call__(self,endpoint,payload,headers,timeout):
        index=len(self.payloads);record,response=self.requests[index]
        self.payloads.append(sha(payload));self.historical_equal.append(sha(payload)==sha(record['payload']))
        return copy.deepcopy(response)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args()
    requests=[];boundaries={}
    for p in sorted((args.archive/'requests').glob('*.request.json')):
        rec=json.loads(p.read_text());response=json.loads(p.with_name(p.name.replace('.request.json','.response.json')).read_text())
        requests.append((rec,response));fields=None
        for item in rec['payload']['input']:
            for content in item.get('content',[]):
                if content.get('type')=='input_text':
                    try:value=json.loads(content['text'])
                    except (json.JSONDecodeError,TypeError):continue
                    if isinstance(value,dict) and value.get('request')=='agent':fields=value
        if fields is None:raise RuntimeError('saved public request fields missing')
        boundaries[fields['step']]=fields
    plan=json.loads((args.archive/'approved_plan.json').read_text());prompt=requests[0][0]['request_meta'].get('task_prompt') or next(iter(boundaries.values()))['task']
    policies=[];transports=[];now=[0.]
    for arm in ['A','B']:
        lock=PlanLock(plan,task_prompt=prompt);transport=ReplayTransport(requests)
        client=AgentFirstV2Client(runner.base.configuration(),approved_plan=plan,lock=lock,task_prompt=prompt,transport=transport,token_provider=lambda:'offline-fixture',clock=lambda:0.)
        policy=AgentFirstV2Policy(client,lock=lock,clock=lambda:0.);policy.reset();policies.append(policy);transports.append(transport)
    gate=LiveSnapshotGate(clock=lambda:now[0]);count=0;historic_matches=0;max_error=0.;mismatches=[];feedback_matches=0;event_matches=0
    def observation(data,step):
        native=PolicyObservation.from_dict(data)
        if step not in boundaries:return native
        fields=boundaries[step];state=fields['context']['measured_state']
        return CalibratedObservation(image=native.image,wrist_image=native.wrist_image,state=native.state,prompt=native.prompt,geometry=fields['geometry'],joint_position=state['joint_position_rad'],joint_velocity=state['joint_velocity_rad_s'])
    obs=None;decisions=None
    try:
        for line in (args.archive/'trace.jsonl').open():
            row=json.loads(line);event=row['event']
            if event=='reset':obs=observation(row['observation'],0)
            elif event=='action':
                decisions=[p.act(obs) for p in policies]
                if not np.array_equal(decisions[0].action,decisions[1].action):raise AssertionError(f'AB action differs at {count}')
                event_matches+=int(decisions[0].diagnostics==decisions[1].diagnostics)
                error=float(np.max(np.abs(np.asarray(decisions[0].action)-row['action'])));max_error=max(max_error,error)
                historic_matches+=int(np.allclose(decisions[0].action,row['action'],atol=1e-12,rtol=1e-11))
                if error>1e-11 and len(mismatches)<5:mismatches.append({'step':count,'max_abs_error':error})
            elif event=='executed':
                after=observation(row['observation'],row['step']+1)
                feedback=[p.observe_transition(obs,d,after) for p,d in zip(policies,decisions)]
                if feedback[0]!=feedback[1]:raise AssertionError(f'AB feedback differs at {count}')
                feedback_matches+=1;obs=after;count+=1;now[0]=count/20
                # Exercise the actual unchanged policy's expensive vs compact
                # display stats path; neither may change subsequent decisions.
                policies[0].agent_stats()
                if gate.due('observed'):gate.policy_stats(policies[1])
                if count%100==0:print(json.dumps({'replayed_actions':count}),flush=True)
        report={'status':'passed','kind':'saved_public_observation_and_frozen_response_diagnostic','actions':count,'AB_actions_exact':count,'AB_feedback_exact':feedback_matches,'AB_decision_diagnostics_exact':event_matches,'historical_action_matches':historic_matches,'max_historical_action_abs_error':max_error,'historical_mismatches':mismatches,'AB_request_payload_hashes_equal':transports[0].payloads==transports[1].payloads,'requests_per_arm':len(transports[0].payloads),'historical_payload_equal_count':sum(transports[0].historical_equal),'payload_hashes':transports[0].payloads,'historical_payload_equal_by_request':transports[0].historical_equal,'floating_tolerance':{'absolute':1e-12,'relative':1e-11,'source':'existing recorded-policy replay numerical convention'},'sensor_provenance':'RGB/state all897 stored; calibrated geometry and measured joints at public request boundaries only; missing intermediate channels omitted, never invented','limitations':['Saved responses are not independent model decisions or task-success trials.','No new simulator state or physical outcomes are generated.','Per-step camera/joint serialization missing outside model boundaries; no claim to recover those missing channels.'],'new_model_requests':0,'new_simulator_rollouts':0}
    except BaseException as exc:
        report={'status':'failed','actions':count,'error':repr(exc),'historical_action_matches':historic_matches,'AB_payloads_equal':transports[0].payloads==transports[1].payloads,'new_model_requests':0,'new_simulator_rollouts':0}
        args.output.write_text(json.dumps(report,indent=2)+'\n');raise
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['payload_hashes']}))
if __name__=='__main__':main()
