"""Saved-artifact-only audit. Never imports policy code, calls models, or starts a simulator."""
from pathlib import Path
import hashlib,json,math,subprocess,sys
from collections import Counter
from datetime import datetime,timezone

ROOT=Path('/workspace/gpt_field')
PREP=ROOT/'output/r1_two_task_20261111_001'
TAG=sys.argv[1]
SESSION=json.loads((PREP/'session.json').read_text())
SPEC=next(x for x in SESSION['runs'] if x['tag']==TAG)
RUN=Path(SPEC['output'])

def read(rel):return json.loads((RUN/rel).read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-10)
checks={}
def check(name,passed,detail=None):
 checks[name]={'passed':bool(passed)}
 if detail is not None:checks[name]['detail']=detail

result=read('result.json'); summary=read('trial/summary.json'); config=read('configuration.json')
archive=ROOT/'runs/foundation_vision_harness_V3_t04_can_obstacle_001'
orig=json.loads((archive/'source_manifest.json').read_text())
manifest=read('source_manifest.json')
sources={k:v for k,v in manifest.items() if not k.startswith('@')}
orig_sources={k:v for k,v in orig.items() if not k.startswith('@')}
check('source_map_equals_original_r1',sources==orig_sources,{'source_count':len(sources)})
for tag,base in [('run_frozen_sources',RUN/'source_at_launch'),('runtime_sources',Path(SESSION['runtime']))]:
 mismatch=[k for k,v in sources.items() if not (base/k).is_file() or digest(base/k)!=v]
 check(tag,not mismatch,{'mismatches':mismatch})
check('terminal_source_audit',read('terminal_source_audit.json')=={'changed_sources':[],'source_count':370,'unchanged':True})
inputs={k:v for k,v in manifest.items() if k.startswith('@input:')}
input_errors=[]
for key,value in inputs.items():
 p=Path(key[len('@input:'):]);
 if not p.is_file() or digest(p)!=value:input_errors.append(str(p))
check('frozen_input_paths_match',not input_errors,{'mismatches':input_errors})
for p,want in SPEC['input_sha256'].items():
 check('input_'+Path(p).name,digest(Path(p))==want)
obs=read('trial/initial_observation.json');initial=read('trial/initial_observation_check.json')
sha=hashlib.sha256(json.dumps(obs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
check('initial_observation_sha',sha==initial['sha256']==initial['expected_sha256']==config['expected_initial_observation_sha256'] and initial['matched'],{'sha256':sha})
rows=[json.loads(x) for x in (RUN/'trial/trace.jsonl').read_text().splitlines()]
actions=[r for r in rows if r['event']=='action'];holds=[r for r in rows if r['event']=='evaluator_same_gripper_hold'];terminal=[r for r in rows if r['event']=='policy_terminal']
check('policy_action_step_sequence',[r['step'] for r in actions]==list(range(1,result['steps']+1)),{'actions':len(actions)})
chain=[]
prev=obs['observation/state']
for r in actions:
 if r['before_state']!=prev:chain.append(r['step'])
 prev=r['after_state']
check('complete_state_chain',not chain,{'mismatch_steps':chain})
check('initial_and_terminal_state_match',actions[0]['before_state']==obs['observation/state'] and actions[-1]['after_state']==read('trial/terminal_observation.json')['observation/state'])
check('finite_valid_actions_and_states',all(len(r['action'])==7 and all(math.isfinite(v) and abs(v)<=1.0000000001 for v in r['action']) and len(r['before_state'])==len(r['after_state'])==8 and all(math.isfinite(v) for v in r['before_state']+r['after_state']) for r in actions))
check('all_feedback_complete',all(r['feedback_error'] is None for r in actions))
check('decision_step_matches',all(r['decision']['step']==r['step']-1 for r in actions))
check('autonomous_terminal',len(terminal)==1 and terminal[0]['claim']==result['terminal_claim'] and terminal[0]['action_withheld'] and terminal[0]['step']==result['steps'])
check('unchanged_gripper_terminal_hold',len(holds)==result['same_gripper_hold_steps']==10 and [r['hold_index'] for r in holds]==list(range(1,11)) and all(r['action'][6]==actions[-1]['action'][6] for r in holds) and result['evaluation_hold_changes_gripper'] is False)
requests=[json.loads(f.read_text()) for f in sorted((RUN/'trial/requests').glob('*.request.json'))]
request_results=[json.loads(f.read_text()) for f in sorted((RUN/'trial/requests').glob('*.result.json'))]
budget=result['client_stats']['budget_snapshot']; stats=result['agent_stats']; model=stats['model_requests']
check('subscription_counts',len(requests)==len(request_results)==len(model)==budget['api_calls']==budget['subscription_inference_calls']==result['client_stats']['history_stats']['accepted_requests']==stats['model_decisions'],{'count':len(requests)})
check('subscription_only',budget['api_key_calls']==0 and budget['billing']=='chatgpt_subscription' and all(r['billing']=='chatgpt_subscription' and r['api_key_fallback'] is False for r in requests+request_results))
check('requested_model_revision_effort',config['revision']=='astra_tools_r1' and config['reasoning_effort']=='high' and all(r['harness_revision']=='astra_tools_r1' and r['payload']['model']=='gpt-6-astra' and r['payload']['reasoning']['effort']=='high' for r in requests))
check('all_requests_valid',all(r['status']=='valid' for r in request_results) and all(r['status']=='accepted' for r in model))
check('request_sequence',[r['request_ordinal'] for r in requests]==list(range(1,len(requests)+1)) and [r['ordinal'] for r in model]==list(range(1,len(model)+1)))
check('no_retries_or_episode_caps',result['automatic_retry'] is False and all(r['automatic_request_retry'] is False and r['attempt']==1 for r in requests) and all(config[k] is None for k in ('episode_action_cap','episode_time_cap_s','episode_model_call_cap')))
check('model_latency_sum',close(sum(r['latency_s'] for r in model),stats['timing']['model_s']))
truth=read('trial/truth_terminal.json');held=read('trial/truth_after_same_gripper_hold.json');evaluation=read('trial/terminal_evaluator_status.json')
check('strict_score_consistency',result['strict_success_at_terminal']==truth['strict_conjunction'] and result['strict_success_after_same_gripper_hold']==held['strict_conjunction'] and result['strict_success']==(truth['strict_conjunction'] and held['strict_conjunction']))
check('raw_score_consistency',result['raw_success_at_terminal']==truth['check_success']==evaluation['libero_success']==actions[-1]['evaluator']['raw_success'] and result['raw_success_after_same_gripper_hold']==held['check_success']==holds[-1]['raw_success'])
check('evaluation_step_counts',truth['steps']==result['steps'] and held['steps']==result['steps']+10)
keys=('status','steps','online_wall_s','strict_success','strict_success_at_terminal','strict_success_after_same_gripper_hold','raw_success_at_terminal','raw_success_after_same_gripper_hold','completion_wall_s','completion_time_eligible')
check('result_summary_consistency',all(result[k]==summary[k] for k in keys))
check('result_launch_timing',close(result['startup_wall_s']+result['online_wall_s']+result['postprocessing_wall_s'],result['total_wall_s']) and close(result['pre_episode_startup_wall_s']+result['episode_startup_wall_s'],result['startup_wall_s']))
check('online_timing_not_video_subtracted',result['video_time_subtracted'] is False and close(result['performance']['accounted_online_components_s']+result['performance']['other_online_wall_s'],result['online_wall_s']))
check('trace_timing_monotonic',all(a['online_wall_s']<=b['online_wall_s'] for a,b in zip(rows,rows[1:])))
for key,field in [('simulator_step','simulator_step_wall_s'),('feedback','feedback_wall_s')]:
 check(key+'_timing_sum',close(sum(r[field] for r in actions),result['performance']['timings'][key]['total_s']))
check('decision_timing_sum',close(sum(r['decision_latency_s'] for r in actions+terminal),result['performance']['timings']['decision']['total_s']))
collision=evaluation['collision'];forbidden=collision['forbidden_classes'];pairs={k:collision['pairs_by_class'].get(k,[]) for k in forbidden};counts={k:collision['counts'].get('points_'+k,0) for k in forbidden}
check('collision_snapshot_matches_trace',actions[-1]['evaluator']['collision']==collision)
check('collision_class_totals',sum(counts.values())==collision['counts']['forbidden_contact_points'])
check('collision_pair_totals',all(sum(x['points'] for x in pairs[k])==counts[k] for k in forbidden))
probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_read_frames,duration,r_frame_rate','-show_entries','format=duration,size','-of','json',str(RUN/'trial/episode.mp4')],capture_output=True,text=True)
video=json.loads(probe.stdout) if probe.returncode==0 else {'error':probe.stderr}
check('video_readable',probe.returncode==0 and video.get('streams') and int(video['streams'][0]['nb_read_frames'])>0,video)
check('video_finalized',result['video']['worker_alive'] is False and result['video']['error'] is None and result['video']['rendered_events']+result['video']['dropped_events']==result['video']['sampled_events'])
out={'audit_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'run':str(RUN),'method':'Saved artifacts only; hashes, trace continuity, metrics, recorded request metadata, and ffprobe frame read. No policy replay, model calls, or simulator reset.','passed':all(x['passed'] for x in checks.values()),'checks':checks,'check_count':len(checks),'metrics':{**{k:result[k] for k in keys},'model_requests':len(requests),'api_key_calls':budget['api_key_calls'],'model_latency_s':stats['timing']['model_s'],'vision_tool_s':stats['timing']['vision_tool_s'],'forbidden_contact_samples':collision['counts']['forbidden_contact_points'],'forbidden_contact_class_counts':counts,'forbidden_contact_pairs':pairs,'trace_event_counts':dict(Counter(r['event'] for r in rows))},'artifact_sha256':{f:digest(RUN/f) for f in ['result.json','trial/summary.json','trial/trace.jsonl','source_manifest.json','trial/episode.mp4']},'limitations':['No full simulator replay or independent physical re-evaluation; raw and strict scores cross-checked against recorded evaluator artifacts.','Contact counts are sampled contact points, not distinct collision incidents.','Video playback duration is not online wall time.']}
dst=PREP/(TAG+'_terminal_audit.json');dst.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'audit':str(dst),'passed':out['passed'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v['passed']],'metrics':out['metrics']},indent=2))
