#!/usr/bin/env python3
"""Read-only historical evidence index and bounded joint audit; never imports policy/simulator."""
import argparse,collections,csv,hashlib,json,math,os,pathlib,statistics
SELECT={
'agent_speedup_v2_user_check_libero_10_t1_001':[1,3,6,9,12,13],
'agent_speedup_v3_t1_008':[1,2,3,10,20,30,39,40],
'agent_speedup_v3_t1_009':[1,8,17,20,24,25],
'agent_speedup_v3_g5_001':[1,2,3,4,5],
'agent_speedup_v3_g5_002':[1,3,6],
'agent_speedup_v3_t1_011':[1,2,3],
'agent_speedup_v3_t1_012':list(range(1,10))}
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def quant(v):
 v=sorted(x for x in v if isinstance(x,(int,float)) and math.isfinite(x))
 def q(p):
  if not v:return None
  i=(len(v)-1)*p;l=int(i);return v[l]+(v[min(l+1,len(v)-1)]-v[l])*(i-l)
 return {'n':len(v),'p50':q(.5),'p90':q(.9),'min':v[0] if v else None,'max':v[-1] if v else None,'small_sample':len(v)<30,'quantile_method':'linear interpolation'}

def texts(value):
 if isinstance(value,dict):
  if value.get('type') in ['input_text','output_text']:yield value.get('text','')
  for k,v in value.items():
   if k not in ['text','image_url','encrypted_content']:yield from texts(v)
 elif isinstance(value,list):
  for v in value:yield from texts(v)

def parse_output(d):
 s=''.join(texts(d.get('output',[])))
 try:return json.loads(s)
 except ValueError:return {'unparsed_text':s}

def leaf_fields(d,keys,prefix=''):
 if isinstance(d,dict):
  for k,v in d.items():
   p=prefix+'.'+k if prefix else k
   if k in keys and v is not None:yield {'path':p,'value':v}
   if k not in ['image_url','rgb','external_rgb','wrist_rgb']:yield from leaf_fields(v,keys,p)
 elif isinstance(d,list):
  for i,v in enumerate(d):yield from leaf_fields(v,keys,f'{prefix}[{i}]')

def run(root,out):
 out.mkdir(exist_ok=True,parents=True);inventory=[];errors=[]
 for directory,dirs,files in os.walk(root/'runs'):
  dirs[:]=[x for x in dirs if x not in ['frozen_source','frozen_runtime','runtime','requests','frames','images','__pycache__','base_v2_artifacts','node_modules']]
  p=pathlib.Path(directory);rel=p.relative_to(root)
  if any(x in files for x in ['summary.json','trace.jsonl','performance.json','review.json','run_state.json']):
   summary=read(p/'summary.json',{});state=read(p/'run_state.json',{});review=read(p/'review.json',{})
   req=p/'requests'; counts=collections.Counter(x.suffixes[-2] if len(x.suffixes)>1 else x.suffix for x in req.glob('*.json')) if req.exists() else {}
   diag=any(x in str(rel).lower() for x in ['diagnos','probe','replay','smoke','preflight','capture','design','preparation']) or summary.get('mode')=='smoke'
   status=summary.get('status',state.get('status',review.get('status','missing_summary')))
   cat='diagnostic_or_preparation' if diag else ('review_only' if 'review.json' in files and 'trace.jsonl' not in files else ('raw_success' if summary.get('libero_success') is True else ('user_stopped' if 'stop' in str(status) else ('infrastructure_or_protocol_interrupted' if str(status) in ['infrastructure_error','subscription_error','error'] else 'incomplete_or_unclassified'))))
   row={'run':str(rel),'category':cat,'status':status,'summary_present':bool(summary),'trace_present':'trace.jsonl' in files,'performance_present':'performance.json' in files,'review_present':bool(review) or 'user_check_review.json' in files,'frozen_source_present':(p/'frozen_source').is_dir(),'frozen_runtime_present':(p/'frozen_runtime').is_dir(),'requests':counts.get('.request',0),'responses':counts.get('.response',0),'results':counts.get('.result',0),'steps':summary.get('steps'),'raw_success':summary.get('libero_success'),'strict_quality_v2_success':summary.get('v2_success'),'task_key':summary.get('task_key'),'episode_index':summary.get('episode_index'),'seed':summary.get('seed'),'elapsed_s':summary.get('elapsed_s'),'control_without_video_s':summary.get('control_without_video_s'),'service_elapsed_s':None,'error':str(summary.get('error') or '')[:500],'missing':'|'.join(x for x in ['summary.json','performance.json','trace.jsonl'] if x not in files),'independent_trial_identity':'unknown; preparation/launch/summary directories must not be summed as independent trials'}
   inventory.append(row)
 inventory.sort(key=lambda x:x['run'])
 with (out/'run_inventory.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
 cases=[];runs={}
 for name,ids in SELECT.items():
  p=root/'runs'/name;summary=read(p/'summary.json',{});stats=read(p/'agent_stats.json',{});reqs=sorted((p/'requests').glob('*.request.json'));timings=[];route_counts=collections.Counter();timing_first=[];timing_later=[]
  # Lightweight request timing index only; semantic audit remains limited to SELECT.
  meta=[]
  for r in reqs:
   d=read(r,{});rsp=read(r.with_name(r.name.replace('.request.','.response.')),{});res=read(r.with_name(r.name.replace('.request.','.result.')),{});dur=(res.get('v3_transport_diagnostics') or {}).get('elapsed_s')
   timing_source='v3_transport_diagnostics.elapsed_s'
   if dur is None:
    before=d.get('budget_elapsed_s_before');after=res.get('budget',{}).get('elapsed_s');dur=after-before if isinstance(before,(float,int)) and isinstance(after,(float,int)) else None;timing_source='result.budget.elapsed_s - request.budget_elapsed_s_before (pipeline interval, includes client work)'
   ordinal=d.get('request_ordinal',int(r.name[:6]));step=d.get('request_meta',{}).get('step');meta.append({'ordinal':ordinal,'step':step,'duration_s':dur,'source':timing_source})
   timings.append(dur);(timing_first if ordinal==1 else timing_later).append(dur)
   if ordinal not in ids:continue
   decision=parse_output(rsp);route_counts[decision.get('type','unknown')]+=1;current={}
   for item in d.get('payload',{}).get('input',[]):
    if item.get('role')=='user':
     for s in texts(item):
      try:val=json.loads(s)
      except ValueError:continue
      if isinstance(val,dict) and val.get('request')=='agent':current=val
   structured_keys={x['path']:x['value'] for x in leaf_fields(current,{'answer_key','physical_truth','contact_identities','object_poses','reward','libero_success','task_predicate'})}
   context=decision.get('motion_context') or {};scene=decision.get('scene_update') or {};entities=scene.get('entities') or []
   row={'case_id':f'{name}:request_{ordinal:06d}','run':str(p.relative_to(root)),'request_ordinal':ordinal,'start_step':step,'end_step_exclusive':None,'request_path':str(r.relative_to(root)),'response_path':str(r.with_name(r.name.replace('.request.','.response.')).relative_to(root)),'result_path':str(r.with_name(r.name.replace('.request.','.result.')).relative_to(root)),'request_sha256':hashlib.sha256(r.read_bytes()).hexdigest(),'request_status':res.get('status'),'request_kind':d.get('request_meta',{}).get('kind'),'route':decision.get('type'),'belief':decision.get('belief'),'plan':decision.get('plan'),'why':decision.get('why'),'target_id':context.get('target_id'),'held_object_id_model_claim':context.get('held_object_id'),'intent':context.get('intent'),'scene_entity_ids':[e.get('id',e.get('entity_id')) for e in entities if isinstance(e,dict)],'scene_update_entity_count':len(entities),'perception':{'identity':'model-declared; no independent pixel identity label in this audit','grasp_region':'unknown; no feasible-contact-region ground truth annotation','metric_error_m':None,'uncertainty_coverage':None,'reason':'No state/controller/image equivalence replay performed; do not compare grasp point against object centroid or claim calibrated error.'},'checker_label':'unknown_counterfactual_not_executed','failure_classification':[],'causal_confidence':'requires window outcome interpretation','duration_s':dur,'duration_source':timing_source,'new_model_requests':0,'new_rollouts':0,'input_audit':{'forbidden_structured_key_hits':structured_keys,'scope':'selected current structured input; not proof by keyword alone'},'raw_run_success':summary.get('libero_success'),'run_autonomous_quality':summary.get('v2_success'),'evidence_paths':[str((p/'trace.jsonl').relative_to(root)),str((p/'summary.json').relative_to(root))]}
   cases.append(row)
  for row in [x for x in cases if x['run']==str(p.relative_to(root))]:
   nxt=[x['step'] for x in meta if x['ordinal']>row['request_ordinal'] and x['step'] is not None];row['end_step_exclusive']=nxt[0] if nxt else summary.get('steps');row['actions_in_window']=max(0,(row['end_step_exclusive'] or 0)-(row['start_step'] or 0))
  ec=collections.Counter();primitive=collections.Counter();timingvals=collections.defaultdict(list);reasons=collections.Counter();windowevents=collections.defaultdict(list)
  selected=[x for x in cases if x['run']==str(p.relative_to(root))]
  with (p/'trace.jsonl').open() as f:
   for ln,line in enumerate(f,1):
    try:d=json.loads(line)
    except ValueError:errors.append({'path':str(p/'trace.jsonl'),'line':ln});continue
    ev=d.get('event','');ec[ev]+=1;step=d.get('step');fb=d.get('feedback') or {};diag=d.get('diagnostics') or {}
    if ev=='action':
     primitive[diag.get('primitive',diag.get('controller','unknown'))]+=1
     for k in ['decision_latency_s','simulator_step_s','video_render_s','observe_transition_s']:
      if isinstance(d.get(k),(int,float)):timingvals[k].append(d[k])
    if ev=='feedback':
     primitive[fb.get('primitive','unknown')+'_feedback']+=1
    for row in selected:
     if isinstance(step,int) and row['start_step']<=step<max(row['end_step_exclusive'],row['start_step']+1):
      if ev not in ['action','feedback','execution','observation','executed_action','sim_step']:
       windowevents[row['case_id']].append({'line':ln,'event':ev,'step':step})
      if ev=='feedback' and step==row['end_step_exclusive']-1:
       row['terminal_feedback']={k:fb.get(k) for k in ['primitive','gripper_command','finger_gap_mm']};row['terminal_feedback']['reasons']=list(leaf_fields(fb,{'reason','status','missing_evidence','missing_evidence_requests'}))[:20];row['terminal_feedback_line']=ln
  for row in selected:row['event_rows']=windowevents[row['case_id']][-20:]
  runs[name]={'summary_path':str((p/'summary.json').relative_to(root)),'status':summary.get('status'),'steps':summary.get('steps'),'raw_success':summary.get('libero_success'),'strict_quality_v2_success':summary.get('v2_success'),'timing_components':summary.get('timing_components'),'elapsed_s':summary.get('elapsed_s'),'control_elapsed_s':summary.get('control_elapsed_s'),'control_without_video_s':summary.get('control_without_video_s'),'model_calls':summary.get('model_calls'),'request_timing':quant(timings),'first_request_timing':quant(timing_first),'subsequent_request_timing':quant(timing_later),'request_timeline':meta,'trace_event_counts':ec,'primitive_counts':primitive,'sampled_action_timings':{k:quant(v) for k,v in timingvals.items()},'operation_counts':stats.get('operation_state',{}).get('counts'),'evidence_view_attempts':stats.get('evidence_view_attempts'),'policy_version':stats.get('policy_version'),'performance_timings':stats.get('performance_timings'),'preparation_s':summary.get('user_check',{}).get('preparation_elapsed_s'),'preparation_request_timing':quant([summary.get('user_check',{}).get('preparation_model_s')]),'completion_target':read(p/'performance.json',{}).get('completion_target'),'autonomous_quality_at_raw_cutoff':summary.get('physical_truth'),'unavailable_distributions':['simulator step latency','per-step video cost','per-step feedback cost','finalization latency (aggregate only)']}
 # Frozen interpretation rules for exactly the selected historical windows.
 for row in cases:
  name=pathlib.Path(row['run']).name; n=row['request_ordinal']; reasons=[str(x['value']) for x in row.get('terminal_feedback',{}).get('reasons',[])]; labels=[]
  if any('stall' in x or 'tracking_mismatch' in x for x in reasons):labels.append('action_tracking_or_reachability')
  if any('geometry' in x or 'coverage' in x for x in reasons):labels.append('geometry_or_checker_dependency')
  if row['route'] in ['acquire_evidence','escalate']:labels.append('recovery_evidence_unavailable')
  row['failure_classification']=labels or ['no_specific_failure_confirmed_in_selected_window']
  row['causal_confidence']='observed route and terminal feedback; physical cause unknown unless separately linked'
  row['new_evidence_status']='unknown; a new image/request alone does not demonstrate useful evidence'
  row['recovery_chain']={'anomaly':None,'observation_proposed':row['route']=='acquire_evidence','observation_executed':None,'effective_new_evidence':None,'state_updated':bool(row['scene_update_entity_count']),'action_changed':None,'operation_completed':None,'attribution':'not established from route label'}
  if name.endswith('t1_008'):
   row['run_level_finding']='149 program actions before gripper phase; 712 ordinary fragments + 12 short motion complete raw task; visual interaction observations=0'
   row['evidence_paths'].append('runs/agent_speedup_v3_t1_008/time_component_analysis.json')
   if n>=10:row['attribution']='ordinary fragment/model fallback after semantic program abandoned; not new-program grasp/release success'
  elif name.endswith('t1_009'):
   row['run_level_finding']='release provenance overwritten by model geometry replacement; raw visual support rejected; unresolved object binding later ignored by ordinary action emission'
   row['evidence_paths'].append('runs/agent_speedup_v3_staged_001/gate_A_analysis.json')
   if n>=17:row['failure_classification']+=['grasp_release_state_provenance','object_lifecycle_binding']
  elif name.endswith('g5_001'):
   row['run_level_finding']='posture closure created possible plate payload despite push intent; accepted hold=0; task failure'
   row['failure_classification']+=['grasp_state_intent_confusion'];row['evidence_paths'].append('runs/agent_speedup_v3_staged_001/gate_C2_analysis.json')
  elif name.endswith('g5_002'):
   row['run_level_finding']='existing revision9 fix: attachment probes/references/active probe actions all zero; task still fails on coverage/view dependency'
   row['evidence_paths'].append('runs/agent_speedup_v3_staged_001/gate_C2_fix_analysis.json')
  elif name.endswith('t1_011'):
   row['run_level_finding']='measured descent stalls; independent old contact audit found finger-table contact; acquire_evidence rejected for missing coverage witness; no closure'
   row['evidence_paths'].append('runs/agent_speedup_v3_local_execution_001/t1_011_analysis.json')
   if n==2:row['new_evidence_status']='empty scene update, same intent, 24 further actions then blocked; no successful correction'
   if n==3:row['recovery_chain'].update(anomaly='measured_tracking_mismatch',observation_executed=False,effective_new_evidence=False,state_updated=False,action_changed=False,operation_completed=False,attribution='interface failed before observation')
  elif name.endswith('t1_012'):
   row['run_level_finding']='automatic view executed zero actions; model fallback retreats and revises estimates; final missing-milk geometry prevents stage goal and view planning'
   row['evidence_paths'].append('runs/agent_speedup_v3_local_execution_001/t1_012_analysis.json')
   if n==4:
    row['recovery_chain'].update(anomaly='tracking stall and unobservable automatic view',observation_proposed=True,observation_executed=True,effective_new_evidence=True,state_updated=True,action_changed=True,operation_completed=False,attribution='47-step model-directed ordinary retreat followed by request5 model update, not automatic view module')
    row['new_evidence_status']='fresh public RGB after actual retreat used by model to revise A/table in request5; subsequent operation incomplete'
   if n==7:row['new_evidence_status']='milk geometry repeated and rejected; subsequent request8 changes shape/correspondence; not all milk requests are equivalent'
   if n==9:row['recovery_chain'].update(anomaly='missing fresh milk geometry before lift goal',observation_executed=False,effective_new_evidence=False,state_updated=False,action_changed=False,operation_completed=False,attribution='semantic_goal_requires_geometry_before_view_planning')
  elif 'v2_user_check' in name:
   row['run_level_finding']='all 897 actions ordinary fragments; explicit model decisions after measured stall; raw success precedes autonomous full release/retreat quality'
   if n==3:row['new_evidence_status']='model explicitly requests bounded grasp after stalled descent; no hardwired stall-to-close rule is established by this trace'
   if n==13:row['attribution']='raw success after six opening actions; following planned retreat not completed at raw-success cutoff'
 cases.sort(key=lambda x:(x['run'],x['request_ordinal']))
 assert len(cases)==40,len(cases)
 with (out/'audit_cases.jsonl').open('w') as f:
  for d in cases:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 metrics={'audit_version':1,'scope':'historical saved records only; no model, no simulator, no heldout truth','index_entries':len(inventory),'index_categories':collections.Counter(x['category'] for x in inventory),'detailed_windows':len(cases),'detailed_window_limit':40,'selection':SELECT,'runs':runs,'read_errors':errors,'new_requests':0,'new_rollouts':0,'candidate_recommendation':'pure_IO_compact_display_and_external_snapshot_throttling_from_frozen_v7','not_claimed':['new task success','S300 improvement','physical clearance','metric perception accuracy','statistical generalization']}
 (out/'audit_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
 print(json.dumps({'index_entries':len(inventory),'detailed_windows':len(cases),'read_errors':len(errors)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();run(a.root.resolve(),a.output.resolve())
