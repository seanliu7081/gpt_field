#!/usr/bin/env python3
"""Offline real-ffmpeg display benchmark using a fixed historical public action/RGB prefix.

No model, policy act, simulator, truth, or new task success. All outputs use fresh names.
"""
import argparse,hashlib,json,pathlib,statistics,subprocess,sys,time,traceback

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=pathlib.Path,required=True);ap.add_argument('--trace',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--steps',type=int,default=120);args=ap.parse_args()
 if not 1<=args.steps<=897:raise ValueError('steps must be 1..897')
 args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.baseline.resolve()/'src'))
 from gpt_field.agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder
 from gpt_field.agent_speedup_v3_video import AgentSpeedUpV3VideoRecorder
 from gpt_field.observation import PolicyObservation
 from gpt_field.policy import Decision
 initial=None;rows={};loaded_until=0
 started=time.monotonic()
 with args.trace.open() as f:
  for ln,line in enumerate(f,1):
   d=json.loads(line);ev=d.get('event');step=d.get('step')
   if ev=='reset':initial=d['observation']
   if isinstance(step,int) and 0<=step<args.steps:
    item=rows.setdefault(step,{'step':step})
    if ev=='action':item['action']=d['action'];item['diagnostics']=d['diagnostics'];item['decision_latency_s']=d.get('decision_latency_s')
    elif ev=='executed':item['observation']=d['observation'] # deliberately omit evaluator sim_status and contact truth
    elif ev=='feedback':item['feedback']=d['feedback']
   if ev=='feedback' and step==args.steps-1:loaded_until=ln;break
 if initial is None or len(rows)!=args.steps:raise ValueError('missing fixed public prefix')
 ordered=[rows[i] for i in range(args.steps)]
 for r in ordered:
  if not {'action','diagnostics','observation','feedback'}<=r.keys():raise ValueError(f'incomplete action chain {r["step"]}')
 raw={'initial_observation':initial,'actions':ordered};input_sha=digest(raw)
 # Observation validation/decoding and trace reads are common loading work, outside renderer timer.
 obs0=PolicyObservation.from_dict(initial);observations=[PolicyObservation.from_dict(r['observation']) for r in ordered]
 decisions=[Decision(tuple(r['action']),r['diagnostics']) for r in ordered]
 load_s=time.monotonic()-started
 sources={str(p.relative_to(args.baseline)):sha(p) for n in ['agent_speedup_v2_video.py','agent_speedup_v1_video.py','agent_speedup_v3_video.py','video.py','direct_history_video.py','observation.py'] for p in [args.baseline/'src/gpt_field'/n]}
 setup={'kind':'offline_display_io_only','trace':str(args.trace.resolve()),'prefix_action_indices':[0,args.steps-1],'trace_lines_read':loaded_until,'input_sha256':input_sha,'steps':args.steps,'common_input_load_s':load_s,'order':['A1','B1','B2','A2'],'source_sha256':sources,'new_model_requests':0,'new_simulator_rollouts':0,'service_success_claim':False,'success_argument':False,'evaluator_truth_forwarded':False,'format_change':'A legacy 1600x960 with decision/action repeats; B existing compact 640x384 one frame per executed action; both initial/terminal frames','clock':'time.monotonic; create + observe + decisions + executed + synchronous close/drain; ffprobe excluded'}
 (args.output/'benchmark_manifest.json').write_text(json.dumps(setup,indent=2));results=[]
 for label in setup['order']:
  cls=AgentSpeedUpV2VideoRecorder if label[0]=='A' else AgentSpeedUpV3VideoRecorder
  video=args.output/(label+'.mp4');rec=None;row={'trial':label,'arm':label[0],'input_sha256_before':digest(raw),'steps':args.steps,'new_model_requests':0,'new_simulator_rollouts':0,'success_claim':False};beg=time.monotonic()
  try:
   rec=cls(video,episode_id='offline historical v7 public RGB prefix / '+label,fps=20)
   rec.observe(obs0,step=0);obs=obs0
   for r,after,decision in zip(ordered,observations,decisions):
    rec.decision(obs,decision,decision_latency_s=r['decision_latency_s'])
    rec.executed(after,step=r['step'],success=False,sim_status={},feedback=r['feedback'],history_stats=r['diagnostics'].get('history_stats'))
    obs=after
   before_close=time.monotonic();rec.close(status='offline_prefix_complete');ended=time.monotonic()
   row.update(status='complete',render_encode_before_close_s=before_close-beg,close_drain_s=ended-before_close,total_display_wall_s=ended-beg,frame_count=rec.frame_count,render_errors=getattr(rec,'render_errors',0),input_sha256_after=digest(raw),video=str(video.resolve()))
   probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,duration','-of','json',str(video)],capture_output=True,text=True,check=True)
   row['ffprobe']=json.loads(probe.stdout)['streams'][0];row['input_unchanged']=row['input_sha256_before']==row['input_sha256_after']==input_sha
   if not row['input_unchanged']:raise RuntimeError('renderer modified shared public inputs')
   if int(row['ffprobe']['nb_frames'])!=row['frame_count']:raise RuntimeError('encoded frame count mismatch')
  except BaseException as exc:
   row.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),total_display_wall_s=time.monotonic()-beg)
   if rec is not None and not rec.closed:
    try:rec.close(status='offline_benchmark_error',error=str(exc))
    except BaseException as close_error:row['close_error']=repr(close_error)
  results.append(row)
  with (args.output/'attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps({k:row.get(k) for k in ['trial','status','total_display_wall_s','frame_count','input_unchanged','error']}),flush=True)
  if row['status']!='complete':break
 medians={arm:statistics.median([r['total_display_wall_s'] for r in results if r['arm']==arm and r['status']=='complete']) for arm in ['A','B'] if any(r['arm']==arm and r['status']=='complete' for r in results)}
 summary={**setup,'trials':results,'median_display_wall_s':medians,'B_median_reduction_fraction':1-medians['B']/medians['A'] if set(medians)=={'A','B'} else None,'all_inputs_unchanged':all(r.get('input_unchanged',False) for r in results),'completed_trials':sum(r['status']=='complete' for r in results),'interpretation':'component performance only; same saved observations/actions, not new episode, no S300 result; frame reduction is display-only and retains all actual actions'}
 summary['source_sha256_after']={n:sha(args.baseline/n) for n in sources}
 summary['recorder_sources_unchanged']=summary['source_sha256_after']==sources
 summary['script_sha256']=sha(pathlib.Path(__file__))
 (args.output/'display_io_benchmark.json').write_text(json.dumps(summary,indent=2))
 with (args.output/'mechanism_trace.jsonl').open('w') as f:
  for r in results:f.write(json.dumps({'kind':'offline_display_IO','candidate':'compact_display_existing_v3_recorder','trial':r['trial'],'input_sha256':input_sha,'public_action_count':args.steps,'output_frames':r.get('frame_count'),'total_display_wall_s':r.get('total_display_wall_s'),'close_drain_s':r.get('close_drain_s'),'raw_success':None,'S300':None,'model_or_macro_fallback':'not_applicable_no_policy_execution','state_or_action_changed':False if r.get('input_unchanged') else None,'status':r['status'],'evidence':str(args.output/'display_io_benchmark.json')})+'\n')
if __name__=='__main__':main()
