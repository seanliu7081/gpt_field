from pathlib import Path
import json,hashlib,collections
import cv2
from PIL import Image,ImageOps
ROOT=Path('/workspace/gpt_field'); OUT=ROOT/'presentations/speedup_v1'; ASSETS=OUT/'assets'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def source(p):return {'path':str(p),'sha256':sha(p)}
keys=['status','task','suite','task_id','episode_index','seed','libero_success','libero_success_observed','task_success','completion_time_eligible','steps','api_calls','calls_by_kind','skill_decision_acts','pointing_results','macro_outcomes','macro_outcome_counts','macro_outcome_sequence','control_without_video_s','control_elapsed_s','elapsed_s','startup_to_initial_observation_s','timing_components','timing_components_note','path_length_m','prompt_version','final_macro','final_primitive','model_declared_complete','local_retries','history_stats']
evidence={'scope':'Recorded speedup v1 runs, not new experiments; each selected task has one episode. Results are not a benchmark success-rate estimate.','sources':[],'runs':{},'historical_runs':[]};frames=[]
for name in ['bowl_zero_shot_002','drawer_zero_shot_003']:
 run=ROOT/'runs'/('agent_speedup_v1_'+name)/'trial';s=load(run/'summary.json');a=load(run/'agent_stats.json');m=load(run/'manifest.json');sim=load(run/'simulator.json')
 tr=[json.loads(l) for l in (run/'trace.jsonl').open()];ev=[json.loads(l) for l in (run/'episode.events.jsonl').open()]
 entry={k:s.get(k) for k in keys};entry.update({'versions':m['configuration']['agent_speedup_v1']['versions'],'model':m['configuration']['api']['model'],'reasoning_effort':s['reasoning_effort'],'initial_status':tr[0]['sim_status'],'initial_observation_sha256':m['configuration']['expected_initial_observation_sha256'],'primitive_action_counts':a['primitive_actions'],'primitive_action_count_sum':sum(a['primitive_actions'].values()),'local_retry_limit':s['local_retries'],'actual_retries_used':a['counts'].get('local_retries_used'),'counts':a['counts'],'collision':s['collision'],'control_freq_hz':sim['metadata']['control_freq_hz'],'simulation_actions_duration_s':s['steps']/sim['metadata']['control_freq_hz'],'video_frames':ev[-1]['frame_start']+ev[-1]['frame_count'],'video_fps':20,'request_files_count':len(list((run/'requests').glob('*.request.json'))),'input_tokens':s['budget']['input_tokens'],'output_tokens':s['budget']['output_tokens']})
 entry['video_duration_s']=entry['video_frames']/20
 entry['skill_decisions']=[{'step':r['step'],'skill':(r.get('routing') or {}).get('skill'),'params':(r.get('routing') or {}).get('params')} for r in ev if r['event']=='skill_decision']
 entry['final_executed']={k:tr[-2].get(k) for k in ['event','step','macro','primitive','success']}
 entry['final_primitive_feedback']=tr[-1]['feedback']['primitive_feedback']
 entry['terminal_semantics']='LIBERO success stops immediately while final place_release primitive is still incomplete (done=false). Bowl ends in lower, still closed; drawer ends on third release action before measured fingers stable. Do not claim completed release+retreat/fully completed macro.'
 entry['completed_macro_note']='Macro outcomes include initial automatic calibrate but exclude final interrupted-by-success place_release. Model decisions exclude calibrate and include final place_release; matching counts are coincidental.'
 evidence['runs'][name]=entry
 for fn in ['summary.json','agent_stats.json','manifest.json','simulator.json','trace.jsonl','episode.events.jsonl','episode.mp4']:
  evidence['sources'].append(source(run/fn))
 chosen=[('initial','observation',0,'初始观测')]
 if name.startswith('bowl'):
  chosen += [('hold_pass','skill_decision',387,'抓取验证通过后；step 387'),('terminal','terminal',448,'LIBERO 成功；449 步；lower 尚未结束')]
 else:
  chosen += [('handle_released','skill_decision',552,'把手已释放、手转为朝下；step 552'),('bowl_hold_pass','skill_decision',818,'碗抓取验证通过后；step 818'),('terminal','terminal',966,'LIBERO 成功；967 步；release 第 3 步')]
 cap=cv2.VideoCapture(str(run/'episode.mp4'));assert cap.isOpened()
 for label,event,step,caption in chosen:
  row=next(x for x in ev if x['event']==event and x['step']==step);index=row['frame_start'];cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();assert ok
  image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));prefix=name+'_'+label
  full=ASSETS/(prefix+'_full.png');image.save(full)
  paths={}
  for camera,rect in [('agent',[44,181,380,517]),('wrist',[44,577,380,913])]:
   p=ASSETS/(prefix+'_'+camera+'.png');ImageOps.mirror(image.crop(rect)).save(p);paths[camera]=str(p)
  frames.append({'run':name,'label':label,'caption_zh':caption,'source_video':source(run/'episode.mp4'),'events_path':str(run/'episode.events.jsonl'),'frame_index':index,'video_time_s':row['video_time_s'],'event':event,'trace_action_index':step,'state_convention':'skill_decision frame is state before that action; terminal state after final executed action','source_dimensions':image.size,'full_frame':str(full),'upright_crops':paths,'crop_rectangles':{'agent':[44,181,380,517],'wrist':[44,577,380,913]},'transform':'Exact crop from recorded MP4, then horizontal mirror of legacy camera panels to match upright model orientation; no new rendering/inference.','macro':row['macro'],'primitive':row['primitive'],'success':row['success']})
 cap.release()
for name in ['agent_speedup_v1_bowl_zero_shot_001','agent_speedup_v1_drawer_zero_shot_001','agent_speedup_v1_drawer_zero_shot_002']:
 p=ROOT/'runs'/name/'trial'/'summary.json';s=load(p);evidence['historical_runs'].append({'run':name,**{k:s.get(k) for k in ['status','libero_success','steps','api_calls','control_without_video_s','elapsed_s','prompt_version','error']},'interpretation':'Earlier prompt version, user-interrupted/uncompleted; not a completed task timing or matched seed sweep.'});evidence['sources'].append(source(p))
p=ROOT/'runs/direct_medium_libero_success_002';s=load(p/'summary.json');m=load(p/'manifest.json');direct={k:s.get(k) for k in keys};direct['video_excluded_control_derived_s']=s['control_elapsed_s']-s['timing_components']['video_render_s'];direct['input_tokens']=s['budget']['input_tokens'];direct['policy']=m['configuration']['policy'];direct['interpretation']='Historical successful same named bowl task and seed 7, different controller/prompt/history/contact monitoring and calibration workload. Not a controlled ablation or general speedup estimate.';direct['derived_ratio_direct_div_v1_video_excluded']=direct['video_excluded_control_derived_s']/evidence['runs']['bowl_zero_shot_002']['control_without_video_s'];evidence['historical_direct_bowl']=direct
for fn in ['summary.json','manifest.json']:evidence['sources'].append(source(p/fn))
evidence['timing_interpretation_zh']=['135.41 秒属于 bowl002，不是 drawer003。','drawer003 去除视频渲染的控制墙钟时间 234.54 秒（3.91 分钟）；含渲染 340.75 秒；总墙钟 348.96 秒（5.82 分钟）。','drawer003 模拟控制时间 = 967 / 20 = 48.35 秒；视频 204.7 秒是添加动作前后帧与决策停顿后的回放长度，不是控制时间。','旧 drawer002 的 1489.35 秒约 24.82 分钟是未完成、被停止的另一条记录。','decision_s 包含 model_pipeline_s，不能叠加计数；video_render_s 可从 control_elapsed_s 扣除。','simulator_step_s 是主机执行模拟器调用的墙钟时间，不是 20 Hz 的模拟时间。']
evidence['result_caveats_zh']=['两个选定成功运行各 1 次，且提示版本不同，不应宣称跨任务 100% 成功率。','两次运行都记录非允许接触；碰撞只记录，不阻断 raw LIBERO success。','forbidden_contact_points 是跨物理子步累积接触点，不能称为独立碰撞次数。','zero-shot 表示这些记录没有示范输入；局部技能、常量、提示已经过此前试验调整，不能解读为完全未经开发调试。']
(OUT/'evidence_results.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n');(ASSETS/'source_frame_manifest.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2)+'\n')
# Contact sheet for human review.
thumbs=[]
for row in frames:
 im=Image.open(row['upright_crops']['agent']);thumbs.append(im)
sheet=Image.new('RGB',(336*4,336*2),'#f1f4f7')
for i,im in enumerate(thumbs):sheet.paste(im,((i%4)*336,(i//4)*336))
sheet.save(ASSETS/'task_frames_contact_sheet.png')
print(json.dumps({'evidence':str(OUT/'evidence_results.json'),'frames':len(frames),'manifest':str(ASSETS/'source_frame_manifest.json')},ensure_ascii=False))
