"""Build an official-task-preserving free-object XY relayout; reset-only preview."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/gpt_field')
sys.path.insert(0,str(ROOT/'src'))
from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config
from gpt_field.libero import _LiberoWorker
from gpt_field.agent_speedup_v2_user_check import initial_observation_sha256
from gpt_field.observation import PolicyObservation
from gpt_field.contact_skills import StepView


def native(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)

def write(p,data):p.write_text(json.dumps(data,indent=2,sort_keys=True,default=native,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def snapshot(env):
 result={}
 for kind,collection in [('object',env.env.objects_dict),('fixture',env.env.fixtures_dict)]:
  for name,obj in collection.items():
   body=env.sim.model.body_name2id(obj.root_body)
   result[name]={'kind':kind,'body_xyz_m':env.sim.data.body_xpos[body].tolist(),
      'body_quat_wxyz':env.sim.data.body_xquat[body].tolist(),
      'joints':{j:np.atleast_1d(env.sim.data.get_joint_qpos(j)).tolist() for j in obj.joints}}
 return result

def contacts(env):
 m,d=env.sim.model,env.sim.data
 roots={m.body_name2id(obj.root_body):name for name,obj in {**env.env.objects_dict,**env.env.fixtures_dict}.items()}
 def owner(g):
  if (m.geom_id2name(g) or '').startswith('table_collision'):return 'table'
  b=int(m.geom_bodyid[g])
  while b:
   if (m.body_id2name(b) or '') in ('living_room_table','living_room_table_col','study_table'):return 'table'
   if b in roots:return roots[b]
   if (m.body_id2name(b) or '').startswith(('robot','gripper')):return 'robot'
   b=int(m.body_parentid[b])
  return 'world'
 out=[]
 for i in range(d.ncon):
  c=d.contact[i];a,b=owner(int(c.geom1)),owner(int(c.geom2))
  if a==b or c.dist>1e-6:continue
  out.append({'a':a,'b':b,'distance_m':float(c.dist),'geom1':m.geom_id2name(int(c.geom1)),'geom2':m.geom_id2name(int(c.geom2)),
    'geom1_id':int(c.geom1),'geom2_id':int(c.geom2),
    'geom1_body':m.body_id2name(int(m.geom_bodyid[int(c.geom1)])),'geom2_body':m.body_id2name(int(m.geom_bodyid[int(c.geom2)]))})
 return out

def images(out,response,prefix):
 obs=response['observation'];view=StepView(PolicyObservation.from_dict(obs))
 write(out/(prefix+'observation.json'),obs)
 Image.fromarray(view.external_rgb_upright).save(out/(prefix+'external.png'))
 Image.fromarray(view.wrist_rgb_upright).save(out/(prefix+'wrist.png'))
 return initial_observation_sha256(obs)

def panel(out,task,prompt,passed):
 path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 title=ImageFont.truetype(path,25);font=ImageFont.truetype(path,18);small=ImageFont.truetype(path,16)
 im=Image.new('RGB',(1060,678),'#101820');d=ImageDraw.Draw(im)
 d.text((18,12),f'LIBERO-10 T{task:02d} | Original vs relayout',font=title,fill='#f1f5f9')
 words=prompt.split();lines=['']
 for word in words:
  trial=(lines[-1]+' '+word).strip()
  if d.textlength(trial,font=font)>1018:lines.append(word)
  else:lines[-1]=trial
 for i,line in enumerate(lines):d.text((18,47+i*22),line,font=font,fill='#cbd5e1')
 d.text((18,92),'ORIGINAL',font=font,fill='#cbd5e1');d.text((546,92),'PROPOSED RELAYOUT',font=font,fill='#5eead4')
 im.paste(Image.open(out/'original_external.png'),(18,122));im.paste(Image.open(out/'relayout_external.png'),(546,122))
 footer='Existing objects rearranged; original task and fixture states preserved.' if passed else 'PREPARATION CHECK FAILED — review validation before using this layout.'
 d.text((18,648),footer,font=small,fill='#cbd5e1' if passed else '#fca5a5')
 im.save(out/'layout_comparison.png')

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-id',type=int,required=True,choices=[3,4,5,7,8,9]);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--xy-json',type=Path,required=True)
 args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'initial_state.npy').exists():raise RuntimeError('Refusing to overwrite an existing prepared state; use another candidate directory')
 xy=json.loads(args.xy_json.read_text());assert xy and all(len(v)==2 and all(np.isfinite(v)) for v in xy.values())
 shutil.copy2(__file__,out/'preparation_source.py')
 if args.xy_json.resolve()!=out/'requested_xy.json':shutil.copy2(args.xy_json,out/'requested_xy.json')
 cfg,_=build_v2_sim_config(f'libero_10_t{args.task_id}');cfg=replace(cfg,collision_check_mode='none')
 w=_LiberoWorker(cfg)
 try:
  original_response=w.reset(args.task_id,0);original_obs_sha=images(out,original_response,'original_')
  reference=ROOT/f'reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning/libero_10_t{args.task_id}/initial_observation.json'
  expected_obs_sha=initial_observation_sha256(json.loads(reference.read_text()))
  assert original_obs_sha==expected_obs_sha,'Original reset differs from preserved official observation'
  original_settled=snapshot(w.env);original_contacts=contacts(w.env)
  task=w.benchmark.get_task(args.task_id);source=w.benchmark_root/'init_files'/task.problem_folder/task.init_states_file
  import torch
  original=np.asarray(torch.load(source,map_location='cpu',weights_only=False)[0]).copy();np.save(out/'original_initial_state.npy',original,allow_pickle=False)
  bddl=Path(w.benchmark.get_task_bddl_file_path(args.task_id));(out/'task.bddl').write_bytes(bddl.read_bytes())
  custom=original.copy();changed=[]
  for name,point in xy.items():
   obj=w.env.env.objects_dict[name];joint=obj.joints[0];addr=w.env.sim.model.get_joint_qpos_addr(joint)
   assert isinstance(addr,tuple) and addr[1]-addr[0]==7,f'{name} must be a free-body object, not an articulated fixture'
   start,end=map(int,addr);before=original[1+start:1+end].copy();custom[1+start:1+start+2]=point
   changed.append({'object':name,'joint':joint,'state_xy_indices':[1+start,2+start],
       'original_xyz_m':before[:3].tolist(),'relayout_xyz_m':custom[1+start:1+start+3].tolist(),
       'xy_displacement_m':float(np.linalg.norm(custom[1+start:1+start+2]-before[:2]))})
  np.save(out/'initial_state.npy',custom,allow_pickle=False)
  w.config=replace(cfg,initial_state_override=str(out/'initial_state.npy'))
  response=w.reset(args.task_id,0);relayout_obs_sha=images(out,response,'relayout_')
  settled=snapshot(w.env);np.save(out/'settled_state.npy',w.env.sim.get_state().flatten(),allow_pickle=False)
  new_contacts=contacts(w.env)
  bad_contacts=[c for c in new_contacts if 'table' not in (c['a'],c['b'])]
  allowed={i for c in changed for i in c['state_xy_indices']};actual=set(map(int,np.flatnonzero(custom!=original)))
  fixtures_unchanged=all(settled[n]==v for n,v in original_settled.items() if v['kind']=='fixture')
  height_error={n:abs(settled[n]['body_xyz_m'][2]-original_settled[n]['body_xyz_m'][2]) for n in xy}
  checks={'original_observation_matches_historical':original_obs_sha==expected_obs_sha,
   'only_selected_free_object_xy_changed':bool(actual) and actual<=allowed,
   'all_requested_objects_moved_at_least_3cm':all(c['xy_displacement_m']>=.03 for c in changed),
   'fixture_poses_and_articulated_states_preserved':fixtures_unchanged,
   'no_cross_object_or_robot_contacts_except_table':not bad_contacts,
   'moved_object_support_heights_stable_2mm':all(v<.002 for v in height_error.values()),
   'initial_goal_false':not bool(w.env.check_success()),'official_goal_bddl_unchanged':sha(out/'task.bddl')==sha(bddl)}
  if args.task_id==4:
   checks['left_right_plates_order_preserved']=bool((original_settled['plate_1']['body_xyz_m'][1]-original_settled['plate_2']['body_xyz_m'][1])*(settled['plate_1']['body_xyz_m'][1]-settled['plate_2']['body_xyz_m'][1])>0)
  manifest={'task_id':args.task_id,'suite':'libero_10','task_prompt':task.language,'layout_kind':'New free-object XY relayout; no added objects',
   'simulator_seed':7,'episode_index':0,'settle_steps':10,'settle_action':[0,0,0,0,0,0,-1],
   'simulator_config':asdict(w.config),'original_bddl':str(bddl),'bddl_sha256':sha(bddl),
   'original_init_source':str(source),'original_init_source_sha256':sha(source),
   'initial_state_sha256':sha(out/'initial_state.npy'),'settled_state_sha256':sha(out/'settled_state.npy'),
   'original_observation_sha256':original_obs_sha,'expected_original_observation_sha256':expected_obs_sha,
   'relayout_observation_sha256':relayout_obs_sha,'changes':changed,
   'original_settled':original_settled,'relayout_settled':settled,'height_errors_m':height_error,
   'original_contacts':original_contacts,'contacts_after_settling':new_contacts,'non_table_contacts':bad_contacts,
   'checks':checks,'passed':all(checks.values()),'initial_raw_success':bool(w.env.check_success()),
   'model_calls':0,'policy_actions':0,'preparation_settling_actions':20}
  write(out/'layout_manifest.json',manifest)
  panel(out,args.task_id,task.language,manifest['passed'])
  print(json.dumps({'task_id':args.task_id,'passed':manifest['passed'],'checks':checks,'changes':changed,'non_table_contacts':bad_contacts,'output_dir':str(out)},indent=2,default=native),flush=True)
  assert manifest['passed'],'Scene preparation checks failed; preserve this candidate and revise separately'
 finally:w.close()

if __name__=='__main__':main()
