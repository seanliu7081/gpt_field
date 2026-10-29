from pathlib import Path
import importlib.util,json,hashlib,sys
import numpy as np
BASE=Path('/workspace/gpt_field/robodojo_uc71');sys.path.insert(0,str(BASE/'src'))
from robodojo_uc71.controller import measured_tcp_pose,orientation_error_world
V=BASE/'validation/new_host_20261029';ATTEMPT=V/'native_open_clearance_preview_001';CHILD=ATTEMPT/'worker/controller';OUT=V/'open_clearance_preview_operator_review_001'
snapshot=ATTEMPT/'scripts_snapshot/controller_characterization.py'
spec=importlib.util.spec_from_file_location('robodojo_uc71._preview_review_frozen',snapshot);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
files={}
def read(path):
 data=path.read_bytes();files[str(path)]=hashlib.sha256(data).hexdigest();return json.loads(data)
def bind(path):files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
launch=read(ATTEMPT/'launch_result.json');result=read(CHILD/'clearance_preview_result.json');ready=read(CHILD/'clearance_preview_preclose_ready.json');prep=read(CHILD/'preparation.json');plan=read(Path(prep['plan_path']));geometry=read(Path(plan['geometry_candidate_path']))
assert launch['clearance_preview_completed'] is True and launch['controller_characterization_completed'] is False
assert launch['returncode']==0 and not launch['termination_signals'] and not launch['remaining_owned_pids']
assert launch['gpu_processes_released'] and launch['gpu_memory_released'] and result['diagnostic_native_actions']==260
metrics={arm:{'max_measured_joint_home_excursion_rad':0.,'max_commanded_joint_home_excursion_rad':0.,'max_tool_home_distance_m':0.,'max_tool_home_orientation_rad':0.,'max_measured_joint_velocity_rad_s':0.,'max_joint_command_lead_rad':0.,'max_inactive_latched_hold_drift_rad':0.,'min_native_joint_position_margin_rad':float('inf')} for arm in helper.ARMS}
holds={};receipts=[]
for row in result['actions']:
 path=CHILD/row['receipt']['path'];r=read(path);assert files[str(path)]==row['receipt']['sha256'];receipts.append(r)
 index=r['stage_index'];stage=plan['stages'][index]
 if index not in holds:holds[index]={arm:np.asarray(r['before']['arms'][arm]['joint_positions']) for arm in helper.ARMS if arm!=stage['active_arm']}
 before=helper.physical_arm_states(r['before'],geometry)
 for arm in helper.ARMS:
  assert r['target'][arm+'_ee_joint_state']==[1.]
  command=np.asarray(r['target'][arm+'_arm_joint_state']);home=np.asarray(plan['home'][arm]['joint_positions'])
  metrics[arm]['max_commanded_joint_home_excursion_rad']=max(metrics[arm]['max_commanded_joint_home_excursion_rad'],float(np.max(np.abs(command-home))))
  metrics[arm]['max_joint_command_lead_rad']=max(metrics[arm]['max_joint_command_lead_rad'],float(np.max(np.abs(command-before[arm].joint_positions))))
 for measurement in (r['before'],r['after']):
  states=helper.physical_arm_states(measurement,geometry)
  for arm,s in states.items():
   q=np.asarray(s.joint_positions);home=plan['home'][arm];p,quat=measured_tcp_pose(s);m=metrics[arm]
   for key,value in {'max_measured_joint_home_excursion_rad':np.max(np.abs(q-home['joint_positions'])),'max_tool_home_distance_m':np.linalg.norm(p-home['tool_pose_wxyz'][:3]),'max_tool_home_orientation_rad':np.linalg.norm(orientation_error_world(quat,home['tool_pose_wxyz'][3:])),'max_measured_joint_velocity_rad_s':np.max(np.abs(s.joint_velocities))}.items():m[key]=max(m[key],float(value))
   m['min_native_joint_position_margin_rad']=min(m['min_native_joint_position_margin_rad'],float(np.min(q-s.joint_lower)),float(np.min(s.joint_upper-q)))
   if arm in holds[index]:m['max_inactive_latched_hold_drift_rad']=max(m['max_inactive_latched_hold_drift_rad'],float(np.max(np.abs(q-holds[index][arm]))))
endpoints=[];images=[]
for stage in result['stages']:
 a=read(CHILD/stage['analysis']['path']);assert a['endpoint_validation']['passed'] is True
 endpoints.append({'stage_index':stage['stage_index'],'label':stage['label'],**a['endpoint_validation']})
 if stage['stage_index'] in (1,5):
  for row in stage['capture']['images']:
   path=CHILD/row['path'];bind(path);assert files[str(path)]==row['sha256'];images.append({'path':str(path),'sha256':files[str(path)]})
assert len(receipts)==260 and len(endpoints)==8 and len(images)==6
read(V/'root_open_clearance_visual_review_001/review.json');read(V/'native_open_clearance_preview_geometry_001/analysis.json');after=read(ATTEMPT/'gpu_after.json');read(ATTEMPT/'gpu_before.json')
for p in (snapshot,BASE/'src/robodojo_uc71/controller.py',BASE/'src/robodojo_uc71/native_characterization.py',Path(__file__)):bind(p)
for path,sha in launch['source_file_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
record={'schema':'robodojo-open-clearance-preview-operator-review-1','reviewer':'delegated root/operator:driver_requirements','source_files':[{'path':p,'sha256':h} for p,h in sorted(files.items())],'actual_native_actions':260,'physics_substeps':2600,'simulated_time_s':result['diagnostic_simulated_time_s'],'all_actual_commands_open':True,'all_eight_fixed_endpoint_checks_passed':True,'endpoint_checks':endpoints,'measured_sampled_limits':metrics,'fresh_endpoint_images':images,'all_six_images_hash_verified_and_viewed':True,'visual_assessment':{'left':'Head view shows tabletop separating the active fingers and swept closing span from the bread rack; near finger surfaces are unobstructed in the wrist view.','right':'Head view now shows a tabletop gap between the active inner fingertip/swept closing span and toaster silhouette. The right wrist view supports the separation but alone would not establish depth.','supports_candidate_for_new_characterization':True,'future_fresh_gates_required_before_any_close':True},'resources':{k:launch[k] for k in ['returncode','termination_signals','remaining_owned_pids','gpu_processes_released','gpu_memory_released','gpu_memory_used_delta_mib','sampled_peak_gpu_used_mib','sampled_minimum_free_mib','elapsed_s']},'source_programs_unchanged':True,'benchmark_attempt_started':False,'full_test_authorized':False,'model_calls':0,'controller_characterization_completed':False,'gripper_timing_measured':False,'calibration_passed':False,'standard_ready':False,'limitations':['Sampled robot states do not bound continuous within-action excursions.','All commands stayed open; this provides no native closing timing or grasp evidence.','Fresh synchronized three-view review remains mandatory before either close in any future characterization.','Visual review is not a geometric collision or loaded-contact certificate.']}
with (OUT/'review.json').open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'review_sha256':hashlib.sha256((OUT/'review.json').read_bytes()).hexdigest(),'metrics':metrics,'resources':record['resources'],'source_files':len(files)},indent=2))
