"""Explicit offline robot-only pose grid; no simulator or model invocation."""
from pathlib import Path
import sys,json,hashlib,shutil,itertools
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path('/workspace/gpt_field/robodojo_uc71')
SNAP=Path(__file__).resolve().parent/'source_snapshot'
sys.path.insert(0,str(SNAP))
from analyze_joint_probe import URDF,pose_matrix
from controller import ArmState,ControllerConfig,joint_target,measured_tcp_pose,orientation_error_world
VAL=BASE/'validation/new_host_20261029'; CHILD=VAL/'native_controller_characterization_008/worker/controller'
OUT=VAL/'controller_clearance_wider_feasibility_001'; assert not OUT.exists()
sources={}
def read(path):
    path=Path(path).resolve(strict=True);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
scope=read(Path(__file__).parent/'scope.json')
assert scope['joint_home_excursion_ceiling_rad']==.50
plan=read(VAL/'controller_characterization_plan_004/proposal.json');geometry=read(plan['geometry_candidate_path'])
for item in geometry['source_files']:
    p=Path(item['path']);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==item['sha256'];sources[str(p)]=h
for p in [SNAP/'controller.py',SNAP/'analyze_joint_probe.py',SNAP/'grid004_analysis_method.py']:
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
urdf=URDF(next(x['path'] for x in geometry['source_files'] if Path(x['path']).suffix=='.urdf'))
tool=np.asarray(geometry['link6_from_tool']);offset=tool[:3,3]
entries={a:read(CHILD/f'native_action_{i:04d}.json')['before'] for a,i in [('left',601),('right',781)]}
observed_response={}
for arm,first,last in [('left',626,665),('right',806,845)]:
    gains=[];lags=[]
    for index in range(first,last+1):
        row=read(CHILD/f'native_action_{index:04d}.json');pre=np.asarray(row['before']['arms'][arm]['joint_positions']);post=np.asarray(row['after']['arms'][arm]['joint_positions']);target=np.asarray(row['target'][arm+'_arm_joint_state']);delta=target-pre
        lags.append(float(np.max(abs(target-post))))
        if np.linalg.norm(delta)>1e-4:gains.append(float((post-pre)@delta/(delta@delta)))
    observed_response[arm]={'median_gain':float(np.median(gains)),'maximum_command_next_state_residual_rad':max(lags)}
root_positions={a:np.asarray(next(r for r in entries[a]['raw_robot_inventory']['robots'] if r['arm_name']==a+'_arm')['root_link_pose_w']['values'][0][:3]) for a in entries}
midpoint=(root_positions['left']+root_positions['right'])/2
for row in scope['imported_source_snapshots']:
    assert hashlib.sha256(Path(row['archived_path']).read_bytes()).hexdigest()==row['sha256']
assert Path(sys.modules[URDF.__module__].__file__).resolve()==SNAP/'analyze_joint_probe.py'
assert Path(sys.modules[ControllerConfig.__module__].__file__).resolve()==SNAP/'controller.py'
grid=list(itertools.product(*(scope['grid_values'][name] for name in ('lift_m','retreat_m','outward_m','outward_yaw_rad'))))
results=[]
for arm in ['left','right']:
    robot=next(r for r in entries[arm]['raw_robot_inventory']['robots'] if r['arm_name']==arm+'_arm')
    ids=robot['native_arm_joint_indices'];names=robot['joint_names'];root=pose_matrix(robot['root_link_pose_w']['values'][0]);full_q=np.asarray(robot['joint_positions']['values'][0])
    q_entry=full_q[ids].copy();home=np.asarray(plan['home'][arm]['joint_positions']);p=np.asarray(plan['home'][arm]['tool_pose_wxyz'][:3]);quat=np.asarray(plan['home'][arm]['tool_pose_wxyz'][3:]);home_rotation=Rotation.from_quat(quat,scalar_first=True)
    tool_x=home_rotation.as_matrix()[:,0];outward=root_positions[arm]-midpoint;outward[2]=0;outward/=np.linalg.norm(outward)
    # World-Z sign rotates the initial tool-X direction toward the base-derived outward direction.
    yaw_sign=1. if np.dot(np.cross([0.,0.,1.],tool_x),outward)>0 else -1.
    limits=np.asarray(robot['joint_position_limits_lower_upper']['values'][0])[ids];velocity=np.asarray(robot['joint_velocity_limits']['values'][0])[ids]
    gain=observed_response[arm]['median_gain']
    def state(q,dq):
        q_all=full_q.copy();q_all[ids]=q;links,frames=urdf.forward(dict(zip(names,q_all)),root);m=links['link6']
        return ArmState(q,dq,m[:3,3],Rotation.from_matrix(m[:3,:3]).as_quat(scalar_first=True),urdf.jacobian(m[:3,3],frames,[names[i] for i in ids]),limits[:,0],limits[:,1],velocity,offset)
    for lift,retreat,lateral,yaw in grid:
        raised=p+[0,0,lift];endpoint=raised-retreat*tool_x+lateral*outward
        endquat=(Rotation.from_rotvec([0,0,yaw_sign*yaw])*home_rotation).as_quat(scalar_first=True)
        q=q_entry.copy();dq=np.zeros(6);maxq=maxcommand=maxdistance=maxrotation=maxrate=maxlead=0.;min_native_position_margin=100.;min_native_velocity_margin=100.;stages=[];failure=None;end_q=None;count=0
        sequence=[('raise',raised,quat,25,'transit',.1),('clearance',endpoint,endquat,40,'contact',.05),('hold',endpoint,endquat,50,'transit',.2),('return',raised,quat,40,'contact',.05),('lower',p,quat,25,'transit',.1)]
        try:
            for label,target,target_quat,n,profile,speed in sequence:
                errors=[]
                config=ControllerConfig(transit_translation_speed_m_s=speed if profile=='transit' else .2,contact_translation_speed_m_s=.05,angular_speed_rad_s=.5)
                for i in range(n):
                    assert config.joint_target_lead_rad==.05
                    command=np.asarray(joint_target(state(q,dq),target,target_quat,profile=profile,config=config).joint_positions)
                    maxlead=max(maxlead,float(np.max(abs(command-q))))
                    next_q=q+gain*(command-q);dq=(next_q-q)/.04;q=next_q;position,orientation=measured_tcp_pose(state(q,dq));count+=1
                    maxq=max(maxq,float(np.max(abs(q-home))));maxcommand=max(maxcommand,float(np.max(abs(command-home))));maxdistance=max(maxdistance,float(np.linalg.norm(position-p)));maxrotation=max(maxrotation,float(np.linalg.norm(orientation_error_world(orientation,quat))));maxrate=max(maxrate,float(np.max(abs(dq))))
                    min_native_position_margin=min(min_native_position_margin,float(np.min(np.r_[q-limits[:,0],limits[:,1]-q,command-limits[:,0],limits[:,1]-command])))
                    min_native_velocity_margin=min(min_native_velocity_margin,float(np.min(velocity-abs(dq))))
                    if maxq>.50 or maxcommand>.50 or maxdistance>.065 or maxrotation>.12 or maxrate>2 or maxlead>.05+1e-12 or min_native_position_margin<0 or min_native_velocity_margin<0:
                        raise ValueError('nominal_declared_robot_envelope_crossed')
                    errors.append((float(np.linalg.norm(position-target)),float(np.linalg.norm(orientation_error_world(orientation,target_quat)))))
                first=next((i+1 for i in range(len(errors)-4) if all(a<=.001 and b<=.01 for a,b in errors[i:])),None)
                stages.append({'stage':label,'first_stable_suffix_sample':first,'samples_remaining_after_first':None if first is None else n-first,'final_position_error_m':errors[-1][0],'final_orientation_error_rad':errors[-1][1]})
                if label=='clearance':end_q=q.copy()
        except Exception as error:
            failure=type(error).__name__+':'+str(error)
        record={'arm':arm,'lift_m':lift,'retreat_m':retreat,'outward_m':lateral,'outward_yaw_rad':yaw,'world_yaw_sign':yaw_sign,'response_gain_sensitivity':gain,'numerical_updates':count,'failure':failure,'max_joint_home_rad':maxq,'max_command_home_rad':maxcommand,'max_tool_home_m':maxdistance,'max_orientation_home_rad':maxrotation,'max_kinematic_joint_rate_rad_s':maxrate,'joint_margin_rad':.50-max(maxq,maxcommand),'production_035_joint_margin_rad':.35-max(maxq,maxcommand),'violates_production_035_joint_envelope':bool(max(maxq,maxcommand)>.35),'minimum_joint_home_ceiling_required_by_nominal_path_rad':max(maxq,maxcommand),'max_command_lead_rad':maxlead,'command_lead_margin_rad':.05-maxlead,'minimum_native_position_margin_rad':min_native_position_margin,'minimum_native_velocity_margin_rad_s':min_native_velocity_margin,'velocity_ceiling_margin_rad_s':2-maxrate,'tool_margin_m':.065-maxdistance,'orientation_margin_rad':.12-maxrotation,'stages':stages,'outward_world_unit':outward.tolist(),'target_tool_pose_wxyz':[*endpoint,*endquat]}
        if end_q is not None:
            record['candidate_endpoint']={'q_rad':end_q.tolist(),'all_joint_names':names,'full_q_rad':[float(end_q[ids.index(i)]) if i in ids else float(full_q[i]) for i in range(len(names))],'world_from_base_link':root.tolist(),'target_tool_pose_wxyz':[*endpoint,*endquat]}
        results.append(record)
sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT.mkdir();shutil.copyfile(__file__,OUT/'analysis_method.py');sources[str(OUT/'analysis_method.py')]=hashlib.sha256((OUT/'analysis_method.py').read_bytes()).hexdigest()
report={'schema':'robodojo-clearance-lift-retreat-outward-yaw-grid-1','scope':'offline_robot_kinematics_only','native_actions_executed':0,'model_calls':0,'execution_authorized':False,'candidate_selected_for_execution':None,'full_test_authorized':False,'standard_ready':False,'actual008_response':observed_response,'grid_values':scope['grid_values'],'hypothetical_analysis_only':True,'analysis_scope':{'path':str(Path(__file__).parent/'scope.json'),'sha256':sources[str((Path(__file__).parent/'scope.json').resolve())]},'production_guards_unchanged':plan['candidate_abort_envelopes'],'analysis_only_guard_change':{'joint_distance_from_home_rad':{'production':.35,'hypothetical':.50}},'imported_source_snapshots':scope['imported_source_snapshots'],'outward_formula':'normalize(horizontal(recorded_robot_base_position - midpoint_of_two_robot_bases))','yaw_sign_formula':'sign(dot(cross(world_Z,initial_tool_X),outward_direction))','unchanged_guards':dict(plan['candidate_abort_envelopes'],joint_distance_from_home_rad=.50),'actual008_entry_states':entries,'results':results,'source_files':[{'path':p,'sha256':h} for p,h in sorted(sources.items())],'limitations':['No scene/object geometry or contact truth used.','Scalar response gain comes from previous near-home measurements and is not a validated model for these directions.','Numerical feasibility does not establish native dynamics or visual clearance.','The unchanged_guards field is retained for the existing rank-reader interface; its0.50rad entry is hypothetical only. Production remains0.35rad and each path reports violations separately.','No plan/source change or new execution is authorized by this grid.']}
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
(OUT/'grid.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('grid',OUT/'grid.json',hashlib.sha256((OUT/'grid.json').read_bytes()).hexdigest())
print('arm_cases',len(results),'completed_nominal_paths',sum(r['failure'] is None for r in results))
