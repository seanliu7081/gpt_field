"""Offline robot-only clearance comparison; no native imports or dispatch."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path('/workspace/gpt_field/robodojo_uc71')
sys.path[:0] = [str(BASE/'src'), str(BASE/'scripts')]
from analyze_joint_probe import URDF, pose_matrix, analyze_state
from robodojo_uc71.controller import ArmState, ControllerConfig, joint_target, measured_tcp_pose, orientation_error_world

VALIDATION = BASE/'validation/new_host_20261029'
ATTEMPT = VALIDATION/'native_controller_characterization_007'
CHILD = ATTEMPT/'worker/controller'
OUT = VALIDATION/'controller_clearance_feasibility_001'
assert not OUT.exists()
sources = {}
def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected is None or digest == expected
    sources[str(path)] = digest
    return path
def read(path):
    return json.loads(bind(path).read_text())
plan_path = VALIDATION/'controller_characterization_plan_003/proposal.json'
plan = read(plan_path)
geometry = read(plan['geometry_candidate_path'])
for row in geometry['source_files']:
    bind(row['path'], row['sha256'])
urdf_path = next(x['path'] for x in geometry['source_files'] if Path(x['path']).suffix == '.urdf')
urdf = URDF(urdf_path)
offset = np.asarray(geometry['link6_from_tool'])[:3,3]
bind(BASE/'scripts/analyze_joint_probe.py')
bind(BASE/'src/robodojo_uc71/controller.py')
bind(BASE/'src/robodojo_uc71/controller_characterization.py')
bind(VALIDATION/'joint_probe_analysis_001/analysis.json')
bind(VALIDATION/'controller_characterization_partial_007/analysis.json')
bind(ATTEMPT/'launch_result.json')
before = read(CHILD/'before/robot_measurement.json')
base_positions = {r['arm_name'].split('_')[0]: np.asarray(r['root_link_pose_w']['values'][0][:3])
                  for r in before['raw_robot_inventory']['robots']}
midpoint = (base_positions['left']+base_positions['right'])/2
measurements = []
actual_retreat = {}
entry_states = {}
for arm, first, last, entry in [('left',626,665,601),('right',806,845,781)]:
    entry_states[arm] = read(CHILD/f'native_action_{entry:04d}.json')['before']
    rows = [read(CHILD/f'native_action_{i:04d}.json') for i in range(first,last+1)]
    residuals, gains, observed_q = [], [], []
    for row in rows:
        pre = np.asarray(row['before']['arms'][arm]['joint_positions'])
        post = np.asarray(row['after']['arms'][arm]['joint_positions'])
        target = np.asarray(row['target'][arm+'_arm_joint_state'])
        residuals.append(float(np.max(abs(target-post))))
        command = target-pre
        if np.linalg.norm(command) > 1e-4:
            gains.append(float((post-pre)@command/(command@command)))
        observed_q.append(post.tolist())
        for phase in ['before','after']:
            measurements.extend(analyze_state(row[phase], geometry, urdf, f'{arm}_{row["native_action_index"]}_{phase}'))
    actual_retreat[arm] = {'command_to_next_state_max_joint_residual_rad':max(residuals),
        'projected_response_gain_min_median_max':[min(gains),float(np.median(gains)),max(gains)],
        'measured_final_joint_positions_rad':observed_q[-1],
        'recorded_entry_state':entry_states[arm]}

traces, summaries = [], []
candidate_endpoints = {}
for arm in ['left','right']:
    robot = next(r for r in entry_states[arm]['raw_robot_inventory']['robots'] if r['arm_name']==arm+'_arm')
    ids, names = robot['native_arm_joint_indices'], robot['joint_names']
    root = pose_matrix(robot['root_link_pose_w']['values'][0])
    full_q = np.asarray(robot['joint_positions']['values'][0])
    q_entry = full_q[ids].copy()
    home = np.asarray(plan['home'][arm]['joint_positions'])
    limits = np.asarray(robot['joint_position_limits_lower_upper']['values'][0])[ids]
    velocity = np.asarray(robot['joint_velocity_limits']['values'][0])[ids]
    outward = base_positions[arm]-midpoint
    outward[2] = 0
    outward /= np.linalg.norm(outward)
    p = np.asarray(plan['home'][arm]['tool_pose_wxyz'][:3])
    quaternion = np.asarray(plan['home'][arm]['tool_pose_wxyz'][3:])
    tool_x = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()[:,0]
    raised = p+[0,0,.03]
    def state(q, dq):
        values=full_q.copy(); values[ids]=q
        links,frames=urdf.forward(dict(zip(names,values)),root)
        link=links['link6']; quat=Rotation.from_matrix(link[:3,:3]).as_quat(scalar_first=True)
        return ArmState(q,dq,link[:3,3],quat,urdf.jacobian(link[:3,3],frames,[names[i] for i in ids]),
                        limits[:,0],limits[:,1],velocity,offset)
    for outward_m in [0.,.01,.015,.02]:
        endpoint=raised-.025*tool_x+outward_m*outward
        for gain in [.5, actual_retreat[arm]['projected_response_gain_min_median_max'][1], 1., 1.25]:
            q=q_entry.copy();dq=np.zeros(6);trace=[];stages=[];failure=None
            sequence=[('raise',raised,25,'transit',.1),('retreat',endpoint,40,'contact',.05),
                ('held_close_pose',endpoint,25,'transit',.2),('held_open_pose',endpoint,25,'transit',.2),
                ('return',raised,40,'contact',.05),('lower',p,25,'transit',.1)]
            try:
                for label,target,count,profile,speed in sequence:
                    stage_rows=[]
                    config=ControllerConfig(transit_translation_speed_m_s=speed if profile=='transit' else .2,
                        contact_translation_speed_m_s=.05, angular_speed_rad_s=.5)
                    for i in range(count):
                        result=joint_target(state(q,dq),target,quaternion,profile=profile,config=config)
                        command=np.asarray(result.joint_positions)
                        next_q=q+gain*(command-q);dq=(next_q-q)/.04;q=next_q
                        position,quat=measured_tcp_pose(state(q,dq))
                        row={'stage':label,'stage_action':i+1,'q_rad':q.tolist(),'command_q_rad':command.tolist(),
                            'kinematic_joint_rate_rad_s':dq.tolist(),'tool_pose_wxyz':[*position,*quat],
                            'position_error_m':float(np.linalg.norm(position-target)),
                            'orientation_error_rad':float(np.linalg.norm(orientation_error_world(quat,quaternion))),
                            'max_joint_home_distance_rad':float(np.max(abs(q-home))),
                            'max_command_home_distance_rad':float(np.max(abs(command-home))),
                            'tool_home_distance_m':float(np.linalg.norm(position-p)),
                            'native_joint_limit_margin_rad':float(min(np.min(q-limits[:,0]),np.min(limits[:,1]-q)))}
                        trace.append(row);stage_rows.append(row)
                    first=next((i+1 for i in range(len(stage_rows)-4) if all(r['position_error_m']<=.001 and r['orientation_error_rad']<=.01 for r in stage_rows[i:])),None)
                    stages.append({'stage':label,'actions':count,'first_continuous_1mm_0_01rad_sample_with_at_least5_samples':first,
                        'samples_remaining_after_first':None if first is None else count-first,
                        'final_position_error_m':stage_rows[-1]['position_error_m'],
                        'final_orientation_error_rad':stage_rows[-1]['orientation_error_rad'],
                        'final_q_rad':stage_rows[-1]['q_rad']})
                    if label=='retreat' and gain==1.:
                        candidate_endpoints.setdefault(str(outward_m),{})[arm]={'q_rad':q.tolist(),
                            'all_joint_names':names,'full_q_rad':[float(q[ids.index(i)]) if i in ids else float(full_q[i]) for i in range(len(names))],
                            'world_from_base_link':root.tolist(),'outward_world_unit':outward.tolist(),
                            'target_tool_pose_wxyz':[*endpoint,*quaternion],'predicted_tool_pose_wxyz':stage_rows[-1]['tool_pose_wxyz']}
            except Exception as error:
                failure=type(error).__name__+':'+str(error)
            summary={'arm':arm,'outward_m':outward_m,'response_gain_sensitivity':gain,'actions_computed':len(trace),
                'failure':failure,'stages':stages,'max_joint_home_rad':max(r['max_joint_home_distance_rad'] for r in trace),
                'max_command_home_rad':max(r['max_command_home_distance_rad'] for r in trace),
                'max_tool_home_m':max(r['tool_home_distance_m'] for r in trace),
                'max_kinematic_joint_rate_rad_s':max(max(abs(x) for x in r['kinematic_joint_rate_rad_s']) for r in trace)}
            summary['joint_guard_margin_rad']=.35-summary['max_joint_home_rad']
            summary['tool_guard_margin_m']=.065-summary['max_tool_home_m']
            summary['nominal_guard_checks_passed']=failure is None and summary['joint_guard_margin_rad']>=0 and summary['tool_guard_margin_m']>=0
            summaries.append(summary);traces.append({'summary':summary,'states_after_each_numerical_update':trace})

metrics=['fk_translation_m','fk_rotation_rad','tool_fk_translation_m','jacobian_linear_max_abs',
         'jacobian_angular_max_abs','tool_jacobian_linear_max_abs']
report={'schema':'robodojo-controller-clearance-offline-comparison-1','scope':'offline_robot_kinematics_only',
    'executed_native_actions':0,'model_calls':0,'benchmark_attempt_started':False,'full_test_authorized':False,
    'execution_authorized':False,'standard_ready':False,'candidate_selected_for_execution':None,
    'source_attempt':str(ATTEMPT),'unchanged_plan':{'path':str(plan_path),'sha256':sources[str(plan_path)]},
    'outward_formula':'normalize(horizontal(root_arm_position - (root_left_position+root_right_position)/2))',
    'base_positions_world_m':{k:v.tolist() for k,v in base_positions.items()},
    'lift_m':.03,'negative_initial_toolX_retreat_m':.025,'explicit_comparison_outward_m':[0,.01,.015,.02],
    'unchanged_guards':plan['candidate_abort_envelopes'],'actual007_retreat_measurements':actual_retreat,
    'actual007_geometry_checks':{'arm_state_checks':len(measurements),'all_passed':all(x['passed'] for x in measurements),
        'max_errors':{k:max(x[k] for x in measurements) for k in metrics}},
    'response_model':'q_next = q + gain*(unchanged_controller_command(q)-q); URDF FK/Jacobian recomputed each update',
    'response_sensitivities_note':'0.5 and1.25 are explicit numerical sensitivity cases, not measured native-response bounds. Per-arm0.8768 values are the007 retreat projected-response medians, not identified dynamics.',
    'summaries':summaries,'candidate_endpoints':candidate_endpoints,
    'limitations':['No native simulation or new motion was run.','Numerical joint-rate proxies do not certify measured velocity or continuous motion.',
        'Prior near-home tracking residuals do not bound the new lateral direction.',
        'Held close/open poses predict arm kinematics only; no finger dynamics or closing-sweep clearance is inferred.',
        'Fresh three-camera operator clearance is mandatory before any future close.',
        'No task object pose, contact truth, reward, or object geometry entered target construction.']}
OUT.mkdir();shutil.copyfile(__file__,OUT/'analysis_method.py');bind(OUT/'analysis_method.py')
(OUT/'numerical_traces.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n');bind(OUT/'numerical_traces.json')
(OUT/'actual_geometry_checks.json').write_text(json.dumps(measurements,indent=2,allow_nan=False)+'\n');bind(OUT/'actual_geometry_checks.json')
report['source_files']=[{'path':p,'sha256':h} for p,h in sorted(sources.items())]
(OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(OUT/'comparison.json'),'sha256':hashlib.sha256((OUT/'comparison.json').read_bytes()).hexdigest(),
    'geometry_checks':report['actual007_geometry_checks'],'summaries':[{k:v for k,v in r.items() if k!='stages'} for r in summaries]},indent=2))
