"""Offline source-bound analysis of actual open-preview receipts; no native calls."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

BASE = Path('/workspace/gpt_field/robodojo_uc71')
sys.path[:0] = [str(BASE/'scripts'), str(BASE/'src')]
from analyze_joint_probe import URDF, THRESHOLDS, analyze_state, pose_matrix
from robodojo_uc71.controller import orientation_error_world

V = BASE/'validation/new_host_20261029'
OUT = Path(__file__).resolve().parent
ATTEMPT = V/'native_open_clearance_preview_001'
CHILD = ATTEMPT/'worker/controller'
sources = {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('source_changed:'+str(path))
    sources[str(path)] = digest
    return path


def read(path, expected=None):
    return json.loads(bind(path, expected).read_text())


def main():
    if (OUT/'analysis.json').exists(): raise ValueError('fresh_output_required')
    for path in (Path(__file__), BASE/'scripts/analyze_joint_probe.py', BASE/'src/robodojo_uc71/controller.py'):
        bind(path)
    launch = read(ATTEMPT/'launch_result.json')
    for key in ('clearance_preview_completed','clean_shutdown_confirmed','gpu_memory_released',
                'gpu_processes_released','open_only_actions_verified','stable_open_endpoints_verified'):
        if launch.get(key) is not True: raise ValueError('incomplete_preview')
    if (launch['returncode'] != 0 or launch['termination_signals'] or launch['remaining_owned_pids']
            or launch['diagnostic_completed'] is not False or launch['controller_characterization_completed'] is not False
            or launch['full_test_authorized'] is not False or launch['model_calls'] != 0):
        raise ValueError('preview_scope_or_release_differs')
    result = read(CHILD/'clearance_preview_result.json', launch['clearance_preview_result_sha256'])
    ready = read(CHILD/'clearance_preview_preclose_ready.json')
    if ready['clearance_preview_result_sha256'] != launch['clearance_preview_result_sha256']:
        raise ValueError('preclose_binding_differs')
    preparation = read(CHILD/'preparation.json', result['preparation_sha256'])
    plan = read(ATTEMPT/'plan.json', result['plan_file_sha256'])
    if result['diagnostic_native_actions'] != 260 or result['diagnostic_physics_substeps'] != 2600:
        raise ValueError('exact260_preview_required')
    for path, digest in preparation['source_file_sha256'].items():
        bind(ATTEMPT/'scripts_snapshot'/Path(path).name, digest)
    geometry = read(plan['geometry_candidate_path'])
    for row in geometry['source_files']: bind(row['path'], row['sha256'])
    urdf = URDF(next(row['path'] for row in geometry['source_files'] if Path(row['path']).suffix == '.urdf'))
    checks, receipts, captures, endpoints = [], [], [], []
    for index, row in enumerate(result['actions'], 1):
        receipt = read(CHILD/row['receipt']['path'], row['receipt']['sha256'])
        if (row['index'] != index or receipt['native_action_index'] != index
                or any(receipt['target'][arm+'_ee_joint_state'] != [1.] for arm in ('left','right'))
                or receipt['before']['native_action_counter'] != [index-1]
                or receipt['after']['native_action_counter'] != [index]):
            raise ValueError('actual_action_chain_or_open_command_differs')
        for phase in ('before','after'):
            checks.extend(analyze_state(receipt[phase], geometry, urdf, f'action_{index:03d}_{phase}'))
        receipts.append(receipt)
    if len(receipts) != 260: raise ValueError('incomplete_receipts')
    for capture in result['captures']:
        state = read(CHILD/capture['robot_measurement']['path'], capture['robot_measurement']['sha256'])
        camera = read(CHILD/capture['camera_inventory']['path'], capture['camera_inventory']['sha256'])
        if (capture['physics_unchanged_during_capture'] is not True
                or state['native_physics_time_s'] != camera['native_physics_time_s']
                or capture['native_physics_time_s'] != state['native_physics_time_s']
                or capture['native_physics_counter'] != state['native_physics_counter']
                or capture['native_action_counter'] != state['native_action_counter']):
            raise ValueError('capture_not_synchronized')
        for image in capture['images']: bind(CHILD/image['path'], image['sha256'])
        checks.extend(analyze_state(state, geometry, urdf, capture['label']))
        captures.append(state)
    if len(captures) != 9: raise ValueError('nine_captures_required')
    maximum_joint_excursion = 0.
    for entry, stage, capture in zip(result['stages'], plan['stages'], captures[1:]):
        last = entry['last_action_index']; tail = receipts[last-5:last]
        if capture['arms'] != tail[-1]['after']['arms']: raise ValueError('endpoint_capture_changed')
        positions, angles, fingers = [], [], []
        for receipt in tail:
            state = receipt['after']; measured = state['arms'][stage['active_arm']]
            transform = pose_matrix(measured['link6_world_pose_wxyz'])@np.asarray(geometry['link6_from_tool'])
            target = stage['target_tool_pose_wxyz']
            positions.append(float(np.linalg.norm(transform[:3,3]-target[:3])))
            angles.append(float(np.linalg.norm(orientation_error_world(measured['link6_world_pose_wxyz'][3:],target[3:]))))
            fingers.extend(float(np.max(np.abs(np.asarray(state['arms'][arm]['finger_positions'])-.044))) for arm in ('left','right'))
        endpoint = {'stage_index':stage['index'],'arm':stage['active_arm'],'last_native_action':last,
            'five_sample_max_position_error_m':max(positions),'five_sample_max_orientation_error_rad':max(angles),
            'five_sample_max_open_finger_error_m':max(fingers),
            'passed':max(positions)<=.002 and max(angles)<=.02 and max(fingers)<=.001}
        endpoints.append(endpoint)
        report = read(CHILD/entry['analysis']['path'],entry['analysis']['sha256'])['endpoint_validation']
        for key,value in (('maximum_position_error_m',max(positions)),('maximum_orientation_error_rad',max(angles)),('maximum_open_finger_error_m',max(fingers))):
            if not np.isclose(report[key],value,atol=1e-12,rtol=1e-10): raise ValueError('endpoint_report_changed')
    for receipt in receipts:
        for arm in ('left','right'):
            q = np.asarray(receipt['after']['arms'][arm]['joint_positions'])
            maximum_joint_excursion=max(maximum_joint_excursion,float(np.max(abs(q-plan['home'][arm]['joint_positions']))))
    keys=[k for k in checks[0] if k not in ('state','arm','native_physics_time_s','measured_arm_joint_positions_rad','measured_finger_positions_m','passed')]
    maxima={k:max(row[k] for row in checks) for k in keys}
    passed=all(row['passed'] for row in checks) and all(row['passed'] for row in endpoints)
    report={'schema':'robodojo-native-open-preview-geometry-analysis-1','geometry_and_endpoints_passed':passed,
        'source_launch_result':{'path':str(ATTEMPT/'launch_result.json'),'sha256':sources[str(ATTEMPT/'launch_result.json')]},
        'native_actions_analyzed':260,'state_records_including_receipt_duplicates':len(checks)//2,'arm_pose_checks':len(checks),
        'thresholds':THRESHOLDS,'thresholds_unchanged_from_predeclared37_action_analysis':True,
        'maximum_errors':maxima,'actual_maximum_joint_home_excursion_rad':maximum_joint_excursion,
        'actual_preview_joint_bound_rad':plan['candidate_abort_envelopes']['joint_distance_from_home_rad'],
        'endpoint_checks':endpoints,'pose_checks':checks,'all260_actual_commands_both_grippers_open':True,
        'source_files':[{'path':p,'sha256':sha} for p,sha in sorted(sources.items())],
        'native_clearance_approved_by_this_analysis':False,'gripper_timing_validated':False,
        'full_controller_characterization_completed':False,'runtime_bundle_issued':False,'full_test_authorized':False,
        'standard_ready':False,'simulator_starts':0,'native_actions_executed_by_analysis':0,'model_calls':0,
        'limitations':['Measured geometry and open endpoints only; no new task or scene-object state is consumed.',
            'This preview contains no closed-gripper timing and cannot replace the full39-stage960-action characterization.',
            'Fresh visual closing clearance remains a separate operator judgment.']}
    path=OUT/'analysis.json'
    with path.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    path.chmod(0o444)
    print(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'passed':passed,
        'arm_pose_checks':len(checks),'maximum_errors':maxima,'max_joint_excursion_rad':maximum_joint_excursion,
        'endpoint_checks':endpoints},indent=2))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
