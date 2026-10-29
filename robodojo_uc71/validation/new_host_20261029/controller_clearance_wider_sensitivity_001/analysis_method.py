"""Independent analysis-only 0.50-rad sensitivity on the exact 130-step open preview.

No simulator imports, native dispatch, policy calls, selection, or authorization.
The scalar gain cases are numerical assumptions, not validated dynamics bounds.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

BASE = Path('/workspace/gpt_field/robodojo_uc71')
sys.path[:0] = [str(BASE / 'src'), str(BASE / 'scripts')]
from analyze_joint_probe import URDF, pose_matrix
from robodojo_uc71.controller import (ArmState, ControllerConfig, joint_target,
    measured_tcp_pose, orientation_error_world)

VALIDATION = BASE / 'validation/new_host_20261029'
OUTPUT = Path(__file__).resolve().parent
CHILD = VALIDATION / 'native_controller_characterization_008/worker/controller'
CASES = ((.03, .04, .025, .08),)
HYPOTHETICAL_JOINT_BOUND_RAD = .50
sources = {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and expected != digest:
        raise ValueError('source_changed')
    sources[str(path)] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o444)


def main():
    if (OUTPUT / 'analysis.json').exists() or (OUTPUT / 'numerical_traces.json').exists():
        raise ValueError('fresh_output_required')
    bind(__file__)
    for path in (BASE / 'scripts/analyze_joint_probe.py', BASE / 'src/robodojo_uc71/controller.py'):
        bind(path)
    projection_path = VALIDATION / 'controller_clearance_wider_projection_001/projection.json'
    projection = read(projection_path)
    if projection['candidate_specifications']['A'] != dict(zip(('lift_m','retreat_m','outward_m','outward_yaw_rad'),CASES[0])):
        raise ValueError('projection_candidate_mismatch')
    for row in projection['source_inputs'] + projection['artifacts'] + [projection['analysis_program']]:
        bind(row['path'], row['sha256'])
    plan_path = VALIDATION / 'controller_characterization_plan_004/proposal.json'
    plan = read(plan_path)
    geometry = read(plan['geometry_candidate_path'])
    for row in geometry['source_files']:
        bind(row['path'], row['sha256'])
    urdf = URDF(next(row['path'] for row in geometry['source_files'] if Path(row['path']).suffix == '.urdf'))
    offset = np.asarray(geometry['link6_from_tool'])[:3, 3]
    fixed = read(plan['fixed_probe_plan_path'])
    inventory = read(fixed['initial_inventory_path'])
    if sources[str(Path(plan['fixed_probe_plan_path']).resolve())] != plan['fixed_probe_plan_file_sha256']:
        raise ValueError('fixed_plan_changed')
    entries = {arm: {'raw_robot_inventory': inventory} for arm in ('left','right')}
    measured = {}
    for arm, first, last in (('left', 626, 665), ('right', 806, 845)):
        gains, lags = [], []
        for index in range(first, last + 1):
            row = read(CHILD / f'native_action_{index:04d}.json')
            before = np.asarray(row['before']['arms'][arm]['joint_positions'])
            after = np.asarray(row['after']['arms'][arm]['joint_positions'])
            command = np.asarray(row['target'][arm + '_arm_joint_state'])
            delta = command - before
            lags.append(float(np.max(abs(command - after))))
            if np.linalg.norm(delta) > 1e-4:
                gains.append(float((after - before) @ delta / (delta @ delta)))
        measured[arm] = {'observed_actions': last-first+1, 'projected_response_samples': len(gains),
            'projected_response_gain_min_median_max': [min(gains), float(np.median(gains)), max(gains)],
            'maximum_command_to_next_state_residual_rad': max(lags)}
    roots = {arm: np.asarray(next(robot for robot in entry['raw_robot_inventory']['robots']
        if robot['arm_name'] == arm + '_arm')['root_link_pose_w']['values'][0][:3])
        for arm, entry in entries.items()}
    midpoint = (roots['left'] + roots['right']) / 2
    all_traces, summaries = [], []
    for arm in ('left', 'right'):
        robot = next(row for row in entries[arm]['raw_robot_inventory']['robots'] if row['arm_name'] == arm + '_arm')
        ids, names = robot['native_arm_joint_indices'], robot['joint_names']
        world_from_base = pose_matrix(robot['root_link_pose_w']['values'][0])
        original_q = np.asarray(robot['joint_positions']['values'][0])
        lower_upper = np.asarray(robot['joint_position_limits_lower_upper']['values'][0])[ids]
        velocity_limits = np.asarray(robot['joint_velocity_limits']['values'][0])[ids]
        home_q = np.asarray(plan['home'][arm]['joint_positions'])
        home_pose = np.asarray(plan['home'][arm]['tool_pose_wxyz'])
        home_rotation = Rotation.from_quat(home_pose[3:], scalar_first=True)
        initial_tool_x = home_rotation.as_matrix()[:, 0]
        outward = roots[arm] - midpoint
        outward[2] = 0.
        outward /= np.linalg.norm(outward)
        yaw_sign = 1. if np.dot(np.cross([0., 0., 1.], initial_tool_x), outward) > 0 else -1.

        def state(q, dq):
            full_q = original_q.copy()
            full_q[ids] = q
            links, frames = urdf.forward(dict(zip(names, full_q)), world_from_base)
            link = links['link6']
            return ArmState(q, dq, link[:3, 3], Rotation.from_matrix(link[:3, :3]).as_quat(scalar_first=True),
                urdf.jacobian(link[:3, 3], frames, [names[index] for index in ids]),
                lower_upper[:, 0], lower_upper[:, 1], velocity_limits, offset)

        for lift, retreat, lateral, yaw in CASES:
            raised = home_pose[:3] + [0., 0., lift]
            endpoint = raised - retreat * initial_tool_x + lateral * outward
            end_quaternion = (Rotation.from_rotvec([0., 0., yaw_sign*yaw]) * home_rotation).as_quat(scalar_first=True)
            sequence = (('raise', raised, home_pose[3:], 25, 'transit', .1),
                ('clearance', endpoint, end_quaternion, 40, 'contact', .2),
                ('return', raised, home_pose[3:], 40, 'contact', .2),
                ('lower', home_pose[:3], home_pose[3:], 25, 'transit', .1))
            for gain in (.5, measured[arm]['projected_response_gain_min_median_max'][1], 1., 1.25):
                q, dq = original_q[ids].copy(), np.zeros(6)
                if not np.array_equal(q, home_q):
                    raise ValueError('fixed_initial_joint_state_mismatch')
                trace, stage_reports, failure = [], [], None
                try:
                    for label, target, target_quaternion, count, profile, transit_speed in sequence:
                        config = ControllerConfig(transit_translation_speed_m_s=transit_speed,
                            contact_translation_speed_m_s=.05, angular_speed_rad_s=.5)
                        stage_rows = []
                        for index in range(count):
                            result = joint_target(state(q, dq), target, target_quaternion, profile=profile, config=config)
                            command = np.asarray(result.joint_positions)
                            following = q + gain * (command - q)
                            dq, q = (following-q)/.04, following
                            position, quaternion = measured_tcp_pose(state(q, dq))
                            record = {'stage': label, 'stage_update': index+1, 'q_rad': q.tolist(),
                                'command_q_rad': command.tolist(), 'kinematic_joint_rate_rad_s': dq.tolist(),
                                'tool_pose_wxyz': [*position, *quaternion], 'target_tool_pose_wxyz': [*target, *target_quaternion],
                                'position_error_m': float(np.linalg.norm(position-target)),
                                'orientation_error_to_target_rad': float(np.linalg.norm(orientation_error_world(quaternion, target_quaternion))),
                                'orientation_from_home_rad': float(np.linalg.norm(orientation_error_world(quaternion, home_pose[3:]))),
                                'max_joint_home_rad': float(np.max(abs(q-home_q))),
                                'max_command_home_rad': float(np.max(abs(command-home_q))),
                                'tool_home_distance_m': float(np.linalg.norm(position-home_pose[:3])),
                                'native_joint_limit_margin_rad': float(min(np.min(q-lower_upper[:, 0]), np.min(lower_upper[:, 1]-q)))}
                            trace.append(record); stage_rows.append(record)
                            if (record['max_joint_home_rad'] > HYPOTHETICAL_JOINT_BOUND_RAD or record['max_command_home_rad'] > HYPOTHETICAL_JOINT_BOUND_RAD
                                    or record['tool_home_distance_m'] > .065 or record['orientation_from_home_rad'] > .12
                                    or max(abs(dq)) > 2. or record['native_joint_limit_margin_rad'] < 0):
                                raise ValueError('hypothetical_robot_guard_exceeded')
                        first = next((i+1 for i in range(len(stage_rows)-4) if all(
                            row['position_error_m'] <= .002 and row['orientation_error_to_target_rad'] <= .02
                            for row in stage_rows[i:])), None)
                        stage_reports.append({'stage': label, 'updates': count, 'first_stable_suffix_update': first,
                            'stable_suffix_samples': None if first is None else count-first+1,
                            'samples_after_first_stable': None if first is None else count-first,
                            'last_five_pose_samples_passed': all(r['position_error_m'] <= .002 and r['orientation_error_to_target_rad'] <= .02 for r in stage_rows[-5:]),
                            'last_five_max_position_error_m': max(r['position_error_m'] for r in stage_rows[-5:]),
                            'last_five_max_orientation_error_rad': max(r['orientation_error_to_target_rad'] for r in stage_rows[-5:]),
                            'final_position_error_m': stage_rows[-1]['position_error_m'],
                            'final_orientation_error_rad': stage_rows[-1]['orientation_error_to_target_rad']})
                except Exception as error:
                    failure = type(error).__name__ + ':' + str(error)
                summary = {'arm': arm, 'lift_m': lift, 'retreat_m': retreat, 'outward_m': lateral,
                    'outward_yaw_rad': yaw, 'world_yaw_sign': yaw_sign, 'response_gain': gain,
                    'numerical_updates': len(trace), 'failure': failure, 'stages': stage_reports,
                    'hypothetical_joint_margin_rad': HYPOTHETICAL_JOINT_BOUND_RAD-max(max(row['max_joint_home_rad'], row['max_command_home_rad']) for row in trace),
                    'tool_margin_m': .065-max(row['tool_home_distance_m'] for row in trace),
                    'orientation_margin_rad': .12-max(row['orientation_from_home_rad'] for row in trace),
                    'kinematic_joint_rate_margin_rad_s': 2.-max(max(abs(x) for x in row['kinematic_joint_rate_rad_s']) for row in trace),
                    'native_joint_limit_margin_rad': min(row['native_joint_limit_margin_rad'] for row in trace),
                    'nominal_guard_checks_passed': failure is None and len(trace) == 130,
                    'all_four_endpoints_settled': len(stage_reports) == 4 and all(r['last_five_pose_samples_passed'] for r in stage_reports),
                    'first_unsettled_stage': next((r['stage'] for r in stage_reports if not r['last_five_pose_samples_passed']), None),
                    'numerical_schedule_steps': 130,
                    'production_joint_bound_would_pass': all(max(r['max_joint_home_rad'],r['max_command_home_rad']) <= .35 for r in trace),
                    'native_dynamic_validation_passed': False}
                summaries.append(summary)
                all_traces.append({'summary': summary, 'states_after_each_numerical_update': trace})
    trace_path = OUTPUT / 'numerical_traces.json'
    write(trace_path, all_traces); bind(trace_path)
    report = {'schema': 'robodojo-hypothetical-open-preview-sensitivity-1',
        'scope': 'offline_robot_kinematics_and_scalar_response_sensitivity_only',
        'source_projection': {'path': str(projection_path), 'sha256': sources[str(projection_path)]},
        'initial_inventory': {'path': str(Path(fixed['initial_inventory_path']).resolve()), 'sha256': sources[str(Path(fixed['initial_inventory_path']).resolve())]},
        'hypothetical_joint_home_ceiling_rad': HYPOTHETICAL_JOINT_BOUND_RAD,
        'production_joint_home_ceiling_rad': .35, 'production_source_or_limits_changed': False,
        'schedule': [{'stage': name, 'updates': count} for name,count in (('raise',25),('clearance',40),('return',40),('lower',25))],
        'endpoint_rule': {'last_samples':5,'position_tolerance_m':.002,'orientation_tolerance_rad':.02,'open_finger_tolerance_m':.001,'extra_settling_actions':0},
        'cases_lift_retreat_outward_yaw': CASES, 'actual008_response_measurements': measured,
        'response_model': 'q_next = q + gain*(unchanged joint_target command(q)-q); recompute source URDF FK/Jacobian each update',
        'production_guards': plan['candidate_abort_envelopes'],
        'hypothetical_guards': {**plan['candidate_abort_envelopes'], 'joint_distance_from_home_rad': HYPOTHETICAL_JOINT_BOUND_RAD},
        'summaries': summaries,
        'all_hypothetical_guard_checks_passed': all(row['nominal_guard_checks_passed'] for row in summaries),
        'all_gain_cases_meet_endpoint_checks': all(row['all_four_endpoints_settled'] for row in summaries),
        'fixed_open_finger_proxy': {r['arm_name']: {'positions_m': [r['joint_positions']['values'][0][r['joint_names'].index(n)] for n in ('joint7','joint8')], 'modeled_as_constant': True, 'within_declared_open_tolerance': all(abs(r['joint_positions']['values'][0][r['joint_names'].index(n)]-.044) <= .001 for n in ('joint7','joint8'))} for r in inventory['robots']},
        'physical_finger_dynamics_validated': False, 'inactive_arm_dynamics_validated': False,
        'counterfactual_continuation_after_unsettled_endpoint': True,
        'source_files': [{'path': path, 'sha256': digest} for path, digest in sorted(sources.items())],
        'packages': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'candidate_selected_for_execution': None, 'native_dynamic_validation_passed': False,
        'fresh_empty_sweep_clearance_passed': False, 'simulator_starts': 0, 'native_actions_executed': 0,
        'model_calls': 0, 'full_test_authorized': False, 'standard_ready': False,
        'limitations': ['The fitted008 response medians and selected gain cases are not error bounds for a new pose.',
            'The exact four-stage130-update path has no endpoint holds. Finger positions and inactive arms are held constant as numerical proxies, without dynamics validation.',
            'Numerical joint-rate proxies do not certify native continuous-time velocities.',
            'A real preview must stop at the first unsettled endpoint; continued numerical paths after such a failure are counterfactual only.',
            'The0.50-rad joint bound is hypothetical. Production remains0.35; all other numerical guards are unchanged.',
            'Fresh unambiguous three-camera clearance and actual native guard/accounting checks remain mandatory.']}
    write(OUTPUT / 'analysis.json', report)
    print(json.dumps({'path': str(OUTPUT / 'analysis.json'), 'sha256': hashlib.sha256((OUTPUT / 'analysis.json').read_bytes()).hexdigest(),
        'actual008_response': measured, 'cases': [{key: row[key] for key in row if key != 'stages'} for row in summaries]}, indent=2))


if __name__ == '__main__':
    main()
