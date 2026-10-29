"""Independent offline sensitivity replay of three explicitly requested poses.

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
CASES = ((.02, .01, .03, .10), (.03, .01, .03, .10), (.02, .01, .025, .10))
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
    grid_path = VALIDATION / 'controller_clearance_feasibility_002/grid.json'
    grid = read(grid_path)
    for row in grid['source_files']:
        bind(row['path'], row['sha256'])
    # Explicitly retain the companion needed for the grid's historical comparison.
    bind(VALIDATION / 'controller_clearance_feasibility_001/archived_sources.json')
    plan_path = VALIDATION / 'controller_characterization_plan_004/proposal.json'
    plan = read(plan_path)
    geometry = read(plan['geometry_candidate_path'])
    for row in geometry['source_files']:
        bind(row['path'], row['sha256'])
    urdf = URDF(next(row['path'] for row in geometry['source_files'] if Path(row['path']).suffix == '.urdf'))
    offset = np.asarray(geometry['link6_from_tool'])[:3, 3]
    entries = {arm: read(CHILD / f'native_action_{index:04d}.json')['before']
        for arm, index in (('left', 601), ('right', 781))}
    if entries != grid['actual008_entry_states']:
        raise ValueError('grid_entry_state_mismatch')
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
            grid_row = next(row for row in grid['results'] if row['arm'] == arm and
                (row['lift_m'], row['retreat_m'], row['outward_m'], row['outward_yaw_rad']) == (lift, retreat, lateral, yaw))
            if not np.allclose(np.r_[endpoint, end_quaternion], grid_row['target_tool_pose_wxyz'], atol=1e-14, rtol=0):
                raise ValueError('candidate_target_mismatch')
            sequence = (('raise', raised, home_pose[3:], 25, 'transit', .1),
                ('clearance', endpoint, end_quaternion, 40, 'contact', .2),
                ('held_close_pose', endpoint, end_quaternion, 25, 'transit', .2),
                ('held_open_pose', endpoint, end_quaternion, 25, 'transit', .2),
                ('return', raised, home_pose[3:], 40, 'contact', .2),
                ('lower', home_pose[:3], home_pose[3:], 25, 'transit', .1))
            for gain in (.5, measured[arm]['projected_response_gain_min_median_max'][1], 1., 1.25):
                q, dq = original_q[ids].copy(), np.zeros(6)
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
                            if (record['max_joint_home_rad'] > .35 or record['max_command_home_rad'] > .35
                                    or record['tool_home_distance_m'] > .065 or record['orientation_from_home_rad'] > .12
                                    or max(abs(dq)) > 2. or record['native_joint_limit_margin_rad'] < 0):
                                raise ValueError('unchanged_nominal_robot_guard_exceeded')
                        first = next((i+1 for i in range(len(stage_rows)-4) if all(
                            row['position_error_m'] <= .001 and row['orientation_error_to_target_rad'] <= .01
                            for row in stage_rows[i:])), None)
                        stage_reports.append({'stage': label, 'updates': count, 'first_stable_suffix_update': first,
                            'stable_suffix_samples': None if first is None else count-first+1,
                            'samples_after_first_stable': None if first is None else count-first,
                            'final_position_error_m': stage_rows[-1]['position_error_m'],
                            'final_orientation_error_rad': stage_rows[-1]['orientation_error_to_target_rad']})
                except Exception as error:
                    failure = type(error).__name__ + ':' + str(error)
                summary = {'arm': arm, 'lift_m': lift, 'retreat_m': retreat, 'outward_m': lateral,
                    'outward_yaw_rad': yaw, 'world_yaw_sign': yaw_sign, 'response_gain': gain,
                    'numerical_updates': len(trace), 'failure': failure, 'stages': stage_reports,
                    'joint_margin_rad': .35-max(max(row['max_joint_home_rad'], row['max_command_home_rad']) for row in trace),
                    'tool_margin_m': .065-max(row['tool_home_distance_m'] for row in trace),
                    'orientation_margin_rad': .12-max(row['orientation_from_home_rad'] for row in trace),
                    'kinematic_joint_rate_margin_rad_s': 2.-max(max(abs(x) for x in row['kinematic_joint_rate_rad_s']) for row in trace),
                    'native_joint_limit_margin_rad': min(row['native_joint_limit_margin_rad'] for row in trace),
                    'nominal_checks_passed': failure is None and len(trace) == 180,
                    'native_dynamic_validation_passed': False}
                summaries.append(summary)
                all_traces.append({'summary': summary, 'states_after_each_numerical_update': trace})
    trace_path = OUTPUT / 'numerical_traces.json'
    write(trace_path, all_traces); bind(trace_path)
    report = {'schema': 'robodojo-clearance-independent-sensitivity-1',
        'scope': 'offline_robot_kinematics_and_scalar_response_sensitivity_only',
        'source_grid': {'path': str(grid_path), 'sha256': sources[str(grid_path)]},
        'cases_lift_retreat_outward_yaw': CASES, 'actual008_response_measurements': measured,
        'response_model': 'q_next = q + gain*(unchanged joint_target command(q)-q); recompute source URDF FK/Jacobian each update',
        'unchanged_guards': plan['candidate_abort_envelopes'], 'summaries': summaries,
        'all_nominal_checks_passed': all(row['nominal_checks_passed'] for row in summaries),
        'source_files': [{'path': path, 'sha256': digest} for path, digest in sorted(sources.items())],
        'packages': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'candidate_selected_for_execution': None, 'native_dynamic_validation_passed': False,
        'fresh_empty_sweep_clearance_passed': False, 'simulator_starts': 0, 'native_actions_executed': 0,
        'model_calls': 0, 'full_test_authorized': False, 'standard_ready': False,
        'limitations': ['The fitted008 response medians and selected gain cases are not error bounds for a new pose.',
            'The held close/open portions model arm kinematics only; finger and inactive-arm dynamics are not simulated.',
            'Numerical joint-rate proxies do not certify native continuous-time velocities.',
            'Fresh unambiguous three-camera clearance and actual native guard/accounting checks remain mandatory.']}
    write(OUTPUT / 'analysis.json', report)
    print(json.dumps({'path': str(OUTPUT / 'analysis.json'), 'sha256': hashlib.sha256((OUTPUT / 'analysis.json').read_bytes()).hexdigest(),
        'actual008_response': measured, 'cases': [{key: row[key] for key in row if key != 'stages'} for row in summaries]}, indent=2))


if __name__ == '__main__':
    main()
