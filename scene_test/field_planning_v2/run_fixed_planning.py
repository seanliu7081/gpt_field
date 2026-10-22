"""Drawer-only planning with explicitly separated environment and robot layers.

All old code and artifacts are read-only. Planning consumes predicted geometry,
native robot-only packages and declared sensor calibration. Evaluation opens the
LIBERO environment only after a complete, hash-sealed prediction run exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = HERE.parent / 'field_planning'
sys.path.insert(0, str(OLD))
import run_planning as previous
from joint_planner import JointPlanner, PlannerSettings
from predicted_fields import SNAPSHOTS
from robot_geometry import RobotGeometry

sha256 = previous.sha256
write_json = previous.write_json


def freeze_sources(output):
    """Include both new code and every reused numerical source in the seal."""
    sources = list(HERE.glob('*.py')) + list(OLD.glob('*.py'))
    sources += [HERE.parent / 'astra_scene' / 'field_csg.py',
                HERE.parent / 'rgb_scene' / 'field_rgb.py']
    records = {}
    for path in sorted(set(sources)):
        relative = path.relative_to(PROJECT)
        target = output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records[str(path)] = {'sha256': sha256(path),
                              'copy': str(target.relative_to(output)),
                              'copy_sha256': sha256(target)}
    return records


def field_check(robot, field, q, details=False):
    result = previous.check_robot_field(robot, field, q, details=details)
    if 'field_points' in result:
        result['field_support_layers'] = dict(getattr(field, 'last_diagnostics', {}))
        result['clearance_interpretation'] = (
            'Astra model zero-set bound; DepthPro axial ray evidence and SELF prior '
            'do not provide a Euclidean obstacle-clearance certificate')
    return result


def settings():
    return {
        'scene': 'drawer003_only', 'snapshots': list(SNAPSHOTS),
        'methods': ['astra', 'depthpro'], 'seeds': list(previous.SEEDS),
        'goals': [[name, list(delta)] for name, delta in previous.GOAL_OFFSETS],
        'planner': asdict(PlannerSettings()),
        'robot_surface_max_spacing_m': previous.SAMPLE_SPACING_M,
        'unknown_policy': 'any unknown or occupied original robot surface sample refuses state',
        'field_geometry': {
            'astra': 'environment-only analytic CSG from original model output; remove complete robot_* union components; retain original field bounds',
            'depthpro': 'original predicted depth and K; restore pre-surface ray support; robot self image and physical self occlusion handled explicitly',
        },
        'robot_geometry': 'all 13 original LIBERO native collision geoms; no replacement body proxies',
        'robot_camera_registration': 'known rig extrinsics plus declared rig intrinsics for pixel self mask only; predicted K and depth unchanged',
        'environment_true_geometry_inputs': [],
        'depthpro_known_initial_self_prior': {
            'scope': 'exact union of native robot collision hulls at the initial q; numerical tolerance only',
            'assumption': 'known initial robot does not interpenetrate external scene objects and rig registration is valid',
            'not_visual_evidence': True,
            'no_external_dilation': True,
            'occupied_conflicts': 'reported separately; never silently counted as camera observations',
        },
        'motion': 'static 7-DOF arm queries with original snapshot finger joints fixed',
        'comparison': 'same endpoints, seeds, original robot surfaces and planner settings as field_planning/planning_001',
        'limitations': [
            'Finite robot surface and joint-edge samples are not continuous or solid-volume collision certificates.',
            'Astra environment completion remains a model hypothesis; removing self geometry does not fix its scene reconstruction error.',
            'DepthPro single-view support cannot establish free space behind occluders or outside its view.',
            'Known initial SELF volume is a proprioceptive prior, not a reconstruction improvement.',
            'These movement queries do not execute a drawer manipulation task.',
        ],
    }


def plan(preparation, output):
    from planning_fields import load_planning_field
    started = time.perf_counter()
    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'diagnostics').mkdir()
    source_records = freeze_sources(output)
    config = settings()
    write_json(output / 'settings.json', config)
    data = {'status': 'running', 'settings': config, 'records': [],
            'start_diagnostics': [], 'cases': [], 'robot_samples': {}}
    consumed, provenance = {}, []
    for snapshot in SNAPSHOTS:
        package = preparation / snapshot
        consumed.update(previous.robot_package_inputs(package))
        robot = RobotGeometry(package, sample_spacing_m=previous.SAMPLE_SPACING_M)
        cases = previous.make_cases(robot, snapshot)
        data['cases'].extend(cases)
        data['robot_samples'][snapshot] = robot.sample_metadata
        for method in config['methods']:
            loading = time.perf_counter()
            field = load_planning_field(method, snapshot, robot=robot,
                                        project_root=PROJECT, output=output / 'derived' / snapshot / method)
            points, gids = robot.samples(robot.q_initial)
            query = field.query(points)
            layers = dict(getattr(field, 'last_diagnostics', {}))
            file = f'diagnostics/{snapshot}_{method}.npz'
            np.savez_compressed(output / file, points_camera_m=points, geom_ids=gids,
                                point_state=query.state, point_support=query.support,
                                clearance_lower_bound_m=query.conservative_clearance_m,
                                raw_phi_m=query.raw_phi_m,
                                source_code=query.source_code,
                                known_initial_self=query.known_initial_self,
                                raw_predicted_state=query.raw_predicted_state)
            diagnosis = {'snapshot': snapshot, 'method': method, 'file': file,
                         'sha256': sha256(output / file),
                         'setup_and_start_diagnostic_seconds': time.perf_counter() - loading,
                         **field_check(robot, field, robot.q_initial, details=True)}
            diagnosis['field_support_layers'] = layers
            data['start_diagnostics'].append(diagnosis)
            print(json.dumps({'start': snapshot, 'method': method,
                              'valid': diagnosis['valid'], 'field_points': diagnosis.get('field_points'),
                              'support_layers': layers}), flush=True)
            planner = JointPlanner(robot.joint_limits, lambda q: field_check(robot, field, q))
            for case in cases:
                for seed in previous.SEEDS:
                    answer = planner.plan(case['start_q'], case['goal_q'], seed)
                    data['records'].append(previous.annotate_record(robot, case, method, answer))
                    write_json(output / 'plans.json', data)
                    print(json.dumps({'snapshot': snapshot, 'method': method,
                                      'case': case['case_id'], 'seed': seed,
                                      'status': answer['status'], 'reason': answer['reason'],
                                      'seconds': round(answer['planning_seconds'], 3)}), flush=True)
            provenance.append(field.provenance(verify=True))
            print(json.dumps({'finished_snapshot': snapshot, 'method': method,
                              'statuses': dict(Counter(r['status'] for r in data['records']
                                                      if r['snapshot'] == snapshot and r['method'] == method))}), flush=True)
    data['status'] = 'complete'
    write_json(output / 'plans.json', data)
    write_json(output / 'field_provenance.json', provenance)
    for entry in provenance:
        consumed.update(entry['consumed_sha256'])
    inputs_ok = all(sha256(path) == digest for path, digest in consumed.items())
    sources_ok = previous.verify_sources(source_records)
    if not inputs_ok or not sources_ok:
        raise RuntimeError('planning input or source changed during run')
    manifest = {'status': 'complete', 'stage': 'corrected_predicted_field_planning',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'preparation': str(preparation), 'sources': source_records,
                'source_unchanged': sources_ok, 'consumed_sha256': consumed,
                'inputs_unchanged': inputs_ok, 'elapsed_s': time.perf_counter() - started,
                'records': len(data['records']), 'artifacts_sha256': previous.artifact_hashes(output)}
    write_json(output / 'manifest.json', manifest)
    return manifest


def evaluate(plans, output, references):
    # The complete scene is inaccessible to the callback above; only enter here
    # after the sealed prediction artifacts have been verified by the reused evaluator.
    from native_distance_evaluation import evaluate as evaluate_native
    return evaluate_native(plans, output, references)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--preparation', type=Path, default=OLD / 'preparation_001')
    p.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--plans', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--references', type=Path, default=HERE.parent / 'rgb_scene/references_004')
    args = parser.parse_args()
    if args.command == 'plan':
        plan(args.preparation, args.output)
    else:
        evaluate(args.plans, args.output, args.references)


if __name__ == '__main__':
    main()
