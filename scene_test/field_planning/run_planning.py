"""Saved-field planning and separately launched LIBERO truth evaluation.

`plan` reads only robot-only packages and saved predicted fields. `evaluate`
reads the completed, hash-verified plan and then opens full LIBERO scenes.
No simulator actions, inference, asset edits, or full-task execution occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

from joint_planner import JointPlanner, PlannerSettings, interpolate_edge
from predicted_fields import SNAPSHOTS, load_predicted_field
from robot_geometry import RobotGeometry

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEEDS = (17, 29)
SAMPLE_SPACING_M = .005
TRUTH_JOINT_STEP_RAD = .005
GOAL_OFFSETS = (
    ('base_positive', (.45,0,0,0,0,0,0)),
    ('base_negative', (-.45,0,0,0,0,0,0)),
    ('shoulder_positive', (0,.30,0,0,0,0,0)),
    ('shoulder_negative', (0,-.30,0,0,0,0,0)),
    ('elbow_positive', (0,0,0,.40,0,0,0)),
    ('elbow_negative', (0,0,0,-.40,0,0,0)),
    ('wrist_positive', (0,0,0,0,0,.40,0)),
    ('wrist_negative', (0,0,0,0,0,-.40,0)),
    ('combined_positive', (.35,-.20,.15,.30,0,.25,0)),
    ('combined_negative', (-.35,-.20,-.15,.30,0,-.25,0)),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def frozen_sources(output):
    folder=output/'sources';folder.mkdir()
    records={}
    for p in sorted(HERE.glob('*.py')):
        dest=folder/p.name;shutil.copyfile(p,dest)
        records[str(p)]={'sha256':sha256(p),'copy':str(dest.relative_to(output)), 'copy_sha256':sha256(dest)}
    return records


def verify_sources(records):
    return all(sha256(path)==row['sha256'] for path,row in records.items())


def artifact_hashes(folder):
    return {str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}


def robot_package_inputs(package):
    # Do not traverse export audit documents containing evaluator provenance.
    data=json.loads((package/'robot.json').read_text())
    paths=[package/'robot.json',package/'robot.xml']+[package/a['file'] for a in data['assets']]
    return {str(p):sha256(p) for p in paths}


def make_cases(robot, snapshot):
    return [{'snapshot':snapshot,'case_id':name,'start_q':robot.q_initial.tolist(),
             'goal_q':(robot.q_initial+np.asarray(delta)).tolist(),'offset_rad':list(delta)}
            for name,delta in GOAL_OFFSETS]


def check_robot_field(robot, adapter, q, details=False):
    own=robot.check(q)
    if not own['valid']:
        return {'valid':False,'reason':'self_collision' if own['self_collision'] else 'joint_limit',
                'robot_self_check':own}
    points,gids=robot.samples(q)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or gids.shape != (len(points),):
        raise ValueError('nonempty original robot geometry samples required')
    if hasattr(robot,'geom_ids') and set(map(int,gids)) != set(map(int,robot.geom_ids)):
        raise ValueError('samples must cover every original robot collision geom')
    query=adapter.query(points)
    if query.state.shape != (len(points),) or not np.isin(query.state,[-1,0,1]).all():
        raise ValueError('field adapter returned invalid state codes or sample count')
    counts=query.summary()
    occupied=counts['occupied']>0;unknown=counts['unknown']>0
    reason=('predicted_occupied_and_unknown' if occupied and unknown else
            'predicted_occupied' if occupied else 'unknown_space' if unknown else 'sampled_clear')
    free=query.state==1
    result={'valid':not occupied and not unknown,'reason':reason,'field_points':counts,
            'minimum_free_sample_clearance_lower_bound_m':float(query.conservative_clearance_m[free].min()) if free.any() else None,
            'collision_test':'finite samples of all original MuJoCo collision surfaces',
            'continuous_or_full_volume_certificate':False}
    if details:
        result['per_geom']=[{'geom_id':row['geom_id'],'geom_name':row['geom_name'],
                            'points':int(np.sum(gids==row['geom_id'])),
                            'occupied':int(np.sum((gids==row['geom_id'])&(query.state==-1))),
                            'unknown':int(np.sum((gids==row['geom_id'])&(query.state==0))),
                            'free':int(np.sum((gids==row['geom_id'])&(query.state==1)))}
                           for row in robot.sample_metadata]
    return result


def eef_path(robot, path):
    if not path:return []
    expanded=[]
    for index,(a,b) in enumerate(zip(path,path[1:])):
        segment=interpolate_edge(a,b,.025)
        expanded.extend(segment if index==0 else segment[1:])
    if not expanded:expanded=path
    return [robot.eef_pose(q)[:3,3].tolist() for q in expanded]


def annotate_record(robot, case, method, result):
    positions=eef_path(robot,result['path_q'])
    return {**case,**result,'method':method,
            'start_eef_world_m':robot.eef_pose(case['start_q'])[:3,3].tolist(),
            'goal_eef_world_m':robot.eef_pose(case['goal_q'])[:3,3].tolist(),
            'eef_path_world_m':positions,
            'eef_path_length_m':float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()) if len(positions)>1 else 0.}


def plan(preparation, output):
    started=time.perf_counter();preparation=Path(preparation).resolve();output=Path(output).resolve()
    output.mkdir(exist_ok=False);(output/'diagnostics').mkdir()
    sources=frozen_sources(output)
    settings={'scene':'drawer003_only','snapshots':list(SNAPSHOTS),'methods':['astra','depthpro'],
              'seeds':list(SEEDS),'goals':[[n,list(d)] for n,d in GOAL_OFFSETS],
              'planner':asdict(PlannerSettings()),'robot_surface_max_spacing_m':SAMPLE_SPACING_M,
              'unknown_policy':'any unknown/ambiguous robot surface sample refuses the state',
              'field_geometry':'unchanged saved 2mm scalar arrays and certificates; no robot carving or scene repair',
              'robot_geometry':'all 13 original LIBERO MuJoCo collision geoms, including native collision hulls and pad boxes',
              'robot_camera_registration':'known rig extrinsic calibration only; no environment GT into collision callback',
              'environment_true_geometry_inputs':[],'motion':'static 7-DOF arm paths; snapshot finger joints fixed',
              'limitations':['Finite robot-surface and joint-edge samples do not certify swept-volume clearance.',
                             'Obstacle wholly inside a sampled robot volume or between samples may be missed.',
                             'Astra completions are hypotheses; DepthPro support is not verified physical free space.',
                             'Robot appearances remain in raw reconstruction fields; no semantic self-removal.',
                             'These are joint-space movement queries, not drawer opening or manipulation success.']}
    write_json(output/'settings.json',settings)
    results={'status':'running','settings':settings,'records':[],'start_diagnostics':[],'cases':[],'robot_samples':{}}
    consumed={};provenance=[]
    for snapshot in SNAPSHOTS:
        package=preparation/snapshot
        consumed.update(robot_package_inputs(package))
        robot=RobotGeometry(package,sample_spacing_m=SAMPLE_SPACING_M)
        results['robot_samples'][snapshot]=robot.sample_metadata
        cases=make_cases(robot,snapshot);results['cases'].extend(cases)
        for method in settings['methods']:
            loading=time.perf_counter()
            adapter=load_predicted_field(method,snapshot,project_root=PROJECT)
            points,gids=robot.samples(robot.q_initial)
            query=adapter.query(points)
            file=f'diagnostics/{snapshot}_{method}.npz'
            np.savez_compressed(output/file,points_camera_m=points,geom_ids=gids,point_state=query.state,
                                point_support=query.support,clearance_lower_bound_m=query.conservative_clearance_m,
                                raw_phi_m=query.raw_phi_m)
            diagnostic={'snapshot':snapshot,'method':method,'file':file,'sha256':sha256(output/file),
                        'setup_and_start_diagnostic_seconds':time.perf_counter()-loading,
                        **check_robot_field(robot,adapter,robot.q_initial,details=True)}
            results['start_diagnostics'].append(diagnostic)
            planner=JointPlanner(robot.joint_limits,lambda q:check_robot_field(robot,adapter,q))
            for case in cases:
                for seed in SEEDS:
                    answer=planner.plan(case['start_q'],case['goal_q'],seed)
                    results['records'].append(annotate_record(robot,case,method,answer))
            provenance.append(adapter.provenance(verify=True))
            write_json(output/'plans.json',results)
            print(json.dumps({'planned_snapshot':snapshot,'method':method,
                              'statuses':dict(Counter(r['status'] for r in results['records'] if r['snapshot']==snapshot and r['method']==method))}),flush=True)
    results['status']='complete';write_json(output/'plans.json',results)
    write_json(output/'field_provenance.json',provenance)
    for p in provenance:consumed.update(p['consumed_sha256'])
    unchanged=all(sha256(p)==v for p,v in consumed.items())
    source_ok=verify_sources(sources)
    if not unchanged or not source_ok:raise RuntimeError('input/source changed during planning')
    manifest={'status':'complete','stage':'predicted_field_planning','created_utc':datetime.now(timezone.utc).isoformat(),
              'preparation':str(preparation),'sources':sources,'source_unchanged':source_ok,
              'consumed_sha256':consumed,'inputs_unchanged':unchanged,'elapsed_s':time.perf_counter()-started,
              'records':len(results['records']),'artifacts_sha256':artifact_hashes(output)}
    write_json(output/'manifest.json',manifest)
    return manifest


def summarize(records):
    out=[]
    for snapshot in SNAPSHOTS:
        for method in ('astra','depthpro','oracle'):
            group=[r for r in records if r['snapshot']==snapshot and r['method']==method]
            planned=[r for r in group if r['status']=='planned']
            truth_valid=[r for r in planned if r['truth']['valid_at_all_samples']]
            keys={(r['case_id'],r['seed']) for r in group if r['status']!='planned'}
            oracle={ (r['case_id'],r['seed']):r for r in records if r['snapshot']==snapshot and r['method']=='oracle'}
            if any(k not in oracle for k in keys):
                raise ValueError('missing paired oracle query')
            false_refusal=sum(oracle[k]['status']=='planned' and oracle[k]['truth']['valid_at_all_samples'] for k in keys)
            out.append({'snapshot':snapshot,'method':method,'queries':len(group),'statuses':dict(Counter(r['status'] for r in group)),
                        'returned_paths':len(planned),'truth_valid_returned_paths':len(truth_valid),
                        'truth_invalid_returned_paths':len(planned)-len(truth_valid),
                        'returned_path_truth_collision_rate':(len(planned)-len(truth_valid))/len(planned) if planned else None,
                        'refusals_with_oracle_valid_path':false_refusal if method!='oracle' else 0,
                        'mean_planning_seconds':float(np.mean([r['planning_seconds'] for r in group])) if group else None,
                        'mean_returned_eef_length_m':float(np.mean([r['eef_path_length_m'] for r in planned])) if planned else None})
    return out


def evaluate(plans, output, references):
    # Full-scene class is imported only after entering the evaluation command.
    from robot_geometry import GroundTruthEvaluator
    started=time.perf_counter();plans=Path(plans).resolve();output=Path(output).resolve();references=Path(references).resolve()
    manifest=json.loads((plans/'manifest.json').read_text())
    if manifest['status']!='complete' or not manifest['source_unchanged'] or not manifest['inputs_unchanged']:
        raise ValueError('requires complete intact plan')
    for rel,digest in manifest['artifacts_sha256'].items():
        if sha256(plans/rel)!=digest:raise ValueError('plan artifact changed')
    data=json.loads((plans/'plans.json').read_text())
    if data['status']!='complete':raise ValueError('incomplete planning records')
    output.mkdir(exist_ok=False);sources=frozen_sources(output)
    consumed={str(plans/'manifest.json'):sha256(plans/'manifest.json'),str(plans/'plans.json'):sha256(plans/'plans.json')}
    result={'status':'running','settings':{**data['settings'],'truth_max_joint_step_rad':TRUTH_JOINT_STEP_RAD,
                     'oracle_role':'reference planner with identical algorithm/endpoints/budget; runs only after prediction paths frozen',
                     'truth_geometry':'original LIBERO native MuJoCo collision geoms, not the visual reconstruction reference surfaces'},
            'plans_directory':str(plans),'records':[],'start_diagnostics':data['start_diagnostics'],'robot_samples':data['robot_samples']}
    for snapshot in SNAPSHOTS:
        source=references/'snapshots'/snapshot/'evaluator'
        for name in ('render_model.xml','camera_sample_state.npz','calibration.json'):
            consumed[str(source/name)]=sha256(source/name)
        package=Path(manifest['preparation'])/snapshot
        consumed.update(robot_package_inputs(package))
        robot=RobotGeometry(package,sample_spacing_m=None)
        truth=GroundTruthEvaluator(source)
        consumed.update(truth.provenance()['consumed_sha256'])
        if not np.allclose(robot.q_initial,truth.q_initial,rtol=0,atol=1e-12):raise ValueError('robot/evaluator state mismatch')
        cases=[c for c in data['cases'] if c['snapshot']==snapshot]
        endpoint_truth={}
        for case in cases:
            endpoint_truth[case['case_id']]={'start':truth.check(case['start_q']),'goal':truth.check(case['goal_q'])}
        for record in data['records']:
            if record['snapshot']!=snapshot:continue
            record={**record,'truth_endpoints':endpoint_truth[record['case_id']]}
            record['truth']=truth.validate_path(record['path_q'],max_joint_step_rad=TRUTH_JOINT_STEP_RAD) if record['status']=='planned' else None
            result['records'].append(record)
        oracle=JointPlanner(robot.joint_limits,lambda q:truth.check(q,distances=False))
        for case in cases:
            for seed in SEEDS:
                answer=oracle.plan(case['start_q'],case['goal_q'],seed)
                record=annotate_record(robot,case,'oracle',answer)
                record['truth_endpoints']=endpoint_truth[case['case_id']]
                record['truth']=truth.validate_path(record['path_q'],max_joint_step_rad=TRUTH_JOINT_STEP_RAD) if record['status']=='planned' else None
                result['records'].append(record)
        write_json(output/'metrics.json',result)
        print(json.dumps({'evaluated_snapshot':snapshot,'oracle_statuses':dict(Counter(r['status'] for r in result['records'] if r['snapshot']==snapshot and r['method']=='oracle'))}),flush=True)
    result['summary']=summarize(result['records']);result['status']='complete'
    write_json(output/'metrics.json',result)
    unchanged=all(sha256(p)==digest for p,digest in consumed.items())
    source_ok=verify_sources(sources)
    if not unchanged or not source_ok:raise RuntimeError('input/source changed during evaluation')
    end={'status':'complete','stage':'evaluation_and_oracle_control','sources':sources,'source_unchanged':source_ok,
         'consumed_sha256':consumed,'inputs_unchanged':unchanged,'elapsed_s':time.perf_counter()-started,
         'artifacts_sha256':artifact_hashes(output)}
    write_json(output/'manifest.json',end)
    return end


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--plans',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    e.add_argument('--references',type=Path,default=PROJECT/'scene_test/rgb_scene/references_004')
    args=parser.parse_args()
    if args.command=='plan':plan(args.preparation,args.output)
    else:evaluate(args.plans,args.output,args.references)


if __name__=='__main__':main()
