"""Finalize hypothetical geometry evidence and a zero-response negative control."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

OUT=Path(__file__).resolve().parent
V=OUT.parent
BASE=V.parents[1]
bound={}


def binding(path,expected=None):
    path=Path(path).resolve(strict=True);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected is None or digest==expected
    assert str(path) not in bound or bound[str(path)]==digest
    bound[str(path)]=digest
    return {'path':str(path),'sha256':digest}


def read(path,expected=None):
    row=binding(path,expected);return json.loads(Path(row['path']).read_text())


grid_path=V/'controller_clearance_wider_feasibility_001/grid.json'
rank_path=V/'controller_clearance_wider_ranking_001/result/ranking.json'
projection_path=V/'controller_clearance_wider_projection_001/projection.json'
grid=read(grid_path,'e6d3149c837bc9988a14212cd17ac4f848cd089ad314f98309cc5f785d7a1911')
rank=read(rank_path,'0ad8dec48501e2287769958dd01ea30cad4b03157f4bc18d057b883e8041417e')
projection=read(projection_path)
current=read(V/'controller_clearance_ranking_001/grid004/ranking.json',
             '63586042051a877d022b2f654e08a6215b745c18719670a11bef0b641bf1b84b')
for document,key in ((grid,'source_files'),(rank,'source_files'),(projection,'source_inputs')):
    for row in document[key]:binding(row['path'],row['sha256'])
for row in projection['artifacts']+[projection['analysis_program']]:binding(row['path'],row['sha256'])
geometry=read(V/'tool_geometry_001/candidate.json')
plan=read(V/'controller_characterization_plan_004/proposal.json')
for row in grid['imported_source_snapshots']:
    binding(row['archived_path'],row['sha256'])
    binding(row['original_path'],row['sha256'])
assert grid['hypothetical_analysis_only'] and rank['hypothetical_analysis_only'] and projection['hypothetical_analysis_only']
parameters={'lift_m':.03,'retreat_m':.04,'outward_m':.025,'outward_yaw_rad':.08}
selected=next(row for row in rank['ranked_candidates'] if row['parameters']==parameters)
assert selected['nominal_reserve_filter'] and not selected['production_035_would_accept_both_nominal_paths']
negative=[]
for arm in ('left','right'):
    row=next(row for row in grid['results'] if row['arm']==arm and all(row[k]==v for k,v in parameters.items()))
    actual=grid['actual008_entry_states'][arm]['arms'][arm]
    pose=np.asarray(actual['link6_world_pose_wxyz'])
    offset=np.asarray(geometry['link6_from_tool'])[:3,3]
    rotation=Rotation.from_quat(pose[3:],scalar_first=True)
    stalled_tcp=pose[:3]+rotation.apply(offset)
    target=np.asarray(row['target_tool_pose_wxyz'])
    position_error=float(np.linalg.norm(stalled_tcp-target[:3]))
    orientation_error=float((Rotation.from_quat(target[3:],scalar_first=True)*rotation.inv()).magnitude())
    q=np.asarray(actual['joint_positions']);home=np.asarray(plan['home'][arm]['joint_positions'])
    joint_excursion=float(np.max(abs(q-home)))
    assert joint_excursion<.35 and position_error>.001 and orientation_error>.01
    negative.append({'arm':arm,'assumed_response_gain':0.,
        'assumption':'All commanded changes have zero response; recorded entry joints and link pose remain fixed.',
        'predicted_fixed_tcp_world_m':stalled_tcp.tolist(),'desired_clearance_tool_pose_wxyz':target.tolist(),
        'home_joint_excursion_rad':joint_excursion,'within_current035_joint_guard':True,
        'clearance_position_residual_m':position_error,'clearance_orientation_residual_rad':orientation_error,
        'offline_reference_position_tolerance_m':.001,'offline_reference_orientation_tolerance_rad':.01,
        'satisfies_offline_clearance_pose_settling':False,
        'native_schedule_replayed':False,'native_actions_executed':0})
binding(__file__)
record={'schema':'robodojo-hypothetical-clearance-analysis-handoff-1',
    'scope':'source_bound_offline_geometry_and_nominal_kinematics_only',
    'candidate_parameters':parameters,'candidate_rank_record':selected,
    'current035_guard_best_worst_arm_gain_px':current['ranked_candidates'][0]['worst_arm_directional_gain_px'],
    'hypothetical050_selected_worst_arm_gain_px':selected['worst_arm_directional_gain_px'],
    'production_joint_home_ceiling_rad':.35,'hypothetical_joint_home_ceiling_rad':.50,
    'actual008_nominal_response_assumptions':grid['actual008_response'],
    'figures':projection['artifacts'],'zero_response_negative_control':negative,
    'exact130_action_preview_replay':'separate independent evidence required; not performed or claimed here',
    'selection_status':'root_selected_A_for_disabled_proposal_only_pending_independent_replay_and_review',
    'limitations':['Image-direction gains and visual-mesh sweeps do not certify native collision clearance.',
                   'Nominal reserve margins are calculated under a fitted scalar response assumption, not a physical error bound.',
                   'The full180-update grid schedule differs from the proposed130-action preview schedule.',
                   'The stalled control demonstrates that joint-envelope compliance alone does not establish arrival.'],
    'source_files':[{'path':p,'sha256':h} for p,h in sorted(bound.items())],
    'simulator_starts':0,'model_calls':0,'native_actions':0,'production_source_or_limits_changed':False,
    'execution_authorized':False,'full_test_authorized':False,'clearance_approved':False,'standard_ready':False}
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bound.items())
path=OUT/'handoff.json'
assert not path.exists()
path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'verified_source_files':len(bound),'negative_control':negative},indent=2))
