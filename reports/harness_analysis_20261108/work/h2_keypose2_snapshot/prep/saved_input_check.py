"""One targeted check on saved observations. No model, simulator or rollout."""
import ast
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
from gpt_field.foundation_harness.surface import planning_surface
from gpt_field.robodojo_direct_sim import PandaKinematics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'runs/foundation_harness_t04_can_obstacle_001'
V=ROOT/'trial/foundation_harness/vision'
# Reuse just the already recorded reconstruction helper, with CURRENT classes.
# No top-level execution or frozen runtime imports from that analysis script.
helper=ROOT/'analysis/reconstruct_rejections.py'
tree=ast.parse(helper.read_text())
restore=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='restore')
exec(compile(ast.Module(body=[restore],type_ignores=[]),str(helper),'exec'))
records=[json.loads(l) for l in (V/'snapshots.jsonl').read_text().splitlines()]
events=[json.loads(l) for l in (ROOT/'trial/foundation_harness/events.jsonl').read_text().splitlines()]
robot=json.loads((ROOT/'robot_kinematics.json').read_text())
kin=PandaKinematics.from_xml(robot['robot_xml'],robot['gripper_xml'])
feedback=next(e for e in events if e['event']=='feedback' and e['step']==13)
record=next(r for r in records if r['snapshot']['evidence']['observation_index']==13)
scene,err=restore(record)
state=np.asarray(feedback['measured_state']);q=np.asarray(feedback['joint_position'])
original=next(e['operation'] for e in events if e['event']=='operation_interrupted' and e['step']==13)
old_root=ROOT.parent/'foundation_vision_t04_can_obstacle_20261105_001'
response=json.loads(sorted((old_root/'trial/requests').glob('*.response.json'))[0].read_text())
text=next(c['text'] for m in response['output'] for c in m.get('content',[]) if c.get('type')=='output_text')
segment=json.loads(text)['execution']['segments'][0]
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
xyzw=matrix_to_quaternion_xyzw(grip_pose(state)[:3,:3]);wxyz=[xyzw[3],*xyzw[:3]]
astra={**original,'displacement':None,'keyposes':[{'name':'pregrasp',
    'pose':{'position':segment['xyz_m'],'quaternion_wxyz':wxyz},
    'gripper':'keep','profile':'transit','observe_after':True}]}
report={'scope':'offline saved-input check only; no model, simulator or robot actions',
        'geometry_source':record['arrays'],'ownership_source_match_max_m':err,
        'saved_depth_precision':'float32; reconstruction is not bitwise replay', 'paths':[]}
for name,op in [('previously_rejected_small_viewpoint_move',original),('previous_success_astra_first_keypose',astra)]:
    planner=CandidatePlanner(kin);start=perf_counter()
    try:
        plan=planner.plan(op,scene,state,q)
        result={'case':name,'planned':True,'joint_path_points':len(plan.joint_paths[0]),'evidence':plan.evidence}
    except PlanningFailure as exc:
        result={'case':name,'planned':False,'reason':exc.reason,'evidence':exc.evidence}
    result['elapsed_s']=perf_counter()-start
    report['paths'].append(result)
    print(json.dumps(result),flush=True)
yellow=next(r for r in records if r['snapshot']['frame_id']==5)
cloud,_=restore(yellow);obj=cloud.objects['yellow_white_mug'];points,evidence=planning_surface(obj)
report['surface']={'object':'yellow_white_mug','raw_count':len(obj.points),'planning_count':len(points),
                   'raw_extent_m':np.ptp(obj.points,axis=0).tolist(),
                   'planning_extent_m':np.ptp(points,axis=0).tolist() if len(points) else None,
                   'evidence':evidence}
print(json.dumps(report['surface']),flush=True)
(HERE/'saved_input_check.json').write_text(json.dumps(report,indent=2)+'\n')
