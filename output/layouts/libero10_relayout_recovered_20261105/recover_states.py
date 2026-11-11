"""Recover saved relayout initial states from preserved evidence; no simulation."""
from pathlib import Path
import hashlib,io,json,shutil
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
LIBERO=Path('/workspace/heading_flow_pub/third_party/LIBERO/libero/libero')
PRES=ROOT/'presentations/speedup_v2_user_check_v71_libero10_en'
PLAN=ROOT/'reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning/libero_10_t6_relayout'
evidence=json.loads((PRES/'evidence.json').read_text())

def digest(b):return hashlib.sha256(b).hexdigest()
def record(p):return {'path':str(p.resolve()),'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size}

tasks={
0:{'stem':'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket', 'xy':{'alphabet_soup_1':[-.1,-.2],'tomato_sauce_1':[-.16,-.02],'cream_cheese_1':[.13,-.25],'orange_juice_1':[-.04,-.33],'milk_1':[.15,-.1],'butter_1':[.08,.02],'ketchup_1':[-.27,-.17],'basket_1':[.04,.24]}},
1:{'stem':'LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket', 'xy':{'cream_cheese_1':[-.13,.02],'butter_1':[-.16,-.19],'alphabet_soup_1':[.12,-.26],'tomato_sauce_1':[-.02,-.08],'milk_1':[.14,-.08],'orange_juice_1':[-.05,-.33],'ketchup_1':[-.27,-.08],'basket_1':[.04,.22]}},
6:{'stem':'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate'}
}
order=['alphabet_soup_1','cream_cheese_1','tomato_sauce_1','ketchup_1','orange_juice_1','milk_1','butter_1','basket_1']
manifest={'kind':'Recovered exact historical LIBERO-10 relayout states','suite':'libero_10','seed':7,'episode_index':0,'scene_kind':'object-position relayout; no obstacle scene','new_simulator_actions':0,'new_model_calls':0,'reconstruction_script':str(Path(__file__).resolve()),'tasks':{}}
for task,info in tasks.items():
    directory=OUT/f'task_{task:02d}';directory.mkdir(exist_ok=True)
    init=LIBERO/'init_files/libero_10'/f'{info["stem"]}.pruned_init'
    bddl=LIBERO/'bddl_files/libero_10'/f'{info["stem"]}.bddl'
    original=np.asarray(torch.load(init,weights_only=False)[0]).copy()
    entry={'task_id':task,'official_init_source':record(init),'official_bddl_source':record(bddl),'official_init_index':0,'seed':7,'episode_index':0,'task_bddl_override':None}
    if task in (0,1):
        historical=evidence['tasks'][f't{task}']
        modified=original.copy()
        change_records=[]
        for i,obj in enumerate(order):
            start=10+7*i
            modified[start:start+2]=info['xy'][obj]
            change_records.append({'object':obj,'flat_state_xy_indices':[start,start+1],'original_xyz_m':original[start:start+3].tolist(),'relayout_xyz_m':modified[start:start+3].tolist(),'orientation_wxyz_preserved':original[start+3:start+7].tolist()})
        f=io.BytesIO();np.save(f,modified);content=f.getvalue()
        expected=historical['provenance']['initial_state.npy']['sha256']
        assert digest(content)==expected, (task,digest(content),expected)
        (directory/'initial_state.npy').write_bytes(content)
        entry.update({'task_prompt':historical['config']['task'],'recovery_method':'Official saved pruned_init[0] plus exact XY assignments printed in preserved comparison PNG; np.save bytes match historical SHA256 exactly. All other state entries preserved bit-for-bit.','relayout_xy_source':record(PRES/'assets'/f't{task}_layout_comparison.png'),'expected_state_sha256_source':record(PRES/'evidence.json'),'expected_initial_state_sha256':expected,'expected_initial_observation_sha256':historical['approval']['initial_observation_sha256'],'expected_initial_observation_hash_source_field':f'tasks.t{task}.approval.initial_observation_sha256','historical_initial_observation_json_file_sha256':historical['provenance']['initial_observation.json']['sha256'],'object_changes':change_records,'expected_external_image':record(PRES/'assets'/f't{task}_initial.png'),'expected_wrist_image':record(PRES/'assets'/f't{task}_initial_wrist.png')})
    else:
        review=json.loads((PLAN/'review.json').read_text())
        source=PLAN/'frozen_inputs/initial_state.npy'
        expected=review['preparation']['execution']['initial_layout']['sha256']
        assert digest(source.read_bytes())==expected
        shutil.copyfile(source,directory/'initial_state.npy')
        modified=np.load(source)
        entry.update({'task_prompt':review['task_prompt'],'recovery_method':'Byte-for-byte copy of retained frozen historical initial_state.npy; source hash matches preserved review metadata.','initial_state_source':record(source),'review_source':record(PLAN/'review.json'),'initial_observation_source':record(PLAN/'initial_observation.json'),'expected_initial_state_sha256':expected,'expected_initial_observation_sha256':review['initial_observation_sha256'],'expected_initial_observation_hash_source_field':'review.initial_observation_sha256','expected_external_image':record(ROOT/'output/visualizations/libero10-relayout-comparison-20261105/task_06/relayout_external.png')})
    assert original.shape==modified.shape
    changed=np.flatnonzero(original!=modified).tolist()
    np.save(directory/'official_initial_state.npy',original)
    entry.update({'initial_state':record(directory/'initial_state.npy'),'official_initial_state':record(directory/'official_initial_state.npy'),'state_shape':list(modified.shape),'state_dtype':str(modified.dtype),'changed_indices_from_official_init0':changed,'changed_entry_count':len(changed),'hash_verified_against_historical':True})
    if task in (0,1):
        xyindices=[i for c in entry['object_changes'] for i in c['flat_state_xy_indices']]
        assert sorted(xyindices)==changed
    (directory/'provenance.json').write_text(json.dumps(entry,indent=2)+'\n')
    manifest['tasks'][str(task)]=entry
(OUT/'recovery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for t,e in manifest['tasks'].items():
 print(f'T{int(t):02d}: {e["initial_state"]["path"]} sha256={e["initial_state"]["sha256"]} changed_indices={e["changed_indices_from_official_init0"]}')
