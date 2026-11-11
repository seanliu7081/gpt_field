"""Read-only independent audit of six saved relayouts; no simulator/model calls."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path('/workspace/gpt_field')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gpt_field.agent_speedup_v2_user_check import initial_observation_sha256
from gpt_field.contact_skills import StepView
from gpt_field.observation import PolicyObservation

# Official BDDL object order, independently read before rendering. Standard
# Panda has seven arm and two finger qpos before these seven-qpos free joints.
OBJECTS = {
    3: ['akita_black_bowl_1', 'wine_bottle_1'],
    4: ['porcelain_mug_1', 'red_coffee_mug_1', 'white_yellow_mug_1', 'plate_1', 'plate_2'],
    5: ['black_book_1', 'white_yellow_mug_1'],
    7: ['alphabet_soup_1', 'cream_cheese_1', 'tomato_sauce_1', 'ketchup_1', 'basket_1'],
    8: ['moka_pot_1', 'moka_pot_2'],
    9: ['porcelain_mug_1', 'white_yellow_mug_1'],
}
EXPECTED_GOALS = {
    3: ['(Close white_cabinet_1_bottom_region)', '(In akita_black_bowl_1 white_cabinet_1_bottom_region)'],
    4: ['(On porcelain_mug_1 plate_1)', '(On white_yellow_mug_1 plate_2)'],
    5: ['(In black_book_1 desk_caddy_1_back_contain_region)'],
    7: ['(In alphabet_soup_1 basket_1_contain_region)', '(In cream_cheese_1 basket_1_contain_region)'],
    8: ['(On moka_pot_1 flat_stove_1_cook_region)', '(On moka_pot_2 flat_stove_1_cook_region)', '(Turnon flat_stove_1)'],
    9: ['(In white_yellow_mug_1 microwave_1_heating_region)', '(Close microwave_1)'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(task_id):
    out = HERE / f'task_{task_id:02d}'
    manifest = json.loads((out / 'layout_manifest.json').read_text())
    original = np.load(out / 'original_initial_state.npy', allow_pickle=False)
    custom = np.load(out / 'initial_state.npy', allow_pickle=False)
    settled = np.load(out / 'settled_state.npy', allow_pickle=False)
    official = np.asarray(torch.load(manifest['original_init_source'], map_location='cpu', weights_only=False)[0])
    requested = json.loads((out / 'requested_xy.json').read_text())
    before = manifest['original_settled']
    after = manifest['relayout_settled']
    checks = {}
    checks['official_init0_exact'] = np.array_equal(official, original)
    checks['official_init_source_hash'] = sha(manifest['original_init_source']) == manifest['original_init_source_sha256']
    checks['initial_state_hash'] = sha(out / 'initial_state.npy') == manifest['initial_state_sha256']
    checks['settled_state_hash'] = sha(out / 'settled_state.npy') == manifest['settled_state_sha256']
    checks['bddl_bytes_exact'] = (out / 'task.bddl').read_bytes() == Path(manifest['original_bddl']).read_bytes()
    checks['bddl_hash'] = sha(out / 'task.bddl') == manifest['bddl_sha256']
    goal = (out / 'task.bddl').read_text().split('(:goal', 1)[1]
    checks['official_goal_predicates_preserved'] = all(x in goal for x in EXPECTED_GOALS[task_id])
    checks['original_object_inventory_unchanged'] = set(before) == set(after)
    checks['only_official_free_objects_relocated'] = set(requested) <= set(OBJECTS[task_id])
    allowed = []
    for index, name in enumerate(OBJECTS[task_id]):
        start = 10 + 7 * index  # flattened time + 9 robot qpos
        if name in requested:
            allowed += [start, start + 1]
            checks[f'{name}_requested_xy_exact'] = np.array_equal(custom[start:start + 2], requested[name])
            checks[f'{name}_moved_at_least_3cm'] = float(np.linalg.norm(custom[start:start + 2] - original[start:start + 2])) >= .03
        qpos = np.asarray(after[name]['joints'][f'{name}_joint0'])
        checks[f'{name}_saved_settled_state_matches_snapshot'] = np.allclose(settled[start:start + 7], qpos, rtol=0, atol=1e-12)
        checks[f'{name}_support_height_preserved'] = abs(after[name]['body_xyz_m'][2] - before[name]['body_xyz_m'][2]) < .002
    actual = np.flatnonzero(custom != original)
    checks['only_independently_mapped_free_xy_entries_changed'] = bool(actual.size) and set(actual.tolist()) <= set(allowed)
    checks['fixture_pose_and_joints_exactly_preserved'] = all(after[n] == v for n, v in before.items() if v['kind'] == 'fixture')
    contacts = manifest['contacts_after_settling']
    checks['only_table_support_cross_contacts_recorded'] = all('table' in (c['a'], c['b']) for c in contacts)
    checks['every_free_object_has_table_support'] = all(any(n in (c['a'], c['b']) and 'table' in (c['a'], c['b']) for c in contacts) for n in OBJECTS[task_id])
    checks['raw_initial_success_recorded_false'] = manifest['initial_raw_success'] is False
    checks['builder_checks_all_pass'] = manifest['passed'] is True and all(manifest['checks'].values())
    checks['zero_model_and_policy_calls_recorded'] = manifest['model_calls'] == 0 and manifest['policy_actions'] == 0
    checks['reset_protocol_preserved'] = manifest['simulator_seed'] == 7 and manifest['episode_index'] == 0 and manifest['settle_steps'] == 10 and manifest['settle_action'] == [0, 0, 0, 0, 0, 0, -1]
    old_obs = json.loads((ROOT / f'reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning/libero_10_t{task_id}/initial_observation.json').read_text())
    original_obs = json.loads((out / 'original_observation.json').read_text())
    custom_obs = json.loads((out / 'relayout_observation.json').read_text())
    checks['original_observation_matches_historical'] = initial_observation_sha256(original_obs) == initial_observation_sha256(old_obs) == manifest['original_observation_sha256']
    checks['relayout_observation_hash'] = initial_observation_sha256(custom_obs) == manifest['relayout_observation_sha256']
    checks['original_and_relayout_observations_differ'] = manifest['original_observation_sha256'] != manifest['relayout_observation_sha256']
    for prefix, obs in [('original_', original_obs), ('relayout_', custom_obs)]:
        view = StepView(PolicyObservation.from_dict(obs))
        for camera in ['external', 'wrist']:
            pixels = np.asarray(Image.open(out / f'{prefix}{camera}.png').convert('RGB'))
            checks[f'{prefix}{camera}_exact_observation_pixels'] = np.array_equal(pixels, getattr(view, f'{camera}_rgb_upright'))
    comparison = np.asarray(Image.open(out / 'layout_comparison.png').convert('RGB'))
    checks['comparison_original_panel_exact'] = np.array_equal(comparison[122:634, 18:530], np.asarray(Image.open(out / 'original_external.png').convert('RGB')))
    checks['comparison_relayout_panel_exact'] = np.array_equal(comparison[122:634, 546:1058], np.asarray(Image.open(out / 'relayout_external.png').convert('RGB')))
    semantic_state = {}
    if task_id == 3:
        value = after['white_cabinet_1']['joints']['white_cabinet_1_bottom_level'][0]
        semantic_state['bottom_drawer_qpos'] = value
        checks['native_bottom_drawer_initially_open'] = value < -.14
    if task_id == 4:
        checks['left_plate_id_and_world_y_order_preserved'] = after['plate_1']['body_xyz_m'][1] < after['plate_2']['body_xyz_m'][1]
    if task_id == 8:
        value = after['flat_stove_1']['joints']['flat_stove_1_button'][0]
        semantic_state['stove_button_qpos'] = value
        checks['native_stove_initially_on'] = value >= .5
    if task_id == 9:
        joints = after['microwave_1']['joints']
        value = next(iter(joints.values()))[0]
        semantic_state['microwave_joint_qpos'] = value
        checks['native_microwave_initially_open'] = value < -1.3
    checks = {key: bool(value) for key, value in checks.items()}
    return {'task_id': task_id, 'passed': all(checks.values()), 'checks': checks, 'semantic_state': semantic_state, 'initial_state_sha256': manifest['initial_state_sha256'], 'manifest_sha256': sha(out / 'layout_manifest.json')}


if __name__ == '__main__':
    results = [audit(task_id) for task_id in OBJECTS]
    result = {'passed': all(r['passed'] for r in results), 'method': 'Independent reads of official init0, saved arrays, BDDL, observations, RGB images, and recorded simulator snapshots. No simulator or model calls.', 'checks_passed': sum(sum(r['checks'].values()) for r in results), 'checks_total': sum(len(r['checks']) for r in results), 'tasks': results, 'audit_source_sha256': sha(__file__)}
    (HERE / 'independent_audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passed': result['passed'], 'checks_passed': result['checks_passed'], 'checks_total': result['checks_total'], 'failed_checks': {r['task_id']: [k for k, v in r['checks'].items() if not v] for r in results}, 'semantic_states': {r['task_id']: r['semantic_state'] for r in results}}, indent=2))
    assert result['passed']
