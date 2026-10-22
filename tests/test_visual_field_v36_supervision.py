"""Fresh V36 original-only preparation; dummy/input tests never run a task."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import test_visual_field_v32_supervision as inherited

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT/'runs/visual_field_v36_zero_shot_001'
PRIOR = ROOT/'runs/visual_field_v35_zero_shot_001'


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.setattr(inherited, 'HELPERS', HELPERS)
    return inherited


def test_dummy_child_exits_once_without_restart(tmp_path, helpers):
    helpers.test_dummy_child_survives_launcher_exit_and_failed_exit_does_not_restart(tmp_path)


def test_unsealed_harness_cannot_launch(tmp_path, helpers):
    helpers.test_real_trial_helper_refuses_unsealed_plan_without_launching(tmp_path)


def test_changed_source_cannot_launch(tmp_path, helpers):
    helpers.test_changed_source_seal_is_refused_without_starting_any_process(tmp_path)


@pytest.mark.parametrize('changed', ['frozen_initial_observation', 'official_bddl',
                                    'official_initial_state', 'missing_seal_check'])
def test_input_binding_is_rechecked_before_launch(tmp_path, changed, helpers):
    helpers.test_sealed_input_is_rechecked_immediately_before_any_supervisor_launch(tmp_path, changed)


def test_complete_matching_seal_is_read_only(tmp_path, helpers):
    helpers.test_unchanged_complete_input_seal_verifies_without_launch(tmp_path)


def test_fresh_paths_preserve_original_input_and_experiment_scope():
    plan = json.loads((HELPERS/'plan.json').read_text())
    old = json.loads((PRIOR/'plan.json').read_text())
    assert len(plan['scenes']) == 1
    scene, previous = plan['scenes'][0], old['scenes'][0]
    for key in ('id', 'task', 'suite', 'task_id', 'seed', 'episode_index', 'task_prompt',
                'bddl', 'initial_state', 'official_inputs', 'expected_initial_sha256'):
        assert scene[key] == previous[key]
    assert Path(scene['expected_initial']).read_bytes() == Path(previous['expected_initial']).read_bytes()
    assert hashlib.sha256(Path(scene['expected_initial']).read_bytes()).hexdigest() == scene['expected_initial_sha256']
    observation = json.loads(Path(scene['expected_initial']).read_text())
    assert set(observation) == {'observation/image', 'observation/wrist_image',
        'observation/state', 'observation/joint_position', 'observation/joint_velocity', 'prompt'}
    assert Path(scene['expected_initial']) == HELPERS/'inputs/bowl_original/expected_initial.json'
    assert Path(scene['output']) == HELPERS/'trials/bowl_original_zero_hybrid_v36'
    args = plan['command_argv']
    assert args[:3] == ['env', '-u', 'OPENAI_API_KEY']
    assert args[args.index('--method-version')+1] == 'v36'
    assert args[args.index('--output')+1] == scene['output']
    assert args[args.index('--expected-initial')+1] == scene['expected_initial']
    assert '--demo' not in args and plan['shots'] == 0 and plan['controller'] == 'hybrid'
    assert plan['model'] == 'gpt-6-astra' and plan['effort'] == 'medium'
    assert plan['billing'] == 'chatgpt_subscription' and plan['api_key_fallback'] is False
    assert plan['max_episode_steps'] is plan['max_episode_wall_time_s'] is None
    assert plan['process_supervision']['autorestart'] is False
    assert plan['process_supervision']['startretries'] == 0
    assert plan['process_supervision']['queue'] is False
    assert plan['process_supervision']['tcp_listener'] is False
    assert plan['operator_stop']['request_file'] == str(Path(scene['output'])/'operator_stop.request.json')
    assert plan['prior_attempt'] == str(PRIOR/'analysis.json')
    assert plan['completion_condition'] == 'Actual LIBERO success with no forbidden contact'
    assert plan['completion_metric'] == 'control_elapsed_s minus timing_components.video_render_s'


def test_four_v36_modules_review_and_independent_ordinary_preview_contract():
    plan = json.loads((HELPERS/'plan.json').read_text())
    old = json.loads((PRIOR/'plan.json').read_text())
    assert set(plan['new_runtime_modules']) == {
        'visual_field_policy_v36.py', 'visual_field_subscription_v36.py',
        'visual_intent_watch_v36.py', 'visual_intent_replan_v36.py'}
    assert len(plan['new_runtime_modules']) == 4
    assert plan['inherited_v35_runtime_modules'] == old['new_runtime_modules']
    assert not plan['modified_shared_runtime_sources']
    assert plan['inherited_shared_preview_hook'] == old['modified_shared_runtime_sources']
    watch = plan['ongoing_objective_review_contract']
    assert watch['feedback_key'] == 'ongoing_objective_review'
    for key in ('acknowledged_noncontact_field_motion_only',
                'current_reference_identity_and_domain_required',
                'same_heading_prospective_validation_required',
                'sustained_measured_worsening_and_edge_proximity_required',
                'explicit_model_decision_required'):
        assert watch[key] is True
    for key in ('reference_reset_requested', 'automatic_motion_or_reverse',
                'physical_budget_refund', 'collision_or_task_success_inferred',
                'permanent_motion_lock'):
        assert watch[key] is False
    ordinary = plan['ordinary_replan_contract']
    assert ordinary['feedback_key'] == 'ordinary_replan_previews'
    assert ordinary['candidate_goal_distance_m'] == .020
    assert ordinary['motion_axes'] == 'xyz' and ordinary['field_proposal_id'] is None
    assert ordinary['full_route_and_first_command_checked_on_clones'] is True
    assert ordinary['actual_execution_rechecks_all_constraints'] is True
    assert ordinary['explicit_ordinary_field_and_same_active_id_replace_required'] is True
    assert ordinary['visual_improvement_or_clearance_certified'] is False
    assert ordinary['automatic_selection'] is ordinary['physical_budget_refund'] is False


def test_inherited_runtime_and_history_provenance_are_bound_to_prior_seal():
    plan = json.loads((HELPERS/'plan.json').read_text())
    contract = plan['inherited_source_contract']
    path = Path(contract['prior_seal_path'])
    assert path == PRIOR/'preflight.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == contract['prior_seal_sha256']
    assert contract['allowed_changed_inherited_paths'] == ['scripts/evaluate_visual_field_subscription.py']
    assert contract['all_other_prior_runtime_sources_must_match'] is True
    assert contract['historical_trial_and_helpers_immutable'] is True
    for source, digest in json.loads(path.read_text())['source_sha256'].items():
        if source not in contract['allowed_changed_inherited_paths']:
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest


def test_new_guidance_is_declared_and_wire_schema_stays_inherited():
    from gpt_field.visual_field_subscription_v36 import output_schema
    from gpt_field.visual_field_subscription_v35 import output_schema as previous_schema
    plan = json.loads((HELPERS/'plan.json').read_text())
    contract = plan['model_contract']
    assert contract['guidance_changed_for_v36'] and contract['wire_schema_inherited_from_v35']
    assert contract['no_demonstration'] and contract['medium_subscription_only']
    assert contract['task_prompt'] == plan['scenes'][0]['task_prompt']
    assert output_schema() == previous_schema()


def test_helper_paths_are_relative_and_prior_sealed_helpers_remain_exact():
    seal = json.loads((PRIOR/'preflight.json').read_text())
    assert len(list(HELPERS.glob('*.py'))) == 7
    for name in ('foreground_trial.py', 'launch.py', 'private_supervisor.py'):
        assert (HELPERS/name).read_bytes() == (PRIOR/name).read_bytes()
    for name, expected in seal['helper_sha256'].items():
        assert hashlib.sha256((PRIOR/name).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('v36_preflight', HELPERS/'preflight.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.DIRECTORY == HELPERS and module.ROOT == ROOT


def _text_audit():
    spec = importlib.util.spec_from_file_location('v36_current_text_audit', HELPERS/'audit_request_text.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_current_text_audit_declares_v36_guidance_and_preserves_unrelated_content():
    from gpt_field.visual_field_subscription_v36 import output_schema, OBJECTIVE_REVIEW_GUIDANCE
    payload = {'instructions': 'old', 'text': {'format': {'name': 'old', 'schema': {}}},
        'input': [{'role': 'assistant', 'content': [{'type': 'output_text', 'text': '{"unchanged":true}'}]},
                  {'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,offline'}]}]}
    result = _text_audit().transform(payload)
    assert result['instructions'].endswith(OBJECTIVE_REVIEW_GUIDANCE)
    assert result['text']['format']['schema'] == output_schema()
    assert result['text']['format']['name'] == 'visual_field_rgb_joints_v36'
    assert result['input'] == payload['input'] and payload['instructions'] == 'old'


def test_actual_archived_v35_packet_keeps_all_current_facts_and_messages_exact():
    from gpt_field.astra import _canonical
    from gpt_field.visual_feedback_compact_v35 import expand_current_feedback
    module = _text_audit()
    path = next((PRIOR/'trials/bowl_original_zero_hybrid_v35/api').glob('000002-*.request.json'))
    raw = path.read_bytes(); payload = json.loads(raw)['payload']; before = copy.deepcopy(payload)
    packed = module.transform(payload)
    _, _, source = module.current_feedback(payload)
    _, _, compact = module.current_feedback(packed)
    assert 'presentation_v35' in source
    assert compact == source
    assert _canonical(expand_current_feedback(compact)) == _canonical(expand_current_feedback(source))
    assert packed['input'] == payload['input']
    assert payload == before and path.read_bytes() == raw


def test_terminal_input_audit_expands_named_tables_before_checking_forbidden_keys(monkeypatch):
    from gpt_field.visual_feedback_compact_v35 import VERSION
    monkeypatch.syspath_prepend(str(HELPERS))
    spec = importlib.util.spec_from_file_location('v36_terminal_audit', HELPERS/'terminal_audit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    packed = {'presentation_v35': VERSION, 'semantics': 'test', 'shared_facts': {
        'table': {'$table_v35': {'columns': ['object_poses', 'uncertainty'], 'rows': [[{'hidden': 1}, .1]]}}},
        'current': {'objects': {'$same_v35': 'table'}}}
    assert '/objects/0/object_poses' in module.forbidden_keys(packed)
