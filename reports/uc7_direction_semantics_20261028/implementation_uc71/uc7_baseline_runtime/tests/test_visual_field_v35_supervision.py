"""Fresh V35 original-only preparation; dummy/input tests never run a task."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import test_visual_field_v32_supervision as inherited

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT/'runs/visual_field_v35_zero_shot_001'
PRIOR = ROOT/'runs/visual_field_v34_zero_shot_002'


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
    assert Path(scene['expected_initial']) == HELPERS/'inputs/bowl_original/expected_initial.json'
    assert Path(scene['output']) == HELPERS/'trials/bowl_original_zero_hybrid_v35'
    args = plan['command_argv']
    assert args[:3] == ['env', '-u', 'OPENAI_API_KEY']
    assert args[args.index('--method-version')+1] == 'v35'
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


def test_v35_modules_and_shared_preview_hook_are_explicit():
    plan = json.loads((HELPERS/'plan.json').read_text())
    assert set(plan['new_runtime_modules']) == {
        'visual_field_policy_v35.py', 'visual_field_subscription_v35.py',
        'visual_field_proposals_v35.py', 'visual_translation_guard_v35.py',
        'visual_navigation_fields_v35.py', 'visual_wrist_goal_v35.py',
        'visual_feedback_compact_v35.py'}
    assert len(plan['new_runtime_modules']) == 7
    shared = plan['modified_shared_runtime_sources']
    assert set(shared) == {'visual_field_proposals_v31.py'}
    assert len(shared['visual_field_proposals_v31.py']['base_source_sha256_at_preparation']) == 64
    assert shared['visual_field_proposals_v31.py']['legacy_default_behavior_preserved']
    assert shared['visual_field_proposals_v31.py']['included_in_final_source_seal']
    contract = plan['wrist_navigation_contract']
    assert contract['prediction_before_training'] and contract['acknowledged_measured_motion_only']
    assert contract['full_route_and_first_command_admission_required']
    assert contract['explicit_model_choice_required'] and not contract['physical_budget_refund']
    assert not contract['contact_or_depth_or_success_certified']


def test_new_guidance_is_declared_and_wire_schema_stays_inherited():
    from gpt_field.visual_field_subscription_v35 import output_schema
    from gpt_field.visual_field_subscription_v34 import output_schema as previous_schema
    plan = json.loads((HELPERS/'plan.json').read_text())
    contract = plan['model_contract']
    assert contract['guidance_changed_for_v35'] and contract['wire_schema_inherited_from_v34']
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
    for path in HELPERS.glob('*.py'):
        assert str(PRIOR) not in path.read_text()
    spec = importlib.util.spec_from_file_location('v35_preflight', HELPERS/'preflight.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.DIRECTORY == HELPERS and module.ROOT == ROOT


def _text_audit():
    spec = importlib.util.spec_from_file_location('v35_current_text_audit', HELPERS/'audit_request_text.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_current_text_audit_declares_v35_guidance_and_preserves_unrelated_content():
    from gpt_field.visual_field_subscription_v35 import output_schema, WRIST_GUIDANCE
    from gpt_field.visual_feedback_compact_v35 import SEMANTICS
    payload = {'instructions': 'old', 'text': {'format': {'name': 'old', 'schema': {}}},
        'input': [{'role': 'assistant', 'content': [{'type': 'output_text', 'text': '{"unchanged":true}'}]},
                  {'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,offline'}]}]}
    result = _text_audit().transform(payload)
    assert WRIST_GUIDANCE in result['instructions'] and result['instructions'].endswith(SEMANTICS)
    assert result['text']['format']['schema'] == output_schema()
    assert result['text']['format']['name'] == 'visual_field_rgb_joints_v35'
    assert result['input'] == payload['input'] and payload['instructions'] == 'old'


def test_current_request_factoring_round_trips_without_rewriting_old_messages():
    from gpt_field.astra import _canonical
    from gpt_field.visual_feedback_compact_v35 import expand_current_feedback
    module = _text_audit()
    path = next((PRIOR/'trials/bowl_original_zero_hybrid_v34/api').glob('000010-*.request.json'))
    raw = path.read_bytes(); payload = json.loads(raw)['payload']; before = copy.deepcopy(payload)
    packed = module.transform(payload)
    current_part, _, source = module.current_feedback(payload)
    _, _, compact = module.current_feedback(packed)
    assert 'presentation_v35' in compact
    assert _canonical(expand_current_feedback(compact)) == _canonical(source)
    for old_message, new_message in zip(payload['input'], packed['input']):
        for old_part, new_part in zip(old_message['content'], new_message['content']):
            if old_part is not current_part:
                assert old_part == new_part
            else:
                old_record, new_record = json.loads(old_part['text']), json.loads(new_part['text'])
                new_record['observable_context']['visual_feedback'] = expand_current_feedback(compact)
                assert old_record == new_record
    assert payload == before and path.read_bytes() == raw


def test_terminal_input_audit_expands_named_tables_before_checking_forbidden_keys(monkeypatch):
    from gpt_field.visual_feedback_compact_v35 import VERSION
    monkeypatch.syspath_prepend(str(HELPERS))
    spec = importlib.util.spec_from_file_location('v35_terminal_audit', HELPERS/'terminal_audit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    packed = {'presentation_v35': VERSION, 'semantics': 'test', 'shared_facts': {
        'table': {'$table_v35': {'columns': ['object_poses', 'uncertainty'], 'rows': [[{'hidden': 1}, .1]]}}},
        'current': {'objects': {'$same_v35': 'table'}}}
    assert '/objects/0/object_poses' in module.forbidden_keys(packed)
