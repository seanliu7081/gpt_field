"""Fresh retry002 provenance and private-supervisor boundaries, no task run."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import test_visual_field_v32_supervision as inherited

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT/'runs/visual_field_v34_zero_shot_002'
PRIOR = ROOT/'runs/visual_field_v34_zero_shot_001'


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
    assert Path(scene['output']) == HELPERS/'trials/bowl_original_zero_hybrid_v34'
    args = plan['command_argv']
    assert args[:3] == ['env', '-u', 'OPENAI_API_KEY']
    assert args[args.index('--method-version')+1] == 'v34'
    assert args[args.index('--output')+1] == scene['output']
    assert args[args.index('--expected-initial')+1] == scene['expected_initial']
    assert '--demo' not in args
    assert plan['shots'] == 0 and plan['controller'] == 'hybrid'
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


def test_post_terminal_preview_modules_and_contract_are_declared():
    plan = json.loads((HELPERS/'plan.json').read_text())
    assert set(plan['new_runtime_modules']) == {
        'visual_field_policy_v34.py', 'visual_field_subscription_v34.py',
        'visual_field_proposals_v34.py', 'visual_reference_review_v34.py',
        'visual_feedback_transaction_v34.py', 'visual_field_v34.py',
        'visual_translation_guard_v34.py'}
    assert len(plan['new_runtime_modules']) == 7
    contract = plan['prospective_field_role_contract']
    assert set(contract['applies_to']) == {'continuous_reverse_preview', 'ordinary_axis_recovery_preview'}
    assert contract['temporary_context_only'] and contract['restored_on_return_or_exception']
    assert contract['explicit_model_selection_required'] and contract['fresh_actual_selection_checks_required']
    assert not contract['physical_debt_refund']


def test_model_prompt_schema_sources_match_trial001_frozen_bytes():
    plan = json.loads((HELPERS/'plan.json').read_text())
    contract = plan['unchanged_prompt_schema_contract']
    frozen = PRIOR/'trials/bowl_original_zero_hybrid_v34/frozen_source'
    assert contract['reference_trial'] == str(PRIOR)
    assert contract['guidance_or_wire_changes_permitted'] is False
    assert set(contract['source_sha256']) == {p.name for p in frozen.glob('visual_field_subscription*.py')}
    for name, expected in contract['source_sha256'].items():
        assert hashlib.sha256((ROOT/'src/gpt_field'/name).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((frozen/name).read_bytes()).hexdigest() == expected


def test_helper_paths_are_relative_and_prior_sealed_helpers_remain_exact():
    seal = json.loads((PRIOR/'preflight.json').read_text())
    assert len(list(HELPERS.glob('*.py'))) == 7
    for name in ('foreground_trial.py', 'launch.py', 'monitor.py', 'private_supervisor.py',
                 'terminal_audit.py', 'audit_request_text.py'):
        assert (HELPERS/name).read_bytes() == (PRIOR/name).read_bytes()
    for name, expected in seal['helper_sha256'].items():
        assert hashlib.sha256((PRIOR/name).read_bytes()).hexdigest() == expected
    for p in HELPERS.glob('*.py'):
        assert str(PRIOR) not in p.read_text()
    spec = importlib.util.spec_from_file_location('retry002_preflight', HELPERS/'preflight.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.DIRECTORY == HELPERS and module.ROOT == ROOT


def test_current_text_audit_retains_same_v34_guidance_wire_and_history():
    from gpt_field.visual_field_subscription_v34 import output_schema, TASK_RESUMPTION_GUIDANCE
    spec = importlib.util.spec_from_file_location('retry002_text_audit', HELPERS/'audit_request_text.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    payload = {'instructions': 'old', 'text': {'format': {'name': 'old', 'schema': {}}},
        'input': [{'role': 'assistant', 'content': [{'type': 'output_text', 'text': '{"unchanged":true}'}]},
                  {'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,offline'}]}]}
    result = module.transform(payload)
    assert result['instructions'].endswith(TASK_RESUMPTION_GUIDANCE)
    assert result['text']['format']['schema'] == output_schema()
    assert result['text']['format']['name'] == 'visual_field_rgb_joints_v34'
    assert result['input'] == payload['input'] and payload['instructions'] == 'old'
