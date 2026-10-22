"""Prepared V34 harness checks; only a harmless local dummy is launched."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import test_visual_field_v32_supervision as inherited

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / 'runs/visual_field_v34_zero_shot_001'
PRIOR = ROOT / 'runs/visual_field_v33_zero_shot_001'


@pytest.fixture
def inherited_helpers(monkeypatch):
    monkeypatch.setattr(inherited, 'HELPERS', HELPERS)
    return inherited


def test_dummy_child_exits_once_without_restart(tmp_path, inherited_helpers):
    inherited_helpers.test_dummy_child_survives_launcher_exit_and_failed_exit_does_not_restart(tmp_path)


def test_unsealed_harness_cannot_launch(tmp_path, inherited_helpers):
    inherited_helpers.test_real_trial_helper_refuses_unsealed_plan_without_launching(tmp_path)


@pytest.mark.parametrize('changed', ['frozen_initial_observation', 'official_bddl',
                                    'official_initial_state', 'missing_seal_check'])
def test_input_binding_is_rechecked_before_launch(tmp_path, changed, inherited_helpers):
    inherited_helpers.test_sealed_input_is_rechecked_immediately_before_any_supervisor_launch(tmp_path, changed)


def test_original_inputs_and_single_uncapped_subscription_scope():
    plan = json.loads((HELPERS / 'plan.json').read_text())
    old = json.loads((PRIOR / 'plan.json').read_text())
    assert plan['method_version'] == 'v34' and plan['shots'] == 0
    assert len(plan['scenes']) == 1
    scene, previous = plan['scenes'][0], old['scenes'][0]
    for key in ('id', 'task', 'suite', 'task_id', 'seed', 'episode_index', 'task_prompt',
                'bddl', 'initial_state', 'official_inputs', 'expected_initial_sha256'):
        assert scene[key] == previous[key]
    assert Path(scene['expected_initial']).read_bytes() == Path(previous['expected_initial']).read_bytes()
    assert hashlib.sha256(Path(scene['expected_initial']).read_bytes()).hexdigest() == scene['expected_initial_sha256']
    args = plan['command_argv']
    assert args[:3] == ['env', '-u', 'OPENAI_API_KEY']
    assert args[args.index('--method-version') + 1] == 'v34'
    assert args[args.index('--output') + 1] == scene['output']
    assert args[args.index('--expected-initial') + 1] == scene['expected_initial']
    assert '--demo' not in args
    assert plan['model'] == 'gpt-6-astra' and plan['effort'] == 'medium'
    assert plan['billing'] == 'chatgpt_subscription' and not plan['api_key_fallback']
    assert plan['max_episode_steps'] is plan['max_episode_wall_time_s'] is None
    assert not plan['process_supervision']['autorestart']
    assert plan['process_supervision']['startretries'] == 0
    assert not plan['process_supervision']['queue']
    assert not plan['process_supervision']['tcp_listener']
    assert plan['operator_stop']['request_file'] == str(Path(scene['output']) / 'operator_stop.request.json')
    assert set(plan['new_runtime_modules']) == {
        'visual_field_policy_v34.py', 'visual_field_subscription_v34.py',
        'visual_field_proposals_v34.py', 'visual_reference_review_v34.py', 'visual_feedback_transaction_v34.py'}
    assert set(plan['inherited_v33_runtime_modules']) == set(old['new_runtime_modules'])
    assert plan['prior_attempt'] == str(PRIOR / 'analysis.json')


def test_unchanged_helpers_and_historical_seal_remain_exact():
    seal = json.loads((PRIOR / 'preflight.json').read_text())
    assert len(list(HELPERS.glob('*.py'))) == 7
    for name in ('foreground_trial.py', 'launch.py', 'monitor.py', 'private_supervisor.py'):
        assert (HELPERS / name).read_bytes() == (PRIOR / name).read_bytes()
    for name, expected in seal['helper_sha256'].items():
        assert hashlib.sha256((PRIOR / name).read_bytes()).hexdigest() == expected


def test_current_text_audit_uses_v34_prompt_but_unchanged_wire_and_raw_history():
    from gpt_field.visual_field_subscription_v34 import output_schema, TASK_RESUMPTION_GUIDANCE
    spec = importlib.util.spec_from_file_location('v34_harness_text_audit', HELPERS / 'audit_request_text.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    assistant = {'role': 'assistant', 'content': [{'type': 'output_text', 'text': '{"unchanged":true}'}]}
    payload = {'instructions': 'old', 'text': {'format': {'name': 'old', 'schema': {}}},
               'input': [assistant, {'role': 'user', 'content': [
                   {'type': 'input_image', 'image_url': 'data:image/png;base64,offline'}]}]}
    result = audit.transform(payload)
    assert result['instructions'].endswith(TASK_RESUMPTION_GUIDANCE)
    assert result['text']['format']['schema'] == output_schema()
    assert result['text']['format']['name'] == 'visual_field_rgb_joints_v34'
    assert result['input'] == payload['input']
    assert payload['instructions'] == 'old'
