"""Offline authorization, source binding and evaluator-boundary checks only."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'robocasa365'
sys.path[:0] = [str(PROJECT), str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from rc365 import runner
from rc365.sim import MobileObservation, MOBILE_SCHEMA, MOBILE_COMMAND_FRAME
from gpt_field.agent_speedup_v2_user_check import ReviewError, ReviewStore, canonical_sha256
from gpt_field.observation import image_to_data_url
from test_robocasa_user_check_geometry import STATE, metadata

PLAN = {
    'scene': 'Counter cans and a cabinet with a designated side.',
    'parts': [{'id': 'A', 'text': 'counter cans', 'move': 'free', 'w': 20, 'x': 30},
              {'id': 'B', 'text': 'designated cabinet side', 'move': 'fixed', 'w': 25, 'x': 35}],
    'hold_true': [],
    'steps': [{'do': 'place_in', 'obj': 'A', 'grip': [], 'ref': 'B',
               'sense': 'any', 'keep': [], 'until': []}],
}


def observation():
    pixels = image_to_data_url(np.zeros((512, 512, 3), dtype=np.uint8), 512, rotate=False)
    mobile = {'schema_version': MOBILE_SCHEMA, 'command_frame': MOBILE_COMMAND_FRAME,
              'base_joint_position': [0., 0., 0.], 'base_joint_velocity': [0., 0., 0.],
              'torso_joint_position': [0.], 'torso_joint_velocity': [0.],
              'base_position_world_m': [0., 0., 0.], 'base_rotation_world': np.eye(3).tolist(),
              'control_mode': 'arm', 'step': 0}
    return MobileObservation(pixels, pixels, tuple(STATE), runner.TASK_PROMPT,
        joint_position=(0.,) * 7, joint_velocity=(0.,) * 7,
        geometry=metadata(), mobile_state=mobile)


class FakeSim:
    def __init__(self, success_at=2):
        self.obs = observation()
        self.success_at = success_at
        self.status = {'success': False, 'robocasa_success': False, 'terminated': False,
                       'contacts': {'EVALUATOR_SECRET_SENTINEL': True}}
        self.metadata = {'fixture': True}
        self.ops, self.actions, self.mobile_commands = [], [], []
        self.closed = False

    def start(self):
        self.ops.append('start')

    def reset(self):
        self.ops.append('reset')
        return self.obs

    def snapshot(self, include_xml=False):
        return {'physical_sha256': 'a' * 64, 'model_xml_sha256': 'b' * 64,
                'hidden_task_state': {'EVALUATOR_SECRET_SENTINEL': True}}

    def step(self, action):
        self.actions.append(list(action))
        success = len(self.actions) >= self.success_at
        self.status.update(success=success, robocasa_success=success, terminated=success)
        return self.obs, success

    def mobile_step(self, action, command):
        self.mobile_commands.append(copy.deepcopy(command))
        return self.step(action)

    def close(self):
        self.closed = True


class FakeClient:
    instances = []

    def __init__(self, config, **kwargs):
        self.kwargs, self.requests, self.checked = kwargs, [], []
        self.instances.append(self)

    def check_observation(self, obs):
        self.checked.append(obs)

    def request_plan(self, external, wrist, **kwargs):
        self.requests.append((external.shape, wrist.shape, kwargs))
        return copy.deepcopy(PLAN)

    def budget_snapshot(self):
        return {'billing': 'chatgpt_subscription', 'api_key_calls': 0,
                'api_calls': len(self.requests)}

    def history_stats(self):
        return {'accepted_requests': len(self.requests)}


class FakePolicy:
    instances = []
    mobile = False

    def __init__(self, client, *, lock):
        self.client, self.lock = client, lock
        self.seen, self.feedback = [], []
        self.instances.append(self)

    def reset(self):
        pass

    def act(self, obs):
        self.seen.append(obs)
        diagnostics = {'model_believes_finished': True}
        if self.mobile:
            diagnostics['mobile_command'] = {'base': [.1, 0., 0.], 'torso': 0.}
        return SimpleNamespace(action=[0., 0., 0., 0., 0., 0., -1.], diagnostics=diagnostics)

    def observe_transition(self, before, decision, after):
        self.feedback.append((before, decision, after))
        return {'observed': True}

    def agent_stats(self):
        return {'observations': len(self.seen)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = PROJECT / 'rc365/runner.py'
    inventory = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    monkeypatch.setattr(runner, 'runtime_inventory', lambda config, sim_config: {
        **inventory, '@configuration': canonical_sha256({'api': asdict(config), 'sim': asdict(sim_config)})})
    FakeClient.instances, FakePolicy.instances = [], []
    args = {'config': runner.reviewed.base.configuration(), 'sim_config': runner.RoboCasaConfig(),
            'token_provider': lambda: 'offline-fixture-no-network'}
    return SimpleNamespace(path=tmp_path, args=args)


def draft(setup):
    sim = FakeSim()
    result = runner.prepare(setup.path / 'review', **setup.args,
                            sim_factory=lambda _: sim, client_factory=FakeClient)
    return ReviewStore(setup.path / 'review'), result, sim


def approved(setup):
    store, data, _ = draft(setup)
    store.approve(expected_revision=data['revision'], reviewed_steps=[0])
    return store


def execute(setup, store, *, sim=None, policy=FakePolicy):
    sim = sim or FakeSim()
    summary = runner.execute(setup.path / 'run', review_dir=store.path, **setup.args,
        sim_factory=lambda _: sim, client_factory=FakeClient, policy_factory=policy, video=False)
    return summary, sim


@pytest.mark.parametrize('command', ['prepare', 'run'])
def test_cli_requires_test_approval_before_credentials_or_simulation(tmp_path, monkeypatch, command):
    from gpt_field import chatgpt_auth
    def forbidden(*args, **kwargs):
        pytest.fail('Unapproved CLI crossed the credential/simulator boundary')
    monkeypatch.setattr(chatgpt_auth, 'SubscriptionCredentials', forbidden)
    monkeypatch.setattr(runner, 'prepare', forbidden)
    monkeypatch.setattr(runner, 'execute', forbidden)
    monkeypatch.setattr(runner, 'new_sim', forbidden)
    review_dir = tmp_path / 'review'
    with pytest.raises(SystemExit) as exc:
        runner.main([command, '--review-dir', str(review_dir), '--output', str(tmp_path / 'run')])
    assert exc.value.code == 2
    assert not review_dir.exists()
    assert not (tmp_path / 'run').exists()


def test_unapproved_draft_blocks_execution_before_simulator_and_model(setup):
    store, result, prepared_sim = draft(setup)
    assert result['approved'] is None
    assert prepared_sim.actions == [] and prepared_sim.closed
    sim = FakeSim()
    with pytest.raises(ReviewError):
        execute(setup, store, sim=sim)
    assert sim.ops == [] and sim.actions == []
    assert len(FakeClient.instances) == 1
    assert not (setup.path / 'run').exists()


def test_only_evaluator_success_ends_trial_and_hidden_truth_never_reaches_policy(setup):
    store = approved(setup)
    summary, sim = execute(setup, store)
    # The fake model claims completion from its very first action, but the
    # independent fake evaluator succeeds only after action two.
    assert summary['steps'] == summary['feedback_count'] == 2
    assert summary['robocasa_success'] and summary['task_success']
    assert summary['official_benchmark_score'] is False
    assert summary['evaluation_profile'] == 'uncapped_exploratory_pilot'
    policy = FakePolicy.instances[-1]
    for obs in policy.seen:
        assert not hasattr(obs, 'status')
        assert 'EVALUATOR_SECRET_SENTINEL' not in json.dumps(runner.sensor_record(obs))
    for before, decision, after in policy.feedback:
        assert 'EVALUATOR_SECRET_SENTINEL' not in json.dumps(runner.sensor_record(after))
    assert not any('evaluator' in key for key in FakeClient.instances[-1].kwargs)
    assert sim.closed
    manifest = json.loads((setup.path / 'run/manifest.json').read_text())
    assert manifest['transit_action_m'] == .05
    assert manifest['transit_target_limit_m'] == .50
    assert manifest['automatic_episode_caps'] is None
    assert manifest['automatic_episode_retries'] == 0
    assert manifest['task'] == 'RestockPantry' and manifest['mobile_control'] is True
    with pytest.raises(ReviewError):
        runner.execute(setup.path / 'second', review_dir=store.path, **setup.args,
            sim_factory=lambda _: FakeSim(), client_factory=FakeClient,
            policy_factory=FakePolicy, video=False)
    assert not (setup.path / 'second').exists()


def test_mobile_diagnostics_route_to_native_mobile_endpoint(setup):
    class MobilePolicy(FakePolicy):
        mobile = True
    summary, sim = execute(setup, approved(setup), policy=MobilePolicy)
    assert summary['task_success']
    assert len(sim.mobile_commands) == summary['steps'] == 2
    assert sim.mobile_commands == [{'base': [.1, 0., 0.], 'torso': 0.}] * 2


def test_review_binds_mobile_robot_state_before_execution_model(setup):
    store = approved(setup)
    sim = FakeSim()
    mobile = copy.deepcopy(sim.obs.mobile_state)
    mobile['base_joint_position'][0] = .01
    sim.obs = replace(sim.obs, mobile_state=mobile)
    summary, sim = execute(setup, store, sim=sim)
    assert summary['status'] == 'infrastructure_error'
    assert summary['error']['type'] == 'ReviewError'
    assert summary['steps'] == 0
    assert len(FakeClient.instances) == 1
    assert sim.actions == [] and sim.closed


def test_entrypoint_enforces_real_transit50_before_policy_import():
    code = """
import os,runpy,sys,json
ns=runpy.run_path(sys.argv[1],run_name='offline_entrypoint_test')
sys.argv=[sys.argv[1],'preflight']
assert ns['main']()==0
from gpt_field.action import USER_CHECK_TRANSIT_ACTION_M
from gpt_field.agent_speedup_v2_user_check_fragments import PROFILES
assert USER_CHECK_TRANSIT_ACTION_M==.05
assert PROFILES['transit']['action_translation_m']==.05
assert PROFILES['transit']['target_translation_m']==.50
print('TRANSIT50_RUNTIME_VERIFIED')
"""
    env = dict(os.environ)
    env.pop('GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-c', code, str(PROJECT / 'run.py')],
        env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith('TRANSIT50_RUNTIME_VERIFIED')


def test_entrypoint_rejects_conflicting_transit_without_creating_trial(tmp_path):
    env = dict(os.environ, GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M='.10', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, str(PROJECT / 'run.py'), 'prepare',
                            '--review-dir', str(tmp_path / 'review')],
        env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'conflicting environment value' in result.stderr
    assert not (tmp_path / 'review').exists()


def test_real_clients_build_high_effort_subscription_payloads_without_credentials():
    from rc365.policy import RoboCasa365PlanClient, RoboCasa365Client
    from gpt_field.agent_speedup_v2_user_check_client import PlanLock
    from gpt_field.astra import ENDPOINT
    from gpt_field.ordered_transport import ordered_dumps, ordered_sse_transport

    def forbidden_credentials():
        pytest.fail('Payload inspection must not read OAuth credentials or send a request')

    config = runner.reviewed.base.configuration()
    lock = PlanLock(PLAN, task_prompt=runner.TASK_PROMPT)
    planning = RoboCasa365PlanClient(config, task_prompt=runner.TASK_PROMPT,
                                     token_provider=forbidden_credentials)
    execution = RoboCasa365Client(config, task_prompt=runner.TASK_PROMPT,
                                  approved_plan=PLAN, lock=lock,
                                  token_provider=forbidden_credentials)
    for client, modes in ((planning, ['plan', 'replan']), (execution, ['agent'])):
        assert client.config.endpoint == ENDPOINT
        assert client.transport is ordered_sse_transport
        assert client.budget_snapshot()['billing'] == 'chatgpt_subscription'
        assert client.budget_snapshot()['api_key_calls'] == 0
        for mode in modes:
            payload = client.build_payload({'fields': {}, 'images': []}, mode=mode)
            wire = json.loads(ordered_dumps(payload))
            assert wire['model'] == 'gpt-6-astra'
            assert wire['reasoning']['effort'] == 'high'
            assert wire['stream'] is True and wire['store'] is False
            if mode == 'agent':
                schema = wire['text']['format']['schema']
                assert 'mobile' in schema['required']
                assert 'mobile' in schema['properties']['type']['enum']
                assert wire['text']['format']['name'] == 'robocasa365_uc71_mobile'
        assert client.budget_snapshot()['api_calls'] == 0


def test_authorized_automatic_plan_approval_is_never_labeled_human_review(setup):
    prepared_sim = FakeSim()
    draft_result = runner.prepare(setup.path / 'review', **setup.args,
        sim_factory=lambda _: prepared_sim, client_factory=FakeClient, auto_approve=True)
    store = ReviewStore(setup.path / 'review')
    assert draft_result['preparation']['approval_mode'] == 'user_authorized_automatic'
    assert draft_result['preparation']['manual_review'] is False
    # Initial draft creation does not bypass ReviewStore's single-use gate.
    assert draft_result['approved'] is None
    assert prepared_sim.actions == [] and prepared_sim.closed
    store.approve(expected_revision=draft_result['revision'], reviewed_steps=[0])
    summary, sim = execute(setup, store)
    assert summary['task_success'] and sim.closed
    assert summary['approval_mode'] == 'user_authorized_automatic'
    assert summary['human_reviewed_plan'] is False
    manifest = json.loads((setup.path / 'run/manifest.json').read_text())
    saved_summary = json.loads((setup.path / 'run/summary.json').read_text())
    saved_review = json.loads((setup.path / 'run/user_check_review.json').read_text())
    for artifact in (manifest, saved_summary):
        assert artifact['approval_mode'] == 'user_authorized_automatic'
        assert artifact['human_reviewed_plan'] is False
    assert saved_review['preparation']['approval_mode'] == 'user_authorized_automatic'
    assert saved_review['preparation']['manual_review'] is False
    assert store.load()['preparation']['manual_review'] is False
    assert len(FakeClient.instances) == 2
