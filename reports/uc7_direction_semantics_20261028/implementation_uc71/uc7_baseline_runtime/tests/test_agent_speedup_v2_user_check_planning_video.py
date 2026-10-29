"""Opt-in recorder wiring and approval identity; fake RGB, no live calls or simulation."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from agent_speedup_v2_fakes import FakeV2Sim
from gpt_field.agent_speedup_v2_user_check import ReviewError
from gpt_field.agent_speedup_v2_user_check_fragments import FragmentSequencePrimitive
from test_agent_speedup_v2_user_check_runner import DraftClient, TASK, runner, settings


def prepare_planning(tmp_path):
    config, sim_config, monitor = settings()
    fake = FakeV2Sim(prompt=TASK.prompt)
    store = runner.prepare_review(
        tmp_path / 'review', config=config, sim_config=sim_config, task_key=TASK.key,
        episode_index=0, monitor_options=monitor, client_factory=DraftClient,
        sim_factory=lambda _: fake, token_provider=lambda: 'offline',
        video=True, planning_video=True)
    return store, fake, config, sim_config, monitor


def test_planning_video_is_opt_in_and_configuration_bound():
    config, sim_config, monitor = settings()
    original = runner.runtime_inventory(config, sim_config, monitor_options=monitor)
    planning = runner.runtime_inventory(config, sim_config, monitor_options=monitor, planning_video=True)
    assert original['@configuration'] != planning['@configuration']
    assert {k: v for k, v in original.items() if not k.startswith('@')} == {
        k: v for k, v in planning.items() if not k.startswith('@')}
    options = runner.execution_options(config, sim_config, planning_video=True)
    assert options['planning_video'] is True
    assert options['video_visualization']['recorder'] == 'AgentSpeedUpV2UserCheckVideoRecorder'
    assert options['video_visualization']['affects_policy'] is False
    assert options['video_visualization']['scalar_planning_cost_field'] is False
    assert runner.execution_options(config, sim_config)['video_visualization']['recorder'] == 'AgentSpeedUpV2VideoRecorder'


def test_planning_video_rejects_disabled_video_before_preparation(tmp_path):
    config, sim_config, monitor = settings()
    constructed = []
    with pytest.raises(ReviewError, match='requires video recording'):
        runner.prepare_review(
            tmp_path / 'review', config=config, sim_config=sim_config, task_key=TASK.key,
            episode_index=0, monitor_options=monitor, video=False, planning_video=True,
            sim_factory=lambda _: constructed.append(True))
    assert constructed == []
    assert not (tmp_path / 'review').exists()


def test_disabling_planning_video_invalidates_existing_approval(tmp_path):
    store, fake, config, sim_config, monitor = prepare_planning(tmp_path)
    assert fake.actions == []
    assert store.load()['preparation']['execution']['planning_video'] is True
    store.approve(expected_revision=0, reviewed_steps=[0])
    with pytest.raises(ReviewError, match='Runtime code changed'):
        runner.execute_reviewed(
            tmp_path / 'run', review_dir=store.path, config=config, sim_config=sim_config,
            task_key=TASK.key, episode_index=0, monitor_options=monitor,
            token_provider=lambda: 'offline', video=True, planning_video=False)
    assert store.load()['status'] == 'approved'
    assert not (tmp_path / 'run').exists()


def test_preflight_preserves_planning_video_flag_without_live_work():
    config, sim_config, monitor = settings()
    result = runner.preflight(config, sim_config, task_key=TASK.key, episode_index=0,
                              monitor_options=monitor, planning_video=True)
    assert '--planning-video' in result['live_command']
    assert result['live_command_first_run'] == result['live_command']
    assert result['planning_video'] is True
    assert result['video_visualization']['resolution'] == [1600, 960]
    assert result['video_visualization']['fps'] == 20
    assert result['simulator_started'] is False
    assert result['api_calls'] == 0


def test_run_restores_planning_video_from_review(tmp_path, monkeypatch):
    import gpt_field.chatgpt_auth as auth
    store, _, _, _, _ = prepare_planning(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    monkeypatch.setattr(runner, '_live_gate', lambda *args: None)
    class NoCredentials:
        def __init__(self, **kwargs):
            pass
        def access_token(self):
            pytest.fail('No credential read is allowed in this fake wiring test')
    monkeypatch.setattr(auth, 'SubscriptionCredentials', NoCredentials)
    calls = []
    def execute(output, **kwargs):
        calls.append(kwargs)
        return {'status': 'fake', 'output': str(output), 'evaluation_completed': True,
                'user_check': {'review_id': store.load()['id']}}
    monkeypatch.setattr(runner, 'execute_reviewed', execute)
    assert runner.main(['run', '--review-dir', str(store.path), '--output', str(tmp_path / 'run')]) == 0
    assert calls[0]['planning_video'] is True
    assert calls[0]['video'] is True


def test_recorder_reads_only_current_fragment_snapshot(tmp_path, monkeypatch):
    store, _, config, sim_config, monitor = prepare_planning(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    fragment_state = {'status': 'running', 'target_xyz_m': [0.1, 0.2, 0.3]}
    class FakeFragment(FragmentSequencePrimitive):
        def __init__(self):
            pass
        def progress(self):
            return deepcopy(fragment_state)
    policy = SimpleNamespace(_mode='fragment', _primitive=FakeFragment())
    callbacks = []
    class FakeRecorder:
        def __init__(self, *args, planning_provider=None, **kwargs):
            callbacks.append(planning_provider)
    monkeypatch.setitem(sys.modules, 'gpt_field.agent_speedup_v2_user_check_video',
                        SimpleNamespace(AgentSpeedUpV2UserCheckVideoRecorder=FakeRecorder))
    def execute(output, **kwargs):
        assert kwargs['policy_factory'](None) is policy
        kwargs['recorder_factory'](tmp_path / 'never_encoded.mp4')
        return {'fake': True}
    monkeypatch.setattr(runner.base, 'execute', execute)
    monkeypatch.setattr(runner, '_normalize_artifacts', lambda output, summary, *args, **kwargs: summary)
    result = runner.execute_reviewed(
        tmp_path / 'run', review_dir=store.path, config=config, sim_config=sim_config,
        task_key=TASK.key, episode_index=0, monitor_options=monitor, video=True,
        planning_video=True, policy_factory=lambda *args, **kwargs: policy,
        token_provider=lambda: 'offline')
    assert result == {'fake': True}
    assert len(callbacks) == 1
    snapshot = callbacks[0]()
    snapshot['target_xyz_m'][0] = 999
    assert fragment_state['target_xyz_m'] == [0.1, 0.2, 0.3]
    policy._mode = 'direct'
    assert callbacks[0]() is None
    policy._mode = 'fragment'
    policy._primitive = object()
    assert callbacks[0]() is None
    assert not (tmp_path / 'never_encoded.mp4').exists()
