"""Fake-only lifecycle checks for the matched RoboCasa Astranomy trial.

No simulator worker, EGL context, live model call or credential read occurs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'robocasa365'
sys.path[:0] = [str(PROJECT / 'tests'), str(PROJECT), str(ROOT / 'src'), str(ROOT / 'scripts')]
from test_runner import observation
from rc365 import astranomy_runner as runner
from rc365.astranomy import validate_mobile_feedback
from gpt_field.astra import APIError, MissingCredentials
from gpt_field.astranomy_agentic_client import validate_feedback

SECRET = 'EVALUATOR_ONLY_SENTINEL'


class Clock:
    def __init__(self):
        self.now = 100.

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeSim:
    def __init__(self, *, success_at=2, events=None, after_step=None,
                 clear_status_on_close=False, step_error=None):
        self.obs = observation()
        self.success_at = success_at
        self.events = events if events is not None else []
        self.after_step = after_step
        self.clear_status_on_close = clear_status_on_close
        self.step_error = step_error
        self.actions, self.mobile_commands = [], []
        self.status = {'success': False, 'robocasa_success': False, 'terminated': False,
                       'steps': 0, 'contacts': {SECRET: True}}
        self.metadata = {'fixture': True}
        self.physical_hash, self.xml_hash = 'a' * 64, 'b' * 64
        self.close_count = 0

    def start(self):
        self.events.append('start')

    def reset(self):
        self.events.append('reset')
        return self.obs

    def snapshot(self, include_xml=False):
        self.events.append('snapshot')
        return {'physical_sha256': self.physical_hash, 'model_xml_sha256': self.xml_hash,
                'hidden_task_state': {SECRET: True}}

    def _advance(self, action, *, mobile=False):
        self.events.append('mobile_step' if mobile else 'step')
        self.actions.append(list(action))
        if self.step_error:
            raise self.step_error
        state = deepcopy(self.obs.mobile_state)
        state['step'] += 1
        state['control_mode'] = 'mobile' if mobile else 'arm'
        self.obs = replace(self.obs, mobile_state=state)
        success = self.success_at is not None and len(self.actions) >= self.success_at
        self.status.update(success=success, robocasa_success=success, terminated=success,
                           steps=len(self.actions))
        if self.after_step:
            self.after_step(self)
        return self.obs, success

    def step(self, action):
        return self._advance(action)

    def mobile_step(self, action, command):
        self.mobile_commands.append(deepcopy(command))
        return self._advance(action, mobile=True)

    def close(self):
        self.close_count += 1
        self.events.append('close')
        if self.clear_status_on_close:
            self.status.clear()


def arm_command(request, *, steps=5, gripper=1):
    target = list(request['robot_state'])
    target[-1] = gripper
    return {'name': 'execute_eef_target', 'arguments': {'target': target, 'steps': steps}}


def mobile_command(request, *, steps=5):
    return {'name': 'execute_mobile', 'arguments': {'base': [.1, -.1, .05], 'torso': .1, 'steps': steps}}


class FakeClient:
    def __init__(self, *, events, commands=None, error=None, on_predict=None, clock=None):
        self.events = events
        self.commands = commands or [arm_command]
        self.error = error
        self.on_predict = on_predict
        self.clock = clock
        self.requests, self.results = [], []
        self.pending = False
        self._used_access_tokens = ()

    def predict(self, request):
        assert not self.pending, 'Previous tool result must be closed before another model request'
        self.events.append('predict')
        self.requests.append(deepcopy(request))
        assert SECRET not in json.dumps(request)
        if self.clock:
            self.clock.advance(7.5)
        if self.on_predict:
            self.on_predict(request)
        if self.error:
            raise self.error
        command = self.commands[min(len(self.requests)-1, len(self.commands)-1)]
        self.last_command = command(request) if callable(command) else deepcopy(command)
        self.pending = True
        return deepcopy(self.last_command)

    def record_result(self, feedback):
        assert self.pending
        assert SECRET not in json.dumps(feedback)
        checked = (validate_mobile_feedback(feedback) if self.last_command['name'] == 'execute_mobile'
                   else validate_feedback(feedback))
        self.events.append('record_result')
        self.results.append(checked)
        self.pending = False

    def budget_snapshot(self):
        return {'billing': 'chatgpt_subscription', 'api_key_calls': 0,
                'api_calls': len(self.requests), 'subscription_inference_calls': len(self.requests)}

    def history_stats(self):
        return {'requests': len(self.requests), 'tool_results': len(self.results),
                'pending_tool_result': self.pending}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    reference = tmp_path / 'reference'
    reference.mkdir()
    initial = FakeSim()
    (reference / 'initial_sensors.json').write_text(json.dumps(runner.common.sensor_record(initial.obs)))
    (reference / 'initial_physics.json').write_text(json.dumps(initial.snapshot(include_xml=True)))
    marker = tmp_path / 'runtime_source.py'
    marker.write_text('original runtime\n')
    def inventory(config, sim_config, reference_run):
        return {str(marker): hashlib.sha256(marker.read_bytes()).hexdigest(),
            '@configuration': runner.common.canonical_sha256({
                'api': asdict(config), 'sim': asdict(sim_config), 'reference': str(reference_run)})}
    monkeypatch.setattr(runner, 'runtime_inventory', inventory)
    events = []
    def no_credentials():
        pytest.fail('Offline runner tests must never read OAuth credentials')
    return SimpleNamespace(path=tmp_path, reference=reference, marker=marker, events=events,
        output=tmp_path / 'output', config=runner.configuration(), sim_config=runner.RoboCasa365Config(),
        token_provider=no_credentials)


def execute(setup, *, sim=None, client=None):
    sim = sim or FakeSim(events=setup.events)
    client = client or FakeClient(events=setup.events)
    def client_factory(config, **kwargs):
        setup.events.append('client_construct')
        assert sim.events.count('reset') == 1
        assert (setup.output / 'initial_match.json').is_file()
        assert json.loads((setup.output / 'initial_match.json').read_text())['matched'] is True
        return client
    result = runner.execute(setup.output, reference_run=setup.reference,
        config=setup.config, sim_config=setup.sim_config, token_provider=setup.token_provider,
        sim_factory=lambda _: sim, client_factory=client_factory, video=False)
    return result, sim, client


def test_matching_fresh_reset_before_client_and_model_then_midsegment_success(setup):
    result, sim, client = execute(setup)
    assert setup.events.index('reset') < setup.events.index('snapshot') < setup.events.index('client_construct') < setup.events.index('predict')
    assert setup.events.count('start') == setup.events.count('reset') == 1
    assert result['status'] == 'success' and result['task_success']
    assert result['steps'] == result['feedback_count'] == result['arm_steps'] == 2
    assert result['mobile_steps'] == 0
    assert result['pending_action'] is None and result['pending_tool'] is False
    assert result['preparation_model_calls'] == 0
    assert result['official_benchmark_score'] is False
    assert len(client.requests) == len(client.results) == 1
    assert client.results[0]['status'] == 'interrupted'
    assert client.results[0]['requested_steps'] == 5 and client.results[0]['executed_steps'] == 2
    assert not client.pending and sim.close_count == 1
    assert SECRET not in json.dumps(client.requests + client.results)


@pytest.mark.parametrize('change', ['sensor', 'physical', 'xml', 'already_successful'])
def test_initial_mismatch_never_constructs_client_or_acts(setup, change):
    sim = FakeSim(events=setup.events)
    if change == 'sensor':
        mobile = deepcopy(sim.obs.mobile_state)
        mobile['base_joint_position'][0] = .01
        sim.obs = replace(sim.obs, mobile_state=mobile)
    elif change == 'physical':
        sim.physical_hash = 'c' * 64
    elif change == 'xml':
        sim.xml_hash = 'c' * 64
    else:
        sim.status['success'] = sim.status['robocasa_success'] = True
    def forbidden(*args, **kwargs):
        pytest.fail('Initial mismatch reached the model client')
    result = runner.execute(setup.output, reference_run=setup.reference,
        config=setup.config, sim_config=setup.sim_config, token_provider=setup.token_provider,
        sim_factory=lambda _: sim, client_factory=forbidden, video=False)
    assert result['status'] == 'infrastructure_error'
    assert not result['task_success'] and not result['completion_time_eligible']
    assert result['steps'] == 0 and sim.actions == [] and sim.close_count == 1
    assert result['control_elapsed_s'] is None


def test_fresh_directory_required_before_sim_or_client(setup):
    setup.output.mkdir()
    untouched = setup.output / 'preserve.txt'
    untouched.write_text('prior artifact')
    def forbidden(*args, **kwargs):
        pytest.fail('Existing output crossed the simulator/client boundary')
    with pytest.raises(FileExistsError):
        runner.execute(setup.output, reference_run=setup.reference,
            config=setup.config, sim_config=setup.sim_config, token_provider=setup.token_provider,
            sim_factory=forbidden, client_factory=forbidden, video=False)
    assert untouched.read_text() == 'prior artifact'


def test_segment_completion_never_substitutes_for_raw_task_success(setup):
    # First five-step tool finishes, yet the task only succeeds during tool two.
    result, sim, client = execute(setup, sim=FakeSim(events=setup.events, success_at=7))
    assert result['steps'] == result['feedback_count'] == 7
    assert len(client.requests) == len(client.results) == 2
    assert [r['status'] for r in client.results] == ['segment_complete', 'interrupted']
    assert [r['executed_steps'] for r in client.results] == [5, 2]
    assert result['task_success'] and len(sim.actions) == 7


def test_mobile_dispatch_retains_previous_close_command_and_acknowledges_interruption(setup):
    client = FakeClient(events=setup.events, commands=[lambda r: arm_command(r, steps=1, gripper=1), mobile_command])
    result, sim, client = execute(setup, sim=FakeSim(events=setup.events, success_at=4), client=client)
    assert result['task_success']
    assert result['arm_steps'] == 1 and result['mobile_steps'] == 3
    assert len(sim.mobile_commands) == 3
    assert all(action[:6] == [0.] * 6 and action[6] == 1. for action in sim.actions[1:])
    assert all(command == {'base': [.1, -.1, .05], 'torso': .1} for command in sim.mobile_commands)
    assert client.results[-1]['status'] == 'interrupted'
    assert client.results[-1]['gripper_command'] == 1.
    assert client.results[-1]['executed_steps'] == 3
    assert client.results[-1]['measured_mobile_state']['step'] == 4
    assert result['pending_tool'] is False


@pytest.mark.parametrize('error', [APIError('offline subscription failure'), MissingCredentials('offline missing credentials')])
def test_subscription_failure_wait_is_counted_and_worker_closed(setup, monkeypatch, error):
    clock = Clock()
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=clock.monotonic))
    client = FakeClient(events=setup.events, error=error, clock=clock)
    result, sim, client = execute(setup, client=client)
    assert result['status'] == 'subscription_error'
    assert result['steps'] == result['feedback_count'] == 0
    assert result['model_responses'] == 0
    assert result['execution_budget']['subscription_inference_calls'] == 1
    assert result['timing_components']['model_pipeline_s'] == pytest.approx(7.5)
    assert result['control_elapsed_s'] == pytest.approx(7.5)
    assert result['control_without_video_s'] == pytest.approx(7.5)
    assert not result['completion_time_eligible']
    assert sim.close_count == 1 and sim.actions == [] and client.results == []


def test_source_change_invalidates_completion_without_erasing_raw_success(setup):
    def changed(sim):
        setup.marker.write_text('changed while executing\n')
    result, sim, client = execute(setup, sim=FakeSim(events=setup.events, after_step=changed))
    assert result['robocasa_success'] is True
    assert result['runtime_unchanged'] is False
    assert result['task_success'] is False and result['completion_time_eligible'] is False
    assert (setup.output / 'frozen_source' / setup.marker.relative_to('/')).read_text() == 'original runtime\n'
    assert sim.close_count == 1


def test_stop_after_model_response_executes_no_extra_action_and_closes_zero_step_tool(setup):
    def stop(request):
        (setup.output / 'operator_stop.request.json').write_text('{}')
    client = FakeClient(events=setup.events, on_predict=stop)
    result, sim, client = execute(setup, client=client)
    assert result['status'] == 'user_stop'
    assert result['steps'] == result['feedback_count'] == 0
    assert sim.actions == [] and sim.close_count == 1
    assert len(client.results) == 1
    assert client.results[0]['status'] == 'interrupted' and client.results[0]['executed_steps'] == 0
    assert result['pending_tool'] is False and result['pending_action'] is None


def test_stop_at_complete_action_boundary_preserves_one_feedback_and_no_extra_actions(setup):
    def stop(sim):
        (setup.output / 'operator_stop.request.json').write_text('{}')
    result, sim, client = execute(setup, sim=FakeSim(events=setup.events, success_at=None, after_step=stop))
    assert result['status'] == 'user_stop'
    assert result['steps'] == result['feedback_count'] == 1
    assert len(sim.actions) == 1 and len(client.requests) == len(client.results) == 1
    assert client.results[0]['status'] == 'interrupted' and client.results[0]['executed_steps'] == 1
    assert result['pending_action'] is None and result['pending_tool'] is False


def test_unknown_interrupted_simulator_step_is_not_synthesized_as_complete(setup):
    sim = FakeSim(events=setup.events, step_error=KeyboardInterrupt('unknown physical step endpoint'))
    result, sim, client = execute(setup, sim=sim)
    assert result['status'] == 'user_stop'
    assert result['steps'] == result['feedback_count'] == 0
    assert result['pending_action'] == 0 and result['pending_tool'] is True
    assert len(sim.actions) == 1
    assert client.results == [] and client.pending
    assert result['task_success'] is False and result['completion_time_eligible'] is False
    assert sim.close_count == 1


def test_evaluator_terminal_survives_simulator_close_clearing_live_status(setup):
    result, sim, client = execute(setup, sim=FakeSim(events=setup.events, clear_status_on_close=True))
    assert sim.status == {}
    assert result['robocasa_success'] is True and result['task_success'] is True
    assert result['evaluator_terminal']['robocasa_success'] is True
    assert result['evaluator_terminal']['contacts'] == {SECRET: True}
    assert SECRET not in json.dumps(client.requests + client.results)
    saved = json.loads((setup.output / 'summary.json').read_text())
    assert saved['evaluator_terminal']['robocasa_success'] is True
    assert sim.close_count == 1


def test_stopfile_present_at_first_decision_sends_no_model_request_or_action(setup):
    class StopAfterReset(FakeSim):
        def reset(self):
            obs = super().reset()
            (setup.output / 'operator_stop.request.json').write_text('{}')
            return obs
    result, sim, client = execute(setup, sim=StopAfterReset(events=setup.events))
    assert result['status'] == 'user_stop'
    assert result['steps'] == result['feedback_count'] == result['model_responses'] == 0
    assert client.requests == client.results == []
    assert sim.actions == [] and sim.close_count == 1
    assert result['pending_action'] is None and result['pending_tool'] is False
