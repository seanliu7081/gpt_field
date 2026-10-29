"""Audit actual direct/wrapper entry points with existing pure fakes; no live work."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'baseline'
RUNNER = BASE / 'scripts/evaluate_speedup_evidence.py'
WRAPPER = ROOT / 'launches/prepare_then_begin_review.py'
sys.path.insert(0, str(BASE / 'src'))
sys.path.append(str(ROOT.parents[1] / 'tests'))
from agent_speedup_v2_fakes import FakeV2Sim
from gpt_field.agent_speedup_v2_client import FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_tasks import resolve_v2_task
from gpt_field.agent_speedup_evidence_clock import EvaluationBudgetExceeded

PLAN = {'scene': 'Offline cup fixture.', 'parts': [{'id': 'A', 'text': 'cup: body',
    'move': 'free', 'w': 20, 'x': 30}], 'hold_true': [], 'steps': [{'do': 'hold',
    'obj': 'A', 'grip': [], 'ref': None, 'sense': 'any', 'keep': [], 'until': []}]}
TASK = resolve_v2_task('libero_10_t1')

def forbidden(*args, **kwargs):
    raise AssertionError('Network or subprocess forbidden in offline wrapper audit')

socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
subprocess.Popen = forbidden

def one(mode, arm, failure):
    with tempfile.TemporaryDirectory(prefix='speedup_wrapper_audit_') as temp:
        temp = Path(temp)
        reservation = {'reservation_id': 'offline_fake', 'stage': 'P5', 'task': TASK.key,
            'episode_index': 2, 'seed': 7, 'arm': arm,
            'effective_service_budget_s': 900, 'score_threshold_s': 300}
        reserve_file = temp / 'reservation.json'
        reserve_file.write_text(json.dumps(reservation))
        session = temp / 'session'
        args = ['prepare', '--session-dir', str(session), '--reservation-file',
            str(reserve_file), '--store-dir', str(temp / 'never-read-credentials'), '--enroll-reset']
        transport = ScriptedV2Transport(FixedAnswers(plans=[PLAN]), effort='high')
        captured, contexts, fakes, commands = [], [], [], []
        def offline_transport(endpoint, payload, headers, timeout):
            captured.append({'endpoint': endpoint, 'payload': payload, 'timeout': timeout,
                'header_names': sorted(headers)})
            if failure:
                raise EvaluationBudgetExceeded('offline injected deadline only')
            return transport(endpoint, payload, headers, timeout)
        offline_transport.offline_mock = True
        def tracer(frame, event, arg):
            if event == 'call' and frame.f_code.co_filename == str(RUNNER) and frame.f_code.co_name == 'main':
                glob = frame.f_globals
                commands.append(list(frame.f_locals.get('argv') or sys.argv[1:]))
                if not glob.get('_OFFLINE_AUDIT_PATCHED'):
                    glob['_OFFLINE_AUDIT_PATCHED'] = True
                    contexts.append({'module_name': glob['__name__'], 'file': glob['__file__'],
                        'sys_argv0': sys.argv[0], 'root': str(glob['ROOT'])})
                    glob['reservation'] = lambda _: dict(reservation)
                    glob['token'] = lambda _: (lambda: 'offline-token')
                    original_runtime = glob['runtime']
                    def runtime(arm, clock=None):
                        module = original_runtime(arm, clock)
                        original_prepare = module.prepare_review
                        def prepare(*pos, **kwargs):
                            sim = FakeV2Sim(prompt=TASK.prompt, task_id=TASK.task_id)
                            fakes.append(sim)
                            kwargs['sim_factory'] = lambda _: sim
                            kwargs['transport'] = offline_transport
                            return original_prepare(*pos, **kwargs)
                        module.prepare_review = prepare
                        return module
                    glob['runtime'] = runtime
            return tracer
        oldargv = sys.argv[:]
        entry = RUNNER if mode == 'direct' else WRAPPER
        sys.argv = [str(entry), *args]
        sys.settrace(tracer)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    runpy.run_path(str(entry), run_name='__main__')
                    exit_code = None
                except SystemExit as exc:
                    exit_code = exc.code
        finally:
            sys.settrace(None)
            sys.argv = oldargv
        clock = json.loads((session / 'service_clock.json').read_text())
        review_file = session / 'review/review.json'
        review = json.loads(review_file.read_text()) if review_file.exists() else None
        evidence_file = session / 'evidence_result.json'
        evidence = json.loads(evidence_file.read_text()) if evidence_file.exists() else None
        payload_hash = hashlib.sha256(json.dumps(captured, sort_keys=True).encode()).hexdigest()
        return {'mode': mode, 'arm': arm, 'failure_injected': failure, 'exit_code': exit_code,
            'loader_context': contexts, 'command_sequence': [c[0] for c in commands],
            'requests_captured': len(captured), 'captured_request_sha256': payload_hash,
            'fake_actions': sum(len(s.actions) for s in fakes),
            'fake_operations': [s.ops for s in fakes], 'plan': review['plan'] if review else None,
            'review_status': review['status'] if review else None,
            'review_approved': review['approved'] if review else None,
            'clock_paused_after_exit': clock['pause_start'] is not None,
            'automatic_review_started': any(e['event'] == 'automatic_review_start' for e in clock['events']),
            'error_status': evidence['status'] if evidence else None}

results = [one(mode, arm, failure) for arm in ['A', 'B'] for failure in [False, True]
           for mode in ['direct', 'wrapper']]
checks = []
for index in range(0, len(results), 2):
    a, b = results[index:index+2]
    expected = ['prepare'] if a['failure_injected'] else ['prepare', 'review-begin']
    check = {'arm': a['arm'], 'failure_injected': a['failure_injected'],
        'request_equal': a['captured_request_sha256'] == b['captured_request_sha256'],
        'same_request_count_one': a['requests_captured'] == b['requests_captured'] == 1,
        'same_outcome': a['exit_code'] == b['exit_code'] and a['plan'] == b['plan']
            and a['error_status'] == b['error_status'],
        'wrapper_expected_command_order': b['command_sequence'] == expected,
        'zero_actions': a['fake_actions'] == b['fake_actions'] == 0,
        'never_approved': a['review_approved'] is None and b['review_approved'] is None}
    checks.append(check)
report = {'scope': 'Existing FakeV2Sim and scripted transport only; socket connect and subprocess Popen blocked.',
    'wrapper_sha256': hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
    'runner_sha256': hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
    'checks': checks, 'results': results,
    'passed': all(all(v for k, v in c.items() if k not in ['arm', 'failure_injected']) for c in checks),
    'limits': 'Offline fake request equivalence does not prove identical network/service timing or rule out every runpy/environment interaction. No real simulator/model call or budget reservation.'}
(ROOT / 'wrapper_offline_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': checks, 'artifact': 'wrapper_offline_audit.json'}))
