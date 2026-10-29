"""One approval-gated planner request on saved images; never starts a simulator."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    spec_path = HERE / 'validation_specification.json'
    spec = read(spec_path)
    for path, digest in spec['files_sha256'].items():
        if sha(path) != digest:
            raise SystemExit('Prepared validation input changed: ' + path)
    if sha(HERE / 'validation_runtime.json') != spec['runtime_inventory_sha256']:
        raise SystemExit('Validation runtime inventory changed')
    for path, digest in read(HERE / 'validation_runtime.json').items():
        if sha(path) != digest:
            raise SystemExit('Prepared runtime changed: ' + path)
    if not args.live:
        print(json.dumps({'status': 'ready_awaiting_approval', 'model': 'gpt-6-astra',
                          'effort': 'high', 'maximum_requests': 1, 'robot_actions': 0,
                          'request_deadline_s': 180, 'specification_sha256': sha(spec_path)}, indent=2))
        return 0
    approval_path = HERE / 'validation_approval.json'
    if not approval_path.exists():
        raise SystemExit('Explicit user approval for this planning-only validation is required')
    approval = read(approval_path)
    if (approval.get('approved') is not True or approval.get('attempts') != 1
            or approval.get('scope') != 'one_planning_request_zero_robot_actions'
            or approval.get('specification_sha256') != sha(spec_path)
            or not approval.get('user_message') or not approval.get('approved_at')):
        raise SystemExit('Approval does not match this prepared validation')
    output = HERE / 'live_validation'
    if output.exists():
        raise SystemExit('This validation already has an output directory; no retry')
    with (HERE / 'validation_claimed.json').open('x') as stream:
        json.dump({'approval_sha256': sha(approval_path), 'specification_sha256': sha(spec_path),
                   'claimed_at': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    output.mkdir()
    sys.path.insert(0, str(ROOT / 'src'))
    from gpt_field.astra import AstraConfig
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    from gpt_field.hybrid_subscription import ChatGPTPlanHybridClient
    from gpt_field.agent_speedup_v2_user_check import plan_validation
    from gpt_field.agent_speedup_v2_user_check_client import UserCheckV2Client
    from gpt_field.agent_speedup_v2_user_check_transport import configure_client_transport
    request = read(HERE / 'validation_request.json')
    client = UserCheckV2Client(AstraConfig(max_api_calls=1), task_prompt=spec['task_prompt'],
        token_provider=SubscriptionCredentials().access_token, reasoning_effort='high',
        log_dir=output / 'requests')
    configure_client_transport(client, request_deadline_s=180, stream_progress=True)
    payload = client.build_payload(request, mode='plan')
    if payload != read(HERE / 'validation_payload.json'):
        raise SystemExit('Actual payload differs from prepared payload')
    started = time.monotonic()
    result = {'model': 'gpt-6-astra', 'reasoning_effort': 'high', 'billing': 'chatgpt_subscription',
              'robot_actions': 0, 'simulator_started': False, 'task_completion_time_s': None,
              'scope': 'planner_only_on_saved_task1_images', 'completed_response': False}
    try:
        # The inherited one-attempt method uses the same payload/validator/logging,
        # but deliberately bypasses the outer client's transient retry loop.
        plan = ChatGPTPlanHybridClient.predict(client, request, mode='plan')
        checks = plan_validation(plan, require_native=False)
        (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        result.update(status='valid_plan' if checks['valid'] else 'invalid_plan',
                      completed_response=True, plan_validation=checks)
    except BaseException as error:
        # Detailed errors stay in the existing sanitized request logs.
        result.update(status='request_failed', error_code=getattr(error, 'error_code', None))
    finally:
        result.update(planning_elapsed_s=time.monotonic() - started,
                      model_calls=len(list((output / 'requests').glob('*.request.json'))))
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'valid_plan' else 1


if __name__ == '__main__':
    raise SystemExit(main())
