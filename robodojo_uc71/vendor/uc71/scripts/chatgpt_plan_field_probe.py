"""One subscription-only Astra field and local plan from the recorded initial scene.

This is an access/inference check, not an episode: it executes no robot actions,
does not score task success, and never falls back to API-key billing or a model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from gpt_field.astra import AstraConfig
from gpt_field.experiment import source_manifest, write_json
from gpt_field.field_history import FieldHistoryPolicy, PROMPT_VERSION
from gpt_field.observation import PolicyObservation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "runs/astra_field_history_001/trace.jsonl"
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
MODEL = "gpt-6-astra"


def initial_observation(trace_path):
    """Read only the reset record and expose exactly the observation allowlist."""
    with Path(trace_path).open("rb") as source:
        raw = source.readline(8 * 1024 * 1024)
    record = json.loads(raw)
    if record.get("event") != "reset":
        raise ValueError("Expected the first reset observation of the bowl episode")
    observation = PolicyObservation.from_dict(record["observation"])
    if observation.prompt != TASK:
        raise ValueError("Recorded observation is for a different task")
    return observation, hashlib.sha256(raw).hexdigest()


def probe(output, token_provider, *, trace_path=DEFAULT_TRACE, model_catalog=None,
          client_factory=None):
    from gpt_field.chatgpt_plan import ChatGPTPlanFieldClient, list_plan_models

    catalog = model_catalog or list_plan_models
    client_factory = client_factory or ChatGPTPlanFieldClient
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh probe directory; existing artifacts are preserved")
    observation, reset_digest = initial_observation(trace_path)
    config = AstraConfig(model=MODEL, max_api_calls=1, timeout_s=180)
    client = client_factory(config, token_provider=token_provider, log_dir=output / "requests")
    policy = FieldHistoryPolicy(client)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "experiment": "chatgpt_subscription_field_access_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "billing": "chatgpt_subscription", "model": MODEL,
        "configuration": asdict(config), "prompt_version": PROMPT_VERSION,
        "recorded_initial_observation": str(trace_path),
        "reset_record_sha256": reset_digest, "software": source_manifest(),
        "model_fallback": False, "api_key_fallback": False,
        "maximum_inference_attempts": 1, "simulator_actions": 0,
        "evaluation_scored": False, "policy_action_step_limit": None,
        "scope": "Static initial-scene field generation and local planning, not an episode.",
        "protocol_changes": ["ChatGPT plan OAuth", "stream=true",
            "unsupported max_output_tokens omitted; service controls output limit"],
    })
    status, error, validated, plans = "starting", None, 0, 0
    try:
        models = catalog(token_provider)
        write_json(output / "models.json", {"models": models,
            "catalog_is_not_entitlement_proof": True})
        if not any(item.get("slug") == MODEL for item in models):
            status = "astra_not_in_account_catalog"
        else:
            write_json(output / "observation.json", observation.to_dict())
            decision = policy.act(observation)
            validated, plans = 1, 1
            write_json(output / "field.json", policy.field.to_dict())
            write_json(output / "plan.json", {"action_not_executed": list(decision.action),
                "diagnostics": decision.diagnostics})
            status = "subscription_field_verified"
    except KeyboardInterrupt:
        status, error = "interrupted", "User interrupted the subscription probe"
    except Exception as exc:
        # Never put credential-provider, HTTP or token-exchange exception text in
        # public artifacts. The transport separately records sanitized failures.
        status = "subscription_probe_failed"
        error = {"type": type(exc).__name__,
                 "message": "Subscription probe did not complete"}
        http_status = getattr(exc, "http_status", None)
        error_code = getattr(exc, "error_code", None)
        if type(http_status) is int and 100 <= http_status <= 599:
            error["http_status"] = http_status
        if isinstance(error_code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", error_code):
            error["error_code"] = error_code
    response = client.last_response
    access_verified = (isinstance(response, dict) and response.get("status") == "completed"
                       and not response.get("error") and response.get("model") == MODEL)
    summary = {"status": status, "model": MODEL, "billing": "chatgpt_subscription",
        "astra_subscription_access_verified": access_verified,
        "field_pipeline_verified": status == "subscription_field_verified",
        "validated_field_generations": validated, "local_plans": plans,
        "simulator_actions": 0, "evaluation_completed": False,
        "success_rate": None, "usage": client.budget_snapshot(),
        "api_key_calls": 0, "api_key_fallback": False, "error": error,
        "scope": "One static field/plan access probe, not a task-performance evaluation."}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    args = parser.parse_args(argv)
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)
    result = probe(args.output, credentials.access_token, trace_path=args.trace)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["field_pipeline_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
