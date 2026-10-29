"""Bounded repeated-observation Astra pilot. Reads credentials without echo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from gpt_field.astra import APIError, AstraClient, AstraConfig, BudgetExceeded, SchemaError, urllib_transport
from gpt_field.baselines import transform_field
from gpt_field.diagnostics import compare_cached_field
from gpt_field.experiment import source_manifest, write_json
from gpt_field.field import GridField
from gpt_field.observation import PolicyObservation
from gpt_field.schema import FieldPrediction, GridSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", default="runs/sim_cli_smoke/observation.json")
    parser.add_argument("--config", default="configs/astra_pilot.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    observation_path = Path(args.observation)
    raw_observation = observation_path.read_bytes()
    observation = PolicyObservation.from_dict(json.loads(raw_observation))
    config = AstraConfig(**json.loads(Path(args.config).read_text()))
    if config.model != "gpt-6-astra" or config.max_api_calls != 3:
        raise ValueError("This pilot requires gpt-6-astra and exactly three maximum API attempts")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    grid = GridSpec()
    write_json(output/"observation.json", observation.to_dict())
    write_json(output/"manifest.json", {
        "experiment": "Repeated predictions on one fixed real simulator observation",
        "model":config.model,"configuration":asdict(config),"grid":grid.to_dict(),
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "input_path":str(observation_path),"input_sha256":hashlib.sha256(raw_observation).hexdigest(),
        "semantics":"unitless spatial potential; no metric distance claim",
        "simulator_actions":0,"max_total_api_attempts_including_retries":3,
        "software":source_manifest(),
    })
    api_key = getpass.getpass("API key (hidden, memory only): ").strip()

    def redact(value):
        if isinstance(value,str):
            return value.replace(api_key,"[REDACTED]") if api_key else value
        if isinstance(value,list):
            return [redact(v) for v in value]
        if isinstance(value,dict):
            return {k:redact(v) for k,v in value.items()}
        return value

    def transport(endpoint,payload,headers,timeout_s):
        try:
            return redact(urllib_transport(endpoint,payload,headers,timeout_s))
        except APIError as exc:
            raise APIError(redact(str(exc)),response_body=redact(exc.response_body)) from None

    client = AstraClient(config,api_key=api_key,transport=transport,log_dir=output/"api")
    samples = []
    status = "running"
    failure = None
    try:
        client.preflight()
        while client.budget_snapshot()["api_calls"] < 3:
            sample_index = len(samples)
            print(json.dumps({"event":"requesting","sample":sample_index,
                              "attempts_used":client.budget_snapshot()["api_calls"]}),flush=True)
            prediction = client.predict(observation.to_dict(),grid,"field")
            field = GridField(grid,observation.state[:3],FieldPrediction.from_dict(prediction,grid))
            write_json(output/f"sample_{sample_index:02d}.field.json",field.to_dict())
            choices = compare_cached_field(field,observation.state[:3])
            write_json(output/f"sample_{sample_index:02d}.plans.json",choices)
            valid = np.asarray(field.valid_indices,dtype=int)
            values = np.asarray(prediction["values"])[valid]
            if len(valid):
                fitted = np.asarray(transform_field(field,"affine").prediction.values)[valid]
                affine_rmse = float(np.sqrt(np.mean((values-fitted)**2)))
                std = float(np.std(values))
                value_summary = {"min":float(values.min()),"max":float(values.max()),
                    "std":std,"affine_rmse":affine_rmse,
                    "affine_rmse_over_std":affine_rmse/std if std>0 else None}
            else:
                value_summary = None
            sample = {"sample":sample_index,"stage":prediction["stage"],
                "motion_mode":prediction["motion_mode"],"gripper_command":prediction["gripper_command"],
                "orientation_target_xyzw":prediction["orientation_target_xyzw"],
                "valid_points":len(valid),"blocked_points":len(prediction["blocked_indices"]),
                "unknown_points":len(prediction["unknown_indices"]),"value_summary":value_summary,
                "choices":{method:{key:result[key] for key in ("delta_xyz","status","valid_count","predicted_improvement")}
                           for method,result in choices.items()},
                "budget_after":client.budget_snapshot()}
            samples.append(sample)
            write_json(output/"summary.json",{"status":"running","samples":samples,
                "budget":client.budget_snapshot(),"model":config.model,"simulator_actions":0})
            print(json.dumps({"event":"sample_complete",**sample}),flush=True)
        status = "complete"
    except (APIError,SchemaError,BudgetExceeded) as exc:
        status = "blocked" if isinstance(exc,APIError) else "failed"
        failure = {"type":type(exc).__name__,"message":redact(str(exc)),
                   "api_error_body":redact(getattr(exc,"response_body",None))}
        print(json.dumps({"event":"stopped","status":status,"error":failure}),flush=True)
    finally:
        result = {"status":status,"model":config.model,"samples":samples,"error":failure,
                  "budget":client.budget_snapshot(),"simulator_actions":0,
                  "scope":"Static repeated-observation pilot; no task success or distance-accuracy claim."}
        write_json(output/"summary.json",result)
        # The full key must not appear in any persisted artifact, including HTTP errors.
        leak = any(api_key and api_key.encode() in p.read_bytes() for p in output.rglob("*") if p.is_file())
        print(json.dumps({"event":"finished","status":status,"valid_samples":len(samples),
                          "api_attempts":client.budget_snapshot()["api_calls"],
                          "credential_in_artifacts":bool(leak),"output":str(output)}),flush=True)
        if leak:
            raise RuntimeError("Credential unexpectedly appeared in pilot artifacts")
    return 0 if status=="complete" else 2


if __name__=="__main__":
    sys.exit(main())

