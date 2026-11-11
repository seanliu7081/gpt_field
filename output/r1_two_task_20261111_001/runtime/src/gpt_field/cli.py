"""Command line entry points; only infer and libero may call a live model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys

from .astra import AstraClient, AstraConfig
from .diagnostics import compare_cached_field, run_demo
from .experiment import run_libero, source_manifest, write_json
from .field import GridField
from .libero import LiberoConfig, SimProcess
from .observation import PolicyObservation
from .planner import PlannerConfig
from .policy import FieldPolicy, OpenPIPolicy
from .report import distance_fidelity, load_results, summarize
from .schema import FieldPrediction, GridSpec


def _json(path):
    return json.loads(Path(path).read_text())


def _api_config(args):
    return AstraConfig(**_json(args.api_config)) if args.api_config else AstraConfig(max_api_calls=3,max_wall_time_s=900)


def _indices(value, limit):
    if value == "all":
        return list(range(limit))
    values = [int(x) for x in value.split(",")]
    if not values or len(values) != len(set(values)) or any(x<0 or x>=limit for x in values):
        raise ValueError(f"indices must be unique integers in [0,{limit-1}], or all")
    return values


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    commands.add_parser("doctor",help="inspect local dependencies and key presence; never sends requests")
    demo = commands.add_parser("demo",help="run synthetic oracle diagnostics, no model/API")
    demo.add_argument("--output",required=True)
    demo.add_argument("--seed",type=int,default=7)
    demo.add_argument("--max-steps",type=int,default=80)
    replay = commands.add_parser("replay",help="compare transformations of one cached field")
    replay.add_argument("input",help="JSON with grid, center, prediction, and optional position")
    replay.add_argument("--output",required=True)
    fidelity = commands.add_parser("fidelity",help="evaluate separate explicitly metric auxiliary references")
    fidelity.add_argument("input")
    fidelity.add_argument("--output")
    report = commands.add_parser("report",help="task-balanced and paired episode summaries")
    report.add_argument("runs",nargs="+")
    report.add_argument("--output")
    infer = commands.add_parser("infer",help="predict from a saved allowlisted observation")
    infer.add_argument("observation")
    infer.add_argument("--mode",choices=("field","waypoint","direct"),default="field")
    infer.add_argument("--api-config")
    infer.add_argument("--output",required=True,help="new directory for prediction, grid and raw logs")
    infer.add_argument("--dry-run",action="store_true",help="write exact request payload without key/network")
    infer.add_argument("--grid-size",type=int,default=9)
    infer.add_argument("--grid-spacing",type=float,default=0.02)
    for name in ("libero", "sim-smoke"):
        command = commands.add_parser(name,help="run benchmark policy" if name=="libero" else "one simulator reset and neutral action without API")
        command.add_argument("--libero-root",default=os.environ.get("LIBERO_ROOT"))
        command.add_argument("--sim-python",default=sys.executable)
        command.add_argument("--suite",choices=("libero_spatial","libero_object","libero_goal","libero_10","libero_90"),default="libero_spatial")
        command.add_argument("--seed",type=int,default=7)
        command.add_argument("--output",required=True)
        command.add_argument("--max-steps",type=int)
        if name=="libero":
            command.add_argument("--method",choices=(*FieldPolicy.METHODS,"pi05"),default="field")
            command.add_argument("--tasks",default="0",help="comma-separated indices or all")
            command.add_argument("--episodes",default="0",help="official initial-state indices, comma-separated or all")
            command.add_argument("--api-config")
            command.add_argument("--phase",choices=("exploratory","frozen"),default="exploratory")
            command.add_argument("--resume",action="store_true")
            command.add_argument("--no-video",action="store_true")
            command.add_argument("--pi05-host",default="localhost")
            command.add_argument("--pi05-port",type=int,default=8000)
            command.add_argument("--reference-manifest",help="JSON recording exact external checkpoint and server revision")
            command.add_argument("--grid-size",type=int,default=9)
            command.add_argument("--grid-spacing",type=float,default=0.02)
            command.add_argument("--step-m",type=float,default=0.01)
            command.add_argument("--horizon",type=int,default=3)
            command.add_argument("--update-steps",type=int,default=5)
    return parser


def dispatch(args):
    if args.command == "doctor":
        return {**source_manifest(),"openai_api_key_present":bool(os.environ.get("OPENAI_API_KEY")),
            "optional_packages":{name:importlib.util.find_spec(name) is not None for name in ("openpi_client","libero","robosuite")},
            "note":"Simulator dependencies may live in a separate --sim-python runtime."}
    if args.command == "demo":
        if args.max_steps<=0:
            raise ValueError("max_steps must be positive")
        return run_demo(args.output,seed=args.seed,max_steps=args.max_steps)
    if args.command == "replay":
        data = _json(args.input)
        grid = GridSpec(**data["grid"])
        field = GridField(grid,data["center"],FieldPrediction.from_dict(data["prediction"],grid))
        result = compare_cached_field(field,data.get("position",data["center"]))
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        write_json(Path(args.output),result)
        return {"output":args.output,"actions":{k:v["delta_xyz"] for k,v in result.items()}}
    if args.command in ("fidelity","report"):
        result = distance_fidelity(_json(args.input)) if args.command=="fidelity" else summarize(load_results(args.runs))
        if args.output:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            write_json(Path(args.output),result)
        return result
    if args.command == "infer":
        obs = PolicyObservation.from_dict(_json(args.observation))
        grid = GridSpec(size=args.grid_size,spacing_m=args.grid_spacing)
        output = Path(args.output)
        output.mkdir(parents=True,exist_ok=False)
        client = AstraClient(_api_config(args),log_dir=output/"api")
        if args.dry_run:
            write_json(output/"request.json",client.build_payload(obs.to_dict(),grid,args.mode))
            return {"dry_run":True,"output":str(output/"request.json"),"api_calls":0}
        try:
            prediction = client.predict(obs.to_dict(),grid,args.mode)
            write_json(output/"prediction.json",prediction)
            if args.mode=="field":
                write_json(output/"field.json",{"grid":grid.to_dict(),"center":list(obs.state[:3]),"prediction":prediction})
            return {"output":str(output),"budget":client.budget_snapshot()}
        finally:
            write_json(output/"budget.json",client.budget_snapshot())
    horizons = {"libero_spatial":220,"libero_object":280,"libero_goal":300,"libero_10":520,"libero_90":400}
    sim_config = LiberoConfig(suite=args.suite,seed=args.seed,max_steps=args.max_steps if args.max_steps is not None else horizons[args.suite],
        libero_root=args.libero_root,python_executable=args.sim_python)
    if args.command=="sim-smoke":
        output = Path(args.output)
        output.mkdir(parents=True,exist_ok=False)
        with SimProcess(sim_config) as sim:
            tasks = sim.list_tasks()
            obs = sim.reset(0,0)
            write_json(output/"observation.json",obs.to_dict())
            _,success = sim.step([0,0,0,0,0,0,-1])
            result = {"api_calls":0,"tasks":len(tasks),"metadata":sim.metadata,"status":sim.status,
                      "observation_keys":list(obs.to_dict()),"success":success}
            write_json(output/"result.json",result)
        return result
    if args.method=="pi05":
        if not args.reference_manifest:
            raise ValueError("pi05 requires --reference-manifest documenting exact checkpoint, server source and config")
        reference = _json(args.reference_manifest)
        if not isinstance(reference,dict) or not all(reference.get(k) for k in ("checkpoint","server_revision","config")):
            raise ValueError("reference manifest requires checkpoint, server_revision and config")
        client = None
        policy = OpenPIPolicy(args.pi05_host,args.pi05_port,args.update_steps)
    else:
        reference = None
        client = AstraClient(_api_config(args),log_dir=Path(args.output)/"api")
        policy = FieldPolicy(client,method=args.method,seed=args.seed,
            grid=GridSpec(size=args.grid_size,spacing_m=args.grid_spacing),
            planner_config=PlannerConfig(step_m=args.step_m,horizon=args.horizon),
            update_steps=args.update_steps)
    return run_libero(args.output,policy,sim_config,
        task_ids=_indices(args.tasks,90 if args.suite=="libero_90" else 10),
        episode_indices=_indices(args.episodes,50),client=client,phase=args.phase,resume=args.resume,
        save_video=not args.no_video,reference_metadata=reference)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 3 if result.get("status")=="budget_exceeded" else 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
