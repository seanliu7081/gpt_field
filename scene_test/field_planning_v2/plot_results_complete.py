"""Before/after drawer planning plots with explicit robot-prior provenance.

Only saved planning records, diagnostics, and reference RGB/calibration are
read. This visualization performs no planning, reconstruction, or simulation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BASE_PLOT_SOURCE = HERE.parent/"field_planning"/"plot_planning.py"
spec = importlib.util.spec_from_file_location("raw_planning_plot_readonly", BASE_PLOT_SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SNAPSHOTS = base.SNAPSHOTS
METHODS = ("astra", "depthpro")
LABELS = base.METHOD_LABELS
COLORS = {"valid":"#25956a", "invalid":"#df472f", "unscored":"#a57bc0", "refused":"#b8bdc4",
          "field_free":"#3aa87c", "self_prior":"#be77c1", "occupied":"#df472f", "unknown":"#77a0c9"}


def record_key(record):
    return record["snapshot"], record["method"], str(record["case_id"]), int(record["seed"])


def indexed_records(records):
    result = {record_key(r):r for r in records}
    if len(result) != len(records):
        raise ValueError("Duplicate snapshot/method/case/seed records")
    return result


def assert_same_queries(before, after):
    """A before/after chart is meaningful only for exactly paired endpoints."""
    left, right = indexed_records(before), indexed_records(after)
    if set(left) != set(right):
        raise ValueError("Before and after must contain the same snapshot/method/case/seed queries")
    for key in left:
        for name in ("start_q", "goal_q"):
            a, b = np.asarray(left[key][name]), np.asarray(right[key][name])
            if a.shape != (7,) or b.shape != (7,) or not np.allclose(a,b,rtol=0,atol=1e-12):
                raise ValueError(f"Before/after endpoint mismatch: {key}, {name}")
    return {"paired_queries":len(left), "endpoint_tolerance_rad":1e-12,
            "meaning":"same original robot joint endpoints and search seeds; representation semantics changed"}


def outcomes(records, snapshot, method):
    rows = [r for r in records if r["snapshot"] == snapshot and r["method"] == method]
    if not rows:
        raise ValueError(f"Missing query group: {snapshot}/{method}")
    accepted = [r for r in rows if base.returned(r)]
    valid = sum(base.truth_clear(r) is True for r in accepted)
    invalid = sum(base.truth_clear(r) is False for r in accepted)
    return {"queries":len(rows),"returned":len(accepted),"valid":valid,"invalid":invalid,
            "unscored":len(accepted)-valid-invalid,"refused":len(rows)-len(accepted)}


def initial_states(diagnostic, *, revised):
    """Split final free classifications by SELF-prior origin without overlap."""
    counts = diagnostic["field_points"]
    total = int(counts["points"])
    free, occupied, unknown = (int(counts[k]) for k in ("free","occupied","unknown"))
    if total <= 0 or min(free,occupied,unknown) < 0 or free+occupied+unknown != total:
        raise ValueError("Initial sample classification counts do not sum to total")
    layers = diagnostic.get("field_support_layers",{})
    if revised and diagnostic["method"] == "depthpro" and "self_prior_free" not in layers:
        raise ValueError("Revised DepthPro diagnostic must identify SELF-prior free samples")
    prior = int(layers.get("self_prior_free",0)) if revised else 0
    if prior < 0 or prior > free:
        raise ValueError("SELF-prior free count must be a subset of final free samples")
    return {"points":total,"field_free":free-prior,"self_prior":prior,
            "occupied":occupied,"unknown":unknown,"layers":layers,
            "fraction_semantics":"fraction of fixed robot surface samples, not scene volume or uniform area"}


def selected_pair(records, snapshot):
    """Expose a new predicted-path failure before displaying a success example."""
    scene = [r for r in records if r["snapshot"] == snapshot]
    order = lambda r:(str(r["case_id"]),int(r["seed"]),str(r["method"]))
    predicted = sorted((r for r in scene if r["method"] in METHODS),key=order)
    failed = [r for r in predicted if base.returned(r) and base.truth_clear(r) is False]
    valid = [r for r in predicted if base.returned(r) and base.truth_clear(r) is True]
    oracle = sorted((r for r in scene if r["method"] == "oracle" and base.returned(r)
                     and base.truth_clear(r) is True),key=order)
    chosen = (failed or valid or oracle or predicted)[0]
    matching = {}
    for method in (*METHODS,"oracle"):
        candidates = [r for r in scene if r["method"] == method and r["case_id"] == chosen["case_id"] and r["seed"] == chosen["seed"]]
        if len(candidates) != 1:
            raise ValueError("Example requires exactly one paired record for each method")
        matching[method] = candidates[0]
    return matching, "predicted path invalid at GT sample" if failed else "predicted path passed GT samples" if valid else "oracle reference"


def outcome_legend(ax):
    """Select every outcome explicitly; Matplotlib artist order mixes lines and bars."""
    ordered = ("Returned / GT samples clear", "Returned / GT sample invalid",
               "Returned / not scored", "No path returned")
    handles, labels = ax.get_legend_handles_labels()
    result = []
    for label in ordered:
        matches = [handle for handle, name in zip(handles, labels) if name == label]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one outcome legend artist for {label}")
        result.append(matches[0])
    return result, list(ordered)


def comparison_figure(output, before, after):
    fig, axes = plt.subplots(2,2,figsize=(12,8.7),constrained_layout=True)
    report=[]
    for row,snapshot in enumerate(SNAPSHOTS):
        for col,method in enumerate(METHODS):
            ax=axes[row,col]
            old,new=outcomes(before,snapshot,method),outcomes(after,snapshot,method)
            reference=outcomes(after,snapshot,"oracle")
            bottom=np.zeros(2)
            for key,label in (("valid","Returned / GT samples clear"),("invalid","Returned / GT sample invalid"),
                              ("unscored","Returned / not scored"),("refused","No path returned")):
                heights=np.array([old[key],new[key]])
                ax.bar([0,1],heights,bottom=bottom,color=COLORS[key],label=label,width=.58)
                for x,value in enumerate(heights):
                    if value:
                        ax.text(x,bottom[x]+value/2,str(value),ha="center",va="center",fontsize=11)
                bottom+=heights
            ax.axhline(reference["valid"],color="#292929",ls=":",lw=1.3,
                       label=f"Oracle passed GT samples: {reference['valid']} / {reference['queries']}")
            ax.set_xticks([0,1],["Original raw field","Revised planning space"])
            ax.set(title=f"{base.SCENE_LABELS[snapshot]} · {LABELS[method]}",ylabel="Fixed planning queries",
                   ylim=(0,max(old["queries"],new["queries"])*1.12))
            ax.spines[["top","right"]].set_visible(False)
            ax.text(.98,.97,f"Oracle: {reference['valid']}/{reference['queries']} pass",transform=ax.transAxes,ha="right",va="top",fontsize=9)
            report.append({"snapshot":snapshot,"method":method,"before":old,"after":new,"oracle":reference})
    handles,labels=outcome_legend(axes[0,0])
    fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=9)
    fig.suptitle("Before / after: does each saved space support the original LIBERO robot?\n"
                 "Same drawer states, joint endpoints, seeds and planner. GT checks finite path samples.\n"
                 "A returned path may be wrong; no returned path is not zero-collision success.",fontsize=12)
    fig.savefig(output/"before_after_outcomes.png",dpi=170);plt.close(fig)
    return report


def initial_figure(output,before,after):
    old={(r["snapshot"],r["method"]):r for r in before["start_diagnostics"]}
    new={(r["snapshot"],r["method"]):r for r in after["start_diagnostics"]}
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    report=[]
    for row,snapshot in enumerate(SNAPSHOTS):
        for col,method in enumerate(METHODS):
            ax=axes[row,col];left=initial_states(old[snapshot,method],revised=False);right=initial_states(new[snapshot,method],revised=True)
            if left["points"]!=right["points"]:
                raise ValueError("Before/after diagnostics must use the same number of robot surface samples")
            bottom=np.zeros(2)
            for key,label in (("field_free","Field / ray-inferred free"),("self_prior","Initial robot SELF prior"),
                              ("occupied","Predicted occupied"),("unknown","Unknown / ambiguous")):
                heights=np.array([left[key]/left["points"],right[key]/right["points"]])*100
                ax.bar([0,1],heights,bottom=bottom,color=COLORS[key],width=.58,label=label)
                for x,value in enumerate(heights):
                    if value>=3:
                        ax.text(x,bottom[x]+value/2,f"{value:.1f}%",ha="center",va="center",fontsize=10)
                bottom+=heights
            ax.set_xticks([0,1],["Original raw field","Revised planning space"])
            ax.set(title=f"{base.SCENE_LABELS[snapshot]} · {LABELS[method]}",ylabel="Original robot surface samples (%)",ylim=(0,110))
            ax.spines[["top","right"]].set_visible(False)
            ax.text(.5,.98,f"{left['points']:,} fixed surface samples",transform=ax.transAxes,ha="center",va="top",fontsize=9)
            report.append({"snapshot":snapshot,"method":method,"before":left,"after":right})
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=9)
    fig.suptitle("What changed at the robot's original starting configuration?\n"
                 "Purple = known initial robot volume prior, NOT newly observed camera free space.\n"
                 "Percentages describe fixed surface samples, not scene volume or uniformly weighted surface area.",fontsize=12)
    fig.savefig(output/"initial_space_states.png",dpi=170);plt.close(fig)
    return report


def path_figure(output,records,inputs):
    fig,axes=plt.subplots(2,3,figsize=(15,10.8),constrained_layout=True);report=[]
    for row,snapshot in enumerate(SNAPSHOTS):
        selected,selection_reason=selected_pair(records,snapshot)
        image,calibration=inputs[snapshot]
        for col,method in enumerate((*METHODS,"oracle")):
            record=selected[method];ax=axes[row,col];base.image_axes(ax,image,calibration)
            accepted=base.returned(record);truth=base.truth_clear(record)
            color="#ff4a31" if truth is False else "#fff000"
            details=base.draw_recorded_path(ax,record["eef_path_world_m"],calibration,accepted=accepted,color=color)
            base.endpoint(ax,record["start_eef_world_m"],calibration,"o","#00d9b3","S")
            base.endpoint(ax,record["goal_eef_world_m"],calibration,"*","#fa5ed0","G")
            outcome="NO PATH RETURNED" if not accepted else "RETURNED / GT SAMPLE INVALID" if truth is False else "RETURNED / GT SAMPLES CLEAR" if truth is True else "RETURNED / UNSCORED"
            ax.set_title(f"{base.SCENE_LABELS[snapshot]} · {LABELS[method]}\n{outcome}",fontsize=10)
            reason=str(record.get("reason") or record["status"])
            text=f"case {record['case_id']} / seed {record['seed']}\n{reason}"
            first=(record.get("truth") or {}).get("first_invalid_sample")
            if first:
                contacts=first.get("contacts",[])
                if contacts:
                    text+=f"\nFirst invalid: {contacts[0].get('geom1','?')}\nvs {contacts[0].get('geom2','?')}"
                else:
                    text+="\nInvalid GT sample (see metrics for reason)"
            first_eef=(record.get("truth") or {}).get("first_invalid_eef_world_m")
            if first_eef is not None:
                details["first_invalid_eef_visible"]=base.endpoint(ax,first_eef,calibration,"X","#ff2f20","first invalid")
            ax.text(.01,.01,text,transform=ax.transAxes,va="bottom",fontsize=7.8,
                    bbox={"facecolor":"white","alpha":.9,"edgecolor":"none","pad":4})
            details.update({"snapshot":snapshot,"method":method,"case_id":record["case_id"],"seed":record["seed"],
                            "returned":accepted,"truth_valid_at_all_samples":truth,"selection_reason":selection_reason})
            report.append(details)
    fig.suptitle("Revised space: recorded end-effector projections, with full-robot GT evaluation\n"
                 "Red marks an entire path containing a GT-invalid sample; it does NOT mark only colliding segments.\n"
                 "Example rule: first predicted-path GT failure, otherwise first success. Same case/seed in each row; no line for refusals.",fontsize=12)
    fig.savefig(output/"path_examples.png",dpi=170);plt.close(fig)
    return report


def load_metrics(path,consumed):
    path=Path(path).resolve();manifest_path=path.parent/"manifest.json"
    base.checked(manifest_path,None,consumed)
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("status")!="complete" or not manifest.get("source_unchanged") or not manifest.get("inputs_unchanged"):
        raise ValueError("Metrics require a completed intact evaluation manifest")
    base.checked(path,manifest["artifacts_sha256"][path.name],consumed)
    data=json.loads(path.read_text())
    if data.get("status")!="complete":
        raise ValueError("Cannot plot incomplete metrics")
    if {r["snapshot"] for r in data["records"]}!=set(SNAPSHOTS):
        raise ValueError("Only the same two drawer snapshots are allowed")
    return data


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics",type=Path,required=True)
    parser.add_argument("--before",type=Path,default=HERE.parent/"field_planning"/"runs"/"evaluation_002"/"metrics.json")
    parser.add_argument("--references",type=Path,default=HERE.parent/"rgb_scene"/"references_004")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError("Output must be a new directory")
    consumed={};source_hash=base.sha(__file__);base.checked(BASE_PLOT_SOURCE,None,consumed)
    before=load_metrics(args.before,consumed);after=load_metrics(args.metrics,consumed)
    pairs=assert_same_queries(before["records"],after["records"])
    inputs={}
    for snapshot in SNAPSHOTS:
        folder=args.references/"snapshots"/snapshot
        rgb=base.checked(folder/"agentview_upright.png",None,consumed)
        calibration_path=base.checked(folder/"evaluator"/"calibration.json",None,consumed)
        calibration=json.loads(calibration_path.read_text())
        with Image.open(rgb) as image:pixels=np.asarray(image.convert("RGB")).copy()
        if pixels.shape!=(calibration["height"],calibration["width"],3):raise ValueError("RGB/calibration size mismatch")
        inputs[snapshot]=(pixels,calibration)
    args.output.mkdir(parents=True);started=time.monotonic()
    comparison=comparison_figure(args.output,before["records"],after["records"])
    initial=initial_figure(args.output,before,after)
    paths=path_figure(args.output,after["records"],inputs)
    unchanged={p:base.sha(p)==digest for p,digest in consumed.items()}
    if not all(unchanged.values()) or base.sha(__file__)!=source_hash:raise RuntimeError("Plot source/input changed")
    report={"status":"complete","source_sha256":source_hash,"consumed_sha256":consumed,"inputs_unchanged":unchanged,
            "query_pairing":pairs,"before_after":comparison,"initial_classifications":initial,"path_examples":paths,
            "artifacts_sha256":{p.name:base.sha(p) for p in args.output.glob("*.png")},
            "scope":"revised planning semantics, not a new camera observation or a geometric-reconstruction improvement claim",
            "elapsed_s":time.monotonic()-started}
    base.dump(args.output/"manifest.json",report)
    print(json.dumps({"status":"complete","output":str(args.output),"figures":list(report["artifacts_sha256"])}))


if __name__=="__main__":main()
