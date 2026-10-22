"""Visualize saved drawer planning results without inventing rejected paths.

This module consumes recorded geometry samples, robot end-effector positions,
and scores. It does not call a field, planner, inverse kinematics, or simulator.
Camera calibration is used only for this evaluator-side visualization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SNAPSHOTS = ("drawer003_closed_step60", "drawer003_open_step552")
SCENE_LABELS = {SNAPSHOTS[0]: "Drawer / closed", SNAPSHOTS[1]: "Drawer / open"}
METHODS = ("astra", "depthpro", "oracle")
METHOD_LABELS = {"astra": "Astra field", "depthpro": "DepthPro field", "oracle": "MuJoCo oracle"}
STATE_COLORS = {0: "#2aa876", 1: "#e34a33", 2: "#268bd2"}
STATE_LABELS = {0: "Predicted free", 1: "Predicted occupied", 2: "Unknown / ambiguous"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def checked(path, expected, consumed):
    path = Path(path).resolve()
    observed = sha(path)
    if expected is not None and observed != expected:
        raise ValueError(f"Artifact SHA mismatch: {path}")
    consumed[str(path)] = observed
    return path


def project_world(points, calibration):
    """Project measured world points through the archived upright CV camera.

    Invalid or behind-camera points become NaN, never reflected to the image.
    The returned mask marks only pixel centres within the reference image.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("World points must have shape (N,3)")
    transform = np.asarray(calibration["T_camera_cv_from_world"], dtype=float)
    intrinsics = np.asarray(calibration["K"], dtype=float)
    if transform.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("Invalid camera matrix shape")
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    forward = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    uv = np.full((len(points), 2), np.nan)
    projected = camera[forward] @ intrinsics.T
    uv[forward] = projected[:, :2] / projected[:, 2:3]
    in_image = (forward & (uv[:, 0] >= -.5) & (uv[:, 0] < calibration["width"]-.5)
                & (uv[:, 1] >= -.5) & (uv[:, 1] < calibration["height"]-.5))
    return uv, camera[:, 2], in_image


def adjacent_segments(uv):
    """Never bridge a missing/behind-camera segment in a recorded path."""
    uv = np.asarray(uv, dtype=float)
    if len(uv) < 2:
        return np.empty((0, 2, 2), dtype=float)
    segments = np.stack((uv[:-1], uv[1:]), axis=1)
    return segments[np.isfinite(segments).all(axis=(1, 2))]


def draw_recorded_path(ax, path_world_m, calibration, *, accepted, color, label=None):
    """Draw only a returned path, never a straight line between refused endpoints."""
    points = np.asarray(path_world_m, dtype=float).reshape((-1, 3))
    if not accepted:
        if len(points):
            raise ValueError("Refused query unexpectedly contains a returned path")
        return {"returned_path_points": 0, "drawn_segments": 0, "in_image_points": 0}
    if len(points) < 2:
        raise ValueError("Accepted query must contain at least two recorded positions")
    uv, _, visible = project_world(points, calibration)
    segments = adjacent_segments(uv)
    ax.add_collection(LineCollection(segments, colors=[color], linewidths=2.0, label=label, zorder=5))
    return {"returned_path_points": len(points), "drawn_segments": len(segments),
            "in_image_points": int(visible.sum())}


def sample_indices(states, limit=12000):
    """Deterministic equal-stride display sampling, preserving rare state classes."""
    states = np.asarray(states)
    if states.ndim != 1 or not np.isin(states, list(STATE_COLORS)).all():
        raise ValueError("Point states must be 0=clear, 1=blocked, or 2=unknown")
    if limit < 3:
        raise ValueError("Display limit must be at least three")
    groups = [np.flatnonzero(states == state) for state in STATE_COLORS]
    budget = max(1, limit // 3)
    selected = [g if len(g) <= budget else g[np.linspace(0, len(g)-1, budget, dtype=int)] for g in groups]
    return np.sort(np.concatenate(selected))


def draw_robot_points(ax, points_world_m, states, calibration, *, limit=12000):
    """Overlay recorded samples of original robot geometry, with explicit clipping."""
    points = np.asarray(points_world_m, dtype=float)
    states = np.asarray(states)
    if points.shape != (len(states), 3) or not np.isfinite(points).all():
        raise ValueError("Expected finite original-geometry points and one state per point")
    selected = sample_indices(states, limit)
    uv, _, visible = project_world(points[selected], calibration)
    # Draw unknown then clear then blocked so rare collision evidence is visible.
    for state in (2, 0, 1):
        mask = visible & (states[selected] == state)
        ax.scatter(uv[mask, 0], uv[mask, 1], s=1.3 if state != 1 else 3.5,
                   color=STATE_COLORS[state], alpha=.75, linewidths=0, zorder=4+int(state == 1))
    return {"original_sample_count": len(points), "display_sample_count": len(selected),
            "in_image_display_samples": int(visible.sum()),
            "all_sample_counts": {STATE_LABELS[k].lower(): int(np.sum(states == k)) for k in STATE_COLORS},
            "display_rule": "deterministic per-state equal-stride sample; no visibility/occlusion filtering",
            "meaning": "recorded samples of original LIBERO collision geometry; x-ray overlay, not a replacement robot"}


def image_axes(ax, image, calibration):
    ax.imshow(image)
    ax.set_xlim(-.5, calibration["width"]-.5)
    ax.set_ylim(calibration["height"]-.5, -.5)
    ax.set_axis_off()


def endpoint(ax, point, calibration, marker, color, label):
    uv, _, visible = project_world(np.asarray(point).reshape((1, 3)), calibration)
    if visible[0]:
        ax.scatter(*uv[0], marker=marker, s=50, color=color, edgecolors="white", linewidths=.8, zorder=7)
        ax.annotate(label, uv[0], xytext=(5, -7), textcoords="offset points", color=color,
                    fontsize=8, weight="bold", bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none", "pad":1})
    return bool(visible[0])


def returned(record):
    """The saved path, rather than a display-specific status synonym, is decisive."""
    path = record.get("eef_path_world_m", [])
    if len(path) == 1:
        raise ValueError("A returned EEF path must have at least two positions")
    return len(path) > 0


def truth_clear(record):
    """The evaluator checks finite joint samples, not a continuous swept volume."""
    truth = record.get("truth") or {}
    if "valid_at_all_samples" in truth:
        return truth["valid_at_all_samples"]
    return truth.get("path_collision_free")


def choose_records(records, snapshot):
    """Predeclared display selection: first oracle-validated case, lowest seed."""
    scene = [r for r in records if r["snapshot"] == snapshot]
    order = lambda r: (str(r["case_id"]), int(r["seed"]))
    oracle = sorted((r for r in scene if r["method"] == "oracle"), key=order)
    if not oracle:
        raise ValueError(f"No oracle records for {snapshot}")
    valid = [r for r in oracle if returned(r) and truth_clear(r) is True]
    selected = (valid or oracle)[0]
    paired = {}
    for method in METHODS:
        match = [r for r in scene if r["method"] == method and r["case_id"] == selected["case_id"] and r["seed"] == selected["seed"]]
        if len(match) != 1:
            raise ValueError(f"Missing or duplicate paired query for {snapshot}, {method}")
        paired[method] = match[0]
    return paired


def summary_counts(records):
    """Count outcomes over all fixed queries, never only the displayed example."""
    oracle = {(r["snapshot"], str(r["case_id"]), int(r["seed"])): r
              for r in records if r["method"] == "oracle"}
    result = []
    for snapshot in SNAPSHOTS:
        for method in METHODS:
            group = [r for r in records if r["snapshot"] == snapshot and r["method"] == method]
            if not group:
                raise ValueError(f"No records for {snapshot} / {method}")
            accepted = [r for r in group if returned(r)]
            collision_free = sum(truth_clear(r) is True for r in accepted)
            colliding = sum(truth_clear(r) is False for r in accepted)
            refused_oracle_valid = 0
            for record in group:
                reference = oracle.get((snapshot, str(record["case_id"]), int(record["seed"])))
                if reference is None:
                    raise ValueError("Every predicted-field query must have a paired oracle query")
                if (not returned(record) and returned(reference)
                        and truth_clear(reference) is True):
                    refused_oracle_valid += 1
            result.append({"snapshot":snapshot,"method":method,"queries":len(group),
                           "returned":len(accepted),"refused":len(group)-len(accepted),
                           "returned_truth_collision_free":int(collision_free),
                           "returned_truth_colliding":int(colliding),
                           "returned_unscored":len(accepted)-collision_free-colliding,
                           "refused_with_oracle_valid_path":refused_oracle_valid,
                           "median_query_seconds":float(np.median([r["planning_seconds"] for r in group]))})
    return result


def path_figure(output, records, inputs):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6), constrained_layout=True)
    report = []
    for row, snapshot in enumerate(SNAPSHOTS):
        selected = choose_records(records, snapshot)
        image, calibration = inputs[snapshot]
        for column, method in enumerate(METHODS):
            record = selected[method]; ax = axes[row, column]
            image_axes(ax, image, calibration)
            accepted = returned(record)
            metadata = draw_recorded_path(ax, record["eef_path_world_m"], calibration,
                                         accepted=accepted, color="#fff200")
            metadata["start_visible"] = endpoint(ax, record["start_eef_world_m"], calibration, "o", "#00e0b0", "S")
            metadata["goal_visible"] = endpoint(ax, record["goal_eef_world_m"], calibration, "*", "#ff63dc", "G")
            outcome = "PATH RETURNED" if accepted else "NO PATH RETURNED"
            if accepted:
                validated = truth_clear(record)
                outcome += " / GT samples clear" if validated is True else " / GT INVALID" if validated is False else " / GT not scored"
            ax.set_title(f"{SCENE_LABELS[snapshot]} · {METHOD_LABELS[method]}\n{outcome}", fontsize=10)
            reason = str(record.get("reason") or record.get("status", ""))
            detail = f"case {record['case_id']} / seed {record['seed']}\n{reason}\nquery time {record['planning_seconds']:.3f} s"
            ax.text(.02,.02, detail, transform=ax.transAxes, fontsize=8, va="bottom", color="#202020",
                    bbox={"facecolor":"white","alpha":.9,"edgecolor":"none","pad":4})
            metadata.update({"snapshot":snapshot,"method":method,"case_id":record["case_id"],"seed":record["seed"],
                             "status":record["status"],"returned":accepted})
            report.append(metadata)
    fig.suptitle("Drawer scene: planned motion of the original LIBERO robot\n"
                 "Yellow = recorded EEF projection; S / G = common start / goal. No line is drawn for a refusal.\n"
                 "Example rule: first oracle case clear at GT samples, lowest seed. Collision tests use the full robot, not this EEF line.", fontsize=12)
    fig.savefig(output/"planned_paths.png", dpi=170); plt.close(fig)
    return report


def diagnostic_figure(output, diagnostic_dir, inputs, consumed, expected_diagnostics=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), constrained_layout=True)
    report = []
    for row, snapshot in enumerate(SNAPSHOTS):
        image, calibration = inputs[snapshot]
        for column, method in enumerate(METHODS):
            ax = axes[row,column]; image_axes(ax, image, calibration)
            path = diagnostic_dir/f"{snapshot}_{method}.npz"
            if not path.exists():
                if method != "oracle":
                    raise FileNotFoundError(path)
                ax.text(.5,.16,"Exact MuJoCo geometry queries\nNo predicted-field point states",transform=ax.transAxes,
                        ha="center",va="center",fontsize=11,bbox={"facecolor":"white","alpha":.9,"edgecolor":"none","pad":8})
                ax.set_title(f"{SCENE_LABELS[snapshot]} · {METHOD_LABELS[method]}", fontsize=11)
                report.append({"snapshot":snapshot,"method":method,"diagnostic":"not applicable; oracle has no predicted field"})
                continue
            expected = (expected_diagnostics or {}).get((snapshot, method))
            checked(path, expected, consumed)
            with np.load(path, allow_pickle=False) as saved:
                camera_points = saved["points_camera_m"].copy()
                original_states = saved["point_state"].copy()
                support = saved["point_support"].astype(bool) if "point_support" in saved else None
            if not np.isin(original_states, [-1,0,1]).all():
                raise ValueError("Adapter states must be -1 occupied, 0 unknown/ambiguous, +1 free")
            states = np.where(original_states == 1, 0, np.where(original_states == -1, 1, 2))
            transform = np.asarray(calibration["T_world_from_camera_cv"])
            world = camera_points@transform[:3,:3].T+transform[:3,3]
            metadata = draw_robot_points(ax, world, states, calibration)
            counts = metadata["all_sample_counts"]; n = len(states)
            detail = "\n".join(f"{STATE_LABELS[k]}: {counts[STATE_LABELS[k].lower()]:,} ({100*counts[STATE_LABELS[k].lower()]/max(n,1):.1f}%)" for k in STATE_COLORS)
            if support is not None:
                if support.shape != states.shape:
                    raise ValueError("Support mask must match point states")
                metadata["unsupported_points"] = int((~support).sum())
                metadata["supported_ambiguous_points"] = int((support & (original_states == 0)).sum())
            ax.text(.02,.02,detail,transform=ax.transAxes,va="bottom",fontsize=8,
                    bbox={"facecolor":"white","alpha":.9,"edgecolor":"none","pad":4})
            ax.set_title(f"{SCENE_LABELS[snapshot]} · {METHOD_LABELS[method]}\nOriginal robot at the common start",fontsize=10)
            metadata.update({"snapshot":snapshot,"method":method,"diagnostic_file":str(path.resolve())})
            report.append(metadata)
    handles = [Line2D([0],[0],marker="o",color="none",markerfacecolor=STATE_COLORS[k],label=STATE_LABELS[k],markersize=7) for k in STATE_COLORS]
    fig.legend(handles=handles,loc="outside lower center",ncol=3,fontsize=10)
    fig.suptitle("Why a field accepts or refuses the actual starting robot\n"
                 "Original collision-geometry samples, x-ray overlay: occluded samples are also projected. Counts include offscreen points.\n"
                 "Astra support includes model-guessed hidden geometry. Blue includes absent support and uncertain sign; it is not free space.",fontsize=12)
    fig.savefig(output/"start_field_diagnostics.png",dpi=170);plt.close(fig)
    return report


def metric_figure(output, rows):
    fig, axes = plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
    for index,snapshot in enumerate(SNAPSHOTS):
        group = [r for r in rows if r["snapshot"] == snapshot]
        x=np.arange(3);labels=[METHOD_LABELS[r["method"]] for r in group]
        valid=np.array([r["returned_truth_collision_free"] for r in group])
        colliding=np.array([r["returned_truth_colliding"] for r in group])
        unscored=np.array([r["returned_unscored"] for r in group])
        refused=np.array([r["refused"] for r in group])
        axes[index,0].bar(x,valid,color="#2aa876",label="Returned / GT samples clear")
        axes[index,0].bar(x,colliding,bottom=valid,color="#e34a33",label="Returned / GT sample invalid")
        axes[index,0].bar(x,unscored,bottom=valid+colliding,color="#bd7ebe",label="Returned / unscored")
        axes[index,0].bar(x,refused,bottom=valid+colliding+unscored,color="#b9bdc4",label="No path returned")
        axes[index,0].set(ylabel="Number of fixed queries",title=f"{SCENE_LABELS[snapshot]}: all query outcomes")
        paired=[r["refused_with_oracle_valid_path"] for r in group]
        axes[index,1].bar(x,paired,color="#e6a23c")
        axes[index,1].set(ylabel="Number of paired queries",title="Refused, while paired oracle path passed GT samples",ylim=(0,max(r["queries"] for r in group)*1.08))
        times=[r["median_query_seconds"] for r in group]
        axes[index,2].bar(x,times,color="#6b83b4")
        axes[index,2].set(ylabel="Median planning query time (s)",title="Time for returned paths and refusals")
        for col in range(3):
            axes[index,col].set_xticks(x,labels,rotation=15)
            axes[index,col].spines[["top","right"]].set_visible(False)
            axes[index,col].grid(axis="y",alpha=.18);axes[index,col].set_axisbelow(True)
        for i,value in enumerate(times):
            axes[index,2].annotate(f"{value:.3g}",(i,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,fontsize=9)
    fig.suptitle("Planning evaluation in two frozen drawer scenes\n"
                 "Same original robot, start/goal pairs and search settings. GT scoring happens after predicted-field planning.\n"
                 "GT tests finite joint samples, not a continuous swept volume. These paths are not a manipulation-success result.",fontsize=12)
    fig.savefig(output/"planning_outcomes.png",dpi=170);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics",type=Path,required=True)
    parser.add_argument("--diagnostics",type=Path)
    parser.add_argument("--references",type=Path,default=HERE.parent/"rgb_scene"/"references_004")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Plot output must be a new directory")
    consumed={}
    run_manifest_path=args.metrics.parent/"manifest.json"
    expected_metrics=None
    if run_manifest_path.exists():
        checked(run_manifest_path,None,consumed)
        run_manifest=json.loads(run_manifest_path.read_text())
        if (run_manifest.get("status") != "complete" or not run_manifest.get("source_unchanged")
                or not run_manifest.get("inputs_unchanged")):
            raise ValueError("Evaluation manifest must be complete and intact")
        expected_metrics=run_manifest["artifacts_sha256"][args.metrics.name]
    checked(args.metrics,expected_metrics,consumed)
    metrics=json.loads(args.metrics.read_text())
    if metrics.get("status") != "complete":
        raise ValueError("Only complete planning/evaluation results may be plotted")
    records=metrics["records"]
    unknown_snapshots={r["snapshot"] for r in records}-set(SNAPSHOTS)
    unknown_methods={r["method"] for r in records}-set(METHODS)
    if unknown_snapshots or unknown_methods:
        raise ValueError("This experiment visualizes only the two drawer snapshots and three declared methods")
    inputs={}
    for name in SNAPSHOTS:
        folder=args.references/"snapshots"/name
        rgb=checked(folder/"agentview_upright.png",None,consumed)
        cal=checked(folder/"evaluator"/"calibration.json",None,consumed)
        calibration=json.loads(cal.read_text())
        with Image.open(rgb) as image:
            pixels=np.asarray(image.convert("RGB")).copy()
        if pixels.shape != (calibration["height"],calibration["width"],3):
            raise ValueError("RGB and calibration dimensions differ")
        inputs[name]=(pixels,calibration)
    args.output.mkdir(parents=True)
    started=time.monotonic();source_hash=sha(__file__)
    paths=path_figure(args.output,records,inputs)
    default_diagnostics=Path(metrics.get("plans_directory",str(args.metrics.parent)))/"diagnostics"
    expected_diagnostics={(r["snapshot"],r["method"]):r["sha256"] for r in metrics.get("start_diagnostics",[])}
    if "start_diagnostics" in metrics and set(expected_diagnostics) != {(s,m) for s in SNAPSHOTS for m in ("astra","depthpro")}:
        raise ValueError("Expected all four original predicted-field start diagnostic records")
    diagnostics=diagnostic_figure(args.output,args.diagnostics or default_diagnostics,inputs,consumed,expected_diagnostics)
    summary=summary_counts(records)
    metric_figure(args.output,summary)
    artifacts={p.name:sha(p) for p in args.output.glob("*.png")}
    unchanged={p:sha(p)==digest for p,digest in consumed.items()}
    if not all(unchanged.values()) or sha(__file__)!=source_hash:
        raise RuntimeError("Plot input/source changed during rendering")
    report={"status":"complete","source_sha256":source_hash,"consumed_sha256":consumed,"inputs_unchanged":unchanged,
            "artifacts_sha256":artifacts,"path_panels":paths,"diagnostics":diagnostics,"summary":summary,
            "selection_rule":"lexicographically first oracle case valid at all GT joint samples, then lowest seed; same pair in all method panels",
            "truth_scope":"finite sampled-state native collision checks; not a continuous swept-volume certificate",
            "scope":"static drawer-scene motion planning with original robot geometry; not drawer manipulation success",
            "elapsed_s":time.monotonic()-started}
    dump(args.output/"manifest.json",report)
    print(json.dumps({"status":"complete","output":str(args.output),"figures":list(artifacts)},allow_nan=False))


if __name__ == "__main__":
    main()
