"""Reproduce real recorded Dijkstra fields and make scientific slide figures.

No models, simulator, hidden object state, or invented obstacles are used.
Reconstruct with the recorded project's NumPy runtime; plot saved arrays using
the existing scientific environment:

  .venv/bin/python reports/hybrid_method_slides_20261012/field_assets/generate_field_assets.py reconstruct
  /workspace/venvs/libero-plus/bin/python reports/hybrid_method_slides_20261012/field_assets/generate_field_assets.py plot
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
RUN = PROJECT / "runs/bowl_obstacle_distance_medium_zero_shot_001"
BLOCKS = ((201, 214), (300, 309))
CHECKED_SOURCE = ("distance_planner.py", "distance_policy.py", "compact_field.py",
                  "field.py", "schema.py", "planner.py", "tracking.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def reconstruct():
    sys.path.insert(0, str(PROJECT / "src"))
    from gpt_field.compact_field import expand_compact_field
    from gpt_field.distance_planner import DistanceFieldPlanner
    from gpt_field.field import GridField
    from gpt_field.planner import PlannerConfig
    from gpt_field.schema import FieldPrediction, GridSpec

    manifest = json.loads((RUN / "manifest.json").read_text())
    config = manifest["configuration"]
    hashes = {}
    for name in CHECKED_SOURCE:
        current = digest(PROJECT / "src/gpt_field" / name)
        recorded = config["software"]["source_sha256"][name]
        assert current == recorded, f"Source version differs: {name}"
        hashes[name] = {"current_sha256": current, "recorded_sha256": recorded,
                        "matches": True}
    rows = [(i, json.loads(line)) for i, line in enumerate((RUN / "trace.jsonl").open(), 1)]
    actions = {r["step"]: (i, r) for i, r in rows if r["event"] == "action"}
    executed = {r["step"]: (i, r) for i, r in rows if r["event"] == "executed"}
    grid = GridSpec(**config["policy"]["grid"])
    metadata = {
        "artifact_type": "recorded_dijkstra_field_scientific_figures",
        "source_run": str(RUN.relative_to(PROJECT)),
        "source_trace_sha256": digest(RUN / "trace.jsonl"),
        "source_manifest_sha256": digest(RUN / "manifest.json"),
        "verified_source_hashes": hashes,
        "reconstruction_python": sys.version,
        "reconstruction_numpy": np.__version__,
        "model_calls": 0, "simulator_calls": 0,
        "coordinate_frame": "world axes; plots use offsets from the field-generation EEF position",
        "grid": grid.to_dict(),
        "distance_definition": {
            "D": "D(v) = min over valid graph paths from v to the seeded goal region of sum c(u,w)",
            "edge_cost": "c(u,w) = ||w-u||_2 * [rho(u)+rho(w)]/2",
            "density": "rho(x) = 1 + sum_j a_j exp(-||x-c_j||^2 / (2 sigma_j^2))",
            "graph": "26-neighbor 3-D lattice; every active diagonal interpolation corner and continuous segment must avoid blocked/unknown boxes",
            "goal_seeding": "Grid vertices in the tolerance ball have D=0; valid virtual edges connect nearby corners to the tolerance-sphere surface when needed.",
            "unreachable": "Infinity at invalid/unreachable nodes, preserved as infinity in NPZ, never converted to zero.",
            "unit": "Meters when rho=1; weighted meters with dimensionless nonconstant density.",
            "selected_examples": "Both recorded examples have no Gaussian repulsors, so rho=1 and D is 3-D graph path length in meters.",
            "not_sdf": "D is cost to the model-specified local goal region, not signed distance to an object surface.",
            "legacy_potential": "The dense analytic attraction values are also reproduced for version verification, but the Dijkstra cost does not use attraction_weights or these legacy values.",
        },
        "scope_limits": [
            "Masks and goals are recorded model estimates, not measured obstacle meshes or object poses.",
            "Each heatmap is an actual plane of the full 3-D solved D field; it is not a 2-D replanning result.",
            "Tracking measurements are end-effector positions only, not object motion or full-arm collision clearance.",
            "The source episode ultimately failed the collision criterion; these are method illustrations, not a successful-task result.",
        ],
        "blocks": [],
    }
    for first, last in BLOCKS:
        line, action = actions[first]
        d = action["diagnostics"]
        compact = d["effective_compact_field"]
        center = np.asarray(d["field_center"], dtype=float)
        goal = np.asarray(d["plan"]["search"]["goal_xyz"], dtype=float)
        expanded = expand_compact_field(compact, grid)
        assert expanded == d["prediction"], "Expanded legacy values/masks differ from recorded response"
        field = GridField(grid, tuple(center), FieldPrediction.from_dict(expanded, grid))
        planner = DistanceFieldPlanner(PlannerConfig(**config["policy"]["planner"]))
        result = planner.plan(field, action["state"][:3], goal_xyz=goal,
            compact_prediction=compact, goal_tolerance_m=d["plan"]["search"]["goal_tolerance_m"])
        assert result.to_dict() == d["plan"], f"Exact recorded plan replay failed at action {first}"
        points = grid.coordinates(center).reshape((grid.size,) * 3 + (3,))
        blocked = np.zeros(grid.count, dtype=bool)
        blocked[expanded["blocked_indices"]] = True
        unknown = np.zeros(grid.count, dtype=bool)
        unknown[expanded["unknown_indices"]] = True
        measured = np.asarray([action["state"][:3]] + [executed[k][1]["next_state"][:3]
                                                       for k in range(first, last + 1)])
        replan_actions = [k for k in range(first, last + 1) if actions[k][1]["diagnostics"].get("plan_refreshed")]
        replan_paths = [np.asarray(actions[k][1]["diagnostics"]["plan"]["candidates"][0]["positions"])
                        for k in replan_actions]
        replan_points = np.concatenate(replan_paths)
        replan_ends = np.cumsum([len(p) for p in replan_paths])
        density = planner._cache["density"].reshape((grid.size,) * 3)
        assert np.all(density == 1.0), "Examples should not be labeled unweighted if density differs"
        np.savez_compressed(OUT / f"block_{first}_data.npz",
            world_xyz_m=points, node_indices_ijk=np.indices((grid.size,) * 3).transpose(1, 2, 3, 0),
            distance_cost_m=planner.distance_grid_m, traversal_density=density,
            legacy_attraction_values=np.asarray(expanded["values"]).reshape((grid.size,) * 3),
            blocked_mask=blocked.reshape((grid.size,) * 3), unknown_mask=unknown.reshape((grid.size,) * 3),
            successor_flat=planner._cache["successor"], goal_seed_flat=np.asarray(sorted(planner._cache["seed_end"])),
            center_world_xyz_m=center, goal_world_xyz_m=goal,
            selected_path_world_xyz_m=np.asarray(result.candidates[0].positions),
            measured_eef_world_xyz_m=measured,
            measured_sample_after_action_index=np.arange(first - 1, last + 1),
            normalized_actions=np.asarray([actions[k][1]["action"] for k in range(first, last + 1)]),
            replan_action_indices=np.asarray(replan_actions), replan_paths_world_xyz_m=replan_points,
            replan_path_end_offsets=replan_ends)
        with (OUT / f"block_{first}_eef_track.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample", "after_action_index_zero_based", "x_world_m", "y_world_m", "z_world_m", "distance_to_model_goal_m"])
            for sample, point in enumerate(measured):
                writer.writerow([sample, first - 1 + sample, *point, float(np.linalg.norm(point-goal))])
        observation_line, prior = executed[first - 1]
        rgb = []
        for key, camera in (("observation/image", "external"), ("observation/wrist_image", "wrist")):
            prefix, encoded = prior["observation"][key].split(",", 1)
            assert prefix == "data:image/png;base64"
            payload = base64.b64decode(encoded, validate=True)
            name = f"block_{first}_input_{camera}.png"
            (OUT / name).write_bytes(payload)
            rgb.append({"artifact": name, "sha256": digest(OUT/name), "source_trace_line_1based": observation_line,
                        "source_event": "executed", "after_action_index_zero_based": first-1,
                        "source_key": key, "transformation": "base64 decode only; original PNG bytes"})
        item = {
            "field_refresh_action_index_zero_based": first, "last_executed_action_index_zero_based": last,
            "source_action_trace_line_1based": line, "source_request": d["api_request_log"],
            "source_request_sha256": digest(PROJECT/d["api_request_log"]),
            "raw_model_compact_prediction": d["compact_prediction"],
            "effective_compact_field": compact, "field_center_world_xyz_m": center.tolist(),
            "goal_world_xyz_m": goal.tolist(), "recorded_plan": d["plan"],
            "expanded_field_exact_match": True, "reconstructed_plan_exact_match": True,
            "executed_control_count": last-first+1,
            "replan_action_indices_zero_based": replan_actions,
            "initial_goal_distance_m": float(np.linalg.norm(measured[0]-goal)),
            "final_goal_distance_m": float(np.linalg.norm(measured[-1]-goal)),
            "measured_path_length_m": float(np.linalg.norm(np.diff(measured, axis=0), axis=1).sum()),
            "measured_net_displacement_m": float(np.linalg.norm(measured[-1]-measured[0])),
            "measured_y_offset_range_m": [float((measured-center)[:, 1].min()), float((measured-center)[:, 1].max())],
            "raw_rgb": rgb,
            "terminal_collision": executed[last][1]["sim_status"]["collision"]["first_event"],
            "data_artifacts": [f"block_{first}_data.npz", f"block_{first}_eef_track.csv"],
        }
        if first == 201:
            assert np.all(np.asarray(result.candidates[0].positions)[:, 1] == center[1])
            item["primary_slice"] = {"axes": ["world X", "world Z"], "fixed_axis": "world Y",
                "fixed_world_coordinate_m": float(center[1]), "fixed_offset_m": 0.0, "grid_index": 16,
                "path_lies_in_slice": True,
                "description": "Exact XZ plane of the 3-D distance field. The selected path lies in this plane; the measured path in the tracking plot is an XZ projection."}
        metadata["blocks"].append(item)
        write_json(f"block_{first}_recorded_compact_and_plan.json", item)
    write_json("metadata.json", metadata)
    print(json.dumps({"reconstruction": "exact", "blocks": [b[0] for b in BLOCKS],
                      "output": str(OUT), "model_calls": 0, "simulator_calls": 0}))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
        "axes.titlesize": 20, "axes.labelsize": 15, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 12, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#758393", "axes.labelcolor": "#25374A",
        "text.color": "#172A3A", "xtick.color": "#3D4D5E", "ytick.color": "#3D4D5E",
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    colors = {"plan": "#116B91", "measured": "#E26036", "blocked": "#485866",
              "unknown": "#EBC976", "goal": "#11644D", "replan": "#805EAC"}
    meta = json.loads((OUT / "metadata.json").read_text())
    plots = []

    def save(fig, stem, description):
        fig.savefig(OUT / (stem + ".png"), dpi=220)
        fig.savefig(OUT / (stem + ".pdf"))
        plt.close(fig)
        plots.append({"png": stem+".png", "pdf": stem+".pdf", "description": description,
                      "png_sha256": digest(OUT/(stem+".png")), "pdf_sha256": digest(OUT/(stem+".pdf"))})

    def axes_style(ax, full=True):
        ax.set_xlabel("World X offset from field center (cm)")
        ax.set_ylabel("World Z offset from field center (cm)")
        if full:
            ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
            ax.xaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_aspect("equal", adjustable="box")

    def slice_boxes(ax, compact):
        for category, style in (("unknown_boxes", {"facecolor":colors["unknown"], "edgecolor":"#9E762B", "hatch":"///"}),
                                ("blocked_boxes", {"facecolor":colors["blocked"], "edgecolor":"white"})):
            for box in compact[category]:
                lo, hi = np.asarray(box["lower_offset_m"])*100, np.asarray(box["upper_offset_m"])*100
                if lo[1] <= 0 <= hi[1]:
                    ax.add_patch(Rectangle((lo[0], lo[2]), hi[0]-lo[0], hi[2]-lo[2], linewidth=.8, zorder=3, **style))

    def point_markers(ax, start, goal):
        ax.scatter(*start, s=130, color="white", edgecolor="#172A3A", linewidth=2, marker="o", zorder=8)
        ax.scatter(*goal, s=230, color=colors["goal"], edgecolor="white", linewidth=1.1, marker="*", zorder=9)

    data = np.load(OUT / "block_201_data.npz", allow_pickle=False)
    block = meta["blocks"][0]
    compact = block["effective_compact_field"]
    center, goal = data["center_world_xyz_m"], data["goal_world_xyz_m"]
    offsets = (data["world_xyz_m"] - center)*100
    axis = offsets[:, 16, 16, 0]
    path = (data["selected_path_world_xyz_m"] - center)*100
    goaloff = (goal-center)*100
    costs_cm = data["distance_cost_m"][:, 16, :].T*100
    costs = np.ma.masked_invalid(costs_cm)

    fig, axs = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios":[1, 1.11]})
    fig.subplots_adjust(top=.79, bottom=.24, left=.065, right=.96, wspace=.26)
    fig.suptitle("A real model-specified local map becomes a 3-D cost-to-go field", fontsize=24, x=.05, ha="left", y=.955)
    fig.text(.05, .904, "Recorded Hybrid decision at action 201  |  33³ grid  |  5 mm spacing  |  16 cm span", fontsize=16, color="#607184")
    axs[0].set_facecolor("#EEF3F6")
    slice_boxes(axs[0], compact)
    axs[0].plot(axis, axis*0, alpha=0)
    point_markers(axs[0], (0,0), goaloff[[0,2]])
    axs[0].annotate("Current EEF", (0,0), xytext=(.6,.55), fontsize=13)
    axs[0].annotate("Model goal", goaloff[[0,2]], xytext=(-7.5,6.0), fontsize=13)
    axs[0].set_title("1   Model-predicted exclusion regions", loc="left", pad=15)
    axs[0].grid(color="white", linewidth=.9, alpha=.6)
    im = axs[1].pcolormesh(axis, axis, costs, cmap="viridis", shading="nearest", rasterized=True, vmin=0)
    finite = np.ma.masked_invalid(costs_cm)
    axs[1].contour(axis, axis, finite, levels=np.arange(2,25,2), colors="white", linewidths=.45, alpha=.36, zorder=2)
    slice_boxes(axs[1], compact)
    axs[1].plot(path[:,0], path[:,2], color="white", lw=5.6, zorder=5)
    axs[1].plot(path[:,0], path[:,2], color="#EE6F37", lw=2.7, marker="o", ms=4, zorder=6)
    point_markers(axs[1], (0,0), goaloff[[0,2]])
    axs[1].set_title("2   Dijkstra distance + selected path", loc="left", pad=15)
    cb=fig.colorbar(im, ax=axs[1], fraction=.052, pad=.035)
    cb.set_label("3-D graph cost-to-go D (cm)", fontsize=13)
    for ax in axs: axes_style(ax)
    handles=[Patch(facecolor=colors["blocked"],label="Predicted blocked"),
             Patch(facecolor=colors["unknown"],edgecolor="#9E762B",hatch="///",label="Predicted unknown"),
             Line2D([],[],color="#EE6F37",lw=3,marker="o",label="Selected path (10 segments)"),
             Line2D([],[],marker="*",color=colors["goal"],lw=0,markersize=13,label="Model goal")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.145), ncol=4, frameon=False)
    fig.text(.065,.093,"Exact XZ slice at world Y = 0.20656 m. The selected path lies in this plane; D was solved in 3-D.",fontsize=13)
    fig.text(.065,.058,"Here ρ = 1 (no repulsors): D(start) = 8.864 cm. Masks are model estimates, not measured object geometry.",fontsize=13,color="#607184")
    save(fig,"field_cost_slice","Exact XZ slice of recorded 3-D Dijkstra cost at decision 201; both exclusion masks and selected path are real recorded data.")

    # Compact page asset: two square scientific panels and a readable fact strip.
    fig=plt.figure(figsize=(14,4.7))
    left=fig.add_axes([.055,.215,.29,.70])
    right=fig.add_axes([.397,.215,.29,.70])
    left.set_facecolor("#EEF3F6")
    slice_boxes(left,compact)
    point_markers(left,(0,0),goaloff[[0,2]])
    left.set_title("Model-predicted map",loc="left",fontsize=16,pad=12)
    left.annotate("Start",(0,0),xytext=(6,7),textcoords="offset points",fontsize=11)
    left.annotate("Goal",goaloff[[0,2]],xytext=(5,8),textcoords="offset points",fontsize=11)
    im=right.pcolormesh(axis,axis,costs,cmap="viridis",shading="nearest",rasterized=True,vmin=0)
    right.contour(axis,axis,finite,levels=np.arange(2,25,2),colors="white",linewidths=.45,alpha=.36,zorder=2)
    slice_boxes(right,compact)
    right.plot(path[:,0],path[:,2],color="white",lw=4.9,zorder=5)
    right.plot(path[:,0],path[:,2],color="#EE6F37",lw=2.3,marker="o",ms=3.4,zorder=6)
    point_markers(right,(0,0),goaloff[[0,2]])
    right.set_title("3-D Dijkstra field + path",loc="left",fontsize=16,pad=12)
    for ax in (left,right):
        axes_style(ax)
        ax.set_xlabel("World X offset (cm)",fontsize=12)
        ax.set_ylabel("World Z offset (cm)",fontsize=12)
        ax.tick_params(labelsize=10)
    cax=fig.add_axes([.710,.23,.012,.655])
    cb=fig.colorbar(im,cax=cax);cb.set_label("Cost-to-go D (cm)",fontsize=11);cb.ax.tick_params(labelsize=10)
    fig.text(.790,.84,"RECORDED FIELD 201",fontsize=12,color="#607184",fontweight="bold")
    fig.text(.790,.69,"8.864 cm",fontsize=26,fontweight="bold")
    fig.text(.790,.63,"Graph cost at the start",fontsize=11,color="#607184")
    fig.text(.790,.50,"33³ nodes · 5 mm spacing",fontsize=12)
    fig.text(.790,.42,"26-neighbor 3-D search",fontsize=12)
    fig.text(.790,.34,"Goal radius: 2.5 mm",fontsize=12)
    fig.text(.790,.26,"ρ = 1; no repulsors",fontsize=12)
    fig.text(.790,.17,"White: invalid / unreachable",fontsize=10,color="#607184")
    fig.legend(handles=handles[:3],loc="lower center",bbox_to_anchor=(.40,.055),ncol=3,frameon=False,fontsize=10.5)
    fig.text(.055,.018,"Actual XZ slice at world Y = 0.20656 m; selected path lies in-plane. Masks are model estimates, not object meshes.",fontsize=10.5,color="#607184")
    save(fig,"field_cost_slice_wide","Slide-width exact XZ slice with real blocked/unknown masks and recorded path; 14 by 4.7 inches.")

    def tracking_figure(first, title, terminal=False):
        a=np.load(OUT/f"block_{first}_data.npz",allow_pickle=False)
        b=next(b for b in meta["blocks"] if b["field_refresh_action_index_zero_based"]==first)
        c=a["center_world_xyz_m"]; g=(a["goal_world_xyz_m"]-c)*100
        p=(a["selected_path_world_xyz_m"]-c)*100
        track=(a["measured_eef_world_xyz_m"]-c)*100
        allp=(a["replan_paths_world_xyz_m"]-c)*100
        splits=np.split(allp,a["replan_path_end_offsets"][:-1])
        steps=np.arange(len(track))
        errors=np.linalg.norm(a["measured_eef_world_xyz_m"]-a["goal_world_xyz_m"],axis=1)*1000
        fig=plt.figure(figsize=(16,9));gs=fig.add_gridspec(1,2,width_ratios=[1.1,1],left=.075,right=.96,bottom=.235,top=.79,wspace=.27)
        ax=fig.add_subplot(gs[0]);err=fig.add_subplot(gs[1])
        fig.suptitle(title,fontsize=24,x=.05,ha="left",y=.955)
        fig.text(.05,.904,f"Recorded actions {first}–{b['last_executed_action_index_zero_based']} (zero-based)  |  {b['executed_control_count']} executed controls  |  EEF measurements only",fontsize=16,color="#607184")
        ax.plot(p[:,0],p[:,2],lw=2.9,color=colors["plan"],marker="o",ms=4.5,label="Initial selected path")
        for idx,other in enumerate(splits[1:],1):
            ax.plot(other[:,0],other[:,2],lw=2.4,color=colors["replan"],ls="--",marker="o",ms=3.5,label=f"Replan at action {int(a['replan_action_indices'][idx])}")
        ax.plot(track[:,0],track[:,2],lw=2.9,color=colors["measured"],marker="s",ms=5,label="Measured EEF (XZ projection)")
        point_markers(ax,(0,0),g[[0,2]])
        ax.annotate("Start",(0,0),xytext=(9,9),textcoords="offset points",fontsize=13)
        ax.annotate("Model goal",g[[0,2]],xytext=(-15,22 if not terminal else -32),textcoords="offset points",fontsize=13,ha="center")
        ax.set_title("Planned geometry and actual motion",loc="left",pad=15)
        axes_style(ax,full=False);ax.grid(alpha=.19)
        ax.margins(x=.13,y=.15)
        if terminal:
            ax.set_xlim(-4,2);ax.set_ylim(-7,1)
        ax.legend(loc="lower left" if first==201 else "upper left",frameon=False,fontsize=12)
        err.plot(steps,errors,color=colors["measured"],lw=3,marker="s",ms=5)
        err.axhline(5,color=colors["goal"],lw=1.6,ls="--",label="Policy arrival tolerance: 5 mm")
        for idx in a["replan_action_indices"][1:]:
            err.axvline(int(idx)-first,color=colors["replan"],lw=1.7,ls=":")
        err.set_title("Distance to the model's goal",loc="left",pad=15)
        err.set_xlabel("Executed controls since field refresh")
        err.set_ylabel("3-D EEF-to-goal distance (mm)")
        err.set_xlim(0,len(track)-1+.4);err.set_ylim(bottom=0);err.grid(alpha=.2)
        err.xaxis.set_major_locator(MultipleLocator(2))
        err.annotate(f"{errors[-1]:.2f} mm",(steps[-1],errors[-1]),xytext=(-8,18),textcoords="offset points",ha="right",fontsize=16,fontweight="bold",color=colors["measured"])
        err.legend(loc="upper right",frameon=False,fontsize=12)
        if terminal:
            event=b["terminal_collision"]
            fig.text(.075,.148,"Forbidden contact: left finger ↔ ramekin at action 309, physics substep 7 (zero-based).",fontsize=15,color="#A53F2A")
            fig.text(.075,.107,"The last control was interrupted after 8 of 25 physics substeps. Contact identity comes from the evaluator.",fontsize=13)
            fig.text(.075,.066,"EEF path validity is not full-gripper collision clearance. No object geometry is inferred from this plot.",fontsize=13,color="#607184")
            stem="terminal_field_tracking"
        else:
            fig.text(.075,.148,f"Initial selected path: {b['recorded_plan']['candidates'][0]['path_length_m']*100:.3f} cm  |  Measured travel: {b['measured_path_length_m']*100:.3f} cm  |  Final error: {errors[-1]:.2f} mm",fontsize=15)
            fig.text(.075,.107,"The controller replanned after 10 actions, then requested a new model field after measured arrival.",fontsize=13)
            fig.text(.075,.066,"Paths at different times are not synchronized point pairs; the left panel is an XZ projection of 3-D data.",fontsize=13,color="#607184")
            stem="field_tracking"
        save(fig,stem,"Recorded planned versus measured EEF motion and measured 3-D goal error; no synthetic trajectories.")
    tracking_figure(201,"A geometric path is followed from successive measured poses")
    tracking_figure(300,"Point-path validity did not certify gripper clearance",terminal=True)

    a=np.load(OUT/"block_201_data.npz",allow_pickle=False)
    c=a["center_world_xyz_m"]
    p=(a["selected_path_world_xyz_m"]-c)*100
    track=(a["measured_eef_world_xyz_m"]-c)*100
    goaloff=(a["goal_world_xyz_m"]-c)*100
    replan=np.split((a["replan_paths_world_xyz_m"]-c)*100,a["replan_path_end_offsets"][:-1])[1]
    error=np.linalg.norm(a["measured_eef_world_xyz_m"]-a["goal_world_xyz_m"],axis=1)*1000
    fig,axs=plt.subplots(1,2,figsize=(10,4),gridspec_kw={"width_ratios":[1,1]})
    fig.subplots_adjust(left=.075,right=.97,bottom=.245,top=.88,wspace=.35)
    ax,err=axs
    ax.plot(p[:,0],p[:,2],color=colors["plan"],lw=2.4,marker="o",ms=3.2,label="Initial plan")
    ax.plot(replan[:,0],replan[:,2],color=colors["replan"],lw=2.2,ls="--",label="Replan at action 211")
    ax.plot(track[:,0],track[:,2],color=colors["measured"],lw=2.4,marker="s",ms=3.4,label="Measured EEF")
    point_markers(ax,(0,0),goaloff[[0,2]])
    ax.set_title("Planned vs. measured (XZ projection)",fontsize=13,loc="left",pad=10)
    ax.set_xlabel("World X offset (cm)",fontsize=10);ax.set_ylabel("World Z offset (cm)",fontsize=10)
    ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.2);ax.tick_params(labelsize=9);ax.margins(.12)
    err.plot(np.arange(len(error)),error,color=colors["measured"],lw=2.5,marker="s",ms=3.5)
    err.axhline(5,color=colors["goal"],ls="--",lw=1.3)
    err.axvline(10,color=colors["replan"],ls=":",lw=1.4)
    err.text(10.15,77,"Replan",fontsize=9,color=colors["replan"])
    err.annotate(f"{error[-1]:.2f} mm",(14,error[-1]),xytext=(-5,11),textcoords="offset points",ha="right",fontsize=12,fontweight="bold",color=colors["measured"])
    err.text(.4,8,"5 mm policy arrival tolerance",fontsize=8.8,color=colors["goal"])
    err.set_title("Measured 3-D distance to model goal",fontsize=13,loc="left",pad=10)
    err.set_xlabel("Controls since field refresh",fontsize=10);err.set_ylabel("EEF-to-goal distance (mm)",fontsize=10)
    err.set_xlim(0,14.4);err.set_ylim(0,90);err.xaxis.set_major_locator(MultipleLocator(2));err.grid(alpha=.2);err.tick_params(labelsize=9)
    handles2,labels2=ax.get_legend_handles_labels()
    fig.legend(handles2,labels2,loc="lower center",bbox_to_anchor=(.5,.070),ncol=3,frameon=False,fontsize=9.5)
    fig.text(.075,.017,"Recorded actions 201–214; 14 controls, two local plans. Positions are relative to the fixed field-generation center.",fontsize=9,color="#607184")
    save(fig,"field_tracking_wide","Slide-width real initial/replanned paths, actual EEF projection, and goal-error history; 10 by 4 inches.")
    meta["plotting_python"]=sys.version
    meta["plotting_numpy"]=np.__version__
    meta["matplotlib_version"]=matplotlib.__version__
    meta["figures"]=plots
    meta["generation_script_sha256"]=digest(__file__)
    meta["numerical_artifact_hashes"]={p.name:digest(p) for p in sorted(OUT.iterdir()) if p.suffix in (".npz",".csv")}
    write_json("metadata.json",meta)
    print(json.dumps({"plots":[p["png"] for p in plots],"output":str(OUT)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("reconstruct","plot"))
    args=parser.parse_args()
    reconstruct() if args.phase=="reconstruct" else plot()
