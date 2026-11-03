"""Extract a compact animation timeline from saved observations; no replay."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001"
OUT = Path(__file__).resolve().parent
OBSTACLE = "ketchup_obstacle_1"


def read(path):
    return json.loads(path.read_text())


def public_text(response):
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    return "".join(part.get("text", "") for item in response.get("output", [])
                   if item.get("type") == "message" for part in item.get("content", [])
                   if part.get("type") == "output_text")


def rounded(value):
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def extract():
    trial = RUN / "trial"
    summary, manifest = read(trial / "summary.json"), read(trial / "manifest.json")
    audit = read(RUN / "independent_audit.json")
    hz = manifest["configuration"]["active_agent"]["framework"]["execution_fragments"]["control_hz"]
    outputs = []
    for path in sorted((trial / "requests").glob("*.response.json")):
        result = read(path.with_name(path.name.replace(".response.json", ".result.json")))
        if result.get("status") == "valid" and result.get("request_kind") == "agent":
            outputs.append({"source": str(path.relative_to(RUN)),
                            "value": json.loads(public_text(read(path)))})

    stages, frames, action_meta, events = {}, {}, {}, []
    maps = [{"revision": 0, "update_control_step": None, "source": "uninitialized",
             "avoid_boxes": [], "clearance_m": 0., "can_box_index": None,
             "obstacle_box_index": None, "box_labels": []}]
    previous_counts, previous_pairs = {}, {}
    latest_map, latest_route = 0, None
    seen_outputs, seen_failures, route_entries = set(), set(), {}
    initial_state = None

    def event(kind, control_step, label, *, action_index=None, **details):
        # Model/controller decisions are detected at an already-observed state.
        # A physics contact during action n first belongs to endpoint state n+1.
        state_index = control_step if action_index is None else action_index + 1
        item = {"id": len(events), "kind": kind, "control_step": control_step,
                "detected_at_state_index": state_index, "frame_index": state_index,
                "action_index": action_index, "label": label, **details}
        events.append(item)
        return item

    def capture(fragment, start_step, evidence):
        if not isinstance(fragment, dict) or not isinstance(start_step, int):
            return
        for raw in fragment.get("segments", fragment.get("segment_trace", [])):
            sid = f"f{start_step}_s{raw['index']}"
            existing = stages.get(sid)
            if existing and existing["executed_actions"] > raw.get("actions", 0):
                continue
            request = raw.get("request", {})
            stage = {"id": sid, "fragment_start_step": start_step, "stage_index": raw["index"],
                     "entry_control_step": start_step + raw.get("entry_at_fragment_step", 0),
                     "skill": request.get("skill"), "profile": request.get("profile"),
                     "status": raw.get("status"), "reason": raw.get("reason"),
                     "executed_actions": raw.get("actions", 0),
                     "entry_xyz_m": raw.get("entry_pose_wxyz", [])[:3],
                     "goal_xyz_m": raw.get("target_xyz_m"), "route_id": None,
                     "gripper": request.get("gripper"), "evidence": evidence}
            if "routing" in raw:
                route = raw["routing"]
                rid = "r_" + sid
                stage["route_id"] = rid
                route_entries[rid] = {
                    "id": rid, "stage_id": sid, "entry_control_step": stage["entry_control_step"],
                    "map_revision": route.get("map_revision"), "map_source": route.get("map_source"),
                    "geometry_source": route.get("geometry_source"),
                    "start_xyz_m": stage["entry_xyz_m"], "goal_xyz_m": stage["goal_xyz_m"],
                    "path_xyz_m": route.get("path_xyz_m", []),
                    "straight_path_valid": route.get("straight_path_valid"),
                    "expanded_boxes": route.get("expanded_boxes", []),
                    "clearance_m": route.get("clearance_m"),
                    "planner_search_calls": (route.get("planner") or {}).get("calls", 0),
                    "planner_methods": (route.get("planner") or {}).get("methods", []),
                    "status": raw.get("status"), "reason": raw.get("reason"),
                    "route_reason": route.get("reason"),
                    "tracking_reason": (raw.get("route_tracking") or {}).get("reason"),
                    "executed_actions": raw.get("actions", 0), "evidence": evidence}
            stages[sid] = stage

    trace = trial / "trace.jsonl"
    for lineno, line in enumerate(trace.open(), 1):
        row = json.loads(line)
        evidence = f"trial/trace.jsonl:{lineno}"
        kind = row.get("event")
        if kind == "reset":
            initial_state = row["observation"]["observation/state"]
            previous_counts = row["sim_status"]["collision"]["counts"]
            frames[0] = {"frame_index": 0, "control_step": 0, "action_index": None,
                         "sim_action_time_s": 0., "grip_xyz_m": initial_state[:3],
                         "map_revision": 0, "stage_id": None, "skill": None, "profile": None,
                         "status": "initial_observation", "route_id": None, "last_route_id": None,
                         "goal_xyz_m": None, "contact_delta": {}, "forbidden_contact_delta": 0,
                         "can_contact_delta": 0, "obstacle_contact_delta": 0,
                         "obstacle_contact_pairs": [], "contact_types": [], "contact_summary": None,
                         "contact_xyz_m": [], "forbidden_contact_total": 0,
                         "libero_success": False, "event_ids": [], "evidence": evidence}
        elif kind == "action":
            step, diag = row["step"], row.get("diagnostics", {})
            params = (diag.get("routing") or {}).get("params")
            match = next((i for i, output in enumerate(outputs) if output["value"] == params), None)
            if match is not None and match not in seen_outputs:
                seen_outputs.add(match)
                output = outputs[match]
                output["control_step"] = step
                update = output["value"].get("obstacle_map")
                if update is not None:
                    latest_map += 1
                    maps.append({"revision": latest_map, "update_control_step": step,
                                 "source": "model_estimate", **update,
                                 "box_labels": [f"B{i + 1}" for i in range(len(update["avoid_boxes"]))],
                                 "can_box_index": None, "obstacle_box_index": None,
                                 "object_identity": "unlabeled_model_estimates",
                                 "response_file": output["source"]})
                    event("map_update", step, f"Map updated: revision {latest_map}",
                          map_revision=latest_map, box_count=len(update["avoid_boxes"]))
            completed = diag.get("feedback") or {}
            if completed.get("action") == "execute":
                start = (completed.get("start") or {}).get("step")
                snapshot = completed.get("evidence") or {}
                capture(snapshot, start, evidence)
                if completed.get("reason") in ("route_unavailable", "segment_stall", "target_envelope_rejected", "segment_budget_exhausted"):
                    observed = completed.get("observed_at_step", step)
                    fail_key = (start, observed)
                    if fail_key not in seen_failures:
                        seen_failures.add(fail_key)
                        for stage in snapshot.get("segments", []):
                            if stage.get("reason") == completed["reason"]:
                                reason = (stage.get("route_tracking") or {}).get("reason") or (
                                    stage.get("routing") or {}).get("reason") or stage.get("detail")
                                sid = f"f{start}_s{stage['index']}"
                                rid = "r_" + sid if "routing" in stage else None
                                if rid:
                                    latest_route = rid
                                profile = stage.get("request", {}).get("profile")
                                if completed["reason"] == "route_unavailable":
                                    event_kind, label = "route_failure", reason.replace("_", " ")
                                elif completed["reason"] == "target_envelope_rejected":
                                    event_kind, label = "target_rejection", "Target rotation rejected before routing"
                                    reason = completed["reason"]
                                elif completed["reason"] == "segment_budget_exhausted":
                                    event_kind, label = "stage_budget_exhausted", "Gripper stage budget ended before completion"
                                    reason = completed["reason"]
                                else:
                                    event_kind = "segment_stall" if profile == "transit" else "contact_stall"
                                    label = "Transit stalled" if profile == "transit" else "Contact approach stalled"
                                    reason = completed["reason"]
                                event(event_kind, observed, label,
                                      reason=reason, detail=stage.get("detail"), route_id=rid,
                                      stage_id=sid, skill=stage.get("request", {}).get("skill"),
                                      profile=profile,
                                      map_revision=(stage.get("routing") or {}).get(
                                          "map_revision", action_meta.get(observed - 1, {}).get("map_revision", latest_map)),
                                      decision_boundary_map_revision=latest_map,
                                      goal_xyz_m=stage.get("target_xyz_m"),
                                      measured_xyz_m=snapshot.get("measured_pose_wxyz", [])[:3],
                                      position_error_m=stage.get("position_error_m"),
                                      orientation_error_rad=stage.get("orientation_error_rad"),
                                      executed_actions=stage.get("actions", 0), evidence=evidence)
            decision_type = (params or {}).get("type")
            motion = (params or {}).get("motion") or {}
            action_meta[step] = {"map_revision": latest_map, "most_recent_failure_route": latest_route,
                                 "model_decision_type": decision_type, "motion_request": motion,
                                 "execution_kind": diag.get("execution_kind"),
                                 "before_xyz_m": row["state"][:3], "evidence": evidence}
        elif kind == "executed":
            step, status = row["step"], row["sim_status"]
            collision = status["collision"]
            counts = collision["counts"]
            delta = {name.removeprefix("points_"): counts.get("points_" + name, 0) - previous_counts.get("points_" + name, 0)
                     for name in collision["forbidden_classes"]}
            delta = {key: value for key, value in delta.items() if value}
            pairs = {(cls, tuple(item["pair"])): item["points"]
                     for cls, items in collision["pairs_by_class"].items() for item in items}
            obstacle_pairs = [{"class": key[0], "pair": list(key[1]),
                               "contact_point_samples": value - previous_pairs.get(key, 0)}
                              for key, value in pairs.items()
                              if key[0] in collision["forbidden_classes"]
                              and any(OBSTACLE in item for item in key[1])
                              and value > previous_pairs.get(key, 0)]
            obstacle_delta = sum(item["contact_point_samples"] for item in obstacle_pairs)
            contact_types = []
            if any(any("gripper0_finger1_collision" in geom for geom in item["pair"]) for item in obstacle_pairs):
                contact_types.append("bottle_left_finger")
            if any(any("moka_pot_" in geom for geom in item["pair"]) for item in obstacle_pairs):
                contact_types.append("bottle_moka_pot")
            if sum(delta.values()) > obstacle_delta:
                contact_types.append("other_forbidden")
            if obstacle_delta:
                finger = any(any("finger" in geom for geom in item["pair"]) for item in obstacle_pairs)
                label = "Bottle–finger contact" if finger else "Bottle–pot contact"
                event("obstacle_contact", step, label, action_index=step,
                      contact_point_samples=obstacle_delta, obstacle_name=OBSTACLE,
                      pairs=obstacle_pairs, contact_xyz_m=None,
                      map_revision=action_meta[step]["map_revision"], evidence=evidence)
            direct_motion = action_meta[step]["model_decision_type"] == "motion"
            frames[step + 1] = {"frame_index": step + 1, "control_step": step + 1,
                               "action_index": step, "sim_action_time_s": (step + 1) / hz,
                               "grip_xyz_m": row["next_state"][:3],
                               "map_revision": action_meta[step]["map_revision"],
                               "stage_id": None, "skill": "direct_motion" if direct_motion else None,
                               "profile": "direct_motion" if direct_motion else None,
                               "status": "executing_direct_motion" if direct_motion else "policy_hold_or_nonfragment", "route_id": None,
                               "last_route_id": None, "goal_xyz_m": None,
                               "contact_delta": delta, "forbidden_contact_delta": sum(delta.values()),
                               "can_contact_delta": 0, "obstacle_contact_delta": obstacle_delta,
                               "obstacle_contact_pairs": obstacle_pairs, "contact_xyz_m": [],
                               "contact_types": contact_types,
                               "contact_summary": "; ".join({"bottle_left_finger": "Bottle–left finger",
                                                              "bottle_moka_pot": "Bottle–moka pot",
                                                              "other_forbidden": "Other forbidden contact"}[item]
                                                             for item in contact_types) or None,
                               "model_decision_type": action_meta[step]["model_decision_type"],
                               "execution_kind": action_meta[step]["execution_kind"],
                               "forbidden_contact_total": counts["forbidden_contact_points"],
                               "libero_success": row.get("success", False), "event_ids": [],
                               "evidence": evidence}
            previous_counts, previous_pairs = counts, pairs
        elif kind == "feedback":
            fragment = (row.get("feedback", {}).get("primitive_feedback") or {}).get("fragment")
            if fragment:
                start = row["step"] + 1 - fragment["executed_steps"]
                capture(fragment, start, evidence)
                # Use the stage which actually produced this action. The next
                # stage can already appear as current with zero actions.
                for segment in fragment.get("segments", []):
                    entry = start + segment.get("entry_at_fragment_step", 0)
                    if entry <= row["step"] < entry + segment.get("actions", 0):
                        frame = frames[row["step"] + 1]
                        sid = f"f{start}_s{segment['index']}"
                        frame.update(stage_id=sid, skill=segment["request"]["skill"],
                                     profile=segment["request"]["profile"], status=segment["status"],
                                     goal_xyz_m=segment.get("target_xyz_m"),
                                     route_id="r_" + sid if "routing" in segment else None)
        elif kind == "recorded_collision":
            first = row["first"]
            owners = [first["geom1"]["owner"], first["geom2"]["owner"]]
            roles = [first["geom1"]["role"], first["geom2"]["role"]]
            bottle = OBSTACLE in owners
            label = ("First bottle–finger contact" if "robot_finger" in roles else "First bottle–pot contact") if bottle else "First forbidden contact"
            event("first_forbidden_contact", row["step"],
                  label,
                  action_index=row["step"], map_revision=action_meta[row["step"]]["map_revision"],
                  contact_class=row["class"], physics_substep=first["physics_substep"],
                  contact_xyz_m=first["contact_pos"],
                  owners=owners, roles=roles, geometries=[first["geom1"]["name"], first["geom2"]["name"]],
                  obstacle_name=OBSTACLE if bottle else None, evidence=evidence)
            frames[row["step"] + 1]["contact_xyz_m"].append(first["contact_pos"])

    assert len(seen_outputs) == len(outputs) == audit["routing_interpretation"]["accepted_agent_outputs"], (len(seen_outputs), len(outputs))
    assert len(maps) == len(audit["routing_interpretation"]["map_updates"]) + 1
    assert len(frames) == summary["steps"] + 1
    assert len(route_entries) == audit["routing_interpretation"]["routing_stages"]
    assert sum(frame["obstacle_contact_delta"] for frame in frames.values()) == audit["collisions"]["policy_obstacle_forbidden_contact_points"]
    assert sum(frame["forbidden_contact_delta"] for frame in frames.values()) == audit["collisions"]["policy_forbidden_contact_points"]
    event("terminal", summary["steps"], "Raw success; strict held/rest checks fail",
          map_revision=latest_map, libero_success=summary["libero_success"],
          v2_success=summary["v2_success"],
          physical_success_at_terminal=summary["physical_truth"]["physical_success_at_terminal"],
          physical_success_after_settle=summary["physical_truth"]["physical_success_after_settle"],
          terminal_failing_checks=summary["physical_truth"]["terminal_failing_checks"],
          terminal_truth_window_note="Strict held/rest checks include the evaluator's recent-state evidence window; they do not prove current instantaneous attachment.",
          diagnostic_frames_included=False, evidence="trial/physical_truth.json")
    for item in events:
        assert item["frame_index"] in frames
        frames[item["frame_index"]]["event_ids"].append(item["id"])
        if item["action_index"] is not None:
            endpoint = frames[item["frame_index"]]
            item.update(stage_id=endpoint["stage_id"], skill=endpoint["skill"],
                        profile=endpoint["profile"], route_id=endpoint["route_id"])
    latest_route = None
    boundary_map = 0
    for index in sorted(frames):
        frame = frames[index]
        for event_id in frame["event_ids"]:
            if events[event_id]["kind"] == "route_failure":
                latest_route = events[event_id]["route_id"]
            if events[event_id]["kind"] == "map_update":
                boundary_map = events[event_id]["map_revision"]
        if frame["route_id"] is not None:
            latest_route = frame["route_id"]
            assert frame["profile"] == "transit"
            assert route_entries[latest_route]["map_revision"] == frame["map_revision"]
        frame["last_route_id"] = latest_route
        frame["decision_boundary_map_revision"] = boundary_map
        frame["incoming_action_map_revision"] = frame["map_revision"]
    data = {
        "schema": "recorded_obstacle_map_video_timeline_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(RUN), "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "frame": "world", "units": "meters", "position_reference": "robot grip_site",
        "control_hz": hz, "policy_actions": summary["steps"], "frame_count": len(frames),
        "time_semantics": "Frame/state index0 is the reset observation; index n+1 is the measured state after action n. Simulator action time only, model latency omitted. Decision/failure events use their exact detected state index, without adding one. Contact events during action n use endpoint state n+1 and retain action_index/substep. A map update at boundary n occurs after state n was observed and before action n; frames.map_revision describes the incoming action, while decision_boundary_map_revision applies accepted updates at that exact state.",
        "image_boundary_semantics": "Align robot images to saved observation state indices, not directly to action indices. A contact during action n appears at state/image n+1. A decision/failure at observed state n uses state/image n. This dataset contains no rendered planning frames and excludes the diagnosis-only release/settle intervention.",
        "geometry_semantics": "All boxes are model estimates. B1/B2/... are positional labels, not object identities; obstacle_box_index and legacy can_box_index remain null. No simulator obstacle geometry or signed distance field is fabricated.",
        "route_semantics": "route_id belongs only to the transit stage that executed the frame's action. Contact stages have null route_id. last_route_id is context only. A failed zero-action route appears as an event, not an executed route. Paths are grip-site paths, not full robot or held-object collision guarantees.",
        "contact_semantics": "Forbidden contact-point samples over physics checks, not unique impacts. First recorded class events contain measured contact XYZ and physics substep. Per-action contact counts and pair identities are cumulative-count differences; no position is invented for those events. obstacle_contact_delta counts forbidden contacts involving ketchup_obstacle_1; can_contact_delta is a legacy compatibility field fixed at zero because no can is present.",
        "obstacle_name": OBSTACLE,
        "initial_state8": initial_state,
        "maps": maps, "routes": list(route_entries.values()), "stages": list(stages.values()),
        "events": events, "frames": [frames[i] for i in sorted(frames)],
        "summary": {"libero_success": summary["libero_success"], "v2_success": summary["v2_success"],
                    "physical_success_at_terminal": summary["physical_truth"]["physical_success_at_terminal"],
                    "physical_success_after_settle": summary["physical_truth"]["physical_success_after_settle"],
                    "diagnostic_settle_included_in_frames": False,
                    "map_updates": len(maps) - 1, "route_attempts": len(route_entries),
                    "direct_routes": sum(r["straight_path_valid"] is True for r in route_entries.values()),
                    "detour_routes": sum(r["straight_path_valid"] is False and len(r["path_xyz_m"]) > 1 for r in route_entries.values()),
                    "forbidden_contact_points": audit["collisions"]["policy_forbidden_contact_points"],
                    "can_contact_points": 0,
                    "obstacle_contact_points": audit["collisions"]["policy_obstacle_forbidden_contact_points"],
                    "route_failures": [event for event in events if event["kind"] == "route_failure"],
                    "transit_stalls": [event for event in events if event["kind"] == "segment_stall"],
                    "contact_stalls": [event for event in events if event["kind"] == "contact_stall"],
                    "target_rejections": [event for event in events if event["kind"] == "target_rejection"],
                    "stage_budget_exhaustions": [event for event in events if event["kind"] == "stage_budget_exhausted"],
                    "terminal_failing_checks": summary["physical_truth"]["terminal_failing_checks"],
                    "terminal_failure_is_hold_only": summary["physical_truth"]["terminal_failure_is_hold_only"],
                    "diagnostic_settle": audit["physical_truth"]["diagnostic_settle"],
                    "final_policy_stage_details": audit["physical_truth"]["final_policy_stage_details"],
                    "command_configuration": audit["command_configuration"]},
    }
    target = OUT / "obstacle_map_timeline.json"
    target.write_text(json.dumps(rounded(data), separators=(",", ":")) + "\n")
    print(json.dumps({"path": str(target), "bytes": target.stat().st_size,
                      "frames": len(frames), "maps": len(maps), "routes": len(route_entries),
                      "map_update_steps": [m["update_control_step"] for m in maps],
                      "events": Counter(e["kind"] for e in events)}, indent=2))


if __name__ == "__main__":
    extract()
