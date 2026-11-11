"""Build a source-linked T08 visualization timeline from saved artifacts only.

No simulator, policy, model, or runtime modules are imported. The completed run is
read-only. Relative goals resolve at measured stage entry, never retroactively.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/foundation_vision_t08_bottle_obstacle_20261105_001"
OUT = Path(__file__).resolve().parent
OBSTACLE = "ketchup_obstacle_1"
MOTION = {"move_to_pose", "move_relative", "approach"}


def read(path):
    return json.loads(path.read_text())


def public_text(response):
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    return "".join(p.get("text", "") for i in response.get("output", [])
                   if i.get("type") == "message" for p in i.get("content", [])
                   if p.get("type") == "output_text")


def quaternion(rotvec):
    angle = math.sqrt(sum(x * x for x in rotvec))
    scale = math.sin(angle / 2) / angle if angle else .5
    return [math.cos(angle / 2), *(x * scale for x in rotvec)]


def extract():
    trial = RUN / "trial"
    source_paths = set()

    def source(path):
        source_paths.add(path)
        return str(path.relative_to(RUN))

    summary = read(trial / "summary.json")
    manifest = read(trial / "manifest.json")
    vision = read(trial / "foundation_vision/summary.json")
    for name in ("summary.json", "manifest.json", "foundation_vision/summary.json"):
        source(trial / name)
    hz = manifest["configuration"]["active_agent"]["framework"]["execution_fragments"]["control_hz"]
    decisions = []
    for path in sorted((trial / "requests").glob("*.response.json")):
        result_path = path.with_name(path.name.replace(".response.json", ".result.json"))
        result = read(result_path)
        if result.get("status") != "valid" or result.get("request_kind") != "agent":
            continue
        request_path = path.with_name(path.name.replace(".response.json", ".request.json"))
        request = read(request_path)
        value = json.loads(public_text(read(path)))
        decisions.append({"call_index": len(decisions) + 1,
                          "control_step": request["request_meta"]["step"],
                          "response_file": source(path), "request_file": source(request_path),
                          "result_file": source(result_path), "value": value})
    by_step = {d["control_step"]: d for d in decisions}
    assert len(by_step) == len(decisions) == summary["api_calls"]
    maps = [{"revision": 0, "update_control_step": None, "source": "uninitialized",
             "avoid_boxes": [], "clearance_m": 0., "box_labels": [],
             "obstacle_box_index": None, "can_box_index": None}]
    stages, routes, frames, action_meta, events = {}, {}, {}, {}, []
    latest_map, latest_call = 0, None
    previous_counts, previous_pairs = {}, {}
    trace_counts = Counter()

    def event(kind, step, label, action_index=None, **details):
        state = step if action_index is None else action_index + 1
        item = {"id": len(events), "kind": kind, "control_step": step,
                "detected_at_state_index": state, "frame_index": state,
                "action_index": action_index, "label": label, **details}
        events.append(item)

    def capture(fragment, start, evidence):
        if not isinstance(fragment, dict) or not isinstance(start, int):
            return
        for raw in fragment.get("segments", fragment.get("segment_trace", [])):
            sid = f"f{start}_s{raw['index']}"
            prior = stages.get(sid)
            if prior and prior["executed_actions"] > raw.get("actions", 0):
                continue
            request = raw.get("request", {})
            end = raw.get("end_at_fragment_step")
            stage = {"id": sid, "fragment_start_step": start, "stage_index": raw["index"],
                     "entry_control_step": start + raw.get("entry_at_fragment_step", 0),
                     "end_control_step": start + end if end is not None else None,
                     "skill": request.get("skill"), "profile": request.get("profile"),
                     "request": request, "status": raw.get("status"), "reason": raw.get("reason"),
                     "executed_actions": raw.get("actions", 0),
                     "entry_xyz_m": raw.get("entry_pose_wxyz", [])[:3],
                     "entry_quaternion_wxyz": raw.get("entry_pose_wxyz", [])[3:7],
                     "goal_xyz_m": raw.get("target_xyz_m"),
                     "target_quaternion_wxyz": raw.get("target_quaternion_wxyz"),
                     "last_pose_wxyz": raw.get("last_pose_wxyz"),
                     "position_error_m": raw.get("position_error_m"),
                     "orientation_error_rad": raw.get("orientation_error_rad"),
                     "route_id": None, "gripper": request.get("gripper"),
                     "object_grasp_verified": raw.get("object_grasp_verified", False),
                     "evidence": evidence}
            if "routing" in raw:
                route = raw["routing"]
                rid = "r_" + sid
                stage["route_id"] = rid
                routes[rid] = {"id": rid, "stage_id": sid,
                               "entry_control_step": stage["entry_control_step"],
                               "end_control_step": stage["end_control_step"],
                               "map_revision": route.get("map_revision"),
                               "map_source": route.get("map_source"),
                               "geometry_source": route.get("geometry_source"),
                               "start_xyz_m": stage["entry_xyz_m"],
                               "goal_xyz_m": stage["goal_xyz_m"],
                               "path_xyz_m": route.get("path_xyz_m", []),
                               "straight_path_valid": route.get("straight_path_valid"),
                               "expanded_boxes": route.get("expanded_boxes", []),
                               "clearance_m": route.get("clearance_m"),
                               "planner_search_calls": (route.get("planner") or {}).get("calls", 0),
                               "planner_methods": (route.get("planner") or {}).get("methods", []),
                               "status": stage["status"], "reason": stage["reason"],
                               "route_reason": route.get("reason"),
                               "tracking_reason": (raw.get("route_tracking") or {}).get("reason"),
                               "executed_actions": stage["executed_actions"], "evidence": evidence}
            stages[sid] = stage

    def make_frame(index, state, evidence, **extra):
        return {"frame_index": index, "control_step": index, "action_index": index - 1 if index else None,
                "sim_action_time_s": index / hz, "state8": state, "grip_xyz_m": state[:3],
                "measured_rotvec_rad": state[3:6], "measured_quaternion_wxyz": quaternion(state[3:6]),
                "map_revision": 0, "stage_id": None, "skill": None, "profile": None,
                "status": "initial_observation" if not index else "policy_hold_or_nonfragment",
                "route_id": None, "last_route_id": None, "goal_xyz_m": None,
                "target_quaternion_wxyz": None, "contact_delta": {},
                "forbidden_contact_delta": 0, "can_contact_delta": 0, "obstacle_contact_delta": 0,
                "obstacle_contact_pairs": [], "contact_xyz_m": [], "contact_types": [],
                "contact_summary": None, "forbidden_contact_total": 0,
                "libero_success": False, "event_ids": [], "evidence": evidence, **extra}

    trace = trial / "trace.jsonl"
    source(trace)
    for lineno, line in enumerate(trace.open(), 1):
        row = json.loads(line)
        kind = row.get("event")
        trace_counts[kind] += 1
        evidence = f"trial/trace.jsonl:{lineno}"
        if kind == "reset":
            initial_state = row["observation"]["observation/state"]
            previous_counts = row["sim_status"]["collision"]["counts"]
            previous_pairs = {(cls, tuple(item["pair"])): item["points"]
                              for cls, items in row["sim_status"]["collision"].get("pairs_by_class", {}).items()
                              for item in items}
            frames[0] = make_frame(0, initial_state, evidence)
        elif kind == "action":
            step, diag = row["step"], row.get("diagnostics", {})
            params = (diag.get("routing") or {}).get("params") or {}
            if step in by_step:
                decision = by_step[step]
                assert params == decision["value"], (step, "accepted response mismatch")
                latest_call = decision["call_index"]
                update = params.get("obstacle_map")
                if update is not None:
                    latest_map += 1
                    maps.append({"revision": latest_map, "update_control_step": step,
                                 "source": "model_estimate", **update,
                                 "box_labels": [f"B{i+1}" for i in range(len(update["avoid_boxes"]))],
                                 "obstacle_box_index": None, "can_box_index": None,
                                 "object_identity": "unlabeled_model_estimates",
                                 "call_index": latest_call, "response_file": decision["response_file"]})
                    event("map_update", step, f"Astra map updated: revision {latest_map}",
                          map_revision=latest_map, box_count=len(update["avoid_boxes"]), evidence=evidence)
                decision["map_revision"] = latest_map
            completed = diag.get("feedback") or {}
            if completed.get("action") == "execute":
                capture(completed.get("evidence"), (completed.get("start") or {}).get("step"), evidence)
            action_meta[step] = {"map_revision": latest_map, "call_index": latest_call,
                                 "model_decision_type": params.get("type"),
                                 "execution_kind": diag.get("execution_kind"), "evidence": evidence}
        elif kind == "executed":
            step = row["step"]
            collision = row["sim_status"]["collision"]
            counts = collision["counts"]
            delta = {name: counts.get("points_" + name, 0) - previous_counts.get("points_" + name, 0)
                     for name in collision["forbidden_classes"]}
            delta = {k: v for k, v in delta.items() if v}
            assert all(v >= 0 for v in delta.values())
            pairs = {(cls, tuple(item["pair"])): item["points"]
                     for cls, items in collision["pairs_by_class"].items() for item in items}
            obstacle_pairs = [{"class": key[0], "pair": list(key[1]),
                               "contact_point_samples": value - previous_pairs.get(key, 0)}
                              for key, value in pairs.items()
                              if key[0] in collision["forbidden_classes"]
                              and any(OBSTACLE in part for part in key[1])
                              and value > previous_pairs.get(key, 0)]
            obstacle_delta = sum(item["contact_point_samples"] for item in obstacle_pairs)
            if obstacle_delta:
                event("obstacle_contact", step, "Bottle contact recorded", action_index=step,
                      contact_point_samples=obstacle_delta, pairs=obstacle_pairs,
                      contact_xyz_m=None, evidence=evidence)
            meta = action_meta[step]
            frames[step + 1] = make_frame(step + 1, row["next_state"], evidence,
                                          map_revision=meta["map_revision"], incoming_call_index=meta["call_index"],
                                          contact_delta=delta, forbidden_contact_delta=sum(delta.values()),
                                          obstacle_contact_delta=obstacle_delta, obstacle_contact_pairs=obstacle_pairs,
                                          contact_types=["other_forbidden"] if delta else [],
                                          contact_summary="Pot–stove forbidden contact" if delta and not obstacle_delta else None,
                                          model_decision_type=meta["model_decision_type"],
                                          execution_kind=meta["execution_kind"],
                                          forbidden_contact_total=counts["forbidden_contact_points"],
                                          libero_success=row.get("success", False))
            previous_counts, previous_pairs = counts, pairs
        elif kind == "feedback":
            fragment = (row.get("feedback", {}).get("primitive_feedback") or {}).get("fragment")
            if fragment:
                start = row["step"] + 1 - fragment["executed_steps"]
                assert start in by_step
                capture(fragment, start, evidence)
                for segment in fragment.get("segments", []):
                    entry = start + segment.get("entry_at_fragment_step", 0)
                    if entry <= row["step"] < entry + segment.get("actions", 0):
                        sid = f"f{start}_s{segment['index']}"
                        frames[row["step"] + 1].update(
                            stage_id=sid, skill=segment["request"]["skill"], profile=segment["request"]["profile"],
                            status=segment["status"], goal_xyz_m=segment.get("target_xyz_m"),
                            target_quaternion_wxyz=segment.get("target_quaternion_wxyz"),
                            route_id="r_" + sid if "routing" in segment else None,
                            position_error_m=segment.get("position_error_m"),
                            orientation_error_rad=segment.get("orientation_error_rad"))
                final_fragment = fragment
        elif kind == "recorded_collision":
            first = row["first"]
            event("first_forbidden_contact", row["step"], "First pot–stove forbidden contact",
                  action_index=row["step"], map_revision=action_meta[row["step"]]["map_revision"],
                  contact_class=row["class"], physics_substep=first["physics_substep"],
                  contact_xyz_m=first["contact_pos"], owners=[first["geom1"]["owner"], first["geom2"]["owner"]],
                  roles=[first["geom1"]["role"], first["geom2"]["role"]],
                  geometries=[first["geom1"]["name"], first["geom2"]["name"]],
                  obstacle_name=None, evidence=evidence)
            frames[row["step"] + 1]["contact_xyz_m"].append(first["contact_pos"])

    keyposes = []
    for i, decision in enumerate(decisions):
        start = decision["control_step"]
        handoff = decisions[i + 1]["control_step"] if i + 1 < len(decisions) else summary["steps"]
        decision["end_control_step"] = handoff
        decision["keypose_ids"] = []
        for index, request in enumerate(decision["value"].get("execution", {}).get("segments", [])):
            sid = f"f{start}_s{index}"
            stage = stages.get(sid)
            motion = request["skill"] in MOTION
            final_status = "not_executed"
            if stage:
                if stage["status"] == "succeeded":
                    final_status = "reached" if motion and stage["reason"] == "pose_arrived" else "completed"
                elif stage["status"] == "failed":
                    final_status = "failed"
                else:
                    final_status = "active"
            history = [{"control_step": start, "status": "requested", "evidence": decision["response_file"]}]
            if stage:
                history.append({"control_step": stage["entry_control_step"], "status": "active", "evidence": stage["evidence"]})
                if stage["end_control_step"] is not None:
                    history.append({"control_step": stage["end_control_step"], "status": final_status,
                                    "reason": stage["reason"], "evidence": stage["evidence"]})
            else:
                history.append({"control_step": handoff, "status": "not_executed",
                                "reason": "earlier_segment_ended_fragment", "evidence": decision["response_file"]})
            key = {"id": f"k{decision['call_index']:02}_s{index}", "call_index": decision["call_index"],
                   "decision_control_step": start, "decision_end_control_step": handoff,
                   "segment_index": index, "fragment_index": index, "keypose_number": index + 1,
                   "skill": request["skill"], "profile": request["profile"], "gripper": request["gripper"],
                   "is_pose_keypose": motion, "requested_xyz_m": request.get("xyz_m"),
                   "relative_delta_m": request.get("xyz_m") if request["skill"] in {"approach", "move_relative"} else None,
                   "absolute_requested_goal_xyz_m": request.get("xyz_m") if request["skill"] == "move_to_pose" else None,
                   "explicit_quaternion_wxyz": request.get("quaternion_wxyz"),
                   "orientation_source": "explicit_model_request" if request.get("quaternion_wxyz") else "inherited_at_measured_entry",
                   "entry_control_step": stage["entry_control_step"] if stage else None,
                   "end_control_step": stage["end_control_step"] if stage else None,
                   "resolved_goal_xyz_m": stage["goal_xyz_m"] if stage else None,
                   "resolved_quaternion_wxyz": stage["target_quaternion_wxyz"] if stage else None,
                   "resolved_goal_visible_from_control_step": stage["entry_control_step"] if stage else None,
                   "status": final_status, "status_reason": stage["reason"] if stage else "earlier_segment_ended_fragment",
                   "status_history": history, "executed_actions": stage["executed_actions"] if stage else 0,
                   "stage_id": sid if stage else None, "route_id": stage["route_id"] if stage else None,
                   "position_error_m": stage["position_error_m"] if stage else None,
                   "orientation_error_rad": stage["orientation_error_rad"] if stage else None,
                   "reached_pose_evidence": stage["evidence"] if final_status == "reached" else None,
                   "evidence": [decision["response_file"]] + ([stage["evidence"]] if stage else [])}
            keyposes.append(key)
            decision["keypose_ids"].append(key["id"])
    for stage in stages.values():
        if stage["status"] != "failed":
            continue
        reason = stage["reason"]
        event_kind = {"route_unavailable": "route_failure", "target_envelope_rejected": "target_rejection",
                      "segment_budget_exhausted": "stage_budget_exhausted"}.get(reason,
                       "segment_stall" if stage["profile"] == "transit" else "contact_stall")
        event(event_kind, stage["end_control_step"], reason.replace("_", " "), reason=reason,
              stage_id=stage["id"], route_id=stage["route_id"], skill=stage["skill"], profile=stage["profile"],
              goal_xyz_m=stage["goal_xyz_m"], measured_xyz_m=(stage["last_pose_wxyz"] or [])[:3],
              position_error_m=stage["position_error_m"], orientation_error_rad=stage["orientation_error_rad"],
              executed_actions=stage["executed_actions"], evidence=stage["evidence"])
    truth = summary["physical_truth"]
    event("terminal", summary["steps"], "Raw LIBERO success; second pot not autonomously released",
          libero_success=summary["libero_success"], physical_success_at_terminal=truth["physical_success_at_terminal"],
          physical_success_after_settle=truth["physical_success_after_settle"],
          terminal_failing_checks=truth["terminal_failing_checks"], diagnostic_frames_included=False,
          evidence="trial/physical_truth.json")
    events.sort(key=lambda e: (e["frame_index"], e["id"]))
    for idx, item in enumerate(events):
        item["id"] = idx
        frames[item["frame_index"]]["event_ids"].append(idx)
        if item["action_index"] is not None:
            frame = frames[item["frame_index"]]
            item.update({k: frame[k] for k in ("stage_id", "skill", "profile", "route_id")})
    latest_route, boundary_map, current_call = None, 0, None
    for index, frame in sorted(frames.items()):
        if index in by_step:
            current_call = by_step[index]["call_index"]
        for eid in frame["event_ids"]:
            if events[eid]["kind"] == "map_update":
                boundary_map = events[eid]["map_revision"]
        if frame["route_id"]:
            latest_route = frame["route_id"]
            assert routes[latest_route]["map_revision"] == frame["map_revision"]
        frame.update(last_route_id=latest_route, decision_boundary_map_revision=boundary_map,
                     incoming_action_map_revision=frame["map_revision"], call_index=current_call)
    vision_requests_path = trial / "foundation_vision/requests.jsonl"
    vision_frames = []
    for lineno, line in enumerate(vision_requests_path.open(), 1):
        row = json.loads(line)
        objects = row["summary"].get("objects", [])
        vision_frames.append({"control_step": row["step"], "request_index": row["request_index"],
                              "consumed": row["consumed"], "age_s": row["summary"]["age_s"],
                              "supported_centroids": sum(o.get("centroid_world_m") is not None for o in objects),
                              "candidate_counts": {o["key"]: o.get("candidate_counts", {}) for o in objects},
                              "snapshot_file": "trial/foundation_vision/" + row["snapshot_path"],
                              "evidence": f"trial/foundation_vision/requests.jsonl:{lineno}"})
    source(vision_requests_path)
    episode_path = trial / "episode.events.jsonl"
    episode_counts = Counter(json.loads(line)["event"] for line in episode_path.open())
    source(episode_path)
    assert len(frames) == summary["steps"] + 1 == 591
    assert trace_counts["action"] == trace_counts["executed"] == trace_counts["feedback"] == 590
    assert sum(s["executed_actions"] for s in stages.values()) == summary["steps"]
    assert sum(f["forbidden_contact_delta"] for f in frames.values()) == summary["collision"]["counts"]["forbidden_contact_points"] == 400
    assert sum(f["obstacle_contact_delta"] for f in frames.values()) == 0
    assert len(decisions) == len(vision_frames) == 11
    assert sum(f["supported_centroids"] for f in vision_frames) == vision["supported_centroids"] == 0
    assert [m["update_control_step"] for m in maps[1:]] == [0, 188, 328]
    assert len(keyposes) == 22 and len(stages) == 21 and len(routes) == 12
    for key in keyposes:
        if key["status"] == "reached":
            tolerance = manifest["configuration"]["active_agent"]["framework"]["execution_fragments"]["profiles"][key["profile"]]
            assert key["status_reason"] == "pose_arrived"
            assert key["position_error_m"] <= tolerance["position_tolerance_m"]
            assert key["orientation_error_rad"] <= tolerance["rotation_tolerance_rad"]
        if key["stage_id"] is None:
            assert key["resolved_goal_xyz_m"] is None and key["executed_actions"] == 0
    data = {"schema": "recorded_obstacle_map_video_timeline_v2", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(RUN), "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "source_files_sha256": {str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)},
            "frame": "world", "units": "meters", "position_reference": "robot grip_site", "orientation_reference": "robot right_hand",
            "control_hz": hz, "policy_actions": summary["steps"], "frame_count": len(frames),
            "time_semantics": "State 0 is reset; state n+1 is after action n. Contacts during action n use state n+1. Decisions, map updates, and failures at boundary n use state n. Incoming-action map and current boundary map are distinct. Model latency omitted from simulator action time.",
            "image_boundary_semantics": "Use frame_mapping.json/control_state_frames.json from episode.events to select recorded RGB. State timeline excludes diagnostic-only release/settle actions.",
            "geometry_semantics": "Boxes are Astra model estimates, not SAM/DA3 geometry. B1/B2/B3 are positional labels with unknown object identity. SAM3.1 + DA3 supplied no supported target 3D centers in all 11 calls. No oracle geometry or reconstructed free space is displayed.",
            "route_semantics": "Transit stage-entry paths are grip-site routes. They do not certify swept robot/payload clearance. Relative goals resolve from measured entry; keypose chains are model intentions, not planner paths. Contact stages have no route. last_route_id is context only.",
            "keypose_semantics": "Each request segment keeps its original call-local number. World absolute requested goals may be shown from decision time. Relative targets and inherited orientation may be shown only at/after entry_control_step. reached requires runtime pose_arrived and measured tolerances; completed gripper means gap_stable, not verified grasp. not_executed means requested but never entered. active at terminal remains unfinished despite raw task success.",
            "contact_semantics": "Contact values count contact-point samples, not independent impacts. XYZ exists only for the recorded first forbidden event; per-action contact pairs/counts do not invent positions. Bottle counts specifically involve ketchup_obstacle_1. Legacy can counts are zero.",
            "obstacle_name": OBSTACLE, "initial_state8": initial_state, "maps": maps,
            "routes": list(routes.values()), "stages": list(stages.values()), "decisions": decisions,
            "keyposes": keyposes, "vision_frames": vision_frames, "events": events,
            "frames": [frames[i] for i in sorted(frames)],
            "summary": {"libero_success": summary["libero_success"], "v2_success": summary["v2_success"],
                        "physical_success_at_terminal": truth["physical_success_at_terminal"],
                        "physical_success_after_settle": truth["physical_success_after_settle"],
                        "terminal_failing_checks": truth["terminal_failing_checks"],
                        "terminal_failure_is_hold_only": truth["terminal_failure_is_hold_only"],
                        "diagnostic_settle_included_in_frames": False, "diagnostic_extra_actions_excluded": 38,
                        "terminal_gripper_command": final_fragment["gripper_command"],
                        "terminal_gap_mm": final_fragment["measured_gap_mm"],
                        "terminal_fragment_status": final_fragment["status"],
                        "terminal_fragment_sequence_complete": final_fragment["sequence_complete"],
                        "map_updates": len(maps)-1, "route_attempts": len(routes),
                        "direct_routes": sum(r["straight_path_valid"] is True for r in routes.values()),
                        "detour_routes": sum(r["straight_path_valid"] is False and len(r["path_xyz_m"]) > 1 for r in routes.values()),
                        "forbidden_contact_points": 400, "obstacle_contact_points": 0, "can_contact_points": 0,
                        "keypose_status_counts": dict(Counter(k["status"] for k in keyposes)),
                        "requested_segments": len(keyposes), "entered_segments": len(stages),
                        "supported_target_centroids": vision["supported_centroids"],
                        "vision_inferences": vision["vision_inferences_completed"], "vision_errors": vision["errors"],
                        "executive_calls": len(decisions), "initial_plan_calls": 1, "api_key_calls": summary["api_key_calls"],
                        "control_without_video_s": summary["control_without_video_s"],
                        "trace_event_counts": dict(trace_counts), "video_event_counts": dict(episode_counts),
                        **{name: [e for e in events if e["kind"] == kind] for name, kind in
                           (("route_failures", "route_failure"), ("transit_stalls", "segment_stall"),
                            ("contact_stalls", "contact_stall"), ("target_rejections", "target_rejection"),
                            ("stage_budget_exhaustions", "stage_budget_exhausted"))}}}
    text = json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ("timeline.json", "obstacle_map_timeline.json"):
        (OUT / filename).write_text(text)
    report = {"path": str(OUT / "timeline.json"), "bytes": len(text.encode()),
              "frames": len(frames), "maps": len(maps), "routes": len(routes),
              "keypose_status_counts": data["summary"]["keypose_status_counts"],
              "map_update_steps": [m["update_control_step"] for m in maps[1:]],
              "event_counts": dict(Counter(e["kind"] for e in events)),
              "forbidden_contact_points": 400, "bottle_contact_points": 0,
              "supported_target_centroids": 0, "validation": "all assertions passed"}
    (OUT / "timeline_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return data


if __name__ == "__main__":
    extract()
