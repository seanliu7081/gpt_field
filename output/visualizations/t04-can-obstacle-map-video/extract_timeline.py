"""Extract a compact animation timeline from saved observations; no replay."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/agent_speedup_v2_uc71_can_obstacle_t04_20261103_001"
OUT = Path(__file__).resolve().parent


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
    hz = manifest["configuration"]["active_agent"]["framework"]["execution_fragments"]["control_hz"]
    outputs = []
    for path in sorted((trial / "requests").glob("*.response.json")):
        result = read(path.with_name(path.name.replace(".response.json", ".result.json")))
        if result.get("status") == "valid" and result.get("request_kind") == "agent":
            outputs.append({"source": str(path.relative_to(RUN)),
                            "value": json.loads(public_text(read(path)))})

    stages, frames, action_meta, events = {}, {}, {}, []
    maps = [{"revision": 0, "update_control_step": None, "source": "uninitialized",
             "avoid_boxes": [], "clearance_m": 0., "can_box_index": None, "box_labels": []}]
    previous_counts, previous_pairs = {}, {}
    latest_map, latest_route = 0, None
    seen_outputs, seen_failures, route_entries = set(), set(), {}
    initial_state = None

    def event(kind, control_step, label, **details):
        item = {"id": len(events), "kind": kind, "control_step": control_step,
                "frame_index": control_step + 1, "label": label, **details}
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
                         "can_contact_delta": 0, "forbidden_contact_total": 0,
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
                    can_index = 2 if latest_map <= 2 else 1
                    maps.append({"revision": latest_map, "update_control_step": step,
                                 "source": "model_estimate", **update,
                                 "box_labels": [f"B{i + 1}" for i in range(len(update["avoid_boxes"]))],
                                 "can_box_index": can_index,
                                 "can_box_identity": "inferred_from_location_and_public_can_clearance_plan",
                                 "response_file": output["source"]})
                    event("map_update", step, f"Map updated: revision {latest_map}",
                          map_revision=latest_map, box_count=len(update["avoid_boxes"]))
            completed = diag.get("feedback") or {}
            if completed.get("action") == "execute":
                start = (completed.get("start") or {}).get("step")
                snapshot = completed.get("evidence") or {}
                capture(snapshot, start, evidence)
                if completed.get("reason") == "route_unavailable":
                    observed = completed.get("observed_at_step", step)
                    fail_key = (start, observed)
                    if fail_key not in seen_failures:
                        seen_failures.add(fail_key)
                        for stage in snapshot.get("segments", []):
                            if stage.get("reason") == "route_unavailable":
                                reason = (stage.get("route_tracking") or {}).get("reason") or (
                                    stage.get("routing") or {}).get("reason") or stage.get("detail")
                                rid = f"r_f{start}_s{stage['index']}"
                                latest_route = rid
                                event("route_failure", observed, reason.replace("_", " "),
                                      reason=reason, route_id=rid,
                                      goal_xyz_m=stage.get("target_xyz_m"), evidence=evidence)
            action_meta[step] = {"map_revision": latest_map, "most_recent_failure_route": latest_route,
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
            can_delta = sum(value - previous_pairs.get(key, 0) for key, value in pairs.items()
                            if key[0] in collision["forbidden_classes"]
                            and any("tomato_sauce_obstacle_1" in item for item in key[1]))
            if can_delta:
                event("can_contact", step, "Can–mug contact", contact_point_samples=can_delta,
                      evidence=evidence)
            frames[step + 1] = {"frame_index": step + 1, "control_step": step + 1,
                               "action_index": step, "sim_action_time_s": (step + 1) / hz,
                               "grip_xyz_m": row["next_state"][:3],
                               "map_revision": action_meta[step]["map_revision"],
                               "stage_id": None, "skill": None, "profile": None,
                               "status": "policy_hold_or_nonfragment", "route_id": None,
                               "last_route_id": None, "goal_xyz_m": None,
                               "contact_delta": delta, "forbidden_contact_delta": sum(delta.values()),
                               "can_contact_delta": can_delta,
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
            event("first_forbidden_contact", row["step"],
                  "First palm–mug contact" if row["class"] == "robot_nonfinger_manipulated_object" else "First can–mug contact",
                  contact_class=row["class"], physics_substep=first["physics_substep"],
                  contact_xyz_m=first["contact_pos"],
                  owners=[first["geom1"]["owner"], first["geom2"]["owner"]], evidence=evidence)

    assert len(seen_outputs) == len(outputs) == 26, (len(seen_outputs), len(outputs))
    assert len(maps) == 6, len(maps)
    assert len(frames) == summary["steps"] + 1 == 1277
    assert len(route_entries) == 15
    assert sum(frame["can_contact_delta"] for frame in frames.values()) == 33
    assert sum(frame["forbidden_contact_delta"] for frame in frames.values()) == 1325
    for item in events:
        assert item["frame_index"] in frames
        frames[item["frame_index"]]["event_ids"].append(item["id"])
    latest_route = None
    for index in sorted(frames):
        frame = frames[index]
        for event_id in frame["event_ids"]:
            if events[event_id]["kind"] == "route_failure":
                latest_route = events[event_id]["route_id"]
        if frame["route_id"] is not None:
            latest_route = frame["route_id"]
            assert frame["profile"] == "transit"
            assert route_entries[latest_route]["map_revision"] == frame["map_revision"]
        frame["last_route_id"] = latest_route
    data = {
        "schema": "recorded_obstacle_map_video_timeline_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(RUN), "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "frame": "world", "units": "meters", "position_reference": "robot grip_site",
        "control_hz": hz, "policy_actions": summary["steps"], "frame_count": len(frames),
        "time_semantics": "Initial reset frame plus one frame after each recorded policy action. Simulator action time only; model latency is omitted. Event control_step is its exact action-boundary onset; frame_index is the following post-action frame.",
        "geometry_semantics": "All boxes are model estimates. B1/B2/... are positional labels, not object identities. can_box_index is a location-based inference supported by the public can-clearance plan. No simulator obstacle geometry or signed distance field is fabricated.",
        "route_semantics": "route_id belongs only to the transit stage that executed the frame's action. Contact stages have null route_id. last_route_id is context only. A failed zero-action route appears as an event, not an executed route. Paths are grip-site paths, not full robot or held-object collision guarantees.",
        "contact_semantics": "Forbidden contact-point samples over physics checks, not unique impacts. No contact position is invented where the trace only records cumulative counts.",
        "initial_state8": initial_state,
        "maps": maps, "routes": list(route_entries.values()), "stages": list(stages.values()),
        "events": events, "frames": [frames[i] for i in sorted(frames)],
        "summary": {"libero_success": summary["libero_success"], "v2_success": summary["v2_success"],
                    "physical_success_at_terminal": summary["physical_truth"]["physical_success_at_terminal"],
                    "physical_success_after_settle": summary["physical_truth"]["physical_success_after_settle"],
                    "diagnostic_settle_included_in_frames": False,
                    "map_updates": 5, "route_attempts": len(route_entries),
                    "direct_routes": sum(r["straight_path_valid"] is True for r in route_entries.values()),
                    "detour_routes": sum(r["straight_path_valid"] is False for r in route_entries.values()),
                    "forbidden_contact_points": 1325, "can_contact_points": 33,
                    "route_failures": [event for event in events if event["kind"] == "route_failure"]},
    }
    target = OUT / "obstacle_map_timeline.json"
    target.write_text(json.dumps(rounded(data), separators=(",", ":")) + "\n")
    print(json.dumps({"path": str(target), "bytes": target.stat().st_size,
                      "frames": len(frames), "maps": len(maps), "routes": len(route_entries),
                      "map_update_steps": [m["update_control_step"] for m in maps],
                      "events": Counter(e["kind"] for e in events)}, indent=2))


if __name__ == "__main__":
    extract()
