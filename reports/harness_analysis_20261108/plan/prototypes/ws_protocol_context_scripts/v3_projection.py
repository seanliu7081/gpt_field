"""Prototype of the v3 per-message projection on saved H1 executive requests (offline, read-only).

Measures: chars of the current observation text under H1 vs. the v3 projection
(static fields moved to the prefix, macro layer removed, duplicate feedback
removed, floats rounded to 4 decimals), and checks that every reason/detail
string of last_feedback survives in the projected outcome.
"""
import json, glob, sys, os, statistics as st

STATIC_CTX = ("capabilities", "macro", "native_evidence", "task_success_source", "region_directions")
DUP_CTX = ("recent_feedback",)

def rnd(v, nd=4):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, list):
        return [rnd(x, nd) for x in v]
    return v

def strings(v, out):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in ("reason", "detail", "status") and isinstance(x, str):
                out.add(x)
            strings(x, out)
    elif isinstance(v, list):
        for x in v:
            strings(x, out)
    return out

def drop_samples(v):
    """Remove per-step measurement arrays (kept in the audit), keep everything else."""
    if isinstance(v, dict):
        return {k: drop_samples(x) for k, x in v.items() if k not in ("measurements",)}
    if isinstance(v, list):
        return [drop_samples(x) for x in v]
    return v

def project(fields):
    ctx = fields["context"]
    out = {"step": fields["step"],
           "measured": rnd({"pose_wxyz": fields["robot_pose_wxyz"][:7], "closure": fields["robot_pose_wxyz"][7],
                            "finger_joints_m": fields["finger_joints_m"],
                            "joints": ctx["measured_state"].get("joint_position_rad"),
                            "gap_mm": ctx["measured_state"].get("finger_gap_mm"),
                            "gripper_command": ctx["measured_state"].get("last_gripper_command")}),
           "wrist_camera": rnd({k: fields["geometry"]["cameras"]["wrist"][k]
                                for k in ("position_world_m", "rotation_world_from_camera")}),
           "model_belief_echo": None,   # belief is in the transcript (assistant message)
           "outcome": rnd(drop_samples(ctx.get("last_feedback"))),
           "execution": rnd({"active": ctx["execution"].get("active"),
                             "obstacle_map": ctx["execution"].get("obstacle_map")})}
    if "reviewed_goal_contract" in fields:
        out["reviewed_goal_contract"] = fields["reviewed_goal_contract"]
    return out

rows = []
for run in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(run, "trial", "requests", "*.request.json"))):
        p = json.load(open(f))["payload"]
        if p["text"]["format"]["name"] != "user_check_agent_first":
            continue
        text = p["input"][-1]["content"][0]["text"]
        fields = json.loads(text)
        v3 = project(fields)
        v3_text = json.dumps(v3, separators=(",", ":"))
        lost = strings(fields["context"].get("last_feedback"), set()) - strings(v3["outcome"], set())
        rows.append({"run": os.path.basename(run)[:45], "h1_chars": len(text), "v3_chars": len(v3_text),
                     "compacted": "context_detail_compaction" in fields, "lost_reason_strings": sorted(lost)})
print("requests", len(rows))
h1 = [r["h1_chars"] for r in rows]; v3 = [r["v3_chars"] for r in rows]
print("H1 per-message chars mean %.0f median %.0f max %d" % (st.mean(h1), st.median(h1), max(h1)))
print("v3 per-message chars mean %.0f median %.0f max %d" % (st.mean(v3), st.median(v3), max(v3)))
print("ratio mean %.2f" % (st.mean(v3) / st.mean(h1)))
print("requests losing any reason/detail/status string:", sum(1 for r in rows if r["lost_reason_strings"]))
print("H1 requests that were compacted (>45k):", sum(r["compacted"] for r in rows))
