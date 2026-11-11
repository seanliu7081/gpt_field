"""Find zero-progress handoffs (target_envelope_rejected, route_unavailable) and their details."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from calls import load, request_fields

def walk(o, path=""):
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from walk(v, path + "." + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + f"[{i}]")

for run in sys.argv[1:]:
    req = sorted(glob.glob(os.path.join(run, "trial", "requests", "*.request.json")))
    for rp in req:
        r = load(rp)
        if r.get("mode") != "agent":
            continue
        f = request_fields(r["payload"]) or {}
        fb = f.get("context", {}).get("last_feedback") or {}
        top = fb.get("summary", fb) if isinstance(fb, dict) else {}
        reason = top.get("reason")
        if reason in ("target_envelope_rejected", "route_unavailable"):
            details = []
            for path, node in walk(fb):
                if node.get("reason") in ("route_goal_blocked", "route_start_blocked", "route_exceeds_local_map",
                                          "route_not_found", "route_length_limit", "route_goal_connector_blocked",
                                          "route_planner_no_progress", "no_valid_reference") or \
                   (node.get("status") == "unavailable"):
                    details.append({k: node.get(k) for k in ("reason", "status", "straight_path_valid") if k in node})
                if node.get("detail") and "envelope" in str(node.get("detail")):
                    details.append({"detail": node.get("detail"), "pos_err": node.get("position_error_m"),
                                    "rot_err": node.get("orientation_error_rad")})
            print(os.path.basename(run)[:70], os.path.basename(rp)[:6], "step", f.get("step"), reason, details[:3])
