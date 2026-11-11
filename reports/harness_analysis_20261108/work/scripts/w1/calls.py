"""Per-call table for uc7.1 runs: decision type, trigger (previous feedback), latency, tokens.
Read-only over saved artifacts."""
import json, glob, os, sys, math
from collections import Counter, defaultdict

def load(p):
    with open(p) as f:
        return json.load(f)

def request_fields(payload):
    last = payload["input"][-1]
    for part in last.get("content", []):
        if part.get("type") == "input_text":
            t = part.get("text", "")
            i = t.find("{")
            if i >= 0:
                try:
                    return json.loads(t[i:])
                except Exception:
                    return None
    return None

def response_answer(resp):
    for o in resp.get("output", []):
        if o.get("type") == "message":
            for c in o.get("content", []):
                if c.get("type") == "output_text":
                    try:
                        return json.loads(c["text"])
                    except Exception:
                        return None
    return None

def latencies(run):
    lat = {}
    ev = os.path.join(run, "trial", "episode.events.jsonl")
    if not os.path.exists(ev):
        return lat
    for line in open(ev):
        d = json.loads(line)
        if d.get("event") == "skill_decision" and d.get("api_request_log"):
            lat[os.path.basename(d["api_request_log"]).split(".")[0]] = d.get("api_latency_s")
    return lat

def analyse(run):
    req_dir = os.path.join(run, "trial", "requests")
    lat = latencies(run)
    rows = []
    for rp in sorted(glob.glob(os.path.join(req_dir, "*.request.json"))):
        stem = rp[:-len(".request.json")]
        r = load(rp)
        if r.get("mode") != "agent":
            continue
        fields = request_fields(r["payload"]) or {}
        ctx = fields.get("context", {})
        fb = ctx.get("last_feedback") or {}
        if "summary" in fb and isinstance(fb["summary"], dict):
            fb = fb["summary"]
        trigger = fb.get("reason") or fb.get("status") or ("initial" if not fb else "?")
        resp = load(stem + ".response.json") if os.path.exists(stem + ".response.json") else {}
        res = load(stem + ".result.json") if os.path.exists(stem + ".result.json") else {}
        ans = response_answer(resp) or {}
        usage = resp.get("usage") or {}
        segs = ((ans.get("execution") or {}).get("segments") or [])
        joint = fields.get("context", {}).get("measured_state", {}).get("joint_position_rad")
        rows.append({
            "file": os.path.basename(stem).split("-")[0], "step": fields.get("step"),
            "trigger": trigger, "trigger_action": fb.get("action"), "type": ans.get("type"),
            "n_segments": len(segs),
            "segments": [(s["skill"], s["profile"], s["max_steps"], s["observation_after"]) for s in segs],
            "obstacle_map_update": ans.get("obstacle_map") is not None,
            "latency_s": lat.get(os.path.basename(stem).split(".")[0]),
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            "output_tokens": usage.get("output_tokens"), "input_tokens": usage.get("input_tokens"),
            "status": res.get("status"),
            "pose_wxyz": fields.get("robot_pose_wxyz"),
            "belief": (ans.get("belief") or "")[:300],
            "why": (ans.get("why") or "")[:200],
        })
    return rows

if __name__ == "__main__":
    out = {}
    for run in sys.argv[1:]:
        rows = analyse(run)
        name = os.path.basename(run.rstrip("/"))
        out[name] = rows
        trig = Counter(r["trigger"] for r in rows)
        types = Counter(r["type"] for r in rows)
        lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
        print(f"== {name}: calls={len(rows)} types={dict(types)}")
        print("   triggers:", dict(trig))
        if lat:
            print(f"   latency sum={sum(lat):.1f} mean={sum(lat)/len(lat):.1f} first={rows[0]['latency_s']}")
        # quaternion sign flips between consecutive calls
        flips = 0
        prev = None
        for r in rows:
            q = (r["pose_wxyz"] or [None]*7)[3:7]
            if prev is not None and None not in q and None not in prev:
                dot = sum(a*b for a, b in zip(q, prev))
                if dot < 0:
                    flips += 1
            prev = q
        print("   consecutive robot_pose quaternion sign flips:", flips)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "calls_table.json"), "w"), indent=1)
