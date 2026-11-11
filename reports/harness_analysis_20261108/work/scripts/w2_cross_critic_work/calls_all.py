"""Extract one row per model call for every run under /workspace/gpt_field/runs.

Read-only. Sources per call: requests/*.response.json (server usage, created/completed),
requests/*.stream.json when present (client stream timing), request.json (image count, size).
Writes data/calls_all.json.
"""
import glob, json, os, re, sys, datetime as dt

RUNS = "/workspace/gpt_field/runs"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "calls_all.json")

LABEL = {
    "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001": "H1_T04",
    "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001": "H1_T08",
    "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001": "H1_stoveoff",
    "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001": "H1_drawer001",
    "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_002": "H1_drawer002",
    "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003": "H1_drawer003",
    "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001": "T20_T04",
    "agent_speedup_v2_uc71_default_routing_t04_20261103_001": "T20_T04official",
    "foundation_vision_t04_can_obstacle_20261105_001": "H3_T04",
    "foundation_vision_t08_bottle_obstacle_20261105_001": "H3_T08",
    "foundation_harness_t04_can_obstacle_001": "H2_T04",
    "astranomy_direct_can_obstacle_t04_high_20261104_001": "AST_001",
    "astranomy_direct_can_obstacle_t04_high_20261104_002": "AST_002",
    "astranomy_direct_can_obstacle_t04_collision_avoidance_high_20261105_001": "AST_CA",
}


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def count_images(req):
    n = 0
    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            if o.get("type") == "input_image":
                n += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(req)
    return n


rows = []
for run, label in LABEL.items():
    for kind, sub in (("plan", "review/requests"), ("exec", "trial/requests")):
        d = os.path.join(RUNS, run, sub)
        if not os.path.isdir(d):
            continue
        for resp_path in sorted(glob.glob(os.path.join(d, "*.response.json"))):
            base = resp_path[: -len(".response.json")]
            idx = int(os.path.basename(base).split("-")[0])
            r = load(resp_path) or {}
            u = r.get("usage") or {}
            st = load(base + ".stream.json") or {}
            req_size = os.path.getsize(base + ".request.json") if os.path.exists(base + ".request.json") else None
            req = load(base + ".request.json") if req_size and req_size < 60_000_000 else None
            row = {
                "run": label, "run_dir": run, "kind": kind, "idx": idx,
                "created_at": r.get("created_at"), "completed_at": r.get("completed_at"),
                "service_tier": r.get("service_tier"), "model": r.get("model"),
                "effort": (r.get("reasoning") or {}).get("effort"),
                "input_tokens": u.get("input_tokens"),
                "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens"),
                "output_tokens": u.get("output_tokens"),
                "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                "req_bytes": req_size,
                "n_images": count_images(req) if req is not None else None,
                "stream_started_at": st.get("started_at"),
                "elapsed_s": st.get("elapsed_s"),
                "headers_s": st.get("headers_received_s"),
                "first_visible_s": st.get("first_visible_output_s"),
                "last_event_s": st.get("last_event_s"),
                "visible_chars": st.get("visible_characters"),
                "status": r.get("status"),
            }
            if row["created_at"]:
                row["server_s"] = (row["completed_at"] or row["created_at"]) - row["created_at"]
                t = dt.datetime.fromtimestamp(row["created_at"], dt.timezone.utc)
                row["utc"] = t.isoformat()
                row["utc_date"] = t.date().isoformat()
                row["utc_hour"] = t.hour + t.minute / 60
            if row["output_tokens"] is not None and row["reasoning_tokens"] is not None:
                row["visible_tokens"] = row["output_tokens"] - row["reasoning_tokens"]
            if row["first_visible_s"] and row["last_event_s"] and row.get("visible_tokens"):
                dur = row["last_event_s"] - row["first_visible_s"]
                row["visible_rate"] = row["visible_tokens"] / dur if dur > 0 else None
            rows.append(row)

# astranomy client-side latency from model_calls.csv
import csv
for run, label in LABEL.items():
    p = os.path.join(RUNS, run, "model_calls.csv")
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for rec in csv.DictReader(f):
            i = int(rec["call"])
            for row in rows:
                if row["run"] == label and row["kind"] == "exec" and row["idx"] == i:
                    try:
                        row["elapsed_s"] = float(rec["logged_request_to_result_s"])
                    except Exception:
                        pass

with open(OUT, "w") as f:
    json.dump(rows, f, indent=0)
print(len(rows), "rows ->", OUT)
from collections import Counter
print(Counter((r["run"], r["kind"]) for r in rows))
