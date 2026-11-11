"""Independent per-call table from raw request logs: stream.json (timing), response.json (usage), request.json (images, size)."""
import json, glob, os, sys
RUNS = {
 "T04": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
 "T08": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
 "stove": "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
 "d001": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
 "d003": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003",
}
BASE = "/workspace/gpt_field/runs"
def count_images(obj):
    n = 0
    if isinstance(obj, dict):
        if obj.get("type") == "input_image": n += 1
        for v in obj.values(): n += count_images(v)
    elif isinstance(obj, list):
        for v in obj: n += count_images(v)
    return n
rows = []
for name, run in RUNS.items():
    for sub in ("review", "trial"):
        for st in sorted(glob.glob(f"{BASE}/{run}/{sub}/requests/*.stream.json")):
            stem = st[:-len(".stream.json")]
            s = json.load(open(st))
            try: resp = json.load(open(stem + ".response.json"))
            except FileNotFoundError: resp = {}
            try: req = json.load(open(stem + ".request.json"))
            except FileNotFoundError: req = {}
            try: res = json.load(open(stem + ".result.json"))
            except FileNotFoundError: res = {}
            u = resp.get("usage") or {}
            out = u.get("output_tokens"); rs = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
            rows.append({"run": name, "phase": sub, "file": os.path.basename(stem),
                "kind": res.get("request_kind"), "status": res.get("status"), "stream_phase": s.get("phase"),
                "latency_s": s.get("elapsed_s"), "headers_s": s.get("headers_received_s"),
                "first_visible_s": s.get("first_visible_output_s"), "last_visible_s": s.get("last_visible_output_s"),
                "completion_s": s.get("completion_s"),
                "input": u.get("input_tokens"), "cached": (u.get("input_tokens_details") or {}).get("cached_tokens"),
                "output": out, "reasoning": rs, "visible": None if out is None or rs is None else out - rs,
                "visible_chars": s.get("visible_characters"),
                "images": count_images(req.get("body", req)), "request_bytes": os.path.getsize(stem + ".request.json") if os.path.exists(stem + ".request.json") else None,
                "prompt_cache_key_sent": ("prompt_cache_key" in json.dumps(req)[:0]) or ('"prompt_cache_key"' in open(stem + ".request.json").read() if os.path.exists(stem + ".request.json") else None),
                "resp_cache_key": resp.get("prompt_cache_key")})
json.dump(rows, open(sys.argv[1], "w"), indent=1)
print(len(rows), "calls")
