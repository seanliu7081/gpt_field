"""Plan A: per-call extraction over the 13 uc7 runs (read-only, stdlib, run with python3 -I).

Sources per run R = /workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_<key>:
  R/requests/NNNNNN-*.request.json   (observation text by context key, trigger = context.last_feedback)
  R/requests/NNNNNN-*.response.json  (usage tokens, output text)
  R_launch/runner.log                ("model_call" lines: client latency_s, step)
Output: plan/A_calls.json (one row per executive call).
"""
import glob, json, os, re, sys

ROOT = "/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
OUT = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/A_calls.json"
RUNS = "t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()


def sz(x):
    return len(json.dumps(x, separators=(",", ":"), ensure_ascii=False))


def runner_calls(key):
    rows = []
    with open(ROOT + key + "_launch/runner.log", errors="replace") as f:
        for line in f:
            if '"event":"model_call"' not in line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


rows = []
for key in RUNS:
    rc = runner_calls(key)
    reqs = sorted(glob.glob(ROOT + key + "/requests/*.request.json"))
    for i, path in enumerate(reqs):
        k = int(os.path.basename(path).split("-")[0])
        req = json.load(open(path))
        resp_path = path.replace(".request.json", ".response.json")
        resp = json.load(open(resp_path)) if os.path.exists(resp_path) else {}
        last = req["payload"]["input"][-1]
        text = last["content"][0]["text"]
        j = text.find("{")
        fields = json.loads(text[j:])
        ctx = fields.get("context") or {}
        lf = ctx.get("last_feedback") or {}
        trig = None
        if lf:
            src = lf.get("summary") if isinstance(lf.get("summary"), dict) else lf
            trig = "%s/%s/%s" % (src.get("action"), src.get("status"), src.get("reason"))
        u = resp.get("usage") or {}
        out_tok = u.get("output_tokens")
        reas = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
        otext = ""
        for o in resp.get("output") or []:
            if o.get("type") == "message":
                for c in o.get("content") or []:
                    if c.get("type") == "output_text":
                        otext += c.get("text") or ""
        try:
            dec = json.loads(otext) if otext else {}
        except ValueError:
            dec = {}
        prose = sum(len(dec.get(x) or "") for x in ("belief", "plan", "why"))
        segs = ((dec.get("execution") or {}).get("segments") or [])
        lat = None
        if k - 1 < len(rc) and rc[k - 1].get("k") == k:
            lat = rc[k - 1].get("latency_s")
        row = dict(run=key, k=k, step=fields.get("step"), latency_s=lat,
                   server_s=(resp.get("completed_at") - resp.get("created_at")) if resp.get("completed_at") else None,
                   trigger=trig, type=dec.get("type"), n_segments=len(segs),
                   seg_skills=[(s.get("skill"), s.get("profile"), s.get("max_steps"), s.get("observation_after")) for s in segs],
                   input_tokens=u.get("input_tokens"),
                   cached_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens"),
                   output_tokens=out_tok, reasoning_tokens=reas,
                   visible_tokens=(out_tok - reas) if (out_tok is not None and reas is not None) else None,
                   out_chars=len(otext), prose_chars=prose,
                   exec_chars=sz(dec.get("execution")) if dec.get("execution") is not None else 0,
                   obs_text_chars=len(text),
                   obs_top={k2: sz(v) for k2, v in fields.items() if k2 != "context"},
                   obs_ctx={k2: sz(v) for k2, v in ctx.items()},
                   n_input_items=len(req["payload"]["input"]))
        rows.append(row)
json.dump(rows, open(OUT, "w"), indent=0)
print("calls", len(rows), "with latency", sum(r["latency_s"] is not None for r in rows))
