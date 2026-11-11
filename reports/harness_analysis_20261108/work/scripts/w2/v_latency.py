import json, glob, os, statistics as st
R = "/workspace/gpt_field/runs/"
runs = ["agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
        "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
        "foundation_vision_t04_can_obstacle_20261105_001",
        "foundation_vision_t08_bottle_obstacle_20261105_001",
        "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001",
        "agent_speedup_v2_uc71_default_routing_t04_20261103_001",
        "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
        "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_002",
        "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003",
        "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001"]
rows_all = {}
for r in runs:
    reqs = sorted(glob.glob(R + r + "/trial/requests/*.request.json"))
    if not reqs: print(r, "no trial requests"); continue
    prev = None; rows = []
    for rq in reqs:
        res = json.load(open(rq.replace(".request.json", ".result.json")))
        stm = json.load(open(rq.replace(".request.json", ".stream.json")))
        resp = json.load(open(rq.replace(".request.json", ".response.json")))
        u = resp.get("usage") or {}
        b = res["budget"]
        it = u.get("input_tokens"); ot = u.get("output_tokens"); rt = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
        ct = (u.get("input_tokens_details") or {}).get("cached_tokens")
        vis = ot - rt if ot is not None and rt is not None else None
        fv = stm.get("first_visible_output_s"); le = stm.get("last_event_s"); hd = stm.get("headers_received_s")
        dec = vis / (le - fv) if vis and fv is not None and le and le > fv else None
        rows.append(dict(it=it, ot=ot, rt=rt, vis=vis, ct=ct, el=stm.get("elapsed_s"), ttft=fv, hdr=hd, dec=dec,
                         started=stm.get("started_at"), status=res.get("status")))
    rows_all[r] = rows
    n = len(rows)
    def m(k): 
        v = [x[k] for x in rows if x[k] is not None]; return (st.mean(v) if v else None)
    def md(k):
        v = [x[k] for x in rows if x[k] is not None]; return (st.median(v) if v else None)
    sum_el = sum(x["el"] for x in rows)
    sum_vis = sum(x["vis"] for x in rows); sum_dec_t = sum(x["el"] - x["ttft"] for x in rows)
    print(f"{r[:70]:70s} n={n:2d} day={rows[0]['started'][:10]} mean_el={m('el'):.1f} med_el={md('el'):.1f} sum_el={sum_el:.1f} "
          f"mean_ttft={m('ttft'):.1f} rt/call={m('rt'):.0f} vis/call={m('vis'):.0f} in/call={m('it')/1000:.1f}k "
          f"dec_mean={m('dec'):.1f} dec_pooled={sum_vis/sum_dec_t:.1f} cached={sum(x['ct'] or 0 for x in rows)} in_sum={sum(x['it'] for x in rows)}")
json.dump(rows_all, open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/latency_rows.json", "w"), indent=0)
