"""Finger gap at the end of every close segment, gap change during the following lift, and the model's next verdict.
Tests whether a measured continuation predicate (gap band + no gap collapse during lift) separates the
grasp failures the visual checkpoints caught."""
import json, glob, os, re
RUNS = {
  "H1_T04": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
  "H1_T08": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
  "H1_stoveoff": "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
  "H1_drawer001": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
  "H1_drawer003": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003",
  "H3_T04": "foundation_vision_t04_can_obstacle_20261105_001",
  "H3_T08": "foundation_vision_t08_bottle_obstacle_20261105_001",
  "T20_T04": "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001",
}
def beliefs(run):
    out = {}
    for p in sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.response.json")):
        r = json.load(open(p))
        for o in r.get("output", []):
            if o.get("type") == "message":
                for c in o["content"]:
                    if c.get("type") == "output_text":
                        try: out[int(os.path.basename(p)[:6])] = json.loads(c["text"]).get("belief", "")
                        except Exception: pass
    return out
FAIL = re.compile(r"did not (lift|follow|rise)|missed|slipped|closed (almost completely|to about 1)|not attached|no attachment|failed grasp|grasp missed", re.I)
OK = re.compile(r"rose with|followed the test lift|remain(s|ed) attached|captured|supports? a successful|securely|stable .*grip|good visual evidence of attachment", re.I)
allrows = []
for name, run in RUNS.items():
    lines = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    acts = [l for l in lines if l.get("event") == "action"]
    call_at_step = {}
    k = 0
    for a in acts:
        if a["diagnostics"].get("api_calls_in_act"):
            k += 1; call_at_step[a["step"]] = k
    fb = [l for l in lines if l.get("event") == "feedback"]
    gaps = {l["step"]: l["feedback"].get("finger_gap_mm") for l in fb}
    cmd = {l["step"]: l["feedback"].get("gripper_command") for l in fb}
    B = beliefs(run)
    steps = sorted(gaps)
    # close events: gripper command transitions from open(-1) to close(+1)
    for i in range(1, len(steps)):
        s0, s1 = steps[i-1], steps[i]
        if (cmd.get(s0) or 0) < 0 and (cmd.get(s1) or 0) > 0:
            # gap when closing settles: first step where gap changes < 0.2 mm over 3 steps after s1
            j = i
            while j + 3 < len(steps) and abs(gaps[steps[j+3]] - gaps[steps[j]]) > 0.2: j += 1
            settled = gaps[steps[min(j+3, len(steps)-1)]]
            # gap 40 steps later (lift window) if still closed
            later = [gaps[s] for s in steps[j:j+60] if (cmd.get(s) or 0) > 0]
            drop = (settled - min(later)) if later else None
            nxt = min([c for st, c in call_at_step.items() if st > s1] or [None]) if call_at_step else None
            verdict_txt = B.get(nxt, "") if nxt else ""
            verdict = "fail" if FAIL.search(verdict_txt) else ("ok" if OK.search(verdict_txt) else "?")
            row = {"run": name, "close_step": s1, "settled_gap_mm": round(settled, 1), "gap_drop_next60_mm": round(drop, 1) if drop is not None else None,
                   "next_call": nxt, "verdict": verdict, "belief": verdict_txt[:120]}
            allrows.append(row); print(row)
json.dump(allrows, open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/cross_critic_work/data/grasp_gap.json", "w"), indent=1)
