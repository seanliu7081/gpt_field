import pickle, numpy as np, json, sys
D = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
H1 = ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003"]
# non-model seconds per action: use agent_stats timing if available, else 0.18
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from extract import RUNLIST, RUNS
for lab in H1:
    S = pickle.load(open(f"{D}/{lab}_steps.pkl","rb"))
    st = json.load(open(f"{RUNS}/{RUNLIST[lab]}/trial/agent_stats.json"))
    # find wall/time keys
    tm = st.get("timing") or {}
    tr_speed = np.mean([np.linalg.norm(S[s]["measured_delta"]) for s in S if (S[s].get("frag") or {}).get("profile") == "transit" and (S[s].get("frag") or {}).get("skill") in ("move_to_pose","move_relative")])
    out = []
    for stand in (.02, .04, .06):
        far = [s for s in S if (S[s].get("frag") or {}).get("profile") == "contact" and (S[s].get("frag") or {}).get("skill") in ("move_to_pose","move_relative","approach") and ((S[s].get("frag") or {}).get("pos_err") or 0) > stand]
        dist = sum(np.linalg.norm(S[s]["measured_delta"]) for s in far)
        need = dist / tr_speed
        out.append((stand, len(far), round(len(far) - need)))
    print(lab, "transit mm/step %.2f" % (1000*tr_speed), "| (standoff, contact steps beyond, steps saved):", out, "| timing keys:", list(tm.keys())[:8])
