import pickle, json, collections, numpy as np, sys
D = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
RUNS = "/workspace/gpt_field/runs"
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from extract import RUNLIST
H1 = ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003"]
SCALE = {"T04_20mm": .05, "T04_default": .05}
for lab in H1:
    S = pickle.load(open(f"{D}/{lab}_steps.pkl","rb")); sc = SCALE.get(lab, .10)
    n = len(S); c = collections.Counter(); mv = collections.defaultdict(list); ratio = []
    far = 0
    for s, st in S.items():
        f = st.get("frag"); md = np.linalg.norm(st.get("measured_delta") or [0,0,0])
        if st.get("fb_primitive") == "execute_fragment" and f:
            prof = f.get("profile"); sk = f.get("skill")
            c[prof] += 1
            k = f"{prof}:{'move' if sk in ('move_to_pose','move_relative','approach') else sk}"
            c[k] += 1; mv[k].append(md)
            if prof == "contact" and sk in ("move_to_pose","move_relative","approach") and (f.get("pos_err") or 0) > .02:
                far += 1
            if prof == "contact" and sk in ("move_to_pose","move_relative","approach"):
                cmd = np.linalg.norm(np.asarray(st["norm_action"][:3]) * sc)
                if cmd > .004: ratio.append(md / cmd)
        else:
            c["direct/other"] += 1; mv["direct"].append(md)
    stats = json.load(open(f"{RUNS}/{RUNLIST[lab]}/trial/agent_stats.json"))
    print(f"{lab:13s} actions={n} contact={c['contact']} ({100*c['contact']/n:.0f}%) [move {c['contact:move']}, set_gripper {c['contact:set_gripper']}, wait {c['contact:wait']}] transit={c['transit']} ({100*c['transit']/n:.0f}%) direct={c['direct/other']}"
          f" | mm/action contact-move {1000*np.mean(mv['contact:move']):.2f} all-contact {1000*np.mean(mv['contact:move']+mv['contact:set_gripper']+mv['contact:wait']):.2f} transit-move {1000*np.mean(mv['transit:move']):.2f}"
          f" | contact moves >20mm from target {far} | realized/commanded contact median {np.median(ratio):.2f}")
