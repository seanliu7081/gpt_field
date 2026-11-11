"""Join the three W1 stall taxonomies on (run, step): h1_speed (error-trace shape), h1_collision (contact data),
h1_zero_shot (belief keywords). Prints a cross-tab."""
import json, ast, re
from collections import Counter, defaultdict
W1 = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1"
W2 = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/cross_critic_work"
# h1_collision
coll = {}
for line in open(f"{W2}/data/stall_causes_out.txt"):
    line = line.strip()
    if line.startswith("("):
        t = ast.literal_eval(line)
        coll[(t[0], t[1])] = t[5]
# h1_speed stall_class (5 runs)
sc = json.load(open(f"{W1}/data/stall_class.json"))
RUNKEY = {"t04_t100": "can_obstacle_transit100_t04_20261", "t08_t100": "bottle_obstacle_transit100_t08_2", "stoveoff": "stove_off_bottle_obstacle_transi",
          "drawer001": "drawer_bowl_moka_transit500_cap10", "drawer003": "drawer_bowl_moka_transit500_cap10"}
speed = {}
for k, v in sc.items():
    for d in v["details"]:
        speed[(k, d["step"])] = d["kind"]
# h1_zero_shot keyword classes from calls_table beliefs (same regex families the report describes)
ct = json.load(open(f"{W1}/h1_zero_shot_work/calls_table.json"))
def kw(b):
    b = (b or "").lower()
    if re.search(r"limit|elbow|extension|singular|kinematic|reach", b): return "kinematic"
    if re.search(r"end stop|travel limit|closed stop|fully (open|closed)", b): return "end_stop"
    if re.search(r"contact|touch|press|blocked by|pushing|collid", b): return "contact"
    if re.search(r"height|depth|too (low|high)|lower than|higher than|short of|above (its|the) requested|stopped about|mm (short|from)", b): return "height_depth"
    return "other"
zs = {}
for run, rows in ct.items():
    for r in rows:
        if r.get("trigger") == "segment_stall":
            zs[(run, r["step"])] = kw(r.get("belief"))
# join on run+step; h1_collision keys use run[26:60]
xt = Counter(); rows = []
for (run, step), zc in zs.items():
    ck = coll.get((run[26:60], step))
    sk = None
    for k, rk in RUNKEY.items():
        if rk in run and ("_001" in run if k == "drawer001" else "_003" in run if k == "drawer003" else True):
            sk = speed.get((k, step))
    rows.append((run[26:60], step, sk, ck, zc)); xt[(sk, ck)] += 1
for r in sorted(rows): print(r)
print("\ncross-tab (h1_speed kind, h1_collision cause):")
for k, v in sorted(xt.items(), key=lambda kv: -kv[1]): print("  ", k, v)
print("h1_zero_shot keyword classes:", Counter(r[4] for r in rows))
print("by h1_collision cause:", Counter(r[3] for r in rows))
print("by h1_speed kind:", Counter(r[2] for r in rows))
