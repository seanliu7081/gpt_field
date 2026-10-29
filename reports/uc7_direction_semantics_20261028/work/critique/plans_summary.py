import json, glob, os, re, sys
RUNS = "/workspace/gpt_field/runs"
pats = ["agent_speedup_v2_uc7_*/task_*/trial/approved_plan.json",
        "agent_speedup_v2_user_check_libero_10_t*/approved_plan.json",
        "robocasa_coffee_user_check_001/**/approved_plan.json"]
files = []
for p in pats:
    files += glob.glob(os.path.join(RUNS, p), recursive=True)
DIRW = re.compile(r"\b(left|right|front|back|behind|middle|top|bottom|\+x|-x|\+y|-y|world)\b", re.I)
for f in sorted(set(files)):
    d = json.load(open(f))
    run = os.path.relpath(f, RUNS)
    parts = {p["id"]: p for p in d.get("parts", [])}
    print("==", run)
    print("   scene:", d.get("scene"))
    for r in d.get("regions", []):
        pid = r["id"]
        print("   REGION", r, "| part text:", parts.get(pid, {}).get("text"), "| w/x:", parts.get(pid, {}).get("w"), parts.get(pid, {}).get("x"))
        if r.get("anchor"):
            a = parts.get(r["anchor"], {})
            print("      anchor:", a.get("text"), "w/x", a.get("w"), a.get("x"))
    for g in d.get("placement_goals", []):
        print("   GOAL", g)
    for p in d["parts"]:
        if DIRW.search(p["text"]):
            print("   dir-word part", p["id"], p["text"], p["w"], p["x"])
    for i, s in enumerate(d["steps"]):
        print(f"   step{i}", s["do"], s["obj"], "->", s["ref"], "grip", s["grip"])
