"""Diagnose free-space shortfall outliers; evaluate per-step relative guard families."""
import pickle, math, os
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}
A, B = 0.7114, 0.0732
def load(name, scale):
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb"))
    rows = []
    for k in sorted(d):
        x = d[k]; b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"])
        act = np.asarray(x["norm_action"]); frag = x.get("frag") or {}
        rows.append(dict(c=act[:3] * scale, cr=act[3:6] * 0.5, s=a[:3] - b[:3], forb=x.get("forb_inc", 0) or 0,
                         allowed=sum((x.get("inc_allowed") or {}).values()),
                         seg=(frag.get("seg_index"), frag.get("skill"), x.get("primitive")), grip=act[6],
                         prim=x.get("primitive"), skill=frag.get("skill"), gap=x.get("finger_gap_mm")))
    return rows
data = {n: load(n, s) for n, s in FILES.items() if os.path.exists(BASE + n + "_steps.pkl")}

def nofc(rows, i, W):  # no contact in window and 2 steps before
    return all(not rows[j]["forb"] and not rows[j]["allowed"] for j in range(max(0, i - W - 1), i + 1))

W, cmin = 3, 0.001
out = {"rot": 0, "grip": 0, "rev": 0, "other": 0}; n_out = 0
for name, rows in data.items():
    for i in range(W + 2, len(rows)):
        if not nofc(rows, i, W): continue
        P = R = 0.0; ok = True; rot = grip = rev = False
        for j in range(i - W + 1, i + 1):
            c = rows[j]["c"]; n = np.linalg.norm(c)
            if n < cmin or rows[j]["seg"] != rows[i]["seg"]: ok = False; break
            u = c / n
            P += A * float(np.dot(rows[j - 1]["s"], u)) + B * n
            R += float(np.dot(rows[j]["s"], u))
            rot |= np.linalg.norm(rows[j]["cr"]) > 0.02
            grip |= rows[j]["grip"] != rows[j - 1]["grip"] or rows[j]["skill"] == "set_gripper"
            rev |= float(np.dot(rows[j - 1]["c"], c)) < 0
        if not ok or not (0.002 <= P < 0.006): continue
        if P - R > 0.002:
            n_out += 1
            if rot: out["rot"] += 1
            elif grip: out["grip"] += 1
            elif rev: out["rev"] += 1
            else: out["other"] += 1
print("free [2,6)mm windows with shortfall > 2 mm:", n_out, out)

# Per-step relative test (verifier family, model-normalised): fire if for W consecutive steps
# real_j < f * pred_j with |c_j| >= cmin, rotation command small or not.
def evaluate(f, W, cmin, require_low_rot):
    eps = det = 0; delays = []; ev_free = 0
    for name, rows in data.items():
        flag = [False] * len(rows)
        for i in range(W, len(rows)):
            good = True
            for j in range(i - W + 1, i + 1):
                c = rows[j]["c"]; n = np.linalg.norm(c)
                if n < cmin or rows[j]["seg"] != rows[i]["seg"]: good = False; break
                if require_low_rot and np.linalg.norm(rows[j]["cr"]) > 0.02: good = False; break
                u = c / n
                pred = A * float(np.dot(rows[j - 1]["s"], u)) + B * n
                real = float(np.dot(rows[j]["s"], u))
                if not (real < f * max(pred, B * n)): good = False; break
            flag[i] = good
        i = 0
        while i < len(rows):
            if rows[i]["forb"] > 0:
                j = i
                while j + 1 < len(rows) and rows[j + 1]["forb"] > 0: j += 1
                eps += 1
                fz = next((k for k in range(i, j + 1) if flag[k]), None)
                if fz is not None: det += 1; delays.append(fz - i)
                i = j + 1
            else: i += 1
        prev = False
        for i in range(len(rows)):
            fr = flag[i] and nofc(rows, i, W)
            if fr and not prev: ev_free += 1
            prev = fr
    return eps, det, (float(np.median(delays)) if delays else None), ev_free

for f in (0.2, 0.3, 0.4, 0.5):
    for W in (2, 3, 4):
        for lowrot in (False, True):
            e, d, dl, fr = evaluate(f, W, 0.001, lowrot)
            print(f"f={f} W={W} lowrot={lowrot}: episodes {e} detected {d} median delay {dl} free fire events {fr}")
