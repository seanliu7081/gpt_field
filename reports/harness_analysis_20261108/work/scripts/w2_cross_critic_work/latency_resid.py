"""Predict H3 calls from a model fitted on 11-02/03 uc7.1 calls only; decode-rate vs context; images per request."""
import json, os, statistics as stt
import numpy as np
D = os.path.join(os.path.dirname(__file__), "..", "data")
rows = json.load(open(os.path.join(D, "calls_all.json")))
ik = lambda r: (r["input_tokens"] or 0)/1000
uc = [r for r in rows if r["kind"]=="exec" and r["run"].startswith(("H1_","T20_"))]
h3 = [r for r in rows if r["kind"]=="exec" and r["run"].startswith("H3_")]
X = np.array([[1, r["reasoning_tokens"], r["visible_tokens"], ik(r)] for r in uc]); y = np.array([r["elapsed_s"] for r in uc])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
print("fit(11-02/03 only):", np.round(b, 5))
for run in ("H3_T04", "H3_T08", "H1_T04", "H1_T08"):
    rs = [r for r in rows if r["kind"]=="exec" and r["run"]==run]
    pred = sum(b @ np.array([1, r["reasoning_tokens"], r["visible_tokens"], ik(r)]) for r in rs)
    obs = sum(r["elapsed_s"] for r in rs)
    print(f"{run}: n={len(rs)} observed {obs:.1f}s predicted {pred:.1f}s residual {obs-pred:+.1f}s ({(obs-pred)/len(rs):+.2f}s/call)")
# decode rate vs context within 11-02/03
pts = [(ik(r), r["visible_rate"]) for r in uc if r.get("visible_rate") and r["visible_tokens"]>=100]
A = np.array([[1, p[0]] for p in pts]); yy = np.array([p[1] for p in pts])
bb, *_ = np.linalg.lstsq(A, yy, rcond=None)
print("decode rate ~ input_k (11-02/03):", np.round(bb, 4), "n", len(pts))
for run in ("H3_T04", "H3_T08"):
    rs = [r for r in rows if r["kind"]=="exec" and r["run"]==run and r.get("visible_rate") and r["visible_tokens"]>=100]
    pr = [bb[0]+bb[1]*ik(r) for r in rs]; ob = [r["visible_rate"] for r in rs]
    print(run, "observed decode median", round(stt.median(ob),1), "predicted at its context", round(stt.median(pr),1))
# reasoning tokens distribution
for run in ("H1_T04","H1_T08","H1_stoveoff","H1_drawer001","H1_drawer003","T20_T04","T20_T04official","H3_T04","H3_T08"):
    rs = [r["reasoning_tokens"] for r in rows if r["kind"]=="exec" and r["run"]==run]
    later = [r["reasoning_tokens"] for r in rows if r["kind"]=="exec" and r["run"]==run and r["idx"]>1]
    zero = sum(1 for x in rs if x==0)
    print(f"{run:16s} reasoning median {stt.median(rs):6.1f} mean {sum(rs)/len(rs):6.1f} first {rs[0]:4d} later-median {stt.median(later):6.1f} zero-reasoning calls {zero}/{len(rs)}")
# images and input growth per request
for run in ("H1_T04","H3_T04","AST_002","AST_CA","H2_T04"):
    rs = [r for r in rows if r["kind"]=="exec" and r["run"]==run]
    print(run, "images/request first..last", [r["n_images"] for r in rs][:3], "...", [r["n_images"] for r in rs][-3:],
          "input_k first..last", round(ik(rs[0]),1), round(ik(rs[-1]),1), "req MB last", round(rs[-1]["req_bytes"]/1e6,2))
