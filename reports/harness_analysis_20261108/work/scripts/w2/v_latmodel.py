import json, numpy as np
rows = json.load(open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/latency_rows.json"))
train = [x for r, v in rows.items() if "foundation_vision" not in r for x in v]
print("n train", len(train))
X = np.array([[1, x["rt"], x["it"] / 1000] for x in train]); y = np.array([x["ttft"] for x in train])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef; r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("TTFT fit: %.2f + %.4f*rt + %.4f*in_k  R2=%.3f" % (*coef, r2))
# also a model with output tokens on total elapsed
X2 = np.array([[1, x["rt"], x["vis"], x["it"] / 1000] for x in train]); y2 = np.array([x["el"] for x in train])
c2, *_ = np.linalg.lstsq(X2, y2, rcond=None); p2 = X2 @ c2; r22 = 1 - ((y2 - p2) ** 2).sum() / ((y2 - y2.mean()) ** 2).sum()
print("elapsed fit: %.2f + %.4f*rt + %.4f*vis + %.4f*in_k R2=%.3f" % (*c2, r22))
dec_train = sum(x["vis"] for x in train) / sum(x["el"] - x["ttft"] for x in train)
print("pooled 11-02/03 visible decode tok/s", round(dec_train, 1))
for r, v in rows.items():
    if "foundation_vision" not in r and "transit100" not in r: continue
    actual = sum(x["el"] for x in v)
    p_ttft = sum(coef @ [1, x["rt"], x["it"] / 1000] for x in v)
    p_dec = sum(x["vis"] / dec_train for x in v)
    p_hdr = 0
    p_el2 = sum(c2 @ [1, x["rt"], x["vis"], x["it"] / 1000] for x in v)
    # counterfactual: same calls but with H1-like reasoning count per call
    print(f"{r[:66]:66s} actual={actual:6.1f} predicted@1103(ttft+decode)={p_ttft + p_dec:6.1f} residual={actual - p_ttft - p_dec:+6.1f} | elapsed-fit pred={p_el2:6.1f} resid={actual - p_el2:+6.1f}")
# reasoning-attributable time for H3 T04 relative to H1 T04 at equal throughput (simple):
h1 = rows["agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001"]; h3 = rows["foundation_vision_t04_can_obstacle_20261105_001"]
print("H1 T04 total rt", sum(x["rt"] for x in h1), "H3 T04 total rt", sum(x["rt"] for x in h3), "diff", sum(x["rt"] for x in h3) - sum(x["rt"] for x in h1))
print("rt coef * diff =", round(coef[1] * (sum(x['rt'] for x in h3) - sum(x['rt'] for x in h1)), 1), "s; intercept*(n diff) =", round(coef[0] * (len(h3) - len(h1)), 1))
# per-call reasoning distribution
for r, v in rows.items():
    rts = sorted(x["rt"] for x in v)
    print(f"{r[:60]:60s} rt per call: median={np.median(rts):.0f} p25={np.percentile(rts,25):.0f} p75={np.percentile(rts,75):.0f} max={max(rts)}")
