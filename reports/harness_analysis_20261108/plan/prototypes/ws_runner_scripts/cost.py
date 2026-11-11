"""Sweep cost model (estimates). Inputs are H1-family measurements from the 2026-11-08 analysis."""
# Finished episodes (H1-family measured): T04 14 calls/305 s, T08 22/490, stove-off 17/459, drawer 43/1301, 37/1259,
# official T04 (20 mm build) 11/358; uc7 LIBERO-10 batch ~17 calls, mean 337.5 s among successes.
# After A2 the per-step overhead falls from ~0.158 s to <=0.06 s; A1 adds ~1-2 calls (+20-40 s) per success.
scenarios = {
    "optimistic": dict(p_finish_success=0.75, give_up_share_of_fail=0.7,
                       succ_calls=18, succ_wall_min=6.5, gu_calls=25, gu_wall_min=9.0,
                       cap_calls=80, cap_wall_min=33.0),
    "pessimistic": dict(p_finish_success=0.50, give_up_share_of_fail=0.3,
                        succ_calls=20, succ_wall_min=7.5, gu_calls=30, gu_wall_min=12.0,
                        cap_calls=80, cap_wall_min=33.0),
}
def per_arm(n, s, no_give_up=False):
    fail = n * (1 - s["p_finish_success"])
    gu = 0 if no_give_up else fail * s["give_up_share_of_fail"]
    cap = fail - gu
    succ = n - fail
    calls = succ * s["succ_calls"] + gu * s["gu_calls"] + cap * s["cap_calls"]
    wall_h = (succ * s["succ_wall_min"] + gu * s["gu_wall_min"] + cap * s["cap_wall_min"]) / 60
    return calls, wall_h
for name, s in scenarios.items():
    print(name)
    for n, label in ((34, "34 scenes (L10+Goal+4 variants+10 relayouts)"), (44, "44 scenes (+10 seeded obstacle relayouts)")):
        c, h = per_arm(n, s)
        cp, hp = per_arm(n - 4, s, no_give_up=True)   # pristine H1: no stove-off/composite, no give_up
        print(f"  {label}: new-build arm {c:.0f} calls {h:.1f} h ; pristine H1 arm ({n-4} scenes) {cp:.0f} calls {hp:.1f} h ;"
              f" both {c+cp:.0f} calls {h+hp:.1f} h serial / {max(h,hp):.1f} h paired-concurrent")
