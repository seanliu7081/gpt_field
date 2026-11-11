import sys, json
sys.path.insert(0, "/workspace/gpt_field/src")
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
src = open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/v_geom_repro.py").read().split("for se, sw in")[0]
exec(src)
D = depths(S0, S0)
cast_src = open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/v_can_raycast.py").read().split("sil = {}")[0]
exec(cast_src)
for n, (Kc, Ec, _) in cams.items():
    hit, Pt, o = cast(Kc, Ec)
    sel = masks[n] & hit
    Pc = Pt[sel] @ Ec[:3, :3].T + Ec[:3, 3]
    ztrue = Pc[:, 2]; zpred = D[n].depth_m[sel]; ok = np.isfinite(zpred); err = zpred[ok] - ztrue[ok]
    print(f"{n}: can px {ok.sum()}, DA3 z-depth error median {np.median(err)*1000:+.1f} mm, |err| p50 {np.median(np.abs(err))*1000:.1f} mm, p90 {np.percentile(np.abs(err),90)*1000:.1f} mm, true z median {np.median(ztrue):.3f} m")
