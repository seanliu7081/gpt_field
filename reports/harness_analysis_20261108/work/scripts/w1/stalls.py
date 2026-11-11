import json, sys, os
run = sys.argv[1]
tr = os.path.join(run, 'trial', 'trace.jsonl')
res = []
with open(tr) as f:
    for line in f:
        if not line.startswith('{"event": "feedback"'):
            continue
        d = json.loads(line)
        fb = d['feedback']; pf = fb.get('primitive_feedback') or {}; frag = pf.get('fragment') or {}
        if frag.get('reason') not in ('segment_stall',): continue
        if frag.get('status') != 'failed': continue
        tgt = frag.get('target_xyz_m'); pose = frag.get('measured_pose_wxyz')
        segs = frag.get('segments') or []
        ci = frag.get('current_segment_index'); cur = segs[ci] if ci is not None and ci < len(segs) else {}
        req = cur.get('request') or {}
        err = [p - t for p, t in zip(pose[:3], tgt)]
        res.append(dict(step=d['step'], skill=frag.get('current_skill'), profile=req.get('profile'), gap_mm=round(frag.get('measured_gap_mm'),1), grip=frag.get('gripper_command'),
                        err_mm=[round(e*1000,1) for e in err], err_norm_mm=round(sum(e*e for e in err)**.5*1000,1), rot_err=round(frag.get('orientation_error_rad') or 0,3),
                        actions=cur.get('actions'), stall=cur.get('stall_evidence')))
for r in res:
    st = r.pop('stall') or {}
    print(r, 'improvement_mm=%.2f' % (1000*(st.get('weighted_error_improvement_m') or 0)))
