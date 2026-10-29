"""Plan A: per-segment motion records from uc7 trace.jsonl feedback rows (read-only; python3 -I).

For each fragment segment: skill, profile, entry translation error E0 = |target - entry position|,
entry rotation error R0, steps executed, end reason, along-path realized displacement per step
(projected on the commanded direction), commanded translation per step, realized rotation per step.
Output: plan/A_segments.json
"""
import json, math, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
OUT = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/A_segments.json"
RUNS = "t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()


def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)


def qang(a, b):
    """rotation angle between wxyz quaternions"""
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    d = abs(sum(x*y for x, y in zip(a, b))) / (na*nb)
    return 2*math.acos(min(1.0, d))


def rotvec_q(v):
    ang = math.sqrt(sum(x*x for x in v))
    if ang < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(ang/2)/ang
    return (math.cos(ang/2), v[0]*s, v[1]*s, v[2]*s)


def proc(key):
    segs = {}
    order = []
    frag_id = 0
    last_fs = None
    with open(ROOT + key + "/trace.jsonl") as f:
        for line in f:
            if not line.startswith('{"event": "feedback"'):
                continue
            r = json.loads(line)
            fb = r["feedback"]
            pf = fb.get("primitive_feedback") or {}
            fr = pf.get("fragment")
            if fr is None:
                last_fs = None
                continue
            fs = fr["observed_at_fragment_step"]
            if last_fs is None or fs <= last_fs:
                frag_id += 1
            last_fs = fs
            b, a = fb["before_state"], fb["after_state"]
            na = fb["normalized_action"]
            cmd = [x*0.05 for x in na[:3]]
            real = [a[i]-b[i] for i in range(3)]
            c = math.sqrt(sum(x*x for x in cmd))
            along = (sum(cmd[i]*real[i] for i in range(3))/c) if c > 0 else 0.0
            rreal = qang(rotvec_q(b[3:6]), rotvec_q(a[3:6]))
            # which segment did this step belong to: the one whose entry < fs <= end (or running)
            cur = None
            for seg in fr["segments"]:
                e = seg["entry_at_fragment_step"]; en = seg.get("end_at_fragment_step")
                if e < fs and (en is None or en >= fs):
                    cur = seg
                    break
            if cur is None:
                continue
            sid = (frag_id, cur["index"])
            if sid not in segs:
                ep = cur["entry_pose_wxyz"]
                tq = cur["target_quaternion_wxyz"]
                E0 = math.dist(ep[:3], cur["target_xyz_m"])
                R0 = qang(ep[3:7], tq)
                segs[sid] = dict(run=key, frag=frag_id, seg=cur["index"], skill=cur["skill"],
                                 profile=cur["request"]["profile"], E0=E0, R0=R0, max_steps=cur["request"]["max_steps"],
                                 obs_after=cur["request"]["observation_after"], steps=0, cmd=[], along=[], rot=[],
                                 err=[], rerr=[], reason=None, first_global_step=fb["step"])
                order.append(sid)
            s = segs[sid]
            s["steps"] += 1
            s["cmd"].append(round(c, 5)); s["along"].append(round(along, 5)); s["rot"].append(round(rreal, 5))
            s["err"].append(round(cur.get("position_error_m") or 0.0, 5))
            s["rerr"].append(round(cur.get("orientation_error_rad") or 0.0, 5))
            if cur.get("reason"):
                s["reason"] = cur["reason"]
            if fr.get("reason") and fr.get("status") in ("failed", "succeeded") and cur["index"] == fr.get("current_segment_index"):
                s["frag_reason"] = fr["reason"]
    return [segs[k] for k in order]


if __name__ == "__main__":
    allsegs = []
    with ProcessPoolExecutor(max_workers=13) as ex:
        for res in ex.map(proc, RUNS):
            allsegs.extend(res)
            print(res[0]["run"] if res else None, len(res), flush=True)
    json.dump(allsegs, open(OUT, "w"))
    print("segments", len(allsegs))
