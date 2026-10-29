"""Plan A: stdlib-only lower bound on per-step image handling cost (no simulator, no PIL).

Takes 20 'executed' rows of uc7 t1_001 trace.jsonl, decodes the two PNG data URLs with base64 + zlib
(the PNG IDAT inflate is the dominant part of PIL's PNG decode), re-deflates the raw scanlines at
zlib level 6 (Pillow's default PNG compress_level) and level 1, and times json.dumps/loads of the full
observation dict that crosses the worker pipe. Numbers are machine-specific LOWER BOUNDS for the
corresponding steps inside simulator_step_s (PIL also un-filters/filters rows; GPU render not measured).
"""
import base64, json, time, zlib, struct, statistics as st

TRACE = "/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_t1_001/trace.jsonl"
rows = []
with open(TRACE) as f:
    for line in f:
        if line.startswith('{"event": "executed"'):
            r = json.loads(line)
            if r["step"] % 40 == 0:
                rows.append(r["observation"])
            if len(rows) >= 20:
                break


def idat(png):
    pos, out = 8, []
    while pos < len(png):
        n = struct.unpack(">I", png[pos:pos+4])[0]
        typ = png[pos+4:pos+8]
        if typ == b"IDAT":
            out.append(png[pos+8:pos+8+n])
        pos += 12 + n
    return b"".join(out)


t_b64d, t_inf, t_def6, t_def1, t_b64e, t_dump, t_load, sizes = [], [], [], [], [], [], [], []
for obs in rows:
    for key in ("observation/image", "observation/wrist_image"):
        url = obs[key]
        a = time.perf_counter(); png = base64.b64decode(url.split(",", 1)[1]); b = time.perf_counter()
        t_b64d.append(b - a)
        data = idat(png)
        a = time.perf_counter(); raw = zlib.decompress(data); b = time.perf_counter()
        t_inf.append(b - a)
        a = time.perf_counter(); z6 = zlib.compress(raw, 6); b = time.perf_counter()
        t_def6.append(b - a)
        a = time.perf_counter(); z1 = zlib.compress(raw, 1); b = time.perf_counter()
        t_def1.append(b - a)
        a = time.perf_counter(); base64.b64encode(z6); b = time.perf_counter()
        t_b64e.append(b - a)
        sizes.append((len(png), len(raw), len(z6), len(z1)))
    a = time.perf_counter(); s = json.dumps(obs); b = time.perf_counter(); t_dump.append(b - a)
    a = time.perf_counter(); json.loads(s); b = time.perf_counter(); t_load.append(b - a)

ms = lambda xs: 1000 * st.median(xs)
print("images", len(t_inf), "png bytes median", st.median(x[0] for x in sizes), "raw", st.median(x[1] for x in sizes))
print("per image (ms, median): b64decode %.2f inflate %.2f deflate6 %.2f deflate1 %.2f b64encode %.2f" % (
    ms(t_b64d), ms(t_inf), ms(t_def6), ms(t_def1), ms(t_b64e)))
print("per observation json dumps %.2f ms loads %.2f ms, json chars %d" % (ms(t_dump), ms(t_load), len(s)))
dec = ms(t_b64d) + ms(t_inf)
enc = ms(t_def6) + ms(t_b64e)
# per control step in uc7: 2 encodes + 2 worker validation decodes (worker), JSON dump in worker + load in host,
# 4 host validation decodes (from_dict + CalibratedObservation), video panels decode 2 (cached per observation)
per_step = 2 * enc + 2 * dec + ms(t_dump) + ms(t_load) + 4 * dec
print("lower-bound image/serialization work inside simulator_step_s per step: %.1f ms "
      "(2 enc %.1f + 6 dec %.1f + json %.1f)" % (per_step, 2 * enc, 6 * dec, ms(t_dump) + ms(t_load)))
print("same with deflate level 1 instead of 6: %.1f ms" % (2 * (ms(t_def1) + ms(t_b64e)) + 6 * dec + ms(t_dump) + ms(t_load)))
