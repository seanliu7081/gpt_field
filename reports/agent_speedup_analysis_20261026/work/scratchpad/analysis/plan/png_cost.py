"""Stdlib-only estimate of per-step PNG work: inflate/deflate the actual 512x512 frames from a uc7 trace.
No PIL, no simulator. PNG = filtered scanlines + zlib; timing zlib on the real filtered stream bounds the
deflate/inflate share of PIL's encode/decode (PIL adds filter selection, CRC and chunking)."""
import base64, json, sys, time, zlib, struct

def idat(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, data, ihdr = 8, b"", None
    while pos < len(png):
        n = struct.unpack(">I", png[pos:pos+4])[0]; typ = png[pos+4:pos+8]
        body = png[pos+8:pos+8+n]
        if typ == b"IHDR": ihdr = struct.unpack(">IIBBBBB", body)
        if typ == b"IDAT": data += body
        pos += 12 + n
    return ihdr, data

def timeit(fn, n=20):
    best = []
    for _ in range(n):
        t = time.perf_counter(); fn(); best.append(time.perf_counter() - t)
    best.sort(); return best[len(best)//2]

path = sys.argv[1]
rows = 0
with open(path, "rb") as fh:
    for line in fh:
        if b'"event": "executed"' not in line[:200] and b'"event":"executed"' not in line[:200]:
            continue
        row = json.loads(line)
        obs = row["observation"]
        out = {}
        for key in ("observation/image", "observation/wrist_image"):
            url = obs[key]
            png = base64.b64decode(url.split(",", 1)[1])
            ihdr, comp = idat(png)
            raw = zlib.decompress(comp)
            out[key] = {
                "ihdr_w_h_depth_colortype": ihdr[:4], "png_bytes": len(png), "b64_chars": len(url),
                "filtered_raw_bytes": len(raw),
                "inflate_ms": 1000*timeit(lambda: zlib.decompress(comp)),
                "b64decode_ms": 1000*timeit(lambda: base64.b64decode(url.split(",", 1)[1])),
                "deflate_level6_ms": 1000*timeit(lambda: zlib.compress(raw, 6), 10),
                "deflate_level1_ms": 1000*timeit(lambda: zlib.compress(raw, 1), 10),
                "level6_bytes": len(zlib.compress(raw, 6)), "level1_bytes": len(zlib.compress(raw, 1)),
            }
        js = json.dumps(row["observation"])
        out["observation_json_chars"] = len(js)
        out["json_dumps_ms"] = 1000*timeit(lambda: json.dumps(row["observation"]))
        out["json_loads_ms"] = 1000*timeit(lambda: json.loads(js))
        out["step"] = row.get("step")
        print(json.dumps(out, indent=1))
        rows += 1
        if rows >= 2:
            break
