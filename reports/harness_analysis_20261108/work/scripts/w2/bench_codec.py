"""Time PNG encode (level 6 default vs 1) and decode on a saved 512x512 observation, mirroring observation.py."""
import base64, io, json, sys, time, zlib
import numpy as np
from PIL import Image
obs = json.load(open(sys.argv[1]))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
def dec(url):
    raw = base64.b64decode(url[len("data:image/png;base64,"):], validate=True)
    im = Image.open(io.BytesIO(raw)); im.load(); return im
res = {}
for key in ("observation/image", "observation/wrist_image"):
    url = obs[key]
    im = dec(url); arr = np.asarray(im)
    t = time.perf_counter()
    for _ in range(N): dec(url)
    d = (time.perf_counter() - t) / N
    enc = {}
    for level in (None, 1):
        t = time.perf_counter()
        for _ in range(N):
            b = io.BytesIO(); fr = Image.fromarray(np.ascontiguousarray(arr))
            (fr.save(b, format="PNG") if level is None else fr.save(b, format="PNG", compress_level=level))
            s = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("ascii")
        enc["default" if level is None else "level1"] = ((time.perf_counter() - t) / N, len(s))
    t = time.perf_counter()
    for _ in range(N): json.loads(json.dumps({"a": url}))
    js = (time.perf_counter() - t) / N
    res[key] = {"shape": arr.shape, "decode_s": d, "encode": enc, "json_roundtrip_s": js}
import PIL
print(json.dumps({"pillow": PIL.__version__, "zlib": zlib.ZLIB_VERSION, "N": N, "res": res}, default=str, indent=1))
