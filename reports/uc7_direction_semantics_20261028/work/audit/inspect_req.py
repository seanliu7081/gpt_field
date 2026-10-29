import json, sys
p = sys.argv[1]
d = json.load(open(p))
def walk(x, path="", depth=0):
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, path + "/" + str(k), depth+1)
    elif isinstance(x, list):
        if len(x) and all(not isinstance(i,(dict,list)) for i in x) and len(x) < 20:
            print(path, "=", json.dumps(x)[:200])
        else:
            for i, v in enumerate(x[:50]):
                walk(v, path + f"[{i}]", depth+1)
    else:
        s = json.dumps(x)
        if len(s) > 300:
            if s.startswith('"data:image'):
                print(path, "= <image", len(s), ">")
            else:
                print(path, "= (len", len(s), ")", s[:300])
        else:
            print(path, "=", s)
walk(d)
