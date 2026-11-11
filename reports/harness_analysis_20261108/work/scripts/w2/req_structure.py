import json, sys, glob, re
f = sys.argv[1]
d = json.load(open(f))
def walk(o, path="", depth=0):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}", depth+1)
    elif isinstance(o, list):
        if len(o) and all(isinstance(x,(int,float)) for x in o):
            print(path, "list[num]", len(o)); return
        for i, v in enumerate(o[:50]):
            walk(v, f"{path}[{i}]", depth+1)
    elif isinstance(o, str):
        s = o if not o.startswith("data:") else "<dataurl>"
        print(path, "str", len(o), repr(s[:100]))
    else:
        print(path, type(o).__name__, o)
walk(d)
