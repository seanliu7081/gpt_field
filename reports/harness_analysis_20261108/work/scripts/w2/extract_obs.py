"""Extract the observation of one executed trace row (default: first executed) to a small JSON file."""
import json, sys
trace, out, want = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
with open(trace) as f:
    for line in f:
        r = json.loads(line)
        if r.get("event") == "executed" and r.get("step") == want:
            json.dump(r["observation"], open(out, "w"))
            print("wrote step", want, "image chars", len(r["observation"]["observation/image"]))
            break
