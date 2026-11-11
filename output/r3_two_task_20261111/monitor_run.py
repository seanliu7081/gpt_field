"""Read saved run progress; optionally extract the latest request's RGB pair."""
import argparse
import base64
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--images", action="store_true")
a = p.parse_args()
t = a.run / "trial"
state_path = t / "run_state.json"
if not state_path.exists():
    print("Starting; run_state not yet written")
    raise SystemExit(0)
state = json.loads(state_path.read_text())
print("STATE", json.dumps(state, ensure_ascii=False))
print("state_age_s", round(time.time() - state_path.stat().st_mtime, 1))
events = []
ep = t / "foundation_vision_harness_V3/events.jsonl"
if ep.exists():
    for line in ep.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
print("accepted_requests", sum(e["event"] == "model_request" for e in events))
for e in events[-8:]:
    answer, outcome = e.get("answer", {}), e.get("outcome", {})
    print(e["event"], e.get("step"), round(e.get("online_s", 0), 1),
          answer.get("status"), answer.get("why"), outcome.get("reason"))
print("result_ready", (a.run / "result.json").exists())
if a.images:
    requests = sorted((t / "requests").glob("*.request.json"))
    if requests:
        request = json.loads(requests[-1].read_text())
        images = []
        for m in request["payload"]["input"]:
            if isinstance(m, dict) and m.get("role") == "user":
                found = [i["image_url"] for i in m.get("content", [])
                         if isinstance(i, dict) and i.get("type") == "input_image"]
                if found:
                    images = found
        (a.run / "monitor").mkdir(exist_ok=True)
        for name, uri in zip(("external", "wrist"), images):
            (a.run / "monitor" / f"latest_{name}.png").write_bytes(
                base64.b64decode(uri.split(",", 1)[1]))
        print("image_frame", request.get("request_meta"))
