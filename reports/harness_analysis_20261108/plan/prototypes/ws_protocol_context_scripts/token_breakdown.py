"""Per-call input token attribution for one H1 run (offline, read-only)."""
import json, glob, sys, os
run = sys.argv[1]
req_dir = os.path.join(run, "trial", "requests")
reqs = sorted(glob.glob(os.path.join(req_dir, "*.request.json")))
rows = []
for rq in reqs:
    base = rq[:-len(".request.json")]
    try:
        resp = json.load(open(base + ".response.json"))
    except FileNotFoundError:
        continue
    req = json.load(open(rq))
    payload = req["payload"]
    usage = resp.get("usage", {})
    attr = (usage.get("attribution") or {}).get("items", {})
    rf = (usage.get("attribution") or {}).get("request_fields", {})
    # classify input items in payload order
    inp = payload["input"]
    n_user = sum(1 for m in inp if m.get("role") == "user")
    n_img = sum(1 for m in inp if m.get("role") == "user" for p in m.get("content", []) if p.get("type") == "input_image")
    n_stub = sum(1 for m in inp if m.get("role") == "user" for p in m.get("content", []) if p.get("type") == "input_text" and p.get("text","").startswith("[Older camera image omitted"))
    # attribution: user msgs text vs images (content index 0 is text, then label/image pairs)
    user_text = img = asst = reason = 0
    for key, item in attr.items():
        if key.startswith("rs_"):
            reason += item.get("input_tokens", 0)
        elif key.startswith("msg_"):
            content = item.get("content") or []
            if len(content) == 5:  # user obs: text, label, image, label, image
                user_text += content[0]["input_tokens"]
                img += content[2]["input_tokens"] + content[4]["input_tokens"]
                user_text += content[1]["input_tokens"] + content[3]["input_tokens"]
            elif len(content) == 1:
                # assistant message replayed or user stubbed message
                asst += item.get("input_tokens", 0)
            else:
                user_text += item.get("input_tokens", 0)
    last_user = inp[-1]
    text = last_user["content"][0]["text"]
    fields = json.loads(text)
    ctx = fields.get("context", {})
    sizes = {k: len(json.dumps(v, separators=(",", ":"))) for k, v in fields.items() if k != "context"}
    csizes = {k: len(json.dumps(v, separators=(",", ":"))) for k, v in ctx.items()}
    rows.append({"call": os.path.basename(rq)[:6], "input": usage.get("input_tokens"),
                 "cached": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                 "instr": (rf.get("instructions") or {}).get("input_tokens"),
                 "format": (rf.get("text.format") or {}).get("input_tokens"),
                 "user_text_hist": user_text, "images": img, "assistant_hist": asst, "reasoning_hist": reason,
                 "n_user": n_user, "n_img": n_img, "n_stub": n_stub,
                 "reasoning_out": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                 "visible_out": usage.get("output_tokens", 0) - ((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
                 "cur_chars": len(text), "cur_fields": sizes, "cur_context": csizes,
                 "prompt_cache_key_sent": "prompt_cache_key" in payload})
json.dump(rows, open(sys.argv[2], "w"), indent=1)
for r in rows:
    print(r["call"], "in", r["input"], "cached", r["cached"], "instr", r["instr"], "fmt", r["format"],
          "usertext", r["user_text_hist"], "img", r["images"], "asst", r["assistant_hist"], "rs", r["reasoning_hist"],
          "nuser", r["n_user"], "nimg", r["n_img"], "stub", r["n_stub"], "curchars", r["cur_chars"], "key_sent", r["prompt_cache_key_sent"])
last = rows[-1]
print("last call field chars:", last["cur_fields"])
print("last call context chars:", last["cur_context"])
print("first call context chars:", rows[0]["cur_context"])
