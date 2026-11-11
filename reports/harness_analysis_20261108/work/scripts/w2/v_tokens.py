import json, glob, numpy as np
R = "/workspace/gpt_field/runs/"
runs = {"H1T04": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
        "H1T08": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
        "H3T04": "foundation_vision_t04_can_obstacle_20261105_001",
        "H3T08": "foundation_vision_t08_bottle_obstacle_20261105_001"}
def vis_chars(text):
    j = json.loads(text)
    fv = j.get("context", {}).get("foundation_vision")
    if fv is None: return 0, j
    # wire substring length: locate the serialized block exactly
    key = '"foundation_vision":'
    i = text.find(key)
    assert i >= 0
    # parse forward with raw_decode to find the end
    dec = json.JSONDecoder(); obj, end = dec.raw_decode(text, i + len(key))
    return end - (i + len(key)), j
samples = []  # (run, nonvis_chars, vis_chars, tokens)
per_req = {}
for tag, r in runs.items():
    reqs = sorted(glob.glob(R + r + "/trial/requests/*.request.json"))
    totals = []
    for rq in reqs:
        d = json.load(open(rq)); resp = json.load(open(rq.replace(".request.json", ".response.json")))
        users = [m for m in d["payload"]["input"] if m.get("role") == "user"]
        att = resp["usage"]["attribution"]["items"]
        # user messages in attribution are those with 5 content parts and input_tokens>0 (msg_ ids); order preserved?
        att_users = [v for k, v in att.items() if k.startswith("msg_") and v.get("input_tokens", 0) > 0 and len(v.get("content", [])) == 5]
        assert len(att_users) == len(users), (tag, rq, len(att_users), len(users))
        vis_tot = 0; text_tok_tot = 0
        rows = []
        for m, a in zip(users, att_users):
            text = m["content"][0]["text"]; tok = a["content"][0]["input_tokens"]
            vc, _ = vis_chars(text)
            rows.append((len(text) - vc, vc, tok))
        samples.extend((tag,) + x for x in rows[-1:])  # each message once (last occurrence = its own request)
        totals.append(dict(input=resp["usage"]["input_tokens"], rows=rows, instr=resp["usage"]["attribution"]["request_fields"]["instructions"]["input_tokens"]))
    per_req[tag] = totals
S = np.array([[x[1], x[2], x[3]] for x in samples], float)
# fit tokens = a*nonvis + b*vis (no intercept) using all messages
A = S[:, :2]; y = S[:, 2]
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
print("tokens per char: nonvision %.4f (%.2f ch/tok)  vision %.4f (%.2f ch/tok)" % (coef[0], 1 / coef[0], coef[1], 1 / coef[1]))
h1 = S[[i for i, x in enumerate(samples) if x[0].startswith("H1")]]
print("H1-only nonvision ch/tok %.2f" % (h1[:, 0].sum() / h1[:, 2].sum()))
for tag in ("H3T04", "H3T08"):
    tot_in = 0; tot_vis = 0
    for i, t in enumerate(per_req[tag]):
        vis_tok = sum(coef[1] * vc for _, vc, _ in t["rows"]); vis_ch = sum(vc for _, vc, _ in t["rows"])
        tot_in += t["input"]; tot_vis += vis_tok
        if i == len(per_req[tag]) - 1:
            print(tag, "last request: vision chars in history =", vis_ch, "~tokens", round(vis_tok), "of input", t["input"], "instr tokens", t["instr"])
    print(tag, "episode vision tokens ~", round(tot_vis), "of", tot_in, "= %.1f%%" % (100 * tot_vis / tot_in))
for tag in ("H1T04", "H1T08"):
    print(tag, "exec instructions tokens", per_req[tag][0]["instr"])
print("--- per-message check: vision tokens = tok - nonvis/3.90")
for x in samples:
    if x[0].startswith("H3"):
        vt = x[3] - x[1] / 3.90
        print(x[0], "nonvis", x[1], "vis", x[2], "tok", x[3], "-> vis tok %.0f, ch/tok %.2f" % (vt, x[2] / vt))
res = y - A @ coef
print("fit residual tokens: mean abs %.1f, max abs %.1f, mean tok %.0f" % (np.abs(res).mean(), np.abs(res).max(), y.mean()))
