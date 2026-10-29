"""Check that review image 1 (what the planner saw) == observation image de-mirrored by [:, ::-1]."""
import json, base64, io
import numpy as np
from PIL import Image
RUN = "/workspace/gpt_field/runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6/review"
def dec(url): return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
rv = json.load(open(RUN + "/review.json"))
imgs = rv["preparation"]["review_images"]
print([i["label"] for i in imgs])
shown = dec(imgs[0]["data_url"])
obs = dec(json.load(open(RUN + "/initial_observation.json"))["observation/image"])
print("review==obs[:, ::-1]:", np.array_equal(shown, obs[:, ::-1]), " review==obs:", np.array_equal(shown, obs),
      " review==obs[::-1]:", np.array_equal(shown, obs[::-1]))
