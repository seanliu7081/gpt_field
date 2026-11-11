"""Offline check: does the H1 approval path accept the recorded uc7.1 t9 draft?"""
import json, sys, copy
sys.path.insert(0, "/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/src")
from gpt_field.agent_speedup_v2_user_check import plan_validation, validate_review_plan, ReviewError
review = json.load(open("/workspace/gpt_field/reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning/libero_10_t9/review.json"))
draft = review["draft_plan"]
print("steps:", [(s["do"], s["obj"], s["sense"]) for s in draft["steps"]])
print("validation:", json.dumps(plan_validation(draft)))
try:
    validate_review_plan(draft)
    print("approve would PASS")
except ReviewError as exc:
    print("approve would RAISE ReviewError:", exc)
