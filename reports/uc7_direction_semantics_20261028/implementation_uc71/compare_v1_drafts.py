"""Read-only V1 draft comparison; writes reports here, never edits a draft.

No simulator/model calls. Baselines use recorded initial drafts, not outcomes or
manual edits. Object identity comparisons use part descriptions and step refs,
not assumed stable part IDs. Exact text equality is descriptive, not a gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from gpt_field.agent_speedup_v2_user_check_direction_display import (
    region_direction_lines, world_axis_wording_diagnostics,
)
from gpt_field.agent_speedup_v2_user_check_spatial import region_directions
from gpt_field.agent_speedup_v2_user_check import plan_validation

EXPECTED = [f"libero_10_t{i}" for i in range(10)] + ["libero_goal_g5", "libero_10_t6_relayout", "robocasa_coffee"]


def _load(path):
    return json.loads(path.read_text())


def _baseline(name):
    if name == "libero_10_t6_relayout":
        return ROOT / "runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6/review/review.json"
    if name == "libero_goal_g5":
        return ROOT / "runs/agent_speedup_v3_g5_001_review/review.json"
    if name == "robocasa_coffee":
        return ROOT / "runs/robocasa_coffee_review_002/review.json"
    candidates = sorted((ROOT / "runs").glob(f"agent_speedup_v2_user_check_{name}_*_review/review.json"))
    # Earliest recorded revision 7 draft, selected by version, not outcome.
    for path in candidates:
        if _load(path).get("version") == "agent-speedup-v2-user-check-7":
            return path
    return ROOT / f"runs/agent_speedup_v2_user_check_{name}_001_review/review.json"


def _text(plan, pid):
    return next((p["text"] for p in plan["parts"] if p["id"] == pid), "")


def _summary(plan):
    return {"parts": plan["parts"], "regions": plan.get("regions", []),
            "placement_goals": plan.get("placement_goals", []),
            "steps": [dict(s, object_text=_text(plan,s["obj"]),
                           reference_text=_text(plan,s.get("ref"))) for s in plan["steps"]]}


def _check(row, name, passed, evidence):
    row["checks"].append({"name": name, "passed": bool(passed), "evidence": evidence})


def _plate_part(plan, side):
    return [p for p in plan["parts"] if "plate" in p["text"].lower()
            and re.search(rf"\b{side}\b", p["text"].lower())]


def _t4_checks(row, plan, baseline):
    mappings = {}
    for side in ("left", "right"):
        old, new = _plate_part(baseline, side), _plate_part(plan, side)
        evidence = {"baseline_candidates": old, "current_candidates": new,
                    "adjacency_definition": "Chebyshev distance <= 1 on the external 8x8 grid (including diagonal neighbors)"}
        matched = len(old) == len(new) == 1 and all(1 <= p.get("x",0) <= 64 for p in old+new)
        if matched:
            before, after = np.array(divmod(old[0]["x"]-1,8)), np.array(divmod(new[0]["x"]-1,8))
            distance = int(np.abs(after-before).max())
            evidence.update(external_cell_before=old[0]["x"], external_cell_after=new[0]["x"], cell_distance=distance)
            mappings[side] = new[0]["id"]
            matched = distance <= 1
        _check(row, f"t4_{side}_plate_same_or_adjacent", matched, evidence)
    placements = [step for step in plan["steps"] if step["do"] == "place_on"]
    white_to_left = any("white" in _text(plan,s["obj"]).lower()
                        and "yellow" not in _text(plan,s["obj"]).lower()
                        and s.get("ref") == mappings.get("left") for s in placements)
    yellow_to_right = any("yellow" in _text(plan,s["obj"]).lower()
                          and s.get("ref") == mappings.get("right") for s in placements)
    _check(row, "t4_intended_mugs_target_the_intended_plates", white_to_left and yellow_to_right,
           {"white_mug_to_left_plate": white_to_left, "yellow_white_mug_to_right_plate": yellow_to_right,
            "resolved_steps": _summary(plan)["steps"]})


def _t5_checks(row, plan):
    targets = [p for p in plan["parts"] if re.search(r"\b(back|rear)\b",p["text"].lower())
               and re.search(r"\b(compartment|section|slot)\b",p["text"].lower())
               and re.search(r"\b(caddy|organizer|storage)\b",p["text"].lower())]
    ids = {p["id"] for p in targets}
    placements = [s for s in plan["steps"] if s["do"] == "place_in"
                  and "book" in _text(plan,s["obj"]).lower() and s.get("ref") in ids]
    _check(row, "t5_book_into_physical_back_compartment", bool(targets and placements),
           {"target_parts": targets, "matching_steps": placements,
            "note": "Checks intended physical identity and step reference, not a new behind half-plane."})


def _direction_checks(row, plan, contract, *, relation, anchor_word, object_word, operation, vector):
    candidates = [spec for spec in plan.get("regions", []) if spec["relation"] == relation
                  and anchor_word in _text(plan,spec.get("anchor")).lower()]
    rows = []
    for region in candidates:
        actual = contract.get("regions", {}).get(region["id"], {}).get("direction_world_xy")
        steps = [s for s in plan["steps"] if s["do"] == operation and s.get("ref") == region["id"]
                 and object_word in _text(plan,s["obj"]).lower()]
        rows.append({"spec": region, "anchor_text": _text(plan,region["anchor"]), "direction": actual,
                     "vector_matches": actual is not None and bool(np.allclose(actual,vector,atol=1e-5,rtol=0)),
                     "object_steps": steps})
    passed = bool(rows) and any(r["vector_matches"] and r["object_steps"] for r in rows)
    _check(row, "directional_region_and_intended_object_reference", passed,
           {"expected_relation": relation, "expected_anchor": anchor_word, "expected_world_xy": vector,
            "candidates": rows, "tolerance_note": "1e-5 numerical comparison only, no runtime threshold"})


def main():
    root = HERE / "v1_planning"
    progress = _load(root / "summary.json")
    progress_rows = {row["name"]:row for row in progress["cases"]}
    rows = []
    for name in EXPECTED:
        path = root / name / "review.json"
        info = progress_rows.get(name,{"status":"not_started"})
        if not path.exists():
            rows.append({"name":name,"status":info["status"],"error":info.get("error"),"checks":[]})
            continue
        review = _load(path)
        plan = review["draft_plan"]
        frame = review["preparation"].get("view_frame")
        contract = region_directions(frame,plan.get("regions",[]))
        base_path = _baseline(name)
        baseline_record = _load(base_path) if base_path.exists() else None
        baseline = baseline_record["draft_plan"] if baseline_record else None
        row = {"name":name,"status":"draft_prepared", "review_path":str(path.relative_to(ROOT)),
               "review_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
               "baseline_path":str(base_path.relative_to(ROOT)) if baseline else None,
               "baseline_kind":"v3 planning draft (shared planner), not uc7 execution evidence" if name=="libero_goal_g5" else "recorded user-check initial draft",
               "baseline_version":baseline_record.get("version") if baseline_record else None,
               "baseline_draft_validation":plan_validation(baseline) if baseline else None,
               "baseline_draft_differs_from_reviewed_plan":bool(baseline_record and baseline_record["draft_plan"]!=baseline_record["plan"]),
               "task_prompt":review["task_prompt"],"version":review["version"],
               "same_initial_observation":bool(baseline_record and review["initial_observation_sha256"]==baseline_record["initial_observation_sha256"]),
               "quarter_turns":review["preparation"].get("quarter_turns"),
               "region_directions":contract,"direction_lines":region_direction_lines(contract),
               "before":_summary(baseline) if baseline else None,"after":_summary(plan),"checks":[]}
        validation = plan_validation(plan)
        row["validation_matches_baseline_draft"] = bool(baseline and validation == row["baseline_draft_validation"])
        diagnostics = world_axis_wording_diagnostics(plan)
        _check(row,"plan_schema_and_semantics_valid",validation["valid"],validation)
        _check(row,"no_world_signed_horizontal_axis_in_part_text",not diagnostics,diagnostics)
        _check(row,"frozen_external_frame_available",contract["available"],{"reason":contract.get("reason"),"frame":contract.get("frame")})
        _check(row,"draft_only_no_approval_or_policy_actions",review["status"]=="draft" and review["approved"] is None and review["preparation"]["policy_actions"]==0,
               {"review_status":review["status"],"approved":review["approved"],"policy_actions":review["preparation"]["policy_actions"]})
        if name in {f"libero_10_t{i}" for i in (3,4,5,9)}:
            _check(row,"physical_parts_do_not_create_regions",not plan.get("regions"),plan.get("regions"))
        if name=="libero_10_t4":
            _t4_checks(row,plan,baseline)
        if name=="libero_10_t5":
            _t5_checks(row,plan)
        if name in ("libero_10_t6","libero_10_t6_relayout"):
            _direction_checks(row,plan,contract,relation="right_of",anchor_word="plate",object_word="pudding",operation="place_on",vector=[0,1])
        if name=="libero_goal_g5":
            _direction_checks(row,plan,contract,relation="front_of",anchor_word="stove",object_word="plate",operation="push_to",vector=[1,0])
        if name=="robocasa_coffee":
            _check(row,"robocasa_synchronized_robot_mounted_camera",frame["external_camera"].get("pose_source")=="synchronized_robot_mounted_camera",
                   {"pose_source":frame["external_camera"].get("pose_source")})
        row["passed"]=all(check["passed"] for check in row["checks"])
        rows.append(row)
    failures = [{"name":row["name"],"check":check["name"],"evidence":check["evidence"]}
                for row in rows for check in row["checks"] if not check["passed"]]
    preparation_errors = [{"name":row["name"],"error":row.get("error")} for row in rows if row["status"]=="preparation_error"]
    complete = all(row["status"] in ("draft_prepared","preparation_error") for row in rows)
    direction_checks = [(row["name"],check) for row in rows for check in row["checks"]
                        if check["name"] not in ("plan_schema_and_semantics_valid","draft_only_no_approval_or_policy_actions")]
    direction_passed = complete and not preparation_errors and all(check["passed"] for _,check in direction_checks)
    validity = [(row["name"],check) for row in rows for check in row["checks"] if check["name"]=="plan_schema_and_semantics_valid"]
    result = {"generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "scope":"V1 planning drafts only; no task success or live execution claim",
              "expected_cases":len(EXPECTED),"prepared_cases":sum(row["status"]=="draft_prepared" for row in rows),
              "complete":complete,"status":"failed" if failures or preparation_errors else "passed" if complete else "pending",
              "task_success_acceptance":"V2/V3 require user approval and actual task success; V1 cannot satisfy that criterion",
              "baseline_selection":"Earliest recorded revision 7 original-layout draft by numeric directory order; fixed historical relayout draft; g5 first v3 shared-planner draft; recorded RoboCasa draft. No selection by outcome.",
              "direction_criteria":{"all_passed":direction_passed,"checks_passed":sum(check["passed"] for _,check in direction_checks),"checks_total":len(direction_checks),"cases_passed":sum(row.get("status")=="draft_prepared" and all(check["passed"] for check in row["checks"] if check["name"] not in ("plan_schema_and_semantics_valid","draft_only_no_approval_or_policy_actions")) for row in rows)},
              "draft_validity":{"valid_count":sum(check["passed"] for _,check in validity),"invalid_cases":[name for name,check in validity if not check["passed"]]},
              "failures":failures,"preparation_errors":preparation_errors,"cases":rows,
              "visual_comparison":_load(HERE/"v1_visual_comparison.json") if (HERE/"v1_visual_comparison.json").exists() else None}
    out=HERE/"v1_draft_comparison.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# uc7.1 V1 规划草稿对比", "",f"状态：**{result['status']}**；已生成 {result['prepared_cases']}/{len(EXPECTED)} 份草稿。", "",
           "本报告只检查规划与冻结方向。没有审批、执行动作或任务成功结论；V2/V3 仍需用户批准，并以任务成功验收。", "",
           f"方向及部件身份检查：{result['direction_criteria']['cases_passed']}/{result['prepared_cases']} 个已完成案例通过，{result['direction_criteria']['checks_passed']}/{len(direction_checks)} 项检查通过。草稿有效性：{result['draft_validity']['valid_count']}/{result['prepared_cases']}。整体 V1 状态保留为 {result['status']}。", "",
           "| 案例 | 状态 | 与历史初始观测相同 | 方向区域 |", "|---|---|---|---|"]
    for row in rows:
        status = "通过" if row.get("passed") else "有偏差" if "passed" in row else row["status"]
        lines.append(f"| {row['name']} | {status} | {row.get('same_initial_observation','—')} | {'; '.join(row.get('direction_lines',[])) or '—'} |")
    lines += ["", "Task 4 按左右盘子的物体描述匹配 ID，再比较外部 8×8 网格；相邻包含对角格。Task 5 检查书本指向收纳盒的实体后侧隔间，不能用 behind 区域代替。", "",
              "原始布局基线按目录顺序选择最早的 revision 7 草稿，不按成功结果挑选；t2/t3/t6 分别为 002/003/009，t4 固定为 001。g5 的历史对照为 v3 保存的规划草稿；其执行结果不作为 uc7 基线。完整前后部件、步骤引用、逐项证据与草稿哈希见 [v1_draft_comparison.json](v1_draft_comparison.json)。人工核对原始网格图像的记录见 [v1_visual_comparison.json](v1_visual_comparison.json)。", ""]
    if failures or preparation_errors:
        lines += ["## 偏差", ""]
        for item in failures:
            matched = next(row for row in rows if row["name"]==item["name"])
            prior = "（历史原始草稿已有相同校验问题；旧审核计划后来已修改）" if item["check"]=="plan_schema_and_semantics_valid" and matched.get("validation_matches_baseline_draft") and matched.get("baseline_draft_differs_from_reviewed_plan") else ""
            lines.append(f"- {item['name']}: {item['check']}{prior} — {json.dumps(item['evidence'],ensure_ascii=False)}")
        for item in preparation_errors:
            lines.append(f"- {item['name']}: preparation_error — {item['error']}")
    else:
        lines.append("目前已生成的草稿未发现上述检查的偏差。" if not complete else "13 份草稿均通过列出的 V1 检查。")
    (HERE/"v1_draft_comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({key:result[key] for key in ("status","prepared_cases","expected_cases","complete","failures","preparation_errors")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
