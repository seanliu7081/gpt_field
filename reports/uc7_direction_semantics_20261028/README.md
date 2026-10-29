# uc7 direction semantics: t6 "right of the plate" (2026-10-28)

**Implementation update:** [Version A / uc7.1 delivery](implementation_uc71/README.md) includes A9 and the user's final decisions in [final_plan.md](final_plan.md). V0 adds 71 passing tests with no new regression failures. V1 made 13 planning calls and zero task executions: direction checks pass, but Task 9 reproduces an existing invalid-draft issue, so V1 is not fully passed. One subsequently approved [Task 6 relayout trial](../../runs/agent_speedup_v2_uc71_relayout_t6_20261028_001/README.md) completed: official LIBERO success and direction criterion pass, but strict terminal physical verification fails because the pudding was still held. Completion time excluding video rendering is 604.86 seconds; 14 execution calls plus one reused planning call. Other full task tests remain pending user approval. The read-only investigation below is preserved as historical evidence.

This folder answers one question: how should the framework fix LIBERO-10 task 6? Task 6 failed (user_stop) in `runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001`. The planner prompt says `right_of = world +X`, but LIBERO's plate-right goal region lies on the plate's world +Y side.

The review was read-only. It made no code changes, ran no simulator and made no model calls. It had two parts:
- **Inline checks in the main session:** run artifacts, LIBERO BDDL and camera definitions, and ray-casts of the planner's pointed cells.
- **Workflow `wf_4f8af1e4-32b` (6 agents):**
  - a code-path audit;
  - a check of LIBERO and RoboCasa direction semantics;
  - two independent fix designs, one general and one minimal;
  - an adversarial critique of the main session's lead proposal (P0);
  - a judge that merged everything into one suggestion list.

## Reading order

| File | Contents |
|---|---|
| [final_plan.md](final_plan.md) | **Start here.** The final plan (in Chinese), written after a second review round and three read-only checks of a draft. It supersedes the suggestion list, decisions and verification plan in this README wherever they differ, notably S6 (now a must), D9/S8 (now a ranking preference, experiment E1), S7 (text fix now; rotation in experiment E2), the roll and g5 numbers, and the image-pipeline test. |
| this README | The first-round verdict, suggestions S1–S9 and evidence. |
| [plan_check_result.json](plan_check_result.json) | Three read-only checks of the draft plan ([work/final_plan_draft.md](work/final_plan_draft.md)): references, plumbing, and rules plus validation design. |
| [workflow_result.json](workflow_result.json) | The full structured output. `judged` holds the merged suggestions, with evidence, effort and risks for each item. `audit`, `semantics`, `designGeneral`, `designMinimal` and `critique` hold each agent's raw result. |
| [workflow_script.js](workflow_script.js) | The brief every agent received: the facts, your framework rules and P0. |
| [work/](work/) | The agents' prototype scripts and evidence files (see [Files in work/](#files-in-work)). |

## Verdict

The framework introduced the error. The model did not.

- **All four direction words are wrong, not only `right_of`.**
  - In every LIBERO problem class, the agentview camera sits at +x, y = 0 and looks toward −x.
  - LIBERO's words follow that viewer: right = +y, left = −y, front (toward the camera) = +x, behind = −x.
  - 41/41 decidable BDDL checks agree, with 0 counterexamples ([work/semantics/verify_claim.json](work/semantics/verify_claim.json)).
  - uc7's constant (right_of = +x, left_of = −x, behind = +y, front_of = −y) is that frame rotated −90°.
- **The constant is written in three places:**
  - the planner prompt, `src/gpt_field/agent_speedup_v2_user_check_spatial.py:59-60`;
  - the region resolver, same file `:211-214`;
  - the review UI, `src/gpt_field/agent_speedup_v2_user_check.html:104-105` ("Right of anchor (world +x)") and `:244` ("follow world axes, not image directions").

  In addition, METRIC_INSTRUCTIONS (`src/gpt_field/agent_speedup_v2_user_check_client.py:134-135`) pushes the model toward world-axis wording. It is appended to plan, replan, repair and review. The executive prompt defines no direction words at all.
- **Both models read the scene correctly; the framework overrode them.**
  - The planner pointed region E at external cell 55, which ray-casts to about (0.11, 0.18), the plate's +y side. Following the rule, it still wrote "table: surface to the plate's world +x side".
  - The executive knew image-right is +y (request 1). Yet request 11 says it "places the package in the reviewed right-of region rather than relying on image-right", and it released the pudding at (0.29, −0.02).
- **It recurs on 2 tasks in 2 suites.** v3's libero_goal plans for "push the plate to the front of the stove" (g5_001/002) used `front_of` the stove. The constant maps that to −y, but the BDDL front region is on the stove's +x side.
- **Swapping the constant does not fix it.** RoboCasa coffee's external camera is mounted on the robot, and its image-right is about (0.979, −0.202), roughly +x. No constant world-axis table is right for both suites, so the frame must come from calibration.
- **Two related gaps:**
  1. The plan-time wrist image is not turned (k = 0), although the prompt says its left/right match image 1. At LIBERO start poses it is 180° off.
  2. A region with null bounds covers the whole half-plane beyond the anchor. Every LIBERO side-goal region is instead a box of about 10×10 cm next to the anchor.

## Suggestions

### Must (ship in one frozen build; 0 extra model calls)

| ID | Change | Where |
|---|---|---|
| S1 | One helper derives the external-camera frame from calibration: `right = unit(R_ext[:2,0])`, `away = up × right`. Then right_of = +right, left_of = −right, behind = +away, front_of = −away. Every consumer uses it, and it never reads the wrist camera. For LIBERO it gives right (0, 1) and away (−1, 0). | `agent_speedup_v2_user_check_spatial.py`: new helpers near `:139-153`; `SPATIAL_VERSION` `:23` |
| S2 | Planner text: replace the world-axis rule with the external-view definition below. The model must never restate directions as world axes. Words that name part of an object are pointed at, not regions. Vertical words mean world z. Drop the image left/right sentence from METRIC for plan and replan. Keep directional bounds null, because the plan request carries no calibration. | `spatial.py:59-63`; `user_check_client.py:24, 131-138`; `scripts/evaluate_robocasa_user_check.py:43-44` |
| S3 | Executive: add the same definition to AGENT_INSTRUCTIONS, plus a code-computed `task_contract.region_directions`, e.g. `E: right_of C → world (0,+1)`. Put it beside `regions` so PlanLock's hash and the rule that replans keep regions verbatim are untouched. **This is the change that alters uc7's t6 behaviour.** | `user_check_agent_client.py:41, 51-192, 477-480`; `user_check_agent.py:205-224` |
| S4 | Resolver: replace the fixed-axis tests with a projection onto the relation direction: keep cells beyond the anchor polygon's maximum projection. LIBERO masks are identical to before, only relabelled. Report `view_frame_unavailable` when there is no frame. This is a consistency fix; the macro path is dormant in uc7 (0 grants). | `spatial.py:129-136, 211-214, 235` |
| S5 | Review UI: labels and the note use the external-view definition. | `user_check.html:104-105, 244` |

Proposed S2 text:

```
left_of/right_of/front_of/behind (and these words in the task) are as seen in image 1,
the external camera: right/left = toward image 1's right/left; behind/front_of = farther
from/nearer to that camera along the support. Not the robot's own left/right; never judged
in wrist images. Use the same viewer words in plan text; never restate them as world axes.
Vertical words mean world up/down. A word naming part of an object (back compartment,
left plate) selects that part by pointing; it is not a region.
```

### Should

| ID | Change | Where |
|---|---|---|
| S6 | Freeze the frame at the reviewed reset, inside the approval hash. The review shows each region's computed world direction and an arrow on image 1. | `scripts/evaluate_agent_speedup_v2_user_check.py:238-275`; RoboCasa prepare `:180-212`; `user_check_client.py:234-301`; review UI |
| S7 | Make the plan-time wrist image match image 1 with a k derived from calibration (k = 2 for LIBERO, k = 0 for RoboCasa), or at least correct the false "turned" text. | `scripts/evaluate_agent_speedup_v2_user_check.py:246`; `agent_speedup_v2_client.py:1097` |
| S8 (= D9) | A "<side> of anchor" region with null bounds means beside the anchor. The lateral band is the anchor's measured half-extent. The aim is the nearest fitting free cell just past the anchor's edge, using the existing M16 clearance. It adds no new constant and no veto. | `spatial.py:208-220, 299-351` |

### Optional

| ID | Change |
|---|---|
| S9 | Diagnostics shown in the review only, never a plan issue: a cell-order cue (the region's cell vs the anchor's cell on image 1), and highlighting of world-axis wording in part texts. It would have flagged this run, but cells are coarse and it gives a false alarm on t6_006. |

### Not recommended

- **Swapping the constant.** It is only correct for LIBERO.
- **Robot-centred left/right.** It is mirrored in LIBERO.
- **Grounding regions from pointed cells.** Cell 55's centre already lies outside the goal box, and 2 of 5 recorded region plans have x = 0.
- **Per-scene numbers in the static planner instructions.** The instructions are built once in `__init__`, so this would break comparability between runs.
- **Renaming the `frame` enum now.** It would break validation of every recorded region plan.

## Open decisions

- **D9 (S8):** with null bounds, does a directional region mean "beside the anchor" or "anywhere on that side"?
  - For "beside": all 7 LIBERO side-goal boxes are aligned with the anchor and sit 3–12 cm past its edge. The official t6 pudding starts at (−0.04, 0.117), already inside the +y half-plane but outside the box. t6_009 ended at y = 0.170, past y_max = 0.15.
  - Against: only t6 uses it in LIBERO-10.
  - S1–S5 do not depend on this answer.
- **S7 timing:** ship the wrist rotation in the same frozen build as S1–S5? It changes images 3–4 of every LIBERO plan, so one sweep would then cover both changes.

## Verification plan

1. **Offline:**
   - S1 unit test on the recorded calibrations;
   - the BDDL convention harness, as a developer test only that never feeds the agent;
   - the resolver on a synthetic heightfield;
   - text snapshots: no `right_of=+x`, `WORLD axes` or `world -x/+x` remain;
   - an executive payload test with a stub transport: `region_directions.E == [0,1]`;
   - every recorded approved_plan still validates.
2. **Plan-only calls (~10, one per LIBERO-10 task):** the t6 region uses viewer words; the t4 and t5 pointing is unchanged.
3. **Live t6, official layout and relayout:** the first pudding release lands on the plate's +y side; post-hoc truth check against the goal box.
4. **Acceptance:** a full LIBERO-10 sweep plus a RoboCasa coffee smoke run, all on one frozen build.

`judged.verification_plan` in [workflow_result.json](workflow_result.json) lists all steps with their expected values.

## Evidence gathered in the main session (not in workflow_result.json)

| Item | Value | Source |
|---|---|---|
| Final pudding COM per t6 run (goal box x [0.10, 0.20], y [0.05, 0.15]) | t6_006 and t6_011 (−0.039, 0.117): never moved. t6_007 (0.145, 0.146): inside, lenient success. t6_009 (0.187, 0.170): +y side, 2 cm past y_max. Relayout (0.276, −0.020): wrong side. | each run's `physical_truth.json`, `terminal.atoms` |
| Ray-casts of the planner's pointed cells (agentview calibration, plane z = 0.44) | E cell 55 → (0.113, 0.180); C cell 61 → (0.199, 0.033); sanity checks: D cell 35 → (−0.138, −0.140) vs true (−0.16, −0.14), A cell 38 → (−0.138, 0.142) vs true (−0.12, 0.18) | `task_6/trial/approved_plan.json`, `trial/requests/000001-*.request.json` |
| Executive quotes | Request 1 belief: the mug is "to the image-right of the plate; its handle extends toward world +y". Request 11 why: "places the package in the reviewed right-of region rather than relying on image-right". | `task_6/trial/requests/000001-*`, `000011-*.response.json` |
| libero_goal stove | Stove region centre (−0.41, 0.21); stove_front_region centre (−0.05, 0.21), so front = +x | `libero_goal/push_the_plate_to_the_front_of_the_stove.bddl` |
| RoboCasa external camera | `robot0_agentview_left` image-right (0.979, −0.202), away (0.202, 0.979) | `runs/robocasa_coffee_user_check_001/trial/requests/000001-*.request.json` |
| Plan-time wrist turn | `frame_alignment_turns(view.R, None)` returns 0; `review.json` `preparation.quarter_turns` = 0 | `scripts/evaluate_agent_speedup_v2_user_check.py:246`, `src/gpt_field/agent_speedup_v2_geometry.py:1533-1534` |

## Corrections the investigators made to the brief

- The plan-time wrist image is **not** turned; only the prompt says it is.
- t6_007 was a lenient success: the pudding was still gripped at the terminal step, 4.6 mm inside y_max. Its first release was at y ≈ 0.198.
- Cell 55's centre lies about 3 cm beyond y_max; cell 54 matches the region.
- The 2026-10-26 report's line references are stale. Current lines: `spatial.py:59-60` and `:211-214`; `user_check_client.py:374-389`.
- `work/bddl_regions.py` prints table-frame centres. It ignores the study table's x offset (−0.2) and yaw. Relative directions are unaffected.
- LIBERO has two language/goal labelling errors (KITCHEN_SCENE2 "front" bowl, STUDY_SCENE3 "right compartment"). Neither contradicts the frame.

## Files in work/

Paths inside `workflow_result.json` and the scripts that start with `/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/` now live under `work/`. The JSON itself is at the top level. All files are unchanged copies; the scripts still name the scratchpad paths, so update those paths before re-running them.

| Path | What it shows |
|---|---|
| [work/judge/axes_check.py](work/judge/axes_check.py) | S1 prototype run on the recorded LIBERO and RoboCasa calibrations. |
| [work/semantics/verify_claim.py](work/semantics/verify_claim.py), [.json](work/semantics/verify_claim.json) | LIBERO convention check: 41/41 decidable checks, 0 counterexamples. |
| [work/semantics/side_geometry.json](work/semantics/side_geometry.json) | Geometry of every "<side> of <anchor>" goal box. |
| [work/semantics/t6_band.json](work/semantics/t6_band.json) | t6: the pudding-centre band inside plate_right_region that also clears the plate. It is y in [0.099, 0.150) with the plate's visual radius (0.069 m) and [0.104, 0.150) with its collision radius (0.073 m); the pudding is at its initial yaw. |
| [work/semantics/footprints.json](work/semantics/footprints.json) | Object footprints computed from the LIBERO meshes. |
| [work/design/dir_signs.txt](work/design/dir_signs.txt) | 31 anchored side regions and the sign of each direction. |
| [work/design/view_frame_check.txt](work/design/view_frame_check.txt) | Ray-casts of pointed cells and the S9 cue on the recorded plans. |
| [work/design/turns_check.txt](work/design/turns_check.txt) | Wrist turns derived from calibration (S7). |
| [work/design/minfix/viewer_axes_check.py](work/design/minfix/viewer_axes_check.py) | S4 resolver equivalence on a synthetic heightfield. |
| [work/design/minfix/bddl_convention.py](work/design/minfix/bddl_convention.py) | Prototype of the convention harness (verification step 1). |
| [work/design/minfix/pointing_agreement.py](work/design/minfix/pointing_agreement.py) | Prototype of the S9 cue. |
| [work/critique/plans_summary.txt](work/critique/plans_summary.txt) | Regions and part texts of every recorded approved plan. |
| [work/audit/](work/audit/) | The four images the t6 planner saw, plus the audit's ray-cast and wrist-turn scripts. |
| [work/bddl_regions.py](work/bddl_regions.py) | The main session's first BDDL parser. |
| [work/rotation/rotation_check.py](work/rotation/rotation_check.py), [output](work/rotation/rotation_check_output.txt) | Camera-rotation check (corrected version): yaw, elevation 8–88°, top-down and roll at the plate pixel, plus the share of visible table positions misclassified under roll. `work/finalcheck/design/rotation_check_copy.py` is the checkers' copy of the earlier, uncorrected version. |
| [work/finalcheck/](work/finalcheck/) | Scripts from the draft-plan checks: roll over the whole table, local vs global directions, projected cells and the double-flip check, approval-hash recomputation, request sizes. |

Line numbers refer to the source as of 2026-10-28; the relayout run's frozen source equals the current `src` for the spatial module.
