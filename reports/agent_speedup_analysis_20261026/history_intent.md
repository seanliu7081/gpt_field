# Agent SpeedUp: history and user intent (v1 → base v2 → user-check uc1–uc7 → v3 r1–r11 → 2026-10-26 campaigns)

Read-only reconstruction, 2026-10-26. Sources: the design docs under `/workspace/gpt_field/`, the user's memory notes under `/root/.claude/projects/-workspace/memory/`, and the run artifacts under `/workspace/gpt_field/runs/` and `/workspace/gpt_field/results/`. Every number comes from a file cited inline. Inferences are labelled **[inference]** and unknowns **[unknown]**.

Conventions:
- uc*N* means `agent-speedup-v2-user-check-N`; r*N* means `agent-speedup-v3-N`. A run number is not a revision: `t1_011` is r10 and `t1_012` is r11.
- **raw** means LIBERO success. **strict** means `v2_success`, which is raw AND the strict physical audit.
- **svc** means full service time:
  - v3 r6+: `performance.json.completion_target.total_service_elapsed_s`.
  - Other runs: ≈ `user_check.preparation_elapsed_s + elapsed_s` from `summary.json`. Where supervisord logs exist, the process wall time minus review wait is 4.7–5.9 s above this sum (t2_001 503.1 vs 497.2; t3_002 1357.2 vs 1351.6; t5_001 274.9 vs 270.2; t6_007 1387.7 vs 1382.9).
- **ctrl** means `control_elapsed_s`; **ctrl_nv** means `control_without_video_s`.
- I extracted every run's fields with a stdlib script into `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/runs_extract.json`. The script is `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/scripts/extract.py`.

---

## 0. Headline findings

1. **uc7 beats v3 on speed and on generality on every comparable measure.**
   - Same t1 initial state (both have init SHA `e92ca405…`; `runs/*/initial_observation_check.json`):
     - uc7 `t1_001` succeeded with 13 calls, 416.3 s elapsed and ≈437.7 s svc.
     - The best v3 success, `t1_008`, used 40 calls, 1,601.9 s elapsed and 1,626.2 s svc.
   - No v3 run has succeeded since r6: 0 of 7 runs (t1_009, t1_010, t7_001, g5_001, g5_002, t1_011, t1_012), plus 0 of 3 in today's v3 r6 baseline.
   - v3 never succeeded on any task except t1: 0 of 3 runs on t7 and g5.
   - uc7 ran on all 10 LIBERO-10 tasks: 11 of 13 runs reached raw success, and the manual sweep reached 9 of 10.

2. **Exactly one LIBERO-10 run in the whole line has raw success with service time ≤ 300 s: uc7 `t5_001`.** The history doc never mentions it.
   - Run facts: auto-approved, 399 steps, 7 calls, 241.7 s elapsed plus 28.4 s preparation.
   - The supervisord process lived 274.9 s, from spawn at 17:02:59.325 to exit at 17:07:34.182 (`runs/agent_speedup_v2_user_check_libero_10_t5_001_launch/supervisord.log`). Review wait was 0.09 s.
   - The success depends on the runner stopping at the first raw success while the book was still held.
     - `physical_truth.raw_libero_success_after_settle = false`.
     - After a diagnostic release, the book sits physically inside the compartment (all strict checks true). But LIBERO's origin-only test fails: the origin is 3.48 mm below the region box (`runs/agent_speedup_v2_user_check_libero_10_t5_001/physical_truth.json`, `margin_to_lb_mm[2] = -3.48`).

3. **v3's dominant failure mode since r4.1 is `evidence_acquisition_failed`. It covers 8 of the 9 v3 failures from r4.1 on, plus today's baseline attempt 3.** The ninth, r5 `t1_006`, was a local `float(None)` bug, so the orchestrator's "every failure from r4.1" is off by one. The stop reasons, from each run's `error_traceback.txt`:
   - 6 of 8 are `model_escalation`. The model gave up because the runtime demanded hand, palm, finger or payload sweep coverage or clearance evidence that it could not establish from RGB (t1_005, t1_009, t1_010, t7_001, g5_001, g5_002).
   - 2 of 8 are view-planner preconditions:
     - t1_011: `targeted_view_unavailable: no_spatial_coverage_witness`.
     - t1_012: `targeted_view_unavailable: semantic_goal_requires_geometry_before_view_planning`.

4. **Output tokens, not input tokens, drive per-call latency, and v3's protocol inflated output.**
   - Across 29 agent-call runs, mean seconds per call correlate with output tokens per call at r = 0.97 and with input tokens per call at r = −0.17.
   - OLS fit: s/call ≈ 1.70 + 0.0222·(input kTok) + 0.0340·(output Tok), R² = 0.96 (my computation from `summary.json` budgets).
   - uc7 t1: 14.0 s/call and 368 output tokens/call. v3 r6–r11 runs: 27.0–51.2 s/call and 721–1,367 output tokens/call.
   - The first v3 execution call (program plus scene_update) carried 2,576–3,588 output tokens and took 96–120 s. uc7's first call took 17.9 s.

5. **Process drift against the user's own rules.**
   - On 2026-10-21 the user ruled that work be organized by mechanisms that recur on two or more tasks, that acceptance be an all-task sweep, and that veto checks be removed rather than thresholds added.
   - Since then:
     - v3 ran 12 of its 15 runs on t1 episode 0; the first 10 were consecutive.
     - Every revision after r1 responded to the previous single run.
     - Gate-like code grew monotonically, for example `_pause(` 5 → 22 and `raise ValueError` 28 → 118.

6. **Plan-preparation requests that stream but never complete recur in both lines.** This is the dominant infrastructure risk for a 300 s target; one stall alone exceeds it. There are at least 7 occurrences plus 2 of 4 diagnostic replays (§4.4).

7. **The general plan's user decisions D1–D5 were never ruled on in any later artifact.**
   - D1 (honest end) was not adopted in either mainline.
   - D4 (autonomous headline via `--auto-approve`) was reversed in v3.
   - Service-time accounting for automated (Codex) review differs between campaigns (§3).

---

## 1. Timeline: design change, then measured outcome

### 1.1 v1 and base v2

| Version (date) | Design change | Measured outcome |
|---|---|---|
| **v1** (2026-10-16/17), prompts 1–5 | The model picks among 6 macro skills (`reach`, `grasp_with`, `pull`, `transport`, `place_release`, `retreat`) and 2 grasp families (rim pinch, frontal bar pinch). Contact, closing and verification run locally (`agent_SpeedUp_implementation_history.md:64-66`). It is task-specific by construction: explicit bowl and drawer targets plus fixed distances (`:76`). | 5 runs, 2 raw successes, both on non-LIBERO-10 tasks (spatial 0, goal 3). bowl_002: 449 steps, 11 calls, 193.1 s elapsed. drawer_003: 967 steps, 21 calls, 349.0 s; the bowl was perched about 56 mm above the drawer floor (`:88`). Failures used 61–108 calls and were stopped by the user (`runs/agent_speedup_v1_*/trial/summary.json`). |
| **base v2** (2026-10-18 to 10-20): eval3 → eval3post → fast | Relation-constraint architecture. The model names objects, parts and relations; `StepRunner`, `SceneModel`, held frame, planner, primitives and observers execute them, with one threshold table (thresholds v7, then v8). | 7 live runs, 0 raw successes. eval3 t2/t6/t8 were stopped by the operator at 444–548 steps. eval3post 0/3 at the 4,000-step cap. fast t2: 4,000 steps and 44 calls, 1,214.5 s elapsed. Root cause was the O-1 5° false refusal of a good handle hold; a fork with the limit raised to 20° succeeded at step 1679 with 3 calls (`agent_SpeedUp_v2_t2_failure_analysis.md:17-26`). |

### 1.2 user-check uc1–uc7 (all on 2026-10-20 18:14 to 10-21 20:11)

| Version | Design change (per `agent_SpeedUp_implementation_history.md:140-148`, spot-checked in frozen sources) | Measured outcome (`runs/agent_speedup_v2_user_check_libero_10_*/summary.json`) |
|---|---|---|
| uc1 | Browser review before base v2. Semantics are locked; any strategy change ends with `plan_conflict`. | t2_001 raw and strict success: 1,886 steps, 6 calls, ≈497 s svc. t6_001 `plan_conflict` at 1,771 steps and 15 calls. |
| uc2 | Approval narrowed to `initial_plan_only` (the string appears first in t6_002's frozen client). The lowering-stall rule was relaxed. | t6_002 stopped by the user: 5,621 steps, 45 calls, 1,625.6 s elapsed. |
| uc3 | Four contracts (task, shared state, skill, execution feedback), persistent IDs and failure memory. | t6_004 stopped by the user: 3,187 steps, 21 calls, 869.3 s. |
| uc4 | `local_motion` short world-frame corrections, each target ≤ 5 cm / 0.35 rad from the measured pose (frozen client:122). | t6_005: 4,078 steps, 37 calls, 1,446.2 s. Labelled `infrastructure_error`, but the real cause is the client-side `ValueError: request text exceeds 60000 characters` (`runs/…t6_005/error_traceback.txt`; `runs/agent_speedup_v2_user_check_libero_10_t6_005/frozen_source/agent_speedup_v2_client.py:136,541`). |
| uc5 | Regions, placement goals and scoped constraints. Explicit `verify_grasp`/`resume_macro` hand-offs, which add handoff preconditions. | t6_006 stopped by the user: 1,076 steps, 38 calls, 625.2 s. |
| uc6 | The model controls continuously; macros become optional accelerators. **The uniform "verify-first" gate was removed.** | Both raw successes, on the first t6 success after 7 failed v2-family t6 runs. t6_007: 464 steps, 84 calls, ctrl 1,338.6 s, ≈1,383 s svc. t3_002: 419 steps, 87 calls, ctrl 1,316.3 s. Model time was 92–93% of ctrl. |
| uc7 (frozen 2026-10-21) | `execute` fragments: 1–6 segments of 5 generic skills (`move_to_pose`, `move_relative`, `approach`, `set_gripper`, `wait`), ≤ 120 steps each and ≤ 240 per execute. Measured closed-loop tracking ends on arrival, stable gap, stall or budget (`src/gpt_field/agent_speedup_v2_user_check_fragments.py:22-34,300-320`). | 13 runs: raw 11/13, strict 3/13 (t2_002, t8_001, t9_001), S300 1/13 (t5_001). The manual 10-task sweep was raw 9/10; t6 failed (t6_011). Auto-approved: t3_003 succeeded (22 calls, ≈650 s), t5_001 succeeded (≈270 s), t6_009 failed. For raw successes the median ≈svc is 437.7 s (range 270.2–650.3). Native macro actions were 0 in all 13 runs (`agent_stats.json`). |

### 1.3 v3 r1–r11 (2026-10-23 19:22 to 10-25 19:09), all `libero_10_t1` episode 0 unless noted

| Revision | Design change (`agent_SpeedUp_v3.md`) | Measured outcome |
|---|---|---|
| r1 | `GeometryMemory` of model OBB hypotheses. Sweep-prefix checks on palm, fingers and payload. Unknown moves limited to 3 mm × 4 steps or 5 mm × 6 steps; 24-step observation cap; 80-step age expiry (`history:245-247`). | t1_001 raw success: 772 steps, **79** calls, 1,739.2 s elapsed. Preparation took 925.0 s because the first plan stream failed with `stream_missing_completion` after 900.8 s (`runs/agent_speedup_v3_t1_review_001/requests/000001-…result.json`). |
| r2 | Removed the fixed 4/6/24-step handoffs on supported paths; validity counts only relevant evidence. | t1_002 raw success: 1,568 steps, 41 calls, 1,285.1 s elapsed. |
| r3 | Event-driven validity (no 80-step expiry), RGB change watch, adaptive speed, **final-interaction target/support/neighbor inventory guard**. | t1_003 HTTP 500 on request 43 after 1,531 steps (1,422.7 s). |
| r4 | Semantic dependencies separated from pose dependencies. Bounded candidate search. `geometry_patch`/`continue_intent`; `missing_evidence_requests` lead to evidence acquisition or a stop. | t1_004 HTTP 503 at step 4 (48.8 s). |
| r4.1 | Bounded retry for subscription 503s. | t1_005 `evidence_acquisition_failed` (model escalation): 210 steps, 25 calls, 579.7 s. Preparation took 1,682.1 s (a stream failure at ~900.5 s, then 771.7 s producing 151,950 whitespace characters). |
| r5 / prompt 5.1 | Tracking and coupled-command checks, `acquire_evidence` view moves, 180 s request deadline, whitespace guard, compact JSON. | t1_006 crashed locally on `float(None)` at action 90. t1_007 raw success: 880 steps, 52 calls, 1,877.1 s; 21 of the calls were `geometry_patch`, costing 465.9 s (`runs/agent_speedup_v3_t1_007/time_cost_details.json`). |
| r6 | Semantic `program` (≤ 24 stages), two-view interaction monitor, I/O cuts (video, state files). | t1_008 raw success, **the last v3 success**: 873 steps, 40 calls, svc 1,626.2 s. The program ran only steps 0–148. 21 `geometry_patch` calls took 568.4 s (`time_component_analysis.json`). |
| r7 | `OperationRuntime`, local RGB feature references, recovery budgets (2 per object, 4 goal updates per stage; `agent_speedup_v3_program.py:24-25`). | t1_009 failed: 743 steps, 25 calls, svc 936.0 s. Release provenance was erased. |
| r8 | Provenance fixes; `program.operations` with 1–3 `pick_place` operations of 8 stages each (`program.py:93,149-158`). | t1_010 failed (190 steps, 7 calls, 535.9 s). t7_001 failed (176, 8, 463.2 s). g5_001 failed (21, 5, 201.2 s). |
| r9 | Closing an empty hand is not a grasp unless grasp intent is declared. | g5_002 failed (71 steps, 6 calls, 242.4 s); the false payload was gone. |
| r10 | `LocalGeometryMemory`: a missing free-space box becomes an advisory, not a veto (`local_geometry.py:201`). Context compaction cut request characters by 57.1%. | t1_011 failed: 194 steps, 3 calls, 214.8 s, gripper never closed. Descent stalled, then `no_spatial_coverage_witness`. |
| r11 (current source; byte-identical to t1_012's `frozen_runtime`) | Tracking failures feed view requests; the observation focus no longer needs a coverage witness. | t1_012 failed: 431 steps, 9 calls, 469.4 s. 12 close actions, 0 accepted hold. Stopped on `semantic_goal_requires_geometry_before_view_planning` (`agent_speedup_v3.py:264-267`; `program.py:249` `fresh_geometry:`). |

### 1.4 Campaigns on 2026-10-26 (driven by Codex)

| Campaign | Design | Measured outcome |
|---|---|---|
| `results/speedup_evidence_20261026T055123Z` (Codex plan P0–P6) | A = uc7 baseline; B = uc7 + compact video and throttled state snapshots. 900 s budget. **Automated Codex review time is billed to service time.** | t1 ep1: A 648.7 s (830 steps, 11 calls, elapsed 475.8 s, review 138.1 s); B 614.0 s (1,019 steps, 13 calls, elapsed 446.5 s, review 138.5 s). Both raw success, strict false, S300 false. t1 ep2: both preparations hit the 900 s cutoff with no completed response. Stopped; P5 inconclusive, P6 not started (`final_report_zh.md`, `paired_results.csv`, `evaluations/trial_00{2,3}/trial/summary.json`). |
| `runs/speedup_next_20261026T101438Z` | Replayed 4 saved preparation requests with stream diagnostics and a 120 s cutoff. The bounded local candidate was `bounded_visual_placement_continuation`. | Both inputs that had previously *succeeded* stalled. They streamed `output_text` deltas to the cutoff (141,110 and 27,853 characters) without `response.completed`. Both previously timed-out inputs completed (17.3 s, 11.2 s; 922 and 924 characters). The candidate stopped before implementation for lack of validated object feedback, and online robot experiments were paused (`decision.md`). |
| `runs/v3_completion_baseline_20261026T133633Z` | Isolated r6 copy plus an autonomous finish endpoint, so raw success no longer truncates the episode. | t1 ep0/1/2: 0 of 3. Attempt 1: a newline broke serialization (140.2 s). Attempt 2: missing initial SHA (0.75 s). Attempt 3: escalated for lack of evidence after 313 actions (188 program + 125 fragment), svc 1,007.4 s; automated review 284.8 s; model 84.5% of ctrl; 7 of 13 responses were `geometry_patch` (`report_zh.md`, `summary.json`). |

---

## 2. The user's explicit goals, constraints and rejected directions

Quotes are verbatim where marked "verbatim". Codex's `authorization_note` fields are Codex's paraphrases of user instructions and may contain embedded quotes.

### Goals
- **G1 (2026-10-18, verbatim):** "Using the same robot arm and RGB image and robot-state inputs, there is no longer a need to modify control code or thresholds for each individual manipulation task when changing objects, layouts, or task combinations." (`memory/agent-speedup-v2-plan.md:17`)
- **G2 (2026-10-20, verbatim):** "we should not write skills or harness specific for a task, we need a general framework for the agent to control the robot. if we need each task a skill this is totally out of the scope." The user then said "yes adopt that rule and order". (`memory/v2-scope-general-framework.md:11`)
- **G3 (2026-10-21, verbatim):** "we should not make our framework be specific for a task, we want a general framework for the agent to speed up and finish tasks." (`memory/v2-scope-general-framework.md:29`)
- **G4 (2026-10-25, Codex paraphrase):** "User authorized the generic five-minute optimization plan…". The 300 s target "includes preparation and complete execution/finalization excluding approval wait; it is not a stop condition" (`runs/agent_speedup_v3_t1_review_008/review.json`; `runs/agent_speedup_v3_t1_009_review/review.json`).
  - The time target evolved:
    1. The v2 spec set per-task targets of ≤ 160–260 s `control_without_video_s` (`agent_SpeedUp_v2.md:88-89`).
    2. The general plan said "≤ 600 s first" (`agent_SpeedUp_v2_general_plan.md:242`).
    3. Since r6 the target is 300 s of full service (`history:41`).
- **G5 (2026-10-26, the user's prompt to Codex, verbatim):** "判断“一个可靠的局部控制机制能否替代昂贵模型交接，并带来净提速”。本轮不扩展完整 v3，不恢复原 P5/P6 大矩阵。" ("Determine whether a reliable local control mechanism can replace expensive model hand-offs and give a net speedup. This round does not extend full v3 and does not resume the original P5/P6 matrix.") (`runs/speedup_next_20261026T101438Z/objective.md:3-6`)
- **G6 (2026-10-26, verbatim):** "User: implement and test it then give me results". The accepted proposal was: "isolated v3 r6 completion baseline; autonomous finishing; known t1 then a small preregistered set of other t1 initial states; preserve 300-second score" (`runs/v3_completion_baseline_20261026T133633Z/user_request.txt`).
- **G7 (the current request, relayed):** show the framework of the *workable or latest* version of user-check and v3; analyse the bottleneck; give a next-step plan. **[inference]** "Workable" points at uc7 and "latest" at v3 r11.

### Constraints
- **C1. Generality rules** (`memory/v2-scope-general-framework.md:13-21`):
  - One frozen code version and one threshold table for every task. "No task key reaches the agent" (the runs record `task_key_reaches_policy: false`, e.g. `runs/…t1_001/manifest.json`).
  - No change may be justified by a single task.
  - Failures are framework gaps.
  - The headline is how many LIBERO-10 tasks one build completes.
- **C2. How to organize work (2026-10-21)** (`v2-scope-general-framework.md:30-32`):
  - "Organise improvement plans by mechanisms that recur on 2 or more tasks, or by structural capability gaps. Never organise them by one run's timeline."
  - "Acceptance is an all-task sweep on one frozen build."
  - "Prefer removing veto checks over adding thresholds."
- **C3. Live-first (verbatim):** "i never ask to have zero model calls for the v2" and "yes model calls" (`memory/v2-live-first-model-calls-ok.md:11`).
- **C4. Fewer checks (verbatim):** "can i have a runable version? we did too much checks and evaluation" (2026-10-20, `agent-speedup-v2-plan.md:115`). "implement this and automatically start the test on t3 and t6; monitor full tasks; avoid excessive checks or smoke tests" (2026-10-21 13:42, `runs/agent_speedup_v2_user_check_v7_t3_t6_001_launch/authorization.json`).
- **C5. Inputs:** RGB, robot state, camera calibration and robot geometry only. No simulator object poses, depth, contact truth or task predicates (`agent_SpeedUp_v2_user_check.md:65`; `agent_SpeedUp_v3.md:218,465`). Model: `gpt-6-astra`, high effort, ChatGPT subscription, no API-key fallback (`agent_SpeedUp_v3.md:28`).
- **C6. Success and timing:** independent raw LIBERO success decides success. 300 s is a reporting target, not an episode cutoff (`agent_SpeedUp_v3.md:30`).
- **C7. Review:** initial-plan review is retained. The user delegated review explicitly and per run ("implement this fix and retry the t1, just approve the plan yourself", `runs/agent_speedup_v3_t1_review_002/review.json`; "just auto approve the plan", `runs/…t3_002_launch/operator_approval_authorization.json`). For the 10-task retest the user wanted manual approval: "retest this 10 tasks, we need the user approve now for the plan" (`runs/agent_speedup_v2_user_check_libero_10_manual_batch_001/queue_manifest.json`).
- **C8 (2026-10-26 to Codex, verbatim):** "禁止 stall→close、测试专用坐标常数和评估真值驱动的触发器" (no stall→close, no test-specific coordinate constants, no truth-driven triggers). "不要为了完成计划强行实现三角化、新几何层或更复杂状态机" (do not force triangulation, a new geometry layer or a more complex state machine just to finish the plan). "负结果是有效结果" (negative results are valid results). (`objective.md`)
- **C9 (verbatim):** "Allow early stop": an agent-side terminal stop is approved, with LIBERO counting unchanged (`agent-speedup-v2-plan.md:113`). "when hit the session limit, pause then after reset, keep run" (`memory/usage-limit-pause-resume.md:11`).

### Rejected directions
- **X1.** The F1–F11 t2 plan (`agent_SpeedUp_v2_t2_failure_analysis.md`) was rejected as task-specific even though no code named a task (`v2-scope-general-framework.md:29`; `agent_SpeedUp_v2_general_plan.md:285-292`).
- **X2.** A zero-model-call scripted gate before live runs (Claude's own addition) was rejected (`v2-live-first-model-calls-ok.md:11-15`).
- **X3.** Per-task skills or harnesses (G2), and v1's hand-designed `rim_pinch`/`frontal_bar_pinch` (`agent-speedup-v2-plan.md:21`).
- **X4.** Excessive checks, evaluations and smoke tests (C4).
- **X5 (today).** Extending full v3, resuming the large P5/P6 matrix, and adding triangulation, new geometry layers or more complex state machines (G5, C8). **[inference]** This is the clearest signal that the user is now sceptical of the v3 direction.

### Tensions to flag
The Codex plan's author is not recorded; Codex executed it today. It deviates from C1/C2 in two places:
- "任务标签可以用于正常的语义条件或技能选择" (task labels may be used for semantic conditioning or skill selection) (`Agent_SpeedUp_Codex_Implementation_Plan_ZH.md:25`). This conflicts with C1's "no task key reaches the agent".
- "第一轮只声明对选定操作族的复用" (the first round claims reuse only within a selected operation family) (`:23`). This narrows C2's all-task acceptance.

---

## 3. Pending user decisions and whether later work resolved them

The general plan (2026-10-21 ~12:45 UTC) asked the user for D1, D4 and D5(a) before M0 (`agent_SpeedUp_v2_general_plan.md:296-317`). `grep` finds no later doc that mentions D1–D5, M0, WP0–WP9 or "sweep 1" (`agent_SpeedUp_v2_user_check.md`, `agent_SpeedUp_v3.md`, the history doc, the Codex plan). **No ruling is recorded.**

| Decision | De facto status (evidence) |
|---|---|
| **D1(a): the episode ends where the agent ends it** | **Not adopted in either mainline.** The base runner still requires `stop_on_success` ("LIBERO success must stop the episode", `scripts/evaluate_agent_speedup_v2.py:205-206`) and breaks at the first success (`:1514-1515`). uc7's prompt says `finish` "never ends the episode or sets success. Only the independent evaluator terminates a successful task. … There is no give_up action." (`runs/…t1_001/frozen_source/agent_speedup_v2_user_check_agent_client.py:173-177`). Autonomous-finish endpoints exist only in today's isolated copies (`speedup_next_endpoint.py`, `v3_completion_endpoint.py`). Consequences: all 18 historical raw successes ended with `physical_success_at_terminal=false`, and uc7 t6_009 and t6_011 ran until the user stopped them. |
| **D1(b): end after N turns with no progress** | Not implemented: no `blocked_no_progress` in the uc or v3 source. v3 has evidence stops instead (`agent_speedup_v3.py:387` repeated blocked action, `:500` invalid continuation). |
| **D2: ratify model-emitted numbers, model camera calibration, high effort** | Ratified in practice. uc7 and v3 take world-frame targets (`agent_SpeedUp_v2_user_check.md:53`); camera calibration is a model input (`:49`; `agent_SpeedUp_v3.md:218`); effort stayed high. No explicit ruling is recorded. |
| **D3: contact policy** | Record-only in all 36 uc and v3 summaries (`collision.collision_policy`). v3 added intent-scoped `allowed_contact_ids` (`program.py:149-158`). Not prevented: uc7 t0_001 had 19,799 forbidden points, t3_005 15,196, t9_001 7,355. |
| **D4: autonomous headline via `--auto-approve`** | **Contradicted.** uc7 gained `--auto-approve` (`scripts/evaluate_agent_speedup_v2_user_check.py:600`) and used it for t3_003, t5_001 and t6_009. The 10-task sweep used manual review at the user's request (C7). v3 rejects auto-approve: "agent speedup v3 requires manual initial-plan approval" (`scripts/evaluate_agent_speedup_v3.py:234-238`). **Accounting is inconsistent:** v3's `performance.json` *excludes* approval wait, including Codex review (`runs/agent_speedup_v3_t1_008/performance.json.completion_target.excludes`), but today's campaigns *bill* automated review to service time (138.1 s, 138.5 s and 284.8 s). |
| **D5: Calibrate+ robot facts** | Moot for the fragment path: "新的动作片段从当前状态直接接手，不要求旧宏的全场景校准/注册" (new fragments take over from the current state and do not need the old macros' full-scene calibration/registration) (`agent_SpeedUp_v2_user_check.md:40`). Native macros ran 0 actions in all 28 uc7 and v3 runs. |
| **M0 items** | **WP1 (ledger and termination):** not implemented (uc7 `ledger.json` is `[]`). **WP6 (≤ 25k input tokens per request):** not met in uc7 (24.7k–82.2k server-reported input tokens per call, 41.6k on t1); v3 r10–r11 reached 15.7k–27.7k. **WP8 (`--no-video`):** not adopted (`video_enabled=true` in all 36 runs). **Sweep 1:** happened as a *manual-review, video-on* uc7 batch, not the autonomous honest-end sweep the plan specified. |

**Other open items:**
1. Whether to merge the I/O branch (`narrow_scope`), which the evidence campaign kept but did not merge into the main execution path.
2. Whether to resume online experiments, which speedup_next paused until the preparation-stream non-completion is explained.
3. Whether to fund a narrowed v3 development budget aimed at "首次抓取的合法视觉证据→动作修正→确认抓住" (legal visual evidence for the first grasp → action correction → confirmed hold) (`v3_completion_baseline…/report_zh.md`).
4. v3 stage 6 (asynchronous pre-planning) is deferred (`history:353`).
5. The held-out t1/t5/t7 episodes 1–6 are untouched (`heldout_results.csv`).
6. The automated-review cost rule (D4 above).

---

## 4. Recurring patterns, with evidence

### 4.1 Do fixes keep adding gates?

**v3: yes, mostly.** I classified the documented changes per revision **[inference]**:
- New checks or stop paths appear in r1, r3, r4, r5, r6, r7, r8 and r10:
  - r1: sweep limits.
  - r3: inventory guard and RGB-watch invalidation.
  - r4: evidence stop.
  - r5: coupled-command check and output guard.
  - r6: co-motion attachment gate.
  - r7: recovery budgets.
  - r8: next-object close stop, release-before-next-operation, attachment≠release.
  - r10: core-conflict correction and grasp intersection constraint.
- The relaxations undid earlier *v3* gates:
  - r2 removed r1's 4/6/24-step handoffs.
  - r3 removed r1's 80-step expiry.
  - r10 made the r1/r3 coverage veto advisory.
  - r11 dropped r5's coverage witness for viewing.

A crude, reproducible proxy, string counts over `frozen_runtime/**/agent_speedup_v3*.py`:

| Token | r1 | r4 | r6 | r7 | r10 / r11 |
|---|---|---|---|---|---|
| `_pause(` | 5 | 5 | 14 | 21 | 22 / 22 |
| `_stop_for_evidence(` | 0 | 4 | 5 | 5 | 6 / 6 |
| `reject` | 0 | 21 | 32 | 47 | 52 / 52 |
| `missing_evidence` | 0 | 19 | 40 | 42 | 48 / 48 |
| `raise ValueError` | 28 | 42 | 68 | 102 | 118 / 118 |

v3 code grew from 69.2 KB to 478.3 KB (6.9×) across 6 → 16 files, on top of 373 KB of uc7 code.

**User-check: the turning point was a gate removal.** uc6 "取消“先走旧验证交接流程才有资格继续”的统一门槛" (dropped the uniform rule that the old verify-and-handoff flow had to run before continuing) (`history:147`). That produced the first t6 and t3 completions; uc7 then added a capability rather than a gate. This matches C2's "prefer removing veto checks".

### 4.2 Evaluation concentrated on one task and one episode

- **All 36 uc and v3 runs before today used episode 0, seed 7** (extraction).
- v3: 12 of 15 runs on t1 ep0. The first 10 (t1_001 → t1_010) were consecutive before any other task was tried. 1 run on t7, 2 on g5.
- The v3 t2 and t3 reviews prepared on 2026-10-23 18:32 were never approved (`runs/agent_speedup_v3_t{2,3}_review_001/review.json`: `approved=None`).
- uc1–uc6: 6 of 8 runs on t6, each revision "tried on t6 first" (`agent_SpeedUp_v2_general_plan.md:254`).
- uc7 is the exception: 13 runs over all 10 tasks.
- In the v1 era the user's stated order was "measure 3-5 inits per task once both tasks have succeeded live" (2026-10-17, `memory/agent-speedup-v1-implementation.md:57`). Before today no LIBERO-10 task had more than one initial state tested.
- v3 contradicts C2 as follows. Each revision after r1 was triggered by the previous single run:
  - "Revision 5 repairs the t1_005 failure mechanisms" (`agent_SpeedUp_v3.md:103`).
  - r4.1 followed "the user's request to fix the t1_004 HTTP 503" (`:240`).
  - r9 says "only g5 is retested for this causal fix" (`:68`).
  - r11 followed t1_011 (`:14`).

### 4.3 Live runs per revision

| Line | Live runs per revision |
|---|---|
| uc | uc1 2; uc2–uc5 1 each; uc6 2; uc7 13 |
| v3 | 1 for each revision except r5 (2) and r8 (3); 15 runs over 12 builds in ~48 h (2026-10-23 19:22 → 10-25 19:09) |

Each v3 revision was backed by large *offline* suites and **no simulator smoke run**: 52 (r2), 91 + 69 (r3), 121 (r4), 48 (r4.1), 177 (r5), 207 (r6), 238 (r7), 234 (r8), 50 (r9), 10 + 2 (r11) (`agent_SpeedUp_v3.md:195,189,228,246,111,80,66,68,46,14`). Every behavioural conclusion therefore rests on n = 1.

### 4.4 Infrastructure versus mechanism

**Execution-stage terminal failures: 18 runs.**

| Category | Runs |
|---|---|
| Remote service | 2: v3 t1_003 HTTP 500, t1_004 HTTP 503 |
| Local code or protocol defect | 2: uc4 t6_005 request > 60,000 characters (mislabelled infrastructure); v3 t1_006 `float(None)` |
| Approval boundary | 1: uc1 t6_001 `plan_conflict` |
| Mechanism | 13: 5 uc user stops (t6_002, 004, 006, 009, 011) and 8 v3 `evidence_acquisition_failed` |

Today adds 3 harness defects (credential directory, newline serialization, missing initial SHA) and 1 more v3 evidence stop.

**Preparation stalls are recurring infrastructure.** Each one of these alone exceeds 300 s:

| Line | Run | What happened |
|---|---|---|
| uc7 | t6_008 | Plan request interrupted by the user after 634.1 s |
| uc7 | t3_004 | Interrupted after 298.1 s ("restart it") |
| uc7 | t6_010 | Interrupted after 297.3 s |
| v3 | t1_001 | `stream_missing_completion` after 900.8 s |
| v3 | t1_005 | ~900.5 s stream failure, then a 771.7 s request with 151,950 whitespace characters |
| today | trial_004, trial_005 | 900 s each |
| today | speedup_next replays | 2 of 4 hit the 120 s cutoff while still streaming |

Sources: `runs/…_{t6_008,t3_004,t6_010}_review/requests/*.result.json` (`KeyboardInterrupt`), `runs/agent_speedup_v2_user_check_libero_10_manual_batch_001/t0{3,6}_restart.json`.

That is 3 of 16 uc7 preparations. **[inference]** The stalled streams emit 30–150× the characters of a normal ~920-character plan, which is consistent with runaway output generation; only t1_005's content was retained.

### 4.5 Speed across versions

Same t1 ep0 initial state:

| Run | Calls | s/call | 1st call (s) | Output tok/call | Model % of ctrl | Elapsed (s) | svc (s) | Result |
|---|---|---|---|---|---|---|---|---|
| uc7 t1_001 | 13 | 14.0 | 17.9 | 368 | 44.8 | 416.3 | ≈437.7 | raw ✓ |
| v3 r1 t1_001 | 79 | 19.1 | — | 370 | 87.5 | 1,739.2 | ≈2,664.2 | raw ✓ |
| v3 r2 t1_002 | 41 | 19.1 | — | 505 | 61.5 | 1,285.1 | ≈1,310.4 | raw ✓ |
| v3 r5 t1_007 | 52 | 26.9 | — | 585 | 74.9 | 1,877.1 | ≈1,901.9 | raw ✓ |
| v3 r6 t1_008 | 40 | 33.4 | 119.6 | 728 | 83.9 | 1,601.9 | 1,626.2 | raw ✓ |
| v3 r8 t1_010 | 7 | 51.2 | 96.0 | 1,247 | 71.9 | 505.8 | 535.9 | fail |
| v3 r10 t1_011 | 3 | 45.3 | 116.3 | 1,367 | 78.8 | 181.1 | 214.8 | fail |
| v3 r11 t1_012 | 9 | 39.4 | 106.8 | 1,052 | 81.4 | 443.2 | 469.4 | fail |

Sources: `summary.json`, and per-request latencies from `runner.log` / `run.log`.

- Within uc, model time per task fell sharply from uc6 to uc7: t3 went from 1,226.6 s to 356.2 s (87 → 22 calls).
- From uc7 to v3, every success was 3.1–4.5× slower in elapsed time (1,285.1–1,877.1 s vs 416.3 s). Shorter failed runs are not speedups.
- What remains of uc7's own time is model plus simulation. For t1_001, model 182.5 s + sim 108.4 s + preparation 21.4 s = 312.3 s, already above 300 s before video (81.8 s) and state files (25.1 s). The simulator costs about 0.10–0.13 s per step (108.4/897, 41.1/399, 110.1/873).

### 4.6 Generality across versions

| Line | Distinct tasks run | Raw success | Strict |
|---|---|---|---|
| uc7 | 10 | 11/13 runs; 9/10 tasks in the sweep (t6 0/2) | 3/13 |
| v3 | 3 | 4/15 runs, all on t1 ep0 (the last at r6); 0/3 off t1; 0/7 since r6, and 0/10 including today's 3 attempts | 0/4 |
| uc6 | 2 | 2/2 | 0/2 |

- uc7's sweep plans were human-reviewed, and 2 of 10 were edited:
  - t2_002: step 0 grip `[] → ["A"]`.
  - t9_001: step 1 grip `["E"] → ["E","D"]` and sense `close → any`.
  - Source: `review.json` `draft_plan` vs `plan`.
- uc7 was also tried outside LIBERO: RoboCasa CoffeePressButton ended in `infrastructure_error` (the arm-only guard tripped on base movement) after 244 steps (`runs/robocasa_coffee_user_check_001/analysis/terminal_audit.json`).

### 4.7 Other recurring facts

- **Macros are dead weight on the executed path:** `native_macro_actions = 0` in all 13 uc7 runs and all 15 v3 runs (`agent_stats.json`). uc6 used one macro in 171 turns (`general_plan:25`).
- **Raw success is always truncated before release:** `physical_success_at_terminal=false` in all 18 raw successes. 17 of 18 stay raw-successful after a diagnostic release and settle; the exception is t5_001. Strict-after-settle fails for t0_001 and t5_002.
- **v3's model calls feed its own gates:** `geometry_patch` was 21 of 52 calls in r5, 21 of 40 in r6 (568.4 s), 8 of 25 in r7, 4 of 9 in r11, and 7 of 13 in today's r6 attempt. uc7 t1_001 used 13 of 13 calls on `execute`.

---

## 5. What the history doc concludes in §11–13, and whether the evidence supports it

| History claim | Status |
|---|---|
| §11.2 limits: 5 fragment skills; 1–6 segments, ≤ 120/240 steps; ≤ 24 program stages or 1–3 × 8 pick_place; recovery 2 per object and 4 per stage; 30 cm/1.2 rad and 8 cm/0.5 rad; contact step adaptive from 5 to 20 mm; no episode cap; record-only contacts | **Supported by the code.** `agent_speedup_v2_user_check_fragments.py:22-34` (inherited by v3 at `agent_speedup_v3_fragments.py:15-17`); `agent_speedup_v3_program.py:24-25,93,149-158,180`; `agent_speedup_v3_fragments.py:325-360`; `scripts/evaluate_agent_speedup_v3.py:234-238`; `collision_policy` in the summaries. |
| §10: "主瓶颈仍是模型往返" (the main bottleneck is still model round-trips) | **Supported for v3** (model 72–95% of ctrl). **Only partly true for uc7:** model is 44–64% of ctrl, and sim + video + state files take 36–56% (t1_001: 215.3 s non-model). |
| §13 row 1: handing measurable action sequences to a local closed loop reduces handoffs | **Supported:** uc6 → uc7 on t3 went 87 → 22 calls; v3 r1 → r2 went 79 → 41. |
| §13 row 2: r2 was faster than r1 | **Supported** for one pair: 1,739.2 → 1,285.1 s. |
| §13 row 3: r6 cut I/O cost | **Supported:** video 110.08 → 37.49 s, state files 72.54 → 0.22 s. The task still took 1,626 s. |
| §13 row 4: the r9 false payload was fixed | **Supported** by g5_002 (zero attachment probes), but the push still failed. |
| §13 row 5: from r10, free-space coverage no longer generally blocks progress | **Partly supported.** Motion permission no longer needs coverage (`local_geometry.py:201`), but t1_011 still ended on `no_spatial_coverage_witness` in view planning, and in r11 the semantic-goal geometry precondition took over that blocking role. |
| §13 rows 6–7: r11 viewing is not proven; no object operation has been locally recovered end to end | **Supported** (0 executed view steps; 0 accepted holds in r10–r11). |
| §13 row 8: the 5-minute target is not met by v3 | **Supported.** **But** the doc's opening line ("尚未实现…5 分钟内成功完成", not yet achieved within 5 minutes) omits uc7 t5_001 (≈274.9 s wall, raw ✓). That success survives only because of truncation at first raw success (§0, item 2). |
| §13 row 9: generalization is unproven | **Supported**, and v3 is weaker than uc7 on it (§4.6). |
| §13 closing: future work should centre on "可用的观测目标、有效的几何修正、持续的对象身份与接触状态" (usable observation targets, effective geometry correction, persistent object identity and contact state) | **This is a recommendation, not a finding.** It continues the v3 direction that today's user prompt restricts (C8, X5), and today's v3 r6 baseline failed 0/3. |

**History-doc omissions and inaccuracies:**
1. uc7 t5_001 is absent, along with its ≤ 300 s service and its fragile raw success.
2. The 3 uc7 preparation stalls and v3 t1_001's 900.8 s stream failure are missing; the history table gives "完整服务 —" (full service: not recorded) for t1_001.
3. uc4 t6_005 is described as an infrastructure stop; the cause was the client's request-size `ValueError`.
4. No forbidden-contact counts are reported for the uc7 sweep (for example t0 19,799, t1 5,738), while v3's (433, 251) are reported. This asymmetry flatters v3's speed-for-safety trade.

---

## 6. The architectural fork: uc7 versus v3

### Philosophy

| | uc7: the model owns continuous control; macros are optional | v3: the runtime gates motion on geometric evidence and compiles semantic programs |
|---|---|---|
| Who sets targets | The model gives world-frame poses, segment order, gripper events and observation points (frozen uc7 `agent_client.py:209`, decision types `execute/motion/macro/observe/finish`). | The model gives intent plus an OBB `scene_update` plus `motion_context` (intent, target, held object, allowed contacts, scene inventory) in *every* response (`agent_speedup_v3_client.py:190,196-206`). Programs derive targets from geometry (`program.py:138-158`) and require fresh geometry (`:249`). |
| Local layer | Measured tracking that stops only on measured events: arrival, stable gap, stall, budget (`fragments.py:300-320`). Fixed envelopes (`:25-31`). | Sweep checks on palm, fingers and payload against model geometry. Final-interaction inventory (`agent_speedup_v3_geometry.py:1154-1202`). RGB watch invalidation. Two-view attachment and release gates. Operation runtime and budgets. Evidence acquisition. A terminal `EvidenceAcquisitionFailed` (`agent_speedup_v3.py:369-375`; call sites at :301, :387, :500, :577, :645). |
| How unknowns are treated | "未知不自动等同于物理失败" (unknown is not automatically physical failure) (`agent_SpeedUp_v2_user_check.md:20-21`). The model decides. | Unknowns must be resolved by evidence before final interactions, goal derivation and viewing. From r10, missing free space is advisory only (`local_geometry.py:201`). |
| Prompt and output | Up to 6 image pairs plus full text history: 24.7k–82.2k input and 274–518 output tokens per call (uc7 runs). | Compact context: 15.7k–27.7k input but 1,052–1,367 output tokens per call (r10–r11); the first call is 3.4–3.6k output tokens. |
| Code | uc7 layer: 20 files, 6,954 lines; core loop is agent 645 + client 505 + fragments 389 lines. | 16 additional v3 files, 7,850 lines (478 KB), on top of uc7. |

### Which design the evidence favours

- **Speed: uc7.**
  - On the same initial state its success was 3.7× faster than v3's best (416.3 vs 1,601.9 s elapsed) with a third of the calls.
  - Per call: 14.0 vs 33.4 s. First call: 17.9 vs 119.6 s.
  - v3 spent 21 of 40 calls (568.4 s) on `geometry_patch` responses that its own gates required.
  - uc7 holds the only ≤ 300 s raw success. With compact I/O on a new episode it still took 614 s of service including 138 s of review (t1 ep1).
- **Generality: uc7.**
  - Raw 9/10 on the one-episode 10-task sweep and 11/13 overall, against v3's 0/3 off t1 and 0/10 since r6 including today.
  - Caveats against uc7:
    - Strict success only 3/13.
    - t6 unsolved in uc7 (0/2; only uc6 solved it, with 84 calls).
    - Raw success depends on truncation.
    - High forbidden-contact counts on some tasks.
    - Most sweep plans were human-reviewed, 2 of 10 edited.
- **v3's one advantage:** fewer recorded forbidden contacts on t1 in r5/r6 (433/251 vs uc7's 5,738). This does not hold for r2 (6,808), and contacts are not part of the success criterion (C6, D3).
- **[inference]** The mechanism: v3 turns RGB-estimation uncertainty into required evidence. That shows up as many repair calls, larger outputs and terminal stops — 8 of the 9 failures since r4.1, 6 of them model escalations over unresolvable sweep or clearance evidence. uc7 lets the model act on uncertain estimates and relies on measured stalls.

---

## 7. Unknowns

- **U1.** The cause of the preparation streams that never complete: possible runaway output, service-side behaviour, or both. Only t1_005's content was retained.
- **U2.** Whether the user ever ruled on D1–D5 outside the repository (in chat). No artifact records a ruling.
- **U3.** Who wrote `Agent_SpeedUp_Codex_Implementation_Plan_ZH.md` and whether the user endorses its task-label and operation-family relaxations.
- **U4.** uc7's success rate on more than one initial state per task. Only t1 ep1 has been tested, twice, both successes and both > 300 s.
- **U5.** Whether t5_001's ≤ 300 s is repeatable. Its manual twin t5_002 took ≈327.5 s with 10 calls.
- **U6.** Exact service time for the uc7 runs without supervisord logs. I used preparation + elapsed, which runs about 5–6 s low where checkable.
- **U7.** What "implement this" referred to in the uc7 authorization of 2026-10-21 13:42. **[inference]** It was the fragment design; the instruction came about an hour after the general plan.
