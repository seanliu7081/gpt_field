# uc7 (`agent-speedup-v2-user-check-7`): the framework as it actually runs

Scope: a map of the code, read only. No simulator, model call or test was run. Unless a path is given, paths are relative to `/workspace/gpt_field`. `uc.py` means `src/gpt_field/agent_speedup_v2_user_check.py`, and `agent.py`, `agent_client.py`, `fragments.py`, `macros.py`, `sim.py`, `client.py` and `motion.py` mean the matching `src/gpt_field/agent_speedup_v2_user_check_*.py` file. "Base runner" means `scripts/evaluate_agent_speedup_v2.py`, and "UC runner" means `scripts/evaluate_agent_speedup_v2_user_check.py`. Numbers marked *computed* are my own aggregation of the run artifacts named next to them.

---

## 0. Identity, builds and the evidence base

- Version constants:
  - `VERSION = "agent-speedup-v2-user-check-7"` (uc.py:31)
  - `POLICY_VERSION = "agent-speedup-v2-user-check-agent-2"` (agent.py:30)
  - `CLIENT_VERSION/PROMPT_VERSION = "...-client-2"/"...-prompt-2"` and `HISTORY_PROTOCOL = "persistent_task_executive_fragments_text_and_six_visual_pairs_v2"` (agent_client.py:40-42)
  - `FRAGMENT_VERSION = "user-check-measured-fragments-1"` (fragments.py:21)
  - framework `"user-check-agent-fragments-framework-2"` (UC runner:57)
- **There are two uc7 builds, and their Python is identical.** Hashing each run's `frozen_source` gives two groups. Runs t3_003 and t6_009 used build A. The other 11 used build B. `diff -rq` between `runs/..._t3_003/frozen_source` and `runs/..._t1_001/frozen_source` reports only `agent_speedup_v2_user_check.html`, which changed from Chinese to English (216 changed lines). All 300 frozen files otherwise match.
- **Outcomes** (from `runs/agent_speedup_v2_user_check_libero_10_<run>/summary.json`):
  - 13 uc7 runs gave 11 raw LIBERO successes across 9 of the 10 tasks. t6 failed twice: t6_009 and t6_011, both `user_stop` by operator stop file.
  - Strict `v2_success` held in 3 of 13 runs (t2_002, t8_001, t9_001).
  - The one-build sweep `runs/agent_speedup_v2_user_check_libero_10_manual_batch_001/batch_state.json` (build B, manual review, episode 0, t0 to t9) gave 9/10 raw successes.
  - The orchestrator's "13/15" matches only if the two uc6 successes (t6_007, t3_002) are added; uc7 alone is 11/13.
- **S300.** I computed service time as `user_check.preparation_elapsed_s + elapsed_s`. Only **t5_001** passes S300: 28.4 + 241.7 = 270.2 s, with raw success, and it was an auto-approved run outside the batch. All other uc7 runs took 327.5 to 962.4 s.
- Approval mode: 10 runs used `manual_browser_review`; t3_003, t5_001 and t6_009 used `user_authorized_automatic` (`summary.json:user_check.approval_mode`).
- Preparation can stall. 3 uc7 preparation attempts never got a plan-draft response and were stopped by KeyboardInterrupt. Each has a request log and a `result.json` with `"error_type": "KeyboardInterrupt"`:
  - t6_008: 634 s
  - t3_004: 298 s
  - t6_010: 297 s

  Sources: `runs/..._{t6_008,t3_004,t6_010}_review/requests/*.result.json` and their file mtimes. The batch logged two of them as `user_requested_restart_of_stalled_plan_preparation` (`manual_batch_001/t03_restart.json`, `t06_restart.json`). That makes 3 stalls in 16 uc7 preparation attempts (*computed*).

  The stall is **not input-determined**. The stalled requests have the same `request_key` as requests that succeeded:
  - t6 key `c24936d2…`: t6_008 and t6_010 stalled; t6_009 and t6_011 were valid.
  - t3 key `426b72b4…`: t3_004 stalled; t3_003 and t3_005 were valid.

---

## 1. Component diagram (as executed in uc7)

```
 CLI  scripts/evaluate_agent_speedup_v2_user_check.py  main():578
  |   subcommands: prepare | start [--auto-approve] | run | review | preflight
  |
  |--(1) PREP, 0 policy actions  prepare_review():131
  |      UserCheckSimProcess (own worker) start + reset -> initial obs SHA check (:170-176)
  |      UserCheckV2Client.request_plan (:188)  4 PNG 512px images (ext clean/grid, wrist clean/grid),
  |                                             reasoning=high, variant plan schema (+regions,+placement_goals)
  |      ReviewStore.create (:214) -> review.json, .review-context.json, initial_observation.json, frozen_source/
  |
  |--(2) REVIEW  _show_review():563 -> ReviewServer 127.0.0.1:8765 (token in URL #fragment)
  |      GET / | /api/schema | /api/review ; POST /api/update | /api/approve  (server.py:201-252)
  |      or --auto-approve: store.approve(all steps) (UC runner:656-663)
  |
  |--(3) RUN  execute_reviewed():426
         ReviewStore.claim (single use, .review-claimed.json)  ->  PlanLock(approved plan)
         base.execute()  scripts/evaluate_agent_speedup_v2.py:822   (synchronous loop :1398-1527)
            |
            |  per control step: state("deciding") -> policy.act -> trace "action" -> video frame
            |                    -> state("executing") -> sim.step -> trace "executed"(+2 PNGs)
            |                    -> policy.observe_transition -> trace "feedback" -> video frame -> state("observed")
            v
   +-------------------------------- AgentFirstV2Policy (agent.py:55) --------------------------------+
   |  modes: agent | fragment | direct | macro          one executive call max per act (agent.py:507)   |
   |                                                                                                    |
   |  need_agent --> AgentFirstV2Client.request_agent (agent_client.py:451)                              |
   |                  transcript: ALL prior text turns + encrypted reasoning, last 6 RGB pairs,          |
   |                  strict json_schema, stream SSE, effort high  ==> gpt-6-astra (~15 s/call)          |
   |                                                                                                    |
   |  type=execute  --> FragmentSequencePrimitive (fragments.py:169)  1-6 segments, <=240 steps          |
   |  type=motion   --> LocalMotionPrimitive (motion.py:267) -> ShortTargetExecutor (1-5 steps, 5cm)     |
   |  type=observe/finish --> 1-step zero relative motion (agent.py:356-370)                             |
   |  type=macro    --> AgentMacroSession (macros.py:33) -> base v2 relation layer                       |
   |                    (hold/turn/place_on/release; CalibratePlus ~190 steps; 1-60 step grants)         |
   |  PlanLock (client.py:209): approved plan, decision log, framework snapshot                          |
   +----------------------------------------------------------------------------------------------------+
            | 7-D normalized OSC action (action.py ActionAdapter)
            v
   UserCheckSimProcess (sim.py:67) --JSONL pipe--> worker (sim.py:93) = V2LiberoWorker + joints
      robosuite OSC_POSE 20 Hz; renders agentview + robot0_eye_in_hand at 512x512 EVERY step;
      PNG-encode both; env.check_success (raw LIBERO); generic contact monitor (record-only);
      evaluator truth window update; + one extra "agent_geometry" IPC per observation (sim.py:86)
            |
   ReviewedRecorder(AgentSpeedUpV2VideoRecorder): 1600x960 canvas x2 per step -> ffmpeg libx264;
      episode.events.jsonl; user_check_framework.json (<=1 Hz)   (UC runner:237-297)
   Finalization: diagnosis-only truth, video close, _normalize_artifacts (UC runner:300-423)
```

---

## 2. Entry points, preparation, review and approval

### 2.1 Subcommands (UC runner:578-680)
- `prepare` creates the draft only.
- `start` runs prepare, then review or `--auto-approve`, then execute.
- `run` consumes an already approved review.
- `review` serves the UI only.
- `preflight` makes no network, credential or simulator calls (:525-560).

`start` requires a loopback host (:615-617). A live run first passes `_live_gate`: T0 lint, render environment and a live build check (:500-508).

### 2.2 Preparation (`prepare_review`, UC runner:131-225)
- The review directory must be fresh (:143-144). The oracle is refused (:153-154 and :165-166).
- It computes `runtime_inventory` (:155). This hashes **every** `src/gpt_field/**/*.py`, the UC runner, the HTML, the base and v1 runners, the helper chain, the data files and an `@configuration` hash (:113-128).
- It starts a simulator worker and resets it (:162-170). The initial-observation SHA must equal the frozen reset SHA (:174-176).
- Images come from `plan_images`: external clean, external 8x8 grid, wrist clean and wrist grid (`agent_speedup_v2_client.py:494-500`).
- One `UserCheckV2Client.request_plan` call (:188-190) uses `reasoning_effort="high"` (client.py:343-347). It sends the base plan instructions plus `SPATIAL_INSTRUCTIONS` and `METRIC_INSTRUCTIONS` (client.py:348-359). The schema is the variant plan schema, which requires `regions` and `placement_goals` (`agent_speedup_v2_user_check_plan.py:82-89`).
- Measured on t1_001: `draft_model_s` 13.7 s and preparation `elapsed_s` 21.4 s. The request had 3,911 input tokens and 426 output tokens. Each 512 px high-detail image cost about 308 input tokens (`runs/..._t1_001_review/review.json`, `requests/000001-*.response.json`).
- The runtime inventory is re-hashed after preparation (:210-213). Then `ReviewStore.create` runs (:214-217) and the whole inventory is copied to `review_dir/frozen_source` (:218-224). The preparation simulator is closed (:207-208); execution later starts a second one.

### 2.3 Review store and approval binding (uc.py:187-408)
- **Context hash** covers schema/version, id, created_at, task_key, episode_index, task_prompt, initial_observation_sha256, runtime_sha256, draft_plan and preparation (uc.py:35-39). It is sealed in `.review-context.json` (uc.py:335-338).
- **Selector-only edits.** Scene text, part text and image cells are immutable. Effect texts must come from the draft (`_selector_edits`, uc.py:173-184). Every write is atomic with fsync (uc.py:212-228).
- **Approval** requires every step index to be checked (uc.py:367-381). It stores `binding_sha256 = sha256(context_sha256, plan, revision, reviewed_steps)` (uc.py:240-246). Validation uses `plan_validation(require_native=False)`, so non-native intents (slide, place_in, push_to) are approvable (uc.py:98-135).
- **Claim, single use** (uc.py:383-408). The review must be `approved`. Task and episode must match. The initial-observation SHA must match. `runtime_sha256` must equal the prepare-time inventory. The claim writes `.review-claimed.json` **before** setting status `consumed`, so a crash fails closed. `_load` rejects replay (uc.py:288-294).
- `execute_reviewed` claims before any action, so a reset mismatch still consumes the attempt (UC runner:456-461).
- **Consequence (inference):** the runtime hash covers all of `src/gpt_field`. Editing any file in that tree between prepare and run, including v3 files, makes the claim fail with "Runtime code changed".

### 2.4 Review UI and server (server.py)
- `ThreadingHTTPServer` binds 127.0.0.1 only (:65-84).
- A 32-byte bearer token travels in the URL fragment (:71, :84). Host and Origin are checked, and cross-site Sec-Fetch is refused (:130-149).
- The HTML is served with a CSP nonce (:103-124). The body limit is 131,072 B (:24).
- Routes: `/api/schema`, `/api/review`, `/api/update`, `/api/approve` (:201-252).
- The runner polls `store.load()` every 0.25 s until the review is approved (UC runner:572-575).
- Wall-clock review time is recorded as `review_phase_elapsed_s`. It is not counted as control time (UC runner:340-343).

### 2.5 Auto-approve (UC runner:656-663)
`store.approve(expected_revision=draft.revision, reviewed_steps=all)`. The mode is recorded as `approval_mode="user_authorized_automatic"` and `human_reviewed_plan=False` (:194-195, :316, :347).

---

## 3. The executive (model) loop

### 3.1 When the model is called
- Calls are synchronous inside `policy.act` (base runner:1438-1444).
- At most **one executive call per act** (agent.py:506-511). A call happens only when `_need_agent` is set:
  - at episode start (agent.py:88);
  - after a fragment, direct motion, observe or finish ends (agent.py:372-401);
  - after a macro yields (agent.py:403-419);
  - after a rejected response (agent.py:264-274).
- The simulation is paused during inference. The manifest records `simulation_pauses_for_api: True` (base runner:934), and the prompt says so (agent_client.py:150-151).

### 3.2 What each request contains (`request_agent`, agent_client.py:451-505; `_context`, agent.py:205-224)

| Element | Content | Source |
|---|---|---|
| Images | 2 new PNGs per call: "External upright RGB, observation step N" (agentview) and "Wrist upright RGB, observation step N" (robot0_eye_in_hand). Both 512x512, `detail:"high"`. Upright means the decoded policy image flipped (`StepView._image`, contact_skills.py:281-292); the geometry calls this `raw_mujoco_rgb_vertical_flip_no_crop`. | agent_client.py:496-502; `agent_speedup_v2_client.py:800` |
| Image history | Last **6 pairs** (12 images). Older images are replaced by the text "[Older camera image omitted by fixed image-retention rule.]" | agent_client.py:43, :364; direct_history.py:45-57 |
| Robot state | `robot_pose_wxyz` = [x,y,z,qw,qx,qy,qz,closure] from state8 (grip_site position, right_hand quaternion); `finger_joints_m` | agent_client.py:475-476; astranomy_agentic_motion.py:89-100 |
| Measured state | `pose_wxyz`, `finger_gap_mm`, **7 joint positions + 7 joint velocities** from the same cached observation as the RGB, `last_gripper_command` | agent.py:212-216; sim.py:40-48, :84-90 |
| Calibration | `geometry`: external `agentview` (fixed world camera, fovy 45 deg, f 618.04 px) and wrist `robot0_eye_in_hand` (pose from state8 plus fixed mount, fovy 75 deg, f 333.63 px), principal point 255.5. It must have `pose_source == observed_robot_pose_fixed_mount` for the wrist. | agent_client.py:464-469; astranomy_agentic_geometry.py:143-178; request 000001 of t1_001 |
| Gripper geometry | hand-to-grip 0.097 m, pad midpoint [0,0,-0.0036], pad 16x8x16 mm, nominal open gap 0.079 m, finger ranges | astranomy_agentic_geometry.py:33-45; also restated in the prompt (agent_client.py:185-190) |
| Context | `capabilities` (fragment, direct-motion and macro contracts), `model_belief`, `last_feedback`, `recent_feedback` (last 8), `macro` (status plus `setup_progress`), `execution` (active progress, last_result, shared_state), `native_evidence` (holding estimate, calibration, parts, observers, outcomes[-8:]), `task_success_source` | agent.py:128-224; CONTEXT_KEYS agent_client.py:45-49 |
| Goal contract | **First call only:** `reviewed_goal_contract = {initial_plan, task_contract, authority}`. It persists afterwards only through text history. | agent_client.py:478-481 |
| Compaction | If size(fields) > 45,000 chars, `execution`, `native_evidence`, `last_feedback` and `recent_feedback` are cut to 7000/5000/7000/4500 chars, with stage outcomes and causes prioritised. If the result is still > 56,000 chars, a `ValueError` is raised and the run ends as infrastructure_error. | agent_client.py:484-495, :261-289 |
| Instructions | `AGENT_INSTRUCTIONS`, 10,060 chars, sent in every request | agent_client.py:51-192; t1_001 request payloads |

Sizes in t1_001 (`runs/..._t1_001/requests/*.request.json`, *computed*):
- fields text: 13.1k chars on call 1, then 17.8k to 19.9k chars;
- input tokens: 6,544 on call 1, rising to 75,354 on call 13;
- request file: up to 4.19 MB on disk.

Pooled over all uc7 runs, input tokens grow about 5.5k per call, reaching 153,901 at call 28 of t6_009.

### 3.3 Transcript protocol (agent_client.py:355-372, :393-449; hybrid_subscription.py:416-489)
- The payload is `store:false, stream:true, service_tier:"default", reasoning.effort:"high", include:["reasoning.encrypted_content"]`, with `text.format` = strict `json_schema` named `user_check_agent_first`.
- The whole transcript is re-sent every call: `_agent_history` plus the new user message, image-trimmed.
- After an accepted response the client commits `payload["input"] + response["output"]`, which includes the opaque encrypted reasoning items (hybrid_subscription.py:465-468; agent_client.py:442-447).
- The resulting layout is user, reasoning, assistant message, user, and so on (request 000003 of t1_001).
- A failed or rejected response is **not** committed.

### 3.4 Output schema and action types (agent_client.py:195-215)
- Required keys: `belief` (≤1800 chars), `plan` (≤1800), `why` (≤600), `type`, `motion`, `macro`, `execution`. Unused payloads must be null.
- The `type` values:
  - `execute`: `execution={segments:[1..6]}`, each segment {skill, xyz_m|null, quaternion_wxyz|null, gripper, profile, max_steps 1..120, observation_after} (fragments.py:46-64).
  - `motion`: {frame:"world", position_mode absolute|relative, xyz_m[3], quaternion_wxyz[4]|null, gripper keep|open|close, steps 1..5}. This is `local_motion_schema` minus `recovery_goal` (motion.py:64-80; agent_client.py:196-198).
  - `macro`: {step_index ≥0, budget_steps 1..60, continuation bool}.
  - `observe`: hold and get fresh images.
  - `finish`: a completion **claim** plus fresh images. It never terminates and there is no give_up (agent_client.py:170-177).
- **Observed in uc7:** of 218 executive decisions, 210 were `execute`, 7 `motion`, 1 `finish` (t6_009) and 0 `macro`/`observe`. Source: `user_check_plan_history.json:model_decisions`, summed over the 13 runs.

### 3.5 Validation (agent_client.py:218-258; fragments.py:80-139)
- Exact key set; non-empty bounded strings; the chosen payload present and the others null.
- Motion: finite numbers and a non-zero quaternion (motion.py:94-119).
- Macro ranges.
- `validate_execution`:
  - skill, profile and gripper enums;
  - arm stages require xyz_m and gripper=keep;
  - `approach` requires profile=contact;
  - a relative or approach translation must fit the profile envelope;
  - set_gripper/wait require null pose fields;
  - set_gripper needs open or close with max_steps ≥ 12 (close) or ≥ 8 (open);
  - wait requires keep;
  - total max_steps ≤ 240;
  - quaternions are renormalised.
- A failure raises `SchemaError`. The policy turns it into a feedback record `{"action":"model_response","status":"rejected","reason":"invalid_action_arguments"}`, issues a 1-step hold and re-asks on the next act (agent.py:264-274, :524-531).
- Observed: no `invalid_action_arguments` appears in any uc7 request (grep across all 13 runs' requests).

### 3.6 Transport and retries
- `ordered_sse_transport` sends one POST to `https://api.openai.com/v1/responses` and reads SSE until `response.completed` (ordered_transport.py:29-53).
- The timeout is `urllib` `timeout=timeout_s` (180 s, from the manifest `api.timeout_s`; astra.py:489; chatgpt_plan.py:136-157). That is a per-socket-operation timeout, not a total deadline. With `max_wall_time_s=None`, nothing bounds a stream that keeps emitting events (inference, consistent with the 3 stalled preparations in section 0).
- Bounded retries apply only when no response envelope arrived (`agent_speedup_v2_client.py:903-914`):
  - overload: 3 attempts, 2 s base with up to 1 s jitter, 60 s maximum wait;
  - transient: ≤2 retries (:139-151; agent_client.py:407-440).
- No uc7 execution run recorded a retry or an error result (requests directory scan).

---

## 4. Execution layer

### 4.1 Fragments: `FragmentSequencePrimitive` (fragments.py:169-389)
- **Entry** (`_enter_segment`, :208-244):
  - Targets resolve once from the **measured** pose at each stage's entry. `move_to_pose` is absolute; `move_relative` and `approach` add a world delta.
  - A null quaternion keeps the entry orientation.
  - If the distance from the measured entry pose exceeds the profile envelope (transit 0.30 m / 1.2 rad, contact 0.08 m / 0.5 rad), the stage fails with `target_envelope_rejected`. It issues no action and the sequence ends (:233-239).
- **Per step** (`_next`, :246-258):
  - `delta = target − measured position`, clamped to 0.02 m (transit) or 0.005 m (contact).
  - Orientation goes through `ActionAdapter(max_rotation_rad=.1)` (:185).
  - The gripper keeps its actual open/close command; set_gripper switches it.
  - The action is normalised to [Δ/0.05, rot/0.5, ±1] and clipped to [−1, 1] (action.py:59-93).
- **After each acknowledged step** (`_observe`, :260-326):
  - *gap-rise interrupt:* during a non-set_gripper stage with close maintained, the gap at ≥6 mm above its floor for 3 consecutive samples gives `maintained_grip_gap_changed` (:278-289).
  - *motion completion:* `pose_arrived` after 2 consecutive samples within 4 mm / 0.06 rad (transit) or 2 mm / 0.04 rad (contact) (:203-206, :291-293).
  - *set_gripper completion:* `gap_stable` needs ≥12 or 8 actions, a 5-sample gap range ≤0.6 mm **and** pose arrival at the entry pose (:294-301).
  - *wait:* `wait_elapsed` after max_steps (:302-303).
  - *budgets:* `segment_budget_exhausted` (:306-308) and `execution_budget_exhausted` at 240 (:309-310). The latter is effectively unreachable because validation already caps the sum at 240 (inference).
  - *stall:* for motion stages with ≥16 actions, if the weighted error (pos + 0.05·rot) improved < 0.8 mm across the last 12 samples, `segment_stall` (:311-325).
- **Advance** (`_complete_segment`, :328-340):
  - The next stage starts in the same step unless the stage had `observation_after=true` or was a `wait`. In that case the result is `observation_required` and the remaining stages are reported but **never auto-resumed**.
  - Completing the last stage gives `fragment_complete`.
- **Handoff:** evidence (progress plus `segment_trace` with per-step measurements) goes to feedback (agent.py:375-391), and the next act asks the model.

### 4.2 Direct targets: `LocalMotionPrimitive` to Astranomy `ShortTargetExecutor`
Sources: motion.py:267-311; astranomy_agentic_motion.py:103-266.
- Relative targets resolve once from the measured pose (motion.py:276-281).
- The target must lie within **0.05 m / 0.35 rad** of the latest measured pose. Otherwise `local_motion_target_rejected` with zero actions (astranomy_agentic_motion.py:158-163; motion.py:292-302).
- It executes **exactly** `steps`, 1 to 5 (:126-127). Each step re-computes the error capped at 0.02 m / 0.1 rad (:208-229). There is no arrival check or retry.

### 4.3 observe and finish
A 1-step relative-zero motion with gripper keep (agent.py:356-362). `finish` also appends `model_completion_claims` with `evaluator_success:"not_observed"`.

### 4.4 Native macros (dormant in uc7 runs)
Sources: `AgentMacroSession`, macros.py:33-304; `_apply_agent`, agent.py:307-339; `_next_macro`, agent.py:437-482.
- Only `hold/turn/place_on/release` are supported (macros.py:90-91). Other intents return `native_skill_not_implemented`.
- The macro is refused with `native_geometry_unavailable_while_hand_closed` if the hand is closed without native geometry (macros.py:96-99).
- **Setup:**
  1. lazy `CalibratePlus`, nominally about 190 steps (`agent_speedup_v2_primitives.py:689`; reported as `estimated_total_steps: 190`, macros.py:227);
  2. a current-view **plan grounding model request**, whose result goes through `PlanLock.check_plan` (a semantic change triggers the `plan_conflict` stop; macros.py:119-123, client.py:294-297);
  3. registration, possibly with pointing calls;
  4. keep-outs, then `run_step`.
- Grants are 1 to 60 actions. When a grant is used up the macro pauses with reason `model_selected_action_budget`, which is explicitly "not failure" (agent.py:503-505; macros.py:252-260). `continuation=true` resumes the same generator.
- Limits: at most 2 non-executive model calls per act (agent.py:33, :474-475); 24 transitions per act (agent_speedup_v2.py:133).
- Any direct or fragment action marks native geometry stale (macros.py:297-304). Taking direct control cancels the macro and clears episode monitors (agent.py:292-303).
- **Usage:** `optional_macro_grants = 0` and `native_macro_actions = 0` in all 13 uc7 `agent_stats.json`. 8 of 13 approved plans contain at least one intent with no native macro (`place_in` or `slide`). The other 5 (t2_002, t4_001, t6_009, t6_011, t8_001) are all-native (`turn`/`place_on`), and the model still never chose a macro (`approved_plan.json`). The ~33k lines of base v2 relations, planner, observers and scene code are therefore instantiated at reset (agent.py:66-84) but never exercised by uc7 control.

### 4.5 Observed execution mix (13 uc7 runs, *computed*)
- Control steps (`agent_stats.json` counts, summed): fragment 11,895 (99.66%); direct 41 (7 `motion` decisions plus 5 hold steps); native 0. Total 11,936 = sum of `summary.json:steps`.
- Terminal outcome of 192 fragment executions that ended with ≥1 action (trace `feedback` rows):
  - `fragment_complete` 133 (69%);
  - `segment_stall` 55 (29%);
  - `segment_budget_exhausted` 2;
  - `observation_required` 1;
  - `target_envelope_rejected` 1 mid-sequence.
- 5 more envelope rejections happened at a sequence's first stage, so they issued zero actions and each produced one hold step (`hold_actions` in `agent_stats.json`).
- Steps per execute: median 56, mean 61.4, max 139.
- The 55 stalls ended with a final position error median of 22.4 mm (range 2.3 to 120.1 mm). They consumed 2,087 of 11,798 fragment steps, about 17.7%. They occurred in transit `move_to_pose` (24), contact `move_to_pose` (19) and `approach` (12).
- `maintained_grip_gap_changed` fired 0 times.
- Stage completion medians (actions):
  - transit `move_to_pose` → pose_arrived: 35;
  - contact `move_to_pose`: 53;
  - `move_relative` contact: 33;
  - `set_gripper` → gap_stable: 15.

---

## 5. One control cycle, step by step
Base runner `execute()`, loop at :1398-1527, with the uc7 policy.

1. **Terminal checks** (:1399-1435):
   - `sim.status["success"]` (raw LIBERO) leads to `success` and break;
   - `operator_stop.request.json` leads to `user_stop`;
   - `--stop-after-steps` cap;
   - (`recovery_stop`, never emitted by the agent policy);
   - a simulator terminated without success raises an error.
2. `state("deciding")` writes **run_state.json** (light: budget plus history_stats, no agent_stats) (:997-1014, :1438).
3. `policy.act(obs)` (:1440), which is `AgentFirstV2Policy.act` (agent.py:484-570):
   1. `plan_lock.check_active()`; build or cache the StepView.
   2. If `_need_agent` and no executive call yet in this act, `_ask_agent`. That runs `_sync_framework`, `_context`, `request_agent` (**blocking model call, about 15 s**), validation, then `_apply_agent`, which builds the Fragment/LocalMotion primitive or starts or resumes a macro.
   3. Otherwise continue the active primitive or macro. If nothing is active, a 1-step hold (:524-531).
   4. `acting.next_action(view)` produces the 7-D action (:532). Diagnostics include history_stats, routing, belief and feedback (:541-566).
4. Timings accumulate (`decision_s`, `model_pipeline_s`, :1441-1444). The trace gets an **"action" row** of about 35 KB (:1473-1475).
5. `recorder.decision` renders **video frame #1**: a 1600x960 canvas repeated 2x, or 12x on a model-refresh step (agent_speedup_v1_video.py:182-213). One events.jsonl line is written. At most once per second this also publishes `user_check_framework.json` (UC runner:248-275).
6. `state("executing")` writes run_state.json (light) (:1480).
7. `sim.step(action)` (:1483) sends a JSONL request to the worker. The worker does:
   - robosuite OSC_POSE step at 20 Hz;
   - contact monitor at every physics substep;
   - **render agentview + eye_in_hand at 512x512**;
   - `check_success` (base and V2 each call it);
   - truth-window update;
   - PNG-encode both images and decode them for validation;
   - attach 7+7 joint values.

   On the host, `PolicyObservation.from_dict` decodes both PNGs again. Then **one extra IPC `agent_geometry` request** builds a `CalibratedObservation`, whose `__post_init__` re-runs `PolicyObservation.__post_init__` and so decodes both PNGs a second time. That is 4 host-side validation decodes per step. Sources: libero.py:868-873, :919-954, :662-667; observation.py:24-45, :114-115, :131-160; agent_speedup_v2_sim.py:155-169; sim.py:55-57, :78-90, :117-122.
8. The trace gets an **"executed" row with the full observation** (two base64 PNGs; average 556 KB per row in t1_001) (:1494-1496), plus collision rows.
9. `policy.observe_transition` (:1506) runs these in turn:
   - base: the primitive's `observe_transition` (fragment predicates and handoff), the `InvariantMonitor.observe` (empty in agent mode) and the feedback record (agent_speedup_v2.py:1913-1947);
   - then agent: gripper-command memory, counters, macro geometry marked stale, `_complete_direct` if done (sets `_need_agent`), and `_sync_framework`, which deep-copies state (agent.py:572-599).
10. The trace gets a **"feedback" row** of about 4.3 KB (:1509). `recorder.executed` renders **video frame #2**, repeated 2x (video.py:227-233; :1510-1513).
11. `state("observed")` writes **run_state.json with full `agent_stats`**, including the framework snapshot. That file was about 0.94 MB at the end of t1_001 (:1516; agent.py:629-645).
12. If raw success, break.

---

## 6. Which checks veto, shorten, end the episode, or only advise

| Check | Class | Where | Effect | Seen in 13 uc7 runs |
|---|---|---|---|---|
| Strict JSON schema plus `validate_agent_decision` and `validate_execution` | **VETO** (whole decision) | agent_client.py:218-258; fragments.py:80-139 | `invalid_action_arguments` feedback, 1-step hold, re-ask | 0 |
| Stage entry envelope (transit 0.30 m/1.2 rad, contact 0.08 m/0.5 rad from measured entry) | **VETO** (stage; ends sequence) | fragments.py:233-239 | `target_envelope_rejected`, control back to the model | 6 (fed back) |
| Direct target envelope 0.05 m / 0.35 rad | **VETO** | astranomy_agentic_motion.py:158-163; motion.py:292-302 | zero actions, `local_motion_target_rejected` | 0 observed |
| Macro refusals (`step_index_out_of_range`, `no_paused_macro_to_continue`, `native_macro_not_started`, `native_skill_not_implemented`, `native_geometry_unavailable_while_hand_closed`, calibration or grounding unavailable) | **VETO** (macro only) | agent.py:310-333; macros.py:90-121 | feedback, model decides | not reached |
| Per-action clamps 0.02 m / 0.005 m / 0.1 rad; OSC clip [−1, 1] | **CLAMP** | fragments.py:252-258; astranomy_agentic_motion.py:215-217; action.py:67-93 | limits speed, never rejects | every step |
| `segment_stall` (≥16 actions; 12-sample window; < 0.8 mm improvement) | **SHORTEN** | fragments.py:311-325 | sequence ends, model re-asked | 55/192 terminations (29%) |
| `maintained_grip_gap_changed` (+6 mm × 3 samples) | **SHORTEN** | fragments.py:278-289 | sequence ends | 0 |
| `segment_budget_exhausted` / `execution_budget_exhausted` | **SHORTEN** | fragments.py:306-310 | sequence ends | 2 / 0 |
| `observation_after` / `wait` checkpoint | **SHORTEN** (model-chosen) | fragments.py:335-337 | `observation_required`; pending stages dropped | 1 mid-sequence |
| Macro grant budget; native observer preemption; native primitive failure; RepairNeed | **SHORTEN** (macro only) | agent.py:440-472, :503-505 | pause or return to model | not reached |
| Raw LIBERO `env.check_success()` | **TERMINATE** (success) | libero.py:936-945; base runner:1399-1404, :1526-1527 | `success` | 11 |
| Operator stop file / `--stop-after-steps` | **TERMINATE** | base runner:1405-1421 | `user_stop` | 2 (t6_009, t6_011) |
| Macro grounding changes approved semantics (`PlanLock.check_plan` → `stop`) | **TERMINATE** | macros.py:123; client.py:285-297; UC runner:352-361 | `plan_conflict` | 0 (uc1 t6_001 hit it) |
| APIError after retries / BudgetExceeded | **TERMINATE** | base runner:1533-1534 | `subscription_error` | 0 |
| Any other exception (e.g. fields > 56,000 chars, image/calibration mismatch, simulator timeout 120 s) | **TERMINATE** | agent_client.py:494-501; base runner:1535-1536 | `infrastructure_error` | 0 |
| `finish` claim | ADVISORY | agent.py:359-362; `stop_on_model_complete: False` (base runner:927) | logged only, episode continues | 1 (t6_009 at step 965; run continued to 1544 and was operator-stopped) |
| InvariantMonitor observers in direct/fragment mode | ADVISORY | agent_speedup_v2.py:1933 (observe); preemption consumed only in `_next_macro` (agent.py:440) | none | `observers: {}` in t1_001 action rows |
| Contact/collision monitor | ADVISORY | V2 worker `collision_policy: record_only`, `stop_on_collision` False (agent_speedup_v2_sim.py:99-107) | recorded | n/a |
| Stuck alert (10 of last 12 relation outcomes failed) | ADVISORY | base runner:37-41, :127 | console alert only; needs relation outcomes, which uc7 never produces | 0 |
| Physical truth / `v2_success` | ADVISORY (post-hoc) | base runner:1198-1199, finalization | diagnosis only | 3/13 strict |
| Visible-token budget 1400 | ADVISORY | agent_client.py:331, :352-353 | counter only | n/a |
| Horizon / max API calls / max wall time | **absent** | `max_steps=None`, `max_api_calls`/`max_wall_time_s` None (manifest) | a failing run continues until an operator stops it | t6 runs: 930 s and 711 s |

---

## 7. Termination, success and per-step logging
- **Success** is raw LIBERO `env.check_success()` evaluated after every step. With `stop_on_success=True` the worker marks the episode terminated (libero.py:936-945; agent_speedup_v2_tasks.py:146-151). The runner then sets `status="success"` and `task_success = (final == "success")` (base runner:1193).
- `v2_success` also requires the strict physical truth check, which is diagnosis-only (base runner:1198-1199).
- Model beliefs and `finish` claims never end the episode. `model_completion_is_libero_success: False` (base runner:927).
- **Per-step artifacts:**

| Artifact | Per-step behaviour | Size (t1_001, 897 steps) |
|---|---|---|
| `trace.jsonl` | 3 rows: action (about 35 KB), executed with full observation (about 556 KB), feedback (about 4.3 KB) | 535 MB total (*computed* by event type) |
| `episode.mp4` and `episode.events.jsonl` | 2.0 canvas renders and 4.23 frames per step, at 1600x960 RGB piped to ffmpeg libx264 `-preset veryfast -threads 1` (agent_speedup_v1_video.py:95-125). That is about 17.5 GB of raw RGB over the run. | events file 15.8 MB (*computed*) |
| `run_state.json` | 3 writes per step; the "observed" write includes full agent_stats | 0.94 MB at the end |
| `user_check_framework.json` | at most 1 write per second, pretty-printed `lock.framework_snapshot()` | 0.74 MB at the end |

- Per model call: `requests/NNNNNN-*.request.json` (the full payload including retained images), `response.json` and `result.json`.
- At the end: summary/runner_summary/episodes, `base_v2_artifacts/`, `user_check_review.json`, `approved_plan.json`, `user_check_plan_history.json` (every decision) and `frozen_source/` (UC runner:300-423).

---

## 8. Structural time costs visible in code (with measured magnitudes)

Pooled over the 11 successful uc7 runs (`summary.json:timing_components`, *computed*):
- 9,195 steps and 163 calls; control time 4,579.7 s;
- **model 2,460.3 s (53.7%)**;
- simulator step 961.6 s (21.0%);
- video 835.2 s (18.2%);
- run_state 232.3 s (5.1%);
- observe_transition 64.2 s (1.4%);
- policy compute 8.8 s (0.2%).

Unaccounted time (trace writes and similar) is 1.4 to 2.7 ms per step across runs (control_elapsed minus the components, *computed*).

1. **Synchronous, stop-and-think loop.** The model call blocks `policy.act`, and the simulator does not advance during it (base runner:1440, :934). Each handoff costs one full model latency. Latency averaged 15.47 s per call (218 calls; `episode.events.jsonl` skill_decision `api_latency_s`) and 15.09 s per call by model_pipeline_s. Simulated robot time is steps/20 Hz: 9,195/20 = 459.8 s, about **10% of control wall time** (*computed*).
2. **Steps per model call:** pooled 56.4, per run 35.4 (t3_005) to 79.1 (t0_001). uc6 managed 4.8 to 5.5 (t3_002, t6_007). Steps per execute decision: median 56.
3. **Call latency is a fixed per-call cost, not a prompt-size cost.** Input tokens grow from about 6.5k to 154k, but Pearson r(input tokens, latency) = −0.13 over 218 pooled calls (*computed* from `requests/*.response.json` usage and the events latencies). Reasoning tokens per call averaged 84 to 318 per run.
4. **The simulator does heavy work on every step, although images feed the model only about once per 56 steps.**
   - Rendering both cameras at 512² (libero.py:868-873).
   - 2 PNG encodes plus 2 validation decodes in the worker (observation.py:24-45, :114-115, :158-160).
   - 4 host-side validation decodes: `from_dict` (libero.py:663), then `CalibratedObservation` construction (sim.py:55-57, :87-90).
   - 2 decodes for the video panels, cached per observation (agent_speedup_v2_video.py:193-206).
   - At decision steps only: 2 more decodes in `check_observation` (direct_history.py:37-38) and the StepView decode for the request.
   - About 613 KB of JSON over the pipe.
   - 2 `check_success` calls (libero.py:936; agent_speedup_v2_sim.py:162).
   - Contact monitoring at each physics substep and an evaluator truth-window update (agent_speedup_v2_sim.py:160-166).
   - An extra `agent_geometry` round trip (sim.py:86).

   Measured: 104.6 ms per step pooled, 85 to 124 ms per run, against a 50 ms real-time step at 20 Hz. The split among these items is **unknown**, because nothing times them separately.
5. **Video.** 2 full 1600x960 renders per step plus raw-frame piping; 90.8 ms per step pooled. Today's offline benchmark on 120 t1_001 steps (`results/speedup_evidence_20261026T055123Z/display_io_benchmark.json`) measured a median 10.69 s for the legacy recorder (about 89 ms per step) and 2.72 s for a compact 640x384 one-frame-per-step recorder (about 23 ms per step), a 74.5% reduction. The `video_render_s` bucket also contains the ≤1 Hz `user_check_framework.json` publication (UC runner:272-280); its share is unknown.
6. **run_state.json** is written 3 times per step, once with full agent_stats; 25.3 ms per step pooled (base runner:997-1014).
7. **Trace** is about 0.6 MB per step, roughly 0.5 GB per run. It costs storage more than time (about 2 ms per step unaccounted).
8. **Preparation and startup.** Preparation takes 19.0 to 32.3 s, of which the plan-draft model call is 11.9 to 24.3 s (`summary.json:user_check.preparation_*`). It starts its own simulator, and execution starts a second one; `startup_to_initial_observation_s` is 4.0 to 7.9 s. Finalization takes about 2.2 to 2.8 s (`elapsed_s − control_elapsed_s − startup`, *computed*). The plan-draft request has no total deadline (section 3.6), and 3 of 16 uc7 preparations stalled for 297 to 634 s.
9. **Convergence tail** (minor, *computed* on 131 sequence-ending arrived stages): a median of 4 (transit) to 9 (contact) steps were spent within 10 mm of the target before the 2-sample arrival predicate fired.

Worked example, t1_001 (`summary.json`):
- service time 21.4 + 416.3 = 437.7 s;
- control 407.2 s = model 182.5 (13 × about 14.0 s) + simulator 108.4 (121 ms/step) + video 81.8 (91 ms/step) + run_state 25.1 + observe 6.4 + about 3.0 more (policy compute 0.9 + unaccounted 2.1);
- the trajectory itself is 897 steps = 44.9 s of simulated time.

---

## 9. What changed uc6 → uc7, and current src vs the uc7 frozen copy

**uc6 (t6_007) → uc7 (t1_001)** (`diff -u` of the two `frozen_source` trees; `fragments.py` exists only in uc7):

1. **New `agent_speedup_v2_user_check_fragments.py`** (389 lines). It adds the `execute` action: 1 to 6 measured-closed-loop stages with transit/contact profiles, 120/240 step budgets and arrival/gap/stall/budget/checkpoint handoffs. This is the main speed change. uc6 had only 1-to-5-step `motion` targets: t6_007 made 82 motion decisions out of 84 at 5.5 steps per call; t3_002 made 87 out of 87 at 4.8. uc7 averages 56.4 steps per call (summaries and plan histories).
2. **Prompt and schema v1→v2** (agent_client.py):
   - `execute` instructions were added, including "do not spend repeated model calls replaying it every five control steps".
   - The uc6 motion advice "use repeated observed segments for longer travel" became "once the route/target is understood, use execute".
   - The schema gained the required `execution` key and the `execute` enum value.
   - The context gained the `execution` key.
   - Compaction moved from a 5000-char summary per key to per-key budgets that prioritise stage outcomes and causes.
   - The macro prompt gained the "about 190-step calibration, a 60-step pause is NORMAL" text.
3. **Policy agent-1→agent-2** (agent.py):
   - fragment mode;
   - **SchemaError is now recoverable feedback** (in uc6 it propagated, which the base runner turns into `subscription_error`, base runner:1533-1534);
   - separate fragment/direct/native action counters;
   - macro status includes `setup_progress`;
   - the hold fallback switches the mode to direct.
4. **Macros** (macros.py): `setup_progress()` (calibration stage detail, 190-step estimate, pause-is-not-failure messaging); runner reset on start; correct action accounting through `_end_step`.
5. **Review validation** (uc.py:98-135): `require_native` defaults to False. Plans with `slide`/`place_in`/`push_to` became approvable and are "executed directly by the model". The native check stays only in `_capability_issues` (client.py:166-168). In the uc6 HTML those options were disabled ("未实现", not implemented). In early uc7 they are labelled "（模型直接执行）", executed directly by the model.
6. **UC runner:** `--auto-approve` (start only) with `approval_mode`/`manual_review` provenance and `human_reviewed_plan=manual_review`; framework version string; `execution_fragments` capability added to `framework_settings`.
7. Late uc7 build B changed only the HTML, from Chinese to English.

**Current `src` vs the uc7 frozen copy.** Comparing all 300 files of `runs/..._t1_001/frozen_source` against the working tree, only 2 differ. They are the files git reports as modified, and `HEAD` equals the frozen copy for both.

```diff
- from .astranomy_agentic_geometry import validate_geometry_metadata          # agent_client.py:30
+ from .robocasa_user_check_geometry import validate_geometry_metadata
- from .astranomy_agentic_geometry import extract_geometry_metadata, validate_geometry_metadata   # sim.py:19
+ from .astranomy_agentic_geometry import extract_geometry_metadata
+ from .robocasa_user_check_geometry import validate_geometry_metadata
```

- The new validator (`robocasa_user_check_geometry.py:44-83`) dispatches on schema. For the LIBERO schema (`astranomy-agentic-panda-camera-v2`) it returns the original validator's result unchanged (:53-54), so LIBERO behaviour should be identical (inference from code). It adds a RoboCasa schema.
- The side effects concern identity and robustness:
  - uc7 now imports `robocasa_user_check_geometry.py`, an **untracked** file (`git status`: `??`).
  - Because `runtime_inventory` hashes all of `src/gpt_field`, a new uc7 run today would carry a different runtime hash and frozen tree. That tree includes 19 untracked files (17 `agent_speedup_v3*`, 2 `robocasa_*`).
  - It is not byte-identical to the 13 uc7 runs.

---

## 10. Module table (uc7 path)

| File | Responsibility in uc7 | Key functions (path:line) | Exercised in uc7 runs |
|---|---|---|---|
| scripts/evaluate_agent_speedup_v2_user_check.py | CLI, preparation, review serving, auto-approve, single-use run, artifact normalisation, live framework publication | `main` 578; `prepare_review` 131; `runtime_inventory` 113; `reviewed_recorder_factory` 237 (`_publish_framework` 248); `execute_reviewed` 426; `_normalize_artifacts` 300 | yes |
| scripts/evaluate_agent_speedup_v2.py (base) | episode loop, timings, run_state, trace, termination, finalization/truth | `execute` 822; `state` 997; loop 1398-1527; `build_summary` 1189 | yes |
| src/gpt_field/agent_speedup_v2_user_check.py | review schema, lossless plan validation, ReviewStore (bind/approve/claim) | `plan_validation` 98; `ReviewStore` 187; `_approval_binding` 240; `approve` 367; `claim` 383 | yes |
| …_user_check_server.py + .html | loopback selector UI | `ReviewServer` 57; handlers 201-252 | yes (manual runs) |
| …_user_check_client.py | `PlanLock`; `UserCheckV2Client` (prep draft); `ReviewedV2Client` (base of the executive); legacy LockedStepRunner/ReviewedV2Policy (unused by uc7) | `PlanLock` 209 (`framework_snapshot` 244, `check_plan` 294); `UserCheckV2Client` 339; `ReviewedV2Client` 394 | PlanLock and prep client yes; legacy runner no |
| …_user_check_agent_client.py | executive prompt/schema/validation, persistent transcript, compaction, retries | `AGENT_INSTRUCTIONS` 51; `agent_schema` 195; `validate_agent_decision` 218; `build_payload` 355; `predict` 393; `request_agent` 451 | yes |
| …_user_check_agent.py | model-owned policy: modes, dispatch, feedback, context, hold fallback, counters | `act` 484; `_ask_agent` 258; `_apply_agent` 305; `_complete_direct` 372; `observe_transition` 572; `_context` 205 | yes |
| …_user_check_fragments.py | measured multi-stage executor | `validate_execution` 80; `FragmentSequencePrimitive` 169 (`_enter_segment` 208, `_next` 246, `_observe` 260, `progress` 355) | yes (99.66% of steps) |
| …_user_check_motion.py + astranomy_agentic_motion.py | direct 1-5-step targets | `LocalMotionPrimitive` 267; `ShortTargetConfig` 103; `validate_target` 144; `ShortTargetExecutor` 188 | yes (41 steps) |
| …_user_check_agent_macros.py | optional native macro adapter | `AgentMacroSession._execute` 87; `setup_progress` 164; `cancel` 284; `invalidate_after_direct` 297 | no (0 grants) |
| …_user_check_sim.py | worker adds 7-joint proprioception and the per-observation `agent_geometry` calibration | `UserCheckLiberoWorker._response` 40; `CalibratedObservation` 52; `UserCheckSimProcess._observation` 84; worker op 117 | yes |
| astranomy_agentic_geometry.py | camera calibration and gripper constants | `extract_geometry_metadata` 143; `_GRIPPER` 33 | yes |
| agent_speedup_v2_sim.py / libero.py / observation.py | LIBERO worker, raw success, rendering, PNG observation | `V2LiberoWorker.step` 155; `_LiberoWorker.step` 919; `image_to_data_url` 24 | yes |
| agent_speedup_v2_video.py / agent_speedup_v1_video.py / video.py | annotated MP4 and events index | v1 `decision` 182, `_emit` 221; video `executed` 227 | yes |
| …_context/_runtime/_control/_spatial/_skills/_local/_constraints/_framework.py | relation-layer extensions and legacy repair/recovery for macros and the uc1-uc5 runner | `build_relation_layer`, `install_spatial_layer`, `StableScene`, `LocalRecoveryFulfiller` (built at reset, agent.py:66-84) | constructed only |
| base v2 relations/planner/observers/scene/geometry/primitives (about 33k lines) | native skills | — | constructed only |

---

## 11. Limits table (every numeric bound on motion, steps or budget)

| Bound | Value | Where | Enforcement |
|---|---|---|---|
| Segments per execute | 1..6 | fragments.py:22, :61, :89-90 | reject |
| max_steps per segment | 1..120 | fragments.py:23, :57, :105-106 | reject |
| Total max_steps per execute | ≤240 | fragments.py:24, :137-138 | reject |
| Transit target envelope (from measured stage entry) | 0.30 m / 1.2 rad | fragments.py:26; :124-125 (relative), :233-239 (all) | reject stage |
| Contact target envelope | 0.08 m / 0.5 rad | fragments.py:29 | reject stage |
| Per-action translation | transit 0.02 m; contact 0.005 m | fragments.py:27, :30, :252-256 | clamp |
| Per-action rotation | 0.1 rad | fragments.py:185 | clamp |
| Arrival tolerance | transit 4 mm / 0.06 rad; contact 2 mm / 0.04 rad; 2 consecutive samples | fragments.py:28, :31, :36, :203-206 | completion |
| set_gripper minimum actions | close 12, open 8 | fragments.py:35, :132-133, :299 | reject/completion |
| Gap-stable predicate | 5 samples, range ≤0.6 mm | fragments.py:37-38, :295-301 | completion |
| Gap-rise interrupt | ≥6 mm above floor, 3 consecutive samples | fragments.py:39-40, :278-289 | interrupt |
| Stall | ≥16 actions; 12-sample window; < 0.8 mm improvement of pos + 0.05·rot | fragments.py:41-43, :311-325 | interrupt |
| Direct target steps | 1..5 | motion.py:76; astranomy_agentic_motion.py:126-127 | reject |
| Direct target distance | ≤0.05 m / ≤0.35 rad | astranomy_agentic_motion.py:104-105, :158-163 | reject |
| Direct per-step | 0.02 m / 0.1 rad | astranomy_agentic_motion.py:106-107, :215-217 | clamp |
| OSC normalisation | 0.05 m and 0.5 rad per unit; clip [−1, 1]; reject per-axis translation > 0.05 m | action.py:17-18, :67-68, :93 | clamp/reject |
| Control rate | 20 Hz | libero.py:870; astranomy_agentic_motion.py:29 | fixed |
| Macro grant | 1..60 actions | agent.py:32, :315-316; agent_client.py:201 | reject |
| Native calibration (nominal) | about 190 actions, not a cap | macros.py:227-229; primitives.py:689 | info |
| Native model calls per act | ≤2 | agent.py:33, :474-475 | defer |
| Transitions per act | 24 | agent_speedup_v2.py:133; agent.py:439, :502 | loop cap |
| Executive calls per act | 1 | agent.py:507-508 | loop cap |
| Image history | 6 pairs | agent_client.py:43 | trim |
| belief / plan / why | ≤1800 / ≤1800 / ≤600 chars | agent_client.py:206-208 | reject |
| Context compaction | trigger > 45,000 chars; budgets 7000/5000/7000/4500 | agent_client.py:484-489 | compact |
| Context hard cap | 56,000 chars | agent_client.py:494-495 | raise (episode ends) |
| Recent feedback in context | last 8; native outcomes last 8 | agent.py:219, :165 | trim |
| Visible output budget | 1400 tokens | agent_client.py:331, :352-353 | statistic only |
| API timeout | 180 s per socket operation; no total deadline | manifest `api.timeout_s`; astra.py:489; chatgpt_plan.py:141 | socket |
| Overload retry | 3 attempts; 2 s base + ≤1 s jitter; ≤60 s wait | agent_speedup_v2_client.py:139-142 | retry |
| Transient retry | ≤2; 2 s base | agent_speedup_v2_client.py:144-145 | retry |
| max_output_tokens / max_input_tokens | 8192 (not sent) / 1,050,000 | manifest `api`; `output_token_limit_sent: None` | none / implicit |
| max_api_calls, max_wall_time_s, max_cost_usd, max_policy_steps | None | manifest | **unbounded** |
| Simulator | seed 7; settle 10; render and policy 512 px; `max_steps=None` (horizon ∞); `response_timeout_s` 120 | agent_speedup_v2_tasks.py:146-151; libero.py:866-867 | config |
| Framework file publish | ≥1 s apart | UC runner:254-256 | throttle |
| Stuck alert | 10 of last 12 failed relation outcomes; re-alert every 12 | base runner:127-131 | alert only |
| Review server | body ≤131,072 B; socket timeout 10 s; approval poll 0.25 s | server.py:24, :97; UC runner:573 | — |

---

## 12. Structural implications (inference, for the bottleneck and next-step analysis)
- Time is dominated by stop-and-think. About 54% of control time is model latency, which behaves as a fixed cost of about 15 s per call. Every fragment end, including 29% that end in a stall, forces a call while the robot is frozen.
- The remaining ~46% is mostly harness overhead per sim step: rendering two 512² cameras with PNG encode/decode every step, a 1600x960 video recorder drawing twice per step, and 3 run_state writes per step. Together this is about 230 ms per step (pooled (control − model)/steps = 2119.4 s / 9195) against 50 ms of simulated time.
- Even with zero model time, t1_001 would need about 225 s of non-model control. Adding preparation of about 21 s and startup of about 7 s leaves little margin under 300 s. S300 therefore needs both fewer or overlapped calls **and** a leaner per-step harness. Any change must keep raw-success semantics and the evaluator's independence.
- The plan-draft call has no total deadline, and 3 of 16 uc7 preparations stalled. This prep-stage risk has nothing to do with manipulation skill.
- Failure runs have no autonomous end. `finish` is advisory and there is no horizon, so the t6 runs ran 711 to 930 s until an operator stopped them.

## Unknowns
- How the ~105 ms simulator step splits between physics, contact monitoring, the truth window, two 512² renders, PNG encode/decode, the pipe transfer and the extra `agent_geometry` IPC. No timer separates them.
- How much of `video_render_s` is the ≤1 Hz `user_check_framework.json` publication.
- Whether the API's per-call latency (about 15 s mean) is dominated by queueing, reasoning or upload of the roughly 4 MB request bodies. Latency does not correlate with input tokens.
- Why the 55 stalls happened (contact, obstruction or controller tracking). The executor deliberately does not record a cause (`contact_or_collision_cause: unknown`).
- The early uc7 build A's exact HTML for t6_008: its preparation stalled before `frozen_source` was written.
