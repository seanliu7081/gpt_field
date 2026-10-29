# agent_speedup_v3: code map of r11 vs r6, gate inventory, and what ended the failed runs

Root for all paths is `/workspace/gpt_field`. "r11" means the current source (`POLICY_VERSION = "agent-speedup-v3-11"`, `src/gpt_field/agent_speedup_v3.py:23`). Its v3 modules are byte-identical to the frozen copy in `runs/agent_speedup_v3_t1_012/frozen_runtime/src` (`diff -rq` lists no v3 file, and `v3_runtime_integrity.json` shows `changed_since_start: []`). "r6" means `runs/agent_speedup_v3_t1_008/frozen_runtime/src` (`POLICY_VERSION = "agent-speedup-v3-6"`), the last successful revision.

I verified the revision behind each run from the frozen copies and `agent_stats.json` (`policy_version`):

| Run | Revision | Geometry class used by the policy |
|---|---|---|
| t1_008 | r6 | GeometryMemory |
| t1_009 | r7 | GeometryMemory |
| t1_010, t7_001, g5_001 | r8 | GeometryMemory |
| g5_002 | r9 | GeometryMemory |
| t1_011 | r10 | LocalGeometryMemory |
| t1_012 | r11 | LocalGeometryMemory |
| completion-baseline attempt_003 | r6 copy, t1 episode 2 | GeometryMemory |

Labels used below: **(U)** inherited from uc7 (`agent-speedup-v2-user-check-7`, `src/gpt_field/agent_speedup_v2_user_check.py:31`); **(V)** added by v3 up to r6; **(7)…(11)** added in that revision.

---

## 1. Architecture of r11

### 1.1 Diagram

```
+--------------------------------------------------------------------------------------------+
| RUNNER  scripts/evaluate_agent_speedup_v3.py                                               |
|  (U) the uc7 runner is loaded as a private module; its client/policy classes are swapped   |
|      (L45-55, L181-231). Manual or explicitly delegated single-use approval (U); no smoke  |
|      runs or step caps (L234-238). Runtime hash freeze (L369-372).                         |
|  (V) performance.json completion_target: 300 s, not a stop condition (performance.py:15,  |
|      127-141). Compact 640x384 video (video.py:16-17).                                     |
+-------------------------------------------+------------------------------------------------+
                                            | one LIBERO episode, 20 Hz
+-------------------------------------------v------------------------------------------------+
| POLICY  AgentSpeedUpV3Policy(AgentFirstV2Policy)                   agent_speedup_v3.py:68   |
|  (U) act(): at most one executive call per act; a rejected answer costs a 1-step hold      |
|      (agent_speedup_v2_user_check_agent.py:499-531). Native macro session (U).             |
|  (V) _context(): adds the shared_geometry summary and the current_decision core            |
|      (plus evidence_acquisition proposals while an intent is pending)          L133-145    |
|  (V) _apply_agent router: program | execute | motion | observe | finish | macro(U)         |
|      geometry_patch/continue_intent -> _continue_intent    acquire_evidence -> view move   |
|      escalate -> EvidenceAcquisitionFailed                             L573-634            |
|  (V) pending intent, repeat guard, continuation guard, 2 view attempts => TERMINAL stops   |
|  (7) _resume_with_local_evidence (L396-447)   (8) act(): binding-conflict stop (L636-646)  |
+----------+-------------------------------+------------------------------+-----------------+
           | request_agent                 | primitives                   | observe_transition
+----------v-----------------+   +---------v-------------------------+  | (geometry first, then
| CLIENT AgentSpeedUpV3Client|   | (V) AdaptiveProgramPrimitive      |  |  the operation runtime;
| (U) AgentFirstV2Client:    |   |  <=24 semantic stages; targets    |  |  L648-663)
|  six image pairs, accepted |   |  derived from OBBs; no XYZ        |  |
|  transcript, v7 7-field    |   |  (8) compact {operations:[1..3]}  |  |
|  schema                    |   |  (7) local recovery, goal updates |  |
| (V) + scene_update,        |   +---------+-------------------------+  |
|  motion_context,           |             | child (one segment at a time)
|  continuation, program;    |   +---------v-------------------------+  |
|  current_decision core     |   | (V) AdaptiveFragmentSequence      |  |
| (10) lean prompt and       |   |     extends (U) FragmentSequence  |  |
|  wire-only history         |   | (V) AdaptiveLocalMotion           |  |
|  projection                |   |     extends (U) LocalMotion       |  |
| (V, from r5) bounded       |   |  tracking evidence, intent        |  |
|  transport: 180 s, output  |   |  watchdog, exact-command verify,  |  |
|  guards                    |   |  prefix/horizon pauses,           |  |
| (4.1) 503 availability     |   |  (7) adaptive contact cap         |  |
|  retry                     |   +---------+-------------------------+  |
+----------------------------+             | assess_motion / verify_command / preview_motion
+------------------------------------------v-----------------------------v-------------------+
| SHARED GEOMETRY  (10) LocalGeometryMemory  extends  (V) GeometryMemory                     |
|  (10) "goal_local": obstacle-core conflicts; blocking_reasons kept separate from           |
|       advisories; execution_allowed; free-region coverage is advisory only                 |
|  (V)  strict sweep / coverage / inventory / candidate checks: still in the code, not       |
|       called in r10-r11                                                                    |
|  (V)  model scene_update; event invalidation (contact, slip, release, appearance);         |
|       attachment/release probes                                                            |
|  (V)  VisualGeometryWatch (fixed external RGB)   (7) LocalRGBGeometryEvidence (stereo)     |
+-------------------------------+---------------------------------------+--------------------+
| (7) OperationRuntime          | (V) InteractionEvidence: two-view     | (V) propose_evidence_views |
|  bound object, close/open     |  hold/release via LK optical flow     |  wrist-camera lateral view |
|  epochs, probe window,        |  (7) v2: per-view re-reference        |  moves; (10)/(11) add a    |
|  recovery budget              |                                       |  blocker/tracking focus    |
+-------------------------------+---------------------------------------+----------------------------+
```

### 1.2 Executive request (client, prompt-10, `CLIENT_VERSION agent-speedup-v3-client-11`; `agent_speedup_v3_client.py:33-35`)

- **Required output fields (11).** The 7 uc7 fields (`belief, plan, why, type, motion, macro, execution`) plus `continuation, scene_update, motion_context, program` (`agent_speedup_v3_client.py:188-200`). The validator is `validate_v3_agent_decision` (L203-241).
- **scene_update** = `{entities[≤12], invalidate_ids, remove_ids}`. Each entity has id, role (obstacle/object/graspable_part/support_surface/destination_region/free_region), parent_id, pose, 1–8 boxes, uncertainty_m (0–2), coverage, attachment (none/held), related_ids and optional anchors (`agent_speedup_v3_geometry.py:39-59`, validated at L93-155). Memory holds at most 24 entities (L28).
- **motion_context** = `{intent ∈ transit/approach/grasp/place/retreat/manipulate/observe, target_id, held_object_id, allowed_contact_ids[≤12], scene_inventory{reviewed, support_ids, nearby_obstacle_ids, unresolved}}` (`agent_speedup_v3_client.py:133-185`).
- **Input built per call.**
  - The uc7 context (approved plan, task contract, calibration, capabilities, measured state, belief, feedback, macro and execution state).
  - `native_evidence.shared_geometry` summary and `operation_state` (`agent_speedup_v3.py:115-119, 133-145`).
  - `fields.current_decision`, inserted outside generic truncation (`agent_speedup_v3_client.py:740-743`). It is built by `_decision_core` (`agent_speedup_v3.py:154-222`) with keys `operation_state, pending_intent, current_requested_action, motion_context, current_tracking, missing_evidence_requests, requirements, whole_goal_assessment, executable_prefix_assessment, blocker, repeated_action_guard, evidence_pause` and, while an intent is pending, `evidence_acquisition`.
  - Logged t1_012 requests carry the user-text keys `request, task, step, robot_pose_wxyz, finger_joints_m, geometry, context, context_detail_compaction, current_decision, wire_projection`.
- **Images.** Six most recent external+wrist pairs (U: `IMAGE_HISTORY = 6`, `agent_speedup_v2_user_check_agent_client.py:43`). t1_012 requests reach `image_count` 12.
- **History projection (10).** `project_agent_history` (`agent_speedup_v3_client.py:587-640`) is wire-only:
  - Older user turns drop `capabilities` and `model_belief`. Their `current_decision` becomes `_historical_decision` and their context becomes `_feedback_events`.
  - Accepted outputs and opaque reasoning are kept. The raw input is restored as accepted history after the call (L690-700).
  - Measured effect, t1_012 request 9: 298,988 → 138,195 prompt characters (`requests/000009-*.request.json` `prompt_sizes`).
- **Instruction and schema size** (from the logged payloads):

| Revision | Instruction chars | Schema chars |
|---|---|---|
| r6 | 25,722 | 7,645 |
| r7 | 27,359 | 7,776 |
| r8 | 18,572 | 10,121 |
| r10–r11 | 10,569 | 10,121 |

- **Input tokens per call.** r6 t1_008 grew from 11,517 to 397,117 by call 40. r11 t1_012 reached 43,836 at call 9 (response `usage`).

### 1.3 Decision types

| type | Handler | Notes |
|---|---|---|
| execute (U) | `AdaptiveFragmentSequencePrimitive` (`agent_speedup_v3.py:615-618`) | 1–6 segments, ≤120 steps each, ≤240 total (U `agent_speedup_v2_user_check_fragments.py:22-24, 80-139`) |
| motion (U) | `AdaptiveLocalMotionPrimitive` (L620-632) | prompt: 1–5 steps, ≤0.05 m / 0.35 rad (`agent_speedup_v3_client.py:268-270`) |
| macro (U) | uc7 native macro (L599-600) | 1..60 budget steps (U agent L315-316) |
| observe / finish (U) | zero-motion hold; finish records a claim (L623-629) | finish does not terminate |
| program (V) | `AdaptiveProgramPrimitive` (L608-614) | `{stages:[1..24]}` (`agent_speedup_v3_program.py:32-46`), or (8) `{operations:[1..3 pick_place]}` (L49-63) compiled to 8 stages per operation (approach, grasp, hold+close, lift_clear+visual_hold, transfer, place, hold+open, retreat+visual_release; L138-171) |
| geometry_patch / continue_intent (V) | `_continue_intent` (L506-546) | resumes the saved intent; resume=false only patches |
| acquire_evidence (V) | `_acquire_evidence` (L289-318) | runtime-chosen viewing move; terminal if 0 candidates |
| escalate (V) | `_stop_for_evidence` (L576-577) | terminal |

### 1.4 GeometryMemory vs LocalGeometryMemory

- **GeometryMemory** (`agent_speedup_v3_geometry.py:404-1870`, version geometry-8; the policy used it through r9).
  - "Supported" requires every sampled hand/finger/payload box along the sweep (≤192 intervals, ≤3 refinements, L22-23) to sit inside the union of model `free_region` boxes eroded by uncertainty (`_union_coverage` L246-286). It also requires no overlap with obstacles, fresh evidence, a resolved inventory for final interactions, and a feasible command candidate (L1559-1817).
  - Anything else is "unknown": a 4-step (contact) or 6-step horizon, ≤1.5 or 2 cm, at most 5 mm / 0.04 rad per command (L1721-1725). Or it is "blocked".
  - Missing evidence is reported as `sweep_coverage:<body>:<region>:<axis>:<face>` plus coverage witnesses (L1737-1741).
- **LocalGeometryMemory(GeometryMemory)** (`agent_speedup_v3_local_geometry.py`, local-geometry-10; the policy has used it since r10, `agent_speedup_v3.py:90`).
  - Overrides `assess_motion`, `verify_command`, `preview_scene_update` and `summary`. Status is always "unknown".
  - Execution is allowed unless `blocking_reasons` is non-empty (L174-279). Free regions are ignored ("Missing free-region boxes never certify or forbid motion", L5-6).
  - Obstacles are checked by "core" (half-extents minus declared uncertainty) with interval-swept boxes (L86-172). The uncertainty shell only produces advisories and a 5 mm speed cap (L151-157, 236-237). Route advice is cached for 8 steps (L23, 99-101).
  - `_sweep_geometry`, `_goal_sweep`, `_command_candidates` and `_interaction_inventory` are never called from the override (grep finds no reference in `agent_speedup_v3_local_geometry.py`), so those gates are inert in r10–r11.
  - Inherited event machinery is still live: `update_from_model`, `observe_transition`, `watch_observation`, probes, `confirm_attachment_evidence`.

### 1.5 Primitives (sweep checker and local planner)

- **AdaptiveFragmentSequencePrimitive** (`agent_speedup_v3_fragments.py:363-628`) and **AdaptiveLocalMotionPrimitive** (L631-831).
  - Each segment entry and each acknowledged step calls `_assess_route`: whole goal plus committed prefix (L246-264).
  - Each emitted coupled command is checked by `verify_command` (L294-298, 500-518).
  - Measured tracking is per channel (`_TrackingEvidence` L79-187). There is a once-learned intent watchdog (L189-200), a stall watchdog (L478-484), and chunk renewal only under allowed, normal, progressing motion (L460-476).
  - (7) `_adaptive_contact_cap` raises the contact command cap from 5 mm toward 20 mm when the estimated target is far away (L324-360).
- **AdaptiveProgramPrimitive** (`agent_speedup_v3_program.py:522-1133`).
  - Derives stage goals from OBBs (`derive_program_goal` L358-519). Runs one child segment at a time (L916-953).
  - Visual hold/release verification uses the operation runtime (L615-657, 960-986).
  - (7) Local recovery: fresh-pose retarget, retreat to a breadcrumb, or a view move. Limited to ≤2 per object (L24, 792-879) and ≤4 goal updates per stage (L25, 770-772).
  - (8) Operation dependency gate (L588-598).

### 1.6 Perception and monitoring

- **VisualGeometryWatch** (V) (`agent_speedup_v3_visual_watch.py`). Uses only the fixed external camera, resized to ≤160 px.
  - A reference is captured on model update (L255-269).
  - "Changed" means more than 18% of visible pixels differ by more than 24 and the mean difference exceeds 10 (L319-321).
  - Visible area below 48 px or below 60% of the projection, or low texture, gives "unverified" (L63, 241-253).
  - It feeds `watch_observation` invalidation (`agent_speedup_v3_geometry.py:750-801`).
- **InteractionEvidence** (V; (7) v2) (`agent_speedup_v3_interaction.py`).
  - Lucas-Kanade (LK) optical-flow tracks in both views compare a hand-rigid hypothesis with a world-stationary one.
  - Hold is supported when separation ≥3 px, held residual ≤ max(1.25, min(3, 0.35·separation)), and the stationary residual exceeds the held residual by more than 1.5 px, in both views for 2 consecutive samples (L169-197).
  - Release requires ≥3 stable samples within 1.25 px (L178-187).
- **OperationRuntime** (7) (`agent_speedup_v3_operation.py`).
  - Keeps a persistent bound object and close/open reference epochs.
  - Starts an attachment probe after a settled close (L223-233).
  - Probe window ≤120 steps and cannot be renewed (L300-329). Recovery claims ≤2 (L369-388).
  - (9) An empty-hand close is treated as posture, not a grasp (L104-124).
- **LocalRGBGeometryEvidence** (7) (`agent_speedup_v3_perception.py`).
  - SIFT stereo references (L131-187). An explicit anchor needs NCC ≥0.85 and passes a uniqueness check (L189-291).
  - Rigid displacement needs ≥6 tracked points, patch NCC ≥0.70, ≥70% RANSAC inliers and spread ≥4 mm (L293-376).
  - Accepted displacements update poses through `apply_visual_displacement` (`agent_speedup_v3_geometry.py:621-715`).
- **propose_evidence_views** (V) (`agent_speedup_v3_observation.py:47-182`).
  - Focus is the first coverage witness. (10) If there is no witness, a blocking entity is used; (11) or the target or observed endpoint after a tracking anomaly (L55-82).
  - Candidates use the wrist camera with 4 lateral directions × {1, 0.5} of a scale of min(0.30, max(2·uncertainty, radius, 0.08·depth)) (L121-124, 141-145).
  - Visibility requires the point in the image, a shift of ≥8 px, and no occlusion by any estimated non-free box (L27-44, 150-153). The move must also pass preview admission (L156-161). Top 3 candidates are returned.

### 1.7 Transport and performance

- **Transport** (V, added r5 per `agent_SpeedUp_v3.md:107-109`; identical r6→r11, diff +0/−0).
  - `V3BoundedSubscriptionTransport` (`agent_speedup_v3_transport.py:224-271`): 180 s total deadline via SIGALRM (L17, 67-89).
  - Output limits: ≤131,072 visible characters (L18), JSON whitespace run ≤2,048 (L19, 92-124), stream ≤64 MiB and event ≤8 MiB (L20-21, 172-201). Partial output is never executed.
- **Performance.** `LiveSnapshotGate` limits live status writes to 1 per second (`agent_speedup_v3_performance.py:14, 31-47`). Video is ffmpeg libx264 at 640×384, one frame per action (`agent_speedup_v3_video.py:16-17`).

---

## 2. Module table (r11 vs r6)

| Module | r11 lines | r6 lines | Role | uc7/base it extends or calls |
|---|---:|---:|---|---|
| `agent_speedup_v3.py` | 693 | 574 | policy; router; pending intent; terminal stops | subclasses `AgentFirstV2Policy` (`agent_speedup_v2_user_check_agent.py:55`) |
| `agent_speedup_v3_client.py` | 767 | 598 | schema, prompt, current_decision, history projection, retry mixin | subclasses `AgentFirstV2Client` and `UserCheckV2Client`; reuses v7 schema/validator |
| `agent_speedup_v3_transport.py` | 271 | 271 | bounded SSE transport | `chatgpt_plan` protocol parser |
| `agent_speedup_v3_geometry.py` | 1870 | 1587 | GeometryMemory: estimates, invalidation, strict sweep, probes | `robot_tool_geometry._local_parts` |
| `agent_speedup_v3_local_geometry.py` | 323 | — (r10) | goal-local admission | subclasses GeometryMemory |
| `agent_speedup_v3_fragments.py` | 831 | 756 | adaptive fragments and motion | subclasses uc7 `FragmentSequencePrimitive` and `LocalMotionPrimitive` |
| `agent_speedup_v3_program.py` | 1133 | 478 | semantic and compact programs, recovery | `contact_skills.Primitive`; uc7 `PROFILES` |
| `agent_speedup_v3_operation.py` | 408 | — (r7) | persistent operation state | uc7 gap constants |
| `agent_speedup_v3_interaction.py` | 219 | 187 | two-view hold/release | — |
| `agent_speedup_v3_perception.py` | 381 | — (r7) | stereo displacement and anchors | — |
| `agent_speedup_v3_visual_watch.py` | 325 | 324 | external-RGB change watch | — |
| `agent_speedup_v3_observation.py` | 182 | 135 | view proposals | uc7 `PROFILES` and `MAX_SEGMENT_STEPS` |
| `agent_speedup_v3_performance.py` | 141 | 141 | live-snapshot throttle, 300 s report | patches the private uc7 evaluator copy |
| `agent_speedup_v3_video.py` | 192 | 192 | compact video | `video.MP4Stream` |
| `agent_speedup_v3_review.py` | 99 | 99 | v3-labelled review store | loads `agent_speedup_v2_user_check.py` |
| `agent_speedup_v3_server.py` | 15 | 15 | review UI | loads `agent_speedup_v2_user_check_server.py` |
| **total** | **7,850** | **5,357** | +2,493 lines (+46.5%) | uc7 user_check modules total 6,955 lines |

---

## 3. Gate inventory: everything that blocks, shortens or vetoes an action, or forces control back to the model

**Classes.**
- **TERM**: terminal stop (raises `EvidenceAcquisitionFailed` → `status=evidence_acquisition_failed`, `scripts/evaluate_agent_speedup_v3.py:283-288`; or an unretried exception).
- **VETO**: the action or gripper command is not executed.
- **SHORT**: step size, prefix or horizon is reduced.
- **HAND**: control returns to the model.
- **ADV**: advisory only.

**What the model receives.** Unless stated otherwise, a pause gives the model `last_feedback` (`{action, status, reason, executed_steps, evidence summary}`) plus `current_decision` (`blocker{status, reason, execution_allowed, blocking_reasons, advisories, …}`, `missing_evidence_requests`, `requirements`, `pending_intent{current_target, remaining_watchdog_steps, …}`, `repeated_action_guard`, `evidence_acquisition{candidates, reason, rejected}`).

### 3.1 Terminal stops (policy)

| # | Gate | r11 path:line | Trigger | Model receives | Class | r6? |
|---|---|---|---|---|---|---|
| T1 | model escalation | `agent_speedup_v3.py:576-577` → `_stop_for_evidence` L369-375 | model returns `type=escalate` | — (run ends; `evidence_pause` holds the blocker) | TERM | yes (r6 L475) |
| T2 | `targeted_view_unavailable` | L300-302 | `acquire_evidence` when `_propose_views` has 0 candidates, or ≥2 view attempts for this intent | — (the previous request already showed `evidence_acquisition.reason` and the candidate count) | TERM | yes (r6 L256-258) |
| T3 | `repeated_blocked_action_without_new_relevant_evidence` | L386-387 | second unchanged retry: same action identity, same blocker identity, same relevant evidence, and the patch does not change the path (`_patch_changes_path` L234-260) | — | TERM | yes (r6 L341) |
| T4 | `repeated_invalid_continuation` | L499-500 | second consecutive invalid continuation | — | TERM | yes (r6 L397-399) |
| T5 | `active_manipulated_object_binding_conflict` | L644-645, `agent_speedup_v3_operation.py:130-133` | a close/open request for a different object while the bound object is mid-operation | — | TERM | **added r8** |
| T6 | transport deadline / output guard | `agent_speedup_v3_transport.py:17-21, 67-201` | >180 s total; >131,072 visible chars; whitespace run >2,048; >64 MiB stream; >8 MiB event | — (exception). These codes are not in `TRANSIENT_ERROR_CODES` (`agent_speedup_v2_client.py:149-150, 903-914`), so uc7 `_ask_agent` re-raises (`agent_speedup_v2_user_check_agent.py:275-280`) | TERM (run error) | yes (unchanged r6→r11) |

### 3.2 Executive vetoes and handoffs (policy and client)

| # | Gate | r11 path:line | Trigger | Model receives | Class | r6? |
|---|---|---|---|---|---|---|
| H1 | schema/semantic rejection | `agent_speedup_v3_client.py:203-241`; uc7 `agent_speedup_v2_user_check_agent.py:263-274` | invalid fields, routes or continuation; execution limits (U fragments L80-139) | `{action: model_response, status: rejected, reason: invalid_action_arguments, detail}`, then a 1-step hold (U L506-531) | VETO+HAND | yes (schema extended r8) |
| H2 | repeat guard, first rejection | `agent_speedup_v3.py:377-394`; armed L481-492; checked L513-515, 581-588 | same action and blocker after a `geometry_*` or `program_geometry_*` pause, with no new relevant evidence (`_relevant_evidence` L320-367) | `{action: rejected_repeat, status: evidence_required, reason: unchanged_blocker_requires_new_evidence_or_intent, required_route: acquire_evidence_or_change_intent}`; `repeated_action_guard.new_evidence_required=true` | VETO+HAND | yes (r6 L332-349) |
| H3 | continuation rejections | L292-297, 507-511, 535-536 → L497-504 | wrong `intent_id`; changed protected context; scene_update attached to `acquire_evidence`; budget/target mismatch (`agent_speedup_v3_fragments.py:208-220`, `agent_speedup_v3_program.py:1125-1129`) | `{action: continuation, status: rejected, reason, detail, executed_steps: 0, required_route}` | VETO+HAND | yes |
| H4 | patch without resume | L521-525 | `geometry_patch` with `resume=false` | `{action: geometry_patch, status: patched_without_motion}` | HAND | yes |
| H5 | evidence-view return | L450-465 | the view move ends | `{action: acquire_evidence, status: fresh_view_requires_interpretation, original_goal_resumed: false}` | HAND | yes |
| H6 | every primitive end | uc7 `_complete_direct` L372-401 via L495 | any pause or finish; the only bypass is H7 | per-primitive feedback | HAND | yes |
| H7 | local-evidence resume (bypass) | L396-447 | accepted local pose or interaction event at this step, and a re-preview is allowed | `{action: local_evidence_resume}` | (removes a HAND) | **added r7**; fired 0 times in r7–r11 runs (every `continuation_events` list) |
| H8 | native macro yields (U) | uc7 agent L310-333, 437-482 | macro rejected or unavailable, observer preemption, primitive failure, RepairNeed, budget | `native_macro` feedback | HAND | uc7 |

### 3.3 Live geometry admission in r10–r11 (`LocalGeometryMemory._evaluate`)

In this layer `missing_evidence_requests` equals `blocking_reasons` (`agent_speedup_v3_local_geometry.py:253`).

| # | Gate | r11 path:line | Trigger | Model receives | Class | r6? |
|---|---|---|---|---|---|---|
| L1 | `obstacle_core:<id>:<body>` | `agent_speedup_v3_local_geometry.py:114-165, 200` | a swept body box penetrates an obstacle's core (half minus uncertainty) by more than 1e-6 m, the pair is not allowed (L66-84), and it is not an escape from an existing overlap | `blocker.blocking_reasons`; feedback `geometry_predicted_block` | VETO+HAND | **added r10** |
| L2 | `measured_tracking_mismatch` | L202-205; source `agent_speedup_v3_fragments.py:79-187` | 3 valid drive samples with response < −0.05, > 2.5, or < 0.03 without improvement, in 2 consecutive anomalous windows | blocking reason; `current_tracking` | VETO+HAND | tracking existed in r6 (GeometryMemory revoked motion, r6 geometry L1400/1415); as a blocking reason, r10 |
| L3 | `payload_shape_required:<held>` | L206-208 | `held_object_id` set but no attached body | blocking reason | VETO+HAND | r10 (r6 analogue: `payload_geometry_unknown` → unknown) |
| L4 | `attachment_verification_required` | L209-211 | an unconfirmed attachment probe outside its finite window, and not a stationary gripper action | blocking reason | VETO+HAND | r10 (probes from r7) |
| L5 | `changed_geometry:<id>` | L212-214 | a relevant entity is stale with `visual_appearance_changed` or `possible_slip` | blocking reason | VETO+HAND | r10 (r6: stale → unknown, SHORT) |
| L6 | `grasp_target_shape_required` | L217-219 | grasp close without an object-role target | blocking reason | VETO | r10 |
| L7 | `current_public_rgb_required` | L220-224 | motion without a current image | blocking reason | VETO | r10 |
| L8 | controller translation/rotation limit | L226-233 | exact command exceeds 0.02 m or 0.1 rad | blocking reason | VETO | r10 |
| L9 | `estimated_obstacle_margin` | L151-157, 236-237 | nominal gap within uncertainty; cap 5 mm when the gap is ≤0 | `advisories` | ADV+SHORT | r10 |
| L10 | `unmeasured_free_space` | L201 | always present | `advisories` | ADV | r10 |
| L11 | exact-command verify | L291-316 (blocker copied into `last_motion` L307-312) | the emitted coupled command is not allowed | pause `geometry_predicted_block` | VETO+HAND | yes (strict version, `agent_speedup_v3_geometry.py:1232-1296`; r6 L990) |

### 3.4 Strict GeometryMemory admission (live r6–r9; inert in r10–r11)

| # | Gate | path:line | Trigger | Model receives | Class | r6? |
|---|---|---|---|---|---|---|
| S1 | sweep coverage | `agent_speedup_v3_geometry.py:1298-1432, 1644, 1686, 1721-1727, 1737-1741` | any sample body not contained in the eroded free-region union | `missing_evidence_requests` (`sweep_coverage:…`), `coverage_witnesses`, `unknown_fraction` | SHORT (4/6-step horizon, 1.5/2 cm, 5 mm/cmd) then HAND | yes (r6 L1449, 1464) |
| S2 | estimated overlap | L1647-1649 | new or worsening overlap, or a starting overlap without escape | `blocked`, `estimated_obstacle_overlap` | VETO+HAND | yes (r6 L1379) |
| S3 | interaction inventory | L1154-1203, 1658-1661, 1688-1689 | final approach/grasp/place within 0.05 m, any gripper action, closing, or stationary release. Requires `reviewed=true`, `unresolved=false`, a fresh target, a support surface within 0.06 m, and fresh listed obstacles | `geometry_inventory_required`, `current_target_envelope:*`, `nearby_support_surface_for:*` | VETO+HAND | yes (r6 L917-961); r7 exempts stationary release and departure probe |
| S4 | tracking revocation | L1670-1676, 1688-1689 | tracking mismatch or lost | `blocked`, `measured_tracking_mismatch` | VETO+HAND | yes |
| S5 | stale / monitoring-unknown / payload-unknown | L861-865, 1683-1696 | relevant evidence stale or unknown | unknown, prefix 0 | SHORT+HAND | yes |
| S6 | no feasible command candidate | L1487-1557, 1709-1715 | all 4 scales fail (1, 0.5, 0.25, 0.1 of 2 cm / 0.1 rad; 1 cm / 0.06 rad when tracking is not normal) | `local_command_sweep_not_supported` | VETO+HAND | yes (r6 L1243, 1441) |

### 3.5 Scene-event invalidations that feed the gates (inherited, live in r11)

| # | Event | path:line | Trigger | Effect | Class | r6? |
|---|---|---|---|---|---|---|
| E1 | predicted possible contact | `agent_speedup_v3_geometry.py:917-933` | measured hand box overlaps an object/obstacle box padded by its uncertainty | entity stale; "local" only for allowed finger contact; intersecting free regions stale | feeds P1/P2, S3, S5 | yes (r6 L730) |
| E2 | slip | L914-916 | held-object gap change >6 mm | `possible_slip` → L5 | feeds VETO | yes |
| E3 | release command | L896-913; (8) provenance L577-619 | open while holding | `release_pending`; carried and world alternatives | feeds L1/S1 | yes (+r8) |
| E4 | appearance change | L750-801; `agent_speedup_v3_visual_watch.py:280-325` | external-RGB change; (7) absorbed only by accepted local displacement (L776-781) | stale + `monitoring_unknown` → L5, P1 | feeds VETO | yes |
| E5 | model revision | L508-516 | changed pose, boxes, attachment, parent, role, or larger uncertainty | stale dependents | feeds | yes |

### 3.6 Fragment and motion primitives

| # | Gate | path:line | Class | r6? |
|---|---|---|---|---|
| F1 | entry block, fraction 0 → `geometry_predicted_block` | `agent_speedup_v3_fragments.py:440-441` / 675-677 | VETO+HAND | yes (r6 L389) |
| F2 | emitted command not allowed | L515-516 / 712-714 | VETO+HAND | yes (r6 L457-458, 644) |
| F3 | blocked during motion | L551-552 / 775-776 | VETO+HAND | yes (r6 L490) |
| F4 | prefix truncation → `geometry_prefix_observed` | L554-556, 570-571 / 769-778 | SHORT+HAND | yes (r6 L509) |
| F5 | unknown horizon → `geometry_observation_horizon` | L67-76, 557-564, 572-573 / 779-780 | SHORT+HAND | yes (r6 L511) |
| F6 | stall: <0.8 mm in 12 steps after ≥16 steps | L478-484; U `agent_speedup_v2_user_check_fragments.py:41-43, 311-325` | HAND | yes |
| F7 | intent watchdog (learned once) | L189-200, 453-458 | HAND | yes |
| F8 | chunk renewal requires allowed motion, normal tracking and progress | L460-476 | HAND | yes (r6 required "supported") |
| F9 | U envelope: transit 0.30 m / 1.2 rad, contact 0.08 m / 0.5 rad | U fragments L230-239 | VETO+HAND | uc7 |
| F10 | U grip-gap rise ≥6 mm ×3 samples | U fragments L278-289 | HAND | uc7 |
| F11 | U direct-motion target rejection | U `agent_speedup_v2_user_check_motion.py:292-302` | VETO+HAND | uc7 |

### 3.7 Program

| # | Gate | path:line | Model receives | Class | r6? |
|---|---|---|---|---|---|
| P1 | stage-entry `MissingGeometry` → `program_geometry_required` | `agent_speedup_v3_program.py:700-704`; kinds L239-251, 367-513 | `pending_intent.missing_evidence_requests` (e.g. `fresh_geometry:milk`); `stage_goal=None` makes `current_target=None` (L1116-1117), which **disables view planning** (`agent_speedup_v3.py:266-268`) | HAND (view route closed) | yes (r6 L308). r10 added `unresolved_observed_geometry_change`, `support_beneath_grasp_target`, `support_clearance_and_grasp_contact_inconsistent` |
| P2 | freshness of every inventory member | L372-379 via `_entity` L236-252 (`_conditional_shape_usable` L220-233) | as P1 | HAND | yes (r6 also required reviewed/unresolved, r6 L123) |
| P3 | visual hold before transfer/place | L695-699 | `visual_common_motion:<id>` | HAND | yes (r6 L303) |
| P4 | visual verification at stage end | L960-968 | `visual_hold:<id>` or `visual_release:<id>` | HAND | yes (r6 L366) |
| P5 | previous operation unverified | L588-598 | `accepted_release:<id>` | HAND | **added r8** |
| P6 | program watchdog | L919-922, 1003-1004, 1065-1066 | `program_watchdog_exhausted` | HAND | yes |
| P7 | child pause → local recovery or `program_<reason>` | L955-958, 1037-1050; recovery L830-914; ≤2 per object (L24, 798; `agent_speedup_v3_operation.py:369-388`) | `program_local_recovery_requires_geometry`, `program_local_view_requires_evidence`, … | HAND | recovery **added r7** (r6 paused directly, r6 L357, 416) |
| P8 | goal-update budget (4 per stage) | L770-772 | `stable_target_pose:<id>` | HAND | **added r7** |

### 3.8 Operation runtime (7)

- **O1** binding conflict → T5 (`agent_speedup_v3_operation.py:130-133`, stop since r8).
- **O2** hold/release evidence preconditions (L253-298): acknowledged command, settled close, release revokes hold, geometry confirmation. These feed P3 and P4.
- **O3** finite probe window (L300-329; `agent_speedup_v3_geometry.py:999-1013`). Outside the window, L4 applies.
- **O4** (9) empty-hand posture is not a grasp (L104-124).

### 3.9 View planner: the only route out of an evidence pause

| # | Zero-candidate reason | path:line | r6? |
|---|---|---|---|
| V1 | `no_pending_goal` / `semantic_goal_requires_geometry_before_view_planning` | `agent_speedup_v3.py:264-268` | yes (r6 L231-233) |
| V2 | `no_spatial_coverage_witness` | `agent_speedup_v3_observation.py:49, 55-84`; (10) blocking-entity focus; (11) tracking focus | yes (r6 L49) |
| V3 | wrist camera missing / point behind camera / bad metadata | L86-103 | yes |
| V4 | visibility rejection: out of image, <8 px shift, or occluded by any estimated non-free box | L141-153, 27-44 | yes (r6 L112) |
| V5 | motion not admitted by preview | L156-161 | yes (r6 required "supported") |
| V6 | `payload_geometry_required_before_view_motion` | r6–r9 `agent_speedup_v3_observation.py:75` | **removed r10** |
| V7 | ≥2 attempts per intent (→ T2) | `agent_speedup_v3.py:300` | yes |

Approval, step caps and runtime hash checks (`scripts/evaluate_agent_speedup_v3.py:234-238, 364-372`) are run-level controls, not motion gates.

---

## 4. Gates in the terminal reasons of the failed runs

Primary sources: `summary.json` (status and error), `agent_stats.json` (`evidence_pause`, `operation_state`, `continuation_events`), and the last two `requests/*.{request,response,result}.json` files.

- Every row has `status = failure_reason = evidence_acquisition_failed`. The summary `error.message` is always the generic "Diagnostic stopped; no retry or API-key fallback"; the specific reason is in `agent_stats.json.evidence_pause.reason`.
- All last-two requests returned `status: valid`.
- The longest single request across these runs was 119.5 s (t1_008 call 1), so T6 never fired.

| Run (rev) | Terminal route | Last two requests: what the model received → what it answered | Root gate chain | View planner at the final call |
|---|---|---|---|---|
| t1_009 (r7), step 743 | T1 escalate | Step 742: feedback `geometry_predicted_block` (0 steps executed); blocker `blocked/estimated_obstacle_overlap`, `payload_geometry_unknown=true`, 2 `sweep_coverage` requests → **execute** `set_gripper open`. Step 743: same blocker; feedback `geometry_predicted_block` with 0 steps → **escalate** | O2 (B release never accepted: 14 raw release samples, 14 confirmation rejections, `release_supported=0`) → S5 payload-unknown → S2 + S1 vetoed even a stationary open (F1/F2) | V6 `payload_geometry_required_before_view_motion`, 0 candidates |
| t1_010 (r8), step 190 | T1 | Step 142: `fresh_view_requires_interpretation` (view move, 26 actions) → **geometry_patch** resume. Step 190: feedback `program_local_recovery_requires_geometry`; 6 `sweep_coverage` requests (fingers, palm); `unknown_fraction` 0.850 → **escalate** | S1 + S2 on the grasp descent → P7 (2 of 2 recovery claims used) | V4 `no_supported_informative_view`; rejected {visibility 8, motion 0} |
| t7_001 (r8), step 176 | T1 | Step 141: `program_local_recovery_requires_geometry` → **execute** (own 2-segment descent). Step 176: `geometry_prefix_observed`; blocked overlap; 2 coverage requests; `unknown_fraction` 0.277 → **escalate** | S1 + S2 (palm/finger next to a neighbouring can) → F4 → P7 (1 claim) | V4/V5: {visibility 6, motion 2} |
| g5_001 (r8), step 21 | T1 | Step 20: `geometry_predicted_block` (3 steps) → **execute** `set_gripper open`. Step 21: blocked, 0 steps executed → **escalate** | Empty-hand pushing close created a plate attachment probe (O3; `attachment_probes`=1) → payload unknown → S2 + S1 vetoed translation and the stationary open | V6, 0 candidates |
| g5_002 (r9), step 71 | T1 | Step 65: feedback `rejected_repeat` (H2 at step 64) → **execute** with a new path. Step 71: `geometry_observation_horizon` after 6 steps; `unknown/unobserved_or_model_uncovered_sweep`; 1 palm coverage request → **escalate** | S1 → F5 horizon, H2 | V4 {visibility 8} |
| t1_011 (r10), step 194 | **T2** | Step 170: `program_local_recovery_requires_geometry` (`no_alternative`) → **continue_intent**. Step 194: feedback `program_geometry_predicted_block`. `current_tracking.status=mismatch` (translation response 1.06e-4 at 8.0 mm error); `executable_prefix_assessment` blocked by `measured_tracking_mismatch`; **but** `blocker.execution_allowed=true`, `blocking_reasons=[]` → **acquire_evidence** | L2 (descent stall) → L11/F2 → P7 no alternative. The blocker is a recomputed preview without tracking (`agent_speedup_v3.py:137-141`) | V2 `no_spatial_coverage_witness`, 0 candidates (already shown to the model) → `targeted_view_unavailable: no_spatial_coverage_witness` |
| t1_012 (r11), step 431 | **T2** | Step 298: feedback `rejected_repeat` (H2 at step 297) → **geometry_patch**(milk) resume. Step 431: feedback `program_geometry_required`; missing `fresh_geometry:milk`; stage 3 `lift_clear`; `current_target` none → **acquire_evidence** | E1: hand/padded-OBB overlap staled `milk` at step 396 (`possible_contact`, `invalidation_local_contact=false`; `agent_stats.json` events) → P2 (milk listed in `nearby_obstacle_ids`) → P1 → V1 | V1 `semantic_goal_requires_geometry_before_view_planning`, 0 candidates (already shown) → T2 |
| attempt_003 (r6, t1 episode 2), step 313 | T1 | Step 312: `geometry_predicted_block` → **geometry_patch**(A, A_approach_air). Step 313: `blocked/geometry_inventory_required`: `current_target_envelope:A` (A stale) plus 4 finger `sweep_coverage` requests → **escalate** | S3 + S1 | V4 {visibility 8} |

**Gate families at the root of the 8 terminal chains**

| Family | Runs | Count |
|---|---|---:|
| strict sweep coverage (S1) | t1_009, t1_010, t7_001, g5_001, g5_002, attempt_003 | 6 |
| strict overlap (S2) | t1_009, t1_010, t7_001, g5_001 | 4 |
| inventory (S3) | attempt_003 | 1 |
| payload/attachment uncertainty (S5/O2/O3, with V6) | t1_009, g5_001 | 2 |
| tracking (L2) | t1_011 | 1 |
| predicted-contact staleness → inventory freshness (E1→P2→P1) | t1_012 | 1 |
| local recovery exhausted or without alternative (P7) before the end | t1_010, t7_001, t1_011 | 3 |
| zero view candidates at the final call (V1/V2/V4/V5/V6) | all | 8 |
| terminal route T1 (escalate) | — | 6 |
| terminal route T2 (acquire_evidence with 0 advertised candidates) | — | 2 |
| T3–T6 | — | 0 |

**Calibration checks**

- **Was the t1_012 staleness justified?** The evaluator-only audit `runs/agent_speedup_v3_t1_012/collision.json` records the first robot–object contact `milk_1_g1` ↔ `gripper0_hand_collision` at action_index 386. The policy invalidated milk at step 396. The invalidation therefore followed a real palm contact; this assumes step and action indices are comparable, which I did not verify.
- **Why did all 8 visibility candidates fail?** (Inference, from `agent_speedup_v3_observation.py:27-44` plus recomputed box margins.) In t1_010 the focus witness lies 15.3 mm inside the uncertainty-padded table box. In attempt_003 it lies 31.2 mm inside the padded stale target A box. `_occluded()` returns true for any ray ending inside an estimated box, so every candidate fails visibility. For t7_001 and g5_002 the witness is not within 1 cm of any box, and the rejection cause is unknown.
- **Did the local-evidence paths ever fire?** Executed view moves before the end occurred only twice: t1_009 step 62 and t1_010 step 116 (`evidence_view_attempts` = 1 each). No request log in r7–r11 mentions an accepted displacement or anchor (`two_view_rigid_material_displacement` / `explicit_local_anchor_triangulated`: 0 mentions). `local_evidence_resume` fired 0 times.

**Which gate preceded each model call** (from `last_feedback` of every request):

| Run | Follow-up calls | After a geometry pause, recovery pause or repeat rejection | After normal `fragment_complete` |
|---|---:|---:|---:|
| t1_008 (r6, success) | 39 | 36 (predicted_block 11+1, observation_horizon 11+1, prefix_observed 8+2, rejected_repeat 2) | 3 |
| t1_009 | 24 | 17 | 5 (+1 `observation_required`, +1 view interpretation) |
| t1_012 | 8 | 7 (4 `program_geometry_required`) | 1 |

**Timing** (`summary.json` `timing_components`; service time from `performance.json`):

| Run | Model pipeline (s) | Share of control time | Mean per call (s) | Service total (s) |
|---|---:|---:|---:|---:|
| t1_008 | 1,336.5 | 0.839 | 33.4 | 1,626.2 |
| t1_009 | 721.4 | 0.802 | 28.9 | 936.0 |
| t1_010 | 358.3 | 0.719 | 51.2 | 535.9 |
| t7_001 | 385.1 | 0.886 | 48.1 | 463.2 |
| g5_001 | 152.9 | 0.952 | 30.6 | 201.2 |
| g5_002 | 162.1 | 0.814 | 27.0 | 242.4 |
| t1_011 | 135.9 | 0.788 | 45.3 | 214.8 |
| t1_012 | 354.7 | 0.814 | 39.4 | 469.4 |
| attempt_003 | 581.8 | 0.861 (0.845 of `T_control` 688.65) | 44.8 | 1,007.4 |

---

## 5. v3 = uc7 + which layers

**Reused or subclassed from uc7**

- **Policy loop.** `AgentSpeedUpV3Policy(AgentFirstV2Policy)`, which itself subclasses `agent_speedup_v2.AgentSpeedUpV2Policy`. v3 inherits:
  - `act()` with one executive call per act and a hold fallback;
  - `_ask_agent` (SchemaError handling) and `_complete_direct`;
  - the native macro session (`AgentMacroSession`, `LocalRecoveryFulfiller`), `StableScene`, `WholeObjectHeldFrameEstimator`, and the relation layer (`agent_speedup_v2_user_check_agent.py:66-113`).
- **Overridden in v3:** `reset`, `_capabilities`, `_native_evidence`, `_execution_state`, `_context`, `_complete_direct`, `_sync_framework`, `_apply_agent` (macro delegated to super at L599-600), `act`, `observe_transition`, `agent_stats`.
- **Primitives.** `AdaptiveFragmentSequencePrimitive(FragmentSequencePrimitive)` and `AdaptiveLocalMotionPrimitive(LocalMotionPrimitive)`. They reuse uc7 `PROFILES`, `MOTION_SKILLS`, `STALL_*`, gap and gripper constants, and `validate_execution`.
- **Clients.**
  - `AgentSpeedUpV3Client(V3SubscriptionAvailabilityRetryMixin, AgentFirstV2Client)` keeps uc7 transcript handling, six image pairs and same-request retry. It replaces the instructions and schema and adds `current_decision` and the projection (`agent_speedup_v3_client.py:660-767`).
  - `AgentSpeedUpV3PlanClient(mixin, UserCheckV2Client)` is the unchanged draft-plan protocol plus compact-output text (L121-130).
- **Review and server.** `importlib` loads `agent_speedup_v2_user_check.py` and `agent_speedup_v2_user_check_server.py` and relabels them (`agent_speedup_v3_review.py:21-31`, `agent_speedup_v3_server.py:8-15`).
- **Runner.** A private copy of `scripts/evaluate_agent_speedup_v2_user_check.py` (which builds clients at L183/466 and the policy at L480). Its `AgentFirstV2Client`, `AgentFirstV2Policy` and `UserCheckV2Client` globals, review store, labels, recorder factory, inventory and normalization are swapped or wrapped (`scripts/evaluate_agent_speedup_v3.py:181-231, 356-413`).

**Layers v3 adds on top of uc7**

1. Extended action contract: `scene_update`, `motion_context`, `continuation`, `program`, and 5 new decision types.
2. Persistent shared geometry: GeometryMemory, replaced in the policy by LocalGeometryMemory from r10.
3. Adaptive primitives: sweep and exact-command checks, tracking, watchdogs, prefixes and horizons.
4. Semantic programs; compact operations from r8; local recovery from r7.
5. Pending intent, repeat and continuation guards, evidence acquisition and escalate, all ending in the terminal `EvidenceAcquisitionFailed`.
6. Visual watch.
7. Two-view interaction monitor.
8. Operation runtime (r7).
9. Local stereo perception (r7).
10. Bounded transport (r5) and 503 retry (r4.1).
11. History projection (r10).
12. Performance and video adapters.

---

## 6. Change list r6 → r11 (verified in the frozen copies)

- **r7** (`agent-speedup-v3-7`, t1_009)
  - New `agent_speedup_v3_operation.py` and `agent_speedup_v3_perception.py`; interaction-2.
  - geometry-7: local displacement acceptance, probe ids (dual hypothesis), stationary-release and departure-probe exemptions, watch re-entry.
  - program-2: local recovery (`MAX_LOCAL_RECOVERIES` 2, `fresh_pose_retarget`) and goal updates (4).
  - fragments-6: adaptive contact cap.
  - Policy: `_resume_with_local_evidence`. prompt-7 (27,359 chars).
- **r8** (t1_010, t7_001, g5_001)
  - Compact `{operations}` contract and `program_previous_operation_unverified` (program-3).
  - geometry-8: `_preserve_pending_release`.
  - Policy: binding-conflict terminal stop. prompt-8 (18,572 chars; schema 10,121).
- **r9** (g5_002): operation runtime distinguishes empty-hand posture from grasp and keeps an unresolved release reference; client-9 (prompt unchanged).
- **r10** (t1_011)
  - The policy switches to `LocalGeometryMemory` (local-geometry-10): coverage becomes advisory, obstacle-core conflicts are added, blocking reasons are separated from advisories.
  - fragments-10: `_execution_allowed`.
  - program-4: goal-local `_entity` and `unresolved_observed_geometry_change`; grasp support and pad-closing consistency.
  - Observation: payload veto removed, blocking-entity focus, admitted (not only "supported") candidates.
  - Lean prompt-10 (10,569 chars) and `project_agent_history`.
- **r11** (t1_012): tracking mismatch is carried into view proposals (`agent_speedup_v3.py:269-283`); robot-endpoint focus (`agent_speedup_v3_observation.py:52-82`); `obstacle_core` id parsing (`agent_speedup_v3.py:161-162`); client-11.

---

## 7. Implications for the bottleneck question

The facts above are verified; the conclusions in this section are inference.

1. **Every v3 failure exits through one funnel.** A gate pauses the primitive, the model is called, the view planner returns zero candidates (8 of 8), and the model escalates (T1) or calls `acquire_evidence` anyway (T2). There is no fallback that keeps the run alive. T2 is terminal by construction even when the request already advertised 0 candidates; this happened in r10 and r11.
2. **Gates are the main driver of model calls.** In the only r6 success, 36 of 39 follow-up calls came after geometry or repeat gates, at 33.4 s mean per call, which put 83.9% of control time in the model. Under this design S300 is out of reach whenever a task needs more than a handful of handoffs.
3. **r10 moved the strictness rather than removing it.**
   - Coverage stopped vetoing.
   - But each hold, attachment, tracking or changed-shape uncertainty became a hard `blocking_reason`.
   - Program goal derivation still requires every inventory neighbour to be fresh (P2).
   - View planning requires a derived goal (V1).
   - Two new terminal modes followed: t1_011 (tracking stall with no witness) and t1_012 (stale neighbour, no goal, no view).
4. **The blocker shown to the model can disagree with the actual pause cause.** The pending-goal preview drops tracking (`agent_speedup_v3.py:137-141`). r11 fixed this only inside view planning.
5. **The evidence channels the gates ask for were never produced locally.** No accepted local pose update, displacement or anchor appears in r7–r11, so in practice only model patches or view moves can clear them, and both are themselves gated (H2, V4).
