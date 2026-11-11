# H1 obstacle harness (uc7.1, 100 mm transit): zero-shot generality and first-try robustness

Analyst key `h1_zero_shot`, 2026-11-07. Read-only analysis. No simulator, model, Astra or GPU work was run, and nothing under `/workspace/gpt_field` was edited. Small offline scripts over saved artifacts are in `h1_zero_shot_work/scripts/` next to this file. The per-call table they produced is `h1_zero_shot_work/calls_table.json`.

Code paths are relative to `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/`. Run paths are relative to `/workspace/gpt_field/runs/`.

## Summary

- **The H1 policy has almost no evidence on the tasks it claims to support.** The exact H1 policy code (every `agent_speedup_v2_user_check_*` module, by hash) was used in 5 completed live runs and 1 launch that aborted. All 5 are added-obstacle or composite scenes: T04 + can, T08 + bottle, T08 stove-off + bottle, and the custom drawer/bowl/moka task twice. None is one of the 21 registered official tasks.
  - Raw LIBERO success: 5/5.
  - Strict physical success at the terminal step: 0/5.
  - Calls per run: 14–43.
  - Control time without video: 287–1249 s. None is under 180 s.
- **The evaluator acts as a two-sided oracle.**
  - The episode stops at the first raw success: 7/7 uc7.1 successes stopped with the last object still held.
  - The prompt tells the agent to treat "the episode continues" after `finish` as a sign that it is not done. T04 at 20 mm used exactly that cue to fix a 3–4 cm misplacement.
  - There is no `give_up` and no cap, so failures never end by themselves.
- **The initial plan is a blocking, unrecoverable gate built for a native-macro layer that is never used.** Macro grants are 0 in 7/7 uc7.1 runs.
  - A real uc7.1 draft (t9, `turn` with `sense=close`) fails H1's auto-approval, so the run exits before any action. The plan prompt itself allows `close`, and the re-ask path in the client is never called.
  - A 401 `token_expired` on the plan request killed composite run 002.
- **The LLM does all metric grounding in its head from calibration; there is no pixel-to-3-D tool.**
  - First estimates in T04 were 3.4–5.7 cm off, and the can's obstacle box was 4.85 cm off along the camera's depth axis.
  - 46 of 164 calls (28%, 939 s of model time) came right after a segment stall.
- **There is no kinematic or reach model in the loop.** In composite run 001 the elbow sat at its joint limit in 7 requests, and an 11-call recovery cost 169 s.
- **"Until contact" or "until end stop" cannot be expressed.** Every expected contact ends the fragment as a failed `segment_stall`, which costs a model call.
- **Two veto checks waste the first calls.**
  - The 1.2 rad rotation envelope rejected the first call in both H1-config runs whose first move needed about 90° of yaw.
  - `route_start_blocked` fired 6 times.
  - The router never actually detoured: 0 detours and 0 search calls.
- **Task, robot and camera identity are closed code registries.** Running a new sentence, BDDL, robot, camera name or gripper needs code edits, and the composite task did.
- **Fixed since the earlier reports:** the direction-word bug (`right_of = +x`), image mirroring and schema-key reordering. Plan-request hangs are mitigated.

## 0. Scope and method

**Read:**
- the launcher and runner (`scripts/launch.py`, `scripts/evaluate_agent_speedup_v2_user_check.py`, plus the episode loop of `scripts/evaluate_agent_speedup_v2.py`);
- all policy-path modules: `*_user_check_{agent,agent_client,client,fragments,routing,spatial,plan,direction_display}.py`, `agent_speedup_v2_user_check.py`, `action.py`, `agent_speedup_v2_tasks.py`, `agent_speedup_v2_client.py` (plan prompt and retry logic), `astranomy_agentic_geometry.py`, `contact_skills.py` (image transform), `observation.py`, `ordered_transport.py`, `chatgpt_auth.py`, and `libero.py` (custom-BDDL contract);
- `robot_facts_v2.json`, `README.md` and `validation/VERIFICATION.md`;
- all 8 `runs/agent_speedup_v2_uc71_*` directories: result summaries, `routing_metrics.json`, approved plans, every executive request/response log, `trace.jsonl`, `agent_stats.json`, and the drawer 002 failure audit;
- the T04 can layout manifest;
- the earlier reports `reports/agent_speedup_analysis_20261026/` (README, `uc7_runs.md`, `history_intent.md`) and `reports/uc7_direction_semantics_20261028/` (README, `implementation_uc71/README.md`, V1 planning drafts);
- the uc7 LIBERO-10 export `exports/gpt_field_libero10_20261022/`;
- `UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md` and `agent_SpeedUp_v2_user_check.md`.

**Computed** with stdlib Python over the saved logs:
- per-call trigger, type, latency and tokens, from `requests/*.request.json`, `*.response.json` and `episode.events.jsonl` skill decisions;
- details of every veto handoff;
- joint-limit margins at each request;
- grasp attempts (open→close transitions of the executed gripper command);
- request sizes and how often requests were compacted;
- the share of each request taken by macro/legacy fields;
- model position estimates against the settled layout truth;
- an offline approval check of the uc7.1 t9 draft with H1's own validator;
- hash comparison of policy modules between H1 and each run.

**Not covered:**
- live H1 behaviour on the official tasks (no such runs exist);
- deep collision accounting and per-step overhead (other analysts);
- H2 and H3;
- the RoboCasa runner, which H1 does not include.

The uc7 runs from October (LIBERO-10 official) were deleted from `runs/`. Numbers about them come from the earlier reports and the export, and are marked "likely" where I could not recompute them.

## 1. How H1 actually works

`run.sh` / `launch.py` always starts `evaluate_agent_speedup_v2_user_check.py start` with these settings:
- `--auto-approve`, `--reasoning-effort high`, `--request-deadline-s 180`, `--planning-video`;
- `GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M=0.10`.

It accepts only registered task keys whose reset hash is already frozen (21 official tasks at init 0, plus the packaged T04 can scene).

**Preparation.** The runner resets the simulator and makes **one** `plan` request:
- **Input:** four images (external and wrist, clean and with an 8×8 grid) and **no calibration**.
- **Output:** a native v2 relation plan:
  - up to 8 parts A–H, each with grid cells;
  - steps `hold/turn/slide/place_on/place_in/push_to/release`, with `keep` and `until`;
  - `regions` and `placement_goals`.

The draft is validated and auto-approved, and any issue aborts the run.

**Execution** is `AgentFirstV2Policy`. Each decision is one Astra call.
- **The call receives:**
  - the task sentence;
  - the current external and wrist RGB (6 image pairs plus the full text history are retained);
  - the measured pose and joints;
  - both camera calibrations and the frozen Panda gripper geometry;
  - `region_directions`, taken from the frozen external-camera frame;
  - compacted fragment feedback.
- **The model returns** belief/plan/why plus exactly one action:
  - `execute`: 1–6 world-frame segments (`move_to_pose`, `move_relative`, `approach`, `set_gripper`, `wait`). Transit allows at most 0.50 m / 1.2 rad per stage and 0.10 m per action; contact allows 0.08 m / 0.5 rad and 5 mm per action.
  - `motion`: a target at most 5 cm away.
  - `macro`: a native v2 relation (never used).
  - `observe`.
  - `finish`: does not end the episode.
- **Obstacles.** The answer may also carry an `obstacle_map` of up to 4 axis-aligned boxes. A local router uses it for the grip-site path only.
- **When fragments end:** on arrival, a stable finger gap, a stall, budget exhaustion, a veto, or an observation checkpoint. The model is then called again.

**Termination.** The episode ends only at the first raw LIBERO success, an operator stop, or an error. Contacts are recorded but never acted on.

## 2. Evidence base

**Table A. Runs whose policy code is byte-identical to H1.**

All `agent_speedup_v2_user_check_*` modules, `action.py`, `agent_speedup_v2_client.py`, `astranomy_agentic_*`, `contact_skills.py` and `observation.py` match. The composite runs differ only in `_sim.py` (ordered evaluation).

| Run | Task | Raw | Strict at terminal | Calls (plan + exec) | Plan s | Control s (no video) | Stall-triggered calls | Veto-ended calls | Close cmds / needed |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `…can_obstacle_transit100_t04_20261103_001` (H1 reference) | T04 + can | ✓ | ✗ (mug held) | 1 + 13 | 17.5 | 287.3 | 3 | 2 | 2 / 2 |
| `…bottle_obstacle_transit100_t08_20261103_001` | T08 + bottle | ✓ | ✗ (pot held) | 1 + 21 | 22.3 | 467.4 | 2 | 3 | 3 / 2 |
| `…stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001` | stove-off T08 + bottle (custom sentence) | ✓ | ✗ | 1 + 16 | 37.9 | 420.7 | 3 | 1 | 4 / 3 |
| `…drawer_bowl_moka_transit500_cap100_20261103_001` | 5-stage composite (custom) | ✓ | ✗ | 1 + 42 | 52.2 | 1249.2 | 17 | 1 | 5 / ≥4 |
| `…drawer_bowl_moka_transit500_cap100_20261103_002` | same | **initial_plan_failed** (HTTP 401 `token_expired`) | — | 1 + 0 | — | 0 actions | — | — | — |
| `…drawer_bowl_moka_transit500_cap100_20261103_003` | same | ✓ | ✗ | 1 + 36 | 50.2 | 1208.4 | 16 | 0 | 5 / ≥4 |

**Table B. Supporting evidence from other builds.**

| Build | Runs | Result | Source |
|---|---|---|---|
| uc7.1 early (prompt-5, 20 mm transit) | `…can_obstacle_t04_…_001` and `…default_routing_t04_…_001` | both raw ✓, strict ✗; 27 and 11 calls; 682.8 and 338.1 s | result_summary.json |
| uc7.1 (2026-10-28, before obstacle maps) | t6 relayout | raw ✓, strict ✗, 14 exec + 1 plan calls, 604.86 s; the first pudding release was outside the goal box, so the agent re-grasped | `reports/uc7_direction_semantics_20261028/implementation_uc71/README.md` (run directory deleted) |
| uc7.1 V1 planning only | 13 drafts (LIBERO-10, g5, t6 relayout, RoboCasa coffee) | 12/13 valid (t9 invalid); mean plan time 28.8 s | `…/implementation_uc71/v1_planning/summary.json` |
| uc7 (older policy) on LIBERO-10 official init 0 | 13 runs | raw 11/13, strict 3/13, service time ≤ 300 s in 1/13; t6 0/2; 3 plan requests hung and were interrupted (t3_004, t6_008, t6_010) | `reports/agent_speedup_analysis_20261026/uc7_runs.md`, `history_intent.md` (likely; runs deleted) |
| uc7 manual 10-task batch | 10 | 9/10 raw; 161 exec calls; mean 337.5 s among successes; the selected runs t2_002, t3_005, t5_002 and t6_011 show that retries happened | `exports/gpt_field_libero10_20261022/README.md` |
| uc7 RoboCasa coffee | 1 | `infrastructure_error` after 244 steps (base-movement guard) | `history_intent.md:301` (likely) |

No uc7 or uc7.1 full-task run exists on LIBERO-Goal, LIBERO-Spatial or RoboCasa beyond the attempt above.

---

## 3. Findings, by mechanism

### F1 (high, verified). The evaluator is a two-sided success oracle, and the agent cannot end an episode honestly

**Mechanism**
- The runner requires `stop_on_success` and forbids an action horizon (`scripts/evaluate_agent_speedup_v2.py:213-216`).
- The loop ends only on raw success, the operator-stop file, a step cap, a recovery stop or an exception (`:1437-1472`).
- `finish` is a single hold step plus a logged claim (`src/gpt_field/agent_speedup_v2_user_check_agent.py:366-372`).
- The prompt says `finish` "never ends the episode … If observation continues, recheck unfinished spatial relations … There is no give_up action" (`agent_speedup_v2_user_check_agent_client.py:205-209`).

| Evidence | Value | Source |
|---|---|---|
| Successes that end with the last object still held | **7/7** uc7.1 runs (5/5 H1-identical). The failing checks are only `not_held`/`at_rest`, and `terminal_failure_is_hold_only=true` | each run's `result_summary.json` → `physical_truth` |
| Policy stage at termination | still in a contact `approach` or `set_gripper`; the release and retreat the agent planned were cut off. Examples: composite 001 had an open plus a 7 cm withdraw pending; T08 had a 17 cm retreat pending; T04's final call contained only a carry and a descent | `physical_audit.final_policy_stage`, `final_pending_stages` |
| H1 reference T04 strict pass | only after the evaluator's separate 38-action release-and-settle diagnostic | `…transit100_t04…/result_summary.json` → `diagnostic_settle.actions=38` |
| Agent uses the negative oracle | T04 at 20 mm: `finish` at call 17 (step 794), then **9 more calls (159.1 s of latency) and 482 more actions**. The belief at call 18: "The episode continues, so at least one final relation may be insufficient…". It then re-centred the yellow mug by 3–4 cm with two re-grasps, and raw success came at step 1276 | `…can_obstacle_t04_20261103_001/trial/requests/000018…` (calls_table) |
| Failure never ends on its own | uc7 t6_009: 12 calls / 579 steps / 399.6 s after `finish`, then stopped by the user. H1 README: "no whole-episode call/action/time cap" | `uc7_runs.md` §5 (likely), `README.md` |

**Why it matters**
- Reported successes are oracle-assisted:
  - the positive oracle stops the episode before the agent's own release;
  - the negative oracle tells the agent it is not finished.
- Neither oracle exists outside this evaluator, so the success rate is not a zero-shot measurement.
- Completion time leaves out the release and retreat that a real task needs.
- Failed attempts are unbounded, which conflicts with the few-attempts goal.

**Improvement (general, not a threshold)**
- Make `finish` terminal, and score raw + strict success at that moment, after the agent's own release and retreat.
- Add `give_up(reason)`.
- Score without `stop_on_success`: log the first raw-success step as a diagnostic only.
- Delete the "if observation continues" sentence.
- Report an evaluation wall-clock bound equal to the user's target (e.g. 180 s) as failure. It is not a policy veto.

**Expected effect**
- An honest success number. 1 of the 7 uc7.1 successes (T04 at 20 mm) would be scored as a failure at its own `finish`.
- Each success costs about 1–2 more calls (+20–40 s) for release, retreat and finish.
- Failed attempts end at `give_up` or the bound, instead of a 740–962 s user stop.

**Validation**
- Offline: rescore the existing runs at their `finish` steps from `trace.jsonl` plus truth.
- Live: an all-task sweep on one frozen build with finish-scoring, where raw/strict-at-finish is the headline metric.

### F2 (high, verified). The initial plan is a mandatory, unrecoverable gate whose content the executive barely uses

**Mechanism**
- The runner makes one `request_plan` call (`scripts/evaluate_agent_speedup_v2_user_check.py:307-309`).
- The launcher always adds `--auto-approve` (`scripts/launch.py:53`).
- `approve()` runs `validate_review_plan` (`src/gpt_field/agent_speedup_v2_user_check.py:367-371`). Any issue raises `ReviewError`, and the runner turns it into `parser.error`, a nonzero exit (runner `:940-948`, `:965-966`).
- The client already has a targeted correction re-ask (`reask_error`, `agent_speedup_v2_client.py:1040-1047`), but preparation never uses it.

| Evidence | Value | Source |
|---|---|---|
| Prompt and validator disagree | the plan prompt says "sense is any, open or close" for every step (`agent_speedup_v2_client.py:219-220`); the validator requires `sense=any` except for `slide` (`agent_speedup_v2_user_check.py:121-122`) | code |
| Real draft fails | uc7.1 V1 t9 draft has steps `(place_in A any)` and `(turn D close)`. H1's validator raises "sense must be any except for slide", so auto-approve exits with 0 actions. The historical uc7 t9 draft had the same error and was fixed by hand | `h1_zero_shot_work/scripts/t9_autoapprove.py`; `implementation_uc71/README.md` |
| Draft validity | 12/13 V1 drafts | `v1_planning/summary.json` |
| Serial cost before the first action | 14.3–52.2 s across 7 live uc7.1 plans (composite: 50.2 and 52.2 s); mean 28.8 s over 13 V1 drafts | result summaries, V1 summary |
| Plan use | **0 macro grants in 7/7 uc7.1 runs** (`agent_stats.json`) and in 13/13 uc7 runs (earlier report). The executive is told the plan is "NOT a mandatory execution program" (`agent_client.py:62-64`) | code + artifacts |
| Representation limits | 8 part IDs A–H (`agent_speedup_v2_types.py:58`); the composite plan uses all 8. None of the 5 obstacle-scene plans lists the added can or bottle as a part or a `clear_of` constraint | approved_plan.json |
| Stale instruction | the planner is told "The robot measures every distance, height, direction and angle itself by moving, touching and looking" (`agent_speedup_v2_client.py:194-196`); in agent-first H1 nobody measures anything for it | code |

**Improvement**
1. Remove the macro-only validation codes from approval while macros are optional. These are `ignored_selection`: the sense, ref and until text rules. This is a veto removal.
2. If a plan call stays, on an invalid draft re-ask once through the existing `reask_error` path instead of exiting.
3. Structurally: delete the separate plan call. The first executive answer already sees the task, images and calibration, so it can emit the goal contract itself (objects, goal relations, regions, constraints).

**Expected effect**
- No abort on t9-type drafts (1 of 13 recorded drafts).
- Merging saves about 14–52 s (≈29 s mean) of serial latency per task and removes the 8-ID ceiling.

**Validation**
- Offline: revalidate the 13 V1 drafts and 8 uc7.1 approved plans with the relaxed validator; expect 21/21 valid.
- Live: run t9 with auto-approve.

### F3 (high, mechanism verified; effect is a hypothesis). Metric grounding is LLM mental arithmetic on calibration, with no geometry tool

**Mechanism**
- The executive gets intrinsics and extrinsics and is told "Pixels specify rays, not known 3-D points; infer depth from the views and observed motion" (`agent_speedup_v2_user_check_agent_client.py:221-226`).
- The action schema (`:237-260`) has no pointing, triangulation or ray–plane query.
- Estimation error turns into extra "view" moves, contact descents that stall short, and re-placements.

| Evidence | Value | Source |
|---|---|---|
| T04, first estimate of the left plate | (−0.055, −0.35, 0.41) against the settled truth (−0.0045, −0.3232, 0.4393): **5.7 cm horizontal**, mostly along x, the external camera's depth axis. After a close wrist view: 2.4 cm | requests 000005 and 000007 beliefs; `output/layouts/libero_10_t4_can_obstacle_20261103_001/layout_manifest.json` |
| T04, right plate | 3.4 cm horizontal, 2.9 cm low | request 000012 |
| T04, can obstacle box | centre **4.85 cm** off along x (box x [−0.015, 0.065], can x = −0.0235) | `routing_audit.map_updates[0]` |
| Composite 003, moka handle | calls 25–35: **11 calls, 297.1 s**. Beliefs: "My earlier assumed grasp height was too low", "triangulate the handle's upper crest" | calls_table |
| Composite, drawer pull | calls 1–9 took 174.6 s in run 001 and 230.8 s in run 003, including 3–5 viewpoint moves to locate the pull by parallax | calls_table |
| Stall-triggered calls, all 7 runs | **46/164 (28%), 938.7 s of 3071.3 s of model latency.** Keyword classification (likely): 25 height/depth misestimate, 9 kinematic, 4 contact, 1 end stop, 7 other | calls_table |

**Improvement (RGB and calibration only, no depth sensor)**
- Add a `locate` field to the executive schema. The model names a feature by pixel (u, v) in two or more retained images: the current pair and any of the 6 kept pairs, each with its synchronized calibration.
- The harness triangulates (linear DLT plus reprojection residual) and returns world XYZ ± uncertainty in the same decision cycle.
- With a single view, fall back to intersecting the ray with a support height the model declares.
- Use the result for grasp points, support tops, region anchors and obstacle boxes.

**Expected effect (hypothesis)**
- 1–2 fewer calls per pick-and-place, 5–8 fewer on the composite.
- First placements and obstacle boxes within about 1–2 cm, which also helps collision avoidance.

**Validation**
- Offline: in saved T04 and composite request images, mark plate centres and handles in two views, triangulate with the saved calibrations, and compare with layout truth against the model's beliefs (5.7 and 3.4 cm).
- Live: A/B on T04, T08 and the composite.

### F4 (high, verified). No kinematic model in the loop: joint limits are discovered by stalling

**Mechanism**
- The executive geometry contains only `schema_version`, `frame`, `image`, `gripper` and `cameras`. There is no robot base pose, link geometry or joint limits; `measured_state` has the joint angles only (composite 001 request 1).
- Fragments send Cartesian OSC targets with no feasibility check.
- Reaching a limit therefore shows up as `segment_stall`, which the model has to diagnose by probing.

| Evidence | Value | Source |
|---|---|---|
| Composite 001 | **7 requests with joint 4 at or past its −0.0698 rad upper limit** (margin −0.0005 to 0.0). Calls 17–27 (**11 calls, 169.0 s**) were spent on posture recovery: "elbow near its extension limit", "local kinematic blockage", "Large simultaneous translation and orientation changes are unreliable in this configuration" | `h1_zero_shot_work/scripts/joints.py`, calls_table |
| Composite 003 | 2 requests at the joint-5 limit | same |
| uc7, LIBERO-10 | 13/218 requests within 0.02 rad of the joint-4 limit (t3, t6); 17/55 stall handoffs blamed the elbow or reach | `uc7_runs.md` §4 (likely) |

**Improvement**
- Put the base pose, joint limits and per-joint margins into `measured_state`.
- At each stage entry, run an IK feasibility check with the known Panda kinematics (robot calibration is an allowed input). It reports "reachable, margins m_i" or "infeasible: nearest reachable pose p".
- Optionally insert an elbow-bent via-posture when the straight Cartesian path crosses a limit, and report it. Nothing is vetoed.

**Expected effect**
- Avoids the roughly 10-call recoveries on high or far targets: composite, t3- and t6-type tasks, any new layout near the edge of the workspace.

**Validation**
- Offline: compute IK margins along the commanded paths in the traces of composite 001/003 and check that stage-entry IK predicts the limit events.
- Live: rerun the composite.

### F5 (medium-high, verified). "Until contact" and "until end stop" cannot be expressed

**Mechanism**
- A motion stage with less than 0.8 mm of progress over 12 steps ends the whole fragment as failed `segment_stall` (`agent_speedup_v2_user_check_fragments.py:404-419`).
- The segments that would follow are discarded, not resumed (`:429-431`, `:476-478`).
- Segments have no termination predicate (`:60-69`).
- So a planned "descend to contact → close → test lift" sequence, or "pull until the drawer stops", always returns to the model at the contact.

| Evidence | Value |
|---|---|
| Composite 001 / 003 | 17/42 and 16/36 executive calls came right after a stall. The beliefs then reinterpret the stall as success: "consistent with reaching its travel limit", "the final push stalled at the expected closed stop" |
| T04 reference, call 11 | "The approach stopped about 8 mm from its requested target … I will not push farther downward", followed by the planned close |
| Grasp verification | every grasp spends an extra call on "close + test lift + checkpoint" (T04 calls 4 and 11) |

**Improvement**
- Add `stop_on: ["arrival", "contact"]` to `approach` and `move_relative`, where contact uses the existing stall predicate. When contact is declared as the terminal condition, the stage succeeds, reports the achieved displacement and the stall location, and the sequence continues.
- Add an optional compliant profile with lower gains for pulling and pushing mechanisms.
- Optionally add a measured continuation predicate for grasps: the gap stays inside the jaw range and stays stable through the test lift.

**Expected effect (hypothesis)**
- One fewer call per contact descent and per mechanism end stop.
- About 8–12 fewer calls on the composite.
- Helps drawers, knobs, doors and buttons (LIBERO-10 t3/t9, LIBERO-Goal g0/g3/g7, RoboCasa press tasks).

**Validation**
- Offline: in the logs, count stall handoffs whose next decision just continued the planned sequence; those would have continued automatically.
- Live: t3, t9, g0, g3, g7.

### F6 (medium, verified). Zero-progress vetoes cost first calls; the router only refuses and never detours

**Mechanism**
- **Rotation envelope.** A transit stage whose target is more than 1.2 rad from the measured entry orientation is rejected before any action (`fragments.py:285-291`), although the action adapter already caps rotation at 0.1 rad per step (`fragments.py:234`, `action.py:66-100`).
- **Start-blocked route.** A route whose start lies inside an expanded box is refused (`routing.py:105-106`).

| Evidence | Value | Source |
|---|---|---|
| First-call rotation rejections | **both H1-config runs whose first move needed about 90° yaw**: T04 transit100 (1.572 rad requested, 32.4 s call) and T08 transit100 (1.569 rad, 30.8 s). In uc7: t4_001, t8_001, t9_001 and t6_011 (twice), 133.1 s | `vetoes.py`; `uc7_runs.md` §3 |
| `route_start_blocked` | 6 events in 5 of 7 runs, often right after the model increased the clearance around the held or just-placed object ("The planner rejected the start after the isotropic clearance increase") | `routing_metrics.json`, beliefs |
| Router behaviour | all 88 routed stages that succeeded went straight; **0 detours, 0 planner search calls** in all 7 runs | `routing_metrics.json` |
| Calls that ended in a veto | **9/164 (213.4 s, 7% of model time)**. H1 reference T04: **2/13 calls, 52.1 s of its 200.3 s** of execution model time | calls_table |

**Improvement (removes vetoes, adds no thresholds)**
- Delete the rotation part of the stage envelope, or split oversized rotations into sub-stages automatically.
- Turn `route_start_blocked` into an escape: route first to the nearest point outside the expanded boxes along the direction of least penetration, then continue, and report what was done.
- The 0.50 m translation envelope was never binding in LIBERO data (0 rejections). In larger scenes it would be, so splitting it automatically too is a cheap portability fix.

**Expected effect**
- About 1 call (≈30 s) saved on tasks whose start needs reorientation.
- About 1 call per run saved on start-blocked routes.
- For the H1 reference: −52 s.

**Validation**
- Offline: replay the 9 vetoed requests through the patched `_enter_segment`/`TransitRoute` (pure Python) and confirm they become executable.
- Live: one sweep.

### F7 (medium, verified). Task identity is a closed code registry: new tasks, layouts and sentences need code edits

**Mechanism**
- The runner accepts only registered keys (`--task choices`, runner `:831`) and checks the simulator prompt against the registry (`:280-282`).
- The launcher refuses any reset without a pre-recorded hash (`launch.py:35-37`).
- A custom BDDL must keep the official `:language`, `:goal` and target assets (`src/gpt_field/libero.py:203-218`).
- The registry itself is `agent_speedup_v2_tasks.py:97-124`.

**Evidence**
- The stove-off T08 and composite runs needed main-tree code: `COMPOSITE_TASK_KEY`, `CUSTOM_TASK_KEYS`, `custom_task_prompt` in `libero.py`, and an ordered-success evaluator (diff of `/workspace/gpt_field/src/gpt_field/agent_speedup_v2_tasks.py` and `libero.py` against H1).
- The label `agent-speedup-v2-user-check-7.1` appears on prompt-5 and prompt-6 builds with different routing and fragment code (`trial/summary.json`), so a "7.1" table mixes builds.

**Why it matters.** The user wants one frozen framework where a new layout or task combination works from the sentence alone. Today each new case is a code change, so the build is no longer frozen.

**Improvement**
- Pass the task as data at runtime: sentence + scene/BDDL + initial state + an evaluator predicate file.
- Compute and record the reset hash instead of requiring it in advance.
- Stamp every result with the build digest (the existing `runtime_sha256`), and bump the version on any change to a policy file.

**Expected effect.** New tasks with zero code edits, and acceptance sweeps that provably ran on one build.

**Validation.** Run 3 LIBERO-90 tasks and 2 new sentences on the frozen build; the source diff should be empty and the digest identical across results.

### F8 (medium, verified). Robot, camera and gripper facts are frozen into validators and prompt text, along with leftovers from LIBERO and the macro layer

| Hard-coded assumption | Where | What breaks |
|---|---|---|
| Camera names `agentview` / `robot0_eye_in_hand`; external camera is a `fixed_world_camera`; images square and uncropped | `astranomy_agentic_geometry.py:23`, `:88`, `:94-102` | any other camera rig, a robot-mounted or moving camera, non-square images. The uc7 RoboCasa coffee run ended in `infrastructure_error` on base motion (likely) |
| The exact `PandaGripper` dict, including the asset hash | `astranomy_agentic_geometry.py:33-37`, `:90-91` | any other gripper |
| "one Franka Panda"; grip site 0.097 m; pad size; 0.079 m gap, all in the prompt text | `agent_speedup_v2_user_check_agent_client.py:53`, `:228-231` | copies the geometry metadata as text, so it silently contradicts the metadata on a new robot |
| "about 79 mm maximum opening", "starts with the hand pointing straight down" | `agent_speedup_v2_client.py:194-195` | other grippers or start poses |
| Joint limits, base pose | absent (see F4) | reasoning about reach |
| Example "(back compartment, left plate)", wording taken from LIBERO-10 t5 and t4 | `agent_speedup_v2_user_check_spatial.py:70` | prompt fitted to benchmark tasks |
| "Initial native calibration normally requires about 190 control steps"; DROID / 15 Hz adaptation notes inside the capabilities | `agent_client.py:190-191`; capabilities `direct_motion.adaptations` in every request | macro and paper leftovers in every call |
| `robot_facts_v2.json` keyed by task (start poses for libero_10_t2, bowl_plate, drawer_bowl) | used only by the macro layer (`agent_speedup_v2_robot.py:32`) | harmless to H1 execution, but a task-keyed artefact inside the frozen tree |

**Improvement**
- Keep one robot-and-sensor description, derived from the MJCF/URDF: gripper offsets and opening, joint limits, base pose, and a camera list with intrinsics and mount type (fixed, robot or moving, with a pose per frame). Check it against the simulator at reset.
- Render every prompt number from it.
- Let validators accept any number of cameras with any names.
- Remove the task-derived examples and the macro and DROID text.

**Expected effect.** Portability with no change to LIBERO behaviour, and fewer tokens per call.

**Validation**
- Preflight against a second robot configuration with a robot-mounted camera.
- A text-snapshot test asserting that the prompts contain no benchmark object words.

### F9 (medium; the fix is verified, the remaining gap is likely). Directions are fixed, but region extent and "on" precision are undefined

**Verified fixed**
- `build_view_frame` takes right = R_ext[:2, 0] and away = up × right (`spatial.py:198-211`).
- Live requests carry `right_xy=(−1.7e−6, 1.0)` and `away_xy=(−1.0, −1.7e−6)` for LIBERO's agentview, i.e. right = +y and front = +x. This holds in T04 and also in the composite, whose camera has a 70° field of view.
- The planner and executive texts use viewer words (`spatial.py:66-72`, `agent_client.py:211-219`).

**Remaining gap**
- The executive receives only unit direction vectors.
- With `bounds=null`, the region resolver is a half-plane (`spatial.py:334`), and it never runs anyway because no macros are used.
- So "beside" and "on/centered" have no metric meaning for the agent, and the hidden evaluator tolerance shows up only through the oracle in F1.
  - uc7.1 t6 relayout: the first release was outside LIBERO's 10 × 10 cm box, so the agent re-grasped; 14 calls, 604.86 s.
  - T04 at 20 mm: the first placement was 3–4 cm off-centre, found only through the "continues" cue, and cost 9 calls.

**Improvement**
- Freeze general default semantics:
  - directional relation = an adjacent band starting at the anchor's edge, about one object-width deep, laterally within the anchor's extent;
  - `on` / `centered_on` = the whole-object centre inside the support's inner half.
- Give these to the executive as metric target regions computed from the anchor extent (from F3 `locate`).
- Have the agent record the measured offset as evidence before it releases.

**Expected effect.** First-try placement on side and centre relations (t6, g5 "front of", LIBERO-Spatial); about 3–9 fewer calls on such tasks.

**Validation**
- Developer-only check against LIBERO's side-goal BDDL boxes (the earlier report found 7 boxes, each 3–12 cm past the anchor edge).
- Live: t6 (official and relayout) and g5.

### F10 (medium, verified). The transport stops on error classes outside its short retry list

**Mechanism**
- `_retry_cause` retries only 503 overload and four stream codes (`agent_speedup_v2_client.py:903-914`).
- Other HTTP errors raise "no retry or API-key fallback" (`chatgpt_plan.py:150`).
- `request_deadline_exceeded` is not retried (`agent_speedup_v2_user_check_transport.py:46,117`).
- Token refresh is proactive only: it happens when the local expiry is within 60 s (`chatgpt_auth.py:450-480`).
- Any of these errors mid-episode ends the episode, and all progress is lost.

**Evidence.** Composite 002: the single plan request returned **HTTP 401 `token_expired`**. Zero actions, 14.0 s of wrapper time, the attempt was lost (`…_002/initial_plan_failure_audit.json`, `runner.log`). That is 1 of the 6 launches of the H1 policy.

**Improvement**
- On 401, force one token refresh and resend.
- Treat 5xx and deadline errors as transient, under the same rule as today: same observation, nothing committed, bounded retries.
- Allow resuming from the last committed observation. Simulation is already paused during inference.

**Expected effect.** Removes infrastructure aborts (1 of the 6 H1-policy launches) and protects long episodes (composite runs take 28–30 minutes of wall time).

**Validation.** Fault-injection tests with a fake transport returning 401, 520 and deadline errors.

### F11 (high, verified). Coverage: the frozen H1 build has never been through the all-task acceptance sweep the user asks for

- The H1 policy has 5 completed live runs on 3 scene families (Table A): 5/5 raw, 0/5 strict at terminal, 0/5 within 180 s; the best is 287.3 s plus a 17.5 s plan.
- It has never run on any of the 21 registered official tasks that README §Run lists.
- `validation/VERIFICATION.md` is a deterministic replay of the single T04 reference: "not a fresh stochastic test, a success-rate estimate".
- The broader LIBERO-10 evidence (raw 9/10 tasks, strict 3/13) comes from an older policy: 20 mm transit, no obstacle map, prompt-5 or earlier, mostly manual review. Those runs also needed restarts after plan hangs.

**Improvement.**
- First land the cheap fixes on a single frozen build: F1 scoring, F2 (a and b), F6 and F10.
- Then run each task once with auto-approve on LIBERO-10 (10 tasks), LIBERO-Goal (10 tasks) and the 2 obstacle variants.
- Report raw, strict-at-finish, wall and control time, and calls.

**Expected effect.** Replaces extrapolation with a measured zero-shot rate.

**Validation.** The sweep itself, with a build digest identical across all results.

### F12 (low, verified). The dead macro layer still shapes every request, and compaction truncates failure causes

- **Macro share.** 15.2–16.4% of each executive request's JSON (about 3.2 k of 20.7 k characters) is macro status, native evidence (always empty), macro capability text, and the plan cells.
- **Compaction.** 155 of 164 requests exceeded 45 k characters and were compacted (`agent_client.py:545-553`). Compaction cuts the evidence:
  - `measured_pose` is reduced to xyz plus `{"more_in_audit": 5}`;
  - in T08 call 15 the route sub-reason `route_start_blocked` was hidden, and the model had to guess the cause.
- **Input growth.** Input grows from about 8 k to 260 k tokens over the composite run. This is a risk for longer tasks.

**Improvement.** In agent-first mode, drop the macro fields and text. Keep outcome reasons and routing sub-reasons verbatim, and summarise older fragment evidence.

**Expected effect.** About 3 kB less per request and clearer failure causes. The effect on latency is small: the earlier analysis found latency was not correlated with input tokens.

**Validation.** Re-render the recorded requests offline and compare sizes, compaction rate and which causes survive.

---

## 4. Older defects: status in H1

| Defect (earlier report) | Status | Evidence |
|---|---|---|
| Policy images mirrored left/right | **Fixed** (verified) | the observation is raw[::-1, ::-1] (`observation.py:31`); `StepView` applies [:, ::-1] (`contact_skills.py:281-283`), so the net result is a vertical flip; the geometry label says `raw_mujoco_rgb_vertical_flip_no_crop` |
| `astra._canonical` `sort_keys` reorders schemas | **Not on the wire** (verified) | `ordered_sse_transport`; the logs show `wire_transport_version=chatgpt-subscription-http-sse-ordered-v1` with schema order belief→…→obstacle_map; `_canonical` is used only for request keys |
| An all-zero rotation keeps a stale OSC goal | **Practically mitigated** (likely) | robosuite updates the goal orientation only for a nonzero rotation (`.venv-sim/…/osc.py:258-261`); the adapter emits exact zeros only when the error sine is below 1e-12 (`action.py:87-89`); the recorded hold action had a 2e-9 rotation |
| Axis-angle sign jumps in state8 | **Still present, shown to the model only** (verified) | `robot_pose_wxyz` is built from the rotation vector with no sign convention (`astranomy_agentic_motion.py:89-99`); the quaternion's sign flips between consecutive requests 3–12 times per run (T04: 3 of 12 transitions; composite 003: 12 of 35); the controller handles the double cover correctly |
| Direction words `right_of = +x` | **Fixed** (verified) | F9 |
| Plan-draft hang | **Mitigated** (verified) | compact-output instruction (`agent_speedup_v2_user_check_client.py:28-36`) plus the 180 s deadline (`launch.py:47`); 0 hangs in 7 live plans and 13 V1 drafts; a missed deadline is still fatal (F10) |
| Wrist image not turned at planning time | **Text fixed, image unchanged** | the request says "not turned; not aligned with image 1" (`agent_speedup_v2_user_check_client.py:417-422`); low impact because the executive never uses the plan's grid cells |
| 1.2 rad envelope veto; raw success mid-release | **Still present** | F6, F1 |
| Request larger than 60 k characters crashes (uc4) | **Bound kept, not hit** | the executive raises above 56 k characters (`agent_client.py:555-556`); the largest compacted request was 26.3 k |

## 5. Recommended order (one frozen build, then the sweep)

1. **Cheap, small:** F1 finish-scoring with `give_up`; F2a + F2b (relax the validator, re-ask once); F6 (remove the rotation veto, add the escape route); F10 (refresh on 401, transient retry classes); F12 (trim the macro fields).
2. **Then:** run the F11 sweep on that frozen build to establish the real zero-shot baseline.
3. **Capability work, medium cost:** F3 `locate` triangulation (also feeds F9 and collision boxes); F5 `stop_on: contact` and the compliant profile; F4 IK and joint-limit feedback.
4. **Generalisation infrastructure:** F7 data-driven task spec and build digest; F8 robot/sensor description; F2c merge the plan into the first executive call.

Each change is a mechanism that recurs across at least two tasks, or a structural gap. None adds a task-specific threshold. Four of them remove a veto: the rotation envelope, the start-blocked refusal, the macro-only plan validation, and the abort on an invalid draft.

## Appendix: scripts and data (in `h1_zero_shot_work/`)

| Script | Purpose |
|---|---|
| `scripts/calls.py` | per-call trigger, type, latency and tokens → `calls_table.json` |
| `scripts/vetoes.py` | details of envelope and route vetoes |
| `scripts/joints.py` | joint-limit margins at each request |
| `scripts/grasps.py` | close commands per run |
| `scripts/ctx_size.py` | share of each request taken by macro/legacy fields |
| `scripts/req_size.py` | request size and compaction |
| `scripts/t9_autoapprove.py` | H1 approval check of the uc7.1 t9 draft |
