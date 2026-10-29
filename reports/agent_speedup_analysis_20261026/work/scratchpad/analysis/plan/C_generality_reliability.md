# Plan C: a reliable all-task uc8, v3 frozen, one lean sweep

Angle: generality and reliability first, plus what to do with v3. Read-only planning on 2026-10-26. No simulator, model, HTTP or supervisor calls were made. Paths are relative to `/workspace/gpt_field` unless absolute. Helper scripts and their outputs are in `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/C/`. "Estimate" marks arithmetic on measured components. "Raw" means LIBERO success, "strict" means `v2_success`, and "service" means prep + execution + finalization without human review wait.

## 0. Thesis

uc7 is already general: 11/13 raw successes, and 9/10 tasks in the one-episode sweep. What it lacks is reliability, plus one wrong framework constant:

- a plan stream with no deadline (5 of 20 uc7-code preparation attempts hung);
- failure runs that never end (t6_009 ran 962 s and t6_011 740 s until an operator stopped them);
- envelope vetoes that cost calls on 5 tasks;
- a hard-coded world-axis convention for spatial words that is wrong in every LIBERO-10 scene. That convention is the root cause of the only uncovered task, t6.

The plan:

- Freeze v3 at r11. Port only its transport and I/O pieces, which do not veto motion.
- Ship uc8 as uc7 plus a few general changes. Each one either removes a check or corrects a fact the model is given.
- Accept or reject uc8 with one unattended sweep: 10 tasks × 2 episodes, reported as a coverage map.

**S300 is not this plan's lever.** By arithmetic, it takes the 11-success mean from 448.6 s to about 370.6 s, and S300 from 1/11 to 2/11. That still leaves a gap of about 5 calls per episode, which belongs to the speed plans. This plan makes that gap measurable on all tasks, without hangs or operator stops.

## 1. Evidence specific to this angle (new computations marked [C])

1. **[C] The framework's spatial-relation convention is a wrong constant.**
   - `src/gpt_field/agent_speedup_v2_user_check_spatial.py:48-49` tells the plan model: "Relations use WORLD axes: right_of=+x, left_of=-x, behind=+y, front_of=-y; they do not mean screen directions."
   - The native resolver hard-codes the same mapping at `:200-203`.
   - In all 10 LIBERO-10 scenes the external camera's image-right axis is world +y, and its forward axis points toward −x. This is the first-request calibration of each uc7 run (`rotation_world_from_camera`). I checked t0, t1, t2, t3, t4, t5, t7, t8 and t9 directly; t6 was checked separately.
   - LIBERO's language follows that viewer's frame:
     - **t4**, left/right plate: left at y∈[−0.325,−0.275], right at y∈[0.275,0.325] (`.../bddl_files/libero_10/LIVING_ROOM_SCENE5_...bddl:5-20`).
     - **t6**, right of the plate: the region is y∈[0.05,0.15] (V6a).
     - **t5**, back compartment: the caddy is yawed by π (`STUDY_SCENE1_...bddl:5-14`), and `back_contain_region` sits at local +x (`assets/turbosquid_objects/desk_caddy/desk_caddy.xml:11`). In world terms that is −x, away from the camera. The truth box is x∈[−0.458,−0.402] (`runs/..._t5_001/physical_truth.json`).
   - Both t6 plans used `relation: right_of`. t6_009's region text reads "surface region to world +x side of plate" (`runs/..._t6_009/approved_plan.json`). Its first placement was at (0.30, 0.02).
   - All 10 of t6_009's joint-4 elbow-limit requests (steps 916-1414) came while it was placing at these +x targets (`C/joint_margins.json`). The wrong axis therefore produced both the region miss and the reach stalls.
   - The plan request carries no calibration at all: only the task text and 4 images (`runs/..._t6_009_review/requests/000001-*.request.json`).
2. **[C] The model's tool envelope leaves out the fingers and the hand body.**
   - The prompt gives only pads: a midpoint at −3.6 mm and a 16 mm span, so the pads end 4.4 mm past the grip site (`agent_speedup_v2_user_check_agent_client.py:186-190`).
   - The compiled-model facts say the fingertips reach 9.25 mm past the grip site, the palm underside is 31.04 mm behind it, and the hand footprint is R1 −104..+100 mm by R0 ±31.6 mm (`src/gpt_field/robot_facts_v2.json` `asset`).
   - Descents stall at grip-site z≈0.446 on t1, t6 and t7 (`uc7_runs.md` §4). The audited table contact is at 0.43675 (V2b). The difference is 9.25 mm, exactly the fingertip reach.
   - In 10 of 56 execute decisions that planned a `set_gripper` stage, the stage was never reached because the preceding descent stalled on a support, a rim or a neighbour. These were t1 ×4, t3_003, t3_005, t4, t6_009 and t7 ×2, so 5 tasks (`C/dropped_gripper.json`). Each one cost a model call; the next decision then issued the dropped close or open.
3. **[C] Stall cause is ambiguous to the model.**
   - Arm joint limits are not in the request; only finger ranges are.
   - Using the Panda ranges in `/venv/fm_dno/lib/python3.10/site-packages/robosuite/models/assets/robots/panda/robot.xml`:
     - 16 of 218 uc7 requests had a joint within 0.05 rad of a limit, always joint 4 (t6_009 ×10, t3_003 ×3, t6_011 ×3).
     - 9 of 55 stall handoffs were true elbow-limit stalls.
     - In 13 stall handoffs the model's text blamed kinematics while no joint was within 0.15 rad of a limit (t1, t3, t6, t7; crude keyword match). Two examples are t7_001 #8 ("Joint 6 is near its upper range", but it was the table) and t6_009 #10.
4. **[C] Failures that recur on only one task, so this plan proposes no change for them:**
   - Grip-gap collapse during a close-maintained carry occurs only in t6_011 (steps 290 and 717; `C/gap_events.json`).
   - Empty closures on real grasp attempts occur only in t6_011 (2). The 3 other empty closes, in t3_003, t3_005 and t9_001, look like closing the gripper to push a door or drawer shut (`C/closures.json`).
   - Handle grasps held 5 of 6 times (t4 ×2, t9 ×2, t6_009 held; t6_011 slipped). Gaps were 16.5-18.5 mm, and the model's beliefs say "handle".
   - `set_gripper` settling that exhausts its budget: only t6_011. The other `segment_budget_exhausted`, in t9, was a `move_to_pose`.
5. **Envelope vetoes recur.**
   - The 1.2 rad rotation veto fired at t4 #1, t6_011 #1 and #9, t8 #1 and t9 #1. The 0.30 m translation veto fired on t2_002 (V4b). That is 5 of 10 tasks.
   - The executor already clamps every action to 0.1 rad (`agent_speedup_v2_user_check_fragments.py:185`).
6. **Honest end.**
   - `finish` never terminates (`agent_client.py:173-177`). No horizon exists (manifest `max_*: None`).
   - The base runner already ends an episode as the failure status `recovery_exhausted` when the policy emits that event (`scripts/evaluate_agent_speedup_v2.py:1422-1430, 1452-1458`). That status keeps terminal truth (`:141-143`).
   - The user approved an agent-side terminal stop on 2026-10-20 ("Allow early stop", memory `agent-speedup-v2-plan.md`).
   - [C] Under stop-at-first-raw-success, every model call happens in a raw-false state, so the model always gets one implicit evaluator bit: the task is not finished yet. The prompt tells it to use that bit ("If observation continues, recheck…", `agent_client.py:175-177`).
   - The only t6 success, uc6 t6_007, used it: `finish` at step 421, then raw success at step 464 (`user_check_plan_history.json`).
7. **Process.**
   - Today: 8.1% robot time, 54.9% protocol/freeze/audit, 18.8% waiting for authorization, and 2,025 s spent on four plan streams that produced nothing (`campaigns_today.md` §7).
   - [C] The 2026-10-21 uc7 manual batch had 43.3% robot time: 4,432 s of control in 10,248 s of wall. Human review wait was 44.0% (4,504 s), because each task waited for its own review (`runs/..._manual_batch_001/batch_state.json`).

## 2. Steps

Everything below is one new version id, `agent-speedup-v2-user-check-8` (uc8). Historical files and frozen copies are not edited.

uc8 follows the existing v3 pattern: a private module copy of the uc7 runner with its classes swapped (`scripts/evaluate_agent_speedup_v3.py:186-231`). That keeps uc7 rebuildable from its files, as C1 did.

### S0. Freeze v3 at r11; port only components that do not veto motion
- **Change.**
  - Do not edit `src/gpt_field/agent_speedup_v3*.py`. They are byte-identical to `runs/agent_speedup_v3_t1_012/frozen_runtime`.
  - uc8 imports three self-contained modules:
    - `agent_speedup_v3_transport.py` (bounded stream, `:224-271`);
    - `agent_speedup_v3_video.py` (640×384 recorder);
    - `agent_speedup_v3_performance.py` (`LiveSnapshotGate`, `adapt_private_evaluator`, `:31-125`).
  - uc8 copies two pieces of text instead of importing `agent_speedup_v3_client.py`, which would pull in v3 geometry and program code: `COMPACT_OUTPUT_INSTRUCTIONS` (`:40-46`) and the 503 availability-retry rule (`:49-118`).
  - Section 4 lists everything that is discarded.
- **Evidence.**
  - All 4 v3 raw successes were at least 81.6% uc7-style `execute_fragment` (V1a).
  - v3 is 0/7 since r6 and 0/3 today. All 9 behavioural failures from r4.1 on ended in `EvidenceAcquisitionFailed` (V2a). Gates triggered 36 of 39 follow-up calls in the last success (V1b).
  - The model's metric geometry is biased 17-42 mm, beyond its declared 8-15 mm uncertainty (V2b).
  - The ported pieces are the ones with independent evidence: compact output, bounded transport and the I/O cuts (V5a, V4c, C1).
- **Expected effect.** No live behaviour of its own. It ends the per-run revision loop of r1-r11: 15 runs over 12 builds in about 48 h (`history_intent.md` §4.3).
- **Acceptance.** Offline: the uc8 inventory lists the three imported v3 files with r11 hashes, and `git diff` shows no v3 change.
- **Cost.** 0.5 h; 0 model calls; 0 runs.
- **Risk.** It gives up v3's one measurable benefit: fewer recorded forbidden contacts on t1 (251 vs 5,738, n=1, and r2 recorded 6,808). Contacts are not part of the success criterion.
- **Constraints.** It removes vetoes and geometry layers rather than adding them (constraints 2 and 3), and it is a new version id (constraint 7).

### S1. Reliability harness: bounded compact plan stream with one retry, 503 retry, compact I/O, multi-episode enrollment
- **Change.**
  1. **uc8 plan client.** A subclass of `UserCheckV2Client` (`agent_speedup_v2_user_check_client.py:339-386`):
     - append `COMPACT_OUTPUT_INSTRUCTIONS` to the plan and replan instructions;
     - use the bounded transport, copied with two parameters for plan requests: a 90 s total deadline (about 3× the slowest completed plan, 30.4 s) and a visible-output cap of 8,192 characters (about 4.9× the largest legitimate plan, 1,686; [C] from `verify/v5_plan_rows.json`). Keep the 2,048-character whitespace-run guard (`agent_speedup_v3_transport.py:16-21, 92-124`).
  2. **Retry policy.** Extend `_retry_cause` (`agent_speedup_v2_client.py:903-914`) in the uc8 plan and executive clients:
     - **one** fresh retry on `subscription_request_deadline`, `subscription_output_whitespace_limit`, `subscription_visible_output_limit`, and the two 503 sharing codes;
     - eligibility is unchanged: no response envelope and nothing committed;
     - the retry is charged to service time.
  3. **Executive transport.** Same bounded transport at the default 180 s / 131,072-character bounds. The execution prompt does not change in S1.
  4. **I/O.** As in `scripts/evaluate_agent_speedup_v3.py:191-193`:
     - `adapt_private_evaluator(module.base)`;
     - `module.base._default_recorder_factory = AgentSpeedUpV3VideoRecorder`.
  5. **Enrollment.** `scripts/enroll_resets.py` resets the simulator only, one reset per fresh worker, and makes no model calls. It writes the initial-observation SHAs for (t0..t9, episodes 0..1). Runs pass them with `--expected-initial-observation-sha256` (UC runner `:591`). This avoids D4 (`scripts/evaluate_agent_speedup_v2.py:411-421`); the registry only knows episode 0 (`agent_speedup_v2_tasks.py:49`).
  6. Contract strings stay single-line, to avoid D3 (`agent_speedup_v2_client.py:518-531`).
- **Evidence.**
  - V5a: 11/38 non-compact plan attempts tailed vs 0/12 compact; on identical t1 inputs 8/17 vs 0/9. Tails are stochastic per sample.
  - V5b: the one completed tail was 99.6% whitespace.
  - The C2 runaway streams passed 4,000 characters at 25.3 s and 26.8 s (`campaigns_today.md` §3), so an 8,192-character cap trips at about 28-44 s ([C] from the logged rates).
  - One 503 ended v3 t1_004 at request 2 (`v3_runs.md`).
  - Compact I/O: 897/897 actions and 13/13 payloads exact in replay (`results/speedup_evidence_20261026T055123Z/baseline_validation.json`). Display plus status took 103.7 s in arm A vs 25.8 s in arm B.
- **Expected effect** (estimate).
  - Preparation worst case goes from 297-900 s to ≤120 s (trip plus one retry). The tail rate falls from roughly 25-29% to near 0.
  - About −0.091 s/step of display and status cost, or about −78 s per episode at uc7 step counts.
  - Multi-episode runs become possible without per-run SHA workarounds.
- **Acceptance.**
  - Offline (no model): replay the per-delta character counts in `runs/speedup_next_20261026T101438Z/request_diagnostics.jsonl` through the guard. Both runaway streams must trip before 60 s, and the two complete streams must pass.
  - Live: 20/20 sweep preparations complete in ≤120 s, and 0 episodes are lost before control.
- **Cost.** About 5 h engineering; 0 model calls; one sim-only enrollment pass (20 resets, about 5 min).
- **Risk.**
  - SIGALRM needs the main thread and no other `ITIMER_REAL`. That holds in the uc runner, as v3 used it.
  - The compact instruction is a formatting change backed only by observational evidence, and it is confounded with the transport and with t1.
  - The two transport bounds are the only new numbers in this plan. They bound a wait; they do not gate motion.
- **Constraints.** The model, effort and subscription are unchanged. No task key is involved, and the bounds are identical for all tasks.

### S2. Episodes end honestly: model `give_up` and an evaluator-side service cap
- **Change.**
  - **Schema.** Add `give_up` to the executive type enum (`agent_client.py:209`) and to the validator (`:229-231`). Replace "There is no give_up action." (`:177`) with this rule: give up only when the official task can no longer be achieved from the current state with any available action. A failed attempt or an unverified placement is not a reason.
  - **Policy.** The uc8 `_apply_agent` (overriding `agent_speedup_v2_user_check_agent.py:305-370`) handles `give_up` like this:
    - it records the claim;
    - it calls `self._events.emit("recovery_exhausted", source="model_give_up", why=...)` (EventSink, `agent_speedup_v2.py:312-330`) and holds for one step;
    - the base loop then ends with status `recovery_exhausted` and records terminal truth plus the release-and-settle counterfactual;
    - uc8 normalization labels it `agent_gave_up`, re-adds the rule that the uc7 runner strips (`scripts/evaluate_agent_speedup_v2_user_check.py:232-234`), and leaves the raw base artifacts intact.
  - **Cap.** The sweep supervisor writes the existing operator-stop file when prep plus run wall time reaches the cap; 900 s is proposed (a user decision). The loop checks that file every iteration (`scripts/evaluate_agent_speedup_v2.py:1405-1411`). The episode ends as `user_stop` with source=file, keeps truth, and is labelled `service_cap` in the sweep table.
  - The evaluator code does not change, and the cap never reaches the policy.
  - `finish` stays advisory, and stop-at-first-raw-success stays as it is (see the user decisions).
- **Evidence.**
  - t6_009 continued 12 calls and 399.6 s after `finish`, then was operator-stopped at 962 s. t6_011 was operator-stopped at 740 s after 20 recovery calls (`uc7_runs.md` §5).
  - uc1-uc5 t6 runs lasted 524-1,643 s; three of them ended by user stop (`uc7_runs.md` §1).
  - The longest uc7 raw success took 650.3 s of service, so a 900 s cap loses none.
- **Expected effect.** No sweep episode needs an operator. A failure costs at most about 900 s plus one call, or less after a `give_up`. Terminal truth is present for 20/20 episodes; today it was skipped in 9/9 v3 evidence stops (D2).
- **Acceptance.** Live: every episode ends as success, `agent_gave_up` or `service_cap`, and each has `physical_truth.json`. Post hoc, from video and truth: count `give_up` decisions made in states that were still recoverable. The target is 0.
- **Cost.** About 2 h; 0 model calls.
- **Risk.**
  - A premature `give_up` could cost a success; uc6 t6_007 succeeded 43 steps after a wrong `finish`.
  - The cap can overshoot by up to one call (≤180 s).
  - The status `user_stop` is conflated with operator stops in the base artifacts; the sweep table separates them.
- **Constraints.** `give_up` is a model decision, not driven by truth. The cap is a uniform evaluation bound, not a 300 s stop. Success is still independent raw LIBERO. No added state machine.

### S3. Remove the fragment envelope vetoes
- **Change.** In a uc8 subclass of `FragmentSequencePrimitive`:
  - delete the stage-entry translation and rotation rejection (`agent_speedup_v2_user_check_fragments.py:233-239`), keeping the non-finite check (`:230-232`);
  - delete the relative-translation envelope check at validation (`:124-125`);
  - keep the per-action clamps (0.02 m / 0.005 m at `:252-258`, 0.1 rad at `:185`), `max_steps`, arrival and stall;
  - report no target envelope in the uc8 capabilities (`:25-32, :146`);
  - replace prompt lines `agent_client.py:118-121` with: "no per-stage distance or rotation envelope; each action is capped at 0.02 m (0.005 m contact) and 0.1 rad; max_steps bounds the stage; do large reorientations in free space".
- **Evidence.**
  - 5 zero-action rotation vetoes (entry error 1.57-1.72 rad) and 1 translation veto, on t2, t4, t6, t8 and t9.
  - Avoidable cost: 5 replacement calls (42.4 s) plus 5 holds. The model's own remedy was to split the rotation into stages, which the clamp already does (V4b).
  - One of the vetoes blocked the t6_011 post-slip recovery (call 9).
- **Expected effect** (estimate). −1 call on about 4 of 13 uc7-like episodes; in the 11 successes, t4, t8 and t9 each lose one replacement call worth 6.4-8.1 s ([C] `C/s300.json`). Reorientations above 1.2 rad, needed to regrasp a fallen object, become possible.
- **Acceptance.** Live: 0 `target_envelope_rejected` events in the sweep, and no new failure class attributable to large single-stage reorientation (from the video). Offline: one fake-view unit test checks that a 1.6 rad target executes at ≤0.1 rad per step.
- **Cost.** About 1 h; 0 model calls.
- **Risk.**
  - Fingers sweep through a larger volume during a combined rotate-and-translate. Contacts are recorded only.
  - A mistaken far target now executes, bounded by `max_steps` and stall, instead of being refused.
- **Constraints.** It removes a veto and adds no threshold (constraint 2).

### S4. Give the model the right frame facts: viewer-centric spatial words, the full tool envelope, joint-limit margins
- **Change.**
  - **(a) Spatial words.**
    - Replace `agent_speedup_v2_user_check_spatial.py:47-49` with a viewer-centric definition: right_of and left_of mean toward image-right and image-left of the external camera (image 1); behind and front_of mean farther from and nearer to that camera; all are measured along the support surface.
    - The uc8 plan request adds `external_camera_axes_world` (image-right = R[:,0], image-up = R[:,1], forward = −R[:,2]) from the observation's calibration. It is fed through the uc8 plan client `_content` (`agent_speedup_v2_client.py:788`), from the call in `prepare_review` (UC runner `:188`). Region text states the derived world direction.
    - The native resolver (`spatial.py:200-203`) uses the same calibration-derived axes. Macros are dormant (0 grants in 28 runs), but it should stay consistent.
    - The executive prompt (calibration paragraph, `agent_client.py:179-186`) gets one sentence with the same definition.
    - It also gets a default for regions whose extent is hidden: "A to the <side> of B, with no distance given, means beside B on that side, aligned with B's centre along the other horizontal axis, with a small clear gap, not far away."
  - **(b) Full tool envelope.** Replace the pad-only geometry text (`agent_client.py:186-190`) with the compiled-model facts in `robot_facts_v2.json` `asset`: fingertip lowest point 9.25 mm beyond the grip site; pads −11.6..+4.4 mm; palm underside 31.04 mm behind; hand footprint R1 −104..+100 mm, R0 ±31.6 mm. Add one rule: after a descending stall, the fingertip height (grip z minus 9.25 mm when pointing down) is a measured support height to reuse.
  - **(c) Joint-limit margins.**
    - The uc8 `_context` override adds `arm_joint_ranges_rad` (the Panda `robot.xml` ranges) and per-joint `joint_limit_margin_rad` to `measured_state` (`agent_speedup_v2_user_check_agent.py:212-216`).
    - Fragment stall evidence (`fragments.py:315-323`) gets the margins at the stall.
    - One prompt sentence: a margin near 0 means a kinematic limit, so change orientation or bring the target inward; a stall with ample margin during a descent is contact.
- **Evidence.**
  - (a) §1.1: the constant is wrong in all 10 scenes; the language of t4, t5 and t6 follows the camera; t6 failed because of it.
  - (b) §1.2: the stall height equals table + 9.25 mm; 10 dropped gripper stages across 5 tasks; rim stalls in t1 and t7 are blamed on hand clearance.
  - (c) §1.3: 9 true limit stalls (t3, t6) and 13 kinematic misattributions (t1, t3, t6, t7).
- **Expected effect** (estimates).
  - (a) t6 places the pudding on the correct side, with targets inside the reach limit. That removes t6_009's +x elbow stalls. t6 coverage could go from 0/2 to 1/2 or better, though the mug grasp and fallen-mug recovery remain open.
  - (b) Removes the systematic 4.85 mm fingertip offset and rim stalls caused by the hand body: maybe −2 to −5 of the 10 dropped-stage calls per 13 episodes. The RGB height bias itself remains.
  - (c) Fewer diagnosis calls (t6_011 #10-14, t7 #8).
  - These facts add about 0.5-1k input characters per call, worth roughly 0.02-0.03 s (input costs 0.018-0.027 s per 1k tokens).
- **Acceptance.**
  - Offline, at no extra cost: the batch-prepared t4, t5 and t6 plans (S5 step 3) state image-side and world directions that match the calibration. The t6 region must read image-right / world +y.
  - Live:
    - t6 first pudding placement on the +y side (truth, post hoc), and raw success on at least 1 of 2 episodes;
    - t4 and t5 do not regress;
    - the planned gripper stage is reached in ≥90% of executes that include one (uc7: 45/56 = 80%);
    - fewer than 13/55 stall handoffs with a kinematic misattribution.
- **Cost.** About 2.5 h; 0 extra model calls.
- **Risk.**
  - Language that is object-centric (e.g. "back of the caddy") could diverge from the camera frame in other suites. In LIBERO-10 it matches, including t5 under the caddy's π yaw.
  - The adjacency default rests on one task (two runs: t6_007 first target y=0.20 and t6_009 y=0.17, both past the far edge y=0.15). It is justified as a structural gap: the region's extent is hidden and there is no completion signal. The user may drop it.
- **Constraints.** Inputs are camera calibration and robot geometry only (constraint 3). It removes a coordinate constant rather than adding one, and adds no geometry layer or gate.

### S5. One lean sweep as acceptance: 10 tasks × episodes {0, 1}, unattended, coverage map
- **Change.**
  - **The protocol is in §5.**
  - **New code is limited to one queue script:** batch `prepare`, then review, then sequential `run`. It reuses the existing `prepare`, `review` and `run` subcommands (UC runner `:578-680`), the S2 cap, and per-episode rows in `sweep.csv`.
  - **Not used:** isolated copies, campaign-local git repositories, Codex delegated review, and mid-sweep amendments.
- **Evidence.**
  - 8.1% robot time today, with 54.9% spent on protocol and audit.
  - 43.3% robot time in the 2026-10-21 batch, where per-task review wait took 44.0% ([C]).
  - Only t1 has been tested on more than one initial state (U4).
- **Expected effect** (estimate).
  - About 20 × 400 s ≈ 2.2 h of control. Over the evaluation window that is ≥55-65% robot time (preparation about 10 min, one review session, sweep, analysis ≤1 h).
  - The first multi-episode, all-task coverage map.
- **Acceptance.** The pre-registered reliability gates in §5 (live), and the coverage map as the result.
- **Cost.** About 1 h of scripting.
  - Model calls: 20 plan calls plus about 300 execution calls (about 15 per episode).
  - Live runs: 20 episodes, about 2.5 h, run sequentially.
  - Analysis: ≤1 h.
- **Risk.**
  - Service latency drifted 35-51% between days (`model_io.md` §1.3), which confounds S300 comparisons with history.
  - 20 episodes give coverage, not tight success rates.
- **Constraints.** One frozen build, all tasks, and a coverage map rather than per-task work items (constraint 1). Live-first, with the minimum of checks; the sweep is announced and started by the user (constraint 6).

Dependencies:

- S1-S4 are all in the uc8 freeze. S2 and S5 rely on S1's runner skeleton.
- S5 depends on S0-S4.
- S0 is only a decision and blocks nothing.

## 3. Deliberately not proposed (constraint discipline)
- **Stall→close and stall→open continuation.** Banned for close (C8). Automatically opening after a stalled placement descent would violate the same rule's intent. The 10 dropped-stage calls are left to S4(b).
- **Gap-collapse interrupt, `set_gripper` settling change, grasp or upright skills.** Each recurs on one task only (§1.4).
- **Measured-contact memory, triangulation, history projection.** The first two would be a new geometry or state layer. History projection gave no latency gain (r10/r11 still 39-45 s per call).
- **Model or effort changes.**
- **Changing first-raw-success truncation.** This is a user decision (§7).

## 4. v3 disposition
| v3 piece | Disposition | Why (evidence) |
|---|---|---|
| Bounded transport (`agent_speedup_v3_transport.py`) | **Port**, parameterized for plans | 0/12 compact tails vs 11/38 (V5a); whitespace guard tripped at 24.9 s |
| `COMPACT_OUTPUT_INSTRUCTIONS` (`agent_speedup_v3_client.py:40-46`) | **Port** (copy) to the plan client | same; formatting only |
| 503 availability retry (`:49-118`) | **Port** (copy) | t1_004 ended on a 503 |
| Compact video and `LiveSnapshotGate` (`agent_speedup_v3_video.py`, `agent_speedup_v3_performance.py`) | **Port** (import) | 897/897 replay-exact; −78 s display and status per episode (C1) |
| Terminal stop (`escalate`) | **Re-implement** as `give_up` → `recovery_exhausted` | the v3 path ends as `infrastructure_error`, so truth is skipped (D2, 9/9) |
| `TerminalActTimingMixin` (`scripts/evaluate_agent_speedup_v3.py:94-131`) | Optional port | honest timing when an act raises |
| GeometryMemory / LocalGeometryMemory, sweep / coverage / inventory / obstacle-core gates | **Discard** | 9/9 behavioural failures ended through them; 36/39 gate handoffs; geometry bias 17-42 mm (V1-V2) |
| Programs and compact operations | **Discard** | used once in a success, abandoned at step 148 before the grasp; first call 96-137 s |
| `scene_update` / `motion_context` output | **Discard** | +5.1 s visible output and +5.7 s reasoning per call; no attributable success |
| Visual watch, interaction monitor, operation runtime, stereo, view proposer, repeat and continuation guards | **Discard** | 0 accepted local pose revisions; candidates offered in 11/130 requests; 2 executed views, neither cleared a blocker |
| History projection (r10), adaptive contact cap (r7) | **Discard here** (speed plans may A/B them on uc8) | no latency gain measured; the cap derives from model geometry |

- v3 stays frozen at r11. Its files stay in place untouched, so r11 can be reproduced.
- If v3 is ever re-run, fix D2 first (`scripts/evaluate_agent_speedup_v2.py:1124-1126, 1535-1536`; `scripts/evaluate_agent_speedup_v3.py:283-289`).

## 5. Evaluation protocol (lean sweep 1 for uc8)

**Steps:**

1. **Freeze once.**
   - Implement S1-S4.
   - Run the unit tests for the changed functions and `preflight` once.
   - Record the runtime-inventory hash in `runs/uc8_sweep_001/manifest.json`.
   - Do not run a live smoke: the first sweep episode is the smoke (live-first).
2. **Enroll.** Sim-only reset SHAs for t0-t9 × episodes {0, 1}, with one reset per fresh worker. That matches prep and run, and avoids the hard-reset-count effect on KITCHEN_SCENE4/6/8. A mismatch excludes that episode; it is never silently re-drawn.
3. **Batch prepare.** 20 plans, about 25 s each. A preparation that fails after its one bounded retry counts as an infrastructure failure. Claude checks the t4, t5 and t6 plans' spatial text (S4 acceptance) before anything executes.
4. **One review session.**
   - Either the user reviews all 20 plans, which is human wait and excluded from service,
   - or the user explicitly authorizes `--auto-approve` for the sweep, which costs 0 s.
   - No Codex review is billed.
5. **Announce, then the user starts it.** Sequential runs, never concurrent; today's +45% latency may have come from concurrency. Execution is unattended, ends on the S2 cap or `give_up`, and the operator acts only in an emergency.
6. **Score every episode.**
   - Fields: task, episode, raw (headline), service s, control s, calls, steps, S300, end cause, raw after release-and-settle, strict after settle, and a failure class.
   - Failure classes are pre-defined: relation grounding, grasp/slip, reach/joint limit, height/contact, plan grounding, infra-prep, infra-model, harness.
   - Do not use `held_ever` contact labels (D5).
7. **Analyse once** (≤1 h). A change for uc9 needs a mechanism recurring on ≥2 tasks. A single-task failure is recorded as a coverage hole.

**Pre-registered gates for sweep 1:**
- **Reliability (must pass):**
  - 20/20 episodes end without an operator;
  - 0 preparations over 120 s;
  - ≤1 infrastructure or harness failure;
  - terminal truth for 20/20;
  - 0 envelope vetoes.
- **Generality (the result):**
  - raw successes out of 20, and tasks with ≥1 raw success out of 10;
  - the episode-0 column is directly comparable with uc7's 9/10;
  - no uc7-covered task goes to 0/2 without a cause that recurs on ≥2 tasks.
- **Speed (secondary):** S300 count, median service and calls per episode.
- **Process KPI:** ≥55% of the evaluation window is robot control, vs 8.1% today.

**Coverage map template.** Rows t0-t9. Columns ep0 and ep1, each cell showing raw ✓/✗, service in seconds and end cause. A final column marks tasks covered.

**Next sweeps.** Speed plans A/B their levers against frozen uc8 on the same 20-episode design.

## 6. S300 arithmetic (estimates; [C] `C/s300.py` → `C/s300.json`)
| Scenario, over the 11 uc7 raw successes | Mean service | ≤300 s |
|---|---:|---:|
| Measured: prep + approval-to-claim + elapsed | 448.6 s (median 438.1) | 1/11 (t5_001 270.2) |
| + compact I/O (trial_003 rates: video 0.0252, run_state 0.0001 s/step) | 372.6 s | 2/11 (t5_001 236.8, t5_002 286.0) |
| + S3 envelope removal (−1 call on t4, t8, t9) | 370.6 s | 2/11 |
| + S1 compact preparation | about −1 s on completed preparations (draft call 17.2 → about 16.5 s) | 2/11 |

- Adding about 5 s of process start-up changes no count.
- **What S1 really buys for S300:** an episode can no longer lose the 297-900 s that a hung preparation costs. Before, 5 of 20 uc7-code preparation attempts hung.
- **The remaining gap.** After this plan's non-model costs, the model budget left under 300 s averages 151.1 s (range 90.5-210.0 s). That is about 10 calls at 15.1 s each, or 7 at today's 21.5 s, against the 14.8 calls actually used.
- **So S300 on most tasks needs about 5 fewer calls per episode, or about 32% lower latency per call,** plus fewer steps on t0, t3 and t9 (non-model costs of 185-210 s). Those levers are outside this plan.
- **S4 may add calls on t6** if the task becomes solvable.

## 7. User decisions needed
1. **Approval mode for sweep 1:** one batched manual review session, or an explicit sweep-wide `--auto-approve`. Either way, confirm that no Codex delegated review is used. If it is used, decide how its time is billed (today 138-285 s was billed to service).
2. **Per-episode cap.** The value (900 s proposed; the longest uc7 success took 650 s), and confirm that a cap ending counts as a failure (`service_cap`).
3. **`give_up`.** Extend the 2026-10-20 "Allow early stop" approval to a model-declared `give_up`, ending as a failure with truth recorded.
4. **End of a successful episode (D1a, pending since 2026-10-21).**
   - Keep stop-at-first-raw-success: official LIBERO semantics, comparable with uc7. Then accept the implicit "not yet successful" bit, and the prompt line that uses it (`agent_client.py:175-177`).
   - Or let the agent end episodes and score the end state. That removes the bit, but uc6 t6_007 would have scored a failure.
5. **Sweep size:** 10 × 2 episodes (proposed, about 2.5 h) or 10 × 3 (about 3.7 h).
6. **Adjacency default in S4(a).** Keep it or drop it; it rests on one task as a structural gap.
7. **Model and effort stay gpt-6-astra / high.** This plan proposes no change.

## 8. Biggest uncertainties
- Generalization to episode 1 is untested on 9 of 10 tasks; t1 episode 1 was 2/2 raw.
- Whether the model uses `give_up` well: too early would lose successes, never would leave the cap to end failures.
- Whether S4(a) is enough for t6. The mug handle slip (t6_011) and fallen-object recovery remain, with no task-specific fix allowed.
- New failure classes from envelope-free large reorientations. Untested; contacts are recorded only.
- The root cause of the preparation hangs is unknown. The compact-instruction effect is observational (11/38 vs 0/12, confounded). The bound limits the cost, not the cause.
- Service latency drifted 35-51% between days. That dominates any S300 comparison; this plan reaches S300 only on t5 by arithmetic.
- The S4(b) and S4(c) effects are unmeasured. The keyword-based misattribution count is crude.

## Appendix: scripts written for this plan (stdlib, read-only on `/workspace`)
- `C/gap_events.py` → `C/gap_events.json`: grip-gap collapse during close; only t6_011.
- `C/closures.py` → `C/closures.json`: settled gap per close episode; 29 closes, 5 empty.
- `C/joint_margins.py` → `C/joint_margins.json`: joint-limit margins at the 218 uc7 requests, with stall attributions.
- `C/dropped_gripper.py` → `C/dropped_gripper.json`: planned `set_gripper` stages never reached; 11/56, 10 after a stall.
- `C/s300.py` → `C/s300.json`: the §6 arithmetic.
