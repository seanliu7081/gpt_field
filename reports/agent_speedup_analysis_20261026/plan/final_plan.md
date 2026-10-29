# Agent SpeedUp: final next-step plan (judge synthesis, 2026-10-26)

**Verdict.** Plan C (reliability and generality first) wins by a narrow margin over Plan A (S300 first). Plan B (asynchronous executive) comes last.

The final plan has four parts:
- It starts from C's uc8 build, which holds the cheapest, highest-certainty, all-task changes.
- It adds A's measured S300 path as two later builds.
- It adds B's recorder thread and B's handoff evidence.
- v3 stays frozen at r11.

Each phase ends with one announced, user-started live sweep of one frozen build across all 10 LIBERO-10 tasks.

**Terms**
- **raw**: independent raw LIBERO success, the success criterion.
- **strict**: `v2_success`, reported as a diagnostic only.
- **service**: preparation + execution + finalization, excluding human review and queue wait.
- **f**: the service-speed factor, Σ measured executive latency / Σ latency predicted by the uc7 fit (model_io.md:52: 3.98 + 0.0274·in_k + 0.0392·reasoning + 0.0212·visible).
- **uc7**: `agent-speedup-v2-user-check-7`.
- **(est.)** marks an estimate from `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/final_budget.py`, which takes Plan A's measured per-run inputs and re-tiers them into the phases below.

---

## 1. Spot checks (code change points opened and confirmed)

| Change point | Where | Finding |
|---|---|---|
| Transit cap, rotation clamp, envelope veto | `agent_speedup_v2_user_check_fragments.py:25-32`, `:252-256` (Euclidean clip), `:185` (`ActionAdapter(max_rotation_rad=.1)`), `:233-239` (stage-entry veto), `:124-125` (relative-translation veto); `action.py:16`, `:52`, `:67-68` | Confirmed. The adapter rejects any per-axis step above 0.05 m and allows a rotation clamp up to 0.5 rad, so 50 mm / 0.25 rad is feasible. A subclass can override `PROFILES` and `_enter_segment` without editing uc7. |
| Spatial-word constant | `agent_speedup_v2_user_check_spatial.py:47-49` ("right_of=+x, left_of=-x, behind=+y, front_of=-y; they do not mean screen directions"); resolver `:200-203` | **Confirmed wrong.** In the first requests of t0, t1, t4, t5, t6 and t9, `rotation_world_from_camera[:,0]` ≈ (0, 1, 0), so image-right = world +y. The forward axis −R[:,2] is (−0.85, 0, −0.53) in t0, t1, t4 and t6 and (−0.78, 0, −0.63) in t5 and t9, so the camera looks toward −x and down. The t6_009 approved plan reads "E: table: surface region to world +x side of plate". Its plan request carries no calibration. `UserCheckV2Client.__init__` appends the text (`agent_speedup_v2_user_check_client.py:348-351`), so a uc8 subclass can replace it. |
| v3 bounded transport | `agent_speedup_v3_transport.py:55-64` (main-thread SIGALRM; refuses if an `ITIMER_REAL` is already active), `:16-19` (module constants), `:224-271` | Confirmed reusable. Per-kind limits need a parameterized copy. **Integration hazard:** C1's `ServiceClock.deadline()` arms `ITIMER_REAL` (`results/speedup_evidence_20261026T055123Z/baseline/src/gpt_field/agent_speedup_evidence_clock.py:65-82`), which would make every bounded request fail. The clock must therefore be reporting-only, and the cap is enforced through the stop file. |
| Compact I/O | `agent_speedup_v3_video.py` and `agent_speedup_v3_performance.py` import no v3 geometry or program code. `adapt_private_evaluator` patches a private module (`:78-125`). | Confirmed. The wrapper pattern is in `scripts/evaluate_agent_speedup_v3.py:181-231`, and the UC runner factories are at `:465-484`. |
| Per-step image work | `observation.py:24-45` (level-6 PNG encode), `:114-115` (decode in `__post_init__`); `libero.py:662-667` (`from_dict`); `agent_speedup_v2_user_check_sim.py:52-57`, `:84-90` (re-decode plus `agent_geometry` IPC) | Confirmed: 6 decodes and 2 encodes per step. The executive re-encodes upright pixels at level 1 (`agent_speedup_v2_user_check_agent_client.py:497-502`, `mark_prompting.py:84-88`), so model bytes do not depend on the worker's PNG level. |
| Truth-keeping end paths | base runner `:141-143` (`TRUTH_STATUSES`), `:1405-1411` (stop file), `:1422-1458` (`recovery_exhausted`); `agent_speedup_v2_types.py:320-322` (`EVENTS`) | Confirmed. A stop-file cap ends the episode as `user_stop` and terminal truth is kept. |
| Plan B's async premise | `hybrid_subscription.py:420` holds `_history_lock` across the whole transport; `chatgpt_plan.py:217-222`: deltas "never supply executable text" | Confirmed. B's lock surgery and the change to this invariant are both real costs. |
| Tool envelope | `robot_facts_v2.json` `asset` (from the compiled robot model): fingertip 9.25 mm beyond the grip site; pads −11.6..+4.4 mm; palm 31.04 mm; footprint R1 −103.99..+100.43 mm, R0 ±31.6 mm. The prompt gives pads only (`agent_client.py:185-190`). | Confirmed. This is an allowed input class (robot geometry). |

---

## 2. Phases

Every build follows the same rules:
- It is a new version id.
- It is built as new modules plus a private copy of the UC runner with its classes swapped (the v3 pattern). No uc7, v3 or frozen file is edited.
- Before a sweep, run `preflight` once. The first sweep episode serves as the smoke test. There are no test suites and no replay gates.

### Phase 1: uc8 (`agent-speedup-v2-user-check-8`). Reliable lean harness, veto removal, frame facts, then Sweep 1

Steps are listed in order of certainty.

**1.1 Compact display I/O (C1 arm B, unchanged).**
- Change: use `AgentSpeedUpV3VideoRecorder` for video, and gate `state()` through `LiveSnapshotGate`.
- Evidence: replay matched on 897/897 actions and 13/13 payloads. In trial_003, video cost 0.0252 s/step against 0.0908, and run_state 0.0001 against 0.0253.
- Effect (est.): about −0.090 s/step, which is −84 s on t1_001 and −76 s on the raw-11 mean.
- Cost: about 2 h.

**1.2 Bounded, compact preparation and a bounded executive.**
- Change:
  - Copy `COMPACT_OUTPUT_INSTRUCTIONS` (`agent_speedup_v3_client.py:40-46`) and append it to the plan and replan instructions.
  - Use a parameterized copy of the v3 bounded transport. Plan request: 90 s total deadline, 8,192 visible characters, 2,048-character whitespace run. Executive: 120 s, 16,384 characters, 2,048.
  - Allow one same-request retry on the deadline and guard codes and on v3's two 503 availability codes, via the `_retry_cause` override (`agent_speedup_v2_client.py:903-914`). Retry only when no response envelope arrived and nothing was committed. The retry time counts as service.
- Evidence:
  - 11 of 38 plan attempts without the compact block tailed, against 0 of 12 with it. On identical t1 input the split was 4/9 against 0/8.
  - The one completed tail was 99.6% syntax whitespace.
  - The uc7 plan path has only a per-socket 180 s timeout. 5 of 20 uc7-code preparations hung for 297–900 s.
  - The v3 guards recorded 0 false trips in 160 calls.
- Effect: worst-case preparation drops to about 2 × 90 s, against 300–900 s today. C2's two runaway streams would trip at about 24 s and about 40 s.
- Cost: about 3 h.

**1.3 Service clock and per-component timers.**
- Change:
  - Merge C1's `ServiceClock` as **reporting only**; its SIGALRM `deadline()` is never used.
  - Add timers in the uc8 worker and host adapter around `env.step`, rendering, PNG encode, validation decodes, JSON/pipe, `from_dict`, `CalibratedObservation` and `agent_geometry`. Report them out of band so the observation schema stays unchanged.
- This replaces Plan A's separate simulator-split session (M1): Sweep 1 measures the split at no extra cost.
- Cost: about 2 h.

**1.4 Remove the fragment envelope vetoes (constraint 2).**
- Change:
  - Delete the stage-entry rejection at `fragments.py:233-239` (translation and rotation, both profiles) and the relative-translation check at `:124-125`.
  - Keep the non-finite check, per-step clamps, `max_steps`, arrival, stall and gap-rise checks.
  - The prompt (`agent_client.py:118-121`) then states only the clamps.
- Evidence: 6 rejections on 5 tasks. Five were rotation rejections at 1.57–1.72 rad (t4, t6_011 ×2, t8, t9); one was a translation rejection at 0.325 m (t2). They cost 5 replacement calls (42.4 s) plus 5 holds. The 0.1 rad clamp already splits large rotations.
- Effect: −6.4 to −8.1 s on t4, t8 and t9, and reorientations above 1.2 rad become possible.
- Cost: about 1 h.

**1.5 Correct the frame facts given to the model** (prompt and context only; allowed inputs).

- **(a) Viewer-centric spatial words.**
  - Definition: right_of/left_of mean toward image-right/left of the external camera; behind/front_of mean farther from or nearer to that camera, along the support.
  - The plan request gains `external_camera_axes_world` (image_right = R[:,0], forward = −R[:,2]) from calibration, and region text must state the derived world direction.
  - The executive prompt gets one sentence with the same definition.
  - The dormant macro resolver is untouched (0 macro grants in 28 runs).
  - No adjacency default is added (user option D9).
- **(b) Full tool envelope.**
  - Fingertip 9.25 mm beyond the grip site, plus the pads, palm and hand footprint, from `robot_facts_v2.json`.
  - Also tell the model that the fingertip height at a descending stall is a measured support height.
- **(c) Arm joint ranges and per-joint margins.** Add them to `measured_state` and to the stall evidence.
- Evidence:
  - (a) The constant contradicts calibration in every checked scene. LIBERO's wording follows the viewer: t4 plates at y ∓0.30; t5 back compartment at world −x; t6 region y ∈ [0.05, 0.15]. t6_009 grounded the region as "+x" and placed the pudding at (0.30, 0.02).
  - (b) Descents stalled at grip z ≈ 0.446 on t1, t6 and t7, which is the audited table height of 0.43675 + 9.25 mm. 10 of 56 planned `set_gripper` stages were dropped after stalls, on 5 tasks.
  - (c) 16 of 218 requests had a joint within 0.05 rad of the joint-4 limit (t3, t6). 13 stall handoffs blamed kinematics with no joint within 0.15 rad of a limit (t1, t3, t6, t7).
- Effect (est.):
  - t6 can place the pudding on the correct side, though the mug grasp is still a separate gap.
  - Fewer dropped gripper stages and fewer misdiagnosis calls.
  - About +0.03 s per call.
- Cost: about 2.5 h.

**1.6 Sweep queue.**
- Change:
  - Prepare all 10 plans in one batch, hold one review session, then run the episodes sequentially.
  - Service cap: the existing operator-stop file is written when service reaches the cap (D2; 900 s proposed). The policy never sees the cap. A capped episode ends as `user_stop`/file with truth kept, and is labelled `service_cap`.
  - Write one CSV row per episode.
- Cost: about 1.5 h.

**Sweep 1 (live).**
- Scope: 10 tasks × episode 0, paired with the uc7 sweep.
- Size: about 10 plan calls + 160 executive calls; about 80–100 min.
- Acceptance:
  - 10/10 plans complete within 90 s with at most 1 retry.
  - 0 episodes lost to harness defects; 0 operator interventions; terminal truth for 10/10.
  - video ≤ 0.03 s/step and run_state ≤ 0.001 s/step on every task.
  - 0 `target_envelope_rejected`.
  - raw ≥ 9/10 (uc7: 9/10).
  - The t6 region reads image-right / world +y at review, and the first pudding placement lands on +y (checked post hoc from truth).
  - The timers sum to `simulator_step_s` within 5%.
  - f is reported.
- Projection (est., f = 1.0): 5th-best 359 s, S300 1/10.

**Constraint check.** No task keys. Vetoes and a wrong constant are removed, and nothing is added except wait bounds. Inputs stay within calibration and robot geometry/state. Model and effort are unchanged.

### Phase 2: uc9. Cheaper and fewer control steps, then Sweep 2

**2.1 One sim-only session** (about 15 min, 0 model calls).
- Motion check: from the t1 episode-0 frozen reset, run scripted free-space fragments at {20, 35, 50} mm × {0.10, 0.25} rad. The moves are a 0.20 m lateral move, a 0.25 m diagonal, 0.8 and 1.6 rad reorientations, and 0.2 m combined with 0.8 rad.
- Image check: replay a recorded uc7 action prefix through the slimmed worker and require equal RGB at every step.
- Rule: adopt the largest pair that reaches steady transit ≥ 10 mm/step and rotation ≥ 0.05 rad/step, with overshoot ≤ 4 mm / 0.06 rad and no torque saturation. If no pair beyond 20 mm / 0.1 rad passes, drop step 2.2.

**2.2 Faster transit.**
- Change: in a uc9 `PROFILES` table, raise the transit cap from 0.02 m to the chosen value (target 0.05 m) and the transit rotation clamp from 0.1 rad to the chosen value (target 0.25 rad). The contact profile is unchanged. The prompt states the new numbers.
- Evidence:
  - Realized transit is 4.80 mm/step at the 20 mm cap (ratio 0.24) across all 13 runs, and transit covers 79% of the path.
  - The OSC accepts 0.05 m per axis.
  - A's OSC model reproduces arrived-segment step counts within 3.2% and predicts 13.3 mm/step at 50 mm.
- Effect (est.): t1_001 goes from 897 to 667 steps. Across the 13 runs, steps fall 15% at 50 mm alone and 19% with the 0.25 rad clamp.

**2.3 Simulator-step slimming** (sized by Sweep 1's timers).
- Change:
  - Replace the 6 full validation decodes with PNG signature and IHDR checks. Decode only where pixels are used: the decision-step `StepView` and the video panel.
  - Encode per-step worker PNGs at compression level 1. The reset response is unchanged, so initial SHAs stay valid.
- Evidence: 6 decodes and 2 level-6 encodes sit inside 0.1046 s/step; decoding alone costs at least 14.4 ms/step; level 1 is about 5× faster (`mark_prompting.py:85`).
- Effect (est.): −15 to −50 ms/step, central −30.

**2.4 Recorder thread** (from B's P0 and A's H1b).
- Change: one thread with a bounded queue draws the canvas and writes to ffmpeg. Framework publication stays on the main thread, and the thread never calls the model.
- Effect (est.): hides about 60% of the remaining 0.025 s/step.

**Sweep 2 (live).**
- Scope: 10 tasks × episode 0, paired with Sweep 1.
- Acceptance:
  - Realized transit ≥ 10 mm/step.
  - Transit steps per metre at least 35% below the paired Sweep 1 run.
  - `simulator_step_s` per step falls by the timer-predicted amount, ±5 ms.
  - Main-thread video ≤ 0.012 s/step, with frames = steps.
  - `maintained_grip_gap_changed` ≤ 1.
  - raw ≥ Sweep 1 − 1, and every lost task is inspected for a transit-speed cause.
- Projection (est., f = 1.0): 5th-best 297 s, S300 5/10.
- Cost: about 9 h + the session; about 150 calls and about 70–90 min.

### Phase 3: uc10. Cheaper and fewer model calls, then Sweep 3

**3.1 Terse prose.**
- Change:
  - Limit belief to 240, plan to 160 and why to 80 characters (today 1800/1800/600), with one instruction.
  - Truncate and log overruns instead of rejecting them (`agent_client.py:226-228`).
- Evidence: prose is 413 of 806 visible characters per call.

**3.2 Append-only de-duplication of the observation text.**
- Change:
  - Send `capabilities` and `macro` only on call 1 or when they change.
  - Drop `recent_feedback`.
  - Keep only active/progress in `execution`.
  - Send the external-camera block only on call 1; the wrist camera stays in every call.
  - Reduce `last_feedback` to one row per segment.
  - Never rewrite an earlier message (unlike v3's projection).
- Evidence: 63% of the non-first observation text is static or duplicated.

**3.3 Checkpoint economy** (A's C2, scoped by B's handoff evidence).
- Change: replace the mandatory post-grasp test-lift checkpoint with guidance:
  - put `observation_after` only where the next stage depends on what the camera would show (before contact, release or a regrasp);
  - after a close whose measured gap shows an object, the lift and the free-space carry may follow in the same execute;
  - each decision pauses the robot for about 15 s.
- Pre-contact checkpoints stay.
- Evidence: in the raw-11, 23 of 26 post-gripper checkpoints were followed by a free-space transit. Pre-contact replies moved the target by more than 20 mm in 23 of 91 cases, so those checkpoints are real decision points.

**Effect of 3.1–3.3 (est.).** About −1.5 s per call (t1: −19 s) and about −1 call per task (t1: −14 s).

**Sweep 3 (live).**
- Acceptance:
  - Prose ≤ 260 characters per call.
  - Non-first observation text ≤ 60% of Sweep 2's.
  - Token-adjusted latency at least 1.2 s per call lower, at Sweep 2's f.
  - 0 schema rejections.
  - At least 1 fewer call per task on tasks that carry an object.
  - Empty carries are reported.
  - raw ≥ Sweep 2 − 1.
- **Headline:** S300 ≥ 5/10 with the 5th-best ≤ 300 s, provided f ≤ 1.1.
- Projection (est.), 5th-best and S300 count:
  - f = 1.0: 261 s, 7/10;
  - f = 1.25: 306 s, 3/10;
  - f = 1.5: 351 s, 0/10.
- Cost: about 5 h; about 140 calls and about 60–80 min.

### Phase 4: decision point and held-out coverage

**4.1 Trigger.** Sweep 3's 5th-best service is above 300 s at the measured f.
- Present U1 (D6): execution reasoning effort high → medium. Projected 294 s, S300 5/10, at f = 1.5.
- Present U2 (D7): a speculative first call during the excluded review wait.
- If either is adopted, build uc11 and run Sweep 4.

**4.2 Held-out sweep.**
- Enroll episode-1 reset SHAs in a sim-only pass, one reset per fresh worker.
- Run the final build on t0–t9, episode 1.
- Output: the coverage map on new initial states.

---

## 3. S300 budget (est., central; uc7 outcomes held fixed; +5 s process start-up included)

**t1_001, the 5th-best task of the uc7 sweep**, at f = 1.0 unless noted:

| Component (s) | uc7 now | P1 uc8 | P2 uc9 | P3 uc10 | Moved by |
|---|---:|---:|---:|---:|---|
| preparation | 21.4 | 21.4 | 21.4 | 21.4 | 1.2 caps the 300–900 s hang tail; the typical time is unchanged |
| process start + start-up + finalization | 14.5 | 14.5 | 14.5 | 14.5 | nothing (the pre-started worker is dropped) |
| model (13 calls) | 182.5 | 182.9 | 182.9 | 149.8 | 3.1–3.2 (−19.4), 3.3 (−13.7) |
| simulator steps | 108.4 | 108.4 | 60.6 | 60.6 | 2.2 (897 → 667 steps), 2.3 (−30 ms/step) |
| video | 81.8 | 22.6 | 6.7 | 6.7 | 1.1, 2.4 |
| run_state | 25.1 | 0.1 | 0.1 | 0.1 | 1.1 |
| observe + policy + residual | 9.4 | 9.4 | 7.0 | 7.0 | 2.2 (fewer steps) |
| **total** | **443.1** | **359.3** | **293.2** | **260.1** | |
| total at f = 1.5 | 534.3 | 450.7 | 384.6 | 335.0 (293.3 with U1) | |

**10-task sweep:** 5th-best / median of 10 / S300 count. The central value comes first, with the low–high band for the 5th-best in brackets.

| Build | f = 1.0 | f = 1.25 | f = 1.5 |
|---|---|---|---|
| uc7 measured | 443 / 449 / 0 | 489 / 500 / 0 | 534 / 551 / 0 |
| P1 uc8 | 359 / 369 / 1 | 406 / 419 / 0 | 456 / 471 / 0 |
| P2 uc9 | 297 / 315 / 5 [282–317] | 352 / 369 / 0 | 407 / 424 / 0 |
| P3 uc10 (3.1–3.2 only) | 276 / 292 / 5 | 323 / 340 / 1 | 372 / 388 / 0 |
| P3 uc10 (+3.3) | 261 / 278 / 7 [216–306] | 306 / 322 / 3 | 351 / 368 / 0 |
| + U1 (user decision) | 232 / 245 / 9 | 263 / 278 / 6 | 294 / 312 / 5 |

**Per task at P3, f = 1.0** (uc7 → uc10):
- t0: 550 → 333
- t1: 443 → 260
- t2: 385 → 260
- t3: 457 → 299
- t4: 454 → 294
- t5: 333 → 230
- t7: 432 → 261
- t8: 432 → 252
- t9: 572 → 345
- t6 failed in uc7; its run time goes 745 → 494.

**Raw-11 mean** (f = 1.0): 453.6 → 376.1 → 324.8 → 301.7 → 286.9 s, then 246.1 s with U1. Runs at or under 300 s: 1 → 2 → 6 → 6 → 8 of 11, then 10 of 11 with U1.

**Reading.**
- The median task reaches ≤ 300 s with no model or effort change only if f ≤ about 1.1.
- At f ≥ 1.25, it also needs U1.
- t0, t3 and t9 stay marginal: they need 14–21 calls and carry a large non-model cost.
- Sweep 1 measures f at no extra cost before any model decision is asked of the user.

---

## 4. Evaluation protocol

- **Builds.** One frozen build per sweep: uc8, uc9, uc10, and conditionally uc11.
  - The runtime inventory is bound at prepare time, and nothing is edited during a sweep.
  - On a harness defect: stop, fix under a new build id, and restart from t0.
  - Manipulation failures are recorded as framework gaps and never patched mid-sweep.
- **Episodes.**
  - LIBERO-10 t0–t9, episode 0, seed 7, on the frozen reset SHAs of `manual_batch_001`. This pairs each task with uc7 and with the previous sweep. 1 episode per task per sweep.
  - The final build also runs the held-out sweep on episode 1.
  - Episodes run strictly sequentially, with no concurrent arms.
- **Review.**
  - Plans are prepared in one batch, then reviewed in one session: either a manual review (wait excluded) or an explicit sweep-wide `--auto-approve`.
  - Codex delegated review is not used.
  - Success is raw LIBERO from the unchanged evaluator, and the first raw success still ends the episode.
- **Per-episode record:**
  - raw (headline), service on the clock, the S300 flag, and the end cause (success / service_cap / infra / harness);
  - calls, steps, and component times (prep, start-up, model, simulator with sub-timers, video, run_state, observe, finalization);
  - f, and per-call input, reasoning and visible tokens with latency;
  - preparation attempts, guard trips and retries;
  - envelope rejections, realized transit and contact mm/step, stalls, gap events, schema rejections;
  - a failure class: relation grounding, grasp/slip, reach/joint limit, height/contact, plan grounding, infra-prep, infra-model or harness;
  - strict and forbidden contacts, as diagnostics only. `held_ever` labels are not used.
- **Headline.**
  - The coverage map: rows t0–t9, each showing raw ✓/✗, service s and end cause.
  - The S300 count, and the 5th-best service at the measured f and normalized to f = 1.0.
- **Attribution.**
  - Paired per-task component deltas.
  - A raw-coverage change of ±1 task is treated as noise unless the trace shows a cause.
  - A change for the next build needs a mechanism that recurs on 2 or more tasks, or a structural gap.

---

## 5. v3 disposition

**Freeze v3 at r11.** `src/gpt_field/agent_speedup_v3*.py` is byte-identical to `runs/agent_speedup_v3_t1_012/frozen_runtime`. There is no r12, no edit, and no frozen copy is touched.

**Why.**
- Every v3 success was a raw-only success on t1 episode 0 (r1, r2, r5, r6). Each ran at least 81.6% uc7-style execute fragments, inside v3's adaptive subclass.
- r6 t1_008 used 40 calls at 33.4 s/call, against uc7's 13 calls at 14.0 s/call on the same reset:
  - model time was 97% of the +1,186 s control gap;
  - 36 of 39 follow-up calls were triggered by a gate or guard;
  - 21 `geometry_patch` calls cost 568 s.
- v3 is 0/7 since r6, plus 0/3 today.
- All 9 behavioural failures since r4.1 ended in `EvidenceAcquisitionFailed`:
  - 7 were model escalations, and 2 were acquire_evidence requests with no view candidate;
  - the zero-candidate ending applies only to the 8 failures from r7 on;
  - in t1_005 the gate actually permitted the motion.
- The model's table-height estimates were 17–42 mm low, beyond its declared 8–15 mm uncertainty.

**Harvest into uc8.**
- Imported unchanged: `agent_speedup_v3_video.py` and `agent_speedup_v3_performance.py`.
- Copied with parameters: the guard and deadline logic from `agent_speedup_v3_transport.py`.
- Copied as text: `COMPACT_OUTPUT_INSTRUCTIONS` and the 503 availability-retry rule.
- Optional: `TerminalActTimingMixin`.

**Discard:**
- geometry memory and every sweep, coverage, inventory, obstacle-core and candidate gate;
- programs and operations;
- `scene_update` and `motion_context`;
- the visual watch, interaction monitor, operation runtime, stereo and view proposer;
- the repeat and continuation guards;
- escalate / `EvidenceAcquisitionFailed`;
- the history projection;
- the adaptive contact cap.

**If v3 is ever re-run,** first fix D2, the truth skip (base runner `:1124-1126`, `:1535-1536`; `scripts/evaluate_agent_speedup_v3.py:283-289`).

---

## 6. User decisions

**Required before Sweep 1**
- **D1. Approval mode for each sweep.** Choose one batched manual review session or an explicit sweep-wide `--auto-approve`. Confirm that Codex delegated review is not used. If you do use it, decide whether its compute counts toward service: today 138–285 s was billed, which alone breaks S300.
- **D2. Per-episode service cap.** 900 s is proposed. It is enforced through the operator-stop file, and a capped episode counts as a failure.
- **D3. Artifacts.** Accept the compact 640×384, one-frame-per-step video instead of 1600×960 (about −59 s on t1). Also choose whether `trace.jsonl` keeps full per-step PNGs (about 0.5 GB per run) or keeps them only at decision steps.
- **D4. End of a successful episode.** Keep stop-at-first-raw-success, with strict reported as a diagnostic. This is decision D1(a), pending since 2026-10-21.
- **D5. Go-ahead.** Approve building uc8 and announcing Sweep 1, which you start. Then the Phase 2 sim-only session, Sweeps 2–3, and the held-out sweep, each announced before it runs.

**Conditional or optional**
- **D6 (constraint 4).** Execution effort high → medium (U1), only if Sweep 3 misses at the measured f. The quality impact is unknown.
- **D7.** A speculative first executive call during the excluded human review wait (U2): an accounting rule. It applies only under manual review, and the call is discarded if the plan is edited.
- **D8.** A model `give_up` action, extending the 2026-10-20 "Allow early stop" approval. Default: off.
- **D9.** An adjacency default for regions with a hidden extent. Default: off, because only one task supports it.
- **D10.** Later, only if f stays at or above 1.25: Plan B's early dispatch of a complete, validated action before the trailing prose. This changes the invariant that no partial answer authorizes action.
- **D11.** Sweep size. Default: 10 × episode 0 per build, plus 10 × episode 1 held out on the final build. Alternative: 10 × {episode 0, episode 1} per build, which doubles live time.

---

## 7. Dropped or deferred

- **B's S1/S3 asynchronous executive.** It needs an in-flight state machine with sync and supersede rules, lock surgery, and acceptance of stale images. That conflicts with the 10-26 instruction against more complex state machines. Change 3.3 gets most of the post-gripper saving without concurrency.
- **B's S2 early dispatch.** Deferred to D10.
- **B's hedged duplicate requests.** Rejected.
- **A's pre-started worker.** Dropped: it saves about 4.6 s, adds a thread around the credential-stripped environment, and does not fit batch preparation.
- **A's M1 session and the R0 replay gate, B's fake-test, scanner and smoke gates, and C's unit-test and guard-replay gates.** Dropped: Sweep 1's timers measure the simulator split, and testing is live-first.
- **C's `give_up` and adjacency default.** Moved to D8 and D9.
- **C's 20-episode first sweep.** Replaced by paired 10-episode sweeps plus one held-out sweep.
- **C's resolver edit.** Dropped: macros are dormant.
- **Contact cap 5 → 8 mm.** Deferred because contact precision is at risk.
- **Lazy rendering and per-control-step contact sampling.** Taken only if the timers show they dominate. The sampling change also alters the contact audit, so it is a user decision.
- **Prompt caching.** Dropped: only about 3% of input is cached today, and the route's cache key is unverified.
- **Fewer or smaller images.** Dropped: at most 0.6% of model time by tokens.
- **Rewriting old history (v3's projection).** Dropped: no latency gain was measured. 3.2 cuts text and stays append-only.
- **Any stall-triggered continuation.** Excluded.
- **v3 r12.** Excluded.

## 8. Main uncertainties

- **f.** Today's calls ran ×1.35–1.51 slow, possibly because of concurrent arms. S300 on the median task holds only at f ≤ about 1.1 without U1.
- **50 mm transit.** It is untested beyond the model and could cost raw coverage. The sim-only session and Sweep 2 decide.
- **The simulator-step split.** It is unmeasured until Sweep 1.
- **Call-economy savings.** The coefficients are observational (±20–35%). Terse context or fewer checkpoints could add calls or empty carries.
- **n = 1 per task per sweep.** That is about ±1 task of noise; uc7's two t5 runs took 270 s and 328 s.
- **t6.** The mug grasp and upright recovery remain a gap even once the side is correct.
- **The compact-instruction effect.** It is observational. The bounds limit the damage of a hang but do not cure its cause.
