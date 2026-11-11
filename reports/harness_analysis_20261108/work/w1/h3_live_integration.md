# H3 (foundation_vision) live integration into uc7.1: what it does and what it measurably changed

Key: `h3_live_integration`. This was a read-only analysis: no simulator runs, no model calls, no GPU work. The only computation was small offline Python over saved run artifacts. The scripts are in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts/`.

Short answer: **as integrated, H3 does not measurably help any of the four goals.**
- **Geometry.** None of the 22 executive requests (T04 + T08) carried a usable 3D target position.
- **Influence on Astra.** No Astra response references the vision output except two that say "Vision geometry is unavailable".
- **Collision.** By construction, H3 cannot affect collision avoidance.
- **Cost.** It adds about 14 s of vision time per episode and about 2.3–2.5k prompt tokens per call. Those tokens are re-sent on every later call and make up 24–26 % of all input tokens.
- **Zero-shot.** It relies on an operator-supplied, per-scene obstacle name (`--vision-query can` / `bottle`).

Underneath this, the raw SAM 2D detections and the camera calibration were already good enough to place the added obstacle to about 1–2 cm. Plain ray triangulation achieved that offline, with no DA3. The pipeline throws that evidence away. The fix is to change **how perception is consumed**, not to add vetoes.

---

## 1. How the integration actually works (code, as executed)

```
scripts/evaluate_foundation_vision_task.py
  ├─ preload SAM 3.1 detector + DA3-LARGE-1.1 on cuda:0      (19.7 s + 4.9 s, runner.log line 13)
  ├─ monkey-patch the uc7.1 runner: execution_options / runtime_inventory / execute_reviewed
  └─ client_factory → FoundationVisionAgentClient(AgentFirstV2Client)          (lines 135-151)
        targets = every reviewed plan part as TargetSpec(id, text)             (live.py:74-82, :79)
                  + --vision-query extras ("can" / "bottle")                    (live.py:182-184)
        request_agent(view, ctx):  captured_at = perf_counter()                (live.py:301-336)
          └─ parent validates → build_payload(mode="agent")                   (live.py:274-299)
               ├─ _get_snapshot: SAM(all targets × 2 views) → DA3(2 views) → build_geometry
               │     target 3D only if: 1 candidate per view in BOTH views, DA3 re-projection
               │     mask agreement ≥ 0.9 in each direction, ≥3 points with 2-view depth support
               │     (engine.py:118-155, :200; geometry.py:32-37, :199-266)
               ├─ _summary: ≤12 objects × ≤3 candidate 2D boxes + 64 sampled scene points
               │     + ~15 disclaimer flags; stale if wall-clock age > 5 s     (live.py:108-170)
               ├─ context.foundation_vision = summary                          (live.py:284)
               └─ payload.instructions = VISION_INSTRUCTIONS (+1,403 chars)    (live.py:34-55, :290)
        Astra still writes its own obstacle_map; routing checks only those boxes,
        only for transit-profile move segments, only at the grip site
        (agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/agent_speedup_v2_user_check_fragments.py:293-310;
         .../agent_speedup_v2_user_check_routing.py:1-5).
        engine.query_path / geometry.assess_path exist but are never called (grep: only definitions).
```

- **Same code and settings as H1.** All 22 uc7.1 modules in the H3 launch snapshot (`runs/foundation_vision_t04_can_obstacle_20261105_001/source_at_launch`) have the same hashes as the H1 extraction. The only exception is `agent_speedup_v2_user_check_sim.py`, where the main tree adds ordered-task evaluation. The H1 transit100 run snapshot matches the H1 extraction too.
- **Same launch flags.** Both arms set `GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M=0.10` and pass `--auto-approve --planning-video --request-deadline-s 180`.
- **Same executive context.** The `profiles` sent to the model are identical: transit 0.5 m target / 0.10 m per action; contact 0.08 m / 0.005 m. The only added context key is `foundation_vision`.
- **Conclusion.** The H1 ↔ H3 comparison is code-controlled. The differences are the vision payload and instructions, model sampling, and the day the runs were made (H1 on 2026-11-02/03, H3 on 2026-11-05).

---

## 2. Evidence table: H3 vs H1 transit100 on the same scenes

| | H1 T04 | H3 T04 | H1 T08 | H3 T08 |
|---|---|---|---|---|
| Calls (plan + exec) | 1+13 | 1+11 | 1+21 | 1+11 |
| Policy actions | 549 | 604 | 843 | 590 |
| Control w/o video (s) | 287.3 | 432.4 | 467.4 | 464.1 |
| Executive model stream time, Σ completion_s (s) | 198.5 | 311.1 | 333.6 | 340.8 |
| Mean / median per call (s) | 15.3 / 13.4 | 28.3 / 24.3 | 15.9 / 12.2 | 31.0 / 28.8 |
| Time to first visible token per call (s) | 10.6 | 21.6 | 11.8 | 22.5 |
| Reasoning tokens per call | 123 | 259 | 142 | 254 |
| Visible output tokens per call | 264 | 261 | 214 | 262 |
| Input tokens per call | 49.4k | 59.3k | 72.8k | 57.1k |
| Visible decode rate (tok/s) | 57.0 | 39.3 | 52.7 | 30.9 |
| Vision processing (s) | – | 14.7 | – | 13.2 |
| Request chain − stream − vision (s) | 1.8 | 2.3 | 2.8 | 2.0 |
| Non-model control (s) / per action (ms) | 87.0 / 158 | 104.3 / 173 | 131.0 / 155 | 108.0 / 183 |
| Non-decision step wall incl. video, median (ms) | 275 | 259 | 296 | 285 |
| Forbidden contact samples | 3372 (hand vs mug) | 0 | 3826 (3504 finger vs bottle) | 400 (pot vs burner) |
| Usable vision 3D target centres | – | 0 / 11 requests | – | 0 / 11 requests |

Sources:
- **Summaries.** `runs/*/result_summary.json` (H1) and `runs/foundation_vision_*/analysis.json` (H3).
- **Per-call figures.** Deltas of the cumulative budgets in `trial/requests/*.result.json` and `*.stream.json`, computed by `scripts/req_stats2.py`.
- **Step walls.** `trial/episode.events.jsonl`, computed by `scripts/step_times.py`.

---

## 3. Findings (by mechanism)

### F1 — CRITICAL (verified): perception never produced usable geometry, because of four distinct gates

**Mechanism.** H3 promotes a target to 3D (`geometry_usable`) only if all of the following hold:
- SAM returns exactly one candidate in **both** the fixed external view and the moving wrist view;
- every DA3 re-projection mask agreement is ≥ 0.9 in each direction (`geometry.py:36`, `:228`, `:240-241`);
- at least 3 points have two-view depth support, i.e. agreement within 2 cm + 2 % (`geometry.py:32-34`, `:236-237`).

Ambiguous candidate sets are dropped, not ranked (`engine.py:153`).

**Evidence.**
- 0 usable target centres in 22 accepted requests. Source: `analysis.json` in both H3 runs, recomputed by `scripts/vision_frames.py`.
- Causes by target-frame, T04 (77) / T08 (66):

| Cause | T04 | T08 | Mechanism |
|---|---|---|---|
| No SAM candidate in either view | 34 | 42 | **Prompt form.** Six of the 11 plan-part prompts returned *zero* candidates in every one of their episode's 11 frames: T04 A "white mug: textured cylindrical body", B "white mug: loop handle on its side", E "yellow and white mug: loop handle on its side"; T08 A "near moka pot: whole faceted body beside stove", B "near moka pot: curved side handle toward stove", E "stove: rectangular top surface around circular burner". These six prompts account for 33 of the 34 (T04) and 33 of the 42 (T08). The bare nouns `can` and `bottle`, supplied as operator extras, were found in the external view in 11/11 and 10/11 frames. |
| Candidate in only one view | 26 | 17 | **Wrist camera coverage.** The wrist view sees a target only when hovering over it, and the two-view rule then rejects it. |
| Several candidates (identity ambiguous) | 11 | 3 | **Relational qualifiers.** SAM ignores words like "left plate" and "right moka pot" and returns both plates or both pots (F7). |
| Cross-view association rejected | 6 | 4 | **The 0.9 gate.** The can at frame 0 had agreement 0.64 (external→wrist) and 0.91 (wrist→external). The bottle had 0.04 / 1.0 at step 243 and 0.83 / 0.85 at step 328. |

- **The rejected detections were geometrically correct.**
  - Method (`scripts/triangulate.py`, offline): take the two SAM mask centroids, back-project them through the public calibration, and take the midpoint of closest approach. No DA3 is used.
  - Results against layout truth (`output/layouts/libero_10_t4_can_obstacle_20261103_001/layout_manifest.json`, `.../libero_10_t8_bottle_obstacle_20261103_001/layout_manifest.json`): the can at frame 0 is **1.2 cm** off (xy), with z error −0.2 cm and a ray gap of 2.1 cm; the bottle is off by **1.7 cm** at step 243 and **0.4 cm** at step 328.
- **DA3 two-view consistency is poor scene-wide.** In the live snapshots:
  - external-view pixels with cross-view support range over 0.1–26 % of valid pixels per frame;
  - contradicted pixels reach 35 % (external, frame 0);
  - per-frame scale alignment is not logged: `provenance.alignment` is `null` at preload (runner.log line 13), and the snapshots do not contain it.

**Improvement (general, removes vetoes).**
1. **Query objects, not parts.** Query SAM with the plan's object head with spatial words removed. The plan schema already defines the `"<object>: <part>"` format; see the `object_words` docstring in `src/gpt_field/agent_speedup_v2_plan.py:15-19`. Segment parts such as handles *inside* the object mask rather than as separate open-vocabulary prompts.
2. **Choose instances with the plan's own cells** (F7), then by continuity.
3. **Locate objects without DA3:**
   - two-view ray triangulation when both views have a mask;
   - otherwise one external view intersected with a support plane. On the saved masks, using the *true* table height, the centre error was 1.5 cm (can) and 0.6 cm (bottle), and the top height came out 1.8–1.9 cm too high, i.e. conservative (`scripts/single_view2.py`). The plane height has to be estimated in deployment.
4. **Replace the 0.9 bidirectional gate and the two-view requirement with inflated boxes that carry an uncertainty**, rather than vetoes.
5. **Restrict DA3** to large planar supports, with the robot masked out using forward kinematics, and log its per-frame scale.

**Validation.**
- Offline replay of SAM and the locator on the 22 saved frame pairs. The request JSONs contain both RGB images and the calibration.
- Metrics: detection recall per plan object, and centre/extent error against layout truth.
- Then the all-task sweep on one frozen build.

**Expected effect.** For the added obstacles, saved detections suggest localisation in ≥ 21 of 22 frames to ≤ 2 cm. Recall for plan targets with object-level prompts is **unverified**: it needs the GPU replay.

### F2 — CRITICAL (verified): consumption is advisory only, so H3 cannot change collision avoidance

**Mechanism.**
- Settings: `automatic_depth_routing: False`, `automatic_action_gate: False`, `routing_geometry_source: astra_estimated_obstacle_boxes` (`scripts/evaluate_foundation_vision_task.py:102-109`; `live.py:164-166`).
- Routing runs only for `profile == "transit"` move segments (`.../agent_speedup_v2_user_check_fragments.py:293`), and only checks the grip-site path against the boxes Astra wrote (`.../agent_speedup_v2_user_check_routing.py:1-5`).
- Contact-profile descents, the held object and the arm are never checked.
- `engine.query_path` and `geometry.assess_path` are unused.

**Evidence that Astra did not use it.**
- Only 2 of 22 outputs mention vision, both to dismiss it:
  - T04 #1: "Vision geometry is unavailable, so initial positions are approximate from the calibrated views."
  - T08 #1: "Vision geometry is unavailable for the targets, so the initial handle location is approximate from the calibrated views."
- The obstacle-map updates (T04 calls 1 and 7, T08 calls 1, 4 and 7) are justified by Astra's own reasoning. Examples: T04 #7 "The new map protects the placed mug and removes the yellow mug from avoidance"; T08 #4 "The stove top is estimated near z=0.93 m from cross-view rays".
- Each updated box contains only 0–5 of the 64 sampled scene points, of which ≤ 2 are cross-view supported (computed). The boxes cannot have been built from vision points.

**Improvement.** Make perception an *executor input*:
- automatically merge perception boxes for every static non-target object into the obstacle map (F1 locator), excluding the held object, the current grasp target and the robot (via forward kinematics);
- route **all** free-space motion against that map, including the approach phase of contact segments (not only `transit`);
- let Astra stop writing `obstacle_map` except for semantic overrides.

This removes an error-prone model output rather than adding a veto.

**Validation.** Paired same-day runs (H1 vs H1 + perception map) across all tasks, with added-obstacle variants. Metrics:
- forbidden samples by class;
- obstacle-box truth coverage (`scripts/box_accuracy.py`);
- number of `route_unavailable` / start-blocked events.

**Expected effect.** Astra's first obstacle boxes in the H1 runs covered 30 % (T04 can) and **0 %** (T08 bottle) of the true obstacle volume. The T08 box was 6.6 cm too low. The hand then clipped the bottle top at action 37: 3,504 robot_object samples, plus a bottle tilt and recovery calls (H1 T08 calls 3–5). A perception box with about 2 cm error would have covered both.

### F3 — HIGH (numbers verified; attribution likely): per-call latency doubled, and vision bought no time back

**Decomposition, T04.** Control time is 145 s higher than H1:
- +112.6 s model stream time;
- +14.7 s vision inside the request path (`live.py:256-264`, synchronous before HTTP);
- +0.5 s request overhead;
- +17.3 s non-model control, of which about 8.7 s is from 55 more actions.

**Two causes inside the +112.6 s stream time:**
1. **The service was slower on 2026-11-05 (likely).**
   - Decode rate: the visible decode rate across the 164 calls of the seven uc7.1 runs is 50.5 tok/s; H3 ran at 30.9–39.3 tok/s, against 48–57 tok/s for every individual uc7.1 run.
   - Reasoning rate: about 15.3 tok/s for H3 against 20.4 tok/s for uc7.1, from a regression of time-to-first-token on reasoning and input tokens.
   - Fitted model: time to first visible token ≈ 3.89 s + 0.049 s × reasoning tokens + 0.0196 s per 1k input tokens, R² = 0.92 (`scripts/latency_model.py`).
   - At 11-02/03 throughput, H3 T04 would have used **252 s**, not 311 s (+59 s residual), and H3 T08 **249 s**, not 341 s (+92 s residual). Both H1 runs had small negative residuals.
   - Independent check: the identical, vision-free initial-plan request decoded at 61–64 tok/s on 11-05 against 84–89 tok/s on 11-03. T08 plan time was 36.3 s vs 20.6 s.
2. **Twice the reasoning per call (verified numbers, cause unknown).** H3 averaged 259 and 254 reasoning tokens per call. That is above all seven uc7.1 runs (111–219). On T04 this explains about +42 s net at equal throughput (+1,246 reasoning tokens, 2 fewer calls).
   - The input-token increase itself costs only about 0.2 s per call.
   - Whether the 7–8k-character vision block and the +1.4k-character caveat instructions *induce* the extra reasoning is a **hypothesis**. With n = 1 per arm it cannot be separated from sampling.

**T08.** The call count fell from 21 to 11, so model time is roughly equal (341 vs 334 s). At equal throughput H3 would have been about 98 s faster. Nothing links those 10 fewer calls to vision (see F10).

**Improvement.**
- Take perception out of the prompt path. Run it concurrently with the LLM call, because the executor consumes it. The model call takes 13–79 s and vision 1.0–2.9 s, so vision adds 0 s to the critical path.
- When a summary is still sent, send only compact facts (F4).
- Record per-call server throughput, and compare arms interleaved on the same day.

**Expected effect.** −13 to −15 s per episode from the vision time. If the reasoning increase is payload-induced, a further −0 to −7 s per call (hypothesis).

### F4 — HIGH (verified): every past (stale) vision summary is re-sent on every call

**Mechanism.** The executive history keeps every textual record and trims only images (`.../direct_history.py:45-57`, `.../agent_speedup_v2_user_check_agent_client.py:417`). The vision block sits inside the user message text, so it is never dropped.

**Evidence.**
- Per-request summary size: 6,946–7,921 characters, about 2.3–2.5k tokens. This was calibrated from the first-call input-token difference against H1 (+2,990 / +2,740 tokens for vision plus +1,403 instruction characters, about 3.1 characters per token).
- By the last call, the request carried 86,135 (T04) and 78,099 (T08) characters of vision text, about 25–28k tokens of a 101–106k-token input.
- Over the episode, about 168k (T04) and 150k (T08) tokens of vision text were sent: **26 % / 24 % of all input tokens**.
- Content of each summary:
  - 0 usable centroids;
  - 1–7 SAM 2D boxes;
  - 64 uniformly subsampled scene points, of which only 1–30 are cross-view supported (computed per request);
  - about 15 constant disclaimer flags (`automatic_action_gate`, `collision_free_certified`, …).
- Prompt caching gives no relief: cached input tokens were 0 of 652k (H3 T04), 3.3k of 628k (H3 T08) and 16k of 642k (H1 T04).

**Improvement.**
- Send only the current snapshot, as an ephemeral context block, and replace older snapshots in the history with a one-line "superseded" stub.
- Send facts only: per-object located/not-located, box, uncertainty and age in robot steps.
- Move the caveats into the instructions once.
- Never send sampled scene points.

**Validation.** Offline: rebuild the 22 payloads and measure tokens. Live: an A/B on reasoning tokens per call.

**Expected effect.** −150 to −170k input tokens per episode (subscription budget), and about −2 to −3 s per episode of direct prefill time. Possibly fewer reasoning tokens (hypothesis, F3).

### F5 — HIGH (verified): zero-shot violation — the obstacle exists in perception only because the operator named it

**Evidence.**
- Launchers pass `--vision-query can` (T04) and `--vision-query bottle` (T08). See `runs/foundation_vision_t04_can_obstacle_20261105_001/run.py:48` and `run_state.json` `vision_query`.
- Without these, vision targets are only plan parts (`live.py:74-82`), i.e. task objects.
- The T08 plan's `scene` text ("Two moka pots beside a single-burner stove on a wooden counter.", `trial/approved_plan.json`) does not mention the bottle. Deriving queries from the plan would therefore miss it.
- In T04 the red mug, which Astra itself boxed as an obstacle (box [−0.22, −0.045]…[−0.09, 0.075] around the true (−0.202, 0.018)), was never queried.

**Improvement.** Class-agnostic obstacle discovery:
- external-view depth or a support-plane fit, followed by connected components above the plane;
- minus the robot (forward kinematics) and the task objects;
- or a SAM "object" sweep labelled afterwards.

Drop `--vision-query` from the protocol.

**Validation.** On the saved T04/T08 frames, without naming the obstacle: can/bottle recall, and false positives on the robot.

**Expected effect.** Restores zero-shot eligibility, which is a precondition for any obstacle benefit.

### F6 — MEDIUM (verified): staleness is measured in wall-clock time, which is the wrong clock for a paused simulator, and it also disables tracking

**Evidence.**
- Freshness is judged at payload construction: age 1.14–2.94 s against `max_age_s = 5` (`live.py:110-111`; analysis.json `age_at_payload_s`).
- When the response is accepted, the evidence is **12.8–78.7 s old** (median 25.6 s for T04, 30.0 s for T08; `scripts`: requests.jsonl `elapsed_s`).
- The simulator is synchronous, so the scene is unchanged during model latency. Validity really depends on whether the robot or held objects moved, not on wall time.
- The engine keeps an identity only if consecutive frames are ≤ `max_age_s` apart (`engine.py:205-206`). The live frames were 24–86 s apart (computed from `captured_at_s`), so **track continuity can never engage**. Every frame is a fresh "new candidate".
- A same-step re-request, such as a schema retry, reuses the snapshot with its old capture time and blanks all geometry as "stale" (`live.py:247-252`; asserted by `tests/test_foundation_vision_live.py:158-175`).
- The engine's `need_observation` events are not forwarded to Astra (`_summary` omits `events`).

**Improvement.** Index validity by observation step and by motion:
- static scene evidence stays valid until any contact or object-motion event;
- evidence about the held object and the robot is invalidated at the next motion;
- on real hardware, compare the sensor timestamp with the action start.

Forward `need_observation` as a view request.

**Expected effect.** Enables tracking of moved and held objects (F8) and placement verification. No time cost.

### F7 — MEDIUM (verified): the plan's own pointing cells are discarded, though they resolve every ambiguity

**Mechanism.** Every plan part carries `x` and `w` cell ids: an 8×8 grid numbered 1–64, with 0 meaning not visible (`src/gpt_field/agent_speedup_v2_plan.py:57-58`, `:91-92`, `:199-201`). `targets_from_plan` keeps only `id` and `text` (`live.py:79`). `TargetSpec.seed_pixels` exists but is unused (`engine.py:137`).

**Evidence (computed, `scripts/cell_disambig.py`).** The part's external cell overlapped exactly one candidate in **11/11** ambiguous T04 frames (C "left plate", cell 41, picks the [0, 308, 58, 375] mask, not the right plate) and **3/3** ambiguous T08 frames (D, cell 38). All 31 single-candidate external detections were consistent with the plan cell.

**Improvement.** Use plan cells as the initial instance prior (mask–cell overlap), then step-indexed continuity (F6). This is general because every plan already produces cells.

**Expected effect.** Removes all 14 ambiguity frames. It does not fix the prompt-form or one-view gates (F1).

### F8 — MEDIUM (verified geometry; partially likely): the T08 400 samples come from placement touchdown at the edge of the cook region, which H3 could not have prevented

**What happened.**
- Steps 243–296 are call 5's contact `approach` of [+0.025, +0.010, −0.060] m from a grip height of 1.095.
- At action 277 (grip z 1.049) the first pot's base touched `flat_stove_1_collision_burner`.
- The controller kept commanding about −4 mm per step while the measured z-rate decayed. The pot slid about 7 mm in y until `segment_stall` at action 296, and the gripper opened at 297 (trace.jsonl, computed).
- So the forbidden samples accrue over about 20 actions of pressing during placement, not during transport.

**Why "forbidden" (computed from truth).**
- The cook-region site is axis-aligned: both pots' `d_site` values from `trial/physical_truth.json` put its centre at (−0.0524, −0.2087).
- The first contact point (−0.1306, −0.2062) is at local x = −78.2 mm, outside the 75 mm half-size plus 2 mm tolerance. That is the "on the region host but outside the region" branch (`src/gpt_field/agent_speedup_v2_contacts.py:233-236`).
- The same geom pair is classified as `support` from action 277/278 (`monitoring/collision_classification_review.json`).
- Astra had chosen the "far-left portion of the stove". It intended a remaining gap ("lower 60 mm slowly, then inspect the remaining support gap"), but touchdown came about 1.4 cm earlier than its estimate of how far the pot hangs below the grip.

**Why H3 could not help.**
- The stove prompt (E) had 0 candidates in all 11 frames.
- Contact segments are never routed or checked (F2).
- Vision runs only at decision points.
- There is no held-object geometry.

**Improvement (general).**
- After a grasp, measure the held object's lowest point relative to the grip site once (wrist + external masks → F1 locator).
- Measure the support-surface height.
- Plan descents to touchdown plus a small margin, and terminate contact descents at touchdown (z-rate drop) instead of at a stall.
- Pick placement points inside the support's visible footprint, using its mask.

**Expected effect.** The pressing window (about 19 actions, during which the 400 samples accrued) should shrink to a few actions. This also applies to every place-on task.

### F9 — MEDIUM (verified): perception is fail-closed — a vision exception aborts the episode

**Evidence.**
- Inference exceptions re-raise (`live.py:257-262`, `:325-328`).
- The executive catches only `SchemaError` and re-raises everything else (`agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/agent_speedup_v2_user_check_agent.py:272-288`).
- The size limits raise `ValueError`: 28,000 characters for the summary and 56,000 for the observation fields (`live.py:168-169`, `:285-286`). The largest T04 observation text was about 34.0k characters (call 8), so the margin is about 22k characters.
- This behaviour is asserted by `tests/test_foundation_vision_live.py:209-218`.
- No errors occurred in either run.

**Improvement.** Degrade gracefully: send `foundation_vision: {status: "unavailable", reason}` and continue. A component that currently adds no value must not be able to end an episode, and the all-task sweep runs on GPUs shared with training.

**Expected effect.** Removes a new single point of failure (fewer episode losses). There is no measured incident yet.

### F10 — MEDIUM (hypothesis): H3's better outcomes on these two episodes are not attributable to vision, but one channel deserves a test

**Evidence.**
- H3 had fewer calls (12 vs 14; 12 vs 22) and fewer forbidden samples (0 vs 3,372; 400 vs 3,826).
- The H1 contacts came from Astra geometry errors: in T04 the hand hit the mug handle during the grasp descent (`robot_nonfinger_manipulated_object`, first at action 114); in T08 the approach at grip z = 1.035 was below the bottle top at 1.046.
- Astra's own obstacle boxes were more accurate in the H3 runs. First box coverage of the true obstacle was **83 %** (T04) and **51 %** (T08) in H3, against **30 %** and **0 %** in H1. With clearance, H3 reached 100 % / 100 % and H1 77 % / 0 % (`scripts/box_accuracy.py`).
- The only vision content that could have helped is the SAM 2D bounding box of the operator-named obstacle. That box was accurate: its back-projected centre is about 1 cm from truth.
- No response cites it, and both H3 first replies call vision "unavailable".
- With n = 1 per arm, a different server day and stochastic sampling, the improvement may be chance.

**Improvement / validation.** Run an ablation on the same day and seeds: H1, H3, and H3 with 2D boxes only. Measure first-box coverage, approach clearance and calls.

**Expected effect.** If it holds, the effect argues for automatic perception boxes (F2), not for longer prompts.

### F11 — LOW (verified): the scene-point sample and the robot-inclusive cloud are noise in the prompt

**Evidence.**
- The 64 points are `linspace` samples (`live.py:143`) of a 2,048-point subsample (`geometry.py:291`) of about 520k valid pixels from both views.
- That cloud includes the robot, the walls and the background; robot removal is a listed limitation (`engine.py:306`).
- Within the T08 stove footprint, the sampled z values mix stove (≈ 0.91–0.92), pot (≈ 1.0–1.05) and arm (1.2–1.4) points.

**Improvement.** Remove the points from the prompt. Use the full cloud only inside the executor, with the robot masked out.

**Expected effect.** About −0.9k tokens per request, removed from every replay (part of F4).

---

## 4. Direct answers to the investigation questions

**(a) What `context.foundation_vision` contained, and whether Astra used it.**
- Content: 6,946–7,921 characters (about 2.3–2.5k tokens) per request; 22 / 22 requests carried it.
- Fields: version and mode, frame and clock fields (`age_s`, `max_age_s`, `status = estimated`), and per-target objects (status, `geometry_usable`, centroid/bounds/normal, which were always null, per-view candidate counts and ≤ 3 2D boxes with scores).
- It also had a scene block (64 points plus support and contradiction counts) and constant disclaimers.
- Usable geometry: 0 / 22.
- Use by Astra: 2 / 22 responses mention vision, only to say it is unavailable. The obstacle-map updates (T04 #1, #7; T08 #1, #4, #7) are explained by Astra's own image reasoning (quotes in F2). The boxes do not coincide with vision points.

**(b) Time.** See the F3 decomposition.
- The settings match H1 transit100: same module hashes, the same 0.10 m transit cap, the same envelope and contact profiles, and the same routing code.
- The 145 s T04 gap breaks down as follows. The stream-time figures (≈ +59 s server, ≈ +42 s reasoning) are model estimates within the +112.6 s; together they leave ≈ +11 s unexplained by the latency model.

| Part | Seconds |
|---|---|
| Slower service on 11-05 | ≈ +59 |
| More reasoning tokens per call, net of 2 fewer calls | ≈ +42 |
| Vision processing | +14.7 |
| More actions and per-action overhead | +17.3 |
| Request overhead | +0.5 |

- The vision payload's direct prefill cost is about 0.2 s per call.
- T08 is a wash, 464 vs 467 s: 10 fewer calls offset slower and longer calls.

**(c) Collision.** The T08 400 samples come from placement touchdown and pressing (actions 277–296) at the edge of the cook region (F8). H3 could not have prevented them (F2, F8). By design, H3's only lever is Astra's own boxes, which matter only for transit segments.

**(d) Few attempts and zero-shot.**
- Vision targets are all plan parts (detector-hostile phrasing) plus per-scene operator nouns, which is a zero-shot violation (F5).
- Staleness uses wall-clock time, which makes tracking impossible (F6).
- Failures crash the episode rather than degrade (F9).

---

## 5. Verdict per goal (H3 as integrated)

| Goal | Effect | Why |
|---|---|---|
| Zero-shot | Negative | Operator obstacle names; plan-part prompts are not detectable (F1, F5). |
| Fast | Negative | +13–15 s vision and +24–26 % input tokens per episode; reasoning per call about 2× (cause open); +24.6 s model preload per process (F3, F4). |
| Collision-avoiding | None by design | Advisory only; 0 usable geometry; contact segments unchecked (F2, F8). |
| Few attempts | Not established | Fewer calls on both scenes, but no causal link to vision (F10). |

## 6. How perception output should be consumed instead (one frozen, general design)

1. **Perception is an executor service, not prompt text.**
   - It runs concurrently with each LLM call on the current frame.
   - Locator: object-level SAM queries (plan object heads plus a class-agnostic obstacle sweep), instance selection from plan cells, then continuity; two-view triangulation, or one view plus a support plane; conservative boxes with an uncertainty instead of the 0.9 / two-view vetoes.
2. **Automatic obstacle injection.**
   - Static non-target objects become routing boxes. The held object, the grasp target and the robot are excluded.
   - All free-space motion is routed, including the approach part of contact segments.
   - Astra's `obstacle_map` becomes an optional semantic override.
3. **Held-object model.** After a grasp, measure the object's extent below the grip and track it through the wrist view. Use it for swept-volume routing and for placement descents to touchdown.
4. **Placement verification.** Use the support mask, footprint and height for target selection, and use touchdown and post-release masks to verify. This replaces stall-until-timeout and the extra inspect calls.
5. **Compact facts to Astra.**
   - Only the current snapshot, with located / not-located per object.
   - Validity counted in robot steps.
   - No history replay and no point samples.
6. **Graceful degradation**, plus per-call throughput logging.

**Acceptance.** An all-task sweep on one frozen build with same-day interleaved arms. Metrics:
- calls per task and seconds per task, with server tok/s logged;
- forbidden samples by class;
- offline box coverage against layout truth;
- vision recall per object class.

## 7. Older defects checked against H3

- **Mirrored policy images: not applicable (verified).** Triangulating StepView-upright SAM masks with the public calibration lands within 0.4–1.7 cm of truth. A mirror would put y on the wrong side.
- **`astra.py` `_canonical` sort_keys: not applicable to the H3 executive wire (verified).** The executive uses `chatgpt-subscription-http-sse-ordered-v1`; `result.json` shows `output_key_order` with belief first. `live.py`'s own `_json(sort_keys=True)` is used only for logs, size checks and fingerprints.
- **Axis-angle sign jumps: harmless (verified).** H3 converts with `_rotation_matrix` (`src/gpt_field/robot_tool_geometry.py:235-244`), and the two representations give the same matrix to 2e-16. The wrist calibration is also cross-checked against state8 (`bridge.py:69-72`).
- **Zero-rotation stale OSC goal:** this is controller-side and inherited unchanged from H1. It was not assessed here.

## 8. Scripts (offline, saved artifacts only)

All under `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts/`:
- `req_stats2.py` — per-call token and latency deltas;
- `latency_model.py` — throughput regression;
- `vision_frames.py` — per-frame vision outcomes;
- `triangulate.py`, `single_view2.py` — DA3-free localisation;
- `cell_disambig.py` — plan-cell instance selection;
- `box_accuracy.py` — obstacle-box coverage against truth;
- `step_times.py` — per-step wall time;
- `outputs.py` — Astra outputs.

## 9. Not covered

- No model was re-run: SAM recall for object-level prompts is untested, and no DA3 depth accuracy was measured because there is no depth ground truth.
- The single-view estimate used the *true* table height.
- H2 and the H1 controller internals were outside this focus.
- Statistical claims rest on one episode per arm.
