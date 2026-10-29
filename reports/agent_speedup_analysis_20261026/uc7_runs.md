# v2 user-check line (focus uc7): where time goes and why runs fail

Read-only analysis, 2026-10-26. No simulator, model or HTTP calls were made. All numbers come from run artifacts parsed with python3 (stdlib). Helper scripts and intermediate JSON are in `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/scripts/` and `.../scratchpad/analysis/` (`runs_metrics.json`, `steps_metrics.json`, `steps2.json`, `mix.json`, `call_tokens.json`, `table_runs.md`).

## 0. Sources, definitions, caveats

- **Run identity.** All 13 uc7 runs report `user_check.version = agent-speedup-v2-user-check-7` (`runs/agent_speedup_v2_user_check_libero_10_*/summary.json`). Their frozen control code is byte-identical. Hashes of all 37 `agent_speedup_v2*` files plus the runner give two groups: {t3_003, t6_009} and the other 11. The only differing file is `agent_speedup_v2_user_check.html`, the review UI. Today's trials 002/003 also report uc7.
- **Where per-call latency lives.**
  - (a) `<run>_launch/runner.log` (`run.log` for today's trials): one `{"event":"model_call",...,"latency_s",...,"step","wall_s"}` line per execution call. The sum of these latencies equals `timing_components.model_pipeline_s` exactly for every uc7 run (for example t1_001: 182.5 = 182.5).
  - (b) `episode.events.jsonl` `skill_decision` events (`api_latency_s`, `decision_latency_s`, `api_request_log`). Each call is duplicated in the next `executed` event, so these must be deduplicated.
  - (c) `trace.jsonl` action diagnostics (`api_calls[].latency_s`).
  - (d) `requests/*.response.json` `completed_at - created_at`: server-side, 1 s resolution.
  - `requests/*.result.json` `budget.elapsed_s` is a **cumulative** session clock, not a per-call latency. `ledger.json` is empty (`[]`).
- **Service time.** Historical runs have no service clock, so I reconstruct it as `preparation_elapsed_s + approval_to_claim_s + elapsed_s`, all from `summary.json`. Preparation is measured from simulator start, through reset and the plan draft, to simulator close (`scripts/evaluate_agent_speedup_v2_user_check.py:160-209`). Human review wait (`review_phase_elapsed_s`) is excluded.
  - Check against supervisor process wall time: t5_001 runs 274.9 s from spawn to exit (`..._t5_001_launch/supervisord.log`) against 270.2 s reconstructed. t6_009 runs 968.1 s against 962.4 s. The reconstruction therefore under-counts by about 5 s of process start-up.
  - Today's trials use the real service clock (`evaluations/trial_00N/service_clock.json`).
- `decision_s` contains `model_pipeline_s`. The additive decomposition used below is model + policy_compute + observe_transition + simulator_step + video_render + run_state + residual = `control_elapsed_s`; `elapsed_s - control_elapsed_s` covers start-up and finalization. This reproduces the history doc's t1_001 split.

## 1. Per-run table

Latency columns use runner.log. Service time is reconstructed except for today's two runs, which use the service clock.

| run | ver | task | raw | v2 | steps | calls | model_s | lat mean/p50/p90/max (first) | sim_s (s/step) | video_s | run_state_s | observe_s | prep_s (model) | elapsed_s | service_s |
|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| t0_001 | uc7 | t0 | Y | N | 1107 | 14 | 226.0 | 16.1/13.0/27.9/31.5 (31.5) | 137.6 (0.124) | 101.8 | 35.4 | 9.1 | 20.3 (12.7) | 524.1 | 544.8 |
| t1_001 | uc7 | t1 | Y | N | 897 | 13 | 182.5 | 14.0/14.0/17.8/18.9 (17.9) | 108.4 (0.121) | 81.8 | 25.1 | 6.4 | 21.4 (13.7) | 416.3 | 438.1 |
| t2_002 | uc7 | t2 | Y | Y | 621 | 15 | 219.4 | 14.6/11.9/22.4/29.6 (23.0) | 60.2 (0.097) | 55.6 | 12.0 | 3.2 | 21.5 (16.3) | 358.8 | 380.4 |
| t3_003 | uc7 | t3 | Y | N | 1207 | 22 | 356.2 | 16.2/15.5/22.8/33.3 (33.3) | 110.7 (0.092) | 106.2 | 29.6 | 9.7 | 27.1 (21.4) | 623.1 | 650.3 |
| t3_005 | uc7 | t3 | Y | N | 744 | 21 | 258.2 | 12.3/10.8/18.4/23.0 (18.4) | 69.2 (0.093) | 72.9 | 13.3 | 4.2 | 24.2 (18.0) | 427.3 | 451.6 |
| t4_001 | uc7 | t4 | Y | N | 770 | 16 | 225.6 | 14.1/11.8/24.3/28.7 (28.7) | 92.0 (0.119) | 71.1 | 17.3 | 4.7 | 25.3 (16.3) | 423.8 | 449.5 |
| **t5_001** | uc7 | t5 | Y | N | **399** | **7** | 146.8 | 21.0/16.4/31.9/38.5 (38.5) | 41.1 (0.103) | 36.5 | 6.9 | 1.7 | 28.4 (21.8) | 241.7 | **270.2** |
| t5_002 | uc7 | t5 | Y | N | 497 | 10 | 189.6 | 19.0/18.7/27.2/32.3 (26.7) | 50.4 (0.101) | 44.6 | 9.8 | 2.5 | 21.6 (15.1) | 306.0 | 327.8 |
| t7_001 | uc7 | t7 | Y | N | 812 | 13 | 199.5 | 15.3/15.9/20.1/20.4 (18.0) | 94.8 (0.117) | 75.4 | 21.4 | 5.4 | 19.0 (11.9) | 407.8 | 427.0 |
| t8_001 | uc7 | t8 | Y | Y | 902 | 13 | 187.6 | 14.4/11.3/24.2/28.8 (25.8) | 91.7 (0.102) | 81.0 | 25.6 | 6.5 | 26.1 (21.4) | 401.2 | 427.4 |
| t9_001 | uc7 | t9 | Y | Y | 1239 | 19 | 268.8 | 14.1/12.5/20.9/26.2 (26.2) | 105.6 (0.085) | 108.3 | 36.0 | 10.6 | 26.5 (20.1) | 540.4 | 567.2 |
| t6_009 | uc7 | t6 | N | N | 1544 | 28 | 495.9 | 17.7/14.6/29.1/39.8 (39.8) | 184.6 (0.120) | 165.7 | 48.0 | 21.4 | 32.3 (24.3) | 930.0 | 962.4 |
| t6_011 | uc7 | t6 | N | N | 1197 | 27 | 415.4 | 15.4/13.6/28.5/30.1 (28.0) | 137.8 (0.115) | 108.8 | 26.5 | 9.8 | 29.0 (21.2) | 711.0 | 740.3 |
| trial_002 (A) | uc7 | t1 ep1 | Y | N | 830 | 11 | 252.3 | 22.9/20.2/35.7/40.5 (35.7) | 101.8 (0.123) | 79.1 | 24.6 | 6.4 | 28.8 (21.5) | 475.8 | 648.7 (clock) |
| trial_003 (B) | uc7 | t1 ep1 | Y | N | 1019 | 13 | 279.5 | 21.5/16.9/38.9/52.9 (42.5) | 122.0 (0.120) | 25.6 | 0.1 | 7.6 | 24.0 (16.6) | 446.5 | 614.0 (clock) |
| t3_002 | uc6 | t3 | Y | N | 419 | 87 | 1226.6 | 14.1/14.6/19.4/28.1 (20.1) | 40.0 (0.095) | 46.1 | 1.9 | 0.8 | 27.9 (22.2) | 1323.7 | 1351.9 |
| t6_007 | uc6 | t6 | Y | N | 464 | 84 | 1229.3 | 14.6/14.1/21.0/36.0 (8.7) | 56.0 (0.121) | 49.3 | 2.0 | 0.8 | 35.3 (27.6) | 1347.6 | 1383.3 |
| t6_001 | uc1 | t6 | N | N | 1771 | 15 | 84.7 | 6.0/5.9/8.1/9.8 | 185.7 (0.105) | 155.9 | 8.0 | 24.1 | 19.0 (11.3) | 505.0 | 524.3 |
| t6_002 | uc2 | t6 | N | N | 5621 | 45 | 347.6 | 7.9/6.4/12.4/18.9 | 583.3 (0.104) | 535.5 | 26.4 | 73.4 | 16.8 (9.2) | 1625.6 | 1642.7 |
| t6_004 | uc3 | t6 | N | N | 3187 | 21 | 155.8 | 7.4/6.3/12.8/13.0 | 331.3 (0.104) | 292.2 | 22.0 | 22.9 | 15.5 (7.8) | 869.3 | 885.1 |
| t6_005 | uc4 | t6 | N | N | 4078 | 37 | 398.1 | 11.1/10.6/15.9/21.1 | 440.2 (0.108) | 407.1 | 36.9 | 74.2 | 28.8 (20.8) | 1446.2 | 1475.2 |
| t6_006 | uc5 | t6 | N | N | 1076 | 38 | 339.2 | 8.9/8.6/14.2/16.6 | 120.2 (0.112) | 113.9 | 9.3 | 13.4 | 39.1 (31.0) | 625.2 | 664.6 |

Older runs (uc1–uc5) mixed in small `point_*` and `repair` calls, which is why their mean latency is low.

**Today's reruns** (`results/speedup_evidence_20261026T055123Z/evaluations/`):

| trial | arm | outcome | service (s) | breakdown |
|---|---|---|---:|---|
| trial_001 | A | `preparation_error` (MissingCredentials, 0 model calls) | 7.6 | — |
| trial_002 | A | raw success at step 830 | 648.7 | prep 29.1, Codex auto-review 138.1, execution start at 168.9 s, first action at 214.7 s, raw at 645.8 s |
| trial_003 | B | raw success at step 1019 | 614.0 | prep 24.3, auto-review 138.5, execution start at 163.7 s, first action at 216.4 s, raw at 612.0 s |
| trial_004 | B | `evaluation_budget_exhausted` | 901.0 | one plan request in flight from 6.6 s to the 900 s deadline, no response, 0 tokens |
| trial_005 | A | `evaluation_budget_exhausted` | 901.0 | same as trial_004 |

- A and B ran concurrently: both were spawned at 07:00:18–19 (`launches/trial_00{2,3}/supervisord.log`).
- B's compact I/O cut video plus run_state from 103.7 s (A) to 25.7 s. B nevertheless used 1019 steps against A's 830 and 13 calls against 11, so its service time was only 5.4% lower.

## 2. Q1 — time decomposition of the 11 uc7 raw successes and the gap to 300 s

These means cover t0, t1, t2_002, t3_003, t3_005, t4, t5_001, t5_002, t7, t8 and t9.

| component | mean (s) | share of service |
|---|---:|---:|
| model_pipeline | 223.7 (14.8 calls × 15.1 s) | 49.9% |
| simulator_step | 87.4 (0.105 s/step) | 19.5% |
| video_render | 75.9 (0.091 s/step) | 16.9% |
| preparation | 23.8 (17.2 s of it is the model draft) | 5.3% |
| run_state | 21.1 (0.025 s/step) | 4.7% |
| start-up + finalization (elapsed − control) | 8.3 | 1.8% |
| observe_transition | 5.8 | 1.3% |
| residual | 1.6 | 0.4% |
| policy_compute | 0.8 | 0.2% |
| **service (reconstructed)** | **448.6** (median 438.1, range 270.2–650.3) | |

- Only **1 of 11 runs is ≤ 300 s**: t5_001 at 270.2 s. The next best is t5_002 at 327.8 s.
- Per-step costs (simulator + video + run_state + observe) total 0.228 s/step × 836 steps = 190 s, which is 42.4% of service.

**Counterfactuals.** These are arithmetic on measured components, not predictions.
- **Model time = 0:** mean 224.9 s, so the non-model floor fits on average. It does not fit everywhere: t0_001 is still **318.7 s** with zero model time, and t3_003 (294.1 s) and t9_001 (298.4 s) sit within 6 s of the limit. For today's trial_002 the zero-model figure is 648.7 − 252.3 = 396.4 s, because the 138 s automated review counts as service.
- **No video and no run_state:** mean 351.5 s; 2 of 11 runs are ≤ 300 s.
- **Compact B-arm I/O** (video 0.0252 s/step as measured in trial_003, run_state ≈ 0): mean **372.6 s**; still only 2 of 11 runs (t5_001 236.8 s, t5_002 286.0 s). The model budget left is 300 − non-model = **151.1 s mean**, about 10 calls at 15.1 s, or about 7 calls at today's ~22 s per call. Actual calls are 7–22, mean 14.8.
- **Compact I/O plus steps cut to the realized-speed lower bound** (§4): mean **317.8 s**; 4 of 11 runs fit (t2_002, t5_001, t5_002, t8_001). Closing the rest needs about 1–2 fewer calls per run.

So no single lever reaches 300 s. The mean case needs all three together: compact I/O, motion at transit speed, and about 25–35% fewer model calls or cheaper ones.

**t5_001 reference** (the only S300 run):
- Breakdown: prep 28.4 s (model 21.8 s), model 146.8 s (7 calls; first call 38.55 s, mean 21.0 s), sim 41.1 s, video 36.5 s, run_state 6.9 s, observe 1.7 s, start-up/finalization 7.8 s, total 270.2 s.
- Process wall time was 274.9 s.
- It had the fewest calls (7), the fewest steps (399) and zero stalls, with one object and one grasp.
- Model time was still 54% of its service time, and its success is fragile (see §6).

**Per-call latency.** Over 205 non-first uc7 calls in the 13 runs (`call_tokens.json`), latency ≈ **10.2 s + 0.0373 s × reasoning_tokens**:
- Correlation of latency with reasoning tokens is 0.84, with output tokens 0.86, and with input tokens 0.04, even though input grows to 150k tokens.
- First calls average **27.4 s** with 438 reasoning tokens; non-first calls average 14.7 s with 120.
- Today the non-first mean was 21.7 s (A) and 19.8 s (B), about 45% higher than the historical 14.7 s, at similar reasoning (107 and 89 tokens). The server-side `created→completed` time was also higher: 20.0 and 18.2 s against 10.5–16.7 s historically. Concurrency between the two arms is a plausible cause, but that is not established.

## 3. Q2 — action mix and decision types

- **Control steps.** `agent_stats.json` counts across the 13 uc7 runs give fragment_actions **11,895**, direct_actions **41**, native_macro_actions **0**, optional_macro_grants **0**.
  - `execution_kind` is `framework_fragment:execute_fragment` for 99.7% of steps; the rest are `agent_owned:local_motion` (t3_003 10, t3_005 15, t9 6, t6_009 6, t6_011 2, t4 1, t8 1).
  - Native macros (hold/turn/place_on/release) were **never used in uc7**.
  - In uc6 t6_007 one macro grant spent 60 `calibrate_plus` steps with no progress ("The macro spent its budget opening for calibration without approaching the mug"); everything else in uc6 was `local_motion` (404 of 464 and 419 of 419 steps).
- **Decision types** (218 response JSONs, 13 uc7 runs): execute 210, motion 7, finish 1, observe 0, macro 0.
  - Segments per execute: mean 1.95 (1 segment 77 times, 2: 79, 3: 41, 4: 13; never more than 4 of the 6 allowed).
  - Segment kinds: `move_to_pose` 229 (156 transit, 73 contact), `move_relative` 74, `approach` 38, `set_gripper` 66, `wait` 3.
  - Every execute put `observation_after: true` on its last segment. By design, pending segments never autoplay (`agent_speedup_v2_user_check_fragments.py:335-337`), so **every fragment ends in a model call**.
- **Steps per call.** In the 11 successes, 9,195 steps over 163 calls is **56.4 steps per call**. uc6 bought 4.8 steps per call (t3_002) and 5.5 (t6_007).
  - Fragments, excluding the one cut short by success, executed mean 63.4 / median 58 steps (range 10–139), using **49%** of the requested step budget.
- **Why each of the 163 calls happened** (11 successes):
  - 104 scheduled checkpoints after `fragment_complete`;
  - **36 after `segment_stall`** (22%);
  - 11 final fragments cut by raw success;
  - 6 direct motion calls;
  - 1 each for `observation_required` and `segment_budget_exhausted`;
  - 1 mid-sequence `target_envelope_rejected`, plus **3 zero-step first calls rejected by the 1.2 rad rotation envelope** (t4_001, t8_001, t9_001).
  - t6_011 lost calls 1 and 9 the same way. These 5 wasted calls cost 28.7 + 25.8 + 26.2 + 28.0 + 24.4 = **133.1 s** of model time. The rejection rule is at `fragments.py:236-239`. The executor already caps rotation at 0.1 rad per step, and the model's own fix was to split the rotation into stages.

## 4. Q3 — step efficiency (11 uc7 successes, `steps2.json`, `episode.events.jsonl`)

- **Realized speed is about 24% of the commanded cap.**
  - The fragment controller clips Δ = target − pose to 20 mm per step (transit) or 5 mm (contact) (`fragments.py:25-32, 252-258`). Normalized action = Δ / 0.05 m (`action.py:16, 92`). The controller is OSC_POSE with kp = 150, damping ratio 1, 20 Hz (`simulator.json`).
  - Measured steady state at the cap over 8,650 motion steps: transit 20 mm commanded → **4.79 mm/step** (n = 2,740, ratio 0.24, about 0.10 m/s); contact 5 mm commanded → **1.17 mm/step** (n = 3,322, ratio 0.23).
  - A steady-state PD estimate (v ≈ kp·Δ/kd ≈ 6 mm/step) is consistent with this. That is inference, not a measurement.
  - 1,479 motion steps (17.1%) were in the proportional tail below the cap.
- **Contact versus transit.** Contact steps are **48.6% of motion steps** (4,203 of 8,650) but cover only **21.2% of the path** (4.73 of 22.28 m). Transit covers 17.55 m in 4,447 steps. The other 5.4% of all 9,195 steps went to the gripper (496).
- **Gripper and wait.** set_gripper used **496 steps** (5.4%) against a hard minimum of 364 (12 per close, 8 per open; `fragments.py:35, 294-301`), so gap-stability settling adds 132 steps. `wait` was used 3 times (16 steps in t3_005).
- **Stalls and no-progress.**
  - 36 stalled segments contain **1,411 steps** (15.3%). Each stall includes at least 12 steps of under 0.8 mm improvement by construction (`fragments.py:41-43, 311-325`).
  - Runs of 10 or more consecutive motion steps under 1 mm/step total **1,115 steps** (12.1%); the longest is 77 steps (t3_003). Per run this ranges from 0 (t5_001) to 322 steps (t3_003).
- **Lower bound on steps.**
  - Same realized path at the measured transit speed (5.1 mm/step) plus minimum gripper steps: **4,752 steps, 51.7% of actual** (per run 40–65%; t1_001 572 vs 897; t5_001 249 vs 399).
  - At the nominal 20 mm/step command cap: 1,483 steps, 16.1%. That bound is physically unreachable with current gains.
  - The OSC accepts up to 0.05 m per axis, so a larger transit command could raise speed. This is an untested inference that needs a sim-only check.
- **Elbow limit.** At 13 of 218 uc7 requests, measured joint 4 was within 0.02 rad of its −0.0698 rad upper limit: t6_009 ×8, t6_011 ×3, t3_003 ×2 (`measured_state.joint_position_rad` in `requests/*.request.json`). In 17 of 55 stall handoffs (keyword match on the next belief) the model itself blamed the elbow, joint limit or reach.
- **Fingertip-to-table stalls.** Descents repeatedly stalled at grip_site z ≈ 0.446 m: t1_001 call 2, t6_009 calls 9–10, t7_001 calls 7 and 9 ("the fingers appear at tabletop level").

## 5. Q4 — t6 failures ("put the white mug on the plate and put the chocolate pudding to the right of the plate")

**Goal.** From `answer_key.json` and the LIBERO BDDL: `On(porcelain_mug_1, plate_1)` and `On(chocolate_pudding_1, plate_right_region)`.
- The region is x ∈ [0.10, 0.20], y ∈ [0.05, 0.15]. The plate starts in x ∈ [0.125, 0.175], y ∈ [−0.025, 0.025], and the pudding in x ∈ [−0.075, −0.025], y ∈ [0.075, 0.125].
- "Right of the plate" therefore means **world +y, 5–15 cm**. The pudding's starting y (0.117) is already inside that band, so it only needs to move about 0.19 m in +x.
- LIBERO's `On(mug, plate)` also needs the mug origin within 30 mm of the plate centre in xy (`physical_truth.json` libero test string).

| run (ver) | stop | mug at terminal | pudding at terminal | where it went wrong |
|---|---|---|---|---|
| t6_001 (uc1) | plan_conflict, 1771 steps | held, 43.3 mm off the plate centre | untouched (−139 mm outside region) | grasp `pointed_not_grasp_point` ×2, then `lateral_unreachable`. The replan was forbidden by the uc1 approval rule |
| t6_002 (uc2) | user stop, 5621 steps | on the table, 281.5 mm off | held at its start position | mug grasp failed 6× (pointed_not_grasp_point ×3, align_no_convergence, footprint_blocked, geometry_unmeasured). Pudding `supported_on` failed 16× with `destination_not_registered`. 3,362 of 5,621 steps were `motionless_repeat_hold` |
| t6_004 (uc3) | user stop, 3187 steps | on the plate, strict pass, but **52.6 mm** off-centre so raw fails, tilted 12.8° | untouched | region E registration failed 16× (`reach_limit` ×10, `part_not_found` ×6). Pudding `precondition_unmet` ×8, `invalid_plan`, then the arm held still (1,363 `hold_recheck` steps) |
| t6_005 (uc4) | infrastructure_error, 4078 steps | no truth file | no truth file | mug placed, then `keepout_contact`, release, and 9 failed regrasps |
| t6_006 (uc5) | user stop, 1076 steps | held in the air, 197 mm off | untouched | `geometry_unmeasured`, then `local_grasp_identity` and local_motion loops |
| t6_007 (uc6) | **raw success** | on the plate (19.9 mm) | inside the region by 4.43 mm, still held | first pudding target (0.15, 0.20) was outside the region (y > 0.15). The model judged the gap "large" and shifted to y = 0.145 |
| **t6_009 (uc7)** | user stop, 1544 steps | **on the plate, strict pass (19.6 mm)** | released at (0.187, 0.170): **20.36 mm outside** (region y ≤ 0.15) | the auto-approved plan bound region E to "surface region to world +x side of plate" (`approved_plan.json`). First placement at (0.30, 0.02) was outside in both x and y. After `finish` (call 16, step 965) the episode continued, and 12 more calls over 579 steps and **399.6 s** moved it to (0.25, 0.17) and then (0.19, 0.17) |
| **t6_011 (uc7)** | user stop, 1197 steps | **tipped over (88.9° tilt)**, 95 mm off | untouched | handle grasp: the mug leaned after lift (call 5) and the handle slipped while lowering (call 8, gap about 1 mm). Recovery: rotation-envelope rejections (calls 1, 9), elbow at its straight limit (calls 10–15), body grasps slipped and rolled the mug off the plate (call 20), empty closes (calls 24, 27). 20 recovery calls over 874 steps and 535 s |

**Time profile of t6_009** (runner.log `wall_s − latency_s`):
- Mug done at step 376, 232.8 s into execution (≈265 s of service including prep).
- Pudding grasped and first placed by the finish at step 965 / 530.4 s. That is already about 563 s of service before the relocation began.

**Diagnosis.**
1. **"Right of the plate" is a general missing capability: grounding a relational region with no observable completion signal.** It is not perception or control error. Execution hit the model's own targets to within a few mm (t6_009 call 27 belief: "within about 2.5 mm"), but the intended targets were outside a hidden 10 × 10 cm box.
   - Both runs that reached the pudding (t6_007, t6_009) missed the region on the first placement.
   - The only feedback was the implicit "episode continued after finish" bit. `finish` does not stop the evaluator (request capability text).
   - The +x reading also drove the arm to its reach limit. The target (0.30, 0.02, 0.45) is about 0.87 m from the shoulder (0.742 m frontal stretch limit, from the shoulder position in `summary.json mount_decision`); this is my calculation.
   - Native-macro versions uc1–uc5 failed at the same point structurally: region E could not be registered (`part_not_found`, `reach_limit`, `destination_not_registered`).
2. **Mug: grasp robustness and recovery gaps, which are control problems.** The same handle-grasp plan succeeded in t6_009 and slipped in t6_011, so the method is marginal. Once the mug tipped, the primitives could not right it: no regrasp or reorientation skill, the 1.2 rad envelope veto, and elbow-limit stalls. In uc1–uc5 the mug consumed the budget in 5 of 5 runs through macro precondition vetoes.
3. Perception and geometry contributed secondary costs: table-contact height misestimates (pudding descent stalled at z = 0.446 for 2 calls; t6_011 empty closes "previous height estimate was too high").

## 6. Q5 — raw successes with v2_success = false

**Rule.** `v2_success` = raw AND strict truth. The only exception is a terminal failure that is *hold-only* (every failing check is in {not_held, robot_not_touching_part, at_rest, joint_at_rest, orientation_kept} and at least one of not_held/robot_not_touching_part); then a release-and-settle counterfactual decides (`agent_speedup_v2_truth.py:74-75, 884-923`). Episodes end at the **first** raw success (`libero.py:944`, `stop_on_success=True`).

**Findings across 15 user-check raw successes** (11 uc7, 2 uc6, today's 2):
- **Strict truth at the terminal step failed in 15 of 15.**
  - Every terminal state still had finger contact: `not_held` failed in 11, `robot_not_touching_part` in 4.
  - Raw success fired **2–27 steps into the final fragment**, before the planned release, retreat and inspection (t0 27, t1 6, t2 3, t3_003 5, t3_005 14, t4 7, t5_001 4, t5_002 6, t7 11, t8 2, t9 10, A 23).
  - Dropped objects were still falling: 0.78 m/s (t0), 0.62 (t1), 0.65 (t5_002), 0.73 (A), 1.07 (B).
- **v2 true in 3 uc7 runs** (t2_002, t8_001, t9_001): their failures were hold-only and the counterfactual settle passed.
- **v2 false in 12.**
  - Containers (t0, t1, t5_001, t5_002, t7, A, B) also failed `supported_by_container_or_contents` and/or `not_perched_on_rim`, which are not deferrable.
  - Plates (t4, t6_007) also failed `supported_by_target`.
  - Drawers (t3_002, t3_003, t3_005): the bowl failed `at_rest` while the drawer was still moving. That atom has no hold check, so it is not deferrable.
- **After release and settle, strict passes in 10 of those 12.**
  - Real placement defects remain in **t0_001** (tomato `not_perched_on_rim`, `supported_by_container_or_contents`) and **t5_002** (book `lowest_point_below_rim`, `not_perched_on_rim`).
  - **t5_001, the only S300 run, would lose raw success after release.** Raw fired with the held book origin 1.16 mm below the region box top; after settling, the origin is 3.48 mm *below* the box floor (`check_success=false`, raw true for only 15.8% of the settle).
- Conclusion: v2 = false mostly measures the evaluator stopping mid-release, not a policy failure. Raw is the user's criterion. Even so, 2 of 11 raw successes are physically poor, and the single S300 result depends on a mid-descent transient.

## 7. Ranked bottlenecks (with numbers)

1. **Model latency × call count.**
   - 223.7 s, 49.9% of service; 14.8 calls × 15.1 s.
   - Latency floor about 10.2 s per call, plus 0.037 s per reasoning token; first call 27.4 s.
   - With compact I/O, only about 151 s (≈10 calls) fits under 300 s.
   - Calls are forced by a checkpoint after every fragment (mean 1.95 segments, 49% of budget used), 36 stalls, and 5 envelope-veto calls (133 s).
2. **Per-step cost × steps.**
   - 0.228 s/step × 836 steps = 190 s (42%).
   - Video and run_state account for 97 s; compact I/O measured −78 s.
   - The simulator step (0.105 s) includes encoding two 512² RGB data-URLs on **every** control step (`observation.py:158-159`, `libero.py:951-954`). The fragment controller uses only proprioception between calls; how the 0.105 s splits between physics, rendering and encoding is unknown.
3. **Slow realized motion.**
   - Arm speed is 24% of the command: 4.8 mm/step in transit, 1.17 in contact.
   - The contact profile takes 48.6% of motion steps for 21% of the path.
   - The same path at transit speed needs 51.7% of the steps; stalls take 15%.
4. **Preparation hangs, stochastic on identical inputs.**
   - The byte-identical t6 prep payload `fe11f64d…` was sent 4 times: 2 hung (t6_008 634 s, t6_010 297 s, both user-interrupted with 0 tokens) and 2 completed (24.3 s, 21.2 s).
   - The t3 payload `655e8267…` was sent 3 times: 1 hung (t3_004, 298 s).
   - Today 2 of 4 hung to the 900 s cutoff.
   - In total **5 of 20 uc7-code prep attempts hung (25%)**, each a certain S300 failure. Prep payloads carry no `max_output_tokens` (`output_token_limit_sent: null`; payload keys omit it).
5. **t6 generality gaps.** The hidden relational region with no completion feedback cost 12 calls and 399.6 s after `finish` in t6_009, ending 20.4 mm outside. Grasp and upright recovery failed in t6_011 (20 calls, 535 s), and elbow and reach limits recur on t3 and t6.
6. **First raw success fires mid-placement.** Strict at terminal 0 of 15; t5_001's S300 depends on it.
7. **Measurement variance.** Today's per-call latency was +45%, and the protocol counted 138 s of automated review. The historical 270 s result is not a stable baseline.

## 8. Unknowns

- How simulator_step_s (0.085–0.124 s/step) splits between physics, two-camera rendering, PNG/data-URL encoding, pipe transfer and the per-substep contact monitor. No per-component samples exist.
- Whether larger transit commands (up to 0.05 m per axis) keep the 0.24 ratio, or overshoot and collide. This needs a sim-only test.
- Why identical prep requests hang (runaway stream versus server stall). The speedup_next replay saw output_text deltas (141,110 and 27,853 chars) without `response.completed`. The root cause is unresolved.
- Whether today's +45% latency comes from the A/B concurrency or from service load.
- Historical service times are reconstructed (about 5 s under process wall time), not clocked.
- t6_005 has no physical truth file; its terminal state is unknown.
- Stall-cause labels come from the model's own beliefs, matched by keyword; they are not verified contact truth.
