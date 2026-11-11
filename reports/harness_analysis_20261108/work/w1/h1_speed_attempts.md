# H1 (uc7.1 obstacle harness, 100 mm transit): where the time and the model calls go

Key: `h1_speed_attempts`. Focus: FAST and FEW-ATTEMPTS. Read-only analysis (2026-11-07). No simulator, model or GPU job was started; nothing under `/workspace/gpt_field` was modified. All numbers below were computed from saved run artifacts with the scripts in `scripts/` next to this file (paths at the end). "Verified" means I read the code line or computed the number myself; "estimate" marks a derived projection.

Code citations are to `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/` (git `3cc84ac`), abbreviated `H1/`. Runs are under `/workspace/gpt_field/runs/`:

| Short name | Run directory | Result | Steps | Exec calls |
|---|---|---|---|---|
| T04 t100 | `agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001` | raw success | 549 | 13 (+1 plan) |
| T08 t100 | `agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001` | raw success | 843 | 21 (+1) |
| T08 stove-off | `agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001` | raw success | 889 | 16 (+1) |
| drawer 001 | `agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001` | raw success | 2364 | 42 (+1) |
| drawer 002 | `..._20261103_002` | initial_plan_failed (401) | 0 | 0 |
| drawer 003 | `..._20261103_003` | raw success | 2131 | 36 (+1) |

All five executed runs use identical profiles (transit: 0.10 m/action cap, 0.50 m / 1.2 rad envelope, 4 mm / 0.06 rad tolerance; contact: 5 mm/action, 0.08 m / 0.5 rad, 2 mm / 0.04 rad) per each `result_summary.json:command_configuration_audit.profiles`, so they pool as one build. Contrast runs: `agent_speedup_v2_uc71_can_obstacle_t04_20261103_001` (20 mm cap) and `agent_speedup_v2_uc71_default_routing_t04_20261103_001`.

## 0. How H1 actually runs (as implemented)

One process. `scripts/evaluate_agent_speedup_v2_user_check.py start --auto-approve` first **prepares**: it starts a simulator worker, resets, sends one stateless *plan* request (compact-output instruction + 180 s SSE deadline + whitespace guard), then **closes that simulator** (`H1/scripts/evaluate_agent_speedup_v2_user_check.py:274-330`). Approval is automatic. `base.execute` then starts a **second** simulator and resets again (`H1/scripts/evaluate_agent_speedup_v2.py:1384-1394`).

The control loop is strictly serial and single-threaded (`evaluate_agent_speedup_v2.py:1437-1566`): `run_state.json` write → `policy.act()` → trace append → video render → `run_state.json` write → `sim.step()` (IPC to the worker, which renders and PNG-encodes both cameras) → `policy.observe_transition()` → video render → `run_state.json` write with full `agent_stats()`. `act()` makes **at most one** executive call (`H1/src/gpt_field/agent_speedup_v2_user_check_agent.py:512-541`). The call blocks; "Simulation pauses during inference" (`..._agent_client.py:182`). The executive answers with 1-6 measured segments (`move_to_pose`, `move_relative`, `approach`, `set_gripper`, `wait`). The fragment executor clips each step's Cartesian command to 0.10 m (transit) or 0.005 m (contact) and converts it to a normalized OSC delta (`..._fragments.py:333-346`, `action.py:20-23,99`). Transit stages follow a local grid route through the model's boxes with a 0.10 m lookahead (`..._routing.py:105-108,154-156`). A segment ends on 2 stable samples inside tolerance, on gripper-gap stability, or on a failure: envelope rejection, `route_unavailable`, `segment_stall` (no 0.8 mm improvement over 12 steps after 16 steps), gap rise, or budget (`..._fragments.py:285-315,368-419`). **Any end of a fragment returns control to the model**, whether it completed or failed (`..._fragments.py:422-434`, `..._agent.py:382-401`). Pending segments are discarded on failure. Each new request resends the full text transcript (all prior observations, accepted outputs and encrypted reasoning) plus the 6 newest image pairs (`..._agent_client.py:44,417-425`). The episode ends only when the evaluator sees raw LIBERO success (`evaluate_agent_speedup_v2.py:1440-1443,1553-1566`). `finish` never stops it, and there is no give-up action or call/time cap (`..._agent_client.py:205-209`).

## 1. Wall-clock budget

### 1.1 Process wall decomposition (seconds; from `result_summary.json:timing` + `trial/summary.json:timing_components`)

| Run | Wall | Prep sim | Plan call | Exec sim start | Exec model | Sim step | Video (inline) | run_state | observe+policy | Other | Finalize+outer | Simulated motion (steps×0.05) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T04 t100 | **402.1** | 10.7 | 17.5 | 8.9 | **200.3** | 68.4 | **68.0** | 13.0 | 4.6 | 0.9 | 9.6 | 27.5 |
| T08 t100 | 634.1 | 6.4 | 22.3 | 4.9 | 336.4 | 94.1 | 122.5 | 24.8 | 10.5 | 1.6 | 10.6 | 42.2 |
| T08 stove-off | 628.1 | 6.1 | 37.9 | 5.4 | 261.8 | 112.6 | 146.7 | 32.7 | 11.9 | 1.8 | 11.2 | 44.5 |
| drawer 001 | 1804.2 | 7.0 | 52.2 | 6.1 | 781.6 | 288.4 | 479.2 | 108.9 | 60.5 | 9.9 | 10.5 | 118.2 |
| drawer 003 | 1676.2 | 6.8 | 50.2 | 5.9 | 830.1 | 246.4 | 395.8 | 85.8 | 42.0 | 4.3 | 9.0 | 106.6 |

Shares of process wall:

| Run | Model (plan+exec) | Video | Sim step | run_state/observe/other | Startup (2 sims) | Robot motion (sim time) |
|---|---|---|---|---|---|---|
| T04 t100 | 54% | 17% | 17% | 5% | 5% | 6.8% |
| T08 t100 | 57% | 19% | 15% | 6% | 2% | 6.7% |
| T08 stove-off | 48% | 23% | 18% | 7% | 2% | 7.1% |
| drawer 001 | 46% | 27% | 16% | 10% | 1% | 6.6% |
| drawer 003 | 52% | 24% | 15% | 8% | 1% | 6.4% |

- The README's "287.3 s control without video + 17.5 s plan = 304.8 s" omits the 68.0 s of inline planning-video rendering. That rendering sits inside the control loop (`evaluate_agent_speedup_v2.py:1515-1517,1549-1552`), so the user actually waited **402.1 s** (`total_wrapper_wall_s`).
- The robot's simulated motion is under 7.1% of every run's wall. The rest is model latency (46-57%) and harness work (37-50%).
- **Time to first robot motion in T04 ≈ 77 s**: prep 28.2 + exec-sim startup 8.9 + call 1 32.4 + envelope re-ask 7.7.

### 1.2 Per-step harness cost (non-model steps)

| Run | sim_step | video | run_state | observe | total/step (incl. video) | without video | wall gap per step by episode quintile (episode.events.jsonl) |
|---|---|---|---|---|---|---|---|
| T04 t100 | 0.1246 | 0.1238 | 0.0237 | 0.0073 | 0.282 | 0.158 | 0.263, 0.268, 0.279, 0.289, 0.289 |
| T08 t100 | 0.1116 | 0.1453 | 0.0295 | 0.0113 | 0.301 | 0.155 | 0.257 → 0.329 |
| T08 stove-off | 0.1266 | 0.1651 | 0.0368 | 0.0120 | 0.344 | 0.179 | 0.293 → 0.388 |
| drawer 001 | 0.1220 | 0.2027 | 0.0461 | 0.0239 | 0.401 | 0.198 | 0.342 → 0.460 |
| drawer 003 | 0.1156 | 0.1858 | 0.0402 | 0.0183 | 0.363 | 0.178 | 0.306 → 0.395 |

The real-time step is 0.05 s, so the harness runs **5.6-8× slower than real time**. The cost grows within an episode (quintile column) because video, `run_state` and observe work are O(history). The prior report's 0.228 s/step (uc7) is now 0.28-0.40 s/step. The planning video is the main addition.

## 2. The model-call channel

### 2.1 Per-call statistics (stream timing from `requests/*.stream.json`; tokens from cumulative `result.json:budget` differences)

| Run | Calls | Σ latency s | mean / median / p90 / max | First call | Input tokens first → last | Σ input | Cached share | Reasoning mean | Visible mean | Images/request (mean) | Request MB (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T04 t100 | 13 | 198.5 | 15.3 / 13.4 / 21.5 / 32.3 | 32.3 | 7,945 → 87,425 | 641,669 | 2.5% | 123 | 264 | 9.7 | 3.2 |
| T08 t100 | 21 | 333.6 | 15.9 / 12.2 / 27.9 / 38.1 | 30.8 | 7,972 → 132,627 | 1,527,818 | 2.1% | 142 | 214 | 10.6 | 3.0 |
| T08 stove-off | 16 | 259.3 | 16.2 / 14.1 / 26.2 / 28.8 | 28.8 | 8,094 → 102,686 | 901,229 | 1.0% | 111 | 273 | 10.1 | 2.8 |
| drawer 001 | 42 | 774.1 | 18.4 / 15.9 / 28.4 / 44.5 | 24.4 | 8,457 → 260,558 | 5,716,129 | 1.2% | 151 | 259 | 11.3 | 3.5 |
| drawer 003 | 36 | 823.8 | 22.9 / 19.6 / 42.0 / 64.0 | 15.1 | 8,476 → 227,014 | 4,275,112 | 1.9% | 219 | 259 | 11.2 | 3.3 |

- Least squares over all 128 execution calls: **L = 6.63 + 0.0179·input_k + 0.0518·reasoning + 0.0074·visible (R² 0.949)**. Of the 2,389 s of model time, the fixed per-call term is 35.5% (848 s), reasoning 44.6%, input/prefill 9.8%, visible tokens 10.1%.
- From the stream timestamps: time to headers is 13.1%, the pre-visible phase (reasoning + prefill) 59.7%, and visible streaming 27.2% at a median 49 visible tok/s.
- Visible output averages 910 chars per response, 44% of it belief/plan/why prose (schema maxLength 1800/1800/600, `..._agent_client.py:250-252`). `max_output_tokens` is never sent (`H1/src/gpt_field/chatgpt_plan.py:37-41`). There were no retries in any run, and no output degeneration in execution calls: longest JSON whitespace run 0.
- Each request carries a different server-assigned `prompt_cache_key` with `prompt_cache_retention: 24h` (`response.json`). The client sends none. Cache hits are at most 3,072 tokens per call (the instructions prefix), apart from a 16,000-token hit (T04 call 3) and a 7,808-token hit (T08 call 5), both before the image window starts rewriting history.

### 2.2 T04 call by call

| # | step | what ended the previous decision | latency s | input tok | reasoning | visible | actions before next call | response (`*` = observation_after) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | initial | 32.3 | 7,945 | 495 | 339 | 1 (hold) | move/t* + map |
| 2 | 1 | **veto: target_envelope_rejected** (1.572 rad > 1.2) | 7.6 | 16,124 | 0 | 226 | 69 | move/t + move/t* |
| 3 | 70 | checkpoint (fragment end) | 22.0 | 22,521 | 230 | 241 | 54 | move/t + approach/c* |
| 4 | 124 | checkpoint | 9.6 | 29,429 | 0 | 194 | 38 | close + approach/c* (test lift) |
| 5 | 162 | checkpoint (after test lift) | 19.5 | 35,953 | 226 | 367 | 1 (hold) | move/t + move/t* + map |
| 6 | 163 | **veto: route_start_blocked** | 10.6 | 44,842 | 16 | 347 | 33 | move/t + move/t* + map |
| 7 | 196 | **stall** (4.8 mm, overshoot return) | 17.6 | 51,044 | 152 | 240 | 57 | move/t + approach/c* |
| 8 | 253 | checkpoint | 10.6 | 57,137 | 29 | 249 | 94 | approach + open + approach* |
| 9 | 347 | checkpoint (after release) | 15.5 | 63,137 | 101 | 305 | 22 | move/t* + map |
| 10 | 369 | checkpoint | 16.2 | 69,304 | 198 | 221 | 64 | move/t + approach/c* |
| 11 | 433 | **stall** (contact approach, 8.2 mm) | 13.4 | 75,454 | 52 | 225 | 39 | close + approach/c* (test lift) |
| 12 | 472 | checkpoint (after test lift) | 12.6 | 81,354 | 50 | 244 | 35 | move/t + move/t* |
| 13 | 507 | **stall** (4.9 mm, overshoot return) | 11.1 | 87,425 | 55 | 228 | 42 → raw success at 549, mug still held |

Two mugs took 13 execution calls, 6.5 per object: approach above the handle → closer view → descend → close + test lift → carry → lower → release + retreat. On average a call bought 42 actions, 2.1 s of simulated motion, for 15.3 s of blocked inference.

### 2.3 Why every execution call was made: all 128 calls in the five runs

Script `scripts/triggers.py` reads `context.last_feedback` in each request. `scripts/stall_class.py` classifies the stalls from the per-step error trace.

| Trigger | T04 | T08 | stove-off | drawer 001 | drawer 003 | Total | Avoidable by mechanism |
|---|---|---|---|---|---|---|---|
| initial | 1 | 1 | 1 | 1 | 1 | 5 | – |
| model checkpoint at fragment end | 7 | 13 | 11 | 20 | 18 | **69 (54%)** | protocol (F2) |
| stall: transit overshoot (passed within 10 mm, error then rose ≥3 mm) | 2 | 0 | 1 | 6 | 8 | **17** | controller + stall predicate (F3) |
| stall: contact approach stopped on contact | 1 | 1 | 2 | 8 | 7 | **19** | stall semantics (F4) |
| stall: transit far / genuine | 0 | 1 | 0 | 3 | 1 | 5 | kinematics (F10) |
| veto: target envelope (rotation) | 1 | 1 | 0 | 0 | 0 | 2 | F5 |
| veto: route_start_blocked | 1 | 2 | 1 | 1 | 0 | 5 | F5 |
| direct 5-step motion finished | 0 | 1 | 0 | 3 | 1 | 5 | F10 |
| segment budget exhausted | 0 | 1 | 0 | 0 | 0 | 1 | – |

- 12 stalls discarded pending planned segments: 5 overshoot, 5 contact, 2 transit near/far. Each directly cost one extra call.
- All **123 of 123** `execute` responses ended with `observation_after=true` on the last segment, and none put a checkpoint mid-sequence. They used a mean of 2.07 segments (cap 6) and 135 requested steps (cap 240).
- Failure-triggered calls (stall, veto, direct motion, budget) were 38% (T04), 33%, 25%, 50% and 47% of calls, and 60, 118, 66, 385 and 427 s of model time.
- Consecutive failure streaks: drawer 001 calls 17-25 (9 calls); drawer 003 calls 24-27 and 29-34; T08 calls 2-5.

## 3. Action accounting: why two mugs need 549 actions

T04 phase timeline (segment boundaries from `trace.jsonl` feedback rows):

| Phase | Mug 1 actions | Mug 2 actions | Comment |
|---|---|---|---|
| Rotate 90° + move above handle (transit) | 69 (51 with zero translation command) | 31 | 90° split into 2×45° by the envelope veto; rotation realizes ~0.023 rad/step |
| Descend to handle (contact approach) | 54 | 55 (stalled on contact) | 5.5-7.2 cm at 1.2 mm/step |
| Close | 12 | 13 | min 12 steps + 5-sample gap stability |
| Test lift 2.9 cm (contact approach) | 26 | 26 | mandated by the prompt before any carry |
| Carry (lift + transit) | 43 (overshoot stall) | 42 (overshoot stall) | 22-25 mm/step peaks |
| Lower to plate (contact approach) | 65 | 35 → raw success while still holding | |
| Open | 19 | – | |
| Retreat 6.9 cm (contact approach) | 57 | – | free-space move run at contact speed |
| Holds after rejected answers | 2 | | |

By profile (all runs):

| Run | Contact `approach` share of actions / of path | Contact speed (mm/step) | Transit share of actions / path | Transit mean / p90 / max mm per step (cap 100) | Transit steps with zero translation (rotation wait) | set_gripper |
|---|---|---|---|---|---|---|
| T04 | 56.6% / 20.7% | 1.17 | 35.0% / 79.2% | 7.3 / 22.4 / 26.3 | 54 (9.8% of all actions) | 8.0% |
| T08 | 58.1% / 21.1% | 1.24 | 30.0% / 77.8% | 8.8 / 25.3 / 26.9 | 16 | 11.0% |
| stove-off | 56.0% / 17.8% | 1.28 | 25.6% / 81.8% | 12.9 / 25.8 / 27.2 | 10 | 12.0% |
| drawer 001 | 51.5% / 17.8% | 1.09 | 36.6% / 79.8% | 6.8 / 21.4 / 26.6 | 211 | 6.2% |
| drawer 003 | 66.8% / 26.4% | 1.12 | 26.3% / 72.9% | 7.9 / 22.6 / 26.9 | 86 | 6.7% |

Realized displacement is about 22-30% of the per-step command in both profiles. That is OSC dynamics: kp 150, critical damping, re-referenced delta goal. The 100 mm cap never produced more than 27.2 mm in one action. Compared with the 20 mm run (1,276 actions, p90 transit 5.25 mm/step), H1 needs 57% fewer actions on T04. Each setting has only one stochastic episode, so this is indicative, not causal.

## 4. Findings

### F1 (critical, FAST). Per-step harness work is 3-6× the robot's real motion time and grows with the episode
**Mechanism.** Every one of the 549-2,364 actions pays for:
- **Inline planning video.** Two full 1600×960 renders per step. Each render deep-copies the whole diagnostics dict (`H1/src/gpt_field/agent_speedup_v1_video.py:182-213,221-226`) and redraws the full measured path in 3 panels (`..._user_check_video.py:178,216-218`): 0.124-0.203 s/step, rising with episode length.
- **Camera rendering and PNG coding every step.** The worker PNG-encodes both 512² cameras at PIL's default level 6 (`H1/src/gpt_field/observation.py:44`) and decodes them for validation (`observation.py:114-115`). The host decodes them twice more (`libero.py:662`, then `CalibratedObservation`). A second `agent_geometry` IPC runs per step (`..._user_check_sim.py:118-125`). Yet the policy uses images only at decision steps (≈1 in 42), because fragments are proprioceptive.
- **State-file writes with growing history.** `run_state.json` is rewritten with full `agent_stats()` every step (`evaluate_agent_speedup_v2.py:1036-1053,1555`); it reaches 0.92 MB in drawer 001. `_sync_framework()` deep-copies the growing feedback history every step (`..._agent.py:234-260,605`).

**Evidence.**
- Non-model control was 154 s of 355 s in T04 (43%) and 943 s of 1,728 s in drawer 001 (55%); see the §1.2 table.
- Offline codec benchmark on a saved T04 observation with the simulator's own Pillow 12.3.0 / zlib-ng 2.3.3 (`.venv-sim`): encode 36-40 ms per image at the default level vs 9.6-10 ms at level 1, decode 5-7 ms. Two encodes plus six decodes ≈ 0.10-0.12 s, about the measured `simulator_step_s` of 0.112-0.127 s/step. The host was loaded (load average 50), so this is likely rather than measured in-run. The code itself notes "level 1 is ~5x faster" (`mark_prompting.py:85`).

**Improvement (general; no policy change).**
1. Render the video **after** the episode from the trace, or with a compact recorder at ≤0.03 s/step. All inputs are already logged.
2. Render and encode cameras **only when the policy asks for an observation** (decision steps), and send state-only replies otherwise. Validate PNGs once. Return calibration in the step reply. Hash raw pixels instead of PNG bytes for the reset identity.
3. Write `run_state` and `agent_stats`, and run `_sync_framework`, only at decision boundaries or at ≤1 Hz. Log per-step deltas instead of full diagnostics.

**Expected effect (estimate).**
- Per-step cost falls from 0.28-0.40 to ~0.04-0.07 s.
- T04: −110 to −125 s (402 → ~280 s). T08: −190 to −210 s. Drawer: −700 to −820 s.
- Overhead stops growing with episode length.
- `control_without_video_s` should be replaced by true wall-clock as the primary metric.

**Validation.** `validation/replay_baseline.py replay` re-executes the 13 saved T04 responses with zero model calls and already checks all 549 actions, state8 values and contact dictionaries bit-exactly. Run it before and after the change, require exact equality, and compare per-step wall time. Then run an all-task sweep.

### F2 (critical, FAST + FEW-ATTEMPTS). Serial checkpoint-per-phase protocol: the model decides every ~2 s of robot motion while the robot is frozen
**Mechanism.**
- `act()` blocks on the SSE stream and the simulation pauses (`..._agent.py:512-541`, `..._agent_client.py:182`).
- Every fragment end returns control, completed or not (`..._fragments.py:422-434`).
- The prompt mandates a blocking visual check after every grasp ("explicit set_gripper=close followed by an appropriately small monitored test lift/checkpoint, then inspect new images", `..._agent_client.py:161-163`) and after every placement (`:163-164`). It also asks for "short informative moves" under uncertainty (`:81-83`).

**Evidence.**
- 69 of 128 calls (54%) were end-of-fragment checkpoints, and 123 of 123 execute responses ended on one.
- Responses averaged 2.07 segments (cap 6) and 135 requested steps (cap 240). They actually executed 42-59 actions per call (2.1-3.0 s of simulated motion) against 15-23 s of inference.
- That is 6.5 calls per object (T04), 10.5 (T08) and 8 (stove-off), and 7-8 per stage (drawer).
- Model time was 46-57% of wall.
- Upper bound for overlapping inference with motion, Σ min(L_k, E_{k-1}): T04 117 s at current per-step cost, 80 s without video, but only **25 s** once the harness runs at real time (`data/overlap.json`). After F1, the call **count** is the lever, not overlap alone.

**Improvement (general).**
1. Let the executive commit a whole pick-and-place (approach → descend → close → lift → carry → lower → open → retreat) as one sequence. The model chooses explicit measured **continuation predicates** for that sequence, for example an expected finger-gap band after close and no gap rise during the lift. The harness yields only when a predicate fails.
2. Make the visual verification **non-blocking**. At a checkpoint, send the request with fresh images while the robot continues the committed next phase. Apply the answer when it arrives, and let it abort or replace the remainder.
3. Rewrite the grasp and placement checkpoint mandates as "verify asynchronously; yield on measured anomaly". `observation_after` becomes a blocking request reserved for genuinely uncertain contact.

**Expected effect (hypothesis until a sweep).**
- T04: 7 checkpoint calls → ~3 (two pre-grasp view refinements and one place-and-next-object), i.e. −4 calls ≈ −60 s. The two test-lift verifications move off the critical path for another ~−30 s.
- T08: −8 calls ≈ −125 s. Drawer: −10 calls ≈ −190 s.
- Risk: undetected grasp failures. Measure the failed-grasp rate and contact samples.

**Validation.** All-task sweep on one frozen build: calls per object, wall time, success, number of late-detected grasp failures and forbidden contact samples, against the current H1.

### F3 (high, FAST + FEW-ATTEMPTS). The 0.10 m cap overshoots, and the error-window stall test turns the return trip into a "failed" segment
**Mechanism.** The transit tracker pulls a carrot 0.10 m ahead (lookahead = max(0.03, cap), `..._routing.py:154-156`) and clips each command to 0.10 m (`..._fragments.py:339-346`). The arm reaches 22-25 mm/step, cannot brake in time, passes 1-6 mm from the target without 2 consecutive in-tolerance samples (`:38,368-369,384-386`), and overshoots by 10-28 mm. The stall predicate (`:404-419`; constants `:43-45`) compares the window's first error, which sits at the overshoot minimum, with later errors. It therefore reports "no progress" while the arm is still moving back at 1-4 mm/step.

**Evidence.**
- T04 steps 173-195: error 175 → 4.2 → 10.7 → 4.8 mm, stall declared while closing at 1.5 mm/step, 1-2 steps before arrival. Steps 481-506 are identical (4.6 → 10.8 → 4.9 mm).
- drawer 003 steps 2065-2087: 1.1 → 27.9 → 12.0 mm. drawer 001 steps 24-46: 1.8 → 13.3 → 11.0 mm.
- 17 of 41 stalls are this kind (T04 2, stove-off 1, drawer 001 6, drawer 003 8), and 5 of them discarded pending segments.
- The 20 mm-cap T04 run (1,276 steps) and the default-routing run (801 steps) have **zero** transit overshoot stalls (`scripts/` comparison). H1 introduced this failure.
- Downstream misreading: drawer 003 call 27's belief is "simultaneous rotation and translation are producing poor tracking here".

**Improvement.**
1. Make the reference braking-aware: limit the commanded offset by a deceleration profile, e.g. offset ≤ k·sqrt(remaining distance), or use a time-scaled min-jerk reference. This keeps the 0.10 m cruise.
2. Define a stall as **measured speed ≈ 0 while out of tolerance**, not "error not improved since window start". The old predicate is replaced, not supplemented, so no threshold is added.

**Expected effect.**
- −5 directly avoidable calls (≈ −92 s: drawer 001 calls 6, 25, 40; drawer 003 calls 5, 33) across these runs.
- −10 to −20 return steps per overshooting transit.
- Correct "succeeded" feedback in 12 more cases, which removes the seed of repair streaks such as drawer 001 calls 22-25.

**Validation.** Scripted transit benchmark (random start/goal pairs, with and without a held object): overshoot, steps to arrival, false-stall rate. Then a sweep counting stall-triggered calls.

### F4 (medium, FEW-ATTEMPTS). A contact approach that stops on contact is reported as a failure, and the planned close/lift is thrown away
**Mechanism.** `approach` must use the contact profile (`..._fragments.py:159-160`). When it is blocked by the object or support, which is often the intended outcome, it ends with `failed: segment_stall` after ≥16 actions plus a 12-step no-progress window. All pending segments are dropped (`:418-419,436-447`).

**Evidence.**
- 19 contact stalls; 5 discarded pending segments (T08 call 13, stove-off 12, drawer 001 36, drawer 003 27 and 34).
- drawer 003 call 34's belief: "reached within about 3 mm ... The close and test-lift stages were not executed". The next call only re-issued "close at the current measured pose".
- Contact stalls end with 0.0-0.2 mm/step of motion, i.e. 12+ steps spent pushing against the contact.

**Improvement.** Treat a measured stop in a contact-profile approach as segment success ("contact_reached", reported with its residual) and continue with the pending segments. Shorten the no-motion window, since speed is the signal. This removes a veto rather than adding one.

**Expected effect.** −5 calls (≈ −114 s of model time) across these runs, ~−12 pushing steps per contact, and probably fewer hand-vs-object contact samples. Collision analysts should check: T04's 2,065 forbidden samples occurred during contact `approach`.

**Validation.** Sweep: pending-discard count (target 0), calls per grasp, forbidden samples per grasp.

### F5 (high, FEW-ATTEMPTS + ZERO-SHOT). Up-front vetoes reject whole answers; the first plan is invalid in 2 of 5 runs
**Mechanism.**
- The 1.2 rad per-stage rotation envelope (`..._fragments.py:27-34,288-291`; prompt `..._agent_client.py:135`) rejects a stage before any action, although each step is already capped at 0.1 rad (`..._fragments.py:234`).
- `route_start_blocked` refuses any transit whose current grip site lies inside a model box after clearance expansion (`..._routing.py:105-106`). That is routine right after a grasp next to an obstacle, or when the model enlarges a margin.
- Either way `act()` allows one call, falls back to a 1-step hold, and asks again (`..._agent.py:516-541`).

**Evidence.**
- T04 and T08 first plans were rejected at 1.572 and 1.569 rad (translation 0.134 and 0.233 m, well inside 0.50 m). This is the same failure the prior report found in uc7 (5 calls); it is unchanged in uc7.1.
- All route failures were `route_start_blocked`: T04 1, T08 2, stove-off 1, drawer 001 1.
- 7 veto-triggered calls cost 72.9 s of model time.

**Improvement.**
1. Drop the rotation envelope, or auto-split large rotations inside the executor.
2. When the start is inside an expanded box, plan an exit path first (shortest or vertical escape) instead of refusing.
3. Validate what remains of the envelope at parse time and report it in the same call, with no hold.

**Expected effect.** T04 −2 calls (−18.2 s), T08 −3 (−29.4 s), stove-off −1 (−10.1 s), drawer 001 −1 (−15.2 s). First answers become valid in all five runs.

**Validation.** Contract tests (`tests/test_validation_contracts.py`) plus a sweep counting `target_envelope_rejected` and `route_unavailable` (target 0).

### F6 (medium, FAST). Motion primitives are slow where the task does not need it: contact speed for all approaches, slow wrist rotation
**Mechanism.** Every `approach`, including free-space lifts and retreats, runs at the 5 mm cap. That realizes 1.1-1.3 mm/step (2.4 cm/s). Rotation commands 0.1 rad/step (`..._fragments.py:234`, `action.py:24`) but realizes ~0.023 rad/step, so a 45° turn takes ~33 steps after translation finished at step ~10 (T04 steps 1-33 and 34-69).

**Evidence.**
- Contact approach is 51-67% of all actions but only 18-26% of the path (§3).
- Rotation-wait steps: T04 54 (9.8%), drawer 001 211.

**Improvement.**
1. Distance-scheduled approach speed: fast until the last ~2 cm before the expected contact, slow only there.
2. Allow the transit profile for vertical lifts and retreats (route-checked).
3. Raise the rotation command cap together with F3's braking.

**Expected effect (estimate, `scripts/` model with 7.5 mm/step fast part and 2 cm slow tail).**
- T04 −143 actions, T08 −217, stove-off −253, drawer −695 to −732 (upper bound: drawer pulls are in contact throughout).
- Rotation: ~−35 actions in T04.
- Worth ~40 s on T04 at today's per-step cost, but only ~9 s after F1. Faster contact may raise contact forces, so validate with forbidden-sample counts.

**Validation.** Scripted approach benchmark (steps, contact samples), then a sweep.

### F7 (medium, FAST). Context grows linearly, is not cached, and resends ~9k chars of boilerplate per observation
**Mechanism.**
- Full text history plus encrypted reasoning plus the 6 newest image pairs are resent on every call (`..._agent_client.py:44,417`).
- Every observation message carries the full `capabilities` (4.6k chars), `macro` status with native-calibration notes (2.3k chars; native macros were used in **0 of 128** decisions), `geometry` (1.9k chars, external camera static), and the same feedback three times (`last_feedback`, `execution.last_result`, `recent_feedback`) (`..._agent.py:211-232`, `..._agent_client.py:536-542`).
- From call 7 on, the sliding image window rewrites message n−6, so the history is not append-only: the common prefix drops to 0 items at call 7 (`scripts/prefix.py`).
- No `prompt_cache_key` is sent, and the server assigns a new one per request.

**Evidence.**
- Input grows by about 6.1-6.6k tokens per call: T04 7.9k → 87.4k, drawer 001 → 260.6k. That totals 0.64-5.7M input tokens per run, 1-2.5% cached.
- At 0.0179 s/ktok: +1.6 s per call at 87k tokens and +4.7 s at 260k.
- Visible output streams for ~5 s per call, and prose is 44% of it.

**Improvement.**
1. Move static capability, macro and geometry text into the instructions or the first message, and drop the unused macro context.
2. Deduplicate feedback.
3. Project older observations to compact (belief, plan, outcome) records.
4. Strip images by a fixed per-message rule at insertion time so the history stays append-only.
5. Send a stable per-episode `prompt_cache_key` (acceptance untested on the subscription route).
6. Cap the prose fields (e.g. belief ≤ 300 chars, drop `why`).

**Expected effect (estimate).** Input: T04 ~−8 s, drawer 001 ~−80 s. Prose: ~−1 to −2.5 s per call (T04 −13 to −30 s). Caching would cut prefill further if honored.

**Validation.** Replay identical saved payloads with and without the projection or cache key (model calls only, no robot) and compare cached tokens and latency. Then a sweep for quality.

### F8 (medium, FAST + FEW-ATTEMPTS). Startup is serial and duplicated; a 401 cost a whole attempt
**Mechanism.**
- Preparation starts and resets a simulator, makes the plan call, then closes the simulator. Execution starts and resets a second one (`evaluate_agent_speedup_v2_user_check.py:274-330`, `evaluate_agent_speedup_v2.py:1384-1394`).
- With `--auto-approve` the review takes 0.13 s, so the plan call (17.5-52.2 s; 514-1,282 output tokens) is pure serial latency.
- Its macro steps were never used, and the first execution call re-analyzes the scene anyway (15-32 s, the longest call in 4 of 5 runs).
- Only overload and stream-break codes are retried (`H1/src/gpt_field/agent_speedup_v2_client.py:149-150,903-914`). `token_expired` (401) and `request_deadline_exceeded` end the episode.

**Evidence.**
- Two simulator startups cost 11.3-19.6 s per run.
- drawer 002 `initial_plan_failure_audit.json`: plan request 401 `token_expired` after 0.48 s, 0 actions. The attempt was consumed, and a new run (003) was needed.

**Improvement.**
1. Keep the preparation simulator for execution, or start the execution simulator concurrently with the plan call.
2. When auto-approved, fold the plan into the first executive call, or let the first approach start while it is drafted.
3. On 401, force one token refresh and resend before any action. This is not an episode retry.

**Expected effect.** −9 to −20 s startup; −17 to −52 s plan (partly offset by a longer first call); no lost attempts to token expiry.

**Validation.** Injected-transport test returning 401 once (the client has `transport=` seams); startup timing on the sweep.

### F9 (medium, FAST + FEW-ATTEMPTS). Termination is the raw LIBERO predicate mid-grasp; there is no completion or failure contract
**Mechanism.**
- The evaluator stops on the first raw success (`evaluate_agent_speedup_v2.py:1440-1443`).
- `finish` only requests an observation, and there is no give-up action (`..._agent_client.py:205-209`).
- There is no call or time cap (README).

**Evidence.**
- In all 5 runs raw success fired while the last object was still grasped. The terminal failing checks were `not_held, at_rest`, and strict success needed the evaluator's separate 38-action release-and-settle diagnostic.
- `finish` was never used.
- Failure has no bound: drawer 001 spent 9 consecutive failure-triggered calls (#17-25) recovering, and H2's T04 had to be user-stopped.

**Improvement.**
1. Make release-and-retreat part of the executive's completion contract.
2. Evaluate strict success (raw + released + at rest) on the fly and terminate on it.
3. Report wall time to strict success.
4. Add an evaluation-side stop rule for runaway episodes (time budget), recorded as a failure.

**Expected effect.** Honest accounting, about +30-40 actions (≈ +2-11 s depending on F1) and possibly one more call. Bounded worst-case time.

**Validation.** Sweep reporting strict-success time; count of episodes hitting the stop rule.

### F10 (medium, FEW-ATTEMPTS + ZERO-SHOT). There is no reachability or joint-limit awareness, so recovering from a stretched arm costs a call per 5-step probe
**Mechanism.** Targets are Cartesian OSC goals with no IK, reachability or joint-limit check. The model receives joint positions but no limits. Its only fine tool is a ≤5-step direct motion, each followed by a new call (`..._agent.py:363-380`).

**Evidence.**
- drawer 001: joint 4 (elbow) at its limit (margin 0.000 rad) at calls 15, 17-21 and 24 (`scripts/` joint-margin check). Calls 17-25 were a 9-call recovery streak, three of them 5-step motions (calls 18-20), costing 141 s of model time plus 143 steps.
- drawer 003: joint 5 at its limit at calls 5 and 9. Calls 26-27 report "arm fairly extended ... poor tracking".
- Not seen in T04 or T08. Evidence is one task family, but the gap is structural.

**Improvement.**
1. Pre-check each target with Panda IK and joint limits (H2's `foundation_harness/robot_geometry.py` already has FK/IK) and report infeasibility in the same call.
2. Include joint-limit margins in feedback.
3. Offer a joint-space posture-recovery primitive the model can request in one call.

**Expected effect.** Removes this class of streak: ≈ −8 calls and −130 s of model time in drawer 001, plus 143 steps (−57 s at today's per-step cost, −7 s after F1).

**Validation.** Offline IK pass over all logged targets of these runs to see which were infeasible; then a sweep counting joint-limit-adjacent calls.

## 5. Prior findings re-checked against H1

| Prior finding | Status in H1 | Evidence |
|---|---|---|
| Model ≈50% of service time; robot idle during calls | **Still applies** (46-57% of wall) | §1.1 |
| ~0.228 s/step harness overhead | **Worse**: 0.28-0.40 s/step, growing in-episode; the planning video adds 0.124-0.203 s | §1.2 |
| Transit cap 20 mm limits speed | Fixed by the 0.10 m cap (−57% actions on T04), but it **introduced overshoot false stalls** | F3 |
| 1.2 rad envelope wastes calls | **Still applies** (T04 and T08 first plans) | F5 |
| Plan-draft whitespace hangs | **Fixed**: compact instruction (`..._user_check_client.py:28-29,412`), 180 s deadline, whitespace guard (`stream.json` settings); the 5 plan calls that reached the model completed in 17-52 s (the sixth, drawer 002, failed with 401 after 0.48 s) | §2 |
| No `max_output_tokens`; history resent; caching ≈0 | **Still applies** | F7 |
| `right_of=+x` direction bug | Fixed in uc7.1 (frozen external-view frame); not re-audited here | – |
| Policy images L-R mirrored | Model images are un-mirrored by `StepView` (`contact_skills.py:283`, `[:, ::-1]`); not re-audited | – |
| `astra._canonical` sort_keys reorders schemas | Not on the wire: requests use `ordered_dumps`, and `_canonical` only hashes the request key (`..._agent_client.py:427-433`); `result.json` output_key_order is preserved | – |
| Axis-angle sign jumps | Present in state8 (e.g. −3.1397 → +3.1360 between reset and step 1) but harmless: control and prompts use quaternions with sign normalization (`action.py:84-86`) | – |

## 6. What it would take to reach ~3 minutes (estimates; levers overlap, so they are applied in order)

| Lever | T04 (402 s) | T08 t100 (634 s) | drawer 001 (1,804 s) |
|---|---|---|---|
| F1 harness (video offline, images on demand, state writes at decisions) | −120 → ~282 | −205 → ~429 | −820 → ~984 |
| F5 vetoes | −18 → ~264 | −29 → ~400 | −15 → ~969 |
| F3+F4 stall semantics (pending-discard calls only) | ~−2 (steps only) → ~262 | −10 → ~390 | −90 → ~879 |
| F10 kinematic recovery | 0 | 0 | −135 → ~744 |
| F2 committed pick-and-place + async verification | −90 → ~172 | −125 → ~265 | −190 → ~554 |
| F6 contact/rotation speed (after F1) | −9 → ~163 | −11 → ~254 | −37 → ~517 |
| F8 startup/plan | −20 → ~143 | −22 → ~232 | −50 → ~467 |
| F7 context/prose | −15 → ~128 | −20 → ~212 | −90 → ~377 |

On these assumptions T04 gets under 3 minutes only if F1 and F2 both land. F1 alone leaves ~280 s, and every lever except F2 leaves ~220 s. T08 lands near ~3.5-4.5 minutes and the 5-stage drawer task near 6-9 minutes. All call-count effects are hypotheses; per the user's acceptance rule, they must be measured in one all-task sweep on a frozen build.

## 7. Scripts and data (scratch)

- `scripts/percall.py`, `scripts/callstats.py` (per-call tokens, latency, fit) → `data/callstats.json`
- `scripts/prefix.py` (prompt prefix stability), `scripts/ctxsize.py` (static vs changing context)
- `scripts/steps.py` (compact per-step extraction from `trace.jsonl`) → `data/*.json`
- `scripts/motion_stats.py`, `scripts/segments.py` (displacement, profiles, convergence tails)
- `scripts/triggers.py` → `data/triggers.json`; `scripts/stall_class.py` → `data/stall_class.json`; `scripts/retry_after_stall.py`
- `scripts/stepgrowth.py`, `scripts/walltable.py`, `scripts/overlap.py` → `data/stepgrowth.json`, `data/walltable.json`, `data/overlap.json`
- Codec benchmark: run inline with `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/.venv-sim/bin/python` and `/venv/fm_dno/bin/python` on `data/obs_sample.json` (one T04 observation)

Not covered: H2 and H3 internals, collision mechanisms beyond their interaction with speed, the direction-semantics fix, the astranomy baseline, and the per-substep cost of the contact monitor (not timed in any artifact).
