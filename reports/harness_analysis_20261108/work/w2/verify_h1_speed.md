# W2 adversarial verification: `h1_speed_attempts` (H1 speed and attempts)

Key: `verify_h1_speed`. Target report: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_speed_attempts.md` (findings F1-F10 plus §1-§3 tables).

## Method and rules

- Read-only for `/workspace/gpt_field`. No simulator, model or GPU job was run.
- Every number below was recomputed from raw run artifacts with my own scripts:
  - `trial/summary.json`, `result_summary.json`, `trial/episode.events.jsonl`, `trial/trace.jsonl`
  - `trial/requests/*.{request,stream,response}.json`, `trial/physical_truth.json`, `trial/simulator.json`
- I did not reuse the analyst's `data/` or `scripts/`. The one exception is the cross-eval `per_call_rows.json`, which I refit only to settle the latency-model conflict.
- I ran two small offline CPU benchmarks:
  - PNG codec on a saved T04 frame, using the run's own simulator Python (`/workspace/venvs/libero-plus/bin/python`, Pillow 12.3.0).
  - ffmpeg/x264 re-encode of 400 frames decoded from T04's own `episode.mp4`.
  - Host load was about 9-10 on 256 cores.
- Abbreviations:
  - `H1/` = `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/`.
  - Runs are under `/workspace/gpt_field/runs/`.
- Scope note: the policy modules are byte-identical across all five runs (`diff -rq source_at_launch`). The stove-off and drawer runs, however, launched a different evaluator and sim worker:
  - files: `evaluate_agent_speedup_v2*.py`, `libero.py`, `agent_speedup_v2_tasks.py`; drawer runs also `agent_speedup_v2_user_check_sim.py` and `ordered_task_evaluation.py`;
  - purpose: custom-task registration and an ordered success evaluator.
  - The report's `H1/` line numbers are exact for T04/T08. For the other runs they are shifted by up to about 65 lines in `evaluate_agent_speedup_v2.py`.
  - Pooling the five runs as "one build" is valid for the policy, but not literally for the evaluator.

## Verdict summary

| Ref | Verdict | Severity after review | One-line reason |
|---|---|---|---|
| Tables §1-§3 | confirmed | low | Reproduced to the last digit. Only exception: the d001 stall split depends on whether position error or route-remaining is used. |
| F1 | confirmed_with_corrections | critical | Mechanism and numbers hold, and the −110..−125 s T04 estimate holds. Corrections: README framing; video cost composition; F1.1 and F1.2 conflict; the replay check is byte-exact on RGB. |
| F2 | confirmed_with_corrections | critical | 69/128 and the overlap bound reproduce. "123/123 observation_after" is protocol-forced, not a model choice. Hard fragment caps (6 segments / 240 steps) block the proposed fix; the analyst missed them. |
| F3 | confirmed_with_corrections | medium | Overshoot exists only with the 100 mm cap (22 events vs 0). Stalls fire about 11 steps after the minimum, often at the apex, not "1-2 steps before arrival". The "misreading" example is refuted. "Speed≈0" alone would still fire at the apex. |
| F4 | confirmed_with_corrections | medium | 19 contact stalls, and 5 discards costing 114.2 s, reproduce. 4 of 19 are not stops and 2 of 19 are not approaches. Under F4's own rule the gain is 2-3 calls, not 5. T04 contacts are 2,117 points in approach and 1,255 in close, not "2,065 in approach". |
| F5 | confirmed | high | 1.5724 / 1.5691 rad, all 5 route vetoes `route_start_blocked`, and 72.9 s all reproduce. |
| F6 | confirmed_with_corrections | low | Shares, speeds and rotation rates reproduce, and the action-saving estimate is in range. The fix adds a 2 cm threshold and carries a force/slip risk. After F1 it is worth about 9 s on T04. |
| F7 | confirmed_with_corrections | medium | Growth, no cache key and the prefix reset reproduce. Overstated: "same feedback three times" and "geometry static". Missed: append-only alone did not yield cache hits. |
| F8 | confirmed_with_corrections | medium | The 401 loss and the retry policy are confirmed. "First call longest in 4 of 5" is false (2 of 5). The avoidable startup is 4.9-8.9 s, not 9-20 s. |
| F9 | confirmed | medium | 5/5 terminal `not_held, at_rest`; the settle is 38 actions (10 open + 8 lift + 20 settle). |
| F10 | confirmed | medium | Joint-limit hits are exact (d001 J4 at calls 15, 17-21, 24; d003 J5 at calls 5, 9). The streak is 141.2 s / 143 steps. |
| LAT | confirmed (h1_speed); cross-eval claim refuted | medium | The input-token effect is real: +14-21 ms per 1k tokens, controlled. A single-variable R²≈0 is an omitted-variable artifact. |

## Tables §1-§3: re-derived from raw artifacts

- **§1.1 wall decomposition** (`trial/summary.json:timing_components`, `result_summary.json:timing`).
  - T04 sums to 401.9 of 402.06 s: prep 28.19 = 10.71 + 17.48 plan; exec startup 8.91; model 200.33; sim 68.43; video 67.98; run_state 13.03; observe+policy 4.63; other 0.90; finalize+outer 9.65.
  - Shares reproduce, e.g. T04 model 54.2% and video 16.9%.
  - Time to first motion is about 77 s: 28.2 + 8.9 + 32.3 (call 1) + 7.6 (call 2). The text writes 32.4 for call 1; the stream says 32.33 s.
- **§1.2 per-step costs** reproduce exactly.
  - T04: 0.1246 / 0.1238 / 0.0237 / 0.0073 s; total 0.282; 0.158 without video.
  - d001: 0.401 / 0.198.
  - The quintile column reproduces from `episode.events.jsonl` wall gaps on no-call steps: T04 0.263, 0.268, 0.279, 0.288, 0.289; d001 0.342 → 0.460.
- **§2.1 per-call statistics** reproduce exactly from `response.json:usage` and `stream.json:elapsed_s`. The analyst differenced budgets, which gives the same numbers.
  - T04: 13 calls, Σ 198.5 s, mean / median / p90 / max 15.3 / 13.4 / 21.5 / 32.3 s, input 7,945 → 87,425, Σ 641,669, cached 2.5%.
  - The other four runs also match.
  - All 128 calls have `attempt` 1 (no retries).
  - 128/128 server-assigned `prompt_cache_key` values are distinct, and no request payload carries one.
  - Cached tokens per call are 0 or 3,072, except 16,000 (T04 call 3) and 7,808 (T08 call 5).
- **§2.2 T04 call table** reproduces: latencies, tokens, actions before the next call (1, 69, 54, 38, 1, 33, 57, 94, 22, 64, 39, 35, 42) and response skills.
- **§2.3 triggers**, from `context.last_feedback` of each request: initial 5, `fragment_complete` 69 (54%), `segment_stall` 41, `target_envelope_rejected` 2, `route_unavailable` 5, direct motion 5, `segment_budget_exhausted` 1.
  - Stall split: with position error, I get 17 overshoot stalls, which matches the analyst. With the predicate's own `route_remaining_m` and a 10 mm cut, d001 is 5/8/4 instead of 6/8/3 (borderline: d001 step 1274 minimum is 10.8 mm).
  - Pending discards: 12. Failure-triggered shares: 38/33/25/50/47%, i.e. 60/118/66/385/427 s.
  - Streaks: d001 17-25, d003 24-27 and 29-34, T08 2-5. Also 3-call streaks at T08 13-15 and stove 5-7, which the report omits.
- **§3 motion** reproduces.
  - T04: approach 56.6% of actions / 20.7% of path at 1.17 mm/step; transit 35.0% / 79.2%, mean / p90 / max 7.3 / 22.4 / 26.3 mm.
  - Zero-translation transit steps: T04 54, T08 16, stove 10, d001 211, d003 86.

## F1: per-step harness work (critical)

**Code checked.**
- Loop order (`H1/scripts/evaluate_agent_speedup_v2.py:1477-1555`): `state()` writes `run_state.json` at deciding/executing/observed (`:1036-1053`); the full `agent_stats()` is written only at "observed".
- `write_json` uses `indent=2, sort_keys=True` with atomic replace (`H1/src/gpt_field/experiment.py:23-26`).
- The video recorder runs synchronously (`:1515-1517`, `:1549-1552`). All five runs used `AgentSpeedUpV2UserCheckVideoRecorder` (`trial/manifest.json`).
- Each step emits twice (`agent_speedup_v1_video.py:213`, `video.py:233`), with 2 repeats each. T04 `episode.mp4` has 2,398 frames = 12 + 536×2 + 13×12 + 549×2 + 60 (exact). That is 4 raw 1600×960 frames per step, piped to `libx264 -preset veryfast -threads 1` (`video.py:89-94`, `agent_speedup_v1_video.py:119-124`).
- Deep copies:
  - The decision render deep-copies the diagnostics (`agent_speedup_v1_video.py:183`).
  - The executed render deep-copies feedback and history_stats (`:217-219`).
  - Every emit deep-copies the planning view (`agent_speedup_v2_user_check_video.py:287`).
  - The full measured path is redrawn in 3 panels (`:178`, `:216-218`).
- Images:
  - The worker encodes at PIL's default level (`observation.py:44`) and validation-decodes (`:114-115`).
  - The host decodes again in `PolicyObservation.from_dict` (`libero.py:662`) and in `CalibratedObservation.__post_init__` (`agent_speedup_v2_user_check_sim.py:84-89`).
  - A second `agent_geometry` IPC runs every step (`:124`). Total per step: 2 encodes + 6 decodes.
- `_sync_framework` deep-copies the full `_feedback_history` every step (`agent_speedup_v2_user_check_agent.py:258`, called at `:605`).
- No native observer was armed: `diagnostics.observers == {}` in the T04 trace. `StepView` decodes lazily (`contact_skills.py:281-284`). So images really are needed only at decision steps (549 steps / 13 calls ≈ 1 in 42).

**My measurements.**
- PNG coding, at load ≈9: encode 33.9 / 32.3 ms at the default level and 9.1 / 8.8 ms at level 1; decode 5.7 / 4.8 ms. So 2 encodes + 6 decodes ≈ 97 ms of the 125 ms `simulator_step_s`.
  - Re-encoding the logged frame at the default level reproduces its base64 length exactly (302,014 chars), which confirms level 6 was used in-run.
- x264 cost: 400 real frames take 6.72 s piped versus 1.38 s for decode alone. That is about 14-17 ms of encoder CPU per frame, or 55-68 ms per step for 4 frames.
- Deep copy: 0.4 ms per history entry, i.e. 5 ms at 13 entries and 18 ms at 42.
- Growth split, from `episode.events.jsonl`:
  - T04: the pre-decision part (observed write + `act()` + decision render) grows 0.061 → 0.088 s, while sim + observe + executed render stays flat at about 0.20 s.
  - d001: 0.108 → 0.165 s and 0.234 → 0.295 s respectively.
- Final file sizes (from the evaluator's final, not per-step, `trial/run_state.json`):
  - `run_state.json`: 0.60-1.07 MB. The largest is stove-off at 1.07 MB, not drawer 001 at 0.92 MB.
  - `agent_stats`: bounded by `recent_segments[-8:]` (`agent_speedup_v2_user_check_client.py:296-300`), so it is not strictly O(history).

**Corrections.**
1. The README does not omit the video. It says "Startup/reset and video overhead are outside that combined figure" (`H1/README.md:127-132`). The substantive point stands: the user waited 402.1 s (`total_wrapper_wall_s`).
2. Video cost composition. A large constant part is full-canvas PIL rendering plus single-threaded x264 encoding of 4 frames per step (back-pressured through a pipe), plus a JSON event record per emit (`episode.events.jsonl` is 11.4 MB for T04). Deep copies and the O(n) path redraw explain the growth, not the bulk.
3. F1.1 conflicts with F1.2. "Render the video after the episode from the trace" works only while per-step images are still logged, and F1.2 removes them. An offline video then needs deterministic re-simulation from the saved actions, which the replay validator already does.
4. The proposed validation would fail as written. `validation/replay_baseline.py` and `validation/VERIFICATION.md:21-22` require all 550 serialized observations, including the encoded RGB bytes, to be exact. The initial-observation identity is a SHA-256 over PNG data URLs (`agent_speedup_v2_user_check.py:66-68`, `agent_speedup_v2_sim.py:671-674`). Any change of PNG level or rendering policy needs image comparison by decoded pixels.

**Estimate check.** Bottom-up for T04, over 536 no-call steps:
- video −68 s (certain by construction);
- images on demand −45..−52 s (if the in-run codec share matches my benchmark);
- sparse `run_state` −10..−12 s.

Total: −117..−131 s, consistent with the analyst's −110..−125 s. T08 (about −220 s upper) and d001 (about −820 s upper) are also consistent.

**Improvement assessment.** Sound, general, with no policy change. Lowest-risk ordering:
- (a) Drop the 4 redundant validation decodes: about −20 ms/step, with zero behaviour change.
- (b) Run the recorder out-of-process or post-hoc by re-simulation, or with fewer repeats and a smaller canvas.
- (c) Lazy camera render and encode at decision steps, with pixel-level replay checks.
- (d) `run_state`, `agent_stats` and `_sync_framework` only at decisions.

PNG level 1 alone would save about 48 ms/step, but it changes the frozen reset hash.

## F2: serial checkpoint-per-phase protocol (critical)

**Confirmed.**
- 69/128 triggers are `fragment_complete`.
- 123 `execute` responses: all end with `observation_after=true`, none sets it mid-sequence, and none has more than 5 segments. Mean 2.07 segments, mean requested steps 135.
- Mean actions per call: 42.2 / 40.1 / 55.6 / 56.3 / 59.2.
- Calls per object: 6.5 (T04, 2 mugs) and 10.5 (T08, 2 pots). For stove-off, "8 per object" ignores the stove subtask; it is 5.3 per subtask.
- Prompt mandates are verbatim: `agent_speedup_v2_user_check_agent_client.py:161-164` (test lift / checkpoint after grasp; inspect after placement), `:81-83`, `:182`.
- Overlap bound Σ min(L_k, E_{k−1}) for T04: 116.2 / 80.1 / 25.4 s at 0.282 / 0.158 / 0.05 s/step (matches 117 / 80 / 25).

**Corrections.**
1. "123/123 ended with observation_after" is not evidence of model behaviour. `_complete_segment` ends with `fragment_complete` whenever no segment is pending, regardless of the flag (`agent_speedup_v2_user_check_fragments.py:426-428`). The informative facts are the short sequences and the absence of mid-sequence checkpoints.
2. The analyst missed hard caps that make the proposed fix impossible without changing them:
   - `MAX_SEGMENTS = 6`, `MAX_SEGMENT_STEPS = 120`, `MAX_EXECUTION_STEPS = 240` (`fragments.py:24-26`, `:174-175`); prompt "sum(max_steps)<=240" (`agent_client.py:133-134`).
   - T04 mug 1 took 346 actions over 8 phases (steps 1-347). A committed pick-and-place therefore cannot fit one fragment.

**Improvement assessment.**
- (1) Commit with measured continuation predicates. This is the right lever and general, but it requires raising the caps and making the predicates model-declared rather than new thresholds.
- (2) Asynchronous verification. This changes the "simulation pauses during inference" contract (`:182`). The model would act on stale images, and a carry could start before an unverified grasp is rejected, which is in tension with COLLISION-AVOIDING when the held object passes near obstacles.
- Prefer (1) first, and restrict (2) to checkpoints that carry no contact risk.
- The expected −4 calls on T04 is a hypothesis; it needs a sweep, as the analyst says.

## F3: 0.10 m cap overshoot and the stall predicate (medium; analyst: high)

**Confirmed.**
- Code: lookahead `max(.03, cap)` (`agent_speedup_v2_user_check_routing.py:154-156`); clip (`fragments.py:339-346`); 2-sample arrival (`:38`, `:368-369`, `:384-386`); window test `errors[0] - min(errors[1:]) < 0.8 mm` after 16 actions (`:404-419`, constants `:43-45`).
- T04 steps 173-195 reproduce exactly: 175.5 → 4.2 (step 184) → 10.7 (step 189) → 4.8 mm, stall at step 195 at 1.5 mm/step. Steps 481-506: 4.6 → 10.8 → 4.9 mm. d003 step 2087 (min 1.3, peak 28.0, end 12.0 mm) and d001 step 46 (1.8 → 13.3 → 11.0 mm) also reproduce.
- Counting transit overshoot events (min < 10 mm, then a rise ≥ 3 mm), whatever ended the segment:

| Run(s) | Overshoot events | Of which ended in `segment_stall` | Peak transit speed (mm/step) |
|---|---|---|---|
| 100 mm runs (T04 2, T08 0, stove 1, d001 9, d003 10) | 22 | 17 (the analyst's count) | 20-27 |
| 20 mm run (`agent_speedup_v2_uc71_can_obstacle_t04_20261103_001`) | 0 | 0 | 5.2-5.9 |
| default-routing run | 0 | 0 | 5.2-5.9 |

  This strongly supports "introduced by the 100 mm cap", although it rests on one episode per setting.
- −5 directly avoidable calls: d001 calls 6 / 25 / 40 and d003 calls 5 / 33, latencies 13.7 + 10.5 + 23.5 + 25.8 + 18.1 = 91.6 s. d001 call 25 is borderline (minimum 10.8 mm).

**Corrections.**
1. The stall fires mechanically 11 steps after the error minimum (12 of 16 overshoot stalls), so often at the overshoot apex or early in the return, not "1-2 steps before arrival".
   - 10 of 16 end at ≤0.65 mm/step, 10-28 mm away.
   - d003 step 27 fires while the arm is still moving away (28.0 → 28.3 → 28.4 mm).
   - Only the T04 cases (and about 4 others) match the "closing at 1-4 mm/step" picture.
2. The "downstream misreading" is refuted. In d003 call 26's fragment (steps 1618-1633), a contact approach with rotation moved the hand away from the target (31.1 → 46.6 mm, 0.7-1.55 mm/step). Call 27's belief, "rotation ... displaced the hand away from the position target", is an accurate reading of those measurements.
3. A second, general mechanism was missed. `PathTracker` commands zero translation once within `endpoint_tolerance_m` (`tracking.py:90-95`), and the route sets that tolerance to the same 4 mm used for arrival (`routing.py:156`). While rotation is still converging, the arm then hovers at the boundary and drifts out, e.g. d001 steps 208-231 sit at 2.4-4.4 mm with commands alternating 0 / 4 mm until the stall at step 241.

**Improvement assessment.**
- (1) A braking-aware reference, `offset ≤ k·sqrt(d)`, is a constant-deceleration profile. It is physically sound and keeps the cruise speed.
- (2) "Stall = speed ≈ 0 while out of tolerance" is insufficient as stated:
  - speed crosses zero at every overshoot apex (0.18 mm/step at d003 step 27), so it would still declare false stalls;
  - it would miss "moving but diverging" cases (d003 step 1633 at 1.45 mm/step away).
- Better: a windowed test of realized progress along the commanded direction. Remove the deadband, or make it smaller than the arrival tolerance. Fix the overshoot first.
- Severity: the measured direct cost is about 92 s of model time across 5,145 s of total wall (1.8%), plus 10-25 return steps per event. That is medium, not high.

## F4: a contact approach that stops on contact is reported as failed (medium)

**Confirmed.**
- `approach` requires the contact profile (`fragments.py:159-160`). A stall ends the fragment (`:418-419`, `:436-447`), and pending segments are not resumed (`agent_speedup_v2_user_check_agent.py:393-401`).
- 19 contact-profile stalls. 5 pending discards: T08 call 13, stove-off call 12, d001 call 36, d003 calls 27 and 34. Their latencies total 114.2 s.
- d003 call 34's belief is verbatim.

**Corrections.**
- 2 of the 19 are contact-profile `move_to_pose`, not `approach` (stove step 160, d001 step 1834).
- Final speeds range 0.0-1.45 mm/step:

| Kind | Count | Behaviour |
|---|---|---|
| Genuine stops | 15 | ≤ 0.3 mm/step |
| Slow creep | 3 | 0.46-0.77 mm/step, 18-20 mm residual |
| Moved away under rotation | 1 | d003 step 1633 |

- Under F4's own rule (a measured stop in a contact `approach`), only stove call 12 and d003 call 34 clearly qualify, plus d001 call 36 if the rule covers every contact-profile motion. The gain is therefore −2..−3 calls (−46..−90 s), not −5 / −114 s.
- Call 34 re-issued both the close and a 2 cm test lift, not only the close.
- T04 forbidden contacts are 3,372 points (`result_summary.json:collision`), all palm vs the grasped mug. They split 2,117 during contact `approach` and 1,255 during `set_gripper` close. "2,065 samples during approach" is roughly right but incomplete.

**Improvement assessment.** Removing the false-failure label is the right direction. It is safest when the model declares intent per segment (for example, "until contact"). A stop is then success only when planned, with its residual reported, and creep or divergence still yields. This adds no threshold beyond the existing stop test.

## F5: up-front vetoes (high)

**Confirmed.**
- Envelope reject before any action (`fragments.py:27-34`, `:285-291`); per-step 0.1 rad cap (`:234`).
- `route_start_blocked` (`routing.py:105-106`).
- One agent call per `act()`, followed by a 1-step hold (`agent_speedup_v2_user_check_agent.py:512-541`).
- First executive answers were rejected in T04 (1.5724 rad, 0.1344 m) and T08 (1.5691 rad, 0.2331 m).
- All 5 route failures are `route_start_blocked`: T04 call 6, T08 calls 5 and 15, stove step 178, d001 step 1581.
- The veto re-asks cost 7.6 + 10.6 + 7.1 + 12.1 + 10.2 + 10.1 + 15.2 = 72.9 s.

**Improvement assessment.**
- Dropping the rotation envelope or auto-splitting is low risk: the envelope is not a collision check, and the per-step cap remains.
- The exit-path proposal is sound only if the exit leg must strictly increase the distance from the unexpanded box. Model boxes are estimates, so do not exit "vertically" by default.
- Severity stays high because it directly violates the user's "valid first plans" goal, although the time cost is small (1.4% of total wall).

## F6: slow approach and rotation primitives (low)

**Confirmed.**
- All shares and speeds (§3).
- Rotation in T04 segment 0 goes from 0.783 to 0.096 rad over steps 1-31, about 0.023 rad/step, while translation finishes by step ~10.
- My re-estimate of the fast-approach saving, with the same 7.5 mm/step and 2 cm tail model: T04 −142 actions (analyst −143); T08 −247 (−217); stove −229 (−253); d001 −602 (−695..−732); d003 −671.

**Assessment.**
- The "last ~2 cm" schedule adds a new threshold, which is against the user's rule.
- Transit-speed lifts and retreats risk slip and contact forces.
- After F1 it is worth only about 9 s on T04.
- Raising the rotation cap is sensible together with F3's braking, but the rotation-translation coupling seen in d003 call 26 needs handling.

## F7: context growth and caching (medium)

**Confirmed.**
- Input grows by +6.6k tokens per call (T04) and +6.1k (d001).
- From call 7 the oldest image pair is stripped, so the common prefix with the previous request drops to 0 items. Request item hashes: call 7 has prefix 0 of 14; call 8 has 3 of 17.
- No `prompt_cache_key` is sent, and every server-assigned key is distinct.
- Context sections in d001 call 21: `capabilities` 4,802 chars, `macro` 2,360, `geometry` 2,000.
- Native macros: 0 of 128 responses (types are only `execute` and `motion`).
- The F7 estimates (T04 about −8 s, d001 about −80 s) are consistent with the history-growth cost I compute in LAT (9.7 s and 96.2 s).

**Corrections.**
- "The same feedback three times" is overstated. `last_feedback` is about 3.3-3.7k chars (compacted). `execution.summary.last_result` is a 225-char summary, and `recent_feedback` is a compacted list of 8 summaries.
- `geometry` includes the wrist camera, which moves every step, so it is only partly static.
- Calls 2-6 were strictly append-only, yet 4 of 5 got 0 cached tokens. Making the history append-only will not by itself yield cache hits; the missing stable key/routing matters more.
- `prompt_cache_retention` and `max_output_tokens` are transport-unsupported fields (`H1/src/gpt_field/chatgpt_plan.py:37-41`), so visible output can only be bounded through the schema.

**Improvement assessment.**
- Static-to-instructions moves and compact projections are sound and general.
- Capping the belief/plan prose risks decision quality, because the belief is the model's cross-call memory. Test it.

## F8: duplicated startup and the 401 loss (medium)

**Confirmed.**
- Preparation closes its simulator and execution starts a new one (`evaluate_agent_speedup_v2_user_check.py:274-330`; `evaluate_agent_speedup_v2.py:1384-1394`).
- Only 503 overload and 4 transient stream codes are retried (`agent_speedup_v2_client.py:140-150`, `:903-914`).
- `..._20261103_002/initial_plan_failure_audit.json` records 401 `token_expired`, 0 actions; the stream `elapsed_s` is 0.37 s (the report says 0.48 s).
- Plan calls took 17.4-50.4 s of stream time (pipeline 17.5-52.2 s), with 514-1,282 output tokens.

**Corrections.**
- "The first execution call is the longest in 4 of 5 runs" is false. By latency it is 2 of 5 (T04, stove); by reasoning tokens 1 of 5.
- Two startups cost 11.3-19.6 s, but the plan needs the first reset's images, so only the second startup (4.9-8.9 s, `startup_to_initial_observation_s`) is avoidable.
- The 0.13 s auto-approval figure is not in any artifact I found.

**Assessment.** A single 401 refresh-and-resend before any action is sound and is not an episode retry. Folding the plan into the first executive call is plausible, but it removes the reviewed goal contract that the executive uses (`approved_plan` and `reviewed_goal_contract` in the context).

## F9: termination at raw success mid-grasp (medium)

**Confirmed.**
- The evaluator breaks on `sim.status["success"]` (`evaluate_agent_speedup_v2.py:1440-1443`). `finish` only observes and there is no `give_up` (`agent_client.py:205-209`). `max_wall_time_s` is null.
- In all five `result_summary.json:physical_truth` records: `physical_success_at_terminal: false`, `terminal_failing_checks: [not_held, at_rest]` on the last object, and success only after settle.
- The settle diagnostic (`trial/physical_truth.json:after_release_settle.settle`) is 38 actions: 10 open + 8 lift + 20 settle.
- The drawer runs use the ordered evaluator (`success_qualification: ordered_custom_goal`), and their last subgoal (moka pot on the stove) was also still held.

**Assessment.** The improvements are sound. Terminating on strict success requires release to be part of the executive contract. The time cap must be task-agnostic.

## F10: no joint-limit awareness (medium)

**Confirmed exactly.**
- Measured `joint_position_rad` in each request against the robosuite Panda ranges (`robosuite/models/assets/robots/panda/robot.xml`):
  - d001 J4 at its limit (margin ≤ 0.0005 rad) at calls 15, 17, 18, 19, 20, 21, 24;
  - d003 J5 at calls 5 and 9;
  - none in T04, T08 or stove.
- The model context contains no joint limits.
- d001 calls 17-25 cost 141.2 s over 143 steps; calls 18-20 were the 5-step direct motions.

**Assessment.** The IK/limit pre-check and margins in feedback are general. "−8 calls" is an upper bound.

## LAT: latency model reconciliation

Fits on the 128 H1 execution calls, from raw usage and stream timing (`w2/data/latency_fit.json`, `w2/data/latency_robust.json`):

| Model | Input coefficient (s per 1k tokens) | Robust SE (HC1) | R² |
|---|---|---|---|
| input only | +0.0117 | 0.0152 | 0.006 |
| reasoning + visible | – | – | 0.934 |
| input + reasoning + visible | **+0.0179** | 0.0038 | 0.9487 |
| + run fixed effects | +0.0167 | 0.0040 | 0.954 |
| calls with k ≥ 6 (steady image window) | +0.0213 | 0.0049 | 0.935 |
| pre-visible phase ~ input + reasoning | +0.0140 | 0.0020 | 0.978 |
| cross-eval's own 186 calls, input + reasoning + output | +0.0111 | 0.0045 | 0.892 |
| cross-eval's own 186 calls, input only | −0.0176 | 0.0159 | 0.009 |

- The analyst's equation `L = 6.63 + 0.0179·input_k + 0.0518·reasoning + 0.0074·visible` (R² 0.949) reproduces exactly, with bootstrap 95% CI [0.011, 0.026] on the input term.
- Partial R² of input is 0.22. Input is uncorrelated with reasoning tokens (r = −0.04).

**Decision.** The h1_speed statement is correct. The cross-eval claim "does not depend on input or history tokens (R² = 0.009)" (`w1/cross_eval_methodology.md:26`, `:204-206`) is wrong. Reasoning tokens explain about 93% of the variance and are independent of input, so the input effect is invisible in a single-variable fit but significant once reasoning is controlled. This holds even on the cross-eval's own data.

One caveat: within a run, input grows almost linearly with call index. With run fixed effects plus call index, the input term is unidentified (SE 0.15). A within-episode server drift cannot be excluded, but the phase fit (prefill sits in the pre-visible phase) supports prefill as the cause.

**What the user should conclude.**
- Context size costs about 11-21 ms per 1k input tokens: about 1.6 s per call at T04's final 87k tokens, and about 4.7 s at d001's final 261k.
- History growth costs about 10 s per T04 run and about 70-96 s per drawer run, roughly 2-5% of wall.
- That is real but second-order. Per-call fixed cost (about 6.6 s) and reasoning tokens (about 52 ms per token) dominate.
- So fewer calls first, then fewer reasoning tokens, then a compact and cacheable context.

## Missed issues (summary; details in the structured result)

1. Fragment caps (6 segments / 120 per segment / 240 total) structurally force at least 2 calls per pick-and-place (T04 mug 1: 346 actions, 8 phases). F2's fix depends on lifting them. Severity high.
2. `PathTracker` zero-translation deadband equal to the 4 mm arrival tolerance, combined with 2-sample arrival and rotation coupling, makes the arm hover at the boundary (d001 steps 208-241). Severity medium.
3. Video cost is mostly a constant per step: full-canvas render plus 4-frame single-thread x264 and per-emit event JSON. A lighter or out-of-process recorder keeps live video. Severity medium.
4. F1's offline video conflicts with on-demand images, and the replay is byte-exact on RGB. Severity medium.
5. Rotation-translation coupling under the contact profile drives the hand away from targets (d003 call 26). Severity medium.
6. The cross-eval latency inference is wrong (see LAT). Severity medium.
7. An append-only history still got no cache hits; the transport rejects `prompt_cache_retention` and `max_output_tokens`. Severity low.
8. Four of six per-step PNG decodes are redundant re-validations, and `trace.jsonl` logs about 0.56 MB of base64 images per step (308 MB for T04, 1.13 GB for d001). Severity low.

## Scripts and data (W2)

All under `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/`:

| Topic | Scripts | Data |
|---|---|---|
| Per-call stats and latency fits | `scripts/percall_w2.py`, `scripts/latency_fit.py`, `scripts/latency_robust.py` | `data/percall_w2.json`, `data/latency_fit.json`, `data/latency_robust.json` |
| Triggers and responses | `scripts/triggers_w2.py`, `scripts/responses_w2.py` | `data/triggers_w2.json`, `data/responses_w2.json` |
| Per-step traces and stalls | `scripts/steps_w2.py`, `scripts/stall_w2.py`, `scripts/overshoot_events_w2.py` | `data/steps_*.json`, `data/stall_w2.json` |
| Motion and approaches | `scripts/motion_w2.py`, `scripts/approach_model_w2.py` | – |
| Wall gaps per step | `scripts/step_gaps.py`, `scripts/gap_split.py` | – |
| Benchmarks | `scripts/bench_codec.py`, `scripts/extract_obs.py`, `scripts/deepcopy_bench_w2.py` | `data/bench_codec_liberoplus.json` |
| Joints, prefixes, routes, contacts | `scripts/joint_margin_w2.py`, `scripts/prefix_w2.py`, `scripts/route_reasons_w2.py`, `scripts/forbidden_by_skill_w2.py` | – |
