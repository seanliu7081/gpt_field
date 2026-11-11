# Cross-harness quantitative comparison and evaluation methodology

Key: `cross_eval_methodology`. Date: 2026-11-07. This was a read-only analysis: no simulator, model or GPU job was started and nothing under `/workspace/gpt_field` was edited. All numbers below come from saved artifacts, recomputed by the scripts listed in the appendix (scratch directory `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/`).

Harness labels:
- **H1**: `agent_speedup_v2_obstacle_harness_100mm` (uc7.1 with transit100).
- **H3**: `foundation_vision` (the uc7.1 runner plus a vision summary in the context).
- **H2**: `foundation_harness`.
- **astranomy**: the direct one-target-per-call baseline.
- **uc7.1-T20**: the same runner with the main-tree default 20 mm transit cap.

## Summary

1. **None of the existing pairwise conclusions is statistically supported.**
   - Every configuration has n = 1 on init 0 / seed 7.
   - The plan requests for the T04+can scene were byte-identical across uc7.1-T20, H1 and H3. Each produced a different approved plan, with latencies of 14.3, 17.5 and 18.8 s.
   - The plan requests for T08+bottle were identical for H1 and H3. They produced different plans (H3's has `placement_goals: []`) and took 22.3 vs 36.4 s.
   - The only true replicate pair (drawer_bowl_moka 001 and 003, identical inputs and build) differs by 43 vs 37 calls and by 4,715 vs 2,068 forbidden samples.
2. **Model latency dominates time in every harness that moved:**
   - 71–76 % of control+plan time for H1 and H3;
   - 69 % for uc7.1-T20;
   - 92 % for astranomy.

   Motion is 2–10 % and other harness overhead is 6–22 %. H1's model time alone on T04 (217.8 s) already exceeds the 180 s target.

   Per-call latency is about 9 s + 50 ms per reasoning token + 6.5 ms per output token (R² = 0.89, n = 186). It does not depend on input or history tokens (R² = 0.009). The levers are therefore fewer calls and fewer reasoning tokens per call, not shorter history.
3. **Forbidden-contact totals rank the harnesses wrongly.**
   - H1 T04's 3,372 samples are all palm-on-grasped-mug contacts, with 0 obstacle contact.
   - uc7.1-T20 T04 (1,325 samples) is the run that actually touched the can (33 samples, one 4-action episode).
   - H1 T08's 3,826 samples include a 47-action gripper press into the ketchup bottle (3,504 samples). It is the only robot-vs-obstacle collision in the whole set.
   - H3 T08's 400 samples are all pot-vs-burner placement contacts, with 0 obstacle contact.
   - Samples scale as duration × contact geoms × 25 substeps. They should be replaced by classified contact episodes.
4. **Calls are structural, not failure-driven.**
   - Across the 9 uc7.1-family episodes, 58 % of executive calls (108/186) come after a fragment that completed successfully.
   - 27 % (51/186) come after `segment_stall`.
   - 3 % (6/186) are repair calls after a framework veto that produced zero motion. These are `target_envelope_rejected` on the first fragment of both H1 runs and `route_start_blocked` by the model's own boxes.
   - The router never ran a detour (`detour_stages = 0` and `planner_search_calls = 0` in every run that has routing metrics).
5. **Per-step overhead is unchanged since the 2026-10-26 report.**
   - Non-model time is 0.158 s/step without video and 0.282 s/step with video, against 0.05 s of simulated motion (H1 T04).
   - `simulator_step` takes 0.125 s/step. Re-encoding two 512² PNGs every step costs about 68 ms/step in the simulator venv's Pillow (offline benchmark), yet only 13 of 549 steps feed a model call.
   - `run_state.json` is rewritten with the full agent stats every step. This cost 13 s in H1 T04 and 109 s in drawer 001, and the per-step cost grows with episode length.
6. **H3's vision context roughly doubled per-call latency and produced no usable geometry.**
   - 0/77 (T04) and 0/66 (T08) target frames yielded a usable 3D centre.
   - Mean call latency was 29.8 / 32.4 s, against 15.4 / 16.0 s for H1, with about 2× the reasoning tokens per call.
   - H3 also received the obstacle's name as a per-task `--vision-query` (`can`, `bottle`), which breaks the zero-shot input rule.
7. **H2 made no task progress, and the code that ran is no longer current.**
   - 8 of 8 operations were rejected by local planning, at about 17.6 s per failed cycle with zero motion.
   - 6 of its 9 source files have changed since that run, and the current revision has no live evaluation.
8. **Measurement definitions differ between runners.**
   - Initial-plan time is inside some timers and outside others.
   - H3's model timer includes vision; H2's control timer includes a withheld call.
   - Video takes 0.046–0.124 s/step depending on the runner.
   - Success is always declared while the last object is still held (strict-at-terminal 0/11).
   - One "T04" run actually used the official layout with no can.
   - The stove-off variant changes the task sentence.
   - Obstacle poses are never recorded, so "can tipped" is a visual claim only.

A compact protocol with a cost estimate is in §8: about 95 episodes, about 1,700 model calls and about 13 h of serial wall time.

---

## 1. Normalised run table (every run in `runs/` dated 2026-11-03..07)

Definitions:
- **Motion** = actions × 0.05 s (20 Hz).
- **Control+plan** = runner `control_without_video_s` + separate initial-plan model time. H2 and astranomy have no separate plan; their first call is inside control.
- **Pure model** = execution model pipeline − vision (H3 only) + plan.
- **Harness overhead** = control+plan − pure model − vision − motion.
- **Strict@T** = the runner's strict physical check at the policy's terminal step.

### 1a. Outcome, calls, time

| # | harness | scene | run (under `runs/`) | status | raw | strict@T | calls plan+exec | actions | act / exec call | pure model s | vision s | plan s | control w/o video s | control+plan s | process wall s | video s | motion s | harness overhead s | overhead s/action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | H1 | T04+can | `agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001` | success | Y | N | 1+13 | 549 | 42.2 | 217.8 | 0 | 17.5 | 287.3 | 304.8 | 402.1 | 68.0 | 27.4 | 59.5 | 0.108 |
| 2 | H3 | T04+can | `foundation_vision_t04_can_obstacle_20261105_001` | success | Y | N | 1+11 | 604 | 54.9 | 332.2 | 14.7 | 18.8 | 432.4 | 451.2 | 569.8 (incl. 24.6 SAM/DA3 preload) | 59.7 | 30.2 | 74.0 | 0.123 |
| 3 | H2 (old revision) | T04+can | `foundation_harness_t04_can_obstacle_001` | user_stop | N | N | 0+12 (1 withheld) | 19 | 1.6 | 112.8 (+12.2 withheld) | (perception inside local compute) | 0 | 202.3 | 202.3 | – | 4.3 | 0.95 | 88.5 (71.0 local compute + 12.2 withheld wait) | 4.66 |
| 4 | uc7.1-T20 | T04+can | `agent_speedup_v2_uc71_can_obstacle_t04_20261103_001` | success | Y | N | 1+26 | 1276 | 49.1 | 477.7 | 0 | 14.3 | 682.8 | 697.1 | 906.1 | 180.6 | 63.8 | 155.6 | 0.122 |
| 5 | uc7.1-T20 | **T04 official, NO can** | `agent_speedup_v2_uc71_default_routing_t04_20261103_001` | success | Y | N | 1+10 | 801 | 80.1 | 217.0 | 0 | 19.4 | 338.1 | 357.5 | 498.0 | 115.1 | 40.0 | 100.5 | 0.125 |
| 6 | astranomy | T04+can | `astranomy_direct_can_obstacle_t04_high_20261104_001` | infra error (HTTP 503) | N | N | 0+34 | 161 | 4.7 | 274.0 | 0 | 0 | 303.2 | 303.2 | 322.0 | 7.5 | 8.1 | 21.2 | 0.132 |
| 7 | astranomy | T04+can | `astranomy_direct_can_obstacle_t04_high_20261104_002` | success | Y | N | 0+63 | 306 | 4.9 | 682.9 | 0 | 0 | 740.2 | 740.2 | 765.4 | 14.2 | 15.3 | 41.9 | 0.137 |
| 8 | astranomy | T04+can, prompt + "…in collision avoidance manner" | `astranomy_direct_can_obstacle_t04_collision_avoidance_high_20261105_001` | user_stop | N (white mug 42 mm off) | strict checks pass, raw fails | 0+87 | 409 | 4.7 | 1059.2 | 0 | 0 | 1137.5 | 1137.5 | 1168.9 | 19.8 | 20.4 | 57.8 | 0.141 |
| 9 | H1 | T08+bottle | `agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001` | success | Y | N | 1+21 | 843 | 40.1 | 358.7 | 0 | 22.3 | 467.4 | 489.7 | 634.1 | 122.5 | 42.1 | 88.9 | 0.105 |
| 10 | H3 | T08+bottle | `foundation_vision_t08_bottle_obstacle_20261105_001` | success | Y | N | 1+11 | 590 | 53.6 | 379.3 | 13.2 | 36.4 | 464.1 | 500.5 | 614.1 | 64.5 | 29.5 | 78.5 | 0.133 |
| 11 | H1 config | T08 stove-off+bottle (**different sentence**: "turn on the stove and put both moka pots on it") | `agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001` | success | Y | N | 1+16 | 889 | 55.6 | 299.7 | 0 | 37.9 | 420.7 | 458.6 | 628.1 | 146.7 | 44.5 | 114.5 | 0.129 |
| 12 | H1 config | drawer_bowl_moka (5 stages, no obstacle) | `…drawer_bowl_moka_transit500_cap100_20261103_001` | success | Y | N | 1+42 | 2364 | 56.3 | 833.8 | 0 | 52.2 | 1249.2 | 1301.3 | 1804.2 | 479.2 | 118.2 | 349.4 | 0.148 |
| 13 | H1 config | drawer_bowl_moka | `…_20261103_002` | initial_plan_failed (HTTP 401 token_expired) | – | – | 1 (error)+0 | 0 | – | 0 | – | – | – | – | 14.0 | – | 0 | – | – |
| 14 | H1 config | drawer_bowl_moka | `…_20261103_003` | success | Y | N | 1+36 | 2131 | 59.2 | 880.3 | 0 | 50.2 | 1208.4 | 1258.6 | 1676.2 | 395.8 | 106.5 | 271.8 | 0.128 |

Sources:
- Rows 1, 4, 5, 9 and 11–14: `result_summary.json` (`timing`, `model_calls`, `physical_truth`) and `trial/summary.json` (`timing_components`).
- Rows 2 and 10: `analysis.json` (`timing_s`, `calls`) and `trial/summary.json`.
- Row 3: `analysis.json` (`timing`), `trial/summary.json` and `trial/foundation_harness/events.jsonl`.
- Rows 6–8: `result_summary.json` (`timing`, `model_calls`, `outcome`).
- Row 13: `initial_plan_failure_audit.json`.
- Transit settings come from `result_summary.json` (`transit_command_limit_m`) and `run.py` (`GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M`): rows 4–5 use 0.02; H1, H3 and rows 11–14 use 0.10.

Attempts (failed launches before success, same configuration):
- H1 T04: 0. H1 T08: 0. H3 T04: 0. H3 T08: 0.
- astranomy T04: 1 (infrastructure: HTTP 503 after 34 calls and 322 s).
- drawer: 0 before the first success. The repeat lost one launch to HTTP 401 before success 003.
- H2: not successful (1 launch, user-stopped). The astranomy collision-avoidance prompt variant was also not successful.
- Over the window: 10 raw successes in 14 launches (10/12 excluding infrastructure). Strict success at the terminal step: 0 of 10.

### 1b. Forbidden contacts: totals, split by meaning, and episodes

An **episode** is a contiguous run of actions that adds samples for the same contact pair; gaps of 6 actions or fewer are merged. The split is:
- **obstacle**: any contact with the added can or bottle;
- **robot vs scene/other object**: `robot_scene` and `robot_object` excluding the obstacle;
- **held object vs scene**: `manipulated_object_scene` excluding the obstacle;
- **palm vs target**: `robot_nonfinger_manipulated_object`.

| # | run | forbidden samples | obstacle | robot vs scene/other | held obj vs scene | palm vs target | pair-episodes | obstacle episodes | first forbidden action |
|---|---|---|---|---|---|---|---|---|---|
| 1 | H1 T04 | 3372 | 0 | 0 | 0 | 3372 | 3 | **0** | 114 |
| 2 | H3 T04 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | – |
| 3 | H2 T04 (19 hold actions) | 0 | 0 | 0 | 0 | 0 | 0 | 0 (no motion) | – |
| 4 | uc7.1-T20 T04 | 1325 | **33** | 0 | 0 | 1292 | 2 | **1** | 120 |
| 5 | uc7.1-T20 T04 official (no can) | 726 | n/a | 0 | 0 | 726 | 1 | n/a | 137 |
| 6 | astranomy 001 | 111 | **111** | 0 | 0 | 0 | 1 | **1** | 45 |
| 7 | astranomy 002 | 580 | **426** | 0 | 0 | 154 | 2 | **1** | 66 |
| 8 | astranomy CA | 634 | **634** | 0 | 0 | 0 | 1 | **1** | 126 |
| 9 | H1 T08 | 3826 | **3615** (gripper 3504 + pot 111) | 0 | 211 (pot–burner) | 0 | 5 | **3** | 37 |
| 10 | H3 T08 | 400 | 0 | 0 | 400 (pot–burner) | 0 | 2 | **0** | 277 |
| 11 | stove-off T08 | 1650 | 0 | 0 | 1650 (pot–burner) | 0 | 3 | 0 | 548 |
| 12 | drawer 001 | 4715 | n/a | 1109 (robot–cabinet) | 697 (bowl–cabinet) | 2909 | 10 | n/a | 148 |
| 14 | drawer 003 | 2068 | n/a | 17 | 611 | 1440 | 5 | n/a | 223 |

Episode spans (action indices, length, samples):
- **H1 T04**: three palm-vs-mug episodes: 114–132 (19 actions, 473 samples), 261–300 (40, 865), 420–466 (47, 2,034).
- **uc7.1-T20 T04**: mug–can 141–144 (4 actions, 33 samples), inside palm–mug 120–153.
- **astranomy 002**: mug–can 66–98 (33 actions, 426 samples).
- **astranomy CA**: mug–can 126–137 (12 actions, 634 samples).
- **H1 T08**:
  - gripper–ketchup bottle 37–83 (**47 actions, 3,504 samples**);
  - pot–bottle 223–232 (10, 110) and 362 (1, 1);
  - pot–burner 510–513 (4, 149) and 540 (1, 62).
- **H3 T08**: pot–burner 277–282 (6, 278) and 299–301 (3, 122).

Samples per action range from about 11 to about 78, so sample totals mix contact intensity with duration. The same pot-vs-`flat_stove_1_collision_burner` geom pairs appear in both the allowed `manipulated_object_support` class and the forbidden `manipulated_object_scene` class within the same action windows (`trial/collision.json` `pairs_by_class`, H3 T08 and H1 T08). Which classifier branch fired (normal more than 35° from vertical, or contact point outside the cook-region footprint) is not recorded. See `agent_speedup_v2_contacts.py:208-239` in H1, and `runs/foundation_vision_t08_bottle_obstacle_20261105_001/monitoring/collision_classification_review.json`, which notes that `force_impulse_or_severity_threshold` is null. `physical_truth.json` does not contain the obstacle's pose in any run, so obstacle displacement or tipping is never measured.

## 2. Time decomposition

| run | control+plan s | pure model | vision | motion | harness overhead | model share | median call s | mean reasoning tokens/call | first exec call s |
|---|---|---|---|---|---|---|---|---|---|
| H1 T04 | 304.8 | 217.8 | 0 | 27.4 | 59.5 | **71 %** | 13.6 | 123 | 32.4 |
| H3 T04 | 451.2 | 332.2 | 14.7 | 30.2 | 74.0 | **74 %** (+3 % vision) | 25.6 | 259 | 50.2 |
| H2 T04 | 202.3 | 125.0 incl. withheld | – | 0.95 | 71.0 local + 5.4 | 62 % (+35 % local) | 9.9 | 23 | 9.9 |
| uc7.1-T20 T04 | 697.1 | 477.7 | 0 | 63.8 | 155.6 | 69 % | 14.7 | 129 | 37.3 |
| astranomy 002 | 740.2 | 682.9 | 0 | 15.3 | 41.9 | **92 %** | 10.0 | 24 | 28.0 |
| H1 T08 | 489.7 | 358.7 | 0 | 42.1 | 88.9 | **73 %** | 12.4 | 142 | 30.8 |
| H3 T08 | 500.5 | 379.3 | 13.2 | 29.5 | 78.5 | **76 %** (+3 %) | 30.1 | 253 | 78.7 |
| stove-off | 458.6 | 299.7 | 0 | 44.5 | 114.5 | 65 % | 14.3 | 110 | 28.9 |
| drawer 001 | 1301.3 | 833.8 | 0 | 118.2 | 349.4 | 64 % | 16.1 | 150 | 24.5 |
| drawer 003 | 1258.6 | 880.3 | 0 | 106.5 | 271.8 | 70 % | 19.8 | 218 | 15.2 |

### Harness overhead components (H1 T04, `trial/summary.json` `timing_components`)

| component | seconds |
|---|---|
| `simulator_step_s` | 68.4 (0.125 s/step, against 0.05 s simulated) |
| `run_state_s` | 13.0 |
| `observe_transition_s` | 4.0 |
| `policy_compute_s` | 0.65 |
| residual | 0.9 |
| `video_render_s`, excluded | 68.0 (0.124 s/step) |

`run_state_s` per step grows with episode length:
- H1 T04: 0.024 s/step;
- H1 T08: 0.029;
- uc7.1-T20: 0.027;
- drawer 003: 0.040;
- drawer 001: 0.046.

The writer serialises `policy.agent_stats()` every step at the `observed` stage (`scripts/evaluate_agent_speedup_v2.py:1036-1053` in H1). `trial/agent_stats.json` is 722 KB and `run_state.json` is 772 KB in H1 T04.

### Per-step image cost

Every step, `PolicyObservation.from_libero` PNG-encodes both 512² images, then decodes both to validate them (`src/gpt_field/observation.py:43-45` and `:114-115`). The controller decodes them again in `from_dict`.

An offline benchmark with the simulator interpreter (`/workspace/venvs/libero-plus/bin/python`, Pillow 12.3.0, zlib 2.3.3) on a saved frame measured:
- 2 × encode + base64: **68.3 ms/step**;
- encode + decode: 75.7 ms/step.

This is about 55–60 % of the 0.125 s `simulator_step` (host shared with training, so treat it as approximate). Only 13 of 549 steps feed a model call.

### Zero-call overhead reference

H1's deterministic saved-response replay (`validation/results/combined_replay_005.json`) reproduced all 549 actions exactly with 0 live calls in `elapsed_s` = 97.5 s, including reset. That is 70 s more than the 27.45 s of motion. The replay is a zero-cost benchmark for any per-step overhead change.

### Comparison with the earlier report

The 2026-10-26 finding (`reports/agent_speedup_analysis_20261026/README.md`) was about 0.23 s/step of per-step overhead and a model share of about 50 %. Both still hold:
- non-model time per step is now 0.158 s without video and 0.282 s with video;
- the model share has risen to about 71 %, because transit100 cut actions but not calls.

### Model-latency drivers

These come from per-call tokens (differenced cumulative budgets in `trial/requests/*.result.json`) and per-call latencies (`trace.jsonl` `diagnostics.api_calls`), n = 186 executive calls in the uc7.1 family:

| model | fit | R² |
|---|---|---|
| tokens only | `latency ≈ 8.98 s + 0.0498 s × reasoning_tokens + 0.0065 s × output_tokens` | 0.888 |
| input tokens only | `latency ≈ 21.8 − 0.018 s per 1k input tokens` | 0.009 |

- Input tokens per call grow with history: H1 T04 first call 7.9k, last call 87k; drawer 001 last call 261k. This has no measurable latency effect.
- Prompt caching is negligible: the cached share is 0–2.5 % in most runs and 10.4 % in row 5.
- Astranomy calls (17–35 reasoning tokens) still take 4.4–28.8 s; the median is 10.0 s. This is a per-call floor of roughly 5–10 s on the subscription transport.

### Model stochasticity on identical inputs

The initial plan requests are byte-identical across runs on the same scene (same SHA over the request JSON):

| scene | runs | latency s | reasoning tokens | approved plans |
|---|---|---|---|---|
| T04+can | uc7.1-T20, H1, H3 | 14.3 / 17.5 / 18.8 | 98 / 219 / 177 | three different (`trial/approved_plan.json` sha 39fbd3…, f77c06…, 46efc3…) |
| T08+bottle | H1, H3 | 22.3 / 36.4 | 235 / 345 | two different; H3's has `placement_goals: []` and the stove marked `fixed` |

## 3. What triggers each executive call (uc7.1 family: H1, H3, uc7.1-T20, stove-off, drawer; 9 episodes, 186 calls)

Fragment endings come from `trace.jsonl` `feedback.primitive_feedback.fragment`. Zero-motion vetoes are rows where a call's fragment was rejected before any action and a repair call followed at step+1.

| run | exec calls | fragment complete | segment_stall | route_unavailable mid-fragment | budget exhausted | zero-motion vetoes (cause) | repair-call s | mean actions/fragment |
|---|---|---|---|---|---|---|---|---|
| H1 T04 | 13 | 7 | 3 | 0 | 0 | 2 (1 `target_envelope_rejected` on the first fragment, 1 `route_start_blocked`) | 18.5 | 50.5 |
| H1 T08 | 21 | 13 | 2 | 1 | 1 | 2 (1 envelope on the first fragment, 1 route) | 19.4 | 49.0 |
| H3 T04 | 11 | 8 | 1 | 0 | 0 | 1 (route) | 15.0 | 61.3 |
| H3 T08 | 11 | 6 | 4 | 0 | 0 | 0 | 0 | 55.1 |
| uc7.1-T20 T04 | 26 | 17 (+1 observation_required) | 4 | 1 | 0 | 1 (route) | 10.5 | 52.7 |
| T04 official (row 5) | 10 | 8 | 1 | 0 | 0 | 0 | 0 | 83.2 |
| stove-off | 16 | 11 | 3 | 1 | 0 | 0 | 0 | 56.3 |
| drawer 001 | 42 | 20 | 17 | 1 | 0 | 0 | 0 | 61.6 |
| drawer 003 | 36 | 18 | 16 | 0 | 0 | 0 | 0 | 61.3 |
| **total** | **186** | **108 (58 %)** | **51 (27 %)** | 4 | 1 | **6 (3 %)** | | 49–83 |

Routing in the three uc7.1-family runs with `routing_metrics.json` (H1 T04, H1 T08, uc7.1-T20 T04):
- `detour_stages` = 0 and `planner_search_calls` = 0 in all three;
- `routing_unavailable_stages` = 1–2 per run.

The H3 runs did not write `routing_metrics.json`. The executive prompt requires a visual checkpoint after every test lift and after every placement (`agent_speedup_v2_user_check_agent_client.py:157-164`). That makes at least about 4 calls per pick-and-place structural. The observed rates are:
- H1: 6.5 (T04) and 10.5 (T08) calls per pick-and-place;
- H3: 5.5;
- astranomy: 31.5.

Stall detection needs at least 16 actions in a segment and less than 0.8 mm of weighted progress over a 12-step window (`agent_speedup_v2_user_check_fragments.py:43-45` and `:404-419`).

In H1 T08 the gripper pressed the ketchup bottle from action 37 to 83:
- commanded about 16–17 mm/step;
- measured 0.1–1.4 mm/step (ratio below 0.1 from step 40);
- the stall fired only at step 81.

A measured-state signal (commanded > 5 mm and measured/commanded < 0.1 for 3 consecutive steps), replayed offline over the saved traces:
- fires at step 42, 5 actions after onset;
- in the other transit100 runs it also fires 0–4 times per run without forbidden contact;
- in the drawer runs it fires 13–19 times on blocked articulated motions.

## 4. Dominant bottleneck per goal per harness

| harness | fast | collision avoidance | few attempts |
|---|---|---|---|
| H1 | Model calls: 71–73 % of control+plan; 13–21 exec calls at a median of 12–14 s; the first exec call is 31–32 s. Then per-step overhead (18–20 %). Motion is only 9 %. | No contact signal until a stall. The model's boxes never caused a detour. T08: a 47-action robot-vs-bottle press plus 2 held-pot-vs-bottle episodes. T04: obstacle-free, but 3 palm-vs-mug episodes. | Executive called at every fragment end (58 % after success); stalls 27 %; 2 zero-motion vetoes per run, including the first fragment in 2 of 2 runs. |
| H3 | The same model share, but per-call latency is about 2× H1 (25.6 / 30.1 s median), consistent with 2× reasoning tokens. The vision context adds about 3 k tokens and 1.2–1.3 s of synchronous vision per call. | Same controller as H1 and no automatic use of vision. 0 obstacle episodes in 2 of 2 runs, but 0 usable 3D centres, so this is not attributable to vision. T08: 2 pot–burner episodes. | Fewer calls (11 vs 13 and 21), but n = 1 and the plans differ. 1 route veto. |
| H2 (old revision) | No progress. Model 62 %; local perception and planning 35 % (`policy_compute_s` = 71.0); about 4.7 s of planning per rejected operation. | Untested (no motion). | 8 of 8 operations rejected locally; repeated near-identical operations; the current revision is untested. |
| astranomy | Model 92 %: 63 calls × 10.5 s mean. Motion 2 %. | Hit the can in 3 of 3 attempts (1 episode each); "can tipped" in 002 is visual only. | 63–87 calls per attempt; 1 infrastructure failure out of 2 launches; the collision-avoidance prompt variant failed after 87 calls and 1,137 s. |

## 5. Measurement gaps and invalid comparisons

1. **n = 1 per configuration with a stochastic model.**
   - Identical inputs gave different plans and ±25–60 % latency (§2).
   - The replicate pair (drawer 001 vs 003) gave:
     - calls 43 vs 37 (±7.5 %);
     - actions 2,364 vs 2,131 (±5 %);
     - control 1,249 vs 1,208 s (±1.7 %);
     - forbidden samples 4,715 vs 2,068 (±39 %);
     - pair-episodes 10 vs 5.

   Consequently none of the following is resolvable:
   - "transit100 vs transit20" (549 vs 1,276 actions; 14 vs 27 calls);
   - "H3 vs H1" (451 vs 305 s on T04, 501 vs 490 s on T08);
   - "transit100 is 2.43× faster than astranomy" (`reports/astranomy_vs_transit100_can_t04_20261104_001/comparison.json`).

   Each compares one draw of the plan distribution with another. The directions are mechanistically plausible (calls × latency), but the magnitudes are not estimates.
2. **Contact samples are not events** (§1b). Totals invert the obstacle ranking in two places:
   - uc7.1-T20 at 1,325 samples "looks safer" than H1 at 3,372, but it is the run that touched the can;
   - H3 T08 at 400 samples has 0 obstacle contact, while H1 T08 hit the bottle with the gripper for 47 actions.

   Palm-on-target contacts (the whole of H1 T04 and of row 5) are not in the user's forbidden set ("robot body or held object vs scene, obstacles, other objects"), yet they dominate the totals. The classifier branch, penetration, force and obstacle displacement are not recorded.
3. **Strict vs raw success.**
   - All 10 uc7.1-family and H3 successes, and astranomy 002, end at the first raw-success action with the last object still gripped (`first_success_action_index = n_actions − 1`; terminal checks fail only `not_held` and `at_rest`). The uc7.1-family "strict" success relies on an evaluator-run 38-action release and settle that is not counted in time. Astranomy 002 never had that diagnostic.
   - The astranomy collision-avoidance run is the reverse case: both mugs pass the strict physical checks, but raw LIBERO fails (white mug 42.0 mm from its target).
   - Time-to-success therefore omits the release (at least one more call, 10–30 s, if the policy must decide to release).
4. **Timing definitions differ.**
   - The official uc7.1 `task_completion_time_s = control_without_video_s` (`UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md` §8.1) excludes the initial plan call (14.3–52.2 s). H2 and astranomy have no separate plan, so their first call is inside control.
   - H3's model timer includes synchronous vision. `live.py:256-264` runs vision inside `build_payload`, inside `request_agent`: 14.7 / 13.2 s.
   - Astranomy's `model_pipeline_s` (682.9 s) exceeds its logged per-call sum (658.7 s).
   - H2's control time includes 12.2 s waiting on a withheld response.
   - Process wall includes video (0.124 s/step in the uc7.1 runner against 0.046 s/step in astranomy), the SAM/DA3 preload in H3 (24.6 / 23.5 s), and preparation (6–11 s outside the plan call).
   - Evaluator-only work runs inside the timed loop: the per-step image encoding used by trace and video, the contact monitor (about 26 checks per action), and `run_state` writes.
5. **Scene and configuration registration.**
   - Row 5, "uc7.1 default routing T04", is the **official layout with no can** (`result_summary.json` `layout = "official"`), but it was listed with the can-obstacle runs.
   - The stove-off run changes the task sentence.
   - The drawer task has no obstacle.
   - The custom obstacle scenes exist only for init 0 (`agent_speedup_v2_obstacle_harness_100mm/scripts/launch.py:22-24`). Their obstacles were hand-placed on the source-to-goal line (T08 `layout_manifest.json`: `fraction_source_pot_to_cook_region = 0.45`, `perpendicular_offset_m = 0`).
   - The transit cap is set only by an environment variable whose main-tree default is 0.02 (`src/gpt_field/action.py:20`, identical in H1). A launch without `run.sh` / `run.py` silently becomes transit20.
   - H3's launch passes `--vision-query can` / `bottle` (`runs/foundation_vision_t0{4,8}_*/run.py:48`). This is a per-task, privileged naming of the added obstacle, which the task sentence never mentions.
   - The H2 run executed an older revision. Six of 9 `foundation_harness/*.py` files differ between `source_at_launch` and the current tree (client, executor, feedback, motion, robot_geometry, types). For example, the executor now forwards structured `evidence` (current `executor.py:292-297`, against run-time `:282` `detail=str(exc)`).
6. **Infrastructure failures end episodes.**
   - Two of 14 launches in the window: astranomy 001 (HTTP 503 after 34 calls; the direct runner has no same-request retry) and drawer 002 (HTTP 401 `token_expired` at the plan call).
   - The uc7.1 client retries 503 and stream truncation, but not 520, `unknown_error` or deadline errors (`agent_speedup_v2_client.py:143-149`; the protocol is in §6 of `UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md`).
   - These must be counted separately from policy failures and relaunched, not mixed into success@1.
7. **Missing artefacts for comparison.**
   - The H3 runs lack `routing_metrics.json`.
   - No run records obstacle pose, contact force or classifier branch.
   - H2 has no per-operation timing summary; it had to be derived from `events.jsonl` timestamps.

## 6. Older verified defects (subset relevant to measurement)

- The state8 axis-angle sign still jumps near π. For example, astranomy 002 has `before_state[3] = -3.13963` and `observation/state[3] = +3.13991` one step later. The cause is that `PolicyObservation.from_libero` uses `quaternion_to_axisangle` (`observation.py:151`; the file is byte-identical in H1 and the main tree), which all three harnesses share.
- No metric in this report uses state8 rotation: path length uses xyz only (`evaluate_agent_speedup_v2.py`). Any future rotation-error metric must use quaternions.
- The mirrored-image and `sort_keys` defects were not re-checked here; they are outside this focus.

## 7. Findings

The structured findings are in the returned result. In short:

1. **M1:** n = 1 with plan sampling makes every pairwise comparison unresolved.
2. **M2:** sample totals invert collision rankings; replace them with classified episodes.
3. **M3:** timing schemas differ, and evaluator work runs inside the timed loop.
4. **M4:** success is declared while the object is still held; strict is not uniform.
5. **M5:** scene, configuration and privileged-input registration problems: an official layout mislabelled as an obstacle run, a changed sentence, an obstacle-name query, environment-variable configuration, and an untested H2 revision.
6. **M6:** infrastructure failures are mixed into attempts.
7. **B1:** model calls dominate; calls are structural (58 % after success); latency is driven by reasoning tokens.
8. **B2:** per-step image encode, decode and `run_state` overhead of 0.1–0.15 s/step.
9. **B3:** zero-motion vetoes and no detours.
10. **B4:** contact persists until a 12–16-step stall; a measured-state blocked signal would have stopped the bottle press within 5 steps.
11. **B5:** H3's vision context doubles call latency with zero usable geometry.
12. **B6:** H2 has a zero-progress rejection loop and an untested revision.
13. **P1:** no decision protocol exists; the proposed one is below.

## 8. Proposed evaluation protocol (decide between H1, H3, H2-current and the astranomy baseline)

### Frozen builds

- One git commit per harness.
- All settings in a hashed configuration file, with no environment-variable defaults. Transit cap, video, reasoning effort and vision queries are explicit and identical across tasks.
- No per-task flags. `--vision-query` is not allowed; queries must come from the task sentence or plan only.
- Any code, prompt or threshold change mid-sweep creates a new build ID and restarts that harness's sweep.

### Scenes (14), all frozen by reset hash before any harness sees them

- **10 unseen LIBERO-10 relayouts.** Regenerate them with the existing generators after freezing: `output/layouts/libero10_remaining_relayouts_20261105_001/prepare_scene.py` and `output/layouts/libero10_relayout_recovered_20261105/recover_states.py` show the procedure. The T00, T01 and T06 relayouts from 2026-11-05 were already used historically, so they are not held-out.
- **2 existing obstacle scenes** (T04+can, T08+bottle) as regression anchors.
- **2 auto-generated obstacle scenes**, for example T00 and T09. Place a non-task LIBERO asset on the BDDL source-to-goal centreline at a random fraction in [0.3, 0.7] with a random lateral offset of ±3 cm. This is the T08 `prepare_layout.py` procedure, generalised and seeded.
- Init 0 / seed 7 per scene. Stochasticity comes from the model, so repeats use the same reset.

### Gates and stages

| stage | content | episodes | model calls (est.) | serial wall (est.) |
|---|---|---|---|---|
| S0 offline | (a) re-score all historical traces with the episode taxonomy (the scripts below already do this); (b) per-step overhead benchmark with H1's zero-call replay, with and without video; (c) H2-current planning feasibility on saved observations (`runs/foundation_harness_t04_can_obstacle_001/analysis/reconstruct_rejections.py` style) plus saved H1 and H3 frames. H2 needs at least 80 % of operations planable to continue. | 0 | 0 | about 2 h engineering, minutes of rendering |
| S1 smoke | T04+can and T08+bottle × {H1, H3, H2 if S0 passes} × 1 | 6 | about 100 | about 1 h |
| S2 acceptance sweep | 14 scenes × harnesses passing S1 × 1 | 28–42 | 500–760 | 4–6 h |
| S3 confirm | the top 2 harnesses × 14 scenes × 2 more repeats (paired) | 56 | about 1,000 | about 7.5 h |
| optional baseline | astranomy on the 4 obstacle scenes × 1 | 4 | about 250 | about 1.2 h |
| **total** | | **about 95–108** | **about 1,650–1,900** | **about 13–15 h serial** (about 7 h at 2-way concurrency if the subscription allows) |

Per-episode estimates use the observed values:
- H1: about 18 calls and about 8 min wall without video;
- H3: 12 calls, about 9 min;
- astranomy: about 63 calls, about 13 min.

The input-token load is about 0.4–1.5 M per uc7.1-family episode (about 6 M for drawer-length tasks), so plan for subscription pacing and the existing pause-and-resume rule.

### Per-episode stop and censoring rules

The cap is an evaluation cap, not a policy veto. Episodes stop at raw success, or are censored at 900 s of policy time or 60 executive calls (5× the 3-minute target). Censored episodes enter the time distributions as censored, Kaplan–Meier style, rather than being dropped.

After raw success, every harness gets the same evaluator release-and-settle diagnostic (38 actions), and both raw and strict results are reported.

An infrastructure error (HTTP 401, 503 or 520, or a deadline) triggers one fresh relaunch in a new directory. It counts toward cost and launches-per-success, not toward the policy failure rate. Check token freshness before every launch.

### Metrics, all reported per harness with paired per-scene differences

| goal | metric |
|---|---|
| zero-shot | One frozen build, with no per-task inputs. Success on unseen relayouts and on generated obstacle scenes is reported separately from the anchors. |
| fast | Policy time from the first observation to raw success: plan, all calls, local compute, vision, motion, and policy-required per-step work. Video, the trace-image copy, the contact monitor and truth checks are excluded and measured separately. Report the median, IQR, P(≤ 180 s), P(≤ 300 s), and the decomposition into model, vision, motion and overhead. |
| collision avoidance | Per episode: obstacle-contact episodes; robot vs scene/other-object episodes; held-object vs scene episodes (split rim/out-of-region vs side impact via a recorded branch); palm-vs-target episodes. Also maximum episode duration, maximum penetration, and obstacle displacement/tilt (evaluator-only, start vs end pose). **Collision-free success** = strict success with 0 obstacle, robot-scene, other-object and held-object side-impact episodes. |
| few attempts | Calls per success; first-plan validity (plan accepted **and** first exec fragment executes at least 1 action); zero-motion vetoes per episode; calls after stalls; repeated-failure count (same operation and reason twice in a row); launches per success, with infrastructure failures shown separately. |

### Decision rules

- **Primary:** collision-free strict success@1 over 42 paired episodes (14 × 3).
  - The Wilson 95 % half-width near p = 0.7 is about ±13 pp, so differences below about 20 pp are ties.
  - Compare harnesses with a paired McNemar test or sign test over scenes.
- **Secondary:** the geometric-mean ratio of policy time over scenes where both succeeded, with a bootstrap 95 % CI over scenes. A harness is "faster" only if the CI excludes 1. The replicate pair suggests a within-configuration time CV of about 2–5 % and a call CV of about 10 %, so a 15 % effect is resolvable with 14 paired scenes.
- **Elimination after S2:** raw success below 50 %, or collision-free success more than 4 scenes below the best harness.
- **H2:** must execute at least 1 operation in S1 before any S2 run.

## Appendix: scripts and scratch outputs

All scripts are in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/`.

| script | inputs | outputs |
|---|---|---|
| `extract_traces.py` | traces | per-action contact deltas and per-call rows → `traces/*.json` |
| `contact_episodes.py` | `traces/` | `contact_episodes.json` |
| `build_table.py` | | `normalised_table.json`, `normalised_table.md` |
| `per_call2.py` | | per-call tokens and latency, plus the regression → `per_call_rows.json` |
| `fragments.py` | | `fragment_outcomes.json` |
| `residual_detector.py` | | offline blocked-motion signal check |
| `bench_png.py` | | PNG cost benchmark; run with the simulator interpreter, CPU only |
| — | | `contact_spans.md` |
| — | | `rejections.json` |
