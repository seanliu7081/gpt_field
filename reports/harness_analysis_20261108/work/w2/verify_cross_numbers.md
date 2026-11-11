# verify_cross_numbers: adversarial check of `w1/cross_eval_methodology.md`

Date 2026-11-08. Read-only: nothing under `/workspace/gpt_field` was edited. No simulator, model or GPU job was run. I did not reuse any w1 script or data file. Every number below was recomputed from raw run artifacts (`trace.jsonl`, `trial/summary.json`, `requests/*.result.json`, `*.stream.json`, `physical_truth.json`, `collision` snapshots) with my own scripts in this directory (see the end of this file).

## Verdicts at a glance

| item | claim | verdict |
|---|---|---|
| (a) | normalised run table (§1a) | **confirmed with minor corrections.** Every status, call, action and timer value reproduces within rounding. H2's process wall is recoverable but missing. |
| (b) | model share 71–76 % for H1/H3 | **confirmed with corrections.** 71.5 / 73.3 / 73.6 / 75.8 %. The motion range is 2–11 % (not 2–10 %) and the overhead range is 6–28 % (not 6–22 %). |
| (c) | contact-episode decomposition | **confirmed.** All totals are exact, and episodes match when merged at the owner-pair level. The claim that totals "invert the ranking in two places" is **partly refuted**: the T08 example is not an inversion. |
| (d) | triggers 58 % / 27 % / 3 %, detour = 0 | **confirmed with a scope correction.** 108 / 51 / 6 of 186 are exact. `routing_metrics.json` exists in 8 runs, not 3, and all of them have 0 detours. |
| (e) | strict-at-terminal 0/10 | **confirmed.** The Summary's "0/11" is an internal inconsistency. |
| (f) | H2 ran an older revision (6/9 files differ) | **confirmed with corrections.** It is now 7/9 files changed, plus a new `surface.py` and a changed evaluator script. |
| (g) | latency independent of input tokens | **refuted.** The univariate fit reproduces exactly, but the multivariate fit on the same 186 calls gives +11 ± 4.6 ms per 1k input tokens (+18 ms with an H3 indicator, matching h1_speed_attempts' 17.9). |
| (h) | infrastructure failures (503, 401) | **confirmed.** The uc7.1 client would **not** have retried this particular 503 either. |
| extra | "per-step overhead unchanged since 2026-10-26" | **partly refuted.** It has grown from 0.23 to 0.27–0.40 s/step (both figures include video). |
| extra | H3 latency doubling "consistent with 2× reasoning" | **confirmed with corrections.** Reasoning explains about 2/3 of the excess. The rest is about 1.3 s of vision plus slower output streaming on the H3 run date (a confound). |
| §8 | evaluation protocol | **partly refuted.** The statistical-resolution claims are optimistic, the cost is underestimated by roughly 1.5–2×, and it is missing time interleaving, a policy release, taxonomy decisions and a multi-goal decision rule. |

---

## (a) Normalised run table: recomputed from raw artifacts

Sources:
- `runs/<run>/trial/summary.json`: `control_elapsed_s`, `control_without_video_s`, `timing_components`, `elapsed_s`.
- `runs/<run>/review/requests/*.result.json`: plan `budget.elapsed_s`.
- Per-call `latency_s` from `trial/trace.jsonl` `action.diagnostics.api_calls`.
- Action counts from `action` (uc7.1) or `step` (astranomy) trace rows.

My script is `v_extract_uc71.py`, with output in `v_uc71_table.json`.

| run | actions (trace rows) | exec calls (trace = result files) | Σ exec latency = model_pipeline_s | plan s | control w/o video | video | wall | matches report? |
|---|---|---|---|---|---|---|---|---|
| H1 T04 | 549 | 13 | 200.33 | 17.49 | 287.32 | 67.98 | 402.06 | yes |
| H3 T04 | 604 | 11 | 328.17 (vision 14.73 inside) | 18.75 | 432.42 | 59.67 | 569.75 | yes |
| uc7.1-T20 T04 | 1276 | 26 | 463.44 | 14.26 | 682.80 | 180.60 | 906.09 | yes |
| default routing T04 | 801 | 10 | 197.58 | 19.42 | 338.08 | 115.05 | 498.04 | yes |
| H1 T08 | 843 | 21 | 336.41 | 22.32 | 467.42 | 122.50 | 634.07 | yes |
| H3 T08 | 590 | 11 | 356.12 (vision 13.21 inside) | 36.43 | 464.08 | 64.46 | 614.06 | yes |
| stove-off T08 | 889 | 16 | 261.80 | 37.90 | 420.73 | 146.74 | 628.09 | yes |
| drawer 001 | 2364 | 42 | 781.61 | 52.16 | 1249.17 | 479.23 | 1804.17 | yes |
| drawer 003 | 2131 | 36 | 830.07 | 50.19 | 1208.44 | 395.84 | 1676.16 | yes |
| astranomy 001 | 161 | 34 (33 ok, 1 HTTP 503) | 273.99 | – | 303.21 | 7.47 | 322.01 | yes |
| astranomy 002 | 306 | 63 | 682.93 (logged sum 658.71) | – | 740.18 | 14.18 | 765.40 | yes |
| astranomy CA | 409 | 87 | 1059.21 | – | 1137.49 | 19.84 | 1168.90 | yes |
| H2 T04 | 19 | 11 counted + 1 withheld | 112.85 (+12.19 withheld) | – | 202.29 | 4.31 | **219.25 runner + 22.0 preload** (report: "–") | yes, except wall |

Mechanism checks:
- **The initial plan is outside the control timer.** `control_started` is set after reset, `policy.reset()` and the first recorder frame (`agent_speedup_v2_obstacle_harness_100mm/scripts/evaluate_agent_speedup_v2.py:1426`). `control_without_video_s = control_stopped − control_started − video_render_s` (`:1298-1300`). The plan is produced earlier, in the review phase (`review/requests`), so "control+plan" is the right sum.
- **H3's model timer includes vision.**
  - `request_agent` wraps `super().request_agent` (`src/gpt_field/foundation_vision/live.py:301-345`). That calls `build_payload` (`:274-281`), which calls `_get_snapshot`, where `vision_engine.process` is timed (`:256-264`). The run's `source_at_launch` copy of `live.py` is byte-identical to the current file.
  - `trial/foundation_vision/summary.json` gives `request_elapsed_s` = 328.098 (T04) and 356.034 (T08), equal to `model_pipeline_s`, with `vision_elapsed_s` = 14.734 and 13.208.
- **Default routing T04 is the official layout with no can.**
  - `result_summary.json` `layout: "official"`.
  - The trace `sim_status` has no BDDL or init override, and `bddl_sha256 = 0c6749b9…`, which is the `original_bddl_sha256` of the custom-can runs.
  - `tomato_sauce_obstacle_1` is absent.
  - I could not trace where it was "listed with the can-obstacle runs".
- **The H2 row** is reproduced from `trial/summary.json` `timing_components` (`policy_compute_s` 70.95) and `analysis.json` `timing` (withheld response 12.19 s).
  - Harness overhead: 202.29 − 125.04 − 0.95 − 70.95 = 5.36 s.
  - The median call (9.95 s) is over the 11 runner-accounted calls. Over all 12 it is 10.02 s.
  - Mean reasoning tokens per call are 27.9 over 12 calls, or 23.8 excluding the withheld call. The report gives 23.
  - "17.6 s per failed cycle" is the mean of the 11 model-to-model gaps, which is **17.50 s** and includes the 3 observe cycles, not only failed operations.
  - "4.7 s planning per rejected operation" reproduces as the mean `feedback → operation_interrupted` gap: **4.69 s** (`trial/foundation_harness/events.jsonl`).
- **Small rounding and truncation differences.** The per-call reasoning means are truncated rather than rounded (astranomy 002: 24.7 → "24"; H3 T08: 253.8 → "253").

## (b) Model share

| run | model / (control+plan) | motion share | overhead share |
|---|---|---|---|
| H1 T04 | 71.5 % | 9.0 % | 19.5 % |
| H1 T08 | 73.3 % | 8.6 % | 18.2 % |
| H3 T04 | 73.6 % (+3.3 % vision) | 6.7 % | 16.4 % |
| H3 T08 | 75.8 % (+2.6 % vision) | 5.9 % | 15.7 % |
| uc7.1-T20 T04 | 68.5 % | 9.2 % | 22.3 % |
| default routing T04 | 60.7 % | **11.2 %** | **28.1 %** |
| stove-off | 65.4 % | 9.7 % | **25.0 %** |
| drawer 001 / 003 | 64.1 / 69.9 % | 9.1 / 8.5 % | **26.8** / 21.6 % |
| astranomy 002 | 92.3 % | 2.1 % | 5.7 % |

The 71–76 % figure for H1 and H3 holds, and so does "model latency dominates". The ranges "motion 2–10 %" and "overhead 6–22 %" are wrong for 3 runs; they should be 2–11 % and 6–28 %.

Definitions to note:
- "Motion" is simulated time (actions × 0.05 s), not wall time.
- "Overhead" therefore contains the simulator's excess over real time. For H1 T04 this is 68.4 − 27.4 = 41 s of the 59.5 s.
- With process wall as the denominator, the model share is 54 % (H1 T04) and 58 % (H3 T04).

## (c) Contact decomposition

My script is `v_contacts.py`, with output in `v_contacts.json`. It differences the exact cumulative class counters (`sim_status.collision.counts.points_<class>`) and the per-pair records step by step.

`pairs_by_class` is truncated to the top 12 pairs per class (`agent_speedup_v2_contacts.py:346`). I found **zero** steps where the pair deltas fail to sum to the class deltas, so the truncation lost nothing in these runs.

| run | forbidden | obstacle | robot vs scene/other | held vs scene | palm vs goal object | owner-pair episodes (gap ≤ 6) | obstacle episodes | first forbidden |
|---|---|---|---|---|---|---|---|---|
| H1 T04 | 3372 | 0 | 0 | 0 | 3372 | 3 | 0 | 114 |
| H3 T04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| H2 T04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| T20 T04 | 1325 | 33 | 0 | 0 | 1292 | 2 | 1 | 120 |
| default routing T04 | 726 | n/a | 0 | 0 | 726 | 1 | n/a | 137 |
| astranomy 001 | 111 | 111 | 0 | 0 | 0 | 1 | 1 | 45 |
| astranomy 002 | 580 | 426 | 0 | 0 | 154 | 2 | 1 | 66 |
| astranomy CA | 634 | 634 | 0 | 0 | 0 | 1 | 1 | 126 |
| H1 T08 | 3826 | 3615 (finger1–ketchup 3504 + pot–ketchup 111) | 0 | 211 (pot–burner) | 0 | 5 | 3 | 37 |
| H3 T08 | 400 | 0 | 0 | 400 (pot–burner) | 0 | 2 | 0 | 277 |
| stove-off | 1650 | 0 | 0 | 1650 | 0 | 3 | 0 | 548 |
| drawer 001 | 4715 | n/a | 1109 | 697 | 2909 | 10 | n/a | 148 |
| drawer 003 | 2068 | n/a | 17 | 611 | 1440 | 5 | n/a | 223 |

Every number and span in §1b matches. The report's episode counts are at the **owner-pair** level; at the geom-pair level H1 T04 has 4, not 3.

The pot–burner ambiguity also checks out. In H3 T08 the same burner geom pairs add samples to *both* `manipulated_object_support` and `manipulated_object_scene` on 9 steps (277–282 and 299–301). The classifier branch at `agent_speedup_v2_contacts.py:228-239` (normal more than 35° from vertical at `:230-231`, or contact outside the region at `:233-236`) is not recorded.

The ranking claim needs correction:
- **T04 inverts.** All four obstacle-touching T04 runs (T20 1,325; astranomy 111, 580, 634) have fewer samples than H1 T04 (3,372, no obstacle contact).
- **T08 does not invert.** The report's second example (H3 T08 400 vs H1 T08 3,826) ranks *correctly*: the run with fewer samples also had no obstacle contact.

The scope of "the only robot-vs-obstacle collision" also needs a caveat. It is true for added obstacles, but drawer 001 has six robot-body-vs-static-cabinet episodes. These include `robot0_link6_collision`, which is the arm link, not the gripper, against `wooden_cabinet_1_g9`, a `fixture_static` part of the cabinet base (1173–1180 and 1227–1232). The narrative never mentions them.

## (d) Call triggers

My script is `v_triggers.py`. For each call I take the last feedback row before the calling action, plus the run's own `result_summary.json` `execution_accounting.direct_controller_steps`.

- 108 / 186 calls follow `succeeded|fragment_complete` (58.1 %).
- 51 follow `failed|segment_stall` (27.4 %).
- 4 follow `route_unavailable`, 1 follows budget exhaustion and 1 follows `observation_required`.
- 9 are the first call after the plan.

**Zero-motion vetoes: 6.** These are H1 T04 at steps 0 and 162, H1 T08 at 0 and 87, H3 T04 at 349, and T20 at 395. T20's step-794 hold is a *model observation hold*, not a veto.

The route vetoes are `route_start_blocked`. That error is raised when the grip site lies inside a model-supplied box after clearance expansion (`agent_speedup_v2_user_check_routing.py:79-106`, scope `model_supplied_boxes_and_grip_site_path`).

**Routing scope correction.** `routing_metrics.json` exists for H1 T04, H1 T08, T20, default routing, stove-off and drawer 001/002/003: 8 runs, not 3. All have `detour_stages = 0` and `planner_search_calls = 0`. `routing_unavailable_stages` ranges 0–2 over these 8 runs (the "1–2" in the report holds only for its 3 runs).

For H3, every routed fragment row in the trace has `straight_path_valid = True` (317 and 420 rows), so it also had no detours.

## (e) Strict at terminal

All 9 uc7.1-family and H3 successes have `physical_success_at_terminal = false`, failing only `not_held` and `at_rest` (`trial/physical_truth.json`). Astranomy 002 has `strict_conjunction = false` for the same reason.

All 10 successes end on their first raw-success action (n − 1), because the runner stops on `sim.status["success"]`. That gives **0/10**; the Summary's "0/11" is inconsistent with §1a.

Astranomy CA is the reverse case: `strict_conjunction = true` and raw false, with the white mug's `xy_dist_mm` 42.01. Its `env_contacts` at rest include `tomato_sauce_obstacle_1` g14 and g22, yet `other_supports = []`. So the strict check **passes while the placed mug leans on the obstacle**. This is a missed evaluator gap.

## (f) H2 revision

Hashes over `runs/foundation_harness_t04_can_obstacle_001/source_at_launch/src/gpt_field/foundation_harness/` compared with the current tree:
- **7 of the 9 launch files differ**: client, executor, feedback, motion, robot_geometry, types and **video**.
- `surface.py` is new.
- `scripts/evaluate_foundation_harness.py` also differs (b3463063… vs f443c86b…).
- The current tree equals the hash-verified keypose-2 snapshot file for file.

The report's "6/9" predates the last edits: `video.py`, `executor.py` and `motion.py` were modified at 23:58 UTC.

The executor citation also moved. Run time had `executor.py:282` `detail=str(exc)`. The structured `evidence` forwarding is now at `executor.py:285-288`, not 292-297.

`prep/implementation.json` reads `implementation_complete_awaiting_full_task_approval` with `full_task_started: false`.

## (g) Latency regression: reconciliation with h1_speed_attempts

My script is `v_latency.py`, with outputs in `v_percall.json` and `v_latency_fits.json`. Tokens come from differenced cumulative `budget` in `trial/requests/*.result.json`; latency from the trace `api_calls`. n = 186.

| model | R² | input coefficient |
|---|---|---|
| A: rsn + out (report) | 0.888 | – (8.98 + 0.0498·rsn + 0.0065·out, exact) |
| B: input only (report) | 0.009 | −0.0176 ± 0.0137 s/ktok |
| C: input + rsn + visible | 0.892 | **+0.0111 ± 0.0046 s/ktok** (t ≈ 2.4) |
| C + H3 indicator | 0.934 | **+0.0181** s/ktok; H3 calls +7.6 s/call |
| C on h1's 5 runs (128 calls) | 0.949 | +0.0183 (h1_speed_attempts: 0.0179 from stream timing) |

The univariate test is confounded. Input size is almost a call index (corr 0.995), and every episode's first call is slow, with high reasoning and the smallest input.

Under fit C, input/prefill accounts for:
- about 5 % of all model time;
- 2–8 % per run;
- 9–13 % in the drawer runs at h1's coefficient;
- about 1 s per call at 87k tokens and about 2.9–4.7 s per call at 260k.

Astranomy gives a clean confirmation. On its zero-reasoning calls (n = 152), latency = 5.15 + **0.067 s/ktok** (R² 0.57). Its so-called per-call "floor" rises from 6.4 to 10.1 s as context grows.

The report's "17–35 reasoning tokens" for astranomy are per-run means. Per-call values are 0–516, and 54/63 calls in astranomy 002 had 0 reasoning tokens.

**Conclusion.** Reasoning tokens (≈ 48 %) and the fixed per-call cost (≈ 38 %) are the main levers, but shorter history is a real secondary lever, contrary to the report.

### H3 latency excess

H3 calls have a residual of +4.9 / +7.7 s per call beyond what their tokens predict. Vision is only 1.2–1.3 s per call.

The stream phases (`requests/*.stream.json`, script `v_stream_phases.py`) locate the rest:
- H3's visible output streamed at **121–137 chars/s**, against 173–215 chars/s in all 7 other uc7.1-family runs.
- The reasoning phase per reasoning token is similar (73–84 ms vs 68–84 ms).
- H3 ran on 2026-11-05; all other uc7.1-family runs ran on 2026-11-02/03.

So "2× reasoning" explains about 2/3 of H3's +14.4 s per call. The remainder is about 1.3 s of vision plus about 2.4–3 s of slower streaming, most likely server load or date. Harness comparisons on latency are therefore confounded by run date.

## (h) Infrastructure failures

- **Astranomy 001.** `trial/error_traceback.txt` and `requests/000034-*.result.json` show `http_status 503`, `error_code "unknown_error"` after 34 requests (33 accepted), with a process wall of 322.0 s.
- **Drawer 002.** `initial_plan_failure_audit.json` shows 401 `token_expired` at the plan call, with a wrapper wall of 14.04 s.

The uc7.1 client retries a 503 **only** when `error_code == "server_is_overloaded"` (`agent_speedup_v2_client.py:910`), or on the stream/transport codes in `TRANSIENT_ERROR_CODES` (`:149`, `:912`). Astranomy 001's 503 `unknown_error` would therefore have ended an H1 or H3 episode too.

This supports the report's "relaunch, count separately" rule, and it also means the failure is not specific to the direct runner. Drawer 003 adds `authentication_verification.json`, which is evidence that the token-freshness check was adopted after 002.

## Other claims re-checked

| claim | result |
|---|---|
| Plan payloads identical within each scene, with different plans as outcomes | **Confirmed.** T04 `payload` sha a6c15554 and `request_key` 29c835d6 match for T20, H1 and H3. T08 c85a311e matches for H1 and H3. Only `budget_elapsed_s_before` differs. Approved-plan shas are 39fbd3, f77c06 and 46efc3. H3 T08 has `placement_goals: []` and stove part E `move: "fixed"`. |
| Drawer 001/003 replicate pair | **Confirmed.** `independent_repeat_precheck.json` reports `all_15_task_inputs_identical` and `all_324_runtime_sources_identical`. Same plan payload; 43/37 calls, 2364/2131 actions, 1249/1208 s, 4715/2068 samples. |
| 0.158 / 0.282 s/step non-model time; `simulator_step` 0.125 s/step; `run_state` 13.0 / 108.9 s | **Confirmed** (H1 T04 and drawer 001 `timing_components`). |
| "Per-step overhead is unchanged since 2026-10-26" | **Partly refuted.** The 10-26 figure of about 0.23 s/step included video (`reports/agent_speedup_analysis_20261026/README.md:44`). The current with-video non-model time is 0.27 s/step (H3 T04) to 0.40 s/step (drawer 001), and without video 0.155–0.198 s/step. It grows with episode length. |
| "Video 0.046–0.124 s/step" | **Corrected.** uc7.1 runner 0.099–0.203 s/step (drawer about 0.19–0.20), H2 0.227, astranomy 0.046. |
| PNG encode/decode citations and offline benchmark | **Code confirmed** (`observation.py:43-45`, `:114-115`; the encode runs in the simulator worker, `libero.py:952`). **The benchmark is environment-sensitive**: the same frame in `.venv` (zlib 1.3.2, host load about 30) gives 205 ms/step for 2× encode + base64, against the analyst's 68 ms in the simulator venv (zlib 2.3.3). Time it in situ instead. |
| Replay benchmark | **Confirmed.** `validation/results/combined_replay_005.json`: 549 actions, 0 live calls, `elapsed_s` 97.47, max action error 0. |
| Bottle press: commanded 16–17 mm/step vs measured < 1.4 mm/step from step 40; stall at step 81 | **Confirmed** (normalized 0.16–0.17 × 0.1 m). Stall constants are at `agent_speedup_v2_user_check_fragments.py:43-45` and the logic at `:404-419`. |
| Measured-state signal: fires at step 42; 0–4 false fires in other runs; 13–19 in drawer | **Confirmed with corrections.** My re-implementation fires at 42, but also 0–5 times in the other obstacle runs (H3 T08: 5) and 25 / 14 times in drawer 001 / 003. This makes it costly as a call trigger. |
| §5.5 citations | **Confirmed**: `launch.py:23-24` (init 0 only); `action.py:20` default 0.02, identical in H1; `run.py:48` `--vision-query can/bottle`; T08 manifest fraction 0.45, offset 0. |
| Obstacle pose never recorded | **Confirmed.** Only contact positions exist (`physical_truth.json`, `task_evaluation.json`). |

## Issues the analyst missed

1. **Run-date and server-throughput confound in latency** (see (g)). H3's output streamed about 40 % slower than every other run. Any cross-harness speed claim needs harnesses interleaved in time plus a fixed latency canary per block.
2. **The input-token effect is real** (11–18 ms/ktok; 67 ms/ktok on astranomy zero-reasoning calls). History compaction is a valid secondary lever.
3. **Robot arm-link collisions with the static cabinet in drawer 001** (`robot0_link6/link7` vs `wooden_cabinet_1_g9`, `fixture_static`). There is also a definitional problem. `robot_scene` absorbs any non-finger contact with articulated parts (`agent_speedup_v2_contacts.py:224-226`). For "close the drawer/microwave" (T03, T09), a palm push would be "forbidden". The protocol must decide whether that counts.
4. **`robot_nonfinger_manipulated_object` covers palm contact with *any* goal object** (`agent_speedup_v2_contacts.py:221-223`), such as plates or the other mug, not only the grasped target. In these runs all palm contacts were with the object being grasped. The protocol's "palm-vs-target is not forbidden" rule must split "current grasp target" from "other goal objects". It also needs the user's sign-off, because the evaluator itself classifies the contact as forbidden.
5. **Strict success accepts a placed object resting against the obstacle** (astranomy CA). "Collision-free success" must also check final-state contacts with obstacles and non-goal objects.
6. **The proposed caps would have censored existing anchors.** The 60-call cap catches astranomy 002 (63 calls, success). The 900 s policy-time cap catches both drawer runs (1,259 / 1,301 s). This affects both the interpretation and the cost.
7. **The uc7.1 client's retry does not cover a 503 with `unknown_error`.** The infrastructure-failure risk is the same across harnesses.

## Assessment of the §8 protocol

What is right:
- Frozen builds, no per-task flags, scenes frozen by reset hash.
- Infrastructure relaunch counted separately.
- Classified contact episodes.
- Censoring instead of dropping.
- An identical release-and-settle diagnostic for every harness.
- Paired per-scene analysis.

Not sufficient as written:

- **Statistical resolution is overstated.**
  - Paired McNemar exact test on 42 episodes (`v_power.py`): a 20 pp gap (0.7 vs 0.5) is detected only **26–38 %** of the time; 30 pp (0.8 vs 0.5) **62–77 %**.
  - "Differences below 20 pp are ties" therefore implies more resolution than exists.
  - S2 single draws (n = 14) detect at most 9–23 %. The "4 scenes below the best" elimination rule wrongly eliminates a truly equal harness roughly 10–20 % of the time.
  - The time CV (2–5 %) comes from one replicate pair. Plan latencies on identical requests already differ by about ±13 %, and date effects are about 40 % on the streaming rate. "A 15 % effect is resolvable" is unsupported.
- **Obstacle coverage is thin.**
  - Only 4 of 14 scenes have an on-path obstacle, and 2 of those are development anchors. The held-out obstacle evidence is 2 scenes × 3 = 6 episodes per harness.
  - A better design generates a seeded on-path obstacle variant for every relayout (the generator already exists) and checks feasibility with an oracle path.
- **No time interleaving or latency normalisation.** Run harnesses back-to-back per scene in randomised order, and add a canary request per block. Record stream phases (headers, pre-visible, visible chars/s), which the artifacts already support.
- **Release is still evaluator-assisted.** Either stop at "raw success and the object not held for N steps" (the policy must release, and the time counts), or report time-to-release separately. Otherwise a harness that places and holds faster is credited for work it did not do.
- **Taxonomy decisions are needed before the sweep:**
  - palm vs current target, against palm vs other goal objects;
  - intended palm contact with the articulated target part;
  - final resting contact with an obstacle.
- **No decision rule across the four goals.** Pre-register a lexicographic or Pareto rule: zero-shot gate, then collision-free strict success, then time, then calls.
- **The cost estimate is optimistic.**
  - The per-episode figures (H1 8 min / 18 calls, H3 9 min / 12 calls) come from successful two-object episodes.
  - Censored failures run to 900 s policy time (about 16–21 min wall) and up to 60 calls.
  - Articulated tasks (T03, T09) and video (0.1–0.2 s/step) add time, as does per-run preparation and audit (several minutes per run here).
  - At 60–80 % success I estimate **about 20–27 h serial and about 2,500–3,000 calls** for about 95 episodes, against the stated 13 h and 1,700.
  - The "7 h at 2-way concurrency" assumes free GPU capacity. SAM/DA3 run on `cuda:0` for H2/H3 (`run.py:48`), the simulator renders with EGL, and the GPUs are currently busy.
- **Unnecessary.**
  - The optional astranomy baseline (4 × 63–87 calls, about 15–28 M input tokens) does not help choose among H1, H3 and H2.
  - The S1 smoke for H1 and H3 duplicates the S2 anchors. Keep S1 only for H2-current.
  - S0(c)'s 80 % "planable on saved observations" gate is an arbitrary threshold on other policies' viewpoints. "H2 executes ≥ 1 operation in S1" is simpler and already present.

## Scripts (all in this directory)

`v_extract_uc71.py`, `v_contacts.py`, `v_triggers.py`, `v_latency.py`, `v_stream_phases.py`, `v_blocked_signal.py`, `v_png_bench.py`, `v_power.py`. Outputs: `v_uc71_table.json`, `v_contacts.json`, `v_triggers.json`, `v_percall.json`, `v_latency_fits.json`.
