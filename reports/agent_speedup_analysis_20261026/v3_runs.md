# agent_speedup_v3: where the time goes and why runs fail (r1-r11 + 2026-10-26 baseline)

Read-only analysis of run artifacts under `/workspace/gpt_field/runs`. Nothing under `/workspace` was modified; no simulator, model or HTTP calls were made. Helper scripts (stdlib only, run with `python3 -I`) and intermediate outputs are in
`/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/scripts/` and `.../scratchpad/analysis/` (`calls.json` = per-call extraction of 376 request files, i.e. 374 valid responses + 2 HTTP errors, `run_table.json`, `tables.md`, `evidence_acq.txt`).

## 0. Method and definitions

* **Per-call latency.** For v3 runs I use `response.v3_transport_diagnostics.elapsed_s` (the bounded transport elapsed). uc7 has no such field, so I use `result.budget.elapsed_s - request.budget_elapsed_s_before` (budget interval). Per t1_008, the two differ by <0.1 s per call (`time_component_analysis.json` timeline). `model_pipeline_s` (from `summary.json`) is the authoritative per-run total; the transport sum is slightly lower (t1_008: 1310.0 s transport vs 1336.5 s pipeline).
* **Request size.** `request_bytes` is the size of the archived `.request.json` file and includes base64 images (12 images ≈ 4 MB), so it says little. Text size = instructions + schema + all `input_text` + replayed assistant `output_text` characters. Token counts come from server `usage`.
* **Trigger of a call** = `context.last_feedback.summary.{action,status,reason}` in the request's last user message (the reason control came back to the model). My classification (`scripts/run_table.py:classify`):
  * *natural completion*: `fragment_complete`, `adaptive_local_segment_complete`, or a model-requested `observation_required` segment that `succeeded`;
  * *physical stall/budget*: `segment_stall`, `segment_budget_exhausted`;
  * *geometry gate*: any reason containing `geometry` (`geometry_observation_horizon`, `geometry_predicted_block`, `geometry_prefix_observed`, `program_geometry_*`, `program_local_recovery_requires_geometry`);
  * *repeat guard*: `rejected_repeat / unchanged_blocker_requires_new_evidence_or_intent`;
  * *evidence follow-up*: `acquire_evidence / fresh_view_requires_interpretation`.
* **Decision types** are the `type` field of the accepted JSON response. I count `program/execute/motion` as *progress* and `geometry_patch/acquire_evidence/continue_intent/observe/escalate` as *geometry/evidence*. This is a coarse split, because many `execute` decisions were themselves re-plans around a gate.
* **Revision ≠ run number.** r1 t1_001, r2 t1_002, r3 t1_003, r4 t1_004, r4.1 t1_005, r5 t1_006+t1_007, r6 t1_008, r7 t1_009, r8 t1_010+t7_001+g5_001, r9 g5_002, r10 t1_011, r11 t1_012 (from `agent_SpeedUp_v3.md` and the runs' `prompt_version`/client version). The 2026-10-26 baseline (`runs/v3_completion_baseline_20261026T133633Z`) runs an isolated copy of r6 plus a completion endpoint (`prompt_version agent-speedup-v3-prompt-6+completion-1`).
* **Success** means raw LIBERO success. Strict `v2_success` is false for every v3 success and for uc7 t1_001.

## 1. Per-run numbers

Total service = `performance.json.completion_target.total_service_elapsed_s` where it exists (r6+). For r1-r5 and uc7 it is approximated as `review.json.preparation.elapsed_s + summary.elapsed_s` and marked `~`. For the baseline attempts it is `attempt_NNN/result.json.timing.T_service_s`, which also counts 284.8 s of automated Codex review for attempt 3.

| run | rev | task/ep | steps | calls | model_s | lat p50/p90/max | 1st call s | transition_s | sim_s | video_s | prep_s | total service s | outcome / terminal |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| uc7 t1_001 | uc7 | L10-t1/ep0 | 897 | 13 | 182.5 | 13.8/17.7/18.7 | 17.8 | 6.4 | 108.4 | 81.8 | 21.4 | ~437.7 | success (raw) |
| t1_001 | r1 | L10-t1/ep0 | 772 | 79 | 1512.7 | 17.8/27.9/42.3 | 29.7 | 26.6 | 93.7 | 77.9 | 925.0* | ~2664.2 | success (raw) |
| t1_002 | r2 | L10-t1/ep0 | 1568 | 41 | 784.8 | 17.5/26.4/42.8 | 29.9 | 68.4 | 196.9 | 170.7 | 25.3 | ~1310.4 | success (raw) |
| t1_003 | r3 | L10-t1/ep0 | 1531 | 42 (+1 err) | 898.1 | 17.6/32.5/47.8 | 46.5 | 115.4 | 184.7 | 147.0 | 24.2 | ~1446.9 | HTTP 500 at request 43 |
| t1_004 | r4 | L10-t1/ep0 | 4 | 1 (+1 err) | 32.4 | 32.3 | 32.3 | 0.2 | 0.6 | 0.5 | 24.5 | ~73.3 | HTTP 503 at request 2 |
| t1_005 | r4.1 | L10-t1/ep0 | 210 | 25 | 484.8 | 18.8/33.7/41.2 | 41.2 | 13.4 | 25.5 | 22.4 | 1682.1* | ~2261.8 | model escalate: occluded basket/milk boundaries of lower-hand sweep |
| t1_006 | r5 | L10-t1/ep0 | 90 | 5 | 138.9 | 25.4/35.5/41.9 | 41.9 | 19.0 | 11.4 | 11.4 | 22.5 | ~218.5 | TypeError in view proposal (null uncertainty), action 90 |
| t1_007 | r5 | L10-t1/ep0 | 880 | 52 | 1399.0 | 22.4/46.8/57.2 | 38.5 | 170.6 | 104.7 | 110.1 | 24.7 | ~1901.9 | success (raw) |
| **t1_008** | **r6** | L10-t1/ep0 | 873 | 40 | 1336.5 | 27.3/63.0/119.5 | 119.5 | 97.8 | 110.1 | 37.5 | 20.3 | 1626.2 | success (raw); last v3 success |
| t1_009 | r7 | L10-t1/ep0 | 743 | 25 | 721.4 | 22.2/59.5/79.2 | 63.0 | 58.5 | 88.0 | 24.9 | 24.6 | 936.0 | model escalate: stale B payload coverage, no view (release provenance erased) |
| t1_010 | r8 | L10-t1/ep0 | 190 | 7 | 358.3 | 44.2/90.4/96.0 | 96.0 | 103.8 | 22.7 | 6.6 | 26.0 | 535.9 | model escalate: fingertip/palm coverage, no view |
| t7_001 | r8 | L10-t7/ep0 | 176 | 8 | 385.1 | 40.7/84.9/96.3 | 80.0 | 20.8 | 21.6 | 5.4 | 17.6 | 463.2 | model escalate: palm/finger coverage near can, no view |
| g5_001 | r8 | goal-g5/ep0 | 21 | 5 | 152.9 | 24.9/55.3/68.3 | 68.3 | 3.8 | 2.5 | 0.6 | 29.8 | 201.2 | model escalate: false plate payload |
| g5_002 | r9 | goal-g5/ep0 | 71 | 6 | 162.1 | 21.8/41.8/56.9 | 56.9 | 25.4 | 7.9 | 2.2 | 32.8 | 242.4 | model escalate: palm x- coverage, patch rejected, no view |
| t1_011 | r10 | L10-t1/ep0 | 194 | 3 | 135.9 | 9.9/94.9/116.2 | 116.2 | 9.0 | 22.4 | 4.5 | 29.0 | 214.8 | runtime stop `targeted_view_unavailable: no_spatial_coverage_witness` |
| t1_012 | r11 | L10-t1/ep0 | 431 | 9 | 354.7 | 26.0/66.8/106.7 | 106.7 | 17.8 | 50.1 | 11.2 | 22.1 | 469.4 | runtime stop `targeted_view_unavailable: semantic_goal_requires_geometry_before_view_planning` (fresh_geometry:milk) |
| base att1 | r6 copy | L10-t1/ep0 | 0 | 0 exec (1 prep) | 0.1 | - | - | 0 | 0 | 0 | - | 140.2 | harness ValueError (U+000A in request text) before 1st execution request |
| base att2 | r6 copy | L10-t1/ep1 | 0 | 0 | 0 | - | - | - | - | - | - | 0.75 | prepare ValueError: no frozen reset SHA for ep1 |
| base att3 | r6 copy | L10-t1/ep2 | 313 | 13 | 581.8 | 33.8/60.3/137.4 | 137.4 | 44.9 | 36.4 | 9.3 | (review 284.8) | 1007.4 | model escalate: A target envelope + 4 fingertip boundaries, no view |

\* Preparation transport failures: `agent_speedup_v3_t1_review_001` request 1 failed after 900.8 s and request 2 then succeeded (`result.json`). `agent_speedup_v3_t1_review_005` request 1 failed after 900.6 s (`stream_missing_completion`), and request 2 took ~772 s with 76,222 output tokens (`review.json.preparation.model_budget`). Every other v3 preparation took 17.6-32.8 s.

Request/response size (the means are over valid calls):

| run | rev | in_tok mean / max | text chars mean / max | req MB mean / max | imgs max | out_tok mean (reasoning) | out chars mean / first call | cached in_tok |
|---|---|---|---|---|---:|---|---|---:|
| uc7 t1_001 | uc7 | 41,616 / 75,354 | 146,571 / 269,597 | 3.34 / 4.19 | 12 | 368 (109) | 986 / 853 | 16,128 of 541,005 |
| t1_001 | r1 | 264,207 / 495,731 | 965,706 / 1,832,385 | 4.86 / 6.38 | 12 | 370 (93) | 976 / 1,190 | 155,904 |
| t1_002 | r2 | 140,070 / 264,809 | 513,207 / 978,614 | 4.26 / 5.09 | 12 | 505 (90) | 1,387 / 1,506 | 46,080 |
| t1_003 | r3 | 136,219 / 262,172 | 517,922 / 999,191 | 4.28 / 5.53 | 12 | 648 (142) | 1,678 / 3,673 | 67,200 |
| t1_005 | r4.1 | 95,814 / 181,541 | 357,811 / 678,557 | 4.18 / 4.81 | 12 | 552 (182) | 1,340 / 3,811 | 23,680 |
| t1_007 | r5 | 233,517 / 455,108 | 876,770 / 1,709,752 | 4.67 / 5.98 | 12 | 585 (145) | 1,547 / 2,651 | 115,456 |
| t1_008 | r6 | 213,075 / 397,117 | 791,071 / 1,480,055 | 4.53 / 5.91 | 12 | 728 (220) | 1,778 / 8,480 | 143,872 of 8,523,018 |
| t1_009 | r7 | 147,853 / 270,162 | 547,966 / 1,005,279 | 4.20 / 5.21 | 12 | 721 (227) | 1,747 / 6,140 | 38,400 |
| t1_010 | r8 | 43,419 / 76,976 | 158,331 / 277,768 | 2.96 / 4.56 | 12 | 1,247 (567) | 2,339 / 4,657 | 4,992 |
| t7_001 | r8 | 45,715 / 82,478 | 167,429 / 298,379 | 3.01 / 4.34 | 12 | 1,297 (542) | 2,640 / 3,864 | 31,104 |
| g5_001 | r8 | 29,755 / 48,750 | 113,809 / 180,945 | 2.09 / 3.44 | 10 | 854 (232) | 2,194 / 4,928 | 15,360 |
| g5_002 | r9 | 34,464 / 58,013 | 130,698 / 213,934 | 2.37 / 3.98 | 12 | 778 (188) | 2,107 / 4,530 | 9,984 |
| t1_011 | r10 | 15,737 / 21,646 | 56,810 / 73,643 | 2.36 / 3.13 | 6 | 1,367 (520) | 2,836 / 6,328 | 1,280 |
| t1_012 | r11 | 27,747 / 43,836 | 90,089 / 138,195 | 3.90 / 5.05 | 12 | 1,052 (478) | 2,080 / 5,801 | 8,576 |
| base att3 | r6 copy | 91,500 / 153,039 | 340,345 / 568,928 | 3.35 / 4.22 | 12 | 943 (264) | 2,384 / 11,866 | 12,288 |

Fixed prompt parts (first request of each run): instructions grew from 10,060 chars (uc7) to 15,476 (r1), 25,722 (r6) and 27,359 (r7), then fell to 18,572 (r8/r9) and 10,569 (r10/r11). The schema grew from 2,140 chars (uc7) to 4,878 (r1), 7,645 (r6) and 10,121 (r8+). All runs use the same `gpt-6-astra`, `reasoning.effort=high`, `stream=true`, timeout 180 s (first request payloads of uc7 t1_001, t1_008 and t1_012). Prompt caching is negligible: 1.7% of input tokens were cached in t1_008 and 3.0% in uc7.

## 2. Q1: why v3 r6 (t1_008) is 3.7-4.8x slower than uc7 on the same t1 reset

Same model and settings, same reset (`initial_reset_check.sha256 e92ca405...` in both), and similar step counts (897 vs 873) and path lengths (2.72 m vs 2.40 m, from `scripts/step_motion.py`).

**Ratios.** Preparation + evaluator elapsed is 437.7 → 1622.2 s (×3.71). Control is 407.2 → 1593.6 s (×3.91). Control without video is 325.4 → 1556.1 s (×4.78).

**Decomposition of the +1186.5 s control difference** (`summary.json.timing_components` of both runs):

| component | uc7 t1_001 | v3 t1_008 | Δ s | share of Δ |
|---|---:|---:|---:|---:|
| model_pipeline | 182.5 | 1336.5 | +1154.0 | 97.3% |
| observe_transition (geometry + feedback) | 6.4 | 97.8 | +91.4 | 7.7% |
| policy_compute | 0.9 | 8.8 | +7.9 | 0.7% |
| simulator_step | 108.4 | 110.1 | +1.7 | 0.1% |
| video_render | 81.8 | 37.5 | −44.3 | −3.7% |
| run_state | 25.1 | 0.2 | −24.9 | −2.1% |
| residual | 2.1 | 2.7 | +0.6 | 0.1% |

Non-model control time is about equal: 224.6 s for uc7 vs 257.1 s for v3. r6's I/O savings (video −44 s, run_state −25 s) were cancelled by +91 s of geometry/feedback work.

**Model time = calls × per-call latency.** Calls went 13 → 40 (+27) and the mean went 14.04 → 33.41 s/call (`model_pipeline_s/api_calls`).
* Symmetric (Shapley) split: the extra calls cost **+640.6 s** and the higher latency **+513.4 s**.
* Sequential bounds: the extra calls alone cost +379 s at uc7 latency or +902 s at v3 latency. 40 calls at uc7 latency would total 561.6 s; 13 calls at v3 latency would total 434.4 s.
* The first call alone (119.5 s vs 17.8 s) contributes +101.7 s.

**Why each call is slower (inference from an observational regression).** Over 374 valid calls from 17 runs, OLS gives latency ≈ 4.38 s + 0.0180 s per 1k input tokens + 0.0203 s per visible output token + 0.0510 s per reasoning token (R² = 0.931; `scripts/latency_regression.py`). Coefficients are stable when first calls (n=357) or uc7 are excluded. Applied to the means (input 41.6k → 213.1k tokens, visible output 259 → 508 tokens, reasoning 109 → 221 tokens), the 18.8 s/call gap splits into:
* +3.1 s from input,
* +5.1 s from visible output,
* +5.7 s from reasoning,
* +5.0 s unexplained (different days, 2026-10-21 vs 2026-10-25; schema/prompt differences).

So longer outputs and more reasoning matter more than the larger context.
* The visible output grew because every v3 response must carry `scene_update` and `motion_context`. In t1_008 these average 646 + 223 chars of a 1,778-char response; uc7 averages 986 chars with neither field.
* The first call's 8,480 chars split into `scene_update` 4,372 + `program` 3,173.
* The only per-event transport log (baseline attempt 3, `attempts/attempt_003/request_diagnostics.jsonl`) shows the same pattern. Call 1 spent 66.5 s before the first visible character and then 70.9 s streaming 11,866 chars. Calls 2-13 spent 7.2-46.1 s (median 20.3 s) before visible output and then 9.4-17.3 s (median 13.3 s) streaming 1,180-2,031 chars, about 87-144 chars/s (call 1: 167 chars/s). Response headers arrived after 1.5-6.1 s, so queueing is not the issue.

**Where the extra calls come from (t1_008 trigger → decision).**

| trigger (reason control returned) | decisions taken | n | transport s |
|---|---|---:|---:|
| initial | program | 1 | 119.5 |
| `geometry_observation_horizon` (11 execute + 1 direct_motion) | 9 geometry_patch, 3 execute | 12 | 316.8 |
| `geometry_predicted_block` | 5 execute, 4 geometry_patch, 1 motion, 1 observe | 11 | 280.1 |
| `geometry_prefix_observed` | 5 geometry_patch, 3 execute | 8 | 260.0 |
| `program_geometry_prefix_observed` | 2 geometry_patch (program resumes) | 2 | 49.5 |
| `program_geometry_predicted_block` | 1 execute (**program replacement**, step 149) | 1 | 66.8 |
| repeat guard `unchanged_blocker_requires_new_evidence_or_intent` | 1 geometry_patch, 1 execute | 2 | 73.2 |
| `fragment_complete` (natural) | 3 execute | 3 | 144.0 |

* **36 of 39 non-initial calls (92%) were gate/guard handoffs, costing 1046.5 s (80% of 1310.0 s transport).** Only 3 handoffs were natural. uc7 had 0 gate handoffs: its 12 non-initial calls were 8 `fragment_complete` and 4 `segment_stall`.
* By decision type, t1_008 made **21 `geometry_patch` decisions (568.4 s, 43% of model transport)**, none of which exist in uc7. It also made 19 progress decisions (1 program, 16 execute, 1 motion, 1 observe) against uc7's 13. So the +27 calls = 21 geometry patches + 6 extra progress decisions, and 12 of the 16 executes were themselves gate-triggered.
* The decision context before the 39 non-initial calls listed these blockers (`current_decision.blocker.reason`): `unobserved_or_model_uncovered_sweep` 14, `geometry_inventory_required` 7, `estimated_obstacle_overlap` 6, `escape_estimated_initial_overlap` 6, `payload_geometry_unknown` 1, `bounded_model_geometry_support` (supported) 5.
* **Evidence acquisition:** 0 calls. The proposer offered candidates in only 4 of the 36 requests where it ran.
* **Program replacement:** 1 call. The 8-stage program (119.5 s to generate) covered actions 0-148 with 2 geometry-patch resumes. At step 148 it stopped on `program_geometry_predicted_block` and was replaced by ordinary fragments.
* **Granularity:** decisions bought a median of 4 steps each in t1_008 (geometry_patch median 4, execute median 5; 20 of 39 inter-call gaps ≤4 steps) against 63 steps per decision in uc7.
* **Release tail:** calls 36-40 (169.2 s) were spent on steps 867-871, one action each, while the open command was already active from step 867. Raw success arrived at step 872 (trace).
* **At the 300 s service mark** (approximate: prep + ~6.3 s startup + budget clock): uc7 was at step ~494-624 of 897, while t1_008 was at step ~149 of 873, i.e. the program had just been abandoned.

## 3. Q2: failing runs from r4.1 on

Of the 12 non-success runs from r4.1 on, 9 ended `evidence_acquisition_failed`. The other three are infrastructure failures: t1_006 (TypeError in `propose_evidence_views`), baseline att1 (U+000A rejected in request text) and att2 (missing reset SHA). Of the 9, 7 are model `escalate` decisions and 2 are runtime stops. Both stop paths are designed to be terminal:
* `escalate` → `_stop_for_evidence` → `raise EvidenceAcquisitionFailed` (`src/gpt_field/agent_speedup_v3.py:576-577`, `369-375`).
* `acquire_evidence` with no candidate (or ≥2 attempts) → terminal stop (`agent_speedup_v3.py:300-302`).
* The prompt itself says "If no useful admissible view or bounded recovery exists, escalate ... This yields evidence_acquisition_failed" and forbids "reduce uncertainty ... or enlarge free space merely to pass a check" (`src/gpt_field/agent_speedup_v3_client.py`, LEAN_RECOVERY / LEAN_GEOMETRY_INSTRUCTIONS).

Terminal requirement per run (parsed from the `EvidenceAcquisitionFailed` JSON in each `error_traceback.txt`; deficit = required margin minus available margin):

| run (rev) | what fired at the end | nominal margin / declared uncertainty / deficit | evidence plausibly obtainable from RGB? | recovery tried → result | decisions progress vs geometry/evidence (model s) |
|---|---|---|---|---|---|
| t1_005 (r4.1) | blocker supported, but the model escalated: occluded basket-side and milk-side boundaries of the whole lower-hand sweep (21-23% uncovered per `agent_SpeedUp_v3.md`). 13 of 24 handoffs were `geometry_observation_horizon` (median inter-call gap 4 steps). | n/a | No: the requirement is free-space coverage of occluded regions, and the design explicitly does not infer free space from visibility. | 7 geometry_patch (6 resumed), 1 continue_intent, 2 repeat-guard hits; no acquire_evidence (not available in r4.1) → escalate at step 210, 0 closes | 16 (296 s) vs 9 (204 s) |
| t1_009 (r7) | `sweep_coverage:B:0:groceries_overhead_air:z:negative` and `A:0:cream_cheese_approach_air:x:negative`, `estimated_obstacle_overlap`, payload unknown | B −159 mm / 12 mm / 181 mm; A −15 mm / 7 mm / 31 mm | Release evidence existed (14 raw supported release samples) but was rejected: a model `scene_update` at request 16 erased release provenance (`gate_A_analysis.json`), so the stale B payload blocked A | 8 patches (7 resumed); 1 acquire_evidence → 15-step view move (actions 62-76) → patch; hold accepted for B at step 302 (the only accepted hold in any v3 run); 0 releases accepted → escalate at 743 | 15 (459 s) vs 10 (250 s) |
| t1_010 (r8) | 6 × `sweep_coverage` (fingers/pads z− below the `package_work_volume` free region; hand y+), `estimated_obstacle_overlap` | fingers −5 to −10 mm / 15 mm / 20-25 mm | No. The planned fingertip z (~0.420) is below the audited table top (0.4368), so the descent was physically infeasible because the OBBs were biased low; no view can certify it | 2 patches (resumed), 1 acquire_evidence → 26-step view move (116-141) → patch; 2 local retreats `alternative_not_supported` → escalate at 190, 0 closes | 3 (227 s) vs 4 (123 s) |
| t7_001 (r8) | `sweep_coverage:finger1_pad:A_upper_grasp_access:y:positive`, `hand:...:z:negative`; finger1 0.9 mm from `other_can` (20 mm combined uncertainty) | hand **+4.0 mm** / 7 mm / 3.3 mm; pad **+7.3 mm** / 7 mm / **0.04 mm** (interpolation bound) | No: deficits are 0.04-3.3 mm, below the ~1.2-1.6 mm/px external-camera footprint, and nominal clearance was positive | 2 patches, 2 programs; proposer gave no candidate in 6/6 requests → escalate at 176, 0 closes | 5 (292 s) vs 3 (89 s) |
| g5_001 (r8) | false plate payload from an empty-hand pushing posture (bug); `A:0` vs table z− | −26 mm (payload vs table) | No: it was a bookkeeping defect (fixed in r9) | 1 patch; proposer `payload_geometry_required_before_view_motion` → escalate at 21 | 3 (114 s) vs 2 (37 s) |
| g5_002 (r9) | `sweep_coverage:hand:plate_overhead_corridor:x:negative` (`single_region_fit_union_not_proven`) | **+6.4 mm** / 18 mm / 14.5 mm | No: the requirement is a free-region union proof, and nominal clearance was positive | 1 patch, rejected (0 continuation events); no candidate → escalate at 71 | 4 (125 s) vs 2 (36 s) |
| t1_011 (r10) | stall on the real table during descent (first finger-table contact at action 144, local retreat at 149 found no alternative) → acquire_evidence → `no_spatial_coverage_witness` | model table top 0.420 vs audited 0.4367 (16.7 mm, `t1_011_analysis.json`) | The informative evidence was the stall (proprioception), not a view | 1 continue_intent (same descent, stalled again), 1 acquire_evidence → terminal at 194, 0 closes | 1 (116 s) vs 2 (19 s) |
| t1_012 (r11) | `fresh_geometry:milk` (stale after possible contact; the audit shows real palm-milk contact from action 386) at `lift_clear` entry → acquire_evidence rejected because `current_target is None` | n/a | Plausibly yes (milk is large and visible; the model patched milk 3 times), but the runtime kept re-invalidating it and blocked view planning before goal derivation (`agent_speedup_v3.py:266-268`) | 4 patches (3 resumed), 1 continue_intent, 47-step model retreat, revised program; 12 close actions on A (419-430), hold not accepted; the earlier tracking-failure view focus was the table center [0,0,0.39] and all 8 candidates failed visibility → terminal at 431 | 3 (216 s) vs 6 (137 s) |
| base att3 (r6 copy, ep2) | `current_target_envelope:A` plus 4 fingertip `sweep_coverage` y± near `A_approach_air`; `geometry_inventory_required` | fingers **+0.5 to +1.6 mm** / 4 mm / 2.4-3.5 mm | No (mm-level deficits); finger-table contact recorded from action 195 | 7 patches (all resumed), 4 executes (withdraw + re-approach), 5 partial close steps; proposer candidates in 1/11 requests → escalate at 313 | 5 (295 s) vs 8 (280 s) |

**Totals over these 9 runs:**
* Triggers: 72 of 92 non-initial calls (78%) were gate/guard handoffs, 16 natural, 2 physical, 2 evidence follow-ups.
* Decisions: 55 progress (2,139 s) vs 46 geometry/evidence (1,175 s, 35% of model time).
* 6 of 9 runs never closed the gripper on an object (t1_005, t1_010, t7_001, t1_011, g5_001/002 never started the task contact; for g5 the closures were posture-only).

**Cross-run evidence facts:**
* **The view proposer almost never offers a move.** In runs r5+ where it was consulted (130 requests over 11 runs), it returned candidates in **11 (8.5%)**. Of 567 candidate rejections, **470 (82.9%) were predicted-visibility failures** and 97 were motion admissions (`scripts/evidence_acq.py`). Reasons:
  * By design it only translates the wrist camera in 4 lateral directions × 2 fractions (`src/gpt_field/agent_speedup_v3_observation.py:86,122-124,147-152`).
  * It rejects any ray occluded by model boxes inflated by their uncertainty, plus 20 mm if stale (`observation.py:27-32`; `agent_speedup_v3_geometry.py:817`).
  * The viewing motion itself must pass the same geometry admission (`observation.py:155-160`).
  * Executed evidence-view moves: 2 in all v3 runs (t1_009, t1_010); neither resolved the blocker.
* **Local metric correction never worked:** zero entities with `local_pose_revision > 0` in t1_008, t1_009, t1_010, t7_001, t1_012 and att3 (`agent_stats.json.shared_geometry`). Triangulation was `unavailable` for 13-19 entities per run. The stage analyses report the causes: `ambiguous_stereo_correspondence` (8 of 10 entities in gate B) and `insufficient_texture`.
* **The model's metric geometry is biased by more than its declared uncertainty.** In all 9 v3 t1 runs that declared a table support entity, the first table-top estimate was 16.8-41.8 mm *below* the audited table-contact height. The audit records z = 0.4367-0.4368 m in every t1 run, uc7 included (evaluator-only; `summary.json.collision` / `collision.json`). Declared uncertainty was only 8-15 mm (error/uncertainty 1.1-4.6×). The table was corrected only after physical contact: 0.400 → 0.432 at call 4 in t1_008 and 0.400 → 0.435 at call 5 in t1_012 (`scripts`, response `scene_update`).
* In 3 of the 5 witness-based terminal blocks (t7_001, g5_002, att3) **the nominal clearance was positive**; only uncertainty padding produced the deficit (0.04-14.5 mm).

## 4. Q3: what drove the r6 t1_008 success

* **Actions by primitive** (trace `executed` events): `semantic_program` 149 (actions 0-148, 17.1%), `execute_fragment` 712 (81.6%), `local_motion` 12 (1.4%).
* **The program never reached its gripper phase.** Its stage-1 grasp goal z was 0.4194 m (grip site; request 4 `last_feedback.stage_goal`), 17 mm below the audited table contact height. The first finger-table contact came at action 144, and at 148 the program ended `program_geometry_predicted_block` with a 22 mm position error. The model then revised the table (0.400 → 0.432) and replaced the program with ordinary fragments (request 4).
* **The visual interaction monitor recorded 0 observations** (`performance.json.policy_subcomponents.visual_interaction_observations = 0`). No accepted hold/release was recorded (`gate_results.json` baseline: `accepted_hold_operations []`). No local pose revision was accepted.
* The run's own `time_component_analysis.json` says: "Program was exercised through approach/local motion, then abandoned before its gripper phase ... do not describe it as tested successfully". `agent_SpeedUp_v3.md` says it "does not ... establish that this chain accelerated the task".
* **The same mechanism produced every v3 success:** model-authored `execute_fragment` control carried 99.5% (t1_001), 99.6% (t1_002), 99.3% (t1_007) and 81.6% (t1_008) of actions. This is the uc7 control mechanism, and uc7 succeeded on the identical reset with 13 calls and no geometry layer.
* **What v3's layers did contribute in t1_008:** they generated 36 of 39 handoffs and 21 geometry patches (above). They also coincided with far fewer recorded forbidden contact points: 251 vs uc7's 5,738 (`summary.json.collision.counts`). That contact reduction is the one measurable benefit, but it is n=1 per arm and noisy: v3 r2 t1_002 recorded 6,808. Contacts are not part of the success criterion.
* **Verdict (inference):** the success is attributable to the uc7-inherited fragment executive plus the model's after-contact correction of its own geometry, not to v3's new layers (program, interaction monitor, evidence acquisition, local stereo). The shared-geometry/sweep layer was active, and its measurable effect was +27 calls / +1154 s of model time relative to uc7.

## 5. Q4: share of calls triggered by gates vs natural boundaries, per revision

Non-initial valid calls. "Gate" = geometry gate + repeat guard (+ evidence follow-up where noted). Time share = transport time of gate/guard-triggered calls over all transport time of the run(s).

| rev | run(s) | non-initial calls | gate/guard | natural | physical stall/budget | evidence follow-up | gate time share |
|---|---|---:|---:|---:|---:|---:|---:|
| uc7 | t1_001 | 12 | **0 (0%)** | 8 | 4 | 0 | 0% |
| r1 | t1_001 | 78 | 67 (86%) | 11 | 0 | 0 | 81% |
| r2 | t1_002 | 40 | 26 (65%) | 10 | 4 | 0 | 58% |
| r3 | t1_003 | 41 | 28 (68%) | 6 | 7 | 0 | 62% |
| r4.1 | t1_005 | 24 | 19 (79%) | 5 | 0 | 0 | 77% |
| r5 | t1_006 + t1_007 | 55 | 39 (71%) | 16 | 0 | 0 | 54% / 65% |
| r6 | t1_008 | 39 | **36 (92%)** | 3 | 0 | 0 | 80% |
| r6 copy | base att3 | 12 | 11 (92%) | 1 | 0 | 0 | 66% |
| r7 | t1_009 | 24 | 17 (71%) | 6 | 0 | 1 | 53% |
| r8 | t1_010 + t7_001 + g5_001 | 17 | 12 (71%) | 3 | 1 | 1 | 49% (29% / 68% / 48%) |
| r9 | g5_002 | 5 | 4 (80%) | 0 | 1 | 0 | 54% |
| r10 | t1_011 | 2 | 2 (100%) | 0 | 0 | 0 | 14% (first call dominates) |
| r11 | t1_012 | 8 | 7 (88%) | 1 | 0 | 0 | 55% |

No v3 revision got the gate share below 65% of handoffs. r10's "coverage advisory, not veto" did lengthen inter-call gaps (median 97 steps in t1_011 and 39.5 in t1_012, vs 4 in t1_008), but the runs then hit physical stalls and evidence-interface stops after 3 and 9 calls.

## 6. Q5: run-level analysis files, key conclusions and checks

| file | key conclusion | check against raw artifacts |
|---|---|---|
| `agent_speedup_v3_t1_008/time_component_analysis.json` | routes program 1 / geometry_patch 21 / execute 16 / motion 1 / observe 1; mean model 33.41 s; model 85.9% of net control; program 149 actions, abandoned before gripper phase; 0 visual observations | ✓ all reproduced from response files, trace and `performance.json` (1336.48/40 = 33.41; 1336.48/1556.12 = 0.859). Its note that geometry has no separate timer is correct: `observe_transition_s` is not pure geometry. |
| `agent_speedup_v3_t1_007/retest_analysis.json` | 52 calls (execute 28 / geometry_patch 21 / observe 1 / motion 2); model 79.57% of control excluding video; settling check of 1 µm vs holding drift causes extra handoffs | ✓ counts and percentage (1398.97/1758.15). The settling claim was not re-checked. |
| `agent_speedup_v3_t1_003/retest_analysis.json` | ~2 mm commands give ~0.5 mm motion per step; 166 forbidden contact points | ✓ median 0.51 mm/step, 75% of steps <1 mm (`scripts/step_motion.py`); 166 ✓ |
| `agent_speedup_v3_t1_005/retest_analysis.json` | prep 1682.1 s with one `stream_missing_completion`; 25 responses; 0 closes; model escalation | ✓ `review.json` (request 1 failed at 900.6 s), trace gripper open 210 / close 0, terminal JSON |
| `agent_speedup_v3_staged_001/final_summary.json` + `gate_results.json/.csv` + `implementation_plan.json` | r7-r9: 0/5 success; total service 935.95 / 535.93 / 463.19 / 201.18 / 242.43 s; no accepted local pose updates; "insufficient swept-volume evidence and no supported informative recovery view repeatedly caused escalation"; dense coverage checks raise feedback cost; gate A 1 hold accepted, 0 release (14 rejections) | ✓ totals equal `performance.json`; 0 `local_pose_revision` ✓; gate A counts ✓ (`agent_stats.json`: hold_supported 1, release_supported 0, confirmation_rejections 14); t1_010 feedback 103.8 s / 190 actions = 0.546 s/action vs 0.079 in t1_009 ✓. "Escalation repeatedly due to no view" is confirmed and quantified here: 11/130 requests had candidates. |
| `agent_speedup_v3_local_execution_001/final_summary.json`, `results.csv`, `prompt_size_comparison.json` | r10/r11 fail at 214.8 / 469.4 s; prompt projection cuts the archived request from 213,934 to 91,777 chars (−57.1%) | ✓ totals; ✓ the "before" request = g5_002 request 6, my max text chars 213,934. The cut is an offline payload measurement. Live r10/r11 per-call latency was still 39-45 s mean because outputs were 1,052-1,367 tokens with 478-520 reasoning tokens (first calls 106.7-116.2 s), so **the compaction did not translate into lower latency**. |
| `.../t1_011_analysis.json` | finger-table contact from action 144; table estimate 0.420 vs contact 0.4367; defect: view proposer needs a coverage witness | ✓ contact action 144 and z 0.4367; ✓ first-call table top 0.42; ✓ terminal reason. The defect is real in the r10 frozen source; r11 added a fallback (diff of `agent_speedup_v3_observation.py`, current lines 51-82). |
| `.../t1_012_analysis.json` | focus = table center [0,0,0.39], 8/8 candidates failed visibility; terminal = semantic-goal prerequisite (`current_target is None`) | ✓ proposer record (visibility 8, motion 0); ✓ terminal JSON; ✓ `agent_speedup_v3.py:266-268` |
| baseline `report_zh.md` | 0/3; att3 T_control 688.653 s, model 581.832 s (84.49%); 1 program / 7 geometry_patch / 4 execute / 1 escalate; 188 program + 125 fragment actions; review 284.756 s; control started at 318.68 s | ✓ all (`result.json`, `service_clock.json`, request files, trace) |

One correction to the shared context: "every failure from r4.1 on ends evidence_acquisition_failed" holds for behavioral failures (9 runs). t1_006 (r5) ended `infrastructure_error`, and baseline att1/att2 ended in harness/preparation errors.

## 7. Ranked bottleneck / failure claims

1. **Gate-driven handoffs, not task phases, set the call count.** 36/39 non-initial calls in t1_008 (92%) and 72/92 across the 9 failing r4.1+ runs (78%) were triggered by geometry gates or the repeat guard. uc7 had 0/12. In t1_008 these calls cost 1046.5 s (80% of model transport), and decisions bought a median of 4 steps vs 63 in uc7.
2. **Per-call latency more than doubled because of output, not context.** t1_008 averaged 33.4 s/call vs uc7's 14.0 s. The regression attributes the 18.8 s gap to +5.1 s visible output (scene_update/motion_context/program), +5.7 s reasoning, +3.1 s input and 5.0 s unexplained. The first call (full scene + program) takes 57-137 s in r6+ (median 96 s) vs 17.8 s in uc7, i.e. 19-46% of a 300 s budget before the robot moves.
3. **The model's metric geometry is biased by more than its declared uncertainty, and the gates require mm-level consistency.** Table top was 16.8-41.8 mm low in 9/9 t1 runs (uncertainty 8-15 mm). Terminal deficits were 0.04-25 mm, and in 3/5 witness cases the nominal clearance was positive. The bias also causes the stalls: grasp targets below the real table in t1_008, t1_010 and t1_011.
4. **Evidence acquisition cannot supply the evidence the gates ask for, and asking for it is terminal.** Candidates appeared in 11/130 proposer calls (82.9% of rejections were visibility). Only 2 view moves were ever executed, and neither cleared a blocker. Both `escalate` and a candidate-less `acquire_evidence` raise `EvidenceAcquisitionFailed` (`agent_speedup_v3.py:300-302, 576-577`). This is the end of 9/9 behavioral failures r4.1+.
5. **v3's new layers did not produce any success.** All 4 v3 successes were ≥81.6% ordinary fragments. The program was abandoned at step 148 in the only success that used it. The interaction monitor and local stereo produced 0 accepted observations or pose updates in every success; the only accepted hold anywhere was gate A, and its release was then rejected by a provenance bug.
6. **Lifecycle/bookkeeping defects end runs that RGB evidence could have continued:** erased release provenance (t1_009, 14 supported samples rejected), false payload (g5_001), stale-neighbor re-invalidation plus a goal-before-view prerequisite (t1_012), a witness-only view focus (t1_011), and a null-uncertainty crash (t1_006).
7. **Geometry/feedback compute is a secondary cost:** +91.4 s in t1_008 vs uc7, and 103.8 s (20.8% of control, 0.546 s/action) in t1_010 from dense coverage refinement.
8. **Preparation transport tails are rare but large:** two of 14 v3 preparations lost ~900 s each to a failed stream (review_001, review_005). Today's baseline att3 also charged 284.8 s of automated review before control started at 318.7 s service.
9. **Even a perfect v3 r6 would miss S300 on t1 with synchronous calls (inference).** Non-model control time was 257 s (sim 110 s for 873 steps) plus ~20 s prep, which leaves ~23 s for all model time against a 57-137 s first call. uc7's own non-model time was 224.6 s.

## 8. Unknowns and caveats

* Single episodes per configuration; no variance estimates. Server load differed by day (uc7 t1_001 ran 2026-10-21, t1_008 2026-10-25).
* The latency regression is observational. Images (12 per request after call 6 in both lines), reasoning replay items and schema complexity are not modeled.
* The per-event split into pre-visible vs streaming time exists only for baseline att3. uc7's split is unknown.
* The audited table height (0.4368 m) comes from evaluator contact points (consistent across 11 runs) and is used here only as a diagnostic. It was never policy input and must not become a constant.
* Whether removing the coverage/uncertainty gates would keep the t1_008 contact reduction (251 vs 5,738 points) is unknown, and contacts are not part of the success criterion.
* `observe_transition_s` mixes geometry with other feedback work; there is no per-method geometry timer.
* For r1-r5 total service is approximated as prep + elapsed. r1-r3 requests lack the `current_decision.blocker` field, so their blocker breakdown is unavailable.
