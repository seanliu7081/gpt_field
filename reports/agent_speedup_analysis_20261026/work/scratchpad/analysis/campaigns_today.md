# Today's three Agent SpeedUp campaigns (2026-10-26): plan, what ran, what it showed

Read-only synthesis. Every number below comes from a cited artifact. "Inference" marks my own reasoning, and "unknown" marks things no artifact establishes. All times are UTC.

- Campaign start and end are the first and last file mtimes in each campaign directory.
- Monotonic service clocks were converted to UTC with an offset calibrated on two evidence files: UTC seconds-of-day = monotonic − 646548.212. The offset agrees within 0.01 s for `trial_001` and `trial_004` (`evaluations/*/service_clock.json` `finished_monotonic` vs the mtime of `evidence_result.json`). Campaign 3 has the same `boot_id` (`899d7468-…`).
- Path roots: `C1 = /workspace/gpt_field/results/speedup_evidence_20261026T055123Z`, `C2 = /workspace/gpt_field/runs/speedup_next_20261026T101438Z`, `C3 = /workspace/gpt_field/runs/v3_completion_baseline_20261026T133633Z`.

## 0. Bottom line

- **Total effort.** Three campaigns used 14,451.5 s, about 4.01 h, of wall-clock:
  - C1: 05:51:23 → 07:48:51
  - C2: 10:14:38 → 11:03:06
  - C3: 13:36:33 → 14:51:30
- **Outcome.** There were **0 S300 successes** and **0 autonomous strict completions**. There were 2 raw LIBERO successes, both in C1 on t1 episode 1, at 614.0 s and 648.7 s full service. The one other executed episode (C3, t1 episode 2) failed with a model `escalate` after 313 actions.
- **Episode slots.** **8 episode slots were consumed and only 3 executed any policy action**, for a total of 1,849 + 313 = 2,162 actions. The other 5 slots ended before control:
  - 1 missing credential store
  - 2 initial-plan streams that ran for 900 s without completing
  - 1 newline rejected by the request serializer
  - 1 missing frozen reset SHA
- **Wall-clock allocation**, all three campaigns:

  | Share of wall-clock | Seconds | Activity |
  |---|---:|---|
  | 8.1% | 1,168.5 | Live robot-execution windows. The model was being waited on during 948.7 s of these windows. |
  | 14.7% | 2,125.9 | Model requests outside robot execution. 2,025.3 s of this was four plan streams that never completed. |
  | 18.8% | 2,710.2 | Waiting for the user's review authorization. |
  | 3.6% | 518.3 | Codex automated plan review. |
  | 54.9% | 7,928.5 | Protocol, implementation, tests, freezes, audits, amendments, reports and gaps. |

- **Main blocker found today: model-request completion.** 4 of the 8 v7 (user-check) initial-plan requests sent today never reached `response.completed`.
  - In C2's two cases the server returned HTTP 200 and text deltas kept arriving, carrying 27,853 and 141,110 public characters. Plans that completed carried 922–924.
  - C1's two cases stayed pending for 893 s, but C1 recorded no stream telemetry, so whether data was flowing there is unknown.
  - The subscription transport strips `max_output_tokens` and applies no client-side cap on output or total time. Only the outer budget stops a stream.
  - This alone stopped C1 and failed C2's service gate.
- **Where control ran, the model dominated.**
  - v7, t1 episode 1: model pipeline 252.3 s (arm A) and 279.5 s (arm B) per episode.
  - v3 r6, t1 episode 2: 581.8 s, which is 84.5% of control time.
  - Even with Codex review time removed, the best run (`trial_003`) is 614.0 − 138.5 = 475.6 s, still over 300 s.
- **Not tested at all today:** generality. Only LIBERO-10 t1 ran. t5 and t7 were preregistered in C1 but never started. None of the campaign fixes or infrastructure exists in the main working tree (see §6).

---

## 1. The plan that drove them: `/workspace/gpt_field/Agent_SpeedUp_Codex_Implementation_Plan_ZH.md`

The plan is dated 2026-10-26 (mtime 05:47 UTC). Codex wrote it. It describes itself as a plan to be executed when the user asks Codex to.

**Core question.** In a fixed observation and task scope, which local capabilities can reliably take over expensive model decisions and raise full-task success within 300 s?

**Metric.** S300 = raw LIBERO success AND full service time ≤ 300 s, on a preregistered set.

**Time budget.** The first round uses a 900 s effective service budget on the same clock as `service_elapsed`. Human review wait is excluded. Automatic review compute is *included*.

**Phases.**
- P0: verify the repository and freeze definitions.
- P1: joint audit of perception, checkers, failure sources and timing. Runs in parallel with P2.
- P2: behavior-preserving light baseline (timing and I/O only).
- P3: implement one minimal candidate.
- P4: local mechanism validation.
- P5: paired full-task screen.
- P6: frozen held-out evaluation.

**Hard caps (plan §2).**
- ≤ 60 detailed audit windows.
- ≤ 40 new offline model queries.
- P4 ≤ 24 short rollouts at 180 s each.
- P5 ≤ 18 episodes (9 paired conditions); P6 ≤ 18 episodes.
- ≤ 4 infrastructure retries in total.
- ≤ 2 repair rounds per causal defect.
- One online candidate per execution.
- Stated upper bound: about 11.2 h of simulated service.

**P5 promotion gate (plan §8).**
- The candidate shows attributable P4 evidence and actually participates in the full task.
- raw900 is not worse than A.
- Then either S300 gains at least 2 paired conditions, or S300 does not drop and the median full-service time falls by ≥ 20% over ≥ 5 common successes.
- Otherwise the result is `inconclusive`.

**Stop rules (plan §10).**
- A repeated failure with no new evidence means no more full episodes.
- Two failed repairs of the same defect freeze the candidate.
- Reaching a budget means stop.

**Scope.** The first round claims only reuse within the pick-place family. t1 episode 0 seed 7 is relabelled development-only.

---

## 2. Campaign 1 — `C1` (05:51:23 → 07:48:51, 7,047.1 s)

**What the user asked.** The campaign directory has no `user_request` file. The plan says it is executed when the user asks Codex to. The only recorded user utterance is the delegated-review authorization `"授权上述具体矩阵逐份代审（推荐）"` (`C1/delegated_review_authorization.json`, written 06:57:59). It covers:
- tasks t1, t5 and t7, seed 7;
- P5 episodes 1–3, arms A and B (18 units);
- P6 episodes 4–6, only if P5 passes;
- every plan inspected individually.

**Hypothesis** (`C1/candidate_spec.md`, type `pure_io`).
- Arm A: user-check v7, rebuilt from 300 frozen files and checked hash by hash, plus a common monotonic service clock and a 900 s budget.
- Arm B: A plus the existing v3 compact display recorder and throttled external status snapshots.
- Claim: removing display and serialization work the policy never consumes keeps requests, events and actions identical, and improves full-service performance in this bounded scope.
- Refuters: any change to policy input or actions; no cost reduction; missing the P5 gate.

**Audit basis** (`C1/audit_summary.md`).
- 229 records indexed and 40 windows audited; 46 of 60 windows after the online pair.
- Historical v7 t1_001 component costs: model 182.511 s, sim 108.384 s, video 81.766 s, live status writes 25.076 s, prep 21.406 s.
- Even deleting all video and status writes leaves about 321.720 s, so an I/O change alone could not reach S300.
- File mtimes show `candidate_spec.md` (05:53:47) was written before `audit_cases.jsonl` (05:58) and `audit_summary.md` (06:01:33). Inference: the pure-I/O candidate was registered before this campaign's P1 audit artifacts existed, possibly carried over from analysis done while writing the plan (plan mtime 05:47). The plan says the P3 candidate must be chosen from the audit results.

**Offline validation (P2/P3).**
- Frozen-response replay: 897/897 actions exact, and 13/13 request payloads identical to history and across A/B (`C1/baseline_validation.json`).
- 11 protocol tests and 11 scoring tests pass (`C1/regression_results.json`).
- Display-I/O benchmark, run in A1/B1/B2/A2 order on the 120-step t1_001 prefix (`C1/display_io_benchmark.json`):
  - A: 10.851 s and 10.526 s, 582 frames at 1600×960.
  - B: 2.705 s and 2.737 s, 122 frames at 640×384.
  - Median 10.689 → 2.721 s, a **74.55%** reduction.

**Live runs** (`C1/paired_results.csv`, `C1/live_audit.csv`, `C1/evaluations/*/service_clock.json`):

| trial | condition / arm | outcome | full service (s) | first raw (s) | steps / exec requests | detail |
|---|---|---|---:|---:|---|---|
| trial_001 | t1 ep1 A | `preparation_error` (MissingCredentials) | 7.567 | — | 0 / 0 | The isolated copy's default credential store was empty (`subscription_status.json`). |
| trial_002 | t1 ep1 A (retry of 001) | raw success | 648.734 | 645.774 | 830 / 11 | prep 29.1 s; Codex review 138.107 s; execution 479.8 s. Model 252.344 s, sim 101.798 s, video 79.135 s, run_state 24.579 s. |
| trial_003 | t1 ep1 B | raw success | 614.011 | 611.987 | 1019 / 13 | prep 24.3 s; review 138.456 s; execution 450.3 s. Model 279.538 s, sim 121.992 s, video 25.642 s, run_state 0.108 s. |
| trial_004 | t1 ep2 B | `evaluation_budget_exhausted` | 901.033 | — | 0 | One plan request pending 893.37 s; cancelled at 900.0004 s. |
| trial_005 | t1 ep2 A | `evaluation_budget_exhausted` | 900.953 | — | 0 | One plan request pending 891.93 s; cancelled at 900.0001 s. |

- **Strict quality at the raw-success instant.** Both raw successes **failed strict terminal quality** when they were truncated. The last-placed object (butter in A, cream cheese in B) was still held: `not_held`, `at_rest`, `supported_by_container_or_contents` and `not_perched_on_rim` were all false. A separate 38-action release/settle diagnostic made both strict-true afterwards; that is not autonomous completion (`C1/online_mechanism_attribution.json`).
- **Exploratory pair comparison, not causal.**
  - B − A full service: −34.72 s (−5.35%).
  - Display + status: 103.714 s (A) vs 25.750 s (B), a difference of −77.96 s.
  - Model: +27.19 s for B. Sim: +20.19 s for B.
  - The trajectories differ (830 vs 1019 actions), so the difference cannot be attributed to I/O.
- **Latency vs history.** Execution-request median was 20.05 s (A) and 16.69 s (B), with maxima of 40.33 s and 52.77 s. Historical v7 t1_001 had p50 13.825 s and p90 17.734 s (`audit_summary.md`). Why today was slower is unknown. Inference: the two arms ran concurrently against the same subscription.
- **Schedule deviation (inference from timestamps).** `paired_schedule_protocol.json` fixed overlapping arms, but pair 2 ran sequentially: `trial_004` 07:09:42.9 → 07:24:43.9, then `trial_005` 07:25:09.8 → 07:40:10.7.
- **Gates.** `p5_gate.json` records 2 of 9 pairs complete. On the primary ledger, A raw900 is 0/2 and B is 1/2; S300 is 0/2 for both; there are 0 common successes. Status `inconclusive`. P6 was not started (`heldout_results.csv`: 18 rows `not_started`).
- **Budget used.**
  - Audit windows 46/60; offline perception queries 0/40; P4 0/24 (not applicable to pure I/O).
  - P5 originals 4/18; P6 0/18; infrastructure retries 1/4.
  - Totals: 28 model requests (4 prep + 24 execution); 1,849 policy actions plus 76 diagnostic actions.
  - Effective service 3,072.298 s; human wait 5,486.579 s (summed per arm); automatic review 276.563 s (`C1/execution_manifest.json`).
- **Conclusion** (`C1/final_report_zh.md`, `C1/online_stop.json` at 07:41:16): stopped under plan §10 because two preparation timeouts matched with no new testable explanation.
  - `narrow_scope`: keep the verified I/O branch as an engineering change, not the default path.
  - `freeze`: online expansion (full v3 recovery, default triangulation, composite programs, async pre-planning).
  - `inconclusive`: S300, full-task speed, autonomous quality and generality.
- **Recommended next step (report):** independently locate the completion latency of the preparation stream, then re-plan the evaluation under a new protocol. Not started automatically.

---

## 3. Campaign 2 — `C2` (10:14:38 → 11:03:06, 2,907.4 s)

**What the user asked** (`C2/objective.md`). Decide whether one reliable local control mechanism can replace an expensive model handoff and give a net speedup. Do not extend v3 and do not resume P5/P6.
- Arms: new A = frozen v7 + the verified compact I/O; B = A + exactly one local candidate.
- Stage 1, bounded diagnosis of the preparation request:
  - ≤ 6 real requests with a 120 s absolute deadline each.
  - The first 4 interleave success-source and timeout-source inputs. The last 2 may only validate an evidence-backed fix.
  - Robot gate: both input classes complete at least once, and the latest two requests complete.
- Stage 2, a common endpoint for A and B: no truncation at raw success, autonomous `finish`, independent quality evaluation, and no automatic release.
- Stage 3, observability first:
  - ≤ 12 historical windows, ≤ 6 offline perception queries.
  - ≤ 8 development short operations, ≤ 2 repairs.
  - Each short operation: T_control ≤ 180 s, T_service ≤ 420 s.
- Stage 4: 6 preregistered pairs (2 normal + 4 trigger), 12 short operations.
- Promotion requires all of:
  - B ≥ 5/6 strict autonomous completions;
  - 0 new regressions;
  - ≥ 4 common successes;
  - ≥ 3 trigger chains with fewer calls and shorter T_control;
  - median T_control_B / T_control_A ≤ 0.90;
  - all added costs counted.

**Hypothesis** (`C2/mechanism_spec.md`): `bounded_visual_placement_continuation`.
- Trigger: a placement descent stalls while the model's open and retreat fragments are still pending.
- Mechanism: one bounded legal visual check confirms the object is inside the container, then the pending fragments run. This saves one model handoff; the historical v7 request 6 cost 10.148901 s.
- Cost gate for a 10% gain: `T_control_A ≤ 101.489010 − 10c` s, where c is the added cost.

**What ran.**
- 4 real plan requests (`C2/request_results.json`, `C2/request_diagnostics.jsonl`).
- 0 simulator starts and 0 robot actions.
- 12/12 historical windows reviewed; 0/6 offline perception queries.
- 52 offline tests + 2 subtests pass (`C2/requirements_audit.json`).

| request (UTC start) | source input | outcome | total (s) | first body chunk (s) | public chars / deltas |
|---|---|---|---:|---:|---|
| prep_001 (10:29:38.9) | trial_002 (completed in 19.8 s in C1) | 120 s deadline, no `response.completed` | 120.004 | 2.83 | 141,110 / 6,493 |
| prep_002 (10:32:15.0) | trial_004 (hung 893 s in C1) | completed; valid plan | 17.308 | 1.40 | 922 / 294 |
| prep_003 (10:34:13.9) | trial_002 | 120 s deadline | 120.003 | 1.86 | 27,853 / 7,115 |
| prep_004 (10:37:01.9) | trial_004 | completed | 11.215 | 1.44 | 924 / 293 |

The two saved payloads are identical except for their images (`C2/decision.md` §1). Which input stalled flipped between C1 and C2. Inference: the stall is not tied to a particular input, though the sample is tiny.

**Additional analysis I ran on `C2/request_diagnostics.jsonl` (inference about a possible fix).**
- The completed plans stopped at 922 and 924 characters after about 6–7 s of deltas, at about 3.1 characters per delta.
- The two non-completing streams passed 4,000 characters at 25.26 s (prep_001) and 26.81 s (prep_003).
- prep_001 averaged 21.7 characters per delta, which is an abnormal delta size.
- A client-side cap at about 4× a normal plan would have cut each stalled stream at about 25–27 s. In practice they ran to 120 s here and 893 s in C1.

**Gates.**
- **Service gate failed.** Success-source inputs completed 0/2 and timeout-source inputs 2/2. The latest two requests did not both complete. Phase evidence was available. The 2 fix-validation slots were left unused because there was no evidence-backed fix.
- **Observability gate failed** (`C2/observability_review.md` and `C2/mechanism_spec.md`, written 10:27:02, before the probes).
  - End-effector tracking is measurable. Examples: 1.670 mm error on approach; 41.063 mm residual at the placement stall.
  - There is no validated feedback on the object, the container interior, or the post-release state. The key regions are occluded.
  - v7 feedback itself reports `grasp_success=unknown` and `contact_or_collision_cause=unknown`.
  - The review found the `held_objects` = `held_ever` trap (§6).
- **Stage 4 not entered.** `paired_results.csv` contains only the header row.

**Conclusion** (`C2/decision.md`).
- Pause online robot experiments in the current service configuration.
- Stop the local-takeover branch; its controller was never implemented.
- The net gain of local control is **unknown**, not 0.

**Recommended next step.** Find out why these frozen inputs produce "continuous public output but no completion event" under the same subscription configuration, and get one verifiable cause and fix. On the mechanism side, first build independently verifiable object/container-region and post-release feedback, tested on occlusion and edge negatives, before writing any controller.

---

## 4. Campaign 3 — `C3` (13:36:33 → 14:51:30, 4,497.0 s)

**What the user asked** (`C3/user_request.txt`): "implement and test it then give me results". The accepted proposal was:
- an isolated v3 r6 completion baseline;
- autonomous finishing;
- the known t1 first, then a small preregistered set of other t1 initial states;
- preserve the 300-second score;
- a separate, larger bounded completion experiment.

**Hypothesis and protocol** (`C3/protocol.json`).
- Hypothesis: frozen r6 plus an autonomous completion endpoint forms a completion-first baseline.
  - r6 = `agent-speedup-v3-6`; source `runs/agent_speedup_v3_t1_008/frozen_runtime`, 317 files.
  - Endpoint: the first raw event is recorded but does not truncate; the model must explicitly `finish`; terminal truth is independent; the evaluator never releases, lifts or settles.
- Qualification: 3/3 runs with autonomous finish, terminal raw success and terminal strict quality, all within 1800 s, with no replacement samples.
- Budget: 3 attempts (t1 episodes 0, 1 and 2, seed 7); 1800 s service each; 180 s per request; ≤ 6 simulator starts; 0 hidden retries.
- Stop rule: a correctness defect freezes the campaign with no repair resit. This rule was amended twice after results were seen (`C3/repair_amendment.json` at 13:58:30; `C3/enrollment_amendment.json` at 14:13:26).

**What ran** (`C3/summary.json`, `C3/attempts/*/result.json`, `C3/budget_ledger.jsonl`):

| slot | episode | outcome | T_service (s) | T_control (s) | actions | HTTP requests |
|---|---|---|---:|---:|---:|---|
| 1 | ep0 | `infrastructure_error`: ValueError. The multi-line `NEW_FINISH` completion rule was rejected by `_safe_strings` before the first execution HTTP request. | 140.170 | 11.268 | 0 | 1 plan request (26.7 s) |
| 2 | ep1 | `failed`: prepare ValueError, "no frozen reset sha256 for libero_10_t1 episode 1". | 0.751 | 0 | 0 | 0 |
| 3 | ep2 | `evidence_acquisition_failed`: model `escalate`. | 1007.386 | 688.653 | 313 | 1 plan request (9.1 s) + 13 execution requests |

**Attempt 3 in detail** (`C3/evidence/execution_attribution.json`, `C3/attempts/attempt_003/service_clock.json`).

Timing:
- Enrollment took 7.9 s; the draft was ready at 25.9 s.
- Codex review took **284.8 s** (approved at 310.7 s). It included failed operator commands: the system `python3` lacked numpy, a premature run launcher was rejected, and an edit of an immutable wrist selector was rejected.
- **Control started at service 318.7 s, so S300 was already impossible before any manipulation.**
- Model pipeline: 581.8 s, which is **84.5% of control**.
- The 13 execution requests took 578.4 s in total (median 34.0 s).
  - The longest was the first: 137.5 s. It produced a 16-stage program of 11,866 public characters, with about 66 s before any text.
- Other components: sim 36.4 s; `observe_transition` 44.9 s; video 9.3 s.

Behavior:
- 188 semantic-program actions ran first. Only stage 0 (approach) of the 16 stages completed. 125 ordinary-fragment actions followed.
- The 13 decisions were 1 program, 7 `geometry_patch`, 4 `execute` and 1 `escalate`.
- Close commands ran at action indices 308–312 (5 steps, the last actions of the episode). The closure never completed and `grasp_verified` stayed false.
- The independent post-hoc contact audit shows:
  - right finger to table contact from action 195 (368 robot–scene contact points);
  - **0 finger–object contact points** across 8,139 checks;
  - empty held and manipulated object lists.
- Inference: the gripper never touched the cream-cheese box. The model's height estimates for the box top and the table were wrong by at least its own predicted 8 mm fingertip clearance.

Terminal strict quality is **unknown**, because the base evaluator skipped terminal truth (§6).

**Totals.**
- 3/3 slots used.
- 15 HTTP requests, plus 1 failure during request construction before any HTTP was sent.
- 5/6 simulator starts.
- 313 actions; 1,148.306 s of service; 699.92 s of control nested within it (`C3/evidence/final_independent_audit.json`).

**Conclusion** (`C3/report_zh.md`).
- No autonomous v3 baseline was established: 0 raw successes, 0 autonomous strict completions, 0 S300.
- Freeze the claim that "r6 + autonomous endpoint gives a repeatable completion baseline".
- Because of the two amendments, the original 3/3 same-build criterion could never have passed.

**Recommended next step**, only if new budget is approved:
- Narrow the work to one chain: legal visual evidence for the first grasp → action correction → grasp confirmation.
- First fix the independent terminal-audit omission.
- Do not loosen geometry or closure thresholds, and do not add blind retries.

---

## 5. Established facts vs unknowns

### Established today, with numbers

1. **Outcomes.** S300 was 0 of 8 episode slots. Autonomous strict completions were 0 of 3 executed episodes. Raw successes were 2 of 3 executed: `C1/trial_002` and `C1/trial_003`, both t1 episode 1, at full service 648.734 s and 614.011 s.
2. **Strict quality at raw success.** Both raw successes failed strict quality at that instant because the last object was still held. The four failing checks were identical in both (`C1/paired_results.csv`).
3. **v7 plan-request reliability.**
   - 4 of 8 v7 plan requests did not complete: `C1/trial_004`, `C1/trial_005` (pending 893.37 s and 891.93 s) and `C2/prep_001`, `C2/prep_003` (cut at 120 s).
   - All 37 execution requests completed (24 in C1, 13 in C3), as did both v3 plan requests (26.7 s and 9.1 s).
4. **Display I/O.** The compact recorder cut display cost by 74.55% on a 120-step prefix (10.689 → 2.721 s). Online, display + status took 103.714 s (A) vs 25.750 s (B) per episode. The whole-episode difference was −5.35%, confounded by different trajectories.
5. **Replay equivalence.** The rebuilt v7 baseline is behavior-equivalent on saved inputs: 897/897 actions and 13/13 payloads.
6. **Where control time goes.**
   - v7, t1 episode 1, model pipeline: 252.344 s (A) and 279.538 s (B) out of 466.856 s and 438.206 s of control.
   - Sim: 0.123 s per step (101.798 s for 830 steps) and 0.120 s per step (121.992 s for 1,019 steps).
   - v3 r6, t1 episode 2: model 581.8 s out of 688.7 s of control (84.5%).
7. **Codex automated review is charged to service time:** 138.107 s and 138.456 s in C1; 93.8 s and 284.8 s in C3. Without review time, the two C1 successes would be 510.6 s and 475.6 s — still over 300 s.
8. **v3 r6 contact (C3 attempt 3).** Never contacted the target object: 0 finger–object contact points. The finger contacted the table from action 195.
9. **Truth skip in historical v3 failures.** Every historical v3 run that ended `evidence_acquisition_failed` and that I checked has its terminal truth skipped with "status infrastructure_error (simulator state not trusted)". That is 8 runs (t1_005, t1_009, t1_010, t1_011, t1_012, g5_001, g5_002, t7_001; field `summary.json.physical_truth.skipped`), plus C3 attempt 3. The r5 run t1_006 (`infrastructure_error`: TypeError) also has its truth skipped, by the same status filter.

### Unknown

1. **Why plan streams fail to complete.** Server-side and client-side causes are not separated. The stream content was never persisted. It is also unknown whether the subscription route accepts any output bound: `max_output_tokens` is deliberately stripped.
2. **Uncapped autonomous finishing (v7).** It is unknown whether v7 would finish strictly if not truncated at raw success. C2's common endpoint was validated only offline with a fake simulator, and no live v7 episode ran with it.
3. **v3 r6 + finish endpoint on t1 episode 0.** Unknown. This is where r6 historically succeeded (t1_008), but attempt 1 never sent an execution request.
4. **Strict terminal quality** of C3 attempt 3 and of the 8 historical v3 `evidence_acquisition_failed` runs.
5. **C1 pair 2 initial state.** Cross-arm initial-state equality is unverified: `trial_004` did not persist its SHA (`C1/launches/trial_005/pair_reset_annotation.json`).
6. **Perception accuracy.** Metric localization error, feasible-grasp-region accuracy and uncertainty coverage were never measured (`C1/audit_summary.md`, `C2/observability_review.md`).
7. **Latency cause.** Whether today's slower v7 request latency (median 16.7–20.1 s vs 13.8 s historically) came from running two arms concurrently.
8. **Other tasks.** Any behavior on t5 or t7: 0 episodes today. Cross-task generality is untested.
9. **Real cancellation latency** of C2's endpoint on a LIBERO worker (`C2/requirements_audit.json` limitations).

---

## 6. Harness and evaluator defects found today

| # | Defect | Evidence (path:line) | Effect today | Fixed? |
|---|---|---|---|---|
| D1 | **The subscription route has no output or total-time bound.** `max_output_tokens` is in `UNSUPPORTED_FIELDS` and is popped from every payload. `read_completed_response` loops until `response.completed` with no length or time cap. `timeout_s=180` is a per-socket-operation urllib timeout, not a total deadline. | `/workspace/gpt_field/src/gpt_field/chatgpt_plan.py:37-41`, `:429-434` (build_payload pops), `:458` (`output_token_limit_sent: None`), `:315-342` (read loop), `:136-141` (`_open_public`: `build_opener(...).open(request, timeout=timeout_s)`, a socket timeout), `:345-373` (transport). The timed-out request payload has no `max_output_tokens` key (`C1/evaluations/trial_004/review/requests/000001-*.request.json`). `C1/stop_rule_review.json` `streaming_semantics`. | Two C1 slots burned 900 s each with 0 actions, and C1 stopped. C2's service gate failed. | **Not fixed in the main tree.** The C2 diagnostic wrapper (120 s) and the C3 isolated copy (`revisions/r2/source/src/gpt_field/agent_speedup_request_diagnostics.py`, 180 s absolute deadline) only bound the wait. No output-length guard exists anywhere. |
| D2 | **The base evaluator treats `EvidenceAcquisitionFailed` as `infrastructure_error`, then skips terminal truth.** The generic `except Exception` sets `infrastructure_error`. `finalize_truth` skips any status not in `TRUTH_STATUSES` (`OUTCOME_STATUSES` + `subscription_error`). The v3 runner relabels the status afterwards, but the truth snapshot is already gone. | `/workspace/gpt_field/scripts/evaluate_agent_speedup_v2.py:1535-1536`, `:141-143`, `:1124-1126`. `/workspace/gpt_field/src/gpt_field/agent_speedup_v3.py:26` (RuntimeError subclass). `/workspace/gpt_field/scripts/evaluate_agent_speedup_v3.py:283-289` (relabel). | C3 attempt 3 strict quality unknown. The same skip is present in 8 of 8 historical v3 `evidence_acquisition_failed` runs. | **Not fixed** (`C3/report_zh.md`). |
| D3 | **Newline in the completion rule.** The new `NEW_FINISH` text contains U+000A and is put into the policy task contract. The inherited `_safe_strings` rejects control characters. | `/workspace/gpt_field/src/gpt_field/agent_speedup_v2_client.py:518-531`. `C3/source/src/gpt_field/v3_completion_endpoint.py:552` (original) vs `C3/revisions/r2/source/src/gpt_field/v3_completion_endpoint.py:552` (`" ".join(NEW_FINISH.splitlines())`). | Attempt 1 (t1 episode 0) failed before the first execution request: 0 actions, 140.2 s charged. The 35 pre-live offline tests missed it. | **Fixed in the C3 isolated copy only** (repair round 1). |
| D4 | **Missing frozen initial-state SHA for t1 episode 1.** `resolve_expected_sha` fails closed when the registry has no entry and no `--expected-initial-observation-sha256` is passed. | `/workspace/gpt_field/scripts/evaluate_agent_speedup_v2.py:411-421`. `C3/attempts/attempt_002/result.json`. | Attempt 2 consumed with 0 requests and 0 simulator starts. Episode 1 never ran. | **Worked around** for attempt 3 by measuring the initial state in a separate enrollment step (SHA `a8429539…`, `C3/enrollment_amendment.json`). The registry itself is unchanged. |
| D5 | **`held_ever` used as a current-hold label.** The contact snapshot's `held_objects` is `sorted(self.held_ever)`: once both fingers ever touched an object, it stays listed after release. | `/workspace/gpt_field/src/gpt_field/agent_speedup_v2_contacts.py:284`, `:347`. Consumed as `generic_held` in `/workspace/gpt_field/scripts/agent_speedup_v2_replay.py:243`. `C2/observability_review.md` §2: still listed at step 353 after release. | An audit-side labeling trap; it does not reach the policy. | **Not fixed**; documented and excluded from labels. |
| D6 | **Isolated copy used its own empty credential store.** | `C1/evaluations/trial_001/infrastructure_annotation.json`; `C1/subscription_status.json`. | `trial_001` failed at 7.567 s; 1 of 4 infrastructure retries used. | **Fixed** with an explicit `--store-dir` in C1 launches. |
| D7 | **Evaluator stops at the first raw success** (`stop_on_success`), so autonomous finishing cannot be scored. | `C1/online_mechanism_attribution.json` (last object still held at the raw-success instant). `C3/source/src/gpt_field/v3_completion_endpoint.py:359-362` patches the validation. | Both C1 successes are strict-false at the raw-success instant. | **Fixed only in isolated copies:** C2 `source/src/gpt_field/speedup_next_endpoint.py` (offline only); C3 r2 copy. |
| D8 | **Pair-2 initial-state hash missing.** `trial_004` timed out before persisting the initial observation SHA. | `C1/launches/trial_005/pair_reset_annotation.json` | Cross-arm initial state for pair 2 unverified. | Not fixed; recorded as a limitation. |
| D9 | **Review-tool friction.** System `python3` lacked numpy; a premature launcher run was rejected; native ReviewStore cells are immutable, so a wrong wrist-cell selector (w25 for the basket) could not be corrected. | `C3/attempts/attempt_003/operator_command_failures.json`; `C3/attempts/attempt_003/individual_review.json`. | Inflated the review to 284.8 s, all charged to service. | Not fixed; no plan-revision capability was added. |
| D10 | **Minor Codex tooling bugs.** `KeyError: stage` in the summary tool; `NameError: Path` in the audit tool. | `C1/reporting_failure_history.json`; `C1/audit_build_attempts.json`. | None on results. | Fixed. |

**Cross-cutting.** None of today's harness pieces exists in the main working tree. That includes the request diagnostics/deadline module, the evidence clock, the completion endpoint and the enrollment launcher. `ls /workspace/gpt_field/src/gpt_field | grep -E "request_diagnostics|evidence_clock|completion_endpoint|speedup_next"` returns nothing. The user's dirty tree was preserved: the tracked modifications in `git status --short` are identical to `C1/initial_git_status.txt` and `C3/initial_git_status.txt`. C1's commits exist only in campaign-local git repositories (`C1/baseline/.git`, `C1/tooling/.git`). C2 and C3 have no git repository; their code lives in `source/` and `revisions/` copies plus `.patch` files.

---

## 7. Process cost

### Per-campaign wall-clock breakdown (first → last file mtime)

**C1 (7,047.1 s):**

| Segment | Time (UTC) | Seconds | Share |
|---|---|---:|---:|
| P0–P3 offline work | 05:51:23.7–06:08:47.5 | 1,043.8 | 14.8% |
| Preparations and launch fix | 06:08:47.5–06:12:49.4 | 241.9 | 3.4% |
| Waiting for user authorization (nothing running) | 06:12:49.4–06:57:59.7 | 2,710.2 | 38.5% |
| Codex review | 06:57:59.7–07:00:19.4 | 139.8 | 2.0% |
| Live execution of two overlapping episodes | 07:00:19.4–07:08:19.3 | 479.8 | 6.8% |
| Two stalled plan streams, sequential (901.0 s each) | 07:09:42.9–07:40:10.7 | 1,802.0 | 25.6% |
| Gaps | — | 109.5 | 1.6% |
| Stop review, reporting and delivery freeze | 07:40:10.7–07:48:50.8 | 520.1 | 7.4% |

Inside the live window, the union of model-request time was 370.3 s.

**C2 (2,907.4 s):**

| Segment | Time (UTC) | Seconds | Share |
|---|---|---:|---:|
| Setup, source review, mechanism spec, observability review, offline tests, freeze | 10:14:38–10:29:38.9 | 900.7 | 31.0% |
| Four probes, wall-clock (268.5 s of it in requests, 9.2% of the campaign; the rest gaps) | 10:29:38.9–10:37:13.1 | 454.2 | 15.6% |
| Analysis, endpoint protocol and tests, decision, requirements audit, freeze | 10:37:13.1–11:03:05.5 | 1,552.4 | 53.4% |

Robot execution: 0 s. Simulator starts: 0.

**C3 (4,497.0 s):**

| Segment | Time (UTC) | Seconds | Share |
|---|---|---:|---:|
| Implement endpoint and launcher, tests, protocol, reviews, freeze | 13:36:33–13:52:18.3 | 945.3 | 21.0% |
| Attempt 1 | 13:52:18.3–13:54:38.5 | 140.2 | 3.1% |
| Repair, amendment and repair freeze | 13:54:38.5–14:06:28.9 | 710.4 | 15.8% |
| Attempt 2 | 14:06:28.9–14:06:29.6 | 0.8 | 0.0% |
| Enrollment amendment and freeze | 14:06:29.6–14:21:47.7 | 918.1 | 20.4% |
| Attempt 3 before control: enrollment, preparation, 284.8 s Codex review | 14:21:47.7–14:27:06.4 | 318.7 | 7.1% |
| Attempt 3 control (578.4 s of it in model requests) | 14:27:06.4–14:38:35.0 | 688.7 | 15.3% |
| Attribution, report, final audit, manifest | 14:38:35.0–14:51:30 | 775.0 | 17.2% |

### Totals across all three campaigns

Total wall-clock: 14,451.5 s. Categories are mutually exclusive.

| Category | Seconds | Share of wall-clock |
|---|---:|---:|
| Live robot-execution windows. The model was being waited on during 948.7 s of these; summed `simulator_step_s` was 260.2 s across the 3 episodes. | 1,168.5 | 8.1% |
| Model requests outside robot execution (C1 1,821.5; C2 268.5; C3 35.9). Includes the four stalled streams, 2,025.3 s. | 2,125.9 | 14.7% |
| Waiting for the user's review authorization (C1) | 2,710.2 | 18.8% |
| Codex automated plan review (C1 139.8; C3 378.5) | 518.3 | 3.6% |
| Everything else: protocol design, implementation, offline tests, freezes, amendments, audits, reports, gaps, small preparation overheads | 7,928.5 | 54.9% |

- Model-request time overall was 3,074.6 s (21.3% of wall-clock), counting overlapping C1 requests once. Of that, **65.9% (2,025.3 s) was spent on four plan streams that produced nothing usable.**
- Calculation scripts: `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/timecalc.py` and `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/budget.py`.

### Live robot episodes executed today

**3 episodes with policy actions, 2,162 policy actions in total.**

| Episode | Steps | Raw result | Full service | Strict at terminal | S300 |
|---|---:|---|---:|---|---|
| C1 `trial_002` (A, t1 ep1) | 830 | success at 645.774 s | 648.734 s | false | false |
| C1 `trial_003` (B, t1 ep1) | 1,019 | success at 611.987 s | 614.011 s | false | false |
| C3 attempt 3 (r6, t1 ep2) | 313 | none (escalated) | 1,007.386 s | unknown (truth skipped) | false |

Episode slots consumed without any policy action: 5.
- C1 `trial_001`: credentials.
- C1 `trial_004` and `trial_005`: stalled plan streams.
- C3 attempt 1: newline; it did start the simulator and control.
- C3 attempt 2: SHA.

C2 started no simulator. The evaluator also ran 76 diagnostic actions in C1 (38 per success); these were not policy actions.

---

## 8. What the user asked for vs what was delivered

| Campaign | Ask | Delivered | Gap |
|---|---|---|---|
| C1 | Execute the plan P0–P6 to deliver "a smaller or more effective execution system that improves full-task performance within the time limit vs a unified baseline" (plan's closing paragraph). Explicit authorization for an 18 + 18 unit delegated-review matrix on t1, t5 and t7. | P0–P3 artifacts; a verified, equivalent pure-I/O branch (−74.55% display cost); 2 executed t1 episode-1 runs; complete accounting. | No S300 improvement. P5 covered 2 of 9 pairs and was `inconclusive`. P6 and t5/t7 were never run. The user's authorization wait was 45 min of the 1.96 h campaign. |
| C2 | Decide whether one local mechanism can replace a model handoff with net speedup. Bounded budgets. Deliverables: `protocol.json`, `request_diagnostics.jsonl`, `mechanism_cases.jsonl`, `paired_results.csv`, `decision.md`. | All files delivered. The request-stall phase was localized: data flows, but `response.completed` never arrives. A negative service gate and a negative observability gate. | The core question is unanswered: net gain unknown. The controller was never implemented and no robot operation ran. Of the 48-minute campaign, 268.5 s was spent on live model requests. |
| C3 | "Implement and test it then give me results": isolated r6 completion baseline, autonomous finishing, known t1 then other t1 initial states, keep the 300 s score, a separate larger bounded completion experiment. | The endpoint adapter and launcher were implemented. 3 attempts were made, with honest results (0/3) and attribution. | **No baseline established.** The known t1 episode 0 never actually ran (newline). Episode 1 never ran (SHA). The larger bounded experiment was not run. Two after-the-fact protocol amendments were needed. |

---

## 9. Implications for the next step (all inference)

1. **Highest value per hour: bound the plan-request stream on the client.**
   - Abort when public `output_text` exceeds a small multiple of a normal plan (normal ≈ 922 characters, about 3.1 characters per delta). Then make one bounded retry, charged to the episode.
   - For C2's two streams it would have cut 240 s to about 52 s. If C1's two 893 s waits followed the same runaway pattern (unknown, since C1 has no telemetry), about 2,025 s of dead waiting would have become about 100 s.
   - It is mechanical and testable offline against `C2/request_diagnostics.jsonl`.
   - It addresses the only failure that blocked C1 and C2, and it needs no robot rollouts.
2. **Fix D2 before any further v3 run.** Otherwise no `evidence_acquisition_failed` run has terminal truth, which is 9 of 9 so far, so failure analysis stays blind. The change is a status-mapping or truth-status edit in `scripts/evaluate_agent_speedup_v2.py`.
3. **Merge the protocol pieces built today into the real tree, once, with tests.** These are the absolute request deadline, the no-truncation completion endpoint and reset enrollment. Today each campaign rebuilt its own isolated copy: about 1,000 s of setup per campaign plus amendments and freezes. The "everything else" bucket was 54.9% of wall-clock.
4. **S300 arithmetic for v7 on t1 (from C1 `trial_003`).**
   - Prep was 24.3 s and Codex review 138.5 s; that review is charged under the plan, while the user's definition excludes *human* review wait.
   - That leaves at most about 137 s of execution with review counted, or about 276 s without it.
   - Today's execution windows were 450–480 s, with 252–280 s of model time and 102–122 s of sim at about 0.12 s per step. Both model time and step count must roughly halve.
   - v3 r6 is further away: 581.8 s of model time in a failed episode, and 137.5 s for its first program request alone.
5. **Generality has had zero tests today.** The user's acceptance criterion is one frozen build swept across all tasks. Every campaign today ran t1 only.
