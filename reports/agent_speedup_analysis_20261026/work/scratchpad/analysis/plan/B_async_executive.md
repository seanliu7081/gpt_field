# Plan B: an asynchronous executive for uc7 (think while moving, act once the action is complete)

Angle: architecture for speed. Per-call latency is assumed to stay at 15-22 s. The question is how small a general change to the uc7 executive (`agent-speedup-v2-user-check-7`) can keep the arm moving while the model thinks, and limit model consultation to genuine decision points.

Scope rules followed: read-only on `/workspace`; no simulator, model or HTTP calls; numbers come from artifacts or from the helper scripts in `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/B/` (`extract.py`, `handoffs.py`, `refine.py`, `stream_order.py`, `s300_budget_B.py`; outputs `*.json` / `*.txt` next to them). Short names: `agent.py` = `src/gpt_field/agent_speedup_v2_user_check_agent.py`, `agent_client.py` = `..._agent_client.py`, `fragments.py` = `..._fragments.py`, "base runner" = `scripts/evaluate_agent_speedup_v2.py`, "UC runner" = `scripts/evaluate_agent_speedup_v2_user_check.py`. "raw" = raw LIBERO success; "strict" = `v2_success`; "service" = prep + approval-to-claim + execution + finalization, excluding human review wait.

---

## 1. Thesis

The uc7 loop is strictly serial. `policy.act` blocks inside `client.request_agent` (agent.py:258-290, the call is at :263). The simulator only steps after `act` returns (base runner:1440, :1483). Every fragment end returns control to the model, because pending segments are never resumed (agent.py:387; fragments.py:160, :335-337, :388). As a result the arm is frozen for every one of the 14.8 calls × 15.1 s of a mean success (V3a).

Three minimal executive changes on top of uc7 shorten that critical path without adding geometry, gates or truth:

1. **S1. Move the request off the control loop.** A single background request runs while the policy keeps a non-blocking view of client state. On its own this changes no behaviour.
2. **S2. Dispatch the action as soon as it is complete in the stream.** The schema puts `type/motion/macro/execution` before `belief/plan/why`.
3. **S3. Make checkpoints non-blocking for model-committed free-space stages.** The arm waits for the reply only before the model's own next contact-profile or gripper stage. The same step drops the stage-entry envelope veto (clamp only).

Together with the already-verified compact I/O and a recorder thread, the arithmetic on the 11 uc7 raw successes gives:

| Latency assumption | Mean service | Median | Runs ≤ 300 s |
|---|---:|---:|---:|
| Historical 15.1 s/call | **299.6 s** | 268.5 s | **7 of 11** |
| Today's 21.5 s/call (×1.42 on execution calls) | 375.4 s | 340.9 s | 1 of 11 |

At today's latency, about 4 more blocking calls per run would have to disappear. This plan cannot promise that, because the remaining blocking calls are evidently genuine decision points (§3).

---

## 2. Verified facts this plan rests on

| # | Fact | Source |
|---|---|---|
| F1 | Loop is serial; the robot is idle about 54% of uc7 control time; the pipelining upper bound is 2,151 s = 34.7% of uc7 control time | V5c; model_io.md §4-5 |
| F2 | Mean raw-success service is 448.3-448.6 s = model 223.7 s (14.8 × 15.1 s) + per-step 190.3 s (0.228 s/step × 836) + prep 23.8 s | V3a |
| F3 | Compact I/O (video 0.0252 s/step, run_state ≈ 0) gives a 372.4-372.6 s mean, 2/11 ≤ 300 s; C1 arm B replay matched 897/897 actions and 13/13 payloads | V3c; `results/speedup_evidence_20261026T055123Z/baseline_validation.json` (via campaigns_today.md §2) |
| F4 | Latency is driven by generated tokens: 3.98 s + 0.0274·in_k + 0.0392·reasoning + 0.0212·visible | model_io.md §1.2; V5c |
| F5 | 5 zero-action envelope vetoes plus 1 mid-sequence veto; avoidable cost was the replacement calls; the executor already clamps rotation to 0.1 rad/step | V4b; fragments.py:185, :233-239 |
| F6 | Plan-request tails: 11/38 non-compact vs 0/12 with v3's compact-output block; the uc7 plan path has no total deadline or output bound | V5a, V5b; chatgpt_plan.py:136-141, :315-342 |
| F7 | Today's execution calls were 42-51% slower than history (21.5-22.9 s/call); the two C1 arms ran concurrently | uc7_runs.md §1-2; model_io.md §1.3 |
| F8 | The v3 bounded transport needs the POSIX main thread (SIGALRM) | `src/gpt_field/agent_speedup_v3_transport.py:55-64` |

### New measurements (helper scripts; read-only over the 13 uc7 runs)

- **Handoff structure** (`B/handoffs.py` → `B/handoffs.txt`). Of 205 handoffs (fragment end, then call), the class is decided by the measured fragment outcome:

  | Class | All 13 runs | Raw-11 (152) | Mean next-call latency |
  |---|---:|---:|---:|
  | pre-contact (arrived, no gripper stage) | 92 (45%) | 76 | 13.9 s |
  | stall | 55 (27%) | 36 | 15.4 s |
  | post-gripper (close/open, then lift/withdraw) | 39 (19%) | 26 | 17.4 s |
  | envelope veto | 6 | 4 | 8.4 s |
  | direct / other | 13 | 10 | — |

  In the raw-11, 23 of 26 post-gripper checkpoints were followed by a decision that began with transit motion that arrived (lift/carry, or move to the next object). The median such run-out was 87 actions, 13.1 s at compact per-step cost, against a median next-call latency of 15.9 s. In 6 of the 23 the run-out would outlast the call.
- **Pre-contact checkpoints are often confirmations, but not reliably** (`B/refine.py` → `B/refine.txt`). After 91 resolvable pre-contact checkpoints, the model's next target moved:
  - ≤ 2 mm in xy in 47 cases (median 1.0 mm);
  - ≤ 5 mm in 56 cases;
  - more than 20 mm in 23 cases, e.g. t5_001 #3, "The new view corrects the destination estimate" (165 mm); t8_001 #1 (65 mm); t2_002 #6 (60 mm).

  These checkpoints change outcomes about a quarter of the time. This plan therefore keeps them as blocking decision points and does not let the arm descend before the reply.
- **Output order** (`B/stream_order.py` → `B/stream_order.txt`).
  - All 218/218 uc7 outputs emit keys in schema order (belief, plan, why, type, motion, macro, execution); uc6 171/171 likewise.
  - belief/plan/why make up 56.4% of the output text (445 of 806 chars on average), all streamed before the action.
  - The prose tail is estimated at 2.44 s per call from the 0.0212 s/visible-token coefficient. It is 3.0-3.4 s per call from the visible streaming rate of 132-150 chars/s measured on the completed C2 plan streams (`runs/speedup_next_20261026T101438Z/request_diagnostics.jsonl`, prep_002/prep_004).

---

## 3. Where the call time sits, and which calls are genuine

- **Pre-contact checkpoints (45%).** These buy the close wrist view. A quarter of them redirect the target by more than 20 mm, so they are **genuine decision points** and stay blocking. A descent committed before the reply would hit exactly the cases where the view mattered.
- **Stalls (27%).** An unexpected measured event: **genuine**. A stall always returns to the model. Model-pre-authorised "continue on stall" is excluded, because it is stall→close by another name (constraint 3).
- **Post-gripper checkpoints (19%).** The model verifies a grasp or release. Its next move is almost always "move away through known free space" (23/26). This is the one class where **the arm can keep moving while the model looks**.
- **Envelope vetoes (3%).** **Pure waste** (F5).
- **Every call.** About 56% of the visible output is prose that precedes the action, so every call has roughly 2.4-3.4 s of **critical-path latency that is not decision time**.

The design follows from this: overlap where the evidence says the next move does not depend on the answer, take the prose off the critical path for every call, and leave genuine decision points blocking.

---

## 4. Design (uc8 = new version id; uc7 files untouched)

New modules hold all new behaviour: `src/gpt_field/agent_speedup_v2_user_check_async.py` (policy, client, fragment subclass, stream reader) and `scripts/evaluate_agent_speedup_v2_user_check_async.py` (the UC runner with uc8 factories). uc7 stays runnable as the baseline. Version strings to bump:
- `VERSION` (`src/gpt_field/agent_speedup_v2_user_check.py:31`) → `agent-speedup-v2-user-check-8`;
- `POLICY_VERSION` (agent.py:30);
- `CLIENT_VERSION` / `PROMPT_VERSION` / `HISTORY_PROTOCOL` (agent_client.py:40-42);
- `FRAGMENT_VERSION` (fragments.py:21).

**Executive state.** It adds only one object: `in_flight = {future, checkpoint_step, view, superseded}`. At most one request is ever outstanding.

```
act():
  if a reply is ready (early action or completed decision)  -> collect it (record, apply per rules)
  if need_agent and no in_flight                            -> fire(view)   # snapshot + context built here
  if in_flight and nothing committed to run, or the active
     fragment is at a sync point (entry of a contact-profile
     or set_gripper stage, or the next checkpoint)          -> block on in_flight (sim paused, as in uc7)
  else                                                      -> next action of the active fragment
observe_transition():
  fragment raised a checkpoint (non-final observation_after) -> fire(after-view) if none in flight
  fragment ended by stall / gap rise / budget while in_flight -> mark in_flight superseded
collect():
  superseded -> keep the turn in history, set last_feedback "not executed: <event> at step N", re-ask
  execute/motion -> replaces the remaining stages at the run-out end / sync point
  observe (finish) -> confirms the remaining stages (claim recorded); with none left: uc7 behaviour
```

The mechanism uses no new thresholds, no geometry, no simulator truth and no task keys. The only new rule is the sync point, and it reuses the profile label the model itself chose: free-space stages after a checkpoint run while the model looks, but a contact or gripper stage never starts without its answer.

---

## 5. Steps

### P0 (prerequisite; shared with the I/O / model-I/O plans): compact I/O, recorder thread and bounded plan request
- **Change.**
  - In the uc8 runner, swap the factories at UC runner:465-484 (`make_client`, `make_policy`, `make_recorder`).
  - Recorder: the compact v3 recorder (`src/gpt_field/agent_speedup_v3_video.py:17` CANVAS 640×384, class :63, `decision` :147, `executed` :165, `close` :172), as in C1 arm B. Wrap it in `ReviewedRecorder` (UC runner:237-297) with a bounded queue and one worker thread; `decision`/`executed` enqueue and return, and `close` drains.
  - `run_state` throttling: reuse v3's `LiveSnapshotGate` / `adapt_private_evaluator` (`src/gpt_field/agent_speedup_v3_performance.py:31`, `:78-125`). It gates `state()` (base runner:997-1014, called at :1438/:1480/:1516) exactly as C1 arm B applied it (`results/speedup_evidence_20261026T055123Z/baseline/scripts/evaluate_speedup_evidence.py:66-68`), so no base-runner edit is needed.
  - Plan request: append `COMPACT_OUTPUT_INSTRUCTIONS` (`src/gpt_field/agent_speedup_v3_client.py:40-46`) and send it through S1's thread-safe bounded reader (120 s total, 2,048-char outside-string whitespace run, visible cap). Allow 1 retry, charged to service.
- **Evidence.** F3, F6; the main thread idles during model waits (F1).
- **Expected effect (estimate).** −76.0 s mean from compact I/O (scenario X0 → X1 in §6). A further −17.3 s mean (X4 → X5) comes from the recorder thread taking critical-path video to about 0. Prep tails, each of which alone exceeds 300 s, are bounded at about 120 s plus one retry.
- **Acceptance (live, in the sweep).**
  - main-thread `video_render_s` + `run_state_s` ≤ 0.005 s/step;
  - frame count = steps;
  - recorder drain at close < 5 s;
  - 0 plan waits > 120 s; guard trips and retries counted.
- **Cost.** About 6 h engineering; 0 extra model calls; no separate live run.
- **Risk.** Recorder backlog or memory (bounded queue, back-pressure, never drop frames); GIL contention (measured through the step wall time); a retry doubles prep when it fires.
- **Constraint check.** Pure I/O and transport. Policy inputs and actions are unchanged (C1 replay). No control thresholds. General.

### S1: asynchronous executive core (no behaviour change)
- **Change.**
  - Add `AsyncAgentFirstPolicy(AgentFirstV2Policy)` (agent.py:55) and `AsyncAgentFirstClient(AgentFirstV2Client)` (agent_client.py:292).
  - Split `_ask_agent` (agent.py:258-290) into:
    - **fire**, on the main thread: `_sync_framework`, a deep-copied `_context(view)`, and the two upright arrays (StepView decodes lazily, contact_skills.py:281-292); then submit `request_agent` to a 1-worker executor;
    - **collect**: `ExecutiveCall` with the full latency, `record_decision` (:287-288), `_apply_agent` (:290).
  - `act` (agent.py:502-523): blocks on the future when nothing is committed, so behaviour matches uc7.
  - Diagnostics (agent.py:541-555): `api_latency_s` = this act's blocking wait, so `model_pipeline_s` (base runner:1444) and `policy_compute_s` (:1201) stay additive. `api_calls[].latency_s` = full latency when a request completes; new field `api_overlapped_s`.
  - **Non-blocking `history_stats()`** (try-acquire the `_history_lock` RLock, direct_history.py:76; else return the snapshot cached at the last commit). This is mandatory. `ChatGPTPlanHybridClient.predict` holds `_history_lock` across the whole transport (hybrid_subscription.py:420, transport call :441, history commit :468; also agent_client.py:398). Meanwhile the main loop calls `client.history_stats()` 5 times per step: `state()` → `history()` at base runner:994-995, ×3 (:1438/:1480/:1516); :1512; agent.py:564. A naive thread would stall the arm on the first step after firing.
  - **Thread-safe bounded stream reader** for execution calls. Reuse v3's `_bounded_lines` / `_checked_records` / `_JsonTextGuard` logic (agent_speedup_v3_transport.py:92-124, :172-223) with a monotonic deadline check per chunk and per event, plus the urllib socket timeout. Leave out the SIGALRM deadline, which refuses non-main threads (:55-64).
  - At episode end with a request in flight: daemon thread; record `abandoned_in_flight_request`.
- **Evidence.** F1, F8; the lock path above (code).
- **Expected effect.** 0 s alone. It enables S2 and S3.
- **Acceptance.**
  - Offline: frozen-response replay of uc7 t1_001 with C1's tooling (`results/speedup_evidence_20261026T055123Z/tooling/baseline_replay.py`), with S2/S3 inert, gives 897/897 identical actions.
  - Offline: about 6 fake-transport tests (fire/collect, non-blocking stats while in flight, exception propagation, abandoned request).
  - Live: step wall time with a request in flight within 10% of steps without one; blocking ≈ full latency when no run-out is committed.
- **Cost.** About 8 h; 0 model calls; replay about 10 min CPU.
- **Risk.**
  - Lock stalls: handled by non-blocking stats.
  - Races on `last_request_key` / `last_log_path`: return them from the future instead.
  - `PlanApprovalRequired` (a KeyboardInterrupt subclass, client.py:200) raised in the worker: re-raised by `future.result()` on the main thread, as today.
- **Constraint check.** Behaviour-preserving; no thresholds; general; new version id.

### S2: action-first output with early dispatch
- **Change.**
  - Reorder `agent_schema()` properties (agent_client.py:205-213) to type, motion, macro, execution, belief, plan, why, and reorder the response description (agent_client.py:84-88). `validate_agent_decision` checks the key set only (:218-258), so it is unchanged.
  - In S1's reader, add an action-prefix scanner: v3's `_JsonTextGuard` plus nesting depth, fed by `response.output_text.delta`. Once the four action keys are complete, it parses that prefix, applies the same checks (`validate_execution` fragments.py:80-139, `_conform_motion`, null rules), and hands the action to the policy. The act unblocks and the fragment starts.
  - Completion still goes through `read_completed_response` (chatgpt_plan.py:315-342) and full validation. The history commit is unchanged (agent_client.py:442). Belief/plan bookkeeping (agent.py:284-288) moves to completion.
  - If the final object differs from the dispatched action, or the stream fails after dispatch: hold at the next step, set `last_feedback` = "dispatched action not committed: <reason>", and re-ask.
- **Evidence.** 218/218 outputs follow schema order; prose is 56.4% of the output chars; F4; streaming at 132-150 chars/s (§2).
- **Expected effect (estimate).** −2.4 s/call (regression) to −3.4 s/call (stream rate). That is −35.0 s per run at 15.1 s/call (scenario X2 → X3) and −49.7 s at ×1.42. It also applies to the first call (26.2 s mean in raw-11).
- **Acceptance.**
  - Per call, log dispatch time vs completion time; median lead ≥ 2 s;
  - final == dispatched in 100% of calls;
  - decision-type and segment mix, stall rate and raw success no worse than the uc7 sweep.
  - Before live: replay the scanner over the 218 saved uc7 outputs offline, plus a 3-call model smoke (no simulator) that rebuilds saved t1_001 contexts under the uc8 schema.
- **Cost.** About 6 h; 3 model calls.
- **Risk.**
  - It deliberately changes the transport invariant "deltas never supply executable text" (chatgpt_plan.py:217-222; v3 `partial_output_executable: False`, agent_speedup_v3_transport.py:46). **User decision.**
  - Quality effect of writing the action before the belief is unknown. Hidden reasoning still precedes all visible output.
- **Constraint check.** Model, effort and inputs unchanged; general; no thresholds.

### S3: non-blocking checkpoints with a model-committed free-space run-out, and clamp-only stage entry
- **Change** (subclass of `FragmentSequencePrimitive`, fragments.py:169, plus policy hooks).
  - `_complete_segment` (fragments.py:328-340): a non-final `observation_after` raises a checkpoint and the sequence continues. Today it ends with `observation_required` at :335-337.
  - `observe_transition` (agent.py:572-599) fires on a checkpoint (S1).
  - **Sync point.** The first action of a `profile=contact` or `set_gripper` stage, or the next checkpoint, waits for an outstanding reply.
  - **Reply rules** in `_apply_agent` (agent.py:305-370): execute/motion replaces the remaining stages at the run-out end or sync point; observe/finish confirms them.
  - **Superseded reply.** A `segment_stall` (fragments.py:311-325), `maintained_grip_gap_changed` (:278-289) or a budget end (:306-310) during a run-out marks the outstanding reply superseded. It is not executed, and the model is re-asked with the event.
  - Context: `_execution_state` (agent.py:181-203) gains `in_flight`: checkpoint step, run-out stages, sync point, apply rule. Feedback reports the step and measured pose at which a reply took effect.
  - **Clamp-only.**
    - Drop the `target_envelope_rejected` branch at fragments.py:236-239 and the relative-translation veto at :124-125.
    - Keep the non-finite check (:230-232), the per-step clamps (:252-258; 0.1 rad ActionAdapter :185), and the stall, gap and budget predicates.
    - In S3, delayed replies take effect from a moved pose. Without this change they would be vetoed against the old 0.30 m / 1.2 rad envelope.
  - Prompt and capability text: agent_client.py:118-121 (envelopes), :125-136 (observation_after), :150-151 ("Simulation pauses during inference" becomes "the arm holds still while you think unless you committed free-space stages after a checkpoint"), :170-172 (observe); fragments.py:151-164.
  - General guidance only: "commit after a checkpoint only motion whose usefulness does not depend on what the checkpoint shows, e.g. moving away from a completed grasp or release through space already seen to be clear".
- **Evidence.** §2 handoff table: 23/26 post-gripper checkpoints continued with a transit run-out; F5.
- **Expected effect (estimate).** −18.0 s per run net: −20.9 s overlap, +2.9 s miss cost (3 misses × a median run-out of travel back) (scenario X3 → X4). A further −2.7 s comes from dropping 4 replacement calls (X1 → X2). The call count is unchanged; the post-gripper calls move off the critical path.
- **Acceptance (live).**
  - Per run-out: overlapped seconds, replace vs confirm, superseded count;
  - 0 `target_envelope_rejected`;
  - `segment_stall` share not above uc7's 29% of fragments;
  - raw success per task ≥ the uc7 sweep (9/10).
- **Cost.** About 8 h including about 8 fake-sim tests; 0 extra model calls.
- **Risk.**
  - Stale images: a reply computed at the checkpoint takes effect 15-22 s later. Mitigated by the sync point (no contact without the reply), absolute targets and the `in_flight` context.
  - Misses (failed grasp): the robot finishes the run-out before going back.
  - The model misreading the new `observation_after` meaning.
  - Larger single-stage rotations without the veto.
- **Constraint check.**
  - Removes a veto (constraint 2) and adds no threshold.
  - The sync point reuses the model's profile label.
  - No truth trigger, and no stall→close: a stall still returns to the model.
  - Not a new state machine: one in-flight flag, and it replaces the rule "pending segments are never resumed".

**Dependencies.** S1 → S2, S3; P0 is independent. If the user rejects S2, S1 and S3 still stand, and so does the reverse.

---

## 6. S300 arithmetic (11 uc7 raw successes; `B/s300_budget_B.py` → `B/s300_budget_B.txt`)

Cumulative arithmetic on measured components. These are estimates, not predictions: trajectories and call counts are held fixed. The reconstruction also omits about 5 s of process start-up (V3b). Scenarios X0..X5 are keys S0..S5 in the script output. Each scenario's rule:
- X1: video 0.0252 s/step (trial_003), run_state 0.
- X2: the replacement call after each envelope veto is removed, plus 1 hold step.
- X3: the per-call prose tail is removed (prose share × visible tokens × 0.0212 s).
- X4: min(effective latency, run-out at compact per-step cost) is subtracted on post-gripper hits, and a median run-out is added per miss.
- X5: compact video of the steps still on the critical path is subtracted, bounded by the remaining idle wait.

| Scenario | Mean (s) | Median | ≤ 300 s |
|---|---:|---:|---|
| X0 measured | 448.6 | 438.1 | 1 (t5_001) |
| X1 + compact I/O (P0) | 372.6 | 353.8 | 2 |
| X2 + clamp-only (step S3) | 369.9 | 353.8 | 2 |
| X3 + action-first dispatch (step S2) | 334.9 | 318.1 | 3 |
| X4 + post-gripper run-outs (step S3) | 316.9 | 285.9 | 6 |
| X5 + recorder thread (P0) | **299.6** | **268.5** | **7** (t1, t2, t4, t5_001, t5_002, t7, t8) |
| X5 at today's latency ×1.42 (execution calls) | 375.4 | 340.9 | 1 (t5_001) |

Two conclusions follow:
- At today's latency the mean gap to 300 s is 75.4 s, about **4.0 fewer blocking calls per run** at about 18.7 s effective each.
- After this plan, 75% of raw-11 calls are still blocking: first call 11, pre-contact 76, stall 36. §3 shows these are genuine. Closing the gap therefore needs lower per-call latency (effort: user decision; output brevity or retained images: other plans) or fewer stalls and pre-contact corrections (perception and motion: other plans). It also needs evaluation under the historical latency regime: run sequentially, never two episodes at once (F7).

---

## 7. What can go wrong (and the guard for each)

| Risk | Where | Guard / detection |
|---|---|---|
| Main loop stalls on `_history_lock` while a request is in flight | hybrid_subscription.py:420; base runner:994-995, :1512; agent.py:564 | non-blocking stats (S1); live step-wall check |
| Stale checkpoint images (reply takes effect after the arm moved) | S3 run-outs | sync point before contact/gripper; absolute targets; `in_flight` context; feedback reports where it took effect |
| Speculation miss (grasp failed, arm carried empty) | S3 | bounded by the model's run-out; counted (replace vs confirm); estimated +2.9 s/run |
| Run-out ends in stall/gap-rise before the reply | S3 | reply marked superseded, not executed; re-ask with the event; counted |
| Early action differs from the final response, or the stream dies after dispatch | S2 | hold, feedback, re-ask; logged per call; expected rare (218/218 uc7 execution calls valid, model_io.md §0) |
| GIL contention from the request/recorder threads | S1, P0 | step wall time with vs without in-flight work |
| Raw success fires mid-run-out with a request in flight | base runner:1401-1404, :1526-1527 | daemon thread; `abandoned_in_flight_request` recorded; service clock unaffected |
| Concurrency inflates latency (today's ×1.42 coincided with concurrent arms) | evaluation | one request in flight per episode; sweep runs sequentially; s/call reported |

---

## 8. Evaluation protocol (live-first, minimal checks)

1. **Offline, no model or simulator calls.**
   - Fake-transport and fake-sim unit tests for S1-S3, about 14 cases.
   - The S2 scanner over the 218 saved uc7 outputs.
   - C1's frozen-response replay of t1_001 with S2/S3 inert: 897/897 actions.
2. **Model-call smoke, no simulator.** 3 calls rebuilt from saved t1_001 contexts with the uc8 schema, through the new bounded reader. Check key order, dispatch lead, final == dispatched, and that the guards stay quiet.
3. **One frozen uc8 build, announced before start.** LIBERO-10 t0-t9, episode 0, seed 7: the resets of `runs/agent_speedup_v2_user_check_libero_10_manual_batch_001`, so the comparison is paired per task.
   - Runs are sequential, never concurrent.
   - Review mode is the user's choice: manual or explicitly authorised auto-approve.
   - The user starts the sweep or explicitly authorises its start.
4. **Report a per-task coverage map, not per-task work.**
   - Raw success (strict shown for information), service time, S300, calls, mean s/call.
   - Blocking vs total model time, dispatch lead per call, run-outs (overlap, confirm/replace, superseded).
   - Envelope clamps (0 vetoes), critical-path video, prep attempts.
   - Compare each task with its uc7 sweep counterpart.
   - Headline: tasks with raw success, and the S300 count.
5. **Stop rule.** A harness defect in the async machinery (thread exception, lock stall, early ≠ final, history mismatch) stops the sweep. One repair under a new build id, then re-sweep from t0. Manipulation failures are recorded as framework gaps and never patched between tasks.

---

## 9. User decisions needed

1. **S2 partial-stream execution.** Accept that a complete, validated action object can start before its trailing prose has streamed (changes chatgpt_plan.py:217-222 and the v3 "no partial answer authorises action" stance).
2. **Review mode and billing.** Choose the sweep's review mode, and decide whether automated (Codex) review time counts toward service (C1 billed 138 s per episode; v3 `performance.json` excluded it).
3. **Optional: early first call.** Fire the first execution call early, at claim, overlapping the execution-simulator start (about 4-8 s; uc7_framework.md §8). Or fire it at plan-draft time during human review, which needs an accounting rule for compute that overlaps the excluded review wait.
4. **Optional, later: hedged duplicate requests.** Hedging gives an estimated −2.6 s per non-first call and −3.6 s on the first call under independence (`B/hedge.py` → `B/hedge.txt`, raw-11 latencies at historical speed). Costs: double credit use, selection toward shorter reasoning, unknown concurrency effects, and a client-lock redesign.
5. **Model and effort stay gpt-6-astra / high.** At ~21.5 s/call this plan does not reach a 300 s mean; whether to trade effort for latency is the user's call. It is not proposed here.

---

## 10. v3 disposition

Freeze v3 at r11 as a documented negative result. Do not edit its frozen copies.

Port only three pieces:
- the compact-output instruction (`agent_speedup_v3_client.py:40-46`);
- the stream-guard logic (`agent_speedup_v3_transport.py:92-124, :172-223`), re-implemented without its main-thread SIGALRM deadline (:55-64);
- the compact recorder (`agent_speedup_v3_video.py`), as in C1 arm B.

Do not port geometry memory, sweep/coverage/inventory gates, programs, evidence acquisition, the interaction monitor or stereo. They coincide with:
- 36/39 gate- or guard-triggered follow-ups and 33.4 s/call in t1_008 (V1b, V1c);
- 8/8 behavioural failures since r7 ending with zero view candidates (V2a);
- no raw success in 7 runs since r6.

v3's deferred "stage 6 asynchronous pre-planning" is superseded by this plan, which builds the asynchrony on uc7's measured fragments: 69% of fragments end `fragment_complete` (uc7_framework.md §4.5).

---

## 11. Biggest uncertainties

1. **Whether the model commits free-space run-outs** after checkpoints, and how often its replies confirm rather than replace. The −18 s figure uses the 23/26 transit-first proxy.
2. **Whether action-first ordering changes decision quality** (success, stall rate, call count). The −35 s figure assumes unchanged decisions.
3. **The latency regime** (15.1 vs 21.5 s/call; cause unknown, possibly concurrency). It decides S300 at the mean: 299.6 vs 375.4 s.
4. **Stale-image and miss effects in contact-rich tasks** (t3 drawer pushes, t6).
5. **Thread contention** on per-step cost.
6. **The arithmetic holds trajectories fixed,** while uc7 itself varies a lot (t5: 270-328 s on two runs).

## 12. Explicitly not proposed

- Continue-on-stall or close-on-stall, including model-pre-authorised variants (constraint 3).
- Descending before the reply at pre-contact checkpoints: 23/91 replies moved the target by more than 20 mm.
- Task-specific skip rules, geometry or triangulation layers, truth-triggered events.
- Raising or adding control thresholds. The only threshold change is the removal of the stage-entry envelope.
