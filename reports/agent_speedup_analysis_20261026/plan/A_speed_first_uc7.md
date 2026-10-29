# Plan A: S300 first, on uc7

**Angle.** Keep uc7's decision protocol and model-owned control. Remove waste from every component of service time. Use the measured decomposition to budget a path to ≤ 300 s on the **median** LIBERO-10 task. Do the cheapest changes first, in this order: infrastructure/harness, then motion, then call economy.

**Scope.** This plan was written read-only. No simulator, model or HTTP call was made, and no test suite was run. Paths are relative to `/workspace/gpt_field` unless absolute. Short names:
- "uc7" means `agent-speedup-v2-user-check-7`.
- `fragments.py`, `agent.py`, `agent_client.py` and `sim.py` mean `src/gpt_field/agent_speedup_v2_user_check_{fragments,agent,agent_client,sim}.py`.
- "base runner" means `scripts/evaluate_agent_speedup_v2.py`; "UC runner" means `scripts/evaluate_agent_speedup_v2_user_check.py`.
- "C1/C2/C3" are today's three campaigns: `results/speedup_evidence_20261026T055123Z`, `runs/speedup_next_20261026T101438Z` and `runs/v3_completion_baseline_20261026T133633Z`.

**Labels.** "Raw" means the independent raw LIBERO success, which is the success criterion. "Strict" means `v2_success`, which is reported as a diagnostic only. "Service" means preparation + execution + finalization, excluding human review wait; it is not control time. Every number marked *(est.)* is an estimate from the arithmetic in §5. All other numbers come from a cited artifact.

**Reproducing the numbers.** Every number I computed comes from stdlib scripts run with `python3 -I`, in `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/`:
- `A_runs.py` → `A_runs.json`: per-run decomposition from `summary.json`.
- `A_calls.py` / `A_calls_stats.py` → `A_calls.json`: 218 executive calls, from `requests/*` and `runner.log`.
- `A_segments.py` → `A_segments.json`: 355 fragment segments, from `trace.jsonl`.
- `A_motion_model.py` → `A_motion_model.json`: OSC motion model.
- `A_png_cost.py`: stdlib PNG/JSON cost lower bounds.
- `A_budget.py`, `A_budget_user.py`, `A_budget_contact.py`: S300 budget.

---

## 0. Bottom line

1. **The median task is about 145 s over budget.**
   - The reference set is the one-build uc7 sweep (`runs/agent_speedup_v2_user_check_libero_10_manual_batch_001/batch_state.json`: t0_001, t1_001, t2_002, t3_005, t4_001, t5_002, t6_011, t7_001, t8_001, t9_001).
   - Its 5th-best service is t1_001 at 438.1 s reconstructed, or about 443 s with the ~5 s of process start-up the reconstruction omits (V3 verification: supervisord wall − reconstruction = 4.56 / 5.52 s).
   - The median of the 10 tasks is 448.8 s, and 0 of 10 tasks meet S300 (`A_runs.py`, `A_budget.py` T0).
2. **Where the time goes.** The mean over the 11 uc7 raw successes is 448.6 s.
   - Model calls: 223.7 s (14.8 calls × 15.1 s).
   - Per-step harness: 0.231 s/step × 836 steps = 193 s. Split per step: simulator 0.105, video 0.091, run_state 0.025, observe 0.007, policy 0.001, residual 0.002 (`A_runs.py`).
   - Preparation: 23.8 s. Start-up and finalization: 8.3 s.
   - The arm is idle during inference, and the realized transit speed is 24–27% of the commanded speed.
3. **The cumulative path, central estimate, at the historical model speed (×1.0).** Each line gives the 5th-best of 10 / the median of 10 / the number of tasks meeting S300 (`A_budget.py`):
   - Baseline: **443 / 449 / 0**.
   - **uc8-H** (compact I/O, bounded preparation, pre-started worker): **354 / 367 / 1**.
   - **+ simulator-step slimming: 327 / 342 / 1**.
   - **+ uc8-M** (transit cap 50 mm, rotation clamp 0.25 rad, rotation-veto removal): **301 / 319 / 4**.
   - **+ uc9-C1** (terse prose, de-duplicated observation text): **281 / 296 / 5**.

   On these estimates, the median task reaches ≤ 300 s with **no change to the model, effort, decision protocol or control ownership**.
4. **The plan is not robust to model service speed, which is the largest uncertainty.**
   - Today's execution calls ran ×1.35 to ×1.51 slower than the historical token-latency fit predicts (model_io.md §1.3).
   - At ×1.25 the same build reaches only 327 s on the 5th-best task. At ×1.5 it reaches 376 s.
   - Getting back under 300 s at ×1.5 needs **both** a user decision (reasoning effort high → medium, est. −30 to −55 s) **and** about 2 fewer calls per task (`A_budget_user.py`).
   - The first live sweep measures this speed factor for free (§6, M3), before any decision is asked of the user.
5. **One reliability fix matters more than any second saved: bounding the plan-draft request.**
   - 5 of 20 preparation attempts on uc7 code hung for 297–900 s (uc7_runs.md:211). Each hang alone exceeds 300 s.
   - The compact-output instruction has a measured 0/12 tail rate, against 11/38 without it (V5a).
6. **Pre-live work is exactly one sim-only session** (about 10–15 min of simulator time, 0 model calls) plus two announced 10-task live sweeps (A on uc8, B on uc9). A third sweep runs only if B misses.

---

## 1. Decomposition this plan budgets against (all measured)

**t1_001, the 5th-best task of the sweep** (`runs/agent_speedup_v2_user_check_libero_10_t1_001/summary.json`, `A_runs.py`):

| component | s | per step | note |
|---|---:|---:|---|
| preparation (sim start + reset + plan draft 13.7 s + hashing) | 21.4 | – | `user_check.preparation_elapsed_s` |
| process start-up (not in the reconstruction) | ~5 | – | supervisord delta on t5_001/t6_009 |
| startup_to_initial_observation (second simulator) | 6.6 | – | `startup_to_initial_observation_s` |
| finalization (truth, video close, normalization) | ~2.6 | – | elapsed − control − startup |
| model pipeline, 13 calls | 182.5 | – | first call 17.9 s; non-first mean 14.0 s |
| simulator_step | 108.4 | 0.1208 | render 2×512², 2 PNG encodes, 6 validation decodes, monitors |
| video_render | 81.8 | 0.0912 | 1600×960 canvas ×2 per step |
| run_state | 25.1 | 0.0280 | 3 writes per step, one with full agent_stats |
| observe_transition + policy + residual | 9.5 | 0.0105 | |
| **total** | **≈443** | **0.250** | 897 steps |

**Pooled facts the levers rest on:**
- **Calls** (218 executive calls in 13 uc7 runs; `A_calls_stats.py`).
  - First call: 27.4 s, 438 reasoning tokens. Non-first calls: 14.7 s, 120 reasoning tokens.
  - Visible output: 806 chars per call, of which 413 (51%) is belief/plan/why prose. The mean field lengths are belief 190, plan 119 and why 104 chars.
  - A non-first observation text averages 18,988 chars:

    | field | chars | share |
    |---|---:|---:|
    | `capabilities` (identical every call) | 3,847 | 20.3% |
    | `last_feedback` | 3,159 | 16.6% |
    | `execution` (largely a duplicate of `last_feedback` and `measured_state`) | 2,972 | 15.7% |
    | `recent_feedback` (already in the transcript) | 2,955 | 15.6% |
    | `macro` (macros were used 0 times in 13 runs) | 2,270 | 12.0% |
    | `geometry` | 1,892 | 10.0% |

  - Call triggers: fragment_complete 133, segment_stall 55, initial 13, local_motion 8, target_envelope_rejected 6, budget 2, observation_required 1.
- **Latency model** (model_io.md:52). Over 218 calls: L ≈ 3.98 + 0.0274·input_k + 0.0392·reasoning + 0.0212·visible, R² 0.81.
- **Motion** (`A_motion_model.py` over `A_segments.json`; 11,895 fragment steps, which equals the `agent_stats` total).
  - Transit at the 20 mm cap realizes 5.15 mm/step (median, n = 2,761 steps after a segment's first 5). Contact at the 5 mm cap realizes 1.25 mm/step.
  - Rotation at the 0.1 rad clamp realizes 0.023 rad/step (41 transit segments with entry rotation > 0.5 rad).
  - A unit-mass OSC stand-in with kp = 150, critical damping, 20 Hz and 25 substeps of 2 ms (from simulator.json) predicts a 0.267 ratio. With a 0.96 gain fit it reproduces the recorded arrived-segment step counts within 3.2% for transit (134 segments, 4,591 steps) and 3.8% for contact (104 segments, 4,310 steps).
- **Image handling** (`A_png_cost.py`, stdlib only, this machine).
  - A PNG base64 decode plus inflate costs 2.4 ms per image, so the 6 per-step validation decodes are ≥ 14.4 ms/step.
  - The worker encodes at the default level 6 (the stored IDAT is 245,667 B, against 246,010 B when re-deflated at level 6; zlib FLEVEL 2).
  - The worker's Pillow links zlib-ng (`ldd …/PIL/_imaging*.so` → `libz-ng.so.2`), so its absolute encode cost is unmeasured.
  - In-repo evidence: "level 1 is ~5x faster than PIL's default 6" (`src/gpt_field/mark_prompting.py:85`).

---

## 2. Rules this plan holds fixed

- **Kept from uc7.** Every handoff still calls the model, and the model chooses every fragment, motion, observe or finish. The same 5 fragment skills and the measured-arrival, stall and gap predicates are kept. The model stays `gpt-6-astra` at effort high on the subscription route. Inputs stay RGB + robot state + calibration + gripper geometry. The independent raw LIBERO evaluator is unchanged and still stops at the first raw success. No new gate, threshold-driven trigger, state machine, triangulation or geometry layer is added. No stall→close.
- **Changed.**
  - Infrastructure and harness work the policy never consumes.
  - Two framework motion constants in the one existing `PROFILES` table: the transit translation cap and the rotation clamp.
  - One veto **removed**: the 1.2 rad stage-entry rotation envelope.
  - The compactness of model input and output text.
- **Versions.** Each version is new: `agent-speedup-v2-user-check-8` (uc8) and `-9` (uc9). Each is built as **new modules that subclass or adapt uc7**, the way v3 wraps the uc7 runner (`scripts/evaluate_agent_speedup_v3.py:181-231`). No uc7, v3 or frozen file is edited, so uc7 stays runnable for paired comparison.
  - Prepare and run use the same tree, because `runtime_inventory` hashes all of `src/gpt_field` (UC runner:113-128).
- **Evaluation discipline.** Live 10-task sweeps are the test loop. Each sweep is announced and started by the user. Tasks run sequentially, never as concurrent arms.

---

## 3. Steps

### S1. One sim-only measurement session (M1 + M2), with no model calls

**Why first.** The two largest uncertain harness and motion levers cannot be sized from artifacts.
- Nothing splits `simulator_step_s` into physics, the per-substep contact monitor, rendering, PNG encode and decode, JSON/IPC, `agent_geometry`, and the truth window (V4c; uc7_framework.md:385).
- Nobody has tested whether larger transit commands keep the linear OSC response or overshoot (V4a).

**M1: simulator-step split.**
- Build a stand-alone tool (scratch tooling, not a uc8 module). It replays the 897 recorded uc7 t1_001 actions (`trace.jsonl` "action" rows) open-loop through an instrumented copy of the uc7 worker and host adapter. There is no policy and no model.
- Put `perf_counter` timers around:
  - `env.step`, with the substep monitor on and off;
  - camera observation production;
  - the 2 `check_success` calls (libero.py:936, agent_speedup_v2_sim.py:162);
  - the generic monitor `finish_action` and the truth-window update (agent_speedup_v2_sim.py:155-169);
  - `image_to_data_url` ×2 (observation.py:24-45);
  - the worker validation decodes ×2 (observation.py:114-115);
  - JSON dump/pipe;
  - host `from_dict` ×2 (libero.py:662-667);
  - the `agent_geometry` IPC (sim.py:86);
  - the `CalibratedObservation` re-decode ×2 (sim.py:52-57).
- Make a second pass of the same 897 actions through the uc8 worker once S3 is coded. That pass gives S3's realized saving and doubles as its pixel check: the decoded RGB at every step must equal the recorded uc7 observation, with the same state and the same render. The final state must also equal the recorded one, which checks determinism.

**M2: OSC step response.**
- From the t1 episode-0 reset (frozen SHA), drive scripted free-space fragments through a configurable-cap copy of `FragmentSequencePrimitive` at {20, 35, 50} mm and {0.10, 0.25} rad/step:
  - a 0.20 m lateral move 0.15 m above the table;
  - a 0.25 m diagonal move;
  - pure 0.8 rad and 1.6 rad reorientations;
  - 0.2 m combined with 0.8 rad.
- Record the steady mm/step and rad/step, the steps to `pose_arrived`, the overshoot past the target, whether actuator commands saturate, and elbow-limit proximity.

**Acceptance (offline, decides S3 and S4).**
- M1 returns ms/step per component, and these sum to within 5% of the measured `simulator_step_s/steps` of the replay.
- For M2, adopt 50 mm / 0.25 rad only if all of these hold:
  - steady transit ≥ 10 mm/step (model prediction 13.3);
  - rotation ≥ 0.05 rad/step;
  - overshoot ≤ the arrival tolerance (4 mm / 0.06 rad);
  - no actuator saturation.
- Otherwise use 35 mm / 0.2 rad. At an unchanged 0.1 rad, the model says 35 mm keeps 74% of the 50 mm translation saving (1,283 against 1,729 steps over 13 runs; `A_motion_model.py`).

**Cost.** About 3–4 engineering hours (reusing C1 `tooling/baseline_replay.py`), one sim-only session of about 10–15 min, and 0 model calls.
**Risk.** M2 cannot test held-object slip at higher acceleration. That is covered live by the unchanged gap-rise interrupt (`fragments.py:278-289`; 0 firings in 13 uc7 runs).

### S2. uc8 harness: I/O, preparation, start-up and the service clock (no behaviour change)

**H1. Compact display I/O.** Port C1 arm B unchanged, since it was verified behaviour-equivalent.
- Change points:
  - `base._default_recorder_factory` (base runner:670-672) → `AgentSpeedUpV3VideoRecorder` (`src/gpt_field/agent_speedup_v3_video.py`), relabelled uc8;
  - `state()` (base runner:997-1014) → gated by `LiveSnapshotGate` via `adapt_private_evaluator` (`src/gpt_field/agent_speedup_v3_performance.py:31-55, 78-125`);
  - wiring as in `results/speedup_evidence_20261026T055123Z/baseline/scripts/evaluate_speedup_evidence.py:66-68`.
- Optional **H1b**: move canvas drawing and the ffmpeg pipe write to a background thread fed from the main loop (`recorder.decision`/`executed`, base runner:1477, 1511). The canvas then overlaps the simulator pipe wait. Not counted centrally; upside of −15 to −25 ms/step *(est.)*.
- Optional, by user choice (**D5**): write per-step PNGs to `trace.jsonl` "executed" rows (base runner:1494-1496; about 556 KB per row) only at decision steps. Saves 0.5 GB per run; about 1–2 s per task.

**H2. Bounded, compact preparation and bounded execution calls.**
- **Compact plan output.** Append `COMPACT_OUTPUT_INSTRUCTIONS` (`agent_speedup_v3_client.py:40-46`; the 396-char block) to the plan instructions, as `AgentSpeedUpV3PlanClient` does (`:121-130`).
- **Bounded transport.** Install the bounded transport (`agent_speedup_v3_transport.py:224-271`) as the default for both the plan client (UC runner:183-185 passes `transport=`) and the executive client (base runner:1373-1374 passes `transport=None`; default at `agent_speedup_v2_client.py:685`), as the v3 mixin does (`agent_speedup_v3_client.py:79-82`).
- **Per-kind limits:**
  - plan: total deadline 60 s (longest completed non-compact draft 30.4 s, V5 other issues); visible cap 8,192 chars (normal plan 920–947 chars);
  - agent: deadline 120 s (uc7 max 39.8 s, today's max 52.8 s); visible cap 16,384 chars (uc7 max 1,261);
  - whitespace-run guard: 2,048 (v3 value), for both kinds.
- **One retry.** Add the guard/deadline codes to the retry eligibility (`agent_speedup_v2_client.py:903-914`; today v3 guard trips are not retried) with **one** same-request retry, charged to service.
  - On C2's two runaway replays, the 8,192-char cap trips at about 24.5 s and about 40.4 s *(est., from the recorded delta rates)*, instead of at 120/900 s.

**H3. Pre-started execution worker** (`start` path only).
- Spawn the execution `UserCheckSimProcess` (sim.py:67-90; `start()` is idempotent, libero.py:580-596) right before `client.request_plan` (UC runner:188), and hand the warm worker to `execute_reviewed` through `sim_factory` (UC runner:489).
- Do the `Popen` synchronously inside `base._simulator_without_credentials()` (UC runner:162). That context pops credential variables from `os.environ` (pattern at `scripts/evaluate_bowl_subscription_close.py:113-119`), so it must not race the model call.
- Wait for the worker's `metadata` reply, which covers the slow imports, in a background thread.
- The evaluation still runs in a fresh process with its own reset and SHA check (base runner:1355-1367).
- The model and the review are untouched.

**H0. Service clock.** Merge C1's `ServiceClock` (`results/speedup_evidence_20261026T055123Z/baseline/src/gpt_field/agent_speedup_evidence_clock.py:19-60`) as a uc8 module.
- It starts at process start (before `_live_gate`, UC runner:500-508), pauses only for human review wait, and marks the first raw success and the end of finalization.
- The clock is reporting-only. The optional per-episode evaluation budget (**D3**) is a separate user decision.

**Evidence.**
- C1 arm B measured video 25.642 s / 1,019 steps (0.0252 s/step) and run_state 0.108 s / 1,019 steps, against uc7's pooled 0.0908 and 0.0253 s/step (`results/speedup_evidence_20261026T055123Z/evaluations/trial_003/trial/summary.json`; `A_runs.py`).
- Its offline display benchmark showed −74.55% (10.689 → 2.721 s per 120 steps).
- Its frozen-response replay was 897/897 actions identical, with 13/13 payloads identical (C1 `baseline_validation.json`).
- Preparation tails: 5/20 uc7-code attempts hung (uc7_runs.md:211). There were 11/38 tails without the compact block against 0/12 with it, and 4/9 against 0/8 on identical t1 inputs (V5a, V5 other issues).
- The plan path has only a per-socket 180 s timeout (`chatgpt_plan.py:136-157`). `max_output_tokens` is rejected by the route (`chatgpt_plan.py:37-41`).
- `startup_to_initial_observation_s` averages 5.76 s (11 successes; `A_runs.py`).

**Expected effect** *(est.)*.
- Per step: −0.0908 s pooled (video −0.0656, run_state −0.0252).
- Per task: mean −80 s; t1_001 −88.8 s (video −59.2, run_state −25.0, worker pre-start −4.6 at 0.7 × 6.58 s).
- Preparation: removes the 300–900 s hang mode (each hang is a certain S300 loss for that task). Typical preparation is unchanged at about 21–24 s.

**Acceptance.**
- *Offline, before sweep A, once (R0, no simulator, no model).* C1's saved-observation, frozen-response replay (`results/speedup_evidence_20261026T055123Z/tooling/baseline_replay.py`; C1 got `actions_exact_to_history` 897 and `model_payloads_exact_to_history_and_AB` 13 in `baseline_validation.json`) is rerun through the uc8 runner and policy wrappers with uc7 motion constants and prompt. It must give 897/897 identical actions and 13/13 identical request payloads.
- *Live, in sweep A:*
  - video_render_s ≤ 0.03 s/step and run_state_s ≤ 0.001 s/step on every task;
  - startup_to_initial_observation_s ≤ 3 s;
  - 10/10 plan drafts complete, each within 60 s plus at most 1 retry;
  - 0 episodes lost to harness defects: empty credential store, newline serialization, or missing reset SHA (today's campaign defects D6/D3/D4 in campaigns_today.md §6).

**Cost.** About 6–8 engineering hours (integration of existing, already-run pieces). R0 takes a few CPU minutes, with no simulator and no model calls.
**Risk.**
- The SIGALRM deadline requires the model call on the main thread and no other `ITIMER_REAL` (`agent_speedup_v3_transport.py:55-64`). H1b and H3 threads must never call the model.
- The compact-output effect is suggestive (p ≈ 0.03) but confounded with the v3 client and the task mix (V5a).
- The compact video is a lower-fidelity artifact (D4).

### S3. uc8 simulator-step slimming: remove repeated work at every step (no behaviour change)

**Changes, ordered by certainty:**
1. **No redundant validation decodes.**
   - Today each step decodes both PNGs twice in the worker (`observation.py:114-115` via `from_libero`, `:143-160`), twice in host `from_dict` (`libero.py:662-667`) and twice more in `CalibratedObservation.__post_init__` (`sim.py:55-57`).
   - uc8 checks PNG signature and IHDR dimensions per step, and decodes fully only where pixels are consumed: `StepView._image` at decision steps and the video panel decode. Pixels, and therefore model inputs, are unchanged.
2. **Per-step PNG at compress level 1.** Change the worker's step response (`libero.py:951-954` → `image_to_data_url` save at `observation.py:44`), but **not the reset response**, so `initial_observation_sha256` (`agent_speedup_v2_sim.py:671-674`) and the frozen-reset registry are unchanged.
   - The encoding is lossless, so pixels are identical. The model receives client re-encodes of the pixels (`agent_client.py:497-502` → `mark_prompting.py:84-88`), which are byte-identical.
3. **Lazy `agent_geometry`.** Request the extra per-observation IPC (`sim.py:86`) only at decision steps.
4. **Per-step bookkeeping.** Call `_sync_framework(view)` (`agent.py:595`) only at handoffs and ≤ 1 Hz. Today it deep-copies policy state every step (observe_transition 7.0 ms/step). Not counted centrally.
5. **Conditional on M1 only:**
   - lazy camera rendering, with images produced only for decision steps and video frames;
   - per-control-step instead of per-substep contact sampling. That one is a **user decision (D6)**, because it changes the fidelity of the evaluator's contact audit, not success.

**Evidence.**
- 6 validation decodes per step plus 2 encodes inside `simulator_step_s` (V4c other issues). The decode lower bound is ≥ 14.4 ms/step (`A_png_cost.py`).
- Level 1 is about 5× faster than 6 (`mark_prompting.py:85`). Stdlib zlib on this machine: level 6 at 75–80 ms per 512² image, level 1 at 8–9 ms. The zlib-ng figures in the worker are lower and unmeasured.
- `simulator_step_s` = 0.1046 s/step pooled; 0.085–0.124 per run.

**Expected effect** *(est., to be fixed by M1)*.
- Per step: −0.030 s central (range −0.015, decodes only, to −0.050).
- Per task: mean −25.1 s (range −12.5 to −42); t1_001 −26.9 s.

**Acceptance.**
- *Offline, in the S1 second pass:* every step's decoded RGB equals the recorded uc7 observation; the reset SHA is unchanged.
- *Live, sweep A:* per-task `simulator_step_s/steps` ≤ the uc7 per-task value minus the M1-predicted saving (±5 ms). The new component timers sum to `simulator_step_s` within 5%.

**Cost.** About 3–5 engineering hours, and no extra session (it reuses S1/S2's).
**Risk.** Skipping full decodes on non-decision steps would let a corrupt PNG reach only the video. Level-1 PNGs are about 14% larger (281 KB against 246 KB), which adds about 70 KB/step to the pipe and is negligible.

### S4. uc8 motion: faster transit, less rotation lag, one veto removed; then **live Sweep A**

**Changes** (new uc8 fragment module subclassing `FragmentSequencePrimitive`):
- **Transit translation cap 0.02 → 0.05 m** (Euclidean). Today it is `PROFILES["transit"]["action_translation_m"]` (`fragments.py:25-28`), clipped at `fragments.py:252-256`. 0.05 m is within the OSC per-axis limit (`action.py:16, 67-68`; simulator.json `output_max` 0.05).
- **Transit rotation clamp 0.1 → 0.25 rad/step.** Today the adapter is fixed at `ActionAdapter(max_rotation_rad=.1)` (`fragments.py:185`). 0.25 rad is within the adapter's (0, 0.5] bound (`action.py:52`) and the OSC 0.5 rad scale. The contact profile is unchanged (5 mm, 0.1 rad).
- **Remove the transit stage-entry rotation envelope (1.2 rad)** at `fragments.py:236-239`. The executor already clamps rotation per step. The translation envelope (0.30 m / 0.08 m) and the contact rotation envelope (0.5 rad) are kept.
- **Prompt and capability text** state the new numbers: replace `agent_client.py:118-120` in a uc8 copy of `AGENT_INSTRUCTIONS`, as v3 derives its prompt from `V7_AGENT_INSTRUCTIONS`. `fragment_capabilities()` reports the uc8 table automatically (`fragments.py:142-166`).

**Evidence.**
- Transit realizes 5.15 mm/step, against 20 mm commanded. Rotation realizes 0.023 rad/step, against 0.1 rad.
- Transit is 51% of motion steps and 79% of the path (V4a).
- The motion model, validated at the recorded caps, predicts 13.3 mm/step at 50 mm with no overshoot (critically damped).
- The 1.2 rad veto produced 5 zero-action calls on 4 tasks (t4, t6 ×2, t8, t9). The avoidable cost was 42.4 s of replacement calls plus 5 one-step holds (V4b; `A_calls.py`). The rule recurs on ≥ 2 tasks, and removing it is a veto removal.

**Expected effect** *(est.)*.
- Steps: arrived transit segments only (stalled segments are not credited):
  - over all 13 runs, −2,291 of 11,936 steps (−19.2%);
  - t1_001: 897 → 667;
  - 50 mm alone gives −14.5%, and 35 mm/0.1 rad gives −10.7% (`A_motion_model.py`).
- Seconds, at the post-S3 per-step cost:
  - mean −17.6 s per task, including about −2 s from veto removal;
  - t1_001 −29 s;
  - t4/t8/t9 additionally save their 6.4–8.1 s replacement call.

**Sweep A** (live; announced and user-started):
- Scope: uc8 = S2 + S3 + S4 on LIBERO-10 t0–t9, episode 0, seed 7, with the same frozen resets as the uc7 sweep, so every task is paired with its uc7 run.
- Run sequentially, with manual review (wait excluded) or user-authorized `--auto-approve` (D2).
- Output:
  - the per-task coverage map;
  - the measured service-speed factor **f = Σ measured latency / Σ uc7-fit predicted latency** (model_io.md:52 coefficients) over the sweep's calls (M3);
  - per-component timings, step counts, realized mm/step, stalls, and slip and envelope events.

**Acceptance (live, sweep A).**
- Raw coverage ≥ 8/10, and every lost task is inspected for a transit-speed cause (forbidden-contact onset inside a transit segment, or a slip). uc7 had 9/10.
- Steady transit ≥ 10 mm/step on every task.
- Transit steps per metre at least 35% lower than the uc7 per-task value.
- `maintained_grip_gap_changed` ≤ 1 across the sweep (uc7: 0).
- Zero `target_envelope_rejected` from rotation.
- Calls per task within ±2 of the paired uc7 run.
- Projected at f = 1.0 *(est.)*: 5th-best ≈ 301 s, median of 10 ≈ 319 s, S300 4/10. At f = 1.5: 411 / 428 / 0.

**Cost.** About 2 engineering hours for code. Sweep A: 10 live episodes, about 150 model calls (10 plans + about 14 executive calls per task), about 75–100 min of wall time. The failing t6 is bounded by D3.
**Risk.**
- Higher transit speed means harder unplanned contacts, knocked neighbours, or slips of held objects at the start of a segment (acceleration about 7.5 m/s² at a 50 mm error, against 3 at 20 mm *(est.)*). Raw coverage could fall.
- Mitigations: the model still chooses transit or contact; the stall and gap-rise interrupts are unchanged; M2 gates the cap.
- n = 1 per task, so a ±1-task change is within noise unless it is attributable.

### S5. uc9 call economy (C1): same decisions, cheaper per call; then **live Sweep B**

**Changes** (new uc9 client module subclassing `AgentFirstV2Client`; protocol and decision types unchanged):
1. **Terse prose.**
   - Keep all 7 keys (`agent_client.py:195-215`), with tighter bounds: belief ≤ 240, plan ≤ 160, why ≤ 80 chars (today 1800/1800/600 at `:206-208`). Add a one-paragraph instruction: "belief/plan/why are terse working notes: key measurements and the next subgoal".
   - Replace the length **rejection** in `validate_agent_decision` (`:226-228`) with truncation of the parsed copy, logged. This is a veto removal: a reject costs a hold plus a re-ask, `agent.py:264-274`.
   - Append `COMPACT_OUTPUT_INSTRUCTIONS`. uc7 execution outputs already had 0 outside-string whitespace, so this is a guard, not a saving.
2. **De-duplicated observation text. The transcript stays append-only and no old message is rewritten**, unlike v3's wire projection (`agent_speedup_v3_client.py:587-640`). Concretely:
   - `capabilities` and `macro` go only in call 1, or when they change;
   - `recent_feedback` is dropped, because each earlier observation in the transcript already carries its own `last_feedback`;
   - `execution` keeps `active`/`progress` and drops `last_result`/`shared_state`, which duplicate `last_feedback`/`measured_state`/`model_belief`;
   - the external-camera block of `geometry` goes only in call 1;
   - `last_feedback` becomes one row per segment (skill, profile, status/reason, actions, final errors, gap). Today the compacted feedback lists every segment twice (`_execution_evidence_view`, `agent_client.py:261-289`).
   - Change points: `_context` (`agent.py:205-224`) and `request_agent` field assembly (`agent_client.py:466-495`).

**Evidence.**
- Prose is 51% of visible output (413 of 806 chars per call).
- 63% of the non-first observation text (12.0k of 19.0k chars) is static or duplicated (§1).
- Fitted costs: 0.0212 s per visible token and 0.0274 s per 1k input tokens (model_io.md:52). The C3 attempt_003 (r6 copy) streaming rate was about 87–144 chars/s (v3_runs.md:104).
- 75.7% of uc7 input tokens are re-sent observation text (model_io.md §3.3).

**Expected effect** *(est.)*.
- Per call: −1.1 s from prose (range 0.6–2.0) and −0.4 to −0.9 s from input (central: 40% of text tokens).
- Per task: mean −23.1 s; t1_001 −19.4 s.
- Sweep B projection at f = 1.0: 5th-best ≈ 281 s, median of 10 ≈ 296 s, S300 5/10 (low-effect assumptions: 308 / 322 / 3).

**Sweep B** (live; same protocol as A, paired per task with sweep A).

**Acceptance (live, sweep B).**
- Mean prose ≤ 260 chars per call (from 413).
- Non-first input tokens per call ≤ 60% of sweep A's.
- Token-adjusted latency saving ≥ 1.2 s per call at sweep A's f.
- 0 schema rejections.
- Raw coverage ≥ sweep A − 1.
- **Headline: S300 ≥ 5/10 (the median task ≤ 300 s) if f ≤ 1.1.**

**Cost.** About 4–6 engineering hours. Sweep B: about 140 model calls and about 70–90 min.
**Risk.**
- Shorter belief/plan text may weaken the model's running state, leading to more calls or wrong next steps.
- Dropped duplicates remain only deep in the transcript, so attention to them may weaken.
- Both risks are measured by calls per task and raw coverage against sweep A.

### S6. Decision point on model time, only if Sweep B misses: U1 / U2 / C2 → uc10 + Sweep C

**Trigger.** Sweep B's 5th-best service is > 300 s, or f > 1.1 measured in A and B.

**Options, in order of expected seconds per unit of risk** *(est., from `A_budget_user.py`; 5th-best at f = 1.0 / 1.25 / 1.5)*:
- **U1 (user decision):** execution reasoning effort high → medium, modelled as −50% reasoning tokens at 0.0392 s/token. Gives 251 / 285 / 321 s. Quality impact unknown.
- **C2 (my lever, flagged speculative):** one cost sentence in the prompt, roughly "each decision pauses the robot about 15 s; when the next stages need no new visual evidence, put them in the same execute". Today's mean is 1.95 segments of 6 allowed and 49% of the step budget used (uc7_runs.md §3). −2 calls gives 251 / 292 / 334 s.
  - **Excluded** on principle: any automatic continuation after a stall (that would be stall→close in disguise).
- **U2 (user decision, accounting):** under manual review, issue the first executive call speculatively during the excluded human review wait, and discard it if the plan is edited (2/10 sweep plans were edited). Saves the 27.4 s mean first call, giving 264 / 304 / 345 s. With auto-approve there is no wait to hide it in.
- U1 + 2 fewer calls gives 224 / 251 / 278 s. This is the only combination modelled that reaches ≤ 300 s at f = 1.5.

**Acceptance (live, sweep C on the chosen uc10).** S300 ≥ 5/10 at the measured f, and raw coverage ≥ sweep B − 1.
**Optional held-out sweep** on the final build (episode 1 for all 10 tasks). It needs reset-SHA enrollment at prepare time, because today's runner fails closed without a frozen SHA (`scripts/evaluate_agent_speedup_v2.py:411-421`; campaign defect D4 in campaigns_today.md §6, which consumed C3 attempt 2).

**Cost.** About 1–2 hours plus one sweep (about 140 calls, about 80 min), and the same again for the held-out sweep.
**Risk.** U1 and C2 trade decision quality for time. Both are measured against B on the same resets.

---

## 4. Which measurement must precede which change

| measurement | how | before | why |
|---|---|---|---|
| **M1** simulator-step split | S1 sim-only replay of 897 recorded actions with per-op timers | S3 items 2, 3 and 5 (encode level, lazy render, contact sampling) | the size of S3 ranges −15 to −50 ms/step; item 1 (decode de-duplication) can proceed without it |
| **M2** OSC response at 35/50 mm and 0.1/0.25 rad | S1 scripted free-space moves | S4 constants | linear-model prediction unverified at higher command (V4a) |
| **R0** replay equivalence | C1 `baseline_replay.py` (saved observations + frozen responses; no simulator, no model) through the uc8 wrappers with uc7 motion constants | Sweep A | C3 lost 2 of 3 slots to harness defects caught only live |
| **M3** service-speed factor f | Sweep A calls against the uc7 token-latency fit; no extra calls | S6 trigger, and whether to ask U1 | today ×1.35–1.51; it decides S300 reachability |
| **M4** per-call tokens, latency and observation-text sizes | Sweep A request logs | S5 bounds | sizes the prose and de-duplication cuts |
| Sweep B result | live | S6 | only spend effort or protocol changes if needed |

(Optional: if the user wants f before sweep A, replay the first-call payloads of 3 historical runs; about 3 model calls, about 90 s, low precision because first calls vary from 17.9 to 39.8 s.)

---

## 5. S300 budget (estimates, `A_budget.py`)

**Method.**
- Per run, take the measured components: prep, approval, start-up/finalization, per-step sim/video/run_state/observe/policy/residual, steps, and per-call latency and tokens.
- Replace each component by tier:
  - **T1:** video 0.0252 and run_state 0.0001 s/step, measured in C1 trial_003; 70% of `startup_to_initial_observation` hidden.
  - **T2:** −30 ms/step in the simulator.
  - **T3:** calibrated motion-model step savings, plus removal of the veto replacement calls.
  - **T4:** −1.1 s per call, minus 0.0274 s × 40% of each call's non-image, non-instruction input tokens.
  - **T5:** −1 mean non-first call.
- Add 5 s of process start-up everywhere.
- Low and high variants: start-up hidden 50/90%; simulator −15/−50 ms; transit at 35 mm only, or 50 mm + 0.25 rad; prose 0.6/2.0 s; text cut 25/55%; calls 0/2.

**10-task sweep, model speed ×1.0** (5th-best / median of 10 / S300 count):

| tier | central | low-effect | high-effect |
|---|---|---|---|
| T0 uc7 measured (+5 s) | 443.1 / 448.8 / 0 | same | same |
| T1 uc8-H1–H3 | 354.3 / 367.2 / 1 | 355.6 / 368.6 / 1 | 351.6 / 364.5 / 1 |
| T2 + S3 simulator slimming | 327.4 / 342.1 / 1 | 342.1 / 356.1 / 1 | 308.3 / 323.5 / 3 |
| T3 + S4 motion = **uc8** | 301.3 / 318.6 / 4 | 318.4 / 334.7 / 1 | 288.4 / 302.5 / 5 |
| T4 + S5 = **uc9** | **280.5 / 296.0 / 5** | 307.5 / 322.3 / 3 | 252.2 / 262.8 / 7 |
| T5 + C2 (−1 call) | 265.4 / 281.7 / 6 | 307.5 / 322.3 / 3 | 222.2 / 235.5 / 9 |

**Per-task projection, central, ×1.0** (uc7 → uc8 → uc9). S300 here means raw success and ≤ 300 s, using uc7's outcomes:

| task | uc7 raw | calls | steps (uc7 → uc8) | T0 | T3 (uc8) | T4 (uc9) |
|---|---|---:|---|---:|---:|---:|
| t0 | ✓ | 14 | 1107 → 919 | 549.8 | 377.7 | 356.4 |
| t1 | ✓ | 13 | 897 → 667 | 443.1 | 298.3 | 278.9 |
| t2 | ✓ | 15 | 621 → 516 | 385.4 | 301.3 | 278.2 |
| t3 | ✓ | 21 | 744 → 622 | 456.6 | 351.5 | 315.7 |
| t4 | ✓ | 16 | 770 → 666 | 454.5 | 335.9 | 311.6 |
| t5 | ✓ | 10 | 497 → 418 | 332.8 | 264.0 | 250.0 |
| t6 | ✗ | 27 | 1197 → 1124 | 745.3 | 569.8 | 520.8 |
| t7 | ✓ | 13 | 812 → 589 | 432.0 | 299.9 | 280.5 |
| t8 | ✓ | 13 | 902 → 710 | 432.4 | 292.0 | 273.5 |
| t9 | ✓ | 19 | 1239 → 1143 | 572.2 | 402.4 | 371.5 |

**Median task (t1_001), central:**
- 443.1;
- − 88.8 (video −59.2, run_state −25.0, worker pre-start −4.6) = 354.3;
- − 26.9 (−30 ms × 897 steps) = 327.4;
- − 29.1 (−230 steps × 126.6 ms; no veto on t1) = 298.3;
- − 19.4 (13 calls × about 1.5 s) = **278.9 s**.

At uc9 the remaining 279 s splits as follows:

| component | s |
|---|---:|
| model | 163 |
| steps (667 × 126.6 ms) | 84 |
| prep | 21 |
| fixed (process start, start-up remainder, finalization) | 10 |

**Speed sensitivity (5th-best).**
- uc9 (T4) gives 280.5 at ×1.0, 327.3 at ×1.25 and 376.4 at ×1.5.
- At ×1.5 the only modelled route under 300 s is U1 + 2 fewer calls: 277.6 s (S300 5/10).

**Upside not counted centrally:**
- H1b asynchronous video drawing: about −15 to −25 ms/step, i.e. −10 to −17 s on t1;
- `_sync_framework` gating: −2 to −4 s;
- contact cap 5 → 8 mm: −5 s on the 5th-best (`A_budget_contact.py`). This is deferred because contact precision is the risk.

---

## 6. Evaluation protocol

- **Builds.** uc8 (S2 + S3 + S4), then uc9 (= uc8 + S5), then a conditional uc10. Each is a new version id with new modules, and its runtime inventory and frozen source are copied at prepare (UC runner:113-128, 218-224). No edits during a sweep.
- **Pre-live.** One sim-only session (M1 on the uc7 and uc8 workers, plus M2) and one saved-observation replay (R0), with 0 model calls. No offline test suites, no smoke runs and no pre-live model calls, unless the user asks for the optional f probe.
- **Sweeps.**
  - LIBERO-10 t0–t9, episode 0, seed 7, frozen reset SHAs. The same initial states as the uc7 one-build sweep, so every task is paired.
  - One episode per task, run **sequentially**, because concurrent arms are a suspected cause of today's ×1.5.
  - Initial-plan review is retained: manual browser review (wait excluded) or user-authorized `--auto-approve` (D2). No Codex delegated review unless the user accepts that its compute is billed (D2).
- **Metrics per task (coverage map).**
  - Raw LIBERO success (independent evaluator, unchanged), measured service from the uc8 service clock, and the S300 flag.
  - Calls; steps; per-component timings (prep, process start, start-up, model, simulator with M1 sub-components, video, run_state, observe, finalization).
  - Per-call input, reasoning and visible tokens and latency.
  - Realized transit and contact mm/step; stalls; envelope, slip and schema-rejection events.
  - Strict `v2_success` and forbidden-contact counts are **diagnostics only**.
- **Headline.** The count of tasks with raw success and service ≤ 300 s, and the 5th-best per-task service (the "median task"). It is reported with f, at the measured f and with the latency normalized to f = 1.0.
- **Attribution.**
  - Paired per-task component deltas (deterministic accounting) for harness and motion.
  - Token-adjusted per-call latency for call economy.
  - Raw-coverage changes of ±1 task are treated as noise unless they are attributable to a cause in the trace.
- **Budget and stop.**
  - Optional per-episode evaluation cap of 900 s of service (D3). It would have censored 0 of 11 uc7 raw successes, whose maximum was 650.3 s.
  - The operator stop file stays available.
  - The first raw success still ends the episode, which is the evaluator's semantics (D1).

---

## 7. v3 disposition

- **Freeze v3 at r11. There is no r12, and its files and frozen copies are not touched.**
- **The evidence:**
  - Every v3 raw success executed ≥ 81.6% uc7-style fragments (V1a).
  - In r6 t1_008, 36 of 39 follow-up calls were gate or guard handoffs (V1b), at 33.4 s against 14.0 s per call.
  - Model time was 97% of the +1,186 s control difference against uc7 on the same reset (V1c).
  - v3 is 0/7 since r6, and 0/3 today.
  - All behavioural failures exit through gate → 0 view candidates → escalate (V2a), on metric geometry biased 17–42 mm below the real table, beyond its declared 8–15 mm (V2b).
- **What uc8 imports or copies under new version ids** is v3's infrastructure, which is independent of the gates:
  - the bounded transport (`agent_speedup_v3_transport.py`);
  - `COMPACT_OUTPUT_INSTRUCTIONS` (`agent_speedup_v3_client.py:40-46`);
  - the compact recorder (`agent_speedup_v3_video.py`);
  - `LiveSnapshotGate`/`adapt_private_evaluator` (`agent_speedup_v3_performance.py`);
  - optionally the 503 availability retry (`agent_speedup_v3_client.py:52-110`).
- **Not ported:** GeometryMemory/LocalGeometryMemory and the sweep, coverage and inventory gates; programs; pending intent, repeat and continuation guards; acquire_evidence/escalate; the visual watch; the interaction monitor; the operation runtime; stereo; and the wire history projection. The projection rewrites history; uc9's de-duplication achieves the text cut and stays append-only.

---

## 8. User decisions needed

- **D1. Raw-success truncation stays.** The episode stops at the first raw success, mid-release, so strict terminal quality is 0/15 (uc7_runs.md §6). An autonomous-finish endpoint would add about 20–40 steps plus about 1 call (≈ +20–25 s *(est.)*), and t5_001's S300 depends on the truncation (V6b). Recommend keeping the current semantics for S300 and reporting strict as a diagnostic.
- **D2. Approval mode and its accounting for sweeps.**
  - Manual review (wait excluded) or user-authorized `--auto-approve`.
  - Whether Codex-delegated review compute is billed to service. Today it was billed at 138–285 s, which alone makes S300 unreachable (campaigns_today.md §5.7).
  - Whether a speculative first call may run during human review wait (U2, −27 s *(est.)*).
- **D3. Per-episode evaluation cap for sweeps,** e.g. 900 s of service. It bounds failing runs: t6 ran 711–930 s until an operator stop. It is cost control only, and 300 s stays a target, not a stop.
- **D4. Video artifact.** Accept the compact 640×384, one-frame-per-step recording (C1 arm B) in place of the legacy 1600×960 one. It saves 66 ms/step and about 55 s per task.
- **D5. Trace audit.** Keep full-resolution PNGs in every `trace.jsonl` row (about 0.5 GB per run), or keep them only at decision steps.
- **D6. Contact-audit sampling.** Conditional on M1: per-control-step instead of per-substep sampling, if M1 shows the monitor is a large share of the simulator step.
- **D7 (constraint 4). Reasoning effort.** high → medium for execution calls, only if S6 triggers (U1). It is the largest single lever at today's service speed.
- **D8. Go-ahead.** Approval to run the S1 sim-only session, and the go-ahead or schedule for Sweep A, B and the conditional C. Live sweeps are user-started or announced.

---

## 9. Biggest uncertainties

1. **Model service speed f.** Today it was ×1.35–1.51 the historical token-adjusted latency, with the cause unknown (concurrent arms or service load). The plan meets the median target only at f ≲ 1.1 without U1 or C2.
2. **The simulator-step split.** S3 ranges −15 to −50 ms/step (−12 to −42 s per task) until M1 runs.
3. **The 50 mm / 0.25 rad transit.** Linear OSC tracking is predicted, but held-object slip, harder unplanned contacts and knocked objects are untested and could cost raw coverage.
4. **Call-economy latency.** The coefficients are observational (±20–35%), and terser prose or de-duplicated context could degrade the model's state tracking.
5. **n = 1 per task.** Coverage and per-task service vary run to run. For example, t5 took 270 s and 328 s on two uc7 runs, and t6 was 0/2.
6. **The preparation-tail fix.** Compact output gave 0/12 tails against 11/38 (p ≈ 0.03), which is confounded. The bound limits the damage but does not cure the cause.

---

## 10. Deliberately not in this plan

- **The t6 gaps:** relational-region grounding ("right of the plate" → world +y) and mug-handle grasp/upright recovery. They are generality work for another plan. S300 on the median task does not require t6, but no change here may lose a uc7-covered task.
- **Pipelining or speculative next calls**, which change what the model sees (model_io.md §5).
- **Lower image resolution or fewer image pairs:** ≤ 0.6% of model time by tokens (model_io.md §5).
- **Prompt caching:** about 3% cached, and the route's cache key is unverified.
- **Faster contact profile**, deferred: about −5 s on the median for the highest-precision phase.
- **Any stall-triggered continuation**, excluded: it is stall→close.
