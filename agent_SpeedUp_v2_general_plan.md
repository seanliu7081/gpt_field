# agent_speedup_v2: a general plan to speed up and finish tasks, agent-first

- **Replaces:** the F1–F11 plan (§4–6 of `agent_SpeedUp_v2_t2_failure_analysis.md`).
- **Builds on:** the user's v6, "模型持续控制任务，宏技能作为可选的加速执行方式" ("the model continuously controls the task; macro skills are optional accelerators"), with its 11:19 UTC review change.
- **Latest runs (read only):**
  - t6_007 (v6) finished with `success` at step 464 after 84 calls. Its runner exited 0 at 11:10:08 UTC, but its supervisord daemon is still idle and can be shut down.
  - t3_002 (v6 on t3) finished with `success` at 11:42:22 UTC.
- **Status:** **No code changed.** No live run and no model call.
- **Date:** 2026-10-21, about 12:45 UTC.

---

## 1. Summary

**The shift: the model finishes tasks; the executor makes its loop end honestly, cost little, move fast and stay safe.** The loop is v6's. This plan adds what it lacks, in one code version for every task, instead of repairing v2's macros along one task's failure chain.

**Why agent-first** (one episode per task so far):
- **v6 completed both tasks it ran.** t6_007 did t6 in 1,289 s and 84 calls from an unedited draft, the first t6 completion by any v2-family build after 7 failed runs and about 27% faster than astranomy's t6 (run on a shared service). t3_002 did t3 in 1,270 s and 87 calls from an auto-approved draft, all by direct control.
- astranomy completed t2, t6 and t8; executor-first v2 completed 0 of 3 autonomously. Six tasks have no agent-first run.

**What holds v6 back** (seen in both runs unless marked "code"):
1. **No honest end.** The runner stops on raw LIBERO success; `finish` is a log line. t6_007's `finish` came with the pudding misplaced, and the continuing episode saved it; t3_002 never said `finish`. A stuck run never ends.
2. **A costly, fragile loop:** +5.3–5.9k input tokens per turn, about 20M per task; a schema slip, an oversize message or a grounding difference ends the run (code).
3. **Macros barely run:** 1 in 171 turns, its 60-step grant spent on calibration; a model-made hold disables them (code).
4. **Free space and actuation are slow:** 2.5 cm per turn; t3_002's drawer close took 35 turns.
5. **No envelope on direct targets:** 1,972 forbidden points in t6_007 (the hand on the target mug), 646 robot-scene points in t3_002.

**What changes, in this order:**
1. Episodes end by themselves, scored where the agent ends them.
2. Turns get cheap: about 750 s instead of about 1,280 s per v6 run at the same turn counts (est.).
3. Macros stop blocking.
4. Free space and contact windows leave the model.
5. Relation macros, cavities and a knob, drawer and door engine come only on sweep evidence.

**What goes away:** the 60-step pause, the end at the first failed primitive, the grounding stop, the 56,000-character crash, the unretried schema error, the runner's success stop and v6's "continuation means unfinished" prompt line. Later: the grasp-point refusal, the `marginal` cascade, O-1's 5° latch, O4's 20° veto and the DD stretch guard as a veto.

**Milestones (est.):** M0, a loop that ends, costs little and runs unattended (3–4.5 d), then sweep 1 on all 10 tasks; M1 moves free space off the model (6.5–10 d); M2–M4 (20–31 d) are a menu opened by sweep evidence. Completions are measured, not promised.

**What to do first.**
1. **The user** rules on D1, D4 and D5(a), and on D3 if robot-scene contact must be prevented (§9). This is the go-ahead for M0.
2. **The user, or Claude at their choice,** builds M0 on the current source (the t3_002 binding) as four separable patches (§7), in about 3–4.5 days (est.): one build whose episodes end by themselves, with requests under 25k tokens and `--auto-approve` for a serial 10-task loop.
3. **The user** starts sweep 1 (t0–t9, episode 0, `--no-video`, while their astranomy agent is idle; about 12–25M tokens, 1.5–3 h, est.). It yields the first all-10 coverage map, every failure tied to a mechanism.

---

## 2. The framework after this plan

The loop stays v6's: observe, then a direct target or an accelerator, then measured feedback. The plan adds a ledger, three ends (`finish`, no progress for N turns, a safety stop), cheap turns and accelerators that never block.

| | v6 today | After M0 and M1 |
|---|---|---|
| End | the runner stops on LIBERO success; `finish` is a log line | `finish` ends the episode; N turns without a new fact end it; success is recorded, not a stop |
| Context | all text kept; about 20M tokens per task | only what changed; the last 10 turns plus a header; ≤ 25k per request |
| Macros | 60-step grants; blocked by a model hold; end at the first failed primitive | run to an event; recover inside the relation; a native hold survives the model's steps |
| Motion | 2.5 cm per turn; no envelope beyond 5 cm / 0.35 rad | `move` and `carry`; a robot-fact envelope and a stall stop |

- **Who decides.** The model: the intent, a direct target or an accelerator, and when it is done. The executor: safe motion, measurements and the ledger (facts measured true for the first time, WP1). The evaluator: success on the end state the agent chose; it neither ends the episode nor signals through continuation (R-4, `agent_SpeedUp_v2.md:70`).
- **The accelerator contract:** check preconditions first, or return at 0 steps with the reason; run to an event under a hang guard, never a pause; return the outcome with its decisive measurement (value, limit, σ, source), unknown kept as unknown; never deadlock or end the run.
- **Speed levers, in order:** cheap turns (M0), phase primitives (M1), then macros that no longer block.
- **Region semantics is a named risk.** t6_007's pudding was "right of the plate" but about 5 cm beyond the benchmark's zone. Such a miss is logged, never tuned against a BDDL zone (R-3).

### 2.1 Rules

- **G1. The test:** would we build this if the task that showed the problem did not exist? Does it help an unseen task without retuning?
- **G2. Admission:** a mechanism seen on 2 or more tasks, or a code-level defect stated as such. The first run is an example, not the reason. Acceptance is an all-10 sweep.
- **G3. One veto rule:** only the IH list vetoes, on robot facts or on measurements whose σ is below the margin. Everything else ranks, guards or informs; unknown is not failure.
- **G4. One table:** no new veto thresholds, no refits; every new number is ROBOT, ONLINE or RULING, registered and linted, the wrapper's too (0 of 17 wrapper modules are linted today).

### 2.2 The closed list of irreversible harms

| Id | Harm | In an accelerator | On a direct target |
|---|---|---|---|
| IH-1 | The arm leaves its ROBOT-fact envelope: full stretch 0.804 m, joint limits, joint-7 yaw, the measured table floor. The DD stretch guards (0.77 / 0.785 m, from a v1 run and t2 smokes) only warn | pose vetoed | clipped and reported |
| IH-2 | The hand opens over a measured drop above the limit. A stall or touch is support; unmeasured support is unknown | release vetoed (cavity: outside a measured opening, M3); no macro stage opens a holding hand | from M1, with a hold fact and a measured plane |
| IH-3 | A confident planned palm contact on a free object | vetoed; the descent stops at first contact (`agent_speedup_v2_relations.py:2515-2525`) | palm contact with the target is recorded (D3) |
| IH-4 | A non-finger link touches a fixture or a non-target object. Parts the intent names (knob, drawer front, door) are targets | keep-outs excluding named parts | keep-outs from registrations (M1); a reporting stall stop |
| IH-5 | Contact disturbs a verified relation. Moving the container that carries the object does not | intent-scoped ledger keep-outs; container-relative in M4 | the same |

- No IH rule on a direct target may need a capability the vocabulary lacks; it falls back to stall, touch and limits.
- The executor never acts on simulator truth, except a camera calibration the user declares for it (D2b). The contact monitor stays evaluator-only (`scripts/evaluate_agent_speedup_v2.py:194-195`).

**What makes it general:** one prompt, code version and threshold table; no task key (`task_key_reaches_policy: false` in both v6 runs); no veto on a one-run constant; every accelerator serves 2 or more tasks; progress and vetoes stay physical for articulated and cavity tasks.

---

## 3. Mechanisms that recur

Rows 1, 3, 4 and 9 recur on t2, t6 and t8 (v2); row 2 on t2 and t6. Rows 5–7 are loop code, measured on t6 and t3 (v6). Row 8 is seen on t2, t3, t6 and t8; row 10 on t3 and a libero_goal task. Unregistrable table spots (39 of 78 v2 registration failures) are mostly t6's, so they get no accelerator.

| # | Mechanism | Evidence | Cost | WP |
|---|---|---|---|---|
| 1 | Macros refuse before contact | 59–60 of 61 failed v2 holds; native mug holds 4 of 25 (t6); agent-first closes held 11 of 14 | 14,738 v2 steps (55%) | WP4 |
| 2 | Placement is never compared with the executor's aim | v2: 5 good holds, 0 `placed`; t6_004 on the plate (strict true) but 52.61 mm from LIBERO's origin; t6_005: another step's keep vetoed the mug's plate | 1,700 steps, 90 calls (t2) | WP5 |
| 3 | No progress measure, no end | v2: 0 `recovery_exhausted`, 3,436 idle steps; t6_004–t6_006: a stop, a crash, a stop; both v6 runs: the success stop | 968 s idle (t6_005) | WP1 |
| 4 | Codes, not measurements | 117 of 149 v2 repairs carried unrelated thresholds (84 in one t2 loop); v6's `last_outcome` is always None (code: `agent.py:383-384` drains it before `agent_macros.py:157-168` reads it) | 110 of 149 answers repeated | WP3 |
| 5 | Macros start cold and cannot continue the model's work | stale geometry after direct steps (`agent.py:502-508`); 205 calibration steps; a grounding call that can stop the run | 1 macro in 171 turns | WP2 |
| 6 | The grant is smaller than a macro with its setup | a place_on is 1,171–1,375 v2 steps, 20–23 grants (uc-t2) | 290–335 s per place_on (est.) | WP2 |
| 7 | A costly, fragile loop | +5.88k and +5.34k tokens per turn, about 85% repeated (est.); t6_005's size crash; `SchemaError` never asked again | 20.6M, 20.1M tokens; 510–550 s of prefill per run (est.) | WP6 |
| 8 | No envelope on direct targets | astranomy 267–1,351 forbidden points per run; t6_007 1,972; t3_002 646 robot-scene; v2 0 in 10 runs | D3 | WP7 |
| 9 | Measurement-heavy macro steps | probes 39.0%, calibration 7.6% of v2 steps | uc-t2: probes 29%, calibration 11%, surveys 335 steps per place | WP8 |
| 10 | Actuation by 5-cm targets is slow | t3_002's drawer close: 35 turns, about 620 s (51% of model time); astranomy's libero_goal drawer task: 77 turns | 1 measured use | WP9 |

**Kept:** v2's calibration (10/10), registration (39/40) and closes (7 holds from 9; 0 forbidden contacts); the model's grasp reading; v3–v6's contract, stable IDs, scoped keeps, compaction, post-release measurement and one request per turn (astranomy's second took 41.6% of its inference time).

### 3.1 The two v6 runs

| | t6_007 (mug on plate; pudding right of plate) | t3_002 (bowl into bottom drawer; close it) |
|---|---|---|
| Result | raw success at step 464; `lenient_success` (pudding held) | raw success at step 419; `lenient_success` (hand on drawer) |
| End | `finish` at turn 75 with the pudding about 5 cm past the zone. Turn 76: "Both objects appear supported, but the episode continues, so my completion estimate is not sufficient." It moved the box about 5.4 cm; success at turn 84 | 87 direct turns, no `finish`; bowl released at turn 51; drawer closed over turns 53–87 |

---

## 4. Work packages

### WP0. Autonomous sweep harness

*0.5–1 d · M0*

- `--auto-approve` approves the draft unchanged with t3_002's provenance, once per sweep; `human_reviewed_plan` follows the mode.
- A serial t0–t9 loop with `--no-video`, `--stop-after-steps` about 3,000 (est.; over 6× either v6 success) and a pause on usage-limit errors.
- Build identity: the binding's `runtime_sha256` (v2's hashes miss v6). New numbers get `LABELS` and a wrapper lint profile.
- **Removes:** ten reviews per sweep; the hard-coded `human_reviewed_plan=True` (`scripts/evaluate_agent_speedup_v2_user_check.py:339, 378, 527`).

### WP1. Progress ledger and termination

*1 d · M0*

- **Facts,** once per object and intent: a hold (a close, with the gap above the empty-close floor (D5a) through a 2.5 cm rise); a release after a hold; a macro's `succeeded`; later a placement measurement (WP5) and a named part's travel (M4). Motion, replans and re-measuring are not progress; beliefs are logged, not counted.
- **Ends:** a terminal `finish` (an open hand first retreats a fixed short distance); `blocked_no_progress` after N turns (D1b); a safety stop. Each is emitted as `recovery_exhausted` with a cause, which the base runner already stops on. The runner records the first success step instead of stopping (D1a).
- **Removes** (with D1): "no episode, retry or repair budget" (`agent.py:426`); the non-terminal `finish` (`agent.py:298-304`); the "continuation means unfinished" prompt line (`agent_client.py:117-121`), which v2 refuses as R-4; the same runner's removal of `recovery_exhausted` (`:227-229`).
- **Risk:** N can cut an actuation phase with no fact; t3_002 ran 36 turns after its last one.

### WP2. Non-blocking accelerators

*0.5–1 d (M0, with WP3) + 1–1.5 d (M1)*

- **M0, recovery inside the relation.** Failed primitives and preemptions return to the relation as in v2 (`agent_speedup_v2.py:1869-1877`), not out of the macro (`agent.py:365-379`). This restores the descent ladder, carry re-anchoring, keep-out back-offs and the turn's reversal (`agent_speedup_v2_relations.py:2520-2525, 1026-1027, 1035, 1042, 3968-3979`).
- **M0, events.** A macro yields at its relation's return, a repair need or a ceiling, plus one optional progress yield. `budget_steps` becomes a hang guard per macro kind (a RULING from t2 and t6 step counts), plus a native-call ceiling (t6_005 made 27 pointing calls). Grounding stops latching: a difference returns `unavailable`, not `plan_conflict` (`agent_macros.py:118`). The capability text gives nominal steps, not hit rates.
- **M1, hand-back.** A native hold survives the model's steps while the gripper stays closed and the gap within σ. A model-made hold is a ledger fact, not a native held frame (registering one would open the hand). `release` means "open here", not "return to the pick support" (`agent.py:134`).
- **Removes:** the 60-step pause; the end at the first failure; the grounding stop; stale geometry after each direct step (`agent_macros.py:184-191`). The closed-hand block (`:87-94`) becomes "no opening stage while a hold fact stands".

### WP3. Honest results

*M0 (in WP2) + 2 d (M2)*

- **M0:** `last_outcome` reaches the yield record, and the decisive record survives compaction.
- **M2:** a measurement record at about 15 decisive exits (value, σ, source, limit, verdict, margin); a no-op reports `not_executed`.
- **Removes:** 3 cause renamings; threshold-name dumps; predicted effects.

### WP4. Act-then-verify in the grasp macro

*4–6 d (M2); variants 3–4.5 d on evidence*

- **A monotone extension in the wrapper namespace:** accepted plans come back byte-identical. Only measurement-gap refusals (`geometry_unmeasured`, `marginal`, `pointed_not_grasp_point`) become guarded attempts: the best yaw, a descent to the first touch, a close, then `VerifyHold`. IH reasons still veto.
- **Holds are accepted by outcome,** whoever closed: the gap holds through the lift. Co-motion is a monitor; undecided means provisional (t6_006 stalled on that gate). Variants only if attempts still fail on 2 or more tasks.
- **Removes:** the grasp-point refusal; the inferred-fill veto; the `marginal` cascade; the 0.5 mm guarded-close abort; O-1's 5° latch.
- **Why:** break-even is 0.27–0.36 at cheap turns; act-then-verify anchors are 0.5–0.78 (est.).

### WP5. Placement measured against the aim

*1–1.5 d (M1) + 2–3 d (M2)*

- **It builds on v5's `_measure_goal`** (`agent_speedup_v2_user_check_spatial.py:422-486`), which in v6 replaces the core check and its no-centre path (`agent_speedup_v2_relations.py:4589-4591`); but `supported_on` passes on "centre inside, share > 0", and the aim tolerance applies only to `centered_on`.
- **M1:** after lower-and-open, the object is registered again with the hand empty, and its offset from the aim is reported at the online 3σ tolerance. **M2:** a macro re-grasps and shifts once, then yields.
- **R-3:** aims and tolerances stay executor-measured (`spatial.py:455-460`), never from LIBERO's origin test, its 30 mm or a BDDL region.
- **Removes:** `supported_on` without an aim comparison; O4's 20° veto. The co-motion, upright and rotation checks and the lowering stall become monitors.

### WP6. A cheap, robust loop

*1–1.5 d · M0*

- **Only what changed:** each feedback record once; no repeated capabilities, geometry or belief (5.5k → 1.5k text tokens per turn, est.).
- **A window:** the last 10 turns, a code-built header (contract, ledger, latest belief and plan) and v6's 6 image pairs, ≤ 25k per request (RULINGs). Short-history competence is untested; sweep 1 reads it against the existing full-history runs.
- **No single-fault end:** a cross-field slip is coerced or asked again once; an oversize message or an overflow is compacted. Effort by decision type waits for a sweep (D2c).
- **Removes:** the 8-record `recent_feedback`; the 56,000-character `ValueError` (`agent_client.py:396-397`); the unretried `SchemaError`.

### WP7. Safety envelope for direct motion

*0.5–1 d · M1 (M0 if D3 says so)*

- IH-1 from ROBOT facts, with the table floor once calibrated (D5b); a violating target is clipped and reported.
- A stall stop ends and reports the segment; it is not a veto. t3_002's robot-scene points came during pushes the model itself saw stall.
- Keep-outs from registrations, scoped to the intent and excluding its named parts, answered with metric feedback.

### WP8. Measurement economy

*0 d (M0) + 2–4 d (M2, on evidence)*

- **M0:** `--no-video`; video is about 4% of v6 control time (31% in v2).
- **M2,** only if macro time exceeds 60% (a development trigger): skip Calibrate+'s robot-only stages (D5a); reuse probes and surveys. Supplied camera calibration needs D2(b) extended to the executor, and its floor is about 84 steps, not 0.

### WP9. Accelerator vocabulary (§5)

- **M1, phase primitives (4–6 d):** `move` and `carry` (a safe-height via-point, arrival detection), lift-and-look, lower-and-open at a model or macro point on any measured support, and retreat-and-look. They reuse existing v2 primitives and need D5(b).
- **M3, destinations (5–8 d):** `place_in` as a destination kind of `place_on` (D5c); joint arrangement in a shared region.
- **M4, actuation (7–12 d):** one DOF engine for knobs, drawers and doors (t2, t3, t9) that follows the axis to a measured end stop (`DOFProbe`, `FollowAxis` and `PushAlong` are stubs today, `agent_speedup_v2_primitives.py:1866-1877`); frontal and pitched hands, with the pitch from measured clearance; container-relative keeps. `push_to` and tilted insertion stay with the model.

---

## 5. Speed plan

### 5.1 Coverage map

Uses come from the BDDL goals; turns per use are astranomy's measured phase costs. This is a coverage map, not a list of work items.

| Phase (accelerator) | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | Uses | Turns per use today | Built in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transit (`move`) | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 19 | 10.0 per leg | M1 |
| Carry (`carry`) | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 16 | 10.2 | M1 |
| Lift- and retreat-and-look | 4 | 4 | 2 | 2 | 4 | 2 | 4 | 4 | 4 | 2 | 32 | 3.3 each | M1 |
| Release on a support (lower-and-open) | · | · | 1 | · | 2 | · | 2 | · | 2 | · | 7 | 2.8 | M1 |
| Release into a cavity (`place_in`) | 2 | 2 | · | 1 | · | 1 | · | 2 | · | 1 | 9 | 2.8 | M3 |
| Grasp (act-then-verify) | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 16 | 1.1, plus 2.6 aligning | M2 |
| Knob, drawer or door (DOF engine) | · | · | 1 | 1 | · | · | · | · | · | 1 | 3 | knob 6; drawer close 35 | M4 |

Non-top-down hands (t3, t9) come with M4; held reorientation (t5 only) is not built.

### 5.2 Priority: time saved × tasks touched

Seconds assume about 8–9 s per turn after M0 (est.: t6_007's latency fit at ≤ 25k tokens, plus simulation).
1. **Free space:** 35 uses on 10 tasks, about 9 turns saved each: 2,000–2,600 s per sweep (est.) → M1.
2. **Looks:** 32 uses on 10 tasks, about 2.3 turns each: 430–630 s (est.) → M1.
3. **Releases:** 16 uses on 10 tasks; speed-neutral at cheap turns (1.8 turns against 11–19 s of v2 lowering), but they give a measured placement → M1, M3.
4. **Actuation:** 3 uses on 3 tasks; one direct drawer close took 35 turns → M4, once a sweep shows it on 2 or more of t2, t3 and t9.
5. **Grasp:** speed-neutral (v2 aligns 4–34 s slower), but it sets macro hit rates → M2.

Completion is bounded by agent competence (unmeasured on 6 tasks), the turn budget, the honest `finish`, region semantics and, on t5, a predicate that fails an upright book [P].

### 5.3 Levers

| Lever | Today | Mechanism | Target (est.) |
|---|---|---|---|
| Context | +5.3–5.9k per turn, up to 0.46–0.49M | WP6 (M0) | ≤ 25k per request; prefill about 50 s per task, not 510–550 s |
| Macro grant | 60 steps; 20–23 per place_on | event yields (WP2, M0) | 1 turn, plus ≤ 1 progress yield |
| Free space | 40 turns per metre | `move`, `carry` (M1) | ≤ 10 per metre |
| Contact windows | 7.1 turns per contact | the looks, lower-and-open (M1) | ≤ 4 per grasp, ≤ 2 per release |
| Effort | high; 36–50 reasoning tokens per turn | medium on routine turns, after a sweep (D2c) | ≤ 2 s per turn |

Input per task is reported, not gated (about 1.2–2.5M after M0, est.).

**Targets** (two-relation task, without video): ≤ 600 s first, about half of v6's 1,270–1,289 s and at least 2× faster than astranomy's 1,203–1,757 s; a stretch from sweep 1's spread (uc-t2's 316 s, one human-edited t2 run, is a reference only); ≤ 30 turns after M1; ≤ 10 s per routine turn, ≤ 15 s near contact.

**The speed model.** t6_007 falsified its seconds (it predicted 2,301 s and an overflow near turn 44; t6_007 took 1,289 s over 84 turns, with no overflow at 492k tokens), but its ordering held in every parameter set:
- per-turn cost is the first lever, now measured: t6_007's fit is 0.90 s + 0.0283 s per 1k input tokens + 0.0372 s per visible and 0.0388 s per reasoning token (R² 0.79; t3_002's is close), and model latency is 95–97% of control time;
- phase primitives beat relation macros: a place_on carries 177–208 s of v2 steps (uc-t2), and macros do not pay at today's hit rate of about 0.025 (0.10 × 0.25).

---

## 6. Evaluation protocol (light)

- **Unit:** one all-10 sweep per milestone (t0–t9, episode 0, seed 7, one `runtime_sha256`, serial, `--no-video`, `--auto-approve`), started by the user while astranomy is idle. Sweep 0 is t6_007 and t3_002 (not re-run); sweeps 1 and 2 follow M0 and M1.
- **Headline:** autonomous completions out of 10, as raw LIBERO success on the end state when the episode ends by itself (D1). Beside it: strict physical success at that state, and success at any step (the runner's first-success step), which is the old count.
- **No smokes and no default task.** The first run after a change is a real run on a task that needs it, never the same task twice in a row, and one run never accepts a change. v3–v6 were each tried on t6 first; sweeps replace that pattern, so no single task drives the design again.
- **Variance:** n = 1. A result that flips between sweeps is rerun once before it counts. Three episodes per completed task are optional, for quoting.
- **Per task, from existing files:** end cause, last fact and turns since it; time; tokens per request (p95); accelerator uses; contacts by class; departures from the model's initial plan draft.
- **Acceptance** (sweep 1 has no earlier sweep, so items 1 and 5 carry it):
  1. Every episode ends by itself, with a named last fact and no crash.
  2. The headline does not drop, paired on the same episodes, after the flip rerun.
  3. Each new accelerator shows a measured outcome where it is needed, or the miss maps to a mechanism.
  4. Time and tokens per completed task fall.
  5. Each failure maps to a mechanism whose fix passes G2.
- **Light:** unit tests only where a live run cannot show the logic; no replay gates or forks; truth analysis only for unattributed failures.
- **Cost per sweep (est.):** v6 as built, about 200M tokens and 3.6 h if every task ends like the two successes, and unbounded otherwise (one stuck task reaches the 1.05M request cap near turn 180, after about 95M). After M0 (N = 50, 60–120 turns per task): about 12–25M and 1.5–3 h. After M1 (30–40 turns): about 6–8M and 1–1.5 h.

---

## 7. Order and milestones

| Milestone | Work | Size (est.) | What the sweep shows |
|---|---|---|---|
| **M0** | WP0; WP1; WP2 events, recovery, grounding; WP3 fix; WP6 | 3–4.5 d | first data on the 6 never-run tasks; clean ends |
| **M1** | D5(b); WP2 hand-back; WP5; WP7; WP9 primitives | 6.5–10 d | free-space turns; time per completion |
| **M2** (menu) | WP3 record; WP4; WP5 in macros; WP8 | 8–11 d (+2–4) | hit rate against break-even |
| **M3** (menu) | `place_in`, arrangement; WP4 variants | 5–8 d (+3–4.5) | measured cavity releases |
| **M4** (menu) | DOF engine, hand families, container keeps | 7–12 d | actuation turns per use |

- **Committed:** M0 and M1, about 9.5–14.5 working days (est.); the menu opens item by item on sweep evidence from 2 or more tasks. A day includes unit tests, not sweeps; the user's pace (v3 to v6 in about three hours) suggests these sizes are generous.
- **M0 as four separable patches:** a loop that ends (WP1); a cheap, robust loop (WP6); macros that neither vanish nor end the run (WP2, WP3); the sweep harness (WP0).
- **Swap rules:** WP4 moves first if sweep 1 shows the model's grasps failing on 2 or more tasks; M4 moves up if actuation dominates on 2 or more of t2, t3 and t9.
- **Coordination:** M0 touches modules the user owns. Build on the current source, equal to the t3_002 binding, whose review change lets six tasks' `slide` and `place_in` intents pass; never edit while a review or preparation is pending (t3_001 was re-prepared for that). The user chooses who applies each patch.

---

## 8. What happens to F1–F11

**Why the old plan was task-specific,** though no code named a task: one t2 run chose its objective, its proof (a t2 fork) and its constants (fitted on t2's moka handle); all 9 validation rows used t2 states, and 10 of 11 fixes were pinned to one t2 step.

- **Reframed:** F1 and F7 as act-then-verify and monitors (WP4, WP5); F4 and F9 as evidence-gated variants (WP4). Their t2 constants (6 cm creep, 2°, the RS-7 refit) are dropped.
- **Folded:** F2 into the measurement record (WP3); F5 and F6 into guarded attempts (WP4); F10 into the ledger (WP1) and the record (WP3).
- **Kept as a code defect only:** F3. `agent_speedup_v2.py:1122` still compares claims with the episode-start image; it is fixed only if a sweep shows a macro checking an effect claim.
- **Dropped:** F8 (the model aligns faster than v2); F11 (t6 only, and lower-and-open covers it); the V0 and t2 forks; "under 10 calls on t2".

---

## 9. Decisions needed from the user

D1, D4 and D5(a) gate M0. D3 gates it only if robot-scene contact must be prevented. D2 can wait.

- **D1. Termination and scoring.**
  - **(a) The episode ends where the agent ends it.** The runner records the first success step instead of stopping; the episode ends at a terminal `finish`, no progress, a safety stop or the operator cap, and is scored there. Continuation tells the agent it has not succeeded, which the spec refuses (R-4); t6_007 used it after a wrong `finish`. Under (a) t6_007 scores a failure, and the old count stays visible beside the headline. It amends v6's "only the independent evaluator terminates a successful task".
  - **(b) End on no progress** after N turns without a new fact, one RULING for all tasks. The gaps were about 25 turns (astranomy t2, t8), 28 (t6_007) and 36 (t3_002's drawer close). N is set again once from the 10-task record.
  - Both amend v6's "no episode, retry or repair budget" (`agent.py:426`). On 2026-10-20 the user approved an agent-side terminal stop for v2 ("Allow early stop"), with LIBERO counting unchanged.
  - **Recommended: yes to both, with N = 50:** about 1.4× the longest gap seen so far (36 turns, t3_002's drawer close).
- **D2. v6's departures from the v2 spec.** (a) The model emits numbers, directions, gripper commands and completion claims, which AC-5 (`agent_SpeedUp_v2.md:81`) and `:207` forbid. (b) It gets simulator camera calibration, against "Nothing is supplied in advance" (`:1214`). (c) High effort, against the spec's medium (`:228`).
  - **Recommended:** ratify (a) inside the envelope; ratify (b) for the model only, as a declared sensor fact, with the executor on its own calibration (D5b) and an online-calibration sweep optional; keep (c) high until a sweep shows medium loses nothing on routine turns (the one medium astranomy run failed).
- **D3. Contact policy.** Forbidden points: v2 0 in 10 runs; astranomy 267–1,351 per run (hand on target, or pot on burner); t6_007 1,972 (hand on the mug, 0 robot-scene); t3_002 646 robot-scene during pushes (hand on cabinet 430 and wine rack 149, arm link 5 on cabinet 67) plus 25 bowl on cabinet; astranomy's libero_goal drawer runs show it too (older monitor).
  - **Recommended:** record only finger and palm contact with the target and the intent's named parts, and held-object contact with its destination; report robot-scene contact by class each sweep and handle it with WP7 in M1, or in M0 (+0.5–1 d) if it must be prevented from sweep 1.
- **D4. The headline, and where to build.**
  - **Headline:** autonomous runs (the draft approved unedited), as the user did for t3_002 for that one episode; `--auto-approve` makes it one authorization per sweep, counted by approval mode, not by `human_reviewed_plan`. Two unedited drafts do not prove drafts reliable elsewhere (reviews edited 5 of 8 user-check plans).
  - **Where:** M0 and M1 in the v6 wrapper; from M2, its relation namespace (`build_relation_layer`, `agent_speedup_v2_user_check_control.py:281-298`), not a package copy that would fork the threshold table. v2 v8 stays byte-frozen.
  - **Recommended: both.**
- **D5. Robot facts and calibration.**
  - **(a) Before M0:** may Calibrate+'s robot-only outputs (121 of 205 steps) be ROBOT facts? M0's hold fact needs the empty-close floor, bit-identical across 5 runs on 3 scenes. If not, M0 runs Calibrate+ at the start.
  - **(b) Before M1:** run Calibrate+ unchanged at episode start (about 205 steps, 31 s, 0 calls) for the table plane and heightfield.
  - **(c) Before M3:** may a cavity release start inside a measured opening at the hand-limited height (a can's bottom then sits at least 2.6 cm above a basket floor [P]), amending "release only on `on_floor`"?
  - **Recommended: yes to all three.**

---

## 10. Evidence index

- **Conventions:** (est.) estimate; [P] prior-session truth analysis; `agent.py`, `agent_client.py`, `agent_macros.py` = `src/gpt_field/agent_speedup_v2_user_check_agent{,_client,_macros}.py`. Each v6 number is from t6_007 or t3_002 (one episode each); each v2 number for a completed relation is from uc-t2 (`…_user_check_libero_10_t2_001`).
- **Scratch reports** in `/tmp/claude-0/-workspace/c2a68f98-7864-498d-82e5-c6d389d85195/scratchpad/genplan/`: `evidence/` (`crosscheck.md` first, then census, gates, loop, coverage, astranomy, priorplans, arch); `design/` (four plans, two judgements, `merged_draft.md`); `final/` (`read_v6.md`, `read_speed.md` with `speed_work/`, `revised_draft.md`, four `critique_*.md`).
- **Runs** (`/workspace/gpt_field/runs/agent_speedup_v2_`…): the 10 retained v2 runs (26,772 steps, 312 calls, 1 success: `eval3*`, `fast_libero_10_t2_001`, `user_check_libero_10_t2_001`, `_t6_001`, `_t6_002`); `_t6_004` to `_t6_007`, `_t3_001_launch`, `_t3_002` (summary, truth, collision, requests, run state, review and launch files); astranomy in `/workspace/gpt_field/astranomy/runs/`.
- **Docs and code** (under `/workspace/gpt_field/`): `agent_SpeedUp_v2.md`, `agent_SpeedUp_v2_user_check.md` (v6), the `src/gpt_field/agent_speedup_v2*` modules cited inline, `scripts/evaluate_agent_speedup_v2{,_user_check}.py`.
