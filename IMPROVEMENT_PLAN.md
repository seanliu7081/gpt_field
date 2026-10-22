# gpt_field improvement plan: reliable, cheap contact

Written 2026-10-16. This plan comes from a read-only, multi-agent analysis of the code and run artifacts; no model calls or simulator runs were made. That analysis covered seven areas, and an independent verifier checked the important claims. The plan was then grounded in the code by four scouts, drafted three ways, merged by a judge, and corrected by a feasibility critic. The intermediate reports are not saved in this repository.

## 0. Goal, principles, metrics

**Goal.** Make the contact stage reliable and cheap within the current constraints: zero-shot, Astra medium via the subscription, RGB plus state8 only. Then confirm that full tasks get faster.

**Principles**
1. **Division of labour.** The model makes discrete choices: target, strategy, side, approach direction, readiness. Local code does the metric part of contact: orientation, depth by touch, close, hold test, retry.
2. **Verified bugs are fixed by default.** A/B tests are only non-inferiority checks, not gates.
3. **Each build is gated by a cheap diagnostic.** Each new mechanism must be shown to fire in a short live segment before any full task.
4. **Lean base.** Build on speed_first v3 (177 output tokens/call), not on the op_v4/contact_region stack (523–550 tokens, about 22 s/call), which we stop extending.
5. **Constants.** Live policies use only robot or online-derived constants unless you rule otherwise (ruling R-b).
6. **No N=1 comparisons, no hand stops.** Decision rules are written down before each run.

**Primary metrics**
- Engagement rate per close: evaluator engagement (DIAGNOSTIC), binary per trial, plus a post-engagement hold tail.
- Model calls and model-seconds per verified engagement. Today: about 6 calls and about 64 s per failed attempt.
- Raw LIBERO success.
- Video-excluded time to success.

**Secondary metrics**
- Output tokens per call (target ≤250).
- First-move sign correctness: 15 of 55 first moves were wrong in no-demo bowl runs.
- Episodes ended by errors.

## 1. Rulings needed from you (before Phase 2/3)

| ID | Question | Default if you don't rule |
|---|---|---|
| R-a | May the policy use (i) wrist-image axis text derived from the robot asset (`robot.xml`: wrist image-up = R[:,0]; jaw always image-horizontal), and (ii) an image→world sign table learned online from measured state8 motion? Both arguably count as "calibration". | (i) yes, it is a robot fact, not scene calibration; (ii) no |
| R-b | May object-specific constants tuned on evaluator data be used live (rim/bar gap bands 3.3–7.7 / 16.5–17.8 mm, fixed descent depth, hold thresholds 3.8/2.1/0.3 mm), labelled? | No. Use only robot/online constants: a free-air close to learn the empty-close curve, pad extent −11.6…+4.4 mm (`robot_tool_geometry.py:47-52`), and measured co-motion |
| R-c | One automatic fresh retry for a short diagnostic trial that ends on a subscription error (about 0.17%/call). This is a PROTOCOL CHANGE against "no automatic retry". | No |
| R-d | Fixed full-task budgets (e.g. 25 min / 1,200 actions) instead of no cap. This is a PROTOCOL CHANGE. | No (keep no cap) |

## 2. Phases

### Phase 0: Offline ledger, settled decisions, frozen baselines (0 calls, no simulator, about 2 days)

**Work**
- **`scripts/close_ledger.py`** (about 250 LOC + 60 test). It reuses `runs/contact_routing_diagnostic_001/preparation/analyze_results.py:96-188`, `scripts/analyze_bowl_chain.py`, `scripts/audit_hybrid_subscription.py:401`.
  - **Per close:** pre-close pose with world jaw axis R[:,1] and approach axis R[:,2]; commanded vs realized motion; finger gap at k=5/10/15; the readiness the model declared; the state8 outcome; the evaluator miss vector where `evaluator.jsonl` exists (DIAGNOSTIC).
  - **Per call:** phase, controller, operation kind, output tokens, latency, how the Field call ended.
  - **Aggregate:** model-seconds per phase (approach, pre-close, failed-close recovery, transport, place).
- **Decisions the existing data already settles** (no longer gates):
  - Build the scripted close/verify. Failed-close cycles and post-close checks took 24.5% of v3's model time.
  - Gate the free close on "supported": 20 of 20 "uncertain" closes failed, and the 1 "supported" close captured.
  - Take Field chaining off the critical path. There were 0 `field_boundary_reached` events in v3 bowl, rg1 bowl and v2_002; Field calls end in `goal_reached`.
- **Calibrate state8 detectors** (in-sample; call this calibration, not validation):
  - empty close: gap ≤3.0 mm at k=15
  - touch: one finger moves inward >0.10 mm per step while commanded open
  - stall: realized/desired ratio <0.08 over 3 steps, disabled while rotating ≥0.05 rad/step. Re-measure under guarded descent in Phase 1 before freezing.
- **Choose checkpoint K3 correctly.** It must be a drawer-open state a few steps before a bowl close onset:
  - `attempt_memory_drawer_zero_shot_002`, before step 425 or 485
  - or `robot_geometry_stage1_drawer_zero_shot_001`, before step 381 or 565
  - Not `contact_region_drawer_zero_shot_001` step 460, where the hand is still at the pulled handle.
- **Choose checkpoint K4:** `contact_region_drawer_zero_shot_001` step 396, handle in the frontal, vertical-jaw pose.
- **Frozen baselines**, defects kept as they are:
  - B0-bowl = `runs/speed_first_v3_bowl_zero_shot_001/launch.json` argv
  - B0-drawer = `robot_geometry_stage1_drawer_zero_shot_001` argv
- **Snapshot workflow.** `verify_manifest` re-hashes all of `src` before every trial (`evaluate_contact_routing_diagnostic.py:287-295`). So develop in a separate git worktree, and run every simulator or live bundle from a frozen copy that is not edited while it runs.

### Phase 1: Oracle grasp basins (APPROVAL #1: simulator only, 0 model calls; about 106 trials, 80–145 min)

**Work**
- **`preparation_v2/extract_checkpoints.py`:** a copy with a new SELECTION, because the original hard-codes it and refuses to overwrite. It extracts:
  - K1 = bowl_b, K2 = bowl_a
  - K3 = the new drawer-open bowl state
  - K4 = handle, frontal pose
  - H = handle_a, wrong orientation
- **`scripts/oracle_grasp_basin.py`** (about 150 LOC). Scripted actions come from `sim.evaluator_metrics` (`evaluate_contact_routing_diagnostic.py:223`), and every output is labelled ORACLE. It needs an opt-in action limit above 200 (`contact_diagnostic_sim.py:141-144`). It records a post-engagement tail (bowl lift 25–30 mm, handle pull 20 mm), because the harness otherwise stops at engagement (`evaluate_contact_routing_diagnostic.py:190-193`).

**Sweeps**

| Sweep | Values |
|---|---|
| Lateral offset, along and across the jaw | {0, ±5, ±10, ±20} mm, plus ±40 mm on one axis (observed misses are 16–72 mm) |
| Bowl grip height relative to the rim | {+8, 0, −5, −10, −15, −20, −30} mm |
| Handle jaw-to-bar angle | {90, 60, 45, 30, 0}° |
| Drawer partial pull from K4 | {80, 120, 160} mm, then a scripted bowl grasp: does an over-pulled drawer block the bowl? |
| Touch-referenced procedure | Descend to touch or stall, back off 3–5 mm, shift **both signs**, descend; plausibility check of touch height against a table height found by touch; a touch-based lateral search. This covers the blind spot where a fingertip lands straight on the rim or the hand is centred and first touches the bowl floor (about 45 mm below the rim top) |
| Handle reorientation from H | Retreat, return to the start orientation, then rotate about x. From H the frontal pose is 120° away about a skewed axis, so direct in-place rotation is borderline (55–67 steps) |

**Pre-registered rules**
- **R1 (bowl lateral half-width W)**
  - W ≥10 mm and the touch procedure captures from seeds 0 and ±10: build the full primitive.
  - 5–10 mm: full primitive with a mandatory touch search.
  - <5 mm, or the touch procedure fails at seed 0: build only the scripted close/verify, and bring tiers T1/T4 to you.
- **R2 (depth window)**
  - ≥15 mm: fixed descent below first touch.
  - 8–15 mm: descend to the window centre, with one hold-test retry.
  - <8 mm: adaptive touch-referenced stop.
- **R3 (handle jaw angle).** Set the jaw-rejection threshold where captures stop.
- **R4 (classifier).** Across all sweep closes, 0 false accepts and ≤10% false rejects before the scripted hold test may replace the model's post-close check.
- **R5 (drawer pull).** Set the pull bound to the longest partial pull that still allows the bowl grasp. Implement it through the model's `max_travel_m`, not "stop at stall", because that stops at the end stop, which is the over-pulled state.
- **R6 (reorientation).** The retreat-and-rotate path must reach within 5° of the frontal pose without a stall. Otherwise use K4 only.

### Phase 2: Interface fixes as opt-in flags (implementation; APPROVAL #2 for about 90 calls, no simulator)

The audited modules must stay byte-identical (`audit_hybrid_subscription.py:135-153`: `action`, `astra`, `chatgpt_plan`, `compact_subscription`, `direct_history`, `observation`, `planner`, `schema`, …). Every fix therefore goes in a new module.

| Flag | Implementation |
|---|---|
| `--policy-image-orientation upright` | New conversion function called from `libero.py:951-954` (not audited). Adds `legacy_equivalent_sha256` to the fingerprints. Rejects demonstration/history continuation across orientations |
| `--wire-key-order insertion` | New transport module injected via `execute_episode(transport=)`, importing `_open_public`/`read_completed_response`. The schema puts a leading `contact_geometry` field first (target and fingertip relation, estimated offset, chosen strategy). Wire order is added to the `request_key` material (`astra.py:465-467`) |
| `--derived-tool-axes` | `build_payload` override in the subclass. Prints world jaw, approach and wrist-up axes; avoids the >3 rad axis-angle jumps |
| `--robot-geometry-guidance-version v2` | "The jaw is image-horizontal in the wrist view; a pinchable bar appears vertical between the fingers" (subject to R-a) |
| `--action-gain-text measured_v1` | About 22% realized per 5-step chunk. Only matters where Direct is used |
| *(dropped)* zero-rotation text flag | The false claim exists only in legacy, audited prompts |

**A/B as a paired non-inferiority check** (`scripts/interface_probe_ab.py`, about 150 LOC)
- Six payloads (K1–K4 contact states, plus the init-0 start), rebuilt with fresh history through the **Phase 4 arm builder**, not the old diagnostic wrapper. Un-mirroring the saved images is pixel-exact.
- Conditions: {legacy, fixed} × 6 × 5 samples (about 60 calls), plus {full, slim} on the 3 contact payloads × 5 (about 30 calls). About 40 min.
- Scoring (DIAGNOSTIC), paired by payload:
  - first-move sign on axes where the true offset is >1 cm, measured against strategy-aware targets (a rim-pinch pregrasp sits about 50 mm off the bowl axis)
  - handle jaw-to-bar angle ≥60°
  - intended grip height relative to the rim
  - output tokens and latency
  - schema failures
- **Rule.** Adopt the fixes unless they are worse by more than 25–30 points (this N cannot resolve smaller differences). Adopt slim if it is non-inferior on the same margin and the median is ≤250 tokens.

### Phase 3: Build the contact primitive on a lean base, then verify in the simulator (implementation; APPROVAL #3: simulator only, 0 calls)

**Base**
- `ContactPrimitivePolicy` on `SpeedFirstV3` (or on v4 with operation_visual, contact_region and attempt_memory bypassed), with a lean schema.
- Arm A0 = the same base without primitives, so A3 − A0 is one factor.
- Not on `ContactRegionPolicy`: `--contact-region-feedback` forces v3 + attempt memory + pregrasp feedback (runner `:1224-1227`), and v4 imports `OperationVisualFeedback`.

**Core module `src/gpt_field/contact_primitive.py`**, offline-testable:
- touch detector (single-finger inward deflection while open; also says which finger)
- stall detector
- empty-close detector learned online from one mandatory free-air close
- co-motion hold test: bowl lift 25–30 mm at ≤3 mm/step; handle pull ≤ `max_travel_m`, stopping on a stall or a gap jump
- object-specific bands only if you approve them under R-b

**Model outputs** (a discriminated `anyOf` branch, pattern from `contact_diagnostic_policy.py:47-93`):
- `strategy ∈ {rim_pinch, frontal_bar_pinch}`. `hook_and_pull` is demo-derived and excluded unless you rule otherwise.
- `side`, `long_axis`, `approach`, `max_travel_m`
- `readiness ∈ {supported, uncertain}`
- The free `grasp_close` requires a dedicated single-value enum readiness field. The gate is enforced through the schema, never by rejection, because a SchemaError ends the episode (`chatgpt_plan.py:560-569`).
- `uncertain` routes to the touch-gated probe, not a blind close.
- Log how often gated closes capture, since the gate may stop predicting once it is the price of closing.

**Code**
1. An axes→quaternion helper; reject any jaw within R3's angle of the long axis.
2. Rotate first, then translate. Use the retreat-and-rotate path for large turns.
3. Guarded descent along R[:,2] at ≤2.5 mm/step.
4. Interpret the touch, using R1/R2 from Phase 1.
5. Settle close with a larger pose-correction cap (configurable up to 0.05 m; the 10 mm cap let the bowl move 6 mm).
6. Hold test.
7. One local retry in checkpoint trials. A full attempt is about 120–150 actions.
8. Emit one result record into the next payload. Model calls inside the primitive: 0.

**Prompt edits, in the new arms only**
- Remove "Repeated attempts are permitted" (`operation_contract.py:242`).
- Remove "within 3cm" (`operation_contract.py:351`).
- Remove the diagnostic wrapper's "Prefer an explicit grasp_close … when ready to test capture" (`contact_diagnostic_policy.py:124-125`).

**Wiring**
- `--contact-primitives` runner flag with `_contact_primitive_metadata()`, which records `oracle_derived_constants`, `tuned_thresholds` and `extra_model_calls: 0`.
- Allow-lists: `speed_first_history.py:150-153`, `attempt_memory.py:370`, `hybrid_video.py:88-109/201/239`, `speed_first_v4.py:472-474`, `distance_subscription.py:28-32,716-718`.

**Harness**
- N×cell orchestrator replacing the hard-coded LIMITS/ARMS (`evaluate_contact_routing_diagnostic.py:27-28,306,361`).
- Stub client.
- Arm adapters for A0/A1/A3 (about 150–250 LOC).
- Opt-in action cap of 350, applied equally to all arms.
- Worker op: replay in the legacy orientation, check the fingerprint, then switch to upright.

**Size:** about 1,500–2,000 LOC including tests.

**Simulator verification** (stub client, about 40 trials, 30–40 min)
- Start from K1–K4 at the oracle pose plus ledger-observed errors (ORACLE).
- Must hold: capture plus hold at zero offset on every target; 0 false accepts; ≤0.3% free-space false triggers; result record present in the next payload (offline test).
- Any failure: fix before any live call.

### Phase 4: Live checkpoint factorial (APPROVALS #4a and #4b; each includes its activation trial)

**Design**
- Fresh history; one interface across all arms (the Phase 2 result).
- Checkpoints K1, K2, K3 (bowl-type) plus K4 (handle, descriptive).
- Caps: 20 requests and 350 actions per trial, equal for all arms.
- Primary outcome: evaluator engagement as a binary per trial (DIAGNOSTIC), plus the hold tail. Report per checkpoint, because trials cluster by checkpoint.

**Activation.** One A3 trial at K1. It must show `primitive_started > 0`, logged touch and stall events, and the result record in the next payload. Otherwise stop. About 12 min.

**4a: A0 vs A3**, N=3 per checkpoint (9 bowl trials per arm). About 250–290 typical calls (≤480), about 3 h.
- **Go:** A3 − A0 ≥4/9 engagements **and** A3 model-seconds per engagement ≤50% of A0's. Then go to 4b and Phase 5.
- **Middle zone** (A3 ahead by 2–3): extend to N=6 per checkpoint. Request this together with #4a.
- **A3 ≈ A0** (difference ≤1) while Phase 3 succeeded at the oracle pose: the bottleneck is the model's discrete choice or pregrasp. Next steps are mark selection (numbered candidate points, PIVOT/MOKA style) or a touch-based lateral search, and tiers T1/T3 come to you. No full tasks.

**4b: A1** (model pregrasp plus scripted close/verify only). About 150–200 calls. Ship A1 instead of A3 if A1 is within 1 engagement of A3: simpler and cheaper.

### Phase 5: Full-task confirmation (APPROVALS #5a, #5b; the init 1–2 simulator resets need a separate simulator-only approval)

**Arms.** B0 (the frozen baselines) vs B1 (the Phase 4 winner plus the adopted interface flags).

**5a.** Init 0, both tasks, N=3 per arm: 12 episodes, about 530–670 calls, about 3.5 h.
- **Gate:** B1 bowl ≥2/3, and on the drawer the bowl is lifted after the drawer opens in ≥2/3.

**5b.** Inits 1–2, both arms: 24 episodes, about 1,070–1,330 calls.
- **Prerequisites:**
  - simulator resets to capture the initial-observation SHAs (the offline preflight uses a synthetic image)
  - re-check `target_identity_guidance.py` against the new images
  - re-validate the detectors on that init's ledger
- **Power.** With N=9 per arm per task, a "more reliable" claim needs about 8/9 vs 3/9 on the bowl, or ≥4/9 vs 0/9 on the drawer (one-sided Fisher, init-stratified). Otherwise report Wilson CIs with no attribution.
- **Speed claim:** median video-excluded time to success is lower **and** successes are no fewer than B0's minus 1.

## 3. Speed track (after Phase 4; each item its own factor)

1. **Output length.** The lean base plus slim schema targets ≤250 tokens/call. Latency ≈ 1.84 s + 36.7 ms/token, so 550 → 250 tokens saves about 11 s per call.
2. **Zero-model local retries** are built into the primitive. This removes the about 6 calls per failed attempt.
3. **Distant-goal Field.** This needs prompt permission for distant goals (the schema already allows ±0.5 m, `compact_field.py:61`), local clipping to an intermediate goal inside the ±80 mm cube, re-centring on arrival, and a larger action budget. It is a prompt factor with its own activation and approval. The lever is modest: about 3 capped-continuation calls per run.

## 4. Constraint-relaxation tiers (each a CONSTRAINT CHANGE; presented only when its trigger fires)

| Tier | Trigger |
|---|---|
| T1: wrist mount + intrinsics (eye-in-hand metric servoing) | R1 fails, or A3 ≈ A0 |
| T2: joint position/velocity, already supported (`observation.py:19-21`) | Unexplained stalls |
| T3: effort=high only for primitive-selection calls | A3 ≈ A0 |
| T4: depth + LIBERO camera matrices; back-project the chosen mark into an analytic grasp | Drawer-bowl capture still low after 5a |
| T5: demo-trained low-level contact policy under Astra routing | Long-term option |

## 5. Stop doing

- N=1 comparisons and hand stops.
- Stacking mechanisms that were never shown to fire. Visual references: 70 unknown / 4 contradicted / 0 supporting.
- Extending `operation_visual.py`, `contact_region_feedback.py` or `attempt_memory.py`.
- Letting the model author quaternions and depths.
- Closing as a free probe.
- Growing the prompt (10.5k → 20.7k characters).
- Prioritising the handle on the drawer task. The bowl after the drawer opens is what kills it.

## 6. Timeline

| Week | Work | Approvals |
|---|---|---|
| 1 | Phase 0; Phase 1 harness + K3/K4; Phase 2 flags; Phase 3 core. Freeze a snapshot, then run Phase 1 (about 1.5–2.5 h of simulator) and the Phase 2 A/B (about 40 min) | R-a…R-d rulings; #1, #2 |
| 2 | Phase 3 wiring, harness, adapters; simulator verification | #3 |
| 3 | Phase 4: activation, 4a, optional middle-zone extension, 4b | #4a, #4b |
| 4–5 | Phase 5a, decision, 5b; report | #5a, #5b, simulator-only resets for inits 1–2 |

**Live model cost before any full task:** about 90 (A/B) + about 400–500 (factorial; up to about 960 at the caps, more if the middle zone extends).
