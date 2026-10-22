# agent_SpeedUp_v1: model-supervised skills with sensor-grounded contact

**Status:** implemented 2026-10-16 (see §17 for the implementation map and the changes driven by real-LIBERO smoke tests). Written 2026-10-16 and revised after independent reviews of its references, geometry/physics, and implementability. Not yet run with live model calls.

**Relation to [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md):**
- Plan Phases 0–3 build and test the shared contact primitives inside the speed_first v3 base. The plan calls the module `contact_primitive.py`; here it is `contact_skills.py`.
- agent_SpeedUp_v1 is a later arm, **B2**. It adds three things to those primitives: the event-driven supervisor, mark pointing, and wrist-image Align.
- Before any full task, B2 needs two phases that the plan does not yet contain (§13):
  - **Phase 3b:** simulator only, zero model calls.
  - **Phase 4b:** a live checkpoint arm.

**Labels used for numbers:**
- **(measured):** taken from recorded runs or code.
- **(design default):** an initial value, to be tuned.
- **(projection):** not yet measured.
- **(rationale only):** derived from scene geometry the policy never sees. It explains a design choice and is never used at run time.

Current method: [METHOD_AND_ARCHITECTURE.md](METHOD_AND_ARCHITECTURE.md).

## 1. Summary

The hosted model (gpt-6-astra, medium, subscription) is a **supervisor**:
- It chooses **macro skills** and their **discrete parameters**: target, grasp strategy, side, approach direction, bounded travel, coarse free-space waypoints.
- It answers short **pointing** requests by selecting numbered marks.
- It never writes contact quaternions, millimetre offsets, or descent depths.

A local **skill executor** runs each macro skill as a fixed sequence of closed-loop primitives at the simulator rate (20 Hz). There are no model choice calls inside a skill.

Each contact quantity goes to the sensor that can observe it:

| Quantity | Observed by | Resolved by |
|---|---|---|
| Object, part, strategy, side, approach direction | model (semantics) | discrete choice; numbered marks instead of coordinates |
| Gripper orientation | strategy geometry | fixed per strategy and computed in code |
| Lateral alignment (target relative to the grip axis) | wrist camera, 2-D | uncalibrated image-based visual servoing; image Jacobian estimated online |
| Depth along the approach | touch on a supported surface | guarded move until stall or finger deflection, then a small robot-defined retreat |
| Grasp success | physics + wrist view | finger gap against an online empty-close curve, plus the object staying fixed in the wrist view during a short scripted lift or pull |
| Free-space motion | existing Field / waypoint planner | unchanged |

**Where the speed comes from:**
- one choice call per macro skill;
- short outputs;
- failed grasps retried locally without a choice call.

## 2. Motivation: what the current system shows

These findings come from recorded runs. Evaluator-only ground truth was used for diagnosis only, never as policy input.

- **Contact fails per attempt, not every time.**
  - Zero-shot bowl runs reached raw LIBERO success 9 times.
  - The drawer opened in 5 of 9 drawer runs, but no drawer task completed.
  - In those runs, the bowl grasp *after* the drawer opened was the step that failed.
  - A failed grasp attempt costs a median of about 6 model calls and about 64 s of model time (45 cycles) (measured).
- **Free-form numeric grasp geometry is often wrong.**
  - The handle bar is 89 mm long, longer than the 79 mm opening. A frontal approach with the jaw vertical captured 6 of 8 closes; all other configurations captured 0 of 25. Many of those 25 also had large position errors. At episode level: 5/5 vs 0/4, Fisher p = 0.008.
  - op_v4 said "vertically separated jaws" but emitted a quaternion with the jaw along the bar, then closed on air four times.
  - The model centred the hand over a bowl (about 107–112 mm) wider than the opening.
- **Depth along the approach axis is misjudged in both directions.** Bowl closes happened in free air above the rim. In handle_a the bar reached the palm without the fingers capturing it.
- **The executor alone is not the fix.** In the paired diagnostic, the Field-tracker arm failed the same way as Direct5. The missing pieces are the *target relation* (where the fingers must go relative to the rim or bar) and *depth by touch*.
- **Closing is an ungated probe.**
  - Of 20 diagnostic and op_v4 closes the model labelled uncertain (19 on alignment), none captured cleanly. One became a slipping pinch.
  - The single close labelled "supported" captured. That is n=1.
  - Failed closes pushed the bowl about 22–36 mm.
- **Visual self-checks did not help.** The 14 body-ROI references in the routing diagnostic produced 70 unknown, 4 contradicted, and 0 supporting observations. The state8 finger gap reveals an empty close within about 10–15 actions.
- **Model time dominates.**
  - Model time is 79–98% of video-excluded time across runs.
  - Client call time ≈ 3.0 s + 38.5 ms per output token, fitted on client latency (server-side: 1.84 s + 36.7 ms, R² 0.876, n=515).
  - Median output: 523 tokens in op_v4 and 550 in contact_region, against 177 in speed_first v3.
- **Interface defects in every run so far:**
  - Policy images are mirrored left-right (robosuite `opengl` convention plus `pixels[::-1, ::-1]`, `observation.py:31`).
  - The schema is sent with sorted keys (`astra.py:118`), so actions are emitted before reasoning.
  - No prompt maps image directions to world axes.
  - In runs launched with `--robot-geometry-guidance` (rg1 onward), the wrist-geometry text is also wrong (`robot_geometry_guidance.py:31-33`).

**Literature.** Every surveyed zero-shot system grounds contact in metric 3-D: calibrated RGB-D, known-camera triangulation, or simulator state. Contact is then executed by a deterministic or learned module.
- Most systems let the model choose only objects, parts, keypoints, marks, or constraints.
- A few let it emit poses or mm-level moves, always on top of metric grounding:
  - Gemini Robotics-ER: 25% zero-shot in the real world.
  - MotorMind: 66.7% on LIBERO-Pro.

v1 keeps this division of labour. Under the no-depth / no-calibration constraint it replaces 3-D lifting with **touch** and with **image Jacobians estimated online**.

## 3. Design principles

1. **The model picks; sensors measure.** No metric contact quantity comes from the model.
2. **Event-driven, macro-level supervision.** The model is called when a macro skill ends (success, or failure after local retries) and for pointing. It is never called every N actions, and never between the primitives of one skill.
3. **Sensor-gated close.** The gripper closes only inside `grasp_with`, after the strategy's approach and depth steps report their sensor events. The model cannot ask for a bare close.
4. **Physical verification.** Success comes from the finger gap plus wrist-view co-motion, not from the model's visual opinion.
5. **Short, stateless requests.** Each request carries the current images, a state summary, and a compact outcome ledger. Targets: choice outputs ≤200 tokens, pointing ≤100.
6. **Every constant is labelled** ROBOT / ONLINE / TUNED / ORACLE (§10). ORACLE values never enter the policy.
7. **Opt-in, legacy-preserving.** New modules sit behind `--agent-speedup-v1`. Audited legacy modules stay byte-identical (`audit_hybrid_subscription.py:135-153`).

## 4. Architecture

```
 Supervisor (hosted model; stateless requests; event-driven)
   decision request: upright external + wrist RGB, state summary, ledger, last outcome
       -> {situation, skill, params}                     (<=200 tokens)
   pointing request: wrist image with numbered marks -> {id | null}   (<=100 tokens)
          │                                         ▲
          ▼                                         │ outcome record
 Skill executor (20 Hz closed loop; no choice calls inside a skill)
   macro skills:  reach | grasp_with | pull | transport | place_release | retreat
   primitives:    Calibrate, Orient, Align, GuardedApproach, Grasp, VerifyHold, Pull, Lower, Release
          │                                         ▲
          ▼                                         │
 Signals: wrist/external RGB; state8 (eef pose, finger joints)
          derived: tool axes, finger gap, touch flag, stall flag, image Jacobian, NCC tracks
```

**Step protocol.** Every model call happens inside `act()`, the existing runner contract.

```text
episode start: Calibrate (free-air close + open; zero model calls)
act(obs):
    if a primitive set needs_pointing on the previous step:  pointing request(s)
    if no macro skill is active:                             decision request -> start macro skill
    action <- active primitive.next_action(obs)
    if the primitive finished with no action: advance to the next primitive / macro skill
        (at most 2 decision calls per act()); if still none, return
        measured_current_pose_hold_action()   (visual_field_v8.py:24)
    report api_latency_s = sum of the calls made in this act()
observe_transition(before, decision, after):
    primitive.observe_transition(before, action, after)  # may set needs_pointing or done
    on macro-skill end: append its outcome record to the ledger
```

LIBERO success can arrive during any step. The runner checks it every step and stops.

## 5. Supervisor contract

### 5.1 When the model is called

| Request | Trigger |
|---|---|
| Decision | episode start (after Calibrate); macro skill ended successfully; macro skill failed after its local retries |
| Pointing | start of each Align (after any Orient); tracked mark lost (NCC peak < 0.6 for 2 consecutive frames; design default), at most 2 re-points per Align |

There is no fixed refresh cadence.

### 5.2 Decision-request inputs

- **Images.**
  - **Orientation fix.** External and wrist images are flipped horizontally in the client, which makes them upright: a vertical-only flip of the raw frame.
  - This is pixel-exact at native 512 px because no resize happens.
  - The worker, traces, initial-observation hashes, and checkpoint fingerprints stay in the legacy orientation.
  - Executor pixel coordinates use the upright frame: `u_legacy = 511 − u`.
- **State summary.**
  - eef position;
  - world-frame jaw axis R[:,1] and approach axis R[:,2], where R is the orientation of the `right_hand` body from state8;
  - finger gap in mm.
- **Robot facts (ROBOT; ruling R-2).**
  - the jaw is image-horizontal in the wrist view;
  - a bar that can be pinched appears vertical between the fingers;
  - maximum opening is 79 mm.
- **Ledger.** One line per finished macro skill, capped at the last 12 lines plus counts, and the last outcome record (§6.9). There is no per-action feedback and no image history.

### 5.3 Decision output

The schema is one flat strict object with keys in this order, sent unsorted:

```json
{
  "situation": "string, maxLength 280 — what is visible; target relative to the fingers",
  "skill": "reach | grasp_with | pull | transport | place_release | retreat",
  "params": {
    "target":            "enum of task objects/parts",
    "strategy":          "rim_pinch | frontal_bar_pinch | null",
    "side":              "wrist_left | wrist_right | null",
    "approach":          "world_-y | world_+y | world_-x | world_+x | null",
    "waypoint_offset_m": "[x,y,z], each within ±0.075 | null",
    "max_travel_m":      "0.02–0.16 | null"
  }
}
```

**Parameters used by each skill.**

| Skill | Parameters |
|---|---|
| `reach`, `transport` | `waypoint_offset_m` |
| `grasp_with` | `target`, `strategy`, plus `side` (for `rim_pinch`) or `approach` (for `frontal_bar_pinch`) |
| `pull` | `max_travel_m` |
| `place_release` | `waypoint_offset_m` (hover above the destination) |
| `retreat` | none |

- `situation` comes first, so the model states the geometry before choosing.
- There is no `close_readiness` field. Closing is gated by sensors (§3.3), so the model's readiness claim is not needed.
- The model does not declare completion. The episode ends only on raw LIBERO success, which is the existing protocol.

### 5.4 Pointing requests (coarse to fine, Set-of-Mark style)

Pointing is a separate, stateless request type, issued only by the executor.

1. **Coarse.** The upright wrist image with an 8×8 grid of 64 px cells, ids 1–64 in row-major order from the top left. Output: `{cell_id | null}`.
2. **Fine.** A 96×96 px crop (the chosen cell plus a 16 px margin, clamped), upsampled ×5, with 8×8 numbered points.
   - Point spacing is 12 px in the original image, so error is at most ±6 px, about ±3 mm at a 6–8 cm hover (rationale only).
   - Pixel = crop origin + ((col + 0.5)·12, (row + 0.5)·12).
   - Output: `{point_id | null}`.

Marks are drawn on a copy; NCC tracking always runs on the unmarked image. A `null` ends Align with `mark_not_visible`.

## 6. Primitives

Each primitive implements:
- `next_action(obs)`
- `observe_transition(before_obs, action, after_obs)`
- `done`
- `outcome_record()`

This extends `LocalGripperSettle`'s protocol (`speed_first_gripper.py:121-190`) with images. Orientation is held with an explicit measured target on every step. A command with exactly zero rotation would keep a stale OSC goal (robosuite `osc.py:259-263`).

### 6.1 Calibrate (episode start; zero model calls)

1. Close at the start pose using `LocalGripperSettle` (≤30 actions).
2. Open and settle (≥15 steps).
3. Record the **empty-close curve** gap(k) for k = 1–30 and the inward finger-step noise σ while settled open.
   - Recorded empty curve: 50.6 / 8.4 / 2.35–2.70 / 1.9 mm at k = 5 / 10 / 15 / 20 (measured).
   - Touch threshold = max(0.10 mm, 3σ) (ONLINE; the recorded noise floor is −0.044 mm).

### 6.2 Orient (frontal_bar_pinch only in v1)

- **Target.**
  - Approach axis = the chosen horizontal world direction.
  - Jaw = world ±z, whichever sign is nearer the current R[:,1].
  - The `right_hand` rotation matrix is built from these axes by a new axes→quaternion helper (none exists today).
  - Reference: approach −y with jaw +z is `[0.7071, 0, 0, 0.7071]` (xyzw). All 6 recorded handle captures were within 2.5° of it.
- **Control.**
  - Rotate with translation held, through `ActionAdapter.adapt` (`action.py:59-93`; cap 0.15 rad per step at `:48`).
  - Rotation realizes about 1.8–2.2° per step (measured), so 90° takes about 45–50 steps.
  - Budget: ⌈angle / 1.8°⌉ + 15 steps.
- **Large turns.** For turns >90°: retreat 50 mm along −R[:,2], return to the start orientation, then rotate.
  - The frontal pose is a pure 90° x-rotation from the start pose, a path observed in 6 runs.
  - A skewed in-place turn in contact_region stalled for about 60 steps.
- **Self-check.**
  - After orienting, the target must be visible in the wrist image and must appear vertical.
  - Otherwise: outcome `approach_inconsistent`.

### 6.3 Align (uncalibrated wrist-image servoing; needs rulings R-1 and R-2)

| | |
|---|---|
| Feature | `s` = upright-image pixel of the selected mark, tracked by NCC (21×21 px patch; design default) |
| Jacobian | 2×2 `J`, mapping in-plane eef translation (along R[:,1] and R[:,0]) to pixel motion. Initialized by two orthogonal ±2 mm probe moves (measured with state8). Broyden rank-one update on steps with no approach motion and measured rotation ≤0.2° |
| Depth-free setpoint | `s* = c + κ·P·o` (see below) |
| Command | `Δp⊥ = −λ J⁺ (s − s*)`, with λ = 0.5 (design default); ≤3 mm realized per step; anchored correction in the style of LocalGripperSettle |
| Stop | ‖s − s*‖ ≤ 6 px for 3 consecutive frames (design default) |
| Fail | no convergence within 60 steps (probes excluded); more than 2 re-points; stall → `align_no_convergence` / `mark_lost` |

Terms in the setpoint formula:

| Symbol | Meaning |
|---|---|
| `c` | principal point = image centre (256, 256) |
| `P` | fixed mount mapping: image right = R[:,1], image up = R[:,0] |
| `o` | lateral offset of the desired contact point from the camera's optical axis. Grip axis: o = (0, −0.05 m), because the camera is mounted 0.05 m along R[:,0] (`robot.xml:239`) |
| `κ` | f/d, read from the online Jacobian (J ≈ −κ·P for pure lateral translation) |

**Why the setpoint needs no depth.** `s*` is recomputed after each Jacobian update. Along the jaw axis the camera has no offset, so that component is exactly `c_u` at every depth. Along R[:,0] it scales with κ.

**Align–descend ladder.**
- GuardedApproach owns the descent. Every 20 mm it pauses for one Align refresh: a κ re-probe, then a servo to `s*`.
- Refreshes stop once `s*_v > 380 px` or the NCC peak drops below 0.6, because the fingertips occlude the mark at rows ≥ ~420.
- After that, lateral position is held fixed with no image feedback.
- Do not servo a mark onto a fingertip's own pixel.

### 6.4 GuardedApproach (depth by touch)

| | |
|---|---|
| Command | a fixed 0.2 normalized command along R[:,2] (10 mm commanded per step; about 2.6 mm realized at the recorded free-motion ratio of 0.26). No adaptive gain. Lateral position held by an anchored correction capped at 3 mm |
| Stall flag | realized/commanded ratio over 3 steps < 0.3 × this move's own free-motion ratio, measured in its first free steps (ONLINE; about 0.08 at 0.26). Disabled for the first 2 steps, while rotating ≥0.05 rad per step, and during gripper settle |
| Touch flag | one finger moves inward by more than the Calibrate threshold in one step, while commanded open and settled ≥15 steps after reopening (ONLINE). It also tells which finger |
| Bound | 150 mm (design default); exceeding it → `no_contact_within_bound` |
| Output | contact type (stall / touch / finger), contact pose |

**Coverage on recorded traces** (open-loop Direct/Field motion; measured):
- Touch: 27 of 64 contact onsets within 3 steps, and 8 of 17 finger-on-bowl/handle onsets.
- Stall: 48–51% of table/cabinet contact windows, but only 36–38% of bowl/handle windows, because a free object slides or tips.
- Stall or touch: 56 of 60 onsets of all kinds within 12 steps (46 within 2 steps).

**Design consequence.** Both v1 strategies are built so that the expected first contact is with a supported surface: the bowl floor inside its support radius, or the cabinet face. There, stall is most reliable. Coverage under the v1 command law is a Phase 3b output.

### 6.5 Grasp (sensor-gated settle close)

| | |
|---|---|
| Precondition | the strategy's approach and depth steps (§7) completed, with no pose change since beyond 2 mm / 2°. Otherwise → `precondition_unmet` (the executor never closes) |
| Control | constant close with pose hold, via `LocalGripperSettle` (`speed_first_gripper.py:66-165`). Correction cap raised from 10 mm to 0.03 m (design default; max 0.05). The 10 mm cap gave too little holding stiffness: one finger's reaction displaced the hand about 6 mm |
| Classify | object present if gap(k=15) > empty_curve(15) + 0.5 mm (TUNED-initial; recorded empty spread 0.35 mm, rim margin 1.35 mm). Settle time is a second cue: empty closes never stabilize before k ≈ 27 |
| Outcome | `empty_close`, `object_between_pads` (gap, settle k), or `stalled_close` |
| Note | "object between pads" is not "held": 11 of 27 such closes were later lost (measured). VerifyHold always follows |

### 6.6 VerifyHold (scripted hold test)

| | |
|---|---|
| Motion | bowl: lift 25 mm at ≤3 mm per step, anchored. Handle: pull 20 mm along −R[:,2], counted inside `max_travel_m` (design defaults) |
| Pass | gap ≥ empty_curve + 1.0 mm throughout (TUNED-initial) **and** the object patch next to the fingers (an NCC template taken at grasp) stays within 10 px of its grasp position in the wrist view (design default). A carried object stays fixed in the wrist view; one left behind recedes |
| Fail | gap reaches the empty level, or the patch recedes. A slipping object counts as a fail in v1 |
| Note | state8 alone cannot show co-motion. Hand progress detects only a jam. The evaluator-fitted 0.3 mm gap-stability rule is used only under ruling R-3 |

### 6.7 Pull, Lower, Release

- **Pull.**
  - Direction: −R[:,2], back along the frontal approach.
  - Stops at `max_travel_m` (measured from the grasp pose), on a stall, or on a single-step gap jump ≥0.2 mm.
  - At the drawer end-stop the recorded gap jump was 0.2–1.5 mm (measured).
  - The travel bound exists because stopping only on a stall pulls to the end-stop, about 170–180 mm of hand travel.
  - In contact_region, commanded pulls of 15–25 cm were followed by a bowl phase with 0 closes in 15 calls. Over-pulling is the suspected cause, a hypothesis that plan Phase 1 tests.
- **Lower.** A guarded move along −z with the GuardedApproach command law, until stall.
- **Release.** Open, settle ≥15 steps, then retreat 30 mm along −R[:,2].

### 6.8 Local retries (no choice call)

- Each `grasp_with` has a `local_retries` budget: 1 in checkpoint trials, 2 in full tasks.
- On `empty_close` or a VerifyHold failure:
  1. reopen and settle (≥15 steps);
  2. retreat along −R[:,2] to the hover height;
  3. re-run Align on the **existing template**, with the setpoint shifted δ = 10 mm along the jaw toward the target centre (design default);
  4. GuardedApproach, then Grasp.
- A pointing call happens only if the track is lost.
- When the budget is spent, the failure record goes to the model.

### 6.9 Outcome record

The outcome record replaces per-action feedback in requests.

```json
{"skill": "grasp_with", "strategy": "rim_pinch", "status": "failed",
 "reason": "empty_close",
 "evidence": {"gap_mm_k15": 1.4, "empty_curve_k15": 2.5, "settle_k": null,
              "contact": "stall", "finger_deflected": null, "local_retries_used": 1},
 "pose": {"eef_xyz": [0.0, 0.0, 0.0], "jaw_axis": [0, -1, 0], "approach_axis": [0, 0, -1]},
 "calls_inside_skill": {"decision": 0, "pointing": 2}}
```

**Reasons by stage**

| Stage | Reasons |
|---|---|
| Motion | `goal_reached`, `no_progress`, `budget`, `rotation_stall`, `approach_inconsistent` |
| Align | `align_no_convergence`, `mark_lost`, `mark_not_visible` |
| GuardedApproach | `contact_stall`, `contact_touch`, `no_contact_within_bound`, `bar_under_finger` |
| Grasp | `precondition_unmet`, `empty_close`, `object_between_pads`, `stalled_close` |
| VerifyHold | `hold_pass`, `hold_fail` |
| Transport, Pull | `lost_in_transport`, `pull_bound`, `pull_stall`, `pull_gap_jump` |

## 7. Grasp strategies (v1)

### 7.1 `rim_pinch`: straddle and floor-touch (bowls and other open containers)

Two recorded facts motivate this strategy:
- 8 of 9 recorded bowl successes closed within 4° of the start orientation, and a radial jaw is not required (measured).
- A free bowl tips or slides when pressed on its rim, which defeats rim-top touch detection (§6.4).

**Sequence**
1. **Orientation.** Keep the start orientation: top-down, jaw along world ∓y. The model's `side` (`wrist_left` / `wrist_right`) picks the rim point on the jaw line. The prompt asks it to choose the side with free space next to the container.
2. **Point.** One mark on the rim at that side (coarse, then fine).
3. **Align.** Put the grip axis over the rim mark (o = (0, −0.05 m)).
   - In the straddle the wall sits between the open fingers.
   - Rationale only: a straddle is feasible for grip-site distances of about 18–86 mm from the bowl axis, so the tolerance around the rim is roughly ±30 mm. It is far wider than for a centred or fingertip-probe placement.
4. **GuardedApproach down.**
   - The expected first contact is the inner fingertip on the bowl floor. The floor lies inside the bowl's support radius, so the bowl cannot tip, and the hand stalls.
   - Rationale only: the grip site is then about 16 mm above the bowl origin. Recorded captures started 17–49 mm above it.
5. **Retreat 4 mm** (design default) for fingertip clearance.
6. **Grasp, then VerifyHold** (lift).

**Never centre the hand over a container wider than the opening.** That placement produced every bowl_a jam, and both fingers end up inside the rim.

**Risks for Phase 3b to measure:**
- a fingertip landing on the rim top, if Align error exceeds the straddle tolerance;
- the bowl sliding toward the outer finger during the close, because the wall at pad height lies inward of the rim mark;
- neighbouring objects on the outer side.

### 7.2 `frontal_bar_pinch` (drawer handle and other horizontal bars)

1. **Reach.** Move to a model-chosen hover waypoint in front of the handle.
2. **Orient.** Horizontal approach, jaw vertical (§6.2). The bar must appear vertical between the fingers in the wrist view.
3. **Point and Align.** Put the grip axis on the bar mark.
4. **GuardedApproach** along R[:,2].
   - With the bar correctly between the open fingers, nothing touches the bar. The 79 mm opening clears the 16.3 mm-tall bar by about 31 mm per side.
   - The expected first contact is the fingertips stalling on the **cabinet face** behind the bar.
   - Rationale only: at that point the grip site is about 22 mm past the bar's front face, and the pads overlap the 15.4 mm bar depth by about 5 mm.
   - **Never advance past the touch.** Fingertips pushed behind the cabinet face jam on the drawer-front edges when closing (attempt_memory_002 step 248, measured).
   - If the bar is not centred between the fingertips in the wrist view at the stall, the bar is under a finger. Outcome `bar_under_finger`: retreat 10 mm and re-Align.
5. **Retreat 5 mm** (design default, robot-only: fingertip clearance).
   - Rationale only: this leaves about 10 mm of pad overlap.
   - Under ruling R-3, use r_bar = 10 mm instead (TUNED-evaluator). That gives about full overlap and matches the verified capture depths: grip site about 10–13 mm past the bar face.
6. **Grasp.** Recorded clean captures plateaued at 16.55–17.76 mm in 5 of 6 (measured).
7. **VerifyHold** (pull 20 mm), then **Pull** to `max_travel_m`.

`hook_and_pull` is excluded from zero-shot v1 because it came from demonstrations.

## 8. Macro skills

| Skill | Sequence | Model calls inside |
|---|---|---|
| `reach` | Field/waypoint tracking to `waypoint_offset_m` via `FineDistancePlanner.plan_analytic` (`fine_distance_planner.py:180`) with a synthesized compact field (no repulsors in v1), limited to the ±80 mm cube (`speed_first_v3.py:39`). A goal outside the grid stops with zero motion (`fine_distance_planner.py:364-365`) | 0 |
| `grasp_with` | §7 sequence: (Orient) → Point → Align → GuardedApproach with ladder → retreat → Grasp → VerifyHold, with local retries | pointing only (2 + 2 per re-point) |
| `pull` | Pull primitive | 0 |
| `transport` | `reach` with the held object; a gap drop or a receding wrist patch aborts with `lost_in_transport` | 0 |
| `place_release` | `reach` to a hover above the destination → Lower until stall → Release | 0 |
| `retreat` | 30 mm along −R[:,2], gripper unchanged | 0 |

Place marks and external-view servoing are deferred to v1.1. A held bowl hides much of the wrist view.

## 9. Sensing reference (state8-derived)

| Signal | Definition | Label |
|---|---|---|
| Finger gap | q1 − q2 (mm); pad separation ≈ gap − 1.0 mm | ROBOT |
| Finger identity | finger 1 at grip + q1·R[:,1]; finger 2 at grip − \|q2\|·R[:,1] | ROBOT |
| Tool axes | R = rotvec→matrix(state8[3:6]) (`right_hand` body); jaw R[:,1], approach R[:,2] | ROBOT |
| Gripper geometry | pads −11.6…+4.4 mm about the grip site; finger travel 0–40 mm (`robot_tool_geometry.py:47-53`); fingertip vertices (`:163-182`) | ROBOT |
| Empty-close curve, touch threshold | learned by Calibrate | ONLINE |
| Stall | 3-step ratio < 0.3 × the move's own free-motion ratio | ONLINE |
| Object-present / hold margins | 0.5 mm / 1.0 mm above the empty curve | TUNED-initial |

## 10. Constraint compliance and rulings

| Item | Source | Label | Status |
|---|---|---|---|
| Upright external + wrist RGB, state8, task text | simulator observation | policy input | allowed (same inputs as today; mirror fixed) |
| Gripper geometry | robot asset | ROBOT | allowed (already used in prompts) |
| Wrist mount facts: 0.05 m offset along R[:,0], image right = R[:,1], image up = R[:,0], principal point at centre | robot asset (`robot.xml:239`) | ROBOT | **R-2** |
| Image Jacobian estimated online for Align | measured robot motion + image displacement | ONLINE | **R-1** (core: without it, Align and this agent are out of scope) |
| Calibrated touch, stall, and empty-curve thresholds | Calibrate + this move's own motion | ONLINE | allowed |
| Object-present / hold margins | fitted to recorded traces | TUNED-initial | allowed with labels; re-validated on new initial states |
| r_bar = 10 mm, 0.3 mm hold stability | fitted against evaluator-labelled grasps | TUNED-evaluator | **R-3**; robot-only defaults are used without it |
| Bowl radius/rim/floor heights, handle position, drawer travel | scene assets / evaluator | ORACLE | never used by the policy (rationale only) |
| Depth, camera projection through fovy, demonstrations, simulator contact truth | — | — | not used (see §14) |

**Rulings** (IDs mapped to IMPROVEMENT_PLAN §1):

| ID | Question | Plan ID | Recommendation |
|---|---|---|---|
| R-1 | Is an image Jacobian estimated online from the robot's own motion within "no camera calibration"? | R-a(ii), extended (plan default: no) | Yes: nothing is supplied in advance. If no, this agent is out of scope and the fallback is plan arm A3 |
| R-2 | May robot-asset wrist-mount facts be used in the prompt **and** in executor code? Setpoints are computed from the online κ, never projected through fovy (that would be tier T1) | R-a(i), extended | Yes: they are robot facts, like the gripper geometry already in use |
| R-3 | May evaluator-labelled TUNED constants be used live, with labels? | R-b | Default no; v1 has robot-only fallbacks |
| R-4 | May the prompt state that a continuing episode implies the task is not complete? This would leak the evaluator's success signal | new | Default no |

## 11. Speed model (projection)

Client call time ≈ 3.0 s + 38.5 ms per output token. Recorded calls of ≤130 tokens had a median of 7.2 s.

| Event (bowl pick-and-place) | Calls | Output tokens |
|---|---|---|
| `reach` to hover (cube-limited) | 1–3 | ≤200 |
| `grasp_with` choice | 1 | ≤200 |
| Pointing (coarse + fine) | 2 (+2 per re-point) | ≤100 each |
| `transport` | 1–2 | ≤200 |
| `place_release` | 1 | ≤200 |
| Recovery decisions | 1–3 | ≤200 |
| **Total** | **about 8–14** | |

**Projected time**
- Model time: about 50–110 s.
- Non-model in-loop time: about 60–135 s. That is 350–500 actions at 0.18–0.27 s per action at 512 px (measured rate); each local retry adds about 120–150 actions.
- Projected video-excluded total: about 110–245 s.

**Comparison.** The best recorded zero-shot success (speed_first v3 bowl) took 24 calls, about 223 s of model time and 270 s video-excluded (measured).
- The projected gain over that run is modest.
- The larger expected effect is on reliability. Most current runs never finish, and each failed attempt today costs about 6 calls.
- All speed claims must be measured under the plan's Phase 5 rules.

## 12. Code mapping

No audited legacy module is edited.

| Module | Contents | Reuses |
|---|---|---|
| `src/gpt_field/agent_speedup_v1.py` | `AgentSpeedUpV1Client(ChatGPTPlanHybridClient)` (`hybrid_subscription.py:166`): stateless. `_history_input` returns `[]`, accepted outputs are not appended. `decide(obs)` and `point(image, stage)` share one predict path, `_api_calls`, logs, and the bounded overload retry (`speed_first_hybrid.py:148-201`). Client-side un-mirroring. `request_key` override includes wire order and transport version. `AgentSpeedUpV1Policy`: its own `reset` / `act` / `observe_transition` with `_pending` / expected-observation invariants. It does not inherit DirectHistoryPolicy or the SpeedFirst* policies, whose per-action history and chunk limits do not fit long skills | transport, OAuth, accounting |
| `src/gpt_field/contact_skills.py` | Calibrate, Orient, GuardedApproach, Grasp, VerifyHold, Pull, Lower, Release, retry queue, detectors, axes→quaternion helper (`right_hand` frame) | `LocalGripperSettle` (`speed_first_gripper.py:66`); `ActionAdapter` (`action.py:40`); `robot_tool_geometry.py:47-53, 163-182, 235-244` |
| `src/gpt_field/wrist_servo.py` | Align: NCC tracking, Broyden Jacobian, κ setpoint, ladder hooks | NCC pattern from `speed_first_pregrasp.py:156-244`; anchored stepping in the style of LocalGripperSettle (not `PathTracker`, which needs a GridField) |
| `src/gpt_field/mark_prompting.py` | coarse/fine overlays, id → pixel mapping | PIL |
| `src/gpt_field/ordered_transport.py` | insertion-order wire serialization | imports `_open_public` / `read_completed_response` from `chatgpt_plan.py` (`:136`, `:315`), not edited |

**Runner** (`scripts/evaluate_bowl_hybrid_subscription.py`)
- `--agent-speedup-v1` gets its own branch.
- It sets the speed_first evaluation protocol only: collisions record-only, operator-stop file, LIBERO-only stopping. It bypasses `_client_type` and the v2/v3 implication chain.
- Every decision's diagnostics carry `controller="agent_skill"`, `routing={skill, params}`, and `execution_kind="agent_skill:<primitive>"`.
- The identity block gains an `agent_speedup_v1` section: prompt/schema sha256, constant labels, pointing-protocol and transport versions. Field/Direct keys are null.
- A new `AgentSpeedUpV1VideoRecorder(HybridVideoRecorder)` is needed, because the existing recorder reads `prediction["actions"]` on non-Field refreshes (`hybrid_video.py:88-89`).
- Offline smoke runs use a separate harness, because speed_first forbids `--offline-smoke`.

**Harness**
- An agent adapter for `evaluate_contact_routing_diagnostic.py` with an opt-in 350-action cap (default 200, `contact_diagnostic_sim.py:141-144`) and a request cap that counts only decision calls.
- `StubAgentClient` with an ORACLE pointing stub (test-only).

**Pitfalls**
- `verify_manifest` re-hashes all of `src` before each trial (`evaluate_contact_routing_diagnostic.py:287-295`). Develop in a separate worktree and run from a frozen copy.
- Client-side un-mirroring keeps `asdict(config)` in the physics fingerprint (`contact_diagnostic_sim.py:72-73`) unchanged.

**Size:** about 3,000–4,000 LOC including tests (projection).

## 13. Validation (additions to IMPROVEMENT_PLAN)

**Prerequisite: plan Phase 1 (oracle basin sweep).** Its lateral half-width W is the go/no-go for Align.
- Align's expected residual is about ±3 mm from mark quantization plus servo error.
- If W < 5 mm, use upgrade T1 instead.

**Phase 3b** (simulator only, 0 model calls; needs approval)
- `StubAgentClient` with ORACLE pointing runs the full `grasp_with` and `pull` macro skills from checkpoints K1–K4.
- It measures:
  - Align's residual lateral error against W;
  - detector coverage under the v1 command law;
  - straddle floor-touch reliability;
  - bar-face stall reliability;
  - VerifyHold false accepts (target 0).

**Phase 4b** (live checkpoint arm; needs approval)
- B2 against the plan's A0 and A3 on the same checkpoints and caps.
- Primary outcome: evaluator engagement (DIAGNOSTIC).
- Also measured: decision and pointing calls per engagement.

**Full tasks.** Then follow plan Phase 5, with B2 as an additional arm.

## 14. Constraint-relaxation upgrades (not part of v1)

| Tier | Upgrade | Replaces | Constraint change |
|---|---|---|---|
| T1 | Wrist camera intrinsics + mount projection | metric eye-in-hand servoing; grasp candidates drawn on the wrist image | camera calibration (robot asset) |
| T2 | Joint position/velocity (`observation.py:19-21`) | better stall and contact detection | state beyond state8 |
| T3 | `effort=high` for strategy and pointing calls only | — | effort setting |
| T4 | Depth + LIBERO camera matrices | lift marks to 3-D (MOKA/ReKep style); analytic grasps or a grasp sampler | depth, calibration |
| T5 | Demo-trained low-level contact policy under supervisor routing | Align / GuardedApproach / Grasp | demonstrations |

## 15. Risks and open questions

1. **Mark tracking on low-texture rims.**
   - An earlier offline replay of NCC/flow tracking kept 47 of 240 observations (8 frozen windows, manual regions).
   - Mitigation: re-point on loss. Pointing calls are short.
2. **Detector blind spots.** A fingertip on a free rim, or sliding objects. The strategies aim first contact at supported surfaces, but this must be measured (Phase 3b).
3. **Lateral tolerance and residual Align error.** These are the key assumptions. They are tested by plan Phase 1 and Phase 3b.
4. **Choosing the horizontal approach direction.** The model's world-direction choice was wrong in 3 of 6 earlier pull checks, under the mirrored interface. Orient's visibility check catches some errors. An online image→world sign map (an R-1 extension) would make it reliable.
5. **Close dynamics in the straddle.** The bowl may shift toward the outer finger before both pads touch. Larger settle stiffness helps; this must be measured.
6. **Strategy coverage.** v1 covers rim and bar pinches only. New object types need new strategies.
7. **TUNED constants come from one scene and initial state.** They must be re-validated on initial states 1–2.
8. **Stateless requests drop the image history.** The ledger must carry enough context for recovery. This must be measured against the history-based baseline.

## 16. References

- **Model chooses skills, with feedback:** Ahn et al., *SayCan*, 2022; Huang et al., *Inner Monologue*, 2022; Liang et al., *Code as Policies*, 2022.
- **Model specifies targets or constraints, a solver executes** (these use RGB-D): Huang et al., *VoxPoser*, 2023; Huang et al., *ReKep*, 2024.
- **Mark-based visual prompting:** Yang et al., *Set-of-Mark Prompting*, 2023; Nasiriany et al., *PIVOT*, 2024; Liu et al., *MOKA*, 2024.
- **Uncalibrated visual servoing:** Hosoda & Asada, *Versatile visual servoing without knowledge of true Jacobian*, IROS 1994; Jägersand, Fuentes & Nelson, *Experimental evaluation of uncalibrated visual servoing for precision manipulation*, ICRA 1997; Chaumette & Hutchinson, *Visual servo control, Part I*, 2006.
- **Model-emitted grasps on metric grounding:**
  - Gemini Robotics-ER, 2025.
  - MotorMind, 2026 preprint (numbers not independently verified): mm-level moves through a deterministic controller, gripper-width grasp checks, and camera triangulation.
- **Demo-trained reference point:** Kim et al., *OpenVLA-OFT*, 2025. It reports 97.6% on LIBERO-Spatial and 97.9% on LIBERO-Goal at 224 px, with about 50 demonstrations per task.

## 17. Implementation (2026-10-16)

**Modules** (all new; no existing file edited):

| File | Contents |
|---|---|
| `src/gpt_field/agent_speedup_v1.py` | `AgentSpeedUpV1Policy`: generator-based macro-skill scheduler, sensor-gated grasps, local retries, ledger/outcome records |
| `src/gpt_field/agent_speedup_v1_client.py` | `AgentSpeedUpV1Client` (stateless decision and pointing requests over the subscription path), schemas, prompts, `ScriptedAgentTransport` |
| `src/gpt_field/contact_skills.py` | Calibrate, AnchoredMove (guarded approach / pull / lower / retreat), Orient, WaypointReach, GripperSettlePrimitive, VerifyHold, geometry helpers |
| `src/gpt_field/wrist_servo.py` | NCC tracking, image Jacobian (probe fit, Broyden, kappa transfer), Align, WristPatchMonitor |
| `src/gpt_field/mark_prompting.py` | coarse 8x8 / fine 8x8 numbered overlays |
| `src/gpt_field/ordered_transport.py` | insertion-order wire serialization |
| `src/gpt_field/agent_speedup_v1_video.py` | annotated video recorder |
| `scripts/evaluate_agent_speedup_v1.py` | runner (loads the existing runner's helpers, never edits it): live, `--preflight`, `--sim-smoke [--smoke-script plumbing|grasp]` |

The runner is a separate script rather than a branch of `scripts/evaluate_bowl_hybrid_subscription.py`, because that file carries uncommitted user work. The model client lives in its own module.

**Changes driven by real-LIBERO scripted smokes** (`runs/agent_speedup_v1_sim_smoke_00*`, no model calls):

1. Align command cap is adaptive: 25% of the estimated remaining lateral error, clipped to 10–30 mm commanded per step (a fixed 10 mm cap realized ~2.5 mm/step and failed a 20 cm offset). Servo budget 90 steps.
2. The NCC template is refreshed during the servo when the match is confident but decaying (0.70 ≤ peak < 0.85); perspective change during a long lateral move otherwise lost the track.
3. Align stops at min(6 px, 3 mm × kappa) (≥2 px); at a high hover 6 px was ~7 mm.
4. Descent refreshes settle 3 steps first and use the depth-free kappa transfer instead of re-probing; ±2 mm probes on a rim edge were ill-conditioned (kappa 2836 vs ~950 px/m).
5. The guarded-descent bound is 300 mm (supporting surfaces stop the hand).
6. Pointing answers use a plain bounded integer, 0 = not visible (live-proven schema construct).

Result on real LIBERO (bowl_plate, init 0, scripted pointing at the target bowl's right rim): align residual 0.4 px, seven successful descent refreshes, stall at the bottom, object detected at close (15 mm at k=15 against a 2.35 mm empty curve), 25 mm lift `hold_pass`, and the bowl carried in the wrist view. The palm touches the rim at the bottom of the descent (recorded by the record-only collision monitor; the bowl depth is about the finger length). Calibration on real physics: empty curve 2.35 mm at k=15, open gap 78.5 mm.

**Changes from the adversarial reviews:**
- frontal jaw sign is chosen by the smaller rotation;
- outcomes carry the macro's params, pose and last-attempt evidence;
- releasing a handle with `place_release [0,0,0]` opens in place;
- `rim_pinch` returns `orientation_restored` after restoring the top-down orientation so the side is chosen from the new view;
- the target description goes only into the bowl's pointing context;
- bar retries re-point;
- total pull travel is shown;
- rejected-decision storms are damped;
- the runner logs every outcome and decision in a step;
- preflight previews the live payload;
- the render GPU is recorded;
- the reset hash defaults to the frozen reference.

**Changes from the control-math review** (checked against recorded real trajectories and a fitted model of the real arm):
- Orient succeeds only once the rotation has settled (≤0.1°/step), followed by a 4-step hold.
- Align restarts its probe (up to twice) instead of failing when frames are rotated.
- The servo keeps its long κ baseline; κ updates are clipped to ±10% each, from baselines of at least 5 mm.
- Stall detection uses a 2-step window at 0.6× the free-motion ratio.
- The hold monitor picks the most textured patch.
- The frontal descent follows the measured camera axis.

**Drawer handle on real LIBERO** (`runs/agent_speedup_v1_sim_smoke_drawer_004_grasp`; scripted ORACLE hover waypoints and pointing, no model calls):
- reach to the hover, then a 90° frontal Orient in 53 steps;
- align 2.6 px, then advance to the cabinet face;
- close plateau 17.4 mm, then `hold_pass`;
- pull 10.1 cm (`pull_bound`), and the drawer opened;
- `place_release [0,0,0]` returns `released_handle`.

The contact registered as a single-finger deflection with the bar correctly between the fingers, so a touch is no longer treated as "bar under a finger" (a real miss shows up as an empty close or a failed hold). The bowl regression after these changes: `hold_pass`, with palm-rim contact down from 2562 to 1740 points.

**Known limitations:**
- The VerifyHold wrist-patch monitor is secondary evidence; on real frames its patch is usually the table, so it reports "moved" during a good lift.
- Pointing on real frames has been exercised only with scripted answers. Model pointing quality is untested.

### 17.1 Live run 001 (bowl_plate, stopped by the user) — analysis and fixes

`runs/agent_speedup_v1_bowl_zero_shot_001/trial`: 2835 actions, 108 calls, Ctrl+C. Four analysis lenses (motion, decisions, contact, runner) each had an adversarial verifier. Reports: session scratch `impl/live001/report_{motion,decisions,contact,runner}.md`.

**What happened**
1. **The grasp worked.**
   - Steps 163-306: rim pinch reached `hold_pass`.
   - Steps 306-415: the bowl was carried cleanly.
   - Step 347: the bowl was 4.4 cm from the plate centre.
2. **The model turned external-view directions into world directions with the wrong sign** (the root cause).
   - It perceived the plate correctly in 16/16 statements.
   - But it converted with the wrist axes. The prompt's only mapping rule was written for the wrist camera.
   - From step 347 the median command error was 153.5°, and the bowl-to-plate distance grew from 4.4 cm to 32 cm.
3. **The +x pushes drove the arm past its top-down envelope** (about 0.79 m horizontally from the base).
   - Outward motion realized 1-2.5% of the command there.
   - The hand pitched up to 61°, the OSC became erratic (jumps up to 66 mm/step), and the held bowl was banged on the table.
   - WaypointReach holds its own start orientation, so the tilt latched from one call to the next.
4. **The held moves were blind to contact.**
   - The transport checked the hold only after its 30-step move, so the drop at step 925 was reported at 945.
   - Gap rises (the bowl pressed into something) were ignored.
5. **The first place could not reach the support.** It started 18.8 cm above the pick height, and the 15 cm lowering bound stopped it about 3 cm above.
6. **The post-loss loop could not succeed:** 68 decisions and 67% of the run.
   - The bowl lay beyond top-down reach.
   - Each grasp_with restored top-down in place, which dragged the hand 4-12 cm back and down.
   - The model saw only `grasp_with failed: budget, evidence {}`.
7. **Two probe_failed outcomes (steps 122, 148) came from a residual swing after a reach**, not from pointing:
   - the swing reached 0.83°, and the 0.2° clean-frame filter left fewer than 2 frames;
   - the marks were on the rim and the NCC peak was 0.97-0.99.
8. **Runner:**
   - ffmpeg shared the terminal's process group, so Ctrl+C was recorded as infrastructure_error;
   - a second Ctrl+C during cleanup could lose summary.json;
   - the last outcome was missing from the summary;
   - failing macros took 86% of the actions.

**Fixes** (policy-2, contact-skills-2, wrist-servo-2, prompt-2, external-axes-1; all measured-only)

| Area | Change |
|---|---|
| External view | `tool_axes_world` apply ONLY to the wrist image. Calibrate makes three net-zero 4 cm probe moves, fits a 2×2 external-image map from Farneback flow of the moving hand (`external_axes.py`), and publishes `external_image_axes_world {image_right, image_up}`. Ledger lines carry `external_image_shift_px_right_down`. Assumed ruling **R-1x** (R-1 applied to the external camera); `--external-axes off` disables it. Real LIBERO: image_right = [0.077, 0.997, 0], image_up = [-0.997, -0.075, 0], residuals < 1°. Live 001 frames: median 5.5° on translations. |
| Workspace edge | WaypointReach `reach_limit`: orientation error ≥ 0.05 rad, rising for 3 steps, while the goal error shrank by less than 10% of the commanded motion. `uncontrolled_motion` for a step > 20 mm. Budget endings are split into `orientation_drift` / `slow_arrival` / `blocked`; `invalid_direction` → `left_planning_grid`. Edges hit by horizontal pushes are remembered, and a further push within 3 cm is rejected as `beyond_recorded_reach_limit`. |
| Held moves | transport, a reach while holding, and the place hover run a held WaypointReach that keeps the grasp-time orientation (only if within 15° of it). It stops on `hold_lost` (→ `lost_in_transport` at once) and on `held_object_contact` (gap +1.0 mm vs the verified grasp gap, or +0.4 mm in one step; the hand is then lifted 1 cm). |
| Place | The hover never descends (dz clipped to ≥ 0). Lowering bound = clip(z − pick z + 3 cm, 0.15, 0.30), measured from the rim pinch's floor-touch stall. Fast lowering (20 mm command) runs to 5 cm above the pick height; slow lowering then stops on a stall or a held-gap rise of 0.3 mm and releases. A fast-phase event only hands over to the slow phase. |
| Restore | For a tilt of 10-60°, move 6 cm back opposite the approach axis's horizontal part and up to the pick height + 6 cm, then Orient. Orient fails fast on a pose drift > 2 cm (`pose_drift_reach_limit`) and accepts 5° at budget. Orient failures are reported as `orient_budget` / `orient_stall` / `pose_drift_reach_limit`, with a detail line. |
| Align | Before the probe, a `probe_settle` phase holds still until 2 quiet steps (≤ 0.03°, ≤ 0.3 mm; at most 14 steps), then re-bases the frozen frame and the hold orientation. |
| Model view | New state_summary fields: `approach_tilt_from_straight_down_deg`, `held_height_above_pick_m`, `hold_quality`, `target_history` (where a held object was lost), `reach_limits`, `loop_warning` (≥ 5 failures in a row), counts. Ledger lines gain `end_xyz`, tilt, progress, angle, gap rise, Orient info and detail; floats are rounded to 4 decimals; `last_outcome` drops its duplicate pose. The prompt adds a reason glossary, external-view rules, carry height, ≤ 3 cm steps near the destination, the drop rule and the situation-while-holding rule. |
| Runner | SIGINT/SIGTERM are deferred during finalization (the 3rd press hard-exits after the pre-video summary). Closes catch BaseException; the summary is written before and after the video. The sim worker and ffmpeg run in their own sessions. Outcomes carry `n` and are reconciled at finalization. A one-line stuck alert is shown (no cap). Console output is compact. The `operator_stop` source is file or sigint. |

**Real-LIBERO scripted smokes** (zero model calls; scripted offsets use evaluator geometry, diagnosis only):
- `runs/agent_speedup_v1_sim_smoke_009_grasp_policy2`: rim pinch `hold_pass`, with probes and calibration in 89 steps.
- `…_010_place_policy2`: grasp, two +7.5 cm lifts, two transports toward the plate, then place_release from about 17 cm above the pick height. Fast then slow lowering ended in **LIBERO success** at step 413; the old 15 cm bound could not have reached the plate.
- `…_011/012_edge`: pushing +x from the start pose:
  - `reach_limit` fired at x ≈ 0.133 (progress 0.84, 3° tilt);
  - the next +x pushes were rejected by the guard.
  - A move back and 3 cm up also lost orientation (the top-down envelope shrinks with height). It is reported but not remembered, so the way back stays open.

**Not done / open:**
- Executor-side external-view servoing for transport.
- A recovery skill for objects beyond top-down reach (drag-back or pitched pinch).
- Lighter traces and PNG level-1 encoding (R4/R6).
- Evaluator-only object poses in the trace.
- The R-1x ruling (assumed). `--target-description` was not passed in live 001; both bowls are the same asset, so passing a description is recommended.

### 17.2 Drawer live run 001 (drawer_bowl, stopped by the user) — analysis and fixes

`runs/agent_speedup_v1_drawer_zero_shot_001/trial`: 867 actions and 61 calls. All 12 handle grasps failed in Align before contact. Three analysis lenses (Align mechanics, pointing and identity, motion and hover) were each adversarially verified, using bitwise real-sim replays and segmentation (diagnosis only). Reports: session scratch `impl/drawer001/`.

**What happened**
1. **Pointing picked the wrong structures.** The pointing model hit the top handle in 1 of 11 pointing rounds. The others landed on the wine-rack post (2), the cabinet's side-wall edge (5), the robot's own hand (1), and two wrong "not visible" answers. With the jaw vertical, the wrist view is turned 90° (world up = image right). The request said "the horizontal handle bar" in an "UPRIGHT" view with "fingertips at the bottom", so the model chose world-vertical structures, which look horizontal in that view.
2. **The tracker slid along the bar (aperture problem).** NCC localizes a point on a uniform bar only across it, not along it. The 2-D probe fit failed even on the one correct mark (6× jacobian_inconsistent_with_mount, 1× ill_conditioned). Two servos tracked a sliding cabinet edge, pushed the hand onto the plate and table for about 70 steps each, and caused 85% of the run's forbidden contacts.
3. **Hovers were off the bar.** 11 of 13 hovers put the grip outside the bar's 89 mm length.
4. **The arm ended up at full stretch.** From step 436 it was pinned at the Panda elbow singularity (stretch 0.9997-1.000). The OSC cannot escape it: 9 maneuvers were tried.
5. **Contacts were mislabelled as workspace edges.** In the top-down phase, contacts with the wine rack, cabinet corner and bottle were reported as "workspace edge: move back toward the robot", which misled the model.
6. **The model mis-signed directions.** It got "low in the wrist view" wrong 5 of 8 times and called +x "toward the robot".

Correction to an earlier statement: the position *along* the bar matters. The vertical jaw has ±31 mm of slack across a 16.4 mm bar, but along it the fingers must land on its length, and the robot-side half also keeps the arm away from full stretch.

**Fixes** (wrist-servo `across_bar` mode, prompt-3, marks `crop96x5|128x4-gravity-turn`):

| Area | Change |
|---|---|
| Pointing frame | When world up lies in the image plane (frontal hand), the coarse and fine images are turned by quarter turns so world up is at the top (`gravity_quarter_turns`, from the measured R only). Answers are mapped back (`unrotate_pixel`). The context names the turn and the fingertip edge; the pointing instructions no longer say UPRIGHT. Bowls (top-down) are unaffected: 0 turns. |
| Bar target | The phrase is now "a clearly visible point on the top drawer handle's bar (not another drawer's) between its two mounting posts, preferably mid-way; never behind the fingertips; 0 if not visible". Bars use a wider fine crop (128 px ×4) because a one-cell coarse slip left the handle 4 px inside the crop edge. |
| Bar check | A zero-shot structure-tensor check of the answered pixel: the dominant gradient must run across a horizontal bar (agree ≥ 0.8, coherence ≥ 0.5, rms ≥ 2), and the point must not be in the fingertip band. On failure, pointing restarts once from the coarse stage with feedback, then fails with `pointed_not_a_bar`. On live frames it rejected 8/8 wrong answers and kept all true centres. About 75-87% of true bar points pass, so the re-ask is needed. |
| Align `across_bar` | Only the across-bar column is servoed; its setpoint (the image centre column) is depth-free. The along-bar position is set once from the pointed pixel and then held by proprioception, with the template row pinned. κ comes from a 6 mm out-and-back probe along R1 (≥ 4 clean samples, ≥ 6 px, residual and leg agreement checks). There is no 2-D radial κ transfer. Commands are capped at 10 mm/step and the approach hold at 8 mm/step. Guards: `align_blocked` (≤ 5% realized for 4 steps), `approach_drift` (> 25 mm), `hover_off_bar_across` (> 4 cm). Convergence requires the hand to be still. |
| Bar descent | A quiet-based settle before the first segment. The guarded segments hold the attempt's along-bar point as a world lateral target. Refreshes re-probe κ and evaluate the cutoff after the update. The stall arms only once the move has realized the threshold, or after 5 steps (bar approach only). After `align_blocked`/`approach_drift` with a known along offset, one local plain reach is made along the bar before re-pointing. |
| Reporting | `finger_contact` (open gap change ≥ 1 mm from a settled gripper). Neutral texts for reach_limit, orient and pose_drift (contact or edge, with measured directions). A pose_drift handle Orient lifts 4 cm and retries once. A per-target loop warning fires after 3 failed grasps. `world_up_in_wrist_image` and the pointed pixel are in the evidence. The glossary covers all grasp_with failure reasons. The prompt states the turned-hand clearance (about 10 cm around the grip) and wrist low/high directions. |

**Not adopted (verified wrong):**
- A translation-only "coupling rescue" for frontal reaches. The stalls were the elbow singularity, so the rescue would turn honest edge stops into tilted false arrivals.
- Holding the exact frontal quaternion in Align. Rotating toward it while the template row was pinned moved the track to the bar's edge (−8.6 mm).
- Masking or shifting the bar-check patch out of the fingertip band. It lowered true-bar pass rates near the band to 0.

**Validation** (real LIBERO, zero model calls):
- **Live-state restarts** (bitwise replay to the recorded step, full `_grasp_bar` + pull, oracle pointing at the visible top-bar point):
  - From live steps 335, 361, 393 and 555: `hold_pass` and the drawer pulled 6-10 cm. 335 and 555 started 5-15 cm off along the bar.
  - From 424, 613 and 744: correctly "not visible" (no bar visible between the posts).
  - From 583 (full stretch): `align_blocked`.
- **Fixed model marks** (live 424, 241, 555): rejected by the bar check.
- **Scripted smokes** `runs/agent_speedup_v1_sim_smoke_drawer_011-016`: grasp, 10 cm pull, release from the verified and off-centre hovers. The bowl rim grasp and carry-and-place were re-checked (`…_016_grasp_drawerfix`, `…_017_place_drawerfix`).

**Full-stretch guard (ruling R-5 granted by the user, 2026-10-17):** robot base (−0.66, 0, 0.912) and shoulder (joint-2 anchor) (−0.66, 0, 1.245), verified in both LIBERO scenes. For a frontal hand, the wrist point grip − 0.204·R2 tracks the arm's stretch; full stretch is ≈ 0.804 m, and live 001 was trapped from 0.788 m on.
- `grasp_with` (frontal_bar_pinch) is refused before turning when the predicted wrist-to-shoulder distance exceeds 0.77 m (`hover_beyond_frontal_reach`).
- Frontal reach goals that go out past 0.77 m are rejected (`beyond_frontal_reach`).
- A frontal move drifting past 0.785 m stops (`frontal_stretch_limit`).
- An along-bar target beyond 0.77 m fails `bar_point_beyond_reach`.
- The state summary carries `robot_base_direction_world` and `frontal_arm_reach`.

The verifier's 0.75 m would have refused the live 361 hover (0.759), from which the grasp and a 10 cm pull succeed. Real-sim checks: live states 335/361/393 still grasp and pull 10 cm; the trap hovers 420 and 555 are refused before any motion; drawer smoke `…_017_guard_B` and the bowl regressions (`…_018_grasp_guard`, `…_019_place_guard`) are unchanged. Top-down moves keep the reach_limit handling.

### 17.3 Drawer live run 002 (drawer_bowl, stopped by the user) — blocked bowl grasp, speed features

`runs/agent_speedup_v1_drawer_zero_shot_002/trial`: 4702 actions and 98 calls. Three lenses (geometry, model decisions, executor) analysed the run, and each was adversarially verified with bitwise real-sim replays (diagnosis only). Reports: session scratch `impl/drawer002/`.

**What happened**
1. **The drawer half worked:** `hold_pass` on the top handle, a 14.1 cm pull, then a release. Reaching the hover took 18 decisions (1.1 m of path) to step around the wine rack.
2. **Every bowl descent stopped on the open drawer.** All 13 bowl descents stalled at grip z 1.092–1.104. The palm (12×) or a finger (1×) landed on the top edge of the drawer's front panel (z 1.122), 14 cm above the bowl rim.
   - **Overlap:** the hand body spans about ±10 cm along the jaw and 3 cm across, with its underside 3 cm above the grip. The bowl lay in front of the cabinet, and the 14.4 cm pull put the drawer front over its far side.
   - **Wrong close and retry:** each stall was taken for a floor touch. The gripper closed on nothing, and the retry shifted the grip *toward* the bowl centre, which is toward the drawer.
3. **Pull window.** Replaying the model's own decisions with only the pull length changed:
   - a 0.10–0.12 m pull succeeds;
   - a 0.13 m pull fails the same way as the live run.

   The window depends on the layout and is about 2 cm wide.
4. **Model-side costs:**
   - **Hover:** "5–10 cm above the rim" could not be met under the overhang.
   - **Loop warning:** `orientation_restored` counted as a failed grasp, so the warning fired after only 2 real failures.
   - **Target description:** none was passed (bowl run 002 had one). Six decisions talked about a phantom second bowl.
   - **Latency:** the model pipeline took 66.6% of the non-video control time, at 0.64 s + 0.051 s per output token.

**Fixes** (prompt-5, marks unchanged):

| Area | Change |
|---|---|
| Blocked descent | A rim-pinch contact while every ladder refresh is still `aligned` (none reached the setpoint cutoff) is reported as `blocked_above_rim`. The executor does not close; it returns to the hover and retries once with the grip shifted 2.5 cm *away* from the container centre, then hands back to the model. Outward is used because the side is chosen with free space outside the container. The shift is bounded by the fingers: the wall must stay between them, so the shift stays below half the ~78 mm opening minus the pad and the pointing spread. Separation: 13/13 recorded genuine `hold_pass` attempts reached the cutoff first, with the cutoff firing 3.2–6.1 cm above the rim and 6.9–9.8 cm of travel to the floor touch after it; none of the 13 drawer stalls did. This evidence comes from only 3 physical situations. |
| After the retry | A `hold_pass` that needed an outward retry reports `away_from_blocked_side_world`. Its detail says to move the container about 4 cm that way before lifting it more than about 5 cm, because its far side can still sit under the obstruction (a straight lift hit the drawer in one geometry test). |
| Reporting | Final `blocked_above_rim` and same-height `empty_close` details name the stall heights and the hand footprint. Blocked stall heights feed the per-target loop warning. `orientation_restored` is procedural and no longer counts as a failure (`PROCEDURAL_REASONS`). |
| Held contact while rising | A `held_object_contact` during an upward held move no longer lifts again; under an overhang the old 1 cm lift repeated the contact until the bowl dropped (executor lens F4). After a blocked-then-held grasp it moves 5 mm down and 10 mm along `away_from_blocked_side_world`; otherwise it moves 1 cm back against the move. Executor verify from 799: lift again lost the bowl, straight down needed 6 contacts to scrape past, down-plus-out needed 2. Contacts during horizontal or downward moves still lift off the support. |
| Prompt-5 | The rim hover says "or just above anything overhanging it, e.g. an open drawer front". `pull` adds that an open drawer front overhangs what lies in front of the cabinet, but gives no pull length. In this layout a 10 cm pull works, a 12 cm pull drops the bowl on a straight lift, and 13–14.4 cm block the palm (handled by the retry); a numeric rule would be layout-specific. Carrying says to go high enough to pass over what is in the way (e.g. above an open drawer's front edge) and to move the object out from under any overhang before lifting it high. The old 3–6 cm carry rule would have driven the bowl into the drawer front. The glossary gains `blocked_above_rim`. The `held_object_contact` text describes the new back-off. |
| Speed (prompt-4) | `more_waypoints_m`: up to 2 extra waypoints in one reach/transport; the path stops at the first failing segment ("waypoint k of n ended …"). `grasp_with` + `max_travel_m` (frontal only) grasps and pulls in one decision. `place_release` on a handle lifts 5 cm and turns top-down in the same decision. The state summary carries the last 3 situations and outcomes (`recent_decisions`). |

**Not adopted:**
- **A hard-coded pull length.** The window is layout-specific (for the standard rim point, 5–7 mm wide geometrically).
- **Documenting `transport` while holding a handle as a way to push the drawer back.** It is untested.
- **A second, 3.5 cm outward shift.** It was validated only with oracle rim points. The live model's rim points were 49–57 mm from the bowl axis, so 3.5 cm would put the grip past the ~86 mm straddle limit. Its one 16 cm-pull "success" was a dropped bowl: the straight lift scraped the drawer underside, and LIBERO scored a 14 cm free fall.
- **A stall re-match check (R2)** that confirms a block by re-finding the rim far above the fingertips. Its thresholds were fitted after the fact. The cutoff rule (R1) alone has known false-negative paths: a refresh that loses the mark, or a κ that grows too fast because the track latched onto a nearer object. Either path falls back to the old close-and-retry. Collect cutoff-height statistics in the 3–5-init runs.
- **A predicted keep-out for the opened drawer** (from the measured handle contact and pull), tested against the palm rectangle before descending, with a 90°-turned jaw as the first remedy (executor lens B1/B2). The reactive check covers this run; the keep-out is kept as an option.
- **A pre-descent palm-footprint overlay** (project the ±10 cm × ±3 cm hand footprint into the wrist image before descending). Kept as an option.

**Validation** (real LIBERO, zero model calls):
- **Live-state restarts from drawer 002 steps 799 and 1316** (bitwise replay, oracle rim pointing): `attempt_reasons` = [`blocked_above_rim`, `hold_pass`]. The bowl lifted 6.4–6.5 cm and the drawer stayed put (q −0.142/−0.145).
- **Full chain with the live model's own recorded rim pointing** (coarse 44 / fine 21, no oracle fallback), single 2.5 cm retry, from 799 and from 1316. Each ran: `blocked_above_rim`, then `hold_pass`; the advised 4 cm move away; lift; carry over the drawer (oracle carry waypoints); `place_release` → `released`. Both ended in LIBERO success with the bowl on the drawer floor (z 1.063), and the drawer stayed at q −0.144. With oracle rim pointing, the earlier two-shift build gave the same result from 799.
- **Lifting straight up from 799**, ignoring the away advice, also succeeded. The grasp had already dragged the bowl clear, so the advice is precautionary here. The new rising back-off is covered only by the fake-sim test `test_held_object_contact_while_rising_backs_off_against_the_move_not_up`.
- **Bowl regressions unchanged:**
  - `…_020_grasp_blocked` and the final-code rerun `…_023_grasp_final`: `hold_pass` at step 355 with 1658 forbidden points, the same as `…_018`.
  - `…_021_place_blocked` and `…_022_place_final`: LIBERO success at 434 steps, the same as `…_019`.
- **Chained skills** (`…_drawer_018_chained`): `hold_pass_then_pull_bound` pulled 0.1005 m; `released_handle` "lifted and turned the hand top-down" ended at 3.0° tilt.

**Contacts:** forbidden contact points still rise by 868–2314 per restart. The geometry lens traced the outward-offset ones to the outer finger on the cream cheese. They are record-only, and LIBERO success is unaffected.

**Open:**
- 3 of the 8 live bowl `grasp_with` calls ended `probe_failed` (`jacobian_inconsistent_with_mount`) at 26–30 cm hovers. No lens has analysed them.
- All sim carries used oracle waypoints. A model-chosen carry is untested.
- The positive set for the cutoff rule comes from one bowl scene.

**Live runs:** pass `--target-description` for the bowl.
