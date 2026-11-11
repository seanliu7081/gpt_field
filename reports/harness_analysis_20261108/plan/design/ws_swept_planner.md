# ws_swept_planner: body and held-object swept planner (replaces the grip-site point router)

Workstream key: `ws_swept_planner`. Design date: 2026-11-08. Read-only design. Nothing under `/workspace/gpt_field` was edited, and no simulator, model call or GPU job was run. The prototypes below were CPU-only, on saved artifacts, with scripts and outputs in the scratchpad (see Appendix A).

`<NEW>` means the new build root. Its location belongs to ws_runner_measurement: a fresh copy of H1 at `3cc84ac`, with H1 itself left untouched. All file paths below are relative to `<NEW>` unless they are absolute.

---

## 0. Summary

**What this replaces.** H1 has a single collision mechanism: `TransitRoute` (`agent_speedup_v2_user_check_routing.py`). It checks one point, the grip site, against model-written boxes inflated by one model-written clearance. It runs only on transit `move_to_pose` and `move_relative` (`_fragments.py:144-145, 293-317`). This design replaces it with **one numpy-only body and held-object swept model, used as a local planner on every arm stage**, running in the controller `.venv` (numpy 2.5.3, no scipy or torch).

**What the planner does at each stage:**
- It sweeps the **executed** motion profile against the harness object map (from ws_geometry_tools and ws_perception_service). The executed profile includes the OSC rotation lag that ws_executor_control models.
- Contact permissions are per robot part and per object ID.
- The held object is attached at grasp.
- Then it chooses, in order:
  1. **escape**, if the start is inside an inflated region (monotone clearance rule);
  2. **direct** motion with a rotation-sequencing mode;
  3. **Cartesian detours**: lift-over, sidestep, or L-shaped;
  4. **a standoff handoff to a contact stage**, if the goal touches an object;
  5. **`no_route`**, only when nothing works.
- It reports clearance with explicit, measured margins.

**Other parts of this workstream:**
- A contact-descent **depth bound** from a measured height map under the exact palm footprint.
- **Clearance-scaled speed**, derived from the fitted OSC response and the measured latencies.
- Feature flags `off`, `shadow` and `on`.
- An offline replay of all recorded stages of the 9 runs.

**Prototype facts behind the design (measured offline, Appendix A):**

| Fact | Value |
|---|---|
| Body model | 13 MJCF collision geoms → convex hulls → bulge-bounded sphere cover. At 12 mm (arm) / 5 mm (gripper) outward error: **372 spheres**, 100 % coverage of hull vertices, 40k surface samples and an interior grid. Build time 92 s in `.venv-sim`. A naive k-means cover bulged 34 mm on the hand and was rejected. |
| FK | Batched numpy FK equals `robodojo_direct_sim.PandaKinematics.forward` to **2.2e-16**. Placing the gripper from state8 alone equals full FK to **4.4e-16**. |
| Palm profile | Hull-derived p(ξ) equals `robot_facts_v2.json palm_profile_mm`: median error 0.003 mm, 31.043 vs 31.04 mm at ξ = 0. |
| T08 press, true bottle box | With the recorded states, the model flags **finger1** at the first post-onset state (recorded onset is the step 37→38 transition; min −7.9 mm). There are 0 negative samples before onset. |
| T08 stage, candidate profiles | `translate_first` (the OSC-lag profile that ran): **−22.0 mm**. `coupled` slerp: −0.8 mm, inside the 5 mm model conservatism. `rotate_first` at the hover: **+15.3 mm**. The planner picks `rotate_first`, which is what H3 did, with 0 contacts. |
| Cost | Per step (K=1, all 319 moving spheres): **≈1.0-1.2 ms**. One stage sweep: K=60, 20 boxes **≈8 ms**; K=200, 20 boxes **≈18 ms**. All-pairs checking without a broad phase costs 54 ms and 257 ms, so the broad phase is mandatory. Gripper-only (no joints) K=60: 1.5 ms. The budget is 155-198 ms of non-model time per step; H2 costs 107-136 ms per configuration. |
| Arm-joint prediction | A minimum-norm resolved-rate lift over the recorded Cartesian states reproduces the measured joints at the next request boundary (median 38-52 steps later). Link-origin error: median **1.4 / 2.5 / 3.1 mm**, p90 17.4 / 9.9 / 16.7 mm (T08 / T04 / drawer 001). Drawer 001 reaches 139 mm max at the joint-4 limit. A nullspace pull toward the reset posture is 3-10× worse. Cost 0.6-0.7 ms per lifted sample. |
| OSC response | s_t = a·s_{t−1} + b·c_t. a = 0.58-0.79 (pooled 0.705), steady realisation 0.24-0.26, R² 0.95-0.99. Braking: a/(1−a) = 1.4-3.8 steps of coasting per unit per-step speed. |
| Depth bound, truth AABBs | At the 3 T04 closes that produced palm contact, the exact hand hull sat **3.2-22.6 mm** below the mug AABB top over the footprint. At 4 contact-free closes (H1-default ×2, H3 ×2) it sat **0.1-11.1 mm** below. The ranges overlap, so an AABB-top bound cannot separate them; it would have stopped H3's second grasp ≥ 11 mm higher. The palm bound must use measured heights per footprint cell. |

---

## 1. Why H1's router never detoured, and why a point router cannot prevent T08

### 1.1 Zero detours in 117 routed stages
Verified: 111 straight, 0 A* calls, 0 detours, 6 + 1 `route_start_blocked`; verify_h1_collision F4. The causes:

1. **Refuse before search.** Start and goal validity are tested before any planning (`_routing.py:105-108`). A start or goal touching an expanded box is refused, never escaped. A* (`:111-143`) runs only when both endpoints are valid *and* the straight segment is blocked.
2. **The model planned around obstacles itself.** It used high waypoints: in T04 it carried at z 0.62-0.64 m ("the elevated transfer avoids…"); in T08 it carried at 1.22-1.26 m. Straight segments were therefore valid against its own boxes.
3. **The boxes were wrong.** The first maps missed by 5-12 cm. T08's first bottle box had IoU 0, its top 6.6 cm below the bottle top, and its bottom 10 cm below the table. Segments that really passed the bottle were "valid" against a box that was elsewhere.
4. **A point is the only checked geometry** (`_routing.py:3-4`). The parts that actually collided were never in the check: finger1 at +52.7 mm along the jaw axis, link6/link7, the palm, and the held mug. No blocked segment could ever arise from them.
5. **A hidden margin, but on the wrong geometry.** The 8-corner rule (`fine_distance_planner.py:101-106`) adds 0-20 mm per axis to the *grip site*. This made 5/7 refusals spurious. Even with the true T08 box it invalidated the straight path, but A* then returned the same descent within 1 mm, because the grip site itself stayed ≥ 27 mm from the bottle (cross_critic G5).
6. **Lookahead tracking was never exercised on a detour.** The 0.10 m lookahead only validates the grip-site chord (`tracking.py:116-141`; G5).

### 1.2 Why a point router cannot stop the T08 finger contact, even with the true box
- **The motion was a rotation in place.** The contact point is on finger1 at R1 = +52.7 mm, R2 ≈ +16 mm from the grip site. It was swept by an **in-place rotation** of the jaw axis, from (0.72, −0.69) to (1.0, −0.01), while z stayed pinned at 1.049-1.054. OSC realises about 0.023 rad per step against a 0.1 rad command, so 90° takes about 68 steps against about 15 for the translation (C5).
  - A point router sees no translation during the rotation, so it has nothing to check.
  - Orientation-dependent geometry cannot be expressed as an isotropic point clearance.
- **Inflating enough would block every grasp.** Covering the open finger body needs ≥ 66 mm (`finger_and_pad_R1_mm` at q = 40 mm). A margin that large would put every grasp target inside an expanded box (IC12).
- **The prototype confirms it.** With the true AABB, the finger spheres go negative from the post-step-37 state while the grip site is ≥ 27 mm away. Among the executable sequencings only `rotate_first` is clear (+15.3 mm); `coupled` (−0.8 mm) and `translate_first` (−22 mm) are not.
- **So the fix needs three things together:** the body model, the *executed* rotation timing (ws_executor_control B1), and the sequencing choice made by the planner.

---

## 2. Architecture (per stage)

```
model stage (skill, target pose, profile, contact declaration)        [ws_protocol_context]
   │
   ▼
StageRequest ──► LocalBodyPlanner.plan_stage ───────────────► StagePlan(legs, speed schedule, report)
                   │  uses:                                         │
                   │  ObjectMapSnapshot (ids, OBBs, bounds) [ws_geometry_tools/ws_perception_service]
                   │  PermissionMatrix (from declaration)           │
                   │  HeldAttachment (attached at grasp)            │
                   │  ExecutedProfileModel (sequencing, lag, band)  [ws_executor_control]
                   │  BodyKinematics.lift (arm joints)              │
                   │  SweptChecker (spheres vs OBB/plane)           │
                   ▼                                                ▼
             body_plan report ──► feedback to model        BodyRoute + BodyChordValidator ──► executor tracker [ws_executor_control]
                                                               per step: monitor_step(measured state8 + joints)
```

The new package is `src/gpt_field/swept/`. It contains only numpy and the standard library, and it is importable in the controller `.venv`.

| Module | Role |
|---|---|
| `assets/panda_body_v1.npz` and `.json` | Built offline. Hash manifest. |
| `kinematics.py` | Vendored PandaKinematics, batched FK, the lift. |
| `body.py` | Sphere sets, placement, part classes. |
| `distance.py` | Sphere–OBB and sphere–plane signed distance, broad phase. |
| `scene.py` | Object map → arrays, permissions. |
| `profile.py` | ExecutedProfileModel adapter plus a measured fallback. |
| `checker.py` | SweptChecker. |
| `planner.py` | LocalBodyPlanner. |
| `held.py` | Attachment. |
| `depth_bound.py`, `speed.py` | Depth bound and speed coupling. |
| `route.py`, `chord.py` | BodyRoute (a TransitRoute-compatible adapter) and the chord validator. |
| `report.py` | `body_plan` JSON. |
| `point_v2.py` | B2 point router. |

---

## 3. Components

Each component lists: Purpose, Files, Interfaces, Algorithm, Veto removed / parameter added, Depends on / interactions, Offline tests (pass criteria), Live validation, Expected effect, Risks, and Effort.

### SP1. Vendored robot geometry and offline body-model build

**Purpose.**
- R2 / C4: "reuse foundation_harness's robot geometry".
- README prerequisites (b) and (c): one interpreter; a vectorised checker.
- G6: the controller venv cannot import H2's `robot_geometry`, which needs scipy (`robot_geometry.py:16`).

**Files (create).**
- `third_party/robot_geometry/panda_robosuite_1.4.1/`: byte copies of `robots/panda/robot.xml` and `grippers/panda_gripper.xml`, plus the collision STLs `robots/panda/meshes/link0-7.stl`, `grippers/meshes/panda_gripper/hand.stl` and `finger.stl`, from `agent_speedup_v2_obstacle_harness_100mm/.venv-sim/lib/python3.10/site-packages/robosuite/models/assets/`. Add `SOURCE.json` with the robosuite version and per-file sha256.
  - robot.xml `263c0f9a…`, panda_gripper.xml `d9d68123…`, link0 `dfc6d943…` … link7 `92ac6afc…`, hand `94493e94…`, finger `2d07a740…`. The full list is in Appendix B.
  - LIBERO's `MountedPanda` loads exactly this robot.xml plus PandaGripper (`third_party/LIBERO/libero/libero/envs/robots/mounted_panda.py:14-15`).
- `third_party/foundation_harness_k2/robot_geometry.py` (sha256 `8fa39c48…`, equal to the pin in `h2_keypose2_snapshot/prep/implementation.json`), `motion.py` (`b15d07a1…`) and `robodojo_direct_sim.py` (`6d2746cb…`, from `/workspace/gpt_field/src/gpt_field/`), each with `SNAPSHOT.json`. **These copies are never imported at runtime.** They are used only by the parity test, which runs in an interpreter with scipy.
- `tools/build_body_model.py`. It runs in `.venv-sim` (scipy 1.15.3) or `.venv-foundation-vision`. It must not import the live `foundation_harness` tree.
- `src/gpt_field/swept/assets/panda_body_v1.npz` and `panda_body_v1.json` (committed build outputs).
- `tests/test_swept_model_build.py`.

**Interfaces.** The manifest `panda_body_v1.json` follows this schema:
```json
{"schema":"swept-body-model-v1","robot_xml_sha256":"…","gripper_xml_sha256":"…","mesh_sha256":{"link0.stl":"…"},
 "builder_sha256":"…","numpy":"1.26.4","scipy":"1.15.3","rng_seed":0,"beta_by_kind_m":{"arm":0.010,"hand":0.004,"finger":0.004,"pad":0.004},
 "bodies":[{"index":0,"name":"base","parent":-1,"xml":"arm","T":[[4x4]],"joint":null|{"name":"joint1","type":"hinge","axis":"0 0 1","pos":"0 0 0","range":"-2.8973 2.8973"}}],
 "parts":[{"index":8,"name":"hand_collision","kind":"hand","body":13,"body_name":"right_gripper","spheres":101,
           "covered_validation":true,"validation_points":40000,"bulge_plane_mm":6.17,"grown_mm":1.1,"max_radius_mm":33.3,
           "hull_extent_mm":[204.4,63.3,91.9]}],
 "palm_profile_mm":[[xi,p],…], "footprints":{"palm_R0_mm":[-31.64,31.62],"palm_R1_mm":[-103.99,100.43]},
 "npz_sha256":"…"}
```
The npz holds, per part: `p{i}_centers (S,3)`, `p{i}_radii (S,)` (body frame), `p{i}_hull_vertices`, `p{i}_hull_equations` (n·x + d ≤ 0 inside, |n| = 1), and `p{i}_bsphere (4,)`.

**Algorithm.**
1. Parse both MJCFs. Use the same visual-only rule as H2: skip a geom when `contype == 0` and `conaffinity == 0` (`robot_geometry.py:194`). Note that finger collision geoms have contype 0 and conaffinity 1, so they are kept. Use the same kind naming: names containing `_pad_` are pads, names containing `finger` are fingers (`:230`), and `hand` gives the hand kind.
2. Read the vertices with the same STL reader (`:25-36`). Apply the geom pos, quat and scale.
3. Compute a ConvexHull to get the vertices and the unit-normal equations.
4. Build a **bulge-bounded greedy sphere cover**:
   - Candidate centres: an interior grid at step clamp(min extent/8, 2 mm, 6 mm), plus hull vertices pulled 0.1 % inward.
   - Candidate radius: r(c) = β + depth(c), where depth(c) = −maxᵢ(nᵢ·c + dᵢ). By construction each ball's outward error against every face plane is exactly β (closed form).
   - Targets: hull vertices, 6,000 area-weighted surface samples and the interior grid.
   - Greedy set cover (most newly covered targets first).
   - Validate with 40,000 fresh surface samples plus the interior grid. Any uncovered point grows its nearest ball; record `grown_mm`.
5. Derive and store the palm profile p(ξ) from the hand hull. This is a 2-D linear program per ξ over the hull planes in the right_hand frame. Also store the footprints for SP8.
6. Store per-part bounding spheres for the broad phase.

**Veto removed / parameter added.** Adds β, the model resolution. β is a build-time, robot-geometry parameter that is reported, never a gate.
- β_gripper = 4 mm, the controller's transit arrival tolerance (`_fragments.py:29`, `position_tolerance_m=.004`). The model is then never more conservative than what the controller already accepts as "arrived".
- β_arm = 10 mm, below the measured p90 arm-prediction error (10-17 mm, SP2), so the sphere error never dominates the arm budget.
- Prototype at 5/12 mm: 372 spheres. At 3/8 mm: 737 (hand alone 233). 4/10 mm is expected to give about 450-600.

**Depends on.** Nothing.

**Offline tests (pass criteria).**
- T1.1: 100 % of validation points are inside the union, for every part.
- T1.2: plane-bulge ≤ β + 2 mm for every part.
- T1.3: H2 parity. In `.venv-sim`, run the vendored `robot_geometry.RobotCollisionModel(kin)` with the vendored `robodojo_direct_sim`. At q = home and 5 random q, the world hull vertices of every collision geom equal ours within 1e-9 m.
- T1.4: p(ξ) equals `robot_facts_v2.json palm_profile_mm` within 0.5 mm at all ξ except the two degenerate endpoints. The prototype got median 0.003 mm.
- T1.5: rebuilding gives the same npz sha256; the manifest hashes equal the vendored files.
- T1.6: a setup check that the vendored sha256 values equal the files in `.venv-sim/.../robosuite/models/assets`. The sim worker reports them (shared interface I10).

**Live validation.** At startup the build digest records `swept_body_model_sha256`. The runner refuses to run in `on` mode if the simulator's asset hashes differ.

**Expected effect.** Enabler only. One model at 1-2 ms per configuration replaces H2's 107-136 ms per configuration (measured).

**Risks.**
- Sphere conservatism β causes false near-contacts. Mitigation: the palm vs the declared grasp target is governed by the exact footprint (SP8), not by the spheres. Clearance is reported as an interval [c, c + β].
- Mesh or version drift. Mitigation: the hash checks in T1.6.

**Effort.** 2.0 d.

---

### SP2. numpy-only kinematics (batched FK, binding, resolved-rate lift)

**Purpose.**
- Arm-link sweeps need joints along *predicted* motion.
- Per-step checks need measured joints. These are already delivered to the policy as `view.observation.joint_position` (`agent_speedup_v2_user_check_sim.py:72-80, 122-128`), but they are stripped from the trace by `to_dict` (`:91-96`).
- K11 / H2 F12: plan in Cartesian space, as OSC executes, rather than in joint space.

**Files.**
- Create `src/gpt_field/swept/kinematics.py`. It contains a vendored, hash-pinned copy of `PandaKinematics` (`robodojo_direct_sim.py:121-262`, which is numpy-only) and the batched code.
- Create `tests/test_swept_kinematics.py`.

**Interfaces.**
```python
class BodyKinematics:
    def __init__(self, asset: BodyModelAsset): ...
    def bind(self, state8: np.ndarray, q_arm: np.ndarray) -> np.ndarray            # world_from_base (4,4), cached per episode
    def check_binding(self, state8, q_arm, tol_m=1e-6) -> dict                     # {"ok": bool, "err_m": float}
    def body_poses(self, q_arm: np.ndarray, q_fingers: np.ndarray, world_from_base) -> dict[int, np.ndarray]   # (K,4,4) per body
    def gripper_poses_from_state8(self, states8: np.ndarray) -> dict[int, np.ndarray]                          # no joints needed
    def lift(self, q0: np.ndarray, states8: np.ndarray, *, every_m=0.02, every_rad=0.10) -> LiftResult
@dataclass(frozen=True)
class LiftResult: q: np.ndarray  # (K,7), lifted at coarse spacing, linearly interpolated between lifts
                  limit_hits: list[dict]   # [{"k": int, "joint": 4, "margin_rad": 0.0}]
                  residual_m: np.ndarray   # (K,) Cartesian residual after each lifted sample
```

**Algorithm.**
- **Binding.** world_from_base = measured ⋅ FK(q)⁻¹, as in `robodojo_direct_sim.py:196-206`. Compute it once per episode, then check it every step (cheap).
- **Gripper from state8.** T_hand = [R(rotvec), p_grip − R·(0, 0, 0.097)]. Then the fixed gripper chain applies, with fingers at state8[6:8].
- **Lift.** Closed-loop minimum-norm resolved rate: qₖ₊₁ = qₖ + J⁺(qₖ)·eₖ, with damping 1e-6, clipped to the limits and **no nullspace term**. This matches OSC's measured behaviour best (see the prototype facts). Lift a sample every 2 cm of grip travel or every 0.1 rad, and interpolate joints between lifts.
- **Joint limits.** A limit hit (|q − limit| < 1e-3 while the residual grows) is reported. It is the input to C5 (shared interface I11).

**Veto removed / parameter added.**
- Parameter: the arm prediction band = the measured p90 lift error (default 17 mm, re-estimated online from the per-step joints in new runs).
- The lift spacing (2 cm / 0.1 rad) is set so the joint-interpolation error stays below β_arm. It is verified in T2.3.

**Depends on.** SP1. The runner's per-step joint sidecar (I10) is needed for the online band.

**Offline tests.**
- T2.1: FK parity with `PandaKinematics.forward` on 64 random q, ≤ 1e-12 m. The prototype got 2.2e-16.
- T2.2: the state8 placement equals the bound FK placement within ≤ 1e-9 m. The prototype got 4.4e-16.
- T2.3: lift reconstruction between request-boundary joints. The fixtures come from the context's `measured_state.joint_position_rad` in `trial/requests/*.request.json` of T08, T04 and drawer 001.
  - Pass: median link-origin error ≤ 5 mm and p90 ≤ 20 mm, excluding intervals flagged `limit_hit`.
  - Pass: drawer 001's 139 mm outlier is flagged `limit_hit`.
  - Pass: ≤ 1 ms per lifted sample.
- T2.4: a +1 cm base error is caught by `check_binding` at step 1.

**Live validation.**
- Every step logs the prediction-vs-measured joint drift for the stage in progress.
- Sweep metric: the lift error distribution per task. It updates the arm band.

**Expected effect.** Enabler. It covers link6/link7 contacts in direct and recovery moves (drawer 001: 414 samples, 135 of them in `local_motion`).

**Risks.**
- Joint limits and singularities; the drawer 001 error was 14 cm. Mitigation: `limit_hits` triggers the C5 posture check and inflates the arm band for the rest of the stage. During execution, the per-step measured joints override the prediction.

**Effort.** 1.5 d.

---

### SP3. Swept checker core (spheres vs OBB and planes, broad phase, permissions)

**Purpose.**
- F1 / C4 (body sweep on all stages).
- The measured time budget.
- README prerequisite (c).

**Files.**
- Create `src/gpt_field/swept/body.py`, `distance.py` and `checker.py`.
- Create `tests/test_swept_checker.py` and the fixtures `validation/fixtures/swept/t08_press.json`. The T08 fixture holds steps 26-84: state8, forbidden increments and pairs, plus the true bottle AABB from `inputs/layout_manifest.json bottle.collision_bounds`.

**Interfaces.**
```python
PART_CLASSES = ("arm", "palm", "finger", "pad", "held")      # link1..7 / hand_collision / finger1,2 / pads / attachment
def sphere_obb_sd(centers: np.ndarray, radii: np.ndarray, c: np.ndarray, R: np.ndarray, h: np.ndarray) -> np.ndarray  # (K,S,M)
def sphere_plane_sd(centers, radii, planes: np.ndarray) -> np.ndarray                                                # (K,S,P)
class SweptChecker:
    def __init__(self, asset, kin: BodyKinematics): ...
    def check(self, cfg: ConfigBatch, scene: SceneArrays, perms: PermissionMatrix, *,
              held: HeldAttachment | None = None, bands: Bands | None = None, parts: Sequence[str] = PART_CLASSES) -> SweepReport
@dataclass class ConfigBatch: states8: np.ndarray (K,8); q_arm: np.ndarray | None (K,7); tracking_band_m: np.ndarray (K,)
```
`SweepReport` (JSON-serialisable):
```json
{"ok":true,"first_violation":null,
 "min_clearance_by_class":{"finger":{"m":0.0153,"object_id":"obj_bottle","part":"finger1_collision","k":59},"arm":{"m":0.21,"object_id":"table","k":0}},
 "clearance_curve_m":[0.031,0.029],"conservatism_m":{"gripper":0.004,"arm":0.010},
 "pairs_evaluated":12,"timing_ms":{"fk":1.0,"place":0.6,"broad":2.1,"narrow":4.5}}
```

**Algorithm.**
1. Place spheres. Gripper parts come from state8 (no joints needed). Arm parts come from `q_arm`; parts are skipped when it is None.
2. Pad every radius by the sample's tracking band, plus half that sphere's displacement to the next sample. This bounds tunnelling. Samples are spaced so that no sphere moves more than its own radius per sample.
3. Broad phase: per-part bounding sphere vs every OBB inflated by its bound, and vs the planes, giving a (K, P, M) array. Keep the (part, object) pairs whose minimum < pad.
4. Narrow phase: exact sphere–OBB signed distance for the surviving pairs only.
5. Apply the permission matrix: `forbidden`, `permitted`, or `permitted_nonincreasing` (allowed only while penetration does not grow).
6. Per sample: minimum over forbidden pairs, the first violation, and per-class minima.

Exactness: the sphere clearance is ≤ the hull clearance, which is ≤ the sphere clearance + β, because the union contains the hull with plane-bulge β. Self-pairs (held vs arm links 1-6, hand/fingers vs links 0-4) are an optional set, enabled by default for the held object.

**Veto removed / parameter added.** Removes H1's hidden lattice margin (`fine_distance_planner.py:101-106`) and the grip-site-only scope (`_routing.py:3-4`). Adds no task parameter: the pads come from measured bands and geometry.

**Depends on.** SP1, SP2.

**Offline tests.**
- T3.1: sphere–OBB signed distance against dense brute force for 1,000 random boxes and rotations, error ≤ 1e-9.
- T3.2: the T08 fixture with the true bottle AABB, measured profile.
  - First negative at the first post-onset state (onset at step 37); nearest part `finger1_collision`.
  - 0 negatives before onset.
  - min ≈ −8 mm (prototype −7.9 mm).
  - Only 14 of the 47 recorded contact steps are negative against the *initial* AABB; the finger kept pressing a bottle that had moved. Truth for later steps needs SP12's enrichment, so score by episode onset, not by sample.
- T3.3: timing in the controller `.venv`, single thread.
  - K ≤ 6, M ≤ 20: p95 ≤ 3 ms.
  - K = 60, M = 20: ≤ 15 ms (prototype 8.2 ms).
  - K = 200, M = 20: ≤ 30 ms (prototype 18.5 ms).
- T3.4: permissions. Finger vs a permitted target is not reported. Palm vs the same target is reported. `permitted_nonincreasing` is reported only when penetration grows.
- T3.5: no tunnelling. A 2 mm-thick box placed between two samples 3 cm apart is detected.

**Live validation.** Shadow mode logs per-step clearances. Sweep metric: per-step check time p95 ≤ 3 ms.

**Expected effect.**
- With truth-quality geometry it catches the routed finger and link contacts: T08 3,504 samples; drawer 001 link6/link7 vs cabinet, 414 samples.
- It extends coverage from the 24 % of samples in routed stages to all arm stages.
- Estimate: ≥ 90 % recall at onset for classes A-C (SP12).

**Risks.**
- Noisy-host timing. Mitigation: a budget (SP6) and measured p95 values in the sweep.
- β false positives at tight grasps. Mitigation: SP8 governs the palm vs the target.

**Effort.** 3.0 d.

---

### SP4. Executed-profile adapter (rotation timing, tracking band)

**Purpose.**
- M1 / IC3 / B1: the sweep must model what OSC *executes*. Rotation realises about 0.023 rad per step against a 0.1 rad command, so stages run as "translate, then rotate in place".
- Checking the commanded straight line with an interpolated orientation is unsound; verify_h1_collision gap 2.

**Files.**
- Create `src/gpt_field/swept/profile.py`, `tests/test_swept_profile.py` and the fixture `validation/fixtures/swept/osc_steps.json` (compact steps of the 7 runs).

**Interfaces.** The provider is ws_executor_control (shared interface I3); this module only adapts it.
```python
class ExecutedProfileModel(Protocol):
    version: str
    def sequencing_options(self, entry_state8, target_xyz, target_wxyz, profile: str) -> list[str]  # subset of
        # ["executor_default", "coupled", "rotate_first", "translate_first"]
    def predict(self, entry_state8, entry_velocity6, waypoints: list[Pose], *, sequencing: str, profile: str,
                speed_cap: Callable[[float], float] | None = None) -> PredictedProfile
    def osc_response(self) -> dict      # {"a":0.705,"b":0.074,"a_p90":0.79,"rotation_rad_per_step":0.023,"updated_at_step":n}
    def guard_latency_steps(self) -> int
@dataclass class PredictedProfile: states8 (K,8); step (K,); arc_m (K,); tracking_band_m (K,); arrival_step: int; sequencing: str
```
If ws_executor_control's model is not yet available, `profile.FallbackProfileModel` implements the same protocol:
- translation by s_t = a·s_{t−1} + b·c_t, with the pooled fit a = 0.705, b = 0.074 (per run R² 0.95-0.99), and commands from H1's carrot law capped at the profile command;
- rotation at a constant realised rate ω (0.023 rad/step at the 0.1 rad command cap);
- band = the p90 residual of the fit;
- the three sequencing modes.

**Algorithm.** For each candidate (waypoints, sequencing), predict a per-step state8 sequence and its band. The checker then sweeps this sequence, never the commanded line.

**Veto removed / parameter added.** Joint with ws_executor_control, this replaces the 1.2 rad stage envelope veto (`_fragments.py:28, 288-291`; it cost 7.2/7.7 s of repair calls in T04/T08) with sequencing modes whose executed sweep is checked. No new threshold: a, b and ω are measured.

**Depends on.** ws_executor_control I3/I4, with the fallback above.

**Offline tests.**
- T4.1: on the recorded T04/T08 transit stages, the measured grip path lies inside the predicted band at p90.
- T4.2: the T08 fixture (entry at step 28, z 1.077; target (−0.025, −0.023, 1.035); 0.97 rad remaining) reproduces the signs: `translate_first` < −15 mm, `coupled` in [−β, +β], `rotate_first` ≥ +10 mm. The prototype gave −22.0, −0.8 and +15.3 mm.
- T4.3: the fallback's predicted arrival step is within ±25 % of the recorded arrival for ≥ 80 % of transit stages.

**Live validation.** Every stage logs the predicted vs measured arrival step and path deviation. Sweep metric: deviation p90.

**Expected effect.** Together with B1 it prevents T08-type rotation sweeps: 3,504 samples, plus the call chain of calls 2-5 (C11). It also saves the veto repair calls, about 7-8 s per occurrence (C6).

**Risks.**
- A wrong lag model. Mitigation: the per-step monitor (SP6) replans on deviation; a, b and ω are estimated online.

**Effort.** 2.0 d.

---

### SP5. Scene adapter and per-part contact permissions by object ID

**Purpose.**
- M3: without object identity, the body sweep cannot cover the palm class (55 %), because the prompt removes the target from the avoid set (`_agent_client.py:101-102`).
- F9: an object-centric map.
- IC13: held, target and robot geometry must not be generic obstacles.

**Files.**
- Create `src/gpt_field/swept/scene.py` and `tests/test_swept_scene.py`.

**Interfaces.** Consumes `ObjectMapSnapshot` (I1) and the stage contact declaration (I8).
```python
@dataclass class SceneArrays: obb_c (M,3); obb_R (M,3,3); obb_h (M,3); bound_m (M,); ids: list[str]; roles: list[set[str]];
                              planes (P,4); plane_bound_m (P,); plane_ids; revision: int; observed_at_step: int
def scene_from_map(snapshot: dict, *, exclude_ids=(), held: HeldAttachment | None = None) -> SceneArrays
def permissions(scene: SceneArrays, decl: ContactDeclaration | None, *, stage_kind: str, held, legacy_target_xyz=None) -> PermissionMatrix
```
The permission rules are general; there are no per-task tables.

| Part class \ Object | Declared contact object (`parts` ∋ class) | Other objects and fixtures | Support planes (table) | Held object |
|---|---|---|---|---|
| finger / pad | permitted | forbidden | forbidden | permitted (gripped) |
| palm | permitted only for intent `push` or `insert` | forbidden | forbidden | forbidden, reported as `palm_on_held` (grasp too deep) |
| arm links | forbidden | forbidden | forbidden | forbidden (self) |
| held | the declared place support in a `place` stage; the pick support in the first lift leg (`permitted_nonincreasing`) | forbidden | as in the column to the left | — |

**Legacy inference, for replay and shadow on the old protocol only.** The declared object is the map object nearest the target of an `approach` or `set_gripper` stage, and the palm is never permitted. In `on` mode a declaration is required for any contact stage. Without one, every contact is forbidden and the guard is the terminator.

**Bounds.**
- Each object's `bound_m` is taken as reported. It is the perception uncertainty bound measured on the truth-scored benchmark (ws_perception_service, I1). No factor is added.
- For legacy model boxes, bound = the model's `clearance_m`, so they remain comparable with H1.

**Veto removed / parameter added.** Removes the single global model clearance (`_fragments.py:58`, range 0-0.20 m) and the "remove the target from the avoid set" workaround. Parameters: none. Bounds are measured upstream.

**Depends on.** I1 (ws_geometry_tools / ws_perception_service) and I8 (ws_protocol_context).

**Offline tests.**
- T5.1: the permission table as unit cases.
- T5.2: legacy inference on the 9 runs' approach stages selects the object the fingers actually touched in ≥ 90 % of `finger_object` contacts.
- T5.3: a snapshot with a stale revision is flagged `stale_map` (event freshness, IC14).

**Live validation.** Every `body_plan` records `map_revision` and `observed_at_step`. Sweep metric: the share of stages planned on a map older than the last robot-object event.

**Expected effect.** It makes palm-on-target, held-object and arm-link contacts expressible. Coverage of the 55 % palm class also needs SP8.

**Risks.**
- Missing declarations make contact impossible to plan. Mitigation: the guard terminator; the protocol requires declarations (I8).

**Effort.** 1.5 d.

---

### SP6. LocalBodyPlanner (escape, direct and sequencing, detours, standoff handoff, `no_route`)

**Purpose.**
- F1 / F4 / F7, B2 (the escape part), C4, IC4, IC12, K3 (clearance must not decrease), K6 (no 12 s RRT waits).
- "Return to the model only when no route exists."

**Files.**
- Create `src/gpt_field/swept/planner.py`, `report.py` and `tests/test_swept_planner.py`.

**Interfaces.**
```python
@dataclass class StageRequest: skill: str; target_xyz; target_wxyz; profile: str; contact: ContactDeclaration | None;
                               stop_on: tuple[str, ...]; next_stage: "StageRequest | None"; stage_key: str
@dataclass class MeasuredState: state8; q_arm; qd_arm; step: int
class LocalBodyPlanner:
    def __init__(self, checker: SweptChecker, kin: BodyKinematics, profiles: ExecutedProfileModel, speed: "SpeedPolicy"): ...
    def plan_stage(self, req: StageRequest, now: MeasuredState, scene: SceneArrays, perms: PermissionMatrix,
                   held: HeldAttachment | None, *, budget_ms: float = 150.) -> "StagePlan"
    def monitor_step(self, plan: "StagePlan", now: MeasuredState, scene: SceneArrays) -> "StepVerdict"  # continue|replan|hold_handoff
@dataclass class Leg: kind: str; waypoints: list[Pose]; sequencing: str; predicted: PredictedProfile; min_clearance_m: float;
                      speed_cap: Callable[[float], float]; reason: dict | None
@dataclass class StagePlan: status: str; legs: list[Leg]; report: dict  # body_plan JSON, schema in §4.1
```

**Algorithm (stage entry).**
0. **Bands.** Per-sample tracking band from `PredictedProfile`. The arm prediction band from SP2 applies to predicted samples only (0 for measured configurations).
1. **Start assessment** on the measured configuration (exact joints): find the set V of forbidden pairs with clearance < 0 against the bound-inflated geometry. Tag each as `in_band` or `penetrating` (inside the unexpanded geometry).
2. **Escape**, when V is non-empty.
   - Candidate directions, ordered by the rate at which they increase min over V of the clearance:
     - the signed-distance gradient of each violating sphere (unit vector from the box's closest point to the sphere centre, averaged over V);
     - −approach (hand −z);
     - +world z;
     - the two horizontal directions perpendicular to the approach.
   - Leg length L ≤ L_max = max over V of (bound + band + penetration) + 1 mm.
   - Accept a candidate only if:
     - (i) every violating part's distance from its **unexpanded** box strictly increases at every sample (README B2);
     - (ii) every other forbidden pair keeps clearance ≥ min(its start value, 0). This is K3's "clearance must not decrease" rule, and it prevents escaping into the object just grasped or released (verify gap 3);
     - (iii) violators reach clearance ≥ 0 by the end of the leg. Otherwise the leg ends at the last increase and the status is `escape_partial`.
   - Vertical is a candidate, not a default (README B2 caution). It wins when the gradient points up, which the beliefs in the replayed refusals suggest ("vertical space … clear"). This reconciles README B2 with IC4.
3. **Main-leg candidates.** The first one that passes wins. The gripper and held object are screened first for all candidates (≈1.5 ms each); the arm is lifted and checked only for the surviving candidates.
   - a. Direct, with `sequencing_options` in the order the executor gives (executor default first), then `rotate_first`, `coupled`, `translate_first`.
   - b. **Lift-over.** z_req = max over blocking objects of (top + bound) + below_grip(orientation, held) + band, where below_grip = max over gripper and held spheres of (grip_z − (centre_z − r)) at the stage orientation. Waypoints: start → (start.xy, z_req) → (goal.xy, z_req) → goal. This is the "minimum safe carry height" of F7, derived from geometry. With the T04 held mug it gives grip ≥ 0.513 + bound + ≈0.065 m, consistent with verify's ≥ 0.57 m.
   - c. **Sidestep.** Offset the segment horizontally, perpendicular to it at the blocking object, by (object half-width along the offset + bound + gripper/held half-width + band). Try the shorter side first.
   - d. **L-shapes** (horizontal-first or vertical-first; for lift-out of containers, e.g. bowl vs cabinet) and lift + sidestep combinations.
   - e. Optional, phase 3: grip-site lattice A* using the existing `FineDistancePlanner`, with boxes inflated by the hand's horizontal half-extent at the stage orientation, as a polyline generator, followed by a body check.
4. **Goal assessment.** If the goal violates a forbidden pair, line-search back along the approach direction (for transit, along −(goal − start)) to the first clear pose. That pose is the standoff. The leg ends there and records `handoff = {object_id, part, standoff_xyz, remaining_offset_m}`.
   - If `req.next_stage` is a contact stage declaring that object, the remaining offset is added to its displacement, and the status is `merged_into_contact_stage`. This avoids shortening relative stages (K8's re-basing issue).
   - Otherwise, return to the model with the record, without moving beyond the standoff.
5. **`no_route`**, only when every candidate fails inside the budget. The evidence lists the blocking pairs per candidate and the nearest feasible alternative (minimum lift z, free side). If the budget runs out, the status is `no_route_budget` with the best candidate evaluated; there are never 12 s waits (K6).
6. **Direct `motion`** (`LocalMotionPrimitive`, ≤ 5 steps, ≤ 5 cm). Same check, no detours. The motion is truncated at the last clear sample, as `truncated_at_clearance`, rather than refused.
7. **`set_gripper`.** Before closing or opening, sweep the jaw (fingers from the current q to 0 or 0.04) against forbidden objects. If it is blocked, return `jaw_sweep_blocked` with the pair; no motion.

**Algorithm (per step, `monitor_step`, ≤ 3 ms).**
- Check the measured configuration (state8 plus measured joints) and the tracker's next chord (SP7).
- If a forbidden clearance is < 0 **and decreasing**: hold, replan once from the current state (with escape semantics), else hand off `blocked_by_geometry` with the nearest pairs.
- Being inside a bound band while clearance is non-decreasing is allowed. This replaces `tracking.py:83-84` (`invalid_measured_start`).
- The nearest-pair list also fills `contact_or_collision_cause` (`_fragments.py:482`, which today is always "unknown") and the guard's `blocked_motion` feedback (I7).

**Veto removed / parameter added.**
- Removes `route_start_blocked`, `route_goal_blocked` and `route_goal_connector_blocked` (`_routing.py:105-108, 147-148`) → escape and standoff handoff.
- Removes `invalid_measured_start` (`tracking.py:83-84`) → the monotone-clearance rule.
- Removes the hidden lattice margin → explicit reported bounds.
- Joint with SP4 and B1: the 1.2 rad envelope.
- Removes `route_exceeds_local_map` (`_routing.py:74-75`); the body planner has no lattice-size limit.
- Parameters, all derived: L_max = bound + band + penetration; budget_ms = 150, under one step of today's non-model time (0.155-0.198 s). The candidate order is fixed and general.

**Depends on.** SP3, SP4, SP5. The tracker comes from ws_executor_control (I6).

**Offline tests.**
- T6.1: the T08 fixture with the true AABB → status `direct` with `rotate_first`; min clearance ≥ +10 mm; planning ≤ 50 ms.
- T6.2: escape cases.
  - (a) the T08 stove-off step-179 start, 52 mm inside the model's own box → an escape leg that is monotone and does not enter the other boxes;
  - (b) a synthetic case of the hand just above a grasped object → the escape never re-enters that object;
  - (c) a start inside the band only → a short escape ≤ L_max.
- T6.3: T04 carry, with the true can and the held mug attached at close. A straight carry at grip z 0.55-0.56 m → the planner returns a lift-over with z_req in [0.57, 0.60] m.
- T6.4: a goal inside the target's inflated box with the next stage `approach` declaring it → `merged_into_contact_stage`, and the remaining offset is propagated.
- T6.5: budget. 20 objects and 8 candidates finish ≤ 150 ms at p95.
- T6.6: on the routed stages of the 9 runs with **model boxes** (SP12 R-B), verify's 21/111 hand-sweep hits (14 at start, 4 at goal, 3 mid-path) resolve locally: ≥ 12 escapes, ≥ 3 handoffs or merges, ≥ 2 detours, `no_route` ≤ 2/111.

**Live validation.**
- Capped smoke L6-a (shadow): T08 + bottle.
- Capped smoke L6-b (`on`, with a perception map): T04 + can.
- Both are in §6. Sweep metrics: 0 `route_start_blocked`; escapes, detours, handoffs and `no_route` per stage type; forbidden episodes in routed stages (target 0); calls per task ≤ baseline.

**Expected effect (estimates).**
- Exact distances fix 5/7 refusals; escape fixes the other 2. That is about −6 calls (≈ −70 s) across the 9 runs (verify F4).
- The 21 would-be veto stages become local actions, so 0 added calls, where a veto would cost about +3 calls per run (≈ +40 s).
- With truth-quality geometry, routed obstacle contacts (T08 3,504 samples) and arm-link vs cabinet contacts (414) are prevented.
- The carry height becomes geometry-chosen. This is a safety gain; the speed gain is only 1-3 s per carry (verify F7).

**Risks.**
- On wrong model boxes it plans confidently around phantom obstacles. Mitigation: `on` only with a perception-backed map (IC12, decision UD2).
- Escape into neighbours. Mitigation: rule (ii).
- Over-long detours. Mitigation: candidate ordering by predicted time; reporting.

**Effort.** 4.0 d.

---

### SP6a. B2 in the legacy point router (early, perception-independent)

**Purpose.**
- README B2 and IC4 step 1: exact point-to-box distances plus an explicit, reported margin plus a checked escape.
- This fixes 5/7 spurious refusals now (verify F4).

**Files.**
- Create `src/gpt_field/swept/point_v2.py` (`TransitRouteV2`, the same interface as `TransitRoute`).
- Modify `agent_speedup_v2_user_check_fragments.py:304-308` to select it under `GPT_FIELD_ROUTER=point_v2`.
- Create `tests/test_point_v2.py`.

**Algorithm.**
- Start, goal and tracking validity use the exact signed distance from the grip point to each expanded box. There is no 8-corner rule.
- The reported `margin_m` equals the model's clearance.
- If the start is inside: build an escape leg along the box's outward normal at the closest point, or +z if that is better by the same criterion as SP6. The distance from the unexpanded box must strictly increase, and the other boxes must not get closer.
- The straight check and A* are unchanged.

**Veto removed / parameter added.** Removes the lattice margin and the start refusal. M4 caution: the lattice was a hidden 0-20 mm margin. The new margin is explicit and reported, and it is validated on contacts in SP12 R-A, not only on refusals.

**Depends on.** None.

**Offline tests.**
- T6a.1: replay the 7 refusals (verify table F4). The 5 with starts 6-20 mm outside become `direct`. T04 20mm @395 (−0.4 mm) and stove-off @179 (−52.4 mm) become escape + direct.
- T6a.2: on all 117 routed stages, the path is identical to `TransitRoute` wherever `TransitRoute` succeeded, to within 1 mm.

**Live validation.** Capped smoke on T04 + can with `point_v2` (≤ 10 calls). Expect 0 `route_start_blocked`.

**Expected effect.** About −6 calls / −70 s across the 9-run set (verify F4), available before any perception work.

**Risks.** A lower effective clearance under 5-12 cm box errors. Mitigation: the margin is reported; this is an A/B arm in the sweep.

**Effort.** 1.0 d.

---

### SP7. Body-aware route and tracking (supersedes `TransitRoute` and `PathTracker`'s field checks)

**Purpose.**
- Item 6: supersede `TransitRoute` and `tracking.py`.
- G5: lookahead chords validated for the body, not the grip site.

**Files.**
- Create `src/gpt_field/swept/route.py` (`BodyRoute`) and `chord.py` (`BodyChordValidator`).
- Modify `tracking.py`:
  - `PathTracker.__init__(…, validator=None)`;
  - lines 83, 92, 127 and 134 call `validator.point_ok` / `validator.segment_clear`. The default validator wraps `FineCompactField`, so `off` behaviour is byte-identical.
- Create `tests/test_swept_route.py`.

**Interfaces.**
```python
class ChordValidator(Protocol):
    def point_ok(self, position, *, state8=None, q=None) -> bool
    def segment_clear(self, start, end, *, state8=None, q=None, orientation_wxyz=None) -> bool
class BodyRoute:      # duck-type compatible with TransitRoute: command(position), progress(position), diagnostics(), path
    def __init__(self, plan: StagePlan, tracker, validator: BodyChordValidator): ...
    def command(self, position, *, state8, q) -> tuple[tuple[float, float, float], dict]
    def orientation_target(self, arc_m: float) -> list[float]   # from the leg's sequencing mode (B1)
```

**Algorithm.**
- The legs' polyline is followed by ws_executor_control's braking-aware tracker (B3, I6).
- Each carrot chord is validated with a ≤ 5-sample body sweep: gripper from the predicted state8, arm at the measured q. This costs ≤ 3 ms.
- If a chord fails, try earlier vertices and then the projection, as `tracking.py:116-141` does today, before replanning.
- `orientation_target` serves the sequencing mode's orientation schedule.

**Veto removed / parameter added.** Removes the grip-site `invalid_measured_start` and the hover at the 4 mm deadband for rotating arms (`tracking.py:90-95`, together with B3). Parameters: none.

**Depends on.** SP6, and ws_executor_control's I6 tracker.

**Offline tests.**
- T7.1: with `off`, `validation/replay_baseline.py` passes byte-exact.
- T7.2: a detour polyline is followed with chord checks that stay clear (synthetic box between start and goal; prototype tracker).
- T7.3: per-step validator ≤ 3 ms p95.

**Live validation.** Per-step chord-failure and replan counts in the shadow and `on` logs.

**Expected effect.** Safe execution of detours, which H1 has never exercised.

**Risks.** Tracker and planner disagree. Mitigation: a single owner per concern (planner = geometry, executor = command law) through I6.

**Effort.** 2.0 d.

---

### SP8. Contact-descent depth bound from a measured height map under the palm footprint

**Purpose.**
- C2 and F3: palm-on-target is 55 % of samples, in 5/7 runs.
- verify F3's correction: the conflict is with the surface *under the palm footprint* (±31.6 × ±100 mm), not with the grasped feature. A rim − 31 mm bound is 5-14 mm too shallow.

**Files.**
- Create `src/gpt_field/swept/depth_bound.py` and `tests/test_swept_depth_bound.py`.
- Create the fixture `validation/fixtures/swept/t04_grasps.json`: close-onset states of the 4 T04 runs plus the truth AABBs from the w2 `truth_boxes.json` method.

**Interfaces.**
```python
@dataclass class PalmFootprint: cells_R0R1_m (N,2); offset_m (N,)   # 2 mm grid; offset = p(xi) behind the grip along -approach
                                tip_cells_R0R1_m (T,2); tip_reach_m=0.00925
def contact_depth_bound(entry_state8, approach_world, fp: PalmFootprint, heights: HeightQuery, perms: PermissionMatrix,
                        *, requested_m: float, insertion: bool = False) -> dict
```
Output JSON:
```json
{"requested_m":0.061,"allowed_m":0.034,"limiting":{"part":"palm","cell_R0_R1_mm":[4,-38],"object_id":"obj_mug_w",
 "surface_along_m":0.040,"sigma_m":0.006},"source":"wrist_height_map|object_map_conservative|none","captured_at_step":118}
```

**Algorithm.**
- Each palm cell c sits at Pc = g0 + R0c·e0 + R1c·e1 − p(ξc)·a, where g0 is the entry grip and a is the approach axis.
- `HeightQuery` (I2) returns, for each cell's ray along a, the first-surface distance tc, its σc and the owner id.
- Palm limit: s_palm = min over cells whose owner is forbidden for the palm of (tc − σc).
- Fingertip limit: from the finger landing footprints (`robot_facts` `finger_landing_footprints_by_finger_q_mm` at the current q, reach 9.25 mm), s_tip = min over tip cells whose owner is forbidden for the fingers (support, neighbours) of (t − σ − 0.00925).
- Allowed descent = min(requested, s_palm, s_tip). Cells over the declared object are exempt when `insertion=True` (intent push or insert).
- Execution:
  - the approach target is truncated to the allowed depth;
  - arriving there is `pose_arrived_at_depth_bound`, not a stall;
  - the guarded descent (C1) stays active as a backstop;
  - feedback reports the remaining depth and the limiting cell, so the model can rotate the jaw axis or change grasp side instead of pressing.

**Sources.**
- Preferred: a wrist-view height map at the pre-contact hover. The wrist camera looks along the approach.
  - It needs ws_perception_service's per-view affine depth correction. Raw wrist DA3 depth is +18 % biased; after the correction the table error is 1.6 cm (h2_perception F1).
  - Better still is two-wrist-pose triangulation (h3_live: 0.4-1.7 cm).
- Fallback: object-map OBBs. This is conservative because it fills hollow interiors, and it is labelled `object_map_conservative`. The default policy is UD4.

**Veto removed / parameter added.** No veto; the descent is truncated and the reason reported. σ is a measured perception quantity (I2). p(ξ) and the footprints are robot geometry (SP1, robot_facts).

**Depends on.** SP1, SP5, I2 (ws_perception_service), and C1 (ws_executor_control).

**Offline tests.**
- T8.1: p(ξ) parity (SP1 T1.4).
- T8.2, which documents the AABB failure mode: on the T04 fixture with truth AABBs and σ = 0, the bound blocks the 3 recorded palm-contact closes. The hull was 14.0 / 22.6 mm below the AABB top at t100 steps 124 / 433, and 3.2 mm at 20mm step 133; verify F3 puts this bound 4-21 mm above contact onset. It **also** blocks 4 contact-free closes: default step 299 (8.8 mm), default 641 (2.2 mm), H3 129 (0.1 mm) and H3 480 (11.1 mm). An AABB fallback can only be labelled conservative.
- T8.3, the real criterion: with synthetic per-cell height maps of the two mugs (rim ring 5 mm thick, interior at the mug floor, handle as its own box), the 4 contact-free closes are allowed and the 3 contact closes stay blocked.
- T8.4: the `insertion` exemption.
- T8.5: ≤ 2 ms per query (about 1,600 palm cells).

**Live validation.**
- Sweep metrics: palm-class episodes per grasp (target ≪ baseline); `depth_bound_reached` count; grasp retry rate and gap-band failures (D1). The risk check is that retries do not rise more than 1 per 5 grasps.
- Capped smoke L8: T04 + can, ≤ 10 calls.

**Expected effect (estimate).** It targets the largest class, 9,739 samples in 10 episodes. In T04 the bound sits 4-21 mm above contact onset (verify F3). It costs 0 calls and a few hundred bytes of feedback.

**Risks.**
- Too-shallow grasps, if heights are taken from AABBs or feature tops. Shown above: 4 of 7 T04 closes were contact-free yet 0.1-11.1 mm below the AABB top, in line with the report's 5-14 mm shallower risk.
  - Mitigations: (1) heights per footprint cell from measured depth, not feature tops; (2) the limiting-cell report so the model can choose another jaw orientation; (3) D1's gap-band predicate catches weak grasps; (4) A/B in the sweep.
- A large σ (1-2 cm uncorrected) makes the bound shallow. Mitigation: triangulated heights, or the guarded descent with the bound only when σ is below the pad height. This is decision UD4.

**Effort.** 2.0 d.

---

### SP9. Speed coupling (clearance-scaled cruise and the distance-scheduled approach, D2)

**Purpose.**
- IC2 and G5: the 0.10 m cap raised p99 speed from 0.12 to 0.53 m/s, and the only robot-obstacle contact began at 0.17-0.19 m/s.
- D2 and IC1: the slow zone = measured uncertainty + braking distance. 3 of 15 contact onsets were 2.2-6.8 cm early.

**Files.**
- Create `src/gpt_field/swept/speed.py` and `tests/test_swept_speed.py`.

**Interfaces.**
```python
class SpeedPolicy:
    def __init__(self, osc: Callable[[], dict], guard_latency_steps: Callable[[], int], s_contact_m: float): ...
    def cruise_cap(self, clearance_unexpanded_m: float, bound_m: float, band_m: float) -> float  # command cap, m/step
    def approach_schedule(self, dist_to_target_m: float, target_bound_m: float) -> float
    def schedule_for(self, leg: Leg, sweep: SweepReport) -> Callable[[float], float]             # arc -> command cap
```

**Algorithm.**
- Let a and b come from `osc_response()` (online, episode p90 of a), n = the guard or perception latency in steps (I4; the measured guard fires a median 4 steps after onset, verify F5), and s_contact = the realised contact-profile speed (1.06-1.24 mm/step).
- For each sample: s_max(c) = max(s_contact, (c − bound − band) / (n + a/(1−a))), where c is the clearance to the *unexpanded* geometry, so the bound is subtracted exactly once.
- The command cap is s_max·(1−a)/b, clipped to the configured 0.10 m.
- **Approach (D2).** c = the distance to the declared target along the approach. The slow zone is L = bound_target + d_brake(s_cruise) + n·s_contact. Inside it the contact profile (5 mm command) runs with the guard on.
- **Worked numbers** (pooled a = 0.705, n = 4, bound 1 cm, band 4 mm): clearance 20 cm gives no restriction (cap-limited); 5 cm gives 5.6 mm/step (≈0.11 m/s); 3 cm gives 2.5 mm/step; 2 cm gives the contact speed.
- Unknown volumes (if I1 provides them) affect speed only and never block (IC11).

**Veto removed / parameter added.** Adds no task threshold. The inputs are measured: OSC response, guard latency, perception bound and tracking band.

**Depends on.**
- ws_executor_control: B3 tracker (I6), C1 guard and latency (I4), OSC fit (I3).
- ws_perception_service: bounds.
- Sequencing (IC1 / IC2): after B3, C1, C2 (SP8) and C3.

**Offline tests.**
- T9.1: the predicted stopping distance against recorded zero-command decays, median error ≤ 20 %.
- T9.2: replay the 15 forbidden contact-approach onsets (cross_critic `contact_onset.py`). With bound := the object's benchmark bound, every onset falls inside the slow zone (speed at onset ≤ 1.5·s_contact). This criterion is reported against bounds measured by ws_perception_service, so it is not tuned to these runs.
- T9.3: the T08 stage under the schedule has speed ≤ 3 mm/step at the closest approach.

**Live validation.**
- Sweep metrics: impact speed at contact onset (target ≤ s_contact·1.5); p99 transit speed vs clearance; time per task against the `on` arm without speed coupling, in the same window.

**Expected effect (estimate).**
- Safety: the impact speed at any unmodelled contact drops from up to 0.53 m/s to ≤ 0.06 m/s within 3 cm of objects.
- Time: D2's −10…−67 s per task (README D2) is available only after C1-C3. Near-object slowdowns cost an estimated +5-15 s per task at σ ≈ 1 cm. Measure both.

**Risks.** Slow near clutter when σ is large. Mitigation: better σ from perception; report the time attributable to speed caps.

**Effort.** 2.0 d.

---

### SP10. Held-object attachment (in-gripper offset, lift-off, place, release)

**Purpose.**
- F7 / K2: the carry window 0.54-0.57 m was accepted by the router.
- Bowl vs cabinet during lifts: 697 and 611 samples. Pot vs bottle: 111.
- "Attach at grasp with its in-gripper offset": a handle grasp offsets the mug body by about 6 cm.

**Files.**
- Create `src/gpt_field/swept/held.py` and `tests/test_swept_held.py`.

**Interfaces.**
```python
@dataclass class HeldAttachment:
    object_id: str | None; T_grip_object: np.ndarray (4,4); half_extents_m: np.ndarray (3,); sphere_centers (S,3); sphere_radii (S,);
    bound_m: float; below_grip_m: float; source: str  # "perception_at_close" | "map_at_close" | "fallback_jaw_column"
    attached_at_step: int; gap_mm_at_close: float; support_id: str | None
def attach(event: GraspEvent, now: MeasuredState, snapshot: dict) -> HeldAttachment
def detach(att: HeldAttachment, now: MeasuredState) -> dict          # world pose + grown bound, written back to the map (I1 write)
```

**Algorithm.**
- **Trigger.** ws_executor_control's grasp continuation predicate (I5). After a close, the gap is stable inside the object band (G7: confirmed grasps settled at 9.3-21.5 mm and lost ≤ 4.4 mm on lift; failures settled at 1.4 mm or collapsed by 12.3 mm).
- **Geometry.** The object's map entry is re-localised at close (IC14) and transformed into the grip frame, so a handle grasp automatically carries the body offset.
  - The vertical hang is cross-checked with a close-time measurement: below_grip ≥ grip_z − support_z at close, since the object rests on its support.
  - The bound is the object bound ⊕ a slip bound (the gap change since close).
- **Fallback, when there is no map entry.**
  - A jaw-column box: R1 extent = gap/2 + pad thickness; R0 = the pad R0 extent (16 mm); along the approach from the pad top down to the support at close.
  - Labelled `fallback_jaw_column`.
  - It does not know lateral offsets; perception supplies those (I1).
- **Spheres.** A closed-form box cover (cells of about 2 cm, r = half the cell diagonal). It is numpy-only, built at attach time, and its outward error is reported.
- **Lift.** The first lift leg uses held vs pick support = `permitted_nonincreasing`.
- **Place.** Held vs the declared support is permitted, with the bottom face only allowed into the support.
- **Release.** After the open completes, `detach` writes the attached pose with a grown bound to the map until perception re-localises.
- Class E (pot vs burner rim) needs the support's part geometry and xy placement precision from ws_geometry_tools; it is out of scope here and reported as a known gap.

**Veto removed / parameter added.** None removed. The cell size equals the measured object bound or less (about 2 cm), so the sphere error stays under the object's own uncertainty.

**Depends on.** SP3, SP5, I5 (ws_executor_control), and I1 attach/detach writes (ws_geometry_tools).

**Offline tests.**
- T10.1: T04 carry with the true can and the held mug at its close-time truth offset. Grip z 0.55-0.56 m is flagged, ≥ 0.58 m is clear, and the planner's z_req is in [0.57, 0.60] m.
- T10.2: drawer 001 and 003 lift legs, with enriched truth (SP12): the bowl vs cabinet contact is flagged at or before onset in ≥ 2 of the 3 recorded episodes.
- T10.3: on every T04/T08 grasp, the fallback envelope's below-grip extent is ≥ the truth object extent below the grip.

**Live validation.**
- Sweep metrics: `manipulated_object_scene` episodes during lift and carry (target ≪ baseline); carry height minus required height; `held_geometry: fallback` rate.

**Expected effect.**
- Carry safety no longer depends on the model's altitude choice; the T04 0.54-0.57 m window is closed.
- Lift-out contacts (697 / 611 / 111 samples) become plannable.
- Speed effect about 1-3 s per carry (verify F7 correction); this is a safety change.

**Risks.** A wrong offset if perception binds the wrong instance. Mitigation: the attach source is recorded; the bound grows when the gap changes; the per-step monitor keeps checking.

**Effort.** 2.0 d.

---

### SP11. Integration, feature flags and shadow mode

**Purpose.**
- Item 6: supersede `TransitRoute` and `tracking.py` in a way that can be A/B tested.
- Route all arm stages: transit, contact approach, lift, retreat, direct motion.

**Files (modify, in `<NEW>`).**

`agent_speedup_v2_user_check_fragments.py`:
- `_enter_segment` (`:259-321`): the transit-only block (`:293-317`) becomes `planner.plan_stage(…)` for every `MOTION_SKILLS` stage of either profile when `on`. The 1.2 rad check (`:288-291`) is dropped only when ws_executor_control's sequencing (B1) is active (a flag coupled to theirs).
- `_next` (`:333-346`): the delta comes from `BodyRoute.command`, capped by the leg speed schedule.
- `_observe` (`:348-420`): `planner.monitor_step` runs before the stall test.
- `progress()` (`:449-483`): `contact_or_collision_cause` comes from nearest pairs (`:482`).
- `fragment_capabilities` (`:179-213`): describes the body planner.
- The `route` schema field (`:56-59, 68, 144-145`) is accepted but ignored when `on`, with a note in the record.

Other files:
- `agent_speedup_v2_user_check_routing.py`: unchanged for `off` and `point_v1`. `point_v2` is imported from `swept/point_v2.py`.
- `tracking.py`: the validator seam (SP7).
- `agent_speedup_v2_user_check_motion.py` `LocalMotionPrimitive.__init__` (`:270-302`): pre-check, and truncate at the last clear sample.
- `agent_speedup_v2_user_check_agent.py` `_apply_agent` (`:313-361`): passes the object-map snapshot (I1), `view.observation.joint_position`, the held attachment and grasp/release events (I5) into the fragment. In `on` mode, model-written `obstacle_map` boxes (`:314`) are logged but not planned on, per "take metric estimation away from the model".

**Files (create).**
- `src/gpt_field/swept/flags.py` (read once and frozen).
- `tests/test_swept_integration.py`.

**Interfaces.**
- `GPT_FIELD_BODY_PLANNER` ∈ {`off`, `shadow`, `on`}, default `off`.
- `GPT_FIELD_ROUTER` ∈ {`point_v1`, `point_v2`}, default `point_v1`, used when `off` or `shadow`.
- Both values and `swept_body_model_sha256` go into the build digest (I10).
- **`off`**: byte-identical to H1 `3cc84ac`.
- **`shadow`**: H1 behaviour. The planner runs at every stage entry and step on the same inputs and writes only `trial/swept_shadow.jsonl` (schema §4.2), never model-visible.
- **`on`**: the planner executes.

**Algorithm.** The seam is a small `StagePipeline` (planner → tracker → monitor) inside the fragment primitive. ws_executor_control's sequencing and guard plug in through I3-I7; that is a shared seam.

**Veto removed / parameter added.** Collects the removals listed in SP6 and SP7.

**Depends on.** SP6 and SP7; I8 (protocol) and I10 (runner).

**Offline tests.**
- T11.1: `validation/replay_baseline.py` passes byte-exact with `off` **and** with `shadow`, because shadow output is never in payloads. H1's 34 tests pass.
- T11.2: in `on`, every MOTION_SKILL stage and every direct motion goes through `plan_stage` (spy test).
- T11.3: `swept_shadow.jsonl` schema validation.
- T11.4: a legacy `route` field in `on` mode is ignored and the record says so.

**Live validation.** Capped shadow smokes (L6-a), then `on` smokes (L6-b, L8, L11). The user's all-task sweep runs `on` and `off` arms in the same window (§6).

**Expected effect.** Safe A/B. Shadow mode measures live recall and precision with no behaviour change.

**Risks.** Merge conflicts in `fragments.py` with ws_executor_control. Mitigation: the shared `StagePipeline` seam, with one owner per hook.

**Effort.** 3.0 d.

---

### SP12. Offline validation tooling: trace enrichment and 9-run replay

**Purpose.**
- Item 6: replay all recorded stages of the 7 H1-family runs and the 2 H3 runs, against model boxes and against diagnostic truth.
- Report recall and precision against actual contact episodes, plus escape and detour rates.
- R8: one saved-observation replay suite.

**Files.**
- Create `validation/enrich_trace.py`. It runs in `.venv-sim` with MUJOCO_GL=egl (or osmesa) and makes **no model calls**.
- Create `validation/swept_replay.py` (`.venv`), `validation/fixtures/swept/*`, `validation/results/swept_replay_<date>.json` and `tests/test_swept_replay_smoke.py`.

**enrich_trace.py.**
- Reset each recorded run with its frozen inputs: task, `inputs/custom_task.bddl`, `inputs/initial_state.npy`, and the OSC translation scale from the run metadata (0.10, or 0.02 for the two older runs).
- Replay the recorded `action` arrays from `trial/trace.jsonl`.
- Assert state8 equality at every step; stop at the first divergence.
- Record per step: arm joints, the pose of every free object and fixture joint, and active contact pairs with positions.
- Output `validation/enriched/<run>.npz`.
- Today the trace has no joints (the `observation` keys are image, wrist_image, state and prompt) and no per-step object poses. The T08 bottle evidently moved during the press, so static truth cannot score steps 53-83.

**swept_replay.py.**
- **Stage extraction.** Fragment segments come from `feedback.primitive_feedback.fragment.segments[*]`: request, entry pose, target, and `routing.expanded_boxes` with `clearance_m`, so the model map is the expanded boxes minus the clearance. Direct motions come from `local_motion`; `set_gripper` stages are included.
- **Maps.**
  - (i) model boxes, with bound = clearance;
  - (ii) truth: enriched per-step object geometry, or for static obstacles `inputs/layout_manifest.json` collision bounds plus `physical_truth.json` / the `truth_boxes.json` method.
- **Joints.** Enriched (exact), or SP2's lift between request-boundary joints (measured accuracy in SP2).
- **R-A detector replay.** The checker on *executed* configurations gives per-pair negative-clearance runs.
- **R-B planner replay.** At each stage entry, `plan_stage` gives the outcome class: direct, escape, detour, handoff, merge, depth-bounded, truncated or `no_route`, plus planned-vs-truth clearance.
- **R-C timing.**

**Metrics.**
- Episode-level recall on the 27 contiguous forbidden episodes (verify §0), by class A-E and routed vs unrouted.
- Precision of flagged runs that match a recorded episode within ±2 steps.
- Onset lead in steps.
- Outcome rates per stage type.
- Timing p50 / p95 / max.

**Pass criteria (development aids, not launch gates; rule 4).**
- R-A with truth and enriched joints: recall ≥ 0.9 at onset ±1 step for classes A (palm), B (finger and link) and C (fixture); ≥ 0.8 for D with held attachment; precision ≥ 0.8.
- R-B with model boxes: `no_route` ≤ 2 % of stages.
- No regressions: no stage whose planned route collides with truth where the recorded route did not.
- Timing within the SP3 and SP6 budgets.

**Depends on.** SP1-SP6 and SP10; simulator access for enrichment (UD7).

**Offline tests.** T12.1: smoke on the T08 fixture (no simulator) runs in < 30 s and reproduces T3.2 and T6.1.

**Live validation.** The same replay runs on every new sweep run, since joints and object truth are logged (I10). The shadow logs give live recall and precision.

**Expected effect.** A measured coverage map of the planner before any `on` sweep.

**Risks.** Enrichment divergence on custom layouts. Mitigation: stop at the first mismatch; fall back to the lift.

**Effort.** 4.0 d.

---

## 4. Data contracts crossing process or model boundaries

### 4.1 `body_plan` (planner → fragment record → model feedback, rendered by ws_protocol_context)
```json
{"$id":"swept-planner/body_plan/v1","type":"object",
 "required":["version","status","legs","min_clearance","margins","map_revision","timing_ms"],
 "properties":{
  "version":{"const":"swept-planner-1"},
  "status":{"enum":["direct","escape_then_direct","detour","escape_then_detour","standoff_handoff","merged_into_contact_stage",
                    "depth_bounded","truncated_at_clearance","jaw_sweep_blocked","escape_partial","no_route","no_route_budget"]},
  "legs":{"type":"array","items":{"type":"object","required":["kind","start_xyz_m","end_xyz_m","sequencing","predicted_steps","min_clearance_m"],
    "properties":{"kind":{"enum":["escape","transit","lift","sidestep","approach","contact","retreat","hold"]},
      "start_xyz_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},
      "end_xyz_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},
      "end_quaternion_wxyz":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},
      "sequencing":{"enum":["executor_default","coupled","rotate_first","translate_first","none"]},
      "predicted_steps":{"type":"integer","minimum":0},"min_clearance_m":{"type":"number"},
      "speed_cap_m_per_step_min":{"type":"number"},"reason":{"type":["object","null"]}}}},
  "min_clearance":{"type":"object","additionalProperties":{"type":"object","required":["m","object_id"],
     "properties":{"m":{"type":"number"},"object_id":{"type":"string"},"part":{"type":"string"},"leg":{"type":"integer"}}}},
  "margins":{"type":"object","properties":{"object_bound_m":{"type":"object","additionalProperties":{"type":"number"}},
     "robot_model_conservatism_m":{"type":"object"},"tracking_band_m":{"type":"number"},"arm_prediction_band_m":{"type":"number"}}},
  "handoff":{"type":["object","null"],"properties":{"object_id":{"type":"string"},"part":{"type":"string"},
     "standoff_xyz_m":{"type":"array"},"remaining_offset_m":{"type":"array"},"requires":{"const":"contact_stage_declaring_object"}}},
  "no_route":{"type":["object","null"],"properties":{"blocking":{"type":"array"},"tried":{"type":"array","items":{"type":"string"}},
     "nearest_feasible":{"type":"object"}}},
  "depth_bound":{"type":["object","null"]},"held":{"type":["object","null"]},
  "map_revision":{"type":"integer"},"map_observed_at_step":{"type":"integer"},
  "timing_ms":{"type":"object","properties":{"total":{"type":"number"},"lift":{"type":"number"},"sweep":{"type":"number"}}}}}
```
A compact model-facing form of at most 400 characters (status, min clearance per class with object id, handoff and no_route evidence) is what ws_protocol_context renders. Reasons are kept verbatim (IC15).

### 4.2 `trial/swept_shadow.jsonl` row (planner → logs; never model-visible)
```json
{"schema":"swept-shadow-v1","step":118,"event":"stage_entry|step_check","stage_key":"fragment:7:2",
 "legacy":{"router":"point_v1","route_status":"direct_segment|unavailable","reason":null},
 "body":{"...body_plan or step verdict..."},"measured":{"state8":[8],"q":[7]},"timing_ms":{"total":12.4}}
```

### 4.3 Consumed: ObjectMapSnapshot (I1), HeightQuery (I2) and ExecutedProfileModel (I3)
The shapes are in `shared_interfaces_needed` in the structured result, and they are repeated in §7.

---

## 5. Rollout, flags and IC compliance

| Phase | What turns on | Needs (other workstreams) | IC rule respected |
|---|---|---|---|
| P0 | SP1-SP3, SP12 R-A on truth (offline only) | none (enrichment needs UD7) | — |
| P1 | SP6a `point_v2` (exact distances + escape); SP11 `shadow` in the new build | runner I10 (joints sidecar, digest) | IC4 step 1 |
| P2 | SP4-SP7 `on` for transit + direct motion with a perception-backed map | executor I3/I6 (B1 sequencing, B3 tracker); geometry/perception I1 (IDs, bounds; held/target/robot excluded) | IC3, IC12, IC13 (exclusions + escape first, then automatic boxes, then the body sweep) |
| P3 | SP10 held object, SP8 depth bound, contact stages routed | executor I4/I5 (C1 guard, D1 grasp predicate); perception I2 | IC1 step 1, IC5 |
| P4 | SP9 cruise scaling, then the D2 approach schedule | executor B3 + C1; perception bounds (C3) | IC1 step 3, IC2 |
| Accept | The user's all-task sweep on one frozen build, `on` vs `off` arms interleaved in the same window | runner (coverage map) | rules 3 and 4 |

---

## 6. Live validation (capped smokes are fine; the user starts full tasks and sweeps)

The launcher flags come from ws_runner_measurement: the evaluation cap A1 provides `--max-model-calls` and `--max-actions`. The commands are indicative:
```bash
# offline (no model, no sim)
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_swept_*.py tests/test_point_v2.py
.venv-sim/bin/python -I tools/build_body_model.py --check          # rebuild + hash equality
.venv/bin/python validation/swept_replay.py --fixtures-only        # T12.1, < 30 s
# offline enrichment (sim, no model; user approval UD7)
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<free> .venv-sim/bin/python validation/enrich_trace.py --run /workspace/gpt_field/runs/<run> --out validation/enriched/
.venv/bin/python validation/swept_replay.py --runs <9 run dirs> --enriched validation/enriched --out validation/results/swept_replay.json
# capped smokes (model calls; user-approved)
GPT_FIELD_BODY_PLANNER=shadow ./run.sh --task t08_bottle_obstacle --gpu <g> --max-model-calls 8 --max-actions 400 --output runs/swept_L6a_t08_shadow_001
GPT_FIELD_ROUTER=point_v2      ./run.sh --task t04_can_obstacle   --gpu <g> --max-model-calls 10 --max-actions 500 --output runs/swept_L6a2_t04_pointv2_001
GPT_FIELD_BODY_PLANNER=on      ./run.sh --task t04_can_obstacle   --gpu <g> --max-model-calls 10 --max-actions 500 --output runs/swept_L6b_t04_on_001
GPT_FIELD_BODY_PLANNER=on      ./run.sh --task drawer_bowl        --gpu <g> --max-model-calls 12 --max-actions 900 --output runs/swept_L11_drawer_on_001
```

**Smoke pass criteria.**
- L6-a (shadow): no behaviour change; no exceptions; per-step p95 ≤ 3 ms; stage entry p95 ≤ 150 ms. If the T08 rotation chain recurs, the shadow log flags finger1 before onset.
- L6-a2: 0 `route_start_blocked`.
- L6-b (`on`): 0 `route_start_blocked`; the carry z is chosen by lift-over and is ≥ the can top + held hang + bound; 0 robot-can and 0 mug-can contacts.
- L8 (depth bound): 0 palm-on-target episodes at the grasps, or each one explained by a σ larger than the bound.
- L11: 0 arm-link vs cabinet contacts; joint-limit reports, if any, go to C5.

**Sweep metrics (by task, as a coverage map).**
- Forbidden contact **episodes** by class, phase and stage kind (routed or unrouted); routed-stage episodes, target 0.
- Impact speed at onset.
- Escapes, detours, handoffs, merges, `no_route`, truncations and depth-bound hits per stage type.
- Calls per task (≤ baseline).
- Planner time p50 / p95 / max.
- Grasp retry rate.
- Carry height minus required height.
- Strict success at the agent's finish (from the runner).

---

## 7. Shared interfaces needed (proposed shapes)

| ID | Name | Provider → Consumer | Shape |
|---|---|---|---|
| I1 | ObjectMapSnapshot (+ attach/detach writes) | ws_geometry_tools + ws_perception_service → ws_swept_planner | `{"schema":"harness-object-map-v1","revision":int,"observed_at":{"step":int,"wall_s":float},"objects":[{"id":str,"label":str,"roles":["target","support","obstacle","fixture","held"],"shape":{"kind":"obb","center_m":[3],"rotation_wxyz":[4],"half_extents_m":[3]} or {"kind":"multi_obb","boxes":[…]},"parts":[{"part_id":str,"shape":{…}}],"bound_m":float (measured uncertainty bound, benchmark-calibrated),"source":"perception_two_view|perception_single_view|located_from_model_pixels|close_time|attached","observed_at_step":int,"movable":bool}],"supports":[{"id":str,"plane":[nx,ny,nz,d],"bound_m":float,"extent_xy_m":[[x0,y0],[x1,y1]]}],"unknown_volumes":[obb…]}`. Writes: `attach(id, T_grip_object, bound, step)`, `detach(id, world_pose, bound, step)`. Robot self-surfaces, the held object and phantom depth must never appear as obstacles (IC13). |
| I2 | HeightQuery | ws_perception_service → ws_swept_planner | Request: `{"origins_m":[[x,y,z]…],"direction":[3],"max_range_m":0.25,"fresh_after_step":int}`. Response: `{"hits":[{"t_m":float|null,"sigma_m":float,"object_id":str|null,"kind":"object|support|unknown"}],"captured_at_step":int,"source":"wrist_affine_depth|triangulated|object_map"}` |
| I3 | ExecutedProfileModel + OSC response | ws_executor_control → ws_swept_planner | the Python Protocol in SP4: `sequencing_options`, `predict` → `PredictedProfile(states8 (K,8), step, arc_m, tracking_band_m, arrival_step, sequencing)`, `osc_response()` → `{"a","b","a_p90","rotation_rad_per_step","updated_at_step"}` |
| I4 | Guard latency and guard events | ws_executor_control ↔ ws_swept_planner | `guard_latency_steps() -> int`; `GuardEvent{"step","blocked_direction":[3],"depth_reached_m","kind":"contact_blocked|joint_limit|…"}`. The planner returns `nearest_pairs(state8,q,scene) -> [{"part","object_id","clearance_m"}]` for the guard's `blocked_motion` feedback. |
| I5 | GraspEvent / ReleaseEvent | ws_executor_control → ws_swept_planner | `{"event":"grasp_established|grasp_lost|released","step":int,"finger_gap_mm":float,"gap_band_mm":[lo,hi],"object_id":str|null}` |
| I6 | Speed schedule and chord validator | ws_swept_planner → ws_executor_control (tracker) | `schedule(arc_m)->cmd_cap_m`, `ChordValidator.point_ok/segment_clear`, `BodyRoute.orientation_target(arc_m)` |
| I7 | Nearest-part attribution | ws_swept_planner → ws_executor_control, ws_protocol_context | `contact_or_collision_cause: {"nearest_pairs":[…],"source":"geometry_model"}` |
| I8 | Stage contact declaration | ws_protocol_context → ws_swept_planner | per arm/gripper stage: `"contact":{"object_id":str|null,"parts":["finger","pad","palm","held"],"intent":"grasp|place|push|insert|release|none"},"stop_on":["contact"],"insertion":bool`; object ids come from the map, and the model never writes metric boxes in `on` mode |
| I9 | body_plan rendering | ws_swept_planner → ws_protocol_context | schema §4.1; a compact form ≤ 400 characters, reasons verbatim |
| I10 | Joints sidecar, logs, digest, asset hashes, episode metrics | ws_runner_measurement → ws_swept_planner | `trial/robot_joints.jsonl` `{"step","q":[7],"qd":[7]}` every step, outside the frozen observation hash; `trial/object_truth.jsonl` (evaluator-only object poses); `trial/swept_shadow.jsonl`; digest fields `swept_body_model_sha256`, `GPT_FIELD_BODY_PLANNER`, `GPT_FIELD_ROUTER`; sim-reported robot asset sha256; contact episodes split by stage kind and routed flag, plus onset speed |
| I11 | Joint-limit and reachability report | ws_swept_planner → owner of C5 (ws_executor_control) | `{"stage_key","limit_hits":[{"k","joint","margin_rad"}],"residual_m_max"}` |

---

## 8. Risks (cross-cutting)

1. **Geometry quality gates the value.**
   - With model boxes (5-12 cm off) the planner escapes or detours around phantoms; verify's hand sweep flagged 21/111 stages.
   - Mitigation: `on` only with a perception-backed map (IC12); `shadow` otherwise; SP6a gives the safe part (exact distances + escape) now.
2. **Executed-profile error.** Rotation lag and tracker overshoot of 10-28 mm at the 0.10 m cap. Mitigation: bands, per-step replan, and the B3 tracker before SP9.
3. **Arm prediction near joint limits** (drawer 001: 14 cm). Mitigation: limit hits inflate the band and trigger C5; measured joints are used per step.
4. **Too-shallow grasps from the depth bound.** Shown concretely with H3's second grasp. Mitigation: per-cell measured heights; the limiting-cell report; UD4.
5. **Compute spikes** on a loaded host (the prototype ran at load average 8-36). Mitigation: a budget with `no_route_budget`, staged screening, and p95 logging.
6. **Merge conflicts with ws_executor_control** in `fragments.py`. Mitigation: the `StagePipeline` seam.
7. **False security.** The sphere model is a geometric model of an estimated scene, not a certificate. Every report states the bounds and the conservatism; the contact monitor remains the judge.

---

## 9. Effort

| Component | Days |
|---|---|
| SP1 build | 2.0 |
| SP2 kinematics | 1.5 |
| SP3 checker | 3.0 |
| SP4 profile | 2.0 |
| SP5 scene/permissions | 1.5 |
| SP6 planner | 4.0 |
| SP6a point_v2 | 1.0 |
| SP7 route/tracking | 2.0 |
| SP8 depth bound | 2.0 |
| SP9 speed | 2.0 |
| SP10 held | 2.0 |
| SP11 integration | 3.0 |
| SP12 replay | 4.0 |
| **Total** | **30.0 engineer-days** (≈ 6 weeks for one engineer; P0 + P1 ≈ 8.5 d) |

---

## 10. Open questions for the user

- **UD1, model resolution β.** 4 mm gripper / 10 mm arm (≈450-600 spheres; recommended) or 5 / 12 mm (372 spheres, as measured).
- **UD2, planner `on` without a perception map.** Recommended: no, keep it in shadow (IC12).
- **UD3, autonomous escape legs** (≤ bound + band + penetration, a few cm) without a model call. Recommended: yes, reported.
- **UD4, depth-bound source policy.** Recommended: apply the bound only with a measured height map whose σ is below the pad height. With object-map boxes only, use guarded descent and report.
- **UD5, speed coupling.** Recommended: log the would-be caps in shadow from P1; enable only after B3 + C1.
- **UD6, direct motions.** Recommended: truncate at the last clear sample, rather than refuse (veto) or not check (H1).
- **UD7, simulator time for enrichment.** About 20-30 min of EGL-GPU (or slower CPU) time for the 9 runs, with no model calls. Recommended: approve, on a free GPU.
- **UD8, model-written obstacle boxes in `on` mode.** Recommended: ignore for planning, log only; missed obstacles come back through pointing → locate (C3).

---

## Appendix A: prototype artifacts (scratchpad; copy into `<NEW>/tools/proto/` if useful)

Root: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/ws_swept_planner_work/`

| Script | Output | What it shows |
|---|---|---|
| `scripts/build_body_model.py` (`.venv-sim`) | `data/panda_body_a12_g5.{npz,json}`, `data/panda_body_a8_g3.{npz,json}`, `data/panda_body_proto.*` (k-means, rejected) | 372 / 737 spheres; coverage; bulge |
| `scripts/body_runtime.py` | — | numpy-only FK, gripper-from-state8, sphere placement, sphere–OBB signed distance |
| `scripts/test_runtime.py` | `data/test_runtime_a12_g5.json` | FK parity; T08 measured and candidate profiles; timing |
| `scripts/test_iklift.py` | `data/test_iklift.json` | lift accuracy at request boundaries (T08, T04, drawer 001) |
| `scripts/test_depth_bound.py` | `data/test_depth_bound.json` | hand-hull vs mug AABB at the 13 T04 close onsets (4 runs) |
| `scripts/test_palm_profile.py` | `data/test_palm_profile.json` | p(ξ) parity |
| `scripts/osc_brake.py` | `data/osc_brake.json` | OSC first-order fit; braking |

Inputs: the compact step pickles in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/*_steps.pkl` (derived from `runs/*/trial/trace.jsonl`), `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/truth_boxes.json`, and `runs/*/trial/requests/*.request.json`.

## Appendix B: code facts checked (H1 `3cc84ac` unless noted)

**H1 routing and tracking:**
- `_routing.py`: `:3-4` scope; `:20-22` grid 0.02 m and detour margin 0.10; `:74-75` map limit; `:77-90` expansion; `:105-108` start/goal refusals; `:111-143` straight/A*; `:147-150` goal connector; `:154-156` lookahead ≥ max(0.03, cap).
- `tracking.py`: `:83-84` `invalid_measured_start`; `:90-95` zero translation inside the tolerance; `:114` lookahead arc; `:116-141` chord fallbacks.
- `fine_distance_planner.py`: `:101-106` 8-corner rule; `:117-130` query; `:135-155` segment_valid.

**H1 fragments:**
- `_fragments.py`: `:24-26` caps; `:27-34` profiles (transit 0.50 m / 1.2 rad, cap `USER_CHECK_TRANSIT_ACTION_M`, tolerance 0.004; contact 0.08 / 0.5, 0.005); `:56-59` route schema; `:90-110` `validate_route`; `:144-145` route only on transit arm moves; `:259-321` `_enter_segment` (envelope `:288-291`, routing `:293-317`); `:333-346` `_next`; `:404-419` stall; `:482` cause "unknown".

**H1 agent, motion and prompt:**
- `_agent.py`: `:110` map; `:186-212` execution state; `:314` map update; `:351-361` fragment creation; `:362-380` direct motion.
- `_motion.py`: `:267-311` `LocalMotionPrimitive`.
- `_agent_client.py`: `:92-105` obstacle-map text (`:101-102` remove the target); `:135-156` planner text (`:150-151` "does not … check the full robot").

**H1 other:**
- `action.py`: `:20-23` transit cap environment variable (default 0.02) and OSC scale; `:84-86` control-error sign.
- `agent_speedup_v2_user_check_sim.py`: `:72-80` joints delivered; `:91-96` popped from serialisation; `:122-128` `CalibratedObservation.joint_position`.
- `contact_skills.py`: `:259-296` StepView (state8 only; joints via `view.observation`).
- `robot_facts_v2.json` (`1f5da32f…`): palm 31.04 mm, footprint R0 ±31.6 / R1 −104…+100 mm, finger landing footprints, fingertip reach 9.25 mm.

**H2 and robodojo:**
- H2 snapshot `robot_geometry.py` (`8fa39c48…`): `:16` scipy; `:25-36` STL; `:127-148` GJK; `:151-292` model and FK; `:194` visual-only rule; `:230` kinds.
- H2 snapshot `motion.py` (`b15d07a1…`): `:16` cKDTree; `:224-333` configuration check; `:271-272` base skip; `:414-426` endpoint veto; `:436-460` RRT; `:518-521` name exemptions; `:532-551` payload.
- `src/gpt_field/robodojo_direct_sim.py` (`6d2746cb…`): numpy-only; `:134-194` `from_xml`, `:196-206` `bind`, `:208-224` `forward`, `:226-262` `solve`.

**Asset sha256** (robosuite 1.4.1 in H1 `.venv-sim`):

| File | sha256 |
|---|---|
| robot.xml | `263c0f9abacbc8686027ec5545bee144481cbcf5cc74aa421cbe3227030cd95e` |
| panda_gripper.xml | `d9d68123909d6910488310d0fae4e358ed4ac831c6d0957f3e30893a5412baac` |
| link0 | `dfc6d94330de8ddb005b311bfdba9f3b8e1aa7c256b71592ee7ff32cb9a9a5aa` |
| link1 | `d24e332dffccf260b91d05dde17c5998bb9559d37da8608a8ee5213d9661f603` |
| link2 | `370f7605a0fae3529db169ded50f52f171024aa792d4d773bc84197301f6a039` |
| link3 | `0a8d638b9349c6c0eefc4e888636ac4838c4b27170f18a51699321118af709c1` |
| link4 | `0180ebb5772ec9840cb049750cffb29a9ddc90311752a16ea34757782ef9e48d` |
| link5 | `dd17e688c7870e722283525879643d53a74c0024d328b0e14b034b54c8b6c31a` |
| link6 | `20b768e99a0e0440b5754dcca108016434e57937cc356acd9c352ccd3cb27f77` |
| link7 | `92ac6afcf7574c034d3170d8a68e95ac9048ab9d0dd5bbd8311b86e551b9ab1c` |
| hand | `94493e94f30fe940f2c8ca2f155c3bbe67bbff406d3edf5e261670d2f0f6e2ed` |
| finger | `2d07a740392f3b9b0816f65d64fff9927d3d57c897870fc4b6ff9c56fff3a0c8` |
