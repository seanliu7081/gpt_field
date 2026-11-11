# ws_geometry_tools: measured geometry the model can point at, name and reference

Workstream key: `ws_geometry_tools`. Date: 2026-11-08. Read-only design: nothing under `/workspace/gpt_field` was edited, and no simulator, model call or GPU job was run. The offline checks behind the new numbers in this document are in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/ws_geometry_tools_work/scripts/` (Appendix A).

**Goal.** The model labels, selects and states intent. The harness measures, and turns that intent into metres at the moment of use. After this workstream, the model never writes a scene coordinate that is not anchored to a measurement (README R1, C3; H1 items 9, 10, 12, 20; H3 V2–V4; cross_critic C4, C5, C11, G4).

**Scope.** In scope:
- the measured-geometry library;
- the object-centric scene map that replaces `obstacle_map`;
- target references and `locate`;
- region semantics;
- overlays;
- the model-facing geometry contract (prompt, schema, robot envelope);
- the plan/first-call object binding.

Out of scope, consumed through interfaces (§4):
- perception inference (ws_perception_service);
- swept collision checking and routing (ws_swept_planner);
- guarded motion, stall and termination predicates (ws_executor_control);
- prompt assembly, history and caching (ws_protocol_context);
- scoring and sweeps (ws_runner_measurement).

---

## 0. Ground rules and where the code lives

- **New build root `<NB>`.** ws_runner_measurement names the fork of H1 `3cc84ac`. All paths below are relative to `<NB>/src/gpt_field/` unless they start with `<NB>/`.
  - H1 stays byte-identical, for A/B runs.
  - Nothing imports `foundation_harness` or `foundation_vision` live trees.
  - This workstream needs no H2 code. The robot envelope comes from `robot_facts_v2.json`, which is already in H1.
- **Interpreter.** Every module here uses only numpy, Pillow and opencv-headless:
  - H1 `.venv`: numpy 2.5.3, cv2 4.13, Pillow 12.3, no scipy.
  - `.venv-foundation-vision`: numpy 1.26.4, cv2 4.11, Pillow 13.0. Checked.
  - So the controller runs it in-process, and the perception service can import the same file. No scipy, no numpy-2-only API.
- **User rules applied.**
  - No per-task code, words or thresholds.
  - Every numeric parameter is one of three kinds:
    - a robot-geometry fact from the MJCF-derived `robot_facts_v2.json`;
    - a measured uncertainty fitted on the truth-scored benchmark (GT11);
    - a statistical convention (the confidence multiplier `k`, decision D2).
  - Model outputs that cannot be resolved produce feedback, never a hidden veto.
  - Raw world targets keep working, with telemetry (D1).
- **What the model still writes in metres:**
  - robot-anchored displacements (`move_relative`, `approach` with `xyz_m`; "lift 5 cm");
  - offsets from a measured anchor ("3 cm above o3's top");
  - absolute orientations;
  - during the transition only, legacy absolute `xyz_m`, flagged in telemetry.
- **What it stops writing:** obstacle boxes, clearances, and target positions inferred from pixels.

## 1. Evidence this design rests on

| Fact | Value | Source |
|---|---|---|
| Model-written first boxes | 5–12 cm off; T08 bottle IoU 0, bottom 10 cm below the table; repeat within 1–2 cm across runs | README R1; h1_collision F2; cross_critic C4, G4 |
| SAM masks + calibration, no DA3 | two-view 1.2 / 1.7 / 0.4 cm; one view + true table 1.5 / 0.6 cm | h3_live F1 |
| Plan cells resolve SAM ambiguity | 11/11 (T04) and 3/3 (T08) ambiguous frames; 31/31 single detections consistent | h3_live F7 |
| Palm-on-target | 55 % of forbidden samples, 10 episodes in 5/7 runs; the envelope is never sent (`_agent_client.py:228-233`) | h1_collision F3; verify F3 |
| Palm bound vs contact onset (T04) | bound 4–21 mm above onset; but rim − 31 mm is 5–14 mm shallower than H3's successful grasps | verify_h1_collision F3 |
| Map | 4 unlabelled boxes; cap hit in 10/36 updates; no IDs, σ or source (`_fragments.py:54-59, 90-110`; `_routing.py:25-48`) | h1_collision F9 |
| Router cannot cover palm-on-target without IDs | the prompt removes the target from the avoid set (`_agent_client.py:101-102`) | verify M3 |
| Image cost | ≈ 893 tokens per 512² image; vision summary 2.3–2.5k tokens per call, re-sent forever (24–26 % of input) | cross_critic C12; h3_live F4 |
| **New: external-camera geometry at the obstacles** | 1 px = 1.3–1.6 mm; an 8×8 plan-cell's quantisation σ (64/√12 px) = 2.4–2.9 cm; one-view ray ∩ plane amplifies a plane-height error by 0.93–1.49× horizontally; wrist 1 px = 0.9–1.2 mm; external–wrist ray angle 45–81° (well conditioned) | A.1 |
| **New: single-view lifting prototype** (external SAM mask + true support z) | centre 1.2 cm (can, 11 frames) and 0.7–0.8 cm (bottle, 10 frames); with a +1 cm plane bias, 1.6 cm. Top: +1.9 to +2.1 cm with the naive rule, **−0.0 / +0.1 cm with the far-edge rule** (§GT1). A naive σ model covered truth in 9/10 frames (bottle) but 0/11 (can, 1.2 cm lateral error against σ = 0.2 cm), so the lateral σ term must be fitted | A.2 |
| **New: DA3-free plane-sweep support height** (no masks) | 3/7 frames within 13 mm (−0.2, +7.5, −12.5 mm). 4/7 failed: 3 had no peak (the external view of the wrist footprint is occluded by the arm at the start pose), 1 locked onto another surface (+130 mm). Not usable without a robot mask and support mask | A.3 |
| Subscription transport supports function tools | `astranomy_agentic_client.py:210-212, 229-246, 308` (main tree) ran `tools` + `tool_choice:"required"` + `function_call_output` live (63 calls). `tools` is not in `UNSUPPORTED_FIELDS` (`chatgpt_plan.py:37-41`); `max_tool_calls` is. H1's extractor needs exactly one output_text (`astra.py:540-541`) | code |
| Request text sizes (H1 T04, call 4) | geometry 1,893 chars; obstacle-map instruction paragraph 1,126 chars; camera/gripper paragraph 1,030 chars; schema 3,232 chars | A.4 |

## 2. Architecture

```
 sim worker ──CalibratedObservation(step, RGB×2, state8, joints, calibration)──► controller
                                                                        │
          ┌──────────────── FrameStore (GT1) ◄──────────────────────────┤ every decision frame
          │                                                             │
 ws_perception_service ──PerceptionSnapshot(frame N)──► SceneMap (GT3) ◄─┼── executor events (grasp_closed,
          (async, fail-open)                              ▲   ▲   ▲      │    released, guarded_stop)
                                     SupportPlane (GT2) ──┘   │   │      │
                         locate (GT5) ◄── model response ─────┘   │      │
                         scene_edits  ◄── model response ─────────┘      │
                                                                        │
 model request ◄── compact scene facts + overlay images (GT7) + GT9 contract
                                                                        │
 fragment stage entry ── TargetResolver (GT4, GT6) ──► ResolvedTarget ──► ws_executor_control
                                     SceneMap.obstacle_set() ──────────► ws_swept_planner
```

The model's decision loop and the fragment state machine are H1's. What changes is which numbers enter and where they come from.

---

## 3. Components

### GT1: metric core and frame store

**Fixes.** R1 and C3 (the harness measures), G3 (real calibration), and the H3 V4 gates (uncertainty, not vetoes). It is the shared maths for GT2–GT7 and for ws_perception_service's lifting.

**Files to create:**
- `geometry_tools/__init__.py` (`GEOMETRY_TOOLS_VERSION = "geometry-tools-1"`)
- `geometry_tools/camera.py`
- `geometry_tools/metric.py`
- `<NB>/tests/geometry_tools/test_camera_metric.py`

**Files to modify:** none. It reads the validated metadata from `astranomy_agentic_geometry.validate_geometry_metadata` (`:79-128`).

**Interfaces**
```python
@dataclass(frozen=True)
class CameraModel:                      # OpenGL axes: +x image-right, +y image-up, -z forward
    name: str; width: int; height: int
    fx: float; fy: float; cx: float; cy: float
    R_world_from_cam: np.ndarray        # 3x3
    t_world: np.ndarray                 # camera centre (m)
    distortion: tuple[float, ...] = ()  # OpenCV k1,k2,p1,p2[,k3]; () = ideal pinhole
    mount: str = "fixed_world"          # | "robot_mounted"
    @classmethod
    def from_geometry(cls, geometry: Mapping, label: str) -> "CameraModel"
    def rays(self, uv) -> tuple[np.ndarray, np.ndarray]          # (N,3) origins, (N,3) unit dirs (undistorts)
    def project(self, xyz) -> tuple[np.ndarray, np.ndarray]      # (N,2) uv, (N,) depth (>0 in front)
    def project_jacobian(self, xyz) -> np.ndarray                # (N,2,3)

@dataclass(frozen=True)
class Estimate3D:
    xyz: np.ndarray; cov: np.ndarray    # 3, 3x3
    method: str                         # triangulated | ray_support | ray_box | lifted | contact | fused
    frames: tuple[int, ...] = (); residual_px: tuple[float, ...] = (); ray_gap_m: float | None = None
    chi2_p: float | None = None         # consistency of a multi-view solution
    sigma = property(lambda s: np.sqrt(np.diag(s.cov)))

@dataclass(frozen=True)
class Box3D:
    lower: np.ndarray; upper: np.ndarray        # world AABB (m)
    centre_cov: np.ndarray; extent_sigma: np.ndarray
    footprint_xy: np.ndarray | None             # convex polygon (K,2), optional
    method: str; frames: tuple[int, ...]

def triangulate(obs: Sequence[tuple[CameraModel, Sequence[float], float]]) -> Estimate3D  # (cam, (u,v), sigma_px)
def ray_gap(cam_a, uv_a, cam_b, uv_b) -> tuple[float, np.ndarray]                       # closest approach, midpoint
def ray_plane(cam, uv, sigma_px, plane: "SupportPlane") -> Estimate3D | None
def ray_box_entry(cam, uv, sigma_px, box: Box3D) -> Estimate3D | None
def lift_silhouette(cam, polygon_uv, plane, *, sigma: "LiftSigmaModel") -> Box3D        # one view on a support
def lift_two_view(cam_a, poly_a, cam_b, poly_b, plane | None, *, sigma) -> Box3D | None
def fuse(estimates: Sequence[Estimate3D]) -> Estimate3D                                  # inverse covariance

@dataclass(frozen=True)
class FrameRecord:
    frame: int                                   # observation step printed on the image label
    sim_step: int; sensor_time_s: float | None; wall_time_s: float
    cameras: Mapping[str, CameraModel]; state8: np.ndarray; joints: np.ndarray | None
    image_sha256: Mapping[str, str]

class FrameStore:
    def __init__(self, keep: int = 16)
    def record(self, frame: int, observation, *, wall_time_s: float) -> FrameRecord
    def get(self, frame: int) -> FrameRecord | None
    def attach_perception(self, frame: int, snapshot: Mapping) -> None   # masks for pixel snapping
    def mark_visible(self, frames: Sequence[int]) -> None                # set by the client from the images it sent
    def visible(self) -> tuple[int, ...]
```

**Algorithms**
- **Projection.** `X_cam = Rᵀ(X − t)`; `u = cx + fx·x/(−z)`; `v = cy − fy·y/(−z)`. This matches `project_world_point` (`astranomy_agentic_geometry.py:186-198`) and the OpenCV conversion in `foundation_vision/bridge.py:84-90`. Distortion uses OpenCV's model; undistortion runs 5 fixed-point iterations.
- **Triangulation.** Weighted DLT (rows `u·p3 − p1` and `v·p3 − p2`, each scaled by 1/σᵢ), then SVD. Then 5 Gauss–Newton steps on the reprojection error.
  - `cov = (JᵀWJ)⁻¹`.
  - χ² = Σ‖rᵢ/σᵢ‖² with 2n − 3 degrees of freedom; `chi2_p` is reported.
  - A small ray angle shows up as a large covariance, never as a threshold.
- **`lift_silhouette` (one view on a support).**
  - **Front-bottom point F.** The lowest mask row, at the median column of the lowest 30 % of rows, gives a ray; F is where it meets the plane z = z_s.
  - **Width w.** The chord between the left and right extremes of that band, back-projected to the plane.
  - **Centre.** C = F + ĥ·w/2, where ĥ is the horizontal ray direction. The depth prior is depth = width.
  - **Top: the far-edge rule.** The topmost silhouette pixel of a top face seen from above is that face's far edge. So the top ray meets the vertical plane through C + ĥ·w/2, not through C.
    - Prototype: top error −0.0 / +0.1 cm, against +1.9 / +2.1 cm for the centre-plane rule (A.2).
    - This is geometry, not a fitted constant.
  - **AABB.** The (ĥ, ĥ⊥) footprint square rotated into world axes, from z_s to z_top.
- **σ model (`LiftSigmaModel`).** Per-axis independent terms:
  - mask-boundary pixel σ, times range/f;
  - support σ, amplified by 1/tan(elevation) along ĥ;
  - depth-prior σ = w/4 along ĥ;
  - a lateral shape term a·w across the ray.

  `σ_px` and `a` are **fitted on GT11** (the prototype needs a ≈ 0.2 for the can). Until then they are marked `provisional` in telemetry.
- **Two views.** Centroid rays are triangulated, and association is checked with `chi2_p`. The expected gap between the two visible-surface centroids is up to about an object radius, so σ includes w/2 (h3 measured 0.5–2.1 cm same-instance gaps). The result is fused with each single-view lift.

**Veto removed / parameter added.** Removes the H3 "unique in both views + 0.9 bidirectional agreement" gate (`foundation_vision/geometry.py:34-36, 213-241`) as the route to 3D: single views produce a box with σ. Adds `σ_px` (mask boundary), `a` (lateral shape) and `σ_point_px` (model pointing). All three are measured in GT11. `k` is D2.

**Depends on:** none.

**Offline tests (pass criteria):**
1. Synthetic cameras: project, then back-project and triangulate, recovers points to 1e-9 m. Analytic covariance is within ±10 % of a 2,000-sample Monte Carlo.
2. Parallel rays return a finite estimate with an eigenvalue above 1 m².
3. Distortion round trip to below 0.01 px.
4. The same test file passes under both venvs (numpy 1.26 and 2.5).
5. `from_geometry` on every saved H1/H3 request frame reproduces `project_world_point` to 1e-9 px.

**Live validation.** None by itself. It runs inside GT3–GT5 smokes.

**Expected effect.** Enabling. Offline accuracy at the T04/T08 obstacles: 0.4–1.7 cm two-view, 0.7–1.2 cm one-view with a correct support (measured), against 5–12 cm model boxes.

**Risks.**
- Real-camera calibration error. Mitigation: distortion support plus a residual report.
- The depth = width prior fails for elongated objects seen end-on. Mitigation: a two-view or wrist-view extent overrides it, and σ_depth = w/4 is honest.

**Effort:** 2.5 days.

---

### GT2: support-plane estimate and fusion

**Fixes:**
- one-view locate and lifting need a measured support height (README C3 caution; h3_live F1 used the *true* height);
- plausibility (h1_collision F2: boxes below the table were accepted);
- place targets.

**Files to create:**
- `geometry_tools/support.py`
- `<NB>/tests/geometry_tools/test_support.py`

**Interfaces**
```python
@dataclass(frozen=True)
class SupportPlane:
    id: str                          # "S1", "S2", ... (S1 = largest/lowest dominant support)
    z: float; sigma: float           # horizontal plane z = const (normal (0,0,1)); tilt reported, not modelled
    extent_xy: np.ndarray | None     # convex polygon of evidence
    sources: tuple[dict, ...]        # [{"method", "z", "sigma", "frame"|"step"}]
def support_candidate(method: str, z: float, sigma: float, *, step: int, extent_xy=None) -> dict
def fuse_supports(candidates: Sequence[dict], *, chi2_p: float = 0.99) -> list[SupportPlane]
def plane_sweep_height(frame: FrameRecord, ext_rgb, wri_rgb, *, robot_mask=None, support_mask=None) -> dict | None  # optional, D8
```

**Algorithm.** Candidates are clustered by z with a χ² gate, so several supports can coexist (table, cabinet top). Each cluster is the inverse-variance mean. Sources:
- **Perception.** DA3 external depth with the robot masked and a RANSAC horizontal plane (ws_perception_service). σ = fit residual ⊕ the known per-view scale bias (external +2.7–3.5 %, README R3) × range × sin(elevation).
- **Contact.** A guarded stop with fingertips or the held object's bottom on a surface (event from ws_executor_control, §4.5) gives z = the tool point's z. σ = kinematic σ ⊕ the guard's detection-lag penetration, which ws_executor_control measures from the OSC realisation.
- **Locate.** An L-point with `surface:"support"` triangulated in two views.
- **Plane sweep (optional, D8).** A homography sweep between wrist and external images. It only counts with a robot mask and support mask, and only when the peak NCC and ambiguity ratio show a single dominant peak. Unmasked it failed 4/7 frames (A.3).

**Veto removed / parameter added.** None removed. `chi2_p` is a statistical convention (D2 family).

**Depends on:** GT1; perception snapshot (§4.1); executor events (§4.5).

**Offline tests:**
1. Synthetic candidates: fusion recovers the weighted mean, and a +5σ outlier forms a second cluster.
2. Truth check on saved frames: the perception-provided or contact-provided z, replayed from GT11's cache, lies within 2σ of the layout table height (T04 0.43676, T08 0.90) in ≥ 90 % of frames. This is diagnostic only.
3. Plane sweep (if D8 says yes): no estimate on the 4 known failing frames, rather than a wrong one. A.3 frames: T04 requests 2, 3, 8; T08 request 4.

**Live validation.** Smoke S2/S3 telemetry `support.sources`; sweep column "support σ at first call".

**Expected effect.** With a support at 5 mm σ, one-view lifting stays within 0.7–1.6 cm (A.2). The single-view `locate` for points on the support becomes available.

**Risks.**
- Multiple support levels. Mitigation: clustering, and each object is assigned to the support under its bottom.
- Tilted surfaces. Mitigation: report a tilt residual; future work.

**Effort:** 1.5 days (+1 day if D8 includes the plane sweep).

---

### GT3: SceneMap, the object-centric map that replaces the model's 4 boxes

**Fixes:**
- H1 items 9 and 12 (unlabelled, capped, unvalidated map; held object not modelled);
- README C3 and C4 (object IDs with per-part contact permissions; attach at grasp);
- verify M3 (palm class needs identity);
- H3 V2 and V3 (automatic, zero-shot obstacles; no `--vision-query`);
- IC13 (exclude held, target and robot).

**Files to create:**
- `geometry_tools/scene_map.py`
- `<NB>/tests/geometry_tools/test_scene_map.py`

**Files to modify**
- `agent_speedup_v2_user_check_agent.py`:
  - `reset()` `:110`: `self._obstacle_map = TransitObstacleMap()` becomes `self._scene = SceneMap(envelope, k_sigma=K_SIGMA)` plus `self._frames = FrameStore()`.
  - `_execution_state()` `:198`: `"obstacle_map"` becomes `"scene_revision"`.
  - `_context()` `:211-232`: add `"scene": self._scene.model_facts()`.
  - `_apply_agent()` `:314`: map update becomes `apply_edits` + `resolve_locates` (GT5).
  - `act()` `:500`: `self._frames.record(...)` before `_ask_agent`, plus a non-blocking poll of perception results.
  - `observe_transition()` `:582-609`: forward executor events to `self._scene.on_robot_event(...)`, then call `predict_held()`.
- `agent_speedup_v2_user_check_routing.py` `:25-48` (`TransitObstacleMap`): delete. The router input becomes `SceneMap.obstacle_set()` (§4.2). ws_swept_planner owns the replacement router.

**Interfaces**
```python
ROBOT_PARTS = ("fingers", "palm", "links", "held_object")
KINDS = ("object", "fixture", "support", "unknown")
STATES = ("static", "held", "held_unverified", "moved_unobserved", "removed_by_model")

@dataclass
class ObjectRecord:
    id: str                      # "o1".."o24" (harness-assigned, stable for the episode)
    name: str | None             # category/label; from perception, plan binding or model edit
    plan_parts: list[str]        # plan part ids A-H bound to this object (aliases for references)
    kind: str; state: str
    box: Box3D | None            # world AABB with sigma; None = bound but unmeasured
    source: str                  # perception | locate | contact | plan_only
    evidence: list[dict]         # [{method, frames, sigma, residual_px|ray_gap_m}], last 4
    observed_step: int | None; updated_step: int; last_moved_step: int | None
    attached: dict | None        # {"grip_lower","grip_upper" (grip frame), "attached_step", "verified": "gap_band"|"unverified"}
    contact_overrides: list[dict]  # model allow/forbid edits with scope
    warnings: list[dict]; revision: int

class SceneMap:
    revision: int
    def bind_plan(self, bindings: Sequence[dict]) -> list[dict]
    def ingest_perception(self, snapshot: Mapping, frame: FrameRecord) -> list[dict]
    def add_points(self, results: Sequence["LocateResult"]) -> None
    def apply_edits(self, edits: Sequence[Mapping], *, step: int) -> list[dict]       # per-edit feedback
    def on_robot_event(self, event: Mapping, *, state8, joints, step: int) -> list[dict]
    def predict_held(self, state8) -> None
    def anchor(self, anchor_id: str) -> "AnchorGeometry"                             # object | part | region | point
    def obstacle_set(self, intent: "ContactIntent | None", *, step: int) -> dict      # §4.2
    def model_facts(self) -> dict                                                    # §GT9 "scene" block
    def snapshot(self) -> dict                                                       # full audit record
```

**Algorithms**
- **Ingest (per snapshot).**
  - Each perception instance is lifted with GT1. Two views use `lift_two_view`; otherwise `lift_silhouette` on its support. If perception supplies a 3D hypothesis, it is fused by covariance.
  - Instances are associated to records by perception track id when given. Otherwise association uses Mahalanobis distance of centres (both covariances), gated at χ²₃(0.99), with category consistency, solved by Hungarian assignment.
  - Unassociated instances get new `o` ids.
  - An associated static record whose update fails the gate becomes `moved_unobserved` (setting `last_moved_step`), and a `moved` event goes to feedback.
  - Records whose projection lies in view, unoccluded by the robot (FK silhouette from ws_swept_planner), but that are not detected for `n_miss` consecutive frames get a warning `not_seen_where_expected`. They are never deleted. `n_miss` is derived from the per-category detection recall measured in GT11.
- **Plan binding.** Plan bindings come from GT10 (noun plus x/w cells). A plan object binds to the unique record whose external projection overlaps the cell (mask–cell overlap, as in h3_live F7) with a compatible category. Ambiguity leaves it unbound, listed in `unbound_plan_parts`, and the model binds it with an edit. Bound records get `plan_parts`, and plan ids work as reference aliases.
- **Attach.** On `grasp_closed` from ws_executor_control's gap-band continuation predicate (G7/IC5):
  - Choose the anchor of the last approach stage. If there is none, choose the record whose box (inflated by kσ) contains the pad midpoint at closure (geometric membership, H2 §4.4 item 4).
  - Store the 8 box corners in the grip frame, and set state `held` (verdict `object_in_jaw`) or `held_unverified` (`unknown`).
  - If no record matches but the verdict is `object_in_jaw`, attach an unnamed envelope. Its width is the measured gap and its depth is the pad extent. Its height runs from the support z (σ) up to the grip: measured bounds, no guess.
  - The record leaves the obstacle list and goes to `held`.
- **Detach.** On `released`, the world box is the last prediction and the state is `moved_unobserved`. σ_z is inflated by the measured height of the release above the support. The next perception observation returns it to `static`.
- **Plausibility (warnings, never vetoes).**
  - `below_support`: bottom < z_s − kσ. A perception-lifted box is clamped onto the support with a note.
  - `floating`: bottom > z_s + kσ with no object under it.
  - `inside_robot`: the box intersects the robot FK silhouette at the current joints. It is excluded from obstacles, because it is likely self-detection (README R2, H2 F3).
  - `duplicate`: two boxes overlap by > 50 % of the smaller one. A merge is proposed to the model.
  - `oversize`: extent > 0.6 m for a movable object.
  - `moved`, `not_seen_where_expected`.
- **Per-part contact permissions** (computed per stage in `obstacle_set(intent)`):
  - Default: every part is forbidden on every object.
  - The stage intent from GT4 allows:
    - `fingers` on the approach anchor (and its parent object);
    - `held_object` on the destination support/region for place stages and on a container for `inside`;
    - nothing for `palm` or `links`.
  - Model `allow_contact` / `forbid_contact` edits add overrides. Default scope: the next `execute` fragment.

**Veto removed / parameter added.**
- Removes the 4-box cap and the single global clearance (`_fragments.py:54-59, 94-100`), and the model's box-writing duty (`_agent_client.py:92-105`).
- Removes "remove the target from the avoid set" (`:101-102`): permissions replace it.
- Adds `n_miss` (measured recall) and `k` (D2). The representation cap of 24 named objects is not a veto: further objects stay in `obstacle_set` as `other`.

**Depends on:** GT1, GT2, GT8; perception snapshot (§4.1); executor events (§4.5); robot FK silhouette (§4.3).

**Offline tests:**
1. Synthetic ingest/attach/detach sequence. The held box follows the grip pose exactly; after release, the box sits where it was released.
2. Replay H1-family first model maps as if they were measurements: `below_support` fires on T08 t100, T08 stove-off and T04 t100 (bottom 0.40 against 0.437). This needs support z at the truth ± 5 mm.
3. Permissions: an approach intent on o1 lets fingers touch o1 only; palm stays forbidden.
4. Perception fail-open: an empty or erroring snapshot leaves the previous records intact, marked `status: perception_unavailable`.
5. Association on H3's saved masks: plates C/D in T04 bind correctly from plan cells in 11/11 frames (h3_live F7 replay).

**Live validation.**
- S2/S3: the obstacle (can or bottle) appears as an `o` id at call 1 without being named by the operator, and its box covers truth with kσ, offline-scored from logs.
- Sweep columns:
  - first-call obstacle coverage per obstacle variant;
  - plausibility warnings per task;
  - attach verdicts against strict truth at release;
  - palm-on-target episodes per task (ws_runner_measurement's episode metric).

**Expected effect.**
- First obstacle boxes within about 1–3 cm (C3 estimate; A.2 measured 0.7–1.6 cm one-view), against 5–12 cm.
- Obstacle coverage with kσ in 2/2 scenes, against 0/4 for H1-family first boxes (estimate, extrapolated from A.2 and h3_live F1).
- The palm class becomes coverable by the body sweep (verify M3).
- Zero-shot obstacle detection without `--vision-query` depends on ws_perception_service's class-agnostic discovery.

**Risks.**
- Wrong instance binding. Mitigation: overlays show ids; `bind` edit; never auto-bind ambiguous cells.
- Perception false positives block paths. Mitigation: plausibility checks plus `remove` with a reason (D6).
- Attach to the wrong object. Mitigation: geometric membership plus the gap verdict; `held_unverified` is still swept, which is conservative.

**Effort:** 3 days.

---

### GT4: target-reference grammar and resolver (resolved at the moment of use)

**Fixes:**
- R1 for fragment targets (T08: Astra aimed at a box 8.5 cm off in x and 6.6 cm low, C5);
- H1 item 10 and README C2 (palm envelope, depth bound by the measured height under the palm footprint);
- IC8 (the harness grounds the metric pose; the model selects);
- IC14 (re-localise at use).

**Files to create:**
- `geometry_tools/references.py`
- `<NB>/tests/geometry_tools/test_references.py`

**Files to modify:** `agent_speedup_v2_user_check_fragments.py`
- `execution_schema()` `:48-74`: add `"target": anyOf[TargetRef, null]` to each segment, and drop `route` (`:56-59, :68`), since the model no longer writes boxes. ws_executor_control may add `stop_on` in the same schema; the field order is agreed in §4.6.
- `validate_execution()` `:113-176`:
  - `move_to_pose` and `approach` need exactly one of `xyz_m` and `target`;
  - `move_relative` needs `xyz_m`, with `target = null`;
  - `set_gripper` and `wait` need both null.
- `FragmentSequencePrimitive.__init__` `:223-229`: `obstacle_map=` becomes `scene=`, `resolver=`, `planner=` (ws_swept_planner).
- `_enter_segment()` `:259-321`: right after `pose = measured_pose(view.state)` (`:262`):
  ```python
  if segment.get("target") is not None:
      try:
          resolved = self.resolver.resolve(segment["target"], skill=segment["skill"], profile=segment["profile"],
                                           entry_state8=view.state, quaternion_wxyz=segment["quaternion_wxyz"],
                                           step=self.global_step)
      except ReferenceUnresolved as exc:                 # mirrors :282-284; zero actions, normal handoff
          self._end("failed", "reference_unresolved", detail=exc.feedback()); return
      self._target_xyz = np.asarray(resolved.xyz_m); self._current["resolution"] = resolved.to_dict()
  elif segment["skill"] in ("move_to_pose", "approach"):
      self._current["resolution"] = self.resolver.annotate(...)   # legacy raw/relative: telemetry + palm clamp
  ```
  The envelope check (`:288-291`) and the router (`:293-317`) then see the resolved target. Whether the envelope stays a veto is ws_executor_control's decision (B1).
- `fragment_capabilities()` `:179-213`: replace the `routing` text (`:188-196`) and add `target` semantics.
- `progress()` `:449-483`: per-segment `resolution` summaries, compact, with reasons verbatim (IC15).

**TargetRef (strict JSON schema, no `$ref` or `pattern`; every key required)**
```json
{"type":"object","additionalProperties":false,
 "required":["anchor","point","offset_m","frame","tool_point"],
 "properties":{
  "anchor":{"type":"string","enum":["A","B","C","D","E","F","G","H",
            "o1","o2","...","o24","L1","L2","L3","L4"]},
  "point":{"type":"string","enum":["center","top","bottom","front","behind","left","right",
           "beside_front","beside_behind","beside_left","beside_right","inside"]},
  "offset_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},
  "frame":{"type":"string","enum":["world","viewer"]},
  "tool_point":{"type":"string","enum":["grip_site","fingertips","palm","held_bottom","held_center"]}}}
```
- The id enum is fixed: A–H plan parts or regions, o1–o24, L1–L4. The schema therefore never changes during an episode, which keeps the cacheable prefix stable (IC9).
- Existence is checked against the live scene at `_apply_agent`. An unknown id produces feedback `reference_unknown` and no motion, mirroring `agent.py:319-323`. It is not a SchemaError.

**ResolvedTarget (to ws_executor_control and ws_swept_planner)**
```json
{"xyz_m":[x,y,z], "quaternion_wxyz":[...]|null, "sigma_m":[sx,sy,sz],
 "anchor":"o3", "anchor_kind":"object|part|region|point", "anchor_revision":7,
 "point":"top", "tool_point":"fingertips", "frame":"world",
 "terms":{"anchor_point_m":[...], "offset_world_m":[...], "tool_offset_m":[...],
          "uncertainty_margin_m":0.012, "palm_clamp_m":0.0, "palm_clearance_m":0.018},
 "contact_intent":{"fingers":["o3"], "held_object":[], "palm":[], "overrides":[]},
 "sigma_along_motion_m":0.006,
 "scene_revision_at_decision":12, "scene_revision_at_use":13,
 "evidence_frames":[118,120], "evidence_age_steps":3, "source":"reference|raw_world|relative",
 "warnings":["part_geometry_unbound: using whole object o1"]}
```

**Resolution algorithm (`TargetResolver.resolve`)**
1. **Anchor.** `SceneMap.anchor(id)` returns one of:
   - an object or part box with σ;
   - an L-point (xyz, cov);
   - a region (GT6);
   - a plan part via its binding.

   A bound sub-part without geometry falls back to its object, with warning `part_geometry_unbound`. Failures:
   - `anchor_unmeasured`;
   - `locate_stale`: the point's containing object moved after the locate;
   - `support_unknown`.
2. **Point P, covariance Σ_P.**
   - `center`, `top`, `bottom`: the box centre and the top or bottom face centre.
   - Faces: the support function of the box (or footprint polygon) along the viewer-frame direction from the frozen view frame (`spatial.py:158-212, 236-252`). `front` is toward the external camera.
   - `beside_*`: GT6.
   - `inside`: the footprint centre at the top (rim) z.
   - L-points accept `center` only; anything else returns `point_not_applicable`.
3. **Offset.** `world`, or `viewer` = [right, away, up] (`right_xy`, `away_xy` of the frozen frame; up = +z).
4. **Tool point.** Using the target orientation `R_t` (the segment quaternion, or the entry orientation if null), with a = R_t·ẑ (approach axis):
   - `fingertips` = grip + a·0.00925;
   - `palm` = grip − a·0.03104;
   - `held_*` = R_t·(attached point in the grip frame).

   The grip target is G = P + offset − tool_vector. Values come from `robot_facts_v2.json:11, 18`.
5. **Uncertainty margin.** On transit-profile stages with offset ≠ 0, G moves along ô = offset/|offset| by k·√(ôᵀΣ_Pô), so the stated gap holds at the k-σ level. No margin on contact-profile stages (contact is intended and guarded). The margin is reported.
6. **Palm envelope.** For any stage with the hand moving along +a toward the scene, including legacy raw and relative approaches:
   - Rays along a are cast from a 1 cm grid over the palm footprint at the target pose: R1 −104…+100 mm, R0 ±31.6 mm (`robot_facts_v2.json:440-447`), with the palm profile p(ξ) (`:19-438`).
   - They are intersected with every palm-forbidden box, inflated by kσ, using the footprint polygon when available, else the AABB.
   - If the palm plane would enter a box, G is clamped back along −a by the deficit. `palm_clamp_m` and `palm_clearance_m` are reported.
   - An `allow_contact palm` override disables the clamp (pushes). Whether the clamp is active or report-only is **D3**.
7. **Contact intent.**
   - An `approach` stage with `tool_point` fingertips or grip_site allows `fingers` on the anchor (and its parent object).
   - Stages with `held_*` allow `held_object` on the anchor support or region.
   - Overrides from edits are added.
8. **Timing.** Resolution runs at **each segment's measured entry**. Later segments of one fragment therefore see newer perception (IC14), and the two scene revisions are logged.

For legacy raw world targets, `annotate()` reports:
- the nearest object and its distance;
- `raw_target_inside_object` when G lies inside a measured box;
- palm clearance, with the same clamp rule as step 6;
- telemetry `source:"raw_world"`.

There is no veto (D1).

**Mapping onto skills**

| Skill | Before | After |
|---|---|---|
| move_to_pose | absolute `xyz_m` from model estimates | `target` (preferred) or `xyz_m` (raw, flagged) |
| approach | displacement `xyz_m` | `target` (absolute, resolved) or displacement |
| move_relative | displacement | unchanged (robot-anchored) |
| set_gripper / wait | — | unchanged |
| direct `motion` | relative/absolute ≤ 5 cm | unchanged; absolute flagged like raw targets |

**Veto removed / parameter added.**
- Removes the model's depth arithmetic for grasps (T04 commanded 5.1–6.2 cm below the rim).
- Adds the palm clamp (robot geometry + measured σ; D3) and the margin (k; D2).
- An unresolvable reference costs one handoff with zero motion. This is not a new veto: the current behaviour for non-finite targets is the same (`_fragments.py:282-284`).

**Depends on:** GT3, GT6, GT8; ws_executor_control (consumes ResolvedTarget, contact intent, `sigma_along_motion_m` for the IC1 slow zone); ws_swept_planner (consumes the target and intent).

**Offline tests:**
1. Hand-computed cases for every point/tool_point/frame combination, including a 90°-yawed hand and a tilted approach axis.
2. The palm clamp on the 4 T04 palm-on-rim episodes uses truth rims (0.542–0.552 m) and the logged commanded grasp poses (`runs/…transit100_t04…/trial/trace.jsonl`). The clamped grip height must lie above the logged contact-onset grip height in 4/4. Prior: the bound is 4–21 mm above onset (verify F3).
3. Legacy replay: every H1 T04/T08 executive response re-validates under the new schema with `xyz_m`. `annotate()` flags the T08 call-1 approach target that sat 1.1 cm below the bottle top (C5) as `raw_target_inside_object` against the GT11 measured bottle box.
4. Unknown id gives `reference_unknown` feedback and 0 actions.
5. Performance: below 2 ms per resolve.

**Live validation.**
- S2: the first grasp uses `target`; `palm_clamp_m` and `palm_clearance_m` are logged; there is no palm-on-mug contact episode in the first grasp (episode metric).
- Sweep columns:
  - share of targets via references;
  - `reference_unresolved` per task;
  - palm-on-target episodes;
  - regrasps per object (the trade-off for D3);
  - stall-after-approach calls.

**Expected effect.**
- Targets inherit the 1–3 cm geometry (C3 estimate).
- The palm class (55 % of forbidden samples) is addressed directly: the T04 bound sits 4–21 mm above contact onset (verified offline).
- 1–2 fewer calls per pick-and-place and 5–8 fewer on the composite (h1_zero_shot F3; hypothesis).

**Risks.**
- The palm clamp makes grasps shallower: rim − 31 mm is 5–14 mm shallower than H3's successful grasps. Mitigation: D3, footprint polygons instead of AABBs, regrasp telemetry.
- The model mixes up `tool_point`. Mitigation: the resolved terms are echoed in feedback; examples in GT9 are generic.

**Effort:** 3 days.

---

### GT5: `locate`, pointing to XYZ ± σ in the same decision cycle

**Fixes:**
- H1 item 20 and h1_zero_shot F3 (no pixel-to-3D tool; "Pixels specify rays; infer depth", `_agent_client.py:224-225`);
- README C3;
- parts that perception does not segment (handles, compartments, drawer pulls; t2, t5, g0, g3 and g7 in the registry `agent_speedup_v2_tasks.py:55-101`).

**Files to create:**
- `geometry_tools/locate.py`
- `<NB>/tests/geometry_tools/test_locate.py`

**Files to modify:**
- `agent_speedup_v2_user_check_agent_client.py`: `agent_schema()` `:237-260` gets `locate`.
- `agent_speedup_v2_user_check_agent.py`: `_apply_agent()` `:313` calls `resolve_locates` before building the fragment, so L-ids are valid in the same response.

**Response field (strict schema, required; `[]` when unused)**
```json
"locate":{"type":"array","maxItems":4,"items":{"type":"object","additionalProperties":false,
  "required":["id","label","surface","observations"],
  "properties":{
   "id":{"type":"string","enum":["L1","L2","L3","L4"]},
   "label":{"type":"string","minLength":1,"maxLength":60},
   "surface":{"type":"string","enum":["support","object","unknown"]},
   "observations":{"type":"array","minItems":1,"maxItems":4,"items":{"type":"object","additionalProperties":false,
     "required":["frame","view","uv","box_uv"],
     "properties":{"frame":{"type":"integer","minimum":0},
      "view":{"type":"string","enum":["external","wrist"]},
      "uv":{"anyOf":[{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},{"type":"null"}]},
      "box_uv":{"anyOf":[{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},{"type":"null"}]}}}}}}}
```

**Algorithm (`resolve_locates(requests, frames, scene, sigma_point_px, step)`)**
1. Each observation needs a `FrameRecord` whose images the model can see (`FrameStore.visible()`). Otherwise the result is `frame_unavailable`.
2. **Snapping.** If perception masks exist for that frame and view, a `uv` inside an instance polygon (or a `box_uv` whose centre is inside one) is snapped. The mask's geometry and σ then replace the pointed pixel, and `snapped_to` is reported. No IoU threshold is used.
3. **Geometry:**
   - **Two or more `uv` (any frames or views):** `triangulate`. If `chi2_p` < 0.01, the result is `views_inconsistent`, reporting per-view residuals plus the best single-view alternative.
   - **One `uv`:**
     - `surface:"support"` uses `ray_plane` with the GT2 support.
     - `surface:"object"` uses `ray_box_entry` on the snapped or containing object, with σ along the ray = half the box depth.
     - Otherwise the result is `ray_only` (the ray is reported; not usable as an anchor).
   - **`box_uv` in one view plus `surface:"support"`:** `lift_silhouette` with the box as the polygon, giving a Box3D.
   - **`box_uv` in two views:** triangulate the box centres plus extents.
4. The result is stored as `L1..L4` in the SceneMap. `on` is set to the containing object, used for invalidation when that object moves.
   - The result is usable by references **in the same response**, because GT4 resolves at stage entry after `_apply_agent`.
   - The model sees the value (xyz, σ, method, residual) in the next request's `scene.points`.
5. σ for unsnapped pixels:
   - pointing σ: `sigma_point_px[view]`, measured in pointing smoke S1;
   - provisional until S1: one plan-cell quantisation σ = 64/√12 ≈ 18.5 px (a property of the 8×8 representation).
   - Snapped pixels use the mask σ from GT11.

**Round trip, tool loop or up-front inventory? Decision: no extra round trip by default.**
- **(a) Up-front facts.** The measured inventory comes up front in `context.scene`. Perception runs asynchronously, so its snapshot is ready at request time. The model references ids directly: zero round trips.
- **(b) Same-response locate.** `locate` inside the same response is resolved locally in milliseconds and is usable by that response's targets via references. The model does not need to read the number to act on it.
- **(c) Native tool loop: not in v1.** The transport can do it (astranomy ran it live; `tools` is accepted, `chatgpt_plan.py:37-41`). But:
  - H1's extractor needs exactly one output_text (`astra.py:540-541`);
  - history commit assumes one response per decision (`agent_client.py:495`);
  - each round trip is a full new request at the transport floor of about 3–5 s plus prefill and new reasoning (cross_critic C12; latency model 6.6–6.9 s + 0.051 s per reasoning token).
- When the model truly needs a value first, a response with `type:"observe"` plus `locate` costs one ordinary call. That is the same price as a tool round trip, on the existing path.
- **D5** asks whether to add the tool loop later.

**Veto removed / parameter added.** Adds `sigma_point_px` (measured, S1). No veto; unresolved locates are reported.

**Depends on:** GT1, GT2, GT3; ws_protocol_context (which frames' images stay visible, §4.6).

**Offline tests:**
1. Synthetic pointing with Gaussian pixel noise gives a χ² coverage of 95 % ± 2 % at the 95 % level.
2. On saved H1/H3 frames, project truth points (can top centre, bottle top, plate centres) and add noise at σ = 18.5 px. Triangulated errors should match the predicted σ (ratio 0.8–1.25).
3. Snapping: a pixel inside H3's saved can mask returns `snapped_to` and an error ≤ 2 cm.
4. Same-cycle use: a response with `locate L1` plus a target on `L1` resolves in one decision.

**Live validation.**
- **S1 (pointing calibration, ≤ 20 model calls, no simulator).** Saved frames; Astra points at named objects; σ_px per view against truth projections.
- **S2.** At least one handle/rim locate, `views_consistent`, used by a target in the same response.
- Sweep columns: locates per task, consistency rate, snapped share.

**Expected effect.**
- With 20 px pointing: about 3 cm lateral at the external camera and 2 cm at the wrist (A.1). Two-view errors about 1–4 cm before snapping; snapped 0.4–1.7 cm (h3_live F1).
- Removes the parallax-probing loops: drawer pull 174.6 / 230.8 s; moka handle 11 calls, 297 s (h1_zero_shot F3). Saving: hypothesis.

**Risks.**
- Astra's pixel accuracy is unknown. Mitigation: S1, snapping, `views_inconsistent` feedback.
- If ws_protocol_context strips images at insertion, only the current pair can be pointed into. Mitigation: `visible()`, plus wrist motion between decisions for multi-frame triangulation.

**Effort:** 1.5 days.

---

### GT6: metric meaning of on / beside / directional / inside

**Fixes:**
- h1_zero_shot F9 ("beside" and "on" have no metric meaning; the half-plane resolver at `spatial.py:333-335` never runs because macros are unused);
- the t6 relayout release outside the 10×10 cm goal box;
- T04-20 mm placed 3–4 cm off centre.

**Files to create:**
- `geometry_tools/regions.py`
- `<NB>/tests/geometry_tools/test_regions.py`

**Files to modify:** none in `spatial.py`. Its frozen view frame (`:158-212`) and `region_directions` (`:236-252`) are reused read-only.

**Interfaces**
```python
@dataclass(frozen=True)
class RegionGeometry:
    id: str; relation: str; support: str; anchor: str | None
    target_xyz: np.ndarray; target_cov: np.ndarray
    polygon_xy: np.ndarray; z: float
    derived: dict    # {"edge_m", "held_half_m", "gap_m", "sigma_place_m", "k"}
def region_geometry(spec: Mapping, scene: SceneMap, view_frame: Mapping, *, held: ObjectRecord | None,
                    k_sigma: float, sigma_place_m: float) -> RegionGeometry
```

**Algorithm.** d is the frozen viewer direction (`right_xy`, `away_xy`), h_X(d) is the support function of X's footprint along d, and H is the held object (or the placement-goal object if nothing is held):
- **left_of, right_of, front_of, behind.**
  - Target centre = anchor centre + d·(h_A(d) + g + h_H(d)), at the support z.
  - Gap g = k·√(σ_A,d² + σ_H,d² + σ_place²).
  - Band polygon: from h_A(d) + g to h_A(d) + g + 2h_H(d) along d, and ± h_A(d⊥) laterally.
- **on.** The support's footprint centre at its top. If that spot is occupied, the nearest point where H's footprint fits inside the support footprint inset by kσ.
- **inside.** The container's footprint centre at its rim z. The descent below the rim is a guarded approach (ws_executor_control).
- **near.** Keeps the plan's explicit `bounds_xy_m` (`spatial.py:112-113`).
- **σ_place** is the measured placement realisation error (the release-to-rest offset statistic from the first sweep). Until then it is the stage position tolerance (`_fragments.py:29-33`: 4 mm transit, 2 mm contact) as a lower bound, flagged provisional.

**Veto removed / parameter added.** None removed. The band depth and gap are derived from measured extents, σ and k. There is no task constant: LIBERO's "on < ~3 cm" and "3–12 cm past the edge" are only used as a diagnostic.

**Depends on:** GT3; GT8 (held footprint); the frozen view frame.

**Offline tests:**
1. Synthetic anchors in rotated viewer frames.
2. Developer-only diagnostic: for the LIBERO side-goal BDDL boxes (7 boxes, 3–12 cm past the anchor edge; h1_zero_shot F9), our `beside_*` target centre falls inside the BDDL box when built from truth extents, in ≥ 6/7. This is not a gate.

**Live validation.** t6 (official and relayout) and g5 in the sweep: first-release relation verified (strict), and calls per placement.

**Expected effect.** First-try side and centre placement; 3–9 fewer calls on such tasks (h1_zero_shot F9; hypothesis).

**Risks.**
- Viewer words in tasks whose camera differs. Mitigation: the frozen frame comes from calibration, and `available=false` is reported.
- A held object with unknown extent. Mitigation: unknown gives h_H = 0 plus a σ term, with warning `held_extent_unmeasured`.

**Effort:** 1.5 days.

---

### GT7: overlays on the current image pair

**Fixes:**
- h1_collision F2 (boxes never shown back to the model);
- IC9 (replace, never add images; about 893 tokens per image);
- identity verification for GT3 and GT10.

**Files to create:**
- `geometry_tools/overlay.py`
- `<NB>/tests/geometry_tools/test_overlay.py`

**Files to modify:** `agent_speedup_v2_user_check_agent_client.py`, `request_agent()` `:557-563`:
- the images become `overlay.render(view.<cam>_rgb_upright, camera, facts, planned=...)`;
- labels become `f"{label} with measured-object overlay, frame {step}"`;
- the client calls `frames.mark_visible(...)` with the frames whose images it actually sent.

Raw images still go to perception and the video recorder.

**Interface**
```python
@dataclass(frozen=True)
class OverlayStyle:
    line_px: int = 1; halo_px: int = 1; font_px: int = 12; draw_sigma_box: bool = False
def render(rgb_upright: np.ndarray, camera: CameraModel, facts: Mapping, *,
           planned: Sequence[Mapping] = (), style: OverlayStyle = OverlayStyle()) -> np.ndarray   # same shape/dtype
```

**Algorithm.** PIL `ImageDraw` (as in `mark_prompting.py`).
- **Boxes.** For each record with a box, project the 8 corners. Draw the 12 edges, clipped at depth > 0.02 m and to the image, 1 px over a 1 px dark halo. Colour by role:
  - plan-bound: green;
  - other obstacles: orange;
  - held: cyan;
  - `moved_unobserved`: dashed.
- **Labels.** Id (plus name ≤ 12 characters) above the projected top-centre on a dark tab.
- **Locates.** L-points as a yellow cross with the id.
- **`planned`.** From the fragment trace and ws_swept_planner (§4.3):
  - the last fragment's executed grip-site trace: thin white;
  - unexecuted planned segments: dashed;
  - a blocked segment: red, naming the blocking id.
- No fills, so the model can still point at the objects under the boxes.

**Veto removed / parameter added.** None. The style is fixed for all tasks.

**Depends on:** GT1, GT3; §4.3 (PlannedSweep).

**Offline tests:**
1. Shape and dtype preserved; deterministic; the input array is never written.
2. Saved T04 frame: the projected can box encloses the SAM can mask's bounding box ± 3 px.
3. Encoded PNG size delta is reported (expected about +5–15 %; at +0.11 s/MB it is negligible).
4. The image count per request is unchanged.
5. Render time ≤ 10 ms per image.

**Live validation.** S2 arms with and without overlays, in the same window (ws_runner_measurement interleaving): reasoning tokens per call, wrong-binding corrections, pointing consistency. D4 decides.

**Expected effect.** No added image tokens. Faster identity and geometry error detection by the model (unmeasured). H3's 2× reasoning is a reason to A/B it (C1).

**Risks.** Clutter, reasoning inflation, and the model trusting a wrong box. Mitigation: thin lines; σ is in the facts; A/B.

**Effort:** 1.5 days.

---

### GT8: robot tool envelope, rendered from one description

**Fixes:**
- H1 item 10 (only pads and gap are given, `_agent_client.py:228-233`; the 31.04 mm palm offset is in `robot_facts_v2.json:18`);
- h1_zero_shot F8 (render every prompt number from one robot description).

**Files to create:**
- `geometry_tools/robot_envelope.py`
- `<NB>/tests/geometry_tools/test_robot_envelope.py`

**Interfaces**
```python
@dataclass(frozen=True)
class ToolEnvelope:
    grip_site_from_hand_m: float          # 0.097  (asset.grip_site_along_R2_m)
    fingertip_reach_m: float              # 0.00925 (:11)
    pad_span_along_approach_m: tuple      # (-0.0116, 0.0044) (:12-15)
    pad_size_m: tuple                     # (0.016, 0.008, 0.016)
    palm_offset_m: float                  # 0.03104 (:18), minimum of the profile
    palm_profile: np.ndarray              # (K,2) xi_R1_m, p_m (:19-438)
    footprint_R1_m: tuple; footprint_R0_m: tuple   # (-0.10399, 0.10043), (-0.03164, 0.03162) (:440-447)
    finger_reach_R1_by_q: dict            # q_mm -> finger_and_pad_R1 extent (:478-621)
    open_inner_gap_m: float               # 0.079
def load_envelope(path, *, geometry: Mapping | None = None) -> ToolEnvelope   # cross-checks hand_to_grip_site_m, gap
def envelope_prompt_text(env: ToolEnvelope) -> str                            # ~0.4 KB
```

Only the `asset` section is read. The task-keyed `start_poses` and `measured_tasks` are ignored (F8). The rendered text, from values only:

> TOOL ENVELOPE (hand frame: +Z approach, +Y jaw). Fingertips reach 9.3 mm beyond the grip site; pads span −11.6…+4.4 mm along the approach. The palm underside is 31.0 mm behind the grip site across the hand (91 mm at the −104 mm end). The hand footprint spans −104…+100 mm along the jaw axis and ±31.6 mm across it. With the fingers fully open, the finger bodies reach ±66 mm along the jaw axis (landing band ±54 mm). Open inner gap 79 mm. The harness applies these numbers when it resolves targets and checks paths.

**Veto removed / parameter added.** None. Facts only.

**Depends on:** none.

**Offline tests:**
1. A snapshot test: every number in the text equals the JSON value.
2. The cross-check against the validated geometry `gripper` dict (`astranomy_agentic_geometry.py:33-48`) passes.
3. No task or benchmark object words in the text.

**Live validation.** S2: the request text contains the envelope; palm-on-target episodes per sweep.

**Expected effect.** C2's prompt half, at 0 calls and about 0.3–0.4 KB. Grasp depth also depends on localisation (verify F3), so the main effect comes with GT4's clamp.

**Risks.** None significant.

**Effort:** 0.5 days.

---

### GT9: the model-facing geometry contract (prompt and schema in `agent_client.py`)

**Fixes:**
- H1 item 20 (delete "Pixels specify rays, not known 3-D points; infer depth", `:224-225`);
- item 10 (envelope);
- item 9 (no model boxes);
- H3 V6 (compact current facts, not a 2.3–2.5k-token summary);
- IC9 and IC15.

**Files to create:**
- `geometry_tools/prompt.py`: `GEOMETRY_INSTRUCTIONS`, `response_fields_schema()`, `scene_edits_schema()`.

**Files to modify:** `agent_speedup_v2_user_check_agent_client.py`
- `CONTEXT_KEYS` `:46-50`: add `"scene"`.
- `AGENT_INSTRUCTIONS` `:52-234`:
  - delete the obstacle_map paragraph (`:92-105`; 1,126 characters);
  - delete "Choose the destination and relevant obstacles using RGB and camera calibration" (`:139`);
  - delete the box/planner sentences (`:140-155`) that describe model boxes;
  - keep the pixel conventions (`:221-223`: u right, v down, integer pixel centres) and delete the projection formula and "infer depth" (`:223-226`);
  - replace the gripper paragraph (`:227-233`) with `envelope_prompt_text()`;
  - append `GEOMETRY_INSTRUCTIONS`.

  The oracle cues at `:205-209` belong to ws_runner_measurement / ws_protocol_context (A1). The belief/plan/why fields are kept unchanged: they are the model's cross-call memory (D3 caution).
- `agent_schema()` `:237-260`: remove `obstacle_map` (`:248, :257`); add `locate` (GT5) and `scene_edits`. Segments get `target` from `execution_schema()`.
- `validate_agent_decision()` `:263-307`: the required set becomes `{belief, plan, why, type, motion, macro, execution, locate, scene_edits}`, with structural checks for the two new arrays. Semantic id checks run in the policy (GT4).
- `_execution_evidence_view()` `:335-342`: stop copying `obstacle_map`; keep `scene_revision`.
- `request_agent()` `:504-566`: images per GT7.

**`scene_edits` (strict, required, `[]` when unused)**
```json
"scene_edits":{"type":"array","maxItems":6,"items":{"type":"object","additionalProperties":false,
  "required":["op","id","value","reason"],
  "properties":{
   "op":{"type":"string","enum":["label","bind","kind","allow_contact","forbid_contact",
                                 "merge","remove","restore","add_from_locate"]},
   "id":{"type":"string","enum":["A","...","H","o1","...","o24","L1","L2","L3","L4"]},
   "value":{"anyOf":[{"type":"string","minLength":1,"maxLength":60},{"type":"null"}]},
   "reason":{"type":"string","minLength":1,"maxLength":120}}}}
```
`value` per op:

| op | value |
|---|---|
| label | name |
| bind | target id (plan part → o-id or L-id) |
| kind | fixture / movable / support |
| allow_contact, forbid_contact | palm / fingers / held_object |
| merge | other o-id |
| remove, restore | null |
| add_from_locate | L-id |

There are **no metric fields**. Each edit returns feedback `{op, id, status: applied|rejected, reason_code}`.

**Model-facing `context.scene` (current only; about 30–45 tokens per object, 300–500 tokens per call for LIBERO)**
```json
{"revision":14,"observed_step":120,"units":"m","sigma":"1-sigma per axis",
 "supports":[{"id":"S1","z":0.437,"s":0.003,"src":["perception","contact"]}],
 "objects":[{"id":"o1","plan":["A","B"],"name":"mug","kind":"object","state":"static",
             "box":[[-0.130,-0.150,0.437],[-0.050,-0.070,0.542]],"s":[0.006,0.006,0.004],"src":"two_view","seen":120}],
 "held":null,
 "points":[{"id":"L1","label":"handle","xyz":[-0.06,-0.11,0.51],"s":[0.004,0.004,0.006],"on":"o1","src":"pointed_two_view","res_px":2.1,"step":118}],
 "regions":[{"id":"D","rel":"right_of","anchor":"C","target":[x,y,z],"half":[hx,hy],"gap":0.014}],
 "warnings":[{"id":"o5","code":"below_support"}],
 "unbound_plan_parts":["E"],
 "perception":"ok|degraded|unavailable"}
```
ws_protocol_context replaces earlier copies in the history with `{"revision":N,"superseded":true}` at insertion. That keeps the history append-only and avoids H3's 24–26 % re-sent vision text.

**`GEOMETRY_INSTRUCTIONS`** (static, cacheable, about 2.4k characters; no task words):
- **MEASURED SCENE.** `context.scene` lists measured objects (ids, plan aliases, box in world metres, 1-σ, source, last observed step, state). The images show the same ids and boxes. Do not estimate positions, heights or extents from pixels; use the scene, targets and locate.
- **TARGETS.** Give `move_to_pose` or `approach` a `target` = {anchor, point, offset_m, frame, tool_point}, or `xyz_m`. The full semantics are as in GT4, with one generic example: "3 cm between fingertips and o3's top" = {anchor:"o3", point:"top", offset_m:[0,0,0.03], frame:"world", tool_point:"fingertips"}. The harness resolves targets from the newest measurements when each stage starts, adds the measured uncertainty on free-space stages, keeps the palm above objects it may not touch, and reports the resolved numbers. Displacements stay displacements from the measured stage entry.
- **LOCATE.** For things not in the scene: up to 4 entries, each with pixel `uv` or `box_uv` in the images labelled frame N (u right, v down, integer pixel centres, 0…width−1). Two views triangulate. One view needs surface `support` or `object`. Results are usable as anchors in the same response and are reported next time with σ.
- **SCENE EDITS.** By id only; no coordinates.
- **CONTACT.** Fingers may touch the current approach anchor, and the held object its destination. Palm and links touch nothing unless allowed. The planner avoids every other measured object with the whole robot and the held object.

**Veto removed / parameter added.**
- Removes `validate_route` on model boxes (`_fragments.py:90-110`) from the decision path.
- Removes the H1 cue that the target must be manually removed from the avoid set.
- No parameters.

**Depends on:** GT3, GT4, GT5, GT8; ws_protocol_context (assembly, history stubs, cache prefix).

**Offline tests:**
1. A strict-schema lint: every object closed, all keys required, no `$ref` / `pattern` / `$defs`. The schema is ≤ 6 KB, against 3,232 characters today.
2. All H1 T04/T08 responses re-validate under the new schema once wrapped with `locate:[]` and `scene_edits:[]`, with `obstacle_map` dropped (legacy adapter used only by the test).
3. Rendered instruction snapshot: no LIBERO object words; the envelope numbers come from GT8.
4. `context.scene` for 12 objects ≤ 2,000 characters.
5. Request text size ≤ H1's at equal history, measured on re-rendered T04 requests.

**Live validation.**
- S2/S3: 0 schema errors in 6 calls; the share of `target` references; visible tokens per call (expected +40–80 for references, about +0.3–0.6 s at 0.007 s per visible token).
- Reasoning tokens against an interleaved H1 arm.

**Expected effect.**
- Instructions: −1.1k characters (obstacle paragraph) −0.3k (projection) +2.4k (geometry) +0.4k (envelope). Net about +1.4k static, cacheable.
- Per call: +300–500 tokens of scene facts, against H3's +2.3–2.5k.
- No model boxes, so no box-related repair calls: the 2 clearance-only map updates in T04 and the start-blocked loops these created (h1_collision F4).

**Risks.**
- A larger schema causes more invalid outputs. Mitigation: lint, plus the existing SchemaError feedback path (`agent.py:272-282`).
- Reasoning inflation. Mitigation: A/B in S2.

**Effort:** 1.5 days.

---

### GT10: plan/first-call object binding (IC7)

**Fixes:**
- IC7 (the merged first call must emit object noun phrases plus external cells);
- H3 V1 (part phrases used as SAM prompts; plan cells discarded, `live.py:79`; `seed_pixels` unused, `engine.py:137`).

**Files to create:**
- `geometry_tools/binding.py`
- `<NB>/tests/geometry_tools/test_binding.py`

**Files to modify:**
- `agent_speedup_v2_user_check_plan.py` `:11`: `EXTENSIONS = ("regions", "placement_goals", "objects")`. The schema merges in `variant_plan_schema()` (`:82-88`) and parses in `parse_plan()` (`:33-41`), the same mechanism as `regions`.
- `agent_speedup_v2_user_check_client.py`: one instruction sentence appended to the plan and replan instruction text (next to `SPATIAL_INSTRUCTIONS`, `:391-396`).
- If ws_protocol_context folds the plan into the first executive call (A4), the same `objects` array moves into that response.

**Plan extension `objects`**
```json
"objects":{"type":"array","maxItems":12,"items":{"type":"object","additionalProperties":false,
  "required":["noun","parts","x","w"],
  "properties":{"noun":{"type":"string","minLength":2,"maxLength":40},
                "parts":{"type":"array","maxItems":8,"items":{"type":"string","enum":["A","B","C","D","E","F","G","H"]}},
                "x":{"type":"integer","minimum":0,"maximum":64},
                "w":{"type":"integer","minimum":0,"maximum":64}}}}
```
Instruction: "objects lists every distinct visible object, task objects first. noun is the plain category a detector would use, with no colour, size, position or part words. parts lists this object's plan part ids ([] for other objects). x and w are its cells in images 2 and 4 (0 if not visible)."

**ObjectBinding (to ws_perception_service and GT3)**
```json
{"key":"b1","noun":"mug","plan_parts":["A","B"],"x_cell":41,"w_cell":0,"x_uv":null,
 "source":"plan_objects|derived_from_parts"}
```
- **Fallback for older plans.** Group parts by `object_words` / `same_object` (`agent_speedup_v2_types.py:329-345`); noun = the head words; x/w from the first part. This is flagged `derived_from_parts`, because head words can carry position words.
- **Perception use.** Unique nouns are SAM category prompts (one per category, all instances kept); cells pick instances (h3_live F7: 11/11, 3/3). Class-agnostic discovery (ws_perception_service) covers objects the plan omits (the T08 bottle).

**Veto removed / parameter added.** None.

**Depends on:** ws_protocol_context (where it is emitted); ws_perception_service (consumer).

**Offline tests:**
1. The 13 V1 drafts and 8 uc7.1 approved plans parse with and without `objects`.
2. The derivation reproduces nouns for T04/T08 plans.
3. Cell binding on H3's saved masks: 14/14 ambiguous frames resolve (replay of h3_live F7).

**Live validation.** S2/S3 and the sweep: binding rate (plan objects bound by call 1), `bind` edits per task.

**Expected effect.**
- Removes the 61–67 % of target-frame failures caused by compound or relational prompts (H3 V1; with ws_perception_service).
- About +60–150 visible tokens in the plan or first call (≈ +0.4–1.0 s once).

**Risks.**
- The model puts colour in the noun. Mitigation: perception strips nothing; it simply prompts more categories, which is harmless.
- Objects beyond 12. Mitigation: class-agnostic discovery.

**Effort:** 1 day.

---

### GT11: truth-scored geometry benchmark and pointing calibration

**Fixes:**
- README R8 and H3 §5.2 item 8 (a saved-observation, truth-scored suite as a development aid, not a launch gate);
- provides the *measured* σ terms that GT1, GT3 and GT5 need (user rule 2).

**Files to create (`<NB>/validation/geometry_bench/`):**
- `build_cache.py`: reads saved runs read-only and writes images, calibration and state per request into a cache. Sources: 7 H1-family, 2 H3 (with SAM masks), H2 v1, astranomy.
- `truth.py`: layout manifests (`output/layouts/*/layout_manifest.json`: obstacle collision bounds, table height, initial object poses) and `physical_truth.json`. An import-guard test fails if any policy module imports it.
- `bench_lift.py`: GT1 lifting and triangulation against truth. Fits `σ_px`, `a` and recall-derived `n_miss`; writes `measured_sigma.json`, which the frozen build loads with its sha recorded in the build digest.
- `bench_plausibility.py`, `bench_resolver.py` (palm clamp vs onsets), `bench_regions.py` (BDDL diagnostic).
- `pointing_smoke.py`: S1, ≤ 20 model calls, user-run, subscription only.

**Pass criteria (development aids; failures are reported, never block a launch):**
1. Two-view centre error ≤ 2 cm in ≥ 90 % of detected frames (prior 0.4–1.7 cm).
2. One-view centre ≤ 2 cm and top |err| ≤ 1 cm with the fused support (A.2: 0.7–1.2 cm, top −0.0 / +0.1 cm).
3. 2σ coverage of truth ≥ 90 % per axis after fitting. The naive model got 9/10 bottle and 0/11 can.
4. Palm clamp above onset in 4/4 T04 episodes.
5. Plausibility fires on the 3 known below-table first maps.
6. Per-decision geometry compute ≤ 30 ms: resolve, locate, ingest and overlays.

**Depends on:** GT1, GT3, GT5; ws_runner_measurement (shared replay suite, truth access policy).

**Live validation.** S1 itself.

**Expected effect.** Turns every σ in the build into a measured number, and catches regressions before paid runs (R8).

**Risks.**
- Little truth beyond initial poses. Mitigation: use frames up to each object's first motion (detectable from the trace).
- Only 2 obstacle scenes have SAM masks. Mitigation: run the perception service offline on cached frames (ws_perception_service, GPU, user-scheduled).

**Effort:** 2 days.

---

## 4. Shared interfaces (proposed shapes)

### 4.1 PerceptionSnapshot (ws_perception_service → GT2, GT3, GT5)
```json
{"version":"…","frame":123,"captured":{"sim_step":123,"sensor_time_s":6.15,"wall_s":…},
 "status":"ok|degraded|unavailable","errors":[],
 "instances":[{"instance_id":"p17","track_id":"t4|null","noun":"mug|null","score":0.91,
   "views":{"external":{"polygon_uv":[[u,v],…≤64],"bbox_uv":[u0,v0,u1,v1],"centroid_uv":[u,v],"area_px":3914,
                        "robot_occluded_fraction":0.0},
            "wrist":null},
   "xyz_hypothesis":{"lower":[…],"upper":[…],"sigma_m":[…],"method":"da3_affine|tri2|lifted"}|null,
   "binding":{"key":"b1","evidence":"x_cell_overlap","score":0.8}|null}],
 "support":{"z":0.437,"sigma":0.006,"method":"da3_plane_robot_masked","inlier_fraction":0.62}|null,
 "held_observation":{"lower_grip":[…],"upper_grip":[…],"sigma_m":[…]}|null}
```
Polygons, not full masks, cross the process boundary. The SceneMap lifts with GT1 when `xyz_hypothesis` is null, so all metric uncertainty logic lives in one place.

### 4.2 ObstacleSet (GT3 → ws_swept_planner)
```json
{"revision":14,"step":123,"supports":[{"id":"S1","z":0.437,"sigma":0.003}],
 "obstacles":[{"id":"o3","kind":"object","aabb":[[…],[…]],"footprint_xy":[[…]]|null,"sigma_m":[…],
               "inflate_m":[kσx,kσy,kσz],"state":"static",
               "contact":{"fingers":"forbidden|allowed","palm":"forbidden","links":"forbidden","held_object":"forbidden|allowed"},
               "source":"perception","observed_step":120}],
 "held":{"id":"o1","grip_lower":[…],"grip_upper":[…],"sigma_m":[…],"verified":"gap_band|unverified"}|null,
 "excluded":[{"id":"o9","why":"inside_robot"}],
 "k_sigma":2.0}
```
The planner uses exact distances to `aabb` + `inflate_m`. There is no lattice margin (IC4/M4). Contact flags are stage-specific (from `ContactIntent`).

### 4.3 RobotBody and PlannedSweep (ws_swept_planner → GT3, GT7)
- `robot_silhouette(joints, finger_q, camera) -> polygon list`: for `inside_robot`, the occlusion test and the plane-sweep robot mask.
- `body_points(joints, finger_q) -> (N,3)`.
- `PlannedSweep = {"stage":i, "path_xyz":[[…]], "status":"planned|executed|blocked", "blocked_by":"o3"|null, "min_clearance_m":0.012}`.

### 4.4 ResolvedTarget and ContactIntent (GT4 → ws_executor_control, ws_swept_planner)
- ResolvedTarget: as in GT4.
- ContactIntent: `{"stage":i, "fingers":["o3"], "held_object":["S1"], "palm":[], "links":[], "overrides":[{"id":"o4","part":"palm","source":"model_edit","reason":"…"}]}`.
- ws_executor_control uses:
  - `contact_intent` to classify a guarded stop (expected contact means arrival; otherwise `blocked_motion`);
  - `sigma_along_motion_m` for the IC1 slow zone (= kσ + braking distance).

### 4.5 Robot events (ws_executor_control → GT2, GT3)
```json
{"type":"grasp_closed","step":140,"anchor":"o1"|null,"settled_gap_mm":12.3,"verdict":"object_in_jaw|empty|unknown"}
{"type":"released","step":260,"gap_mm":78.9}
{"type":"guarded_stop","step":231,"part":"fingertips|palm|held_object|link","position_m":[…],"direction":[…],
 "expected":true,"anchor":"o2"|null,"penetration_sigma_m":0.002}
```

### 4.6 Protocol hooks (GT9/GT10 ↔ ws_protocol_context)
- Static `GEOMETRY_INSTRUCTIONS` + `envelope_prompt_text()` go in the cacheable prefix.
- The response-field schemas (`target`, `locate`, `scene_edits`) are merged into one fixed schema. ws_executor_control's `stop_on` field order is agreed there.
- `context.scene` is current-only, stubbed at history insertion.
- `objects` is emitted by the plan or the merged first call.
- `visible_frames()`: which image pairs the model can see. GT5 accepts only those.

### 4.7 Measurement (GT* → ws_runner_measurement)
- **Per call:** `{scene_revision, n_objects, n_3d, sources, support{z,σ,sources}, locate[{id,status,method,res_px,σ}], edits[{op,id,status}], warnings, facts_chars, overlay_ms}`.
- **Per segment:** `resolution` (GT4).
- **Coverage-map columns:**
  - reference share;
  - `reference_unresolved`;
  - first-call obstacle coverage (offline truth);
  - palm-on-target episodes;
  - regrasps;
  - locates;
  - binding rate.
- The build digest includes `geometry_tools/*` and `measured_sigma.json`.
- The real-calibration path (A5) builds `CameraModel` from a calibration file.

---

## 5. Sequencing (respects cross_critic §2 order; IC numbers in brackets)

1. **Now, offline only:** GT1, GT8, GT11 cache/benchmark, GT2 fusion maths, GT10 derivation. There is no live effect. This runs in parallel with ws_runner_measurement's step 1 (finish-scoring, give_up, cap; IC6), so every later effect is measured against finish-scored baselines.
2. **With ws_executor_control's sensing step:** the GT8 envelope text and GT4's palm clearance in report-only mode. This is IC1 step 1: guarded stops plus the envelope depth bound.
3. **"Geometry out of the LLM" (step 4), as one frozen change:** GT3, GT5, GT4, GT6, GT9, and GT7 (A/B'd).
   - **Preconditions:**
     - ws_swept_planner's exact distances and checked vertical-first escape are live [IC4, IC13]: more boxes near the gripper means more start/goal refusals otherwise;
     - ws_executor_control's guarded motion and continuation predicates are live [IC1, IC5]: approach references aim at contact.
   - Automatic perception obstacles go into the planner only after the escape exists [IC13]. ws_swept_planner's body sweep consumes `ObstacleSet` only together with this geometry [IC12].
   - The palm clamp (D3) switches from report-only to active here.
4. **After step 3 is live:** ws_executor_control's distance-scheduled approach uses `sigma_along_motion_m` [IC1 step 3]. Cruise speed scaled to swept clearance uses the ObstacleSet [IC2].
5. **Plan merge (A4, ws_protocol_context):** must carry `objects` [IC7]. If it lands before step 3, GT10's derivation keeps H3-style binding working.
6. **Throughout:**
   - resolution at stage entry, with revisions and frame ages [IC14];
   - compact, verbatim resolution and edit feedback [IC15];
   - overlays replace images [IC9].

---

## 6. Live validation (the user starts these; capped smokes are allowed)

Commands use `<NB>`'s launcher (flags owned by ws_runner_measurement; no per-task flags):

```bash
# S1: pointing calibration (no simulator; <= 20 subscription calls; writes measured pointing sigma)
<NB>/.venv/bin/python -I <NB>/validation/geometry_bench/pointing_smoke.py \
    --cache <NB>/validation/geometry_bench/cache --max-calls 20 --output <NB>/validation/geometry_bench/results/s1

# S2: T04 + can, capped at 6 executive calls (stop after the first grasp and lift), interleaved with an H1 arm
<NB>/.venv/bin/python <NB>/scripts/launch.py --task t04 --output <NB>/runs/s2_t04_geometry_001 --max-executive-calls 6
/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/run.sh   # H1 arm, same window, same cap via the runner's stop file

# S3: T08 + bottle, capped at 6 calls (bottle must appear without --vision-query)
<NB>/.venv/bin/python <NB>/scripts/launch.py --task t08_bottle --output <NB>/runs/s3_t08_geometry_001 --max-executive-calls 6
```

**Smoke pass signals:**
- 0 schema errors.
- Call-1 `scene` contains the added obstacle with kσ coverage of truth, scored offline by `bench_lift.py --from-run`.
- The first grasp uses a `target`.
- `palm_clearance_m` is logged.
- No palm-on-target episode in the first grasp.
- Reasoning tokens per call within the H1 arm's range (overlay A/B, D4).

**Sweep (one frozen build; LIBERO-10 + LIBERO-Goal + obstacle variants and relayouts), per-task coverage-map columns:**
- strict success at finish;
- palm-on-target and obstacle contact episodes by phase;
- first-call obstacle coverage;
- reference share and `reference_unresolved`;
- locate count and consistency;
- regrasps per object;
- calls per object;
- scene tokens per call;
- plausibility warnings.

---

## 7. User decisions

| Id | Question | Options | Recommendation |
|---|---|---|---|
| D1 | Raw absolute world targets during transition | (a) accept with telemetry; (b) remove once references cover ≥ 95 % of targets in a sweep; (c) remove now | (a) for the first sweep, then decide on (b) from the measured share |
| D2 | Confidence multiplier k for σ margins and gates | 2 (one-sided ≈ 97.7 %); 3; per-axis χ² | 2, recorded in the build digest |
| D3 | Palm-envelope depth clamp | (a) active, the model may allow palm contact per fragment; (b) report-only; (c) active only when the anchor footprint certainly overlaps | (b) in sequencing step 2, (a) in step 3, with regrasp telemetry; fall back to (b) if regrasps rise (rim − 31 mm is 5–14 mm shallower than H3's successful grasps) |
| D4 | Overlays | (a) both images: ids + boxes + last path; (b) ids only; (c) external only; (d) none | (a), confirmed by an (a) vs (d) A/B in S2 (reasoning tokens) |
| D5 | Native tool-call loop for `locate` | (a) none (references + next-call report); (b) function-tool round trip on demand | (a); revisit only after a capped transport smoke shows function calls plus strict text format plus encrypted reasoning replay work |
| D6 | May the model remove perception-detected obstacles? | (a) yes, with reason, kept in audit; (b) only "dispute" (σ inflated); (c) no | (a) |
| D7 | Object listing in the plan or first call | (a) every distinct visible object ≤ 12 with category nouns; (b) task objects only, rely on class-agnostic discovery | (a) (+60–150 tokens once) |
| D8 | DA3-free plane-sweep support height | (a) include behind a robot mask, support mask and peak-quality gate; (b) skip in v1 | (b): unmasked it failed 4/7 frames (A.3); rely on perception + contact + locate |

---

## Appendix A: offline checks run for this design

Scripts are in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/ws_geometry_tools_work/scripts/`. They ran under H1's `.venv` with `-I -B` and read saved artifacts only.

- **A.1 `geo_factors.py`** (H1 T04 and T08 request 2; truth obstacle boxes from the layout manifests):

  | | ext range | elevation | horizontal / plane-z | ext 1 px | cell σ | wrist 1 px | ray angle |
  |---|---|---|---|---|---|---|---|
  | can | 0.80–0.85 m | 34–38° | 1.27–1.49 | 1.30–1.37 mm | 2.40–2.53 cm | 1.0–1.2 mm | 45–48° |
  | bottle | 0.87–0.97 m | 40–47° | 0.93–1.17 | 1.41–1.57 mm | 2.60–2.90 cm | 0.86–1.23 mm | 64–81° |

- **A.2 `lift_eval.py` / `lift_eval_far.py`** (H3 saved SAM masks; key `extra_0`; support z = truth, σ 5 mm):

  | | can (T04, 11 frames) | bottle (T08, 10 frames) |
  |---|---|---|
  | one-view centre error | 1.2 cm | 0.7–0.8 cm |
  | with +1 cm plane bias | 1.6 cm | 1.6 cm |
  | top error, naive rule | +2.1 cm | +1.9 cm |
  | top error, far-edge rule | −0.0 cm | +0.1 cm |
  | two-view error | 0.9 cm (gap 2.1 cm) | 1.7 cm (gap 2.0) and 0.4 cm (gap 0.5) |
  | naive 2σ coverage | 0/11 (lateral 1.2 cm vs σ 0.2 cm) | 9/10 |

  The mask's u-extent matches the truth projection within 2–4 px (`mask_offset.py`).

- **A.3 `plane_sweep.py`** (no masks; median block-NCC):

  | Frame | Estimate error | Peak NCC | Result |
  |---|---|---|---|
  | T08 request 6 | −0.2 mm | 0.744 | ok |
  | T08 request 2 | +7.5 mm | 0.299 | ok |
  | T04 request 5 | −12.5 mm | 0.431 | ok |
  | T04 requests 2, 3, 8 | wrong | 0.10–0.11 (no peak) | failed: arm occludes the overlap at the start pose |
  | T08 request 4 | +130 mm | 0.458 | failed: locked onto another surface |

  About 3–5 s per frame, unoptimised.

- **A.4** H1 T04 request 4 text sizes: fields 21,568 characters; geometry 1,893; obstacle-map paragraph 1,126; camera/gripper paragraph 1,030; schema 3,232.
