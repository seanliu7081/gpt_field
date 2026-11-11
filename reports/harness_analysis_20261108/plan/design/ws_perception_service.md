# ws_perception_service — foundation_vision as an executor perception service

Design for one workstream of the implementation plan. Date 2026-11-08. Read-only work: nothing under
`/workspace/gpt_field` was edited; no simulator, model call or GPU job was run.

`<NEW>` below is the new build root: a fresh copy of H1 at `3cc84ac` (the frozen H1 repo itself stays untouched).
Its location is set by the overall plan. All new perception code lives under `<NEW>/src/gpt_field/perception_service/`.

Numbers marked **(est.)** are estimates that the benchmark (PS-12) or a live smoke must confirm. Unmarked numbers
come from the verified analysis (`reports/harness_analysis_20261108/`) or from code I read.

---

## 0. Summary

The service is a separate GPU process in `.venv-foundation-vision`. It keeps SAM 3.1 and DA3 loaded for a whole sweep
and reads raw frames from a shared-memory ring. It returns an object inventory with stable IDs, a measured support
plane, FK-rendered robot masks, metric primitives with uncertainty (σ) and a held-object model. The executor's router
and swept planner consume these; the model receives at most a compact fact block. It never stops the episode: on any
failure, the snapshot says `unavailable` and H1 behaviour continues.

What changes against H3 (`src/gpt_field/foundation_vision`):

| H3 problem (report) | This design |
|---|---|
| Plan part phrases are SAM prompts (V1; 87/143 target-frame failures, corrected count) | Category nouns from the plan, one prompt per string, masks de-duplicated, every instance kept (PS-4) |
| Two-instance categories dropped (engine.py:144-153) | Instance inventory; roles bound once by plan cells, then viewer-frame words, then an optional set-of-marks pick (PS-7) |
| Two-view + 0.9 mask-agreement gate: 0 usable centres in 22 requests (V4) | DA3-free epipolar ray-gap association; single-view hypotheses with along-ray σ; multi-view primitive fit (PS-6) |
| No metric plane; offline single-view used the *true* table height (verify_h3 missed issue 7) | Support plane measured by plane-sweep parallax (wrist motion; external↔wrist), DA3 RANSAC prior, contact anchors (PS-5) |
| DA3 robot depth off by decimetres; robot in the obstacle cloud (V5, H2 F4, K2 F3) | FK-rendered robot raster with depth replacement (PS-3) |
| Obstacle exists only via `--vision-query` (V2) | Class-agnostic discovery: corrected-DA3 height above the support, minus robot and known instances, refined by a SAM box exemplar (PS-8) |
| Advisory only, never reaches the router (V3) | Conservative world AABBs with roles and σ in the snapshot, for ws_swept_planner (PS-10) |
| Synchronous in the request path; ~2.9k tokens of stale summary per call (V6) | Asynchronous; decision frames are processed during the model call; compact facts only (PS-1, PS-11) |
| Wall-clock staleness in a paused simulator (V7) | Step/event-indexed validity, change detection on the static camera, re-localisation at the moment of use (PS-9) |
| Fail-closed (V8) | Fail-open everywhere (PS-1, PS-11) |
| Never validated against truth (V9) | Truth-scored offline benchmark from saved runs plus deterministic replay truth (PS-12) |

Total effort: **36.5 engineer-days**. A minimum viable slice of about 24 days is defined in §6.

---

## 1. Code facts this design relies on (lines checked)

| Fact | Where |
|---|---|
| H3 targets are plan `parts[].text` verbatim, plus operator extras | `src/gpt_field/foundation_vision/live.py:74-82` (`:79`), `:176-187`; wrapper flag `scripts/evaluate_foundation_vision_task.py:46-47` |
| Vision exception re-raised (fail-closed) | `live.py:257-262`; the executive catches only `SchemaError`: H1 `agent_speedup_v2_user_check_agent.py:266-298` |
| Same-step re-request blanks geometry as stale; wall-clock age | `live.py:247-252`, `:110-111`; `engine.py:23` (`max_age_s=1.0`), wrapper `5.0` (`evaluate_foundation_vision_task.py:45`) |
| Multi-candidate target dropped; `visible` needs every camera | `engine.py:144-153`, `:150-151` |
| Track continuity needs a frame gap ≤ max_age; interaction always unknown | `engine.py:205-206`, `:268-269` |
| Gates: support ≥ 2 views, 0.9 bidirectional agreement, zero contradictions | `geometry.py:32-37`, `:213-241`, `:236-238`; min/max bounds `:252-253` |
| One baseline-ratio DA3 scale per frame | `backends.py:315-354` (`:333`) |
| SAM: per-view `set_image`, per-prompt `set_text_prompt`, threshold 0.5, cache keyed on the RGB hash | `backends.py:139`, `:205-261` (`:217-223` cache key) |
| SAM 3.1 processor also offers `set_image_batch`, `add_geometric_prompt` (box exemplar), `set_confidence_threshold`; resolution 1008 | `artifacts/foundation_vision/vendor/sam3/sam3/model/sam3_image_processor.py:17`, `:76-111`, `:129-153`, `:173-182` |
| Vision routing disabled; routing source is model boxes | `evaluate_foundation_vision_task.py:102-109` |
| H1 obstacle map: model-written, ≤ 4 boxes, replaces the cache | H1 `agent_speedup_v2_user_check_agent_client.py:92-109`; schema `agent_speedup_v2_user_check_fragments.py:48-72` (`maxItems 4` at `:57`), `validate_route` `:90-109` |
| Map applied each answer; fragments get a snapshot of it | H1 `agent_speedup_v2_user_check_agent.py:313-314`, `:356-359` |
| Routing only on transit move stages, grip-site point | H1 `_fragments.py:293-310`; `_routing.py:25-52` (map), `:60-106` (route; `route_start_blocked` at `:106`) |
| Controller receives 7 arm joints with each observation | H1 `agent_speedup_v2_user_check_sim.py:73-80`, `:107-114` |
| Camera validator demands an ideal centred pinhole and exact Panda gripper dict | H1 `astranomy_agentic_geometry.py:79-128` (`:108-112`); wrist pose from state8 `:130-142` |
| Object head rule "`<object>: <part>`" | H1 `agent_speedup_v2_types.py:329-345` (`object_words`, `same_object`) |
| Plan cells: 8×8, 1-64 row-major from top-left, 0 = not visible | H1 `agent_speedup_v2_plan.py:57-58`; centre `mark_prompting.py:95-102` |
| Frozen viewer frame with `right_xy` / `away_xy` | H1 `agent_speedup_v2_user_check_spatial.py:158-212`, `:236-251` |
| Upright raster = PNG decoded and flipped `[:, ::-1]`; the transport PNG is rotated 180° | H1 `contact_skills.py:259-302`, `observation.py:24-45` |
| Evaluator-only oracle segmentation (owner/robot masks, metric depth), refused unless enabled | H1 `agent_speedup_v2_sim.py:213-220`, `:551-600` |
| Deterministic replay is bit-exact (549 actions, 550 observations) | H1 `validation/VERIFICATION.md` |
| K2 robot model: FK chain, collision hulls, merged visual envelopes (98 planes), ray entry | `h2_keypose2_snapshot/.../robot_geometry.py:60-81`, `:151-292`; FK `src/gpt_field/robodojo_direct_sim.py:122-224` |
| K2 silhouette self-filter skips envelopes that contain the camera | `h2_keypose2_snapshot/.../motion.py:96-208` |
| Env versions | `.venv` (controller): Python 3.12.14, numpy 2.5.3, cv2 4.13, no scipy. `.venv-foundation-vision`: Python 3.12.14, numpy 1.26.4, scipy 1.17.1, cv2 4.11, torch 2.10+cu128, sam3, depth_anything_3, trimesh |
| Panda asset hashes | `robot.xml` sha256 `263c0f9a…`, `panda_gripper.xml` `d9d68123…` (same as K2's `robot_kinematics.json`); LIBERO uses `MountedPanda` with `RethinkMount` (`third_party/LIBERO/libero/libero/envs/robots/mounted_panda.py:23-25`) |

Two newer observations, outside the verified analysis:
- `runs/foundation_harness_t04_can_obstacle_002` (K2, 2026-11-08) was user-stopped at 306 steps after 1,615 s. Its policy
  compute was 1,042 s, i.e. **3.41 s per step**, which corroborates K1. It saved 350 vision frames (RGB, K/E, DA3 depth,
  SAM masks) and per-step joints (`foundation_harness/events.jsonl`). That makes it the richest moving-camera benchmark
  source (PS-12).
- In that run "white mug" fired only 2× externally, against 114× for "textured mug". Adjective prompts are fragile, which
  supports head-noun fallbacks (PS-4).

---

## 2. Architecture

### 2.1 Processes

```
 sim worker (.venv-sim, py3.10)            perception service (.venv-foundation-vision, py3.12, 1 GPU)
   renders both cameras each step   ─raw─▶ FrameRing (POSIX shm, 16 slots)  ◀─reads latest/pinned─┐
   + state8 + q7 + qd7 + gripper cmd        provided by ws_runner_measurement (A2)                 │
        ▲ actions                                                                                   │
        │                                   unix socket, JSON lines                                 │
 controller (.venv, numpy only) ─────────── reset_episode / set_plan / observe / snapshot ────────▶│
   AgentFirstV2Policy + PerceptionBridge     notify / relocalize / bind_roles / get_masks           │
   router + swept planner (other ws)  ◀──── PerceptionSnapshot v1 (JSON), RelocalizeResult ─────────┘
```

- One service process per sweep, started by the sweep runner, with models resident across episodes. `reset_episode`
  clears all episode state but keeps the weights.
- The service imports nothing from the controller's `agent_speedup_*` modules. Shared code is either vendored (hash-pinned)
  or stdlib-only (`perception_service/protocol.py`, `queries.py`, `compact.py`), importable from both venvs.
- The controller-side client is stdlib plus numpy only, so it runs in `.venv`. This satisfies prerequisite (b), "one
  interpreter or a service", without touching the controller environment.

### 2.2 When perception runs

| Frame kind | Trigger | Work | Who waits |
|---|---|---|---|
| Frame 0 (reset) | `reset_episode`. Ideally the review-phase initial observation, whose reset hash equals execution frame 0 | Encode both views, DA3, robot raster, support sweep (ext↔wrist), discovery. Plan prompts are added on the cached SAM features when `set_plan` arrives | Nobody: it overlaps the plan call (14–52 s) |
| Decision | controller `observe(frame, "full", "decision")` just before each model call | Full refresh | Executor reads it after the model returns (≥ 7 s later), with `wait_for(timeout=0)` |
| Event | closure, release, contact notify, end of transit | Full minus DA3 | Nobody by default (non-blocking); optional barrier, user decision UD9 |
| Track | latest ring frame, when idle | Robot raster, change detection, held co-motion | Nobody |
| Relocalise | executor at stage entry, for instances the stage uses | Change check (ms); targeted SAM refit only if invalid | Executor, bounded by a timeout |

### 2.3 Latency budget (est., free RTX 4090; from H3 timings: SAM image ≈0.17 s per view, ≈0.03–0.07 s per prompt-view, DA3 0.22–0.34 s)

| Stage | Budget |
|---|---|
| SAM encode, 2 views (`set_image_batch`) | 0.25–0.34 s |
| Prompts: 4 strings × 2 views | 0.24–0.56 s |
| DA3, 2 views (decision frames only) | 0.22–0.34 s |
| Robot raster, 2 views (CPU) | ≤ 0.06 s |
| Support sweep (CPU, when a new baseline exists) | 0.1–0.3 s |
| Association, fit, discovery (CPU) | 0.1–0.3 s |
| **Full refresh** | **≈1.0–1.9 s**, inside the model call |
| Track refresh | ≈0.05–0.1 s |
| Relocalise: confirmed / targeted | ≤ 5 ms / 0.3–0.6 s |
| Preload (once per sweep), plus warm-up | 24.6 s measured, plus ≈3 s |

Added critical-path time per decision: **≈0 s** (H3: +1.1–1.3 s per call, +1.6 s on the first call). Per stage entry:
≤ 5 ms typical.

### 2.4 Fail-open rules

1. Every client call has a timeout and catches every exception. It returns a snapshot with `status="unavailable"` and a
   reason, and never raises into the episode.
2. A service-side job error yields `status="degraded"` for that frame (stage-level reason), and the service keeps running.
   Repeated CUDA OOM (2 in a row after cache clearing) marks the device unhealthy. Every later snapshot in the episode is
   then `unavailable`, and `health` reports it to the sweep runner.
3. If the socket breaks, the client marks perception unavailable for the rest of the episode. The sweep runner restarts the
   service before the next episode.
4. When perception is unavailable, the executor uses H1 behaviour (model boxes). The episode and its coverage-map cell
   record `perception_status`.

### 2.5 Determinism and replay

Consumption points are logged: `trial/perception/consumed.jsonl` holds `{step, consumer, snapshot_seq, sha256}`, and the
snapshots are stored under `trial/perception/snapshots/`. `RecordedPerceptionClient` serves exactly those snapshots, so
ws_runner_measurement's deterministic replay stays bit-exact without a GPU (R8). Live timing remains nondeterministic,
because asynchronous arrival is inherent; the decision-frame barrier makes decision-time consumption deterministic given
the snapshot content.

---

## 3. Components

### PS-1 Vendored package and service process

**Fixes.** V8 / L-F9 (fail-closed); V6 / F9 (preload per process, synchronous); improvement 6; prerequisite (b); G6 (no shared interpreter).

**Files (create).**
- `<NEW>/src/gpt_field/perception_service/__init__.py` (`VERSION = "perception-service-1"`)
- `<NEW>/src/gpt_field/perception_service/vendor/VENDOR_MANIFEST.json`
- `<NEW>/src/gpt_field/perception_service/vendor/foundation_vision/{__init__,types,backends,geometry}.py`. These are byte-identical
  copies with source sha256 `e3f38f16…`, `98eae4ec…`, `17a3d225…`, `d5fc4136…`. `engine.py`, `bridge.py` and `live.py` are not vendored:
  they are the parts being replaced.
- `<NEW>/src/gpt_field/perception_service/vendor/foundation_harness/{__init__.py,robot_geometry.py}`, from the K2 snapshot
  (`robot_geometry.py` `8fa39c48…`, `__init__.py` VERSION only).
- `<NEW>/src/gpt_field/perception_service/vendor/robodojo_direct_sim.py` (`6d2746cb…`). It sits at the relative position that
  `robot_geometry.py`'s `from ..robodojo_direct_sim import …` resolves to, so no byte changes are needed.
- `<NEW>/src/gpt_field/perception_service/{server.py,protocol.py,pipeline.py,health.py}`, `<NEW>/scripts/perception_service.py`, `<NEW>/scripts/check_vendor.py`.

**Interfaces.**
```
python -m gpt_field.perception_service.server --socket PATH --device cuda:N|auto --min-free-gib 10
   --sam-checkpoint FILE --da3-model DIR --robot-model NPZ --log-dir DIR
   [--persist] [--record-frames decision|all|none] [--no-da3]
```
Socket: Unix stream, UTF-8 JSON lines. Request `{"id":int,"op":str,"args":{...}}`, reply `{"id":int,"ok":bool,"result"|"error":...}`.
Ops are `hello, reset_episode, set_plan, observe, snapshot, relocalize, notify, bind_roles, get_masks, health, shutdown`
(schemas in §4.3).

**Algorithm.**
1. At start: verify `VENDOR_MANIFEST.json`; a mismatch refuses to start, which is visible before the sweep. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
   Pick the device: explicit `cuda:N`, or `auto` = the GPU with the largest `torch.cuda.mem_get_info()` free memory ≥ `--min-free-gib`.
   Record the GPU UUID and name.
2. Load `Sam31Segmenter(..., local_files_only=True)._load()` and `DA3DepthEstimator(...)._load()` (as in `evaluate_foundation_vision_task.py:70-86`).
3. Warm up: one 2-view SAM pass with 2 prompts and one DA3 pass on a bundled warm-up frame. Store the measured per-stage
   p50 latencies; they become the latency model the client uses for timeouts (PS-11).
4. Threads: one socket I/O thread and one processing thread with a priority queue
   (decision/event full > relocalise > track). Superseded track jobs are coalesced (latest wins); pinned frames are never dropped.
5. Job isolation: an exception produces a `degraded` snapshot. OOM → `torch.cuda.empty_cache()`, one retry with prompts processed
   per view, then the unhealthy flag (§2.4).
6. `reset_episode`: clear inventory, tracks, planes and caches; set `episode_id`; keep the models.
7. Artifacts: `log-dir/service.jsonl` (requests, latencies, errors) and `log-dir/frames/<episode>/<frame_seq>.npz`
   (RGB pair, masks, robot raster, DA3 float16, snapshot) for decision frames. Live runs extend the benchmark this way.

**Veto removed / parameter.** Removes the fail-closed path (`live.py:257-262`) and the 28k/56k-character abort (`live.py:168-169`, `:285-286`).
New parameters: `--min-free-gib` (measured SAM peak 5.98 GB + DA3 + margin; user decision UD1).

**Tests (offline).**
- `check_vendor.py` passes, and fails after a 1-byte edit.
- With fake segmenter/DA3 backends (injected): ops round-trip; a job exception yields `degraded` and the next job succeeds;
  a killed processing thread → `health.unhealthy`.
- Two episodes on one process: no state leak (instance IDs restart at 1, planes cleared).
- Each test < 5 s on CPU.

**Live validation.** First capped smoke: `hello` shows models warm before the plan call returns; 0 episode aborts caused by
perception; `service.jsonl` latency p95 per stage. Sweep: availability ≥ 99 % of decision frames; one preload per sweep.

**Expected effect.** Removes the 24.6 s preload per task (paid once per sweep) and the abort path.

**Risks.** GPU shared with training: OOM or slowdowns. Mitigation: pick a free GPU (UD1), fail open, record the GPU UUID.
Python < 3.13 `SharedMemory` attach registers the segment with the resource tracker, which unlinks it at exit; the reader
must call `resource_tracker.unregister`.

**Effort.** 3.0 d.

---

### PS-2 Frame ingestion and calibration adapter

**Fixes.** IC10 (raw frames in shared memory); G3 / R9 / A5 (the ideal-pinhole-only validator); improvement 4 (per-camera
geometry).

**Files (create).** `<NEW>/src/gpt_field/perception_service/frames.py`, `<NEW>/src/gpt_field/perception_service/calibration.py`.

**Interfaces.**
```python
class FrameRingReader:
    def __init__(self, shm_name: str): ...
    def latest(self) -> Frame | None
    def get(self, frame_seq: int) -> Frame | None          # None if overwritten
@dataclass(frozen=True)
class Frame:
    frame_seq: int; step: int; capture_time_s: float; flags: int
    state8: np.ndarray; q7: np.ndarray; qd7: np.ndarray; gripper_cmd: int
    rgb: dict[str, np.ndarray]          # 'external','wrist': HxWx3 uint8, upright raster
class Calibration:
    @classmethod
    def from_h1_geometry(cls, geometry: dict, state8) -> "Calibration"   # astranomy-agentic-panda-camera-v2 / robocasa v1
    @classmethod
    def from_file(cls, data: dict) -> "Calibration"                     # camera-calibration-1 (§4.2)
    def camera(self, view: str, state8, q7=None) -> CameraModel          # wrist pose = T_world_hand(state8) @ T_hand_cam
@dataclass(frozen=True)
class CameraModel:
    name: str; K: np.ndarray; dist: np.ndarray | None; T_cam_world: np.ndarray; width: int; height: int
    def rays(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]      # origins, unit directions (world)
    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]    # uv, depth
```

**Algorithm.**
- H1 geometry → OpenCV: `R_cw = diag(1,-1,-1) @ R_world_from_camera.T`, `t = -R_cw p`. This is the same conversion as
  `bridge.py:74-89`, re-implemented without importing `bridge.py`.
- `T_hand_cam` is derived once from frame 0 (state8 plus the wrist pose given in the geometry). Every later frame that carries
  geometry is checked to 1e-6.
- No equality checks on fovy, square images or principal point. Distortion, if present, uses `cv2.undistortPoints` and `cv2.projectPoints`.
- The reader uses the FrameRing v1 seqlock (§4.1), copies the RGB (≈0.3 ms) and validates the slot.

**Veto removed / parameter.** The service no longer depends on the ideal-pinhole validator (`astranomy_agentic_geometry.py:108-112`).
No new parameters.

**Tests (offline).**
- Round trip: project, then ray, on random points: < 1e-9 px for pinhole, < 1e-3 px with distortion.
- On the 22 H3 request frames, the true can centre projects inside the external SAM can mask, reproducing (87.9, 316.8) at T04 frame 0.
- A ring reader against a test writer, with torn writes injected: torn slots are never returned.

**Live validation.** Calibration hash in every snapshot; zero `calibration_mismatch` events in the smoke.

**Expected effect.** Enables real calibrations; per-frame wrist calibration from proprioception.

**Risks.** The writer may lag the controller. The snapshot carries `frame_seq` and `step`, and consumers check them.

**Effort.** 1.5 d.

---

### PS-3 FK-rendered robot raster with depth replacement

**Fixes.** V5 / E-F5 (self-surfaces block every path from the gripper); H2 F4 (DA3 robot depth −29.9 to +13.2 cm); K2 F3
(27–38k floating points; link4 pixels at the top edge); R3; H2 §4.4 item 2; improvement 4.

**Files (create).** `<NEW>/src/gpt_field/perception_service/robot_render.py`; `<NEW>/scripts/build_robot_model.py` (offline, service
venv, uses vendored `RobotCollisionModel(include_visual=True)` and the robosuite assets); `<NEW>/assets/robot_model_panda_v1.npz` plus `.json`
provenance (XML and mesh sha256s, RethinkMount included). If ws_geometry_tools provides the shared robot-model artifact (§6),
it is consumed instead.

**Interfaces.**
```python
class RobotRenderer:
    def __init__(self, model_npz: Path): ...
    def render(self, cam: CameraModel, state8, q7, *, payload: Primitive | None = None,
               dilation_px: int = 2) -> RobotRaster
@dataclass
class RobotRaster:
    label: np.ndarray      # int16 HxW: -1 none, 0..N-1 body index, 100 payload
    depth_m: np.ndarray    # float32 HxW camera z of first envelope entry, +inf elsewhere
    mask: np.ndarray       # bool HxW, label>=0 dilated by dilation_px
    edge: np.ndarray       # bool HxW dilation ring (treated as unknown, never as scene)
def replace_robot_depth(depth: np.ndarray, raster: RobotRaster) -> np.ndarray
```

**Algorithm.**
1. `PandaKinematics.bind(state8, q7)` recovers `world_from_base` from the measured grip pose. Per body, the visual envelope
   (hull vertices plus 98 supporting planes, from the npz) is posed by FK; fingers use `state8[6:8]`.
2. Per view and body: project the vertices, take `cv2.convexHull`, then `cv2.fillConvexPoly`. Inside the polygon, compute the
   per-pixel ray entry distance (the vectorised form of `ray_entry_distances`) and turn it into camera z. Keep the per-pixel
   minimum across bodies.
3. Skip envelopes that contain the camera centre (as K2 `motion.py` does). In the wrist view, the hand and fingers are rigid
   relative to the camera. A precomputed **wrist self-mask atlas**, indexed by finger opening (41 samples over 0–0.04 m, rendered
   once from the same model), gives their silhouette and depth exactly, with no per-frame cost.
4. A held payload (PS-9) is rendered the same way with label 100.
5. Depth replacement: DA3 depth on `mask` pixels is replaced by `depth_m`; `edge` pixels are set to NaN.

**Veto removed / parameter.** Replaces H2's 3 mm / 6 mm depth margins and K2's "entry − 6 mm" depth test. Robot pixels are removed
whatever DA3 says. `dilation_px` is derived from the image-time skew of the joint reading: 0 in simulation, because joints and RGB
come from the same cached observation (`_user_check_sim.py:73-80`), plus 1–2 px for silhouette rounding. On a real robot it is
`ceil(v_img·Δt)+1`, from measured latency.

**Tests (offline).**
- Synthetic: render at known joints; projected hull points ⊂ mask.
- K2 run 002 (350 frames with joints): the raster covers ≥ 95 % of the 29 saved external "robot"-query SAM masks (target).
- With replay truth (PS-12): robot recall ≥ 0.98 and precision ≥ 0.85 per view (targets).
- On K2 saved DA3: floating points above z 0.60 m over the table go from 27–38k to < 500 per frame (target, K2 F3).
- Timing ≤ 60 ms per view pair on CPU.

**Live validation.** The smoke's recorded frames are scored post hoc against replay truth with the same targets. Sweep:
robot-pixel false obstacles = 0 in the obstacle list.

**Expected effect.** Removes self-blocking, which was 4/4 blocking probes on the test path (E-F5), and the phantom link5/link6
obstacles (K2 F3). It is a prerequisite for discovery (PS-8) and for any free-space use.

**Risks.**
- Convex envelopes over-cover concave regions, which hides a few scene pixels near the arm; these are reported as `occluded_by_robot`.
- The RethinkMount and any unmodelled visual geoms show up as static obstacles. Mitigation: include the mount; benchmark residuals.
- Real-robot calibration error widens the masks; measure it.

**Effort.** 2.5 d.

---

### PS-4 Category prompts, SAM detection and mask de-duplication

**Fixes.** V1 / E-F1 / L-F1 (part phrases as prompts: 6 prompts with 0 candidates, 87/143 failures); K2 F8 / H2 F9 (grammar
seeds; "white mug" 0 candidates); improvement 2.

**Files (create).** `<NEW>/src/gpt_field/perception_service/queries.py` (stdlib only), `<NEW>/src/gpt_field/perception_service/detect.py`.

**Interfaces.**
```python
@dataclass(frozen=True)
class CategoryQuery:
    category_id: str; primary: str; head: str | None
    plan_parts: tuple[str, ...]; binding_hints: tuple[str, ...]; move_prior: str
@dataclass(frozen=True)
class PartQuery: part_id: str; category_id: str; noun: str | None   # None = whole object
def derive_queries(plan: dict) -> tuple[list[CategoryQuery], list[PartQuery]]
class Detector:
    def encode(self, frame: Frame) -> None                     # cached by RGB sha256
    def text(self, view: str, prompt: str) -> list[Candidate]  # SAM threshold 0.5 (upstream default)
    def box(self, view: str, xyxy) -> list[Candidate]          # add_geometric_prompt exemplar
@dataclass
class Candidate:
    view: str; mask: np.ndarray; score: float; prompts: dict[str, float]
    bbox: tuple[int, int, int, int]; pixels: int; truncated: bool; parent: int | None
def dedupe(cands: list[Candidate], robot_mask: np.ndarray) -> list[Candidate]
```

**Algorithm.**
1. Group plan parts into objects with `object_words` / `same_object` (the same rule as H1 `agent_speedup_v2_types.py:329-345`).
2. Strip a fixed, task-independent list of spatial and ordinal words (`left right front back behind near nearer nearest far
   farther farthest top bottom upper lower middle center centre first second third other leftmost rightmost closest`).
   Removed words become `binding_hints`. Conjunctions inside modifiers are kept ("yellow and white mug" stays one phrase).
3. `primary` = the remaining phrase. `head` = its last word when the phrase has more than one word ("moka pot" → "pot",
   "white mug" → "mug"). Identical strings across objects are run once.
4. Part noun = the words after ':' up to the first preposition (`on with of at in near beside toward towards around under over
   along from by to into onto`), spatial words stripped, last word. Whole-object words are not special-cased: a part mask
   covering ≥ 80 % of its instance is treated as "whole".
5. `encode()` both views with `set_image_batch`. For each prompt string and view, run `set_text_prompt`, keep masks with
   score > 0.5 (upstream default) and record `(prompt, score)`. The head prompt always runs; it costs ≈0.05 s per prompt-view.
6. De-duplicate: merge two masks with IoU > 0.5 (union-find; keep the highest-scoring mask and every label). Rule: each pixel
   shows one surface, so masks of distinct opaque objects are disjoint apart from boundary error; IoU > 0.5 therefore means the
   same object. A mask contained ≥ 90 % in a larger mask (smaller than half its size) becomes that mask's `parent`-linked part.
   Candidates whose mask is ≥ 50 % inside the robot raster are dropped as robot.
7. At reset, before the plan exists, the images are encoded and discovery runs (PS-8). The plan prompts then cost only
   ≈0.05 s per prompt-view on the cached features.

**Veto removed / parameter.** Removes the multi-instance drop (`engine.py:144-153`), the all-views visibility rule (`:150-151`), plan
text as prompts (`live.py:79`) and `--vision-query` (`live.py:176-187`). No new thresholds: 0.5 is SAM's default; the IoU 0.5 rule
comes from surface uniqueness.

**Tests (offline).**
- `derive_queries` on all saved approved plans (T04, T08, stove-off, drawer 001/003, H3, K2) and on synthetic plans for the 21
  registered sentences. Expected: T04 → {"white mug","mug","yellow and white mug","plate"} with hints {left, right};
  T08 → {"moka pot","pot","stove"} with hints {near, right}. Never split "yellow and white"; 0 per-task strings in code
  (enforced by a grep test for task nouns).
- GPU replay on the 22 H3 frames and the K2 frames: external whole-object recall for task objects ≥ 0.9 against truth masks
  (target, est. from the 21/22 simple-noun rate); no retained pair with IoU > 0.5.
- Timing ≤ 0.9 s for 2 views × 4 prompts.

**Live validation.** Smoke: plan objects detected in the frame-0 external view; candidates per category logged. Sweep:
per-category recall from post-hoc replay truth, shown in the coverage map.

**Expected effect.** Detection of plan objects approaching the simple-noun rate (`can` 11/11, `bottle` 10/11 external) (est.).
Removes the 61 % of target-frame failures caused by prompt form. Prompts fall from 6–7 to about 3–4 per scene.

**Risks.**
- Head nouns can be too generic ("pot" also matches a stove pot). Binding (PS-7) resolves which instance is which; extra
  instances are kept and become obstacles, which is correct.
- Recall on LIBERO-Goal and LIBERO-10 nouns is unmeasured; the benchmark measures it before any sweep.

**Effort.** 2.0 d.

---

### PS-5 Support-plane estimation

**Fixes.** verify_h3 missed issue 7 (plane height is the crux of single-view localisation: ±2 cm plane → 1.3–3.3 cm centres);
E-F4 / H2 F1 / K2 F4 (DA3 table bias: wrist +18–20 %, external +2.7–3.5 % with a range tilt of +3.7 % at 0.65 m to −4.7 % at
1.3 m); C10; improvement 4 (correct depth per camera).

**Files (create).** `<NEW>/src/gpt_field/perception_service/support.py`.

**Interfaces.**
```python
@dataclass
class SupportEvidence:
    kind: Literal["sweep_wrist_motion","sweep_ext_wrist","da3_external_ransac","contact_anchor"]
    height_m: float; sigma_m: float; n_patches: int; frames: tuple[int, ...]; step: int
@dataclass
class SupportPlane:
    id: str; normal: tuple[float, float, float]; height_m: float; sigma_m: float
    extent_xy: list[tuple[float, float]]; sources: list[SupportEvidence]; observed_at_step: int
class SupportEstimator:
    def update(self, frames: list[Frame], cams, rasters, instance_masks, da3: dict | None) -> list[SupportPlane]
    def add_contact_anchor(self, step: int, z_m: float, sigma_m: float, xy) -> None
    def depth_affine(self, view: str, da3_depth, cam) -> tuple[np.ndarray, dict]   # per-view a·z+b
```

**Algorithm.**
1. **Plane sweep** for a view pair (A, B) with known K and poses. Heights h run from base z − 0.30 m to base z + 0.70 m (the Panda reach
   envelope from FK; no task data): first in 5 mm steps, then ±5 mm in 1 mm steps around the minimum. The plane z = h induces
   the homography `H_h = K_A (R_AB − t_AB nᵀ/d_B) K_B⁻¹`.
   - Warp B into A with `cv2.warpPerspective` and compute per-patch ZNCC on a 16 px grid with `cv2.boxFilter`.
   - Mask out robot pixels (raster, both views), known instance masks (dilated 2 px) and invalid pixels.
   - Each patch gets its best height `h_p` (parabolic sub-step) and its peak sharpness.
2. **Pairs.**
   - `sweep_ext_wrist`: external and wrist at the same step. Available at step 0; both pixel footprints are ≈1 mm at the home pose (est.).
   - `sweep_wrist_motion`: two processed wrist frames with baseline ≥ 1 cm, so it is available after the model's first motion.
     Its precision is `σ_h ≈ z²·σ_px/(f·b)`: about 2–6 mm for b = 2–5 cm at z ≈ 0.35 m, f = 333.6 px, σ_px = 0.3–0.5 (est.).
3. **Clustering.** 1-D mean-shift over the `h_p`, with per-patch bandwidth equal to that patch's `σ_h`. The dominant cluster is the
   primary support. Further clusters whose area is at least one gripper footprint (0.08 m × 0.08 m, robot facts) are secondary
   supports (stove top, cabinet top). For each cluster, RANSAC a plane over the patch 3-D points (normal free, horizontal prior).
   `height_m` is the weighted median; `σ = 1.4826·MAD/√n ⊕` the calibration term.
4. **DA3 prior.** RANSAC plane on robot-masked external DA3, with σ = max(MAD, the tilt effect at the workspace range); about 1.5–3 cm by default (est.).
5. **Contact anchors.** `notify(contact, support=true)` supplies the fingertip/pad lowest point from FK, σ = 2 mm.
6. **Fusion.** Per cluster, inverse-variance weighting; a source more than 3·√(σᵢ²+σ_f²) from the fused value is excluded and logged.
7. **Per-view DA3 affine.** Fit `z_true = a·z_DA3 + b` (Huber) on support inliers whose true depth is ray ∩ fused plane, and apply it to
   that view. On the binned external errors this leaves ≤ ≈1 cm residual from 0.65 to 1.3 m (est.; the binned DA3-vs-true depths are
   near-linear). Robot pixels are excluded (DA3 is decimetres off there), so robot pixels are **not** used as DA3 anchors.

**Support source assessment (the user's options):**

| Option | Expected plane-height accuracy | Verdict |
|---|---|---|
| DA3 external RANSAC, affine anchored on the external view itself | ≈1.5–3 cm: keeps the external +2.7–3.5 % bias and ±4 % tilt (est.) | Prior only |
| Robot self-pixel anchors for DA3 | Not usable: DA3 is +8 to −30 cm off on the robot (H2 F4) | Reject as an anchor. Robot pixels are used for depth replacement and as a calibration check (projected silhouette vs SAM "robot" mask) |
| Wrist-motion parallax (plane sweep) | ≈2–6 mm after ≥ 2–5 cm of wrist motion (est.) | Primary after the first motion |
| External↔wrist plane sweep, same step | ≈1–5 mm if the support is textured (est., unverified) | Primary at step 0 |
| Contact anchor | ≈2 mm, only after a support contact | Refinement |

User decision UD2: whether the harness may make a short "look" motion (≤ 5 cm wrist translation, ≈1 s, no model call) when the
step-0 sweep fails (σ above a target such as 1 cm).

**Veto removed / parameter.** Replaces the truth-height assumption of the offline analysis and the DA3-only scale (`backends.py:333`).
Parameters, all derived: sweep range (robot reach), per-patch bandwidth (σ_h formula), secondary-support minimum area (gripper footprint).
Patch size 16 px is a modelling choice, ablated on the benchmark.

**Tests (offline).**
- Synthetic: a textured plane at a known h, two synthetic cameras: |err| ≤ 1 mm, robust to 30 % occluders.
- Saved request frames (T04/T08/drawer/stove-off, RGB + calibration only), `sweep_ext_wrist` vs truth support top (T04
  0.43676 m): |err| ≤ 5 mm on ≥ 80 % of frames (target). `sweep_wrist_motion` on consecutive K2 002 frames: ≤ 5 mm (target).
- σ calibration: |err| ≤ 2σ in ≥ 90 % of frames. External DA3 affine residual on the support ≤ 1 cm (target).

**Live validation.** Smoke: `supports[0].sigma_m` at frame 0 and after the first fragment; post-hoc truth error. Sweep: per task,
support σ and error; fraction of tasks where the step-0 sweep failed.

**Expected effect.** Single-view localisation is near the true-plane case: can 1.3–1.5 cm, bottle 0.6–0.7 cm (verify_h3), instead of
1.3–3.3 cm at ±2 cm plane error (est.). Gives metric DA3 per view for discovery and height fields.

**Risks.**
- Textureless or specular supports make the sweep fail; it then falls back to the DA3 prior with σ ≈2–3 cm, conservative boxes grow,
  and route refusals rise (IC13, mitigated by ws_swept_planner's escape).
- Moving shadows from the arm add matching noise; ZNCC is gain-invariant, and patches near the robot raster are masked.

**Effort.** 3.0 d.

---

### PS-6 Association and 3-D estimation (single view, triangulation, multi-view primitive fit)

**Fixes.** V4 / E-F2 / E-F3 / L-F1 (0.9 gate, two-view rule; correct detections rejected); E-F7 (min/max extents); R1, C3
(measured geometry instead of LLM metres); improvement 3.

**Files (create).** `<NEW>/src/gpt_field/perception_service/localize.py`, `<NEW>/src/gpt_field/perception_service/primitives.py`.

**Interfaces.**
```python
@dataclass
class ViewObservation:
    view: str; frame_seq: int; step: int; mask: np.ndarray; cam: CameraModel
    truncated: bool; dontcare: np.ndarray        # robot raster | other instances (dilated)
@dataclass
class Primitive:
    kind: Literal["cylinder","box"]; center_m: np.ndarray; radius_m: float | None
    half_extents_m: np.ndarray | None; yaw_rad: float; z_min_m: float; z_max_m: float
    cov_center: np.ndarray; sigma_extent_m: np.ndarray
    source: Literal["single_view_support_contact","ray_pair_triangulation",
                    "multiview_silhouette_fit","held_rigid","depth_component"]
    views_used: list[tuple[str, int]]
def associate(ext: list[ViewObservation], wrist: list[ViewObservation], cams, sigma_cal_rad) -> list[Pair]
def single_view_hypothesis(obs: ViewObservation, support: SupportPlane, local_support=None) -> Primitive
def fit_primitive(obs: list[ViewObservation], prior: Primitive, support: SupportPlane) -> Primitive
def conservative_aabb(p: Primitive, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]
```

**Algorithm.**
1. **Ray-gap association** (DA3-free).
   - Take the ray through each candidate's 1 px eroded mask centroid. For each cross-view pair at the same step, compute the
     closest-approach distance g and the midpoint X.
   - The pair is plausible if g ≤ max(W_a, W_b) + 3·σ_cal·range, where W is each mask's physical lateral width at X's range.
     Two lines that both pass through a convex body are closer than its diameter. X must also lie in
     [support − 2σ_s, support + reach] and project inside both dilated bboxes.
   - Cost = g / gate. Hungarian assignment; category labels act as a soft prior.
   - A border-truncated mask still gives a ray through the object (its visible centroid lies inside the object's silhouette), so it
     is associated normally, but its width is not used for the gate; the other view's width is.
   - Observed separation: same instance 0.5–2.1 cm vs wrong instance 22–49 cm (n = 10).
2. **Single-view support-contact hypothesis** (external, or wrist).
   - Lower mask boundary per column: if the pixel below belongs to the robot or another instance, the contact is hidden and the
     ray-interval fallback is used.
   - Intersect those rays with the local support (the fused plane; or a secondary support; or the corrected-DA3 height of the
     strip just below the contact). The result is the footprint front arc.
   - Lateral width W = p2–p98 of the arc perpendicular to the view ray. Depth along the ray: D ≈ W with σ_D = W/2.
     Centre = front-arc midpoint + horizontal ray direction · D/2.
   - Top: the upper-boundary ray evaluated at the footprint front (a conservative upper bound) and at the centre (estimate).
   - σ_along = √((W/4)² + (σ_s/tan α)²), with α the ray depression angle; σ_lateral = 1 px·range/f ⊕ boundary noise.
   - Ray-interval fallback (hidden contact): centre on the centroid ray between the occluder depth and the support intersection,
     σ = half the interval.
3. **Multi-view primitive fit** (render and compare).
   - Parameters: cylinder (x, y, r, H) or yaw box (x, y, h_x, h_y, ψ, H); z_min = local support height.
   - Cost = Σ_v w_v (1 − IoU(render_v(θ), mask_v)) over every static observation since the last invalidation, on 256² rasters.
     `dontcare` pixels are excluded; truncated masks are scored inside the image only; w_v is lower for partial masks.
   - Nelder–Mead (scipy, service venv), ≤ 300 evaluations from the best of the single-view and pair estimates.
     Cylinder and box are both fitted; the lower cost wins, with one-parameter-per-view penalties.
   - σ_θ = the parameter step whose IoU drop equals the IoU change of a ±1 px boundary shift (≈ perimeter/area), combined over views.
     With ≥ 3 views, the leave-one-view-out spread, ⊕ plane-induced σ.
4. **Robust extents.** Masks are eroded 1 px for contact and fit. Lateral extents use p2/p98. The conservative AABB is the primitive's
   world AABB inflated by k·σ per axis, with k = 2 (reported; the consumer adds robot clearance on top).

**Veto removed / parameter.** Removes the 0.9 bidirectional agreement (`geometry.py:36`, `:228-229`, `:240-241`), the two-view support
rule (`:34`, `:236-238`) and min/max bounds (`:252-253`). New parameters: the association gate, derived from object width and
calibration σ; k = 2 (a stated confidence convention, ≈95 % if Gaussian).

**Tests (offline).**
- Synthetic: exact masks → centre ≤ 3 mm, extent ≤ 5 mm; a 50 %-coverage wrist mask → centre ≤ 1 cm.
- H3 saved masks:
  - association accepts the 3 same-instance pairs (T04 f0 can; T08 f243, f328 bottle) and rejects the 7 wrong-instance pairs;
  - pair centres ≤ 1.7 cm xy (the analysis got 0.4–1.7 cm), z ≤ 2.5 cm;
  - single view with the PS-5 plane: can ≤ 1.5 cm, bottle ≤ 0.7 cm when the plane error ≤ 5 mm.
- K2 002 moving frames: the multi-view fit error is non-increasing with the number of views (target p90 ≤ 1.5 cm for compact objects).
- Coverage of the truth AABB by the conservative AABB ≥ 0.95, and σ calibration (|err| ≤ 2σ in ≥ 90 %).

**Live validation.** Smoke: shadow-mode first boxes vs layout truth (post hoc): full coverage of the can with conservative AABBs.
Sweep: first-obstacle-box coverage per task (the cross_critic box_truth metric), centre-error distribution by source.

**Expected effect.** First boxes and targets within about 1–3 cm (C3 estimate), against 5–12 cm model-written errors.
Single-view geometry exists in ≥ 21/22 frames, against 3/22 for two-view triangulation alone (verify_h3).

**Risks.**
- Epipolar ambiguity for large adjacent objects: the gate grows with W. Mitigation: cell and category priors; the benchmark
  confusion matrix.
- Partial SAM masks (wrist can 53 %) under-fit height. Mitigation: per-view weights, and the external upper-boundary bound for
  the conservative top.

**Effort.** 4.0 d.

---

### PS-7 Instance inventory, identity, role binding and parts

**Fixes.** E-F1 / L-F7 (plan cells resolve 11/11 and 3/3 ambiguous frames and are consistent with 31/31 single detections);
E-F8 (no IDs ever assigned); K2 F8 / H2 F9 (identity costs about 1 call in 4); H2 F5 / K2 F5 (wrong auto-bound wrist instance);
H2 §4.4 item 12; improvement 2.

**Files (create).** `<NEW>/src/gpt_field/perception_service/inventory.py`, `<NEW>/src/gpt_field/perception_service/binding.py`.

**Interfaces.**
```python
class Inventory:
    def update(self, frame: Frame, cands: dict[str, list[Candidate]], rasters, supports) -> list[Event]
    def instances(self) -> list[Instance]
class Binder:
    def bind(self, plan: dict, inventory: Inventory, plan_frame: Frame, view_frame: dict) -> dict[str, Binding]
    def apply_model_bind(self, picks: dict[str, int], marks: list[Mark]) -> dict[str, Binding]
@dataclass
class Binding:
    part_id: str; instance_id: str | None; part: str              # "whole" | part noun
    source: Literal["plan_cell","direction_hint","set_of_marks","model_bind","unbound"]
    margin: float; alternatives: list[str]; part_center_m: list | None; part_sigma_m: float | None
```

**Algorithm.**
1. **Frame update.**
   - Predict each instance's silhouette in the new views: the world primitive if static, or hand-attached if held (PS-9).
   - Cost = 1 − IoU(predicted, candidate), plus a category-label mismatch term and an RGB-histogram χ² tie-break. Hungarian assignment.
   - Unmatched candidates → new instances; IDs `inst_NNN` are never reused within an episode.
   - An instance predicted visible (≥ 50 % unoccluded) but missed for n consecutive frames → `possibly_moved`. n = the smallest
     integer with (1 − r)ⁿ < 1 %, where r is the benchmark-measured per-frame recall (r ≥ 0.9 → n = 2). It stays in the inventory
     as an obstacle with inflated σ; obstacles are never silently deleted.
2. **Role binding**, once, when `set_plan` arrives. Uses the plan frame, which is frame 0 or the frame the plan was written on.
   - Cell score(O, I) = Σ over parts p of O of |M_I^ext ∩ cell(p.x)| + |M_I^wrist ∩ cell(p.w)|, normalised by cell area. A cell is
     64 × 64 px at 512 px, (row, col) = divmod(c − 1, 8); cell 0 is ignored.
   - Direction hints: rank the category's instances along the frozen `right_xy` (left/right) or `away_xy` (front/behind;
     near = front, far = behind). The extreme instance matching the word scores 1.
   - Assignment: Hungarian over plan objects × same-category instances. Order of evidence: cell score, then direction hint, then
     primary-prompt hit. `margin` = best − second best.
   - Margin 0, or a cell/hint conflict, → `unbound` plus `marks` (numbered anchors and bboxes on the current external image) for a
     one-time set-of-marks pick. ws_protocol_context renders the marks; the reply arrives via `bind_roles`.
3. **Roles of instances.**
   - `target_object` = an object in some step's `obj`; `reference_object` = a `ref`; `fixture` = parts with `move: fixed`;
     `plan_object` = in the plan but in no step; `non_plan` = everything else (distractor or obstacle).
   - The **current** stage target is not known to perception; the executor passes `exclude_instance_ids` per stage.
4. **Parts.**
   - The part noun is prompted on views where the bound instance is visible. Keep masks ≥ 90 % inside the instance mask dilated by
     the part margin (handles protrude: dilation = 0.25·W in pixels).
   - Several hits → choose with the part's own cells.
   - 3-D: ray pair if seen in both views; otherwise ray ∩ the instance's conservative primitive, interval midpoint, σ = half the interval.

**Veto removed / parameter.** Removes wall-clock track expiry (`engine.py:205-206`) and the unique-candidate auto-binding that conflicts
with the other view. n comes from measured recall; the part dilation from measured width.

**Tests (offline).**
- Cell binding on the H3 frames: C (cell 41) → the left-plate mask [0, 308, 58, 375] in 11/11; T08 D (cell 38) in 3/3.
- Direction hints: "left plate" → minimum along `right_xy`.
- Every saved plan: bound instance = replay-truth owner for 100 % of objects with a nonzero cell (target).
- K2 002 (350 frames): ≤ 1 ID switch per object per 100 frames (target).
- Handle part: the T04 "B / E" handle found inside the bound mug in ≥ 70 % of frames where the handle is visible (target; handle
  prompts were 0/4 correct as global prompts).

**Live validation.** Smoke: bindings A–F at plan time vs post-hoc truth; zero identity-only model calls. Sweep: binding accuracy per
task; set-of-marks requests per task; identity-caused calls (target 0).

**Expected effect.** Removes all 14 ambiguity frames and the identity calls: about −2 to −3 calls (−20 to −30 s) per cluttered task (H2 F9, est.).

**Risks.**
- Plan cells are written at plan time, at 64 px resolution, external view mostly (verify_h3 L-F7 limits).
- If A4 folds the plan into the first executive call, binding happens only after that call (IC7).

**Effort.** 3.0 d.

---

### PS-8 Class-agnostic obstacle discovery and height field

**Fixes.** V2 / L-F5 (operator-named obstacle; the T08 plan never names the bottle; red mug never queried); K2 F2 (non-role
objects non-blocking); H2 §4.4 item 12; improvement 5; C2 (measured height map under the palm footprint).

**Files (create).** `<NEW>/src/gpt_field/perception_service/discovery.py`, `<NEW>/src/gpt_field/perception_service/heightfield.py`.

**Interfaces.**
```python
def discover(frame: Frame, cam_ext: CameraModel, da3_ext_corrected, raster_ext, known_masks,
             supports, detector: Detector) -> list[Candidate]
class HeightField:     # §4.9
    def update(self, frame, cams, depths_corrected, rasters, instances, supports) -> None
    def export(self) -> HeightFieldRef     # npz path + header in the snapshot
```

**Algorithm.**
1. External DA3, affine-corrected per view (PS-5), with robot pixels replaced (PS-3), gives world points and height above the
   local support.
2. Fit a robust quadratic surface in (u, v) to the support inliers to absorb residual tilt; `h_rel` = height − surface.
   `σ_res` = 1.4826·MAD of the inliers, measured per frame.
3. Candidate pixels: `h_rel > 3σ_res`, not robot (dilated), not a known instance mask (dilated 2 px), valid depth.
   Then morphological open (1 px) and 8-connected components. Keep a component if its lateral extent ≥ the 8 mm finger-pad
   width (robot facts) and its height extent ≥ 3σ_res.
4. For each component, a SAM box-exemplar prompt (`add_geometric_prompt`, positive) on the cached external features. Take the
   returned mask that contains ≥ 50 % of the component; otherwise keep the component pixels. The result is an instance with category
   `unknown_object`, source `sam_box_exemplar` or `depth_component`, and role `non_plan`.
5. Wrist-view discovery (phase 2): patches from `sweep_wrist_motion` clustered above the support, not robot, not known, then a
   SAM box exemplar on the wrist image.
6. Optional SAM generic prompt sweep (`discovery.generic_prompts`, default empty); the benchmark decides (UD4).
7. Height field: a 5 mm grid over the observed support extent holding z_max from the corrected external (and wrist) depth and
   the instance primitives. It carries per-cell σ, an `observed` flag (direct ray hit) and a `shadow` flag (behind an observed
   surface along the external ray, unknown up to the ray height).

**Veto removed / parameter.** Removes `--vision-query`. Parameters: 3σ_res (measured noise), 8 mm minimum extent (robot pad width),
quadratic order 2 (a modelling choice).

**Tests (offline).**
- K2 saved DA3 frames, T04 layout: can found in ≥ 90 % of frames where ≥ 200 px are visible; red mug (not a plan object) ≥ 90 %
  (targets); 0 components on robot pixels; ≤ 1 table-texture false component per frame.
- GPU replay with DA3 on H1/H3 T08 frames: bottle found ≥ 90 % (target).
- Height field vs replay-truth depth: error under the T04 mugs' footprint ≤ 1 cm p90 (target).

**Live validation.** Smoke: the can appears as `unknown_object` or `can` without any operator input. Sweep: non-plan obstacle recall
by post-hoc truth; false obstacles per frame; route refusals attributable to discovered boxes (IC13).

**Expected effect.** Restores zero-shot eligibility for obstacle scenes, a precondition for any obstacle benefit (L-F5). Gives the
router the added obstacles in every task.

**Risks.**
- Fixtures (cabinet, basket, stove) become large obstacles near targets. Mitigation: roles plus per-part contact permissions in
  ws_swept_planner.
- DA3 failure on thin or transparent objects. Mitigation: SAM exemplar, and the wrist sweep in phase 2.

**Effort.** 2.5 d (+1.0 d height field, included in the total).

---

### PS-9 Tracking, freshness, held-object state and re-localisation

**Fixes.** V7 / L-F6 / E-F8 (wall-clock staleness; tracks never engage; no held state); K2 F1 / F6 (payload self-blocking or
silently absent); L-F8 (held-object extent for touchdown); IC14 (re-localise at the moment of use); R9; H2 §4.4 items 1 and 4;
improvement 6.

**Files (create).** `<NEW>/src/gpt_field/perception_service/tracking.py`, `held.py`, `relocalize.py`.

**Interfaces.**
```python
class Validity:   # per instance
    state: Literal["static","held","engaged","released_settling","possibly_moved","missing"]
    valid_from_step: int; invalidated_by: dict | None; needs_check: bool
class HeldTracker:
    def on_closure(self, step, state8, q7, gap_mm, inventory, supports) -> HeldState
    def on_frame(self, frame, inventory, cams) -> list[Event]       # co-motion / slip / failure
    def on_release(self, step) -> list[Event]
def relocalize(ids: list[str], frame: Frame, deadline_s: float) -> RelocalizeResult   # §4.5
```

**Algorithm.**
1. **Event invalidation** (no wall clock in simulation):
   - E1 robot near: minimum distance between the robot hull vertices (FK, every ring frame) and the instance AABB < d_near → `needs_check`;
     penetration → `possibly_moved`. d_near = 2σ_instance + the measured OSC tracking error (from ws_executor_control).
   - E2 contact notify naming the instance → `possibly_moved`.
   - E3 closure → held check. E4 release → `released_settling`.
   - E5 change detection on the static external camera: D = mean |gray_t − gray_ref| over the instance's projected footprint,
     excluding robot pixels in either frame. Threshold τ_t = median + 3·MAD of the same statistic over static background patches away
     from the robot (measured per frame). D > τ_t → `possibly_moved`. A false trigger (for example a robot shadow) costs one
     re-localisation, never a failure.
2. **Closure.**
   - From `notify(grasp_closed, gap_mm)` or command and gap stability, compute the inter-finger box from FK.
   - Members = instances whose conservative AABB intersects it; the maximum-overlap instance becomes `held`, and
     `hand_from_object` = T_world_hand⁻¹ · T_world_object is recorded.
   - No member: an `unknown_held` fallback envelope in the hand frame. Width = gap + 2·8 mm pad thickness; depth = 16 mm pad
     extent + 2σ; length from the fingertip (+9.25 mm reach) down to the support height at closure (the object cannot extend
     below its support). This keeps the held-object check from ever being skipped.
3. **Co-motion.**
   - Once the hand has moved ≥ 2× the mask-boundary noise (≈5 mm, est.), compare the attached and static predictions with the new
     external masks.
   - Attached wins for k = 2 frames → `held_confirmed`; static wins → `grasp_not_confirmed`.
   - While held, the payload is rendered into the robot raster (PS-3), excluded from obstacles, and refined in the hand frame from
     every external frame of the carry (multi-view fit in hand coordinates).
4. **Release.** The instance returns to the world at its last attached pose with state `released_settling`; the next full refresh
   refits it on its local support (for example the plate top).
5. **Re-localise at use** (IC14).
   - (a) Valid, no `needs_check` and D ≤ τ → `confirmed` (≈2–5 ms).
   - (b) Otherwise a targeted SAM pass: encode, category prompt plus box exemplar at the predicted bbox, on predicted-visible
     views → refit → `updated` (0.3–0.6 s, est.).
   - (c) Deadline exceeded → `timeout` with the old primitive and `sigma_inflated` (σ grows by the distance the robot body moved near it).
   - Every result carries `age_at_use_s` = stage start − capture time. On a real robot, the executor may require a maximum age (UD6).

**Veto removed / parameter.** Replaces `max_age_s` (`engine.py:23`, wrapper 5 s), same-step blanking (`live.py:247-252`) and
"interaction unknown" (`engine.py:268-269`); never skips the payload check (K2 F1 / F6).
Parameters: d_near (σ plus measured OSC error), τ_t (measured noise), k = 2 frames (from the mask noise and the per-frame recall
argument, as in PS-7), fallback-envelope sizes (robot facts, measured gap, support height).

**Tests (offline).**
- Synthetic sequences: a robot passing near → `needs_check`; a shifted mask → `possibly_moved`; a hand-attached object →
  `held_confirmed` within 2 frames after 5 mm of motion; an object left behind → `grasp_not_confirmed`.
- Replay-truth frames around closures in the H1 T04 reference (549 actions, two grasps): held state correct ≥ 95 % of carry frames;
  payload hand-frame pose error ≤ 1.5 cm (targets).
- Static frames: change detection gives 0 false `possibly_moved` on ≥ 95 % of untouched instances (target).

**Live validation.** Smoke: events `held_confirmed` after each lift; zero payload-absent stages; relocalise latency p95; `age_at_use_s`
logged. Sweep: held-state accuracy (post-hoc truth); share of stage entries needing the targeted path.

**Expected effect.** Avoids the K2-style per-step refresh (3.41 s per step in run 002) through event-indexed refresh. A held object
is always modelled, which enables carry-height checks: T04's router accepted carries at z 0.55–0.56 m with the mug hanging below
the can top. Placement descents can be bounded by measured object extent (L-F8).

**Risks.**
- The real robot has no paused world. Mitigation: change detection plus re-localisation at use, and a maximum age (UD6).
- Articulated targets (drawers): the closure member is a fixture that does not co-move rigidly → state `engaged`; no payload.

**Effort.** 3.5 d.

---

### PS-10 Output contract and compact facts

**Fixes.** V3 / L-F2 (advisory only; boxes never reach the router); V6 / L-F4 / E-F9 (≈2.9k-token summaries re-sent; 31 % / 29 %
of input tokens); L-F11 (scene-point noise); IC9, IC15; improvement 6.

**Files (create).** `<NEW>/src/gpt_field/perception_service/snapshot.py`, `<NEW>/schemas/perception_snapshot_v1.schema.json`,
`<NEW>/schemas/perception_compact_v1.schema.json`, `<NEW>/src/gpt_field/perception_service/compact.py` (stdlib only).

**Interfaces.** PerceptionSnapshot v1 (§4.4); `compact_facts(snapshot: dict, plan: dict, *, max_chars=1500) -> dict` (§4.6);
`router_boxes(snapshot, exclude_ids) -> list[dict]` (stdlib) returning `{lower_xyz_m, upper_xyz_m, id, role, source, sigma_m}`
with the held instance, robot and excluded IDs removed.

**Algorithm.**
- Serialise with fixed key order and rounding (mm for metres, 1e-3 for σ); compute a sha256 per snapshot.
- Compact facts: roles → located / not located, centre and size to the cm, top z, σ, source, state, age in steps; obstacles
  (≤ 8 nearest the hand, by conservative AABB); support z and σ; unbound parts. No points and no disclaimers; the caveats go once
  into the static instructions (ws_protocol_context).

**Veto removed / parameter.** n/a. Parameter: `max_chars` (≈500 tokens; the token-cost basis is 18–20 ms per 1k input tokens).

**Tests (offline).**
- Schema validation of every snapshot produced in PS-12 runs.
- Compact facts ≤ 1,500 chars on every benchmark frame; a deterministic hash for identical inputs.
- `router_boxes` never contains held, robot or excluded IDs.

**Live validation.** Smoke: input-token growth per call vs H1 (target ≤ +0.6k per call, against H3's +2.9k). Sweep: tokens per call and
reasoning tokens per call, as an A/B with and without facts (UD8, G10 confound).

**Expected effect.** −2.3 to −2.4k input tokens per call against H3 (est.); boxes reach the planner.

**Risks.** Consumers may want more fields; schema versioning with additive-only changes.

**Effort.** 1.5 d.

---

### PS-11 Controller-side client, bridge hooks and recorded replay

**Fixes.** V6 / L-F3 (synchronous request path); V8 (fail-open); R4 (perception asynchronous); R8 (replay without a GPU); IC14 (stage-entry stamping).

**Files (create).** `<NEW>/src/gpt_field/perception_client.py` (stdlib plus numpy, `.venv`), `<NEW>/src/gpt_field/perception_recorded.py`,
`<NEW>/src/gpt_field/perception_bridge.py`.

**Files (modify, in `<NEW>` only).**
- `<NEW>/src/gpt_field/agent_speedup_v2_user_check_agent.py` — calls the bridge in `reset` (`:67`), `_ask_agent` (`:266-298`),
  `observe_transition` (`:582-610`). Other workstreams own the router and swept-planner stage hooks; they call `bridge.stage_obstacles` and `bridge.relocalize`.
- `<NEW>/scripts/evaluate_agent_speedup_v2_user_check.py` — flags `--perception-socket`, `--perception off|shadow|on`; client construction
  in `execute_reviewed` (`:574-693`); hashes of the perception options in `runtime_inventory` (`:214-235`).
- `<NEW>/scripts/launch.py` — pass-through flags (`build_command` `:41-58`).

**Interfaces.**
```python
class PerceptionClient(Protocol):
    def reset_episode(self, spec: dict) -> dict
    def set_plan(self, plan: dict, plan_frame_seq: int) -> None
    def request(self, frame_seq: int, mode: str, reason: str) -> None              # non-blocking
    def latest(self) -> dict                                                       # never raises
    def wait_for(self, frame_seq: int, timeout_s: float) -> dict
    def relocalize(self, ids: list[str], frame_seq: int, stage_start_s: float, timeout_s: float) -> dict
    def notify(self, step: int, events: list[dict]) -> None
    def bind_roles(self, picks: dict[str, int], source: str) -> dict
    def get_masks(self, snapshot_seq: int, ids: list[str], views: list[str]) -> dict  # COCO RLE
class SocketPerceptionClient(PerceptionClient): ...   # live; timeouts from the service latency model
class RecordedPerceptionClient(PerceptionClient): ...  # serves trial/perception/consumed.jsonl
class NullPerceptionClient(PerceptionClient): ...      # perception off arm: status=unavailable
class PerceptionBridge:
    def on_reset(self, episode_spec) -> None
    def before_model_call(self, step) -> None            # request(step, 'full', 'decision')
    def after_model_call(self, step) -> dict             # wait_for(step, 0.0)
    def on_transition(self, step, state8, action, controller_events) -> None   # notify
    def stage_obstacles(self, stage_ctx) -> list[dict]   # router_boxes(latest, exclude=stage targets) when mode=='on'
    def relocalize(self, ids, stage_start_s) -> dict
```

**Algorithm.**
- A background reader thread keeps `latest` current.
- Timeouts: `wait_for` after the model returns uses 0 s, because the snapshot was requested ≥ 7 s earlier; `relocalize` uses 2 × the
  warm-up p95 targeted latency.
- Modes:
  - `off`: NullPerceptionClient.
  - `shadow`: snapshots are logged and scored, router boxes are computed and logged, but the router uses model boxes only.
  - `on`: router boxes are merged with model boxes (ws_swept_planner decides the merge; UD5).
- Every consumption is logged to `trial/perception/consumed.jsonl`.

**Veto removed / parameter.** Removes the synchronous request-path coupling (`live.py:274-299`). Timeouts derived from measured latency.

**Tests (offline, `.venv`).**
- A fake service socket that drops, delays and returns garbage: no exception reaches the policy; status is `unavailable`.
- Replay: a recorded run's `consumed.jsonl` reproduces the same `stage_obstacles` outputs.
- The H1 deterministic replay passes in `off` mode with an unchanged action trace (549/549 equal), proving that the hooks are inert.

**Live validation.** Smoke: per-decision perception wait p95 ≤ 50 ms; 0 exceptions; `consumed.jsonl` complete. Sweep: availability and
latency per episode.

**Expected effect.** −1.1 to −1.3 s per call and −1.6 s on the first call against H3 (14.6 / 13.1 s per episode).

**Risks.** Hook placement collides with other workstreams' edits to the same files. Mitigation: a single bridge object and three
call sites, agreed in §6.

**Effort.** 2.0 d.

---

### PS-12 Truth-scored offline perception benchmark

**Fixes.** V9 / E-F10 (never validated); R8 (tests cannot catch live blockers); H2 §4.4 item 7; improvement 7; §6 item 7.

**Files (create).**
- `<NEW>/bench/perception/build_dataset.py` (CPU, read-only on `runs/`). Sources:
  - request JSONs: the last user turn's two `input_image` data URLs, `geometry`, `robot_pose_wxyz`, `finger_joints_m`,
    `context.measured_state.joint_position_rad`;
  - K2 vision npz plus `events.jsonl` joints;
  - H3 saved SAM candidate npz, as legacy masks.
  Writes `bench_data/v1/<run>/frame_<step>.npz` and `index.json` with sha256 values.
- `<NEW>/bench/perception/export_truth.py` (H1 `.venv-sim`, `MUJOCO_GL=egl`, oracle-enabled worker, zero model calls; the user
  starts it).
  - Reset with the run's frozen BDDL and initial state, then replay `trace.jsonl` actions. Assert `next_state` equality at every
    step (≤ 1e-12, else the run is marked not replayable).
  - At each benchmark step, dump per camera the BDDL-owner masks, the robot mask and metric depth (`oracle segment`,
    `agent_speedup_v2_sim.py:551-600`), plus object collision geoms (type, size, pose), support-geom top heights and pad poses.
  - Static fallback without replay: layout-manifest settled poses plus asset geoms (the `truth_boxes.py` method), valid until the
    first contact recorded for that object.
- `<NEW>/bench/perception/run_pipeline.py` (service venv): `--sam live|legacy-masks|truth-masks`, `--da3 live|saved|none`. The
  truth-masks mode isolates geometry error from segmentation error (development only).
- `<NEW>/bench/perception/score.py`, `<NEW>/bench/perception/README.md`, `<NEW>/tests/perception/test_bench_score.py`.

**Data (est. counts from the directory listing).**

| Source | Frames |
|---|---|
| H1-family decision frames: T04 13 + T20 T04 26 + T08 21 + stove-off 16 + drawer 001 42 + drawer 003 36 + official T04 10 | 164 |
| H3 frames with saved masks | 22 |
| K2 001 refresh frames | 20 |
| K2 002 vision frames (moving wrist, DA3, joints) | 350 |
| Astranomy T04 requests (if their format carries both images and geometry) | 184 |

Truth comes from layout manifests (T04 can, T08 bottle, stove-off, drawers) and from replay export for all of them, including the
official-layout run.

**Metrics** (`score.py`).
1. **Detection.** Per (frame, view, truth object with ≥ 200 visible px): matched if an instance mask has IoU ≥ 0.5 with the
   truth owner mask. Recall is reported by class (target, reference, fixture, non-plan obstacle, distractor); precision counts robot
   overlaps separately.
2. **Identity.** Binding accuracy per plan object; ID switches and fragmentation per 100 frames.
3. **Association.** Confusion matrix of cross-view pair decisions vs truth owners; ray-gap distributions.
4. **3-D.** Centre error (xy, z) vs the truth geom-AABB centre, by geometry source; extent ratio per axis; coverage (truth AABB volume
   inside the conservative AABB); over-inflation ratio; σ calibration (share of |err| within 1σ and 2σ).
5. **Support.** Height error by source and fused; σ calibration.
6. **Robot raster.** Per-view IoU, recall and precision vs the truth robot mask; residual robot pixels.
7. **Free / occupied.** Probes on a 1 cm grid over the support extent × [support, support + 0.35 m], plus 2,000 probes in 0–3 cm shells
   around each truth object. Truth occupied = inside any non-robot collision geom. Predicted occupied = inside a non-held conservative
   AABB, or below the support − σ. Reports precision, recall and the **false-free rate within 2 cm of true surfaces**, the
   safety-relevant error.
8. **Held.** State accuracy over carry frames; hand-frame payload pose error.
9. **Latency** per stage, with the GPU recorded; **compact-facts size**.

**Use.** A development aid, not a launch gate (user rule 4). It runs per change: CPU geometry metrics on saved masks and depth take
minutes; GPU detection metrics run when a GPU is free. The scorecard diff against the previous commit is attached to each change.
Targets (not gates): detection recall ≥ 0.9; compact-object centre p90 ≤ 2 cm; coverage ≥ 0.95; robot recall ≥ 0.98; support p90
≤ 5 mm; false-free rate near surfaces ≤ 1 %. Every live run's recorded frames, plus replay truth, extend the dataset.

**Veto removed / parameter.** n/a (measurement). Matching parameters (IoU 0.5, 200 px visibility, 2 cm shell) are scorer conventions,
fixed and versioned.

**Tests (offline).**
- Scorer unit tests on synthetic truth and predictions (matching, IoU, σ calibration).
- The dataset build on 2 runs is hash-stable.
- An `export_truth` dry run on the H1 T04 reference reproduces 549/549 states exactly (as VERIFICATION.md did).

**Live validation.** None live. After each smoke, `score.py` runs on the smoke's recorded frames, with truth from `export_truth`.

**Expected effect.** Perception iteration drops from about 10 min plus 12 Astra calls per live episode to minutes offline (E-F10). It catches
H2-style blockers before paid runs (R8).

**Risks.**
- The replay oracle renders at post-refresh kinematics, one substep ahead: up to about 4 px wrist lag while the hand moves. Only
  settled frames are scored for wrist metrics (|q̇| < ε).
- Main-tree runs may not replay in the H1 package's simulator. The per-step state check detects this; fall back to static truth.

**Effort.** 7.0 d (dataset 1.5, truth export 3.0, scorer and runner 2.5).

---

## 4. Contracts

### 4.1 FrameRing v1 (POSIX shared memory; writer ws_runner_measurement, reader PS-2)

Name: `gpt_field_frames_<run_id>`. Little-endian. Header: 4,096 bytes.

| Offset | Field |
|---|---|
| 0 | magic `b"GFFRAME1"` |
| 8 | version u32 = 1 |
| 12 | header_bytes u32 = 4096 |
| 16 | slot_count u32 (16) |
| 20 | view_count u32 (2) |
| 24 | height u32 |
| 28 | width u32 |
| 32 | channels u32 (3) |
| 40 | slot_bytes u64 |
| 48 | latest_seq u64 (written last) |
| 56 | writer_pid u32 |
| 64 | calibration_sha256 [32] |
| 96 | view names 2 × 16 B ASCII (`external`, `wrist`) |
| 128 | raster code u32 (1 = upright, as the calibration describes) |

Slot k at `4096 + k·slot_bytes`, with a 256 B slot header:

| Offset | Field |
|---|---|
| 0 | seq_begin u64 |
| 8 | frame_seq u64 |
| 16 | step i64 |
| 24 | capture_time_s f64 (monotonic) |
| 32 | sim_time_s f64 |
| 40 | flags u32 (bit0 decision, bit1 pinned_event) |
| 44 | gripper_cmd i32 (−1 open, 0 none, +1 close) |
| 48 | state8 f64[8] |
| 112 | q7 f64[7] |
| 168 | qd7 f64[7] |
| 224 | seq_end u64 |

Then the view images (H·W·3 uint8 each).

Writer: write seq_begin, then data, then seq_end, then header `latest_seq`. Reader: valid iff seq_begin == seq_end == the expected seq,
re-checked after the copy. 16 slots × 1.57 MB ≈ 25 MB.

### 4.2 camera-calibration-1 (consumed; provider ws_runner_measurement A5)

```json
{"schema":"camera-calibration-1","frame":"world","image":{"width":512,"height":512},
 "cameras":{
  "external":{"model":"pinhole","K":[[618.04,0,255.5],[0,618.04,255.5],[0,0,1]],"dist":null,
              "mount":"world","T_world_camera":[[...4x4...]],"convention":"opencv"},
  "wrist":{"model":"pinhole","K":[[333.63,0,255.5],[0,333.63,255.5],[0,0,1]],"dist":null,
           "mount":"hand","T_hand_camera":[[...4x4...]],"convention":"opencv"}},
 "time_offset_s":{"external":0.0,"wrist":0.0},"source":"sim_mujoco|checkerboard|...","sha256":"..."}
```
`dist` is null or OpenCV `[k1,k2,p1,p2,k3]`. The service also accepts H1's `astranomy-agentic-panda-camera-v2` geometry, converted internally.

### 4.3 Socket operations (args → result)

| op | args | result |
|---|---|---|
| `hello` | `{}` | `{service_version, vendor_sha256, device, gpu_uuid, models:{sam:{sha256}, da3:{sha256}}, warm: bool, latency_model:{encode_s, prompt_s, da3_s, full_p95_s, targeted_p95_s}}` |
| `reset_episode` | `{episode_id, task_sentence, calibration (4.2 or H1 geometry), frame_ring, view_frame:{right_xy, away_xy}, robot_model_sha256, config:{router_mode, record_frames}}` | `{ok, snapshot_seq: 0}` |
| `set_plan` | `{plan (approved plan JSON), plan_frame_seq}` | `{categories:[...], parts:[...]}` |
| `observe` | `{frame_seq, mode: full\|track, reason}` | `{queued: bool, job_id}` |
| `snapshot` | `{min_frame_seq?, wait_s?}` | PerceptionSnapshot v1 |
| `relocalize` | `{instance_ids, frame_seq, stage_start_s, deadline_s}` | RelocalizeResult (4.5) |
| `notify` | `{step, events:[Event]}` (4.5) | `{ok}` |
| `bind_roles` | `{picks:{part_id: mark_number}, source}` | `{roles}` |
| `get_masks` | `{snapshot_seq, instance_ids, views}` | `{masks:{id:{view:{size:[h,w], counts:[...]}}}}` (COCO RLE, row-major) |
| `health` | `{}` | `{status: ok\|degraded\|unhealthy, queue, last_latency_s, errors, gpu_free_gib}` |
| `shutdown` | `{}` | `{ok}` |

### 4.4 PerceptionSnapshot v1 (JSON Schema)

```json
{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"perception-snapshot-1","type":"object",
 "required":["schema","episode_id","snapshot_seq","status","status_reason","frame","processing","calibration_sha256",
             "supports","robot","instances","roles","held","marks","events","height_field","limits"],
 "additionalProperties":false,
 "properties":{
  "schema":{"const":"perception-snapshot-1"},
  "episode_id":{"type":"string"},"snapshot_seq":{"type":"integer","minimum":0},
  "status":{"enum":["ok","degraded","unavailable"]},"status_reason":{"type":["string","null"]},
  "frame":{"type":"object","required":["frame_seq","step","capture_time_s","mode","reason"],"additionalProperties":false,
    "properties":{"frame_seq":{"type":"integer"},"step":{"type":"integer"},"capture_time_s":{"type":"number"},
                  "mode":{"enum":["full","track","relocalize"]},"reason":{"type":"string"}}},
  "processing":{"type":"object","required":["started_s","finished_s","stage_s","device","service_version","vendor_sha256"],
    "properties":{"started_s":{"type":"number"},"finished_s":{"type":"number"},
                  "stage_s":{"type":"object","additionalProperties":{"type":"number"}},
                  "device":{"type":"string"},"gpu_uuid":{"type":["string","null"]},
                  "service_version":{"type":"string"},"vendor_sha256":{"type":"string"}}},
  "calibration_sha256":{"type":"string"},
  "supports":{"type":"array","items":{"$ref":"#/$defs/support"}},
  "robot":{"type":"object","required":["step","source","dilation_px","model_sha256"],
    "properties":{"step":{"type":"integer"},"source":{"const":"fk_visual_envelopes"},"dilation_px":{"type":"integer"},
                  "model_sha256":{"type":"string"},"payload_rendered":{"type":"boolean"}}},
  "instances":{"type":"array","items":{"$ref":"#/$defs/instance"}},
  "roles":{"type":"object","additionalProperties":{"$ref":"#/$defs/binding"}},
  "held":{"$ref":"#/$defs/held"},
  "marks":{"type":"array","items":{"$ref":"#/$defs/mark"}},
  "events":{"type":"array","items":{"$ref":"#/$defs/event"}},
  "height_field":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/height_field_ref"}]},
  "limits":{"type":"object","required":["frame","units","sigma_is_one_sd","conservative_k"],
    "properties":{"frame":{"type":"string"},"units":{"const":"m"},"sigma_is_one_sd":{"const":true},
                  "conservative_k":{"type":"number"},"unseen_space":{"type":"string"}}}},
 "$defs":{
  "vec3":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},
  "aabb":{"type":"object","required":["lower_m","upper_m"],"additionalProperties":false,
          "properties":{"lower_m":{"$ref":"#/$defs/vec3"},"upper_m":{"$ref":"#/$defs/vec3"}}},
  "support":{"type":"object","required":["id","normal","height_m","sigma_m","extent_xy","sources","observed_at_step"],
    "properties":{"id":{"type":"string"},"normal":{"$ref":"#/$defs/vec3"},"height_m":{"type":"number"},
      "sigma_m":{"type":"number"},"extent_xy":{"type":"array","items":{"type":"array","items":{"type":"number"},
      "minItems":2,"maxItems":2}},"observed_at_step":{"type":"integer"},
      "sources":{"type":"array","items":{"type":"object","required":["kind","height_m","sigma_m","step"],
        "properties":{"kind":{"enum":["sweep_wrist_motion","sweep_ext_wrist","da3_external_ransac","contact_anchor"]},
                      "height_m":{"type":"number"},"sigma_m":{"type":"number"},"n_patches":{"type":"integer"},
                      "step":{"type":"integer"},"used":{"type":"boolean"}}}}}},
  "primitive":{"type":"object","required":["kind","center_m","z_min_m","z_max_m","yaw_rad","sigma_center_m",
                 "sigma_extent_m","conservative_aabb","source","views_used"],
    "properties":{"kind":{"enum":["cylinder","box"]},"center_m":{"$ref":"#/$defs/vec3"},
      "radius_m":{"type":["number","null"]},"half_extents_m":{"anyOf":[{"$ref":"#/$defs/vec3"},{"type":"null"}]},
      "yaw_rad":{"type":"number"},"z_min_m":{"type":"number"},"z_max_m":{"type":"number"},
      "sigma_center_m":{"$ref":"#/$defs/vec3"},"sigma_along_external_ray_m":{"type":["number","null"]},
      "sigma_extent_m":{"$ref":"#/$defs/vec3"},"conservative_aabb":{"$ref":"#/$defs/aabb"},
      "source":{"enum":["single_view_support_contact","ray_pair_triangulation","multiview_silhouette_fit",
                        "held_rigid","depth_component","unknown_held_envelope"]},
      "views_used":{"type":"array","items":{"type":"object","required":["view","frame_seq"],
        "properties":{"view":{"type":"string"},"frame_seq":{"type":"integer"}}}}}},
  "instance":{"type":"object","required":["id","category","labels","detect_source","role","plan_parts","state",
                "primitive","views","observed_at_step","observed_frame_seq","capture_time_s","valid","invalidated_by","parts"],
    "properties":{"id":{"type":"string","pattern":"^inst_[0-9]{3,}$"},"category":{"type":"string"},
      "labels":{"type":"array","items":{"type":"object","required":["prompt","score"],
        "properties":{"prompt":{"type":"string"},"score":{"type":"number"}}}},
      "detect_source":{"enum":["sam_text","sam_box_exemplar","depth_component"]},
      "role":{"enum":["target_object","reference_object","fixture","plan_object","non_plan"]},
      "plan_parts":{"type":"array","items":{"type":"string"}},
      "state":{"enum":["static","held","engaged","released_settling","possibly_moved","missing"]},
      "primitive":{"$ref":"#/$defs/primitive"},
      "views":{"type":"object","additionalProperties":{"type":"object",
        "required":["frame_seq","bbox_xyxy","pixels","truncated","occluded_fraction"],
        "properties":{"frame_seq":{"type":"integer"},"bbox_xyxy":{"type":"array","items":{"type":"integer"},
          "minItems":4,"maxItems":4},"pixels":{"type":"integer"},"truncated":{"type":"boolean"},
          "occluded_fraction":{"type":"number"}}}},
      "observed_at_step":{"type":"integer"},"observed_frame_seq":{"type":"integer"},"capture_time_s":{"type":"number"},
      "valid":{"type":"boolean"},"needs_check":{"type":"boolean"},
      "invalidated_by":{"anyOf":[{"type":"null"},{"type":"object","required":["type","step"],
        "properties":{"type":{"enum":["robot_near","contact","grasp","release","change_detected","missed"]},
                      "step":{"type":"integer"}}}]},
      "association":{"type":"array","items":{"type":"object","properties":{"views":{"type":"array"},
        "ray_gap_m":{"type":"number"},"gate_m":{"type":"number"},"decision":{"enum":["same","different"]}}}},
      "parts":{"type":"array","items":{"type":"object","required":["noun","center_m","sigma_m","source"],
        "properties":{"noun":{"type":"string"},"center_m":{"$ref":"#/$defs/vec3"},"sigma_m":{"type":"number"},
                      "source":{"enum":["ray_pair","ray_instance_interval"]},"plan_part":{"type":["string","null"]}}}}}},
  "binding":{"type":"object","required":["instance_id","part","source","margin","alternatives"],
    "properties":{"instance_id":{"type":["string","null"]},"part":{"type":"string"},
      "source":{"enum":["plan_cell","direction_hint","set_of_marks","model_bind","unbound"]},
      "margin":{"type":"number"},"alternatives":{"type":"array","items":{"type":"string"}}}},
  "held":{"type":"object","required":["state","instance_id","since_step"],
    "properties":{"state":{"enum":["none","candidate","held_confirmed","grasp_not_confirmed","engaged"]},
      "instance_id":{"type":["string","null"]},"since_step":{"type":["integer","null"]},
      "hand_from_object":{"anyOf":[{"type":"null"},{"type":"array","items":{"type":"array","items":{"type":"number"},
        "minItems":4,"maxItems":4},"minItems":4,"maxItems":4}]},
      "primitive_in_hand":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/primitive"}]}}},
  "mark":{"type":"object","required":["number","instance_id","view","anchor_px","bbox_xyxy"],
    "properties":{"number":{"type":"integer","minimum":1},"instance_id":{"type":"string"},"view":{"type":"string"},
      "anchor_px":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},
      "bbox_xyxy":{"type":"array","items":{"type":"integer"},"minItems":4,"maxItems":4}}},
  "event":{"type":"object","required":["type","step"],
    "properties":{"type":{"enum":["instance_new","instance_missing","possibly_moved","needs_check","held_confirmed",
      "grasp_not_confirmed","released","binding_changed","binding_needed","support_updated","calibration_mismatch",
      "degraded_stage"]},"step":{"type":"integer"},"instance_id":{"type":["string","null"]},"detail":{"type":"string"}}},
  "height_field_ref":{"type":"object","required":["path","sha256","cell_m","origin_xy","shape","step"],
    "properties":{"path":{"type":"string"},"sha256":{"type":"string"},"cell_m":{"type":"number"},
      "origin_xy":{"type":"array","items":{"type":"number"}},"shape":{"type":"array","items":{"type":"integer"}},
      "step":{"type":"integer"}}}}}
```

### 4.5 Notify events and RelocalizeResult

```json
{"step":412,"events":[
  {"type":"gripper_command","command":"close"},
  {"type":"grasp_closed","gap_mm":14.2,"intent_part":"B"},
  {"type":"released"},
  {"type":"contact","source":"progress_guard","nearest_part":"finger|pad|palm|link|held","direction_world":[0,0,-1],
   "depth_m":0.004,"support":false},
  {"type":"stage_start","stage_id":"f12.s2","uses_instances":["inst_004"]},
  {"type":"decision"}]}
```
```json
{"frame_seq":433,"capture_time_s":1782.41,"age_at_use_s":0.03,
 "results":[{"instance_id":"inst_004","status":"confirmed|updated|moved|lost|timeout|unknown_instance",
             "primitive":{"...":"primitive v1"},"sigma_inflated":false}]}
```

### 4.6 Compact facts (model boundary; ws_protocol_context decides inclusion and placement)

```json
{"perception":"ok","step":120,"support":{"z_m":0.437,"sigma_m":0.003},
 "roles":{"A":{"id":"inst_002","cat":"mug","state":"static","center_m":[-0.12,0.05,0.49],"size_m":[0.09,0.08,0.10],
               "top_z_m":0.540,"sigma_m":0.008,"source":"multiview","age_steps":0},
          "F":{"id":null,"unbound":true}},
 "obstacles":[{"id":"inst_007","cat":"unknown_object","box_m":[[-0.06,-0.25,0.43],[0.01,-0.18,0.53]],"sigma_m":0.012}],
 "held":{"id":null},"notes":["inst_005 not seen since step 96"]}
```
Schema `perception_compact_v1`: ≤ 1,500 chars, ≤ 8 obstacles, cm-rounded metres. It never contains points, disclaimers or history.

### 4.7 Set-of-marks answer (proposal to ws_protocol_context; model boundary)

The executive schema gains a nullable `bind` field:
`{"type":["array","null"],"items":{"type":"object","properties":{"part":{"type":"string"},"mark":{"type":"integer","minimum":1}},
"required":["part","mark"],"additionalProperties":false},"maxItems":8}`. It is sent only when the snapshot has `binding_needed` events.
Marks are drawn on the **current** external image (replace, do not add; IC9).

### 4.8 Consumed-snapshot log (replay; consumer ws_runner_measurement)

`trial/perception/consumed.jsonl` rows: `{"step":int,"consumer":"decision|stage_entry|relocalize","snapshot_seq":int,
"sha256":str,"wait_s":float}`. Snapshots are stored at `trial/perception/snapshots/<seq>.json`.

### 4.9 HeightField v1

An npz with `z_max` (float32 H×W), `sigma` (float32), `observed` (bool), `shadow` (bool), `source_mask` (uint8 bits:
external, wrist, primitive). Header in the snapshot `height_field_ref`: 5 mm cells over the observed support extent.

---

## 5. Shared interfaces needed (summary; detail in §4)

| Interface | Provider → consumer | Shape |
|---|---|---|
| FrameRing v1 | ws_runner_measurement (A2) → PS-2 | §4.1; every step, raw upright RGB, state8, q7, qd7, gripper command; plus review-phase frame 0 publication |
| camera-calibration-1 | ws_runner_measurement (A5) → PS-2, ws_geometry_tools | §4.2 |
| Robot model artifact (numpy-only FK + collision hulls + visual envelopes + RethinkMount) | ws_geometry_tools → PS-3, ws_swept_planner. Fallback: PS-3 builds it | `robot_model_panda_v1.npz`: chain (fixed transforms, axes, anchors, limits) plus per-body hull vertices, plane equations, kind and group; sha256 recorded |
| PerceptionSnapshot v1, `router_boxes`, `get_masks` | PS-10 → ws_swept_planner, ws_geometry_tools | §4.4; conservative AABBs with roles; held payload in the hand frame; support planes; height field |
| PerceptionClient / PerceptionBridge hooks | PS-11 → ws_executor_control, ws_swept_planner | `before_model_call`, `after_model_call`, `on_transition`, `stage_obstacles(stage_ctx{target_instance_ids})`, `relocalize` |
| Controller events | ws_executor_control → PS-9 | §4.5: `contact` (guard), `grasp_closed` (gap-band predicate), `released`, `stage_start`; plus the measured OSC tracking error for d_near |
| Plan object spec | ws_protocol_context → PS-4, PS-7 | Every plan or merged first call keeps `parts[].text` as "`<object>: <part>`" with `x` and `w` cells (IC7); optional explicit `noun` field |
| Set-of-marks bind field and mark overlay | ws_protocol_context ↔ PS-7 | §4.7 |
| Compact facts | PS-10 → ws_protocol_context | §4.6 |
| Consumed-snapshot log + RecordedPerceptionClient | PS-11 → ws_runner_measurement (deterministic replay) | §4.8 |
| Service lifecycle, health and build digest | ws_runner_measurement → PS-1 | Start once per sweep; `hello` / `health` probe; include `vendor_sha256` and model sha256s in the build digest; restart on unhealthy |
| Evaluation cap flag `--max-model-calls` (for capped smokes) | ws_runner_measurement (A1) → smoke commands | CLI flag |

---

## 6. Sequencing (respecting cross_critic IC1–IC15)

1. **Phase 0, offline and CPU only, can start now.**
   - PS-12 dataset builder and scorer with static truth; PS-1 vendoring.
   - PS-2, PS-3 (on K2 002 joints), PS-5 (sweep on saved RGB), PS-6 (on H3 and K2 saved masks), PS-7 cell binding (saved masks).
   - This is improvement 7: "benchmark before changing anything else". The truth export (PS-12) needs a user go-ahead (UD7).
2. **Phase 1, GPU replay when a GPU is free.** PS-4 category prompts on saved RGB; PS-8 discovery (K2 frames already carry DA3);
   scorecard v1. In parallel, the user-started same-window ablation (H1 / H1 + H3 instruction text / H1 + SAM 2D boxes, improvement 1)
   decides whether perception boxes should also be drawn into the prompt (IC9), which is ws_protocol_context's choice.
3. **Phase 2, the service.** PS-1 server, PS-11 client in **shadow** mode, PS-9, PS-10.
   - Integration hooks agreed with ws_executor_control and ws_swept_planner.
   - Measurement first (IC6): smokes are scored under finish-scoring from ws_runner_measurement.
4. **Phase 3.** Capped smoke in shadow mode, then `--perception on` **only after** ws_swept_planner ships target/held/robot exclusion and
   escape semantics (IC13: exclude, then escape, then automatic boxes, then the body sweep) and the exact-distance router (IC4).
   The body sweep (C4) uses PS-6 geometry plus the PS-9 payload (IC12: only together with better geometry).
5. **Constraints.**
   - Distance-scheduled speed (IC1, D2) may use `sigma_center_m` plus braking only after guarded motion (C1) lands.
   - Async consumption requires PS-9 re-localisation at stage entry (IC14).
   - History compaction keeps the verbatim reasons (IC15): compact facts are current-only and never summarise feedback.
   - IC7: if A4 merges the plan into the first executive call, that call has no plan-bound facts. Binding happens after it (ms),
     and ws_geometry_tools' `locate` snaps the model's pointed pixels to inventory instances.

**Minimum viable slice (≈24 d):** PS-1, PS-2, PS-3, PS-4, PS-5 (ext↔wrist sweep plus DA3 prior), PS-6 (single view plus pair), PS-7 (cells),
PS-10, PS-11, PS-12 (static truth plus scorer). **Second slice (+12.5 d):** PS-8 with the height field, PS-9, the PS-5 wrist-motion sweep and
contact anchors, the PS-6 multi-view fit, PS-12 replay truth.

---

## 7. Live validation (the user starts every run; commands for the plan)

```bash
# Service, once per sweep, on a free GPU (UD1). Example: GPU 7 is idle at the time of writing.
cd <NEW> && CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /workspace/gpt_field/.venv-foundation-vision/bin/python -m gpt_field.perception_service.server \
  --socket /tmp/gpt_field_perception.sock --device cuda:7 --min-free-gib 10 \
  --sam-checkpoint /workspace/gpt_field/artifacts/foundation_vision/models/sam3.1/sam3.1_multiplex.pt \
  --da3-model /workspace/gpt_field/artifacts/foundation_vision/models/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c \
  --robot-model assets/robot_model_panda_v1.npz --log-dir runs/perception_service_$(date -u +%Y%m%dT%H%M%SZ) \
  --persist --record-frames decision

# Smoke 1: shadow mode, T04 can, capped at 6 executive calls
cd <NEW> && .venv/bin/python scripts/launch.py --task t04 --perception-socket /tmp/gpt_field_perception.sock \
  --perception shadow --max-model-calls 6 --output runs/smoke_perception_shadow_t04_$(date -u +%Y%m%dT%H%M%SZ)

# Post hoc (no model calls): replay truth, then score the smoke's recorded frames
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<minor of a free GPU> <NEW>/.venv-sim/bin/python <NEW>/bench/perception/export_truth.py \
  --run <smoke dir> --out <NEW>/bench_truth/v1
/workspace/gpt_field/.venv-foundation-vision/bin/python <NEW>/bench/perception/score.py --frames <service log-dir>/frames --truth <NEW>/bench_truth/v1 \
  --out <smoke dir>/perception_scorecard.json
```

**Smoke 1 pass criteria.**
- Models warm before the plan call returns; frame-0 snapshot before the first executive request.
- 0 episode aborts from perception; decision wait p95 ≤ 50 ms.
- Can instance present without `--vision-query`; bindings A–F correct (post hoc).
- Robot recall ≥ 0.98; conservative AABB coverage of the can ≥ 0.95; compact facts ≤ 1.5k chars.

**Smoke 2** (after ws_swept_planner's escape lands): the same with `--perception on` on T04 can and T08 bottle.
Pass criteria: no `route_start_blocked` caused by a perception box enclosing the start; 0 robot-vs-obstacle contacts; calls not above
the shadow arm by more than 1.

**Sweep metrics (coverage map, per task).**
- Perception availability; snapshot latency p50/p95; relocalise path share.
- Detection recall by class and binding accuracy (post hoc).
- First-obstacle-box coverage; perception-caused route refusals; forbidden contact episodes with perceived vs unperceived obstacles.
- Held-state accuracy; tokens and reasoning tokens per call (A/B with facts on or off, UD8).

---

## 8. User decisions

| ID | Question | Options | Recommendation |
|---|---|---|---|
| UD1 | Which GPU runs the service? | Fixed index per sweep (e.g. an idle 4090) / auto-pick by free memory at preload | Fixed, idle GPU; auto only as a fallback. Training on the same GPU slows SAM and risks OOM |
| UD2 | May the harness make a short "look" motion to measure the support plane when the step-0 sweep fails? | Never / ≤ 5 cm, ≈1 s, no model call / always at episode start | Allow it only on step-0 failure (σ > 1 cm); otherwise wait for the model's first motion |
| UD3 | Role-binding fallback when cells and words do not resolve | Direction words only / one-time set-of-marks pick via a `bind` field / both | Both; marks only when unresolved |
| UD4 | Class-agnostic discovery method | DA3 above-support components + SAM box exemplar / + SAM generic prompt sweep; fixtures as obstacles yes/no | DA3 components + exemplar; decide the generic sweep from the benchmark; fixtures yes, with roles |
| UD5 | How perception boxes reach the router in the first build | Shadow only / merge with model boxes / replace model boxes (model keeps semantic overrides) | Shadow until escape semantics land, then merge with provenance; replace only after a measured A/B |
| UD6 | Real-robot freshness at the moment of use | Event-indexed only / plus maximum age in seconds | Simulation: event-indexed. Real robot: also a maximum age, set once hardware latency is measured |
| UD7 | May the benchmark truth exporter run (offline simulator replay in oracle mode, zero model calls, one EGL GPU)? | Yes, on all saved runs / only H1-package runs / no (static truth only) | Yes, on all saved runs; non-replayable runs fall back to static truth |
| UD8 | Send compact perception facts to the model in the first sweep? | Executor-only / facts / A/B arms | A/B arms within the sweep design (G10 instruction confound) |
| UD9 | Block on event frames (closure/release) in the simulator for deterministic consumption? | Non-blocking / bounded barrier (≈1 s wall per event) | Non-blocking, with the conservative fallback envelope; barrier only for replay experiments |

---

## 9. Effort

| Component | Days |
|---|---|
| PS-1 Vendored package and service | 3.0 |
| PS-2 Frames and calibration | 1.5 |
| PS-3 Robot raster | 2.5 |
| PS-4 Category prompts and detection | 2.0 |
| PS-5 Support plane | 3.0 |
| PS-6 Association and 3-D primitives | 4.0 |
| PS-7 Inventory and binding | 3.0 |
| PS-8 Discovery and height field | 3.5 |
| PS-9 Tracking, held, relocalise | 3.5 |
| PS-10 Snapshot contract and compact facts | 1.5 |
| PS-11 Client, bridge and recorded replay | 2.0 |
| PS-12 Benchmark | 7.0 |
| **Total** | **36.5** |

(The minimum viable slice is about 24 d; §6.)
