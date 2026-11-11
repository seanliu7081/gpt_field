# Workstream design: `ws_executor_control`

**Scope:** the motion primitives, sensing and kinematics inside the H1 fragment executor, for the approved direction: H1's loop, with foundation_vision perception run as a service, foundation_harness robot geometry used for path checks, and metric estimation taken away from the model.

**Covers:** B1, B2, B3, C1, C5, D1 (executor side) and D2 (executor side, later), as defined in `reports/harness_analysis_20261108/README.md`.

**How this was produced:**
- Read-only design work. Nothing under `/workspace/gpt_field` was edited.
- No simulator, model call or GPU job was used.
- Every number marked *measured here* comes from new offline scripts. They read saved run logs only and are in `…/scratchpad/impl_plan/ws_executor_control_work/`; the outputs are in `results.txt` there.

Path conventions:
- `H1/` = `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/`, at git `3cc84ac`. It stays frozen.
- `<NEW>/` = the new build root. ws_runner_measurement owns it; the proposal is a fresh repo cloned from H1 `3cc84ac`, so H1 itself is never modified. Every file below is relative to `<NEW>/`.

---

## 0. What the executor does today (code read, line-exact)

| Mechanism | Code | Problem it causes |
|---|---|---|
| Hard caps 6 segments / 120 steps per segment / 240 steps per fragment | `H1/agent_speedup_v2_user_check_fragments.py:24-26`, `:174-175`, `:402-403`; prompt `_agent_client.py:119,133-134` | A pick-and-place needs ≥2 calls (T04 mug 1: 346 actions, 8 phases). |
| Stage rotation veto: 1.2 rad transit, 0.5 rad contact | `fragments.py:27-34` (`target_rotation_rad`), `:285-291`; prompt `:135-136` | Both first plans rejected: T04 1.572 rad, T08 1.569 rad. Each veto costs a repair call (7.2–7.7 s). |
| Rotation runs concurrently with translation; the per-step rotation command is capped at 0.1 rad | `fragments.py:234` (`ActionAdapter(max_rotation_rad=.1)`), `:345-346`; `action.py:87-92` | Stages execute as "translate, then rotate in place". This is the T08 finger-on-bottle press (cross_critic C5). |
| Stall = `errors[0] − min(errors[1:]) < 0.8 mm` over 12 steps after ≥16 steps, on `route_remaining + 0.05·orientation_error` | `fragments.py:404-419`, constants `:43-45` | Fires at the overshoot apex, about 11 steps after the error minimum. Misses arms that move but diverge. Rotation masks contact (T08: 45 steps pinned). |
| Zero translation inside the 4 mm arrival tolerance | `tracking.py:90-95`; tolerance set at `routing.py:154-156`; also the single-point path `routing.py:165-180` | The arm hovers and drifts at the boundary while rotation converges (d001 steps 208–241). |
| Lookahead `max(.03, cap)` = 0.10 m, clipped to 0.10 m, no braking | `routing.py:154-156`, `tracking.py:114-141`, `fragments.py:339-344` | Overshoot of 10–28 mm. 17 of 41 stalls are false overshoot stalls. |
| 8-corner lattice validity (hidden margin of 0–20 mm per axis) | `fine_distance_planner.py:101-106` (`_point_valid`), `:117-130` (`query`), `:135-155` (`segment_valid`); used at `routing.py:105-108`, `tracking.py:83` | Of 7 start refusals, 5 were spurious; their starts were 6.3–20.3 mm outside the expanded box. |
| `route_start_blocked` refusal | `routing.py:105-106` | A repair call of 10–15 s each. |
| `approach` stall on contact means failure, and the pending segments are dropped | `fragments.py:159-160`, `:418-419`, `:426-434`; `agent.py:382-401` | 19 contact stalls, 5 discards. |
| Feedback "contact_or_collision_cause: unknown" | `fragments.py:482` | The model misdiagnoses. |
| No joint limits or IK; joints are sent but limits are not | `agent.py:191,222,237`; `observation.py:86-113` | drawer 001: a 9-call J4-limit streak (141 s). |
| Any fragment end hands control back to the model | `agent.py:382-401` (`_need_agent = True`) | 54 % of calls follow a completed fragment. |

**Prior art already in this repo (re-used, not re-invented):**
- `contact_skills.py:577-730` `AnchoredMove`. A guarded move that stalls on a collapse of the realised/commanded ratio (0.6 × a free ratio of 0.26, 2-step window), detects open-finger touch from inward deflection, and detects gap rise.
- `contact_skills.py:209-251` `GripperCalibration`. The measured empty-close curve and the touch threshold.
- `contact_skills.py:556-574` `classify_close`. An object is present if the gap exceeds the empty curve at the same k.
- `contact_skills.py:180-206` `HoldingActionAdapter`. It never emits an exactly-zero rotation (the robosuite stale-goal issue).

---

## 1. Measured facts this design rests on (all new, offline)

All of these come from scripts in `ws_executor_control_work/`:
- The compact per-step pickles of 9 H1/H3 runs (`harness_analysis/w2/data/*_steps.pkl`). Contact labels (`forb_inc`, `inc_allowed`) were used only to exclude or evaluate steps, never as a runtime input.
- Request and response logs.
- The pinned robosuite 1.4.1 Panda XMLs.

| # | Fact | Value | Script |
|---|---|---|---|
| M1 | The translation OSC response is first order. `s_t = a·s_{t−1} + b·c_t`, where `s` is the realised grip-site step and `c` the commanded offset `= action[:3]·TRANSLATION_SCALE_M`. | Contact-free pooled fit over 9 runs, 5,593 steps: **a = 0.711, b = 0.0732, ρ = b/(1−a) = 0.254, R² = 0.973**. Per-run a = 0.53–0.77; ρ = 0.248–0.262 (stable). Bootstrap 95 % CI of a: [0.59, 0.75]. The zero-command decay ratio, median 0.72 (n = 69), matches a. Residual std 0.64 mm per axis per step; rest-noise p99 0.39 mm. | `osc_fit.py`, `plant_perrun.py`, `guard_eval.py` |
| M2 | The rotation OSC response is first order and slow. | **a_r = 0.90, b_r = 0.027, ρ_r = 0.27, R² = 0.95.** At the 0.1 rad command cap, realised rotation is a median 0.0229 rad/step (p10 0.019, p90 0.027). | `osc_fit2.py` |
| M3 | Rotation lags translation in combined stages. | In 24 combined stages, half the rotation took 16–26 steps against 3–10 steps for half the translation. Orientation re-crossed tolerance in 7 of 220 segments. | `rot_profile.py` |
| M4 | The current tracker overshoots; a braking law derived from M1 does not. | Plant simulation, 0.23 m move, σ = 0.6 mm: current tracker overshoot median **17.4 mm** (p90 21.2), 25 steps. Braking law `s ≤ (1−a)(r − tol/2)`: **0 mm**, 21 steps. Robust for true a in [0.53, 0.77] with a_brake = 0.71. With a_brake = 0.77 it overshoots 6 mm when the true a is 0.53, so use the pooled estimate, not the maximum. | `rot_profile.py`, `brake_robust.py` |
| M5 | The current rotation law "arrives" within 0.06 rad but then overshoots. | Error after arrival: p90 0.10–0.13 rad. A plant-derived brake switch keeps it ≤ 0.062 rad, at +8…11 steps. | `rot_brake2.py` |
| M6 | Translation progress per step relative to the plant prediction is a clean free-space signal when no rotation is commanded. | Evaluable contact-free steps (n = 3,319): ratio realised/predicted p0.1 = **0.53**, p0.5 = 0.64, p1 = 0.75, median 0.998. Fraction < 0.5 = 0.09 %. | `guard_calib.py` |
| M7 | Simultaneous rotation corrupts that signal. | Fraction of free steps with ratio < 0.5, by rotation command: 0.03 % (< 0.02 rad), 0.9 % (0.02–0.04), 4.7 % (0.04–0.07), 6.7 % (≥ 0.07). 64 of 66 free-space shortfall outliers coincide with rotation > 0.02 rad. | `rot_coupling.py`, `guard_eval3.py` |
| M8 | Guard performance. Fire when realised < 0.5 × predicted on 3 consecutive evaluable steps, where evaluable means a command of at least 1 mm and less than 0.02 rad of rotation. | Detected **15 of 29** forbidden-contact episodes (H1 + H3 logs), a **median 3 steps after onset** (today's stalls fire at a median of 13). There were 9 contact-free fire events across 9 runs: 3 during `set_gripper` holds, which the design excludes, and the rest in overshoot returns or near-target moves that M4's braking removes. | `guard_eval3.py`, `guard_eval4.py` |
| M9 | Open-finger inward deflection is a zero-false-positive contact channel. | Free open-settled steps (n = 4,907): p99.9 0.033 mm, max 0.062 mm. The existing 0.10 mm touch threshold fired on 0 free steps and on 7.8 % of contact steps. | `finger_touch.py` |
| M10 | Numpy-only Panda FK from the main-tree `robodojo_direct_sim.py` runs in H1's controller `.venv` (numpy 2.5.3, no scipy). Binding the base from measured state8 and joints is consistent. | Base pose spread **0.27 mm / 0.033°** (T04, 13 requests; base (−0.51, 0, 0.42)) and **0.61 mm / 0.038°** (d001, 42 requests; base (−0.66, 0, 0.912)). | `fk_check.py` |
| M11 | Local IK (DLS from the measured posture, which mimics the OSC) flags exactly the joint-limit cases. | Flags d001 calls 16, 17, 21, 22, 23 (J4 at limit, 5–326 mm residual) and 24, and d003 calls 4 (J5) and 8 (J5 margin 0.013). **0 of 31** T04/T08 targets flagged. With multiple seeds, d001 calls 16/17/21/22/23 are **reachable with 0.55–1.19 rad margin from another posture** (posture-limited); d003 call 4 is unreachable (24 mm / 16.4°). IK takes 6.5–7.5 ms median and ≈ 20 ms p95. | `ik_pass.py` |
| M12 | A Cartesian-only posture recovery is not shown to work offline. | In a kinematic emulation of the OSC (DLS with limit clamping, no dynamics), none of three Cartesian strategies left the J4 limit in the d001 states: (i) J·e_j, (ii) the Cartesian image of a joint-interpolated path, (iii) retracting toward the shoulder, with or without a null-space pull. The emulation converged at 0.0 mm on non-limit targets. The real OSC did leave J4 between calls 21 and 22, so the emulation is pessimistic but inconclusive. | `posture_emul*.py` |
| M13 | Grasp continuation is gap-detectable without new constants. | Both observed failures fail the existing empty-close-curve rules: the miss settled at 1.4 mm, at the empty floor; the slip went 13.3 mm → 1.0 mm. The 19 visually confirmed grasps never went below 4.9 mm, which is ≫ the 1.2–1.5 mm floor. Deliberate closed-finger pushes also read 1.4 mm, so intent must be declared. | data `cross_critic_work/data/grasp_gap.json` |

**Consequence for "k·√remaining".** The data reject the constant-deceleration plant that a square-root law assumes. The OSC is first order (viscous): the zero-command decay is geometric with ratio a. The correct braking law is linear in the remaining distance, with gain **k = 1 − a = 0.29**, derived from the measured decay. §3 gives the exact command law.

---

## 2. Architecture

**Where it runs.** All components run inside the controller process, in the `.venv` interpreter (numpy only). No scipy or torch is needed (M10). Perception and the body sweep are other workstreams; they plug in through interfaces (§5).

New package `src/gpt_field/executor/`:

```
executor/
  osc_plant.py          EXC-0  fitted plant, braking/rotation laws, profile simulation
  osc_response.json     EXC-0  calibration artefact (hash recorded in the build digest)
  progress_monitor.py   EXC-B3 guard / no-progress / in-band / finger-touch / rotation channels
  braking.py            EXC-B3 BrakingPathTracker (no deadband; brakes to the endpoint and to sharp vertices)
  sequencing.py         EXC-B1 rotation sequencing: modes, site selection, phase plan
  executed_profile.py   EXC-B1 ExecutedProfile dataclass + JSON (to ws_swept_planner)
  box_clearance.py      EXC-B2 exact, vectorised point/segment↔AABB distances and gradients
  clearance_oracle.py   EXC-B2 ClearanceOracle protocol + GripSiteBoxOracle (interim)
  transit_route.py      EXC-B2 TransitRoute2: exact field, explicit margin, checked escape
  kinematics.py         EXC-C5 PandaModel/BoundPanda over the vendored chain; reachability
  posture.py            EXC-C5 PostureRecoveryPhase (disabled until sim-validated)
  guard_policy.py       EXC-C1 response to monitor events: classify locally, hold or back off, feedback
  predicates.py         EXC-D1 continuation predicates (object present, held gap, gap band, contact window)
  feedback.py           shared record builders + JSON schemas (schemas/*.json)
vendor/
  robodojo_direct_sim.py  byte copy, sha256 6d2746cb…c9971bb (main tree, 266 lines)
  panda_assets/robot.xml          sha256 263c0f9a…030cd95e (robosuite 1.4.1)
  panda_assets/panda_gripper.xml  sha256 d9d68123…5412baac
  VENDOR.json                     source paths, hashes, copy date, licence note
```

**Per-step flow** inside `FragmentSequencePrimitive` (modified `agent_speedup_v2_user_check_fragments.py`, version `user-check-measured-fragments-5`):

```
enter_segment(view):
  target pose (as today) ─► reachability = kin.check(target, profile samples)          [C5a]
  plan = sequencing.plan_stage(entry, target, profile, oracle, plant)                  [B1]
       ─► phases: rotate | translate | coupled | (escape prepended by B2) | hold | gripper
  if transit arm stage: route = TransitRoute2(..., oracle, plant, s_entry)            [B2]
  monitor = ProgressMonitor(plant, expect, phases)                                     [B3]
  budget = max_steps or plant.predicted_steps(...) × overrun                           [D1]
next(view):
  rotate     → hold XYZ (continuous) + plant.rotation_command (with brake switch)
  translate  → BrakingPathTracker.command (braking law) + orientation hold
  coupled    → translation progress ≤ rotation progress + m_track/L
  escape/backoff/posture → their leg, guarded
observe(before, action, after):
  monitor.update(...) → event ∈ {arrival, arrived_in_band, blocked, rotation_blocked,
                                 finger_touch, no_progress, diverging, predicate_fail, budget}
  guard_policy.resolve(event) → succeed (contact_reached / in-band) | local action
                                (back-off, posture) | end (structured feedback)
```

Per-step compute added (estimate): monitor and braking about 30 µs; FK bind about 0.1 ms (only at entry and on events). The IK reachability check costs 7–20 ms once per stage (M11). All of this is well inside the 50 ms control period (R9).

---

## 3. Components

### EXC-0 — Measured OSC plant model and calibration

- **Fixes:** the precondition for user rule 2. Every executor parameter below is derived from this artefact: B3 braking, guard and in-band rules; B1 rotation braking and profile prediction; D1 budgets; D2 slow zone. It covers README B3 ("k derived from measured OSC response") and C1 ("relative to the measured OSC realisation ≈ 0.24"; measured here as 0.254).
- **Files:** create `executor/osc_plant.py`, `executor/osc_response.json`, `scripts/calibrate_osc_response.py`, `tests/test_osc_plant.py`.

**Contract (`osc_response.json`):**
```json
{"version":"osc-response-1",
 "conditions":{"controller":"OSC_POSE","kp":150,"damping_ratio":1,"control_hz":20,
               "translation_scale_m":0.10,"rotation_scale_rad":0.5,"robosuite":"1.4.1","mujoco":"3.2.6"},
 "corpus":[{"run":"<dir>","trace_sha256":"…","steps_used":231,"source":"run_log|scripted_benchmark"}],
 "translation":{"a":0.711,"b":0.0732,"rho":0.254,"r2":0.973,"n_steps":5593,"a_ci95":[0.593,0.754],
                "a_per_run":{"…":0.568},"residual_std_m":0.00064,"rest_noise_p99_m":0.00039},
 "rotation":{"a":0.900,"b":0.0271,"rho":0.271,"r2":0.954,"saturated_rate_rad_per_step":[0.019,0.0229,0.027]},
 "guard":{"f":0.50,"window_steps":3,"c_eval_m":0.0015,"rotation_eval_max_rad":0.02,
          "free_ratio_quantiles":{"p0.1":0.53,"p0.5":0.64,"p1":0.75,"p50":0.998},
          "free_event_rate_per_1000_evaluable":null},
 "gripper":{"empty_curve_mm":[…30 values…],"inward_noise_mm":0.02,"touch_threshold_mm":0.10},
 "derived":{"braking_gain":0.289,"settle_steps_translation":4,"settle_steps_rotation":10}}
```

**API:**
```python
@dataclass(frozen=True)
class OSCPlant:
    a: float; b: float; a_r: float; b_r: float; rho: float; rho_r: float
    residual_std_m: float; rest_noise_p99_m: float
    guard_f: float; guard_window: int; c_eval_m: float; rot_eval_max_rad: float
    source_sha256: str
    @classmethod
    def load(cls, path: Path | None = None, *, controller_metadata: dict | None = None) -> "OSCPlant"
    def predict_step(self, s_prev: np.ndarray, c: np.ndarray) -> np.ndarray
    def predict_rot_step(self, w_prev: np.ndarray, c_r: np.ndarray) -> np.ndarray
    def coast_m(self, s_along: float) -> float                     # a·s/(1−a)
    def braking_offset_m(self, remaining_m, s_along_m, cap_m, band_m) -> float
    def rotation_offset_rad(self, err_rad, w_along_rad, cap_rad, band_rad) -> float
    def predicted_steps(self, distance_m, rotation_rad, *, cap_m, cap_rad, tol_m, tol_rad, mode) -> int
    def simulate(self, phases: list[Phase], s0, w0) -> list[ProfileSample]
```

**Algorithm:**
- **Fit.** Least squares per axis, on contact-free steps (the excluded steps and the step before them).
  - Translation: `s_t[k] ~ a·s_{t−1}[k] + b·c_t[k]`. Rotation uses the same form on world rotation vectors.
  - `guard.f` = the p0.1 of the free per-step ratio, rounded down to 0.05.
  - `c_eval` = rest-noise p99 / ρ.
  - `rotation_eval_max_rad` = the largest rotation-command bin edge whose free fraction below f is ≤ 0.1 % (M7).
  - `window_steps` = the smallest W in {2, 3, 4} with ≤ 1 contact-free fire event per 1,000 evaluable steps on the corpus.
- **Preferred source:** `scripts/benchmark_executor_sim.py` (EXC-INT), scripted free-space moves that need no contact labels. Run logs are the bootstrap source. Contact labels are used only to exclude steps from the fit (a dev-time diagnostic, never a runtime input).
- **Load check.** The plant refuses to load if the live worker's `controller` metadata (kp, damping, scale, control frequency) differs from `conditions`. This is a consistency check of the build, not a task veto.

**Parameters added:** none tuned. Every field is a fitted or quantile statistic of measured OSC realisation, measured noise or the measured gripper empty curve.

**Depends on:** none.

**Offline tests (`tests/test_osc_plant.py`):**
1. The refit on a 9-run fixture (about 150 KB extracted from the pickles; diagnostic labels kept only in the fixture) gives a in [0.55, 0.80], ρ in [0.23, 0.27], R² > 0.95, and a zero-command decay median within ±0.1 of a.
2. Loading with mismatched controller metadata raises.
3. `simulate` reproduces the recorded T04 steps 1–33 (combined 0.78 rad / 124 mm stage) to within 15 mm and 0.1 rad at each step.

**Live validation:** the first capped smoke logs predicted against realised steps per phase (telemetry, §5). The prediction residual should stay within the calibration's p99 (2.96 mm per axis per step). Otherwise refit from that smoke plus the scripted benchmark.

**Expected effect:** enabling only. The parameters become traceable to measurements.

**Risks:**
- a depends on posture: the drawer runs fit 0.69–0.77 against 0.53–0.64 elsewhere. Mitigation: use the pooled a for braking (M4 shows this is robust across [0.53, 0.77]) and log per-phase residuals.
- The contact plant differs. Mitigation: the guard uses the free plant on purpose, so contact appears as a shortfall.

**Effort:** 2.0 engineer-days.

---

### EXC-B3 — Progress monitor (stall replacement), no deadband, braking-aware tracking, `stop_on: contact`

**Fixes:**
- README §3.1 item 3 (overshoot and false stalls) and item 4 (contact counted as failure).
- B3 in full.
- verify_h1_speed F3 corrections 1–3 (stall fires about 11 steps after the minimum; tracker deadband; "speed ≈ 0" still fires at the apex and misses divergence).
- verify_h1_speed F4 (a declared stop is success, with its residual).

**Files:**
- Create: `executor/progress_monitor.py`, `executor/braking.py`, `tests/test_progress_monitor.py`, `tests/test_braking_tracker.py`.
- Modify: `agent_speedup_v2_user_check_fragments.py`. Delete constants `:43-45` and the stall block `:404-419`; replace the rotation pull in `_next` `:339-346`; add the `expect` handling. H1's `tracking.py` and `routing.py` stay unchanged in `<NEW>` for legacy replays; the new code uses `braking.py`.

**(a) `ProgressMonitor`** replaces `fragments.py:404-419`.
```python
class ProgressMonitor:
    def __init__(self, plant: OSCPlant, gripper_cal: GripperCalibration, *, stop_on: str,
                 contact_window_m: float | None, tol_m: float, tol_rad: float): ...
    def update(self, *, phase: str, skill: str, c_m: np.ndarray, c_rad: np.ndarray,
               s_m: np.ndarray, w_rad: np.ndarray, s_prev_m: np.ndarray, w_prev_rad: np.ndarray,
               remaining_m: float, rot_err_rad: float, gripper_open_settled: bool,
               finger_inward_mm: tuple[float, float]) -> MonitorEvent | None
    def evidence(self) -> dict          # window values, predictions, ratios (≤ 600 B)
@dataclass
class MonitorEvent:
    kind: Literal["blocked","rotation_blocked","finger_touch","no_progress","diverging","arrived_in_band"]
    at_step: int; window: list[dict]; detail: dict
```

Rules (all constants come from `osc_response.json`):
- **Evaluable translation step.** The skill moves the arm, |c| ≥ `c_eval_m`, ⟨s_prev, û⟩ ≥ 0 (no reversing momentum; a parameter-free rule), and |c_rad| < `rotation_eval_max_rad`. `set_gripper` and `wait` are never evaluated; this removes the 3 hold false-fires in M8.
- **blocked.** On each of the last W evaluable steps, `real < f·pred`, where `pred = a⟨s_prev,û⟩ + b|c|` and `real = ⟨s,û⟩`.
- **rotation_blocked** (rotate phases). The same rule on the rotation plant, about the commanded rotation axis.
- **finger_touch.** The gripper is open and settled, and max inward deflection > `touch_threshold_mm` (M9; the existing `_inward_mm`, `contact_skills.py:175-177`).
- **no_progress** (the catch-all for non-evaluable phases). Over the plant settling window n_s (4 translation, 10 rotation), Σ⟨s,û⟩ ≤ n_s·rest_noise_p99, while Σ|c| ≥ n_s·c_eval. **diverging**: Σ⟨s,û⟩ < −n_s·rest_noise_p99. This replaces "error not improved since the window start": it never fires while the arm moves toward the target (the T04 return at 1.5 mm/step), and it does fire on d003-type divergence.
- **arrived_in_band.** Any of the events above is converted into arrival if `remaining_m ≤ tol_m + Σ_window pred` and `rot_err ≤ tol_rad`. The guard cannot tell a block from arrival inside the distance its own window would have covered, so this is in-band creep counted as arrival, derived from the plant.

**(b) Deadband.** `BrakingPathTracker.command` always returns the braking-limited offset toward the reference, with no zero zone inside the tolerance. This replaces `tracking.py:90-95` and the arrival zeroing in `routing.py:173-174` for the new tracker. Arrival is still "2 consecutive samples within tolerance" (`fragments.py:38`, `:368-369`).

**(c) Braking-aware tracking.**
```python
class BrakingPathTracker:
    def __init__(self, path_xyz: np.ndarray, plant: OSCPlant, *, cap_m: float, tol_m: float,
                 lookahead_m: float, validity: Callable[[np.ndarray, float], bool]): ...
    def command(self, position: np.ndarray, s_prev: np.ndarray) -> tuple[np.ndarray, dict]
```
- The direction is toward the lookahead reference, keeping H1's earlier-vertex validity fallback (`tracking.py:116-141`).
- The magnitude along the tangent is `min(cap, plant.braking_offset_m(r_eff, s_along, cap, tol/2))`, with
  `braking_offset = clip(((1−a)·(r_eff − tol/2) − a·s_along)/b, 0, cap)`.
- `r_eff` is the remaining arc to the endpoint, or to the next vertex with a turn > 30°, whichever is closer. At such a vertex the band is `m_track` (B2). Escape-leg corners are sharp.
- The lateral component is uncapped by braking (limited only by the cap).
- Rotate phases use `rotation_offset_rad`: capped-proportional (today's law) plus a switch to the braking law once the zero-command coast `a_r·ω/(1−a_r)` would pass the band (M5).

**(d) `stop_on: contact`.** This is a segment field inside `expect` (EXC-INT schema).
- If a `blocked`, `finger_touch` or `no_progress` event arrives while `stop_on == "contact"`, and either `contact_window_m` is null or `remaining ≤ window + tol`, then:
  - the segment **succeeds** with reason `contact_reached` and a residual;
  - the next step commands a zero offset (hold, which releases the pressing force);
  - pending segments continue.
- An event before the window gives `unexpected_contact_early`, which is handled as undeclared (EXC-C1).
- An event under `stop_on == "arrival"` goes to EXC-C1.

**Vetoes and parameters:**
- **Removed:** the stall constants STALL_WINDOW_STEPS = 12, STALL_MIN_STEPS = 16 and STALL_PROGRESS_M = 0.0008, the 0.05 rad→m weight, and the tracker deadband.
- **Added, all derived from EXC-0:** f, W, c_eval, rotation_eval_max, braking gain 1−a, and n_s.

**Depends on:** EXC-0.

**Offline tests:**
1. `test_progress_monitor.py` replays the 9 saved step logs (state8 before and after, actions) through the monitor (`validation/replay_executor_logs.py`):
   - fires inside ≥ 13 of 29 forbidden episodes, with median delay ≤ 4 steps (M8: 15/29, 3);
   - fires earlier than the legacy stall in ≥ 90 % of the 19 contact stalls;
   - 0 events during `set_gripper` and `wait`;
   - every remaining contact-free event lies in a recorded overshoot return (minimum < tol + 10 mm, then a rise) or in reversing momentum.
2. On the legacy T04 steps 173–195 (175 → 4.2 → 10.7 → 4.8 mm), `no_progress` does **not** fire. On d003 step 27 (28.0 → 28.4 mm, moving away) `diverging` fires.
3. `test_braking_tracker.py`, in a plant simulation with σ = 0.64 mm, over 300 random moves of 0.03–0.45 m and a in {0.53, 0.63, 0.71, 0.77}:
   - p99 overshoot ≤ tol/2;
   - median steps to 2-sample arrival ≤ the legacy tracker's (M4: 21 vs 25);
   - 0 `no_progress` events;
   - with rotation pending, position held within tolerance (no deadband drift).
4. A synthetic block, where the plant output is zeroed after k steps, gives `blocked` within W + 1 steps for contact (5 mm cap) and transit (100 mm cap) commands.
5. `stop_on contact` + pending `set_gripper` → the contact succeeds and the close executes (fake simulator).

**Live validation:**
- Capped smoke T04 + can: 0 `segment_stall`; overshoot p99 ≤ 2 mm (telemetry); steps per transit at least 10 % below H1's 7.3 mm/step mean-speed equivalent; 0 pending discards after declared contacts.
- Sweep: false `no_progress` (fired, no contact, later arrival) = 0 per task; stall-triggered calls by class against H1 (17 overshoot + 19 contact + 5 genuine in 5 runs).

**Expected effect (estimates):**
- −5 directly avoidable overshoot-stall calls (≈ 92 s) and 10–25 return steps per overshoot event over the 5 H1 runs (verify_h1_speed F3).
- `stop_on` contact: −2…−3 calls (−46…−90 s) (verify F4).
- Plant simulation: transit arrival 25 → 21 steps for 0.23 m, overshoot 17 → 0 mm (M4).

**Risks:**
- The braking law is slow when a is overestimated. Mitigation: pooled a, robustness test 3.
- The contact profile also overshoots at contact speed (coast 3 mm > 2 mm tolerance). Braking applies to all profiles.
- `stop_on` misuse (contact declared on a transit). Validation rejects it (EXC-INT).

**Effort:** 4.5 engineer-days.

---

### EXC-B1 — Rotation sequencing replaces the 1.2 rad / 0.5 rad veto; the executed profile is exposed

**Fixes:**
- README B1 and §3.1 item 13.
- cross_critic IC3 and C5.
- verify_h1_collision F6 and M1 (rotation lags about 3×; the swept check must use the *executed* profile).
- h1_speed F5.1.

**Files:**
- Create: `executor/sequencing.py`, `executor/executed_profile.py`, `tests/test_rotation_sequencing.py`.
- Modify: `fragments.py`. Remove `target_rotation_rad` from `PROFILES` `:27-34` and the angle clause of `:288-291`; the translation envelope (0.50 / 0.08 m) is **kept**, because it bounds the local map (`routing.py:73-75`). Also change `_enter_segment` (phase plan) and `_next` (phase commands).

**Interfaces:**
```python
@dataclass(frozen=True)
class Phase:
    kind: Literal["rotate","translate","coupled","hold","gripper","escape","backoff","posture"]
    site_xyz: np.ndarray | None; target_wxyz: np.ndarray; path_xyz: np.ndarray | None
    exit_rot_tol_rad: float; exit_pos_tol_m: float
@dataclass(frozen=True)
class StagePlan:
    mode: Literal["translate_only","rotate_then_translate","translate_then_rotate","translate_rotate_translate","coupled"]
    phases: tuple[Phase, ...]; rotation_rad: float; site: dict; profile: "ExecutedProfile"
    candidates: tuple[dict, ...]          # mode, site, predicted steps, oracle clearance
def plan_stage(entry_pose_wxyz, target_xyz, target_wxyz, *, profile: str, path_xyz: np.ndarray | None,
               plant: OSCPlant, oracle: ClearanceOracle, gripper_gap_m: float, held) -> StagePlan
```

**ExecutedProfile** (JSON, version `executed-profile-1`; quaternions canonical w ≥ 0, cf. cross_critic C9):
```json
{"version":"executed-profile-1","stage_index":0,"skill":"move_to_pose","profile":"transit",
 "mode":"rotate_then_translate","rotation_site":{"kind":"start","xyz_m":[…]},
 "dt_s":0.05,"plant_sha256":"…","predicted_steps":{"rotate":66,"translate":21,"total":87},
 "samples":[{"k":0,"phase":"rotate","xyz_m":[…],"quaternion_wxyz":[…],"gap_mm":77.6}, …],
 "envelope":{"lateral_m":0.004,"along_m":0.0,"orientation_lag_rad":0.06},
 "gripper":{"command":"open","gap_mm":77.6},"held":null}
```
Samples are taken every step (≤ 240) for checks, downsampled to every 4th step in logs. The measured profile is appended at the end of the segment for audit (`measured_samples`).

**Algorithm:**
1. Let θ = |rotation_error(target, measured)|. If θ ≤ the profile's rotation tolerance (0.06 / 0.04 rad), the mode is `translate_only` and orientation is held toward the target. This is today's behaviour minus the veto.
2. Otherwise the candidate sites are {start, goal, interior route vertices}.
   - **Interim** (oracle `grip_site_boxes_v1`): choose the site with the largest exact grip-site clearance to the unexpanded model boxes; ties go to the start. Site start gives `rotate_then_translate`, goal gives `translate_then_rotate`, a vertex gives `translate_rotate_translate`. `coupled` is never chosen in the interim.
   - **With ws_swept_planner** (oracle `body_sweep_v1`): simulate an `ExecutedProfile` per candidate, including `coupled`, and call `oracle.profile_clearance(profile, held)`. Pick the fastest candidate whose minimum clearance is ≥ the required margin. If none qualifies, pick the one with the largest clearance; there is no veto, the guard stays active, and the choice is reported.
3. **Rotate phase.** XYZ is held at the site continuously (no deadband). The rotation command is `plant.rotation_offset_rad` (M5). The phase exits when `err ≤ exit_rot_tol` **and** the coast `a_r·|ω|/(1−a_r) ≤ exit_rot_tol − err`. `exit_rot_tol` is the profile tolerance, or `min(tol, rotation_eval_max_rad)` when the next phase is a contact-profile phase, so that the guard can evaluate from the first contact step (M7).
4. **Translate phase.** EXC-B2 route plus EXC-B3 tracker; orientation is held at the target.
5. **Coupled phase** (only when selected): translation arc progress ≤ rotation progress fraction × L + `m_track`, where L is the path length and `m_track` comes from B2. The allowed lead is therefore the tracking margin, not a new constant.
6. **Rotate-phase monitoring.** `rotation_blocked`, `finger_touch` and `no_progress` (rotation) are active. Translation drift during rotation is held by the continuous hold command.

**Vetoes and parameters:**
- **Removed:** the stage rotation veto (1.2 rad transit, 0.5 rad contact).
- **Added:** none. Exit tolerances are existing tolerances or the EXC-0 `rotation_eval_max_rad`; the coupled lead is `m_track`.
- **Optional B1-opt** (user decision UD4): a higher rotation command cap inside rotate phases at cleared sites. It needs the scripted benchmark to refit (a_r, b_r) at that cap.

**Depends on:** EXC-0, EXC-B3, EXC-B2 (oracle protocol and clearance).

**Offline tests:**
1. The recorded T04 and T08 first decisions (1.5724 / 1.5691 rad, 0.134 / 0.233 m) are accepted with no veto, mode `rotate_then_translate` at the start, and `predicted_steps.rotate` within ±15 % of 1.57/0.0229 ≈ 68.
2. A profile simulation on the plant matches the recorded rotation-in-place segment (T04 steps 1–33) within 0.1 rad per step.
3. **T08 geometry fixture**, using diagnostic truth for the test only: the bottle box from `layout_manifest.json`, plus finger keypoints from `robot_facts_v2.json` (finger_and_pad_R1 ±66.4 mm, tip 9.25 mm). The rotate-at-start profile of the second stage (start z 1.077, target z 1.035) gives fingertip clearance ≥ 0, while the recorded (measured) profile intersects at steps 37–45, matching verify_h1_collision F1. The check is done by the keypoint sweep in the test, not at runtime.
4. Contact-profile stages with θ > 0.04 rad exit the rotate phase at ≤ 0.02 rad, and the first approach step is guard-evaluable.
5. The JSON schema is valid and samples are canonical (w ≥ 0).

**Live validation:**
- Capped smoke T08 + bottle: `target_envelope_rejected` = 0. The first decision executes. Telemetry shows the rotate phase at z ≈ 1.07–1.18, then translation. 0 finger–bottle contact episodes (the H1 baseline had one 47-step episode, 3,504 points).
- Sweep: envelope vetoes = 0; contact episodes during rotate phases by class; added steps per large-rotation stage.

**Expected effect (estimates):**
- −7…−8 s per veto occurrence (repair calls 7.6 s and 7.2 s; verify F6, cross C6); first plans valid in T04 and T08.
- Removes the rotation-lag finger sweep mechanism of T08 when the stage starts above the obstacle. This is supported by the verifier's truth sweep (the final pose is clear at 0 clearance; the measured in-place rotation is what hit).
- **Cost:** in the interim, `rotate_then_translate` adds the translation steps after the rotation, ≈ +10–25 steps (+0.5–1.25 s simulated) per large-rotation stage. This is about 2 stages per T04/T08 run and is recovered once `coupled` is certified by the body sweep.

**Risks:**
- The interim site choice uses model boxes (R1 errors). It still works for descents, because the start is the high pose, but the held object is not modelled. Mitigation: the rotate phase is guarded, and the body sweep comes next.
- The rotate-first time cost (above).
- Rotation of a carried object swings it. Mitigation: deferred to ws_swept_planner's held-object model via `held` in the profile.

**Effort:** 3.0 engineer-days.

---

### EXC-B2 — Exact distances, explicit margin, checked escape, and the oracle hand-over

**Fixes:**
- README B2 and §3.1 item 12 (lattice rule) and item 13 (`route_start_blocked`).
- cross_critic IC4 and IC13 (escape semantics before automatic boxes).
- verify_h1_collision F4 and M4 (replace the hidden margin rather than just delete it).
- verify_h1_speed F5.2 (the exit leg must strictly increase distance from the unexpanded box; do not exit vertically by default).

**Files:**
- Create: `executor/box_clearance.py`, `executor/clearance_oracle.py`, `executor/transit_route.py`, `tests/test_box_clearance.py`, `tests/test_transit_route2.py`.
- Modify: `fragments.py`. `_enter_segment` (`:293-317`) uses `TransitRoute2`; `_prepare_route`/`_observe` (`:323-331`, `:362-367`) do one local replan before handing off.

**`box_clearance.py`** (numpy, vectorised over N points and M boxes):
```python
def point_box_signed_distance(p: (N,3), lo: (M,3), hi: (M,3)) -> (N,M)   # >0 Euclidean outside, <0 −penetration
def point_box_gradient(p, lo, hi) -> (N,M,3)        # unit: outward from the closest point; inside: nearest-face normal
def segment_box_distance(a: (N,3), b: (N,3), lo, hi) -> (N,M)   # exact min over t∈[0,1]; slab test for 0,
                                                                 # else golden-section on the convex distance (≤1e-9 m)
```
`ExactBoxField(FineCompactField)` overrides `_point_valid` and `query` with the exact closed-box test. The 8-corner rule at `fine_distance_planner.py:101-106` is gone. `segment_valid` (`:135-155`) and the lazy A* (`:191-327`) run unchanged on boxes expanded by the effective margin.

**`ClearanceOracle`** (in-process protocol; the hand-over point to ws_swept_planner):
```python
class ClearanceOracle(Protocol):
    oracle_id: str          # "grip_site_boxes_v1" (this ws) | "body_sweep_v1" (ws_swept_planner)
    def point_clearance(self, pose: PoseWXYZ, *, gripper_gap_m: float, held: HeldObject | None) -> Clearance
    def segment_clearance(self, a: PoseWXYZ, b: PoseWXYZ, *, gripper_gap_m: float, held) -> Clearance
    def profile_clearance(self, profile: ExecutedProfile, *, held) -> ProfileClearance
    def gradient(self, pose: PoseWXYZ, *, gripper_gap_m: float, held) -> np.ndarray   # unit translation direction
@dataclass(frozen=True)
class Clearance:
    signed_m: float          # unexpanded geometry; negative = penetration
    obstacle_id: str | None; robot_part: str; required_margin_m: float
    source: str              # "model_box" | "perception_box" | "occupancy"
    captured_at_step: int | None
```
`ProfileClearance` (JSON for logs):
```json
{"oracle_id":"…","min_clearance_m":0.012,"argmin_k":41,"robot_part":"finger1","obstacle_id":"obj_3","violations":[{"k":41,"part":"finger1","obstacle_id":"obj_3","depth_m":0.004}],"time_ms":0.8}
```
`GripSiteBoxOracle` (interim) uses the grip-site point against the obstacle-map boxes (ids `model_box_<i>` until object ids exist), with `robot_part = "grip_site"`.

**TransitRoute2:**
```python
class TransitRoute2:
    def __init__(self, start_xyz, goal_xyz, orientation_wxyz, obstacle_map: dict, *, plant: OSCPlant,
                 oracle: ClearanceOracle, cap_m: float, tol_m: float, s_entry_m: np.ndarray,
                 gripper_gap_m: float, held=None): ...
    def command(self, position, s_prev) -> tuple[np.ndarray, dict]
    def progress(self, position) -> float
    def diagnostics(self) -> dict     # margins, escape, path, oracle_id, planning_s
```

**Algorithm:**
1. **Margins.**
   - `model_clearance` = `clearance_m` from the map, or per-object uncertainty once ws_geometry_tools supplies it.
   - `m_track` = `a·|s_entry,⊥|/(1−a) + 3·σ_track`, where σ_track = residual_std·(1−a²)^−½. This is the plant-predicted lateral excursion from the measured entry momentum plus the closed-loop noise: derived per stage, no constant.
   - `effective = model_clearance + m_track`. All three are reported.
2. **Start.**
   - Compute `d0 = oracle.point_clearance(start).signed_m`. If `d0 ≥ effective`, there is no escape.
   - Otherwise plan an **escape leg**:
     - Candidates are unit directions: (i) the gradient of the most violated box; (ii) toward the goal; (iii) +z; (iv) the reverse of the last measured motion.
     - Keep only candidates whose dot product with the violated-box gradient is > 0, i.e. that strictly increase distance from the **unexpanded** box. Distance along a ray from outside a convex box with a positive initial slope is non-decreasing (a convex function of t).
     - Exit length L = the smallest L with `d_i(p0 + L·u) ≥ effective_i` for all boxes, while no other box's segment distance falls below `min(d_j(p0), effective_j)`.
     - Choose the smallest L; break ties by goal alignment. +z gets no preference (README B2: "Don't escape vertically by default").
     - If no candidate passes, refuse with `route_unavailable/start_blocked_no_escape`, carrying the exact numbers.
3. **Goal.** `dg < model_clearance` raises `route_goal_blocked`, with exact distances in the evidence. The veto is retained but becomes exact.
4. **Path.** Use `ExactBoxField` with boxes expanded by `effective`, straight or A* from the exit point.
5. **Tracking validity.** The measured `d(p) ≥ required(arc)`, where `required` = `min(model_clearance, d0)` on the escape leg and `model_clearance` afterwards. The path has `m_track` of slack, so dynamic deviation inside the slack never aborts.
   - On violation: hold, then **one local replan** (escape + route from the measured point), reported.
   - A second violation gives a `route_unavailable` handoff.
6. **Records.** `route = {status: direct|detour|escape+direct|escape+detour|unavailable, effective_margin_m, model_clearance_m, tracking_margin_m, start_clearance_m, goal_clearance_m, min_path_clearance_m, escape:{from_xyz,to_xyz,direction,length_m,violated_ids,d_before_m,d_after_m}, oracle_id}`.

**Vetoes and parameters:**
- **Removed:** the hidden 0–20 mm lattice inflation, and the `route_start_blocked` refusal (it becomes an escape; refusal only when no strictly distance-increasing exit exists).
- **Added:** `m_track`, derived per stage from the plant and the measured entry momentum.
- **Interim option (UD2):** add +10 mm, the mean of the removed hidden inflation, to keep H1's average effective clearance until the body sweep lands.

**Hand-over to ws_swept_planner:**
- The config key `executor.clearance_oracle ∈ {grip_site_boxes_v1, body_sweep_v1}` is recorded in the build digest. Route, escape, rotation-site choice and EXC-C1's nearest-part feedback call only the protocol, so the swap needs no executor code change.
- The swap is made **only together with perception boxes**, per IC12/IC13. With the model's own boxes a hand sweep refuses or escapes 21–23 of 111 accepted stages (verify F1).
- `validation/replay_executor_logs.py --oracle both` reports per-stage disagreement before the swap.
- After the swap, `model_clearance` becomes per-object perception uncertainty, and `m_track` stays.

**Depends on:** EXC-0 (for `m_track`), EXC-B3 (tracker).

**Offline tests:**
1. Exact distances against brute force (10⁵ random points and segments, dense sampling at 0.1 mm): |error| ≤ 1e-9 m for points and ≤ 1e-6 m for segments; the intersection flag agrees with `DistanceFieldPlanner._avoids_boxes`.
2. **Replay of the 7 recorded start refusals** (maps and starts from the logs):
   - the 5 with signed distance +6.3…+20.3 mm route without escape, or with an escape of ≤ `m_track`;
   - T04-20mm (−0.4 mm) and stove-off (−52.4 mm) get escapes whose leg is monotone in distance to the violated box, with length ≤ effective + penetration;
   - no leg reduces any other box's distance below its effective margin.
3. **Replay of the 111 straight-valid routed stages:** `TransitRoute2` is straight in all of them, except those listed with exact margin numbers; every A* path is valid under exact checks.
4. The T08 true-box replay is reported, not asserted: point routing still allows the finger sweep (verify F1). This test documents the limitation that ws_swept_planner removes.

**Live validation:**
- Capped smokes T04 + can and T08 + bottle: `route_unavailable` = 0, unless it is `start_blocked_no_escape` with numbers; escapes are logged with lengths; 0 contact episodes during escape legs.
- Sweep: refusals and escapes per task; effective-margin distribution; contact episodes in routed stages (class B) against H1.

**Expected effect (estimates):**
- 5/7 recorded refusals disappear; 2/7 become reported escapes.
- About 6 repair calls (≈ 70 s) fewer across the 9 runs (verify F4).
- The router is not a body model; collision gains come with the body sweep and perception boxes.

**Risks:**
- The interim effective margin drops by 0–20 mm against H1. The hidden margin gave no demonstrated protection (the true-box T08 replay ended with A* within 1 mm of the colliding descent). Mitigation: UD2 and the guard.
- An escape could push into an unmodelled object. Mitigation: legs are short (≤ margin + penetration), guarded and reported.

**Effort:** 3.5 engineer-days.

---

### EXC-C5 — Vendored Panda kinematics, reachability, joint margins, posture recovery

**Fixes:**
- README C5 and §3.1 item 15.
- h1_speed F10.
- verify_h1_collision M2.
- The drawer 001 9-call J4 streak.

**Files:**
- Create: `vendor/robodojo_direct_sim.py` (byte copy of `/workspace/gpt_field/src/gpt_field/robodojo_direct_sim.py`, sha256 `6d2746cbf1d52cb0b4ebc8de80e5e4dafa6449dc95a9df27d887c6fb3c9971bb`), `vendor/panda_assets/robot.xml` (`263c0f9abacbc8686027ec5545bee144481cbcf5cc74aa421cbe3227030cd95e`), `vendor/panda_assets/panda_gripper.xml` (`d9d68123909d6910488310d0fae4e358ed4ac831c6d0957f3e30893a5412baac`), `vendor/VENDOR.json`, `executor/kinematics.py`, `executor/posture.py`, `tests/test_kinematics_vendor.py`, `tests/test_reachability_replay.py`.
- Modify: `agent.py` `_execution_state`/`_context` (`:186-232`), to add `joint_margins_rad` and the limiting joint.
- **Not** vendored for C5: `foundation_harness/robot_geometry.py` (sha256 `8fa39c48…`, scipy). Its hulls belong to ws_swept_planner, which should precompute them offline into numpy arrays and reuse this FK chain.

**Interfaces:**
```python
class PandaModel:                                   # loads vendored XMLs; verifies the 3 hashes
    VERSION = "panda-model-1"; low: np.ndarray; high: np.ndarray
    @classmethod
    def load(cls) -> "PandaModel"
    def bind(self, state8, q7) -> "BoundPanda"      # world_from_base from measurement (M10)
class BoundPanda:
    world_from_base: np.ndarray; base_residual_m: float      # vs reset bind; health metric
    def fk(self, q) -> tuple[np.ndarray, np.ndarray]          # T_world_grip (right_hand R), J 6×7
    def link_origins(self, q) -> np.ndarray                   # (8,3) joint origins + hand
    def joint_margins(self, q) -> np.ndarray                  # min(q−low, high−q)
    def ik_local(self, xyz, R, seed) -> IKReport              # vendored DLS, 140 it., OSC-like
    def ik_global(self, xyz, R, seeds) -> IKReport            # seeds: measured, reset, 6 fixed perturbations
    def reachability(self, samples: list[tuple[np.ndarray, np.ndarray]], q_now, q_reset) -> ReachabilityReport
```

**IKReport:**
```json
{"ok":true,"position_error_m":0.0,"orientation_error_deg":0.0,"iterations":23,"joints_rad":[…],"min_margin_rad":0.79,"limiting_joint":4,"clamped_joints":[],"seed":"current"}
```

**ReachabilityReport:**
```json
{"version":"reach-1","classification":"reachable|posture_limited|unreachable|unavailable","local":{IKReport},"global":{IKReport}|null,"path_samples":6,"path_min_margin_rad":0.21,"path_limiting_joint":4,"time_ms":11.2}
```

**Algorithm:**
1. **Health check at every reset and every 20 steps.** Bind, then track `base_residual_m`. The measured spread is 0.27–0.61 mm (M10). Above 1 cm, the classification becomes `unavailable` (fail open, reported). This is a sanity bound on the model, not a task threshold.
2. **At stage entry (arm skills).** Take ≤ 6 samples of the `ExecutedProfile` (entry, rotate end, ¼, ½, ¾ of the path, goal) and run `ik_local` sequentially, seeded by the measured q and then by each previous solution.
   - `reachable`: everything ok and no clamped joint.
   - `posture_limited`: local fails or clamps, but `ik_global` of the goal is ok.
   - `unreachable`: global fails.
3. **No veto.** The stage executes in all cases. The report rides in the segment record. If EXC-C1 later gets a block or no-progress event with reachability ≠ reachable, the cause is `reachability_limited`, with joint margins, the limiting joint, the global solution margin and the nearest reachable pose (FK of the best global solve).
4. **`joint_margins_rad` (7)** and the limiting joint appear in `measured_state` every call. The static limits go in `capabilities` once.
5. **C5b `PostureRecoveryPhase`**, implemented but **disabled** by config until the scripted simulator benchmark validates it (M12). Candidates, selected by the benchmark:
   - (A) reduce reach along the shoulder→grip line with orientation held;
   - (B) follow `FK(q(λ))`, the Cartesian image of a joint interpolation to the global IK solution;
   - (C) a temporary joint-space controller in the worker (UD5).
   - Termination: `ik_local` from the measured q succeeds without clamping (checked every 4 steps, ~7 ms), or a guard event, or a plant-derived budget.
   - The phase is guarded and oracle-checked like an escape.
6. **Optional (UD6) jaw-symmetric equivalent.** If the target is `unreachable` but `target·Rz(π)` about the approach axis is reachable, use it and report `jaw_symmetric_equivalent_used`.

**Vetoes and parameters:**
- **Added:** none tuned. The classifications use the solver's own success criterion (≤ 1.5 mm / 1°; `robodojo_direct_sim.py:239`) and joint-limit clamping, which is parameter-free. The 1 cm health bound is a model sanity check.
- **Removed:** the model's limit-blindness (it now sees the margins).

**Depends on:** EXC-B1 (profile samples; it can start with the goal sample only), EXC-0 (recovery budget).

**Offline tests:**
1. Vendored hashes equal the pins. The module imports in the controller `.venv` with numpy 2.5.3 and without scipy.
2. FK bound on every logged request of T04, T08, d001 and d003 reproduces the logged grip pose to ≤ 1e-6 m and ≤ 1e-6 rad. Base spread ≤ 1 mm per episode.
3. **Reachability replay** over the 107 logged targets:
   - d001 calls 16, 17, 21, 22, 23 are `posture_limited` (J4); call 24 is flagged;
   - d003 call 4 is `unreachable` (J5); call 8 has margin < 0.02;
   - 0 of 31 T04/T08 targets are flagged;
   - p95 time ≤ 25 ms.
4. Limits equal the robot.xml ranges (J4 [−3.0718, −0.0698], etc.).
5. C5b has no offline pass criterion; the benchmark is its gate (EXC-INT).

**Live validation:**
- Capped smoke `drawer_bowl` (5-stage, where the d001 streak happened): record reachability classes, joint margins at every handoff, and calls spent at margin < 0.05 rad.
  - Target with C5a: ≤ 2 calls per joint-limit event (H1: 9-call streak).
  - Target with C5b enabled: 0 calls for posture-limited events.
- Sweep: reachability flags per task; false `unreachable` (the stage later arrived) = 0.

**Expected effect:**
- C5a turns "cause unknown" into "posture-limited at J4; the target is reachable from another posture (0.55–1.19 rad margin)". That plausibly shortens the 9-call / 141 s streak (hypothesis).
- C5b, if the benchmark validates a mechanism, gives about −8 calls / −130 s on d001-type tasks (h1_speed F10, an upper bound).

**Risks:**
- No Cartesian recovery is shown to work offline (M12). Mitigation: ship C5a, and gate C5b on the benchmark.
- DLS false `unreachable` from local minima. Mitigation: multi-seed global solve; `unreachable` never vetoes.
- Vendored drift against robosuite. Mitigation: hash pins and the base health check.

**Effort:** 3.0 engineer-days (C5a 2.0; C5b 1.0, plus benchmark time in EXC-INT).

---

### EXC-C1 — Guarded-motion response: hold or back off, local classification, `blocked_motion` feedback

**Fixes:**
- README C1, R5, and §3.1 item 11 (no contact sensing; "cause unknown" at `fragments.py:482`).
- h1_collision F5.
- verify_h1_collision F5. A naive handoff would add ~24 calls; the guard should treat in-band creep as arrival, resolve contact-free (joint-limit) blocks locally, and use thresholds relative to the measured OSC realisation. The guard cannot see contacts during `set_gripper`.
- The direct-motion gap (135 of the 414 drawer link samples happened in `local_motion`).

**Files:**
- Create: `executor/guard_policy.py`, `executor/feedback.py`, `executor/schemas/segment_result.schema.json`, `tests/test_guard_policy.py`.
- Modify:
  - `fragments.py` `_observe` (`:348-420`) dispatches monitor events to `guard_policy`; `progress()` (`:449-483`) carries a `cause` instead of `"unknown"`.
  - `agent_speedup_v2_user_check_motion.py` `LocalMotionPrimitive._observe` (`:307-311`) attaches monitor evidence (reported, not stopped, given its ≤ 5 steps).
  - `agent.py` `_complete_direct` (`:382-401`) copies the structured result.

**Interface:**
```python
class GuardPolicy:
    def resolve(self, event: MonitorEvent, ctx: SegmentContext) -> Resolution
@dataclass
class Resolution:
    action: Literal["succeed","backoff_then_end","posture_then_retry","replan_once","end"]
    reason: str; record: dict           # segment_result fields below
```

**`segment_result`** (schema; ≤ 1.2 KB per segment; reason codes stable and kept verbatim through compaction, IC15):
```json
{"index":2,"skill":"approach","status":"succeeded|failed|preempted",
 "reason":"pose_arrived|arrived_in_band|contact_reached|gap_stable|wait_elapsed|preempted|blocked_motion|rotation_blocked|no_progress|diverging|unexpected_contact_early|grasp_object_absent|grasp_gap_out_of_band|held_object_lost|maintained_grip_gap_changed|segment_budget_exhausted|route_unavailable|reachability_limited",
 "residual":{"position_m":0.003,"orientation_rad":0.01,"along_command_m":0.002},
 "sequencing":{"mode":"rotate_then_translate","site":"start","rotation_rad":1.57,"rotate_steps":66,"translate_steps":21},
 "route":{…EXC-B2 record…},
 "reachability":{"classification":"reachable","limiting_joint":4,"min_margin_rad":0.45},
 "contact":{"declared":true,"channel":"progress_guard|finger_touch|no_progress","at_step":31,"depth_from_entry_m":0.052,"remaining_to_target_m":0.006,"within_window":true},
 "blocked":{"direction_world":[0,0,-1],"commanded_offset_m":0.005,"depth_reached_m":0.041,"remaining_m":0.019,
            "guard":{"window_steps":3,"predicted_m":0.0036,"realised_m":0.0004,"ratio":0.11},
            "gripper":{"command":"open","gap_mm":77.5,"gap_change_mm":-0.1},
            "nearest_robot_part":{"part":"finger1|palm|link6|…","obstacle_id":"model_box_1","distance_m":0.004,"source":"grip_site_boxes_v1|body_sweep_v1"},
            "joint_margins_rad":[…7…],
            "back_off":{"executed_steps":3,"from_xyz":[…],"to_xyz":[…],"status":"done|blocked"}},
 "grasp":{"settled_gap_mm":16.9,"settle_k":14,"empty_curve_gap_mm":2.5,"object_present":true,"gap_band_mm":null}}
```

**Algorithm (classification order, all local, no model call unless stated):**
1. `arrived_in_band` (EXC-B3 rule) → success.
2. Declared contact (`stop_on: contact`) within its window → `contact_reached` success, hold, continue.
3. `reachability ≠ reachable` and the limiting-joint margin ≤ its value at entry (the arm is moving into the limit). If C5b is enabled and has not yet run in this segment: `posture_then_retry`. Otherwise end with `reachability_limited`, which costs one call.
4. A route-validity violation (B2): `replan_once`.
5. Otherwise, an undeclared block or contact:
   - **hold** (zero offset, 1 step; this releases the OSC pressing force ∝ offset);
   - **back off** to the measured pose at the start of the guard window. That is the last nominal pose, so the distance is measured, not a constant. The back-off uses the contact profile, is guarded, and ends early if blocked;
   - end with `blocked_motion`, carrying the structured record.
   - `nearest_robot_part` comes from the oracle. In the interim this is FK link origins plus gripper keypoints (`robot_facts_v2.json` hand and finger footprints) against the map boxes, labelled as a model-box estimate.
6. `rotation_blocked` gets the same treatment, with the rotation channel's evidence.
7. **Limits stated plainly in the record and the capabilities:** the guard fires after onset, so it shortens contacts rather than preventing them. It cannot see `set_gripper` contacts (no commanded motion). Jaw-sweep checks before closing belong to ws_swept_planner (shared interface).

**Vetoes and parameters:**
- **Removed:** "cause unknown"; the stall-as-failure for declared contacts.
- **Added:** none beyond EXC-0. The back-off distance is measured (the window start pose).

**Depends on:** EXC-B3, EXC-C5 (C5a), EXC-B2.

**Offline tests:**
1. Replay of the 29 forbidden episodes through monitor and policy (counterfactual, logs only):
   - classification counts are produced;
   - ≥ 13 detected, with median delay ≤ 4 steps;
   - each projected handoff is mapped to the H1 call it replaces (stall or veto), and the report counts *additional* handoffs. Pass: ≤ 3 additional handoffs over all 9 runs (the verifier's naive design: +24).
2. Synthetic fake simulator:
   - a declared contact continues into the pending `set_gripper`;
   - an undeclared contact backs off to the window-start pose within ±1 mm;
   - a joint-limit block with C5b disabled ends as `reachability_limited`, with margins.
3. Each record validates against `segment_result.schema.json` and stays ≤ 1.2 KB.

**Live validation:**
- Capped smokes T04, T08 and `drawer_bowl`, with sweep metrics:
  - guard fires by classification;
  - pressing duration per contact episode, onset to stop (target median ≤ 4 steps; H1 stall median 13);
  - `blocked_motion` handoffs with no contact in the evaluator log (false handoffs; target ≤ 1 per task);
  - calls after `blocked_motion` against calls after H1 stalls.

**Expected effect (estimates):**
- About 70 % less pressing time, an upper bound (h1_collision F5).
- Net calls ≈ 0, against +24 for a naive handoff.
- Model diagnoses from stated causes rather than "unknown".

**Risks:**
- False blocks near joint limits or clutter. Mitigation: in-band and reachability classification; the false-handoff metric.
- A back-off into a neighbour. Mitigation: the back-off retraces the measured approach, so its path was free a few steps earlier.
- Soft pushes (palm on rim while the object slides) are not detected. M8: the 3 T04 palm episodes were in close, release and lift phases, not blocks. That class needs the palm envelope and geometry from the other workstreams.

**Effort:** 2.5 engineer-days.

---

### EXC-D1 — Committed multi-phase sequences: caps and measured continuation predicates (executor side)

**Fixes:**
- README D1 and §3.1 item 2.
- verify_h1_speed F2 correction 2 (the caps make the fix impossible).
- cross_critic IC5 and G7 / bottom line 5 (grasp failures are gap-detectable; the predicate must key on intent).

**Files:**
- Create: `executor/predicates.py`, `tests/test_continuation_predicates.py`.
- Modify `fragments.py`:
  - constants `:24-26`;
  - schema `:48-74`;
  - `validate_execution` `:113-176` (delete the total-budget check `:174-175`);
  - `_observe` gap logic `:370-398`;
  - delete `:402-403`;
  - `_complete_segment` `:422-434` (unchanged continuation, plus predicate gating);
  - add `preempt()`.
- Also modify `agent.py` `_apply_agent` `:351-362` to pass the held-band state between fragments.

**Caps:**
- `MAX_SEGMENTS` 6 → **12**. A pick-and-place is 8 phases (approach, descend, close, lift, carry, lower, open, retreat), plus room for rotate, hover and escape.
- `MAX_SEGMENT_STEPS` 120 → **240**. Plant-derived: a π rotation at 0.0229 rad/step is 137 steps, a 0.5 m transit plus braking about 30, arrival 2; ≈ 170 × 1.4.
- `MAX_EXECUTION_STEPS` **removed** (UD3). Per-segment budgets plus the predicates bound the fragment. ws_runner_measurement's evaluation cap is the runaway guard.
- `max_steps` becomes nullable. Null means a budget of `ceil(plant.predicted_steps(...) × overrun)`. The overrun factor is the p99 of actual/predicted steps from the benchmark (logged; 2.0 until measured).

**Predicates (`predicates.py`):**
```python
def object_present_after_close(gap_curve_mm: list[float], settle_k: int, cal: GripperCalibration) -> dict
    # present ⇔ settled_gap > cal.empty_gap_at(settle_k) + cal.touch_threshold_mm   (classify_close semantics)
def held_gap_ok(gap_mm: float, cal: GripperCalibration, band: tuple[float, float] | None) -> dict
    # lost ⇔ gap ≤ cal.empty_floor_mm + cal.touch_threshold_mm ; band: lo ≤ gap ≤ hi if declared
def contact_in_window(remaining_m: float, window_m: float | None, tol_m: float) -> bool
```

**Semantics:**
- `set_gripper close` with `expect.object_in_gripper = true`: at `gap_stable` (`:387-394`), `object_present_after_close` must hold, else `grasp_object_absent`. If `gap_band_mm` is given, the settled gap must also lie in it, else `grasp_gap_out_of_band`.
- On success the fragment enters a **held state**. Every later arm segment, while the close command persists, checks `held_gap_ok` each step; a failure holds and yields `held_object_lost`.
- `object_in_gripper = false` (a deliberate closed-finger push) disables both checks; the drawer 1.4 mm pushes need this.
- The legacy gap-rise yield (`:371-382`, 6 mm / 3 samples) stays, unless a band is declared; then a rise above `hi` replaces it.
- Pending segments continue on success (`:432-434`, as today). The executor yields only on a predicate failure, a guard handoff, `observation_after` (kept for placement relations and articulated engagement) or budget.

**Preemption hook** (for ws_protocol_context's asynchronous verification):
```python
def preempt(self, reason: str, replacement: dict | None = None) -> None
    # effective at the next step boundary; current segment → status "preempted"; replacement validated and
    # started from the measured state; evidence records the preempted remainder
```

**Vetoes and parameters:**
- **Removed:** the 6/120/240 caps as constants (two raised by plant derivation, one removed).
- **Added:** no tuned thresholds. The predicates use the measured empty-close curve and noise (the existing `GripperCalibration`) and model-declared, per-task intent data (`object_in_gripper`, an optional band).

**Depends on:** EXC-C1 (IC5: guard before commit), EXC-B3 (`stop_on`), EXC-0.

**Offline tests:**
1. On the 27 recorded closes (`grasp_gap.json`), with intent labels from the recorded beliefs:
   - both failures are flagged (`grasp_object_absent` for the 1.4 mm settle; `held_object_lost` for the 13.3 → 1.0 mm collapse);
   - 0 of 19 confirmed grasps are flagged;
   - the deliberate pushes are not flagged when `object_in_gripper = false`.
2. A 12-segment pick-and-place on a fake simulator runs end to end without a handoff.
3. A 13-segment execution is rejected.
4. `preempt` mid-segment ends at the next boundary with a correct record.
5. Legacy H1 decisions (no `expect`, integer `max_steps`) still validate, which keeps legacy replays possible.

**Live validation:**
- Capped smoke T04 + can, **only together with ws_protocol_context's prompt change**: calls per object (H1: 6.5); late-detected grasp failures (a failure the model saw only after a carry) = 0; predicate yields by reason.
- Sweep: calls per object; strict success at finish (ws_runner_measurement).

**Expected effect:** about −4 calls (−60 s) on T04 (README D1; a hypothesis until the sweep).

**Risks:**
- A committed carry pushes through an undetected contact for longer (IC5). Mitigation: C1 ships first; late failures are counted.
- Mis-declared intent. Mitigation: the yield reason names the predicate; a wrong band costs one call.

**Effort:** 2.0 engineer-days.

---

### EXC-D2 — Distance-scheduled approach speed (executor side, later)

**Fixes:**
- README D2.
- cross_critic IC1 (3 of 15 onsets began 2.2–6.8 cm early; the slow zone must be measured uncertainty plus braking distance).
- verify_h1_collision F10 (the standoff must also cover stopping distance).
- h1_speed F6.

**Files:** modify `executor/sequencing.py` (an approach split into a fast phase and a contact phase) and `fragments.py` (`stop_on` with a measured window); create `tests/test_distance_schedule.py`.

**Interface (input):**
```json
{"target_id":"obj_2","expected_contact_xyz_m":[…],"approach_axis":[0,0,-1],"uncertainty_95_m":0.012,
 "source":"triangulated|support_plane|perception_box","captured_at_step":431}
```
Provided by ws_perception_service and ws_geometry_tools; see §5.

**Algorithm.** For a segment with `stop_on: contact` whose contact expectation has a **measured** source:
- `slow_zone = uncertainty_95 + coast(s_fast) + W·ρ·c_contact`, i.e. measured uncertainty + braking distance + guard detection distance (all from EXC-0).
- The fast phase runs at the transit cap with the braking law targeting `remaining − slow_zone`.
- The contact phase runs at the contact cap with the guard.
- Without a measured source, the whole approach stays at contact speed (today's behaviour).

**Vetoes and parameters:** none added; the slow zone is entirely derived.

**Depends on:** EXC-C1, EXC-B3, EXC-0, plus the external uncertainty input (IC1 order: guard → measured geometry → speed).

**Offline tests:**
1. A plant simulation with an obstacle placed at `uncertainty_95` before the expected contact: the contact-onset speed is ≤ the contact-profile steady speed (ρ·5 mm = 1.27 mm/step).
2. Replay of the 15 recorded onsets with their actual distances: under a measured window, no onset falls inside the fast phase.

**Live validation:** sweep. Contact-approach duration; onset speed distribution; onsets inside fast phases = 0; forbidden contact episodes not above the EXC-C1 build.

**Expected effect:** about −10…−67 s per task at a 4 cm standoff (verify F10). Larger with tighter measured uncertainty.

**Risks:**
- Wrong uncertainty means a fast impact. Mitigation: gate on the source type; the guard is active in both phases.

**Effort:** 1.5 engineer-days.

---

### EXC-INT — Integration: fragments v5, wiring, telemetry, offline replay suite, scripted simulator benchmark

**Fixes:**
- Makes the above one coherent build.
- R8: an executor replay that is a development aid, not a launch gate.
- Gives C5b and the plant a non-model calibration path.

**Files:**
- Modify:
  - `agent_speedup_v2_user_check_fragments.py`: `FRAGMENT_VERSION = "user-check-measured-fragments-5"`, schema v5, the `fragment_capabilities()` text `:179-213`.
  - `agent_speedup_v2_user_check_agent.py`: construct with `OSCPlant`, `PandaModel`, oracle and calibration (`:351-362`); per-step telemetry into `diagnostics["executor"]` (`:551-576`).
  - `agent_speedup_v2_user_check_routing.py`: keep `TransitObstacleMap`, and export the v2 map fields when ws_geometry_tools provides them.
- Create:
  - `validation/replay_executor_logs.py`: counterfactual over saved H1/H3 traces and requests; CPU-only, minutes, no model;
  - `scripts/benchmark_executor_sim.py`: scripted, no-model simulator scenarios, run via the `.venv-sim` worker. A benchmark-only `set_arm_joints` operation is added to `agent_speedup_v2_user_check_sim.py` for C5b reproduction; it is unreachable from evaluation runs;
  - `tests/test_fragment_schema_v5.py`.

**Segment schema v5** (model boundary; to ws_protocol_context):
```json
{"skill":"move_to_pose|move_relative|approach|set_gripper|wait","xyz_m":[3]|null,"quaternion_wxyz":[4]|null,
 "gripper":"keep|open|close","profile":"transit|contact","max_steps":1..240|null,"observation_after":bool,
 "route":{…}|null,
 "expect":{"stop_on":"arrival|contact","contact_window_m":0..0.5|null,
           "object_in_gripper":true|false|null,"gap_band_mm":[lo,hi]|null}|null}
```
- All keys are required, as strict mode demands.
- Validation:
  - `stop_on = "contact"` only on arm skills with profile contact (or transit when EXC-D2 holds a measured window);
  - `object_in_gripper` only on `set_gripper close`;
  - `gap_band_mm` only with `object_in_gripper = true`, and `lo < hi`;
  - segments 1..12;
  - legacy segments without `expect` validate as `expect = null`.

**Benchmark scenarios** (no model; GPU only for EGL if camera observations stay on):
1. Free-space plant refit: 60 random straight moves and 30 rotations in the workspace above the table of a LIBERO-10 reset.
2. A guarded approach pressing into the table at the contact cap: guard delay and pressing steps.
3. Large-rotation stages under each B1 mode: predicted against measured profile.
4. Reproduce d001 J4-limit configurations via `set_arm_joints`: C5b candidates A and B succeed or fail.
5. Overshoot statistics, legacy tracker against braking.

**Offline tests:** the full `tests/` suite passes in `.venv`; the replay report reproduces M1–M13 to within rounding; schema contract tests.

**Live validation:** capped smokes (§6); the user then runs the sweep on one frozen build.

**Effort:** 3.0 engineer-days.

---

## 4. Vetoes removed and parameters added (one table, rule 2)

| Removed | Where |
|---|---|
| Rotation envelope, 1.2 rad / 0.5 rad | `fragments.py:27-34`, `:288-291` |
| Hidden 0–20 mm lattice margin | `fine_distance_planner.py:101-106`, `:117-130` |
| `route_start_blocked` refusal (now an escape; refusal only without a strictly distance-increasing exit) | `routing.py:105-106` |
| Stall constants 12 / 16 / 0.8 mm and the 0.05 m/rad weight | `fragments.py:43-45`, `:404-419` |
| Tracker deadband | `tracking.py:90-95`, `routing.py:173-174` |
| Contact stall counted as failure (when declared) | `fragments.py:418-419` |
| `MAX_EXECUTION_STEPS` = 240 | `fragments.py:26`, `:174-175`, `:402-403` |
| "cause unknown" | `fragments.py:482` |

| Added | Derivation |
|---|---|
| a, b, a_r, b_r, ρ | Fitted OSC response (M1, M2) |
| Braking gain 1−a; rotation brake switch | Plant |
| Guard f, W, c_eval, rotation_eval_max | Quantiles of free-space realisation and measured noise (M6, M7) |
| In-band rule | Plant prediction over the guard window |
| `m_track` | Plant and the measured entry momentum |
| Slow zone | Measured uncertainty + plant coast + guard distance |
| MAX_SEGMENTS 12, MAX_SEGMENT_STEPS 240 | Structural, and plant-derived |
| Gripper presence and held predicates | The existing measured empty-close curve and touch threshold |
| `object_in_gripper`, `gap_band_mm`, `contact_window_m` | Per-task data declared by the model or perception, not harness constants |

No per-task code or threshold is added.

---

## 5. Shared interfaces (provider → consumer)

| # | Name | Provider | Consumer | Shape |
|---|---|---|---|---|
| S1 | Segment schema v5 + capability text + reason codes | ws_executor_control | ws_protocol_context | §EXC-INT schema; reason vocabulary in §EXC-C1. Compaction (`_agent_client.py:316-320`, the `retained` tuple) must add `contact`, `blocked`, `reachability`, `sequencing`, `grasp`, `route.escape`, `cause`. Prompt deltas needed: delete `:135-136` (rotation envelopes); replace `:133-134` (caps); replace the mandated test-lift checkpoint at `:161-163` with "declare `expect.object_in_gripper`; the executor verifies by gap and yields on failure"; explain `stop_on` and `contact_window_m`; state the guard's limits. |
| S2 | `segment_result` feedback record | ws_executor_control | ws_protocol_context, ws_runner_measurement | `executor/schemas/segment_result.schema.json` (§EXC-C1), ≤ 1.2 KB |
| S3 | `ExecutedProfile` | ws_executor_control | ws_swept_planner | §EXC-B1 JSON, plus a Python dataclass with `samples` as an (N,8) array `[x,y,z,qw,qx,qy,qz,gap]` and `phase` labels |
| S4 | `ClearanceOracle` protocol (+ `Clearance`, `ProfileClearance`) | ws_swept_planner (`body_sweep_v1`); ws_executor_control (interim `grip_site_boxes_v1`) | ws_executor_control | §EXC-B2 Python protocol; per-call budget ≤ 5 ms for points and ≤ 20 ms for a profile, so it fits stage entry |
| S5 | Object map v2 (ids, roles, per-part contact permissions, uncertainty, timestamps) | ws_geometry_tools / ws_perception_service | ws_executor_control (oracle, margins), ws_swept_planner | `{"revision":int,"objects":[{"id","role":"obstacle|target|held|support|unknown","aabb":{"lower_xyz_m","upper_xyz_m"},"uncertainty_95_m","source":"model|perception","captured_at_step","contact_permissions":{"fingers":bool,"palm":bool,"links":bool}}],"clearance_m"}`. The executor re-checks `captured_at_step` at stage entry and records the age (IC14). |
| S6 | Contact expectation | ws_perception_service / ws_geometry_tools | ws_executor_control (EXC-D2, `contact_window_m` default) | §EXC-D2 JSON |
| S7 | Grasp and release events (attach or detach the held object) | ws_executor_control | ws_swept_planner, ws_geometry_tools | `{"event":"grasp|release","step","segment_index","settled_gap_mm","object_present":bool,"grip_pose_wxyz":[7],"declared_object_id":str|null}` |
| S8 | `PandaModel` / `BoundPanda` (vendored numpy FK/IK/Jacobian; the one FK source for the build) | ws_executor_control | ws_swept_planner (body FK), ws_geometry_tools (robot mask) | §EXC-C5 API; hashes in `vendor/VENDOR.json` |
| S9 | `osc_response.json` + plant API | ws_executor_control | ws_runner_measurement (build digest), ws_swept_planner (profile envelope) | §EXC-0 JSON |
| S10 | Per-step executor telemetry | ws_executor_control | ws_runner_measurement | `diagnostics["executor"] = {"phase","segment","c_m":[3],"c_rad":[3],"pred_m","real_m","evaluable","ratio","event","remaining_m","brake_limited","joint_min_margin_rad","limiting_joint","clearance_m","oracle_id"}` |
| S11 | Preemption hook for asynchronous verification | ws_executor_control | ws_protocol_context | `FragmentSequencePrimitive.preempt(reason, replacement=None)`; next-boundary semantics |
| S12 | Build root, launcher caps, task data, sweep tool | ws_runner_measurement | ws_executor_control | `<NEW>` repo cloned from H1 `3cc84ac`; `./run.sh --task … --gpu N --max-calls K --max-wall-s S --output …`; build digest including the executor config keys `executor.clearance_oracle`, `executor.posture_recovery`, `executor.rotation_cap_rad`, and the `osc_response.json` sha256 |
| S13 | Pre-close jaw-sweep check (the guard cannot see `set_gripper` contacts) | ws_swept_planner | ws_executor_control | `oracle.profile_clearance(close_profile)`, where `close_profile` samples the finger gap from the empty-close curve at a fixed pose |

---

## 6. Sequencing, IC compliance, commands

**Order** (inside the cross_critic safe order; each step lands in the same `<NEW>` repo, and the user sweeps one frozen build at the end):
1. **Now, offline, in parallel with ws_runner_measurement's measurement work:** EXC-0 (log-based fit), `validation/replay_executor_logs.py`, and the vendoring and C5a code.
2. **Remove spurious vetoes safely** (cross_critic step 2):
   - EXC-B3 (a–c), first, because braking removes the overshoot the guard would otherwise misread;
   - then EXC-B2;
   - then EXC-B1;
   - then C5a.
   - Capped smokes follow.
3. **Sensing** (step 3): EXC-C1, B3(d) `stop_on`, and the EXC-D1 predicates. These need S1 from ws_protocol_context in the same build.
4. **Commit** (EXC-D1 caps and prompt) only after C1 (IC5).
5. **Geometry** (steps 4–5, other workstreams): oracle swap to `body_sweep_v1` only together with perception boxes (IC12/IC13). Cruise speed scaled by the swept clearance through `plant.coast_m` (IC2); this is a ws_swept_planner input on the stage cap.
6. **EXC-D2** last, gated on measured uncertainty (IC1).
7. C5b only after benchmark scenario 4 passes.

**IC compliance:**

| Conflict | How this design complies |
|---|---|
| IC1 | D2 last, with slow zone = measured uncertainty + braking + detection |
| IC2 | Braking first; cruise scaling waits for the body sweep |
| IC3 | The veto becomes sequencing, and the executed profile is exposed |
| IC4 | Exact distances first, then a checked, reported escape; no vertical default |
| IC5 | Guard and gap predicates before commit; late failures are counted |
| IC12 | The oracle is used as a planner (escape, site choice), never as a new veto |
| IC13 | Escape semantics precede automatic boxes |
| IC14 | Map capture step is checked at use |
| IC15 | Verbatim reason codes |

**Commands** (`<NEW>` is the new build root):
```bash
# offline unit + contract tests (controller venv, CPU, no model)
cd <NEW> && PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_osc_plant.py tests/test_progress_monitor.py \
  tests/test_braking_tracker.py tests/test_rotation_sequencing.py tests/test_box_clearance.py tests/test_transit_route2.py \
  tests/test_kinematics_vendor.py tests/test_reachability_replay.py tests/test_guard_policy.py \
  tests/test_continuation_predicates.py tests/test_fragment_schema_v5.py
# plant calibration from saved runs (dev aid)
cd <NEW> && PYTHONPATH=src .venv/bin/python scripts/calibrate_osc_response.py \
  --runs /workspace/gpt_field/runs/agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001 … \
  --out src/gpt_field/executor/osc_response.json
# counterfactual executor replay over saved H1/H3 logs (minutes, CPU)
cd <NEW> && PYTHONPATH=src .venv/bin/python validation/replay_executor_logs.py --runs <9 run dirs> --oracle grip_site_boxes_v1 \
  --report runs/_dev/executor_replay_$(date +%Y%m%d).json
# scripted simulator benchmark, no model calls (choose a free GPU)
cd <NEW> && PYTHONPATH=src .venv/bin/python scripts/benchmark_executor_sim.py --sim-python .venv-sim/bin/python \
  --libero-root third_party/LIBERO --gpu <N> --scenarios plant,guard_table,rotation_modes,joint_limit,overshoot \
  --out runs/_dev/executor_bench_$(date +%Y%m%d)
# capped smokes with model calls (flags from ws_runner_measurement)
cd <NEW> && ./run.sh --task t04_can_obstacle   --gpu <N> --max-calls 20 --max-wall-s 1200 --output runs/exec_smoke_t04_001
cd <NEW> && ./run.sh --task t08_bottle_obstacle --gpu <N> --max-calls 20 --max-wall-s 1200 --output runs/exec_smoke_t08_001
cd <NEW> && ./run.sh --task drawer_bowl        --gpu <N> --max-calls 30 --max-wall-s 1800 --output runs/exec_smoke_drawer_001
# full sweep: started by the user with ws_runner_measurement's sweep tool on the frozen build
```

---

## 7. User decisions

| ID | Question | Options | Recommendation |
|---|---|---|---|
| UD1 | May the executor insert short, reported motions the model did not request (an escape leg, a back-off after an undeclared block, a local replan)? | allow all / escape only / none (refuse as today) | Allow. Each is short, guarded, distance-increasing and reported. |
| UD2 | Interim explicit margin before the body sweep | plant-derived `m_track` only / `m_track` + 10 mm (mean of the removed lattice inflation) | `m_track` only. The hidden margin showed no protection, and the escape handles starts inside the margin. |
| UD3 | Fragment budget | remove `MAX_EXECUTION_STEPS` (per-segment plant budgets) / raise it to 720 | Remove; keep 12 segments × 240 steps. |
| UD4 | Higher rotation command cap inside rotate-in-place phases at cleared sites (speed) | keep 0.1 rad / raise after benchmark recalibration | Keep for now; decide on benchmark data. |
| UD5 | Posture-recovery mechanism | report-only (C5a) / Cartesian candidates A or B after benchmark / temporary joint-space controller in the worker (changes the control interface) | Ship C5a; benchmark A and B; joint-space only with explicit approval if A and B fail. |
| UD6 | Use the jaw-symmetric orientation (π about approach) when the requested orientation is unreachable | never / when unreachable, reported | When unreachable, reported. |
| UD7 | Schema shape | a nullable `expect` object + nullable `max_steps` / flat fields | `expect` object + nullable `max_steps` (agree with ws_protocol_context). |
| UD8 | D2 gate | measured contact expectation only / also model-declared windows | Measured only (IC1). |

---

## 8. Effort

| Component | Engineer-days |
|---|---|
| EXC-0 | 2.0 |
| EXC-B3 | 4.5 |
| EXC-B1 | 3.0 |
| EXC-B2 | 3.5 |
| EXC-C5 | 3.0 |
| EXC-C1 | 2.5 |
| EXC-D1 | 2.0 |
| EXC-D2 | 1.5 |
| EXC-INT | 3.0 |
| **Total** | **25.0** |

## 9. Evidence files

- Analysis scripts and outputs: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/ws_executor_control_work/`:
  - `osc_fit.py`, `osc_fit2.py`, `plant_perrun.py`: M1, M2;
  - `rot_profile.py`, `brake_robust.py`, `rot_brake2.py`: M3–M5;
  - `guard_calib.py`, `rot_coupling.py`, `guard_eval3.py`, `guard_eval4.py`: M6–M8;
  - `finger_touch.py`: M9;
  - `fk_check.py`, `ik_pass.py`: M10, M11;
  - `posture_emul*.py`: M12;
  - `results.txt`: all outputs.
- Inputs: `…/harness_analysis/w2/data/*_steps.pkl`, `steps_*.json`; `…/w2/cross_critic_work/data/grasp_gap.json`; `/workspace/gpt_field/runs/*/trial/requests/*`.
