> **Read this first.** This is the detailed integrated specification (interfaces I-01…I-20, flags, invariants, file lists, tests, commands). Two adversarial reviews (`reviews/review_feasibility.md`, `reviews/review_rules_safety.md`) found 10 must-fix and ~35 should-fix items. **`../IMPLEMENTATION_PLAN.md` §9 lists every correction and the section of this document it supersedes — where they conflict, `../IMPLEMENTATION_PLAN.md` wins.** Paths below that point into a scratch folder (`/tmp/claude-0/...`) refer to the design files now copied to `design/`.

# uc7.2 harness: integrated implementation plan

Date: 2026-11-08. Integration of six workstream designs into one plan. Read-only work: nothing under `/workspace/gpt_field` was edited, and no simulator, model call or GPU job was run.

**Direction (approved by the user).** Build on the obstacle harness's loop (H1, uc7.1 transit100, git `3cc84ac`). Run foundation_vision's perception as a service inside the controller. Reuse foundation_harness's robot geometry for path checks. Take metric estimation away from the model.

**Sources.**
- Analysis: `/workspace/gpt_field/reports/harness_analysis_20261108/README.md` (§0, §2–§6) and `work/w2/cross_critic.md` §2 (IC1–IC15).
- Designs (all in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/design/`): `ws_runner_measurement.md` (R1–R11), `ws_executor_control.md` (EXC-*), `ws_perception_service.md` (PS-1–PS-12), `ws_geometry_tools.md` (GT1–GT11), `ws_swept_planner.md` (SP1–SP12), `ws_protocol_context.md` (PC1–PC7).
- Component IDs below are the designs' IDs. `INT-*` are integration components added by this plan.

**Conventions.**
- "(est.)" marks an estimate that a sweep must confirm. Unmarked numbers come from the verified analysis or from code checked for this plan (§1).
- "Report X" cites the README.
- "Offline check" means a development aid with stated pass criteria. It is never a launch gate (user rule 4): a failure is recorded in the build tag and the user decides.
- "Capped smoke" means a model-calling run with an operator cap that Claude may start. Full tasks and sweeps are started by the user.

---

## 0. Summary

**What gets built.** One new repository, cloned from H1. H1 itself stays byte-identical. The build adds five layers on top of H1's loop.
1. **Honest measurement and a fast evaluation loop.** Scoring at the policy's own `finish`, `give_up`, evaluation caps, contact episodes, per-call service speed, a frame ring instead of per-step PNG round trips, and video out of the loop.
2. **A measured executor.** A fitted OSC plant, braking tracking, a progress guard with `stop_on: contact`, exact box distances with a checked escape, rotation sequencing instead of the 1.2 rad veto, reachability reports and grasp predicates.
3. **Perception as a resident, asynchronous, fail-open GPU service.** SAM 3.1 + DA3, an FK-rendered robot mask, a measured support plane, association by geometry, an instance inventory bound from plan cells, class-agnostic discovery, and held-object tracking. It feeds **one harness object map** with stable IDs and σ.
4. **Harness-side metric grounding.** The model references objects (`o3`, plan part `A`) and points at pixels (`locate`). The harness resolves metres at the moment of use. Model-written obstacle boxes are removed.
5. **A numpy-only body + held-object swept planner.** It is built from H2's robot geometry and used as a planner (escape, sequencing, detours, standoff handoff), never as a veto. After it: committed sequences, asynchronous verification and measured speed scheduling.

| M | Name | Main content | Ends with | Effort (d) |
|---|---|---|---|---|
| M0 | Foundation and equivalence | repo/config/digest, task-as-data, frame ring, overhead fixes, decoded-pixel replay | offline: bit-exact replay of H1 T04 (549 actions, 13 requests) under `h1_equivalent`; reset preflight table | 18.5 |
| M1 | Honest measurement baseline | finish/give_up, scoring at finish, contact episodes, timing, transport retries, relaxed plan gate, sweep harness | **S1**: all-task coverage map; pristine H1 (oracle metric) and measured H1 (honest metric) | 19.5 |
| M2 | Veto replacements, sensing, compact context | plant, braking + progress guard + `stop_on`, exact distances + escape, rotation sequencing, reachability, guarded motion, grasp predicates, context v3 | **S2**: b002 vs b001, same window | 36.5 |
| M3 | Perception service and measured geometry | resident SAM/DA3 service; SceneMap with IDs; target references, `locate`, regions, overlays; `obstacle_map` removed; body planner in shadow | **S3**: b003 vs b002 | 67.5 |
| M4 | Body + held-object swept planner | LocalBodyPlanner, body chord checks, palm depth bound, clearance-scaled cruise, held-object spheres | **S4**: b004 vs b003 | 18.0 |
| M5 | Committed sequences, async, approach speed | `then`/`resume`, 12-segment commits, async verify on a virtual clock, measured approach schedule, cache key | **S5 = acceptance**: frozen b005 vs b001, 3 repeats | 13.0 |
| | **Total** | | | **173.0** |

- **Effort.** The six designs sum to 161.5 engineer-days. Removing duplicated work saves about 14 d (§12.2). Integration and milestone operations add 25.5 d.
- **Critical path.** The perception track (§8): about 57 working days (≈ 11.5 weeks) with five engineers in parallel, about 35 weeks for one.
- **Sweep cost (est.).** About 10,000–16,700 subscription calls and 64–113 h serial sweep time for S1–S5 under the recommended scope (§7.5).
- **Decisions needed first.** UD-01–UD-03 before M0 work starts. UD-04–UD-15 before the S1 sweep (§10).

---

## 1. Facts verified for this plan

These are integration-relevant facts. I re-read each one for this plan, on top of the designs' own checks.

| Fact | Where (lines checked) |
|---|---|
| H1 is git `3cc84ac` on `main`, clean; origin is the public GitHub repo `seanliu7081/agent-speedup-v2-obstacle-harness-100mm` | `git log/status/remote` in `agent_speedup_v2_obstacle_harness_100mm/` |
| The main tree has staged changes, including `A agent_speedup_v2_obstacle_harness_100mm` (H1 staged as a gitlink). The new build must never touch that index | `git -C /workspace/gpt_field status --porcelain` |
| H1 `.gitignore` ignores `.venv/`, `.venv-sim/`, `.local/`, `runs/`, `validation/results/`. The patterns with a trailing slash do not match symlinks | `.gitignore` |
| Fragment caps and profiles: `FRAGMENT_VERSION="user-check-measured-fragments-4"`, caps 6 / 120 / 240, `target_rotation_rad` 1.2 / 0.5, stall constants 12 / 16 / 0.0008 | `agent_speedup_v2_user_check_fragments.py:23-46` |
| Envelope veto, transit-only routing, stall test, checkpoint on `observation_after`, `contact_or_collision_cause: "unknown"` | same file `:288-291`, `:293-317`, `:404-419`, `:423-434`, `:482` |
| Model-written `obstacle_map`, including "Remove the object you will manipulate from the avoid set" | `agent_speedup_v2_user_check_agent_client.py:92-105` (`:101-102`) |
| Grasp test-lift mandate; "Simulation pauses during inference"; the two oracle cues plus "There is no give_up action"; "Pixels specify rays … infer depth"; gripper facts without the palm | same file `:161-163`, `:182-183`, `:205-209`, `:224-225`, `:227-233` |
| Sliding 6-image window rebuilt on every call | same file `:417` (`_trim_images(..., IMAGE_HISTORY)`) |
| `completion_rule` cue (policy and framework); `finish` = 1-step hold + logged claim; every fragment end hands control back | `agent_speedup_v2_user_check_agent.py:128-130`, `:366-372`, `:394-400`; `agent_speedup_v2_user_check_framework.py:112` |
| The base loop refuses `stop_on_success=False` and stops at the first raw success | `scripts/evaluate_agent_speedup_v2.py:213-214`, `:1440-1443` |
| Transit cap is an environment variable defaulting to `"0.02"`; OSC scale = max(0.05, cap) | `src/gpt_field/action.py:20`, `:23` |
| The worker delivers 7 joints with each observation but drops them from the reset-identity dict; a second `agent_geometry` IPC and the ideal-pinhole validator run every step | `agent_speedup_v2_user_check_sim.py:72-80`, `:91-96`, `:122-128` |
| The validator accepts only focal = H/(2·tan(fovy/2)) to 1e-10 and an exactly centred principal point | `astranomy_agentic_geometry.py:104-112` |
| Retries cover only `server_is_overloaded` and 4 stream codes; 401, other 5xx, 520 and deadlines end the episode | `agent_speedup_v2_client.py:903-914` |
| The worker prompt comes from the official benchmark task, not from a BDDL override | `libero.py:837-838` |
| Strict-rest window `WINDOW_STEPS = 10` | `agent_speedup_v2_truth.py:68` |
| H1's launcher sets `MUJOCO_EGL_DEVICE_ID=str(args.gpu)` (so `--gpu` is the EGL minor) and pops `OPENAI_API_KEY` | `scripts/launch.py:97-102` |
| EGL minors are reversed against nvidia-smi on this host: index→minor 0→3, 1→2, 2→1, 3→0, 4→7, 5→6, 6→5, 7→4 | `/proc/driver/nvidia/gpus/*/information` and `nvidia-smi --query-gpu=index,pci.bus_id` |
| GPUs are shared: at writing, 7 of 8 were busy (48–100 % util); only GPU 7 was idle, and GPU 6 had about 15 GB free | `nvidia-smi` |
| Interpreters: H1 `.venv` py3.12.14, numpy 2.5.3, cv2 4.13, Pillow 12.3, **no scipy or torch**. H1 `.venv-sim` py3.10.21, numpy 1.26.4, scipy 1.15.3, MuJoCo 3.2.6, robosuite 1.4.1, torch 2.10 CPU. `/workspace/gpt_field/.venv-foundation-vision` py3.12.14, numpy 1.26.4, scipy 1.17.1, cv2 4.11, torch 2.10+cu128 | import checks with `-I` |
| `gpt_field/__init__.py` is 4 lines with no imports, so the service venv can import shared numpy-only modules from the same tree | `src/gpt_field/__init__.py` |
| Vendor pins: `robodojo_direct_sim.py` `6d2746cb…`; K2 `robot_geometry.py` `8fa39c48…` (the live tree currently equals it but is edited by another session); K2 `motion.py` `b15d07a1…`; foundation_vision `types.py` `98eae4ec…`, `backends.py` `17a3d225…`, `geometry.py` `d5fc4136…`; robosuite `robot.xml` `263c0f9a…`, `panda_gripper.xml` `d9d68123…` | `sha256sum` |
| K2 `robot_geometry.py` imports scipy `ConvexHull` (`:16`) and `..robodojo_direct_sim` (`:18`). It reads `.stl` and `.obj` meshes, including visual meshes when `include_visual=True` (`:30`, `:152-251`) | K2 snapshot |
| Weights: `artifacts/foundation_vision/models/sam3.1/sam3.1_multiplex.pt` (3.3 GB) and the DA3-LARGE-1.1 snapshot `0e109ae3…` | `ls` |
| Disk: 86 GB free on `/workspace`; 251 GB free in `/dev/shm`. H1 is 339 MB excluding venvs | `df`, `du` |

---

## 2. Build strategy

### 2.1 Repository, branches and tags
User decision UD-01.

```bash
git clone /workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm /workspace/gpt_field/agent_speedup_v2_uc72_harness
cd /workspace/gpt_field/agent_speedup_v2_uc72_harness
git remote rename origin h1-local                      # never push to H1's public GitHub origin
git tag -a h1-baseline-3cc84ac 3cc84ac -m "Frozen H1 baseline (uc7.1 transit100)"
git switch -c uc72/dev
printf '.venv\n.venv-sim\n.local/\narms/\n' >> .gitignore   # .venv without a slash so venv symlinks stay ignored
ln -s /workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/.venv .venv
ln -s /workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/.venv-sim .venv-sim
```

- **Clone, not a branch in H1.** H1's working tree and its `.git` stay untouched, so `3cc84ac` remains reproducible for A/B (user rule 6).
- **Main tree.** The new directory appears as untracked in the main tree. Whether to stage it as a gitlink, as was done for H1, is the user's choice. Nothing in this plan writes to the main tree's index or tracked files.
- **Branches.** `uc72/dev` is the integration branch. Each workstream works on `uc72/ws-<key>` (runner, executor, perception, geometry, swept, protocol) and merges into `uc72/dev` only with its offline tests green and the `h1_equivalent` replay still bit-exact.
- **Tags.** Annotated, immutable build tags `uc72-bNNN`. `uc72-b000` is M0 (offline only); `uc72-b001` … `uc72-b005` are the milestone builds M1 … M5. Any code, prompt or config-schema change after a freeze creates a new tag (`uc72-b003` → `uc72-b003a` → …). A lightweight alias `uc72-m3` points at the tag that ran the milestone sweep.
- **Arms run only from worktrees at tags, never from `uc72/dev`.** `scripts/uc72_build.py worktrees` runs `git worktree add arms/<tag> <tag>`, creates the venv symlinks and an empty `.local/` (LIBERO writes its `libero_config` there via `LIBERO_CONFIG_PATH`, as H1's launcher does). That isolates each arm from the global `~/.libero/config.yaml`.

### 2.2 Naming and versioning

| Item | Value |
|---|---|
| Build id | `uc72-bNNN` (the protocol design's "v3-a…v3-e" map to b001…b005; the name "v3" is not used for builds) |
| Arm id | `<build_tag>:<config_name>` (e.g. `uc72-b003:m3_geometry`); the pristine baseline is `h1-baseline-3cc84ac:pristine` |
| Component versions | runner `uc72-runner-1`; fragments `user-check-measured-fragments-5`; protocol `protocol-uc72-1`; perception `perception-service-1`; geometry `geometry-tools-1`; swept `swept-planner-1`; robot model `panda-chain-1` / `panda-body-1` |
| Schemas | `uc72-build-config-1`, `uc72-task-spec-1`, `UC72FRM1` (frame ring), `camera-calibration-1`, `perception-snapshot-1`, `harness-object-map-1`, `target-ref-1`, `decision-uc72-m1/m2/m3/m5`, `segment-uc72-1`, `robot-event-1`, `executed-profile-1`, `swept-planner/body_plan/v1`, `segment-result-1`, `outcome-record-1` |

Path map from the designs (only two normalisations):

| Design path | Path in the build | Why |
|---|---|---|
| protocol `src/gpt_field/v3/` | `src/gpt_field/protocol/` | one version namespace (`uc72`) |
| executor `vendor/…`, swept `third_party/robot_geometry/…` and `third_party/foundation_harness_k2/…`, perception `perception_service/vendor/…` | `src/gpt_field/vendor/` (one tree, one `MANIFEST.json`) | one copy of each pinned file |
| executor `executor/kinematics.py` (FK part), swept `swept/kinematics.py`, swept `swept/assets/panda_body_v1.*`, perception `assets/robot_model_panda_v1.npz` | `src/gpt_field/robot/` (`kinematics.py`, `body.py`, `assets/panda_robot_model_v1.npz`+`.json`) | one FK source, one robot-model artifact |
| all other design paths | unchanged (`uc72/`, `executor/`, `swept/`, `geometry_tools/`, `perception_service/`, `perception_client.py`, `perception_bridge.py`, `perception_recorded.py`, `validation/`, `tools/`, `config/`, `tasks/`, `sweeps/`, `tests/`) | traceability to the designs |

### 2.3 Frozen configuration and build digest
The owner is ws_runner_measurement (R1); interface I-01.
- **One JSON file per arm** under `config/arms/`. The runner reads its path from `--config`; `UC72_CONFIG` is a pointer, never a value. The loader is py3.10-compatible, because the sim worker imports it.
- **Every numeric leaf is `{"value": x, "source": "..."}`**, with source ∈ `inherited:<file:line>` | `measured:<artifact>` | `derived:<formula>` | `user_decision:<UD-id>`. The loader rejects unsourced numbers and undeclared flags. There are no defaults: `action.py` reads the transit cap from config and raises without it (this replaces the `"0.02"` default at `action.py:20`).
- **`build.json` in every run directory** and `summary.json["build"]` record: git commit, tree, tag and dirty flag; the config path, sha and flag values; interpreter fingerprints (controller, sim, perception: python, packages sha, numpy, Pillow zlib); the task-pack sha; the vendor-manifest sha; the robot-model npz sha; `osc_response.json` sha; `measured_sigma.json` sha; the weights sha (SAM, DA3); the LIBERO manifest sha.
- **`--require-frozen` (always used by sweeps).** The runner recomputes the digest from its own worktree at every episode start and aborts on a dirty tree or a mismatch.

### 2.4 Feature flags (one registry, no environment or CLI behaviour switches)
Every behaviour switch is a config flag. CLI arguments are only pointers and operator caps: `--task-spec`/`--task`, `--config`, `--service`, `--gpu`, `--max-exec-calls`, `--max-steps`, `--max-wall-s`, `--output`, `--prepare-only`. The swept design's `GPT_FIELD_BODY_PLANNER` / `GPT_FIELD_ROUTER` environment variables and the protocol/perception CLI mode flags become the flags below.

| Flag | Owner | Values | h1_eq | M1 | M2 | M3 | M4 | M5 | Report item |
|---|---|---|---|---|---|---|---|---|---|
| `runner.single_simulator` | runner | bool | T | T | T | T | T | T | F8 |
| `runner.lazy_frames` / `compact_trace` / `throttled_state` | runner | bool | T | T | T | T | T | T | A2, H1 #1 |
| `runner.video` | runner | off / posthoc / raw_live / inline | off | posthoc | posthoc | posthoc | posthoc | posthoc | A2 (UD-07) |
| `runner.terminal_actions` / `eval_caps` | runner | bool | F | T | T | T | T | T | A1, R6 |
| `runner.contact_records` / `object_truth` | runner | bool | T / F | T / T | T / T | T / T | T / T | T / T | §6.2 (evaluator-only) |
| `protocol.termination` / `transport` | protocol | h1 / uc72 | h1 | uc72 | uc72 | uc72 | uc72 | uc72 | A1, A3 |
| `protocol.plan_gate` | protocol | h1 / relaxed | h1 | relaxed | relaxed | relaxed | relaxed | relaxed | A4, H1 #14 |
| `protocol.plan_inventory` | protocol | bool | F | F | F | T | T | T | IC7 |
| `protocol.plan_mode` | protocol | separate / folded | sep | sep | sep | sep | sep | sep | A4 fold (UD-15) |
| `protocol.context` | protocol | h1 / uc72 | h1 | h1 | uc72 | uc72 | uc72 | uc72 | D3, H1 #6 |
| `protocol.image_retention` | protocol | sliding6_h1 / current_only | s6 | s6 | cur (UD-20) | cur | cur | cur | IC9 |
| `protocol.drop_why` | protocol | bool | F | F | T (UD-21) | T | T | T | D3 |
| `protocol.checkpoints` | protocol | h1 / committed | h1 | h1 | h1 | h1 | h1 | committed | D1, H1 #2 |
| `protocol.verify_mode` | protocol | blocking / async | blk | blk | blk | blk | blk | async (UD-39) | R4, R9 |
| `protocol.cache_key` | protocol | off / episode | off | off | off | off | off | episode (UD-40) | D3 |
| `executor.tracker` | executor | h1 / braking | h1 | h1 | braking | braking | braking | braking | B3, H1 #3 |
| `executor.monitor` | executor | h1_stall / progress | h1 | h1 | progress | progress | progress | progress | B3 |
| `executor.stop_on_contact` | executor | bool | F | F | T | T | T | T | B3, H1 #4 |
| `executor.rotation_sequencing` | executor | bool (removes envelope veto) | F | F | T | T | T | T | B1, IC3 |
| `executor.router` | executor | point_v1 / point_v2 | v1 | v1 | v2 | v2 | (body) | (body) | B2, IC4 |
| `executor.reachability` / `guard_policy` / `grasp_predicates` | executor | bool | F | F | T | T | T | T | C5, C1, D1 |
| `executor.segment_budgets` | executor | h1 / plant | h1 | h1 | plant | plant | plant | plant | D1 caps (UD-18) |
| `executor.max_segments` | executor | 6 / 12 | 6 | 6 | 6 | 6 | 6 | 12 | D1, IC5 |
| `executor.jaw_symmetric` / `posture_recovery` | executor | bool | F | F | T (UD-19) / F | T / F | T / UD-37 | T / UD-37 | C5 |
| `geometry.envelope_text` | geometry | bool | F | F | T | T | T | T | C2, H1 #10 |
| `geometry.scene_map` / `references` | geometry | bool | F | F | F | T | T | T | R1, C3, H1 #9, #20 |
| `geometry.overlays` | geometry | off / ids_boxes_path | off | off | off | UD-26 | UD-26 | UD-26 | IC9 |
| `geometry.raw_world_targets` | geometry | allowed_flagged / removed | allowed | allowed | allowed | UD-27 | UD-27 | UD-27 | R1 |
| `geometry.calibration` | geometry | h1_validator / calibration1 | h1 | h1 | h1 | cal1 | cal1 | cal1 | A5, G3, R9 |
| `perception.mode` | perception | off / shadow / on | off | off | off | on | on | on | V2, V3, C3 |
| `perception.record_frames` | perception | none / decision / all | none | none | none | decision | decision | decision | V9, R8 |
| `perception.look_motion` / `event_barrier` | perception | bool | F | F | F | UD-23 / F | same | same | PS-5, PS-9 |
| `swept.mode` | swept | off / shadow / on | off | off | off | shadow | on | on | C4, R2 |
| `swept.depth_bound` | swept | off / report / active | off | off | off | report | UD-34 | UD-34 | C2 |
| `swept.speed` | swept | off / shadow / cruise / cruise_approach | off | off | off | shadow | cruise (UD-35) | cruise_approach (UD-41) | IC1, IC2, D2 |

`executor.clearance_oracle` is derived, not a flag: `body_sweep_v1` iff `swept.mode=on`, else `grip_site_boxes_v1`. Numeric component parameters live under `components.<ws>` with sources: caps 1,800 s / 80 calls (UD-04), `post_finish_hold_steps` 10, contact-episode merge gap 6, ring slots 16, `geometry.k_sigma` (UD-28), `swept.beta_mm` (UD-33), the latency reference fit, and the sha of `osc_response.json` / `measured_sigma.json`.

### 2.5 Safe-order invariants (IC rules enforced by the config loader)
The loader refuses a config that violates any rule below, so no ablation arm can break the safe order by accident. `h1_equivalent` (all defaults) is exempt from INV-6 only.

| Rule | Requires | IC |
|---|---|---|
| INV-6 | Any non-default executor, geometry, perception, swept, `protocol.context` or `protocol.checkpoints` flag ⇒ `runner.terminal_actions`, `runner.eval_caps`, `protocol.termination=uc72` | IC6 (finish-scoring first) |
| INV-3 | `executor.rotation_sequencing` ⇒ `executor.tracker=braking` | IC3 |
| INV-4 | `geometry.scene_map` ⇒ `executor.router=point_v2` or `swept.mode=on` | IC4, IC13 (escape before automatic boxes) |
| INV-13 | `perception.mode=on` ⇒ `geometry.scene_map` and `geometry.references` (model boxes leave the schema only when automatic boxes exist) | IC13 |
| INV-12 | `swept.mode=on` ⇒ `perception.mode=on`, `executor.rotation_sequencing`, `executor.guard_policy` | IC12 (body sweep only with better geometry), IC3 |
| INV-2 | `swept.speed ∈ {cruise, cruise_approach}` ⇒ `swept.mode=on` and `executor.tracker=braking` | IC2 |
| INV-1 | `swept.speed=cruise_approach` ⇒ `executor.guard_policy`, `executor.stop_on_contact`, `perception.mode=on`, `swept.depth_bound=active` | IC1 (guard + envelope bound → measured geometry → speed) |
| INV-5 | `protocol.checkpoints=committed` ⇒ `executor.guard_policy`, `executor.grasp_predicates`, `executor.stop_on_contact`, `executor.max_segments=12` | IC5 |
| INV-7 | `protocol.plan_mode=folded` ⇒ `protocol.plan_inventory` (the folded first response must carry `objects`) | IC7 |
| INV-10 | `perception.mode≠off` or `protocol.verify_mode=async` ⇒ `runner.lazy_frames` (frame ring) | IC10 |
| INV-14 | `protocol.verify_mode=async` ⇒ `protocol.checkpoints=committed` and `protocol.transport=uc72` (thread-safe deadline) | IC14 |
| INV-D | `swept.depth_bound=active` ⇒ `perception.mode=on` (heights from measured depth) | report C2 caution |
| INV-9 | `geometry.overlays≠off` ⇒ `protocol.context=uc72` (overlays replace the current pair, never add images) | IC9 |

IC8, IC11 and IC15 are enforced structurally rather than by flags:
- IC8: the model never writes absolute metric keyposes for objects; references are resolved by the harness.
- IC11: no "unknown space is traversable" rule exists; unknown volumes affect speed only.
- IC15: a test checks that outcome records keep every reason string verbatim.

### 2.6 Arms (configs) shipped with every tag

| Config | Purpose |
|---|---|
| `h1_equivalent.json` | Replay anchor: all behaviour flags off; must replay H1 T04 bit-exactly (R9) |
| `measured_h1.json` | M1 honest baseline: H1 policy + A1/A2/A3/A5 + relaxed plan gate |
| `m2_executor.json` | M2 candidate |
| `m3_shadow.json` | M2 behaviour + perception and swept in shadow (smoke and diagnostic arm) |
| `m3_geometry.json` | M3 candidate |
| `m4_swept.json` | M4 candidate |
| `m5_blocking.json` / `m5_candidate.json` | M5 with blocking / asynchronous verification (UD-39 A/B smoke); `m5_candidate` is the acceptance arm |

### 2.7 Rollback and equivalence
1. **Pristine baseline.** Worktree `arms/h1-baseline-3cc84ac`, tree identical to H1 `main`, run with H1's own `run.sh` (which sets the 0.10 m cap) and capped externally through its operator stop file. No H1 code changes.
2. **Flags-off equivalence on every build.** `config/arms/h1_equivalent.json` must replay the T04 reference exactly (actions, state8, contact counters, raw-success flags, decoded pixels of every observation and every model image, request text except measured timers). That proves the refactors changed no behaviour.
3. **Per-capability rollback.** A new config with the flag at its previous value (within the invariants). No code change.
4. **Per-milestone rollback.** Run the previous milestone's tag; tags are immutable.
5. **Vendor isolation.** `scripts/check_vendor.py` verifies every pinned hash. An import-guard test fails if any module imports the live `gpt_field.foundation_harness` or `gpt_field.foundation_vision` packages, which are edited by another session.

### 2.8 Engineering rules (binding on every workstream)
- **No per-task code, prompt words, thresholds or inputs.** One shared `tests/uc72/test_no_task_words.py` greps policy, prompt and perception query code for benchmark nouns (the PC4 T4.6 and PS-4 lists).
- **Every numeric parameter has a recorded source** (the config loader enforces it).
- **New behaviour lives in new modules.** H1 modules on the `h1_equivalent` path stay byte-identical: `tracking.py`, `agent_speedup_v2_user_check_routing.py`, `fine_distance_planner.py`, the prompt-6 text and the legacy schema. Edits to shared H1 files are flag-gated and covered by the replay.
- **Evaluator-only data never reaches the policy.** This covers task-spec `scene`/`evaluator` fields, truth, contacts, object poses and oracle masks, and is enforced by an import guard plus a leakage test (R3).
- **No edits to the main tree, H1 or the live foundation_harness tree.**
- **Each milestone build passes the full offline suite and the `h1_equivalent` replay before its freeze.** Results are recorded in the tag message; a failure informs the user and does not block them (rule 4).
- **Subscription only** (GPT-6 Astra, reasoning high). Never an API key or a fallback model; the launcher pops `OPENAI_API_KEY`, as H1 does.

---

## 3. Architecture

### 3.1 Processes

| Process | Interpreter | Runs | GPU | Lifetime |
|---|---|---|---|---|
| **Controller** (one per episode) | H1 `.venv` (numpy 2.5.3, cv2, Pillow; no scipy or torch) | runner loop (`uc72/runner.py`), policy, protocol client + call scheduler, executor (fragments v5 + `executor/*`), swept planner (`swept/*`), robot model (`robot/*`), geometry tools (`geometry_tools/*`), perception client and bridge, recorder writers | none | episode |
| **Simulator worker** (one per episode, started once per episode, R2) | H1 `.venv-sim` (MuJoCo 3.2.6, robosuite 1.4.1, bundled Pillow/zlib-ng encoder) | LIBERO env step, rendering both cameras, FrameRing writer, record-only contact monitor, evaluator ops (truth, hold settle, ordered stages, object poses), `camera-calibration-1` at reset | EGL on `g_sim` (small) | episode |
| **Perception service** (resident) | `/workspace/gpt_field/.venv-foundation-vision` (torch cu128, scipy) | SAM 3.1 + DA3 loaded once; robot raster, support, association, inventory and binding, discovery and height field, tracking, held state, re-localisation | CUDA on `g_perc`: SAM peak ≈ 6 GB + DA3; require ≥ 10 GB free | one per sweep (`reset_episode` per episode) |
| **Sweep runner** | `.venv` | `plan / run / status / report`; launches episode controllers, keeps the perception service resident, pauses on infra or quota, writes the coverage map | none | sweep |
| Live raw recorder (optional, UD-07) | `.venv` + ffmpeg | reads the ring, pipes to x264, drops frames if behind | none | episode |
| Offline tools | `.venv-sim`: reset preflight, replay-truth export, post-hoc video re-simulation, robot-model build, scripted executor benchmark. Service venv: perception benchmark. `.venv`: geometry/swept/executor replays, PC7 prompt A/B | — | EGL / CUDA as noted | on demand |

The controller environment never imports scipy or torch, which satisfies prerequisite (b), "one interpreter or a service". Every module shared across interpreters (`uc72/frames.py`, `uc72/config.py`, `geometry_tools/camera.py`, `geometry_tools/metric.py`, `robot/kinematics.py`, `robot/body.py`, `perception_service/protocol.py`) is numpy/stdlib-only. Each is tested under both numpy 1.26 and 2.5.

### 3.2 Data flows

```
   ChatGPT subscription (GPT-6 Astra, reasoning high)  ◄── HTTPS/SSE: protocol client (retries, telemetry, cache key)
                         ▲
┌────────────────────────┴───────────── controller (.venv) ─────────────────────────────────────────────┐
│ uc72.runner ─act/step/observe─► policy: prompt prefix + append-only transcript + OutcomeRecord + scheduler│
│   Timers + VirtualClock         geometry_tools: FrameStore · SceneMap (o-ids, aliases, held, permissions)  │
│   recorder (steps/calls/events/   · TargetResolver · locate · regions · overlays · context.scene          │
│    contacts/trace/summary)      executor: StagePipeline [planner → BrakingPathTracker → ProgressMonitor]   │
│   ServiceSupervisor               · GuardPolicy · predicates · CommittedSequence · OSC plant               │
│        │                        swept: LocalBodyPlanner · SweptChecker · depth bound · SpeedPolicy         │
│        │                        robot: kinematics (FK/J/bind/lift) · body model npz                        │
│        │                        perception client/bridge (fail-open) ──────────────┐                       │
└────────┼───────────────▲──────────────── step/reset JSON pipe ────────────────────┼───────────────────────┘
         │ unix socket   │ actions / step replies (state8, joints, contacts → recorder only)  │ unix socket JSONL
         │               │                                                                     │ observe/notify/relocalize
┌────────┴──── sim worker (.venv-sim, EGL g_sim) ────┐          ┌─ perception service (.venv-foundation-vision, CUDA g_perc) ─┐
│ env.step · render both cameras · contact monitor │          │ SAM 3.1 + DA3 resident · robot raster · support plane ·      │
│ evaluator ops · calibration-1 at reset           │          │ association · inventory/binding · discovery + height field ·  │
└──────────────┬───────────────────────────────────┘          │ tracking · held · relocalise → PerceptionSnapshot (async)   │
               │ publishes                                     └───────────────▲─────────────────────────────────────────────┘
               ▼                                                               │ reads
   /dev/shm/uc72_<run>_frames.bin  (FrameRing v1: 16 slots, raw RGB ×2 + state8 + q7 + qd7) ──────┘  (also read by the controller's
                                                                                                       lazy frames and the optional recorder)
```

**Reset.**
1. The worker resets and publishes frame 0; the controller records the reset identity.
2. The controller sends `reset_episode` and `observe(frame 0, full, reset)` to the service; the service encodes the images, runs DA3, the robot raster, the support estimate and discovery.
3. The plan call (14–52 s) runs in parallel.
4. `set_plan(objects)` binds roles from cells on the cached features.

**Per step (controller).**
1. `act` → StagePipeline: the planner leg is tracked by BrakingPathTracker, then `ProgressMonitor.update` runs.
2. The action goes to the worker; `env.step` runs; the frame is published to the ring; the step reply carries state8, joints, per-frame camera geometry and `contacts_action` (to the recorder only).
3. `observe_transition` turns executor events into `robot-event-1`. These go to the SceneMap (attach/detach, support anchors), the perception notify queue and `events.jsonl`.

**Per decision.**
1. `bridge.before_model_call` requests a full perception refresh on the decision frame. The model call follows: blocking in M1–M4; asynchronous `verify_*` in M5.
2. `after_model_call` waits up to 0 s for the snapshot. It was requested at least 7 s earlier, while the full refresh takes about 1.0–1.9 s (est.).
3. The SceneMap ingests the snapshot. `context.scene` and the overlays are built for the next request.

**Per stage entry.**
1. The SceneMap supplies anchors and `obstacle_set(intent)`; the TargetResolver produces `ResolvedTarget` + `ContactIntent`; `relocalize(ids)` runs (≤ 5 ms when confirmed).
2. `StagePlanner.plan_stage` (point planner in M2–M3, body planner in M4+) uses candidates and predicted profiles from the executor's `ExecutedProfileModel`, then returns `StagePlan` legs to the tracker.

### 3.3 Package layout

```
src/gpt_field/
  uc72/            runner, services (ServiceSupervisor), recorder, timing (Timers, VirtualClock), frames (FrameRing),
                   taskspec, config (+ invariants), build (digest), gpu (index→EGL minor), metrics/, sweep/
  protocol/        protocol_schema, policy (AgentFirstV3Policy), prompt, context, client, transport, telemetry,
                   plan_gate, scheduler
  executor/        osc_plant (+ osc_response.json), progress_monitor, braking, sequencing, executed_profile,
                   box_clearance, clearance_oracle, transit_route (TransitRoute2), reachability (IK on robot.kinematics),
                   posture, guard_policy, predicates, stage_pipeline, events, feedback, schemas/
  swept/           body (part classes), distance, scene, profile (adapter), checker, planner, held, depth_bound,
                   speed, route, chord, report, flags
  robot/           kinematics (PandaChain: FK, Jacobian, bind, batched FK, lift), body (RobotBodyModel),
                   assets/panda_robot_model_v1.npz + .json
  geometry_tools/  camera, metric, support, scene_map, references, locate, regions, overlay, robot_envelope,
                   prompt, binding (derivation fallback)
  perception_service/  server, protocol, pipeline, health, frames (reader), robot_render, queries, detect,
                   support, localize, primitives, inventory, binding, discovery, heightfield, tracking, held,
                   relocalize, snapshot
  perception_client.py · perception_bridge.py · perception_recorded.py
  vendor/          MANIFEST.json · robodojo_direct_sim.py · panda_robosuite_1_4_1/ (robot.xml, panda_gripper.xml,
                   collision + visual meshes) · foundation_harness_k2/ (robot_geometry.py, motion.py; offline only)
                   · foundation_vision/ (__init__, types, backends, geometry; service only)
tools/             build_robot_model.py (offline, scipy)
validation/        build_pixel_fixture, replay_runner, replay_executor_logs, swept_replay, prompt_ab (+report),
                   truth/export_replay_truth.py, bench/ (perception dataset + scorer), geometry_bench/
config/            schema/, arms/, invariants.py      tasks/   sweeps/specs/   tests/ (per package)
```

`vendor/foundation_harness_k2/robot_geometry.py` keeps its byte-identical `from ..robodojo_direct_sim import …`. That resolves to `vendor/robodojo_direct_sim.py`, so the offline parity tests run on unmodified copies.

### 3.4 GPU placement and addressing
- **One GPU address everywhere: the nvidia-smi index (or UUID).** `uc72/gpu.py` maps it to the EGL device minor through `/proc/driver/nvidia/gpus/<bus>/information` and exports `MUJOCO_EGL_DEVICE_ID=<minor>` for the worker. The perception service uses `cuda:<index>` with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. On this host the map is 0→3, 1→2, 2→1, 3→0, 4→7, 5→6, 6→5, 7→4. Both the index and the UUID are recorded in `build.json` and in each snapshot.
- **Service GPU (UD-22).** A fixed GPU per sweep with ≥ 10 GB free (SAM peak 5.98 GB + DA3 + margin). At writing only GPU 7 was idle; GPU 6 had about 15 GB free. The sweep runner checks free memory before each episode. On OOM the service degrades: one retry per view, then it reports unhealthy and episodes continue fail-open.
- **Simulator EGL.** Rendering needs about 1–2 GB (est.). It can share a GPU with training jobs but should avoid the service GPU, so render latency does not slow SAM.

### 3.5 Time model
The owner is ws_runner_measurement; interface I-19.
- **`VirtualClock`.** `t_v = steps·dt + Σ idle_s`, with `dt = 0.05 s` (`control_freq = 20`). Every blocking wait adds its wall duration to `idle_s` through `Timers.block(kind)`: model calls in blocking mode, a perception `wait_for` that actually waits, and planner overruns.
- **Policy time** (pre-registered) = `t_v` at the terminal step + Σ max(0, compute_step − dt). Excluded and reported separately: simulator wall beyond real time, rendering and encoding for evaluation, IPC, trace and state writes, video, truth, the hold settle, failed transport attempts and quota waits (`infra_wait_s`).
- **Asynchronous verification (M5, PC6).** A request dispatched at `t_v0` with wall latency `L` applies at the first step with `t_v ≥ t_v0 + L`. The simulator never runs ahead of wall time while a request is in flight (pacing). An early answer is buffered until its virtual arrival. Barriers (contact, gripper change, checkpoint, end of sequence) hold the robot.
- **Asynchronous perception.** A snapshot captured at `t_capture` with compute `c` becomes usable at `t_capture + c` on the same clock.
- **Freshness (IC14).** Every target is re-resolved at stage entry from the newest snapshot, and every consumption logs `age_at_use = action_start − capture`.

### 3.6 Failure isolation

| Failure | Handling | Scored as |
|---|---|---|
| HTTP 401 | one forced token refresh, identical payload resent; a second 401 → infra `auth`, the sweep pauses for re-login | — |
| 5xx / 520–524 / stream `server_error`/`unknown_error` / deadline | ≤ 2 retries with backoff; identical payload; nothing committed | infra if exhausted (one relaunch per cell) |
| 429 quota | wait Retry-After if ≤ 60 s, else `QuotaPause`: the simulator is already paused during inference; the request is resent after the quota reset, and the wait goes to `infra_wait_s` (UD-12) | not a failure |
| Simulator crash or timeout | infra `simulator`; one relaunch in a fresh directory | — |
| Perception service error, OOM, socket loss | fail-open: snapshot `degraded` or `unavailable`. The SceneMap keeps the last records marked `perception_unavailable`; new objects can still enter only through `locate` + `scene_edits add_from_locate` (measured), never through model-written boxes. `context.scene.perception` tells the model | episode continues; cell records `perception_status` |
| Exception in policy, executor or planner code | `harness_exception`, finalisation still writes `summary.json` | scored failure `E`, flagged, never relaunched blindly |
| Schema error in a model answer | existing re-ask path (H1 `agent.py:272-282`) | — |

---

## 4. Shared interfaces (one owner each; frozen at the stated milestone)

"Frozen" means the shape is fixed at that milestone. Later changes are additive only, with the schema version bumped; consumers ignore unknown fields.

| ID | Interface | Owner | Producers → consumers | Frozen at |
|---|---|---|---|---|
| I-01 | Config, flags, invariants, build digest, vendor manifest | ws_runner_measurement | all → runner, sweep | M0 (format; flags added per milestone) |
| I-02 | Task spec `uc72-task-spec-1` | ws_runner_measurement | importer → runner, worker | M0 |
| I-03 | Raw frame stream `FrameRing v1` | ws_runner_measurement | worker → controller, perception, recorder | M0 (layout); first live consumer M3 |
| I-04 | Service protocol (socket envelope + lifecycle) | ws_runner_measurement | runner ↔ perception service | M3 |
| I-05 | Camera calibration `camera-calibration-1` + `CameraModel` | ws_geometry_tools (worker emission implemented by ws_runner_measurement) | worker → geometry, perception | M3 |
| I-06 | Perception snapshot / object inventory `perception-snapshot-1` (+ notify, relocalise, height field) | ws_perception_service | service → SceneMap, swept | M3 |
| I-07 | Plan scene inventory (`objects` plan extension) | ws_protocol_context | model (plan) → perception, SceneMap | M3 |
| I-08 | Harness object map: IDs, records, `ObstacleSet`, held record, contact permissions, model-facing `context.scene` | ws_geometry_tools | SceneMap → planners, protocol, overlays | M3 |
| I-09 | Geometry reference grammar: `TargetRef`, `locate`, `scene_edits`, `ResolvedTarget`, `ContactIntent` | ws_geometry_tools | model → resolver → executor, swept | M3 |
| I-10 | Robot model API: kinematics (FK, Jacobian, bind, lift), body model (spheres, hulls, visual envelopes, silhouettes), `SweptChecker`, IK reachability | ws_swept_planner (`robot/`, `swept/checker`); the IK module is executor code built on it | robot/ → executor, perception, geometry, swept | M2 (kinematics); M3 (body model + checker) |
| I-11 | Executed profile + OSC plant (`ExecutedProfileModel`, `osc_response.json`) | ws_executor_control | executor → swept, runner digest | M2 |
| I-12 | Stage-planner seam (`StagePlanner`, `StagePlan`, `ClearanceOracle`, `body_plan`) | ws_executor_control | point planner (executor) and body planner (swept) → executor | M2 (point); M4 (body semantics) |
| I-13 | Robot event stream `robot-event-1` | ws_executor_control | executor → SceneMap, perception, swept, recorder | M2 |
| I-14 | Executive decision + segment schema (wire) | ws_protocol_context (wire); segment semantics by ws_executor_control; target fields by ws_geometry_tools | model ↔ policy | M1 → M5 (`decision-uc72-m5` is final) |
| I-15 | Feedback/failure taxonomy: reason codes, `segment_result`, failure signature | ws_executor_control | executor, swept, resolver, perception → protocol, runner | M2 (codes added at M3–M5) |
| I-16 | `OutcomeRecord` (model-facing projection) | ws_protocol_context | protocol → model | M2 |
| I-17 | `CommittedSequence` API | ws_executor_control | executor → protocol | M2 (core); M5 (resume, barriers) |
| I-18 | Termination + scoring contract | ws_runner_measurement | policy → runner → reports | M1 |
| I-19 | Timing and telemetry (`Timers`, `VirtualClock`, policy time, steps/calls/events/contacts rows) | ws_runner_measurement | all → recorder, report | M1 (async extension M5) |
| I-20 | `HeightQuery` (palm/fingertip depth bound input) | ws_perception_service | height field (npz) → swept depth bound | M3 (report-only); M4 (active) |

### I-01 Config, flags, invariants, build digest, vendor manifest (owner ws_runner_measurement)

```json
{"schema": "uc72-build-config-1", "arm": "m3_geometry",
 "control": {"transit_action_m": {"value": 0.10, "source": "inherited:H1 scripts/launch.py:101"},
             "contact_action_m": {"value": 0.005, "source": "inherited:H1 agent_speedup_v2_user_check_fragments.py:31"},
             "control_period_s": {"value": 0.05, "source": "inherited:robosuite control_freq=20"}},
 "model": {"model": "gpt-6-astra", "reasoning_effort": "high", "transport": "chatgpt_subscription_ordered_sse",
           "request_deadline_s": {"value": 180, "source": "inherited:H1 scripts/launch.py:47"}},
 "evaluation": {"cap_policy_time_s": {"value": 1800, "source": "user_decision:UD-04"},
                "cap_exec_calls": {"value": 80, "source": "user_decision:UD-04"},
                "cap_wall_s": {"value": 5400, "source": "derived:3*cap_policy_time_s"},
                "post_finish_hold_steps": {"value": 10, "source": "inherited:H1 agent_speedup_v2_truth.py:68"},
                "contact_episode_merge_gap_actions": {"value": 6, "source": "inherited:w1/cross_eval_methodology.md §1b"}},
 "runtime": {"frame_ring_slots": {"value": 16, "source": "derived:readers held concurrently + writer lead"},
             "frame_publish_every_n_steps": {"value": 1, "source": "derived:min over subscribers' declared intervals"},
             "run_state_min_interval_s": {"value": 1.0, "source": "inherited:H1 evaluate_agent_speedup_v2_user_check.py:381"}},
 "transport": {"max_transient_retries": {"value": 2, "source": "inherited:H1 agent_speedup_v2_client.py:144"},
               "overload_max_attempts": {"value": 3, "source": "inherited:H1 agent_speedup_v2_client.py:139"},
               "quota": {"mode": "pause_and_resume", "max_pause_s": {"value": 21600, "source": "user_decision:UD-12"}}},
 "telemetry": {"latency_reference": {"intercept_s": 6.89, "per_reasoning_tok_s": 0.0510, "per_visible_tok_s": 0.0071,
               "per_input_ktok_s": 0.0192, "source": "measured:w2/cross_critic.md C1 fit, 164 calls 11-02/03, R2 0.939"}},
 "flags": {"runner.terminal_actions": true, "executor.tracker": "braking", "perception.mode": "on", "...": "..."},
 "components": {"executor": {"osc_response_sha256": "…"}, "geometry": {"k_sigma": {"value": 2.0, "source": "user_decision:UD-28"}},
                "swept": {"beta_mm": {"gripper": {"value": 4, "source": "derived:transit arrival tolerance fragments.py:29"},
                                      "arm": {"value": 10, "source": "measured:SP2 p90 arm prediction error 10-17 mm"}}},
                "perception": {"min_free_gib": {"value": 10, "source": "measured:SAM peak 5.98 GB + DA3 + margin"}}}}
```

- **`config/invariants.py`** implements INV-1 … INV-D (§2.5). `BuildConfig.load()` raises `ConfigInvariantError(rule, flags)`.
- **API:** `BuildConfig.load(path, *, expected_sha256=None)`, `.value(dotted)`, `.source(dotted)`, `.flag(name)` (KeyError for an undeclared flag), `.section(ws_key)`; `current()`.
- **`vendor/MANIFEST.json`:** `{"schema": "uc72-vendor-1", "entries": [{"path", "sha256", "source_path", "source_sha256", "source_snapshot": "work/h2_keypose2_snapshot|main_tree|robosuite-1.4.1", "use": "runtime_controller|runtime_service|offline_only", "copied_utc"}]}`. Its sha enters the build digest.

### I-02 Task spec (owner ws_runner_measurement)
As R3: `{"schema": "uc72-task-spec-1", "task_id", "family": "official|obstacle_variant|relayout|relayout_obstacle|composite", "held_out", "sentence", "scene": {"suite", "source_task_id", "bddl": {"path", "sha256"}|null, "initial_state": {"path", "sha256"}, "official_episode_index", "camera_overrides"}, "evaluator": {"goal": "bddl", "ordered_stages": [{"id", "all": [[pred, args...]]}]|null, "added_obstacles": [..], "settle_rule"}, "reset_identity": {"known_png_sha256", "known_pixel_sha256", "source"}, "provenance": {...}}`.
- Only `sentence` crosses to the policy (leakage test).
- The BDDL `:language` must equal `sentence`, and the evaluator uses the BDDL's own `:goal`. This one rule replaces both H1's official-language veto (`libero.py:203-218`) and the main tree's code registry.

### I-03 FrameRing v1: the raw frame stream (owner ws_runner_measurement)
This merges R7 and the perception design's §4.1.
- **File:** `/dev/shm/uc72_<run_id>_frames.bin` (`np.memmap`; avoids the pre-3.13 `SharedMemory` resource-tracker unlink).
- **Lifecycle:** the runner creates it and unlinks it at exit. The sweep runner removes stale files whose owner PID is gone.
- **Byte order:** little-endian.

Header (4,096 B):

| Off | Type | Field |
|---|---|---|
| 0 | char[8] | magic `UC72FRM1` |
| 8 | u32 | version = 1 |
| 12 | u32 | header_bytes = 4096 |
| 16 | u32 | n_slots (16) |
| 20 | u32 | n_views (2) |
| 24 / 28 / 32 | u32 | height, width, channels (512, 512, 3) |
| 36 | u32 | raster_code = 1: raw robosuite array; upright raster = `raw[::-1, :]` (H1's decode chain `[::-1, ::-1]` then `[:, ::-1]`) |
| 40 | u64 | slot_bytes |
| 48 | i64 | latest_frame_id (written last; −1 before the first write) |
| 56 | u32 | writer_pid |
| 60 | u32 | publish_every_n_steps (current) |
| 64 | u8[32] | static_calibration_sha256 (`camera-calibration-1`) |
| 96 | char[16]×2 | view names `external`, `wrist` |
| 128 | u8[64] | run_id |

Slot k starts at byte `4096 + k·slot_bytes`; slot_bytes = 8,448 + 2·512·512·3 = 1,581,312 B, so 16 slots ≈ 25 MB.

| Off | Type | Field |
|---|---|---|
| 0 | u64 | seq_begin |
| 8 | i64 | frame_id (monotonic per episode; reset = 0) |
| 16 | i64 | step |
| 24 | i64 | capture_monotonic_ns (`CLOCK_MONOTONIC`, system-wide, so comparable across processes) |
| 32 | f64 | sim_time_s |
| 40 | u32 | flags (bit0 reset, bit1 decision, bit2 event) |
| 44 | i32 | gripper_cmd (−1 open, 0 keep, +1 close) |
| 48 | f64[8] | state8 |
| 112 | f64[7] | q7 |
| 168 | f64[7] | qd7 |
| 224 | u32 | geometry_len |
| 232 | u64 | seq_end |
| 256 | u8[8192] | geometry_json (per-frame camera poses, the H1 `agent_geometry` dict) |
| 8448 | u8[2·H·W·3] | pixels, view-major (external, then wrist) |

- **Writer** (sim worker, after each `env.step` every `publish_every_n_steps`, at reset, and always for decision or event frames): write seq_begin = s, then the payload, then seq_end = s, then the header `latest_frame_id`.
- **Reader:** copy the slot, then accept only if seq_begin == seq_end == s and frame_id matches; otherwise retry. `get(frame_id)` raises `StaleFrame` if the slot was overwritten. That is a bug detector, never a silent substitute.
- **Perception copies a frame out of the ring when it receives `observe(frame_id)`**, before queuing work, so event frames cannot be lost to overwrite during a 1–2 s decision refresh.
- **Publish rate.** `publish_every_n_steps` = the minimum interval any subscriber declares. Perception declares its measured track cycle at warm-up; the recorder declares its rate.

```python
class FrameRing:                                     # uc72/frames.py, py3.10-compatible, numpy only
    @classmethod
    def create(cls, path, *, n_slots, views, height, width, run_id, calibration_sha256) -> "FrameRing"
    @classmethod
    def open(cls, path) -> "FrameRing"
    def publish(self, *, frame_id, step, sim_time_s, flags, gripper_cmd, state8, q7, qd7, geometry: dict, pixels: dict) -> None
    def latest(self) -> "Frame | None"
    def get(self, frame_id: int) -> "Frame"          # StaleFrame if overwritten
@dataclass(frozen=True)
class Frame:
    frame_id: int; step: int; capture_monotonic_ns: int; sim_time_s: float; flags: int; gripper_cmd: int
    state8: np.ndarray; q7: np.ndarray; qd7: np.ndarray; geometry: dict
    def rgb_raw(self, view: str) -> np.ndarray       # copy, HxWx3 uint8
    def rgb_upright(self, view: str) -> np.ndarray   # == H1's decoded StepView raster, bit-exact (PNG is lossless)
```

- **`FrameObservation`** (R7) is a lazy, `PolicyObservation`-compatible view on a frame. It produces data-URL images on demand for legacy callers, and its `__eq__` compares `(frame_id, state, prompt, joints)`.
- **`FrameProvider.capture_now()`** (protocol SI-13) returns the latest ring frame for the current step, or issues a worker `render` op without a physics step.

### I-04 Service protocol (owner ws_runner_measurement; operations owned by each service)

```
transport  unix stream socket, UTF-8, one JSON object per line
request    {"id": int, "op": str, "args": {...}}
reply      {"id": int, "ok": bool, "result": {...}|null, "error": {"type": str, "message": str}|null,
            "timing": {"queue_s": float, "compute_s": float}}
mandatory  hello         -> {"service","version","pid","interpreter":{python,packages_sha256,numpy},"code_sha256",
                             "vendor_sha256","weights":{name: sha256},"device":{"index","uuid","name"},"warm":bool,
                             "load_s","latency_model":{...}}
           health        -> {"status":"ok|degraded|unhealthy","queue":int,"last_latency_s","errors":[...],"gpu_free_gib"}
           reset_episode -> {"ok": true}   (args are service-specific)
           shutdown      -> {"ok": true}
ServiceSpec {name, argv, env, socket_path, required: bool, resident: bool, health_timeout_s}
```

- Every client call has a timeout.
- A `required=false` service is fail-open: client calls catch every exception and return a status, never raising into the episode.
- `ServiceHandle.submit(op, args) -> Future` is non-blocking. `call(op, args, timeout_s)` is blocking, and the caller wraps it in `timers.block("perception")`.
- Perception operations: `hello, reset_episode, set_plan, observe, snapshot, relocalize, notify, get_masks (diagnostic), health, shutdown`. The design's `bind_roles` op is dropped: binding overrides arrive as `notify` events.

### I-05 Camera calibration (owner ws_geometry_tools; the worker emits it)

```json
{"schema": "camera-calibration-1", "frame": "world", "image": {"width": 512, "height": 512},
 "cameras": {
   "external": {"model": "pinhole", "K": [[618.04,0,255.5],[0,618.04,255.5],[0,0,1]], "dist": null,
                "mount": "world", "T_world_camera": [[4x4]], "convention": "opencv"},
   "wrist":    {"model": "pinhole", "K": [[333.63,0,255.5],[0,333.63,255.5],[0,0,1]], "dist": null,
                "mount": "hand", "T_hand_camera": [[4x4]], "convention": "opencv"}},
 "time_offset_s": {"external": 0.0, "wrist": 0.0}, "source": "sim_mujoco|checkerboard|file", "sha256": "…"}
```

- **Emission.** The worker emits it at reset from MuJoCo (fovy → K; camera poses), including `camera_overrides`. The wrist pose per frame = T_world_hand(state8) · T_hand_camera, checked against the per-frame geometry JSON to 1e-6.
- **Validation in the new path.** It replaces the ideal-pinhole validator (`astranomy_agentic_geometry.py:104-112`) with a consistency check: finite values, positive focal length, `dist` either null or OpenCV `[k1,k2,p1,p2,k3]`. The validator stays only on the `h1_equivalent` path.

```python
@dataclass(frozen=True)
class CameraModel:                      # geometry_tools/camera.py; OpenGL axes internally (+x right, +y up, −z forward)
    name: str; width: int; height: int; fx: float; fy: float; cx: float; cy: float
    R_world_from_cam: np.ndarray; t_world: np.ndarray; distortion: tuple[float, ...] = (); mount: str = "fixed_world"
    @classmethod
    def from_calibration(cls, cal: dict, view: str, *, state8=None) -> "CameraModel"   # OpenCV → OpenGL conversion
    @classmethod
    def from_geometry(cls, geometry: dict, label: str) -> "CameraModel"               # H1 metadata adapter
    def rays(self, uv) -> tuple[np.ndarray, np.ndarray]
    def project(self, xyz) -> tuple[np.ndarray, np.ndarray]
    def project_jacobian(self, xyz) -> np.ndarray
```

The perception service uses this same class (it does not implement its own; PS-2 keeps only the ring reader).

### I-06 Perception snapshot and object inventory (owner ws_perception_service)

The full JSON Schema (Draft 2020-12) lives in `schemas/perception_snapshot_v1.schema.json`. It is the perception design's §4.4 with four integration amendments:
1. Per-view `polygon_uv` (≤ 64 vertices) and `centroid_uv`, for locate snapping and overlays without mask round trips.
2. `roles` keyed by plan part ids (A–H) and inventory ids (O1–O12).
3. Binding sources limited to cells, direction hints and prompt hits (model binds go through SceneMap `scene_edits`).
4. No model-facing compact facts.

```json
{"schema": "perception-snapshot-1", "episode_id": "…", "snapshot_seq": 17,
 "status": "ok|degraded|unavailable", "status_reason": null,
 "frame": {"frame_id": 412, "step": 412, "capture_monotonic_ns": 0, "mode": "full|track|relocalize",
           "reason": "reset|decision|event|track"},
 "processing": {"started_monotonic_ns": 0, "finished_monotonic_ns": 0, "stage_s": {"sam_encode": 0.31, "da3": 0.27},
                "device": {"index": 7, "uuid": "GPU-…"}, "service_version": "perception-service-1", "vendor_sha256": "…"},
 "calibration_sha256": "…",
 "supports": [{"id": "ps1", "normal": [0,0,1], "height_m": 0.437, "sigma_m": 0.003, "extent_xy": [[x,y]],
               "sources": [{"kind": "sweep_ext_wrist|sweep_wrist_motion|da3_external_ransac|contact_anchor",
                            "height_m": 0.437, "sigma_m": 0.004, "n_patches": 312, "step": 0, "used": true}],
               "observed_at_step": 0}],
 "robot": {"step": 412, "source": "fk_visual_envelopes", "dilation_px": 2, "model_sha256": "…", "payload_rendered": false},
 "instances": [{
   "id": "inst_007", "category": "mug|unknown_object", "labels": [{"prompt": "mug", "score": 0.91}],
   "detect_source": "sam_text|sam_box_exemplar|depth_component",
   "plan_role": "target_object|reference_object|fixture|plan_object|non_plan",
   "plan_parts": ["A", "B"], "inventory_id": "O1",
   "state": "static|held|engaged|released_settling|possibly_moved|missing",
   "primitive": {"kind": "cylinder|box", "center_m": [3], "radius_m": 0.04, "half_extents_m": null, "yaw_rad": 0.0,
                 "z_min_m": 0.437, "z_max_m": 0.541, "sigma_center_m": [3], "sigma_along_external_ray_m": 0.009,
                 "sigma_extent_m": [3], "conservative_aabb": {"lower_m": [3], "upper_m": [3]},
                 "source": "single_view_support_contact|ray_pair_triangulation|multiview_silhouette_fit|held_rigid|depth_component|unknown_held_envelope",
                 "views_used": [{"view": "external", "frame_id": 412}]},
   "views": {"external": {"frame_id": 412, "bbox_xyxy": [4], "polygon_uv": [[u,v]], "centroid_uv": [u,v],
                          "pixels": 3914, "truncated": false, "occluded_fraction": 0.0},
             "wrist": null},
   "observed_at_step": 412, "capture_monotonic_ns": 0, "valid": true, "needs_check": false,
   "invalidated_by": null,
   "association": [{"views": ["external","wrist"], "ray_gap_m": 0.012, "gate_m": 0.06, "decision": "same|different"}],
   "parts": [{"noun": "handle", "center_m": [3], "sigma_m": 0.006, "source": "ray_pair|ray_instance_interval", "plan_part": "B"}]}],
 "roles": {"A": {"instance_id": "inst_002", "part": "whole", "source": "plan_cell|direction_hint|primary_prompt|unbound",
                 "margin": 0.6, "alternatives": ["inst_005"]}},
 "held": {"state": "none|candidate|held_confirmed|grasp_not_confirmed|engaged", "instance_id": null, "since_step": null,
          "hand_from_object": null, "primitive_in_hand": null},
 "events": [{"type": "instance_new|instance_missing|possibly_moved|needs_check|held_confirmed|grasp_not_confirmed|released|binding_changed|binding_needed|support_updated|calibration_mismatch|degraded_stage",
             "step": 412, "instance_id": null, "detail": ""}],
 "height_field": {"path": "trial/perception/hf/000017.npz", "sha256": "…", "cell_m": 0.005, "origin_xy": [x,y], "shape": [ny,nx], "step": 412},
 "limits": {"frame": "world", "units": "m", "sigma_is_one_sd": true, "conservative_k": 2.0, "unseen_space": "unknown, not free"}}
```

- **`invalidated_by`** is null or `{"type": "robot_near|contact|grasp|release|change_detected|missed", "step": int}`.
- **`notify` events** are `robot-event-1` (I-13) plus `{"type": "bind_override", "part": "A", "instance_id": "inst_005"}`.
- **`RelocalizeResult`:** `{"frame_id", "capture_monotonic_ns", "age_at_use_s", "results": [{"instance_id", "status": "confirmed|updated|moved|lost|timeout|unknown_instance", "primitive": {...}, "sigma_inflated": bool}]}`.
- **HeightField npz:** `z_max` and `sigma` (float32 H×W), `observed` and `shadow` (bool), `source_mask` (uint8 bits: external, wrist, primitive); 5 mm cells over the observed support extent.
- **Consumption log:** `trial/perception/consumed.jsonl` rows `{"step", "consumer": "decision|stage_entry|relocalize", "snapshot_seq", "sha256", "wait_s"}`, with snapshots stored under `trial/perception/snapshots/`. `RecordedPerceptionClient` serves them for GPU-free deterministic replays.

### I-07 Plan scene inventory (owner ws_protocol_context)
Plan extension `objects` (PC3), conformed by `agent_speedup_v2_user_check_plan.py` (`_conform` gains `integer` with min/max and `boolean`).

```json
"objects": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "object", "additionalProperties": false,
  "required": ["id", "noun", "attributes", "x", "w", "parts", "role"],
  "properties": {
    "id": {"type": "string", "enum": ["O1","O2","O3","O4","O5","O6","O7","O8","O9","O10","O11","O12"]},
    "noun": {"type": "string", "minLength": 2, "maxLength": 32},
    "attributes": {"type": "string", "maxLength": 32},
    "x": {"type": "integer", "minimum": 0, "maximum": 64},
    "w": {"type": "integer", "minimum": 0, "maximum": 64},
    "parts": {"type": "array", "maxItems": 8, "items": {"type": "string", "enum": ["A","B","C","D","E","F","G","H"]}},
    "role": {"type": "string", "enum": ["manipulated", "destination", "support", "fixture", "other"]}}}}
```

- Instruction text (about 70 tokens, A/B'd by PC7): every distinct physical object on or above the work surface, task objects first; `noun` is the plain category; `attributes` are colour or pattern words; `x`/`w` are cells as for parts; `parts` lists its part ids.
- Advisory issues, never aborts: `inventory_part_unbound`, `inventory_noun_has_direction_word`, `inventory_duplicate_id`.
- **Fallback.** A plan without `objects` gets an inventory derived by GT10 (the `object_words` / `same_object` grouping), flagged `derived_from_parts`.

### I-08 Harness object map (owner ws_geometry_tools)

**One model-facing id namespace**, assigned by the SceneMap:

| Id | Meaning |
|---|---|
| `o1` … `o24` | measured objects, stable for the episode; further objects stay in the obstacle set as `other` |
| `A` … `H` | plan part ids, aliases bound to an o-id (and part) |
| `O1` … `O12` | plan inventory ids, aliases bound to an o-id |
| `L1` … `L4` | `locate` points |
| `S1` … `S4` | supports (S1 = dominant) |
| `inst_NNN` | perception track ids; internal, never shown to the model, mapped 1:1 to o-ids by `track_id` |
| `mb1` … `mb4` | model-written boxes, M1–M2 only (H1 `obstacle_map`) |

```python
@dataclass
class ObjectRecord:                       # geometry_tools/scene_map.py
    id: str; name: str | None; aliases: list[str]                      # ["A", "B", "O1"]
    kind: Literal["object", "fixture", "support", "unknown"]
    plan_role: Literal["manipulated", "destination", "support", "fixture", "other", "none"]
    state: Literal["static", "held", "held_unverified", "moved_unobserved", "removed_by_model"]
    box: Box3D | None; obb: dict | None; footprint_xy: np.ndarray | None
    source: Literal["perception_two_view", "perception_single_view", "perception_multiview", "locate", "contact", "attached"]
    track_id: str | None; evidence: list[dict]; observed_step: int | None; updated_step: int; last_moved_step: int | None
    attached: dict | None; contact_overrides: list[dict]; warnings: list[dict]; revision: int
class SceneMap:
    revision: int
    def ingest_perception(self, snapshot: dict, frame: FrameRecord) -> list[dict]   # adopts bindings as aliases
    def add_points(self, results: list["LocateResult"]) -> None
    def apply_edits(self, edits: list[dict], *, step: int) -> list[dict]            # per-edit feedback
    def on_robot_event(self, event: dict, *, state8, q7, step: int) -> list[dict]   # attach / detach / support anchors
    def predict_held(self, state8) -> None
    def anchor(self, anchor_id: str) -> "AnchorGeometry"
    def obstacle_set(self, intent: "ContactIntent | None", *, step: int) -> dict    # ObstacleSet below
    def held(self) -> dict | None
    def model_facts(self) -> dict                                                   # context.scene below
    def snapshot(self) -> dict                                                      # audit
```

**Held record (single owner).**
- **Attach** on `grasp_established` (I-13). Membership is the last approach stage's anchor; if there is none, the record whose box inflated by kσ contains the pad midpoint at closure.
- **Fallback jaw-column envelope**, defined once for the whole build: width = settled gap + 2·pad thickness (8 mm); depth = pad extent (16 mm) + 2σ; height from the fingertip (+9.25 mm) down to the support z at closure.
- **Verification.** `verified = gap_and_comotion` after perception's `held_confirmed`, `gap_only` otherwise. `held_unverified` is still swept, which is conservative.
- **Detach** on `released`: the world box is the last prediction, state `moved_unobserved`, σ_z inflated by the measured release height until perception re-localises it.
- **Never skipped.** The held-object check is never skipped (H2 K2/F1).

**`ObstacleSet`** (`harness-object-map-1`, SceneMap → planners):

```json
{"schema": "harness-object-map-1", "revision": 14, "step": 123, "k_sigma": 2.0,
 "supports": [{"id": "S1", "z": 0.437, "sigma_m": 0.003, "extent_xy": [[x,y]], "sources": ["perception_sweep", "contact"]}],
 "objects": [{"id": "o3", "aliases": ["C", "O2"], "name": "plate", "kind": "object", "plan_role": "destination",
              "state": "static", "aabb": {"lower_m": [3], "upper_m": [3]},
              "obb": {"center_m": [3], "rotation_wxyz": [4], "half_extents_m": [3]},
              "footprint_xy": [[x,y]], "sigma_m": [3], "inflate_m": [3], "source": "perception_single_view",
              "observed_step": 120, "track_id": "inst_004", "movable": true,
              "contact": {"fingers": "forbidden|permitted", "palm": "forbidden|permitted", "links": "forbidden",
                          "held_object": "forbidden|permitted|permitted_nonincreasing"}}],
 "held": {"id": "o1", "T_grip_object": [[4x4]], "grip_box": {"lower_m": [3], "upper_m": [3]}, "sigma_m": [3],
          "verified": "gap_and_comotion|gap_only|unverified", "attached_step": 140, "support_id": "S1",
          "source": "map_at_close|fallback_jaw_column"},
 "excluded": [{"id": "o9", "why": "inside_robot|held|removed_by_model"}],
 "unknown_volumes": []}
```

- `aabb` is the estimate (unexpanded); `inflate_m = k·σ` per axis. Planners measure exact distances to `aabb + inflate_m`.
- Escape and "strictly increasing distance" rules use the unexpanded box. There is no lattice margin.
- `held` is null when nothing is held.

**Per-stage contact permissions** (one rule set, merging GT3 and SP5; computed from `ContactIntent`):

| Robot part \ object | Intent anchor (`ContactIntent.with`) | Other objects and fixtures | Supports | Held object |
|---|---|---|---|---|
| fingers / pads | permitted | forbidden | forbidden | permitted (gripped) |
| palm | permitted only for intent `push`/`insert` (a `scene_edits allow_contact palm`) | forbidden | forbidden | forbidden; reported `palm_on_held` (grasp too deep) |
| arm links | forbidden | forbidden | forbidden | forbidden (self) |
| held object | the destination support or region on place stages; the pick support on the first lift leg (`permitted_nonincreasing`) | forbidden | as left | — |

Overrides come from `scene_edits allow_contact` / `forbid_contact`, scoped to the next `execute` fragment.

**Model-facing `context.scene`** (current only; about 300–500 tokens per call for LIBERO; ≤ 2,000 chars; stripped to `{"revision": N, "superseded": true}` at transcript insertion by PC4):

```json
{"revision": 14, "observed_step": 120, "units": "m", "sigma": "1-sigma per axis", "perception": "ok|degraded|unavailable",
 "supports": [{"id": "S1", "z": 0.437, "s": 0.003, "src": ["perception", "contact"]}],
 "objects": [{"id": "o1", "aliases": ["A", "B", "O1"], "name": "mug", "kind": "object", "state": "static",
              "box": [[-0.130,-0.150,0.437],[-0.050,-0.070,0.542]], "s": [0.006,0.006,0.004], "src": "two_view", "seen": 120}],
 "held": null,
 "points": [{"id": "L1", "label": "handle", "xyz": [-0.06,-0.11,0.51], "s": [0.004,0.004,0.006], "on": "o1",
             "src": "pointed_two_view", "res_px": 2.1, "step": 118}],
 "regions": [{"id": "D", "rel": "right_of", "anchor": "C", "target": [0.10, -0.20, 0.44], "half": [0.04, 0.04], "gap": 0.014}],
 "warnings": [{"id": "o5", "code": "below_support|floating|inside_robot|duplicate|oversize|moved|not_seen_where_expected"}],
 "unbound_plan_parts": ["E"]}
```

This is the only model-facing perception/geometry text block. PS-10's compact facts and protocol SI-4 `PerceptionFacts` are not sent.

### I-09 Geometry reference grammar (owner ws_geometry_tools)

Strict structured-output fragments. The wire schema inlines everything (no `$ref`, no `pattern`), and every key is required.

```json
"target-ref-1": {"type": "object", "additionalProperties": false,
  "required": ["anchor", "point", "offset_m", "frame", "tool_point"],
  "properties": {
    "anchor": {"type": "string", "enum": ["A","B","C","D","E","F","G","H","O1","…","O12","o1","…","o24","L1","L2","L3","L4","S1","S2","S3","S4"]},
    "point": {"type": "string", "enum": ["center","top","bottom","front","behind","left","right",
                                         "beside_front","beside_behind","beside_left","beside_right","inside"]},
    "offset_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
    "frame": {"type": "string", "enum": ["world", "viewer"]},
    "tool_point": {"type": "string", "enum": ["grip_site", "fingertips", "palm", "held_bottom", "held_center"]}}}

"locate-1": {"type": "object", "additionalProperties": false, "required": ["id", "label", "surface", "observations"],
  "properties": {"id": {"enum": ["L1","L2","L3","L4"]}, "label": {"type": "string", "minLength": 1, "maxLength": 60},
    "surface": {"enum": ["support", "object", "unknown"]},
    "observations": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "object", "additionalProperties": false,
      "required": ["frame", "view", "uv", "box_uv"],
      "properties": {"frame": {"type": "integer", "minimum": 0}, "view": {"enum": ["external", "wrist"]},
        "uv": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}]},
        "box_uv": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}]}}}}}}

"scene-edit-1": {"type": "object", "additionalProperties": false, "required": ["op", "id", "value", "reason"],
  "properties": {"op": {"enum": ["label","bind","kind","allow_contact","forbid_contact","merge","remove","restore","add_from_locate"]},
    "id": {"type": "string", "enum": ["A","…","H","O1","…","O12","o1","…","o24","L1","…","L4"]},
    "value": {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 60}, {"type": "null"}]},
    "reason": {"type": "string", "minLength": 1, "maxLength": 120}}}
```

`scene_edits` values by op:

| op | value |
|---|---|
| label | a name |
| bind | the o-id or L-id for a plan part or inventory id |
| kind | fixture / movable / support |
| allow_contact / forbid_contact | palm / fingers / held_object |
| merge | the other o-id |
| remove / restore | null |
| add_from_locate | an L-id |

There are no metric fields. Each edit returns `{op, id, status: applied|rejected, reason_code}`.

```json
"ResolvedTarget": {"xyz_m": [3], "quaternion_wxyz": [4], "sigma_m": [3], "anchor": "o3", "anchor_kind": "object|part|region|point|support",
  "anchor_revision": 7, "point": "top", "tool_point": "fingertips", "frame": "world",
  "terms": {"anchor_point_m": [3], "offset_world_m": [3], "tool_offset_m": [3], "uncertainty_margin_m": 0.012,
            "palm_clamp_m": 0.0, "palm_clearance_m": 0.018},
  "sigma_along_motion_m": 0.006, "scene_revision_at_decision": 12, "scene_revision_at_use": 13,
  "evidence_frames": [118, 120], "evidence_age_steps": 3, "source": "reference|raw_world|relative",
  "warnings": ["part_geometry_unbound: using whole object o1"]}

"ContactIntent": {"stage_key": "f12.s2", "with": ["o3"], "parts": ["fingers"],
  "intent": "grasp|place|push|insert|release|none", "held_object_to": ["S1"],
  "overrides": [{"id": "o4", "part": "palm", "source": "model_edit", "reason": "…"}], "derived_from": "target|legacy_nearest|edit"}
```

**Resolution, done at each segment's measured entry** (GT4):
1. Find the anchor.
2. Take the point and its covariance; face points use the support function along the frozen viewer frame.
3. Apply the offset (`viewer` = [right, away, up]).
4. Apply the tool point: fingertips +9.25 mm along the approach axis; palm −31.04 mm (`robot_facts_v2.json`).
5. Add the uncertainty margin k·√(ôᵀΣô), only on transit stages with a nonzero offset.
6. Apply the palm depth bound from SP8 (`swept.depth_bound`).
7. Derive the contact intent.
8. Log both scene revisions.

Raw world `xyz_m` targets go through `annotate()`, which reports the nearest object and `raw_target_inside_object` and applies the same depth-bound rule; there is no veto (UD-27). An unknown id costs one zero-motion handoff (`reference_unknown`).

### I-10 Robot model API (owner ws_swept_planner)

```python
# robot/kinematics.py — numpy only, both venvs; loads vendor/panda_robosuite_1_4_1/*.xml through vendor/robodojo_direct_sim.py
class PandaChain:
    VERSION = "panda-chain-1"; low: np.ndarray; high: np.ndarray          # joint limits from robot.xml
    @classmethod
    def load(cls) -> "PandaChain"                                       # verifies the manifest hashes
    def bind(self, state8, q7) -> np.ndarray                            # world_from_base = measured · FK(q)^-1
    def check_binding(self, state8, q7, *, tol_m=1e-6) -> dict          # {"ok", "err_m"}; health check every 20 steps
    def fk(self, q7, *, world_from_base) -> tuple[np.ndarray, np.ndarray]   # T_world_grip (4x4), J (6x7)
    def fk_batch(self, q (K,7), q_fingers (K,2), *, world_from_base) -> dict[str, np.ndarray]   # per body (K,4,4)
    def gripper_from_state8(self, states8 (K,8)) -> dict[str, np.ndarray]   # no joints needed
    def joint_margins(self, q7) -> np.ndarray
    def lift(self, q0, states8, *, every_m=0.02, every_rad=0.10) -> "LiftResult"   # min-norm resolved rate, no nullspace
# robot/body.py
class RobotBodyModel:                                                   # assets/panda_robot_model_v1.npz (+ .json manifest)
    VERSION = "panda-body-1"; parts: list["PartInfo"]; beta_m: dict     # {"gripper": 0.004, "arm": 0.010} (UD-33)
    def spheres(self, poses) -> tuple[np.ndarray, np.ndarray, np.ndarray]   # centers (K,S,3), radii (S,), part_index (S,)
    def collision_hulls(self, poses) -> list[dict]                      # vertices, unit-normal planes per part
    def visual_envelopes(self, poses) -> list[dict]                     # for the robot raster (PS-3), incl. the mount
    def silhouette(self, camera: "CameraModel", q7, q_fingers, *, world_from_base) -> list[np.ndarray]   # polygons
    palm_profile_mm: np.ndarray; footprints: dict                       # parity with robot_facts_v2.json (≤ 0.5 mm)
# swept/checker.py
class SweptChecker:
    def check(self, cfg: "ConfigBatch", scene: "SceneArrays", perms: "PermissionMatrix", *, held=None, bands=None,
              parts=("arm", "palm", "finger", "pad", "held")) -> dict      # SweepReport
# executor/reachability.py (executor-owned consumer of PandaChain)
def ik_local(chain, xyz, R, seed) -> dict                               # IKReport, DLS 140 iterations, OSC-like
def ik_global(chain, xyz, R, seeds) -> dict
def reachability(chain, samples, q_now, q_reset) -> dict                # reach-1
```

JSON contracts:
- IKReport: `{"ok", "position_error_m", "orientation_error_deg", "iterations", "joints_rad", "min_margin_rad", "limiting_joint", "clamped_joints", "seed"}`.
- `reach-1`: `{"version", "classification": "reachable|posture_limited|unreachable|unavailable", "local", "global", "path_samples", "path_min_margin_rad", "path_limiting_joint", "time_ms"}`.
- SweepReport: `{"ok", "first_violation", "min_clearance_by_class": {class: {"m", "object_id", "part", "k"}}, "clearance_curve_m", "conservatism_m": {"gripper", "arm"}, "pairs_evaluated", "timing_ms": {"fk", "place", "broad", "narrow"}}`.

**Body model asset** (built once by `tools/build_robot_model.py` in `.venv-sim`, committed, sha in the digest):
- Per collision part: sphere centres and radii in the body frame, hull vertices, unit-normal plane equations and a bounding sphere.
- Per body: a visual envelope (hull vertices + ≤ 98 supporting planes; the RethinkMount is included).
- The kinematic chain, the palm profile p(ξ) and the footprints.
- `panda_robot_model_v1.json` records the XML and mesh sha256s, the builder sha, numpy/scipy versions, β, the RNG seed, and per-part coverage, bulge and growth.

**Budgets** (controller `.venv`, single thread): point ≤ 5 ms; profile K = 60, M = 20 ≤ 15 ms; K = 200 ≤ 30 ms; per step ≤ 3 ms (p95).

### I-11 Executed profile and OSC plant (owner ws_executor_control)

`executor/osc_response.json` (`osc-response-1`):
- `conditions`: controller, kp, damping, control_hz, translation_scale_m, rotation_scale_rad, robosuite, mujoco.
- `corpus`: runs with trace sha and steps used.
- `translation`: `{a, b, rho, r2, n_steps, a_ci95, a_per_run, residual_std_m, rest_noise_p99_m}`.
- `rotation`: `{a, b, rho, r2, saturated_rate_rad_per_step}`.
- `guard`: `{f, window_steps, c_eval_m, rotation_eval_max_rad, free_ratio_quantiles, detection_delay_steps_median}`.
- `gripper`: `{empty_curve_mm, inward_noise_mm, touch_threshold_mm}`.
- `derived`: `{braking_gain, settle_steps_translation, settle_steps_rotation}`.
- The plant refuses to load if the live worker's controller metadata differs from `conditions`.
- The log fit gave a = 0.711, b = 0.0732, ρ = 0.254, R² = 0.973; rotation a_r = 0.90, ρ_r = 0.27.

```python
@dataclass(frozen=True)
class ExecutedProfile:                       # executed-profile-1; one representation for executor and swept
    stage_key: str; mode: Literal["legacy_concurrent", "translate_only", "rotate_then_translate",
                                  "translate_then_rotate", "translate_rotate_translate", "coupled"]
    states8: np.ndarray        # (K,8) state8 layout [x,y,z, rotvec(3), finger1, finger2] — places the gripper directly
    step: np.ndarray           # (K,)
    phase: tuple[str, ...]     # per sample: rotate|translate|coupled|hold|gripper|escape|backoff|posture
    arc_m: np.ndarray          # (K,)
    tracking_band_m: np.ndarray  # (K,) = a·|s⊥|/(1−a) + 3·σ_track (plant-predicted lateral excursion + noise)
    arrival_step: int; predicted_steps: dict; rotation_site: dict; gripper: dict; held_id: str | None; plant_sha256: str
class ExecutedProfileModel(Protocol):
    version: str
    def candidates(self, entry_state8, target_xyz, target_wxyz, *, profile: str, path_xyz=None) -> list[str]  # modes
    def predict(self, entry_state8, entry_velocity6, waypoints, *, mode: str, profile: str,
                speed_cap=None) -> ExecutedProfile
    def osc_response(self) -> dict          # {"a","b","a_p90","rotation_rad_per_step","updated_at_step"}
    def guard_latency_steps(self) -> int    # measured detection delay (osc_response.json guard)
    def coast_m(self, s_along_m: float) -> float
```

The swept design's mode names map to these: `rotate_first` = `rotate_then_translate`; `translate_first` = `translate_then_rotate`; `executor_default` = the executor's first candidate. JSON logs downsample to every 4th step and carry canonical quaternions (hemisphere rule, I-16).

### I-12 Stage-planner seam (owner ws_executor_control; the body planner implementation is owned by ws_swept_planner)

```python
@dataclass(frozen=True)
class StageRequest:
    stage_key: str; skill: str; profile: str; stop_on: str; contact_window_m: float | None
    target: dict | None                  # ResolvedTarget (I-09) or None for legacy raw/relative targets
    target_xyz: np.ndarray; target_wxyz: np.ndarray
    intent: dict                         # ContactIntent (I-09)
    next_stage: "StageRequest | None"; max_steps: int | None
@dataclass(frozen=True)
class MeasuredState: state8: np.ndarray; q7: np.ndarray; qd7: np.ndarray; s_prev: np.ndarray; step: int
@dataclass(frozen=True)
class Leg:
    kind: Literal["escape", "transit", "lift", "sidestep", "approach", "contact", "retreat", "hold"]
    waypoints: list; mode: str; profile: ExecutedProfile; min_clearance_m: float
    speed_cap: "Callable[[float], float] | None"; reason: dict | None
@dataclass(frozen=True)
class StagePlan: status: str; legs: tuple[Leg, ...]; report: dict          # body_plan v1 (both planners emit it)
class StagePlanner(Protocol):
    planner_id: str                                                        # "point_v2" | "body_sweep_v1"
    def plan_stage(self, req: StageRequest, now: MeasuredState, scene: dict, held: dict | None,
                   profiles: "ExecutedProfileModel", *, budget_ms: float) -> StagePlan
    def monitor_step(self, plan: StagePlan, now: MeasuredState, scene: dict) -> Literal["continue", "replan", "hold_handoff"]
class ClearanceOracle(Protocol):                                           # GripSiteBoxOracle (interim), SweptOracle (M4)
    oracle_id: str
    def point_clearance(self, pose, *, gripper_gap_m, held) -> dict        # {"signed_m","obstacle_id","robot_part","required_margin_m","source","captured_at_step"}
    def segment_clearance(self, a, b, *, gripper_gap_m, held) -> dict
    def profile_clearance(self, profile: ExecutedProfile, *, held) -> dict # {"min_clearance_m","argmin_k","robot_part","obstacle_id","violations","time_ms"}
    def gradient(self, pose, *, gripper_gap_m, held) -> np.ndarray
    def nearest_pairs(self, state8, q7, scene) -> list[dict]               # for blocked_motion feedback
```

- **`StagePipeline`** (`executor/stage_pipeline.py`) owns per-stage execution: `planner.plan_stage` → legs → `BrakingPathTracker` (per-arc `speed_cap` from `SpeedPolicy` in M4+, the minimum with the braking law) → per step `ProgressMonitor.update` and `planner.monitor_step`.
- **Per-step precedence:**
  1. A monitor event goes to `GuardPolicy.resolve` (in-band → arrival; declared contact → `contact_reached`; undeclared → hold and back-off → `blocked_motion`).
  2. Otherwise the planner verdict applies: `replan` re-plans once from the measured state with escape semantics; a second violation → `route_unavailable` / `blocked_by_geometry` handoff.
  3. Otherwise grasp/held predicate failures yield.
  4. Otherwise arrival or budget.
- **`body_plan` v1** is the swept design's §4.1 schema, `status ∈ {direct, escape_then_direct, detour, escape_then_detour, standoff_handoff, merged_into_contact_stage, depth_bounded, truncated_at_clearance, jaw_sweep_blocked, escape_partial, no_route, no_route_budget}`. Its compact model-facing form is ≤ 400 chars, reasons verbatim. The point planner emits the same schema, with `legs[*].min_clearance_m` for the grip site.
- **`trial/swept_shadow.jsonl`** (`swept-shadow-v1`): `{"step", "event": "stage_entry|step_check", "stage_key", "legacy": {...}, "body": {...}, "measured": {"state8", "q"}, "timing_ms"}`. It is never model-visible.

### I-13 Robot event stream (owner ws_executor_control)
One stream replaces the four event shapes in the designs (executor S7, geometry §4.5, swept I5, perception notify). Only the sub-object matching `type` is present.

```json
{"schema": "robot-event-1", "type": "gripper_command|grasp_established|grasp_absent|grasp_lost|released|guarded_stop|stage_start|stage_end|decision",
 "step": 412, "t_v": 20.6,
 "gripper_command": {"command": "open|close"},
 "grasp_established": {"settled_gap_mm": 14.2, "settle_k": 14, "empty_curve_gap_mm": 2.5, "gap_band_mm": null,
                       "anchor_id": "o1", "grip_state8": [8]},
 "grasp_absent": {"settled_gap_mm": 1.4, "anchor_id": "o1"},
 "grasp_lost": {"gap_mm": 1.0, "gap_change_mm": -12.3, "held_id": "o1"},
 "released": {"gap_mm": 78.9, "held_id": "o1", "grip_state8": [8]},
 "guarded_stop": {"channel": "progress_guard|finger_touch|no_progress|rotation_guard", "expected": true, "anchor_id": "o2",
                  "nearest_part": "finger|pad|palm|link|held", "position_m": [3], "direction_world": [3],
                  "depth_from_entry_m": 0.041, "penetration_sigma_m": 0.002, "support_candidate": true},
 "stage_start": {"stage_key": "f12.s2", "skill": "approach", "uses_ids": ["o4"], "planner_id": "point_v2"},
 "stage_end": {"stage_key": "f12.s2", "status": "succeeded", "reason": "contact_reached"}}
```

Consumers:
- SceneMap: attach, detach, support contact anchors with σ = kinematic σ ⊕ guard penetration σ.
- Perception bridge: forwards via `notify`.
- Swept: held geometry.
- Recorder: `events.jsonl` (envelope `{step, t, source, type, payload}`, plus `planner_time`, `escape_executed`, `swept_clearance`, `route_refused`, `late_failure_detected`).

### I-14 Executive decision and segment schema (owner ws_protocol_context)

The wire schema is versioned per milestone build. Each version is the previous one plus or minus the listed fields, with strict mode, all keys required and everything inlined.

| Version | Build | Change from the previous |
|---|---|---|
| `decision-uc72-m1` | b001 | H1 prompt-6 schema + `type` gains `give_up`; `finish`/`give_up` require `termination`; oracle cues deleted (PC1) |
| `decision-uc72-m2` | b002 | `macro` type/property removed; `why` dropped (UD-21); segments become `segment-uc72-1` without `target` and with `then ∈ {continue, decide}`; `route`/`observation_after` removed; `obstacle_map` kept (H1 semantics, ids `mb1…mb4`) |
| `decision-uc72-m3` | b003 | `obstacle_map` removed; segment `target` added; `locate` and `scene_edits` added (I-09) |
| `decision-uc72-m5` | b005 | `type` gains `resume`; `then` gains `verify_grasp`, `verify_view`, `verify_placement`, `verify_engagement`; segments 1..12 |

Final shape (`decision-uc72-m5`):

```json
{"type": "object", "additionalProperties": false,
 "required": ["belief", "plan", "type", "execution", "motion", "termination", "locate", "scene_edits"],
 "properties": {
  "belief": {"type": "string", "maxLength": 1800},
  "plan": {"type": "string", "maxLength": 600},
  "type": {"type": "string", "enum": ["execute", "resume", "motion", "observe", "finish", "give_up"]},
  "execution": {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": false, "required": ["segments"],
                 "properties": {"segments": {"type": "array", "minItems": 1, "maxItems": 12, "items": "segment-uc72-1 (inlined)"}}}]},
  "motion": {"anyOf": [{"type": "null"}, "H1 direct-motion object, unchanged (≤ 5 steps, ≤ 5 cm)"]},
  "termination": {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": false, "required": ["code", "evidence"],
                   "properties": {"code": {"type": "string", "enum": ["goal_satisfied", "unrecoverable", "unsafe", "other"]},
                                  "evidence": {"type": "string", "minLength": 1, "maxLength": 300}}}]},
  "locate": {"type": "array", "maxItems": 4, "items": "locate-1 (inlined)"},
  "scene_edits": {"type": "array", "maxItems": 6, "items": "scene-edit-1 (inlined)"}}}

"segment-uc72-1": {"type": "object", "additionalProperties": false,
 "required": ["skill", "xyz_m", "target", "quaternion_wxyz", "gripper", "profile", "stop_on", "contact_window_m", "grasp", "then", "max_steps"],
 "properties": {
  "skill": {"enum": ["move_to_pose", "move_relative", "approach", "set_gripper", "wait"]},
  "xyz_m": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}]},
  "target": {"anyOf": [{"type": "null"}, "target-ref-1 (inlined)"]},
  "quaternion_wxyz": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}]},
  "gripper": {"enum": ["keep", "open", "close"]},
  "profile": {"enum": ["transit", "contact"]},
  "stop_on": {"enum": ["arrival", "contact"]},
  "contact_window_m": {"anyOf": [{"type": "null"}, {"type": "number", "minimum": 0, "maximum": 0.5}]},
  "grasp": {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": false, "required": ["gap_band_mm"],
            "properties": {"gap_band_mm": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}]}}}]},
  "then": {"enum": ["continue", "decide", "verify_grasp", "verify_view", "verify_placement", "verify_engagement"]},
  "max_steps": {"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 1, "maximum": 240}]}}}
```

Semantic validation is done by the policy; errors are feedback, never schema errors:
- `move_to_pose` and `approach` need exactly one of `xyz_m` and `target`. `move_relative` needs `xyz_m`. `set_gripper` and `wait` need both null.
- `grasp` is non-null only on `set_gripper close` and means "an object is expected". `gap_band_mm` is optional. A deliberate closed-finger push has `grasp = null`.
- `stop_on = contact` is allowed only on motion skills with the contact profile (from M5, also transit with a measured window, D2).
- `max_steps = null` means the plant-derived budget.
- `then` values beyond `continue` / `decide` are only valid in `decision-uc72-m5`.
- `finish` ⇔ `termination.code = goal_satisfied`; `give_up` ⇔ any other code; every other type ⇔ `termination = null`.
- The legacy H1 schema stays on the `h1_equivalent` path, so the 13 saved T04 responses still validate there.

### I-15 Feedback and failure taxonomy (owner ws_executor_control)

`segment-result-1` (≤ 1.2 KB per segment, `executor/schemas/segment_result.schema.json`):
- `{"index", "skill", "status": "succeeded|failed|preempted|yielded", "reason", "residual": {"position_m", "orientation_rad", "along_command_m"}`
- `"sequencing": {"mode", "site", "rotation_rad", "rotate_steps", "translate_steps"}, "route": body_plan(compact), "reachability": {"classification", "limiting_joint", "min_margin_rad"}`
- `"contact": {"declared", "channel", "at_step", "depth_from_entry_m", "remaining_to_target_m", "within_window"}`
- `"blocked": {"direction_world", "commanded_offset_m", "depth_reached_m", "remaining_m", "guard": {...}, "gripper": {...}, "nearest_robot_part": {"part", "obstacle_id", "distance_m", "source"}, "joint_margins_rad", "back_off": {...}}`
- `"grasp": {"settled_gap_mm", "settle_k", "empty_curve_gap_mm", "object_present", "gap_band_mm"}, "resolution": ResolvedTarget(compact)}`

| Source | Code | Meaning | Handling |
|---|---|---|---|
| executor | `pose_arrived`, `arrived_in_band`, `contact_reached`, `gap_stable`, `wait_elapsed`, `pose_arrived_at_depth_bound` | success | continue; no call |
| executor | `blocked_motion`, `rotation_blocked`, `unexpected_contact_early` | undeclared block after hold + back-off | handoff with structured cause |
| executor | `no_progress`, `diverging` | monitor catch-all | handoff |
| executor | `grasp_object_absent`, `grasp_gap_out_of_band`, `held_object_lost`, `maintained_grip_gap_changed` | predicates | yield |
| executor | `reachability_limited` | moving into a joint limit (posture recovery first if enabled) | handoff |
| executor | `segment_budget_exhausted`, `preempted` | budget / preemption | handoff / record |
| planner | `route_unavailable` (sub: `start_blocked_no_escape`, `goal_blocked`, `no_route`, `no_route_budget`, `blocked_by_geometry`), `jaw_sweep_blocked`, `standoff_handoff`, `escape_partial` | geometry | local first (escape, detour, merge, `replan_once`); handoff only when stated |
| planner | `truncated_at_clearance`, `depth_bounded`, `merged_into_contact_stage`, `escape_then_*`, `detour` | executed with modification | reported, no call |
| resolver | `reference_unknown`, `reference_unresolved` (sub: `anchor_unmeasured`, `locate_stale`, `support_unknown`, `point_not_applicable`); warnings `part_geometry_unbound`, `raw_target_inside_object` | grounding | zero-motion handoff / warning |
| locate | `ok`, `snapped`, `frame_unavailable`, `views_inconsistent`, `ray_only` | pointing | reported next call |
| scene edit | `applied`, `rejected` (+ reason_code) | edits | reported |
| perception | `ok`, `degraded`, `unavailable` (+ `status_reason`) | service health | fail-open |
| protocol | `schema_error` (re-ask), `stale_rejected` (sub: `relative_target_after_motion`, `object_not_in_snapshot`, `gripper_state_changed`), `initial_plan_failed` | protocol | re-ask / fresh decide / scored failure |
| runner | termination kinds (I-18) | episode end | scored or unscored per I-18 |

- **Failure signature** (PC1): `skill|reason|sub_reason|gripper_command|held_object` with `repeat_count` and `consecutive_repeats` in every OutcomeRecord. The model decides whether to give up; there is no automatic threshold.
- **Verbatim reasons.** Reason, detail and sub-reason strings are never truncated or rephrased in the model view (IC15).

### I-16 OutcomeRecord (owner ws_protocol_context)

One record per decision, projected from the executor evidence (`progress()` + segment trace), as PC4:

```json
{"schema": "outcome-record-1", "decision": 8, "based_on_step": 347, "ended_at_step": 369, "executed_steps": 22,
 "status": "succeeded|yielded|failed|aborted|stale_answer", "reason": "fragment_complete", "detail": null,
 "segments": [{"i": 0, "skill": "move_to_pose", "status": "succeeded", "reason": "pose_arrived", "steps": 22,
               "residual_mm": 2.1, "residual_rad": 0.01, "route": {"status": "direct", "reason": null},
               "evidence": {"contact_evidence|gap_change_evidence|blocked": "verbatim, executor-sized"}}],
 "remainder": "none|kept|discarded", "pending_segments": [],
 "measured_end": {"pose_wxyz": [7], "finger_gap_mm": 77.5},
 "events": [{"step": 360, "kind": "guarded_stop", "detail": "verbatim"}],
 "failure_signature": null, "repeat_count": 0, "consecutive_repeats": 0}
```

- **Quaternion hemisphere.** In the model view only, every quaternion is replaced by `q` if `q·q_ref ≥ 0`, else `−q`, where `q_ref` is the episode's first measured orientation. On the saved runs, `w ≥ 0` left all 33 sign flips (|w| < 0.011 at the tool-down pose); the hemisphere rule left 0 in 5 runs.
- Floats in the model view use 4 decimals for metres and radians.

### I-17 CommittedSequence API (owner ws_executor_control)

```python
class CommittedSequence:                      # FragmentSequencePrimitive v5
    def enter(self, view, execution: dict, *, keep_gripper: bool) -> None
    def next_action(self, view) -> np.ndarray
    def observe(self, before, action, after) -> list[dict]           # robot-event-1 events
    def state(self) -> Literal["running", "checkpoint", "yielded", "complete"]   # + kind / reason in evidence()
    def next_is_barrier(self) -> bool         # contact profile or stop_on contact, gripper change, own checkpoint, end
    def remainder_summary(self) -> list[dict]
    def resume(self, view) -> None            # M5
    def replace_remainder(self, view, execution: dict) -> None       # M5 (= the design's preempt + replacement)
    def abort(self) -> None
    def evidence(self) -> dict                # progress + segment trace, verbatim reasons
```

The M2 core is `enter`, `next_action`, `observe`, `state`, `evidence` and `abort`. `resume`, `replace_remainder` and `next_is_barrier` land in M5 together with PC5 and PC6.

### I-18 Termination and scoring contract (owner ws_runner_measurement)

- **Wire.** `termination` in I-14.
- **Policy diagnostics.** On the answering step the policy sets `Decision.diagnostics["terminal"] = {"kind": "finish|give_up", "code", "evidence", "step", "decision_index", "virtual_time_s", "request_key"}`. The runner executes that decision's 1-step hold, never calls `act` again, then finalises.
- **Termination kinds:** `finish, give_up, cap_policy_time, cap_exec_calls, cap_wall, raw_success_stop (h1_equivalent only), operator_cap, user_stop, infra_error (auth|transient|simulator|service), harness_exception`.
- **Evaluator side** (worker ops, evaluator-only):
  1. `settle_hold(post_finish_hold_steps)` with the policy's last gripper command (UD-05).
  2. `truth()` at the settled state: raw LIBERO, strict sub-checks, plus `not_resting_on_obstacle_or_distractor` per placed atom.
  3. The release-and-settle counterfactual `would_succeed_if_released`, as a diagnostic only, never in a score.
  4. Ordered stages as data, with `check_success()` evaluated last.

```json
"scoring": {"scored": true, "censored": false, "success_at_finish_strict": false, "success_at_finish_raw": true,
  "collision_free_success": false, "ordered_complete": null,
  "strict_checks": {"<atom>": {"not_held": true, "at_rest": true, "supported_only_by_target": true,
                               "not_resting_on_obstacle_or_distractor": false}},
  "failing_checks": {"On white_mug_1 plate_1": ["not_resting_on_obstacle_or_distractor"]},
  "at_finish_instant": {"raw": true, "strict": false}, "would_succeed_if_released": null,
  "libero_first_success": {"step": 549, "policy_time_s": 271.3}}
```

- `scored` is true for finish, give_up and caps (caps: `censored = true`), and false for user stop, operator cap and infra.
- **Headline:** `success_at_finish_strict`. `collision_free_success` additionally requires zero forbidden contact episodes under the UD-06 taxonomy.
- `libero_first_success` is kept for comparability with published numbers, never as a terminator.
- **Coverage-map glyphs:** `S` strict at finish; `R` raw-only at finish; `F` finished but failed; `G` gave up; `C` capped; `I` infra unresolved; `E` harness exception; `U` user stop; `S*`/`R*` oracle-stopped (pristine arm).

### I-19 Timing and telemetry (owner ws_runner_measurement)

```python
class Timers:                                                    # uc72/timing.py
    def block(self, kind: str): ...       # context manager; kind ∈ {model, perception, geometry, planner}; robot frozen;
                                          # the wall duration is added to VirtualClock.idle_s
    def compute(self, kind: str): ...     # overlappable policy compute
    def evaluator(self, kind: str): ...   # excluded from policy time
    def async_done(self, kind: str, compute_s: float, frame_capture_ns: int, used_at_step: int) -> None
    def step_record(self) -> dict
@dataclass
class VirtualClock:                                              # shared with protocol (PC6) and perception (usable-at)
    control_period_s: float; steps: int = 0; idle_s: float = 0.0
    def now(self) -> float: ...           # steps·dt + idle_s
    def on_step(self) -> None: ...
    def add_idle(self, seconds: float) -> None: ...
    def idle_until(self, t_virtual: float) -> float: ...
```

- **`steps.jsonl`** (≈ 260 B per step): `{"s", "t_v", "a": [7], "x": [8], "q": [7], "fid", "ph": PhaseLabel, "raw": bool, "tm": {"act_s", "model_s", "block_s", "compute_s", "sim_s", "obs_s", "io_s"}}`.
- **`PhaseLabel`** (executor): `{"mode": "fragment|direct|hold|terminal", "fragment", "segment", "skill", "profile", "declared_contact", "stage_target_xyz", "planner_id"}`.
- **`calls.jsonl`**: one row per logical call. The protocol client emits it; the runner owns the schema; it merges R6 and PC2:

```json
{"k": 7, "kind": "decide|verify|plan|canary|ab", "step": 196, "applied_step": 196, "mode": "blocking|async",
 "trigger": "segment_stall|fragment_complete|veto|initial|checkpoint|yield|verify", "utc_start": "…",
 "attempts": [{"n": 1, "elapsed_s": 0.37, "outcome": "auth_expired", "http_status": 401, "error_code": "token_expired", "wait_s": 0.0},
              {"n": 2, "elapsed_s": 17.6, "outcome": "ok"}],
 "latency_s": 17.6, "headers_s": 2.1, "first_visible_s": 11.8, "last_visible_s": 17.2, "completion_s": 17.6,
 "input_tokens": 51044, "cached_tokens": 3072, "output_tokens": 392, "reasoning_tokens": 152, "visible_tokens": 240,
 "visible_tok_s": 44.4, "request_bytes": 3210000, "images": 2, "predicted_latency_s": 16.6,
 "service_speed_index": 1.06, "latency_residual_s": 1.0, "prompt_cache_key_sent": false,
 "status": "applied|stale_rejected|schema_error|error", "response_type": "execute"}
```

- **`contacts.jsonl`** (R5): rows per action with ≥ 1 non-ignored contact, `{"step", "phase", "c": [{"class", "subclass", "owners", "geoms", "samples", "max_penetration_m", "max_normal_force_n"}]}`.
- **Contact episodes:** maximal runs per (subclass, owner pair), merging gaps of ≤ 6 actions, with `onset_phase`, `onset_speed_m_s`, `onset_target_distance_m`, `intended`, `counted_as`.
- **Summary timing block** as R6: `wall_s`, `startup_s`, `policy_time_to_terminal_s`, the decomposition (model, blocking, motion, compute excess), async, evaluator, `infra_wait_s`, `per_step_overhead_s`, `first_raw_success`.

### I-20 HeightQuery (owner ws_perception_service)

A numpy lookup in the controller on the snapshot's height-field npz, with no socket round trip per query:

```python
def height_query(hf_ref: dict, origins_m: np.ndarray, direction: np.ndarray, *, max_range_m: float = 0.25,
                 fresh_after_step: int) -> dict
# -> {"hits": [{"t_m": float|None, "sigma_m": float, "object_id": str|None, "kind": "object|support|unknown"}],
#     "captured_at_step": int, "source": "wrist_affine_depth|external_affine_depth|triangulated|object_map"}
```

- `object_id` is mapped from perception instance ids to o-ids by the SceneMap.
- With no height field (perception unavailable, or a stale one), the depth bound falls back to `object_map_conservative` (the OBBs) and labels it so.

---

## 5. Conflicts between the designs, and how they are resolved

| # | Between | Issue | Resolution |
|---|---|---|---|
| X1 | runner R10 ↔ protocol PC2 | Both implement A3: 401 refresh, 5xx/520/deadline retry, `force_refresh` in `chatgpt_auth.py` | **PC2 owns the whole model-call path**: client, transport, retry policy (+ the `quota` class from R10, raising `QuotaPause` to the runner), cache key, telemetry emission. The runner owns the `calls.jsonl` schema and the sweep-level infra policy (relaunch, pause). R10's 1.5 d is removed. |
| X2 | runner R6 ↔ protocol PC2 | Two telemetry files (`calls.jsonl` vs `call_telemetry.jsonl`); two latency reference fits (h1_speed 128 calls vs cross_critic C1 164 calls) | **One `calls.jsonl`**, one row per logical call with an `attempts[]` array. **One reference fit**: cross_critic C1 (`6.89 + 0.0510·reasoning + 0.0071·visible + 0.0192·input_k`, n = 164, the fit behind the H3 residual analysis), recorded with its source. Both `service_speed_index` and `latency_residual_s` are reported. |
| X3 | runner `Timers`/policy time ↔ protocol `VirtualClock` | Two definitions of episode time | **One clock class in `uc72/timing.py`**, owned by the runner. Policy time = `VirtualClock` time at the terminal step + Σ compute excess. `Timers.block` adds idle. The protocol's `CallScheduler` and perception's usable-at rule consume it. |
| X4 | runner R4 ↔ protocol PC1 | Termination payloads differ (`reason_class` taxonomy vs `code`/`evidence`); PC1 drops the evaluator's release-and-settle, R4 keeps it as a diagnostic | **Wire** = PC1's `termination{code, evidence}`; diagnostics unified in I-18. **Headline** = strict truth after a 10-step hold settle with the policy's last gripper command (UD-05). `would_succeed_if_released` is kept as a diagnostic only, never in any score. |
| X5 | runner R7 ↔ perception §4.1 | Two FrameRing layouts: `/dev/shm` memmap, 8 slots, raw pixels vs POSIX shm, 16 slots, upright pixels, pinned/decision flags | **One `FrameRing v1`** (I-03), owned by the runner. Memmap file (avoids the resource-tracker unlink). 16 slots (25 MB). Union of slot fields. **Raw robosuite pixels** with `raster_code` and an `rgb_upright()` helper, bit-exact with H1's decode, so the replay stays exact. Perception copies a frame on `observe` receipt, so pinned slots are unnecessary. |
| X6 | runner `ServiceProtocol` ↔ perception socket ops | Envelope key `payload` vs `args`; `identity` vs `hello` | `{"id","op","args"}` → `{"id","ok","result","error","timing"}`. Mandatory ops: `hello`, `health`, `reset_episode`, `shutdown`. The runner owns the envelope and lifecycle; the perception service owns its ops (I-04). |
| X7 | perception PS-2/PS-5/PS-6 ↔ geometry GT1/GT2 | Duplicate camera models, single-view lifting, triangulation, ray-gap and support fusion in two processes | **GT1** (`geometry_tools.camera`, `.metric`, numpy-only, tested under both numpy versions) is the single camera and metric library for both processes. **GT2** `fuse_supports` is the single fusion function. PS-5 produces support evidence (masked sweeps, DA3 RANSAC prior, per-view affine). PS-6 keeps ray-gap association and the scipy multi-view fit. The **SceneMap's fused support is canonical**, because it also receives contact anchors and `locate` support points. |
| X8 | perception PS-7 ↔ geometry GT3/GT10 ↔ protocol PC3 | Three binding mechanisms, two inventory schemas (PC3 `objects` with ids, GT10 `objects` without), a separate `bind` field with numbered marks (PS-7 §4.7) | **Inventory schema** = PC3 (ids O1–O12, noun, attributes, x, w, parts, role; owner protocol). **Binding** is computed once in the service (PS-7: cells, then direction hints, then primary prompt) and reported in snapshot `roles`; the SceneMap adopts these as aliases. **Ambiguity**: the model binds with `scene_edits {op: "bind"}`, using the overlay's o-ids as the marks. No separate `bind` field or mark rendering. GT10 survives only as the derivation fallback for plans without `objects`. |
| X9 | all designs | Five id namespaces: `inst_NNN`, `oN`, `O#`/`U#`, `obj_*`, `model_box_i` | **One model-facing namespace owned by the SceneMap** (I-08): `o1–o24`, aliases `A–H` and `O1–O12`, `L1–L4`, `S1–S4`. `inst_NNN` is internal. `mb1–mb4` exist only in M1–M2. `U#` (unlisted) is unnecessary: unlisted objects are o-ids. |
| X10 | perception PS-10 ↔ geometry GT9 ↔ protocol SI-4 | Three model-facing "facts" blocks (compact facts ≤ 1.5k chars, `context.scene`, `PerceptionFacts` ≤ 2k chars) | **One block, `context.scene`**, produced by `SceneMap.model_facts()` (owner geometry). It is ≤ 2,000 chars and current-only; PC4 strips it at insertion (`scene` is in `EPHEMERAL_KEYS`). PS-10's compact facts become a diagnostic only. This also leaves one instruction confound to A/B (G10). |
| X11 | perception PS-9 ↔ geometry GT3 ↔ swept SP10 | Three held-object attach/detach implementations and three fallback envelopes | **The SceneMap owns the held record** (attach on `grasp_established`, membership rule, one fallback jaw-column envelope; detach on `released`). Perception supplies co-motion confirmation (`held_confirmed` / `grasp_not_confirmed`) and in-hand refinement. Swept builds sphere covers from the held record. |
| X12 | geometry GT4 palm clamp ↔ swept SP8 depth bound | Two implementations of C2's depth bound. SP8 shows an AABB-top bound cannot separate the 3 contact closes from 4 contact-free ones (0.1–11.1 mm below the AABB top) | **One implementation (SP8)**, using `HeightQuery` per palm/fingertip footprint cell from the PS-8 height field. The OBB fallback is labelled `object_map_conservative`. GT4 calls SP8 and reports `palm_clamp_m`. One user decision (UD-34): report-only in M3, active in M4 only with measured per-cell heights whose σ is below the pad height. |
| X13 | executor EXC-B2 ↔ swept SP6a | Two exact-distance point routers with escape (`TransitRoute2` vs `point_v2`) | **One implementation, EXC-B2 `TransitRoute2`** (`executor.router=point_v2`). It already carries the plant-derived `m_track` and the braking tracker. SP6a is dropped (−1.0 d); SP6 generalises the same escape rules to the body. |
| X14 | executor `sequencing.plan_stage` + `ClearanceOracle` ↔ swept `LocalBodyPlanner.plan_stage` | Two stage planners choose rotation sequencing, escape and routes | **One `StagePlanner` seam owned by the executor** (I-12), with two implementations: `PointStagePlanner` (TransitRoute2 + GripSiteBoxOracle + interim site choice; M2–M3) and `LocalBodyPlanner` (swept; M4). **Candidates and predicted profiles always come from the executor's `ExecutedProfileModel`.** Legs are always executed by `BrakingPathTracker`. Per-step precedence is fixed (guard → planner verdict → predicates → arrival). The oracle protocol has two implementations (GripSiteBoxOracle, SweptOracle). |
| X15 | executor `ExecutedProfile` ↔ swept `PredictedProfile` | Different sample layouts (xyz+quaternion+gap vs state8) and mode names (`rotate_then_translate` vs `rotate_first`, …) | **One `ExecutedProfile`** (I-11), samples in **state8 layout** (the gripper is placed directly from state8) plus phase, arc, tracking band, steps. The executor's mode names are used; swept's names map 1:1, and `legacy_concurrent` names H1's behaviour. |
| X16 | executor EXC-C5 ↔ swept SP1/SP2 ↔ perception PS-1/PS-3 | Three vendor trees with the same pinned files (robodojo FK, Panda XMLs, K2 `robot_geometry.py`); two FK wrappers; two robot-model builds (collision spheres vs visual envelopes) | **One `src/gpt_field/vendor/` tree with one MANIFEST** (I-01). **One `robot/` package** owned by swept: `PandaChain` FK/J/bind/lift and `RobotBodyModel` (collision spheres + hulls + visual envelopes + palm profile + footprints). **One offline builder** `tools/build_robot_model.py`; the robosuite visual meshes are vendored too, because K2's model reads `.obj`/`.stl` visual meshes. The executor's IK and reachability sit on `robot.kinematics`. Perception's raster reads the npz visual envelopes. Geometry's `inside_robot` uses `robot.body.silhouette`. |
| X17 | swept I8 per-stage `contact` declaration ↔ geometry `ContactIntent` | Two ways for the model to declare contact | **The `ContactIntent` is derived by the TargetResolver** from skill, profile, `stop_on`, target anchor, tool point and held state, plus `scene_edits allow_contact/forbid_contact` (scope: the next fragment). There is no separate wire field. Swept's legacy nearest-object inference is kept only for shadow mode and replays. |
| X18 | executor schema v5 ↔ protocol segment v3 ↔ geometry GT4 | `expect{stop_on, contact_window_m, object_in_gripper, gap_band_mm}` vs flat `stop_on` + `grasp: bool` + `then`; `max_steps` ≤ 240/null vs ≤ 120; 12 vs 16 segments; `route` and `observation_after` kept vs removed; `target` xor `xyz_m` | **One `segment-uc72-1`** (I-14): flat `stop_on`, `contact_window_m`, `grasp: null|{gap_band_mm}`, `then`, `target` xor `xyz_m`, `max_steps 1..240|null`. **12 segments**: placement checkpoints are blocking barriers, so a 16-segment two-object commit saves no call. `route` (used in 0 of 255 segments) and `observation_after` (→ `then`) are removed. The legacy schema stays for `h1_equivalent`. Semantics are owned by the executor, the wire by the protocol, the target grammar by geometry. |
| X19 | protocol SI-11 `scene_labels` ↔ geometry `scene_edits` | Two semantic-override fields | **`scene_edits`** (a superset, id-only, no metres) is kept; `scene_labels` is dropped. |
| X20 | perception UD5 (merge model boxes) ↔ geometry/swept (remove or ignore) ↔ protocol (keep through v3-b) | When and how model-written boxes leave | **Keep `obstacle_map` through M2** (honest baseline and executor changes stay comparable with H1). **At M3, remove it from the schema** and plan only on the SceneMap (perception + `locate` + contact). There is **no merge mode**: it would re-import R1's reproducible 5–12 cm bias. Missed objects come back through `locate` + `scene_edits add_from_locate`. The M2 build (b002) is the comparison arm. |
| X21 | geometry GT1 `from_geometry` ↔ perception PS-2 ↔ runner A5 | GT1 still depends on H1's ideal-pinhole validator; the others replace it | **`camera-calibration-1`** (I-05) is emitted by the worker and loaded by GT1's single `CameraModel`. The new path uses a consistency check instead of the validator (distortion and non-centred principal points accepted). The validator remains only on the legacy path (INT-4). |
| X22 | perception PS-12 `export_truth` ↔ swept SP12 `enrich_trace` ↔ geometry GT11 `build_cache`/`truth.py` ↔ perception `build_dataset` | Two simulator replay-truth exporters and two dataset builders | **One replay-truth exporter** (INT-5, `validation/truth/export_replay_truth.py`, `.venv-sim`, zero model calls, owner runner). Per step it writes joints, object poses and contact pairs; at benchmark steps, oracle masks, metric depth and geoms. **One dataset builder** (PS-12), consumed by the perception scorer, GT11 and SP12. An import-guard test keeps truth unimportable by policy modules. |
| X23 | executor EXC-D2 ↔ swept SP9 approach schedule | Two distance-scheduled approach implementations | **SP9 `SpeedPolicy` owns cruise and approach scheduling.** The executor tracker applies a per-arc command cap (I-12) and supplies the plant's coast and guard latency (one source: `osc_response.json`). EXC-D2 is reduced to the tracker-cap hook. |
| X24 | swept environment variables, protocol CLI mode flags, perception CLI mode flags ↔ runner frozen JSON config | Behaviour switched outside the frozen, hashed config | **All behaviour switches are config flags** (§2.4); the CLI carries only pointers and operator caps. The loader enforces the IC invariants (§2.5). |
| X25 | runner `--max-exec-calls` ↔ perception `--max-model-calls` ↔ geometry `--max-executive-calls` ↔ protocol `--stop-after-calls` ↔ swept `--max-model-calls --max-actions` ↔ executor `--max-calls --max-wall-s` | Six names for the operator caps | **`--max-exec-calls N`, `--max-steps N`, `--max-wall-s S`** (operator caps, unscored), plus `--prepare-only` for plan-only smokes. |
| X26 | H1 / runner `--gpu <egl_minor>` ↔ perception `cuda:N` | Two GPU address spaces; minors are reversed relative to nvidia-smi on this host | **`--gpu <nvidia-smi index or UUID>` everywhere.** `uc72/gpu.py` resolves the EGL minor from `/proc/driver/nvidia/gpus/*/information`; perception uses `cuda:<index>` with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Both are recorded. |
| X27 | executor (caps in D1) ↔ protocol PC5 (caps at the commit step) ↔ IC5 | When the 6 / 120 / 240 caps change | **Step budgets change in M2 with B1**: `MAX_SEGMENT_STEPS` 240 or a null plant budget, and `MAX_EXECUTION_STEPS` removed. A π rotate phase needs 137 steps at 0.0229 rad/step, which the 120-step cap would cut. **`MAX_SEGMENTS` 6 → 12 and removal of the grasp/placement visual mandates (`agent_client.py:161-164`) land in M5** with PC5, after the guard and predicates are measured (IC5). |
| X28 | protocol `v3/` package and v3-a…e names ↔ runner `uc72-bNNN` | Two versioning schemes | Package `src/gpt_field/protocol/`; builds named only `uc72-bNNN` (v3-a ≈ b001, v3-b ≈ b002, v3-c ≈ b003, v3-d+e ≈ b005). |
| X29 | report H2 §4.4 item 2 ("robot pixels as metric anchors") ↔ PS-5 (reject robot pixels as DA3 anchors) | Whether robot self-pixels anchor DA3 depth | **Follow PS-5.** Robot pixels get FK-rendered depth replacement and serve as a calibration check, not as DA3 anchors (DA3 is +8 to −30 cm off on the robot, H2 F4). The per-view affine is anchored on support inliers (ray ∩ fused plane). |
| X30 | geometry D8 (skip the plane sweep: failed 4/7 unmasked) ↔ PS-5 (plane sweep primary at step 0) | Whether the DA3-free plane sweep is a support source | **Include PS-5's sweep only with robot-raster + instance masks and a peak-quality gate** (no estimate is better than a wrong one). It enters GT2 fusion only if the benchmark shows σ calibration (|err| ≤ 2σ in ≥ 90 % of frames). Otherwise fusion uses the DA3 prior + contact anchors + `locate` (UD-31). |
| X31 | executor UD1 ↔ swept UD3 | Two decisions about motions the model did not request | One decision, **UD-16**: escape legs, back-off after an undeclared block, one local replan; each short, guarded, distance-increasing and reported. |
| X32 | swept SP9 cruise ↔ executor B3 braking | Two laws limiting the per-step command | The command = min(braking-law offset, SpeedPolicy cap). IC2 order: braking (M2) → clearance-scaled cruise with the body sweep (M4) → approach schedule (M5). |
| X33 | runner R7 phase 2 (render only on demand) ↔ IC10 / perception tracking | Rendering only at decision steps would starve perception | Publish interval = min over subscribers' declared intervals (I-03). Render-on-demand is used only if the M0 overhead benchmark shows rendering still dominates. `FrameProvider.capture_now()` returns the latest ring frame or a render op. |
| X34 | geometry GT9 (keep `why`) ↔ protocol PC4 (drop `why`) | Prose fields | The protocol owns the schema: drop `why`, keep `belief` (≤ 1,800 chars; the cross-call memory) and `plan` (≤ 600), subject to the PC7 reasoning-token check (UD-21). |
| X35 | perception's numbered marks + `bind_roles` op ↔ geometry overlays (GT7) | Two image annotations for identity | **One overlay renderer (GT7)** draws o-ids, boxes and the last path on the current pair (replace, never add; IC9). Perception supplies polygons and bboxes. Numbered marks are not rendered. |
| X36 | executor `osc_response` guard latency (median 3 steps, M8) ↔ swept SP9 (4 steps, verify F5) | Two guard-latency values for the slow zone | **One source**: `osc_response.json` `guard.detection_delay_steps_median`, from EXC-0's fit, re-measured by the scripted benchmark. SP9 reads it through `ExecutedProfileModel.guard_latency_steps()`. |

---

## 6. Milestones

The order follows the cross_critic safe order and the user's sequence:
1. honest measurement;
2. overhead and cheap fixes;
3. veto replacements and sensing;
4. perception service and measured geometry;
5. the swept planner;
6. committed sequences, async and speed.

**Each milestone M1–M5 ends with a sweep of one frozen build** (`uc72-b00N`) against the previous milestone's frozen build, in the same service window (interleaved blocks). The sweep protocol, scene sets, power and cost are in §7. Offline checks are development aids (rule 4); capped smokes are runs Claude may start; sweeps and full tasks are started by the user.

**Common launcher** (from the new repo root; `<idx>` is an nvidia-smi index):
```bash
arms/<tag>/run.sh --task <family>/<id> --config config/arms/<arm>.json --gpu <idx> \
    [--service perception=unix:/tmp/uc72_perception_<tag>.sock] --max-exec-calls <N> [--max-wall-s <S>] \
    --output runs/<name>
```

### M0 — Foundation and equivalence (offline; no model calls)

**Goal.** A new build whose flags-off configuration is provably H1, with the evaluator overhead removed and tasks as data, so every later change can be measured against an exact anchor.

**Includes.**

| Component | What | Fixes |
|---|---|---|
| R1 | clone, tags, worktrees, venv symlinks, `.local/` isolation; config loader with `{value, source}`; flag registry + invariants (§2.4–2.5); build digest; `uc72_build.py freeze/worktrees` | §6.5, F7; the hidden 0.02 m default (`action.py:20`) |
| R2 (core) | new episode runner preserving H1's loop order (`act → step → observe_transition`, the hold step after a rejected answer); one simulator for plan + execution; `ServiceSupervisor` and `ServiceProtocol` (I-04) skeleton; `h1_equivalent` mode stops on raw success | F8 (second simulator start, −4.9…−8.9 s), A1 scaffolding |
| R3 | task spec (I-02), task pack (20 official + 3 obstacle variants + composite + 10 relayouts; generator for seeded on-path obstacles, UD-10), reset identity (PNG + pixel hashes), no-model reset preflight | A5, H1 #18–19 |
| R7 | FrameRing v1 (I-03), lazy `FrameObservation`, geometry in the step reply (one IPC per step), compact trace, `run_state` at decisions or ≤ 1 Hz, no per-step `_sync_framework` / `history_stats` | A2, H1 #1, IC10, IC14 |
| R8 | video out of the loop: post-hoc re-simulated dashboard (default) or a live raw recorder | A2 |
| R9 | pixel fixture from the pristine worktree; runner-level zero-call replay on decoded pixels; overhead benchmark | A2 caution, R8, §6.7 |
| INT-7 | `uc72/gpu.py`: nvidia-smi index/UUID → EGL minor; free-memory probe | X26 |

**Depends on.** Nothing.

**Offline checks.**
1. **Equivalence:** `validation/replay_runner.py --config config/arms/h1_equivalent.json` reproduces the T04 reference exactly. That covers 549/549 actions and state8, raw-success flags, contact class counters, decoded-pixel hashes of every observation and of all 13 request images, and request text except the measured timers.
2. **Overhead** (target, not a gate): median per-step wall ≤ 0.06 s, p95 ≤ 0.10 s, last-quintile median ≤ 1.10 × first-quintile. The H1 reference is 0.158 s/step without video and 0.282 s with it.
3. **FrameRing:** a py3.10 writer and a py3.12 reader over 10⁴ frames give no torn reads; `StaleFrame` on overwrite; `rgb_upright` equals H1's decoded arrays for the saved T04 reset.
4. **Config:** unsourced numbers rejected; undeclared flags rejected; every INV rule has a failing and a passing example; `action.py` raises without a config; controller and worker report the same `TRANSLATION_SCALE_M`.
5. **Digest:** stable across two clean checkouts; changes on any tracked file, config byte, package set, task-pack or vendor change.
6. **Task specs:** schema + `sentence == BDDL :language`; leakage test (no `scene`/`evaluator` key reaches the policy or any request); the importer reproduces `H1/tasks/t04_can_obstacle` hashes.
7. **Video:** post-hoc re-simulation of T04 is state8-exact.
8. **Reset preflight** (simulator, no model; UD-02): a full table for every scene, each mismatch resolved (UD-03) or listed.

**Live checks.** None (no model calls).

**Acceptance.** Checks 1 and 8 complete. The tag `uc72-b000` records the replay result, the overhead numbers and the preflight table.

**Expected effect (est.).**
- Per-step non-model cost falls from 0.28–0.40 s to ≈ 0.04–0.07 s, and stops growing with episode length.
- T04 wall falls 402 → ≈ 275 s (−117…−131 s, report A2).
- −4.9…−8.9 s per episode from a single simulator (verified).
- Drawer-type tasks −700…−820 s.

**Risks.**
- Hidden coupling in the old runner → caught by the replay.
- PNG-encoder byte differences → pixel hashes.
- Official reset hashes were measured in the main-tree environment → preflight + UD-03.
- `/dev/shm` leaks after crashes → PID sweeper.

**User decisions before M0.** UD-01 (location), UD-02 (no-model simulator jobs), UD-03 (hash adoption).

**Effort.** R1 2.5, R2 3.0, R3 3.5, R7 4.0, R8 2.0, R9 2.0, INT-7 0.5, INT-6 (freeze, preflight run, replay record) 1.0 = **18.5 d**.

### M1 — Honest measurement baseline (sweep S1)

**Goal.** A finish-scored, all-task, zero-shot baseline of H1 behaviour on one frozen build. Infrastructure losses are removed and every later comparison's instruments are in place (IC6: finish-scoring first).

**Includes.**

| Component | What | Fixes |
|---|---|---|
| R2 (rest) | terminal dispatch; caps (UD-04) checked at step boundaries; exception classes | A1, R6 |
| R4 | hold settle with the last gripper command; strict + raw at the settled finish; `not_resting_on_obstacle_or_distractor`; ordered goals as data; counterfactual as a diagnostic only | A1, §6.1, H1 #17 |
| R5 | per-action contact records with subclasses (palm on current target vs other goal objects; static vs articulated; obstacle vs distractor; held-object branches); episodes by subclass and phase; obstacle displacement; historical rescoring | §6.2, cross_eval M2 |
| R6 | policy-time definition (I-19), `steps.jsonl` (with joints), `calls.jsonl`, service-speed index, latency canary per block | §6.3, G2 |
| R11 | sweep harness: plan (blocked, randomised, counterbalanced), run (digest check, GPU check, resident services), stop/censor/relaunch rules, coverage map, report, cost model | rule 3, §6.4–6.6 |
| PC1 | `give_up` and terminal `finish` with `termination{code, evidence}`; deletes `agent_client.py:205-209`, `agent.py:128-130`, `:151`, `:231` and `framework.py:112`; completion contract (release and withdraw before finish); failure signature + repeat counts; banned-phrase test | A1, R6, H1 #17 |
| PC2 | thread-safe transport; 401 forced refresh once; 5xx/520–524/deadline ≤ 2 retries; 429 `QuotaPause`; per-call telemetry; cache key **off** | A3, H1 #16, G2 |
| PC3 (a) | relaxed plan gate (`ignored_selection` rules skipped), one re-ask with `previous_plan`, sanitise instead of abort, `initial_plan_failed` as a scored status | A4 (relaxation), H1 #14 |

**Depends on.** M0.

**Offline checks.**
- **R4:** the rest-contact check fails on the astranomy CA record (white mug resting on `tomato_sauce_obstacle_1`) and passes on the H1/H3 after-settle records; the hold settle keeps the gripper command; the ordered-stage evaluator reproduces `ordered_success.py` events; there is a scoring unit test per termination kind.
- **R5:** rescoring reproduces the verified episode counts exactly. H1 T04: 3 owner-pair episodes, 0 obstacle. T20 T04: 1 obstacle (mug–can, actions 141–144). H1 T08: 5 episodes, 3 obstacle, including gripper–bottle actions 37–83 (47 actions, 3,504 samples). H3 T08: 2. Drawer 001: 10; drawer 003: 5. The class counters are unchanged in the replay.
- **R6:** the call-record builder on the 13 saved T04 requests reproduces 13 calls, Σ 198.5 s, median 13.4 s, input 7,945 → 87,425, cached 2.5 %. The service-speed index median is 0.95–1.05 on the fit set.
- **PC1:** T1.1–T1.5 (0 banned phrases; no `act` after finish).
- **PC2:** T2.1–T2.7 (fault injection; resent payload bytes identical; deadline from a worker thread without SIGALRM).
- **PC3:** T3.1–T3.6 (20/20 fixture plans approvable; the t9 draft passes; never `parser.error`).
- **R11:** plan determinism, relaunch logic, adapters on the 7 historical runs.
- **Replay:** the M0 replay is still bit-exact under `h1_equivalent`.

**Live checks.**
- Capped smokes Claude may run (≈ 25 calls in total):
  ```bash
  arms/uc72-b001/run.sh --task obstacle_variant/t04_can_obstacle --config config/arms/measured_h1.json --gpu <idx> \
      --max-exec-calls 3 --output runs/m1_smoke_t04_001
  arms/uc72-b001/run.sh --task official/libero_goal_g8 --config config/arms/measured_h1.json --gpu <idx> \
      --max-exec-calls 20 --output runs/m1_smoke_g8_001
  arms/uc72-b001/run.sh --task official/libero_10_t9 --config config/arms/measured_h1.json --gpu <idx> \
      --prepare-only --output runs/m1_smoke_t9_plan_001
  ```
  Pass signals:
  - t04: one simulator start, plan + 3 calls, `termination.kind=operator_cap`, complete `summary.json`/`steps.jsonl`/`calls.jsonl`/`contacts.jsonl`, `summary.build` equal to the tag digest.
  - g8: reaches `finish`; 0 calls after it; hold settle executed; strict and raw computed at the settled state; `libero_first_success` logged.
  - t9: the plan is approved, possibly after one re-ask.
- **Sweep S1 (user):** spec `sweeps/specs/s1_measured_baseline.json`. Arms: `h1-baseline-3cc84ac:pristine` (H1's own runner, `--no-video`, external caps through the stop file; oracle metric) and `uc72-b001:measured_h1` (honest metric). Scenes `full34` (or `full44`, UD-10); 1 repeat; interleaved blocks; canary per block.
  ```bash
  cd /workspace/gpt_field/agent_speedup_v2_uc72_harness
  .venv/bin/python scripts/uc72_sweep.py plan   --spec sweeps/specs/s1_measured_baseline.json --output sweeps/s1_001
  .venv/bin/python scripts/uc72_sweep.py run    --plan sweeps/s1_001/sweep_plan.json
  .venv/bin/python scripts/uc72_sweep.py status --plan sweeps/s1_001/sweep_plan.json
  .venv/bin/python scripts/uc72_sweep.py report --plan sweeps/s1_001/sweep_plan.json
  ```

**Acceptance metrics (S1; this sweep defines the baseline, so these are completeness criteria and reported values).**
- **Zero-shot:**
  - every cell has a terminal glyph and every infra failure is explained;
  - one digest per arm, with no per-task input (structural);
  - per task: strict success at finish (honest), raw at finish, and LIBERO first success (diagnostic);
  - reset identity status per scene;
  - the give_up / cap / finish mix.
- **Fast:**
  - policy time to terminal (median, P(≤ 180 / 300 / 600 s));
  - per-step overhead median ≤ 0.06 s;
  - per-arm median `visible_tok_s` and service-speed index, with a confound flag if they differ by > 10 %.
- **Collision:**
  - forbidden contact episodes by subclass and phase per task (counted under UD-06);
  - obstacle displacement;
  - onset-distance and onset-speed distributions (needed by IC1 and IC2 later).
- **Few attempts:** calls per success; post-finish calls = 0; infra-terminated episodes = 0 after retries; `initial_plan_failed` = 0.

**Expected effect (est.).**
- The honest success rate is unknown (the bound for the 7 retained uc7.1 runs is 0/7–6/7); S1 sets it.
- A1 costs +1–2 calls (+20–40 s) per success.
- T04 wall ≈ 275 s plus the finish-scoring cost (+20–40 s).
- The T04-20 mm post-finish loop (9 calls, 482 actions) cannot recur.
- The 2–3 infra losses seen in about 14 launches are removed.

**Risks.**
- The model does not finish, so successes become caps. Visible in the termination mix; the completion-contract text is A/B-able with PC7.
- Sweep cost (§7.5).
- A service-window drift between arms. Mitigated by interleaving and the canary.

**User decisions before the S1 sweep.** UD-04 … UD-15.

**Effort.** R2 1.0, R4 2.0, R5 3.0, R6 2.0, R11 4.0, PC1 1.5, PC2 2.5, PC3 (a) 1.5, INT-6 2.0 = **19.5 d**.

### M2 — Veto replacements, sensing and compact context (sweep S2)

**Goal.** Replace the zero-motion vetoes and false stalls with measured executor mechanisms, add contact sensing and grasp predicates, and make the context append-only and compact. **Model-written boxes are still used**, so the geometry effect is not mixed in.

**Includes.**

| Component | What | Fixes |
|---|---|---|
| EXC-0 | fitted OSC plant (`osc_response.json`; log fit a = 0.711, b = 0.0732, ρ = 0.254, R² = 0.973; rotation ρ_r = 0.27), refit by the scripted benchmark | prerequisite of rule 2 for B3/C1/D1/D2 |
| EXC-B3 | `BrakingPathTracker` (braking law linear in the remaining distance, gain 1 − a; no deadband); `ProgressMonitor` replaces the 12/16/0.8 mm stall test; in-band arrival; `stop_on: contact` with a window | B3; H1 #3, #4 |
| EXC-B2 | exact point/segment–box distances (8-corner lattice rule removed); explicit margin `m_track` (plant-derived) + model clearance; checked escape whose exit strictly increases distance from the unexpanded box, no vertical default; `route_goal_blocked` stays, now exact | B2; H1 #12–13; IC4 |
| EXC-B1 | rotation sequencing (`rotate_then_translate` / `translate_then_rotate` / `translate_rotate_translate`; `coupled` only once certified in M4); the 1.2 / 0.5 rad envelope veto is removed; `ExecutedProfile` (I-11) exposed | B1; IC3; H1 #13 |
| EXC-C5a | `executor/reachability.py` on `robot.kinematics`: local/global IK at stage entry, joint margins in `measured_state` every call, classification without veto | C5; H1 #15 |
| EXC-C1 | `GuardPolicy`: classify locally (in-band, declared contact, reachability, route validity); undeclared → hold 1 step, back off to the guard-window start, `blocked_motion` with structured cause (replaces "cause unknown", `fragments.py:482`) | C1; R5; H1 #11 |
| EXC-D1 (a) | grasp predicates (`object_present_after_close` from the measured empty-close curve; `held_gap_ok` while the close persists; `grasp` intent from the segment); plant-derived budgets (`MAX_SEGMENT_STEPS` 240 or null; `MAX_EXECUTION_STEPS` removed, UD-18) | D1 predicates; IC5 step 1 |
| EXC-INT | fragments v5 wiring, executor telemetry (PhaseLabel), `validation/replay_executor_logs.py`, scripted no-model simulator benchmark (plant refit, guard on table, rotation modes, overshoot) | R8 |
| INT-1 | `src/gpt_field/vendor/` + MANIFEST; `robot/kinematics.py` (`PandaChain`: FK, J, bind, batched FK, lift; health check); vendored robosuite XMLs and meshes; `check_vendor.py`; import guard | prerequisites (b), G6; X16 |
| INT-2 | `StagePipeline` seam with `PointStagePlanner` (I-12); `decision-uc72-m2` / `segment-uc72-1` assembly | X14, X18 |
| INT-3 | `robot-event-1` bus → `events.jsonl` (perception and SceneMap subscribe in M3) | X11, X17 |
| PC4 | context v3: static cacheable prefix rendered from capabilities and robot/camera facts; append-only transcript; current-pair images (UD-20); one OutcomeRecord per decision with verbatim reasons; macro layer removed; `why` dropped (UD-21); quaternion hemisphere | D3; H1 #6; IC9; IC15; C9 |
| PC7 | model-only prompt A/B harness; cache-key probe; **H3 channel ablation** (below) | IC9, G2, G10, report §5.2 item 1 |
| GT8 | tool-envelope prompt text from `robot_facts_v2.json` (palm 31.04 mm behind the grip, fingertip reach 9.25 mm, footprints, open finger reach ±66 mm) | C2 (prompt half); H1 #10 |

**H3 channel ablation** (report §5.2 item 1; cross_critic C11, G10). Diagnostic only; never part of a build. With PC7, replay the **first executive request** of H1 T04 and T08 under three arms × 10 samples (≤ 60 calls, no simulator):
1. H1 text;
2. H1 + H3's instruction block (`foundation_vision/live.py:34-55`);
3. H1 + the saved SAM 2D obstacle box drawn on the current external image.

Score the first `obstacle_map` boxes against layout truth offline (coverage, IoU). This tells whether H3's better first boxes came from the instruction text or the 2D box. It informs UD-26 (what the model sees) before M3.

**Depends on.** M1 (finish-scored baseline, telemetry, contact onset statistics).

**Offline checks.**
- **EXC-0:** a refit on the 9-run fixture gives a ∈ [0.55, 0.80], ρ ∈ [0.23, 0.27], R² > 0.95.
- **EXC-B3:**
  - the monitor fires inside ≥ 13 of 29 forbidden episodes with median delay ≤ 4 steps (logs: 15/29, 3 steps);
  - it fires earlier than the legacy stall in ≥ 90 % of the 19 contact stalls;
  - 0 events during `set_gripper` / `wait`;
  - no `no_progress` on the T04 steps 173–195 return; `diverging` fires on d003 step 27;
  - braking plant simulation: p99 overshoot ≤ tol/2 and steps ≤ legacy (21 vs 25 steps on 0.23 m).
- **EXC-B2:**
  - of the 7 recorded start refusals, the 5 at +6.3…+20.3 mm route without escape, and the 2 inside boxes get monotone escapes;
  - the 111 straight-valid routed stages stay straight;
  - point and segment distances match brute force (≤ 1e-9 m / 1e-6 m).
- **EXC-B1:**
  - the T04 and T08 first decisions (1.572 / 1.569 rad) are accepted as `rotate_then_translate` with predicted rotate steps within ±15 % of 68;
  - T08 truth fixture: the fingertip clearance of the rotate-first profile is ≥ 0, while the measured profile intersects at steps 37–45.
- **EXC-C5a:** d001 calls 16/17/21/22/23 are `posture_limited`; d003 call 4 is `unreachable`; 0 of 31 T04/T08 targets are flagged; p95 ≤ 25 ms.
- **EXC-C1:** the counterfactual over 9 runs gives ≤ 3 additional handoffs (a naive handoff gives +24).
- **EXC-D1:** both recorded grasp failures are flagged; 0 of 19 confirmed grasps; deliberate closed-finger pushes are not flagged.
- **INT-1:** FK parity ≤ 1e-12 m against the vendored `PandaKinematics`; FK bound on every logged request reproduces the logged grip pose to ≤ 1e-6; base spread ≤ 1 mm per episode; hashes equal the pins.
- **PC4:** T4.1–T4.7: mean per-message chars ≤ 25 % of H1 (prototype 18 %); 0 reason strings lost over 128 requests; 0 quaternion flips; append-only prefix byte equality; no benchmark nouns in the prefix.
- **Replay:** the `h1_equivalent` replay is still bit-exact.

**Live checks.**
- Model-only (Claude may run):
  ```bash
  .venv/bin/python validation/prompt_ab.py --source-run /workspace/gpt_field/runs/agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001 \
     --source-run /workspace/gpt_field/runs/agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001 \
     --arms A,B --pairs 20 --max-calls 40 --order interleaved --output validation/results/prompt_ab_m2_<UTC>.json
  .venv/bin/python validation/prompt_ab.py --source-run <T04 run> --arms B,C --sequence-calls 1-7 --cache-key episode \
     --max-calls 14 --output validation/results/cache_probe_<UTC>.json
  .venv/bin/python validation/prompt_ab.py --source-run <T04 run> --source-run <T08 run> --first-request-only \
     --arms H1,H1+H3TEXT,H1+SAMBOX --samples 10 --max-calls 60 --output validation/results/h3_channel_ablation_<UTC>.json
  ```
  Pass signals (dev aids): median paired reasoning-token difference ≤ 0; validity 100 %; input tokens −60 % or more at later calls; whether the cache key is echoed and cached tokens rise; the ablation's first-box coverage per arm is reported.
- Capped smokes Claude may run:
  ```bash
  arms/uc72-b002/run.sh --task obstacle_variant/t04_can_obstacle    --config config/arms/m2_executor.json --gpu <idx> --max-exec-calls 20 --output runs/m2_smoke_t04_001
  arms/uc72-b002/run.sh --task obstacle_variant/t08_bottle_obstacle --config config/arms/m2_executor.json --gpu <idx> --max-exec-calls 20 --output runs/m2_smoke_t08_001
  arms/uc72-b002/run.sh --task composite/drawer_bowl_moka          --config config/arms/m2_executor.json --gpu <idx> --max-exec-calls 30 --output runs/m2_smoke_drawer_001
  ```
  Pass signals:
  - 0 `segment_stall`; 0 `target_envelope_rejected`; `route_start_blocked` = 0 except reported `start_blocked_no_escape` with numbers;
  - overshoot p99 ≤ 2 mm;
  - T08: the first decision executes, rotating at z ≈ 1.07–1.18 m and then translating; 0 finger–bottle contact episodes;
  - drawer: reachability classes logged; ≤ 2 calls per joint-limit event.
- **Sweep S2 (user):** `uc72-b002:m2_executor` vs `uc72-b001:measured_h1`, interleaved, 1 repeat, scene set per UD-11 (recommended `regression12`; `full34` optional).
  ```bash
  .venv/bin/python scripts/uc72_sweep.py plan --spec sweeps/specs/s2_executor.json --output sweeps/s2_001
  .venv/bin/python scripts/uc72_sweep.py run  --plan sweeps/s2_001/sweep_plan.json
  .venv/bin/python scripts/uc72_sweep.py report --plan sweeps/s2_001/sweep_plan.json
  ```

**Acceptance metrics (S2, against b001 in the same window).**
- **Zero-shot:** coverage complete; one build digest; strict success at finish per task (differences under ≈ 30 pp are labelled unresolved at 1 repeat, §7.4).
- **Fast:**
  - policy time to finish, median and paired ratio;
  - stall-triggered calls by class (the overshoot class → ≈ 0);
  - overshoot p99 ≤ 2 mm;
  - input tokens per call (−60 % at later calls);
  - reasoning tokens per call not above b001.
- **Collision:**
  - forbidden contact episodes by subclass and phase not above b001;
  - median pressing duration onset → stop ≤ 4 steps (b001 ≈ 13);
  - 0 episodes during escape legs;
  - contact episodes in rotate phases reported.
- **Few attempts:**
  - `target_envelope_rejected` = 0; `route_start_blocked` = 0 (except `start_blocked_no_escape`);
  - false `blocked_motion` (no evaluator contact) ≤ 1 per task;
  - calls after `blocked_motion` vs after H1 stalls;
  - pending discards after a declared contact = 0;
  - joint-limit call streaks ≤ 2 calls.

**Expected effect (est.; report and verify estimates, n = 1 evidence).**
- −18…−19 s per T04/T08 run from the removed rotation-veto repair calls (C6).
- ≈ −6 repair calls (≈ −70 s) across the 9-run set from exact distances and escape.
- −5 overshoot-stall calls (≈ −92 s) across 5 runs, and −2…−3 calls (−46…−90 s) from `stop_on: contact` (B3).
- Up to −70 % pressing time (C1, upper bound).
- Context: T04 −8 s, drawer 001 −80 s; dropping `why` −2.6…−7 s per T04.
- Cost: rotate-first adds +10–25 steps per large-rotation stage (+0.5–1.25 s simulated).
- T04 honest policy time ≈ 240–290 s, ± 60–90 s by service window.

**Risks.**
- The braking law is slow if a is overestimated → pooled a; robustness test across a ∈ [0.53, 0.77].
- The interim margin is lower than H1's hidden 0–20 mm → UD-17; the guard; the margin is reported.
- The context change moves reasoning tokens → PC7 gate before freeze.
- False blocks near joint limits → reachability classification; false-handoff metric.
- `rotate_then_translate` costs time until `coupled` is certified in M4.

**User decisions before M2.** UD-16, UD-17, UD-18, UD-19, UD-20, UD-21 (the S2 scope is set by UD-11).

**Effort.** EXC-0 2.0, EXC-B3 4.5, EXC-B2 3.5, EXC-B1 3.0, EXC-C5a 1.5, EXC-C1 2.5, EXC-D1(a) 1.0, EXC-INT 3.0, INT-1 3.0, INT-2 2.5, INT-3 1.5, PC4 4.0, PC7 2.0 (incl. ablation), GT8 0.5, INT-6 2.0 = **36.5 d**.

### M3 — Perception service and measured geometry (sweep S3)

**Goal.** The model stops writing metric geometry.
- Perception runs as a resident, asynchronous, fail-open service.
- The harness object map gives every object an id, a box and σ.
- Targets resolve from references at the moment of use.
- Obstacles reach the (point) router automatically, without operator naming.
- The body planner runs in shadow on live maps.

**Includes.**

| Component | What | Fixes |
|---|---|---|
| PS-1 | service process: vendored foundation_vision backends, SAM 3.1 + DA3 resident, priority queue, job isolation, OOM handling, `hello`/`health`, decision-frame recording | V6, V8; prerequisite (b) |
| PS-2 | FrameRing reader; calibration via GT1 `CameraModel` | IC10; G3 |
| PS-3 | FK-rendered robot raster from the npz visual envelopes + depth replacement; wrist self-mask atlas | V5; H2 F4; K2 F3; R3 |
| PS-4 | category prompts from inventory nouns (+ head noun), mask de-duplication (IoU > 0.5 merge), every instance kept | V1; K2 F8 |
| PS-5 | support evidence: external↔wrist plane sweep with robot + instance masks and a peak-quality gate (UD-31); DA3 RANSAC prior; per-view affine depth on support inliers | V5; R3; C10 |
| PS-6 | DA3-free ray-gap association; single view and pairs through GT1; conservative AABBs | V4; R1 |
| PS-7 | instance inventory with stable track ids; role binding from cells → direction hints → prompt hits; parts inside instances | V1; E-F1/L-F7; K2 F5/F8 |
| PS-8 | class-agnostic discovery (corrected DA3 height above support + SAM box exemplar); height field (5 mm) | V2; K2 F2; C2 input |
| PS-9 | event-indexed invalidation, change detection, co-motion held confirmation, relocalise at use | V7; IC14; K2 F1 |
| PS-10 | snapshot contract (I-06), schemas, hashes | V3 |
| PS-11 | `SocketPerceptionClient`, `RecordedPerceptionClient`, `NullPerceptionClient`, `PerceptionBridge` (three call sites in `agent.py`) | V6; R8 |
| PS-12 | perception benchmark: dataset from saved runs (164 H1-family decision frames, 22 H3, 20 + 350 K2, 184 astranomy) + scorer | V9; R8; §5.2 item 8 |
| INT-5 | one replay-truth exporter (`.venv-sim`, oracle ops, zero model calls; UD-02): per-step joints, object poses, contact pairs; oracle masks/depth/geoms at benchmark steps | X22 |
| INT-4 | `camera-calibration-1` emission in the worker; validator replacement on the new path | A5; G3; R9 |
| GT1 | `CameraModel`, triangulation, ray–plane, ray–box, `lift_silhouette` (far-edge top rule), `fuse`; σ models fitted on GT11 | R1; C3 |
| GT2 | support fusion (χ² clustering; perception evidence + contact anchors + locate) | C3 caution |
| GT3 | SceneMap: o-ids, aliases, states, plausibility warnings, held record, `ObstacleSet`, permissions, `model_facts` | H1 #9, #12; C3/C4; V2/V3; IC13 |
| GT4 | TargetResolver at stage entry; legacy `annotate`; palm clearance via SP8 (report-only) | R1; IC8; IC14 |
| GT5 | `locate` (same-response use; snapping to perception masks) | H1 #20; C3 |
| GT6 | metric regions (on, inside, left/right/front/behind of, near) | h1_zero_shot F9 |
| GT7 | overlays on the current pair (ids, boxes, last path) (UD-26) | IC9; h1_collision F2 |
| GT9 | model-facing geometry contract; `decision-uc72-m3` | H1 #9, #20; V6 |
| GT10 | inventory derivation fallback | IC7 |
| GT11 | geometry benchmark + measured σ (`measured_sigma.json`) + pointing calibration S1 (≤ 20 calls) | R8; rule 2 |
| PC3 (b) | plan inventory `objects` (I-07) | IC7; V1/V2 inputs |
| SP1 | robot model build (spheres β = 4/10 mm, UD-33; hulls; visual envelopes) | R2; C4; prerequisite (c) |
| SP2 | batched FK + resolved-rate lift on `robot.kinematics` | R2 |
| SP3 | `SweptChecker` (sphere–OBB, broad phase, permissions) | C4; K6 |
| SP4 | profile adapter on `ExecutedProfileModel` | IC3; M1 |
| SP5 | scene adapter (ObstacleSet → arrays) + permission matrix | M3; IC13 |
| SP8 (report) | depth bound computed and logged, not applied | C2 |
| SP11 (shadow) | planner and per-step checks write `trial/swept_shadow.jsonl`; never model-visible | C4 |
| SP12 | swept replay over the 9 runs on truth and model boxes (R-A, R-B), and on S3's recorded runs | R8 |

**Depends on.**
- M2: exact distances and escape before automatic boxes (IC4, IC13); guard (IC1); robot events; `ExecutedProfile`; `robot.kinematics`.
- M1: finish scoring (IC6).

**Offline checks** (development aids; targets from the designs).
- **PS-12 scorecard:**
  - external detection recall ≥ 0.9 for plan objects;
  - compact-object centre error p90 ≤ 2 cm;
  - conservative-AABB coverage ≥ 0.95;
  - robot raster recall ≥ 0.98 and precision ≥ 0.85;
  - support p90 ≤ 5 mm (or σ-calibrated fallback);
  - false-free rate within 2 cm of true surfaces ≤ 1 %;
  - association accepts the 3 same-instance pairs and rejects the 7 wrong ones;
  - cell binding 11/11 (T04) and 3/3 (T08);
  - discovery finds the T04 can and red mug in ≥ 90 % of frames with ≥ 200 visible px, with 0 components on robot pixels.
- **GT11:**
  - one-view centre ≤ 2 cm and top |err| ≤ 1 cm with the fused support (prototype: 0.7–1.2 cm centre; top −0.0 / +0.1 cm with the far-edge rule);
  - 2σ coverage ≥ 90 % per axis after fitting;
  - plausibility fires on the 3 below-table first maps;
  - geometry compute per decision ≤ 30 ms.
- **GT4:**
  - the legacy replay flags the T08 call-1 approach target (1.1 cm below the bottle top) as `raw_target_inside_object`;
  - an unknown id → `reference_unknown` with 0 actions.
- **GT9:** strict-schema lint, ≤ 6 KB; all H1 T04/T08 responses re-validate through the legacy adapter.
- **SP1:** T1.1–T1.6 (100 % coverage; bulge ≤ β + 2 mm; H2 parity ≤ 1e-9 m in `.venv-sim`; p(ξ) within 0.5 mm of `robot_facts_v2.json`).
- **SP3:** on the T08 fixture with the true AABB, the first negative clearance is at the first post-onset state, on `finger1`; K = 60, M = 20 runs in ≤ 15 ms.
- **SP12 R-A:** recall ≥ 0.9 at onset ±1 for classes A–C on truth geometry, precision ≥ 0.8.
- **PS-11:** with the bridge `off`, the `h1_equivalent` replay stays bit-exact (hooks inert); `RecordedPerceptionClient` reproduces `stage_obstacles` outputs.

**Live checks.**
- Service (Claude may start it for smokes; the sweep runner starts it for sweeps):
  ```bash
  cd /workspace/gpt_field/agent_speedup_v2_uc72_harness/arms/uc72-b003 && CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
    /workspace/gpt_field/.venv-foundation-vision/bin/python -m gpt_field.perception_service.server \
    --socket /tmp/uc72_perception_b003.sock --gpu <idx_free> --min-free-gib 10 \
    --sam-checkpoint /workspace/gpt_field/artifacts/foundation_vision/models/sam3.1/sam3.1_multiplex.pt \
    --da3-model /workspace/gpt_field/artifacts/foundation_vision/models/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c \
    --robot-model src/gpt_field/robot/assets/panda_robot_model_v1.npz \
    --log-dir runs/perception_service_$(date -u +%Y%m%dT%H%M%SZ) --record-frames decision
  ```
- Capped smokes Claude may run:
  ```bash
  .venv/bin/python -I validation/geometry_bench/pointing_smoke.py --cache validation/geometry_bench/cache \
      --max-calls 20 --output validation/geometry_bench/results/pointing_001                     # no simulator
  arms/uc72-b003/run.sh --task obstacle_variant/t04_can_obstacle --config config/arms/m3_shadow.json --gpu <idx> \
      --service perception=unix:/tmp/uc72_perception_b003.sock --max-exec-calls 6 --output runs/m3_smoke_shadow_t04_001
  arms/uc72-b003/run.sh --task obstacle_variant/t04_can_obstacle --config config/arms/m3_geometry.json --gpu <idx> \
      --service perception=unix:/tmp/uc72_perception_b003.sock --max-exec-calls 8 --output runs/m3_smoke_t04_001
  arms/uc72-b003/run.sh --task obstacle_variant/t08_bottle_obstacle --config config/arms/m3_geometry.json --gpu <idx> \
      --service perception=unix:/tmp/uc72_perception_b003.sock --max-exec-calls 8 --output runs/m3_smoke_t08_001
  ```
  Then post hoc, with no model calls:
  ```bash
  MUJOCO_GL=egl .venv-sim/bin/python validation/truth/export_replay_truth.py --run runs/m3_smoke_t04_001 --gpu <idx> --out bench_truth/v1
  /workspace/gpt_field/.venv-foundation-vision/bin/python validation/bench/score.py --frames <service log-dir>/frames \
      --truth bench_truth/v1 --out runs/m3_smoke_t04_001/perception_scorecard.json
  ```
  Pass signals:
  - models warm before the plan call returns;
  - 0 episode aborts from perception; decision wait p95 ≤ 50 ms;
  - the can and the bottle appear as o-ids at call 1 with no operator input, and kσ-cover truth (post hoc);
  - bindings correct (post hoc);
  - the first grasp uses a `target`;
  - 0 schema errors;
  - no `route_start_blocked` caused by a perception box; 0 robot–obstacle contacts;
  - `context.scene` ≤ 2,000 chars; reasoning tokens per call within the b002 arm's range;
  - an overlay on/off PC7 A/B (≤ 40 calls) for UD-26.
- **Sweep S3 (user):** `uc72-b003:m3_geometry` (perception resident on the service GPU) vs `uc72-b002:m2_executor`, interleaved, 1 repeat, `full34` (recommended) or `full44`. Swept shadow and the depth bound log on the b003 arm.
  ```bash
  .venv/bin/python scripts/uc72_sweep.py plan --spec sweeps/specs/s3_geometry.json --output sweeps/s3_001
  .venv/bin/python scripts/uc72_sweep.py run  --plan sweeps/s3_001/sweep_plan.json        # starts/attaches the perception service
  .venv/bin/python scripts/uc72_sweep.py report --plan sweeps/s3_001/sweep_plan.json
  MUJOCO_GL=egl .venv-sim/bin/python validation/truth/export_replay_truth.py --sweep sweeps/s3_001 --gpu <idx> --out bench_truth/s3_001
  .venv/bin/python validation/swept_replay.py --sweep sweeps/s3_001 --truth bench_truth/s3_001 --out validation/results/swept_replay_s3.json
  ```

**Acceptance metrics (S3, against b002).**
- **Zero-shot:**
  - obstacle scenes run with no operator naming (structural: no `--vision-query`, no per-task input);
  - perception availability ≥ 99 % of decision frames; `perception_status` per cell;
  - strict success at finish vs b002 (unresolved below ≈ 30 pp);
  - binding accuracy (post hoc).
- **Fast:**
  - per-decision perception wait p95 ≤ 50 ms;
  - input tokens per call ≤ +0.6k vs b002 (H3 added ≈ 2.9k);
  - reasoning tokens per call vs b002 (G10 confound);
  - policy time vs b002;
  - one service preload per sweep.
- **Collision:**
  - first-call obstacle coverage with kσ in obstacle scenes (post-hoc truth; target 2/2, vs 0/4 for H1-family first boxes);
  - first-target error distribution by source;
  - obstacle contact episodes not above b002;
  - palm-on-target episodes per grasp, with the would-be depth bound logged;
  - shadow planner live recall and precision against contact episodes.
- **Few attempts:**
  - identity-only calls = 0;
  - `reference_unresolved` per task;
  - route refusals and `route_goal_blocked` attributable to perception boxes;
  - locate count and consistency;
  - calls per object vs b002;
  - `scene_edits` use per task.

**Expected effect (est.).**
- First boxes and targets within ≈ 1–3 cm (report C3; offline one-view 0.7–1.6 cm with a correct support), instead of 5–12 cm.
- Obstacle coverage with kσ in the obstacle scenes.
- −2…−3 identity calls per cluttered task (H2 F9).
- ≈ 0 s added critical path per decision (H3 added +1.1–1.3 s per call and 24.6 s preload per task).
- +0.3–0.5k tokens per call.
- 1–2 fewer calls per pick-and-place and 5–8 fewer on the composite (h1_zero_shot F3; hypothesis).
- The collision gain is limited, because the point router still checks the grip site (IC12). T08-class finger and held-object contacts are addressed in M4.

**Risks.**
- **GPU contention or OOM.** Fixed GPU (UD-22), fail-open, free-memory check per episode.
- **Support sweep failure on textureless supports.** Larger σ → more refusals; escape handles starts; `look_motion` (UD-23).
- **Fixtures as large obstacles near targets.** Contact permissions; `kind` edits.
- **Wrong binding.** Overlay ids + `scene_edits bind`; never auto-bind ambiguous cells.
- **Larger schema → invalid outputs.** Lint; the existing SchemaError re-ask.
- **Reasoning inflation from the new context.** PC7 A/B; S3 reasoning-token metric.
- **Disk.** Decision-frame recording ≈ 1–3 GB per perception arm per sweep (est.).

**User decisions before M3.** UD-22 … UD-33.

**Effort.**

| Group | Components | d |
|---|---|---|
| Perception | PS-1 3.0, PS-2 1.0, PS-3 2.0, PS-4 2.0, PS-5 2.0, PS-6 1.5, PS-7 2.5, PS-8 3.5, PS-9 3.0, PS-10 1.0, PS-11 2.0, PS-12 4.0 | 27.5 |
| Integration | INT-5 3.5, INT-4 1.0 | 4.5 |
| Geometry | GT1 2.5, GT2 1.5, GT3 2.5, GT4 2.5, GT5 1.5, GT6 1.5, GT7 1.5, GT9 1.5, GT10 0.5, GT11 1.5 | 17.0 |
| Protocol | PC3 (b) | 1.0 |
| Swept (shadow) | SP1 2.0, SP2 1.0, SP3 3.0, SP4 2.0, SP5 1.5, SP8 (report) 1.0, SP11 (shadow) 1.5, SP12 3.0 | 15.0 |
| Operations | INT-6 | 2.5 |
| **Total** | | **67.5** |

### M4 — Body + held-object swept planner (sweep S4)

**Goal.** Collision reasoning covers the whole robot and the held object along the executed motion. It works as a planner (escape, sequencing choice, detours, standoff handoff), with the palm depth bound and clearance-scaled cruise, and never as a new veto.

**Includes.**

| Component | What | Fixes |
|---|---|---|
| SP6 | `LocalBodyPlanner`: start assessment and escape (monotone clearance; others non-decreasing), direct with sequencing candidates (incl. `coupled`), lift-over / sidestep / L-shapes, goal standoff and `merged_into_contact_stage`, `no_route` only when all fail within 150 ms; direct motions truncated at the last clear sample (UD-36); pre-close jaw sweep | C4; IC4; IC12; K3; K6; H1 #8, #12 |
| SP7 | `BodyRoute` + body chord validator (≤ 5-sample sweep per carrot chord) | G5; H1 #8 |
| SP8 (active) | depth bound applied per UD-34 (per-cell heights from PS-8 with σ below the pad height; otherwise report and guarded descent) | C2; H1 #10 |
| SP9 (cruise) | clearance-scaled command cap `s_max(c) = max(s_contact, (c − bound − band)/(n + a/(1−a)))` | IC2; G5 |
| SP10 | held-object sphere cover from the SceneMap held record; lift leg `permitted_nonincreasing` against the pick support | H1 #12; K2 F1 |
| SP11 (on) | every motion stage and direct motion goes through `plan_stage`; executor oracle → `body_sweep_v1`; `coupled` mode certified by the sweep | C4 |
| EXC-D2 hook | tracker per-arc command cap from `SpeedPolicy` | X23 |
| PS-5/PS-6 (refinement) | wrist-motion plane sweep and contact anchors; scipy multi-view silhouette fit (tighter σ during episodes) | R3; C3 |
| INT-2 (swap) | `StagePipeline` runs `LocalBodyPlanner` when `swept.mode=on` | X14 |
| EXC-C5b (conditional) | posture recovery, only if benchmark scenario 4 validates candidate A or B (UD-37) | C5 |

**Depends on.**
- M3: perception-backed map (IC12, INV-12) and held record.
- M2: rotation sequencing (IC3), braking tracker (IC2) and guard (IC1).

**Offline checks.**
- **SP6:**
  - T08 true AABB → `direct` with `rotate_then_translate`, min clearance ≥ +10 mm (prototype +15.3 mm vs −22.0 mm for the profile that ran), planning ≤ 50 ms;
  - escapes monotone and never re-entering a just-grasped object;
  - T04 carry with the held mug → lift-over with z_req ∈ [0.57, 0.60] m;
  - the 21/111 model-box stages that a hand sweep would refuse resolve locally: ≥ 12 escapes, ≥ 3 handoffs or merges, ≥ 2 detours, `no_route` ≤ 2/111;
  - 20 objects × 8 candidates ≤ 150 ms p95.
- **SP7:** T7.1–T7.3 (`off` replay byte-exact; detours followed clear; ≤ 3 ms per step).
- **SP8:** synthetic per-cell height maps of the two T04 mugs allow the 4 contact-free closes and block the 3 contact closes; ≤ 2 ms per query.
- **SP9:** stopping distance within 20 % of recorded zero-command decays; all 15 recorded onsets inside the slow zone with benchmark bounds; T08 stage ≤ 3 mm/step at closest approach.
- **SP10:** T04 carry at 0.55–0.56 m flagged, ≥ 0.58 m clear; the fallback envelope's below-grip extent ≥ the truth extent on every T04/T08 grasp.
- **SP12 R-B on the S3 recorded runs (live perception maps):** `no_route` ≤ 2 % of stages; no planned route collides with truth where the recorded one did not.

**Live checks.**
- Capped smokes Claude may run (perception service running):
  ```bash
  arms/uc72-b004/run.sh --task obstacle_variant/t04_can_obstacle    --config config/arms/m4_swept.json --gpu <idx> --service perception=unix:/tmp/uc72_perception_b004.sock --max-exec-calls 10 --output runs/m4_smoke_t04_001
  arms/uc72-b004/run.sh --task obstacle_variant/t08_bottle_obstacle --config config/arms/m4_swept.json --gpu <idx> --service perception=unix:/tmp/uc72_perception_b004.sock --max-exec-calls 10 --output runs/m4_smoke_t08_001
  arms/uc72-b004/run.sh --task composite/drawer_bowl_moka          --config config/arms/m4_swept.json --gpu <idx> --service perception=unix:/tmp/uc72_perception_b004.sock --max-exec-calls 12 --output runs/m4_smoke_drawer_001
  ```
  Pass signals:
  - 0 `route_start_blocked`;
  - T04: carry z chosen by lift-over ≥ can top + hang + bound; 0 robot–can and mug–can contacts;
  - T08: rotate-first chosen at the hover; 0 finger–bottle episodes;
  - drawer: 0 arm-link vs cabinet contacts;
  - depth bound: 0 palm-on-target episodes at grasps, or each explained by σ > bound;
  - planner p95 ≤ 150 ms per stage and ≤ 3 ms per step.
- **Sweep S4 (user):** `uc72-b004:m4_swept` vs `uc72-b003:m3_geometry`, interleaved, 1 repeat, scene set per UD-11 (recommended `regression12`, which holds every obstacle and articulated scene; `full34` optional).

**Acceptance metrics (S4, against b003).**
- **Zero-shot:** coverage; strict success at finish vs b003.
- **Fast:** planner time p95 ≤ 150 ms per stage and ≤ 3 ms per step; time attributable to speed caps; policy time vs b003 (expect +5–15 s per task near objects).
- **Collision:**
  - forbidden episodes in routed stages → 0 (target);
  - obstacle episodes in obstacle variants → 0 (target);
  - palm-on-target episodes per grasp well below b003;
  - held-object episodes during lift and carry well below b003;
  - impact speed at onset ≤ 1.5 · s_contact (≈ 1.9 mm/step);
  - carry height minus required height ≥ 0.
- **Few attempts:**
  - `route_start_blocked` = 0;
  - `no_route` ≤ 2 % of stages;
  - calls per task ≤ b003;
  - grasp retries up by ≤ 1 per 5 grasps (the depth-bound shallow-grasp risk).

**Expected effect (est.).**
- With truth-quality geometry, the routed obstacle class (T08 finger press, 3,504 samples) and arm-link vs cabinet (414 samples) are prevented (offline replay).
- The 21/111 would-be veto stages become local actions: 0 added calls, against ≈ +3 per run for a veto (≈ +40 s).
- The T04 carry window 0.54–0.57 m is closed.
- The palm class (55 % of forbidden samples) is targeted; in T04 the bound sits 4–21 mm above contact onset.
- Impact speed at unmodelled contacts ≤ 0.06 m/s within 3 cm of objects (up to 0.53 m/s at the 0.10 m cap).
- Time cost +5–15 s per task near objects.

**Risks.**
- **Geometry quality gates the value.** INV-12: on only with a perception map.
- **Shallow grasps from the depth bound.** Per-cell heights, limiting-cell feedback, grasp predicates, UD-34 fallback.
- **Arm prediction near joint limits** (drawer 001: 139 mm). Limit hits inflate the band; measured joints per step.
- **Compute spikes on the loaded host.** Budget with `no_route_budget`; staged screening.
- **False security.** Every report states the bounds; the contact monitor stays the judge.

**User decisions before M4.** UD-34 … UD-38.

**Effort.** SP6 4.0, SP7 2.0, SP8 (active) 1.0, SP9 (cruise) 1.5, SP10 1.5, SP11 (on) 1.5, EXC-D2 hook 0.5, PS-5/PS-6 refinement 2.0, INT-2 swap 1.0, EXC-C5b 1.0 (conditional), INT-6 2.0 = **18.0 d**.

### M5 — Committed sequences, asynchronous verification and approach speed (sweep S5 = acceptance)

**Goal.** Call the model only at semantic decisions:
- commit whole sub-tasks with measured continuation predicates;
- verify asynchronously where it is safe, and keep blocking checks only for placement relations and articulated engagement;
- speed approaches up inside a measured slow zone.

Then freeze the acceptance build and run the all-task acceptance sweep.

**Includes.**

| Component | What | Fixes |
|---|---|---|
| PC5 | checkpoint protocol: `then` (continue, decide, verify_grasp, verify_view, verify_placement, verify_engagement), `resume`, remainder kept on yields; grasp test-lift and post-placement mandates removed (`agent_client.py:161-164`); `decision-uc72-m5` | D1 protocol; H1 #2; IC5 |
| EXC-D1 (b) | `MAX_SEGMENTS` 12; `CommittedSequence.resume`, `replace_remainder`, `next_is_barrier` | D1 |
| PC6 | asynchronous verification on the `VirtualClock`: arrival at `t_v0 + L`, pacing, barriers (contact, gripper change, checkpoint, end), staleness rules, invalidation by yields; `max_in_flight = 1` (UD-39) | R4; R9; IC14 |
| SP9 (approach) | distance-scheduled approach: slow zone = measured σ (`sigma_along_motion_m`) + braking distance + guard detection distance, for `stop_on: contact` stages with a measured contact expectation only (UD-41) | D2; IC1 step 3 |
| PC2 (cache) | per-episode `prompt_cache_key` if the M2 probe showed it echoed and honoured (UD-40) | D3 |

**Depends on.**
- M2: guard and predicates before commit (IC5).
- M3: measured σ before speed (IC1).
- M4: body planner, for `coupled` and cruise.
- M1: finish scoring (IC6).

**Offline checks.**
- **PC5:** T5.1–T5.4 (16-segment validation logic adapted to 12; `continue` chains 8 segments with 0 calls; `decide` keeps the remainder; `resume` continues it; `verify_placement` holds; no mandate text).
- **PC6:** T6.1–T6.7:
  - exact arrival step under fast and slow simulator stubs;
  - pacing timestamps;
  - barrier idle exactly `t_v0 + L − t_v(barrier)`;
  - a guard yield mid-flight → answer not committed, notes give the reason;
  - a relative-first replacement after motion is rejected;
  - lock-free stats < 5 ms;
  - blocking-mode `virtual_time_s == steps·dt + ΣL`.
- **EXC-D1:** a 12-segment pick-and-place on a fake simulator runs end to end without a handoff; 13 segments are rejected; replacement at the next boundary.
- **SP9:** with benchmark bounds, no recorded onset falls inside a fast phase.
- **Regression:** the full offline suite of M0–M4, and the `h1_equivalent` replay still bit-exact.

**Live checks.**
- Capped smokes Claude may run:
  ```bash
  arms/uc72-b005/run.sh --task obstacle_variant/t04_can_obstacle --config config/arms/m5_blocking.json  --gpu <idx> --service perception=unix:/tmp/uc72_perception_b005.sock --max-exec-calls 6 --output runs/m5_smoke_t04_blocking_001
  arms/uc72-b005/run.sh --task obstacle_variant/t04_can_obstacle --config config/arms/m5_candidate.json --gpu <idx> --service perception=unix:/tmp/uc72_perception_b005.sock --max-exec-calls 6 --output runs/m5_smoke_t04_async_001
  ```
  Pass signals:
  - mug 1 grasp-to-release in ≤ 2 calls;
  - ≥ 1 verify answer applied asynchronously;
  - 0 contacts and 0 gripper changes while a request is in flight (from the trace);
  - stale rejections ≤ 1 in 10 verify calls;
  - late-detected grasp failures = 0.
  - Optional (UD-42 b): one capped hold-step fidelity smoke, to check gap drift during waits.
- **Sweep S5, the acceptance sweep (user):** the frozen `uc72-b005:m5_candidate` vs `uc72-b001:measured_h1`, plus `h1-baseline-3cc84ac:pristine` for the oracle-metric table. `full34` or `full44`; 3 repeats for the A/B decision (UD-09); interleaved.
  ```bash
  .venv/bin/python scripts/uc72_build.py freeze --tag uc72-b005 && .venv/bin/python scripts/uc72_build.py worktrees --tags uc72-b005
  .venv/bin/python scripts/uc72_sweep.py plan   --spec sweeps/specs/s5_acceptance.json --output sweeps/s5_001
  .venv/bin/python scripts/uc72_sweep.py run    --plan sweeps/s5_001/sweep_plan.json
  .venv/bin/python scripts/uc72_sweep.py report --plan sweeps/s5_001/sweep_plan.json
  ```

**Acceptance metrics (S5; pre-registered lexicographic decision rule, UD-09).**
1. **Zero-shot gate:** one digest for every cell of the arm; no per-task inputs; every scene runs from data; reset identity verified or flagged.
2. **Collision-free strict success at finish:** paired by (task, repeat), McNemar exact vs b001; the per-task coverage map.
3. **Fast:** geometric-mean ratio of policy time over pairs where both succeeded (bootstrap 95 % CI over tasks); P(strict ≤ 180 / 300 / 600 s).
4. **Few attempts:**
   - calls per success; calls per object (target ≤ 3, from 6.5–10.5 in H1);
   - checkpoint calls by kind;
   - late-detected grasp failures = 0; stale rejections ≤ 1 in 10 verify calls;
   - approach onsets inside fast phases = 0.

Differences that the design cannot resolve are labelled "unresolved", never "tie".

**Expected effect (est.).**
- **D1:** −4 calls (−60 s) on T04, −8 (−125 s) on T08, −10 (−190 s) on drawers (report; hypotheses). The net saving is ≈ 10 s per avoided call, because committed answers are longer.
- **Async verification:** ≈ −7 s on T04.
- **D2:** −10…−67 s per task, only with measured σ.
- **Cache key:** −3 s on T04, −27 s on drawer 001, if honoured.
- **Overall, honest metric:** T04 ≈ 150–230 s; T08 ≈ 3.5–4.5 min; drawer ≈ 6–9 min (report §3.2 extrapolations plus the finish-scoring cost).
- The 3-minute target needs nearly every lever, including A4's plan fold (deferred), and a normal service window (a slow window adds 60–90 s).

**Risks.**
- **A committed carry pushes through an undetected contact** (IC5). The guard ships first; late failures are counted.
- **Stale rejections add calls.** Measured; blocking mode is the fallback.
- **Concurrency bugs.** One request in flight, deterministic virtual-time tests, a switch to blocking.
- **Longer committed calls.** Accounted for in the estimate.

**User decisions before M5.** UD-39, UD-40, UD-41, UD-42; UD-09 (repeats) before S5.

**Effort.** PC5 3.0, EXC-D1(b) 1.0, PC6 5.0, SP9 (approach) 1.0, INT-6 (acceptance freeze, smokes, S5 supervision, final report) 3.0 = **13.0 d**.

---

## 7. Sweep protocol (common to all milestones)

### 7.1 Scene sets (frozen at M1)
- **`full34`** (34 scenes):
  - 20 official: LIBERO-10 `libero_10_t0…t9` and LIBERO-Goal `libero_goal_g0…g9`;
  - 3 obstacle variants: `t04_can_obstacle`, `t08_bottle_obstacle`, `t08_stove_off_bottle_obstacle`;
  - 1 composite: `drawer_bowl_moka`, with a 70° external camera override and ordered stages as data;
  - 10 relayouts `t00…t09`. `t00`, `t01` and `t06` are flagged not held-out (used historically); `t02` is generated.
- **`full44`** = `full34` + 10 seeded on-path obstacle relayouts (`relayout_obstacle/t0N_seed1`): a non-task LIBERO asset on the source→goal centreline at a seeded fraction in [0.3, 0.7], with a ±3 cm lateral offset. Placements that touch anything at reset or drift > 2 mm during the settle are refused (UD-10).
- **`regression12`** (pre-registered): `obstacle_variant/t04_can_obstacle`, `obstacle_variant/t08_bottle_obstacle`, `obstacle_variant/t08_stove_off_bottle_obstacle`, `composite/drawer_bowl_moka`, `official/libero_10_t0`, `official/libero_10_t4`, `official/libero_10_t6`, `official/libero_10_t9`, `official/libero_goal_g3`, `official/libero_goal_g5`, `official/libero_goal_g8`, `relayout/t06`.
  - Rule: every scene behind the analysed evidence; every relation type (in, on, beside/directional, articulated open/close, turn on); a relational relayout.
  - It is a measurement choice made once, never a policy input.

### 7.2 Arms, interleaving and services
- **Blocks.** Each (task, repeat) is one block. The arms in a block run back-to-back in a randomised, counterbalanced (ABBA) order with a recorded seed. One canary call opens each block.
- **Per-episode checks.** Each episode re-verifies its arm's digest (abort on mismatch), checks GPU free memory for perception arms, resolves the EGL minor, and starts in a fresh output directory.
- **Perception service.** Resident per arm and per sweep: one preload, then `reset_episode` per episode.
- **Pristine H1 arm.** Runs through H1's own runner with `--no-video`, capped externally through its stop file at the call and wall caps. It cannot run changed-sentence or changed-goal scenes (stove-off T08, composite), which get `n/a` cells.

### 7.3 Stop, censor and relaunch

| Outcome | Rule |
|---|---|
| `finish` / `give_up` | scored |
| cap | scored and censored (a failure for success@1; right-censored for time) |
| user stop / operator cap | unscored |
| `infra_error` (transport after retries, simulator crash, required-service crash) | one immediate relaunch in the same block; both runs kept; the cell takes the relaunch's outcome; a second failure → `I` |
| auth infra, or 3 consecutive infra failures | pause the sweep and alert |
| quota | pause all arms until reset (keeps the arms in one window) |
| `harness_exception` | scored failure `E`, flagged, not relaunched |

### 7.4 Power (paired McNemar, α 0.05, logit scene heterogeneity σ ∈ {0, 1, 2}; from R11's simulation)

| Scenes × repeats | 10 pp | 20 pp | 30 pp |
|---|---|---|---|
| 14 × 3 | 0.11–0.15 | 0.38–0.54 | 0.78–0.92 |
| 34 × 1 | 0.08–0.11 | 0.31–0.45 | 0.67–0.82 |
| 34 × 2 | 0.16–0.25 | 0.60–0.80 | 0.95–0.99 |
| 34 × 3 | 0.25–0.35 | 0.80–0.94 | ≥ 0.99 |

**Consequence.**
- One-repeat sweeps (S1–S4) are coverage maps plus mechanism metrics with high counts: vetoes, stalls by class, pressing duration, contact episodes per class, tokens per call, planner outcomes.
- Success-rate differences under ≈ 30 pp are reported as unresolved.
- The success-rate A/B claim is made only at S5 with 3 repeats.

### 7.5 Cost (est. ±30 %, from H1-family call and time distributions; later builds should need fewer calls)

| Sweep | Arms | Scenes × repeats | Calls | Serial wall |
|---|---|---|---|---|
| S1 | pristine H1 + `measured_h1` | full34 × 1 | 1,800–2,950 | 11.6–19.8 h |
| S2 (recommended `regression12`) | b002 + b001 | 12 × 1 | 580–1,020 | 3.6–6.8 h |
| S3 | b003 + b002 | full34 × 1 | 1,620–2,900 | 10.2–19.4 h |
| S4 (recommended `regression12`) | b004 + b003 | 12 × 1 | 580–1,020 | 3.6–6.8 h |
| S5 | b005 + b001 | full34 × 3 | 5,400–8,800 | 35–60 h |
| **Total (UD-11 option b)** | | | **≈ 10,000–16,700** | **≈ 64–113 h** |
| Total if every sweep is full34 (option a) | | | ≈ 12,100–20,500 | ≈ 77–138 h |

Additional costs:
- `full44` adds ≈ 30 %.
- Paired-concurrent execution (UD-08 b) cuts wall time by ≈ 40–50 % if the subscription allows two streams.
- Perception arms hold one GPU for the arm's duration.
- Decision-frame recording costs ≈ 1–3 GB per perception arm per sweep (est.); 86 GB are free.
- Small additions: PC7 A/Bs (≤ 40–60 calls each), the pointing calibration (≤ 20) and the cache probe (≤ 14).

### 7.6 Coverage-map columns (cumulative by milestone)
- **Base (M1):**
  - glyph;
  - policy time to terminal;
  - calls;
  - forbidden and obstacle contact episodes;
  - obstacle displacement;
  - `service_speed_index`;
  - reset-identity status;
  - footer per arm: success@finish strict and raw, collision-free success, P(≤ 180 / 300 / 600 s), calls per success, give_up / cap / infra counts.
- **M2:**
  - envelope vetoes;
  - `route_start_blocked` and escapes;
  - stall-class counts;
  - median pressing duration;
  - false `blocked_motion`;
  - joint-limit streaks;
  - tokens per call.
- **M3:**
  - perception status and availability;
  - first-call obstacle coverage;
  - reference share and `reference_unresolved`;
  - identity calls;
  - locate count and consistency;
  - scene tokens per call;
  - shadow-planner recall and precision.
- **M4:**
  - escapes, detours, handoffs and `no_route` per stage type;
  - palm-on-target and held-object episodes;
  - impact speed at onset;
  - depth-bound hits;
  - grasp retries;
  - carry margin.
- **M5:**
  - calls per object;
  - checkpoint calls by kind;
  - resumes;
  - stale rejections;
  - late-detected failures;
  - overlap and idle seconds;
  - approach onsets in fast phases.

---

## 8. Critical path and staffing

**Recommended staffing: five engineers or coding agents.**

| Engineer | Track | Load (d) |
|---|---|---|
| E1 | runner + integration (R1–R5, R7–R9, INT-4/5/6/7) | ≈ 40.5 |
| E2 | protocol (PC1–PC7) + R6 and R11 (adjacent to PC2 telemetry) + INT-2/INT-3 | ≈ 31.5 |
| E3 | executor (EXC-*, INT-1), then the swept planner `on` (SP6, SP7, SP10, SP11-on) | ≈ 35.5 |
| E4 | perception (PS-1–PS-12) | 29.5 |
| E5 | geometry (GT1–GT11) + swept shadow (SP1–SP5, SP8, SP9, SP11 shadow, SP12) | ≈ 36.0 |

**Critical path: the perception track.** Assuming the decisions arrive on time, the chain is:

```
R1 config/digest (2.5) ─► R7 FrameRing + R2 ServiceSupervisor/ServiceProtocol (≈ 5) ─┐
INT-1 vendor + SP1 robot-model npz with visual envelopes (≈ 5, parallel) ─────────────┤
                                                                                      ▼
PS-1 service (3) ─► PS-2/PS-3 robot raster (3) ─► PS-4 prompts (2) ─► PS-5 support (2) ─► PS-6 association (1.5)
 ─► PS-7 binding (2.5) ─► PS-8 discovery + height field (3.5) ─► PS-9 tracking/held/relocalise (3) ─► PS-10/PS-11 (3)
 ─► GT3/GT4/GT9 integration on live snapshots (≈ 3) ─► M3 offline suite + freeze + shadow/on smokes (≈ 2.5) ─► S3 (1–2 days)
 ─► SP6/SP7/SP10/SP11-on integration and SP12 R-B on S3's recorded runs (≈ 5) ─► M4 freeze + smokes (≈ 2) ─► S4 (≤ 1 day)
 ─► PC5/PC6/EXC-D1(b) integration (≈ 4; developed earlier against fakes) ─► M5 freeze + smokes (≈ 2) ─► S5 (2–3 days)
```

- **Duration.** ≈ 57 working days (≈ 11.5 weeks) from the start of M0 to the end of S5. For one engineer, the 173 d total is ≈ 35 weeks.
- **Off-path tracks** (each must finish before the stage it feeds):
  - M0/M1 runner + protocol (≈ 23 d) → S1 in weeks 5–6;
  - the M2 executor track (≈ 24 d) → S2 in week 6;
  - the geometry track (17.5 d) → needed at M3 integration, week 7;
  - the swept shadow track (15 d) → needed by M3, week 7;
  - the protocol M5 track (8 d) → needed by week 10.
- **Sweeps on the critical path.** Only S3 and S5: M4's offline planner replay uses S3's recorded runs, and S5 is the end. S1, S2 and S4 run while the next milestone is being built.
- **External dependencies.** User decisions (§10); user-started sweeps; a free GPU for the service (only GPU 7 was idle at writing); subscription quota.

**What may start on day 1 (offline, read-only on `runs/`).**
- PS-12 dataset builder and INT-5 exporter design; EXC-0 log fit; INT-1 vendoring;
- SP1 model build; GT1/GT8 library work;
- PC7 dry-run harness; R1/R9.

This follows report §5.2 item 7: the truth-scored benchmark exists before perception changes are consumed.

---

## 9. Deferred backlog

| Item | Why deferred |
|---|---|
| Fold the plan call into the first executive call (A4 fold; UD-15 b) | Removes human review; the first decision would have no inventory-seeded perception (IC7); −14…−52 s per episode. A/B on 3–4 tasks after acceptance. |
| Native function-tool loop for `locate` (geometry D5) | H1's extractor needs exactly one output text (`astra.py:540-541`) and the history commits one response per decision (`agent_client.py:495`). A round trip costs as much as an `observe` call. Revisit after a capped transport smoke. |
| Joint-space posture controller in the worker (C5b option C) | Changes the control interface. Only if Cartesian candidates A/B fail benchmark scenario 4 (UD-37). |
| Higher rotation command cap inside rotate-in-place phases (B1-opt) | Speed only; needs a plant refit at that cap. After acceptance. |
| Persistent fused occupancy map with ray carving and shadow volumes (H2 §4.4 item 3; IC11) | Object-centric conservative boxes + the height field + discovery cover the measured failure classes. Unknown space affects speed only. Revisit if discovery misses obstacles in S3–S5. |
| Harness-generated grasp/place candidates as numbered marks (H2 §4.4 item 9; IC8 second half) | References, regions, the depth bound and the palm envelope address the measured error classes. Revisit if regrasps or placement failures dominate S5. |
| SAM generic prompt sweep for discovery (PS-8 step 6) | The benchmark decides; default empty (UD-24). |
| Wrist-view class-agnostic discovery (PS-8 phase 2) | External discovery plus wrist association comes first. |
| Real-robot items: maximum frame age at use, checkerboard calibration, simulated hold steps by default | No hardware in scope. The contracts already carry capture timestamps, `age_at_use` and a calibration file path. |
| Paired-concurrent sweep execution (UD-08 b) | Needs a two-stream subscription smoke. |
| Render-on-demand (R7 phase 2) and the live raw recorder as default | Only if the M0 overhead benchmark shows rendering still dominates; post-hoc video suffices. |
| Support-part geometry for placement on rimmed supports (SP10 class E, pot vs burner rim) | Needs fixture part segmentation; measured in sweeps first. |
| `prompt_cache_retention` / `max_output_tokens` | Rejected by the transport (`chatgpt_plan.py:37-41`). |
| 16-segment, two-object commits (protocol's original `MAX_SEGMENTS`) | Placement checkpoints are blocking barriers, so no call is saved. Revisit if verify answers are mostly `resume`. |
| Removing raw world targets entirely (UD-27 b) | Decide from the measured reference share in S3–S5 (remove at ≥ 95 %). |
| A K2 (foundation_harness keypose-2) launch | Outside this build, which vendors only K2's robot geometry. If the pending K2 T04 launch runs, treat it as a diagnostic per report §4.3, with the two small fixes first. |

---

## 10. Consolidated user decisions

Original IDs from the designs are in brackets.

| ID | Question | Options | Recommendation | Needed by |
|---|---|---|---|---|
| UD-01 | Where does the new build live? [runner D1] | (a) local clone `/workspace/gpt_field/agent_speedup_v2_uc72_harness` + worktrees; (b) branch + worktrees inside H1's repo; (c) directory copy | (a). H1's tree and `.git` stay untouched. It appears untracked in the main tree; staging it as a gitlink, as was done for H1, is your choice | M0 |
| UD-02 | Standing approval for no-model simulator jobs on a free GPU: reset preflight, pixel fixture and replays, replay-truth export, scripted executor benchmark, post-hoc video [runner preflight, perception UD7, swept UD7] | (a) approve, free GPU, zero model calls; (b) per-job approval; (c) static truth only | (a), about 1–3 GPU-hours per milestone (est.) | M0 |
| UD-03 | Official reset hashes that differ only in PNG bytes and have no stored observation [runner D10] | (a) adopt the pixel hash after a glance at the preflight images; (b) refuse those tasks | (a) | M0 |
| UD-04 | Evaluation caps [runner D2] | (a) 1,800 s policy time + 80 executive calls; (b) 900 s + 60 (censors both drawer successes); (c) 180 s | (a), reporting P(≤ 180 / 300 / 600 s) | M1 |
| UD-05 | Score after a 10-step hold settle or at the finish instant? [runner D3] | (a) hold settle with the last gripper command, plus report the instant; (b) instant only | (a) | M1 |
| UD-06 | Contact taxonomy for collision-free success [runner D4] | (i) palm on the current grasp target; (ii) non-finger push on an articulated part; (iii) resting contact with other goal objects; (iv) held-object support outside the goal region | (i) reported, not forbidden; (ii) forbidden unless the stage declared that contact; (iii) reported, not failing; (iv) forbidden | M1 |
| UD-07 | Video during sweeps [runner D5] | (a) post-hoc re-simulated dashboard; (b) live raw recorder; (c) both | (a) | M1 |
| UD-08 | A/B execution mode [runner D6] | (a) sequential interleaved blocks; (b) paired concurrent on two GPUs | (a); (b) after a two-stream smoke | M1 |
| UD-09 | Repeats and pre-registered decision rule [runner D7] | (a) 1 repeat for S1–S4 coverage + 3 repeats for the S5 A/B; lexicographic rule: zero-shot gate → collision-free strict success → time → calls; (b) 2 repeats everywhere | (a). Pre-register at M1; repeats needed at M5 | M1 (rule), M5 (repeats) |
| UD-10 | Scene set [runner D8] | (a) `full34`; (b) `full44` (+10 seeded on-path obstacle relayouts) | (b) if the budget allows; otherwise obstacle evidence rests on 3 scenes | M1 |
| UD-11 | Sweep scope per milestone | (a) full set at every milestone; (b) full at S1, S3, S5 + `regression12` at S2, S4; (c) full at S1 and S5 only | (b): ≈ 10,000–16,700 calls in total (§7.5) | M1 |
| UD-12 | Subscription quota during an episode [runner D9] | (a) pause and resume the same request (excluded from policy time); (b) fail as infra and relaunch | (a) | M1 |
| UD-13 | Latency canary per block [runner D11] | yes / no | yes (1 call per block) | M1 |
| UD-14 | Baseline arms in S1 [runner D12] | (a) pristine H1 (oracle metric) + measured H1 (honest metric); (b) measured only | (a) | M1 |
| UD-15 | Separate plan call or folded into the first executive call? [protocol UD1, report A4] | (a) separate (human review possible; inventory before decision 1); (b) folded (−14…−52 s; no review) | (a) for all milestone sweeps; (b) as a post-acceptance A/B | M1 |
| UD-16 | May the executor insert short motions the model did not request: escape legs, back-off after an undeclared block, one local replan? [executor UD1, swept UD3] | allow all / escape only / none | Allow all; each is short, guarded, distance-increasing and reported | M2 |
| UD-17 | Interim margin before the body sweep [executor UD2] | plant-derived `m_track` only / `m_track` + 10 mm | `m_track` only (the hidden margin showed no protection) | M2 |
| UD-18 | Fragment step budgets [executor UD3] | remove `MAX_EXECUTION_STEPS` with per-segment plant budgets (`MAX_SEGMENT_STEPS` 240/null) / raise it to 720 | Remove | M2 |
| UD-19 | Use the jaw-symmetric orientation (π about the approach axis) when the requested one is unreachable? [executor UD6] | never / when unreachable, reported | When unreachable, reported | M2 |
| UD-20 | Images in the transcript [protocol UD2] | (a) current pair only (append-only, cacheable); (b) H1 sliding 6 pairs | (a), if PC7 and the M2 smokes show no regression | M2 |
| UD-21 | Drop the `why` prose field? [protocol UD5] | drop / keep | Drop, unless PC7 shows reasoning tokens rising | M2 |
| UD-22 | GPU placement [perception UD1] | fixed GPU per sweep / auto-pick by free memory | Fixed GPU with ≥ 10 GB free for the service (GPU 7 was idle at writing), simulator EGL on another GPU; auto-pick only as a fallback | M3 (M0 for offline jobs) |
| UD-23 | May the harness make a short "look" motion (≤ 5 cm, ≈ 1 s, no model call) when the step-0 support sweep fails? [perception UD2] | never / only on step-0 failure (σ > 1 cm) / always | Only on step-0 failure | M3 |
| UD-24 | Class-agnostic discovery [perception UD4] | DA3 above-support components + SAM box exemplar / + generic prompt sweep; fixtures as obstacles yes/no | DA3 components + exemplar; the generic sweep only if the benchmark shows a gain; fixtures yes, with roles and permissions | M3 |
| UD-25 | How perception boxes reach the planner [perception UD5, swept UD8] | shadow only / merge with model boxes / replace model boxes | Replace: `obstacle_map` leaves the schema at M3; no merge (it would re-import R1's bias); b002 is the comparison arm | M3 |
| UD-26 | What the model sees from perception [perception UD3, UD8; geometry D4] | `context.scene` facts on/off; overlays (ids + boxes + last path) / ids only / none | Facts on; overlays (ids + boxes + last path) on the current pair, confirmed by the PC7 overlay A/B and the H3 channel ablation (M2) | M3 |
| UD-27 | Raw absolute world targets during the transition [geometry D1] | accept with telemetry / remove once references cover ≥ 95 % / remove now | Accept with telemetry in M3–M5; decide removal from the measured share | M3 |
| UD-28 | Confidence multiplier k for σ margins [geometry D2] | 2 / 3 / per-axis χ² | 2, recorded in the digest | M3 |
| UD-29 | May the model remove perception-detected obstacles? [geometry D6] | yes with reason / "dispute" (σ inflated) / no | Yes, with reason, kept in the audit | M3 |
| UD-30 | Plan inventory content [geometry D7] | every distinct visible object ≤ 12 / task objects only | Every distinct visible object ≤ 12 (+60–150 tokens once) | M3 |
| UD-31 | DA3-free plane sweep as a support source [geometry D8 vs PS-5] | include behind robot + instance masks and a peak gate / skip | Include; it enters fusion only if the benchmark shows σ calibration (|err| ≤ 2σ in ≥ 90 % of frames) | M3 |
| UD-32 | Block on event frames (closure/release) for deterministic consumption? [perception UD9] | non-blocking / bounded barrier | Non-blocking, with the conservative fallback envelope | M3 |
| UD-33 | Body-model resolution β [swept UD1] | 4 mm gripper / 10 mm arm (≈ 450–600 spheres) / 5 / 12 mm (372 spheres) | 4 / 10 mm | M3 |
| UD-34 | Palm/fingertip depth bound [geometry D3, swept UD4] | active / report-only / active only with measured per-cell heights whose σ is below the pad height | Report-only in M3; in M4 active only with measured per-cell heights (σ < pad height), otherwise guarded descent + report; revert to report-only if grasp retries rise by > 1 per 5 grasps | M4 |
| UD-35 | Clearance-scaled cruise [swept UD5] | enable in M4 / keep the 0.10 m cap | Enable in M4 after reviewing the M3 shadow logs | M4 |
| UD-36 | Direct motions (≤ 5 steps) against geometry [swept UD6] | truncate at the last clear sample / refuse / unchecked (H1) | Truncate | M4 |
| UD-37 | Posture-recovery mechanism [executor UD5] | report-only (C5a) / Cartesian candidates A or B after the benchmark / joint-space controller in the worker | C5a; enable A or B only if benchmark scenario 4 passes; joint-space only with explicit approval | M4 |
| UD-38 | Keep the rule "body planner only with a perception-backed map" (INV-12)? [swept UD2] | keep / allow model boxes | Keep | M4 |
| UD-39 | Asynchronous verification in the acceptance build [protocol UD3] | async / blocking | Async only if the M5 capped A/B shows no rise in late-detected failures and ≤ 1 stale rejection per 10 verify calls; otherwise blocking | M5 |
| UD-40 | Per-episode `prompt_cache_key` [protocol UD4] | per episode / off | On, if the M2 probe shows the key echoed and cached tokens rising | M5 |
| UD-41 | Approach-speed gate [executor UD8] | measured contact expectation only / also model-declared windows | Measured only (IC1) | M5 |
| UD-42 | Simulate the robot's hold steps during blocking waits? [protocol UD6] | pause + idle accounting / step hold actions for the full latency | Pause + idle accounting; one capped fidelity smoke | M5 |

---

## 11. Traceability: report problem or step → components → milestone

| Report item | Components | Milestone |
|---|---|---|
| R1 LLM writes metric geometry | GT1–GT6, GT9, PS-4–PS-8, PC3 (b) | M3 |
| R2 robot modelled partially | EXC-B1 profile; SP1–SP5 (shadow); SP6–SP11; GT3 permissions | M2, M3, M4 |
| R3 perception backend | PS-3, PS-4, PS-5, PS-6, PS-7, PS-12 | M3 |
| R4 serial blocking loop | PS-11 (async perception); PC5, PC6, EXC-D1 (b) | M3, M5 |
| R5 no contact sensing | EXC-B3 `stop_on`, EXC-C1 | M2 |
| R6 no honest termination | PC1, R2, R4, R11 caps; failure signature | M1 |
| R7 tolerances below the errors | EXC-B2 exact distances + `m_track`; GT1 σ models; PS-6 uncertainty instead of gates; SP5 measured bounds | M2, M3 |
| R8 tests cannot catch live blockers | R9; EXC-INT replay; PS-12, GT11, SP12, INT-5 | M0, M2, M3 |
| R9 simulator-only privileges | INT-4 calibration; PC1 removes "Simulation pauses"; R6 policy time; PS-9 relocalise at use; PC6 virtual clock | M1, M3, M5 |
| H1 #1 per-step overhead | R7, R8 | M0 |
| H1 #2 checkpoint after every phase | PC5, EXC-D1 (b) | M5 |
| H1 #3 cap overshoot and false stalls | EXC-B3; SP9 cruise | M2, M4 |
| H1 #4 contact counts as failure | EXC-B3 `stop_on: contact` | M2 |
| H1 #5 slow primitives | EXC-B1 sequencing; SP9 approach (D2); rotation-cap raise (backlog) | M2, M5 |
| H1 #6 context cost | PC4; PC2 cache key | M2, M5 |
| H1 #7 serial startup | R2 one simulator; plan fold (backlog) | M0 |
| H1 #8 router checks one point | SP3, SP6, SP7 | M4 |
| H1 #9 model boxes wrong | GT3 + PS-* (SceneMap from perception) | M3 |
| H1 #10 palm envelope never given | GT8; SP8 (report → active) | M2, M3, M4 |
| H1 #11 no contact sensing | EXC-C1 | M2 |
| H1 #12 unmodelled motion parts, hidden lattice margin | EXC-B1, EXC-B2; SP10 held object | M2, M4 |
| H1 #13 vetoes waste calls | EXC-B1, EXC-B2 | M2 |
| H1 #14 initial plan unrecoverable gate | PC3 (a) | M1 |
| H1 #15 no reachability | EXC-C5a; C5b (conditional) | M2, M4 |
| H1 #16 transport ends on recoverable errors | PC2 | M1 |
| H1 #17 two-sided oracle | PC1, R4 | M1 |
| H1 #18 never swept | R3, R11, S1 | M0, M1 |
| H1 #19 closed registries, hard-coded facts | R3 task-as-data; INT-4; GT8 from data | M0, M2, M3 |
| H1 #20 no metric grounding tool | GT4, GT5 | M3 |
| H3 V1 part phrases as prompts | PS-4, PS-7, PC3 (b) | M3 |
| H3 V2 operator-named obstacle | PS-8 | M3 |
| H3 V3 advisory only | GT3 ObstacleSet → point router (M3) → body planner (M4) | M3, M4 |
| H3 V4 gates discard evidence | PS-6, GT1 | M3 |
| H3 V5 depth bias and robot self-surfaces | PS-3, PS-5 | M3 |
| H3 V6 synchronous, re-sent summaries | PS-11; GT9 `context.scene` current-only | M3 |
| H3 V7 wall-clock staleness | PS-9 | M3 |
| H3 V8 fail-closed | PS-1, PS-11 | M3 |
| H3 V9 never validated | PS-12, INT-5 | M3 |
| K2 K1 perception per step | PS-11 async, event-indexed | M3 |
| K2 K2/F1 held object self-blocked or absent | GT3 held record; PS-9; SP10 | M3, M4 |
| K2 K3 contact trap | SP6 monotone clearance | M4 |
| K2 K4/F2 uncertain space traversable | not adopted: conservative object boxes; unknown space speed-only | M3 |
| K2 K5 metric keyposes verbatim | GT4 references | M3 |
| K2 K6 RRT budget | SP3 vectorised checker (1–2 ms per configuration vs 107–136 ms) | M3, M4 |
| K2 K8 failures to Astra, no cap | PC1, R4, failure signature | M1 |
| K2 F3 phantom depth | PS-3 | M3 |
| K2 F5 targets block their own grasps | SP5 permissions by id; PS-7 | M3 |
| K2 F8 grounding calls | PS-7, PC3 (b) | M3 |
| Quaternion sign flips (C9) | PC4 hemisphere | M2 |
| Step A1 | PC1, R2, R4 | M1 |
| Step A2 | R7, R8, R9 | M0 |
| Step A3 | PC2 | M1 |
| Step A4 | PC3 (a); fold deferred | M1 |
| Step A5 | R1, R3; S1; INT-4 | M0, M1, M3 |
| Steps B1, B2, B3 | EXC-B1, EXC-B2, EXC-B3 | M2 |
| Step C1 | EXC-C1 | M2 |
| Step C2 | GT8; SP8 | M2, M3, M4 |
| Step C3 | GT1–GT7, PS-* | M3 |
| Step C4 | SP1–SP12, GT3 | M3, M4 |
| Step C5 | EXC-C5a; C5b | M2, M4 |
| Step D1 | EXC-D1 (a) predicates; PC5 + EXC-D1 (b) | M2, M5 |
| Step D2 | SP9 approach | M5 |
| Step D3 | PC4; PC2 cache key | M2, M5 |
| H2 §4.4 items 1–12 | 1: PS-9/PS-11; 2: PS-3/PS-5; 3: partial (height field + discovery), rest backlog; 4: GT3; 5: SP6; 6: PC1/PC3; 7: PS-12/GT11/SP12; 8: SP3/SP9/EXC-B3; 9: GT4 + SP6 standoff, candidates backlog; 10: SP5 + PS-7; 11: object-centric map (N/A); 12: PS-7/PS-8 | M1–M4 |
| H3 §5.2 items 1–8 | 1: PC7 H3 channel ablation; 2: PS-4/PS-7; 3: PS-6; 4: PS-3/PS-5; 5: GT3/SP; 6: PS-11/GT9; 7: PS-9/PS-1; 8: PS-12 | M2, M3, M4 |
| §6 measurement items 1–7 | 1: R4/PC1; 2: R5; 3: R6/R11; 4: R11 power + UD-09; 5: R1/R3 + UD-10; 6: R11 decision rule; 7: R9, INT-5/PS-12 | M0, M1, M3 |

**IC compliance at a glance.**

| IC | Where it is respected |
|---|---|
| IC1 | guard + envelope bound (M2–M4) → measured geometry (M3) → approach speed (M5); INV-1 |
| IC2 | braking (M2) → clearance-scaled cruise with the body sweep (M4); INV-2 |
| IC3 | veto → sequencing + executed profile (M2); INV-3 |
| IC4 | exact distances + escape (M2) before automatic boxes (M3); INV-4 |
| IC5 | guard + predicates (M2) before commits (M5); INV-5; late failures counted |
| IC6 | finish-scoring in M1 before any capability; INV-6 |
| IC7 | inventory in the plan (M3); INV-7 for a fold |
| IC8 | references resolved by the harness (M3) |
| IC9 | overlays replace images; append-only transcript; INV-9 |
| IC10 | frame ring, never decision-only rendering; INV-10 |
| IC11 | no traversable-unknown rule |
| IC12 | body planner only with a perception map (M4); INV-12 |
| IC13 | exclude → escape → automatic boxes → body sweep (M2 → M3 → M4); INV-13 |
| IC14 | relocalise at stage entry, age at use logged, async staleness rules (M3, M5) |
| IC15 | verbatim reasons, enforced by a test (M2) |

---

## 12. Effort

### 12.1 By milestone and by owner (engineer-days)

| Owner | M0 | M1 | M2 | M3 | M4 | M5 | Total | Design total |
|---|---|---|---|---|---|---|---|---|
| ws_runner_measurement | 17.0 | 12.0 | — | — | — | — | 29.0 | 30.5 |
| ws_executor_control | — | — | 21.0 | — | 1.5 | 1.0 | 23.5 | 25.0 |
| ws_perception_service | — | — | — | 27.5 | 2.0 | — | 29.5 | 36.5 |
| ws_geometry_tools | — | — | 0.5 | 17.0 | — | — | 17.5 | 19.5 |
| ws_swept_planner | — | — | — | 15.0 | 11.5 | 1.0 | 27.5 | 30.0 |
| ws_protocol_context | — | 5.5 | 6.0 | 1.0 | — | 8.0 | 20.5 | 20.0 |
| integration (INT-1…INT-7) | 1.5 | 2.0 | 9.0 | 7.0 | 3.0 | 3.0 | 25.5 | — |
| **Total** | **18.5** | **19.5** | **36.5** | **67.5** | **18.0** | **13.0** | **173.0** | 161.5 |

The integration row: INT-1 vendor + `robot/` 3.0; INT-2 stage pipeline 3.5; INT-3 event bus 1.5; INT-4 calibration 1.0; INT-5 truth exporter 3.5; INT-6 milestone operations 12.5; INT-7 GPU resolver 0.5.

### 12.2 Where the duplicated work was removed (−14.5 d, +0.5 d)

| Owner | Change | Δ d |
|---|---|---|
| runner | R10 moved into PC2 | −1.5 |
| executor | EXC-D2 reduced to the tracker-cap hook | −1.0 |
| executor | C5 vendoring moved into INT-1 | −0.5 |
| perception | PS-2 camera model (GT1) | −0.5 |
| perception | PS-3 model build (SP1) | −0.5 |
| perception | PS-5 fusion (GT2) | −0.5 |
| perception | PS-6 single-view and pair lifting (GT1) | −1.0 |
| perception | PS-7 no mark/bind field | −0.5 |
| perception | PS-9 held state (GT3) | −0.5 |
| perception | PS-10 no model-facing facts | −0.5 |
| perception | PS-12 truth export moved into INT-5 | −3.0 |
| geometry | GT3 binding (PS-7) | −0.5 |
| geometry | GT4 palm clamp (SP8) | −0.5 |
| geometry | GT10 schema (PC3) | −0.5 |
| geometry | GT11 cache (PS-12) | −0.5 |
| swept | SP6a dropped | −1.0 |
| swept | SP2 consolidated in `robot/` | −0.5 |
| swept | SP10 attach (GT3) | −0.5 |
| swept | SP12 enrichment (INT-5) | −1.0 |
| swept | SP9 absorbs the D2 approach | +0.5 |
| protocol | PC2 absorbs the quota class | +0.5 |

---

## Appendix A. Source designs and evidence

- Workstream designs: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/design/{ws_runner_measurement,ws_executor_control,ws_perception_service,ws_geometry_tools,ws_swept_planner,ws_protocol_context}.md`. Each holds the full component detail (algorithms, per-component tests, risks) that this plan references by ID.
- Their offline scripts: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/{ws_executor_control_work,ws_geometry_tools_work,ws_swept_planner_work,ws_protocol_context_scripts,ws_runner_scripts}/`.
- Analysis: `/workspace/gpt_field/reports/harness_analysis_20261108/README.md`, `work/w1/`, `work/w2/` (`cross_critic.md` §2 for IC1–IC15), and `work/h2_keypose2_snapshot/` (hash-pinned K2 source).
- Fixture runs: `/workspace/gpt_field/runs/agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001` (T04 replay reference), `…_bottle_obstacle_transit100_t08_20261103_001`, and the stove-off, drawer 001/003, T20, H3 and `foundation_harness_t04_can_obstacle_00{1,2}` runs named in the designs.
