# ws_runner_measurement: build, runner, measurement and runtime overhead

Workstream key: `ws_runner_measurement`. Date 2026-11-08. Read-only design. No file under `/workspace/gpt_field` was edited; no simulator, model call or GPU job was run. The only computation was two small offline numpy scripts (power and cost models) in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/impl_plan/ws_runner_scripts/`.

Abbreviations: `H1/` = `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/` at `3cc84ac` (clean, branch `main`, origin `github.com/seanliu7081/agent-speedup-v2-obstacle-harness-100mm`). `NB/` = the new build (location is decision D1; recommended `/workspace/gpt_field/agent_speedup_v2_uc72_harness/`). "Report" = `reports/harness_analysis_20261108/README.md`. IC*n* = `work/w2/cross_critic.md` §2.

## 0. What this workstream delivers, in one table

| id | component | report steps / problems fixed |
|---|---|---|
| R1 | Build strategy, frozen hashed config, feature flags, build digest | §6.5, F7 (build label mixes 3 generations), cross_eval §8 "frozen builds" |
| R2 | New episode runner core (one simulator, terminal dispatch, caps, service supervisor) | A1, A2 (run_state), F8 (two sim starts), R6 |
| R3 | Task as data: task spec, task pack, reset identity, no-model reset preflight | A5, H1 item 18/19, F7, F11, verify F11 prerequisites |
| R4 | Honest termination and scoring at the policy's own finish; tightened strict check; generic ordered goals | A1, R6, H1 item 17, §6.1, verify_cross (e) gap |
| R5 | Contact episodes by class and phase; obstacle displacement | §6.2, cross_eval M2, verify_cross missed 3–5 |
| R6 | Timing schema (policy time to finish + decomposition) and per-call records (tok/s, service-speed index, canary) | §6.3, G2, cross_eval M3 |
| R7 | Runtime overhead: shared-memory frame ring, lazy observations, compact trace, throttled state | A2, H1 item 1, IC10, IC14 |
| R8 | Video out of the loop (separate live recorder, or post-hoc re-simulation) | A2, verify F1 corrections 2–3 |
| R9 | Zero-call replay on decoded pixels; runner-level equivalence; overhead benchmark | A2 caution, R8, §6.7, verify F1 correction 4 |
| R10 | Transport robustness: forced refresh on 401, transient 5xx/520/unknown_error/deadline | A3, H1 item 16, F10 |
| R11 | Sweep harness: plan, interleaving, stop/censor/relaunch rules, coverage map, cost | §6.4–6.6, rule 3, cross_eval §8 + verify §8 corrections |

Design principle shared by every component: **a "flags-off" configuration of the new build must replay the T04 reference bit-exactly** (actions, state8, contact counters, raw-success flags) and pixel-exactly (every observation and every model image), using the saved responses. That one test proves the runner/overhead/measurement refactor changed no behaviour, and lets each capability flag of the other workstreams be measured in isolation.

---

## 1. Code facts this design rests on (all checked on `3cc84ac`)

| fact | location |
|---|---|
| Transit cap is an env var with default 0.02; OSC scale = max(0.05, cap) | `H1/src/gpt_field/action.py:20-23`; set to 0.10 only by `run.sh:11` and `scripts/launch.py:101` |
| Runner refuses `stop_on_success=False` and any horizon | `H1/scripts/evaluate_agent_speedup_v2.py:213-216`; registry builds `stop_on_success=True` at `agent_speedup_v2_tasks.py:151` |
| Loop ends only on raw success, stop file, step cap, recovery_stop, smoke, exception | `evaluate_agent_speedup_v2.py:1437-1472` (raw success break `:1440-1443`) |
| `finish` is a 1-step hold plus a logged claim; prompt says only the evaluator ends a successful task, "If observation continues…", no give_up | `agent_speedup_v2_user_check_agent.py:366-372`; `agent_speedup_v2_user_check_agent_client.py:205-209` |
| Prepare starts a simulator, resets, makes the plan call, closes it; execute starts a second simulator and resets again | `evaluate_agent_speedup_v2_user_check.py:274-330`; `evaluate_agent_speedup_v2.py:1384-1394` |
| `run_state.json` rewritten at deciding/executing/observed every step, full `agent_stats()` at "observed"; `write_json` is `indent=2, sort_keys=True` | `evaluate_agent_speedup_v2.py:1036-1053`, `:1477`, `:1519`, `:1555`; `experiment.py:23-26` |
| `agent_stats()` deep-copies the framework snapshot | `agent_speedup_v2_user_check_agent.py:639-655`; `agent_speedup_v2_user_check_client.py:285-313, 867-870` |
| `_sync_framework` deep-copies the full feedback history at every `observe_transition` | `agent_speedup_v2_user_check_agent.py:234-260`, call at `:605` (also `:267`) |
| Per-step diagnostics carry `client.history_stats()`; the recorder gets it again | `agent_speedup_v2_user_check_agent.py:574`; `evaluate_agent_speedup_v2.py:1550-1551` |
| Video recorder runs inline twice per step | `evaluate_agent_speedup_v2.py:1515-1517`, `:1549-1552` |
| Trace `executed` row stores the full observation dict (two base64 PNGs) every step; `action` row stores full diagnostics | `evaluate_agent_speedup_v2.py:1512-1514`, `:1533-1535` |
| Worker PNG-encodes both views at PIL default level; `PolicyObservation.__post_init__` decodes both | `observation.py:24-45` (`:44`), `:114-115`; encode reached from `libero.py:951-954` |
| Host decodes again in `from_dict` and again in `CalibratedObservation.__post_init__`; a second `agent_geometry` IPC per step | `libero.py:662-668`; `agent_speedup_v2_user_check_sim.py:83-96`, `:122-128` (IPC `:124`) |
| Images are needed only at decision steps; `StepView` decodes lazily; model PNGs are re-encoded by the controller at level 1 | `contact_skills.py:259-292`; `agent_speedup_v2_user_check_agent_client.py:559-563`; `mark_prompting.py:84-88` |
| Reset identity = sha256 over the observation dict incl. PNG data URLs | `agent_speedup_v2_sim.py:671-674`; replay requires it at `validation/replay_baseline.py:287-292` |
| PNG bytes differ across encoders with identical pixels (the standalone build needed a bundled Pillow/zlib-ng wheel) | `validation/results/reset_mismatch_diagnosis.json` (pixels exact, PNG bytes differ); `setup.sh` encoder pin |
| Replay requires exact serialized observations and image-URL hashes | `validation/replay_baseline.py:48-57`, `:311-312`, `:327-328` |
| Retry: 503 `server_is_overloaded` ≤3 attempts; 4 stream codes ≤2 retries; precondition "no response envelope, nothing committed" | `agent_speedup_v2_client.py:139-150`, `:903-914`; loops at `:916-960` and `agent_speedup_v2_user_check_agent_client.py:446-502` |
| HTTP errors become `APIError(http_status, error_code)`; stream `response.failed` becomes `APIError(error_code)` with no status | `chatgpt_plan.py:136-157` (`:150`), `:330-334` |
| Token refresh only when local expiry < 60 s | `chatgpt_auth.py:450-480` (`:455`) |
| Deadline is SIGALRM per request, `request_deadline_exceeded`, never retried | `agent_speedup_v2_user_check_transport.py:103-128` (`:116-117`), settings `:68-94` |
| Stream progress files hold `headers_received_s`, `first/last_visible_output_s`, `completion_s` | `agent_speedup_v2_user_check_transport.py:433-462`, `:271-287` |
| Worker prompt comes from the official benchmark task, not from the BDDL override | `libero.py:838` |
| Custom BDDL must preserve the official `:language`, `:goal`, `:obj_of_interest` | `libero.py:203-218` |
| Contact classes; palm class merges current target and other goal objects; `robot_scene` absorbs non-finger contact with articulated parts; `manipulated_object_scene` branch not recorded; pair list truncated to 12 | `agent_speedup_v2_contacts.py:66-68`, `:221-223`, `:224-226`, `:228-239`, `:346` |
| Strict "on" checks count only support contacts (below COM, near-vertical normal), so a lean on an obstacle passes | `agent_speedup_v2_truth.py:537-540`, `:671-702`, `:705-757` |
| Release-and-settle counterfactual always commands open; hold-only failures are deferred to it | `agent_speedup_v2_truth.py:74-77`, `:829-834`, `:893-923` |
| Registry: 21 official keys (2 + 10 + 9) with PNG-recipe reset hashes measured in the main-tree env | `agent_speedup_v2_tasks.py:52-108` |
| Launcher: custom can scene init 0 only; refuses resets without a known hash | `scripts/launch.py:20-38` |
| Main tree registers stove-off T08 and the composite by code, with a 70° camera override and a task-specific ordered evaluator | `/workspace/gpt_field/src/gpt_field/agent_speedup_v2_tasks.py:110-138`; `src/gpt_field/libero.py:1003`; `src/gpt_field/ordered_task_evaluation.py:118-131`; `output/layouts/drawer_bowl_right_moka_stove_left_no_obstacles_20261103_002/ordered_success.py` |
| The composite's ordered predicates are all stock LIBERO atoms (`on`, `in`, `open`, `close`, `turnon`, `turnoff`) | `ordered_task_evaluation.py:118-131` |
| H1 venvs carry no project code (`setup.sh` installs requirements only); code comes from `PYTHONPATH` | `H1/setup.sh`, `H1/run.sh:9` |
| Interpreters: H1 `.venv` py3.12.14, numpy 2.5.3, Pillow 12.3.0 (zlib 1.3); `.venv-foundation-vision` py3.12.14, numpy 1.26.4, scipy 1.17.1, Pillow 13.0.0 (zlib 1.3.2) | checked by import |
| Host: 256 cores, 503 GB RAM, `/dev/shm` 251 GB free, `/workspace` 87 GB free | `nproc`, `free`, `df` |

---

## 2. Components

### R1. Build strategy, frozen hashed config, feature flags, build digest

**Purpose.** Report §6.5 (frozen hashed configuration; the transit cap is an env var defaulting to 0.02), cross_eval §8 "frozen builds", F7 (the label `user-check-7.1` covers three code generations). Every result must say exactly which code, configuration, interpreters, task inputs and weights produced it.

**Repository strategy (D1).** Recommended: a **local clone** of H1 as the new build repository, so H1's working tree *and* its `.git` stay untouched:

```bash
git clone /workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm /workspace/gpt_field/agent_speedup_v2_uc72_harness
cd /workspace/gpt_field/agent_speedup_v2_uc72_harness
git remote rename origin h1-local            # no accidental push to the public GitHub repo
git tag -a h1-baseline-3cc84ac 3cc84ac -m "Frozen H1 baseline (uc7.1 transit100)"
git switch -c uc72/dev
```

- Arms always run from **worktrees at immutable tags**, never from the dev checkout: `arms/h1-baseline-3cc84ac/` (detached at the tag, tree identical to H1 `main`) and `arms/uc72-b001/` etc. (`git worktree add arms/<tag> <tag>`; `arms/` is git-ignored in the clone).
- Alternative D1(b): a branch plus worktrees inside the H1 repo itself (adds refs and `.git/worktrees` to H1, H1's checkout stays on `main@3cc84ac`). D1(c): plain directory copy (loses history; not recommended).
- Venvs: worktrees get **symlinks** `.venv`, `.venv-sim` → H1's existing venvs (read-only use; they contain no project code). The clone's `.gitignore` must add `.venv` and `.venv-sim` **without** a trailing slash (git's `.venv/` pattern does not match a symlink, which would make every worktree "dirty").
- Naming: build tags `uc72-bNNN` (b001, b002, …); arm id = `<build_tag>:<config_name>` (e.g. `uc72-b001:measured_h1`); the pristine baseline arm id is `h1-baseline-3cc84ac:pristine`. Any code, prompt, threshold or config change creates a new tag and restarts that arm's sweep (cross_eval §8).

**Files (NB).** Create `src/gpt_field/uc72/__init__.py`, `src/gpt_field/uc72/config.py`, `src/gpt_field/uc72/build.py`, `scripts/uc72_build.py`, `config/schema/uc72_config.schema.json`, `config/arms/h1_equivalent.json`, `config/arms/measured_h1.json`, `config/arms/candidate.json`, `tests/uc72/test_config.py`, `tests/uc72/test_build_digest.py`. Modify `src/gpt_field/action.py` (read the cap from config, no env default), `.gitignore`, `run.sh`, `scripts/launch.py`.

**Config contract.** One JSON file per arm (JSON because the sim worker is py3.10 without `tomllib`). Every numeric leaf is `{"value": x, "source": "<provenance>"}`; the loader **rejects a numeric parameter without a source** (makes user rule 2 auditable). Allowed sources: `inherited:<H1 file:line>`, `measured:<artifact or test id>`, `derived:<formula over measured quantities>`, `user_decision:<Dn>`.

```json
{
  "schema": "uc72-build-config-1",
  "arm": "measured_h1",
  "control": {
    "transit_action_m": {"value": 0.10, "source": "inherited:H1 scripts/launch.py:101 (reference run)"},
    "contact_action_m": {"value": 0.005, "source": "inherited:H1 agent_speedup_v2_user_check_fragments.py:31-33"},
    "control_period_s": {"value": 0.05, "source": "inherited:robosuite control_freq=20 (libero.py:870)"}
  },
  "model": {"model": "gpt-6-astra", "reasoning_effort": "high", "transport": "chatgpt_subscription_ordered_sse",
            "request_deadline_s": {"value": 180, "source": "inherited:H1 scripts/launch.py:47"},
            "image_history_pairs": {"value": 6, "source": "inherited:H1 agent_client.py:44"}},
  "evaluation": {
    "cap_policy_time_s": {"value": 1800, "source": "user_decision:D2 (>=1.38x longest H1-family success, 1301 s)"},
    "cap_exec_calls": {"value": 80, "source": "user_decision:D2 (>=1.86x max H1-family calls, 43)"},
    "cap_wall_s": {"value": 5400, "source": "derived:3*cap_policy_time_s; hang guard on wall minus recorded infra waits; H1 worst wall/(control+plan) = 1.39 (drawer 001)"},
    "post_finish_hold_steps": {"value": 10, "source": "inherited:H1 agent_speedup_v2_truth.py:68 WINDOW_STEPS"},
    "contact_episode_merge_gap_actions": {"value": 6, "source": "inherited:w1/cross_eval_methodology.md §1b convention"}
  },
  "runtime": {"video": "posthoc", "trace": "compact",
              "frame_ring_slots": {"value": 8, "source": "derived:frames readers may hold at once (controller 1 + perception 2 + recorder 2) + 3 writer lead; a buffer size, not a behaviour parameter"},
              "frame_publish_every_n_steps": {"value": 1, "source": "derived:min over subscribers (R7)"},
              "run_state_min_interval_s": {"value": 1.0, "source": "inherited:H1 scripts/evaluate_agent_speedup_v2_user_check.py:381 framework publish rate"}},
  "transport": {"force_refresh_on_401": true, "retry_deadline": true,
                "transient_http_statuses": [500, 502, 503, 504, 520, 521, 522, 523, 524],
                "max_transient_retries": {"value": 2, "source": "inherited:H1 agent_speedup_v2_client.py:144"},
                "overload_max_attempts": {"value": 3, "source": "inherited:H1 agent_speedup_v2_client.py:139"},
                "max_wait_s": {"value": 60, "source": "inherited:H1 agent_speedup_v2_client.py:140"},
                "quota": {"mode": "pause_and_resume", "max_pause_s": {"value": 21600, "source": "user_decision:D9"}}},
  "telemetry": {"latency_reference": {"intercept_s": 6.63, "per_input_ktok_s": 0.0179, "per_reasoning_tok_s": 0.0518,
                "per_visible_tok_s": 0.0074, "source": "measured:w1/h1_speed_attempts.md §2.1 fit, 128 calls, R2 0.949"},
                "canary": {"enabled": true, "every": "block"}},
  "flags": {"runner.single_simulator": true, "runner.terminal_actions": true, "runner.eval_caps": true,
            "runner.lazy_frames": true, "runner.compact_trace": true, "runner.throttled_state": true,
            "runner.contact_records": true, "runner.object_poses": true, "transport.a3": true,
            "protocol.oracle_cues_removed": true, "protocol.give_up": true,
            "perception.service": false, "geometry.locate": false, "swept.planner": false, "executor.guarded_motion": false},
  "components": {}
}
```

- `h1_equivalent.json`: all `flags` false except the evaluator-only ones that do not change behaviour (`runner.lazy_frames`, `runner.compact_trace`, `runner.throttled_state`, `runner.contact_records`, `runner.single_simulator`), and `runner.terminal_actions=false` (raw-success stop as in H1). This is the replay anchor (R9).
- `measured_h1.json`: H1 policy plus all measurement/infra flags (A1, A2, A3, A5) plus `protocol.oracle_cues_removed` and `protocol.give_up` from ws_protocol_context. This is the **honest zero-shot baseline** for the first sweep (IC6: finish-scoring first).
- `candidate.json`: everything on; other workstreams add their sections under `components.<ws>` and their flags.
- `action.py` in NB: `USER_CHECK_TRANSIT_ACTION_M = config.value("control.transit_action_m")`; the config path comes from `UC72_CONFIG` (a pointer, not a value); missing pointer or sha mismatch is a hard error, never a default. The sim worker receives the same path and sha inside its `--config-json` and re-verifies.

**Interfaces.**

```python
# src/gpt_field/uc72/config.py  (py3.10-compatible: imported by the sim worker)
class BuildConfig:
    path: Path; sha256: str; data: dict
    @classmethod
    def load(cls, path: str | Path | None = None, *, expected_sha256: str | None = None) -> "BuildConfig"
    def value(self, dotted: str): ...            # unwraps {"value","source"}
    def source(self, dotted: str) -> str
    def flag(self, name: str) -> bool            # KeyError if a flag is not declared (no silent default)
    def section(self, ws_key: str) -> dict       # components.<ws_key>
def current() -> BuildConfig                     # process-wide singleton from UC72_CONFIG

# src/gpt_field/uc72/build.py
def interpreter_fingerprint(python: str) -> dict  # {python, platform, packages_sha256, pillow_zlib, numpy}
def git_identity(root: Path) -> dict              # {commit, tree, tag|None, dirty: bool}
def build_digest(root: Path, config: BuildConfig, *, interpreters: dict, task_pack_sha256: str,
                 vendor_manifest_sha256: str | None, weights: dict) -> str
```

`build.json` written into every run directory and copied into `summary.json["build"]`:

```json
{"build_id": "uc72-b001", "arm_id": "uc72-b001:measured_h1", "build_digest": "<sha256>",
 "git": {"commit": "...", "tree": "...", "tag": "uc72-b001", "dirty": false},
 "config": {"path": "config/arms/measured_h1.json", "sha256": "...", "flags": {"...": true}},
 "interpreters": {"controller": {"python": "3.12.14", "packages_sha256": "...", "pillow_zlib": "1.3"},
                  "sim": {"...": "..."}, "perception": null},
 "task_pack_sha256": "...", "vendor_manifest_sha256": null,
 "weights": {"sam3.1": null, "da3": null}, "libero_manifest_sha256": "..."}
```

**Algorithm (`scripts/uc72_build.py freeze --tag uc72-bNNN`).** (1) refuse if `git status --porcelain` is non-empty; (2) run `pytest tests/uc72` and the flags-off replay (R9) and record the results in the tag message (a development aid, not a gate: user rule 4; failures are recorded, the user decides); (3) compute `build_digest`; (4) create the annotated tag with the digest JSON; (5) `worktrees` subcommand creates `arms/<tag>/` and the venv symlinks. At every episode start the runner recomputes the digest from its own worktree; in `--require-frozen` mode (sweeps) a mismatch or a dirty tree aborts before any reset.

**Veto/parameter.** Removes the hidden 0.02 default (no defaults at all). Adds no behaviour parameter; every numeric value now carries a recorded source.

**Interactions.** Every workstream registers its flags and its `components.<ws>` section, each numeric with a source (shared interface "config registration"). ws_geometry_tools and ws_perception_service provide a vendor manifest (hash-pinned copies) that enters the digest.

**Offline tests.** (a) Loader rejects a numeric leaf without `source`; rejects unknown flags; (b) `action.py` raises without `UC72_CONFIG`; the controller and the sim worker report the same `TRANSLATION_SCALE_M` for the same file; (c) digest changes when any tracked file, config byte, interpreter package or task-pack file changes, and is stable across two clean checkouts of the same tag; (d) a worktree with venv symlinks reports `dirty=false`. Pass: all assertions.

**Live validation.** None needed beyond R2's capped smoke, whose `summary.json["build"]` must equal the tag's digest.

**Expected effect.** Every result is attributable to one build and one config; per-task flags are structurally impossible (the task spec carries no config, R3).

**Risks.** Symlinked venvs drift if someone upgrades H1's venvs: detected by `packages_sha256` in the digest. Disk: one worktree ≈ 205 MB tracked files (`third_party/`), shared `.git`.

**Effort.** 2.5 engineer-days.

---

### R2. New episode runner core

**Purpose.** A1 (terminal actions, caps), A2 (state writes at decisions, video out of the loop), F8 (one simulator: the plan stage reuses the episode's simulator; −4.9…−8.9 s per episode, verified in verify_h1_speed F8), R6 (bounded failures), the service lifecycle needed by the direction (perception as a service inside the controller). The H1 base runner (`evaluate_agent_speedup_v2.py`, 1,727 lines) carries v1/v2 macro-era machinery (stuck alerts, relation outcomes, ORACLE smokes, lint gates); the new runner is a focused re-implementation that **preserves H1's loop order exactly** (`act → step → observe_transition`, one executive call per `act`, the hold step after a rejected answer), verified by the runner-level zero-call replay (R9). The old scripts stay in NB untouched for comparison.

**Files (NB).** Create `src/gpt_field/uc72/runner.py`, `src/gpt_field/uc72/services.py`, `src/gpt_field/uc72/recorder.py`, `scripts/uc72_run.py`, `tests/uc72/test_runner_termination.py`, `tests/uc72/test_services.py`. Modify `scripts/launch.py` (`--task-spec`, `--config`, `--max-exec-calls` operator cap, `--service name=unix:/path`), `run.sh`.

**Interfaces.**

```python
# src/gpt_field/uc72/runner.py
@dataclass(frozen=True)
class EpisodeRequest:
    task_spec: Path; config: Path; output: Path; gpu_egl_minor: int
    store_dir: Path; profile: str | None = None
    operator_caps: dict | None = None            # {"exec_calls": int, "steps": int}; smoke only, unscored
    services: dict[str, str] | None = None        # resident services from the sweep runner: name -> socket
    require_frozen: bool = False

class EpisodeRunner:
    def __init__(self, request: EpisodeRequest, *, policy_factory=None, client_factory=None,
                 sim_factory=None, transport=None, clock=time.monotonic): ...   # factories = test seams
    def run(self) -> dict                          # returns summary (R4/R5/R6 schema); never raises for policy failures

# Termination kinds (summary["termination"]["kind"])
TERMINATION = ("finish", "give_up", "cap_policy_time", "cap_exec_calls", "cap_wall", "raw_success_stop",
               "operator_cap", "user_stop", "infra_error", "harness_exception")
```

Loop (pseudocode; timers per R6, records per R5/R7):

```
t0 = reset observation time
plan = PlanStage.run(sim, obs)                       # ws_protocol_context API, same simulator (no restart)
while True:
    if stop_file or operator cap: kind = user_stop/operator_cap; break
    if caps exceeded (policy_time, exec_calls, wall): kind = cap_*; break      # checked at step boundary only
    decision = timed("act", policy.act(obs))           # model calls inside are timed by the client hooks
    record decision row if it made a call or changed primitive
    obs2, status, worker_extras = timed("sim", sim.step(decision.action))
    contacts_record(step, phase_label, worker_extras.contacts_action)
    first_raw_success = first_raw_success or (status.libero_success and step)
    timed("observe", policy.observe_transition(obs, decision, obs2)); obs = obs2
    if decision.diagnostics.get("terminal"): kind = terminal.kind; break       # R4 contract
    if flags.terminal_actions is False and status.success: kind = raw_success_stop; break   # H1-equivalent mode
finalize: evaluator hold settle (R4) -> truth -> counterfactual (diagnostic) -> object poses -> summary
```

- Exceptions are classified: `APIError` after R10 retries / `MissingCredentials` → `infra_error` (class from R10); `SimulatorError` (worker died, timeout `libero.py:614-617`) → `infra_error/simulator`; any other exception in policy/client code → `harness_exception` (scored failure, flagged, never relaunched blindly).
- `run_state.json` (R7): written only when an `act` made a model call, on terminal/finalization, or at most every `run_state_min_interval_s`; `agent_stats()` only at decisions and finalization.
- Operator stop file name kept (`trial/operator_stop.request.json`, as `launch.py:139-143`), so the sweep runner and the user stop both arms the same way.

```python
# src/gpt_field/uc72/services.py  (controller side, py3.12)
@dataclass(frozen=True)
class ServiceSpec:
    name: str; argv: list[str]; env: dict[str, str]; socket_path: str
    required: bool                 # False = fail-open (perception)
    resident: bool                 # kept alive by the sweep runner across episodes
    health_timeout_s: float
class ServiceHandle:
    def start(self) -> dict        # identity: {name, pid, interpreter fingerprint, code_sha, weights_sha, device, load_s}
    def health(self) -> dict
    def submit(self, op: str, payload: dict) -> "Future[dict]"      # non-blocking (async perception)
    def call(self, op: str, payload: dict, *, timeout_s: float) -> dict   # blocking; caller wraps in timers.block()
    def reset_episode(self, episode_id: str, task_meta: dict) -> None
    def close(self) -> None
class ServiceSupervisor:
    def attach(self, name: str, socket_path: str) -> ServiceHandle   # resident, started by the sweep runner
    def spawn(self, spec: ServiceSpec) -> ServiceHandle
    def failures(self) -> dict      # {name: [{"t", "kind", "detail"}]} -> summary["services"]
```

Wire protocol (unix domain socket, newline-delimited JSON): request `{"id": int, "op": str, "payload": {...}}`, reply `{"id": int, "ok": bool, "result": {...} | null, "error": {"type": str, "message": str} | null, "timing": {"queue_s", "compute_s"}}`. The ops themselves belong to the service owner (ws_perception_service, ws_geometry_tools).

**Veto/parameter.** Removes the "stop at first raw success" oracle (in non-equivalent configs). Adds the evaluation caps (R4/D2), each with a recorded source.

**Offline tests.** (a) Scripted policy fixtures: finish at step 5 → `kind=finish`, exactly 5+1 steps executed, no further `act`; give_up → `kind=give_up`; a policy that never finishes with an injected clock → `cap_policy_time` at the first step boundary after the cap; (b) the operator stop file ends at a step boundary; (c) an exception in `act` → `harness_exception`, finalization still writes `summary.json`; (d) a worker killed mid-episode → `infra_error/simulator`; (e) the plan stage and the loop use one simulator process (one `start`, one `reset` in the worker log). Pass: all, with a fake simulator (no MuJoCo).

**Live validation.** Capped smoke (user-started, about 4 calls): `./run.sh --task-spec tasks/obstacle/t04_can_obstacle/task.json --config config/arms/measured_h1.json --max-exec-calls 3 --gpu <egl_minor>`. Expect: one simulator start; plan + 3 exec calls; `termination.kind=operator_cap`; complete `summary.json`, `steps.jsonl`, `calls.jsonl`, `contacts.jsonl`; startup-to-first-observation ≤ the H1 prepare+exec startup minus 4.9 s.

**Expected effect.** −4.9…−8.9 s per episode (second simulator start, verified). Bounded failures. Estimates of A1's +1–2 calls per success are in R4.

**Risks.** Hidden behavioural coupling in the old runner (e.g., hold-step order) → caught by the runner-level replay (R9). The H1-equivalent mode must still stop on raw success for the replay to end where the fixture ends.

**Effort.** 4 engineer-days.

---

### R3. Task as data, task pack, reset identity, no-model reset preflight

**Purpose.** A5, H1 items 18–19, F7 (closed registry), F11 (never swept), verify F11 prerequisites (reset preflight; variants not packaged). A task is **data**: sentence + scene (BDDL, initial state, optional camera overrides) + evaluator goal; only the sentence crosses to the policy.

**Files (NB).** Create `src/gpt_field/uc72/taskspec.py`, `config/schema/uc72_task.schema.json`, `scripts/uc72_import_task.py`, `scripts/uc72_preflight_resets.py`, `tasks/` pack (below), `tests/uc72/test_taskspec.py`. Modify `src/gpt_field/libero.py` (worker: prompt from the spec sentence; BDDL validation by spec family; camera overrides), `src/gpt_field/agent_speedup_v2_user_check_sim.py` (pass the spec to the worker), `src/gpt_field/agent_speedup_v2_sim.py` (ordered evaluator hook, R4).

**Task spec contract (`task.json`).**

```json
{
  "schema": "uc72-task-spec-1",
  "task_id": "obstacle/t08_bottle_obstacle",
  "family": "official | obstacle_variant | relayout | relayout_obstacle | composite",
  "held_out": true,
  "sentence": "put both moka pots on the stove",
  "scene": {
    "suite": "libero_10", "source_task_id": 8,
    "bddl": {"path": "custom_task.bddl", "sha256": "..."},
    "initial_state": {"path": "initial_state.npy", "sha256": "..."},
    "official_episode_index": null,
    "camera_overrides": {"agentview": {"fovy_deg": 70.0}}
  },
  "evaluator": {
    "goal": "bddl",
    "ordered_stages": null,
    "added_obstacles": ["ketchup_obstacle_1"],
    "settle_rule": "default"
  },
  "reset_identity": {"known_png_sha256": "7f96963551a6...", "known_pixel_sha256": null,
                     "source": "output/layouts/libero_10_t8_bottle_obstacle_20261103_001/initial_observation.sha256"},
  "provenance": {"imported_from": "...", "files_sha256": {"...": "..."}, "generator": null}
}
```

Rules: (1) `official` family: `bddl=null` means the vendored official BDDL and `official_episode_index` is required; the worker checks the BDDL language equals `sentence`. (2) Other families: the BDDL may change language and goal, but **its `:language` must equal `sentence`** and the evaluator uses the BDDL's own `:goal` (LIBERO evaluates the loaded problem). This replaces both the H1 "must preserve the official language/goal" veto (`libero.py:203-218`) and the main tree's code registry (`agent_speedup_v2_tasks.py:110-138`) with one general consistency rule. (3) `added_obstacles` and `ordered_stages` are evaluator-only; the runner never passes them to the policy (unit test asserts that the policy constructor and every request payload contain no key from `scene` or `evaluator`). (4) `camera_overrides` is applied by the worker after env creation (as the main tree's `libero.py:1003`), generic, recorded in metadata.

**Reset identity.** At reset the worker returns (a) the legacy PNG dict (encoded once, with the bundled encoder) and `png_sha256` (recipe `agent_speedup_v2_sim.py:671-674`), and (b) `pixel_sha256` = sha256 over raw external bytes ‖ raw wrist bytes ‖ state8 as float64 ‖ prompt, plus image shapes. Comparison policy ("record, refuse only on mismatch with a known hash", verify F7): known pixel hash → compare pixel; else known PNG hash → compare PNG; else record as `first_observed` and mark the episode `reset_identity: unverified` (allowed for development, flagged in the coverage map). A PNG mismatch with a stored source observation (layout dirs keep `initial_observation.json`) is re-checked on decoded pixels; equal pixels → adopt `known_pixel_sha256` (recorded as `adopted_from_png_mismatch`); different pixels → refuse (a real scene difference).

**Task pack (initial content).**

| family | ids | source (read-only) |
|---|---|---|
| official (20 for rule 3, +1 optional) | `libero_10_t0..t9`, `libero_goal_g0..g9` (g3 = `drawer_bowl`), optional `bowl_plate` | `H1/src/gpt_field/agent_speedup_v2_tasks.py:52-108` hashes; official BDDL/init in `H1/third_party/LIBERO` |
| obstacle_variant | `t04_can_obstacle`, `t08_bottle_obstacle`, `t08_stove_off_bottle_obstacle` | `H1/tasks/t04_can_obstacle/`; `/workspace/gpt_field/output/layouts/libero_10_t8_bottle_obstacle_20261103_001/`; `.../libero_10_t8_stove_off_bottle_obstacle_20261103_001/` (sentence "turn on the stove and put both moka pots on it", goal adds `Turnon`, known hash `863d4e72…`) |
| composite | `drawer_bowl_moka` | `.../drawer_bowl_right_moka_stove_left_no_obstacles_20261103_002/` (70° external camera; known hash `32fe45e3…`; ordered stages as data, R4) |
| relayout | T00, T01, T06 (`held_out: false`, used historically), T03, T04, T05, T07, T08, T09; T02 to generate | `.../libero10_relayout_recovered_20261105/task_0{0,1,6}/`, `.../libero10_remaining_relayouts_20261105_001/task_0{3,4,5,7,8,9}/` (`relayout_observation_sha256` per `batch_manifest.json`) |
| relayout_obstacle (optional, D8) | one seeded on-path obstacle per relayout | generator below |

`scripts/uc72_import_task.py --from <layout dir> --family <f> --id <id> [--ordered-stages <json>]` copies only the needed files into `tasks/<family>/<id>/`, hashes them, records the source path and source hashes. `--generate-obstacle --relayout <id> --seed <s>` generalises the T08 `prepare_layout.py` procedure (place a non-task LIBERO asset on the source-to-goal centreline at a seeded fraction, lateral offset seeded; refuse placements that touch any object at reset or drift > 2 mm during the 10-step settle — the existing relayout batch checks "no_cross_object_or_robot_contacts_except_table" and "moved_object_support_heights_stable_2mm"). The fraction/offset ranges are a user decision (D8); defaults are the ones cross_eval §8 proposed ([0.3, 0.7], ±3 cm).

**No-model reset preflight.** `scripts/uc72_preflight_resets.py --tasks all --arm-root <worktree> --gpu <egl_minor> --output preflight/resets_<build>.json`: for each spec start the worker, reset, compute both hashes, compare per the policy above, never read credentials, never call a model (network blocked as in `validation/replay_baseline.py:174-179`). It also runs against the pristine worktree's own code (`PYTHONPATH=arms/h1-baseline-3cc84ac/src`, driving `UserCheckSimProcess` directly; nothing is added to that worktree). Output row: `{task_id, png_sha256, pixel_sha256, known, match, resolution, reset_s}`. About 10–15 s per task, about 6–8 min for 34 scenes on one GPU.

**Veto/parameter.** Removes the closed-registry veto (`--task choices`, `launch.py:35-37` hash-required refusal, the official-language/goal BDDL veto). Adds none.

**Offline tests.** (a) Schema validation incl. the sentence = BDDL `:language` rule; (b) importer round trip: the T04 can spec reproduces `H1/tasks/t04_can_obstacle/task.json` hashes; (c) leakage test: no `scene`/`evaluator` key reaches the policy or any request; (d) preflight dry run with a fake worker; (e) the real preflight (simulator, no model) — expected: T04+can PNG hash matches (validated before), others unknown until run. Pass: (a)–(d) green; (e) produces a full table and every mismatch is resolved or listed for the user.

**Live validation.** The preflight table itself; a capped smoke on `t08_stove_off_bottle_obstacle` (changed sentence) and on `drawer_bowl_moka` (70° camera) proving they run with zero code edits (`git diff` of the tag is empty between the two smokes).

**Expected effect.** New layouts and sentences need zero code edits; the sweep provably runs on one build (verify F7). Preflight converts "unknown reset portability" into a measured table before any paid call.

**Risks.** Official hashes were measured in the main-tree simulator env; H1's `.venv-sim` validated only T04+can. Without stored official observations, a PNG-only mismatch cannot be pixel-checked; resolution then needs D10. Generated obstacle scenes may be infeasible; the coverage map marks them `generated` and reports them separately.

**Effort.** 3.5 engineer-days (incl. converting the composite's ordered stages).

---

### R4. Honest termination and scoring at the policy's own finish

**Purpose.** A1, R6, H1 item 17, report §6.1, IC6 ("do it first"), verify_cross (e) gap (strict accepts a mug leaning on the obstacle). The prompt-side cue removal is ws_protocol_context's; this component defines the contract they must emit and the evaluator side.

**Termination contract (emitted by the policy; ws_protocol_context owns prompt and schema).**
- Agent action types gain `give_up`; `finish` becomes terminal. The decision for the step on which the model answered carries:

```json
{"terminal": {"kind": "finish | give_up",
              "reason_class": "task_complete | infeasible | repeated_failure | object_lost | perception_unavailable | other",
              "reason": "<= 600 chars, model text",
              "claim": "<model belief at finish>",
              "decided_at_step": 412, "request_key": "..."}}
```

- The runner executes that decision's action (the existing 1-step hold, `agent_speedup_v2_user_check_agent.py:366-372`) and never calls `act` again. Nothing after a terminal decision reaches the model, so no "episode continues" cue can exist.
- The completion contract the protocol text must state: the object is released and the hand is clear **before** `finish` (release is part of the task). Caps are not disclosed to the policy (they are an evaluation property; disclosing a budget would be a policy feature owned by ws_protocol_context).

**Evaluator side (worker ops, evaluator-only).**
1. **Hold settle** `settle_hold(steps=post_finish_hold_steps)`: zero motion with the policy's last gripper command (new `hold_settle_actions(last_gripper, steps)` next to `release_settle_actions`, which always commands open, `agent_speedup_v2_truth.py:829-834`). It cannot complete the task for the policy: a still-closed gripper keeps the object held and `not_held` fails. Also report the "at the finish instant" verdict (no settle) as the stricter variant (D3).
2. `truth()` at the settled finish state: raw LIBERO (`env.check_success()`), strict sub-checks, plus one new check per placed atom: **`not_resting_on_obstacle_or_distractor`** = no environment contact (any normal direction) between the placed object and (a) any `added_obstacles` object or (b) any movable object that is neither a goal object nor the atom's support/container/contents. Contacts with other goal objects are reported (`rest_contacts_other_goal_objects`) but not failing (D4-iii). Implemented in `_obj`/`explain_*` of `agent_speedup_v2_truth.py` from the existing `o["env"]` contact list; the worker gets `added_obstacles` from the spec.
3. Counterfactual `settle_truth()` (open, lift, settle; unchanged) runs **only as a diagnostic** (`would_succeed_if_released`), never deferring the headline verdict (the H1 deferral, `agent_speedup_v2_truth.py:902-903`, is not used for scoring).
4. Ordered goals (composite): `evaluator.ordered_stages` = list of `{"id": str, "all": [[pred, arg...], ...]}`; a stage completes on a false→true edge of its conjunction observed after the previous stage completed; `ordered_complete` requires all stages plus the final BDDL goal at finish. The worker evaluates stage atoms with LIBERO's own `_eval_predicate` each step, then calls `env.check_success()` **last**, so turn-predicate visual side effects end in the same state H1 produces (the main-tree author's note in `ordered_task_evaluation.py:128-130`). The composite's stages are exactly the main tree's five (`ordered_success.py` STAGES) written as atoms.

**Scores (summary["scoring"]).**

```json
{"scored": true, "censored": false,
 "success_at_finish_strict": false, "success_at_finish_raw": true, "collision_free_success": false,
 "ordered_complete": null,
 "strict_checks": {"<atom>": {"not_held": true, "at_rest": true, "supported_only_by_target": true,
                              "not_resting_on_obstacle_or_distractor": false, "...": true}},
 "failing_checks": {"On white_mug_1 plate_1": ["not_resting_on_obstacle_or_distractor"]},
 "at_finish_instant": {"raw": true, "strict": false},
 "would_succeed_if_released": null,
 "libero_first_success": {"step": 549, "policy_time_s": 271.3},
 "raw_success_at_finish": true}
```

- `scored`: true for finish, give_up and caps (caps: `censored=true`, a failure for success@1 and right-censored for time); false for user stop, operator cap, infra.
- Headline: `success_at_finish_strict`; `collision_free_success` additionally requires zero forbidden contact episodes under the D4 taxonomy (R5). `libero_first_success` is the standard LIBERO metric, kept for comparability with published numbers (verify F1), never a terminator outside the H1-equivalent config.

**Veto/parameter.** Removes the stop-at-first-raw-success oracle and the "evaluator terminates success" cue (with ws_protocol_context). Adds: caps (D2: 1,800 s policy time / 80 executive calls; sourced as ≥1.38× and ≥1.86× the longest/most-called H1-family success: drawer 001 at 1,301 s control+plan and 43 calls, so no historical success would have been censored; cross_eval's 900 s / 60 would have censored both drawer successes, verify_cross missed issue 6), and `post_finish_hold_steps=10` (inherited `WINDOW_STEPS`, the strict rest window, not a new tolerance). Time-to-success is reported as empirical P(strict success at finish ≤ t) for t ∈ {180, 300, 600, cap}; with one uniform cap no Kaplan–Meier estimate is needed below the cap.

**Offline tests.** (a) Pure-function test of the new rest-contact check on the recorded `env_contacts` of `runs/astranomy_direct_can_obstacle_t04_collision_avoidance_high_20261105_001` (white mug with `tomato_sauce_obstacle_1:g14/g22` → must fail) and of the H1/H3 after-settle records (must not fail where no distractor/obstacle contact is listed); (b) hold-settle action builder keeps the last gripper command; (c) ordered-stage evaluator on a synthetic predicate sequence reproduces `ordered_success.py`'s stage events (same inputs from its own tests, edge semantics, early actions retryable); (d) scoring table unit tests for each termination kind. Pass: all.

**Live validation.** Capped smoke on a single-relation LIBERO-Goal task (`libero_goal_g8` "put the bowl on the plate", cap 20 exec calls) expected to reach `finish`; check: no model call after the finish decision, hold settle executed, strict and raw computed at the settled state, `libero_first_success` logged. In the sweep: success@finish strict and raw, give_up count and reasons, cap count, P(≤180 s).

**Expected effect (report estimates).** An honest baseline at about +1–2 calls (+20–40 s) per success (A1). uc7.1-T20 T04 would have failed at its own finish (h1_zero_shot F1). The honest rate of the 7 retained uc7.1 runs is unknown (0/7–6/7, verify F1), so the first measured sweep sets it.

**Risks.** The agent may disturb a completed relation before finishing (legitimate failure, lowers the headline). A model that never uses `finish` turns successes into caps — mitigated by the protocol's completion contract and visible in the give_up/cap/finish mix. Turn-predicate side effects (render state) are covered by the "check_success last" rule and a pixel-equality smoke on T02/T08.

**Effort.** 2 engineer-days (evaluator side; the schema/prompt side is ws_protocol_context's).

---

### R5. Contact episodes by class and phase; obstacle displacement

**Purpose.** Report §6.2 (sample totals inverted the T04 ranking), cross_eval M2, verify_cross missed issues 3–5 (palm class merges current target and other goal objects; articulated pushes classified forbidden; arm links on the static cabinet unreported; classifier branch unrecorded). Also provides the onset statistics IC1/IC2 require for later speed changes.

**Files (NB).** Modify `src/gpt_field/agent_speedup_v2_contacts.py` (return a subclass and branch from `classify`; per-action records; **existing class counters unchanged**, so the replay stays exact), `src/gpt_field/agent_speedup_v2_sim.py` (include `contacts_action` and `object_poses` op), `src/gpt_field/agent_speedup_v2_user_check_sim.py` (worker op wiring). Create `src/gpt_field/uc72/metrics/contact_episodes.py`, `src/gpt_field/uc72/metrics/rescore_history.py`, `tests/uc72/test_contact_episodes.py`.

**Per-action worker record (evaluator-only; step reply `contacts_action`).**

```json
[{"class": "robot_nonfinger_manipulated_object", "subclass": "palm_current_target",
  "branch": null, "owners": ["robot0", "white_mug_1"], "geoms": ["robot0_hand_collision", "white_mug_1_g3"],
  "samples": 26, "max_penetration_m": 0.0021, "max_normal_force_n": 3.4, "first_substep": 0}]
```

Subclasses (worker-side; class counts stay as today):
- `robot_nonfinger_manipulated_object` → `palm_current_target` (owner finger-touched within `MANIPULATED_MEMORY_SUBSTEPS`) | `palm_other_goal_object`.
- `robot_scene` → `robot_static_scene` (arena/static fixture) | `robot_articulated_nonfinger` (jointed fixture part, `:224-226`).
- `robot_object` → `robot_obstacle` (owner in `added_obstacles`) | `robot_distractor`.
- `manipulated_object_scene` → `held_obstacle` | `held_side_impact` (normal > 35°, `:229-231`) | `held_outside_goal_region` (`:233-236`) | `held_on_distractor` (`:237-238`).
- Force: `mujoco.mj_contactForce` only for contacts in forbidden classes (rare; bounded cost).

**Runner join.** `contacts.jsonl` row per action with ≥1 non-ignored contact: `{"step", "phase": PhaseLabel, "c": [...]}` where `PhaseLabel` comes from ws_executor_control (shared interface; H1-equivalent fallback derived from `policy._primitive.progress()` like the existing `planning_snapshot`, `evaluate_agent_speedup_v2_user_check.py:658-667`).

**Episode algorithm.** For each (subclass, owner pair): maximal runs of actions with ≥1 sample, merging gaps ≤ `contact_episode_merge_gap_actions` (6, the convention under which the report's baseline tables were computed; sensitivity at 0 and 12 reported). Episode record:

```json
{"subclass": "robot_obstacle", "owners": ["robot0_gripper", "ketchup_obstacle_1"], "start_step": 37, "end_step": 83,
 "n_actions": 47, "samples": 3504, "max_penetration_m": 0.006, "max_normal_force_n": 41.0,
 "onset_phase": {"mode": "fragment", "skill": "move_to_pose", "profile": "transit", "segment": 1, "declared_contact": null},
 "phases": ["transit"], "onset_speed_m_s": 0.18, "onset_command_m": 0.017, "onset_target_distance_m": 0.052,
 "intended": false, "counted_as": "forbidden"}
```

`intended` is true only if the active stage declared contact with that part (ws_executor_control's `declared_contact`); `counted_as` follows the D4 taxonomy. Summary: `contacts.episodes_by_subclass`, `contacts.max_episode_actions`, `contacts.first_forbidden_step`, plus `obstacle_displacement` from the new `object_poses` op at reset and after the hold settle: `{name: {"displacement_mm", "tilt_change_deg"}}` for every added obstacle and every non-goal movable object (reported as severity, no threshold).

**Historical rescoring.** `rescore_history.py --run <dir>` rebuilds episodes from old traces by differencing cumulative `sim_status.collision` counters and pair records step by step (as the analysts did); subclasses that need worker data (branch) are `unknown` for old runs.

**Veto/parameter.** None removed (record-only stays record-only). The merge gap is an inherited measurement convention with sensitivity rows.

**Offline tests.** (a) Rescoring reproduces cross_eval §1b / verify_cross (c) exactly: H1 T04 3 owner-pair episodes, 0 obstacle; uc7.1-T20 T04 2 episodes, 1 obstacle (mug–can 141–144); H1 T08 5 episodes, 3 obstacle (gripper–bottle 37–83, 47 actions, 3,504 samples); H3 T08 2 (pot–burner); drawer 001 10, drawer 003 5. (b) The modified monitor's class counters equal the old ones in the zero-call replay (all 549 steps). (c) Subclass unit tests on synthetic contact tuples. Pass: exact equality for (a) and (b).

**Live validation.** Every sweep episode reports episodes by subclass and phase; the coverage map shows obstacle and forbidden episodes per cell; IC1's onset-distance and IC2's onset-speed distributions are produced per build.

**Expected effect.** Rankings no longer invert (T04: 3,372 samples of palm-on-mug vs a 4-action can contact are now different subclasses). It makes the collision goal measurable per class.

**Risks.** Forbidden-class definitions are a user decision (D4). The goal-aware tier's goal-object set comes from the BDDL; for changed-goal tasks it changes accordingly (correct by construction).

**Effort.** 3 engineer-days.

---

### R6. Timing schema and per-call records

**Purpose.** Report §6.3 (log tok/s per call; decode speed 31.8–61.1 tok/s), G2, cross_eval M3 (timing definitions differ; evaluator work inside timers), C1 (H3's 2× latency mostly service window), verify LAT (input tokens cost 11–21 ms per 1k).

**Files (NB).** Create `src/gpt_field/uc72/timing.py`, `src/gpt_field/uc72/metrics/calls.py`, `scripts/uc72_canary.py`, `tests/uc72/test_timing.py`. Modify `src/gpt_field/agent_speedup_v2_user_check_agent_client.py` and `src/gpt_field/agent_speedup_v2_client.py` only to emit one call record per request (hook; no behaviour change), and make `--stream-progress` mandatory (config).

**Policy time (pre-registered definition).** From the reset observation (t0) to the terminal decision's step:

```
policy_time = Σ_calls latency(successful attempt)                      # robot frozen during inference (H1 contract)
            + Σ blocking waits (perception/geometry/planner, only while the robot is frozen)
            + Σ_steps max(control_period, compute_step)                 # compute overlaps motion on a real robot
compute_step = act_wall − model_latency − blocking_waits + observe_transition_wall
```

Excluded and reported separately: simulator wall beyond real time, rendering/encoding for evaluation, IPC, trace/run_state/contacts writes, video, truth, hold settle, failed transport attempts and quota waits (`infra_wait_s`). Plan call: inside policy time. Startup before t0 and finalization: reported in wall only.

**Timers API (shared with all workstreams).**

```python
class Timers:
    def block(self, kind: str):    # context manager; kind in {"model","perception","geometry","planner"}; robot frozen
    def compute(self, kind: str):  # policy compute inside act/observe that overlaps motion
    def evaluator(self, kind: str):# evaluator-only work (excluded from policy time)
    def async_done(self, kind: str, compute_s: float, frame_capture_ns: int, used_at_step: int)  # async results
    def step_record(self) -> dict  # per-step: {"s","act_s","model_s","block_s":{...},"compute_s":{...},"sim_s","obs_s","io_s","worker":{...}}
```

`steps.jsonl` row per step (≈200 B): `{"s": 412, "t": 251.03, "a": [7 floats], "x": [8 floats], "fid": 412, "ph": {...}, "raw": false, "tm": {...}}`.

**Per-call record (`calls.jsonl`).**

```json
{"k": 7, "kind": "agent | plan | canary", "step": 196, "trigger": "segment_stall | fragment_complete | veto | initial | ...",
 "attempts": [{"n": 1, "elapsed_s": 0.37, "outcome": "auth_expired", "http_status": 401, "error_code": "token_expired", "wait_s": 0.0},
              {"n": 2, "elapsed_s": 17.6, "outcome": "ok"}],
 "latency_s": 17.6, "headers_s": 2.1, "first_visible_s": 11.8, "last_visible_s": 17.2, "completion_s": 17.6,
 "input_tokens": 51044, "cached_tokens": 3072, "output_tokens": 392, "reasoning_tokens": 152, "visible_tokens": 240,
 "visible_tok_s": 44.4, "previsible_ms_per_reasoning_token": 63.8, "request_bytes": 3210000, "images": 12,
 "predicted_latency_s": 16.6, "service_speed_index": 1.06, "response_type": "execute"}
```

- `visible_tok_s = visible_tokens / (last_visible_s − first_visible_s)`; `service_speed_index = latency_s / predicted_latency_s` with the frozen reference fit `L = 6.63 + 0.0179·input_k + 0.0518·reasoning + 0.0074·visible` (measured on 128 H1 calls, R² 0.949; verified in verify_h1_speed LAT). A run-level median index lets arms run on different days be normalised, and flags slow windows (H3's 2026-11-05 window).
- Canary (D11): one fixed text-only request (fixed JSON schema, about 300 visible tokens, reasoning high, no images) at the start of each sweep block; recorded with `kind=canary`, not part of any episode. Cost: 1 call per block.

**Summary timing block.**

```json
{"timing": {"wall_s": 402.1, "startup_s": 10.7, "policy_time_to_terminal_s": 291.0,
            "decomposition": {"model_s": 217.8, "model_headers_s": 25.0, "model_previsible_s": 129.0, "model_visible_s": 63.8,
                              "perception_blocking_s": 0.0, "geometry_blocking_s": 0.0, "planner_blocking_s": 0.0,
                              "motion_s": 27.45, "compute_excess_s": 0.0},
            "async": {"perception_compute_s": 0.0, "median_frame_age_at_use_s": null},
            "evaluator": {"sim_wall_s": 15.0, "trace_io_s": 1.0, "run_state_s": 0.4, "contact_monitor_s": 3.0,
                          "truth_s": 1.2, "hold_settle_s": 0.6, "video_s": 0.0},
            "infra_wait_s": 0.0, "per_step_overhead_s": {"median": 0.045, "p95": 0.07, "q1_median": 0.044, "q5_median": 0.046},
            "first_raw_success": {"step": 549, "policy_time_s": 271.3}}}
```

**Veto/parameter.** None. The latency reference coefficients are measured quantities (recorded with source).

**Offline tests.** (a) Policy-time formula on synthetic step records (compute below/above the control period; blocking vs async); (b) call-record builder on the 13 saved T04 `requests/*.stream.json` + `response.json` reproduces h1_speed §2.1 (13 calls, Σ 198.5 s, median 13.4 s, input 7,945 → 87,425, cached 2.5 %); (c) service-speed index ≈ 1 on the fit set (median within 0.95–1.05). Pass: exact (b), (a) unit, (c) range.

**Live validation.** Capped smoke shows all fields populated; the sweep report includes per-arm medians of `visible_tok_s` and `service_speed_index`, and arms differing by > 10 % in median index get a "service-window confound" flag in the report.

**Expected effect.** Comparisons stop being confounded silently by service speed (31.8–61.1 tok/s range, G2); "fast" is reported on a real-robot-meaningful clock.

**Risks.** Stream timestamps depend on `--stream-progress` (now mandatory). Async perception compute is reported but not in policy time unless the executor actually waits; ws_perception_service must use `Timers.block` for any wait.

**Effort.** 2 engineer-days.

---

### R7. Runtime overhead: frame ring, lazy observations, compact trace, throttled state

**Purpose.** A2, H1 item 1 (0.28–0.40 s per 0.05 s step), IC10 (raw frames at a fixed rate into shared memory; PNG only for the model; perception subscribes), IC14 (capture time stamped against action start), verify F1 corrections and lowest-risk order (drop redundant decodes; recorder out of process; lazy encode; state only at decisions).

**Files (NB).** Create `src/gpt_field/uc72/frames.py` (py3.10-compatible), `tests/uc72/test_frames.py`. Modify `src/gpt_field/libero.py` (worker `_response` in lazy mode: no PNG; publish frame), `src/gpt_field/agent_speedup_v2_user_check_sim.py` (geometry in the step reply instead of a second IPC; `FrameObservation`), `src/gpt_field/contact_skills.py` (`StepView._image` takes arrays from a `FrameObservation`), `src/gpt_field/agent_speedup_v2_user_check_agent.py` (drop `_sync_framework` at `:605`; `history_stats` only at decision steps `:574`), `src/gpt_field/uc72/recorder.py` (compact trace; throttled `run_state`).

**Frame ring contract (`/dev/shm/uc72_<run_id>_frames.bin`, `np.memmap`, created by the runner, unlinked at exit).** Plain memmap rather than `multiprocessing.shared_memory`, to avoid the pre-3.13 resource-tracker unlinking an attached segment.

```
header (4096 B): magic "UC72FRM1", version u32, n_slots u32, n_cams u32, H u32, W u32, latest_frame_id i64
slot i: seq i64 (odd while writing), frame_id i64, step i64, sim_time_s f64, capture_monotonic_ns i64,
        state8 f64[8], joint_pos f64[7], joint_vel f64[7], geometry_len u32, geometry_json u8[8192],
        pixels u8[n_cams][H][W][3]   (raw robosuite obs arrays: obs["agentview_image"], obs["robot0_eye_in_hand_image"])
```

- Writer (sim worker, after each `env.step` or every `frame_publish_every_n_steps`, and at reset): seqlock write (`seq+=1`, copy, `seq+=1`), then `latest_frame_id`. 2 × 512² × 3 = 1.57 MB memcpy per publish (sub-millisecond).
- Readers (controller, perception service, live recorder): `FrameRing.open(path).latest()` / `.get(frame_id)`; retry while `seq` is odd or changed; `get` raises `StaleFrame` if the slot was overwritten (bug detector, never silent).
- `CLOCK_MONOTONIC` is system-wide on Linux, so `capture_monotonic_ns` is comparable across processes; the runner records each step's action-start time in `steps.jsonl`, giving IC14's freshness check (`frame age at use = action_start − capture`).

**Lazy observation.** `FrameObservation(PolicyObservation-compatible)`: fields `state`, `prompt`, `joint_position`, `joint_velocity`, `geometry`, `frame_id`, `ring`; `rgb(camera)` returns the **same arrays H1's decode path yields** (raw → `[::-1, ::-1]` in `image_to_data_url` → `[:, ::-1]` in `StepView` = `raw[::-1, :]`; PNG is lossless, so equality is exact); arrays are copied out of the ring at first access in that step. `image`/`wrist_image` (data-URL properties) are produced on demand for legacy callers. `__eq__` compares `(frame_id, state, prompt, joints)` (needed by `act`'s `observation != self._expected_observation` check, `agent_speedup_v2_user_check_agent.py:498`). Geometry (camera calibration) travels in the step reply (one IPC per step instead of two, `agent_speedup_v2_user_check_sim.py:124`). The reset observation still carries the legacy PNG dict once (R3).

- Removed per step: 2 PNG encodes (level 6) and 6 decodes (`observation.py:44`, `:114-115`; `libero.py:663`; `agent_speedup_v2_user_check_sim.py:88-89`); the second `agent_geometry` IPC; the per-step trace observation copy (≈0.56 MB/step, 308 MB for T04); per-step `run_state` + `agent_stats`; per-step `_sync_framework` deep copy and `history_stats`.
- Kept: the level-1 PNG encode of the two decision images by the controller (`agent_speedup_v2_user_check_agent_client.py:559-563`), i.e. "PNG only for frames sent to the model".
- Phase 2 (optional, only if R9 shows render ≥ the remaining overhead): disable robosuite camera observables and render on demand (`op: render` at the current state when a `FrameObservation` is accessed without a frame, plus every `n` steps for subscribers). `n` = the smallest frame interval any subscriber declares it consumes (perception's measured cycle in control steps; recorder's rate), so the rate is derived, never fixed per task.

**Compact trace.** `trace.jsonl`: `reset` row (legacy observation once, reset identity, task spec sha), `decision` rows only for steps with a model call or primitive change (full diagnostics, verbatim feedback), `feedback` rows only on fragment end/veto/stall. Per-step data go to `steps.jsonl` (R6) and `contacts.jsonl` (R5). `run_state.json` written at decisions, terminal, finalization, or ≤ 1 Hz.

**Veto/parameter.** None removed (no behaviour change). Parameters: ring slots (derived from concurrent readers), publish interval (derived from subscribers), state interval (inherited 1 Hz).

**Offline tests.** (a) Ring: concurrent writer/reader processes (py3.10 writer, py3.12 reader) with 10⁴ frames, no torn reads, `StaleFrame` on overwrite; (b) `FrameObservation.rgb` equals H1's decoded arrays for the saved T04 reset (decode `tasks/t04_can_obstacle` initial observation PNGs); (c) runner-level zero-call replay (R9) bit-exact with lazy frames; (d) overhead targets in R9. Pass: (a)–(c) exact; (d) as R9.

**Live validation.** Capped smoke: `timing.per_step_overhead_s.median ≤ 0.06 s` and no growth between first and last quintile; sweep: per-step overhead per task family.

**Expected effect (report/verify estimates).** Per-step non-model cost from 0.28–0.40 s to about 0.04–0.07 s; T04 −117…−131 s (402 → about 275 s wall); T08 about −200 s; drawer −700…−820 s; overhead no longer grows with episode length.

**Risks.** A code path that reads images on non-decision steps (macro-era code) → `StaleFrame` or a replay mismatch makes it visible; `/dev/shm` leaks after crashes → the runner and sweep runner sweep stale `uc72_*` files whose owning PID is gone; CPU contention on a loaded host (load 9–50 observed) shifts absolute numbers, so compare arms in the same window.

**Effort.** 4 engineer-days.

---

### R8. Video out of the loop

**Purpose.** A2 (inline planning video 0.12–0.20 s/step; 68 s of T04's 402 s), verify F1 correction 3 (post-hoc video needs re-simulation once per-step images leave the trace).

**Files (NB).** Create `scripts/uc72_render_video.py` (post-hoc), `src/gpt_field/uc72/video_recorder.py` (live raw recorder process), `tests/uc72/test_video_posthoc.py`. Reuse `agent_speedup_v2_user_check_video.py` drawing code unchanged.

**Modes (config `runtime.video`, D5).**
- `posthoc` (recommended for sweeps): after the episode, re-simulate with zero model calls from the task spec + `steps.jsonl` actions; assert state8 equality at every step (the replay property, `validation/replay_baseline.py:302-308`); render the 1600×960 three-panel planning dashboard from the trace's decision rows. Output `episode.mp4` + `video_verification.json {"steps", "state8_max_error", "verified": true}`; a mismatch marks the video `unverified` but never touches the scored result.
- `raw_live`: a separate, `nice`d process reading the frame ring and piping side-by-side raw views to x264; drops frames if behind (records `dropped_frames`); never blocks the loop.
- `off`.

**Veto/parameter.** None.

**Offline tests.** Post-hoc renderer on the saved T04 fixture: 549 steps re-simulated, state8 exact, video frame count = expected; recorder process killed mid-run leaves the episode unaffected. Pass: exact state8, no loop slowdown (per-step overhead unchanged within noise with `raw_live` on vs off in the replay).

**Live validation.** One sweep episode per family rendered post hoc and inspected by the user.

**Expected effect.** −0.12…−0.20 s/step of loop time (verified video cost, verify F1); video still available.

**Risks.** Post-hoc re-simulation is CPU-deterministic (verified bit-exact on T04); a perception or asynchronous component that ever wrote into the simulator would break it (none does).

**Effort.** 2 engineer-days.

---

### R9. Zero-call replay on decoded pixels; runner-level equivalence; overhead benchmark

**Purpose.** A2 caution ("the replay validator compares encoded RGB; switch it to decoded pixels"), R8 (tests cannot catch live blockers; H1's deterministic replay is the cheap check), §6.7, verify F1 correction 4 (byte-exact RGB makes any encoder change fail). It is the development aid every other workstream uses to prove a flag-off change is behaviour-neutral. Not a hard launch gate (rule 4).

**Files (NB).** Create `validation/build_pixel_fixture.py`, `validation/fixtures/t04_baseline_pixels.json`, `validation/replay_runner.py`, `tests/uc72/test_replay_compare.py`. Keep `validation/replay_baseline.py` unchanged (it remains the H1-identity check).

**Algorithm.**
1. `build_pixel_fixture.py` runs the existing H1 replay through the **pristine worktree's code** (`PYTHONPATH=arms/h1-baseline-3cc84ac/src`, saved responses, network blocked) and records, for the reset and each of the 549 steps, `sha256` of the decoded external and wrist arrays (+shape), and for each of the 13 model requests the decoded-pixel hashes of every `input_image` (image URLs decoded, not hashed as bytes). ≈100 s, zero model calls.
2. `replay_runner.py --config config/arms/h1_equivalent.json` drives the **new runner** (R2) with a `SavedTransport` (as `validation/replay_baseline.py:227-255`) and compares: actions (exact), state8 (exact), raw-success flags (exact), contact class counters (exact), decoded-pixel hashes of every observation and every model image (exact), request text except the documented measured timers (as `normalize_planning_timing`, `:77-129`). Protocol-changing flags must be off for the text comparison; actions/state/pixels are compared for any flag set that keeps the saved responses valid.
3. Overhead benchmark (same run, no model latency): per-step wall split from `steps.jsonl` and worker timers; report median, p95, quintile medians.

**Pass criteria.** Equivalence: all exact for 549/549 steps and 13/13 requests. Overhead (development target, not a gate): median per-step wall ≤ 0.06 s and p95 ≤ 0.10 s; last-quintile median ≤ 1.10 × first-quintile median. Reference "before": H1 live T04 0.158 s/step without video, 0.282 with video (verified); H1's own replay took 97.5 s for 549 steps incl. reset (`validation/results/combined_replay_005.json`).

**Veto/parameter.** Removes the PNG-byte equality requirement (replaced by pixel equality); the overhead numbers are targets reported, not thresholds that change behaviour.

**Offline tests.** The replay itself; plus unit tests that the comparer flags a single changed pixel, a changed action bit, and a changed contact counter.

**Live validation.** n/a (offline by design); each build tag records its replay result.

**Expected effect.** A zero-cost check that A2 (and any flags-off refactor by other workstreams) keeps behaviour; a reproducible per-step overhead number per build.

**Risks.** If ws_protocol_context's schema rejects H1's 13 saved responses, the replay cannot run with protocol flags on (shared interface: keep the H1 response shape valid when protocol flags are off).

**Effort.** 2 engineer-days.

---

### R10. Transport robustness (A3)

**Purpose.** A3, H1 item 16, F10 (verify: stronger than stated): 401 `token_expired` killed drawer 002 at the plan call; HTTP 520 killed the uc7.1 t8 relayout's first execution request; astranomy 001's 503 `unknown_error` would also have ended an H1 episode (`agent_speedup_v2_client.py:910`).

**Files (NB).** Modify `src/gpt_field/agent_speedup_v2_client.py` (`_retry_cause` `:903-914`, loop `:916-960`), `src/gpt_field/agent_speedup_v2_user_check_agent_client.py` (loop `:446-502`), `src/gpt_field/chatgpt_auth.py` (`access_token(force_refresh=False)`, `:450-480`). Create `tests/uc72/test_transport_faults.py`.

**Classification (`_retry_cause(exc, initial_count, retries) -> str | None`).** Precondition unchanged and essential: no response envelope (`self.last_response is None`) and no committed observation (`_observation_count` unchanged), so a schema-valid answer is never sent twice and nothing is committed.

| class | condition | action | budget |
|---|---|---|---|
| `auth` | `http_status == 401` (any code) | `token_provider.force_refresh()` then resend immediately | 1 per request; a second 401 → `MissingCredentials` → infra `auth` (sweep pauses for re-login) |
| `overload` | 503 + `server_is_overloaded` (as today) | backoff 2^k + jitter, honour Retry-After | 3 attempts (inherited) |
| `transient` | the 4 stream codes (as today) **plus** HTTP 500/502/503/504/520–524 with any code (incl. `unknown_error`, `server_error`), and stream `response.failed`/`error` events with `server_error`/`unknown_error` | same request, same observation, backoff | 2 retries (inherited) |
| `deadline` | `request_deadline_exceeded` (SIGALRM, `transport.py:116-117`) | resend | counted with `transient` |
| `quota` | 429 (`rate_limit_exceeded`, `usage_limit_reached`, `insufficient_quota`) | wait Retry-After if ≤ 60 s (inherited max wait); else raise `QuotaPause` | runner pauses the episode (simulation is already paused during inference), polls a canary with doubling waits up to `quota.max_pause_s`, resends the same request; wait recorded as `infra_wait_s` (D9) |
| none | 400/403/404, model mismatch, `SchemaError` (existing re-ask path) | raise | — |

```python
# chatgpt_auth.py (NB)
class SubscriptionCredentials:
    def access_token(self, *, force_refresh: bool = False) -> str     # force skips the 60 s freshness shortcut
    def force_refresh(self) -> str                                     # = access_token(force_refresh=True)
    def ensure_fresh(self, min_remaining_s: float) -> None             # launch-time check, min_remaining = cap_wall_s
```

Every retry writes `<request id>.<cause>_retry.json` (existing pattern, `agent_client.py:474-480`) and an `attempts[]` entry in `calls.jsonl` (R6). Infra summary: `{"auth_refreshes", "transient_retries": {code: n}, "deadline_retries", "quota_waits_s", "final_error": null | {"class", "http_status", "error_code"}}`.

**Veto/parameter.** Removes "end the episode on a recoverable error". Adds no new numbers (budgets inherited; the quota pause bound is D9). The launch-time freshness horizon is derived (`cap_wall_s`).

**Offline tests (fault injection via the existing `transport=` seam).** (a) 401 once → refresh called once, request resent, history committed once, one accepted call; (b) 401 twice → `MissingCredentials`, nothing committed; (c) 520 `unknown_error` once → retried, same payload bytes; (d) deadline once → retried; (e) 503 `unknown_error` three times → raised after 2 retries; (f) a response envelope received then a failure (e.g. model mismatch) → never retried; (g) 429 with Retry-After 5 → waited and resent; Retry-After 3600 → `QuotaPause`. Pass: all, and payload bytes of each resend identical to the first attempt.

**Live validation.** None forced (no live 401 injection); the sweep reports the infra block per episode and the count of episodes saved by a retry.

**Expected effect.** Removes the infrastructure losses seen (drawer 002, uc7.1 t8 relayout, astranomy 001-type 503) from the policy failure rate (estimate: 2–3 of about 14 launches in the analysed window).

**Risks.** Retries add hidden latency; it is excluded from policy time and reported. A persistent outage could loop: budgets are bounded; the sweep runner pauses after consecutive infra failures (R11).

**Effort.** 1.5 engineer-days.

---

### R11. Sweep harness: plan, interleaving, stop/censor/relaunch, coverage map, cost

**Purpose.** User rule 3 (all-task sweep on one frozen build, coverage map), report §6.3–6.6 (same-window interleaving, repeats, frozen config, pre-registered decision rule), verify_cross §8 corrections (power overstated, cost underestimated 1.5–2×, no interleaving, release still evaluator-assisted, taxonomy decisions, no cross-goal rule).

**Files (NB).** Create `scripts/uc72_sweep.py` (`plan | run | status | report`), `src/gpt_field/uc72/sweep/{plan.py,run.py,report.py,adapters.py,coverage.py}`, `sweeps/specs/*.json`, `tests/uc72/test_sweep_plan.py`, `tests/uc72/test_coverage_report.py`.

**Sweep spec.**

```json
{"schema": "uc72-sweep-spec-1", "sweep_id": "s2_measured_baseline_001",
 "arms": [
   {"arm_id": "h1-baseline-3cc84ac:pristine", "kind": "h1_frozen", "worktree": "arms/h1-baseline-3cc84ac",
    "runner_flags": ["--auto-approve", "--reasoning-effort", "high", "--request-deadline-s", "180", "--stream-progress", "--no-video"],
    "env": {"GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M": "0.10"},
    "external_caps": {"exec_calls": 80, "wall_s": 5400}},
   {"arm_id": "uc72-b001:measured_h1", "kind": "uc72", "worktree": "arms/uc72-b001", "config": "config/arms/measured_h1.json"}],
 "tasks": {"pack": "tasks", "families": ["official", "obstacle_variant", "composite", "relayout"]},
 "repeats": 1, "order": {"scheme": "blocked_random_counterbalanced", "seed": 20261108},
 "concurrency": {"mode": "sequential | paired_concurrent", "gpus_egl_minor": [2, 3]},
 "canary": true,
 "stop_rules": {"pause_after_consecutive_infra": 3, "max_relaunches_per_cell": 1, "quota": "pause"},
 "budget": {"max_calls": 3000, "max_wall_h": 30}}
```

**Algorithm.**
- `plan`: expands arms × tasks × repeats into cells; groups each (task, repeat) into a **block** run back-to-back (or concurrently, D6); randomises block order and the arm order inside blocks with the recorded seed, counterbalancing (ABBA) across consecutive blocks; writes `sweep_plan.json` with the cost estimate (below) and each arm's build digest. Arms that cannot run a task (the pristine H1 cannot run changed-sentence/changed-goal tasks: stove-off T08, composite) get `n/a` cells.
- `run`: before each episode, re-verify the arm's worktree digest (abort the sweep on mismatch: frozen-build integrity), check GPU free memory if the arm uses perception (GPUs are shared with training jobs), pin `MUJOCO_EGL_DEVICE_ID` to the device **minor number** (the minors are reversed versus `nvidia-smi` on this host), launch a fresh output directory per episode, and keep resident services (perception) alive across that arm's episodes (`reset_episode` per episode; saves the 23.5–24.6 s preload per episode H3 paid). For the pristine H1 arm the frozen runner is invoked directly with its own flags (custom layouts via `--initial-state-override`/`--task-bddl-override`/`--expected-initial-observation-sha256`, which H1 already supports for official-goal scenes) and capped externally by touching its operator stop file at the call or wall cap (no H1 code change).
- Stop/censor/relaunch: terminal finish/give_up → scored; cap → scored, censored; user stop/operator cap → unscored; `infra_error` (transport after retries, simulator crash, required service crash) → one immediate relaunch in a fresh directory inside the same block, both kept, the cell takes the relaunch's outcome, both count toward cost and launches-per-success; a second infra failure → cell `I` (unresolved); auth infra or 3 consecutive infra failures → pause the whole sweep and alert (re-login or outage); quota → pause all arms until reset (keeps arms in the same window); `harness_exception` → scored failure `E`, flagged, no relaunch.
- `status`: progress, spent calls/hours, budget remaining, current pause reason.
- `report` (offline, no model calls): adapters map the pristine arm's `result_summary.json` and the new arm's `summary.json` into one cell record; outputs `coverage_map.{json,md,html}` + `report.md`.

**Coverage map.** Rows = tasks grouped by family (official LIBERO-10, LIBERO-Goal, obstacle variants, composite, relayouts held-out / historical, generated obstacles); columns = arms × repeats; cell glyph + `policy_time_s`, calls, forbidden/obstacle episodes. Glyphs: `S` strict success at finish, `R` raw-only success at finish, `F` finished but failed, `G` gave up, `C` capped (censored), `I` infra unresolved, `E` harness exception, `U` user stop, `S*`/`R*` oracle-stopped (pristine arm: raw-success stop; strict from the evaluator's release-and-settle). Footer per arm: success@finish strict and raw, collision-free success, P(strict ≤ 180/300/600 s), median policy time, calls per success, give_up/cap counts, infra relaunches, median `service_speed_index`, reset-identity status. Two comparison tables are never mixed: (1) **honest metric** (new-build arms only; finish-scored), (2) **oracle metric** (all arms; first raw success, calls and policy time to first raw success, contact episodes up to it).

**Decision rule (pre-registered, D7).** Lexicographic: (1) zero-shot gate (structural: digest shows no per-task inputs, same build for all cells); (2) collision-free strict success@finish, paired by (task, repeat), McNemar exact; (3) geometric-mean ratio of policy time over pairs where both succeeded, bootstrap 95 % CI over tasks; (4) calls per success. Differences the design cannot resolve are labelled "unresolved", not "tie".

**Power (offline Monte Carlo, `ws_runner_scripts/power.py`; paired McNemar, α 0.05, logit scene heterogeneity σ ∈ {0, 1, 2}).**

| scenes × repeats (episodes per arm) | 10 pp | 20 pp | 30 pp |
|---|---|---|---|
| 14 × 3 (42) | 0.11–0.15 | 0.38–0.54 | 0.78–0.92 |
| 34 × 1 (34) | 0.08–0.11 | 0.31–0.45 | 0.67–0.82 |
| 34 × 2 (68) | 0.16–0.25 | 0.60–0.80 | 0.95–0.99 |
| 34 × 3 (102) | 0.25–0.35 | 0.80–0.94 | ≥0.99 |

The σ = 0 row for 14 × 3 (0.38) matches the upper end of the verified 26–38 % (verify_cross §8). So: one repeat over 34 scenes is a coverage map, not an A/B decision below about 30 pp; A/B decisions at 20 pp need 3 repeats.

**Cost model (`ws_runner_scripts/cost.py`, estimates ±30 %).** Assumptions from H1-family runs: finished episodes 18–20 calls and 6.5–7.5 min wall after A2; give_up failures 25–30 calls, 9–12 min; capped failures 80 calls, about 33 min; success at finish 50–75 %; the pristine arm has no give_up so all its failures run to the cap.

| scene set (1 repeat) | new-build arm | pristine H1 arm | both, serial | both, paired-concurrent |
|---|---|---|---|---|
| 34 scenes | 810–1,450 calls, 5.1–9.7 h | 1,000–1,500 calls (30 scenes), 6.6–10.1 h | 1,800–2,950 calls, 11.6–19.8 h | 6.6–10.1 h |
| 44 scenes (+10 generated obstacle relayouts) | 1,050–1,870 calls, 6.5–12.5 h | 1,340–2,000 calls (40), 8.8–13.5 h | 2,400–3,900 calls, 15.3–26 h | 8.8–13.5 h |

Plus about 1 canary call per block. Three repeats of a two-arm A/B on 34 scenes: about 5,400–8,800 calls and 35–60 h serial.

**Veto/parameter.** None in the policy. Sweep-level budgets (max calls, max hours) and repeats are user decisions.

**Offline tests.** (a) `plan` determinism for a seed; every block contains all runnable arms; counterbalancing holds; (b) relaunch logic on synthetic outcomes; (c) report on fixtures: the 7 historical uc7.1 runs that have `result_summary.json` (H1 T04, H1 T08, stove-off T08, drawer 001/003, T20 T04, official T04) pass through the adapter with each run's raw success and after-settle strict verdict exactly as recorded, and strict-at-terminal false in all 7; (d) digest-mismatch abort. Pass: all.

**Live validation.** S2 (measured baseline) itself, user-started; acceptance of the harness = complete coverage map with every cell in a terminal glyph and every infra failure explained.

**Expected effect.** Turns "never swept" into a measured zero-shot coverage map (F11); same-window pairing removes the service-day confound that produced H3's apparent 2× latency (C1); costs are known before launching.

**Risks.** Subscription concurrency limits for paired-concurrent mode; GPU availability (shared); long pauses; a frozen build defect discovered mid-sweep forces a new tag and a restart of that arm (by design).

**Effort.** 4 engineer-days.

---

## 3. Shared interfaces

| name | provider → consumer | proposed shape |
|---|---|---|
| `TerminalDecision` | ws_protocol_context → runner (R2/R4) | agent action types `finish` (terminal) and `give_up` with `{reason_class, reason}`; policy sets `Decision.diagnostics["terminal"] = {"kind","reason_class","reason","claim","decided_at_step","request_key"}` on the answering step; completion contract text "release and move clear before finish"; prompt contains no evaluator cue (`agent_client.py:205-209` removed) |
| `PlanStage` | ws_protocol_context → runner | `PlanStage.run(sim_handle, obs, timers) -> PlanRecord{"plan": dict, "view_frame": dict, "calls": [call ids], "draft_model_s": float, "reasks": int, "approved": bool, "issues": [...]}`; uses the episode's simulator; at most one re-ask (A4) inside; may be a no-op if the plan is folded into the first call |
| `PhaseLabel` | ws_executor_control → runner (R5/R6) | `policy.phase_label() -> {"mode": "fragment|direct|hold|terminal", "fragment": int|None, "segment": int|None, "skill": str|None, "profile": "transit|contact"|None, "declared_contact": None|"object"|"articulated"|"support", "stage_target_xyz": [3]|None, "routed": bool|None}` per step |
| `ExecutorEvents` | ws_executor_control, ws_swept_planner, ws_geometry_tools → runner | `recorder.event(kind, payload)` with kinds `guard_fired`, `blocked_motion`, `stall`, `grasp_predicate_failed`, `late_failure_detected`, `route_refused`, `escape_executed`, `swept_clearance`, `planner_time`, `ik_infeasible`; each `{"step", "t", ...}`; → `events.jsonl`, counted in the report (IC5, IC12, IC13) |
| `FrameRing` | runner → ws_perception_service (and recorder) | `/dev/shm/uc72_<run_id>_frames.bin` layout in R7; `FrameRing.open(path).latest() -> Frame{frame_id, step, sim_time_s, capture_monotonic_ns, state8, joints, geometry, rgb: {"external","wrist"}}`; consumers declare `min_frame_interval_steps` to the runner |
| `ServiceProtocol` | runner → ws_perception_service, ws_geometry_tools | `ServiceSpec`/`ServiceHandle` in R2; unix-socket JSON lines `{"id","op","payload"}` → `{"id","ok","result","error","timing"}`; mandatory ops `health`, `identity`, `reset_episode`, `shutdown`; `required` (fail-closed) or fail-open; resident across episodes |
| `Timers` | runner → all | `timers.block(kind)` for anything that freezes the robot; `timers.compute(kind)` for overlappable compute; `timers.async_done(kind, compute_s, frame_capture_ns, used_at_step)`; nothing else enters policy time |
| `ConfigRegistration` | runner → all | `components.<ws_key>` section + flags in the arm config; every numeric `{"value","source"}`; flags default off in `h1_equivalent.json`; the loader rejects unsourced numbers |
| `VendorManifest` | ws_geometry_tools, ws_perception_service → runner (R1) | `vendor/MANIFEST.json {"entries": [{"path", "sha256", "source_path", "source_sha256", "source_snapshot": "work/h2_keypose2_snapshot"}]}` + interpreter requirement + weights sha; enters the build digest |
| `ReplayCompatibility` | ws_protocol_context, ws_executor_control → runner (R9) | with their flags off, the H1 response shape (13 saved T04 responses) stays valid and the payload text is unchanged; with flags on, actions/state/pixels comparisons still run where responses remain valid |
| `ContactEpisodes` + replay harness | runner → ws_executor_control, ws_swept_planner | `metrics/contact_episodes.py` episode records (R5) incl. `onset_speed_m_s`, `onset_target_distance_m`; `validation/replay_runner.py` usable with any config |

## 4. Sequencing (respects IC1–IC15)

1. **M0, offline only (week 1).** R1 (clone, tags, config, digest) → R10 (fault-injection tests) → R9 step 1 (pixel fixture from the pristine worktree) → R2 skeleton + R7 + R8 under `h1_equivalent.json` until the runner-level replay is bit-exact → R3 importer + no-model reset preflight. No model calls.
2. **M1, measurement (week 2).** R4 (needs `TerminalDecision` from ws_protocol_context), R5, R6, R11 `plan/report` on historical fixtures. Freeze `uc72-b001` (`measured_h1.json`). User-run capped smokes (T04+can at 3 calls; LIBERO-Goal g8 at 20 calls).
3. **S2, measured baseline sweep (user-started).** `uc72-b001:measured_h1` vs `h1-baseline-3cc84ac:pristine`, 34 scenes, 1 repeat, interleaved. This realises IC6 ("finish-scoring first; every later speedup measured against finish-scored baselines") and A5's "then run a baseline all-task sweep".
4. **Later builds.** Other workstreams' capability flags land in `uc72-b00N`; each build gets the replay, a capped smoke, then a sweep against `uc72-b001:measured_h1` (3 repeats for decisions at 20 pp).

Notes on IC constraints: IC10 is met by the frame ring (never images only at decisions). IC14 by capture time stamps + action-start times. IC6 by ordering R4 before any capability sweep. IC1/IC2 need R5's onset statistics before ws_executor_control turns on distance-scheduled speed or cruise. IC5 needs `late_failure_detected` events counted in the sweep. IC12/IC13 refusal and escape counts come from `ExecutorEvents`. The cross_critic order lists "evaluator overhead moved off the loop" under step 5 (Speed); this plan does it in M0 because it is evaluator-only and proven behaviour-neutral by the bit-exact replay, and it saves about 2 minutes per T04 episode in every later sweep.

## 5. Commands the user runs (none are run by this design)

```bash
# one-time setup, no model calls
git clone /workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm /workspace/gpt_field/agent_speedup_v2_uc72_harness
cd /workspace/gpt_field/agent_speedup_v2_uc72_harness && git remote rename origin h1-local
.venv/bin/python scripts/uc72_build.py freeze --tag uc72-b001          # after implementation; refuses a dirty tree
.venv/bin/python scripts/uc72_build.py worktrees --tags h1-baseline-3cc84ac uc72-b001
.venv/bin/python scripts/uc72_preflight_resets.py --tasks all --arm-root arms/uc72-b001 --gpu <egl_minor> \
    --output preflight/resets_uc72-b001.json
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<egl_minor> .venv/bin/python validation/replay_runner.py \
    --config config/arms/h1_equivalent.json --output validation/results/replay_uc72-b001.json

# capped smokes (model calls; user-started)
arms/uc72-b001/run.sh --task-spec tasks/obstacle_variant/t04_can_obstacle/task.json \
    --config config/arms/measured_h1.json --max-exec-calls 3 --gpu <egl_minor>
arms/uc72-b001/run.sh --task-spec tasks/official/libero_goal_g8/task.json \
    --config config/arms/measured_h1.json --max-exec-calls 20 --gpu <egl_minor>

# measured-baseline sweep (user-started)
.venv/bin/python scripts/uc72_sweep.py plan --spec sweeps/specs/s2_measured_baseline.json --output sweeps/s2_001
.venv/bin/python scripts/uc72_sweep.py run --plan sweeps/s2_001/sweep_plan.json
.venv/bin/python scripts/uc72_sweep.py status --plan sweeps/s2_001/sweep_plan.json
.venv/bin/python scripts/uc72_sweep.py report --plan sweeps/s2_001/sweep_plan.json
```

## 6. User decisions

| id | question | options | recommendation |
|---|---|---|---|
| D1 | Where does the new build live? | (a) local clone at `/workspace/gpt_field/agent_speedup_v2_uc72_harness` + worktrees; H1 dir and `.git` untouched; (b) branch + worktrees inside the H1 repo; (c) directory copy | (a) |
| D2 | Evaluation caps | (a) 1,800 s policy time + 80 exec calls (no historical H1-family success censored); (b) 900 s + 60 (censors both drawer successes); (c) 180 s (fails every current run; keep only as P(≤180 s)) | (a), with P(≤180/300/600 s) reported |
| D3 | Score after a 10-step hold settle or at the finish instant? | (a) hold settle (gripper command kept) + report instant; (b) instant only | (a) |
| D4 | Contact taxonomy for "collision-free success" | (i) palm on current grasp target; (ii) non-finger push on an articulated part; (iii) resting contact with other goal objects; (iv) held-object support outside the goal region | (i) reported, not forbidden; (ii) forbidden unless the stage declared that contact; (iii) reported, not failing (obstacles/distractors failing); (iv) forbidden |
| D5 | Video during sweeps | (a) post-hoc re-simulated dashboard on demand; (b) live raw recorder process; (c) both | (a) |
| D6 | A/B execution mode | (a) sequential interleaved blocks; (b) paired concurrent on two GPUs (needs subscription concurrency) | (a) first; (b) once a smoke shows two concurrent streams are allowed |
| D7 | Repeats and decision rule | (a) acceptance coverage 1 repeat; A/B decisions 3 repeats; lexicographic rule zero-shot → collision-free strict success → time → calls; (b) 2 repeats everywhere | (a) |
| D8 | Scene set | (a) 34 scenes (20 official + 4 variants/composite + 10 relayouts, T00/T01/T06 flagged non-held-out, T02 generated); (b) + 10 seeded on-path obstacle relayouts (44 scenes; fraction [0.3, 0.7], offset ±3 cm) | (b) if budget allows; obstacle evidence is otherwise 3 scenes |
| D9 | Subscription quota during an episode | (a) pause and resume the same request (excluded from policy time); (b) fail as infra and relaunch | (a) |
| D10 | Official reset hashes that mismatch only in PNG bytes and have no stored observation | (a) adopt the new pixel hash after a user glance at the preflight images; (b) refuse those tasks | (a) |
| D11 | Latency canary per block | (a) yes (1 call per block); (b) no, rely on per-call service index | (a) |
| D12 | Baseline arms in S2 | (a) both pristine H1 (oracle metric, external caps, `--no-video`) and measured H1 (honest metric); (b) measured H1 only | (a) |
