# Feasibility review: uc7.2 integrated implementation plan

Reviewer key: `review_feasibility`. Date: 2026-11-08. Scope: `INTEGRATED_PLAN.md` (all 2,173 lines), checked against the H1 code at `3cc84ac` (`/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm`), the three interpreters, the K2 snapshot, `foundation_vision`, the T04 reference run and fixture, and the live host (GPUs, `/proc/driver/nvidia`). Read-only: nothing under `/workspace/gpt_field` was edited, and no simulator, model call or GPU job was run (one `nvidia-smi` status query only).

## 0. Verdict

The plan is buildable. Its code facts are accurate: every cited file:line I re-checked says what the plan says it does (table in §1). The architecture fits the interpreter constraints:
- the controller stays numpy-only;
- scipy is needed only offline, in `.venv-sim`, or in the service;
- the vendored closures are self-contained.

The plan's other hard problems are already identified: the SIGALRM deadline, `_history_lock`, PNG-encoder-dependent reset hashes, and the EGL minor reversal.

Three defects should be fixed before M0/M1 freeze, because they would silently corrupt results rather than fail loudly:
1. **The FrameRing v1 seqlock, as specified, accepts torn frames.** The stress test the plan gives cannot detect this. I-03 is frozen at M0.
2. **The pristine H1 arm's external caps are recorded by H1 as `user_stop`.** Under the plan's own rules that is unscored, so every pristine failure would drop out of the S1/S5 oracle table.
3. **Request 1 is built before the plan-bound perception snapshot can exist.** More generally, the model sees perception from at most frame k−1 at decision k. Two M3 pass signals depend on bindings at call 1, and the integrated plan never schedules the wait the perception design asked for.

The effort is optimistic by roughly 15–20 % (§5). The largest gaps are:
- the M0 runner rewrite;
- the M3 integration buffer;
- the M5 asynchronous client split over a 9-level client inheritance chain.

Several migration steps are missing (§6): credential store, pristine-arm invocation, `action.py` import order, `.gitignore` vs `--require-frozen`, vendor tree layout, and controller dependencies with symlinked venvs. Neither those nor the test gaps in §7 block the direction.

---

## 1. Plan facts re-checked (all hold unless noted)

| Plan claim (§1 and inline) | Checked | Status |
|---|---|---|
| H1 `3cc84ac`, clean; origin is the public GitHub repo | `git log/status/remote` | ✓ |
| Main tree has `A agent_speedup_v2_obstacle_harness_100mm` staged | `git status --porcelain` (10,227 lines) | ✓ |
| `FRAGMENT_VERSION`, caps 6/120/240, profiles 1.2/0.5 rad, stall 12/16/0.0008 | `fragments.py:23-46` | ✓ |
| Envelope veto, transit routing, stall, checkpoint, `"unknown"` cause | `fragments.py:288-291, 293-317, 404-419, 423-434, 482` | ✓ |
| Model-written `obstacle_map`; "Remove the object you will manipulate…" | `agent_client.py:92-105` (`:101-102`) | ✓ |
| Test-lift mandate, "Simulation pauses", oracle cues + no give_up, "Pixels specify rays", gripper facts | `agent_client.py:161-163, 182-183, 205-209, 224-225, 227-233` | ✓ |
| `_trim_images(…, IMAGE_HISTORY)` per call | `agent_client.py:417` | ✓ |
| `completion_rule` cues; `finish` = 1-step hold; fragment end hands back | `agent.py:128-130, 366-372, 394-400`; `framework.py:112` | ✓ |
| Base loop refuses `stop_on_success=False`; breaks on first raw success | `evaluate_agent_speedup_v2.py:213-214, 1440-1443` | ✓ |
| Transit cap env default `"0.02"`; OSC scale `max(0.05, cap)` | `action.py:20, 23` | ✓ |
| Joints dropped from reset identity; second `agent_geometry` IPC; validator every step | `user_check_sim.py:72-80, 91-96, 122-128` | ✓ |
| Ideal-pinhole validator (focal to 1e-10, centred principal point) | `astranomy_agentic_geometry.py:104-112` | ✓ |
| Retry policy covers only overload + 4 stream codes | `agent_speedup_v2_client.py:903-914` (constants `:139`, `:144`) | ✓ |
| Worker prompt = official benchmark language | `libero.py:837-838` | ✓ |
| `WINDOW_STEPS = 10` | `agent_speedup_v2_truth.py:68` | ✓ |
| `MUJOCO_EGL_DEVICE_ID=str(args.gpu)`; pops `OPENAI_API_KEY` (and `CUDA_VISIBLE_DEVICES`) | `launch.py:97-103` | ✓ |
| EGL minors reversed: 0→3, 1→2, 2→1, 3→0, 4→7, 5→6, 6→5, 7→4 | `/proc/driver/nvidia/gpus/*/information` | ✓ |
| GPUs shared; only GPU 7 idle, GPU 6 ≈ 15 GB free | `nvidia-smi` now: GPU 7 24,109 MiB free; GPU 6 15,109; GPUs 0–3 ≈ 8.4 GB; GPUs 4–5 739 MiB | ✓ (state unchanged) |
| Interpreters (controller: numpy 2.5.3, no scipy/torch; sim: py3.10, numpy 1.26.4, scipy 1.15.3; service: numpy 1.26.4, scipy 1.17.1, torch cu128) | `-I` imports | ✓. Also: controller Pillow uses zlib 1.3, the sim uses zlib-ng 2.3.3, the service Pillow 13.0 uses zlib 1.3.2. **`jsonschema` is absent in the controller and service venvs** (see SF-13). |
| Vendor pins `6d2746cb…`, `8fa39c48…`, `b15d07a1…`, `98eae4ec…`, `17a3d225…`, `d5fc4136…`, `263c0f9a…`, `d9d68123…` | `sha256sum` | ✓ |
| K2 `robot_geometry.py` imports scipy (`:16`) and `..robodojo_direct_sim` (`:18`); own STL/OBJ parser | snapshot `:16-36` | ✓. No trimesh, so `.venv-sim` can build the model. |
| `foundation_vision` vendor closure = `__init__`, `types`, `backends`, `geometry` | imports | ✓. `sam3` and `depth_anything_3` are **editable installs** from `artifacts/foundation_vision/vendor/` (see SF-16). |
| `prompt_cache_retention`, `max_output_tokens` rejected; `prompt_cache_key` allowed locally | `chatgpt_plan.py:37-41` (denylist), `:351`, `user_check_transport.py:418` | ✓. `previous_response_id` is also rejected, so the transcript is always re-sent; the plan assumes this. |
| Strict-schema features in I-14 (`anyOf`, `enum`, min/max, min/maxItems, min/maxLength, no `$ref`/`pattern`) | H1's live `agent_schema()` already uses all of them | ✓ supported by the subscription |

Additional facts the plan relies on implicitly. Each was verified:
- **Request images are encoded in the controller.** `encode_png_data_url(view.external_rgb_upright)` at `agent_client.py:559-563` builds them from decoded upright pixels. With the FrameRing, the h1_equivalent replay can therefore keep **byte-exact request-image data URLs** (same controller Pillow), not only decoded-pixel equality. Only the observation PNGs and the reset identity depend on the sim's zlib-ng encoder.
- **`rgb_upright = raw[::-1, :]` is correct.** The worker applies `[::-1, ::-1]` (`observation.py:31-32`) and `StepView` applies `[:, ::-1]` (`contact_skills.py:283`). PNG is lossless, so the result is bit-exact.
- **Worker IPC is JSON lines over pipes** (`libero.py:598-637`, 16 MB reply limit, 120 s timeout). The replay's `block_network()` patch (`replay_baseline.py:174-179`) therefore does not affect it. It *would* block a unix-socket perception client, which is fine with perception off or recorded.
- **JSON floats are not rounded on the IPC path** (`jsonable`, `truth.py:110-128`). Storing state8/q7 as f64 in the ring reproduces the request values exactly.

---

## 2. Must fix

### MF-1. FrameRing v1 seqlock: the specified read order accepts torn frames, and the planned test cannot catch it

**Evidence.** The I-03 slot layout puts `seq_begin` at offset 0 and `seq_end` at **offset 232, before the payload**. The payload is `geometry_json` at 256 and pixels at 8448. The reader rule is "copy the slot, then accept only if seq_begin == seq_end == s". A copy runs low → high in address. So the reader:
1. reads `seq_begin` (old s);
2. reads `seq_end` (old s);
3. only then copies 1.5 MB of pixels, which a writer that has just started frame s+16 may be overwriting (`seq_begin = s'` then payload).

Both sequence numbers match, and the torn frame is **accepted**. M0 offline check 3 ("10⁴ frames give no torn reads") is a stochastic stress test. The race window is one ~1 ms copy against a slot reuse every 16 frames, so the test will pass while the defect remains. The defect bites exactly in the plan's riskiest consumers:
- the perception service copying event frames while busy (1–2 s refreshes);
- M5's paced, overlapping execution.

I-03 is frozen at M0 and has three consumers (perception, recorder, lazy frames).

**Fix (cheap now, costly after M0).** Specify the protocol explicitly.

- **Writer:**
  1. `seq_begin = s`;
  2. payload;
  3. `seq_end = s`;
  4. header `latest_frame_id`.
- **Reader:**
  1. read `seq_end` **first**;
  2. copy the payload;
  3. read `seq_begin` **last**;
  4. accept iff they are equal and equal the expected s for `frame_id`.

  On x86 TSO this is correct with plain numpy stores and loads.

Equivalently, move `seq_end` to the last 8 bytes of the slot and read it after the copy.

Replace the stochastic test with deterministic ones:
- a writer hook that pauses mid-pixel-write while a reader copies (must reject);
- a reader paused between payload and the last sequence read while the writer completes a full rewrite (must reject).

Keep the 10⁴-frame run as a smoke test only.

### MF-2. Pristine H1 arm: external caps become `user_stop`, which the plan scores as unscored. All pristine failures would vanish.

**Evidence.**
- The plan caps the pristine arm "externally through its operator stop file" (§2.7 item 1, §7.2).
- H1 records a stop-file stop as `status = "user_stop"` (`evaluate_agent_speedup_v2.py:1444-1445`, and `:1453` for the step cap).
- The plan's own rules score `user stop / operator cap` as **unscored** (§7.3; I-18).
- H1 has no `give_up` and no cap. The runner design's cost model states "the pristine arm has no give_up so all its failures run to the cap".
- Applied literally, the S1 (and S5) oracle-metric table would therefore contain the pristine arm's successes only.

**Fix.**
- The sweep runner writes its own cell record `{stop_reason: cap_exec_calls|cap_wall, at_calls, at_wall_s}` **before** it touches H1's stop file. It counts calls from `trial/requests/*.request.json` or H1's `run_state.json`.
- The pristine adapter maps H1 `user_stop` plus a runner-recorded cap reason to `cap_*` (scored, censored, glyph `C`). A genuine user stop stays unscored.
- The adapter computes the pristine arm's policy time post hoc as steps·dt + Σ request latencies from `requests/*.result.json`. H1 has no virtual clock, and the oracle table reports "policy time to first raw success".
- Add an adapter test on a synthetic H1 run directory for each of the three cases.

### MF-3. Perception ordering: request 1 cannot contain plan-bound instances, and the model's scene lags by one decision

**Evidence.**
- §3.2 Reset: `set_plan(objects)` "binds roles from cells on the cached features" after the 14–52 s plan call. No wait is specified before the first executive request is built.
- The perception design's own timeline (`ws_perception_service.md` §2.2) adds the noun prompts *when `set_plan` arrives*: 0.24–0.56 s plus association (est.). Its smoke criterion is "frame-0 snapshot before the first executive request".
- §3.2 Per decision: `before_model_call` requests the refresh, `after_model_call` waits 0 s after the call, and "context.scene and the overlays are built **for the next request**". So request k carries perception from at most frame k−1 (plus track updates and robot events).
- After a place, push or knock that emits no `released` event, overlays drawn "on the current pair" (GT7) can show a box where the object used to be.
- Several M3 live pass signals and S3 metrics depend on the first call: "bindings correct", "the first grasp uses a `target`", "can/bottle appear as o-ids at call 1", "identity-only calls = 0".

**Fix.**
1. Before request 1 only, a bounded `wait_for(plan-bound snapshot)` charged to `Timers.block("perception")`. Its timeout comes from the service's measured `latency_model` in `hello`, so no new constant is introduced. On timeout the request goes out with `unbound_plan_parts` filled.
2. For later requests, choose one and record it in the frozen config:
   - (a) a bounded wait for the decision-frame refresh (≈ 1.0–1.9 s, est., blocking) — the safe default for M3/M4;
   - (b) keep the 0 s wait, but stamp `context.scene.observed_step` versus the request step, and never draw overlay boxes for records not re-observed since the last robot event or contact on them.
3. Log `scene_snapshot_frame` and `request_frame` per call in `calls.jsonl`.
4. Add a `RecordedPerceptionClient` test on the T04 place step that asserts the overlay does not show the pre-move box.

---

## 3. Should fix

| ID | Where | Issue | Evidence | Fix |
|---|---|---|---|---|
| SF-1 | §2.7 item 1, §7.2 (pristine arm) | The plan says the pristine arm runs "with H1's own `run.sh`". `run.sh` → `launch.py` supports only `t04_can_obstacle` plus registry keys. It passes `--gpu` verbatim as the EGL minor and defaults the store to `ROOT/.private/chatgpt_plan`, which does not exist in H1. Via `run.sh`, `t08_bottle_obstacle` and the 10 relayouts would be unrunnable (not just stove-off and the composite), and an nvidia-smi index would select the wrong GPU (index 7 → minor 7 = GPU 4, 739 MiB free). | `launch.py:20-38, 77-78, 97-103, 120-122`. The H1 evaluator itself accepts `--initial-state-override`, `--task-bddl-override`, `--store-dir`, `--expected-initial-observation-sha256` (`evaluate_agent_speedup_v2_user_check.py` argparse). | Adopt R11's design text in the integrated plan. Invoke `scripts/evaluate_agent_speedup_v2_user_check.py start` directly, with `launch.py`'s environment (PYTHONPATH, transit 0.10, `MUJOCO_GL=egl`, EGL **minor** from `uc72/gpu.py`, `LIBERO_CONFIG_PATH`; pop `OPENAI_API_KEY` and `CUDA_VISIBLE_DEVICES`) and `--store-dir`. Pass the PNG hash produced by the pristine-code preflight. Enumerate `n/a` cells exactly (stove-off T08 and the composite only). |
| SF-2 | §2.1, all run commands | Missing migration step for the credential store. Clones and worktrees have no `.private/`. The real store is `/workspace/gpt_field/.private/chatgpt_plan`. None of the plan's commands pass `--store-dir`, so every live smoke would refuse at start. | `launch.py:77-78` default, refusal at `:120-122`; `ls` of H1 (no `.private`) and of the main-tree store | Add a required `--store-dir` (a pointer, recorded as path only in `build.json`) to `uc72_run`/`uc72_sweep`. `uc72_sweep.py plan` runs `TokenStore(store).status()`. Document that worktrees never hold credentials. |
| SF-3 | R1 (`action.py` "raises without config") | About 25 modules import `action.py` at import time and copy constants: `tracking.py:10`, `contact_skills.py:21`, `fragments.py:18`, `agent_client.py:19` (prompt text built at import, `:234`), and the **py3.10 worker** `user_check_sim.py:18`. Collection of H1's 34 tests, the existing `replay_baseline.py` (which pins only the env var, `initialize_runtime` `:159-170`, env at `:163`), and any service that touches a shared module would crash. R1 lands before R9's new replay exists, leaving M0 without an equivalence check. | `grep "from .action import"` (25 importers) | Keep the env var as the carrier, but have the runner set it *from* the config and make both processes verify equality with the config value (R1 test b). Raise only inside the uc72 runner when no config is given. Or land the hard failure only after `replay_runner.py` passes. Port or retire H1's `tests/` explicitly. |
| SF-4 | §3.4, §7.2, UD-22 | The per-episode "≥ 10 GB free" check measures nvidia-smi free memory, which drops by the resident service's own allocation. SAM weights (3.3 GB), DA3, the CUDA context and the caching allocator hold several GB (peak 5.98 GB for SAM, est. 7–9 GB in total). On GPU 6 (15.1 GB free now) later episodes would be refused. S4 (b004 vs b003) needs **two** resident services. A 35–60 h S5 run has no leak guard. | `nvidia-smi` now; §3.4 text; §7.2 "checks GPU free memory for perception arms" | Check headroom through the service: `torch.cuda.mem_get_info()` plus reserved-but-unused, reported in `health`. Pin service placement per sweep (two services on GPU 7 fit at ≈ 2 × 8 GB, est.). Unset `CUDA_VISIBLE_DEVICES` for the service. Log `memory_allocated` per episode, call `empty_cache()` at `reset_episode`, and restart between blocks if allocation grows. |
| SF-5 | §2.7 item 2, R9 (equivalence anchor) | The only bit-exact anchor is T04, whose 13 responses are **all `execute`**: 14 `move_to_pose/transit`, 8 `approach/contact`, 3 `set_gripper/contact`. There is no `motion`, `observe`, `finish`, `move_relative`, `wait`, SchemaError re-ask or transport retry, yet A1/I-18 build on the `finish` path. Nothing pins M2–M4 behaviour once later milestones refactor shared code (StagePipeline, fragments v5). | Counted from `validation/fixtures/t04_baseline.json` | Add a **differential zero-call replay**: scripted `SavedTransport` answers covering finish/observe/motion/move_relative/wait/re-ask, run through the pristine worktree's code and the new runner under `h1_equivalent`, comparing actions, states and payloads. Freeze each milestone's capped smokes (saved responses plus actions) as golden fixtures, replayed on later tags under that milestone's config. |
| SF-6 | PC1 row (M1), §2.8 | PC1 "deletes `agent_client.py:205-209`, `agent.py:128-130`, `:151`, `:231` and `framework.py:112`". This contradicts §2.8 ("prompt-6 text … byte-identical") and the replay's contract checks (`executive_instructions_exact`, `executive_schema_exact`, `fragment_capabilities_exact`; `replay_baseline.py:207-213`). Module-level constants cannot be flag-gated in place: `AGENT_INSTRUCTIONS` is built at import (`agent_client.py:52-234`), and `MAX_*` in `fragments.py:23-26` feeds `execution_schema()`/`fragment_capabilities()`, which are sent every call. Five engineers editing `agent.py` (655 lines), `fragments.py` (483) and `agent_client.py` (566) will conflict. | Code as cited | In M0, add one set of no-op seams to these files: policy hooks (bridge, resolver, scheduler), an executor stage hook, and a prompt/schema/capabilities provider object chosen at policy construction. Prove them inert with the replay. Afterwards all behaviour goes into new modules only. Name one owner per shared H1 file (suggest INT-2's owner). |
| SF-7 | R2/R3 (terminal actions) | With finish-scoring the worker must not terminate at first raw success. `stop_on_success=True` sets `terminated` (`libero.py:898, 944`), and `SimProcess.step` then raises "inactive episode" (`libero.py:656-657`). The registry hard-codes `stop_on_success=True` (`agent_speedup_v2_tasks.py:151`). The integrated plan never states the worker setting. | Code as cited | Set `LiberoConfig.stop_on_success = not runner.terminal_actions` explicitly in R2/R3. Test that a scripted policy steps past first raw success without `SimulatorError`, and that `settle_hold` → `truth` → counterfactual still run afterwards. |
| SF-8 | PC2/PC6 (M5 async) | Threading couplings beyond those already listed: (1) H1's client is a 9-level inheritance chain (`AgentFirstV2Client` → … → `AstraClient`) with per-request instance state (`_request_kind`, `last_request_key`, `last_log_path`, `last_response`, `_observation_count`). (2) `act()` calls `client.history_stats()` every step (`agent.py:574`), which takes `_history_lock` (`v2_client.py:731`) while `predict` holds it for the whole call (`agent_client.py:451`). (3) SIGTERM or the stop file during a flight: `concurrent.futures` workers are joined at interpreter exit, so finalisation can hang up to the 180 s deadline. | Code as cited; transport SIGALRM `user_check_transport.py:104-128` | Raise PC6 to 7–8 d. Return per-call metadata in `CallResult` instead of instance attributes. Make the in-flight call cancellable: close the socket, then `shutdown(cancel_futures=True)`. Test that SIGTERM during a 30 s fake stream finalises within 5 s. Add a loopback HTTPS/SSE server test (stdlib `ssl` + `http.server`) for the worker-thread deadline, beyond the fake-stream T2.5. |
| SF-9 | §8 staffing, INT-5 | INT-5 (truth exporter) sits with E1, the most loaded engineer (40.5 d, M0/M1 first), and is scheduled in M3. PS-12, GT11 and SP12, and report §5.2 item 7 ("benchmark before perception changes are consumed"), need truth while PS-1…PS-9 are developed (weeks 1–6). | §8 table; §6 M3 effort row | Move INT-5 to E4 or E5 in weeks 1–3, or into M0's offline tools; it needs only `.venv-sim` and saved runs. |
| SF-10 | §6, §12 (effort) | Effort is optimistic; see §5. M0 R2 (3 d) re-implements a 1,727 + 970-line runner pair bit-exactly with plan-stage reuse. The M3 integration buffer (2.5 d) is thin for three tracks meeting a live service. M5's PC6 is underestimated (SF-8). | §5 below | Re-baseline to ≈ 200 d (≈ +15–20 %, est.) with a critical path of ≈ 65–72 working days. Start M0 with a 2-day "equivalence spike": FrameObservation + one-sim + replay on T04 through the new runner skeleton. |
| SF-11 | §2.1 `.gitignore`, `--require-frozen` | Commands write into the repo or worktrees: `bench_truth/…`, `validation/geometry_bench/{cache,results}/…`, `sweeps/s*_00N/`, `preflight/…`. Only `.venv .venv-sim .local/ arms/` are added; H1 ignores `validation/results/` but not these. A dirty tree makes `--require-frozen` abort every later episode in that worktree. | §2.1 snippet; M3/M4 commands | Put all outputs under a root outside the repo (e.g. `/workspace/gpt_field/runs/uc72/…`), add ignore patterns, and compute `dirty` with `git status --porcelain --untracked-files=no`, keeping untracked files as a separate warning. |
| SF-12 | I-10, INT-1 vendor layout | The flat `vendor/panda_robosuite_1_4_1/(robot.xml, panda_gripper.xml, meshes)` breaks both byte-identical consumers. `PandaKinematics.from_xml` derives the gripper as `robot_xml.parents[2]/"grippers/panda_gripper.xml"` (`robodojo_direct_sim.py:136`). K2's `_load` resolves meshes relative to the XML (`robot_geometry.py:182-185`). The XMLs use `meshes/…`, `obj_meshes/…`, `meshes/panda_gripper/…`, `meshes/rethink_mount/…`. Referenced meshes total 41.6 MB. | XML greps; `du` | Mirror robosuite's `models/assets/{robots/panda, grippers, mounts}` tree under `vendor/` and always pass explicit paths. Budget about 42 MB per worktree checkout. Save the body-model npz without object arrays (`allow_pickle=False`) and load-test it in numpy 1.26 and 2.5. |
| SF-13 | Controller dependencies vs symlinked H1 venvs | The plan specifies JSON-Schema documents (snapshot, segment result, config, task) and lints. `jsonschema` exists only in `.venv-sim`, not in the controller or service venvs. Any package installed into H1's `.venv` (symlinked by every arm) changes the **pristine** arm's environment and fingerprint. | `-I` import checks; §2.1 symlinks | State "no new controller dependencies": hand-written validators in the style of H1's `_conform`, with schema documents validated in tests under `.venv-sim`. If a dependency is unavoidable, build a separate, non-symlinked uc72 controller venv that enters the digest. |
| SF-14 | §7.5 S5 cost | §6 M5 includes `h1-baseline-3cc84ac:pristine` in S5, but the §7.5 S5 row counts only b005 + b001. 34 × 3 pristine episodes add ≈ 2,700–4,400 calls and ≈ 17–30 h serial (est., using R11's per-episode figures). | §6 M5 vs §7.5 | Either drop the pristine arm from S5 (S1 already measured it on the same scenes) or add its cost. Recommend dropping it, or keeping it at 1 repeat. |
| SF-15 | I-03 / PS-1 frame copy-on-receipt | With pinned slots removed (X5), event frames survive only if the service copies within 16 frames: ≈ 0.8 s at M5's 20 Hz pacing, ≈ 0.6–1.1 s at the M0 target 0.04–0.07 s/step. The receiver thread competes for the GIL with Python post-processing during a 1–1.9 s refresh. A miss loses closure/release frames (held confirmation) as `StaleFrame`. | §4 I-03 text; PS-2.2 timeline | The controller copies event and decision frames into a small per-episode event store (e.g. `/dev/shm/uc72_<run>_events/<frame_id>.npy`, deleted at episode end), or the writer honours pin bits set by the controller. Test with a service stub whose reader is delayed 2 s. |
| SF-16 | §2.3 digest, I-04 `hello` | (1) Hashing the 3.3 GB SAM checkpoint plus DA3 at every episode start costs ≈ 5–10 s per episode (est.). (2) `sam3` and `depth_anything_3` are editable installs from `artifacts/foundation_vision/vendor/` (main tree), so `packages_sha256` will not see code edits there. | `.pth` files in the service venv; `backends.py` `_file_sha256` | Take weight hashes from `hello` (computed once at load) and cache by (path, size, mtime). Add the editable source trees' git commit or tree hash to `hello.code_sha256` and the digest. |
| SF-17 | §2.4 flag registry, §3.5 time model | Small spec inconsistencies that will trip the loader or the metrics. (1) `executor.router` takes `point_v1/point_v2`, but the M4/M5 columns say "(body)", and INV-4 reads it. (2) The plan call happens before step 0, and §3.5 never says it is wrapped in `Timers.block("model")`; the report's numbers are "control + plan", and the UD-15 fold A/B needs it. (3) The `flags` bit for "decision" frames cannot be known by the worker at write time. | §2.4 table; §3.5; I-03 flags | (1) Derive the router from `swept.mode`, as `clearance_oracle` is, or add `body` to the enum. (2) State that the plan call is model idle time inside policy time. (3) Set decision and event marks controller-side (SF-15 store) and drop the slot bit. |

---

## 4. Hidden-coupling checks requested

| Area | Finding |
|---|---|
| **Replay validator byte-exact checks** | Feasible. The current validator drives `policy.act` / `worker.step` directly (`replay_baseline.py:296-322`), not the base loop. R9's runner-level replay is new work. It requires (`:253`, `:302`, `:308`, `:310`, `:312`, `:327-328`): exact actions and state8; exact contact-count dicts (so R5 subclasses must live in a separate field, as `contacts_action` does); request payloads equal except images and the four `routing.planning_s` paths; and serialized observations equal. **Request images can remain byte-exact** (controller-side encoding, §1). Observation PNGs cannot, which is why the pixel fixture is needed (from the pristine worktree, simulator, zero calls; UD-02). `FrameObservation` must provide: `to_dict()` with exactly 4 keys (`check_observation` → `PolicyObservation.from_dict` decodes PNGs at every decision, `v2_client.py:707-709`); `joint_position`/`joint_velocity` (sent in requests, `agent.py:222-223`); and `__eq__` for `act()`'s `_expected_observation` check (`agent.py:498`). Coverage gap: SF-5. |
| **Reset identity hashing PNG data URLs** | Correctly handled by R3 (PNG hash via the bundled encoder at reset, plus the pixel hash). One addition: the pristine arm can only verify PNG hashes produced by *its* encoder. Every pristine cell therefore needs the hash from the pristine-code preflight; main-tree layout hashes may differ (UD-03). |
| **Sim IPC + shared memory across py3.10/py3.12** | `np.memmap` in `/dev/shm` is the right choice (no resource-tracker unlink). Aligned u64 stores and x86 TSO make a seqlock work, **but the specified read order is wrong (MF-1)**. Folding `agent_geometry` into the step reply is value-identical (same physics state, no rounding). The ring file is created by the controller before the worker starts, so `static_calibration_sha256` must be written by the worker after reset (I-03 header). |
| **Controller venv without scipy** | Holds. FK/IK (`robodojo_direct_sim.py`, numpy only), the sphere–OBB checker, and GT1's χ² p-values (closed form via `math.erf`) are numpy/stdlib. The robot-model build and K2 parity run in `.venv-sim` (scipy 1.15.3, no trimesh needed). The multi-view fit runs in the service. New risk: any extra dependency (SF-13). |
| **GPU memory for SAM/DA3 next to training** | Feasible only on GPU 7 today (24 GB free; GPUs 0–3 ≈ 8.4 GB, GPUs 4–5 0.7 GB). Two services in S4 need ≈ 16 GB (est.). The per-episode free-memory rule is wrong as written (SF-4). `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes `cuda:<index>` equal the nvidia-smi index. Unset `CUDA_VISIBLE_DEVICES`, as H1's launcher does. |
| **Subscription transport unsupported fields** | `prompt_cache_key` passes the local denylist (`chatgpt_plan.py:37-41`); whether the server honours it is correctly left to the M2 probe. `previous_response_id` is denied, so append-only means re-sending the full transcript. PC7 replays reuse **week-old encrypted reasoning items** from request logs (they are stored in full in `trial/requests/*.request.json`). If the server rejects them, the A/B arms must strip reasoning items equally, and the report should say so. |
| **Threading the loop for async calls** | The plan identifies SIGALRM (`user_check_transport.py:108-110`) and `_history_lock`. Not identified: per-request instance state across the 9-level chain, per-step `history_stats()` in `act()`, and cancellation on SIGTERM (SF-8). Sim IPC stays on the main thread (pipes, `selectors`), which is safe. The perception client's reader thread is safe with SIGALRM, because the handler runs on the main thread. |

---

## 5. Effort realism (engineer-days; my estimates are marked est.)

| M | Plan | Realistic (est.) | Why |
|---|---|---|---|
| M0 | 18.5 | 24–27 | R2 3 → 5: new runner preserving loop order, plan-stage reuse with ReviewStore/PlanLock, finalisation, services skeleton. R7 4 → 5.5: FrameObservation must satisfy `check_observation`, the `act()` equality check, `StepView`, joints and geometry. R9 2 → 3: pixel fixture via an external hook on the pristine worktree plus the runner-level SavedTransport harness. R3 3.5 → 5: composite and stove-off need ports of a 685-line main-tree patch, plus camera overrides, the ordered evaluator, the obstacle generator and the preflight. |
| M1 | 19.5 | 21–23 | R11 +1.5: pristine adapter (MF-2, SF-1). PC2 +1: retry and quota logic is duplicated in two `predict` overrides (`agent_client.py:446-502`, `v2_client.py:916-960`). |
| M2 | 36.5 | 38–41 | Flag-gating module-level constants in `fragments.py` and `agent_client.py` (SF-6) +2. EXC-INT scripted simulator benchmark +1. |
| M3 | 67.5 | 75–80 | INT-6 2.5 → 6: three tracks meet a live GPU service. MF-3 timing fix +1. PS-1 +1: priority queue, job isolation, OOM, recording. Vendor tree (SF-12) +0.5. |
| M4 | 18.0 | 19–21 | SP6 at 4 d is plausible given SP3/SP4 in M3. Add ≈ 1 d for the swap's replay and fixtures. |
| M5 | 13.0 | 16–18 | PC6 5 → 7–8 (SF-8). EXC-D1(b) 1 → 2: `resume`/`replace_remainder`/`next_is_barrier` are the core of committed sequences. |
| **Total** | **173** | **≈ 193–210** | Critical path ≈ 57 → ≈ 65–72 working days (est.), assuming S3 starts immediately at the M3 freeze with a free GPU and quota. |

The arithmetic inside the plan is internally consistent: milestone, owner and staffing sums all reconcile; I re-added every row.

---

## 6. Missing migration steps

1. **Credential store** for clones, worktrees and the pristine arm (SF-2, SF-1).
2. **Pristine-arm invocation:** evaluator rather than `run.sh`, EGL minor, overrides, preflight hash (SF-1).
3. **`action.py` cut-over:** sequence it after `replay_runner.py`; port or retire H1's `tests/` and `validation/replay_baseline.py` in the new build (SF-3).
4. **`.gitignore` and output roots** versus `--require-frozen` (SF-11).
5. **Vendor tree mirroring robosuite's asset layout;** npz without pickles (SF-12).
6. **Dependency policy** given the symlinked H1 venvs (SF-13).
7. **Editable `sam3`/DA3 source pins and weight-hash caching** (SF-16).
8. **Task-spec evaluator field for the chained v1 monitor.** H1 runs `drawer_bowl` (g3) and `bowl_plate` with `task_manipulation_v1` chained (`agent_speedup_v2_tasks.py:115`, `:136-137`, `collision_base_mode`). R3's `uc72-task-spec-1` has no field for it. Add `evaluator.collision_base_mode`, or record the change, so S1's pristine and measured contact counts on g3 stay comparable.
9. **Worker `stop_on_success` derived from the flag** (SF-7).
10. **Scripts run with `python -I`** (e.g. `pointing_smoke.py`) ignore `PYTHONPATH` and the script directory. They must insert `src` themselves, as `replay_baseline.py:167` does.

---

## 7. Tests that would not catch the targeted failure

| Test in the plan | Why it misses | Replacement |
|---|---|---|
| M0 check 3 "10⁴ frames, no torn reads" | Stochastic, with a tiny race window; the specified read order is itself wrong (MF-1) | Deterministic paused-writer and paused-reader interleavings |
| `h1_equivalent` T04 replay as the only bit-exact anchor | Exercises only `execute` decisions; there are no anchors for M2–M4 behaviour (SF-5) | Differential zero-call replay with scripted answers; per-milestone golden replays |
| PS-11 "bridge off → replay bit-exact" | Proves the hooks are inert when off, says nothing about the ON-path timing (MF-3) | `RecordedPerceptionClient` test asserting each request's `scene_snapshot_frame` and overlay content |
| Leakage test "no `scene`/`evaluator` key reaches the policy" | Key-based. Values such as obstacle instance names (`ketchup_obstacle_1`), BDDL file paths, family or task ids could leak through metadata or feedback strings | Also grep every request payload and policy-visible file for all strings in the task spec's `scene`/`evaluator`/`provenance` sections |
| M0 overhead benchmark (≤ 0.06 s/step) | Measured only under `h1_equivalent` with perception off. Per-step budgets (swept ≤ 3 ms, client threads, SceneMap ingest, relocalise) are never measured end to end | Re-run the overhead benchmark under `m3_geometry`/`m4_swept` with `RecordedPerceptionClient` |
| PC2 T2.5 deadline from a worker thread (fake stream) | Misses the real urllib/TLS read path (socket idle timeout vs heartbeat lines) | Loopback HTTPS/SSE server test (SF-8) |
| PS-1 fail-open (fake socket drop, delay, garbage) | Misses a service that accepts but never replies, and a CUDA OOM inside the service process | Hang-stub and injected `torch.cuda.OutOfMemoryError` tests on the real server loop |
| EXC-C1 counterfactual over 9 logged runs | Open loop on H1's tracker trajectories; the braking tracker changes them | Keep as a development aid. Judge with the EXC-INT scripted simulator benchmark and S2 metrics. |

---

## 8. Coverage of report items

Most items are traced in §11 and are covered. Items with gaps:

| Report item | Status | Note |
|---|---|---|
| H1 #5 slow primitives | partially covered | Rotation lag is handled by B1 sequencing (raising the rotation cap is backlogged). Model-chosen contact-profile lifts and retreats (51–67 % of actions) have no mechanism: SP9 cruise is transit-only (M4) and D2 covers only `stop_on: contact` approaches (M5). |
| H1 #7 serial startup | partially covered | The second simulator is removed. The 14–52 s plan call remains (fold deferred with a reason, UD-15). MF-3 adds a bounded wait at call 1. |
| H1 #19 closed registries / hard-coded facts | partially covered | Registry, validator and LIBERO prompt examples are handled. The robot model stays Panda-specific (vendored Panda, `robot_facts_v2.json`); acceptable for LIBERO, but not general. |
| H3 §5.2 item 1 (same-window arms: H1, +H3 text, +SAM boxes) | partially covered | Replaced by a model-only first-request replay (≤ 60 calls) that scores first boxes. It cannot show contact or call effects. Acceptable as the cheapest test, but it should be labelled as narrower than the report's item. |
| H2 §4.4 item 2 (robot pixels as metric anchors) | covered | A deliberate, reasoned deviation (X29: FK depth replacement plus support-anchored affine). |
| H2 §4.4 item 3 (persistent fused occupancy) | partially covered | Deferred with a reason. The height field, discovery and conservative boxes cover the measured classes. |
| H2 §4.4 item 9 (harness grasp/place candidates) | partially covered | Snapping and references are covered. Candidates as marks are deferred with a reason. |
| K2 K7, K9, K10, K11; F4, F6, F7, F9 | partially covered | Covered implicitly (EXC-B3; PS-3/PS-5; PS-12/GT11/SP12; SP4 executed profile; PS-5 affine; GT2/GT3 never-empty supports; PS-11; current-only `context.scene`), but missing from the §11 traceability table. Add rows. |
| R9 / G3 "keep per-action local compute inside the control period" | partially covered | Budgets are stated (≤ 3 ms per step) but never measured end to end in perception-on configurations (§7). |
| IC14 / H3 V7 freshness | partially covered | Re-localisation at stage entry is covered. The model-facing scene lags one decision (MF-3). |
| §6 item 5 (seeded on-path obstacle for every relayout) | partially covered | Only with UD-10 `full44`. The recommended scope leaves obstacle evidence on 3 scenes. |
| §6 item 6 (cost for a pre-registered sweep) | partially covered | The S5 cost omits the pristine arm (SF-14). |
| §6 item 7 (H1 replay on decoded pixels) | partially covered | Planned (R9), but the fixture covers `execute` decisions only (SF-5). |
| A2 (raw frames in shared memory) | covered | FrameRing, with the MF-1 defect fixed. |
| A1/A3/A4/A5, B1–B3, C1–C5, D1–D3 | covered | Milestone mapping in §11 checked. The A4 fold is deferred with a reason. |
| R1–R8 root causes; H1 #1–4, #6, #8–18, #20; H3 V1–V6, V8–V9; IC1–IC13, IC15 | covered | Checked against §6 and §11 and the INV rules in §2.5. |
| Bottom line item 1 (pending K2 launch) | covered | Explicitly outside this build (backlog), with the report's diagnostic framing. |
