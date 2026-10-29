# Native RoboDojo Make Toast — implementation and test plan

Status: planning only, 2026-10-28. No implementation, simulator run or model call is authorized by this document. The user requested a plan first, a dedicated folder under `gpt_field`, and the same information categories used for LIBERO-10. The user subsequently selected **native two-arm control and three RGB cameras**.

## 1. Experiment definition

Build a separately versioned **uc7.1 RoboDojo adapter** rooted in this folder. Preserve the uc7.1 decision process, direction semantics, approval workflow, history policy and Astra-high subscription transport. Adapt the embodiment and simulator interface explicitly; do not describe a dual-ARX/Isaac implementation as the byte-identical Panda/MuJoCo policy.

| Item | Proposed first test |
|---|---|
| Environment | Native RoboDojo / Isaac Sim; exact task key `make_toast` |
| Variant | Standard; no `make_toast_random`, custom relayout or object replacement |
| Robot / cameras | Official `arx_x5` / `dual_x5`; head, left-wrist and right-wrist RGB |
| Episode selection | Seed-directory0, first official saved layout (layoutID0), one environment, one attempt |
| Model | `gpt-6-astra`, reasoning `high`, initial planning and execution |
| Billing | Existing ChatGPT subscription/OAuth; API-key calls0; no model fallback |
| Information | Same categories as LIBERO uc7.1, extended to both arms/three views as selected by the user; exact boundary in INPUT_CONTRACT.md |
| History | Fresh per attempt; all accepted execution text, six recent synchronized three-view sets; no earlier-task execution history |
| Review | Initial plan reviewed before launch; `human_reviewed_plan=true`, `zero_shot=false`; truthful chat/browser review metadata |
| Success | Native full-success signal, with native partial score reported separately |
| Native horizon | Preserve1400 native control actions; no added wall-time, model-call or fragment-count episode cap |
| Test authorization | Actual initial preview and plan first; explicit user approval for one attempt; no automatic whole-task retry |

Seed-directory0 and layoutID0 are proposed deterministic pilot selections, not copies of LIBERO's init0/seed7. Verify the first layout's actual filename before freezing it. Preserve the standard task's assets, robot configuration, physics and task overrides. The standard variant itself contains official layout/asset variation; “Standard” does not mean one universal hard-coded scene. The first attempt must use the selected official layout, without selecting a different scene after seeing the model's plan or result.

The [official task page](https://robodojo-benchmark.com/doc/sim-tasks/make-toast/) identifies Standard and Random separately. The pinned [task metadata](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/f57e1633ffed59761bc8e568cfe7530c4b33381c/task/RoboDojo/config/_task.yml) assigns25 episodes to each variant and disables robot self-collision for this task. Preserve that override and disclose it. One pilot episode establishes neither a benchmark success rate nor a full acceptance sweep.

## 2. Verified basis and limitations

The [linked Astra report](https://robodojo-benchmark.com/report/gpt-6-astra-eval) specifies Astra **medium as the default reasoning effort** with its RoboProbe interface; the report's actual campaign request logs have not been audited here. Its native observations include three RGB views and dual-arm proprioception. It reports50 episodes per base task, with Standard/Random entries for Generalization. Our high/uc7.1/controller/calibration/review/history settings differ, so treat the report as a reference rather than a matched baseline. Its100-planner-call limit appears in a particular robustness ablation; do not assume it is a universal benchmark budget.

The existing [robodojo_direct.md](../robodojo_direct.md) and [runner](../scripts/evaluate_robodojo_direct.py) implement a RoboDojo-style tool loop **inside LIBERO**. They cannot execute native Make Toast. The existing [RoboCasa adapter](../scripts/evaluate_robocasa_user_check.py) offers a useful process/review pattern but changes prompts and disables native macros; copying it would not establish uc7.1 parity.

The frozen uc7.1 client, geometry validation, state layout and primitives contain single-Panda assumptions. Native ARX has six arm joints plus a gripper per arm. The adapter must describe real ARX geometry and channels, not pad/relabel them as Panda measurements or disable validation to accept them.

Planning source pin: RoboDojo commit `f57e1633ffed59761bc8e568cfe7530c4b33381c`, XPolicyLab gitlink `5d0be1499634d3176258c8f2e31f5fe816497b0a`. Lock IsaacLab/CuRobo gitlinks, simulator packages and asset revisions during installation. These are inspected planning pins, not evidence that the environment is installed or operational. Freeze any later change as a reviewed plan revision.

Machine inspection found8×RTX4090, CUDA12.8, driver610.43.02 and reported GL/OptiX/Vulkan availability. No native RoboDojo/Isaac installation was found in the inspected locations. Approximately191GiB is free; `/workspace` is not a persistent host volume on this instance. Native rendering and exact asset disk requirements remain to be tested.

## 3. Files and isolation

Only planning documents exist now. The implementation should use this structure:

```text
robodojo_uc71/
  README.md, PLAN.md, INPUT_CONTRACT.md, planning_manifest.json
  configs/              # pinned standard task, sensor and adapter settings
  src/robodojo_uc71/    # sensor schema, dual-arm policy adapters, worker, evaluator
  scripts/              # preflight, prepare, review and single-use run entrypoints
  tests/                # contract, geometry, timing, history and lifecycle tests
  vendor/               # pinned RoboDojo and submodules; immutable before tests
  runtime/              # isolated simulator Python environment and cache locations
  validation/           # offline results and clearly labeled simulator diagnostics
  previews/             # real initial images, reset hashes, initial planning review
  runs/<attempt>/       # approval, logs, trace, truth, video, audits and cleanup
```

Import the existing uc7.1 core read-only, or snapshot its exact release into a versioned local dependency. All compatibility wrappers/subclasses and any unavoidable prompt/schema differences live here. Save a core-to-adapter diff and separate core/runtime hashes. No edits to shared LIBERO code, its environments, protocol, past reviews or run artifacts. Avoid broad `sys.path` or global monkey patches whose effects reach another runner. Keep credentials outside this project and out of the simulator process.

## 4. Implementation stages and completion checks

### A. Provision the native environment

Use a separate Python3.11 simulator environment. The pinned [installer](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/f57e1633ffed59761bc8e568cfe7530c4b33381c/scripts/install.sh) selects Isaac Sim5.1.0 and a CUDA12.8 PyTorch stack. Inspect and reproduce its needed setup in isolation; its submodule `--remote` update must not silently replace our pinned commits. The container cannot run Docker-in-Docker, so use native installation inside the existing container. Do not change the host-injected NVIDIA driver.

Download the assets and official evaluation-layout subset needed for the standard environment and Make Toast. First resolve asset revisions, transitive dependencies and sizes against available storage. Training demonstrations and policy checkpoints are unnecessary. The task's `data_source: teleop` metadata is not permission to feed demonstrations to Astra. Record package, submodule, USD/config and layout checksums.

Completion check: a model-free native load/reset/render succeeds on one selected GPU; no credentials or model transport are involved; native score is initially incomplete. Any rendering/dependency failure is an installation result, not a failed Astra task attempt.

### B. Implement a strict observation boundary

Separate the simulator worker from the model process. Serialize model observations through an explicit allowlist; produce evaluator records through a separate channel. Preserve the exact official task instruction. Initial planning uses three clean views plus their three grid-marked versions (six images), with separate camera selectors and truthful wrist-orientation metadata. Retain uc7.1's distinction between the limited initial-planning request and richer calibrated execution requests; do not add state/calibration to planning merely because it is available in simulation.

Provide synchronized head/left-wrist/right-wrist RGB and real robot-only measurements/calibration. Keep the model-visible geometry allowlist at the LIBERO categories: camera calibration and fixed tool/gripper geometry with provenance. Full ARX link geometry, URDF, kinematic chains and complete joint limits belong inside the robot-only controller, not in Astra's observations merely because the controller needs them. Derive each wrist transform from its actual mount and current robot pose. Define one world frame and one external/head-camera-derived direction contract; neither wrist may redefine left/right. Maintain `external_view_v1` semantics with the new calibration, and repeat the frozen head-camera direction contract in every execution request independently of current sensor metadata/history trimming. Show real image crops, aspect ratio, camera roll and any resizing in the preview; compute intrinsics for the delivered pixels. Do not claim native512×512 if it required resizing. Prefer aspect-preserving512×512 model canvases with explicitly adjusted intrinsics and padding if native camera dimensions differ.

Bind both the original approval observation hash and a complete sensor/config hash covering all three images, both arm states, camera matrices and robot geometry. Existing legacy observation hashing can omit geometry/joints. Attempt exact repeated resets first; if native renderer/state nondeterminism prevents exact matches, stop preparation and specify reviewed deterministic state/tolerance checks before any model run. Do not silently weaken the binding after seeing a failed comparison.

Completion check: byte-level payload inspection, denylisted-field tests and robot-only calibration/projection checks prove the declared boundary. Full object/reward data may exist in evaluator artifacts but cannot enter planning, execution, feedback, exceptions or motion-planner obstacle inputs.

### C. Adapt uc7.1 execution to two ARX arms

Keep the existing high-level model loop and reviewed-goal authority. Add explicit arm identity to selected targets/fragments and keep measured execution state separate per arm. Start with one active arm per fragment while the other holds its last commanded state; either arm can be selected at the next model decision. This preserves the native two-arm embodiment without inserting autonomous bimanual coordination or task-specific handoffs. Do not add a bread/toaster-specific controller or hard-coded insertion/lever/home trajectory.

Audit every state8, seven-joint Panda, finger sign/range, grasp-site offset, axis-angle/quaternion and workspace assumption. Use actual ARX frame/kinematic definitions and expose genuine measurements or clearly identified robot-derived quantities. Never fabricate a second measured finger sensor or seventh arm joint to satisfy an old schema. Build a separately versioned geometry/state validator for these real channels.

Implement robot-only inverse/differential kinematics and local measured Cartesian control as the native equivalent of uc7.1 execution. Preserve target-error control-field intent and limits in physical units. Respect the native250Hz physics/25Hz control timing instead of transplanting20Hz gains or treating one native action as one whole model fragment. Record all gain, duration and tolerance conversions. No scene mesh, object pose, semantic grasp site, collision map or reward-driven waypoint may guide execution. Before choosing CuRobo or another backend, prove that it uses only permitted robot/static robot-calibration inputs; the official report's controller is not an automatic substitute for uc7.1.

Audit native skill/macro dependencies individually. Retain generic behavior where the new embodiment supports it; any omitted, newly implemented or changed capability must appear in the reviewed adapter diff and preflight. If a required primitive cannot be ported faithfully, return an explicit implementation gap rather than quietly run a reduced method under the original label.

Completion check: model-free robot-frame and controller diagnostics validate each arm, left/right image semantics, gripper limits, orientation continuity, inactive-arm holding, interruption feedback and error handling. These are bounded diagnostics, never full task attempts or task demonstrations sent to the model.

### D. Preserve native scoring and action accounting

Keep native `make_toast` reward/success and partial scores unchanged. Report official success, native partial score, termination cause, policy/control actions, physics substeps and model requests independently. A partial25/50 score, model “done” response, or successful process exit is not full task success.

Evaluator-only details: full success checks both filled slots, bread orientations/placement, exactly two slices on the shelf, lever position and return to robot origin. These conditions are more extensive than the environment's generated instruction. The [task implementation](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/f57e1633ffed59761bc8e568cfe7530c4b33381c/task/RoboDojo/tasks/make_toast.py) is authoritative; its numeric boxes/thresholds and hidden finishing conditions stay out of model prompts. A missed return-home condition must be reported honestly, not fixed by an automatic hidden retreat or an evaluator-derived prompt hint.

Preserve the native1400-action horizon. `take_action_cnt` counts each per-environment action dictionary, and each action expands into10 physics substeps at the standard25Hz observation rate /0.004s physics timestep. This is about56 simulated seconds, not56s wall time or1400 model calls. Multi-tick uc7.1 fragments must be charged one native action per tick; never submit an entire long fragment as one counted action. Route commands through the native action path and preserve its interpolation/hold schedule and end-of-action reward-sampling boundaries. Matching counts/rates alone is insufficient if the wrapper bypasses those dynamics. Freeze simulation while awaiting Astra or approval. If the harness cannot pause without advancing physics/action accounting, fix that before launch. Source: [native evaluation client](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/f57e1633ffed59761bc8e568cfe7530c4b33381c/src/eval_client/eval_env.py).

This preserves a benchmark-inherent limit and differs explicitly from the unlimited LIBERO episodes. Do not add a wall-time/model-call cutoff. Any later uncapped continuation would be a separately authorized nonstandard diagnostic, never a Standard success or a replacement result.

LIBERO's strict physical evaluator and hold-only `v2_success` exception are not automatically valid for Isaac/Make Toast. Initially use `strict_physical_success: null`, `v2_success: null` and `post_settle_success: null`, each with “not implemented for this adapter,” rather than reusing the raw native result as a strict pass. Plan a separate, versioned evaluator-only check of release, support and rest where the simulator exposes sufficient evidence; unsupported checks must be unknown. Report it separately from native scoring. Do not replay LIBERO's38 diagnostic actions, invent a hold-only exception or alter the policy state to repair an official result. Contacts remain record-only; log identities only in evaluator artifacts, with native self-collision settings and sampling resolution disclosed. Use physics-substep sampling where available; an endpoint-only log cannot support a collision-free claim or a directly matched contact-count comparison with LIBERO.

### E. Validate, freeze and show the preview

Run focused contract tests for three-view history retention, reset between attempts, settings consistency, dual-arm frame conversions, controller/action accounting, denied truth fields, native horizon/success, approval binding and final cleanup. Regression-check the imported frozen core without editing it. Do not run every historical repository suite by default.

Then perform two model-free resets of the selected official Standard layout. Preserve native reset preparation, including its saved-layout stability procedure, ten render passes and the explicit200-step `setup_scene` settling loop, plus any preceding environment/reset stability preparation; record these separately from the1400 policy-action allowance. Do not replace them with LIBERO's ten preparation actions. Save real head and both wrist images, native layout ID/config/asset hashes, robot initial state and evaluator preparation checks. This is an original standard layout, so label it accordingly; no relayout comparison is required unless the layout is changed. Confirm target visibility, initial stability and initially incomplete success, without claiming future paths collision-free.

Prepare one Astra-high initial plan under the planning input contract. This planning request is a separately recorded operation after implementation is requested; it does not execute policy actions. Preserve its original output, validation result and any proposed review edits. Do not sample repeatedly for a better plan. A review correction requires disclosure, original-output retention and user approval as in the LIBERO protocol.

Present the concrete preview, plan, native1400-action limit, adapter/prompt differences, selected layout and single-attempt launch configuration. Obtain the user's explicit start approval, required by the user's standing preview-before-test instruction and [existing protocol §3–4](../UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md). Implementation approval alone is not test-start approval. Create a fresh review/output and consume approval once.

### F. Run once, monitor and report

Use a private supervisor with `autostart=false`, `autorestart=false`, `startretries=0`, a short0600 socket and a single-use attempt guard. Bind model, high effort, schema/prompts, both robot states, three-view calibration, layout, assets, code/config and request settings. Retain180s per-request deadline and existing finite request-retry semantics; distinguish those retries from whole-episode restarts.

Provide progress about once a minute and on significant transitions/errors. Monitor actual request/response logs, native action count and simulator state; do not present model belief as evaluator truth. Preserve user stop at complete feedback boundaries, immediate error finalization, request/token accounting and all failed results. No queued task or automatic new attempt.

Record all three cameras plus XY/XZ/ZY explanatory plots in the1600×960 dashboard, using separate arm colors/traces. Keep20fps output if matching the prior video format, but label native25Hz control correctly; video frame rate must never set simulation speed. Clear targets/arrows at terminal. Record video-rendering overhead separately.

Only official full success qualifies for completion time: control elapsed minus measured video-rendering overhead, including model waits and local control. Horizon/failure/error/user-stop gets completion timeN/A plus elapsed attempt time. Report request sends/accepted/errors/retries, planning separately, tokens, subscription/API-key accounting, official score, physical diagnostics, collisions and strict-evaluator availability. Check video with ffprobe and active/terminal frames; re-hash all frozen inputs/core/adapter/assets; shut down only this attempt's private processes/socket. Preserve the trial even if it fails.

## 5. Reviewable deliverables before first test

- Pinned native environment and a functioning credential-free simulator worker.
- Separate dual-ARX adapter, explicit prompt/schema/capability diff, and the selected information contract implemented as tests.
- Demonstrated25Hz native action accounting,1400-action termination and native score handling without hidden truth in model context.
- Model-free reset/controller checks, actual three-view preview and frozen official layout.
- One high-effort planning draft, original validation and any disclosed corrections.
- A launch manifest that is still unstarted until the user approves the concrete test.

No executable launch command is supplied in this planning document because the adapter does not exist yet. It will be generated and validated against the installed environment in stageE. Remaining technical unknowns are asset disk footprint, native render compatibility, exact camera pixel transforms, native reset determinism, geometry/skill port fidelity and the availability of sufficient physical diagnostic signals. Resolve these during implementation without changing the declared sensor boundary or standard task to make a result pass.
