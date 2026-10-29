# Implementation status — Make Toast Standard, layout 0

**Partial implementation; procedural and native layout-0 diagnostic rendering have passed.** There is no runnable, approved full-task harness yet. The native reset produced real head and wrist views, with a sampled peak of 5,726 MiB VRAM and at least 18,356 MiB free. No model request or benchmark action has been sent, and no test-start approval has been requested or consumed.

The pinned native source and submodules are verified. The selected 214 asset files total 457,282,229 bytes, with an additional audited 15.2 MiB of NVIDIA material resources. One connected normal texture is absent from the published camera-stand package; robot materials also contain out-of-scope bindings. The reset-only diagnostics preserve original assets and explicitly leave Standard readiness and asset fidelity false. An isolated compatibility overlay repaired both startup import errors (`hid` and `pkg_resources`); the second native launch has clean extension startup. See the [latest native review](validation/new_host_20261029/native_preview002_review.json).

The second native diagnostic also passed a 300.0005-second physics-paused wait with unchanged joints, physics time and step counts. Two independent fresh resets produced bitwise-equal recorded robot state. Pixel differences remain small but nonzero; their acceptance policy must be frozen before a model call or approval consumption. Source geometry and operator-only projections support the proposed physical finger-pad tool point at home; dynamic calibration remains pending.

The current host has one RTX 4090, driver 580.82.09, and 24,065 MiB free VRAM at idle. A separate hash-pinned Python 3.11.16 / Isaac Sim 5.1.0.0 environment installed successfully (140 packages, dependency check passed). Task-local graphics support libraries repair missing X11/GLU dependencies without changing the host driver. New-host scripts bind GPU 0 by full UUID/PCI address and preserve all historical evidence.

The final procedural diagnostic saved eight valid 640×480 images and exited with code 0 using NVIDIA's default fast shutdown. Its GPU memory returned to 17 MiB with no processes. Independent review identified the raw failure flag as an exit-time ownership-monitor race involving only the verified child PID; raw records remain unchanged. The separate full-teardown diagnostic rendered successfully but required operator termination after Python failed to exit. See [new-host evidence](validation/new_host_20261029/). The subsequent native diagnostic measures scene memory; control and task performance remain unvalidated.

The user authorized implementation and subsequently accepted NVIDIA's licence for isolated setup and the GPU 4 camera check. That acceptance is not approval to start Make Toast. The intended selection remains native `make_toast`, Standard, seed-directory 0 / layout ID 0, one attempt with Astra high and a 1,400-native-action horizon.

## Historical runtime result on the previous host

The initial A0 setup and crash evidence remain preserved. The user's subsequent request to fix Isaac Sim reopened model-free diagnostics. A fresh Python 3.11 environment now contains the complete official `isaacsim[all,extscache]==5.1.0.0` package set (140 distributions), and its dependency check passed.

Removing `CUDA_VISIBLE_DEVICES` did not repair the minimal Kit launch: GDB recorded a SIGSEGV in `librtx.scenedb.plugin.so`. The full stock-application diagnostic also exited with SIGSEGV (`-11`) during viewport startup, before any image. It explicitly pre-created a GPU4 primary CUDA context to verify host/container PID mapping; this bootstrap variation is disclosed. Three other A1 launches stopped on monitoring guards and are not counted as renderer failures. The final stock result's post-exit ownership label is corrected in a separate interpretation without changing raw evidence.

See the [A1 conclusion](validation/provisioning/A1_REPORT.md), [machine-readable conclusion](validation/provisioning/a1_conclusion.json), and [durable stage status](validation/provisioning/stage_status.json). NVIDIA components consistently report 610.43.02. Driver/RTX compatibility is the leading hypothesis; a controlled comparison on the documented 580.65.06 baseline is needed to establish a fix. No host driver or global environment was modified.

GPU4 was the only active Kit renderer. Model-free startup created bounded contexts on the other idle GPUs; all diagnostic processes and GPU allocations were released. Final evidence reports all eight GPUs at 1 MiB with no GPU processes. Earlier A0 training occupancy is historical; this work did not stop or modify those workloads.

## Prepared work

- Archived the seven original planning/review documents without modification; revised the active plan and information contract to incorporate the review decisions.
- Recorded native commit/gitlink/asset/layout pins. The exact frozen uc7.1 snapshot passed all 851 release file hashes; it is isolated in `vendor/uc71`.
- Implemented offline planning/execution schemas and client wrappers for two arms and three native 640×480 views, with the generic public home rule and an explicit Panda-macro capability reduction. These reuse the frozen subscription/history machinery; no live request has validated the new schema.
- Implemented single-use approval primitives, action/terminal accounting, dedicated socket IPC and a candidate robot-only differential controller. These are components, not native integration.
- Added a pure fragment executor with inactive-arm hold, latched grippers, bounded segments, observation checkpoints and lifecycle accounting. Its calibration remains external and unvalidated.
- Added native action accounting and a worker RPC session with independent approval checks. Runnable bootstrap, calibrated measurement callbacks and parent loop remain to be integrated.
- Added synthetic and fault-injection tests. The latest combined run passed **209 tests** ([XML](validation/new_host_20261029/offline_tests_native_components_002.xml)); the original [84-test evidence](validation/offline_validation.json) remains intact. These tests do not establish physical tracking or task performance.

The controller's speed mapping and geometric tool assumptions remain **uncalibrated candidates**. Its world-frame Jacobian, six-joint ordering, live limits and tool offset need validation against the native ARX articulation. Do not use synthetic test results to declare it calibrated.

## Work still required before test approval

1. The new-host procedural camera gate is satisfied using the documented default fast-shutdown mode. Carry its exact package/library/device bindings into native setup; full plugin teardown remains unproven.
2. Resolve or explicitly review the published asset fidelity defects and finish the dedicated one-episode worker. Native dependency repairs, pinned checkout and selected asset downloads are complete. Preserve native configuration/reset preparation and reject layout substitution/restart.
3. Complete dynamic camera/tool geometry checks, dual-arm/gripper timing, native action interpolation/counting and the reset-image comparison policy. The 300-second paused wait and exact robot reset repeatability have passed. Freeze measured controller settings before a live attempt.
4. Generic fragment/controller integration, parent-main-thread execution loop, evaluator-only diagnostics, video, private supervisor, monitored cleanup and a fully bound launch manifest. Current environment filtering is not filesystem isolation.
5. Actual layout-0 images and one initial Astra-high planning draft; display the preview, plan, capabilities and measured settings, then obtain explicit user approval for one full attempt. A live execution smoke or extended continuation requires its own approval.

No full-test launch command is provided because these prerequisites have not passed. The implementation authorization and licence record cannot substitute for the pending preview/start approval.
