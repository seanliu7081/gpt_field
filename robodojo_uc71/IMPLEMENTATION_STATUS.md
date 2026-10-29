# Implementation status — Make Toast Standard, layout 0

**Partial implementation; still blocked after the A1 renderer repair diagnostics.** There is no runnable, approved full-task harness yet. No real scene preview exists, no model request or benchmark action has been sent, and no test-start approval has been requested or consumed.

The user authorized implementation and subsequently accepted NVIDIA's licence for isolated setup and the GPU 4 camera check. That acceptance is not approval to start Make Toast. The intended selection remains native `make_toast`, Standard, seed-directory 0 / layout ID 0, one attempt with Astra high and a 1,400-native-action horizon.

## Runtime result

The initial A0 setup and crash evidence remain preserved. The user's subsequent request to fix Isaac Sim reopened model-free diagnostics. A fresh Python 3.11 environment now contains the complete official `isaacsim[all,extscache]==5.1.0.0` package set (140 distributions), and its dependency check passed.

Removing `CUDA_VISIBLE_DEVICES` did not repair the minimal Kit launch: GDB recorded a SIGSEGV in `librtx.scenedb.plugin.so`. The full stock-application diagnostic also exited with SIGSEGV (`-11`) during viewport startup, before any image. It explicitly pre-created a GPU4 primary CUDA context to verify host/container PID mapping; this bootstrap variation is disclosed. Three other A1 launches stopped on monitoring guards and are not counted as renderer failures. The final stock result's post-exit ownership label is corrected in a separate interpretation without changing raw evidence.

See the [A1 conclusion](validation/provisioning/A1_REPORT.md), [machine-readable conclusion](validation/provisioning/a1_conclusion.json), and [durable stage status](validation/provisioning/stage_status.json). NVIDIA components consistently report 610.43.02. Driver/RTX compatibility is the leading hypothesis; a controlled comparison on the documented 580.65.06 baseline is needed to establish a fix. No host driver or global environment was modified.

GPU4 was the only active Kit renderer. Model-free startup created bounded contexts on the other idle GPUs; all diagnostic processes and GPU allocations were released. Final evidence reports all eight GPUs at 1 MiB with no GPU processes. Earlier A0 training occupancy is historical; this work did not stop or modify those workloads.

## Prepared work

- Archived the seven original planning/review documents without modification; revised the active plan and information contract to incorporate the review decisions.
- Recorded native commit/gitlink/asset/layout pins. The exact frozen uc7.1 snapshot passed all 851 release file hashes; it is isolated in `vendor/uc71`.
- Implemented offline planning/execution schemas and client wrappers for two arms and three native 640×480 views, with the generic public home rule and an explicit Panda-macro capability reduction. These reuse the frozen subscription/history machinery; no live request has validated the new schema.
- Implemented single-use approval primitives, action/terminal accounting, dedicated socket IPC and a candidate robot-only differential controller. These are components, not native integration.
- Added synthetic and fault-injection tests. **All 84 combined offline tests passed**; results and source hashes are recorded in [offline_validation.json](validation/offline_validation.json). These tests do not establish simulator compatibility, physical tracking or task performance.

The controller's speed mapping and geometric tool assumptions remain **uncalibrated candidates**. Its world-frame Jacobian, six-joint ordering, live limits and tool offset need validation against the native ARX articulation. Do not use synthetic test results to declare it calibrated.

## Work still required before test approval

1. A working Isaac Sim 5.1.0 RTX environment that passes the real camera gate. The next practical comparison is a compatible host using the documented 580.65.06 driver baseline. No further identical diagnostic launches are scheduled.
2. Pinned native checkout and asset dependency installation, then the dedicated one-episode simulator worker and native joint-action/reward integration. Preserve native configuration and reset preparation and reject layout substitution/restart.
3. Measured camera/tool geometry, robot projection, dual-arm/gripper timing, native action interpolation/counting, reset repeatability and at least 300 seconds of paused model/approval waiting. Freeze controller settings before a live attempt.
4. Generic fragment/controller integration, parent-main-thread execution loop, evaluator-only diagnostics, video, private supervisor, monitored cleanup and a fully bound launch manifest. Current environment filtering is not filesystem isolation.
5. Actual layout-0 images and one initial Astra-high planning draft; display the preview, plan, capabilities and measured settings, then obtain explicit user approval for one full attempt. A live execution smoke or extended continuation requires its own approval.

No full-test launch command is provided because these prerequisites have not passed. The implementation authorization and licence record cannot substitute for the pending preview/start approval.
