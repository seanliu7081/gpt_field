# Isaac Sim repair conclusion — A1

**The local repairs did not restore rendering. Make Toast Standard layout 0 remains blocked before its preview.** Two A1 diagnostics reached actual renderer failures; neither produced a camera image. No Astra request, benchmark action, or full-task attempt ran, and no test-start approval was consumed.

The user reopened model-free repair with “try to fix the IsaacSim problem and give me a conclusion.” Existing NVIDIA licence acceptance remained valid. All package changes and diagnostic files are isolated under `robodojo_uc71`; the host driver, shared LIBERO environment, and other workloads were not changed.

## Repairs and observed results

| Change or check | Result |
| --- | --- |
| Remove `CUDA_VISIBLE_DEVICES`; select physical GPU4 explicitly | Enumeration warning disappeared, but the minimal Kit diagnostic still segfaulted. GDB located the fault in `librtx.scenedb.plugin.so` during Hydra initialization. |
| Install the complete official `isaacsim[all,extscache]==5.1.0.0` package set in a fresh Python 3.11 environment | 140 packages installed with hash enforcement; dependency check passed. Standard `SimulationApp` became available. |
| Run standard `SimulationApp` and unmodified base Python experience with renderer/physics GPU4 and multi-GPU rendering disabled | Attempt 003 exited with SIGSEGV (`-11`) during viewport initialization, before its constructor returned or the procedural camera was created. |
| Check graphics installation and GPU identity | Inspected kernel and NVIDIA userspace components consistently report 610.43.02. Actual loaded graphics libraries were recorded and hashed. CUDA and Kit GPU4 both identify UUID `GPU-e1a460e9-e272-b227-222e-0dbb77c248c5`, PCI `0000:81:00.0`. |
| Repair diagnostic process ownership monitoring | NVML reports host PIDs while `/proc` exposes container PIDs. A same-child CUDA context ON/OFF/ON handshake established the mapping before attempt 003. |

The final stock-application diagnostic pre-created and retained GPU4's primary CUDA context for that identity handshake. Thus it is a standard application/package control with a disclosed bootstrap variation, not a completely untouched startup. Its Python stack establishes a viewport-startup fault; it does **not** identify the same native instruction as the separate minimal-Kit GDB stack.

## Preserve the distinction between crashes and guard stops

A1 consisted of five separately initiated diagnostic launches, not an automatic retry loop:

- `a1_mask_only_001`: stopped by an overly strict startup-allocation guard before a renderer verdict.
- `a1_mask_only_002`: actual native SIGSEGV; GDB shows `librtx.scenedb.plugin.so`; no image.
- `a1_attempt_001` and `a1_attempt_002`: stopped by process-ownership guards before a renderer verdict. These are harness issues, not evidence of driver failure.
- `a1_attempt_003`: actual SIGSEGV after approximately 29.39 seconds, including identity calibration. The raw result's final `gpu_process_ownership_unresolved` label is a post-exit monitoring race: the child had already died, while NVML briefly retained its verified host PID during cleanup. No live owned process remained for that guard to signal. Raw artifacts are unchanged; the [separate interpretation](a1_attempt_003_interpretation.json) records this correction. An independent audit verified the hashes, process timeline, GPU identity, and cleanup.

The current launcher now checks process exit before applying GPU guards and records actual cleanup signals. Five offline classification checks passed, including the captured attempt-003 terminal sample; live polling timing was not rerun. See [monitor repair validation](a1_terminal_race_offline_validation.json).

Only GPU4 was marked active for Kit rendering. Unmasked startup allocated small contexts on the other otherwise-idle GPUs; the diagnostic explicitly allowed up to 768 MiB per non-target GPU. The highest observed non-target owned allocation in the stock crash was 700 MiB. Afterward, all eight GPUs returned to 1 MiB with no reported GPU processes, and no owned simulator process remained. Resource cleanup is confirmed; clean application shutdown is not.

## Cause and next practical step

**Driver/RTX compatibility is the leading hypothesis, not a proven unique cause.** Removing the CUDA mask and using the full official package set were insufficient. The evidence does not show a mixed-version NVIDIA installation or establish that every possible Isaac configuration fails on this machine.

NVIDIA lists Linux **580.65.06** as a tested driver for Isaac Sim 5.1.0. An NVIDIA support reply specifically states that **610.43.02 is outside the validated production branch for 5.1.0**. This makes a controlled test on the documented R580 baseline more useful than simply installing a newer driver. Sources checked 2026-10-29: [5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html), [NVIDIA's 610.43.02 response](https://forums.developer.nvidia.com/t/isaacsim-crash-when-update-gpu-driver/371975/7).

The next practical route is a provider-managed compatible host driver, or an RTX host already using the tested 580.65.06 stack, followed by the same real 640×480 camera gate. The driver belongs to the host and cannot safely be replaced from this unprivileged container. No driver packages, CUDA compatibility shims, or alternate driver libraries were installed here. A successful R580 comparison is still required before claiming that a driver change fixes this failure.

Keep Isaac Sim at the RoboDojo-pinned 5.1 release for that comparison. Moving to a newer Isaac release is a separate compatibility change requiring RoboDojo validation. Before transferring to another machine, preserve the project and evidence externally: this workspace is not backed by a persistent host volume. No migration, provider contact, or additional rental was performed.

## Evidence and continuation

- [Full installation and package provenance](a1_installation_report.md)
- [Mask-removal comparison](a1_mask_comparison.json) and [native GDB log](a1_mask_only_002/stdout_stderr.log)
- [Stock launch result](a1_attempt_003/launch_result.json), [crash log](a1_attempt_003/stdout_stderr.log), and [corrected interpretation](a1_attempt_003_interpretation.json)
- [Graphics audit](a1_graphics_audit.md) and [loaded-library evidence](a1_attempt_003/loaded_graphics_libraries.json)
- [Final resource snapshot](a1_final_resources.json)

No further identical launches are scheduled. Native assets, controller calibration, layout-0 preview, and the full harness remain pending. Once a real camera gate passes, resume the implementation checklist; present the actual preview and obtain explicit approval before starting Make Toast. The previous 84 offline tests validate only the adapter components, not native rendering or task performance.
