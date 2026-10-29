# A1 isolated stock Isaac Sim installation

Recorded 2026-10-29. This report covers package setup, the first two stock-launch guard stops, and the third stock diagnostic's SIGSEGV during viewport initialization. It does not establish a unique driver cause or claim that every possible Isaac Sim configuration fails on this host.

The renewed user request was “try to fix the IsaacSim problem and give me a conclusion.” The existing NVIDIA licence acceptance remains recorded in `vendor_authorization.json`; the renewed model-free scope is in `a1_authorization.json`. No model call, RoboDojo episode, Make Toast asset download, host driver change, or global environment modification was performed by this setup.

## Installed environment

- Interpreter: `runtime/a1_env/bin/python`, Python 3.11.16.
- NVIDIA stock package set: `isaacsim[all,extscache]==5.1.0.0`, including `isaacsim-core` and the standard `SimulationApp` launcher and base Python experience.
- Exactly 140 installed distributions. Distribution metadata is preserved in `a1_installed_package_inventory.json`.
- PyTorch `2.7.0+cu128`, torchvision `0.22.0+cu128`, torchaudio `2.7.0+cu128`; Triton `3.3.0`.
- Exact official wheel URLs and published SHA256 values are recorded in `a1_wheel_sources.json`.
- Final lock SHA256: `0ae7040817518bee2d44c969760935495e8e049ddf3918a474a62579eac8ffbc`.
- Installation used `--require-hashes`; `uv pip check` passed. The successful continuation log is `a1_install_20261029T051117Z.log`.
- Successful continuation declared approximately 8.779 GiB of downloads, excluding any already cached content and earlier partial downloads. Free disk after installation was 158,278,795,264 bytes.

`a0_env` remains separate. A subsequent distribution audit found all 62 original A0 packages, no extra packages, and no changes to their METADATA or RECORD hashes (`a1_a0_preservation_check.json`). This is a distribution-metadata check, not a fresh hash of every installed file. A0 attempts and their copied sources were not edited.

## Preserved setup failures and repair

The first install stopped on a Triton wheel hash mismatch. The generated lock contained the PyPI distribution hashes, while installation with `--torch-backend cu128` selected different bytes from PyTorch's index. The computed SHA256 `8cf975f47838c495a6a1a71007abad31b56312924a97d78874412de3232a7993` matches the official PyTorch index's Triton 3.3.0 CPython 3.11 Linux wheel. The PyPI build has SHA256 `3161a2bf073d6b22c4e2f33f951f3e5e3001462b2570e6df9cd57565bdec2984`.

The repair bound the three CUDA 12.8 PyTorch packages to exact official wheel URLs and Triton to the exact PyPI URL/hash. The next compile reused the previous output's multiple platform hashes, so installation correctly rejected the direct URL/hash conflict before downloading. A fresh output lock eliminated that reuse. Each of the four direct wheel entries was then checked to contain exactly its published source hash before the successful install.

Hash enforcement was never disabled, and neither mismatched wheel was accepted. Both failed logs and lock/status snapshots remain under `a1_*failed_backend_resolution*` and `a1_*failed_stale_direct_hashes*`.

## Diagnostic implementation and guard outcomes

The camera helper uses NVIDIA's unmodified `SimulationApp` and `isaacsim.exp.base.python.kit`, explicitly selects renderer GPU4 and physics GPU4, disables multi-GPU rendering, omits `CUDA_VISIBLE_DEVICES`, and builds only a procedural red/green-cube scene with a 640×480 RGB camera. It requires a validated image and clean shutdown for a successful diagnostic. The launcher preserves source/configuration snapshots, limits each attempt to 600 seconds, monitors GPU allocation, and collects actual loaded graphics-library paths and hashes when observed.

After a separate comparison established that unmasked initialization creates small CUDA contexts on all enumerated GPUs, the coordinated stock diagnostic permitted at most 768 MiB of total owned context memory on each otherwise-idle non-target GPU. Actual renderer and physics selection must still identify physical GPU4. Unknown process ownership remains a stop condition.

1. `a1_attempt_001` used GDB and stopped after 4.47 seconds on a 6 MiB graphics allocation reported as NVML PID 88394. The initial process-group-only ownership accounting omitted a debugger inferior with a different process group. The record originally called this an external job, but that attribution is unproven. It is a monitoring/ownership stop, not an observed renderer crash.
2. Before `a1_attempt_002`, the launcher was repaired to track recursive PPid descendants, original process-group members, retained PID/start-time identities, and available NSpid aliases. CPU-only subprocess validation confirmed discovery and cleanup of a child in another session. Root then launched the second attempt directly, without GDB. It stopped after 4.57 seconds because NVML PID 95555 could not be mapped to the container's Python PID. Its approximately 386 MiB contexts across GPUs are consistent with unmasked initialization, but the guard did not infer ownership from that pattern. The process was terminated by the guard (`-15`); no RGB image or RTX crash verdict was produced.

Both stock attempts ended before a valid camera frame. Their sources and evidence are preserved. The direct attempt confirms that `/proc`/NVML PID translation requires explicit handling even without a debugger; neither of these two guard stops establishes a driver or renderer failure. Final renderer interpretation belongs to the separately coordinated diagnostic assessment.

## Stock attempt 003: identity calibrated, then viewport SIGSEGV

Root coordinated a third direct attempt after reviewing the namespace-identity repair. Before importing Isaac Sim, the same child initialized only physical CUDA device4 (UUID `GPU-e1a460e9-e272-b227-222e-0dbb77c248c5`, PCI `0000:81:00.0`). Over a dedicated inherited Unix socket, the parent challenged the child to retain/set, release, and recreate that GPU's primary context. Two NVML samples in each phase verified its presence, disappearance, and return with the same NVML PID. Six samples bound container PID `3136451`, kernel start ticks `92534791`, and NVML PID `111192`; the nonce and device identity agreed on both sides.

The child then continued into the standard `SimulationApp` and stock `isaacsim.exp.base.python.kit`, retaining the primary CUDA4 context. This is explicitly a **stock SimulationApp diagnostic with a precreated GPU4 primary CUDA context**, rather than an unchanged stock startup sequence. No model or benchmark task was involved.

Kit's GPU table identified GPU4 as the sole active renderer. The child subsequently exited with signal11 (`returncode=-11`) before `SimulationApp(...)` returned. Its preserved Python fault stack locates the main thread in `SimulationApp._wait_for_viewport`, called from its constructor. The log contains `Fatal Python error: Segmentation fault`; no camera image was created. The launcher completed after approximately 29.39 seconds. This direct attempt provides a Python fault stack; the separate minimal, unmasked GDB comparison supplies the native `librtx.scenedb` stack and should not be represented as this attempt's native stack.

The raw launch result also contains `termination=gpu_process_ownership_unresolved`. That label reflects a late monitor race, not another failed identity challenge. The preceding monitor sample still shows the live child and its accepted alias. The final sample has no live owned process, while NVML briefly retains the already-verified PID111192 with declining GPU memory. The monitor drops an alias after the corresponding container process disappears, then classifies those stale NVML entries as unmapped. Its termination helper signals only live owned processes, so the final empty-owned-process branch cannot explain the already-exited child's SIGSEGV. The raw record remains unchanged; this interpretation and supporting file hashes are preserved separately in `a1_attempt_003_interpretation.json`.

Observed peak memory for that verified PID was 694 MiB on GPUs0,1,2,5,6,7; 700 MiB on GPU3; and 752 MiB on GPU4. Non-target peaks stayed within the reviewed 768 MiB context allowance. These are actual observed non-target context allocations, not a claim of zero allocation outside GPU4. Final monitoring found all eight GPUs back at 1 MiB with no listed processes and no remaining owned process.

Actual mapped NVIDIA driver libraries—including `libcuda`, `libGLX_nvidia`, `libEGL_nvidia`, and `libnvidia-rtcore`—resolved to version `610.43.02`; full paths and hashes are in `a1_attempt_003/loaded_graphics_libraries.json`. Standard GLVND/Mesa and bundled Vulkan/runtime libraries also appear in that inventory. There was no host-driver or graphics-library replacement.

The full package setup is repaired and dependency-checked, but this approved stock camera diagnostic did **not** produce a working image. No further launch was performed after attempt003. The preserved evidence supports a continuing renderer-initialization failure on the current setup, while leaving driver causality and alternative configurations unproven.

## Subsequent monitor repair, checked offline

After preserving the attempt003 source snapshot and raw results, the current launcher was repaired for future diagnostics. It checks the child's exit status immediately after GPU sampling and again before a guard shutdown. Once the child exits, it stops interpreting lingering NVML entries instead of extending the alias beyond that process's lifetime. It also records signals actually sent during cleanup and gives an independently observed child signal exit precedence over a late monitor label when no guard signal was sent to that child. Cleanup signals to a separate descendant do not overwrite the primary child's already-observed signal exit.

Five offline checks passed: the captured attempt003 terminal sample classifies as signal11; normal completion and nonzero error exit retain their meanings; an actual guard signal to the child remains a monitor-initiated shutdown; and cleanup of a different descendant cannot replace the child's SIGSEGV. Source syntax also passed. The exact source hash and validation limits are in `a1_terminal_race_offline_validation.json`. No GPU launch or live timing rerun was used to validate this later repair.
