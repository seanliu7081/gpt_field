# A1 independent graphics-stack audit

The archived A0 failure does **not establish a mismatched NVIDIA installation or a host-driver defect**. Inspected kernel and userspace NVIDIA components consistently report 610.43.02. A0 also contains an explicit CUDA-visible-device/Omniverse-enumeration warning and uses a custom minimal Kit bootstrap; both remain configuration variables until root-owned experiments separate them.

This audit performed static filesystem/loader inspection, NVML inventory through `nvidia-smi`, and one explicitly authorized CUDA Driver API device inventory. It did not start Isaac Sim, create a CUDA context, run kernels, create a Vulkan instance, install packages, change drivers, or stop workloads. Full command outputs, hashes, paths, and timestamps are in [a1_graphics_audit.json](a1_graphics_audit.json).

## Driver and device identity

- Kernel `/proc/driver/nvidia/version`, `nvidia-smi`, and 11 inspected CUDA/NVML/GLX/EGL/OptiX/RTX libraries agree on **610.43.02**. `/opt/nvidia-drivers` is absent, so A0's corresponding loader-path entry cannot select an older alternate installation there.
- The sole discovered Vulkan ICD is `/etc/vulkan/icd.d/nvidia_icd.json`, selecting `libGLX_nvidia.so.0`; that symlink resolves to `libGLX_nvidia.so.610.43.02`. System GLX/EGL and the packaged Vulkan graphics plugin have no unresolved dependencies in the isolated A0-style loader check.
- At the audit snapshot all eight RTX 4090 GPUs reported 1 MiB used and 0% utilization; the compute-process query was empty. This differs from earlier A0 occupancy and is only a timestamped observation. This audit changed no workload.
- At **2026-10-29 05:06:14 UTC**, a credential-free subprocess with `CUDA_VISIBLE_DEVICES` removed and `CUDA_DEVICE_ORDER=PCI_BUS_ID` returned eight CUDA devices. All inventory API calls succeeded. **CUDA ordinal 4 = PCI `0000:81:00.0` = UUID `GPU-e1a460e9-e272-b227-222e-0dbb77c248c5`**, matching archived Kit GPU4. The inventory calls were `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetPCIBusId`, and `cuDeviceGetUuid`; no CUDA contexts or kernels were created.

## What A0 actually proves

[A0 attempt 002 log](a0_attempt_002/stdout_stderr.log) lines 24–26 explicitly warn that CUDA and Omniverse enumeration differ and that setting `CUDA_VISIBLE_DEVICES` can cause undesired behavior or crashes. Lines 29–56 contain 14 CUDA-bad-state GPU-skip messages. A0 masks CUDA to the GPU4 UUID but selects renderer index 4. Nonetheless, the Vulkan table at lines 81–83 marks physical GPU4/PCI81/UUID e1a460e9 active. Thus the evidence shows an interop/enumeration confound; it does not show the Vulkan renderer selected the wrong physical GPU, or prove this warning caused the crash.

Kit startup returned at line 252. Exit -11 followed during `context.new_stage()`, with Python frames entering a viewport stage callback and `__enable_hydra_engine` (lines 294–303), before camera-frame validation. There is no saved native faulting-instruction stack in this evidence. Assigning the fault to a particular native plugin or the host driver from this Python stack alone would overstate the evidence.

A0 uses `KitApp` and a custom `a0_camera.kit`, and logs a missing `isaacsim.simulation_app` extension-path warning. This is an additional setup variable to compare with a validated stock SimulationApp bootstrap. Headless GLFW/X-display warnings, IOMMU warnings, and null audio output are present; their presence alone does not establish causation.

## Loader inspection limits

The two packaged `librtx.scenedb.plugin.so` copies are byte-identical: SHA256 `70066ea3da0f67d2d32fe6ab66bae1cb34ec62b9ddc8647b8915515204b99b90`.

Standalone `ldd` on RTX/Hydra plugins under only A0's explicit `LD_LIBRARY_PATH` reports some Python/USD/Carb dependencies unresolved. **Every directly listed missing name exists in the installed runtime.** Adding the existing Kit/USD/RTX dependency directories resolves the scenedb checks; the Hydra check then reveals an additional audio-schema dependency, also present in the package. These diagnostic paths were used only for `ldd`; no launcher was modified. Kit extension configs declare native libraries and plugins, so isolated `ldd` is not equivalent to the original process's dynamic loading.

Several extension packages bundle Vulkan loaders. Their presence on disk is not proof that different versions were concurrently mapped. The GPU Foundation config itself restricts explicit native-library preloading on Linux x86 and comments on duplicate-loader hazards. Actual mapped-library evidence from a reproducing process would be needed to establish a loader collision.

The evidence supports testing the CVD/renderer configuration and stock-bootstrap variables separately before declaring a host-driver incompatibility. Root owns any authorized live diagnostics; later A1 results should be evaluated separately from this archived-A0 audit. A successful native camera frame remains required to demonstrate a repair.
