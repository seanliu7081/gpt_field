# New-host handoff

The selected migration contains project source and the RoboDojo/uc7.1 preparation, frozen core, package locks, reviews, and diagnostic evidence. Historical ignored run archives, installed environments, caches, downloaded datasets/model weights, and credentials are excluded. References to old campaign paths or `/tmp` research caches are historical provenance, not files to expect on the new host.

**Make Toast has not started. No model request or benchmark action ran, no actual layout preview exists, and full-test approval is still pending.** The latest result is [A1_REPORT.md](validation/provisioning/A1_REPORT.md): removing CUDA masking and installing the complete Isaac Sim 5.1 package set did not restore rendering. Two diagnostics crashed; three stopped on monitoring guards. Driver compatibility is the leading hypothesis, not a proven cause. The final stock diagnostic retained a primary CUDA context for PID verification; that bootstrap variation and its post-exit monitoring race are recorded separately.

## Host and source checks

Prefer cloning at `/workspace/gpt_field` to reduce path changes. Read the new host's agent guide and inspect its persistence policy. Use an RTX-capable host with the documented Isaac Sim 5.1 Linux test baseline, **580.65.06**; verify the actual host driver and NVIDIA graphics userspace versions rather than relying on the rental label. An R580 comparison still needs a real camera pass before it establishes a fix. Do not replace the host-injected driver from inside a container.

These checks are read-only and do not launch a simulator:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used --format=csv
python3 /workspace/gpt_field/robodojo_uc71/scripts/verify_core.py
```

The core check must report 851 files and zero mismatches against `vendor/uc71/release_manifest.json`. Preserve this snapshot. [source_pins.json](configs/source_pins.json) records the native repository, gitlinks, asset revision, and layout-0 hash; these are not proof that native dependencies have been installed. The asset dependency closure is still pending. Do not advance submodules to their latest remote versions.

## Rebuild the isolated runtime

Reinstall from the preserved pins; do not copy a virtual environment across hosts. `runtime/` is intentionally absent. Bootstrap **CPython 3.11.16** in the task-local `runtime/python-builds` directory first: both installer scripts currently expect `cpython-3.11.16-linux-x86_64-gnu/bin/python3.11` there and do not download it themselves. Verify the new interpreter and the host OS/architecture before creating the environment.

Use [a1_requirements.lock](validation/provisioning/a1_requirements.lock), its [input](validation/provisioning/a1_requirements.in), and [wheel-source record](validation/provisioning/a1_wheel_sources.json) for the complete `isaacsim[all,extscache]==5.1.0.0` stack and pinned CUDA 12.8 PyTorch wheels. Preserve hash enforcement, run the dependency check, and save a fresh package inventory. The previous setup required at least 60 GiB free. Old failed-resolution locks are evidence, not installation inputs.

**The existing provisioning scripts are reviewed historical implementations, not an unchanged new-host launch recipe.** Before using adapted copies:

- Bootstrap the interpreter above; verify `uv` and any selected debugging tools are available.
- Give new installation/status records their own output location. `provision_a1_install.py` otherwise overwrites `a1_install_status.json`; its resume flag applies only to the recorded old resolution failure. `provision_a0_install.py` recompiles and overwrites its lock and is not the complete-stock reproduction path.
- Review `runtime/sysroot` and library lookup paths. The old helper extracts a particular Ubuntu `libXt` package locally; it is not a portable system dependency solution. Verify new-host Vulkan ICD and graphics libraries before reuse.
- Keep the original authorization/licence records as provenance and bind the new diagnostic to the new host within the authorized setup scope. Those records do not approve a full task.

## Rebind one bounded camera diagnostic

The old launcher, camera script, and identity helper hardcode GPU index/CUDA ordinal **4**, UUID `GPU-e1a460e9-e272-b227-222e-0dbb77c248c5`, and PCI bus `81:00.0`. Review all these checks together with their authorization bindings. Identify the new device by UUID/PCI address; Kit renderer indices and CUDA ordinals must be verified separately. Do not simply replace every `4` or assume `nvidia-smi` ordering equals Kit ordering.

Use fresh attempt names and output directories. The launcher accepts only old `a1_attempt_001`–`003` names, all already occupied in the preserved evidence. Never delete those directories to bypass the one-attempt guard. Retain the unmasked CUDA/Vulkan diagnostic design, single selected renderer/physics device, finite camera updates, deadline, process ownership verification, and cleanup. The current unmasked diagnostic requires other GPUs to be idle and permits bounded context allocation on them; it is not strict GPU-memory isolation. The current launcher contains an offline-checked repair for the recorded post-exit NVML ownership race; revalidate its live behavior on the new host.

A pass requires saved, inspected 640×480 RGB showing the procedural objects, successful bounded updates, and clean termination. Startup messages or a dependency check alone are insufficient. Save the host mapping, script/package hashes, logs, image, and resource cleanup evidence together.

## Continue only after the camera gate

Follow [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md): install the pinned native source/assets, complete the dedicated simulator worker and execution loop, validate native ARX geometry/control, camera calibration, action counting and paused waits, then freeze the measured settings. The 84 offline tests cover components only; native integration remains incomplete.

The intended pilot remains Make Toast **Standard, layout 0**, native dual ARX arms, three 640×480 views, Astra high, and a 1,400-native-action horizon. Present the actual layout preview, initial plan, and capabilities, then obtain explicit approval for one full attempt. No live execution smoke or extended continuation is authorized.

Authenticate the subscription/OAuth transport separately on the new host. Do not commit or copy login tokens, `.env` secrets, or private provider files into this public repository. Preserve the existing licence acceptance as a record; do not treat authentication, setup authorization, or migration as test-start approval.
