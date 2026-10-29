# Stage A0: current native setup blocked

Status: **host compatibility gate failed; `ready_for_native=false`**.

The approved model-free camera check did not produce an image. No further Kit launch or download will run under this setup path. This result does not establish that the machine is categorically incapable of running Isaac Sim, or that its driver is the unique cause.

The user accepted the NVIDIA license and setup with “Accept the licence and proceed with setup.” That acceptance covers the isolated setup and GPU 4 diagnostic. The full Make Toast test remains unapproved.

## Installed environment

- Python: `runtime/a0_env/bin/python`, Python 3.11.16.
- Vendor stack: Isaac Sim/kernel/Kit caches/SDK caches/physics caches 5.1.0.0; no RoboDojo checkout or task assets.
- 62 Python packages; the final `uv pip check` passed.
- Requirements lock SHA-256: `4e12559814d5db9cd75bcb498f9cc77aba935e5cad6695f09669b8316e21081b`.
- `libXt` was unpacked from a hash-verified Ubuntu package into `runtime/sysroot`; no system package registration, global environment edit or driver change.
- GPU: physical index 4, `GPU-e1a460e9-e272-b227-222e-0dbb77c248c5`; Kit reported this GPU alone as active. Multi-GPU rendering was disabled.

## Preserved diagnostic attempts

1. `a0_attempt_001`: failed before scene creation because Replicator lacked `botocore` and SyntheticData required `fabricDefaultStageFrameHistoryCount >= 3`. After the Python import failure and shutdown sequence, the child exited with SIGSEGV (`-11`). No frame was produced. This secondary teardown crash remains recorded.
2. `a0_attempt_002`: separately reviewed setup repair added the documented import dependencies at Isaac Sim 5.1 package versions and the required history setting. Kit startup returned successfully, then the child exited with SIGSEGV (`-11`) after **7.582 seconds**, during `context.new_stage()`. No procedural-scene-created or first-frame marker was reached.

The second attempt's Python fault stack reaches `omni.kit.widget.viewport.impl.texture.__enable_hydra_engine`, through its stage-open callback and `omni.usd.new_stage`, to `provision_a0_camera.py:59`. No C-level `librtx.scenedb.plugin.so` backtrace was captured. Renderer initialization is the observed failure point; the unique underlying cause is unproven.

The second failure triggers the agreed stop rule. There was no driver workaround, settings sweep, automatic retry or third launch.

## Terminal audit and remaining limits

No owned setup or Kit process remains running. GPU 4 returned to 1 MiB. Training GPUs 0–3 retained their original 21,394 MiB allocations; their jobs were not stopped or modified. The terminal GPU and compute-process snapshots are retained beside this report.

Model requests: **0**. Benchmark actions: **0**. Full task attempts: **0**. Native preview: **unavailable**. RoboDojo physics, reset determinism, 300-second idle behavior, ARX control and Make Toast execution remain untested.

See `host_blocked_report.json` for exact attempt hashes, stack and limitations; `stage_status.json` for the durable readiness gate; `installed_package_inventory.json` and `first_launch_registry_extensions.json` for environment provenance. Original attempt directories and installer logs are preserved.
