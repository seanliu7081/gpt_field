# Local Conda setup — updated 2026-10-30

Current-host fixes are verified in [the setup record](../validation/conda_host_20261030/setup_result.json).
The launcher selects GPU 0 by UUID on this two-GPU host, and its bounded camera
diagnostic passed with eight captures and clean GPU release. ChatGPT subscription
sign-in is connected, with plan use granted. The root parent owns this app's
credential store; a real worker running as UID/GID 65534 proved it cannot read
the store or the parent environment. The privileged suite passed 593 tests,
including all previously blocked ownership fixtures. A final 103-test targeted
run passed after the final changes (594 distinct tests passed across both runs).

Supervisor **4.3.0** is installed. It runs the parent process independently of a
terminal and collects logs. Generated configurations disable automatic start,
restart and retry. A service is prepared only with the fresh reviewed task
packet; no full-task service was started during setup.

The `robodojo_uc71` environment is installed at
`/home/haotian/miniforge3/envs/robodojo_uc71`. Its Python executable is the IDE
interpreter to select:

```text
/home/haotian/miniforge3/envs/robodojo_uc71/bin/python
```

Activate it and enter this project:

```bash
conda activate robodojo_uc71
cd /home/haotian/code/gpt_field/robodojo_uc71
python -m pip check
```

Installed versions include Python **3.11.16**, Isaac Sim **5.1.0.0** with all
extension caches, PyTorch **2.7.0+cu128**, NumPy **1.26.0**, pytest **8.4.2**,
and FFmpeg/ffprobe. The preserved native dependencies and compatibility pins
are installed in the same environment. `setuptools==80.9.0` supplies the
`pkg_resources` API required by Kit; `wheel==0.45.1` is compatible with the
preserved `packaging==23.0` pin.

Activation supplies the project-local GLU/OpenGL libraries, disables user-site
packages, selects PCI ordering for CUDA, and applies the already recorded NVIDIA
EULA acceptance. Conda restores prior environment variables on deactivation.
The environment's `.pth` file exposes this adapter, the frozen uc7.1 core, and
the downloaded native source. The original project and other environments were
not installed over or upgraded.

The exact native commit and all three pinned submodules are checked out under
`runtime/native/RoboDojo`. All **214 files** in the recorded Make Toast asset
subset were downloaded from the pinned dataset revision and SHA256-verified.
The published missing camera-stand texture and material-reference defects remain
as documented in the original asset assessment; these downloads do not establish
complete asset fidelity. All **851 frozen core files** still match their manifest.

The model-free camera check passed on GPU 0 (RTX 4090, driver 580.178.04). Eight
captures produced an inspected 640×480 image of the procedural red and green
cubes. CUDA tensor execution passed, the process exited with code 0, and both
GPUs returned to 0 MiB used. This used NVIDIA's stock fast shutdown, which does
not establish full plugin teardown. The first camera attempt and its missing
`libGLU.so.1` failure are preserved beside the successful attempt.

The original 2026-10-29 offline validation returned **543 passed / 34 failed**. Installing
FFmpeg fixed the video failure; all four video tests then passed. The combined
result was **544 passing tests**, with **33 root-dependent tests then failing**
because their fixtures call `os.chown(..., 0, 65534)` or equivalent operations.
No tests were changed, mocked around, or silently skipped. Run the ordinary
controller/schema/video checks as the current user:

```bash
python -m pytest -q -o pythonpath= \
  tests/test_controller.py tests/test_schema_native.py tests/test_attempt_evidence.py
```

The complete suite is `python -m pytest -q -o pythonpath= tests`; its ownership
fixtures require a suitable privileged test environment. They now pass under
the verified root setup. Clearing pytest's
inherited `pythonpath` setting prevents the parent repository's mutable `src`
directory from taking precedence over the frozen core.

Package setup is repeatable from this directory:

```bash
bash scripts/setup_conda_env.sh
```

The script creates a missing environment, combines the existing hash-locked
runtime requirements with the reviewed compatibility override, installs the
locked test tooling, configures imports/libraries, and checks dependencies.
It writes a versioned Conda runtime record with exact package inventory, interpreter,
source import file and requirement hashes. The native launchers accept this
consolidated layout explicitly; their historical base/overlay validation remains
available. It does not download native source/assets or launch a simulator. `environment.yml`
also provides the Conda bootstrap. Native downloads on this host are recorded
in `runtime/setup_conda_20261029/source_verification.json` and
`runtime/setup_conda_20261029/assets_verification.json`.

To repeat only the bounded procedural camera check, use a new output directory
while the GPUs are idle:

```bash
python scripts/check_conda_camera.py --output runtime/camera_check_new
```

[Setup results](../validation/conda_host_20261029/setup_result.json) summarize the
evidence. Full logs, package inventories and the successful image are under
`runtime/setup_conda_20261029/` (ignored local artifacts). The camera image is
`camera_002/rgb.png`.

No model request or Make Toast episode was launched. Native qualification still
needs fresh evidence from the actual Make Toast scene: reset layout 0, verify all
three cameras and paused physics, measure both arms and grippers, and validate the
controller before building a runtime bundle. Then prepare a fresh task preview,
plan and review packet for this host. Published asset-reference defects remain
disclosed; a procedural camera pass does not establish task or contact performance.

The new Conda record is
`validation/conda_host_20261030/runtime_001/install_status.json`. Use that same file
for `--install-record`, `--dependency-record` and `--compatibility-record` in the
native diagnostic launchers. The empty `runtime/conda_imports` directory is an
explicit compatibility search path; installed dependencies live in Conda, and the
validator rejects any file added to that directory. For review-packet preparation,
pass `--runtime-record` and `--evidence-root` pointing to fresh qualified evidence
with the required source/assets/preview files. Use a short service directory such
as `runtime/services/toast001` so its Unix control socket fits the path limit.

Worker setup was completed with `scripts/configure_host_worker.py --run-tests`
under sudo. It changes only this app's credential ownership and creates fresh
probe directories; it does not change ownership of the code or Conda environment.
UID 65534 has traversal-only access through `/home/haotian`, via a targeted ACL.
The subscription store is `/home/haotian/code/gpt_field/.private/robodojo_chatgpt_plan`.
Future sign-in/status operations on that root-owned store require sudo and the
frozen `vendor/uc71/scripts/chatgpt_plan_login.py --store-dir ...` command. Keep
port 1455 forwarded privately for browser sign-in; it can be removed afterward.
See [OpenAI's OAuth flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
and [VS Code's port forwarding instructions](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel).

GPU availability and credentials are checked before an attempt is claimed.
Current Conda records are bound into the review manifest and revalidated before
launch. Historical consumed approvals remain unusable; automatic retry stays off.

The Python 3.11 Conda approach follows the pinned
[Isaac Sim 5.1 installation documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).
