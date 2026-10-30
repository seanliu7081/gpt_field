# Official RoboDojo control in UC7.1

The default for new runtime bundles and review packets is `official_ee`.
The pinned native RoboDojo source is unchanged. UC71 now supplies native EEF
actions to its official cuRobo IK, joint interpolation and actuators.
`legacy_joint` remains explicit for historical evidence and comparisons;
missing or failed official dependencies never trigger a fallback to it.

## Action contract

1. UC71 still selects one arm and supplies its calibrated tool/TCP target.
2. The adapter computes the native world link6 target using the **target**
   orientation: `p_link6 = p_tool - R(q_target) @ tcp_offset_link6`.
   The approved tool transform has translation only, so the quaternion is
   unchanged (`wxyz`). No guessed tool offset is introduced.
3. Both native EEF pose fields are sent. The inactive arm keeps the measured
   Cartesian link6 pose captured at handoff; IK may select different joint
   angles. This is not a guarantee of exact joint-position hold.
4. The official solver uses its native robot model, fixed table cuboid and
   self-collision checks. There is no added full-scene reconstruction or
   simulator object-pose input to UC71.
5. Native binary gripper commands, joint interpolation and ten 0.004-second
   physics substeps remain unchanged: 25 actions/second. On native IK failure,
   the previous commanded arm joints are retained and gripper commands still
   execute. UC71 records that action and the resulting measurements before
   reporting `ik_unavailable`; it neither undoes nor retries the action.

The target envelopes, action budget and measured arrival/gripper monitoring
remain adapter responsibilities. `transit`/`contact` retain target/tolerance
meaning; they do not impose different native speed profiles. Official requests
send the final target directly to native IK. Legacy DLS increments, speed and
joint-step caps are omitted from official controller capabilities and prompts.
The executive prompt and its diff are included in each new review packet.

## Dependency setup and validation

See [setup instructions and installed evidence](official_control_setup.md).
The verified task-local overlay is activated before Isaac/cuRobo imports, with
private Warp/CUDA caches. Package files, native source and the generated native
`curobo.yml` are hash-bound; loaded cuRobo/Warp module provenance is captured.

On 2026-10-30 the real pinned native `CuroboPlanner` initialized and completed
an FK-to-IK roundtrip on GPU 0. This CUDA-only check used no simulator, robot
actions or model calls. Its raw result is
[`ik_check_003/result.json`](../runtime/official_control/ik_check_003/result.json).
The earlier failed check is preserved: the upstream non-editable wheel omitted
its configuration data. Setup now copies the exact pinned content into the
overlay; no solver code was changed. Use `runtime/official_control/setup_v5.json`.
The final check uses exactly the production activation path, including native
cuRobo's own output cache. Repeated dependency integrity checks take about
11–12 ms on this host, with file and directory changes still checked.

Offline tests exercise conversion, both-arm hold, binary grippers, native
failure accounting, worker/parent transport, prompt disclosure and evidence
binding. They use synthetic native environments and do not establish physical
controller performance. Test logs are in
[`validation/official_control_20261030`](../validation/official_control_20261030/).
The current account cannot run the suite's root-owned isolation fixtures;
those failures are recorded separately rather than treated as successful tests.

## Native qualification and launch

Reset tooling accepts `--control-backend official_ee --official-setup PATH`.
It enables real native planners and keeps reset-only/no-policy scope; a reset
does not qualify motion or grant task authorization.

`prepare_runtime_bundle.py` defaults to the official backend and additionally
requires `--official-setup` and `--official-qualification`. The latter must be a
fresh `robodojo-official-control-qualification-1` record bound to the exact setup,
geometry, reset, monitoring and execution limits. Required measured checks
include both arms, TCP conversion, inactive Cartesian hold, gripper timing,
IK failure, native action/physics accounting, and host/GPU cleanup. Actual
native EEF receipts and their source hashes are required.

Installation, the CUDA-only IK check, a reset, or old Jacobian diagnostics cannot
satisfy that gate. New native motion/gripper qualification and a fresh reviewed
packet remain outstanding. No full Make Toast episode or live model call was
started by this implementation work.
