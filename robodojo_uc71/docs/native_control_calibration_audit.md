# Native robot and camera calibration audit

Status: **uncalibrated candidates; full-test READY is false**. No simulator was
launched, no robot was moved, and no model was called for this audit. Preview001
was launched separately by the supervisor. Its recorded robot state was analyzed
offline. Identity at the native end-link origin is not a validated physical
fingertip or grasp-center tool frame.

## Evidence and live mapping

Pinned RoboDojo: `f57e1633ffed59761bc8e568cfe7530c4b33381c`.
Pinned IsaacLab: `afca7b09d60d8beb9c1cb28b43066499940b969b`.
The installed camera/PhysX APIs are from Isaac Sim 5.1.0.0.
Recorded input: [preview001 robot inventory](../validation/new_host_20261029/native_preview_001/preview/robot_inventory.json),
SHA256 `e176f20200af6f55c4b5b971196ef33c97f6a53b95a9cc099a53f65954a0ae85`.

Both live articulations are fixed-base. Their actual joint order is joint1..8,
arm indices are `[0,1,2,3,4,5]`, and fingers are `[6,7]`. Body order is
`base_link,link1,link2,link3,link4,link5,link6,camera_base,link7,link8,camera`.
The full Jacobian shape is `[1,10,6,8]`: link6 body index6 maps to fixed-base
Jacobian row5. These are recorded facts for this initialization, not constants
to substitute for live name/index validation.

| Consumer field | Native measurement / interpretation |
| --- | --- |
| Six arm positions/velocities | `articulation.data.joint_pos/joint_vel[0, native_arm_ids]`, in radians and radians/second |
| Link pose | `data.body_link_pose_w[0, live_link6_id]`, world actor/link origin, quaternion wxyz |
| Arm limits | `data.joint_pos_limits[0, arm_ids]` and `joint_vel_limits[0, arm_ids]` |
| Raw Jacobian | `root_physx_view.get_jacobians()[0, body_id-1][:, arm_ids]` for the verified fixed base |
| COM offset | `data.body_com_pos_b[0, body_id]`, expressed in the actor/link frame |
| Finger measurement | `data.joint_pos[0, native_gripper_ids]`, metres; preserve measured values |
| Gap proxy | Sum of the two measured finger positions; not physical pad clearance or grasp success |

Preview001 limits are approximately ±10rad for joints1..5 and ±3.14rad for
joint6, with live arm velocity limits5rad/s. Fingers have `[0,0.044]m` position
limits and reported velocity limits1000m/s; that large source limit is not a
calibrated control speed. Both measured fingers were0.0439997055m at reset.
Native X5 actuator configuration overrides the much larger USD arm velocity
limits. Use the live arrays, not generic ±pi or URDF velocity assumptions.

Source anchors: [RobotManager](../runtime/native/RoboDojo/env/robot_manager/robot_manager.py)
(`get_link_pose`, `get_real_endpose`, `_setup_robot_key`, `_set_robot_obs_name`),
[X5 actuator configuration](../runtime/native/RoboDojo/env/robot_manager/robot_config/x5.py),
[IsaacLab articulation data](../runtime/native/RoboDojo/third_party/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py)
(`body_link_pose_w`, `body_com_pos_b`, joint limits).
`get_link_pose(is_relative=True)` subtracts the environment origin only; it does
not rotate into the robot base frame. The worker should use world measurements.

## The raw Jacobian needs a COM-to-link-origin correction

PhysX supplies world-frame rows `[vx,vy,vz,wx,wy,wz]`. Its linear component in
the recorded ARX state is at the link COM. The controller expects the actor
origin. With `c` the live COM position in link6 coordinates and `R` its measured
world rotation:

```text
J_link.linear = J_raw.linear + skew(R*c) @ J_raw.angular
J_link.angular = J_raw.angular
J_tool.linear = J_link.linear - skew(R*t_link_to_tool) @ J_link.angular
```

Do not rotate this world Jacobian into the base frame. The current controller
already applies the second equation for its declared fixed TCP translation;
feeding a pre-shifted tool Jacobian would apply that correction twice. Use
separate array indexing as above to avoid NumPy advanced-index axis reordering.
A floating-base articulation would require different body/column handling; the
new candidate adapter rejects it rather than assuming the fixed-base convention.

Offline check: pinned URDF FK at the recorded measured joints, using the
recorded root pose, matches both link6 positions within5.84e-8m and rotations
within1.73e-7rad. Central differences with joint epsilon1e-5rad and world angular
derivative `vee(dR/dq @ R.T)` give these maximum absolute residuals:

| Comparison, both arms | Residual |
| --- | --- |
| Raw linear Jacobian against link-origin FK derivatives | 0.04169704m/rad |
| COM-corrected linear Jacobian against those derivatives | 7.22e-8m/rad |
| Angular Jacobian against world rotation derivatives | 3.73e-7rad/rad |

Original URDF and USD both declare link6 COM
`[0.041697,0.000024368,0.00014464]m`; USD principal axes are identity. The runtime
adapter reads the actual live COM offset instead of embedding that number. This
is strong evidence for the reference-point correction at one recorded pose;
it does not replace multi-pose engine finite differences or actuator tracking.

## Physical tool and finger conventions remain to be validated

[robot_config.yml](../runtime/native/RoboDojo/Assets/Robots/x5/robot_config.yml)
sets `ee_link=link6`, `delta_matrix=I`, and `gripper_bias=0.145`. Native
`get_real_endpose` reads link6 directly. `get_delta_endpose` and
`_trans_from_endlink_to_gripper` change orientation only; neither translates by
that bias. Therefore0.145m is not an established grasp TCP offset.

Both finger coordinates increase when opening. Joint7 moves along link6 +Y;
joint8's USD joint-frame rotation reverses its effective axis to -Y, matching
the URDF. Their rest origins already differ by about0.049796m in Y, before pad
geometry. Native normalized open=1 maps both joints to0.044m; close=0 maps to
-0.01m although the physical lower limit is0. This command target is not the
measured finger position. Preserve native mapping and report actual measurements.

The present candidate explicitly declares the native link6 actor origin with an
identity transform. It is useful for geometry diagnostics only. The execution
prompt requests a calibrated tool frame, so this identity candidate must never
be promoted to final model-facing tool calibration merely because FK matches.
Next derive a physical point/axes from the two authored finger-pad contact
surfaces, including mesh and joint transforms, bind the exact geometry hashes,
and verify the resulting midpoint/axes against actual projected robot landmarks.
An arbitrary mesh bounding-box center or the unused bias is insufficient.

There is also a contract boundary: [schema.py](../src/robodojo_uc71/schema.py)
accepts a full `link6_from_tool` SE(3), but [controller.py](../src/robodojo_uc71/controller.py)
currently supports translation only and uses link6 orientation. A validated
tool with nonidentity rotation needs an explicit controller/schema update and
tests. The new candidate adapter rejects nonidentity tool transforms.

## Camera metadata must come from the actual Camera objects

Map through `env.camera_manager.camera_names[0]` to `cameras[0]`; do not assume
list order. The reviewed names are head, left wrist, and right wrist, each
native640×480. The native head is unmounted; wrists mount to
`robot0/camera` and `robot1/camera`. Their actual child camera prim, not just its
parent Xform, defines the view.

Use `Camera.get_intrinsics_matrix(device="cpu")` and
`world_from_optical = inverse(Camera.get_view_matrix_ros(device="cpu"))` after
the same native render synchronization used for RGB. Optical axes are +X right,
+Y down, +Z forward. The SDK method applies the USD-to-optical axis conversion
and handles USD row-matrix transposition. Default `Camera.get_world_pose()` uses
a different camera-axis convention and should not be substituted without an
explicit conversion.

Two upstream convenience methods are unsuitable for direct schema binding:

- `CameraManager.get_camera_intrinsics` forces fy=fx. The head template has
  focal10, horizontal aperture22.212 and vertical aperture14.266 at640×480, so
  the live API should produce distinct fx≈288.1325 and fy≈336.4643. Wrist D435
  template apertures give fx≈fy≈397.0413. Record actual values, not these checks.
- `CameraManager.get_camera_extrinsics` returns the parent Xform's USD pose;
  it does not return the child camera's optical frame.

Source anchors: [native CameraManager](../runtime/native/RoboDojo/env/camera_manager/camera_manager.py):
camera creation310–332, intrinsics454, extrinsics469;
[native camera templates](../runtime/native/RoboDojo/env_cfg/camera/template.py);
installed `isaacsim/sensors/camera/camera.py`: `get_view_matrix_ros`1664,
`get_intrinsics_matrix`1703, `get_image_coords_from_world_points`1745.
The candidate records a fixed head frame but does not claim that constancy or
pixel alignment has been tested. Wrist world extrinsics must be refreshed with
each synchronized robot/image snapshot.

Native [render_sync.py](../runtime/native/RoboDojo/env/camera_manager/capture/render_sync.py)
can fall back to counting render calls when dispatcher timestamps are missing,
and may return at its bound without raising for stale delivery. Thus a returned
capture call alone does not certify image freshness. Validate projected visible
robot landmarks, image row/channel orientation, timestamps, and wrist movement.

## Implemented candidate API and remaining bounded probes

[native_measurements.py](../src/robodojo_uc71/native_measurements.py) exposes
`native_link6_tool_candidate(native_root)`,
`NativeMeasurements(env, tool_candidate=...)`, `.sample()` and `.arm_states()`.
`sample()` returns both ArmStates and measured finger pairs together, plus
operator-only provenance. `require_ready()` always rejects this uncalibrated
candidate. It has no control, policy, scene-object, contact, or reward calls.
`capture_camera_inventory(env)` records optional camera candidates; extraction
failures remain separate from RGB success. Raw robot inventory now includes
`body_com_pos_b` without applying an inferred TCP.

The preview additionally supports an explicitly authorized300-second pause.
It reads raw `root_physx_view.get_dof_positions()` before and after (avoiding
cached joint positions), checks native time/counters/state equality, and emits
Python log heartbeats only during sleep. It performs no Kit update, render,
physics step, model call, or benchmark action in that interval, then recaptures
the final three RGBs. Offline mocked-clock tests are evidence of implementation,
not evidence that the live300-second pause has passed.

Remaining setup work, before any full-test approval is consumed:

1. Capture the live COM and camera inventories, confirm exact module provenance,
   and finish the bounded paused-physics/re-render check under the supervisor.
2. Define the physical tool from pinned finger-pad geometry and document its
   transform; reconcile the translation-only controller if axes differ.
3. In a separate model-free calibration session, verify all six Jacobian columns
   at several robot-only poses. Use symmetric small measured joint changes and
   world quaternion differences; compare actor and tool point predictions. Do
   not count commanded displacement as measured displacement, or use a stale
   cache before a native synchronization boundary. Preserve failed probes.
4. Validate real finger open/close signs, measured displacement and stable-gap
   thresholds without asserting contact/grasp truth. Validate Cartesian and
   angular tracking, speed/lead limits, arrival/stall thresholds, both arms,
   inactive-arm hold, and latched grippers at native25Hz/10×0.004s timing.
5. Check rendered robot landmark projections in all cameras and wrist poses,
   and fresh reset repeatability. Keep evaluator/object truth out of model input.
6. Bind those measured results into the final calibration/readiness artifacts;
   integrate the main-thread worker, one acknowledged native action per ticket,
   privacy-filtered observations and terminal handling. The missing upstream
   camera-stand normal texture remains a separate Standard asset-fidelity gate.

Offline regression result at this handoff:44 focused measurement, controller,
and preview tests pass. No live calibration or full benchmark result is claimed.
