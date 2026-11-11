# Foundation harness

This opt-in implementation connects one complete manipulation loop: calibrated
public RGB and robot joints → SAM/DA3 surface hypotheses → Astra semantic
contracts and executable keyposes → local planning toward the next keypose →
the existing measured robot controller
→ local visual outcome feedback. It leaves existing runners and historical runs
unchanged. Implementation does **not** authorize a full-task launch.

The current revision is `foundation-harness-keypose-2` with runner version
`foundation-harness-runner-2`. Astra owns manipulation pose generation. Geometry
supports execution and diagnosis; an unavailable complete reconstruction or
complete pick-and-place route does not prevent proposing an initial approach.
This revision has not been validated in a full task.

## Entry and authorization

`scripts/evaluate_foundation_harness.py` defaults to `inspect`. Inspection reads
local configuration and checks paths; it does not read credentials, load vision
models, reset a simulator, or call Astra. A run requires both the explicit `run`
command and `--approved`, after the user approves the concrete full task.

```bash
PYTHONPATH=src .venv-foundation-vision/bin/python scripts/evaluate_foundation_harness.py inspect \
  --task TASK_PROFILE --episode-index 0 \
  --sam-checkpoint /absolute/path/to/sam_checkpoint.pt \
  --da3-model /absolute/path/to/local_da3_model
```

For a user-approved single full task, replace `inspect` with `run --approved`
and add `--output /absolute/path/to/a/fresh/run_root`. Custom scenes/layouts retain
the existing `--task-bddl-override`, `--initial-state-override`, and matching
`--expected-initial-observation-sha256` requirements. `--sim-python` and
`--libero-root` select the existing simulator environment. The run creates
`trial/`; an existing run root is refused. There is no automatic episode retry.

This entry currently targets the existing LIBERO/Panda public simulator adapter.
It does not silently adapt the fixed-base motion model to a mobile robot.

The task uses Astra high reasoning over the existing ChatGPT subscription path,
with no API-key fallback and no episode time, action, or model-call cap. Local
operation bounds remain control events, not task deadlines. SAM/DA3 are loaded
from local weights before authentication and reset. The first paid request is the
actual semantic request after initial perception; there is no discarded legacy
initial-plan request. The first request remains part of control timing.

To stop a running episode, create
`RUN_ROOT/trial/operator_stop.request.json`. A stop arriving during inference
withholds the returned decision before dispatch. Raw evaluator completion and
record-only collision accounting remain with the existing runner. A model's
`finish` claim or a local operation result does not establish task success.

## Contracts and execution

`src/gpt_field/foundation_harness/client.py` owns the strict task/keypose schema and
validation. The client receives two current original RGB views, public camera
calibration, measured robot state, the perception summary, and local outcome
history. It retains accepted text and the six most recent image pairs through
the existing subscription transport. Dense scene geometry stays local.

The response contains `status`, `why`, `targets`, `operations`, and
`completion_conditions`:

- Targets have a stable `id`, a short segmentation `query`, current candidate
  selections for `external`/`wrist`, and optional `region_pixels` boxes in each
  original image. Null values preserve unknown or unavailable evidence. A region
  identifies a visible surface patch, not a full object volume.
- Operations are `pick_place`, `grasp`, `place`, `move`, `contact_move`, `release`,
  or `observe`. Object and support are target IDs. `grasp_region` is the target ID
  of a separately selected material region, or null for the object's surface.
  `placement_relation` distinguishes `on` from `inside` and preserves the intended
  outcome throughout partial execution. Reaching a pose is not evidence that the
  requested relation holds.
- `contact_mode` is `free_object` or `articulated`. Articulated grasp does not
  automatically lift. A later `contact_move` explicitly states displacement or
  keyposes;
  visible mechanism movement is engagement evidence, not whole-fixture attachment.
- **Astra generates executable keyposes** from RGB, public calibration and any
  usable geometry. Missing SAM masks or DA3 target geometry do not prohibit a
  proposed pose. The model can propose a closer approach/viewpoint and inspect
  its outcome without first obtaining a complete object shape.

Each operation includes a `keyposes` array of zero to twelve entries. Every
entry has these required fields:

| Field | Meaning |
| --- | --- |
| `name` | `pregrasp`, `grasp`, `close`, `lift`, `transport`, `preplace`, `place`, `release`, `retreat`, or `move` |
| `pose.position` | Three absolute world coordinates in metres, locating the grip site |
| `pose.quaternion_wxyz` | Four normalized components of the absolute right-hand orientation |
| `gripper` | `open`, `close`, or `keep` |
| `profile` | `transit` or `contact` |
| `observe_after` | A checkpoint requiring fresh evidence before continuation |

`grasp`, `pick_place` and `place` require a nonempty array, but **do not require a
complete operation chain**. A single pregrasp pose ending with
`observe_after=true` is a valid partial plan. A longer understood sequence can
contain up to twelve poses; there is no one-to-three-pose preference. Stage names
express intent. They do not automatically close/open the gripper or establish
contact, attachment, release or completion.

`move` and `contact_move` use exactly one representation: nonempty `keyposes`,
`goal_pose`, or `displacement`. These representations cannot be mixed in an
operation. A displacement is relative to the measured operation-entry position
in world axes; pose positions remain absolute. Other kinds use null
`goal_pose`/`displacement`. `release` and `observe` may have empty `keyposes`.

The local planner addresses the **next keypose** from current measured state.
It does not require a feasible future placement before starting an approach.
An observation checkpoint ends the current committed portion: remaining poses
are unexecuted proposals whose continuation depends on fresh feedback. Partial
sequence completion must not be reported as completed grasp, transfer or
placement. In particular, reaching pregrasp does not mean the object is held.

Unknown or occluded space remains explicitly unknown. The revised policy uses
bounded motion followed by observation in such space, rather than treating
unknown coverage alone as a blanket motion prohibition. The bound limits the
next commanded step; it does not certify clearance. New observed obstacles,
motion discrepancies and geometric uncertainty remain evidence for replanning.

The scene retains both strong and uncertain surface hypotheses. Cross-view
supported points without contradictions, and points belonging to an object's
coherent planning surface, are strong obstacle evidence. Other predicted points
remain uncertain; intersections are counted and shown to Astra, rather than
silently converted to free space. Strong intersections, self-collision and joint
limits still reject motion. Video displays the planner's self-filtered scene:
blue denotes strong surface evidence and grey denotes uncertain predictions.

Self masking uses current measured robot configuration and static visual meshes
projected into each camera. Predictions behind that robot silhouette are treated
as self/occluded evidence; assigned object points and predicted foreground are
preserved. The visual depth envelope uses 98 supporting planes per body for fast
ray checks. It does not replace collision meshes or use future poses to erase
obstacles. `surface.py` derives coherent object patches using source-view density,
normals and connectivity; raw observations and provenance remain archived.

Planning/execution failures retain their structured reason and available inner
evidence in logs and planner feedback. IK failure, predicted collision, unknown
coverage, tracking uncertainty and operation outcome are distinct diagnoses.
They must not all collapse into an opaque `no_complete_operation_path` string.
The model should receive enough context to change the next pose or obtain a new
view instead of repeating an unchanged operation.

Bootstrap queries come only from generic grammar applied to the official task
sentence. They are weak perception seeds, not a task-specific plan. Astra can
replace them with short queries or use `observe` to resolve candidate identity
before requesting manipulation when identity is unresolved. A missing mask alone
does not require a query-only call. Local outcome handling and exceptional
replanning are implemented in `executor.py`; model descriptions remain estimates.

## Evidence and limits

SAM scores are not calibrated probabilities. DA3 estimates visible surfaces;
unseen or occluded space remains unknown. A partial-surface centroid is not an
object centre or a grasp point. Model-proposed keyposes, IK solutions and
collision checks are hypotheses, not guarantees of contact, grasp,
support, or collision freedom. Gripper closure and pose arrival alone do not
prove attachment. The controller must use fresh local observations and preserve
uncertain outcomes.

The Panda FK/IK model comes from static robot assets and measured joints, with
the simulator Python used only to locate the installed assets. No scene object
poses or evaluator contact identities enter policy. Full-body/held-object
geometric checks are bounded by the implemented robot model and observed scene
coverage; predicted clearance is not a physical safety certificate.

The runner archives current source/configuration, dependency versions, local
model provenance, inputs and terminal source-integrity results. It preserves
the existing raw request/response logs, evaluator trace and measured video cost.
`client_elapsed_s` measures payload/auth/transport/parsing/logging and excludes
local perception; it is not pure model compute time. Completion under three
minutes is a performance objective to measure on complete tasks, not an automatic
cutoff or an achieved result of implementation.

The optional recorder keeps both RGB panels and adds XY/XZ/ZY projections of
bounded predicted surface samples, current local keyposes, planned and measured
EEF paths. The frame index stores the same planning snapshot. Yellow geometry is
planned; pink is measured. These displays do not assert full scene visibility or
collision freedom. Video playback omits API waits and is not wall-clock timing.

Configuration records make the revised responsibilities explicit:
`keypose_owner=astra`, `planning_scope=next_keypose`,
`unknown_space_policy=bounded_step_with_observation`, and
`geometry_hypotheses_not_collision_certificate=true`. These fields describe the
method; they do not record a successful evaluation.

## 本次修订状态

此前 `runs/foundation_harness_t04_can_obstacle_001` 已由用户停止，
原始 LIBERO 判定未完成；其封存源码与运行记录保留为历史证据。
此前单次任务的授权不延续为本次修订的重试授权。

本次修订把关键位姿生成交回 Astra，局部执行按下一关键位姿规划，
允许接近后观察的部分序列，并保留结构化失败原因。
这次修订尚未进行完整任务验证，不能声称成功率、抓取效果、碰撞率或速度收益。

精简集成文件的五项检查通过；后续实际受影响的执行、规划、视频项单独重查通过。
保存观测上的两个定点检查已生成轨迹：原先受阻的 5 cm 换视角移动约 1.98 秒，
旧成功版本的首次 Astra 接近位姿约 2.07 秒。后者仍报告不确定几何交叠，
这些是离线连接结果，不是新的机器人执行或完整任务成功。
黄白杯的规划表面从原始 16,677 点提取出 16,095 点，XY 范围从约
26.3×23.4 cm 收缩到 12.3×14.2 cm；这不是物体真实尺寸的测量。
证据和之前的失败诊断保存在
`output/foundation_harness_keypose_v2_preparation_001/`。

当前授权是实现修订，不包含模型调用、模拟器 rollout 或新的完整任务。
下一次完整任务仍需在说明最终改动后获得用户明确批准。
本文及 `inspect` 输出均不构成启动授权；启动逻辑仍要求显式
`run --approved`，使用新的运行目录并保留原始评估和录像。
