# Model-input contract: LIBERO uc7.1 → native RoboDojo

The user selected **the same information types**, with **two native ARX arms and all three RGB cameras**. Sensor count and robot dimensions change with the embodiment; access to task/environment truth does not expand. Initial planning and execution are different request types and must be checked separately.

| Information | LIBERO uc7.1 basis | Proposed RoboDojo mapping / restriction |
|---|---|---|
| Task text | Official reset task instruction | Exact native `gen_instruction`; no evaluator-derived additions |
| Initial planning images | Two clean RGB views plus their two grid-marked versions and existing review orientation/status metadata | Three clean head/left-wrist/right-wrist RGB views plus three grid-marked versions (six planning images); camera selectors and wrist orientation labels; preserve planner-only fields |
| Initial planning state/calibration | Full execution calibration/state is not present in the ordinary planning payload | Do not add it to the initial planning prompt |
| Execution RGB | External and wrist,512×512 delivery | Head and both wrists; synchronized, declared transforms; native resolution/crop/padding recorded |
| Measured robot state | EEF pose, finger state, arm joints/velocities where the existing execution schema exposes them | Actual poses and joint/gripper channels for both ARX arms; measured versus kinematically derived values labeled |
| Model-visible geometry | Allowlisted camera calibration and fixed tool/gripper geometry with provenance | Equivalent ARX tool/gripper calibration fields; no fabricated Panda channels or added full-body geometry |
| Controller-internal robot model | Robot kinematics/control machinery is not automatically part of Astra's geometry payload | Full ARX URDF/link geometry, kinematic chains and complete joint limits remain internal to robot-only control; do not forward them wholesale |
| Camera calibration | Intrinsics, external transform, wrist mount and current robot-derived transform | Actual three-camera calibration adjusted to delivered pixels; no scene/object calibration |
| Directions | Frozen external-camera `external_view_v1` | One frozen head-camera frame; neither wrist redefines world/image left/right |
| Approved plan | Review-bound goal contract/constraints | Same authority; any arm-aware schema change explicitly versioned |
| Execution feedback | Robot-observed motion/error, gripper state and generic primitive status | Same evidence categories for selected arm; other arm's measured/held command state explicit |
| History | All executive text; latest six image-bearing observation turns | All accepted text; latest six synchronized three-view sets (18 images), no adaptive extra views |
| Across attempts/tasks | New client and empty execution history | Same; old traces/results/lessons never automatically loaded |
| Reasoning transport | Existing opaque accepted response items | Preserve unchanged; do not inspect or synthesize hidden reasoning |
| Hidden scene state | Evaluator only | No object poses/IDs from simulator state, bounding boxes, object frames, semantic sites or velocities in policy payloads |
| Other vision channels | Not used | No depth, point cloud, segmentation or inferred ground-truth image overlays |
| Contacts/support/held truth | Independent evaluator only | Never send contact identities, exact grasp detection, support labels or reward-derived completion hints |
| Remaining native budget | No equivalent remaining-episode-budget field in the uncapped LIBERO contract | Evaluator/operator only by default; do not forward the upstream reference policy's remaining-step field into Astra |
| Reward/success details | Evaluator controls stopping; not an answer source | Native reward/partial score/numeric criteria separated from model observations and controller feedback |
| Demonstrations | Not used in these reviewed tests | No teleop videos/trajectories, report solution videos, training examples or oracle waypoints |
| Planner scene model | No privileged object-truth planner | Robot-only kinematics; no simulator scene collision world or object-specific grasp/lever target injection |

The human-facing plan may describe scoring so the user understands the experiment. That description must not be concatenated into Astra's instructions. In particular, the exact remaining-bread count, return-home check and numeric task thresholds are not to be imported from `run_reward` into the model's task text. Record any public benchmark-wide instruction separately if later discovered; compare it to LIBERO parity before using it.

Implementation invariants:

1. Initial-planning and execution payload builders accept explicit allowlisted schemas, not a raw native observation dictionary.
2. Native reward/hidden object state and their derived labels cannot reach requests, local control targets, error messages or model-visible feedback through helper calls.
3. Camera metadata has synchronized timestamp/state and real robot/optical conventions. No alias of ARX as Panda and no artificial zero-padded measured state.
4. Active-arm selection is explicit. Each arm's motion results and gripper command belong to that arm. The head camera defines shared directions.
5. Six retained observation turns means all three views from each of six turns; do not trim one wrist inconsistently. Preserve older text with image-omission markers.
6. A fresh attempt's first request has no accepted responses from any previous run. A reused initial planning draft is allowed only with exact version/layout/sensor/config bindings and explicit provenance, not execution-memory transfer.
7. Approval binds complete sensor geometry and both arms in addition to legacy initial-observation hashes. Runtime sources, adapter diffs and model settings remain frozen during an approved attempt.
8. Unit/contract tests inspect both successful and error payloads. Simulator diagnostics may use ground truth for checking correctness but must be labeled offline and must never become model demonstrations or tuned task trajectories.

Local source basis: [execution history and payload](../src/gpt_field/agent_speedup_v2_user_check_agent_client.py), [reviewed runner](../scripts/evaluate_agent_speedup_v2_user_check.py), [current test protocol](../UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md). The implementation must use the frozen uc7.1 release, not silently inherit unrelated later changes from the shared tree.
