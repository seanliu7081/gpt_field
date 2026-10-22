# robodojo_direct

Independent zero-shot baseline implementing RoboDojo's generic absolute end-effector tool loop on the existing LIBERO Panda environment. Historical Direct5, Field, and SpeedUp implementations are unchanged.

## Interface and execution

GPT-6 Astra (`medium`) uses the existing ChatGPT subscription OAuth transport, with no API-key or model fallback. Each turn produces exactly one native Responses `move_eef` or `give_up` call, grouped in the `robot` tool namespace supported by the subscription route. `move_eef` fields are absolute world `x/y/z` in metres, `pitch_deg/roll_deg/yaw_deg`, and `gripper` (`0` closed, `1` open). Null arm fields preserve the measured pose. A null gripper preserves the previous command, including squeeze while holding an object. RoboDojo's angle names are retained: `Rz(yaw) Ry(roll) Rx(pitch) Rx(pi)` for the Panda downward reference. Position already describes LIBERO's grip site; orientation uses the measured `right_hand` frame.

Static Panda/gripper XML and measured arm joints provide robot-only forward/inverse kinematics. The complete Cartesian path is checked before motion. No IK solution found rejects the tool without physical execution; this does not prove global unreachability. Accepted targets are interpolated and tracked by native 20 Hz OSC, with measured position/orientation feedback. A new jaw command runs only after measured arm arrival. A local timeout returns actual error and skips a new jaw command. The maximum per-tool budget is 170 arm steps plus 30 jaw steps, at most 10 simulated seconds. There is no scene collision planner, visual servo, contact detector, grasp skill, automatic retry, or learned motor policy in this adapter.

Two native 512-pixel upright RGB views, the grasp-point state, and seven measured arm joint angles are supplied. All textual observations, accepted native outputs and robot-motion results are retained; only the newest two image turns remain. Task-stage descriptions are instructions, not demonstrations. Object state, depth, scene calibration, reward, collision identities and benchmark success never enter model observations.

## Evaluation

One fresh run each: `bowl_plate` = LIBERO-Spatial task 0; `drawer_bowl` = LIBERO-Goal task 3. Both use official initial state 0, seed 7, ten reset settling steps. Initial RGB/state hashes are checked against the existing 512-pixel references. Collisions are recorded separately; raw LIBERO success is the sole success criterion. `give_up` ends an unsuccessful attempt. Existing user policy retains no whole-episode call/action/wall-time cap, unlike RoboDojo's finite call budget.

Artifacts include immutable source snapshots, exact requests/responses, every executed action and observation, robot-only tool results, simulator metadata, video, and summary. `control_without_video_s` subtracts only recorder overhead from control wall time; startup/reset and final video encoding are outside that control interval. Model request pipeline time (including failed or interrupted waits), local planning, local control and simulator time are separately recorded. Request attempts and accepted model responses are counted separately. A duration is a successful completion time only when `completion_time_eligible=true`.

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_robodojo_direct.py --preflight --task bowl_plate
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/evaluate_robodojo_direct.py \
  --task bowl_plate --output runs/robodojo_direct_bowl_zero_shot_001/trial \
  --store-dir /workspace/gpt_field/.private/chatgpt_plan
```

Use `--task drawer_bowl` with a separate fresh directory for drawer. `--sim-smoke` uses only scripted diagnostic moves and never reads subscription credentials. Touch `<output>/operator_stop.request.json` to stop at the next complete physical-step boundary; SIGINT also interrupts a pending model request. No run is resumed or overwritten automatically.

## Reproduction boundary

Reference: [RoboDojo report](https://arxiv.org/html/2609.24170v1) and [XPolicyLab implementation](https://github.com/XPolicyLab/XPolicyLab/tree/562c14cf08b9e6e411e7d76f192740afced77160/policy/GPT_6_Astra_Direct_EEF), pinned commit `562c14cf08b9e6e411e7d76f192740afced77160`.

This is a **LIBERO OSC adaptation**, not a CuRobo reproduction. The original uses a dual-arm ARX embodiment, three cameras, geometric grasp-to-flange conversion and CuRobo joint trajectories at 25 Hz. This adapter retains a single Panda, two cameras and the established 20 Hz physics, with robot-only IK and Cartesian/rotation tracking. It also waits for measured arm arrival before applying a new jaw command; a timed-out move reports non-arrival. These differences are explicit in each manifest and preclude treating cross-benchmark success rates as a matched comparison.
