# t08 obstacle task: turn on the stove and place both moka pots

**Instruction:** “turn on the stove and put both moka pots on it.”

Task key: `libero_10_t8_stove_off_bottle_obstacle`.

This is a new custom task derived from
[`libero_10_t8_bottle_obstacle_20261103_001`](../libero_10_t8_bottle_obstacle_20261103_001/README.md).
The bottle obstacle, both moka pots, stove location, robot state, cameras, and
controller are preserved. The stove starts **off**, with button angle
`-0.0025 rad`. Both the BDDL initialization and saved simulator state specify off.

Success requires all three conditions simultaneously: stove on, moka pot 1 on
the stove cook region, and moka pot 2 on that region. The original t08 already
had this final conjunction; the new starting state makes turning on the stove
an action the robot must perform. This uses LIBERO's final-state semantics and
does not impose an additional temporal ordering constraint.

## Files and previews

- [Task definition](custom_task.bddl) and [initial state](initial_state.npy): use together.
- [Stove-off external preview](custom_external_1024.png), [top-down preview](custom_top_down.png), and [wrist preview](custom_wrist.png).
- [Source stove-on preview](source_external_1024.png) for comparison.
- [Layout manifest](layout_manifest.json) and [goal validation](goal_validation.json).
- [Loader](load_task.py) and [reproducible builder](prepare_layout.py).

## Load the new task

From this directory, perform a reset-only smoke check:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 PYOPENGL_PLATFORM=egl \
  /workspace/venvs/libero-plus/bin/python load_task.py
```

Use the same Python environment for a controller integration:

```python
from load_task import load_task

with load_task() as (env, observation, instruction):
    # instruction is read from the new BDDL, including "turn on the stove".
    # observation contains raw LIBERO channels; select RGB and robot state
    # for policy inputs, keeping object truth evaluator-only.
    assert instruction == "turn on the stove and put both moka pots on it"
    assert not env.check_success()
    # observation, reward, done, info = env.step(action)
    # success = env.check_success()
```

The loader uses the official LIBERO checkout at
`/workspace/heading_flow_pub/third_party/LIBERO`, seed 7, the saved state,
ten passive settling actions, Panda / OSC_POSE, and 20 Hz control. Rendering
defaults to 512 pixels; `load_task(render_size=128)` also supports policy RGB.
Choose an available EGL device when running elsewhere. `settled_state.npy` is
validation evidence; the loader uses `initial_state.npy` before settling.

This task is separate from the official `libero_10_t8` registry entry. UC7.1 now
accepts it explicitly as `--task libero_10_t8_stove_off_bottle_obstacle`, with
the new instruction checked against the BDDL. The original official task and
layout-only override contract remain unchanged. For the requested test,
`GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M=0.10` sets a **10 cm per-step command cap**;
the **transit target envelope is 50 cm**. The UC7.1 worker applies the matching
OSC scale after reset; this standalone loader retains stock OSC scaling.

The frozen PI05 weight-space evaluator hardcodes the original task UID 38, so
changing its layout path alone does not condition that policy on the new
instruction. Existing checkpoints and experiment artifacts remain unchanged.

## Validation and reproduction

All **18 scene/reset checks** passed. Only the stove-button qpos differs in the
saved state; every other raw and settled state value, all pot/bottle poses, and
the stove fixture/cook region remain identical to the source. The bottle is
upright, supported, and stable under the passive hold. Two independent resets
produce the same full policy observation hash:

```text
863d4e72bcb3cda4164625bd0043163c276b64e63cd1f4053add8774710d5561
```

All **eight goal truth-table cases** passed in the real simulator. Stove on
alone, one pot plus stove on, and both pots with stove off all fail; only both
pots plus stove on succeed. These are evaluator-only counterfactual states,
restored after checking, and are not robot demonstrations or task rollouts.

Rebuild and revalidate from this directory:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 PYOPENGL_PLATFORM=egl \
  /workspace/venvs/libero-plus/bin/python prepare_layout.py > build.log 2>&1
```

This regenerates this directory's derived artifacts, reading the source layout
and its rendering helpers without modifying them. The manifest records hashes
of every source file and verifies them afterward. Validation makes no model
calls and runs no task policy.
