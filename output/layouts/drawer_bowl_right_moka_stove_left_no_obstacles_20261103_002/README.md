# Right-side drawer bowl, then left-side moka pot on stove — no obstacles

Task key: `drawer_bowl_right_moka_stove_left_no_obstacles`.

The cabinet and bowl are on the right side of the robot base (world +Y in
the external preview). The stove and single moka pot are on the left (world
−Y). In this revision the stove moves forward 11 cm and the moka pot forward
5 cm (world +X). The stove burner and cabinet are aligned side by side at
X = −0.08 m; their working fronts both face the robot (−X). The bowl starts inside the
closed top drawer. The stove starts off. The only movable objects are the bowl and
moka pot; there are no added obstacles or distractors.

1. Open the top drawer, take out the bowl and place it on the cabinet's top
   surface, then close the drawer.
2. Turn on the stove, then place the moka pot on its burner.

“Top of the drawer” is implemented as the top surface of the drawer cabinet.
The cabinet faces the robot, so the external camera sees its back. The drawer
opens toward the robot, clear of the other task objects.

## Previews

- [Initial external view](custom_external_1024.png)
- [Initial top-down view](custom_top_down.png)
- [Bowl revealed in the open drawer](drawer_open_diagnostic_top_down.png)

The open-drawer image is an **evaluator-only layout illustration**, made by
changing the drawer position and translating the contained bowl. It is not
the initial state and is not evidence of an agent opening the drawer.

## Files and loading

`custom_task.bddl` defines the new instruction, assets, initial relations and
four final predicates. `initial_state.npy` freezes the seed-7 scene.
`settled_state.npy` and `initial_observation.json` record the standard ten-step
passive reset. `load_task.py` loads this separate task using the official LIBERO
source checkout and the existing simulator environment.

```bash
cd /workspace/gpt_field/output/layouts/drawer_bowl_right_moka_stove_left_no_obstacles_20261103_002
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=4 \
  /workspace/venvs/libero-plus/bin/python load_task.py
```

```python
from load_task import load_task

with load_task() as (env, observation, instruction):
    # observation is raw simulator data: restrict policy input to the allowed
    # RGB and robot proprioception channels, plus the instruction.
    # observation, reward, done, info = env.step(action)
    ordered_success = env.check_success()
    final_arrangement_only = env.raw_goal_success()
```

The default wrapper checks the requested order: drawer opened → bowl placed on
cabinet top while open → drawer closed → stove turned on → moka pot placed on
stove. It also requires all four final predicates to remain true. Each stage
requires an observed transition after its prerequisite; early actions can be
retried. `raw_goal_success()` is LIBERO's unordered final-state conjunction.
Raw step rewards/done values are forwarded unchanged and must not be used as
the ordered completion signal. Predicate/progress diagnostics are evaluator-only.

State replacement invalidates ordered progress. Direct mutation through the
underlying `env.env` or `env.sim` is reserved for evaluator diagnostics and must
not be used to claim policy completion. `ordered=False` returns raw LIBERO for
these diagnostics.

## Validation and scope

Scene validation passes all 20 checks, including all 16 final-goal combinations,
14 sampled drawer positions, physical support, no initial unwanted contacts,
and passive stability. The bowl has about 10.46 mm clearance below the cabinet
roof. Two fresh 512-pixel resets reproduce the same policy observation hash.
See `scene_validation.json`, `layout_revision_audit.json`, and
`layout_manifest.json` for evidence. The revision audit includes a fresh check
of the default ordered loader and the stove/drawer alignment.

The closed drawer position is +0.001 m: it satisfies the stock strict positive
closed predicate without cabinet self-contact. The stove knob starts at
−0.0025 rad. The [original cabinet probe](../drawer_bowl_moka_stove_right_no_obstacles_20261103_001/closed_joint_probe.json)
records the tested drawer alternatives; this revision preserves that cabinet
and bowl configuration.

The previous left-stove layout is preserved separately; `revision_source.json`
records its file hashes and this requested change.

The robot reset is mapped by joint name from the previous t08 initial state.
Both policy cameras use 512-pixel images. The external camera retains its pose
with a 70° vertical field of view so the right-side cabinet fits in frame; the
wrist camera is unchanged. The loader uses stock OSC_POSE scaling. The prior
UC7.1 50 cm transit / 10 cm command-cap settings belong to its evaluator, not
this standalone scene loader.

No model calls or full agent episode were run. This is a new custom composite
task, not an official LIBERO benchmark task. The existing UC7.1 CLI accepts
only its registered task contracts; launching this new task through that CLI
will require explicitly adding this changed asset/goal contract and ordered
evaluator. Shared evaluation code and prior task artifacts were left unchanged.

Rebuild the scene and previews with the simulator Python and the same EGL
environment using `prepare_layout.py`. Run `validate_scene.py` for the separate
physical audit and `validate_loader.py` for loader/ordered-wrapper integration.
