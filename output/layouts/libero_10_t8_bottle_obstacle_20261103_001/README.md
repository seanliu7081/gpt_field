# t08: moka pots with a bottle obstacle

Custom layout for **`libero_10_t8` (zero-based task 8)**, `KITCHEN_SCENE8`: “put both moka pots on the stove.” The original task language, goals, objects, stove position and on-state are preserved.

A stock `ketchup` bottle stands 45% along the line from `moka_pot_2` to the actual stove cook-region center. Its initial root position is `[-0.00077523, -0.07048846, 0.97262]` m, with upright wxyz quaternion `[0.70710678, 0.70710678, 0, 0]`. Collision dimensions are approximately **5.62 × 3.68 × 14.56 cm**. The tabletop collision surface is at `z=0.90 m`.

## Previews

- [Custom external view, 1024px](custom_external_1024.png)
- [Custom top-down view](custom_top_down.png)
- [Original external view, 1024px](original_external_1024.png)
- Exact policy images: [external](custom_external.png), [wrist](custom_wrist.png)

## Reproduce the layout and previews

From `/workspace/gpt_field`:

```bash
python3 output/layouts/libero_10_t8_bottle_obstacle_20261103_001/prepare_layout.py
```

The script uses `/workspace/venvs/libero-plus/bin/python` and EGL GPU 3. It performs scene construction, the standard reset, and ten additional passive holds; it makes **no model calls or task-policy rollout**. Re-running overwrites the generated artifacts in this directory.

## Load the saved layout

Use **both** files together with the UserCheck simulator:

- `task_bddl_override`: [custom_task.bddl](custom_task.bddl)
- `initial_state_override`: [initial_state.npy](initial_state.npy)

Keep task `libero_10_t8`, **seed 7, episode 0, and 10 standard reset-settling steps**. The BDDL bootstrap region only constructs the model; the saved state defines the bottle's final placement. Original named-joint qpos/qvel values are mapped exactly into the expanded model. The added initialization is appended after the original entries to preserve seeded stove placement.

The expected post-reset policy observation hash is:

```text
7f96963551a64fc826f1cbf549e81497ddb1f40258ad9ee7438bb81dc5411080
```

[layout_manifest.json](layout_manifest.json) records file hashes, poses, fixture comparisons, and validation results. All ten layout checks passed: original pots and stove are unchanged, the stove remains on, the task is initially incomplete, and the upright bottle is table-supported without new object/robot contact or meaningful passive drift. The top-down camera is an evaluator-only preview and does not alter the saved policy observation.
