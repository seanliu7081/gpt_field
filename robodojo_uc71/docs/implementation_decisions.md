# Implementation decisions — 2026-10-29

This is the decision record at creation. Later scoped licence acceptance and the failed A0 renderer check are recorded in [current implementation status](../IMPLEMENTATION_STATUS.md).

Implementation is authorized for native `make_toast` Standard, seed-directory 0 / layout ID 0. The full test remains unstarted until its concrete preview and plan receive separate approval. The unchanged reviewed documents are in [planning_archive_20261029](../planning_archive_20261029/); [the JSON record](implementation_decisions.json) is the structured decision record.

| Decision | Implementation choice |
|---|---|
| D1: completion instruction | Add one disclosed, task-agnostic public rule: “A task counts as complete only when both arms are back at the pose they started the episode in.” Preserve the native task text. Do not add numeric success thresholds, task recipes, hidden retreats or automatic homing. Version and review the prompt addition. |
| D2: native horizon | Keep the official 1,400-action hard stop. Each action is one per-environment action dictionary, expanded to ten 0.004 s physics steps at 25 Hz: 56 s simulated policy motion. Keep remaining-budget information out of Astra. Uncapped continuation is disabled unless separately and explicitly approved. Preregister measured controller timing before a model attempt. |
| D3: images | Deliver synchronized native 640×480 head, left-wrist and right-wrist RGB without crop, resize or square padding. Version the rectangular grid/schema/geometry changes and verify actual dimensions, optical orientation and calibration. |
| D4: live smoke | A live execution smoke needs its own concrete test approval and is disabled now. Model-free implementation checks do not authorize model-driven actions. No automatic task retry, including after a zero-action infrastructure failure. |
| D5: runtime | GPU 4 is the candidate, UUID `GPU-e1a460e9-e272-b227-222e-0dbb77c248c5`, PCI `81:00.0`. This is inventory evidence, not verified Kit/CUDA assignment. EULA acceptance is pending; this record does not accept it. After acceptance, verify the single-GPU renderer/physics assignment with a model-free RTX camera smoke before task assets or execution. |

E1 correction: the archived review overstates command echoing. Native arm joint state is measured through `get_joint`; only `ee_joint_state` keys enter the `prev_control` overwrite. Native gripper state is therefore command-derived, not evidence of a measured opening or grasp. Read six measured arm joints/velocities and both real finger joints per arm, and label measurements separately from requested commands. The cached pinned [observation source](source_evidence/native_obs_manager.py.txt) verifies this distinction.

At arm handoff, freeze the inactive arm's **measured pose** as its hold reference. Keep its **last requested gripper command** latched. Every native action carries both arms. Do not keep chasing a stale motion target after a stall or echo observed opening back as the gripper command.

The reviewed native commit, submodule gitlinks, HF asset revision, exact layout hash and cached evidence are recorded in [source_pins.json](../configs/source_pins.json). These pins are not an installation claim. Checkout validation, asset dependency closure, package hashes, GPU identity and native rendering remain implementation checks. Files under `source_evidence` are provenance/evaluator material and must never be included in policy payloads.
