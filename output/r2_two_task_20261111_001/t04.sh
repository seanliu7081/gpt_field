#!/bin/bash
set -e
export WORKSPACE=/workspace
. /opt/supervisor-scripts/utils/environment.sh
. /opt/supervisor-scripts/utils/logging.sh /workspace/gpt_field/runs/foundation_vision_harness_V3_t04_can_obstacle_009.run.log
export PYTHONPATH=/workspace/gpt_field/output/r2_two_task_20261111_001/runtime/src
export PYTHONDONTWRITEBYTECODE=1 GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M=0.10 CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
cd /workspace/gpt_field/output/r2_two_task_20261111_001/runtime
exec /workspace/gpt_field/.venv-foundation-vision/bin/python /workspace/gpt_field/output/r2_two_task_20261111_001/runtime/scripts/evaluate_foundation_vision_harness_V3.py run --approved --task libero_10_t4 --episode-index 0 --vision on-demand --output /workspace/gpt_field/runs/foundation_vision_harness_V3_t04_can_obstacle_009 --task-bddl-override /workspace/gpt_field/output/r2_two_task_20261111_001/inputs/t04/custom_task.bddl --initial-state-override /workspace/gpt_field/output/r2_two_task_20261111_001/inputs/t04/initial_state.npy --expected-initial-observation-sha256 06f623f31ee2ea2e8ef600ce29d299413d81a8647d8f59663f3aff2e660da187 --sam-checkpoint /workspace/gpt_field/artifacts/foundation_vision/models/sam3.1/sam3.1_multiplex.pt --da3-model /workspace/gpt_field/artifacts/foundation_vision/models/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c --store-dir /workspace/gpt_field/.private/chatgpt_plan
