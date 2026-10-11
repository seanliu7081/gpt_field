import math

import numpy as np
import pytest

from gpt_field.action import ActionAdapter


STATE = [0.2, -0.1, 0.9, 0.0, 0.0, 0.0, 0.04, -0.04]
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def test_physical_translation_scale_and_gripper_memory():
    adapter = ActionAdapter()
    assert adapter.adapt([0.01, -0.01, 0], IDENTITY, "hold", STATE) == pytest.approx([0.2, -0.2, 0, 0, 0, 0, -1])
    assert adapter.adapt([0, 0, 0], IDENTITY, "close", STATE)[-1] == 1
    assert adapter.adapt([0, 0, 0], IDENTITY, "hold", STATE)[-1] == 1
    adapter.reset()
    assert adapter.adapt([0, 0, 0], IDENTITY, "hold", STATE)[-1] == -1


def test_world_frame_rotation_with_nonidentity_current_orientation():
    # Target = Rx(0.1) @ Rz(pi/2). A body-frame error would point elsewhere.
    sine, cosine = math.sin(0.05) / math.sqrt(2), math.cos(0.05) / math.sqrt(2)
    target = [sine, -sine, cosine, cosine]
    state = STATE.copy()
    state[5] = math.pi / 2
    action = ActionAdapter().adapt([0, 0, 0], target, "hold", state)
    assert action[3:6] == pytest.approx([0.2, 0, 0], abs=1e-10)


def test_rotation_cap_preserves_direction_and_quaternion_sign_equivalence():
    axis = np.array([1, 2, 3], dtype=float)
    axis /= np.linalg.norm(axis)
    target = np.r_[axis * math.sin(0.8), math.cos(0.8)]
    adapter = ActionAdapter()
    action = adapter.adapt([0, 0, 0], target, "hold", STATE)
    assert np.array(action[3:6]) == pytest.approx(axis * 0.15 / 0.5)
    assert adapter.adapt([0, 0, 0], -target, "hold", STATE) == pytest.approx(action)


@pytest.mark.parametrize("delta", [[0.0501, 0, 0], [float("nan"), 0, 0], [0, 0]])
def test_invalid_translation_is_rejected_instead_of_clipped(delta):
    with pytest.raises(ValueError):
        ActionAdapter().adapt(delta, IDENTITY, "hold", STATE)


def test_invalid_orientation_and_gripper_rejected():
    adapter = ActionAdapter()
    with pytest.raises(ValueError, match="nonzero"):
        adapter.adapt([0, 0, 0], [0, 0, 0, 0], "hold", STATE)
    with pytest.raises(ValueError, match="gripper"):
        adapter.adapt([0, 0, 0], IDENTITY, "release", STATE)
