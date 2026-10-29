import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, OBSERVATION_KEYS, decode_image


def example():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[0, 0] = [255, 0, 0]
    return {"agentview_image": pixels, "robot0_eye_in_hand_image": pixels,
            "robot0_eef_pos": [0, 0, 0.5], "robot0_eef_quat": [0, 0, 0, 1],
            "robot0_gripper_qpos": [0.03, -0.03], "object_pose": "FORBIDDEN",
            "reward": 1, "contact": True}


def test_filtered_serialization_and_axisangle():
    obs = PolicyObservation.from_libero(example(), "move the mug", image_size=4)
    assert set(obs.to_dict()) == OBSERVATION_KEYS
    assert obs.state == (0, 0, 0.5, 0, 0, 0, 0.03, -0.03)
    assert np.asarray(decode_image(obs.image))[3, 3].tolist() == [255, 0, 0]
    assert PolicyObservation.from_dict(obs.to_dict()) == obs


def test_reject_policy_side_extra_state():
    data = PolicyObservation.from_libero(example(), "reach").to_dict()
    data["object_pose"] = [1, 2, 3]
    with pytest.raises(ValueError, match="four"):
        PolicyObservation.from_dict(data)


def test_axisangle_state_and_invalid_values():
    raw = example()
    raw["robot0_eef_quat"] = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]
    assert PolicyObservation.from_libero(raw, "reach").state[5] == pytest.approx(np.pi/2)
    raw["robot0_eef_pos"][0] = float("nan")
    with pytest.raises(ValueError):
        PolicyObservation.from_libero(raw, "reach")

