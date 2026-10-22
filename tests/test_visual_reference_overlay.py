import copy
import json

import numpy as np

from gpt_field.observation import decode_image
from gpt_field.visual_reference_overlay import (
    LABEL, append_reference_diagnostic, reference_diagnostic_image,
)
from test_visual_field_protocol import observation


def context(repair=True):
    return {"visual_feedback": {"visual": {
        "reference_health": {"repair_required": repair},
        "tracking": {"moving": {"tracked_point_uv": [.5, .25], "reliable": True},
                     "target": {"tracked_point_uv": [.25, .75], "reliable": True}},
    }}}


def test_current_claim_marker_preserves_raw_images_and_observation_history():
    obs = observation().to_dict()
    contents = [{"type": "input_text", "text": json.dumps({"type": "observation"})},
                {"type": "input_image", "image_url": obs["observation/image"]},
                {"type": "input_image", "image_url": obs["observation/wrist_image"]}]
    payload = {"input": [{"role": "user", "content": copy.deepcopy(contents)},
                         {"role": "assistant", "content": [{"type": "output_text", "text": "old"}]},
                         {"role": "user", "content": copy.deepcopy(contents)}]}
    prior = copy.deepcopy(payload["input"][:-1])
    result = append_reference_diagnostic(payload, obs, context())
    assert result is payload and result["input"][:-1] == prior
    assert result["input"][-1]["content"][:3] == contents
    assert result["input"][-1]["content"][3]["text"] == LABEL
    image = np.asarray(decode_image(result["input"][-1]["content"][4]["image_url"]))
    assert image.shape == np.asarray(decode_image(obs["observation/image"])).shape
    # Ring sits at the track's current point, using the same normalized UV axes.
    assert tuple(image[round(.25*(image.shape[0]-1))-5,
                       round(.5*(image.shape[1]-1))]) == (255, 40, 210)
    append_reference_diagnostic(payload, obs, context())
    assert len(result["input"][-1]["content"]) == 5


def test_healthy_reference_adds_no_prompt_image_or_data_change():
    obs = observation().to_dict()
    payload = {"input": []}
    assert append_reference_diagnostic(payload, obs, context(False)) == {"input": []}
    assert reference_diagnostic_image(obs, {}) is None


def test_lost_reference_marked_as_estimate_and_invalid_coordinates_not_invented():
    obs = observation().to_dict()
    ctx = context()
    track = ctx["visual_feedback"]["visual"]["tracking"]
    track["moving"]["reliable"] = False
    track["target"]["tracked_point_uv"] = [float("nan"), .2]
    image = np.asarray(decode_image(reference_diagnostic_image(obs, ctx)))
    assert tuple(image[round(.25*(image.shape[0]-1)),
                       round(.5*(image.shape[1]-1))]) == (255, 40, 210)
    track["moving"]["tracked_point_uv"] = [2., .2]
    assert reference_diagnostic_image(obs, ctx) is None


def test_failed_retain_can_show_explicitly_stale_rejected_reference_without_fabricating_a_track():
    obs = observation().to_dict()
    ctx = {"visual_feedback": {"visual": {"tracking": {}, "reference_health": {
        "repair_required": True, "rejected_current_uv": [.5, .5]}}}}
    original = copy.deepcopy(ctx)
    image = np.asarray(decode_image(reference_diagnostic_image(obs, ctx)))
    assert tuple(image[round(.5*(image.shape[0]-1)),
                       round(.5*(image.shape[1]-1))]) == (255, 40, 210)
    assert ctx == original
