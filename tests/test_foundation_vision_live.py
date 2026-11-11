"""Actual executive payloads and failure boundaries without models or simulation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_user_check_client import PlanLock
from gpt_field.contact_skills import StepView
from gpt_field.foundation_vision import live
from gpt_field.foundation_vision.engine import EngineConfig
from gpt_field.foundation_vision.types import Detection
from gpt_field.observation import PolicyObservation, image_to_data_url
from test_foundation_vision_bridge import metadata


TASK = "Move the chosen objects."
PLAN = {"scene": "Two objects on a surface.", "parts": [
    {"id": "A", "text": "first object: body", "move": "free", "w": 1, "x": 1},
    {"id": "B", "text": "second object: body", "move": "free", "w": 2, "x": 2}],
    "hold_true": [], "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
                                "sense": "any", "keep": [], "until": []}]}
STATE = (.2, -.1, 1., .2, -.3, .1, .04, -.04)


class Transport:
    offline_mock = True

    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(deepcopy(payload))
        answer = {"belief": "Attachment is unverified.", "plan": "Observe the scene.",
                  "why": "Inspect prediction uncertainty.", "type": "observe",
                  "motion": None, "macro": None, "execution": None,
                  "obstacle_map": {"avoid_boxes": [{"lower_xyz_m": [0., 0., 0.],
                                                     "upper_xyz_m": [.1, .1, .1]}], "clearance_m": .01}}
        return {"id": f"offline-{len(self.calls)}", "status": "completed", "model": payload["model"],
                "reasoning": {"effort": "high"}, "offline_mock": True, "billing": "offline_mock",
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}


class Segmenter:
    def __init__(self, candidates=1):
        self.candidates = candidates

    def segment(self, frame, targets):
        return [Detection(camera.name, target.key,
                          np.ones(camera.rgb.shape[:2], dtype=bool), .9 - index * .1)
                for target in targets for camera in frame.cameras for index in range(self.candidates)]


class Engine:
    def __init__(self, clock, *, candidates=1, fail=False):
        self.clock, self.fail = clock, fail
        self.config = EngineConfig(max_age_s=5.)
        self.segmenter = Segmenter(candidates)
        self.calls = []

    def process(self, frame, targets, *, now_s):
        self.calls.append((frame, targets, now_s))
        detections = self.segmenter.segment(frame, targets)
        self.clock[0] += 1.
        if self.fail:
            raise RuntimeError("inference unavailable")
        objects = []
        for target in targets:
            counts = {camera.name: sum(item.prompt_key == target.key and item.view == camera.name
                                      for item in detections) for camera in frame.cameras}
            objects.append({"key": target.key, "text": target.text,
                "status": "new_candidate" if max(counts.values()) == 1 else "identity_ambiguous",
                "candidate_counts": counts, "centroid_world_m": [.2, -.1, .8],
                "identity_evidence": "crossview_geometry_hypothesis",
                "geometry": {"status": "supported_surface_hypothesis", "supported_point_count": 30,
                    "bounds_world_m": {"min": [.1, -.2, .7], "max": [.3, 0., .9]},
                    "normal_world": [0., 0., 1.], "normal_kind": "planar_supported_surface_fit"}})
        data = {"frame_id": frame.frame_id, "captured_at_s": frame.captured_at_s,
            "generated_at_s": self.clock[0], "age_s": self.clock[0] - frame.captured_at_s,
            "max_age_s": self.config.max_age_s, "status": "estimated", "objects": objects,
            "geometry": {"surface_hypothesis_count": 100,
                "surface_points_world": [[i / 100., 0., .8] for i in range(100)],
                "surface_crossview_support_counts": [1] * 100,
                "surface_contradiction_counts": [0] * 100,
                "views": {camera.name: {"valid_depth_pixels": 100, "invalid_depth_pixels": 0,
                    "crossview_supported_points": 100, "contradicted_points": 0,
                    "occluded_in_other_views_points": 0} for camera in frame.cameras}}}
        return SimpleNamespace(to_dict=lambda: deepcopy(data))


def setup(tmp_path, monkeypatch, *, candidates=1, fail=False, extras=()):
    clock = [100.]
    monkeypatch.setattr(live, "perf_counter", lambda: clock[0])
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    url = image_to_data_url(image, size=512, rotate=False)
    view = StepView(PolicyObservation(url, url, STATE, TASK))
    context = {"step": 0, "geometry": metadata(list(STATE), size=512)}
    engine, transport = Engine(clock, candidates=candidates, fail=fail), Transport()
    lock = PlanLock(deepcopy(PLAN), task_prompt=TASK)
    client = live.FoundationVisionAgentClient(task_prompt=TASK, approved_plan=lock.plan(),
        lock=lock, transport=transport, vision_engine=engine,
        vision_output_dir=tmp_path / "vision", vision_extra_targets=extras)
    return client, view, context, engine, transport, clock


def fields(payload):
    message = next(item for item in reversed(payload["input"]) if item.get("role") == "user")
    return json.loads(message["content"][0]["text"])


def test_real_wire_payload_includes_review_targets_user_query_and_honest_instructions(tmp_path, monkeypatch):
    client, view, context, engine, transport, _ = setup(tmp_path, monkeypatch, extras=("can",))
    before = deepcopy(context)
    assert client.request_agent(view, context)["type"] == "observe"
    assert context == before
    assert [(item.key, item.text) for item in engine.calls[0][1]] == [
        ("A", "first object: body"), ("B", "second object: body"), ("extra_0", "can")]
    assert all(not item.seed_pixels for item in engine.calls[0][1])
    payload = transport.calls[0]
    summary = fields(payload)["context"]["foundation_vision"]
    assert len(summary["objects"]) == 3
    assert len(summary["scene"]["surface_points_world"]) == 64
    assert summary["scene"]["sampled_context_only"] is True
    assert summary["objects"][0]["geometry_usable"] is True
    assert summary["grasp_contact_attachment_evidence"] == "unknown"
    assert summary["automatic_action_gate"] is summary["full_robot_collision_check"] is False
    assert "No object poses, depth, segmentation" not in payload["instructions"]
    assert payload["instructions"] == client._instruction_text["agent"] == live.VISION_INSTRUCTIONS
    assert "checking it against both RGB views" in payload["instructions"]
    stats = json.loads((tmp_path / "vision/summary.json").read_text())
    assert stats["vision_requests"] == stats["consumed_requests"] == 1
    assert stats["supported_centroids"] == 3 and stats["nonempty_returned_obstacle_map"] == 1
    assert stats["targets"][-1]["source"] == "user_extra_query"
    record = json.loads((tmp_path / "vision/requests.jsonl").read_text())
    assert record["consumed"] and record["request_key"]


def test_all_candidates_saved_even_when_bounded_prompt_is_ambiguous(tmp_path, monkeypatch):
    client, view, context, _, transport, _ = setup(tmp_path, monkeypatch, candidates=5)
    client.request_agent(view, context)
    record = json.loads((tmp_path / "vision/frame_000000_sam_candidates.json").read_text())
    assert record["candidate_count"] == 20
    with np.load(tmp_path / "vision" / record["masks"]["path"], allow_pickle=False) as arrays:
        assert len(arrays.files) == 20 and arrays["mask_0000"].dtype == np.bool_
    obj = fields(transport.calls[0])["context"]["foundation_vision"]["objects"][0]
    assert obj["status"] == "identity_ambiguous" and obj["geometry_usable"] is False
    assert obj["views"]["external"]["candidate_count"] == 5
    assert obj["views"]["external"]["candidates_omitted"] == 2
    assert len(obj["views"]["external"]["candidates"]) == 3
    assert client.vision_stats()["supported_centroids"] == 0


def test_same_step_reuses_snapshot_but_becomes_stale_without_refreshing_capture_time(tmp_path, monkeypatch):
    client, view, context, engine, transport, clock = setup(tmp_path, monkeypatch)
    client.request_agent(view, context)
    first = fields(transport.calls[0])["context"]["foundation_vision"]
    clock[0] = 110.
    client.request_agent(view, context)
    second = fields(transport.calls[1])["context"]["foundation_vision"]
    assert len(engine.calls) == 1
    assert first["captured_at_s"] == second["captured_at_s"] == 100.
    assert first["generated_at_s"] == second["generated_at_s"] == 101.
    assert second["age_s"] == 10. and second["status"] == "stale"
    assert second["objects"][0]["centroid_world_m"] is None
    assert second["scene"]["surface_points_world"] == []
    assert client.vision_stats()["cache_reuses"] == 1
    context["step"] = 1
    client.request_agent(view, context)
    assert len(engine.calls) == 2
    assert fields(transport.calls[2])["context"]["foundation_vision"]["status"] == "estimated"


def test_age_includes_delay_before_payload_and_inference(tmp_path, monkeypatch):
    client, view, context, engine, transport, clock = setup(tmp_path, monkeypatch)
    original = client.preflight

    def delayed_preflight():
        result = original()
        clock[0] += 6.
        return result

    monkeypatch.setattr(client, "preflight", delayed_preflight)
    client.request_agent(view, context)
    summary = fields(transport.calls[0])["context"]["foundation_vision"]
    assert engine.calls[0][0].captured_at_s == 100.
    assert engine.calls[0][2] == 106.
    assert summary["age_s"] == 7. and summary["status"] == "stale"
    assert summary["objects"][0]["geometry_usable"] is False


def test_changed_input_or_older_step_cannot_reuse_current_frame_id(tmp_path, monkeypatch):
    client, view, context, engine, transport, _ = setup(tmp_path, monkeypatch)
    context["step"] = 2
    client.request_agent(view, context)
    changed = deepcopy(context)
    changed["geometry"]["cameras"]["external"]["position_world_m"][0] += .01
    with pytest.raises(ValueError, match="repeated step"):
        client.request_agent(view, changed)
    with pytest.raises(ValueError, match="frame IDs"):
        client.request_agent(view, {**context, "step": 1})
    assert len(engine.calls) == len(transport.calls) == 1


def test_inference_failure_propagates_before_executive_and_keeps_raw_candidates(tmp_path, monkeypatch):
    client, view, context, _, transport, _ = setup(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="inference unavailable"):
        client.request_agent(view, context)
    assert transport.calls == []
    assert (tmp_path / "vision/frame_000000_sam_candidates.npz").is_file()
    assert not (tmp_path / "vision/frame_000000_snapshot.json").exists()
    record = json.loads((tmp_path / "vision/requests.jsonl").read_text())
    assert record["consumed"] is False and record["error_type"] == "RuntimeError"
    assert client.vision_stats()["errors"] == 1


def test_caller_cannot_inject_vision_or_bypass_public_input_validation(tmp_path, monkeypatch):
    client, view, context, engine, transport, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unexpected fields"):
        client.request_agent(view, {**context, "foundation_vision": {"collision_free": True}})
    assert engine.calls == transport.calls == []
    with pytest.raises(RuntimeError, match="validated request_agent"):
        client.build_payload({"fields": {}, "images": []}, mode="agent")


def test_target_bound_and_existing_output_fail_before_inference(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="at most 12"):
        setup(tmp_path, monkeypatch, extras=tuple(f"object {i}" for i in range(11)))
    assert not (tmp_path / "vision").exists()
    setup(tmp_path, monkeypatch)
    before = (tmp_path / "vision/summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        setup(tmp_path, monkeypatch)
    assert (tmp_path / "vision/summary.json").read_bytes() == before
