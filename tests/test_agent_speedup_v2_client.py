"""T1 tests of the v2 model-interface client (contract §C12; spec §3.2, §3.5, §3.6, §4.2; AC-1(b), AC-5).
Offline only: every request goes through ScriptedV2Transport or a fake transport; no network, no simulator."""

import ast
import base64
import copy
import inspect
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import agent_speedup_v2_fakes as fakes
import gpt_field.agent_speedup_v2_client as C
from gpt_field import agent_speedup_v1_client as v1_client
from gpt_field import agent_speedup_v2_plan as P
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field import agent_speedup_v2_types as kernel
from gpt_field import mark_prompting
from gpt_field.agent_speedup_v2_tasks import list_v2_tasks, resolve_v2_task
from gpt_field.astra import ENDPOINT, APIError, MissingCredentials, SchemaError
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS, _protocol_error
from gpt_field.contact_skills import StepView
from gpt_field.direct_history import TASK as V1_BOWL_TASK
from gpt_field.ordered_transport import ordered_dumps

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
CLIENT_PATH = th.PACKAGE_DIR / "agent_speedup_v2_client.py"
PROMPT = resolve_v2_task("libero_10_t2").prompt
VOCAB = set(th.lint_vocabulary(LIBERO)["vocabulary"])
KNOWN = set(th.LINT_ALLOWLIST)
REPAIR_RETRY = {"why": "the grip slipped on the ridge", "action": "retry", "part": None, "step": None}
OUTCOME = {"reason": "hold_lost", "evidence_text": "part C: the gap fell to 0.4 mm, below the hold floor 0.5 mm",
           "thresholds": [["G6_hold_floor_mm", 0.5]]}


def image(seed=0, size=512):
    """A smooth, cheap-to-encode test image with some structure (gradient plus a seeded square)."""
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    rgb = np.stack([(u * 255 // (size - 1)), (v * 255 // (size - 1)), ((u + v + 37 * seed) % 256)], axis=-1)
    rgb = rgb.astype(np.uint8)
    rgb[100 + seed:140 + seed, 200:260] = (10, 200, 90)
    return rgb


def answers(**kw):
    defaults = {"plans": [C.WORKED_EXAMPLE], "points_fine": [28], "points_coarse": [36], "repairs": [REPAIR_RETRY]}
    return C.FixedAnswers(**{**defaults, **kw})


def make(scripted=None, *, prompt=PROMPT, **kw):
    transport = C.ScriptedV2Transport(scripted or answers())
    return C.AgentSpeedUpV2Client(task_prompt=prompt, transport=transport, **kw), transport


def status_block():
    plan = P.parse_plan(C.WORKED_EXAMPLE)
    states = [kernel.StepState(0, plan.steps[0], "done", "turned_half", {}, {}, 10, 1, [1]),
              kernel.StepState(1, plan.steps[1], "failed", "hold_lost", {}, {}, 20, 1, [2])]
    return P.plan_status_block(plan, states, holding=None)


def every_request(client, *, k=2):
    """Exercise every request kind once; returns the replies."""
    ext, wrist = image(1), image(2)
    edge = mark_prompting.FINGERTIP_EDGE[k]
    return [client.request_plan(ext, wrist, quarter_turns=k, fingertip_edge=edge),
            client.request_plan(ext, wrist, quarter_turns=k, fingertip_edge=edge, status_block=status_block(),
                                reask_error="undefined_pid: step 1 grip names part H, which is not defined",
                                holding="C", replan=True),
            client.request_point(wrist, stage="fine", camera="wrist", text="toy car: roof ridge", pid="C",
                                 center_uv=(300.0, 200.0), quarter_turns=k,
                                 previous_rejected="pointed_not_grasp_point: 1.2 cm below the landing columns"),
            client.request_point(ext, stage="coarse", camera="external", text="rug: patch with the star print",
                                 pid="E", quarter_turns=0),
            client.request_repair(ext, wrist, quarter_turns=k, status_block=status_block(), last_outcome=OUTCOME,
                                  holding="C"),
            client.request_repair(ext, wrist, quarter_turns=k, status_block=status_block(), last_outcome=None,
                                  holding=None, review=True)]


def request_text(payload):
    return json.loads(payload["input"][0]["content"][0]["text"])


ECHOED_KEYS = {"task", "part", "scene"}


def executor_strings(value, prompt, out):
    """Every executor-written string of a payload (dict keys and string values, JSON text decoded); skips images,
    the echoed model texts (ECHOED_KEYS and the texts of status-block part pairs) and the official sentence."""
    if isinstance(value, dict):
        for name, item in value.items():
            out.append(str(name))
            if name in ECHOED_KEYS:
                continue
            if name == "parts" and isinstance(item, list) and all(isinstance(p, list) and len(p) == 2 for p in item):
                out.extend(pair[0] for pair in item)
                continue
            executor_strings(item, prompt, out)
    elif isinstance(value, list):
        for item in value:
            executor_strings(item, prompt, out)
    elif isinstance(value, str):
        if value.startswith("data:"):
            return
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                executor_strings(json.loads(stripped), prompt, out)
                return
            except ValueError:
                pass
        out.append(value.replace(prompt, " "))


def vocabulary_hits(text):
    return set().union(*(th.token_hits(t, VOCAB, KNOWN) for t in th.tokens(text)))


# --------------------------------------------------------------------------------- 1. the official sentence
def test_constructor_requires_the_official_sentence():
    with pytest.raises(TypeError):
        C.AgentSpeedUpV2Client()
    with pytest.raises(TypeError):
        C.AgentSpeedUpV2Client(None, token_provider=lambda: "t")
    for bad in ("", "   ", 3, None):
        with pytest.raises(ValueError):
            C.AgentSpeedUpV2Client(task_prompt=bad)
    parameters = inspect.signature(C.AgentSpeedUpV2Client.__init__).parameters
    assert parameters["task_prompt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["task_prompt"].default is inspect.Parameter.empty
    assert not {"task_key", "task_id", "task_spec", "target_description"} & set(parameters)
    from gpt_field.astra import AstraConfig
    with pytest.raises(ValueError):
        C.AgentSpeedUpV2Client(AstraConfig(model="gpt-other"), task_prompt=PROMPT)
    client, _ = make()
    assert client.task_prompt == PROMPT and client.reasoning_effort == "medium" and client.image_history == 1
    assert client.prompt_version == C.PROMPT_VERSION and client.offline_mock is True
    assert client.config.model == "gpt-6-astra"


def test_client_identity_is_readable_from_the_class():
    """The runner writes the manifest's client block from the class it is handed, before construction
    (scripts/evaluate_agent_speedup_v2.py: getattr(client_class, "prompt_version" / "reasoning_effort"))."""
    cls = C.AgentSpeedUpV2Client
    block = {"class_factory": cls.__qualname__, "prompt_version": getattr(cls, "prompt_version", None),
             "reasoning_effort": getattr(cls, "reasoning_effort", None)}
    assert block == {"class_factory": "AgentSpeedUpV2Client", "prompt_version": C.PROMPT_VERSION,
                     "reasoning_effort": "medium"}
    json.dumps(block, allow_nan=False)                                   # digestible (no property objects)
    client, _ = make()
    assert (client.prompt_version, client.reasoning_effort) == (C.PROMPT_VERSION, "medium")


# ------------------------------------------------------------------------------------- 2. every registry key
@pytest.mark.parametrize("spec", list_v2_tasks(), ids=lambda s: s.key)
def test_every_key_payloads_carry_only_the_official_sentence(spec):
    prompts = {s.key: s.prompt for s in list_v2_tasks()}
    client, transport = make(prompt=spec.prompt)
    every_request(client)
    assert transport.kinds == ["plan", "replan", "point_fine", "point_coarse", "repair", "review"]
    others = [p for key, p in prompts.items() if key != spec.key and p not in spec.prompt]
    for payload in transport.calls:
        text = ordered_dumps(payload)
        assert request_text(payload)["task"] == spec.prompt
        assert not [p for p in others if p in text], spec.key
        if spec.prompt != V1_BOWL_TASK:
            assert V1_BOWL_TASK not in text
        assert payload["instructions"] == C.instructions(C._KIND_BY_FORMAT[payload["text"]["format"]["name"]])
        strings = []
        executor_strings(payload, spec.prompt, strings)
        hits = {s[:60]: sorted(vocabulary_hits(s)) for s in strings if vocabulary_hits(s)}
        assert not hits, (spec.key, hits)
    for payload in transport.calls[4:]:
        assert set(request_text(payload)["last_outcome"] or {}) <= {"reason", "evidence_text", "thresholds"}
    assert set(request_text(transport.calls[4])["last_outcome"]) == {"reason", "evidence_text", "thresholds"}
    assert request_text(transport.calls[5])["last_outcome"] is None


@pytest.mark.parametrize("spec", list_v2_tasks(), ids=lambda s: s.key)
def test_check_observation_accepts_the_reset_observation_only(spec):
    client, _ = make(prompt=spec.prompt)
    sim = fakes.FakeV2Sim(prompt=spec.prompt, task_id=spec.task_id)
    observation = sim.reset(spec.task_id, 0)
    parsed = client.check_observation(observation)
    assert parsed == observation and parsed.prompt == spec.prompt
    assert client.check_observation(observation.to_dict()) == observation
    other = fakes.FakeV2Sim(prompt=spec.prompt + " now", task_id=spec.task_id).reset(spec.task_id, 0)
    with pytest.raises(ValueError):
        client.check_observation(other)


# ------------------------------------------------------------------------------------ 3. one identity
def test_instruction_and_schema_identity_is_the_same_for_every_key():
    assert not inspect.signature(C.instructions_sha256).parameters
    assert not inspect.signature(P.schemas_sha256).parameters
    digests = set()
    for spec in list_v2_tasks():
        client, _ = make(prompt=spec.prompt)
        digests.add((json.dumps(client._instruction_text, sort_keys=True), json.dumps(client._schemas)))
    assert len(digests) == 1
    assert len(C.instructions_sha256()) == 64 and C.instructions_sha256() == C.instructions_sha256()
    assert set(C._INSTRUCTIONS) == set(C.REQUEST_KINDS)
    with pytest.raises(ValueError):
        C.instructions("decision")


def test_instruction_texts_and_worked_example_carry_no_vocabulary_and_cover_the_spec():
    for kind in C.REQUEST_KINDS:
        assert not vocabulary_hits(C.instructions(kind)), kind
        assert C.instructions(kind).startswith(f"{C.PROMPT_VERSION}, {kind} request.")
    assert not vocabulary_hits(json.dumps(C.WORKED_EXAMPLE))
    example = P.parse_plan(C.WORKED_EXAMPLE)
    assert P.validate_plan(example) == [] and P.compact_chars(C.WORKED_EXAMPLE) <= 834
    plan_text = C.instructions("plan")
    for word in (*kernel.DO_KINDS, *kernel.UNTIL_KINDS, *kernel.KEEP_KINDS, *kernel.MOVE_PRIORS, "hold_true",
                 "about 79 mm", "fingertips", "Never give directions, distances, heights, angles or coordinates"):
        assert word in plan_text, word
    assert ordered_dumps(C.WORKED_EXAMPLE) in plan_text
    assert "world" not in th.tokens(plan_text)                     # no world-axis text (spec §3.5)
    for kind in ("repair", "review"):
        for action in kernel.REPAIR_ACTIONS:
            assert action in C.instructions(kind), (kind, action)
    review = C.instructions("review")
    assert "If the planned relations hold in the images, answer hold_still." in review
    for text in (C.instructions(kind).lower() for kind in C.REQUEST_KINDS):   # R-4 stays off
        assert "incomplete" not in text and "not complete" not in text and "continu" not in text
    assert "_legacy_instructions" not in CLIENT_PATH.read_text()


def test_worked_example_teaches_the_prompt_2_rules_and_is_no_gated_task_template():
    """The one-shot example (spec §3.5 item 5; Appendix L 14 BC-R3) shows the rules the planck_001 plans broke and
    is no slice or regression task's nominal plan (spec §3.9: task 2 turn -> place_on until effect-or-stop;
    bowl_plate place_on; drawer_bowl slide -> place_in)."""
    example = C.WORKED_EXAMPLE
    steps = example["steps"]
    kinds = [s["do"] for s in steps]
    assert kinds not in (["turn", "place_on"], ["place_on"], ["slide", "place_in"])
    assert not any(a == "turn" and b in ("place_on", "place_in") for a, b in zip(kinds, kinds[1:]))
    turns = [sorted(u["o"] for u in s["until"]) for s in steps if s["do"] == "turn"]
    assert turns and ["effect", "stop"] not in turns                     # not task 2's effect-or-stop turn
    # no rotation criterion: the model copies the example's criteria, and the relation layer refuses a turn whose
    # quarter_turn / half_turn exceeds the yaw budget left even when ORed with an effect (contract §C10.3 step 0)
    assert not any(u["o"] in ("quarter_turn", "half_turn") for s in steps for u in s["until"])
    effects = [(s, u["p"]) for s in steps for u in s["until"] if u["o"] == "effect"]
    assert effects and all(p != s["obj"] and p not in s["grip"] for s, p in effects)     # the effect-part rule
    assert any(s["do"] in ("place_on", "place_in") and s["grip"] == [] for s in steps)   # the whole object fits
    assert "hold" not in kinds                                           # so no hold before a place either
    assert example["hold_true"] == []                                    # no claim that would add a keep-out
    assert len(example["scene"]) <= P.SCENE_MAX_CHARS
    assert all(P.TEXT_MIN_CHARS <= len(p["text"]) <= P.TEXT_MAX_CHARS for p in example["parts"])
    assert all(len(u["txt"]) <= P.TXT_MAX_CHARS for s in steps for u in s["until"] if u["txt"] is not None)
    assert ordered_dumps(example).isascii()
    plan = P.parse_plan(example)
    assert P.validate_plan(plan) == [] and plan.to_json() == example    # every object used, nothing repeated


def test_plan_instructions_state_the_limits_the_effect_and_grip_rules_and_the_correction():
    for kind in ("plan", "replan"):
        text = C.instructions(kind)
        assert text.isascii() and C._CORRECTION in text and C.PROMPT_VERSION == "agent-speedup-v2-prompt-4"
        for phrase in (f"at most {P.SCENE_MAX_CHARS} characters",
                       f"{P.TEXT_MIN_CHARS} to {P.TEXT_MAX_CHARS} characters",
                       f"txt at most {P.TXT_MAX_CHARS} characters", "plain ASCII",
                       "p is the part whose look changes, never the step's obj or a grip part",
                       "a hold step before them is unnecessary", "List each criterion once",
                       "every object you list is used by a step as obj, grip, ref, keep or criterion part, or by "
                       "hold_true",
                       "every part of one, since it moves with the object",
                       "leave grip empty when obj fits between the fingers",
                       "never invent a thin edge, rim or face of an object that fits whole",
                       "change only what those issues name", "copy every other part and step unchanged",
                       "steps by index from 0", "When previous_plan is null"):
            assert phrase in text, (kind, phrase)
        assert "thin enough" not in text and "thin protrusion" not in text    # prompt 1's 'thin part first' rule
    assert C._CORRECTION not in C.instructions("repair") and C._CORRECTION not in C.instructions("point_fine")
    for kind in ("repair", "review"):                                    # the repair texts' limits are stated too
        text = C.instructions(kind)
        assert text.isascii(), kind
        for phrase in (f"one short sentence of at most {P.WHY_MAX_CHARS} characters", "plain ASCII",
                       f"until txt at most {P.TXT_MAX_CHARS} characters"):
            assert phrase in text, (kind, phrase)


# ------------------------------------------------------------------------------------------- 4. wire order
def test_wire_order_keeps_instructions_task_images_and_schema_order():
    client, transport = make()
    every_request(client)
    plan_wire = ordered_dumps(transport.calls[0])
    assert plan_wire.index('"instructions"') < plan_wire.index('"input"') < plan_wire.index('"text":{"format"')
    assert list(request_text(transport.calls[0])) == ["request", "task", "holding", "images", "status",
                                                      "previous_plan", "previous_answer_rejected"]
    labels = [part["text"] for part in transport.calls[0]["input"][0]["content"][1::2]]
    assert [label.split(":")[0] for label in labels] == ["Image 1", "Image 2", "Image 3", "Image 4"]
    assert "fingertips at the top edge" in labels[2] and "top edge" in request_text(transport.calls[0])["images"]
    kinds = [part["type"] for part in transport.calls[0]["input"][0]["content"]]
    assert kinds == ["input_text"] + ["input_text", "input_image"] * 4
    schema_wire = ordered_dumps(transport.calls[0]["text"]["format"]["schema"])
    assert [schema_wire.index(f'"{key}":{{') for key in ("scene", "parts", "hold_true", "steps")] == sorted(
        schema_wire.index(f'"{key}":{{') for key in ("scene", "parts", "hold_true", "steps"))
    repair_schema = ordered_dumps(transport.calls[4]["text"]["format"]["schema"])
    assert repair_schema.index('"why"') < repair_schema.index('"action"') < repair_schema.index('"part"')
    assert list(request_text(transport.calls[4])) == ["request", "task", "plan", "last_outcome", "step_history",
                                                      "holding", "previous_answer_rejected"]
    assert list(request_text(transport.calls[2])) == ["request", "task", "part", "camera", "image",
                                                      "previous_answer_rejected", "purpose"]
    for payload in transport.calls:                  # the request text is the ordered wire text of that dict
        assert payload["input"][0]["content"][0]["text"] == ordered_dumps(request_text(payload))
    for payload in transport.calls:
        assert payload["store"] is False and payload["stream"] is True and payload["reasoning"] == {"effort": "medium"}
        assert payload["text"]["format"]["strict"] is True and not UNSUPPORTED_FIELDS.intersection(payload)
        assert len(payload["input"]) == 1 and payload["input"][0]["role"] == "user"


# --------------------------------------------------------------------------- 5. fine overlay and mapping
def test_fine_point_pixel_round_trip_with_border_clamping():
    rng = np.random.default_rng(5)
    centres = [(0.0, 0.0), (511.0, 511.0), (3.0, 500.0), (256.0, 256.0), *rng.uniform(-20, 530, (40, 2))]
    for centre in centres:
        origin = C.crop_origin(centre, C.FINE_CROP_PX, 512)
        assert all(0 <= o <= 512 - C.FINE_CROP_PX for o in origin)
        for point in range(1, 65):
            uv = C.fine_point_pixel_at(origin, point)
            assert all(origin[i] < uv[i] < origin[i] + C.FINE_CROP_PX for i in (0, 1))
            nearest = min(range(1, 65), key=lambda q: math.dist(C.fine_point_pixel_at(origin, q), uv))
            assert nearest == point
            assert uv == mark_prompting.fine_point_pixel(origin, point, 512, mark_prompting.WIDE_MARGIN_CELLS)
    inside = C.crop_origin((300.0, 200.0), 128, 512)
    assert inside == (236, 136)                           # centred: the crop centre is the requested pixel
    assert C.fine_point_pixel_at(inside, 1) == (244.0, 144.0) and C.fine_point_pixel_at(inside, 64) == (356.0, 256.0)
    for bad in ((1.5, 2), (1,), "ab"):
        with pytest.raises(ValueError):
            C.fine_point_pixel_at(bad, 1)
    for bad in (0, 65, True, 2.0):
        with pytest.raises(ValueError):
            C.fine_point_pixel_at((0, 0), bad)


def test_fine_overlay_reproduces_v1_drawing_at_the_same_crop():
    rgb = image(3)
    for cell in (1, 10, 28, 37, 64):
        url_v1, origin_v1 = mark_prompting.fine_overlay(rgb, cell, margin_cells=mark_prompting.WIDE_MARGIN_CELLS)
        centre = (origin_v1[0] + C.FINE_CROP_PX / 2.0, origin_v1[1] + C.FINE_CROP_PX / 2.0)
        url, origin = C.fine_overlay_at(rgb, centre)
        assert origin == origin_v1 and url == url_v1, cell
    url, origin = C.fine_overlay_at(rgb, (5.0, 509.0))
    assert origin == (0, 384) and decode(url).shape == (512, 512, 3)
    assert rgb[0, 0].tolist() == image(3)[0, 0].tolist()                       # the input is never drawn on


def decode(url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))


def drawn_point_centre(overlay, expected_up):
    """Centroid (upsampled px, pixel centres at integers) of the point fill nearest ``expected_up``."""
    fill = np.array(C.FINE_STYLE["point_rgba"][:3])
    mask = np.all(np.abs(overlay.astype(int) - fill) <= 2, axis=-1)
    x0, y0 = (int(round(c)) - 12 for c in expected_up)
    window = mask[max(y0, 0):y0 + 25, max(x0, 0):x0 + 25]
    vs, us = np.nonzero(window)
    assert len(us) > 10
    return (float(us.mean()) + max(x0, 0), float(vs.mean()) + max(y0, 0))


@pytest.mark.parametrize("k", range(4))
def test_request_point_fine_maps_back_within_one_pixel_of_the_answered_point(k):
    for point, centre in ((28, (300.0, 200.0)), (1, (20.0, 40.0)), (64, (500.0, 490.0)), (45, (128.0, 380.0))):
        client, transport = make(answers(points_fine=[point]))
        rgb = image(4)
        reply = client.request_point(rgb, stage="point_fine", camera="wrist", text="toy car: roof ridge", pid="C",
                                     center_uv=centre, quarter_turns=k)
        shown_centre = mark_prompting.rotate_pixel(centre, k, 512)
        assert reply.origin == C.crop_origin(shown_centre, 128, 512) and reply.answer_id == point
        assert (reply.stage, reply.camera, reply.quantization_px, reply.crop_px, reply.upsample,
                reply.quarter_turns) == ("fine", "wrist", 8.0, 128, 4, k)
        meta = transport.metas[-1]
        assert meta["origin"] == list(reply.origin) and meta["quarter_turns"] == k and meta["pid"] == "C"
        assert meta["center_uv"] == [centre[0], centre[1]] and meta["crop_px"] == 128 and meta["image_size"] == 512
        # independent check against the pixels the model saw: the drawn point, mapped crop -> shown -> upright
        overlay = decode(transport.calls[-1]["input"][0]["content"][2]["image_url"])
        row, col = divmod(point - 1, 8)
        up = drawn_point_centre(overlay, ((col + 0.5) * 64, (row + 0.5) * 64))
        shown = (reply.origin[0] + (up[0] + 0.5) / 4 - 0.5, reply.origin[1] + (up[1] + 0.5) / 4 - 0.5)
        upright = mark_prompting.unrotate_pixel(shown, k, 512)
        assert math.dist(upright, reply.uv_upright) <= 1.0, (k, point, upright, reply.uv_upright)
        # the crop shown is the turned image's pixels around the turned centre
        turned = mark_prompting.rotate_quarter(rgb, k)
        x0, y0 = reply.origin
        expected = np.asarray(Image.fromarray(turned[y0:y0 + 128, x0:x0 + 128]).resize((512, 512),
                                                                                        Image.Resampling.BILINEAR))
        assert np.abs(overlay[300:310, 3:9].astype(int) - expected[300:310, 3:9].astype(int)).max() <= 1


@pytest.mark.parametrize("k", range(4))
def test_request_point_coarse_maps_cells_back(k):
    for cell in (1, 8, 36, 57, 64):
        client, transport = make(answers(points_coarse=[cell]))
        reply = client.request_point(image(5), stage="coarse", camera="wrist", text="lamp: shade", pid="B",
                                     quarter_turns=k)
        expected = mark_prompting.unrotate_pixel(mark_prompting.coarse_cell_center(cell, 512), k, 512)
        assert reply.uv_upright == pytest.approx(expected) and reply.quantization_px == 32.0
        assert reply.origin is None and reply.crop_px is None and reply.answer_id == cell
        assert decode(transport.calls[-1]["input"][0]["content"][2]["image_url"]).shape == (512, 512, 3)


def test_point_requests_reject_bad_arguments_and_map_zero_to_none():
    client, transport = make(answers(points_fine=[0], points_coarse=[None]))
    reply = client.request_point(image(), stage="fine", camera="external", text="lamp: shade", pid="B",
                                 center_uv=(10.0, 10.0), quarter_turns=0)
    assert reply.answer_id is None and reply.uv_upright is None and reply.origin == (0, 0)
    assert client.request_point(image(), stage="coarse", camera="wrist", text="lamp: shade", pid="B",
                                quarter_turns=3).answer_id is None
    common = {"text": "lamp: shade", "pid": "B"}
    with pytest.raises(ValueError):
        client.request_point(image(), stage="fine", camera="external", center_uv=(1, 1), quarter_turns=1, **common)
    with pytest.raises(ValueError):
        client.request_point(image(), stage="fine", camera="wrist", quarter_turns=0, **common)
    with pytest.raises(ValueError):
        client.request_point(image(size=256), stage="coarse", camera="wrist", quarter_turns=0, **common)
    with pytest.raises(ValueError):
        client.request_point(image(), stage="coarse", camera="wrist", quarter_turns=0, previous_rejected="x" * 121,
                             **common)
    with pytest.raises(ValueError):
        client.request_point(image(), stage="medium", camera="wrist", quarter_turns=0, **common)
    with pytest.raises(ValueError):
        client.request_point(image(), stage="coarse", camera="ceiling", quarter_turns=0, **common)
    with pytest.raises(ValueError):
        client.request_point(image(), stage="coarse", camera="wrist", quarter_turns=4, **common)
    with pytest.raises(ValueError):
        client.request_point(image(), stage="coarse", camera="wrist", quarter_turns=0, text="", pid="B")
    for marker in ((3.0,), (float("nan"), 4.0), "ab"):                    # BC-2: the marker is checked, not reserved
        with pytest.raises(ValueError):
            client.request_point(image(), stage="coarse", camera="wrist", quarter_turns=0, marker_uv=marker,
                                 **common)
    with pytest.raises(ValueError):                                      # BC-1: a POINT_PURPOSES value or None
        client.request_point(image(), stage="coarse", camera="wrist", quarter_turns=0, purpose="lift", **common)
    assert len(transport.calls) == 2                                     # rejected requests never reach the transport


def test_plan_images_turn_the_wrist_view_only():
    ext, wrist = image(6), image(7)
    rendered = C.plan_images(ext, wrist, 3)
    assert [name for name, _ in rendered] == ["external clean", "external grid", "wrist clean", "wrist grid"]
    assert np.array_equal(decode(rendered[0][1]), ext)
    assert np.array_equal(decode(rendered[2][1]), mark_prompting.rotate_quarter(wrist, 3))
    assert rendered[3][1] == mark_prompting.coarse_overlay(mark_prompting.rotate_quarter(wrist, 3))
    assert rendered[1][1] == mark_prompting.coarse_overlay(ext)


# ------------------------------------------------------------------------------- 6. scripted transport
def test_scripted_transport_envelope_metas_and_counts(tmp_path):
    client, transport = make(log_dir=tmp_path)
    every_request(client)
    stats = client.history_stats()
    assert stats["calls_by_kind"] == {kind: 1 for kind in C.REQUEST_KINDS}
    assert stats["accepted_by_kind"] == {kind: 1 for kind in C.REQUEST_KINDS}
    assert stats["protocol"] == C.HISTORY_PROTOCOL and stats["retained_image_pairs"] == 0 and stats["messages"] == 0
    assert [meta["kind"] for meta in transport.metas] == transport.kinds
    assert transport.metas[1]["replan"] is True and transport.metas[5]["review"] is True
    assert transport.metas[0]["quarter_turns"] == 2 and transport.metas[0]["image_size"] == 512
    envelope = C.ScriptedV2Transport(answers())(ENDPOINT, transport.calls[0], {}, 1.0)
    assert envelope["status"] == "completed" and envelope["model"] == "gpt-6-astra"
    assert envelope["reasoning"] == {"effort": "medium"} and envelope["billing"] == "offline_mock"
    assert json.loads(envelope["output"][1]["content"][0]["text"]) == C.WORKED_EXAMPLE
    assert client.budget_snapshot()["billing"] == "offline_mock"
    requests = sorted(tmp_path.glob("*.request.json"))
    assert len(requests) == 6
    record = json.loads(requests[2].read_text())
    assert record["mode"] == "point_fine" and record["request_meta"]["origin"] is not None
    assert record["offline_mock"] is True and record["agent_speedup_v2_client_version"] == C.CLIENT_VERSION
    results = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.result.json"))]
    assert all(r["status"] == "valid" for r in results) and results[0]["request_kind"] == "plan"


def test_scripted_transport_refuses_bad_payloads_and_mismatched_meta():
    client, transport = make()
    client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")
    good = transport.calls[0]
    for bad in (dict(good, store=True), dict(good, stream=False), dict(good, temperature=0.2),
                dict(good, text={"format": {"name": "unknown"}})):
        with pytest.raises(ValueError):
            transport(ENDPOINT, bad, {}, 1.0)
    with pytest.raises(ValueError):
        transport("https://example.invalid/v1/responses", good, {}, 1.0)
    transport.bind_meta({"kind": "repair"})
    with pytest.raises(ValueError):
        transport(ENDPOINT, good, {}, 1.0)
    with pytest.raises(TypeError):
        transport.bind_meta(["kind"])
    with pytest.raises(TypeError):
        C.ScriptedV2Transport(object())


def test_scripted_transport_consumes_each_binding():
    client, transport = make()
    client.request_point(image(), stage="fine", camera="wrist", text="toy car: roof ridge", pid="A",
                         center_uv=(100.0, 100.0), quarter_turns=0)
    assert transport.metas[-1]["origin"] == [36, 36]
    kind, request, meta = client._point_request(image(), stage="fine", camera="wrist", text="toy car: roof ridge",
                                                pid="A", center_uv=(400.0, 400.0), quarter_turns=0)
    assert meta["origin"] == (336, 336)
    client.predict(request, mode=kind)                                   # public plumbing, sent without a binding
    assert transport.metas[-1] == {"kind": "point_fine"}                 # never the earlier request's geometry
    transport.bind_meta({"kind": "point_fine", "origin": [1, 2]})
    client.predict(request, mode=kind)
    assert transport.metas[-1]["origin"] == [1, 2]
    client.predict(request, mode=kind)
    assert transport.metas[-1] == {"kind": "point_fine"}


def test_fixed_answers_repeat_the_last_item_and_fail_when_empty():
    fixed = C.FixedAnswers(points_fine=[3, 9], repairs=[REPAIR_RETRY])
    assert [fixed.answer("point_fine", {})["point_id"] for _ in range(4)] == [3, 9, 9, 9]
    assert fixed.answer("review", {}) == REPAIR_RETRY and fixed.served[-1]["kind"] == "review"
    with pytest.raises(LookupError):
        fixed.answer("plan", {})
    with pytest.raises(ValueError):
        fixed.answer("decision", {})
    client, transport = make(C.FixedAnswers())
    with pytest.raises(APIError):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")
    assert client.history_stats()["calls_by_kind"]["plan"] == 1 and client.history_stats()["accepted_by_kind"][
        "plan"] == 0


def test_requests_are_stateless_and_the_meta_never_reaches_the_model():
    client, transport = make()
    every_request(client)
    every_request(client)
    for payload in transport.calls:
        assert len(payload["input"]) == 1                                  # no transcript
        text = payload["input"][0]["content"][0]["text"]
        for key in ("center_uv", "origin", "crop_px", "upsample", "image_size", "quarter_turns", '"pid"'):
            assert key not in text, key
    assert client.history_snapshot() == []


# ------------------------------------------------------------------------------------------- 7. preflight
def test_preflight_payloads_for_every_key_without_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("preflight must never send a request")
    monkeypatch.setattr(C, "ordered_sse_transport", refuse)
    for spec in list_v2_tasks():
        payloads = C.preflight_payloads(spec.prompt)
        assert list(payloads) == ["plan", "point_fine", "point_coarse", "repair"], spec.key
        for kind, payload in payloads.items():
            assert payload["reasoning"] == {"effort": "medium"} and payload["store"] is False
            assert request_text(payload)["task"] == spec.prompt and request_text(payload)["request"] == kind
            assert payload["text"]["format"]["name"] == C.FORMAT_NAMES[kind]
        assert set(request_text(payloads["repair"])["last_outcome"]) == {"reason", "evidence_text", "thresholds"}
        json.dumps(payloads, allow_nan=False)


# ------------------------------------------------------------------------------------------ 8. SchemaError
@pytest.mark.parametrize("kind, bad", [
    ("plan", {"scene": "x", "parts": [], "hold_true": [], "steps": []}),
    ("plan", {k: v for k, v in C.WORKED_EXAMPLE.items() if k != "steps"}),
    ("plan", dict(C.WORKED_EXAMPLE, parts=[dict(C.WORKED_EXAMPLE["parts"][0], w=99)])),
    ("plan", dict(C.WORKED_EXAMPLE, extra="x")),
    ("point_fine", {"point_id": 65}),
    ("point_fine", {"point_id": True}),
    ("point_fine", {"point_id": 3, "why": "x"}),
    ("point_coarse", {"point_id": 3}),
    ("point_coarse", {"cell_id": 2.0}),
    ("repair", dict(REPAIR_RETRY, action="teleport")),
    ("repair", dict(REPAIR_RETRY, why="")),
    ("repair", ["retry"]),
])
def test_malformed_scripted_output_raises_schema_error(kind, bad):
    group = {"plan": "plans", "point_fine": "points_fine", "point_coarse": "points_coarse", "repair": "repairs"}
    client, transport = make(answers(**{group[kind]: [bad]}))
    call = {"plan": lambda: client.request_plan(image(), image(), quarter_turns=1, fingertip_edge="right"),
            "point_fine": lambda: client.request_point(image(), stage="fine", camera="wrist", text="lamp: shade",
                                                       pid="B", center_uv=(50, 50), quarter_turns=0),
            "point_coarse": lambda: client.request_point(image(), stage="coarse", camera="wrist", text="lamp: shade",
                                                         pid="B", quarter_turns=0),
            "repair": lambda: client.request_repair(image(), image(), quarter_turns=0, status_block=status_block(),
                                                    last_outcome=OUTCOME, holding=None)}[kind]
    with pytest.raises(SchemaError):
        call()
    assert client.history_stats()["accepted_by_kind"][kind] == 0 and client.history_snapshot() == []


def test_validate_response_structure_only():
    client, _ = make()
    client._request_kind = "point_fine"
    assert client.validate_response({"point_id": 0}, None) == {"point_id": None}
    assert client.validate_response({"point_id": None}, None) == {"point_id": None}
    client._request_kind = "review"
    assert client.validate_response(dict(REPAIR_RETRY, action="hold_still"), None)["action"] == "hold_still"
    client._request_kind = None
    with pytest.raises(SchemaError):
        client.validate_response({"point_id": 1}, None)


# ----------------------------------------------------------------------------------- request validation
def test_request_arguments_are_checked_before_any_call():
    client, transport = make()
    with pytest.raises(ValueError):
        client.request_plan(image(), image(), quarter_turns=2, fingertip_edge="bottom")      # k=2 -> top
    with pytest.raises(ValueError):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom", holding="Q")
    with pytest.raises(ValueError):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom", reask_error="x" * 301)
    with pytest.raises(ValueError):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom", status_block={"parts": []})
    bad_block = status_block()
    bad_block["steps"][1]["reason"] = "the toy fell off"
    with pytest.raises(ValueError):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom", status_block=bad_block)
    for outcome in (dict(OUTCOME, evidence={"dx": 0.1}), {"reason": "hold_lost", "evidence_text": "x"},
                    dict(OUTCOME, reason="hold_dropped"), dict(OUTCOME, evidence_text="e" * 201),
                    dict(OUTCOME, thresholds=[["G6_hold_floor_mm", float("nan")]]),
                    dict(OUTCOME, thresholds=[["G6_hold_floor_mm"]])):
        with pytest.raises(ValueError):
            client.request_repair(image(), image(), quarter_turns=0, status_block=status_block(),
                                  last_outcome=outcome, holding=None)
    with pytest.raises(ValueError):
        client.request_repair(image(), image(), quarter_turns=0, status_block=None, last_outcome=None, holding=None)
    with pytest.raises(ValueError):
        client.request_plan(image(size=256), image(), quarter_turns=0, fingertip_edge="bottom")
    assert transport.calls == []


# ----------------------------------------------------------------------------------- retry and accounting
class Clock:
    def __init__(self):
        self.now, self.waits = 100.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class OverloadedOnce:
    offline_mock = True

    def __init__(self, inner, failures=1, status=None, code="server_is_overloaded"):
        self.inner, self.failures, self.status, self.code = inner, failures, status, code
        self.calls = 0

    def bind_meta(self, meta):
        self.inner.bind_meta(meta)

    def __call__(self, *args):
        self.calls += 1
        if self.calls <= self.failures:
            raise _protocol_error("provider is busy", http_status=self.status, error_code=self.code)
        return self.inner(*args)


def test_overload_retry_is_the_v1_bounded_policy(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(C.time, "sleep", clock.sleep)
    monkeypatch.setattr(C.random, "uniform", lambda low, high: 0.25)
    transport = OverloadedOnce(C.ScriptedV2Transport(answers()), status=503)
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=transport, clock=clock)
    assert client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom") == C.WORKED_EXAMPLE
    assert transport.calls == 2 and clock.waits == [2.25]
    stats = client.history_stats()
    assert stats["overload_retry_requests"] == 1 and stats["calls_by_kind"]["plan"] == 2
    always = OverloadedOnce(C.ScriptedV2Transport(answers()), failures=9)
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=always, clock=clock)
    with pytest.raises(APIError):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")
    assert always.calls == C.OVERLOAD_MAX_ATTEMPTS
    other = OverloadedOnce(C.ScriptedV2Transport(answers()), code="rate_limited")
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=other, clock=clock)
    with pytest.raises(APIError):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")
    assert other.calls == 1
    source = Path(v1_client.__file__).read_text()                     # the restated v1 numbers
    assert "self._overload_attempt >= 3" in source and "delay > 60." in source
    assert "2. ** self._overload_attempt + random.uniform(0., 1.)" in source and "(None, 503)" in source
    assert (C.OVERLOAD_MAX_ATTEMPTS, C.OVERLOAD_MAX_WAIT_S, C.OVERLOAD_BACKOFF_BASE_S, C.OVERLOAD_JITTER_S,
            C.OVERLOAD_HTTP_STATUS) == (3, 60.0, 2.0, 1.0, 503)


class EnvelopeTransport:
    offline_mock = True

    def __init__(self, data, **updates):
        self.data, self.updates, self.calls = data, updates, []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(payload)
        envelope = C.ScriptedV2Transport(C.FixedAnswers(plans=[self.data]))(endpoint, payload, headers, timeout_s)
        return {**envelope, **self.updates}


def test_model_and_effort_mismatch_are_rejected_and_tokens_are_counted_per_kind():
    for updates in ({"model": "gpt-other"}, {"reasoning": {"effort": "high"}}):
        client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=EnvelopeTransport(C.WORKED_EXAMPLE, **updates))
        with pytest.raises(APIError):
            client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")
    usage = {"input_tokens": 5000, "output_tokens": 420, "input_tokens_details": {"cached_tokens": 0},
             "output_tokens_details": {"reasoning_tokens": 60}}
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=EnvelopeTransport(C.WORKED_EXAMPLE, usage=usage))
    client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")
    stats = client.history_stats()
    assert stats["output_tokens_by_kind"]["plan"] == 420 and stats["reasoning_tokens_by_kind"]["plan"] == 60
    assert stats["visible_tokens_by_kind"]["plan"] == 360 and stats["over_visible_budget_by_kind"]["plan"] == 1
    assert stats["visible_token_budgets"] == C.VISIBLE_TOKEN_BUDGETS


class RepairEnvelope:
    offline_mock = True

    def __init__(self, data, visible):
        self.data, self.visible = data, visible

    def __call__(self, endpoint, payload, headers, timeout_s):
        envelope = C.ScriptedV2Transport(C.FixedAnswers(repairs=[self.data]))(endpoint, payload, headers, timeout_s)
        return {**envelope, "usage": {"input_tokens": 3000, "output_tokens": self.visible + 40,
                                      "input_tokens_details": {"cached_tokens": 0},
                                      "output_tokens_details": {"reasoning_tokens": 40}}}


@pytest.mark.parametrize("step, visible, over", [
    (None, 80, 0), (None, 81, 1), ({"do": "release", "obj": "C", "grip": [], "ref": None, "sense": "any", "keep": [],
                                    "until": []}, 300, 0),
    ({"do": "release", "obj": "C", "grip": [], "ref": None, "sense": "any", "keep": [], "until": []}, 301, 1),
])
def test_repair_visible_budget_depends_on_a_new_step(step, visible, over):
    """Spec §3.2: a repair or review answer has 25-80 visible tokens, <= 300 only when it carries a new step."""
    action = "retry" if step is None else "insert_step"
    data = {"why": "the grip slipped on the ridge", "action": action, "part": None, "step": step}
    for review in (False, True):
        client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=RepairEnvelope(data, visible))
        client.request_repair(image(), image(), quarter_turns=0, status_block=status_block(),
                              last_outcome=None if review else OUTCOME, holding=None, review=review)
        kind = "review" if review else "repair"
        stats = client.history_stats()
        assert stats["visible_tokens_by_kind"][kind] == visible and stats["over_visible_budget_by_kind"][kind] == over
        assert stats["repair_step_visible_token_budget"] == C.REPAIR_STEP_TOKEN_BUDGET == 300
    assert C.VISIBLE_TOKEN_BUDGETS["repair"] == C.VISIBLE_TOKEN_BUDGETS["review"] == 80


def test_token_provider_defaults():
    client, _ = make()                                                   # offline: the fixture token
    client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")

    class LiveLike:
        offline_mock = False

        def __init__(self):
            self.calls = 0

        def __call__(self, *args):
            self.calls += 1
            raise AssertionError("never reached")
    live = LiveLike()
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=live)
    assert client.offline_mock is False
    client.build_payload({"fields": {}, "images": []}, mode="plan")      # previews need no credentials
    with pytest.raises(MissingCredentials):
        client.request_plan(image(), image(), quarter_turns=0, fingertip_edge="bottom")
    assert live.calls == 0
    seen = []
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=C.ScriptedV2Transport(answers()),
                                    token_provider=lambda: seen.append(1) or "offline-agent-oauth-fixture")
    client.request_point(image(), stage="coarse", camera="wrist", text="lamp: shade", pid="B", quarter_turns=0)
    assert seen == [1]
    assert C.AgentSpeedUpV2Client(task_prompt=PROMPT).transport is C.ordered_sse_transport


def test_build_payload_rejects_task_fields_and_bad_images():
    client, _ = make()
    with pytest.raises(ValueError):
        client.build_payload({"fields": {"task": "x"}, "images": []}, mode="plan")
    with pytest.raises(ValueError):
        client.build_payload({"fields": {}, "images": [("label", "not a url")]}, mode="plan")
    with pytest.raises(ValueError):
        client.build_payload({"fields": {}}, mode="plan")
    with pytest.raises(ValueError):
        client.build_payload({"fields": {}, "images": []}, mode="decision")
    with pytest.raises(ValueError):
        client.build_payload({"fields": {"blob": "x" * 60_001}, "images": []}, mode="plan")


# ---------------------------------------------------------------- the fake ORACLE through the real client
def test_fake_oracle_plan_and_points_through_the_client():
    world = fakes.FakeTwoViewSim(fakes.task2_like_scene(), prompt=PROMPT)
    oracle = fakes.FakeOracleAnswers(world)
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=C.ScriptedV2Transport(oracle))
    view = StepView(world.reset())
    k = 2                                                                # the task-2 start alignment (spec §3.6)
    data = client.request_plan(view.external_rgb_upright, view.wrist_rgb_upright, quarter_turns=k,
                               fingertip_edge=mark_prompting.FINGERTIP_EDGE[k])
    plan = P.parse_plan(data)
    assert P.validate_plan(plan) == []
    for part in plan.parts:
        assert (part.w_cell, part.x_cell) == oracle.cells(part.pid, k), part.pid
    for pid in ("A", "B"):
        truth = oracle.truth_uv(pid, "wrist")
        reply = client.request_point(view.wrist_rgb_upright, stage="fine", camera="wrist", text=plan.part(pid).text,
                                     pid=pid, center_uv=(truth[0] + 20.0, truth[1] - 12.0), quarter_turns=k)
        assert reply.answer_id is not None and math.dist(reply.uv_upright, truth) <= math.sqrt(2) * 8 + 0.5, pid
        coarse = client.request_point(view.external_rgb_upright, stage="coarse", camera="external",
                                      text=plan.part(pid).text, pid=pid, quarter_turns=0)
        assert math.dist(coarse.uv_upright, oracle.truth_uv(pid, "external")) <= math.sqrt(2) * 32 + 0.5, pid
    truth = oracle.truth_uv("A", "wrist")
    far = client.request_point(view.wrist_rgb_upright, stage="fine", camera="wrist", text=plan.part("A").text,
                               pid="A", center_uv=(truth[0] + 150.0, truth[1]), quarter_turns=k)
    assert far.answer_id is None                                         # outside the crop -> 0 [C-37]
    assert oracle.mapping is C.fine_point_pixel_at
    assert oracle.report()["fine_mapping"] == "agent_speedup_v2_client.fine_point_pixel_at"


# --------------------------------------------------------------------------------------------- T0 and facts
def test_client_module_passes_the_t0_lint_firewall_and_literal_audit():
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    assert th.lint_module(CLIENT_PATH, vocab, allow_task_text=True) == []
    assert th.literal_audit(CLIENT_PATH) == []
    if hasattr(th, "firewall_violations"):
        assert th.firewall_violations(ast.parse(CLIENT_PATH.read_text()), "agent_speedup_v2_client") == []
    assert "agent_speedup_v2_client.py" in th.PRESENT_INTERFACE_MODULES
    assert all(key in th.constants_register() for key in th.module_labels(CLIENT_PATH))
    source = CLIENT_PATH.read_text()
    for spec in list_v2_tasks():
        assert spec.prompt not in source
    assert "start_poses" not in source and "measured_tasks" not in source


def test_restated_constants_match_their_sources():
    assert C.MAX_REQUEST_TEXT_CHARS == v1_client.MAX_REQUEST_TEXT_CHARS
    assert C.MARK_COUNT == v1_client.MARK_COUNT == P.CELL_MAX
    assert C.FINE_CROP_PX == mark_prompting.fine_crop_size(512, mark_prompting.WIDE_MARGIN_CELLS)
    assert C.FINE_UPSAMPLE == mark_prompting.fine_upsample(512, mark_prompting.WIDE_MARGIN_CELLS)
    assert C.FINE_STYLE["point_rgba"] == mark_prompting._POINT_FILL
    assert C.EVIDENCE_MAX_CHARS == kernel.EVIDENCE_TEXT_MAX_CHARS
    assert C.PREVIOUS_REJECTED_MAX_CHARS == kernel.PREVIOUS_REJECTED_MAX_CHARS
    assert C.COARSE_QUANT_PX == mark_prompting.coarse_cell_size(512) / 2.0
    assert C.FINE_QUANT_PX == C.FINE_CROP_PX / mark_prompting.FINE_GRID / 2.0
    assert set(C.VISIBLE_TOKEN_BUDGETS) == set(C.REQUEST_KINDS)
    assert C.REQUEST_KINDS == kernel.NEED_KINDS and list(P.schemas()) == list(C.REQUEST_KINDS)
    threshold = C.crop_threshold()
    assert threshold.name == "V6_crop_px" and threshold.value == 128.0 and threshold.label == "DD"
    assert th.threshold_id(threshold.name) == "V6"


# ----------------------------------------------------------------------------------- targeted re-ask (prompt 2)
def operand_effect():
    """The worked example with the turn's effect on its own obj (effect_part_is_operand, step 1)."""
    bad = copy.deepcopy(C.WORKED_EXAMPLE)
    effect = next(u for u in bad["steps"][1]["until"] if u["o"] == "effect")
    effect["p"] = bad["steps"][1]["obj"]
    return bad


def plan_call(client, **kw):
    return client.request_plan(image(1), image(2), quarter_turns=0, fingertip_edge="bottom", **kw)


def reask_text(data):
    issues = P.validate_plan(P.parse_plan(data))
    assert issues
    return P.issues_text(issues)


def test_targeted_reask_carries_the_rejected_plan_and_its_issues(tmp_path):
    first = operand_effect()
    clean_scene = first["scene"]
    first["scene"] = clean_scene + "\u200e\u200d"                          # residue at a cut: sent back sanitized
    client, transport = make(answers(plans=[first, C.WORKED_EXAMPLE]), log_dir=tmp_path)
    assert plan_call(client) == first
    text = reask_text(first)
    assert text.startswith("effect_part_is_operand: step 1")
    assert plan_call(client, reask_error=text) == C.WORKED_EXAMPLE
    initial, reask = request_text(transport.calls[0]), request_text(transport.calls[1])
    assert initial["previous_plan"] is None and initial["previous_answer_rejected"] is None
    assert reask["previous_plan"] == P.parse_plan(first).to_json() and reask["previous_answer_rejected"] == text
    assert reask["previous_plan"]["scene"] == clean_scene
    assert [m["previous_plan"] for m in transport.metas] == [False, True]
    assert client.last_reask_check["checked"] is True and client.last_reask_check["dropped"] == []
    assert client.last_reask_check["named_steps"] == [1]
    assert client.last_reask_check["unnamed_steps_not_verbatim"] == []
    stats = client.history_stats()
    assert stats["reask_requests"] == 1 and stats["reask_rejected_for_dropped_steps"] == 0
    assert stats["accepted_by_kind"]["plan"] == 2
    results = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.result.json"))]
    assert "reask_check" not in results[0] and results[1]["reask_check"]["dropped"] == []


def test_the_previous_plan_is_only_the_plan_answer_right_before_the_reask():
    first, text = operand_effect(), reask_text(operand_effect())
    client, transport = make(answers(plans=[first, C.WORKED_EXAMPLE]))
    plan_call(client)
    client.request_point(image(), stage="coarse", camera="wrist", text="toy car: whole car", pid="A",
                         quarter_turns=0)                                # any other request clears it
    plan_call(client, reask_error=text)
    assert request_text(transport.calls[2])["previous_plan"] is None
    client, transport = make(answers(plans=[first, C.WORKED_EXAMPLE]))
    plan_call(client)
    plan_call(client, reask_error=text, replan=True)                      # another kind
    assert request_text(transport.calls[1])["previous_plan"] is None
    client, transport = make(answers(plans=[first, C.WORKED_EXAMPLE]))
    plan_call(client)
    client.reset_history()                                               # a new episode
    plan_call(client, reask_error=text)
    assert request_text(transport.calls[1])["previous_plan"] is None
    client, transport = make(answers(plans=[first, first, C.WORKED_EXAMPLE]))
    plan_call(client)
    plan_call(client)                                                    # no re-ask: nothing is carried
    plan_call(client, reask_error=text)                                  # the answer right before is carried
    assert request_text(transport.calls[1])["previous_plan"] is None
    assert request_text(transport.calls[2])["previous_plan"] == P.parse_plan(first).to_json()
    schema_bad = {"scene": "x", "parts": [], "hold_true": [], "steps": []}
    client, transport = make(answers(plans=[schema_bad, C.WORKED_EXAMPLE]))
    with pytest.raises(SchemaError):
        plan_call(client)
    plan_call(client, reask_error="the previous answer did not match the required answer schema")
    assert request_text(transport.calls[1])["previous_plan"] is None     # unreadable: a new plan is asked for
    with pytest.raises(ValueError):
        client._plan_request(image(), image(), quarter_turns=0, fingertip_edge="bottom",
                             previous_plan=P.parse_plan(first))           # previous_plan only with a re-ask


def test_a_reask_answer_that_drops_an_unnamed_step_is_rejected(tmp_path):
    first = operand_effect()                                              # issue names step 1 (the turn)
    dropping = copy.deepcopy(C.WORKED_EXAMPLE)
    dropping["steps"] = dropping["steps"][1:]                             # step 0 (place_in A) gone, unnamed
    client, transport = make(answers(plans=[first, dropping]), log_dir=tmp_path)
    plan_call(client)
    with pytest.raises(SchemaError, match="step_dropped: step 0 \\(place_in A\\)"):
        plan_call(client, reask_error=reask_text(first))
    stats = client.history_stats()
    assert stats["reask_rejected_for_dropped_steps"] == 1 and stats["accepted_by_kind"]["plan"] == 1
    assert [d["code"] for d in client.last_reask_check["dropped"]] == ["step_dropped"]
    results = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.result.json"))]
    assert results[1]["status"] == "error" and results[1]["error_type"] == "SchemaError"
    assert results[1]["reask_check"]["dropped"][0]["step"] == 0
    assert len(transport.calls) == 2                                      # never re-sampled by the client
    # the named step itself may be replaced or removed, and an unnecessary hold may go
    keeping = copy.deepcopy(C.WORKED_EXAMPLE)
    keeping["steps"] = keeping["steps"][:1]
    client, _ = make(answers(plans=[first, keeping]))
    plan_call(client)
    assert plan_call(client, reask_error=reask_text(first)) == keeping


def test_the_policy_reask_path_sends_the_targeted_correction():
    """StepRunner.obtain_plan -> NeedFulfiller -> request_plan, as the live plan check drives it."""
    from types import SimpleNamespace
    import gpt_field.agent_speedup_v2 as policy_module
    client, transport = make(answers(plans=[operand_effect(), C.WORKED_EXAMPLE]))
    events = policy_module.EventSink(lambda: 0)
    fulfiller = policy_module.NeedFulfiller(client, events)
    view = SimpleNamespace(external_rgb_upright=image(1), wrist_rgb_upright=image(2))
    ctx = SimpleNamespace(view=view, cal=SimpleNamespace(external=None), events=events, holding=None,
                          active_relation=None, monitor=SimpleNamespace(clear=lambda scope: None))
    runner = policy_module.StepRunner(ctx, relations=SimpleNamespace())
    runner.turns = lambda _view: 0
    generator, calls = runner.obtain_plan("plan"), []
    need = next(generator)
    try:
        while True:
            need = generator.send(fulfiller.fulfil(need, view, calls))
    except StopIteration as stop:
        plan, _answer = stop.value
    assert plan is not None and plan.to_json() == C.WORKED_EXAMPLE
    sent = [request_text(call) for call in transport.calls]
    assert sent[0]["previous_plan"] is None and sent[1]["previous_plan"] == operand_effect()
    assert sent[1]["previous_answer_rejected"].startswith("effect_part_is_operand: step 1")
    validations = [row for row in events.drain() if row["event"] == "plan_validation"]
    assert [row["ok"] for row in validations] == [False, True]


# ------------------------------------------------------------------------------- transient stream retries
class BrokenStream:
    offline_mock = True

    def __init__(self, inner, failures=1, message="Subscription stream ended without response.completed",
                 code=None):
        self.inner, self.failures, self.message, self.code = inner, failures, message, code
        self.calls = 0

    def bind_meta(self, meta):
        self.inner.bind_meta(meta)

    def __call__(self, *args):
        self.calls += 1
        if self.calls <= self.failures:
            if self.code is None:
                raise APIError(self.message)
            raise _protocol_error(self.message, error_code=self.code)
        return self.inner(*args)


def test_transient_stream_failures_are_retried_boundedly_and_logged(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(C.time, "sleep", clock.sleep)
    monkeypatch.setattr(C.random, "uniform", lambda low, high: 0.25)
    transport = BrokenStream(C.ScriptedV2Transport(answers()))
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=transport, clock=clock, log_dir=tmp_path)
    assert plan_call(client) == C.WORKED_EXAMPLE
    assert transport.calls == 2 and clock.waits == [2.25]
    stats = client.history_stats()
    assert stats["transient_retry_requests"] == 1 and stats["overload_retry_requests"] == 0
    assert stats["transient_retry_codes"] == {"stream_missing_completion": 1}
    assert stats["transient_retry_policy"] == C.TRANSIENT_RETRY_POLICY and stats["calls_by_kind"]["plan"] == 2
    (record,) = [json.loads(p.read_text()) for p in tmp_path.glob("*.transient_retry.json")]
    assert record["protocol"] == C.TRANSIENT_RETRY_POLICY and record["cause"] == "transient"
    assert record["error_code"] == "stream_missing_completion" and record["max_attempts"] == 3
    requests = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.request.json"))]
    assert [r["attempt"] for r in requests] == [1, 2] and requests[0]["payload"] == requests[1]["payload"]
    for path in tmp_path.iterdir():                                       # no token material in any log
        assert C.OFFLINE_TOKEN not in path.read_text() and "Bearer" not in path.read_text()
    always = BrokenStream(C.ScriptedV2Transport(answers()), failures=9)
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=always, clock=clock)
    clock.waits.clear()
    with pytest.raises(APIError):
        plan_call(client)
    assert always.calls == 1 + C.TRANSIENT_MAX_RETRIES == 3 and clock.waits == [2.25, 4.25]


@pytest.mark.parametrize("message, code, retried", [
    ("Subscription stream ended before response.completed", None, True),
    ("Subscription stream ended within an SSE event", None, True),
    ("Subscription stream interrupted (TimeoutError); no retry", "subscription_stream_interrupted", True),
    ("Subscription transport failure (OSError); no retry", "subscription_network_error", True),
    ("Subscription stream event is not valid JSON", None, False),
    ("Subscription stream response.failed: server_error; no retry or fallback", "server_error", False),
    ("ChatGPT subscription HTTP 401: unknown_error; no retry or API-key fallback", "unknown_error", False),
])
def test_only_transient_codes_are_retried(monkeypatch, message, code, retried):
    clock = Clock()
    monkeypatch.setattr(C.time, "sleep", clock.sleep)
    transport = BrokenStream(C.ScriptedV2Transport(answers()), message=message, code=code)
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=transport, clock=clock)
    if retried:
        assert plan_call(client) == C.WORKED_EXAMPLE and transport.calls == 2
    else:
        with pytest.raises(APIError):
            plan_call(client)
        assert transport.calls == 1 and client.history_stats()["transient_retry_requests"] == 0


def test_a_received_response_is_never_retried(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(C.time, "sleep", clock.sleep)
    monkeypatch.setattr(C, "TRANSIENT_ERROR_CODES", C.TRANSIENT_ERROR_CODES + ("response_model_mismatch",))
    transport = EnvelopeTransport(C.WORKED_EXAMPLE, model="gpt-other")    # an envelope arrived, then was refused
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=transport, clock=clock)
    with pytest.raises(APIError):
        plan_call(client)
    assert len(transport.calls) == 1 and clock.waits == []
    bad = EnvelopeTransport({"scene": "x", "parts": [], "hold_true": [], "steps": []})   # schema error: no retry
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=bad, clock=clock)
    with pytest.raises(SchemaError):
        plan_call(client)
    assert len(bad.calls) == 1


def test_overload_and_transient_retries_keep_separate_bounds(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(C.time, "sleep", clock.sleep)
    monkeypatch.setattr(C.random, "uniform", lambda low, high: 0.0)
    inner = OverloadedOnce(C.ScriptedV2Transport(answers()), failures=2, status=503)
    transport = BrokenStream(inner, failures=2)
    client = C.AgentSpeedUpV2Client(task_prompt=PROMPT, transport=transport, clock=clock)
    assert plan_call(client) == C.WORKED_EXAMPLE
    stats = client.history_stats()
    assert (stats["transient_retry_requests"], stats["overload_retry_requests"]) == (2, 2)
    assert transport.calls == 5 and clock.waits == [2.0, 4.0, 2.0, 4.0]


# ------------------------------------------------------------------------------------ request text safety
def test_requests_refuse_control_and_zero_width_characters():
    client, transport = make()
    with pytest.raises(ValueError, match="U\\+200B"):
        client.request_point(image(), stage="coarse", camera="wrist", text="toy car:\u200b whole car", pid="A",
                             quarter_turns=0)
    with pytest.raises(ValueError):
        plan_call(client, reask_error="needs_ref: step 1\x07")
    block = status_block()
    block["parts"][0][1] = "toy car: whole car\u200e"
    with pytest.raises(ValueError):
        plan_call(client, status_block=block, replan=True)
    with pytest.raises(ValueError):
        client.request_repair(image(), image(), quarter_turns=0, status_block=status_block(),
                              last_outcome=dict(OUTCOME, evidence_text="part C\ud800"), holding=None)
    assert transport.calls == [] and transport._meta is None              # nothing sent, nothing bound
    client.request_point(image(), stage="coarse", camera="wrist", text="toy car: caf\u00e9 roof", pid="A",
                         quarter_turns=0)                                # a printable letter is not refused


# ------------------------------------------------------------- handle lens BC-1 / BC-2 / BC-3, BX-1 / BX-2 re-asks
# The measured texts of the handle lens (scratchpad/handle/point_check.py, 24 live calls), restated here so a change
# of the module's copy is caught.
GRASP_POINT_CLAUSE = ("For a grasp_point purpose choose the point on the named part where two fingers closing from "
                      "above would hold it. Seen from above, a part can look different than in the external camera "
                      "(its top may have another colour). Anything under the fingertip band at the named edge is "
                      "hidden: if the named part lies only under the fingertips or outside this crop, answer 0.")


def ring_share(rgb, centre, *, radius=None):
    """Share of samples on the drawn ring (just inside its outer radius) that carry the marker colour."""
    radius = C.MARKER_STYLE["radius_px"] if radius is None else radius
    colour = np.array(C.MARKER_STYLE["rgb"], dtype=int)
    inner = radius - C.MARKER_STYLE["width_px"] / 2.0
    hits = total = 0
    for angle in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
        u = int(round(centre[0] + inner * math.cos(angle)))
        v = int(round(centre[1] + inner * math.sin(angle)))
        if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
            total += 1
            hits += int(np.abs(rgb[v, u].astype(int) - colour).max() <= 40)
    return hits / max(total, 1)


def marker_colour_px(rgb):
    return int(np.all(np.abs(rgb.astype(int) - np.array(C.MARKER_STYLE["rgb"])) <= 40, axis=-1).sum())


def test_point_requests_carry_the_purpose_and_the_grasp_point_clause():
    """BC-1: every pointing request names its purpose (POINT_PURPOSES, or null), and both pointing instructions end
    with the measured grasp-point clause verbatim (framed hovers: top bar 4/5 with it, 2/6 without)."""
    assert C.GRASP_POINT_CLAUSE == GRASP_POINT_CLAUSE and not vocabulary_hits(GRASP_POINT_CLAUSE)
    for kind in ("point_fine", "point_coarse"):
        assert C.instructions(kind).endswith(" " + GRASP_POINT_CLAUSE), kind
    for kind in ("plan", "replan", "repair", "review"):
        assert GRASP_POINT_CLAUSE not in C.instructions(kind), kind
    assert C.POINT_PURPOSES == kernel.POINT_PURPOSES
    client, transport = make()
    client.request_point(image(), stage="fine", camera="wrist", text="toy car: roof ridge", pid="C",
                         center_uv=(300.0, 200.0), quarter_turns=1, purpose="grasp_point")
    client.request_point(image(), stage="coarse", camera="external", text="rug: patch with the star print", pid="E",
                         quarter_turns=0, purpose="region_seed")
    client.request_point(image(), stage="coarse", camera="wrist", text="rug: patch with the star print", pid="E",
                         quarter_turns=0)
    sent = [request_text(call) for call in transport.calls]
    assert [s["purpose"] for s in sent] == ["grasp_point", "region_seed", None]
    assert list(sent[0]) == ["request", "task", "part", "camera", "image", "previous_answer_rejected", "purpose"]
    assert [m["purpose"] for m in transport.metas] == ["grasp_point", "region_seed", None]
    preflight = C.preflight_payloads(PROMPT)
    assert request_text(preflight["point_fine"])["purpose"] == "grasp_point"
    assert request_text(preflight["point_coarse"])["purpose"] == "region_seed"


def test_marker_uv_rings_the_rejected_point_in_the_fine_crop_and_the_coarse_view():
    """BC-2: marker_uv (upright pixels of the camera, like center_uv) is ringed where the model sees it: in the
    upsampled fine crop (crop coordinates x FINE_UPSAMPLE) or on the coarse view, after the frame alignment turn.
    The label and the image description say so only when the ring is drawn; without a marker the image is the
    unmarked one, byte for byte."""
    common = {"text": "toy car: roof ridge", "pid": "C", "purpose": "grasp_point"}
    rgb = image(8)
    for k in range(4):
        client, transport = make()
        centre, marker = (300.0, 200.0), (310.0, 215.0)
        plain = client.request_point(rgb, stage="fine", camera="wrist", center_uv=centre, quarter_turns=k, **common)
        client.request_point(rgb, stage="fine", camera="wrist", center_uv=centre, quarter_turns=k, marker_uv=marker,
                             previous_rejected="the pointed point lies on part D itself", **common)
        unmarked, marked = (decode(call["input"][0]["content"][2]["image_url"]) for call in transport.calls)
        shown = mark_prompting.rotate_pixel(marker, k, 512)
        up = ((shown[0] - plain.origin[0]) * C.FINE_UPSAMPLE, (shown[1] - plain.origin[1]) * C.FINE_UPSAMPLE)
        assert ring_share(marked, up) >= 0.8 and ring_share(unmarked, up) <= 0.2, k
        assert marker_colour_px(unmarked) == 0 < marker_colour_px(marked)
        labels = [call["input"][0]["content"][1]["text"] for call in transport.calls]
        assert "ringed" not in labels[0] and "ringed" in labels[1]
        texts = [request_text(call) for call in transport.calls]
        assert "ring" not in texts[0]["image"] and "ring" in texts[1]["image"]
        assert texts[1]["previous_answer_rejected"] == "the pointed point lies on part D itself"
        assert transport.metas[0]["marker_drawn"] is False and transport.metas[1]["marker_drawn"] is True
        assert transport.metas[1]["marker_uv"] == list(marker) and transport.metas[0]["marker_uv"] is None
        # the coarse whole view: the ring sits at the turned pixel itself
        client.request_point(rgb, stage="coarse", camera="wrist", quarter_turns=k, marker_uv=marker, **common)
        coarse = decode(transport.calls[-1]["input"][0]["content"][2]["image_url"])
        assert ring_share(coarse, shown) >= 0.8 and transport.metas[-1]["marker_drawn"] is True
    # outside the fine crop: nothing is drawn and nothing is claimed
    client, transport = make()
    client.request_point(rgb, stage="fine", camera="wrist", center_uv=(300.0, 200.0), quarter_turns=0,
                         marker_uv=(40.0, 480.0), **common)
    client.request_point(rgb, stage="fine", camera="wrist", center_uv=(300.0, 200.0), quarter_turns=0, **common)
    first, second = transport.calls
    assert first["input"][0]["content"][2]["image_url"] == second["input"][0]["content"][2]["image_url"]
    assert "ringed" not in first["input"][0]["content"][1]["text"] and transport.metas[0]["marker_drawn"] is False
    url, origin = C.fine_overlay_at(rgb, (300.0, 200.0))
    assert C.fine_overlay_at(rgb, (300.0, 200.0), marker_uv=None) == (url, origin)    # unchanged default drawing
    assert rgb[205, 305].tolist() == image(8)[205, 305].tolist()                      # the input is never drawn on


def test_repair_reask_carries_the_issue_text_and_the_wrist_image_rings_the_rejected_point():
    """BX-1 / BRR-2 / BX-2 transport: request_repair sends ``reask_error`` as previous_answer_rejected (null
    otherwise) and rings ``marker_uv`` (upright wrist pixels) on the turned wrist image only; the instructions explain
    both."""
    for kind in ("repair", "review"):
        text = C.instructions(kind)
        assert "previous_answer_rejected" in text and "ring" in text and not vocabulary_hits(text)
    ext, wrist = image(1), image(2)
    for k in range(4):
        client, transport = make()
        client.request_repair(ext, wrist, quarter_turns=k, status_block=status_block(), last_outcome=OUTCOME,
                              holding="C")
        client.request_repair(ext, wrist, quarter_turns=k, status_block=status_block(), last_outcome=OUTCOME,
                              holding="C", reask_error="repoint_limit: repoint of part C failed 2 times",
                              marker_uv=(120.0, 330.0))
        plain, marked = (request_text(call) for call in transport.calls)
        assert plain["previous_answer_rejected"] is None
        assert marked["previous_answer_rejected"] == "repoint_limit: repoint of part C failed 2 times"
        images = [[decode(c["image_url"]) for c in call["input"][0]["content"] if c["type"] == "input_image"]
                  for call in transport.calls]
        assert np.array_equal(images[0][0], images[1][0]) and np.array_equal(images[0][0], ext)
        shown = mark_prompting.rotate_pixel((120.0, 330.0), k, 512)
        assert ring_share(images[1][1], shown) >= 0.8 and ring_share(images[0][1], shown) <= 0.2
        assert np.array_equal(images[0][1], mark_prompting.rotate_quarter(wrist, k))
        labels = [[c["text"] for c in call["input"][0]["content"][1:] if c["type"] == "input_text"]
                  for call in transport.calls]
        assert "ringed" not in labels[0][1] and "ringed" in labels[1][1] and "ringed" not in labels[1][0]
        assert transport.metas[1]["marker_drawn"] is True and transport.metas[0]["marker_drawn"] is False
        assert transport.metas[1]["previous_rejected"].startswith("repoint_limit")
    client, transport = make()
    for bad in ({"reask_error": "x" * (P.ISSUES_TEXT_MAX_CHARS + 1)}, {"reask_error": ""},
                {"marker_uv": (1.0, float("inf"))}, {"marker_uv": (1.0,)}):
        with pytest.raises(ValueError):
            client.request_repair(ext, wrist, quarter_turns=0, status_block=status_block(), last_outcome=OUTCOME,
                                  holding=None, **bad)
    assert transport.calls == []


def test_plan_instructions_name_parts_by_shape_and_relative_position():
    """BC-3: a part text names the part by its shape and its position relative to its object or a neighbouring
    object, colour only secondary (the wrist and external views differ). Relational text found the hidden handle 1/1
    in the coarse view (baseline 0/2) and the top bar 2/2 framed."""
    rule = C._NAMING
    assert not vocabulary_hits(rule) and rule.isascii()
    for phrase in ("shape", "position relative to its own object or a neighbouring object", "colour",
                   "secondary"):
        assert phrase in rule, phrase
    for kind in ("plan", "replan"):
        assert rule in C.instructions(kind)
    for kind in ("point_fine", "point_coarse", "repair", "review"):
        assert rule not in C.instructions(kind)
    assert C.PROMPT_VERSION == "agent-speedup-v2-prompt-4" and C.CLIENT_VERSION == "agent-speedup-v2-client-4"
