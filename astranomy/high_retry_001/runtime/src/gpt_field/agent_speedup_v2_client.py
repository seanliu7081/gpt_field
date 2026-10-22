"""agent_SpeedUp_v2 model-interface client (contract §C12; spec §3.2, §3.5, §3.6, §4.2; AC-1(b), AC-5).

``AgentSpeedUpV2Client`` is a stateless subscription client on v1's audited request path (the pattern of the v1
agent client): one user message per request, no retained transcript or image history, the insertion-order wire
(``ordered_sse_transport``) so ``scene`` and ``why`` precede the choices, reasoning effort hard-set to medium, and
the bounded same-request overload retry. Its only task input is the episode's official sentence, a required
constructor keyword ``task_prompt``; every payload carries it as ``"task"`` [C-6], and observations are checked
against it with the base chain's prompt check. It never receives a task key. One instruction text per request kind
serves every task; the instructions and the worked example carry no lint-vocabulary word.

Image protocol (§3.6): the external image clean and with the 8x8 numbered grid; the wrist image turned by k
counter-clockwise quarter turns (frame alignment) clean and with its grid; for fine pointing a crop of FINE_CROP_PX
centred on a pixel of the shown (turned) image, upsampled by FINE_UPSAMPLE with 8x8 numbered points
(``fine_overlay_at``). Answers are mapped back to upright pixels (``fine_point_pixel_at`` or the coarse cell
centre, then ``unrotate_pixel``). Crop origins are in the shown frame. The model outputs no number other than a
cell or point id, no direction and no completion claim (AC-5): the schemas are the plan module's.

Offline: ``ScriptedV2Transport`` returns completed envelopes from a ``ScriptedAnswers`` object
(``answer(kind, meta)``); the client binds the executor-side request meta (crop geometry, turns, part id) to it
before every request [C-11]. The meta is never sent to the model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image, ImageDraw

from .agent_speedup_v2_plan import (ISSUES_TEXT_MAX_CHARS, PLAN_SCHEMA_VERSION, STATUS_BLOCK_KEYS, TEXT_MAX_CHARS,
                                    PlanParseError, conform, parse_plan, plan_status_block, pointing_stage,
                                    schemas, validate_plan)
from .agent_speedup_v2_types import (CAMERAS, PIDS, REASONS, WRIST_IMAGE_PX, PointReply, StepState, jsonable,
                                     th_value)
from .astra import ENDPOINT, APIError, SchemaError, _canonical
from .chatgpt_plan import UNSUPPORTED_FIELDS, _failure_metadata
from .direct_history import _allowed
from .hybrid_subscription import ChatGPTPlanHybridClient
from .mark_prompting import (COARSE_GRID, FINE_GRID, FINGERTIP_EDGE, _font, _grid_lines, _label, coarse_cell_center,
                             coarse_overlay, encode_png_data_url, rotate_pixel, rotate_quarter, unrotate_pixel)
from .ordered_transport import ORDERED_TRANSPORT_VERSION, ordered_dumps, ordered_sse_transport
from .speed_first_hybrid import OVERLOAD_RETRY_POLICY

CLIENT_VERSION = "agent-speedup-v2-client-1"
PROMPT_VERSION = "agent-speedup-v2-prompt-1"
HISTORY_PROTOCOL = "agent-speedup-v2-stateless"
REQUEST_KINDS = ("plan", "replan", "point_fine", "point_coarse", "repair", "review")
FORMAT_NAMES = {kind: f"agent_speedup_v2_{kind}" for kind in REQUEST_KINDS}
POINTING_KEYS = {"point_fine": "point_id", "point_coarse": "cell_id"}
MODEL = "gpt-6-astra"
REASONING_EFFORT = "medium"
OFFLINE_TOKEN = "offline-oauth-fixture"

LABELS = {
    "FINE_CROP_PX": ("DD", "V6 fine crop side 128 px: v1's wide crop, one 64 px cell plus half a cell on each side "
                           "(mark_prompting.fine_crop_size(512, WIDE_MARGIN_CELLS)); spec §3.6 item 3"),
    "FINE_UPSAMPLE": ("DD", "V6 upsample x4: the 128 px crop is shown at 512 px (mark_prompting.fine_upsample for the "
                            "wide crop)"),
    "MARK_COUNT": ("DD", "8x8 numbered cells or points, ids 1-64, 0 = not visible (v1 agent_speedup_v1_client.py:49)"),
    "COARSE_QUANT_PX": ("DD", "half a 64 px coarse cell of a 512 px image (spec §3.6 item 1; §C4.1 Seed)"),
    "FINE_QUANT_PX": ("DD", "half the 16 px point spacing of a 128 px crop with 8x8 points (§C4.1 Seed)"),
    "MAX_REQUEST_TEXT_CHARS": ("DD", "request text bound, restated from v1 agent_speedup_v1_client.py:57"),
    "EVIDENCE_MAX_CHARS": ("DD", "spec §3.7 evidence rule: executor-written evidence text is at most 200 characters"),
    "PREVIOUS_REJECTED_MAX_CHARS": ("DD", "§C4.7 PointNeed.previous_rejected: the measured reason sent with a re-point "
                                          "is at most 120 characters"),
    "OVERLOAD_MAX_ATTEMPTS": ("DD", "bounded same-request overload retry: at most 3 attempts (v1 "
                                    "agent_speedup_v1_client.py:563, speed-first policy)"),
    "OVERLOAD_MAX_WAIT_S": ("DD", "a scheduled overload retry waits at most 60 s (v1 agent_speedup_v1_client.py:571)"),
    "OVERLOAD_BACKOFF_BASE_S": ("DD", "overload backoff base**attempt seconds (v1 agent_speedup_v1_client.py:569)"),
    "OVERLOAD_JITTER_S": ("DD", "uniform overload backoff jitter up to 1 s (v1 agent_speedup_v1_client.py:569)"),
    "OVERLOAD_HTTP_STATUS": ("DD", "HTTP 503 service unavailable: the overload status that may be retried (v1 "
                                   "agent_speedup_v1_client.py:565)"),
    "VISIBLE_TOKEN_BUDGETS": ("DD", "spec §3.2 visible-token budgets [P], upper ends: plan and replan 340, pointing 6, "
                                    "repair and review 80 without a new step; diagnosis only (history_stats counts "
                                    "answers above them), never a cap: the subscription route sends no output limit"),
    "REPAIR_STEP_TOKEN_BUDGET": ("DD", "spec §3.2 visible-token budget [P] of a repair or review answer that carries a "
                                       "new step (insert_step): at most 300; diagnosis only, never a cap"),
    "FINE_STYLE": ("DD", "fine overlay drawing restated from v1 mark_prompting.fine_overlay (:240-252): faint grid "
                         "lines, point fill, font size and point radius per point spacing"),
    "PREFLIGHT_GREY": ("DD", "synthetic preflight observation: uniform mid-grey images (no scene content)"),
    "WORKED_EXAMPLE": ("DD", "spec §3.5 item 5 worked example; its w / x cells are illustrative grid ids, not "
                             "thresholds"),
}

FINE_CROP_PX = 128
FINE_UPSAMPLE = 4
MARK_COUNT = 64
COARSE_QUANT_PX = 32.0
FINE_QUANT_PX = 8.0
MAX_REQUEST_TEXT_CHARS = 60_000
EVIDENCE_MAX_CHARS = 200
PREVIOUS_REJECTED_MAX_CHARS = 120
OVERLOAD_MAX_ATTEMPTS = 3
OVERLOAD_MAX_WAIT_S = 60.0
OVERLOAD_BACKOFF_BASE_S = 2.0
OVERLOAD_JITTER_S = 1.0
OVERLOAD_HTTP_STATUS = 503
VISIBLE_TOKEN_BUDGETS = {"plan": 340, "replan": 340, "point_fine": 6, "point_coarse": 6, "repair": 80,
                         "review": 80}
REPAIR_STEP_TOKEN_BUDGET = 300
FINE_STYLE = {"grid_core_rgba": (255, 255, 255, 70), "point_rgba": (255, 225, 20, 255), "outline_rgba": (0, 0, 0, 255),
              "font_scale": (13, 60), "min_font_px": 8, "radius_divisor": 15, "min_radius_px": 2,
              "label_gap_px": 2}
PREFLIGHT_GREY = 128

_KIND_BY_FORMAT = {name: kind for kind, name in FORMAT_NAMES.items()}
_PNG_PREFIX = "data:image/png;base64,"
_PLAN_KINDS = ("plan", "replan")
_REPAIR_KINDS = ("repair", "review")


# --------------------------------------------------------------------------------------------------- example
# Neutral objects (spec §3.5 item 5) and a relation mix that is no slice or regression task's plan (turn by a
# rotation criterion, then hold with a two-part grip preference list): the example teaches the vocabulary without
# being a template of a gated task's nominal plan.
WORKED_EXAMPLE = {
    "scene": "music box with a crank on its side on a rug; toy car beside a glass vase",
    "parts": [
        {"id": "A", "text": "toy car: roof ridge", "move": "free", "w": 36, "x": 28},
        {"id": "B", "text": "toy car: rear spoiler", "move": "free", "w": 37, "x": 28},
        {"id": "C", "text": "toy car: whole car", "move": "free", "w": 36, "x": 28},
        {"id": "D", "text": "glass vase: whole vase", "move": "free", "w": 21, "x": 19},
        {"id": "E", "text": "music box: small crank on the side", "move": "hinged", "w": 52, "x": 45}],
    "hold_true": [],
    "steps": [
        {"do": "turn", "obj": "E", "grip": ["E"], "ref": None, "sense": "any", "keep": [{"c": "clear_of", "p": "D"}],
         "until": [{"o": "half_turn", "p": None, "txt": None}, {"o": "stop", "p": None, "txt": None}]},
        {"do": "hold", "obj": "C", "grip": ["A", "B"], "ref": None, "sense": "any",
         "keep": [{"c": "upright", "p": "C"}, {"c": "clear_of", "p": "D"}], "until": []}]}


# ------------------------------------------------------------------------------------------------ instructions
_ROBOT = (
    "You plan for a robot arm with one parallel-jaw gripper (about 79 mm maximum opening) that starts with the hand "
    "pointing straight down. The robot measures every distance, height, direction and angle itself by moving, "
    "touching and looking; you name what matters and which physical relations to establish, never how to move.")
_IMAGES = (
    "IMAGES. Four images, all upright (not mirrored): 1 the external camera, clean; 2 the same with an 8x8 grid of "
    "numbered cells; 3 the wrist camera, clean; 4 the same with the grid. The wrist image is turned so that its left "
    "and right match the external image; the request names the edge where the robot's own fingertips appear (they "
    "are never a part). Cells are numbered 1-64 row-major from the top left.")
_VOCABULARY = (
    "WRITE ONE JSON PLAN.\n"
    "- scene: one short line on what you see.\n"
    "- parts: every object part the steps, keeps and criteria need, with ids A-H. text is \"<object>: <visible "
    "part>\", with distinguishing appearance or position when similar objects exist. move is your prior for how the "
    "part moves: free (a loose object), fixed, hinged (rotates about its mount), sliding (moves along a track) or "
    "unknown; the robot verifies it by probing. w is the part's cell in image 4 and x its cell in image 2, 0 when "
    "it is not visible in that image.\n"
    "- steps, in order; each establishes one physical relation: hold (obj is lifted and moves with the hand); turn "
    "(a rotary part obj, held at grip, is rotated about its own axis); slide (a sliding part obj, held at grip, "
    "moves along its own track; sense open = away from the body it slides in, close = into it); place_on (obj rests "
    "on the top surface of ref, released and at rest); place_in (obj rests inside the cavity of ref, below its "
    "rim); push_to (obj slides on its support to the spot ref without lifting); release (the hand opens and leaves "
    "obj). grip lists up to two parts to hold, in preference order (an empty list means obj itself); ref is the "
    "reference part or null; sense is any, open or close. The robot adds the grasps and releases a step needs.\n"
    "- keep: constraints held during a step: upright (part p keeps its in-hand pose) and clear_of (the hand and the "
    "held object stay clear of part p).\n"
    "- until: completion criteria, the first one met ends the step: effect (part p visibly changes as txt says), "
    "stop (the mechanism reaches its end stop), fits (the opening is long enough for the free part p), quarter_turn "
    "and half_turn (of the turned part). turn takes effect, stop, quarter_turn or half_turn and needs one; slide "
    "takes stop, fits or effect; place_on and place_in take only effect; hold, push_to and release take none. "
    "Unused p and txt are null.\n"
    "- hold_true: relations the images clearly show as already true and that must stay true (on, in, open, closed, "
    "effect; a is the part, b the other part or null).")
_RULES = (
    "RULES. Name only things visible in at least one image. As the first grip give a part thin enough to fit "
    "between the fingers (for example a thin protrusion, a ridge or a thin wall), and a second choice if one exists. "
    "Include enabling steps (make an opening before placing into it). Describe completion only with the criteria "
    "above. List a relation as already true only when the images clearly show it. Never give directions, "
    "distances, heights, angles or coordinates. Write scene first.")
_REPLAN = (
    "REPLAN. status gives your previous plan as the robot executed it: the parts as [id, text], every step with its "
    "status (pending, active, done, failed, stays, skipped) and reason code, the already-true relations and the held "
    "part. Write a complete new plan for the work that remains; done steps stay true. A part keeps its measured "
    "registration while its id and text are unchanged.")
_POINT_COMMON = (
    "You help a robot locate one part named by part (your own text from the plan). camera names the view; image "
    "describes it, including the edge where the robot's own fingertips appear in a wrist view (they are never the "
    "part). If previous_answer_rejected is present, your earlier answer failed the robot's measured check; choose "
    "again. Answer only this question.")
_POINT_FINE = (
    "The image is an enlarged crop of one camera view (not mirrored) with 64 numbered points in an 8x8 pattern, "
    "1-64 row-major from the top left. Choose the single point that lies on the part (the nearest point if none "
    "lies exactly on it). Return {\"point_id\": <id>}, or {\"point_id\": 0} if the part is not visible in this crop.")
_POINT_COARSE = (
    "The image is one camera view (not mirrored) with an 8x8 grid of numbered cells, 1-64 row-major from the top "
    "left. Choose the single cell that contains the part; if it spans several cells, the cell holding most of it. "
    "Return {\"cell_id\": <id>}, or {\"cell_id\": 0} if the part is not visible in this image.")
_ACTIONS = (
    "ACTIONS (choose one): retry (rerun the failed step from its last satisfied precondition); repoint (drop the "
    "registration of part and point it again); use_part (grasp part instead; it must be an existing part); "
    "insert_step (put step before the failed step, for example to clear an obstacle); skip_step (the failed step "
    "becomes a kept relation if it currently holds, otherwise it is dropped); swap_next (exchange the failed step "
    "with the next one); replan (you will be asked for a new plan); hold_still (hold position). Set part only for "
    "repoint and use_part and step only for insert_step (the same step object as in a plan); otherwise null. Write "
    "why first: one short sentence. Never give directions, distances, heights, angles or coordinates.")
_REPAIR_INPUTS = (
    "plan gives your relation plan as the robot executes it: the parts as [id, text], every step with its status "
    "(pending, active, done, failed, stays, skipped) and reason code, the already-true relations and the held part. "
    "last_outcome gives the measured reason code, a short evidence text (magnitudes in cm and degrees, parts by id) "
    "and the thresholds it was compared against, or null. Two clean images follow: the external camera and the "
    "wrist camera, turned as named.")
_REPAIR = "A step of your relation plan failed after the robot's own local retries. "
_REVIEW = ("Every step of your relation plan has been carried out and the robot's observers report each planned "
           "relation holding. If the planned relations hold in the images, answer hold_still. ")


def _instruction_texts() -> dict[str, str]:
    example = "EXAMPLE (other objects; cells illustrative): " + ordered_dumps(WORKED_EXAMPLE)
    plan = "\n".join((_ROBOT, _IMAGES, _VOCABULARY, _RULES, example))
    point = f"{PROMPT_VERSION}, " + "{kind} request. " + _POINT_COMMON + " "
    repair_head = (f"{PROMPT_VERSION}, " + "{kind} request. A robot executes your relation plan and measures "
                   "everything itself. ")
    return {"plan": f"{PROMPT_VERSION}, plan request. " + plan,
            "replan": f"{PROMPT_VERSION}, replan request. " + plan + "\n" + _REPLAN,
            "point_fine": point.format(kind="point_fine") + _POINT_FINE,
            "point_coarse": point.format(kind="point_coarse") + _POINT_COARSE,
            "repair": repair_head.format(kind="repair") + _REPAIR + _REPAIR_INPUTS + "\n" + _ACTIONS,
            "review": repair_head.format(kind="review") + _REVIEW + _REPAIR_INPUTS + "\n" + _ACTIONS}


_INSTRUCTIONS = _instruction_texts()


def instructions(kind: str) -> str:
    """§3.5 items 1-6: one text per request kind, identical for every task; no task text."""
    if kind not in REQUEST_KINDS:
        raise ValueError(f"kind must be one of {', '.join(REQUEST_KINDS)}")
    return _INSTRUCTIONS[kind]


def instructions_sha256() -> str:
    """sha256 over the instruction texts of all kinds in REQUEST_KINDS order (the manifest's instruction id)."""
    material = json.dumps([[kind, _INSTRUCTIONS[kind]] for kind in REQUEST_KINDS], ensure_ascii=False,
                          separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def crop_threshold():
    """V6 runtime record of the fine crop (contract §C16); pointing evidence may carry it."""
    return th_value("V6_crop_px", FINE_CROP_PX, "DD", "fine crop side, shown upsampled with 8x8 numbered points",
                    upsample=FINE_UPSAMPLE)


# --------------------------------------------------------------------------------------------- image protocol
def _square_rgb(rgb) -> np.ndarray:
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.shape[0] != pixels.shape[1]:
        raise ValueError("images must be square H x W x 3 uint8 RGB arrays")
    if pixels.shape[1] % COARSE_GRID:
        raise ValueError(f"image width must be divisible by {COARSE_GRID}")
    return pixels


def _protocol_image(rgb) -> np.ndarray:
    pixels = _square_rgb(rgb)
    if pixels.shape[1] != WRIST_IMAGE_PX:
        raise ValueError(f"the pointing protocol uses {WRIST_IMAGE_PX} px upright images")
    return pixels


def _turns(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < len(FINGERTIP_EDGE):
        raise ValueError("quarter_turns must be an integer in 0..3")
    return int(value)


def _crop_px(value, width: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not FINE_GRID <= int(value) <= width:
        raise ValueError(f"crop_px must be an integer in {FINE_GRID}..{width}")
    return int(value)


def crop_origin(center_uv, crop_px: int, width: int) -> tuple[int, int]:
    """Top-left of a crop_px crop centred on ``center_uv`` (u right, v down), clamped inside the image."""
    centre = np.asarray(center_uv, dtype=float)
    if centre.shape != (2,) or not np.isfinite(centre).all():
        raise ValueError("center_uv must be two finite numbers")
    limit = width - crop_px
    return tuple(int(min(max(round(float(c) - crop_px / 2.0), 0), limit)) for c in centre)


def fine_point_pixel_at(origin, point_id: int, *, crop_px=FINE_CROP_PX) -> tuple[float, float]:
    """Pixel (of the image the crop was taken from, i.e. the shown frame) of fine point ``point_id`` (1..64, row-major
    8x8, v1 layout ``mark_prompting.fine_point_pixel``) of a crop_px crop at integer ``origin``."""
    if (not isinstance(origin, (tuple, list, np.ndarray)) or len(origin) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in origin)):
        raise ValueError("crop origin must be two integers")
    if isinstance(point_id, bool) or not isinstance(point_id, (int, np.integer)) or not 1 <= int(point_id) <= MARK_COUNT:
        raise ValueError(f"point id must be an integer in 1..{MARK_COUNT}")
    if isinstance(crop_px, bool) or not isinstance(crop_px, (int, np.integer)) or int(crop_px) < FINE_GRID:
        raise ValueError(f"crop_px must be an integer >= {FINE_GRID}")
    row, col = divmod(int(point_id) - 1, FINE_GRID)
    spacing = int(crop_px) / FINE_GRID
    return (float(int(origin[0]) + (col + 0.5) * spacing), float(int(origin[1]) + (row + 0.5) * spacing))


def _encode(frame: Image.Image) -> str:
    return encode_png_data_url(np.asarray(frame))


def fine_overlay_at(rgb_upright, center_uv, *, crop_px=FINE_CROP_PX,
                    upsample=FINE_UPSAMPLE) -> tuple[str, tuple[int, int]]:
    """Crop crop_px x crop_px centred on ``center_uv`` (clamped inside the image), upsample, draw 8x8 numbered points
    (v1's fine-overlay drawing). Returns (PNG data URL, crop origin in the given image's pixels). Point id k sits at
    ``fine_point_pixel_at(origin, k, crop_px=crop_px)`` of the given image."""
    pixels = _square_rgb(rgb_upright)
    width = pixels.shape[1]
    crop = _crop_px(crop_px, width)
    if isinstance(upsample, bool) or not isinstance(upsample, (int, np.integer)) or int(upsample) < 1:
        raise ValueError("upsample must be a positive integer")
    x0, y0 = origin = crop_origin(center_uv, crop, width)
    size = crop * int(upsample)
    frame = Image.fromarray(np.array(pixels[y0:y0 + crop, x0:x0 + crop], dtype=np.uint8, order="C"), mode="RGB")
    frame = frame.resize((size, size), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(frame, "RGBA")
    _grid_lines(draw, size, FINE_GRID, halo=None, core=FINE_STYLE["grid_core_rgba"])
    spacing = size / FINE_GRID
    numerator, denominator = FINE_STYLE["font_scale"]
    font = _font(max(FINE_STYLE["min_font_px"], round(spacing * numerator / denominator)))
    radius = max(FINE_STYLE["min_radius_px"], round(spacing / FINE_STYLE["radius_divisor"]))
    gap = FINE_STYLE["label_gap_px"]
    for index in range(FINE_GRID * FINE_GRID):
        row, col = divmod(index, FINE_GRID)
        x, y = (col + 0.5) * spacing, (row + 0.5) * spacing
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=FINE_STYLE["point_rgba"],
                     outline=FINE_STYLE["outline_rgba"], width=1)
        _label(draw, (x + radius + gap, y - radius - 1), str(index + 1), font, frame.size, anchor="bottom_left")
    return _encode(frame), origin


def plan_images(external_rgb, wrist_rgb, quarter_turns: int) -> list[tuple[str, str]]:
    """[("external clean", url), ("external grid", url), ("wrist clean", url), ("wrist grid", url)]; the wrist image
    is turned by ``quarter_turns`` counter-clockwise quarter turns before both of its renderings."""
    external = _protocol_image(external_rgb)
    wrist = rotate_quarter(_protocol_image(wrist_rgb), _turns(quarter_turns))
    return [("external clean", encode_png_data_url(external)), ("external grid", coarse_overlay(external)),
            ("wrist clean", encode_png_data_url(wrist)), ("wrist grid", coarse_overlay(wrist))]


_IMAGE_LABELS = {"external clean": "Image 1: external camera, clean:",
                 "external grid": "Image 2: external camera with the 8x8 grid of numbered cells:",
                 "wrist clean": "Image 3: wrist camera, clean (turned; fingertips at the {edge} edge):",
                 "wrist grid": "Image 4: wrist camera with the 8x8 grid of numbered cells:"}
_PLAN_IMAGES_TEXT = ("1 external clean, 2 external grid, 3 wrist clean (fingertips at the {edge} edge), "
                     "4 wrist grid; cells 1-64 row-major from the top left")


# ------------------------------------------------------------------------------------------ request checks
def _png_data_url(value, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(_PNG_PREFIX):
        raise ValueError(f"{name} must be an inline PNG data URL")
    return value


def _json_text(value, name: str) -> str:
    try:
        text = ordered_dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-safe (plain types, finite numbers)") from exc
    if len(text) > MAX_REQUEST_TEXT_CHARS:
        raise ValueError(f"{name} exceeds {MAX_REQUEST_TEXT_CHARS} characters")
    return text


def _pid_or_none(value, name: str):
    if value is not None and value not in PIDS:
        raise ValueError(f"{name} must be a part id or None")
    return value


def _short_text(value, limit: int, name: str):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters or None")
    return value


def _status_block(value, name: str = "status_block"):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(STATUS_BLOCK_KEYS):
        raise ValueError(f"{name} must be the plan status block (keys {', '.join(STATUS_BLOCK_KEYS)})")
    for row in value["steps"]:
        reason = row.get("reason") if isinstance(row, Mapping) else None
        if reason is not None and reason not in REASONS:
            raise ValueError(f"{name} step reasons must be REASONS codes")
    _json_text(value, name)
    return jsonable(dict(value))


def _last_outcome(value):
    """[C-45] exactly {"reason", "evidence_text", "thresholds": [[name, value], ...]} or None."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"reason", "evidence_text", "thresholds"}:
        raise ValueError("last_outcome must have exactly the keys reason, evidence_text, thresholds")
    if value["reason"] not in REASONS:
        raise ValueError("last_outcome.reason must be a REASONS code")
    if not isinstance(value["evidence_text"], str) or len(value["evidence_text"]) > EVIDENCE_MAX_CHARS:
        raise ValueError(f"last_outcome.evidence_text must be a string of at most {EVIDENCE_MAX_CHARS} characters")
    pairs = []
    for pair in value["thresholds"]:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or not isinstance(pair[0], str)
                or isinstance(pair[1], bool) or not isinstance(pair[1], (int, float, np.integer, np.floating))
                or not math.isfinite(float(pair[1]))):
            raise ValueError("last_outcome.thresholds must be [name, finite value] pairs")
        pairs.append([pair[0], float(pair[1])])
    return {"reason": value["reason"], "evidence_text": value["evidence_text"], "thresholds": pairs}


def _validate_pointing(data, kind: str) -> dict:
    key = POINTING_KEYS[kind]
    if not isinstance(data, dict) or set(data) != {key}:
        raise SchemaError(f"{kind} output must contain exactly {key}")
    value = data[key]
    if value is None:
        return {key: None}
    if type(value) is not int or not 0 <= value <= MARK_COUNT:
        raise SchemaError(f"{key} must be an integer in [0, {MARK_COUNT}]")
    return {key: value or None}


def _no_credentials() -> str:
    raise RuntimeError("no ChatGPT subscription token provider was configured")


def _offline_token() -> str:
    return OFFLINE_TOKEN


# ------------------------------------------------------------------------------------------------- client
class AgentSpeedUpV2Client(ChatGPTPlanHybridClient):
    """Stateless plan / point / repair requests over the audited subscription path (contract §C12)."""

    history_protocol = HISTORY_PROTOCOL
    # Plain class attributes, so the runner can read the client identity from the class it is handed (the manifest
    # is written before the client is constructed); they shadow the base chain's computed prompt_version property.
    prompt_version = PROMPT_VERSION
    reasoning_effort = REASONING_EFFORT

    def __init__(self, config=None, *, task_prompt: str, token_provider=None, transport=None, log_dir=None,
                 clock=time.monotonic):
        if config is not None and getattr(config, "model", None) != MODEL:
            raise ValueError(f"agent_SpeedUp_v2 requires {MODEL}")
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise ValueError("task_prompt must be the episode's official sentence (a nonempty string)")
        transport = transport if transport is not None else ordered_sse_transport
        if token_provider is None:
            # offline transports get the fixture token; a live transport without a provider fails at its first
            # request (construction and payload previews need no credentials)
            token_provider = (_offline_token if getattr(transport, "offline_mock", False) is True
                              else _no_credentials)
        # Everything reset_history / history_stats touch must exist before super(): the base calls reset_history().
        self._request_kind = None
        self.last_output_key_order = None
        self.last_request_meta = None
        self._instruction_text = {kind: instructions(kind) for kind in REQUEST_KINDS}
        self._schemas = schemas()
        super().__init__(config, reasoning_effort=REASONING_EFFORT, token_provider=token_provider,
                         transport=transport, log_dir=log_dir, clock=clock, task_prompt=task_prompt, image_history=1)

    @property
    def offline_mock(self) -> bool:
        return getattr(self.transport, "offline_mock", False) is True

    def check_observation(self, observation):
        """The parsed observation (direct_history._allowed); ValueError when its sentence is not ``task_prompt``."""
        data = observation.to_dict() if callable(getattr(observation, "to_dict", None)) else observation
        return _allowed(data, self.task_prompt)

    def reset_history(self) -> None:
        with self._history_lock:
            super().reset_history()
            self._overload_attempt = 1
            self._overload_retry_requests = 0
            self._overload_retry_wait_s = 0.0
            self._kind_calls = {kind: 0 for kind in REQUEST_KINDS}
            self._kind_accepted = {kind: 0 for kind in REQUEST_KINDS}
            self._kind_output_tokens = {kind: 0 for kind in REQUEST_KINDS}
            self._kind_reasoning_tokens = {kind: 0 for kind in REQUEST_KINDS}
            self._kind_over_budget = {kind: 0 for kind in REQUEST_KINDS}

    def history_stats(self) -> dict:
        with self._history_lock:
            visible = {kind: self._kind_output_tokens[kind] - self._kind_reasoning_tokens[kind]
                       for kind in REQUEST_KINDS}
            return {"protocol": HISTORY_PROTOCOL, "accepted_requests": self._observation_count,
                    "messages": len(self._history), "retained_image_pairs": 0,
                    "calls_by_kind": dict(self._kind_calls), "accepted_by_kind": dict(self._kind_accepted),
                    "output_tokens_by_kind": dict(self._kind_output_tokens),
                    "reasoning_tokens_by_kind": dict(self._kind_reasoning_tokens),
                    "visible_tokens_by_kind": visible, "visible_token_budgets": dict(VISIBLE_TOKEN_BUDGETS),
                    "repair_step_visible_token_budget": REPAIR_STEP_TOKEN_BUDGET,
                    "over_visible_budget_by_kind": dict(self._kind_over_budget),
                    "overload_retry_policy": OVERLOAD_RETRY_POLICY,
                    "overload_retry_requests": self._overload_retry_requests,
                    "overload_retry_wait_s": self._overload_retry_wait_s}

    def budget_snapshot(self) -> dict:
        snapshot = super().budget_snapshot()
        if self.offline_mock:
            snapshot = {**snapshot, "billing": "offline_mock", "offline_mock": True}
        return snapshot

    def _record_usage(self, response, reservation=None) -> None:
        super()._record_usage(response, reservation)
        kind = self._request_kind
        usage = response.get("usage") if isinstance(response, Mapping) else None
        if kind not in REQUEST_KINDS or not isinstance(usage, Mapping):
            return
        output = usage.get("output_tokens")
        details = usage.get("output_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
        if type(output) is int and output >= 0:
            self._kind_output_tokens[kind] += output
            spent = reasoning if type(reasoning) is int and 0 <= reasoning <= output else 0
            self._kind_reasoning_tokens[kind] += spent
            if output - spent > self._visible_budget(kind, response):
                self._kind_over_budget[kind] += 1

    def _visible_budget(self, kind: str, response) -> int:
        """The §3.2 visible-token budget of this answer: a repair or review answer that carries a new step gets
        REPAIR_STEP_TOKEN_BUDGET (an answer that cannot be read is given the larger bound: diagnosis only)."""
        if kind not in _REPAIR_KINDS:
            return VISIBLE_TOKEN_BUDGETS[kind]
        try:
            data = self._extract(response)
        except Exception:
            return REPAIR_STEP_TOKEN_BUDGET
        if isinstance(data, Mapping) and data.get("step") is None:
            return VISIBLE_TOKEN_BUDGETS[kind]
        return REPAIR_STEP_TOKEN_BUDGET

    # payload ---------------------------------------------------------------------------------------------
    def _content(self, request, mode: str) -> list[dict]:
        if not isinstance(request, Mapping) or set(request) != {"fields", "images"}:
            raise ValueError("a v2 request must contain exactly fields and images")
        fields = request["fields"]
        if not isinstance(fields, Mapping) or {"request", "task"} & set(fields):
            raise ValueError("request fields must be a JSON object without request / task (the client sets them)")
        text = _json_text({"request": mode, "task": self.task_prompt, **fields}, "request text")
        content = [{"type": "input_text", "text": text}]
        for item in request["images"]:
            if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError("images must be (label, PNG data URL) pairs")
            content.append({"type": "input_text", "text": item[0]})
            content.append({"type": "input_image", "detail": "high", "image_url": _png_data_url(item[1], "image")})
        return content

    def build_payload(self, request, grid=None, mode="plan") -> dict[str, Any]:
        """Preview-safe: no authentication, no history, no network."""
        if mode not in REQUEST_KINDS:
            raise ValueError(f"mode must be one of {', '.join(REQUEST_KINDS)}")
        if grid is not None and grid != self.grid:
            raise ValueError("Requested grid must match the configured client grid")
        content = self._content(request, mode)
        self._request_kind = mode
        return {
            "model": self.config.model, "service_tier": "default", "store": False,
            "instructions": self._instruction_text[mode],
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": FORMAT_NAMES[mode], "strict": True,
                                "schema": copy.deepcopy(self._schemas[mode])}},
            "stream": True, "reasoning": {"effort": self.reasoning_effort},
            "include": ["reasoning.encrypted_content"],
        }

    def request_key(self, payload, grid, mode) -> str:
        material = {"wire_sha256": hashlib.sha256(ordered_dumps(payload).encode("utf-8")).hexdigest(),
                    "mode": mode, "client_version": CLIENT_VERSION, "prompt_version": PROMPT_VERSION,
                    "plan_schema_version": PLAN_SCHEMA_VERSION, "history_protocol": HISTORY_PROTOCOL,
                    "transport_version": ORDERED_TRANSPORT_VERSION, "endpoint": self.config.endpoint}
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def validate_response(self, data, grid) -> dict:
        """Structure only (the schema of the request kind; SchemaError). Semantics belong to the policy."""
        kind = self._request_kind
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        try:
            if kind in POINTING_KEYS:
                return _validate_pointing(data, kind)
            if kind not in REQUEST_KINDS:
                raise SchemaError("no request kind is pending")
            conform(data, self._schemas[kind])
            return copy.deepcopy(data)
        except SchemaError:
            raise
        except PlanParseError as exc:
            raise SchemaError(f"{kind} output: {exc}") from None
        except Exception as exc:
            raise SchemaError(f"Malformed agent output ({type(exc).__name__})") from None

    def _history_input(self, payload) -> list[dict]:
        return []

    def _write_log(self, path, data: Any) -> None:
        request_record = isinstance(data, dict) and "payload" in data and "request_ordinal" in data
        if isinstance(data, dict):
            extra = {"wire_transport_version": ORDERED_TRANSPORT_VERSION, "agent_speedup_v2_client_version":
                     CLIENT_VERSION}
            if self.offline_mock:
                extra["offline_mock"] = True
            if request_record:
                data = {**data, "attempt": self._overload_attempt, "overload_retry_policy": OVERLOAD_RETRY_POLICY,
                        "request_meta": self.last_request_meta, **extra}
            elif data.get("status") in ("valid", "error"):
                data = {**data, "request_kind": self._request_kind, "output_key_order": self.last_output_key_order,
                        **extra}
                if data["status"] == "error":
                    data.update(error="Subscription attempt failed; no API-key fallback",
                                overload_retry_policy=OVERLOAD_RETRY_POLICY)
            else:
                data = {**data, **extra}
        super()._write_log(path, data)
        if request_record:
            self._kind_calls[data["mode"]] += 1

    # requests --------------------------------------------------------------------------------------------
    def predict(self, request, grid=None, mode="plan") -> dict[str, Any]:
        """One stateless request with the bounded same-request overload retry (restated from the v1 agent
        client). History is cleared after every call, accepted or failed."""
        if mode not in REQUEST_KINDS:
            raise ValueError(f"mode must be one of {', '.join(REQUEST_KINDS)}")
        with self._history_lock:
            same_request = copy.deepcopy(request)
            initial_observation_count = self._observation_count
            self._overload_attempt = 1
            try:
                while True:
                    self.last_output_key_order = None
                    try:
                        result = super().predict(same_request, grid, mode)
                    except APIError as exc:
                        if (self._overload_attempt >= OVERLOAD_MAX_ATTEMPTS
                                or getattr(exc, "error_code", None) != "server_is_overloaded"
                                or getattr(exc, "http_status", None) not in (None, OVERLOAD_HTTP_STATUS)
                                or self.last_response is not None
                                or self._observation_count != initial_observation_count):
                            raise
                        delay = OVERLOAD_BACKOFF_BASE_S ** self._overload_attempt + random.uniform(0.0,
                                                                                                    OVERLOAD_JITTER_S)
                        delay = max(delay, _failure_metadata(exc).get("retry_after_s", 0.0))
                        if delay > OVERLOAD_MAX_WAIT_S:
                            raise
                        self._reserve()
                        if (self.config.max_wall_time_s is not None
                                and self.clock() - self._started + delay >= self.config.max_wall_time_s):
                            raise
                        record = {
                            "protocol": OVERLOAD_RETRY_POLICY, "status": "bounded_retry_scheduled",
                            **_failure_metadata(exc), "attempt": self._overload_attempt,
                            "next_attempt": self._overload_attempt + 1, "max_attempts": OVERLOAD_MAX_ATTEMPTS,
                            "wait_s": delay, "same_observation": True, "request_kind": mode,
                            "observation_index": initial_observation_count,
                            "failed_request_key": self.last_request_key,
                            "failed_request_ordinal": self._api_calls,
                            "actions_executed_from_failed_response": 0,
                            "budget": self.budget_snapshot(),
                        }
                        path = (Path(self.last_log_path).with_suffix(".overload_retry.json")
                                if self.last_log_path is not None else None)
                        self._write_log(path, record)
                        started = self.clock()
                        try:
                            time.sleep(delay)
                        finally:
                            self._overload_retry_wait_s += max(0.0, self.clock() - started)
                        self._overload_attempt += 1
                        self._overload_retry_requests += 1
                    else:
                        self._kind_accepted[mode] += 1
                        return result
            finally:
                self._overload_attempt = 1
                self._history = []

    def _send(self, kind: str, request: dict, meta: dict) -> dict:
        self.last_request_meta = jsonable(meta)
        bind = getattr(self.transport, "bind_meta", None)
        if self.offline_mock and callable(bind):
            bind(copy.deepcopy(self.last_request_meta))
        return self.predict(request, mode=kind)

    def _plan_request(self, external_rgb, wrist_rgb, *, quarter_turns, fingertip_edge, holding=None,
                      status_block=None, reask_error=None, replan=False):
        kind = "replan" if replan else "plan"
        k = _turns(quarter_turns)
        if fingertip_edge != FINGERTIP_EDGE[k]:
            raise ValueError(f"fingertip_edge must be {FINGERTIP_EDGE[k]!r} for {k} quarter turns")
        holding = _pid_or_none(holding, "holding")
        block = _status_block(status_block)
        reask = _short_text(reask_error, ISSUES_TEXT_MAX_CHARS, "reask_error")
        images = plan_images(external_rgb, wrist_rgb, k)
        labelled = [(_IMAGE_LABELS[name].format(edge=fingertip_edge), url) for name, url in images]
        fields = {"holding": holding, "images": _PLAN_IMAGES_TEXT.format(edge=fingertip_edge), "status": block,
                  "previous_answer_rejected": reask}
        meta = {"kind": kind, "stage": None, "camera": None, "pid": None, "text": None, "center_uv": None,
                "origin": None, "crop_px": None, "upsample": None, "quarter_turns": k,
                "image_size": WRIST_IMAGE_PX, "previous_rejected": reask, "replan": bool(replan), "review": False,
                "holding": holding, "fingertip_edge": fingertip_edge}
        return kind, {"fields": fields, "images": labelled}, meta

    def request_plan(self, external_rgb, wrist_rgb, *, quarter_turns: int, fingertip_edge: str,
                     holding: str | None = None, status_block: dict | None = None, reask_error: str | None = None,
                     replan: bool = False) -> dict:
        """One plan (or replan) call; returns the schema-valid plan JSON (``_plan.parse_plan`` builds the Plan)."""
        kind, request, meta = self._plan_request(external_rgb, wrist_rgb, quarter_turns=quarter_turns,
                                                 fingertip_edge=fingertip_edge, holding=holding,
                                                 status_block=status_block, reask_error=reask_error, replan=replan)
        return self._send(kind, request, meta)

    def _point_request(self, rgb_upright, *, stage, camera, text, pid, center_uv=None, quarter_turns,
                       previous_rejected=None, marker_uv=None):
        stage = pointing_stage(stage)
        kind = "point_" + stage
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {', '.join(CAMERAS)}")
        k = _turns(quarter_turns)
        if camera == "external" and k:
            raise ValueError("external pointing is never turned (quarter_turns 0)")
        if not isinstance(text, str) or not text.strip() or len(text) > TEXT_MAX_CHARS:
            raise ValueError(f"text must be the plan's part text (nonempty, at most {TEXT_MAX_CHARS} characters)")
        if pid not in PIDS:
            raise ValueError("pid must be a part id")
        previous = _short_text(previous_rejected, PREVIOUS_REJECTED_MAX_CHARS, "previous_rejected")
        if marker_uv is not None:
            raise NotImplementedError("marker_uv (the LookAt predicted hit) is reserved (contract §C0.3)")
        pixels = _protocol_image(rgb_upright)
        width = pixels.shape[1]
        shown = rotate_quarter(pixels, k)
        origin = crop = up = None
        if stage == "fine":
            if center_uv is None:
                raise ValueError("fine pointing needs center_uv (upright pixels of the camera)")
            center_uv = tuple(float(v) for v in np.asarray(center_uv, dtype=float).reshape(2))
            url, origin = fine_overlay_at(shown, rotate_pixel(center_uv, k, width))
            crop, up = FINE_CROP_PX, FINE_UPSAMPLE
            what = "an enlarged crop with 64 numbered points"
        else:
            center_uv = None if center_uv is None else tuple(float(v) for v in np.asarray(center_uv,
                                                                                          dtype=float).reshape(2))
            url = coarse_overlay(shown)
            what = "the whole view with an 8x8 grid of numbered cells"
        if camera == "wrist":
            view = (f"wrist camera, turned so that its left and right match the external camera; the robot's "
                    f"fingertips are at the {FINGERTIP_EDGE[k]} edge of the full view; {what}")
        else:
            view = f"external camera, upright; {what}"
        fields = {"part": text, "camera": camera, "image": view, "previous_answer_rejected": previous}
        meta = {"kind": kind, "stage": stage, "camera": camera, "pid": pid, "text": text, "center_uv": center_uv,
                "origin": origin, "crop_px": crop, "upsample": up, "quarter_turns": k, "image_size": width,
                "previous_rejected": previous, "replan": False, "review": False}
        label = (f"{camera.capitalize()} camera " + ("crop with 64 numbered points:" if stage == "fine"
                                                     else "view with 64 numbered grid cells:"))
        return kind, {"fields": fields, "images": [(label, url)]}, meta

    def request_point(self, rgb_upright, *, stage: str, camera: str, text: str, pid: str, center_uv=None,
                      quarter_turns: int, previous_rejected: str | None = None, marker_uv=None) -> PointReply:
        """One pointing call on the upright image of ``camera`` (wrist: shown turned by quarter_turns; external:
        quarter_turns 0). Fine: a crop centred on ``center_uv`` (upright). The answer is mapped back to upright
        pixels; ``answer_id`` None means 0 (not visible). ``origin`` is the crop origin in the shown frame."""
        kind, request, meta = self._point_request(rgb_upright, stage=stage, camera=camera, text=text, pid=pid,
                                                  center_uv=center_uv, quarter_turns=quarter_turns,
                                                  previous_rejected=previous_rejected, marker_uv=marker_uv)
        result = self._send(kind, request, meta)
        answer_id = result[POINTING_KEYS[kind]]
        k, width = meta["quarter_turns"], meta["image_size"]
        uv = None
        if answer_id is not None:
            shown = (fine_point_pixel_at(meta["origin"], answer_id, crop_px=meta["crop_px"]) if meta["stage"] == "fine"
                     else coarse_cell_center(answer_id, width))
            uv = unrotate_pixel(shown, k, width)
        return PointReply(meta["stage"], meta["camera"], answer_id, uv,
                          FINE_QUANT_PX if meta["stage"] == "fine" else COARSE_QUANT_PX, meta["origin"],
                          meta["crop_px"], meta["upsample"], k)

    def _repair_request(self, external_rgb, wrist_rgb, *, quarter_turns, status_block, last_outcome, holding,
                        review=False):
        kind = "review" if review else "repair"
        k = _turns(quarter_turns)
        if status_block is None:
            raise ValueError("a repair or review request needs the plan status block")
        block = _status_block(status_block)
        outcome = _last_outcome(last_outcome)
        holding = _pid_or_none(holding, "holding")
        external = _protocol_image(external_rgb)
        wrist = rotate_quarter(_protocol_image(wrist_rgb), k)
        images = [("Image 1: external camera, clean:", encode_png_data_url(external)),
                  (f"Image 2: wrist camera, clean (turned; fingertips at the {FINGERTIP_EDGE[k]} edge):",
                   encode_png_data_url(wrist))]
        fields = {"plan": block, "last_outcome": outcome, "holding": holding}
        meta = {"kind": kind, "stage": None, "camera": None, "pid": None, "text": None, "center_uv": None,
                "origin": None, "crop_px": None, "upsample": None, "quarter_turns": k, "image_size": WRIST_IMAGE_PX,
                "previous_rejected": None, "replan": False, "review": bool(review), "holding": holding}
        return kind, {"fields": fields, "images": images}, meta

    def request_repair(self, external_rgb, wrist_rgb, *, quarter_turns: int, status_block: dict,
                       last_outcome: dict | None, holding: str | None, review: bool = False) -> dict:
        """One repair (or review) call; returns the schema-valid answer JSON (``_plan.parse_repair`` builds it).
        ``last_outcome`` carries exactly reason, evidence_text and thresholds [C-45]."""
        kind, request, meta = self._repair_request(external_rgb, wrist_rgb, quarter_turns=quarter_turns,
                                                   status_block=status_block, last_outcome=last_outcome,
                                                   holding=holding, review=review)
        return self._send(kind, request, meta)


# --------------------------------------------------------------------------------------------- offline fixtures
class ScriptedAnswers(Protocol):
    def answer(self, kind: str, meta: dict) -> dict: ...     # schema JSON for the request kind


class FixedAnswers:
    """ScriptedAnswers for tests: one queue per kind group (plan / replan, point_fine, point_coarse, repair /
    review); each answer pops the queue's head and the last item repeats. Pointing items may be ids (or None)."""

    _QUEUE_OF = {"plan": "plans", "replan": "plans", "point_fine": "points_fine", "point_coarse": "points_coarse",
                 "repair": "repairs", "review": "repairs"}

    def __init__(self, *, plans=(), points_fine=(), points_coarse=(), repairs=()):
        self.queues = {"plans": [copy.deepcopy(item) for item in plans],
                       "points_fine": [copy.deepcopy(item) for item in points_fine],
                       "points_coarse": [copy.deepcopy(item) for item in points_coarse],
                       "repairs": [copy.deepcopy(item) for item in repairs]}
        self.served: list[dict] = []

    def answer(self, kind: str, meta: dict) -> dict:
        if kind not in self._QUEUE_OF:
            raise ValueError(f"unknown request kind {kind!r}")
        queue = self.queues[self._QUEUE_OF[kind]]
        if not queue:
            raise LookupError(f"no scripted answer for {kind}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if kind in POINTING_KEYS and (item is None or isinstance(item, (int, np.integer))):
            item = {POINTING_KEYS[kind]: None if item is None else int(item)}
        self.served.append({"kind": kind, "meta": copy.deepcopy(dict(meta or {})), "answer": copy.deepcopy(item)})
        return copy.deepcopy(item)


class ScriptedV2Transport:
    """Offline transport (v1 scripted-transport pattern): payload checks mirror the ordered transport; each call
    asks ``answers.answer(kind, meta)`` with the request meta the client bound before it; billing offline_mock.
    A binding is consumed by the next call: a call made without a fresh ``bind_meta`` gets ``{"kind": kind}``."""

    offline_mock = True

    def __init__(self, answers: ScriptedAnswers, *, model=MODEL, effort=REASONING_EFFORT):
        if not callable(getattr(answers, "answer", None)):
            raise TypeError("answers must provide answer(kind, meta)")
        self.answers = answers
        self.model, self.effort = model, effort
        self.calls: list[dict] = []
        self.kinds: list[str] = []
        self.metas: list[dict] = []
        self._meta: dict | None = None

    def bind_meta(self, meta: dict) -> None:
        if not isinstance(meta, Mapping):
            raise TypeError("request meta must be a dict")
        self._meta = copy.deepcopy(dict(meta))

    def __call__(self, endpoint, payload, headers, timeout_s) -> dict:
        # one binding serves one call: an unbound later call (e.g. a bare client.predict) gets {"kind": kind}, never
        # the crop geometry of an earlier request
        bound, self._meta = self._meta, None
        if endpoint != ENDPOINT:
            raise ValueError("Only the public OpenAI Responses endpoint is supported")
        if payload.get("store") is not False or payload.get("stream") is not True:
            raise ValueError("ChatGPT plan requests require store=false and stream=true")
        if UNSUPPORTED_FIELDS.intersection(payload):
            raise ValueError("Subscription request contains unsupported parameters")
        ordered_dumps(payload)
        kind = _KIND_BY_FORMAT.get(payload.get("text", {}).get("format", {}).get("name"))
        if kind is None:
            raise ValueError("Unknown agent_speedup_v2 request format")
        meta = bound if bound is not None else {"kind": kind}
        if meta.get("kind", kind) != kind:
            raise ValueError("the bound request meta belongs to another request kind")
        self.calls.append(copy.deepcopy(payload))
        self.kinds.append(kind)
        self.metas.append(copy.deepcopy(meta))
        data = self.answers.answer(kind, meta)
        text = ordered_dumps(dict(data) if isinstance(data, Mapping) else data)   # any JSON; the client validates
        ident = len(self.calls)
        return {"id": f"offline-agent-v2-{ident}", "status": "completed", "model": self.model,
                "reasoning": {"effort": self.effort}, "offline_mock": True, "billing": "offline_mock",
                "output": [{"type": "reasoning", "id": f"rs_offline_{ident}", "summary": []},
                           {"type": "message", "id": f"msg_offline_{ident}", "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}


# ------------------------------------------------------------------------------------------------- preflight
def preflight_payloads(task_prompt: str) -> dict:
    """The plan, point_fine, point_coarse and repair payloads for a synthetic 512 px observation (uniform grey
    images; the worked example as the plan in the repair status block). No credentials, network or files: payloads
    are built, never sent."""
    client = AgentSpeedUpV2Client(task_prompt=task_prompt)
    grey = np.full((WRIST_IMAGE_PX, WRIST_IMAGE_PX, 3), PREFLIGHT_GREY, dtype=np.uint8)
    centre = (WRIST_IMAGE_PX / 2.0, WRIST_IMAGE_PX / 2.0)
    example = parse_plan(WORKED_EXAMPLE)
    if validate_plan(example):
        raise AssertionError("the worked example must validate")
    states = [StepState(0, example.steps[0], "done", "turned_half", {}, {}, 0, 1, []),
              StepState(1, example.steps[1], "failed", "hold_lost", {}, {}, 0, 1, [])]
    block = plan_status_block(example, states, holding=None)
    outcome = {"reason": "hold_lost", "evidence_text": "part A: the gap fell to the hold floor during the lift",
               "thresholds": [["G6_hold_floor_mm", 0.5]]}
    requests = {
        "plan": client._plan_request(grey, grey, quarter_turns=0, fingertip_edge=FINGERTIP_EDGE[0]),
        "point_fine": client._point_request(grey, stage="fine", camera="wrist", text=example.parts[0].text,
                                            pid=example.parts[0].pid, center_uv=centre, quarter_turns=0),
        "point_coarse": client._point_request(grey, stage="coarse", camera="wrist", text=example.parts[4].text,
                                              pid=example.parts[4].pid, quarter_turns=0),
        "repair": client._repair_request(grey, grey, quarter_turns=0, status_block=block, last_outcome=outcome,
                                         holding=None)}
    return {name: client.build_payload(request, mode=kind) for name, (kind, request, _meta) in requests.items()}
