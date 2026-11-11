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

Targeted re-ask (prompt 2). A plan request with ``reask_error`` that immediately follows a schema-valid plan answer of
the same kind carries that answer as ``previous_plan`` (sanitized, as parsed), and the instructions ask the model to
change only what the issues name. The client keeps exactly that one answer, for exactly the next plan request: any
other request clears it, and ``reset_history`` does too. The re-ask answer must keep every step no issue names
(``_plan.reask_retention_issues``); an answer that drops one is rejected as a SchemaError (logged as
``reask_check``). Without a readable previous answer (a schema error) the re-ask asks for a new plan.

Pointing purpose and marker (prompt 3; grip-part lens BC-1 / BC-2, blocker 2). A pointing request names its
purpose (``POINT_PURPOSES``: the PointNeed's), and both pointing instructions end with the measured grasp-point clause
(``GRASP_POINT_CLAUSE``). ``marker_uv`` (upright pixels of the camera, like ``center_uv``) rings the point an
earlier answer gave and the robot rejected, on the image the model sees (the upsampled fine crop, or the coarse view),
after the frame-alignment turn; the label and the image description say so only when the ring is drawn (a marker
outside the fine crop is not drawn). Repair and review requests carry ``previous_answer_rejected`` (the StepRunner's
re-ask text after an answer it refused, else null) and may ring the rejected pointed point on the wrist image. The
plan instructions ask for part texts by shape and relative position, colour only secondary (BC-3, ``_NAMING``).

Recovery information (prompt 4; gap round RL-C1, ruling L4). Repair and replan requests carry ``step_history``
(right after last_outcome in a repair, after status in a replan; null when there is none): the failed step's
earlier attempts with the grips tried, the answer that followed and its effect, this attempt's grips, the part the
held object is held by and the step's parts that could not be registered; codes, part ids and counts only, at most
REPAIR_HISTORY_H earlier rows (``_step_history``). One paragraph of the repair and replan instructions describes it
(``_HISTORY``). The answer schemas are unchanged (AC-5).

Retries. The bounded same-request retry covers the v1 overload case and transient stream / transport failures
(TRANSIENT_ERROR_CODES, at most TRANSIENT_MAX_RETRIES): only when no response envelope arrived, so a schema-valid
answer is never retried. Every scheduled retry is logged; token material never reaches a log.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image, ImageDraw

from .agent_speedup_v2_plan import (ISSUES_TEXT_MAX_CHARS, PLAN_MAX_GRIP, PLAN_MAX_PARTS, PLAN_SCHEMA_VERSION,
                                    REPAIR_HISTORY_H, SCENE_MAX_CHARS, STATUS_BLOCK_KEYS, TEXT_MAX_CHARS,
                                    TEXT_MIN_CHARS, TXT_MAX_CHARS, WHY_MAX_CHARS, PlanParseError,
                                    conform, issues_text, parse_plan, plan_status_block, pointing_stage,
                                    reask_retention_issues, schemas, unsafe_characters, validate_plan)
from .agent_speedup_v2_types import (CAMERAS, PIDS, POINT_PURPOSES, REASONS, REPAIR_ACTIONS, REPAIR_EFFECTS,
                                     WRIST_IMAGE_PX, PointReply, StepState, jsonable, th_value)
from .astra import ENDPOINT, APIError, SchemaError, _canonical
from .chatgpt_plan import UNSUPPORTED_FIELDS, _failure_metadata
from .direct_history import _allowed
from .hybrid_subscription import ChatGPTPlanHybridClient
from .mark_prompting import (COARSE_GRID, FINE_GRID, FINGERTIP_EDGE, _font, _grid_lines, _label, coarse_cell_center,
                             coarse_overlay, encode_png_data_url, rotate_pixel, rotate_quarter, unrotate_pixel)
from .ordered_transport import ORDERED_TRANSPORT_VERSION, ordered_dumps, ordered_sse_transport
from .speed_first_hybrid import OVERLOAD_RETRY_POLICY

CLIENT_VERSION = "agent-speedup-v2-client-4"
PROMPT_VERSION = "agent-speedup-v2-prompt-4"
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
    "TRANSIENT_MAX_RETRIES": ("DD", "planck_001: 1 of 61 live calls ended without response.completed; at most 2 "
                                    "same-request retries of a transient stream / transport failure (3 attempts, "
                                    "as the overload policy)"),
    "TRANSIENT_BACKOFF_BASE_S": ("DD", "transient backoff base**retry seconds (2 s, 4 s), the overload backoff "
                                       "restated; plus the overload jitter, at most OVERLOAD_MAX_WAIT_S"),
    "VISIBLE_TOKEN_BUDGETS": ("DD", "spec §3.2 visible-token budgets [P], upper ends: plan and replan 340, pointing 6, "
                                    "repair and review 80 without a new step; diagnosis only (history_stats counts "
                                    "answers above them), never a cap: the subscription route sends no output limit"),
    "REPAIR_STEP_TOKEN_BUDGET": ("DD", "spec §3.2 visible-token budget [P] of a repair or review answer that carries a "
                                       "new step (insert_step): at most 300; diagnosis only, never a cap"),
    "FINE_STYLE": ("DD", "fine overlay drawing restated from v1 mark_prompting.fine_overlay (:240-252): faint grid "
                         "lines, point fill, font size and point radius per point spacing"),
    "PREFLIGHT_GREY": ("DD", "synthetic preflight observation: uniform mid-grey images (no scene content)"),
    "MARKER_STYLE": ("DD", "grip-part lens BC-2 (blocker 2, live pointing check M2 on the live 003 view): a ring of "
                           "radius 22 px and width 4 px in RGB (255, 0, 0) at the shown 512 px scale (64 px point "
                           "spacing) moved the answer onto the part 1/1 with the grasp-point clause"),
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
TRANSIENT_MAX_RETRIES = 2
TRANSIENT_BACKOFF_BASE_S = 2.0
TRANSIENT_RETRY_POLICY = "agent-speedup-v2-transient-retry-1"
# chatgpt_plan failure codes of a stream or connection that broke before a completed response arrived (no output to
# discard): the request may be sent again unchanged.
TRANSIENT_ERROR_CODES = ("stream_missing_completion", "stream_truncated_event", "subscription_stream_interrupted",
                         "subscription_network_error")
VISIBLE_TOKEN_BUDGETS = {"plan": 340, "replan": 340, "point_fine": 6, "point_coarse": 6, "repair": 80,
                         "review": 80}
REPAIR_STEP_TOKEN_BUDGET = 300
FINE_STYLE = {"grid_core_rgba": (255, 255, 255, 70), "point_rgba": (255, 225, 20, 255), "outline_rgba": (0, 0, 0, 255),
              "font_scale": (13, 60), "min_font_px": 8, "radius_divisor": 15, "min_radius_px": 2,
              "label_gap_px": 2}
PREFLIGHT_GREY = 128
MARKER_STYLE = {"rgb": (255, 0, 0), "radius_px": 22, "width_px": 4}

_KIND_BY_FORMAT = {name: kind for kind, name in FORMAT_NAMES.items()}
_PNG_PREFIX = "data:image/png;base64,"
_PLAN_KINDS = ("plan", "replan")
_REPAIR_KINDS = ("repair", "review")


# --------------------------------------------------------------------------------------------------- example
# Neutral objects (spec §3.5 item 5; Appendix L 14 BC-R3: never the task-2 nominal plan). Prompt 2 shows the rules
# the planck_001 plans broke: a place step with an empty grip list (the whole object fits) and no hold before it, an
# effect criterion on a part that is neither obj nor grip, short texts, and every listed object in use. The place
# comes first and the turn ends by the effect alone, so the example is not task 2's turn-then-place_on shape nor its
# effect-or-stop turn. The example teaches no rotation criterion: the model copies the example's criteria into its
# own turns (planck_004: quarter_turn appeared in the task-2 turn), and the relation layer refuses a turn whose
# quarter_turn / half_turn exceeds the yaw budget left, even when ORed with an effect (contract §C10.3 step 0 and
# the free-sense check; half_turn, pi, always exceeds it), so a copied rotation criterion can fail a turn whose
# effect alone would have ended it.
WORKED_EXAMPLE = {
    "scene": "toy car by an open shoe box; music box with a side crank; glass vase",
    "parts": [
        {"id": "A", "text": "toy car: whole car", "move": "free", "w": 36, "x": 28},
        {"id": "B", "text": "shoe box: open cavity", "move": "free", "w": 44, "x": 37},
        {"id": "C", "text": "music box: small crank on the side", "move": "hinged", "w": 52, "x": 45},
        {"id": "D", "text": "music box: dancer figure on top", "move": "hinged", "w": 51, "x": 44},
        {"id": "E", "text": "glass vase: whole vase", "move": "free", "w": 21, "x": 19}],
    "hold_true": [],
    "steps": [
        {"do": "place_in", "obj": "A", "grip": [], "ref": "B", "sense": "any", "keep": [{"c": "clear_of", "p": "E"}],
         "until": []},
        {"do": "turn", "obj": "C", "grip": [], "ref": None, "sense": "any", "keep": [{"c": "clear_of", "p": "E"}],
         "until": [{"o": "effect", "p": "D", "txt": "the dancer spins"}]}]}


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
    "WRITE ONE JSON PLAN. Write every text in plain ASCII English and keep it within its limit; a longer text is cut "
    "off.\n"
    f"- scene: one short line on what you see, at most {SCENE_MAX_CHARS} characters.\n"
    "- parts: exactly the parts the steps, keeps, criteria and hold_true use, with ids A-H; every object you list is "
    "used by a step as obj, grip, ref, keep or criterion part, or by hold_true. text is \"<object>: <visible part>\", "
    f"{TEXT_MIN_CHARS} to {TEXT_MAX_CHARS} characters, with the same <object> words for every part of one object and "
    "distinguishing appearance or position when similar objects exist. move is your prior for how the part moves: "
    "free (a loose object, and every part of one, since it moves with the object), fixed, hinged (rotates about its "
    "mount), sliding (moves along a track) or unknown; the robot verifies it by probing. w is the part's cell in "
    "image 4 and x its cell in image 2, 0 when it is not visible in that image.\n"
    "- steps, in order, numbered from 0; each establishes one physical relation: hold (obj is lifted and moves with "
    "the hand; use it only when holding obj is itself wanted); turn (a rotary part obj, held at grip, is rotated "
    "about its own axis); slide (a sliding part obj, held at grip, moves along its own track; sense open = away from "
    "the body it slides in, close = into it); place_on (obj rests on the top surface of ref, released and at rest); "
    "place_in (obj rests inside the cavity of ref, below its rim); push_to (obj slides on its support to the spot "
    "ref without lifting); release (the hand opens and leaves obj). grip lists up to two parts of obj for the "
    "fingers, in preference order (an empty list means obj itself); ref is the reference part or null; sense is any, "
    "open or close. The robot adds the grasps, carries and releases a step needs: place_on, place_in and push_to "
    "take obj themselves, so a hold step before them is unnecessary.\n"
    "- keep: constraints held during a step: upright (part p keeps its in-hand pose) and clear_of (the hand and the "
    "held object stay clear of part p).\n"
    "- until: completion criteria, the first one met ends the step: effect (part p visibly changes as txt says, txt "
    f"at most {TXT_MAX_CHARS} characters; p is the part whose look changes, never the step's obj or a grip part), "
    "stop (the mechanism reaches its end stop), fits (the opening is long enough for the free part p), quarter_turn "
    "and half_turn (of the turned part). turn takes effect, stop, quarter_turn or half_turn and needs one; slide "
    "takes stop, fits or effect; place_on and place_in take only effect and usually need none, since the robot "
    "checks the placement itself; hold, push_to and release take none. List each criterion once. Unused p and txt "
    "are null.\n"
    "- hold_true: relations the images clearly show as already true and that must stay true (on, in, open, closed, "
    "effect; a is the part, b the other part or null).")
_NAMING = (
    "NAMING. In a part text, name the visible part by its shape and by its position relative to its own object or a "
    "neighbouring object, for example \"<object>: <shape> on its side toward <neighbour object>\"; use colour only "
    "as a secondary cue, since the wrist and external views can show a part in different colours.")
_RULES = (
    "RULES. Name only things visible in at least one image. GRIP: leave grip empty when obj fits between the "
    "fingers, so obj itself is gripped. Name a sub-part only when it is the feature made to be held or turned (a "
    "protruding bar or loop, a small control) or the rim of an open container too wide for the fingers; never invent "
    "a thin edge, rim or face of an object that fits whole. Give a second grip only when a second such feature "
    "exists. Include enabling steps (make an opening before placing into it). Describe completion only with the "
    "criteria above. List a relation as already true only when the images clearly show it. Never give directions, "
    "distances, heights, angles or coordinates. Write scene first.")
_CORRECTION = (
    "CORRECTION. previous_answer_rejected, when not null, says why the robot rejected your last answer. When "
    "previous_plan is not null, it is that rejected plan and previous_answer_rejected lists its issues as \"code: "
    "text\" (steps by index from 0, parts by id): change only what those issues name, copy every other part and "
    "step unchanged with the same ids, add a part or step only when an issue asks for one, and return the whole "
    "plan. When previous_plan is null, your last answer could not be read: write the plan anew.")
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
# Grip-part lens BC-1 (blocker 2), verbatim: appended to both pointing instructions, with the request's purpose field
# (framed hovers: the graspable bar 4/5 with it against 2/6 without; on the part 5/5).
GRASP_POINT_CLAUSE = (
    "For a grasp_point purpose choose the point on the named part where two fingers closing from above would hold it. "
    "Seen from above, a part can look different than in the external camera (its top may have another colour). "
    "Anything under the fingertip band at the named edge is hidden: if the named part lies only under the fingertips "
    "or outside this crop, answer 0.")
_ACTIONS = (
    "ACTIONS (choose one): retry (rerun the failed step from its last satisfied precondition); repoint (drop the "
    "registration of part and point it again); use_part (grasp part instead; it must be an existing part); "
    "insert_step (put step before the failed step, for example to clear an obstacle); skip_step (the failed step "
    "becomes a kept relation if it currently holds, otherwise it is dropped); swap_next (exchange the failed step "
    "with the next one); replan (you will be asked for a new plan); hold_still (hold position). Set part only for "
    "repoint and use_part and step only for insert_step (the same step object as in a plan, its until txt at most "
    f"{TXT_MAX_CHARS} characters); otherwise null. Write why first: one short sentence of at most {WHY_MAX_CHARS} "
    "characters. Write every text in plain ASCII English within its limit; a longer text is cut off. Never give "
    "directions, distances, heights, angles or coordinates.")
_REPAIR_INPUTS = (
    "plan gives your relation plan as the robot executes it: the parts as [id, text], every step with its status "
    "(pending, active, done, failed, stays, skipped) and reason code, the already-true relations and the held part. "
    "last_outcome gives the measured reason code, a short evidence text (magnitudes in cm and degrees, parts by id) "
    "and the thresholds it was compared against, or null. Two clean images follow: the external camera and the "
    "wrist camera, turned as named; when the wrist image's label says ringed, the ring marks the pointed point the "
    "robot rejected. previous_answer_rejected, when not null, says why the robot refused your last answer to this "
    "request; choose an action it allows.")
_REPAIR = "A step of your relation plan failed after the robot's own local retries. "
# RL-C1 / RL-C2 / RL-C3 (gap round, ruling L4): the step history paragraph of the repair and replan instructions
_HISTORY = (
    "step_history, when not null, describes the failed step since it last succeeded (kept across a replan of the "
    "same step): attempts counts its runs; earlier lists its earlier failed attempts, oldest first, each with its "
    "reason, the grip parts tried with the reason each failed, the repair action that followed and its effect "
    "(no_change: nothing changed, the step only ran again; grip_order_changed; reregistered; replanned; regrasp: the "
    "held object was set down on its support and grasped by the other part; skipped); this_attempt lists the grip "
    "parts tried in the failed attempt with their reasons; held_by names the part the hand holds the held object by; "
    "unregistered lists parts of the step the robot could not register, with the reason. An answer that changes "
    "nothing (retry, or use_part of the part already tried first or already held) is refused after it was given "
    "twice in a row for the same step; after repeated failures the robot replans or skips the step itself.")
_REVIEW = ("Every step of your relation plan has been carried out and the robot's observers report each planned "
           "relation holding. If the planned relations hold in the images, answer hold_still. ")


def _instruction_texts() -> dict[str, str]:
    example = "EXAMPLE (other objects; cells illustrative): " + ordered_dumps(WORKED_EXAMPLE)
    plan = "\n".join((_ROBOT, _IMAGES, _VOCABULARY, _NAMING, _RULES, example, _CORRECTION))
    point = f"{PROMPT_VERSION}, " + "{kind} request. " + _POINT_COMMON + " "
    repair_head = (f"{PROMPT_VERSION}, " + "{kind} request. A robot executes your relation plan and measures "
                   "everything itself. ")
    return {"plan": f"{PROMPT_VERSION}, plan request. " + plan,
            "replan": f"{PROMPT_VERSION}, replan request. " + plan + "\n" + _REPLAN + "\n" + _HISTORY,
            "point_fine": point.format(kind="point_fine") + _POINT_FINE + " " + GRASP_POINT_CLAUSE,
            "point_coarse": point.format(kind="point_coarse") + _POINT_COARSE + " " + GRASP_POINT_CLAUSE,
            "repair": (repair_head.format(kind="repair") + _REPAIR + _REPAIR_INPUTS + "\n" + _HISTORY + "\n"
                       + _ACTIONS),
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


def _marker_uv(value, name: str = "marker_uv"):
    """None, or two finite pixel coordinates (u right, v down) as floats."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be two finite numbers or None")
    try:
        uv = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be two finite numbers or None") from None
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError(f"{name} must be two finite numbers or None")
    return (float(uv[0]), float(uv[1]))


def _ring(frame: Image.Image, xy) -> bool:
    """MARKER_STYLE ring centred on ``xy`` (pixels of ``frame``) when that pixel lies inside the frame; returns
    whether it was drawn."""
    if xy is None:
        return False
    x, y = float(xy[0]), float(xy[1])
    if not (0.0 <= x < frame.size[0] and 0.0 <= y < frame.size[1]):
        return False
    radius = MARKER_STYLE["radius_px"]
    ImageDraw.Draw(frame).ellipse((x - radius, y - radius, x + radius, y + radius), outline=MARKER_STYLE["rgb"],
                                  width=MARKER_STYLE["width_px"])
    return True


def _ringed_url(url: str, xy) -> tuple[str, bool]:
    """The PNG data URL ``url`` with the marker ring at ``xy`` (its pixels); unchanged when the ring is not drawn."""
    if xy is None:
        return url, False
    frame = Image.open(io.BytesIO(base64.b64decode(url[len(_PNG_PREFIX):]))).convert("RGB")
    if not _ring(frame, xy):
        return url, False
    return _encode(frame), True


def _fine_frame(rgb_upright, center_uv, *, crop_px, upsample) -> tuple[Image.Image, tuple[int, int], int]:
    """(overlay frame, crop origin, upsample) of ``fine_overlay_at`` before encoding."""
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
    return frame, origin, int(upsample)


def fine_overlay_at(rgb_upright, center_uv, *, crop_px=FINE_CROP_PX, upsample=FINE_UPSAMPLE,
                    marker_uv=None) -> tuple[str, tuple[int, int]]:
    """Crop crop_px x crop_px centred on ``center_uv`` (clamped inside the image), upsample, draw 8x8 numbered points
    (v1's fine-overlay drawing). Returns (PNG data URL, crop origin in the given image's pixels). Point id k sits at
    ``fine_point_pixel_at(origin, k, crop_px=crop_px)`` of the given image. ``marker_uv`` (a pixel of the given image)
    is ringed when it lies inside the crop (``fine_marked_overlay_at`` also says whether it was)."""
    url, origin, _drawn = fine_marked_overlay_at(rgb_upright, center_uv, crop_px=crop_px, upsample=upsample,
                                                 marker_uv=marker_uv)
    return url, origin


def fine_marked_overlay_at(rgb_upright, center_uv, *, crop_px=FINE_CROP_PX, upsample=FINE_UPSAMPLE,
                           marker_uv=None) -> tuple[str, tuple[int, int], bool]:
    """``fine_overlay_at`` plus whether the marker ring was drawn (BC-2): the marker pixel maps into the upsampled
    crop as (marker - origin) x upsample, the same mapping ``fine_point_pixel_at`` inverts."""
    frame, origin, up = _fine_frame(rgb_upright, center_uv, crop_px=crop_px, upsample=upsample)
    marker = _marker_uv(marker_uv)
    drawn = False
    if marker is not None:
        drawn = _ring(frame, ((marker[0] - origin[0]) * up, (marker[1] - origin[1]) * up))
    return _encode(frame), origin, drawn


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


def _safe_strings(value, name: str) -> None:
    """ValueError when any string in ``value`` (keys included) holds a control, zero-width / direction mark,
    surrogate half, private-use or unassigned code point: such text never reaches a request."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _safe_strings(key, name)
            _safe_strings(item, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _safe_strings(item, name)
    elif isinstance(value, str):
        bad = unsafe_characters(value)
        if bad:
            raise ValueError(f"{name} holds control or zero-width characters ({', '.join(sorted(set(bad))[:3])})")


def _json_text(value, name: str) -> str:
    _safe_strings(value, name)
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


STEP_HISTORY_KEYS = ("step", "attempts", "earlier", "this_attempt", "held_by", "unregistered")
STEP_HISTORY_ROW_KEYS = ("attempt", "reason", "grips", "then", "effect")


def _count(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _part_reasons(value, name: str, limit: int) -> list:
    """[[pid, REASONS code], ...] with at most ``limit`` pairs."""
    if not isinstance(value, (list, tuple)) or len(value) > limit:
        raise ValueError(f"{name} must be a list of at most {limit} [part id, reason code] pairs")
    pairs = []
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] not in PIDS
                or pair[1] not in REASONS):
            raise ValueError(f"{name} must hold [part id, reason code] pairs")
        pairs.append([pair[0], pair[1]])
    return pairs


def _step_history(value):
    """RL-C1: the step history of a repair or replan request, or None. Exactly the keys STEP_HISTORY_KEYS; codes,
    part ids and counts only: step (plan index or null), attempts (runs of the step), earlier (at most
    REPAIR_HISTORY_H rows {attempt, reason, grips, then: [action, part], effect}), this_attempt and unregistered
    ([part id, reason code] pairs), held_by (a part id or null)."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(STEP_HISTORY_KEYS):
        raise ValueError(f"step_history must have exactly the keys {', '.join(STEP_HISTORY_KEYS)}")
    step = None if value["step"] is None else _count(value["step"], "step_history.step", minimum=0)
    earlier = value["earlier"]
    if not isinstance(earlier, (list, tuple)) or len(earlier) > REPAIR_HISTORY_H:
        raise ValueError(f"step_history.earlier must be a list of at most {REPAIR_HISTORY_H} rows")
    rows = []
    for row in earlier:
        if not isinstance(row, Mapping) or set(row) != set(STEP_HISTORY_ROW_KEYS):
            raise ValueError(f"step_history rows must have exactly the keys {', '.join(STEP_HISTORY_ROW_KEYS)}")
        then = row["then"]
        if (not isinstance(then, (list, tuple)) or len(then) != 2 or then[0] not in REPAIR_ACTIONS
                or (then[1] is not None and then[1] not in PIDS)):
            raise ValueError("step_history row then must be [repair action, part id or null]")
        if row["reason"] not in REASONS or row["effect"] not in REPAIR_EFFECTS:
            raise ValueError("step_history row reason / effect must be REASONS / REPAIR_EFFECTS codes")
        rows.append({"attempt": _count(row["attempt"], "step_history row attempt", minimum=1),
                     "reason": row["reason"], "grips": _part_reasons(row["grips"], "step_history row grips",
                                                                     PLAN_MAX_GRIP),
                     "then": [then[0], then[1]], "effect": row["effect"]})
    held_by = _pid_or_none(value["held_by"], "step_history.held_by")
    return {"step": step, "attempts": _count(value["attempts"], "step_history.attempts", minimum=0), "earlier": rows,
            "this_attempt": _part_reasons(value["this_attempt"], "step_history.this_attempt", PLAN_MAX_GRIP),
            "held_by": held_by,
            "unregistered": _part_reasons(value["unregistered"], "step_history.unregistered", PLAN_MAX_PARTS)}


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
        self.last_reask_check = None
        self._reask = None                  # the rejected plan and its issues while a re-ask is in flight
        self._previous_plan = None          # {"kind", "plan"}: the last plan answer, for the next plan request only
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
            self._transient_retry_requests = 0
            self._transient_retry_wait_s = 0.0
            self._transient_retry_codes = {}
            self._reask_requests = 0
            self._reask_rejected = 0
            self._previous_plan = None
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
                    "overload_retry_wait_s": self._overload_retry_wait_s,
                    "transient_retry_policy": TRANSIENT_RETRY_POLICY,
                    "transient_retry_requests": self._transient_retry_requests,
                    "transient_retry_wait_s": self._transient_retry_wait_s,
                    "transient_retry_codes": dict(self._transient_retry_codes),
                    "reask_requests": self._reask_requests,
                    "reask_rejected_for_dropped_steps": self._reask_rejected}

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
        """Structure (the schema of the request kind; SchemaError), plus the re-ask protocol: a plan answer to a
        targeted re-ask must keep every step of the rejected plan that no issue names (``reask_retention_issues``).
        Every other semantic check belongs to the policy."""
        kind = self._request_kind
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        try:
            if kind in POINTING_KEYS:
                return _validate_pointing(data, kind)
            if kind not in REQUEST_KINDS:
                raise SchemaError("no request kind is pending")
            conform(data, self._schemas[kind])
            if kind in _PLAN_KINDS and self._reask is not None:
                self._check_retention(kind, data)
            return copy.deepcopy(data)
        except SchemaError:
            raise
        except PlanParseError as exc:
            raise SchemaError(f"{kind} output: {exc}") from None
        except Exception as exc:
            raise SchemaError(f"Malformed agent output ({type(exc).__name__})") from None

    def _check_retention(self, kind: str, data) -> None:
        """SchemaError when the re-ask answer dropped a step no issue named; the finding goes to ``last_reask_check``
        (and the call log). An answer the plan parse refuses is left to the policy's parse (its text is the error)."""
        previous, issues = self._reask["plan"], self._reask["issues"]
        check = {"kind": kind, "previous_steps": len(previous.steps),
                 "named_steps": sorted({i.step for i in issues if i.step is not None}),
                 "named_parts": sorted({i.pid for i in issues if i.step is None and i.pid is not None}),
                 "dropped": [], "checked": False}
        self.last_reask_check = check
        try:
            corrected = parse_plan(data, source=kind)
        except PlanParseError:
            return
        dropped = reask_retention_issues(previous, corrected, issues)
        # diagnosis only: unnamed steps not copied verbatim (reworded keeps / criteria, renamed ids, or dropped)
        verbatim = [i for i, step in enumerate(previous.steps)
                    if i not in check["named_steps"] and step not in corrected.steps]
        check.update(checked=True, corrected_steps=len(corrected.steps), dropped=[i.to_dict() for i in dropped],
                     unnamed_steps_not_verbatim=verbatim)
        if dropped:
            self._reask_rejected += 1
            raise SchemaError(f"{kind} output: {issues_text(dropped)}")

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
                        "transient_retry_policy": TRANSIENT_RETRY_POLICY, "request_meta": self.last_request_meta,
                        **extra}
            elif data.get("status") in ("valid", "error"):
                data = {**data, "request_kind": self._request_kind, "output_key_order": self.last_output_key_order,
                        **extra}
                if self._reask is not None:
                    data["reask_check"] = copy.deepcopy(self.last_reask_check)
                if data["status"] == "error":
                    data.update(error="Subscription attempt failed; no API-key fallback",
                                overload_retry_policy=OVERLOAD_RETRY_POLICY,
                                transient_retry_policy=TRANSIENT_RETRY_POLICY)
            else:
                data = {**data, **extra}
        super()._write_log(path, data)
        if request_record:
            self._kind_calls[data["mode"]] += 1

    # requests --------------------------------------------------------------------------------------------
    def _retry_cause(self, exc, initial_observation_count: int, retries: Mapping[str, int]) -> str | None:
        """"overload" (v1 policy: server_is_overloaded, HTTP 503 or none), "transient" (TRANSIENT_ERROR_CODES) or None
        (raise). Only a request that produced no response envelope and committed nothing is retried, so a
        schema-valid answer is never sent again."""
        if self.last_response is not None or self._observation_count != initial_observation_count:
            return None
        code = getattr(exc, "error_code", None)
        if code == "server_is_overloaded" and getattr(exc, "http_status", None) in (None, OVERLOAD_HTTP_STATUS):
            return "overload" if retries["overload"] + 1 < OVERLOAD_MAX_ATTEMPTS else None
        if code in TRANSIENT_ERROR_CODES:
            return "transient" if retries["transient"] < TRANSIENT_MAX_RETRIES else None
        return None

    def predict(self, request, grid=None, mode="plan") -> dict[str, Any]:
        """One stateless request with the bounded same-request retry: the v1 overload policy (restated from the v1
        agent client) and, for a stream or connection that broke before any response arrived, at most
        TRANSIENT_MAX_RETRIES transient retries. History is cleared after every call, accepted or failed."""
        if mode not in REQUEST_KINDS:
            raise ValueError(f"mode must be one of {', '.join(REQUEST_KINDS)}")
        with self._history_lock:
            same_request = copy.deepcopy(request)
            initial_observation_count = self._observation_count
            self._overload_attempt = 1
            retries = {"overload": 0, "transient": 0}
            try:
                while True:
                    self.last_output_key_order = None
                    self.last_reask_check = None
                    try:
                        result = super().predict(same_request, grid, mode)
                    except APIError as exc:
                        cause = self._retry_cause(exc, initial_observation_count, retries)
                        if cause is None:
                            raise
                        base = OVERLOAD_BACKOFF_BASE_S if cause == "overload" else TRANSIENT_BACKOFF_BASE_S
                        delay = base ** (retries[cause] + 1) + random.uniform(0.0, OVERLOAD_JITTER_S)
                        delay = max(delay, _failure_metadata(exc).get("retry_after_s", 0.0))
                        if delay > OVERLOAD_MAX_WAIT_S:
                            raise
                        self._reserve()
                        if (self.config.max_wall_time_s is not None
                                and self.clock() - self._started + delay >= self.config.max_wall_time_s):
                            raise
                        policy = OVERLOAD_RETRY_POLICY if cause == "overload" else TRANSIENT_RETRY_POLICY
                        limit = (OVERLOAD_MAX_ATTEMPTS if cause == "overload"
                                 else TRANSIENT_MAX_RETRIES + 1)
                        record = {
                            "protocol": policy, "status": "bounded_retry_scheduled", "cause": cause,
                            **_failure_metadata(exc), "attempt": self._overload_attempt,
                            "next_attempt": self._overload_attempt + 1, "max_attempts": limit,
                            "wait_s": delay, "same_observation": True, "request_kind": mode,
                            "observation_index": initial_observation_count,
                            "failed_request_key": self.last_request_key,
                            "failed_request_ordinal": self._api_calls,
                            "actions_executed_from_failed_response": 0,
                            "budget": self.budget_snapshot(),
                        }
                        path = (Path(self.last_log_path).with_suffix(f".{cause}_retry.json")
                                if self.last_log_path is not None else None)
                        self._write_log(path, record)
                        started = self.clock()
                        try:
                            time.sleep(delay)
                        finally:
                            waited = max(0.0, self.clock() - started)
                            if cause == "overload":
                                self._overload_retry_wait_s += waited
                            else:
                                self._transient_retry_wait_s += waited
                        self._overload_attempt += 1
                        retries[cause] += 1
                        if cause == "overload":
                            self._overload_retry_requests += 1
                        else:
                            self._transient_retry_requests += 1
                            code = getattr(exc, "error_code", None)
                            self._transient_retry_codes[code] = self._transient_retry_codes.get(code, 0) + 1
                    else:
                        self._kind_accepted[mode] += 1
                        return result
            finally:
                self._overload_attempt = 1
                self._history = []

    def _send(self, kind: str, request: dict, meta: dict) -> dict:
        if kind not in _PLAN_KINDS:
            self._previous_plan = None         # a re-ask's previous plan is the answer right before it, nothing older
        self.last_request_meta = jsonable(meta)
        bind = getattr(self.transport, "bind_meta", None)
        if self.offline_mock and callable(bind):
            bind(copy.deepcopy(self.last_request_meta))
        return self.predict(request, mode=kind)

    def _plan_request(self, external_rgb, wrist_rgb, *, quarter_turns, fingertip_edge, holding=None,
                      status_block=None, reask_error=None, replan=False, previous_plan=None, step_history=None):
        """(kind, request, meta) of a plan or replan request; ``previous_plan`` (a Plan) is the rejected plan a
        targeted re-ask corrects, sent as its wire JSON (texts as parsed, so sanitized). A replan carries
        ``step_history`` right after status (RL-C1; null when none); a first plan has no such field."""
        kind = "replan" if replan else "plan"
        k = _turns(quarter_turns)
        if fingertip_edge != FINGERTIP_EDGE[k]:
            raise ValueError(f"fingertip_edge must be {FINGERTIP_EDGE[k]!r} for {k} quarter turns")
        holding = _pid_or_none(holding, "holding")
        block = _status_block(status_block)
        reask = _short_text(reask_error, ISSUES_TEXT_MAX_CHARS, "reask_error")
        history = _step_history(step_history)
        if history is not None and not replan:
            raise ValueError("step_history is sent only with a replan request")
        if previous_plan is not None and reask is None:
            raise ValueError("previous_plan is sent only with a re-ask (reask_error)")
        previous = None if previous_plan is None else previous_plan.to_json()
        images = plan_images(external_rgb, wrist_rgb, k)
        labelled = [(_IMAGE_LABELS[name].format(edge=fingertip_edge), url) for name, url in images]
        fields = {"holding": holding, "images": _PLAN_IMAGES_TEXT.format(edge=fingertip_edge), "status": block}
        if replan:
            fields["step_history"] = history
        fields.update(previous_plan=previous, previous_answer_rejected=reask)
        meta = {"kind": kind, "stage": None, "camera": None, "pid": None, "text": None, "center_uv": None,
                "origin": None, "crop_px": None, "upsample": None, "quarter_turns": k,
                "image_size": WRIST_IMAGE_PX, "previous_rejected": reask, "previous_plan": previous is not None,
                "replan": bool(replan), "review": False, "holding": holding, "fingertip_edge": fingertip_edge,
                "purpose": None, "marker_uv": None, "marker_drawn": False}
        _safe_strings(fields, f"{kind} request")
        return kind, {"fields": fields, "images": labelled}, meta

    def request_plan(self, external_rgb, wrist_rgb, *, quarter_turns: int, fingertip_edge: str,
                     holding: str | None = None, status_block: dict | None = None, reask_error: str | None = None,
                     replan: bool = False, step_history: dict | None = None) -> dict:
        """One plan (or replan) call; returns the schema-valid plan JSON (``_plan.parse_plan`` builds the Plan).

        With ``reask_error`` right after a plan answer of the same kind that parsed, the request is a targeted
        correction: it carries that plan as ``previous_plan`` (the issues are its ``validate_plan`` findings, which
        the policy's ``reask_error`` text reports), and the answer must keep every step no issue names. A replan
        carries ``step_history`` (RL-C1)."""
        kind = "replan" if replan else "plan"
        remembered, self._previous_plan = self._previous_plan, None
        previous = None
        if reask_error is not None and remembered is not None and remembered["kind"] == kind:
            previous = remembered["plan"]
        kind, request, meta = self._plan_request(external_rgb, wrist_rgb, quarter_turns=quarter_turns,
                                                 fingertip_edge=fingertip_edge, holding=holding,
                                                 status_block=status_block, reask_error=reask_error, replan=replan,
                                                 previous_plan=previous, step_history=step_history)
        if previous is not None:
            self._reask = {"plan": previous, "issues": validate_plan(previous)}
            self._reask_requests += 1
        try:
            data = self._send(kind, request, meta)
        finally:
            self._reask = None
        try:
            self._previous_plan = {"kind": kind, "plan": parse_plan(data, source=kind)}
        except PlanParseError:
            self._previous_plan = None
        return data

    def _point_request(self, rgb_upright, *, stage, camera, text, pid, center_uv=None, quarter_turns,
                       previous_rejected=None, marker_uv=None, purpose=None):
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
        if purpose is not None and purpose not in POINT_PURPOSES:
            raise ValueError(f"purpose must be one of {', '.join(POINT_PURPOSES)} or None")
        marker = _marker_uv(marker_uv)
        pixels = _protocol_image(rgb_upright)
        width = pixels.shape[1]
        shown = rotate_quarter(pixels, k)
        shown_marker = None if marker is None else rotate_pixel(marker, k, width)
        origin = crop = up = None
        if stage == "fine":
            if center_uv is None:
                raise ValueError("fine pointing needs center_uv (upright pixels of the camera)")
            center_uv = tuple(float(v) for v in np.asarray(center_uv, dtype=float).reshape(2))
            url, origin, drawn = fine_marked_overlay_at(shown, rotate_pixel(center_uv, k, width),
                                                        marker_uv=shown_marker)
            crop, up = FINE_CROP_PX, FINE_UPSAMPLE
            what = "an enlarged crop with 64 numbered points"
        else:
            center_uv = None if center_uv is None else tuple(float(v) for v in np.asarray(center_uv,
                                                                                          dtype=float).reshape(2))
            url, drawn = _ringed_url(coarse_overlay(shown), shown_marker)
            what = "the whole view with an 8x8 grid of numbered cells"
        if drawn:
            what += "; the point the robot rejected is ringed"
        if camera == "wrist":
            view = (f"wrist camera, turned so that its left and right match the external camera; the robot's "
                    f"fingertips are at the {FINGERTIP_EDGE[k]} edge of the full view; {what}")
        else:
            view = f"external camera, upright; {what}"
        fields = {"part": text, "camera": camera, "image": view, "previous_answer_rejected": previous,
                  "purpose": purpose}
        meta = {"kind": kind, "stage": stage, "camera": camera, "pid": pid, "text": text, "center_uv": center_uv,
                "origin": origin, "crop_px": crop, "upsample": up, "quarter_turns": k, "image_size": width,
                "previous_rejected": previous, "replan": False, "review": False, "purpose": purpose,
                "marker_uv": marker, "marker_drawn": bool(drawn)}
        label = (f"{camera.capitalize()} camera " + ("crop with 64 numbered points" if stage == "fine"
                                                     else "view with 64 numbered grid cells")
                 + (" (the rejected point is ringed):" if drawn else ":"))
        _safe_strings(fields, f"{kind} request")
        return kind, {"fields": fields, "images": [(label, url)]}, meta

    def request_point(self, rgb_upright, *, stage: str, camera: str, text: str, pid: str, center_uv=None,
                      quarter_turns: int, previous_rejected: str | None = None, marker_uv=None,
                      purpose: str | None = None) -> PointReply:
        """One pointing call on the upright image of ``camera`` (wrist: shown turned by quarter_turns; external:
        quarter_turns 0). Fine: a crop centred on ``center_uv`` (upright). The answer is mapped back to upright
        pixels; ``answer_id`` None means 0 (not visible). ``origin`` is the crop origin in the shown frame.
        ``purpose`` (POINT_PURPOSES) is sent as the request's purpose; ``marker_uv`` (upright) is ringed (BC-2)."""
        kind, request, meta = self._point_request(rgb_upright, stage=stage, camera=camera, text=text, pid=pid,
                                                  center_uv=center_uv, quarter_turns=quarter_turns,
                                                  previous_rejected=previous_rejected, marker_uv=marker_uv,
                                                  purpose=purpose)
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
                        review=False, reask_error=None, marker_uv=None, step_history=None):
        kind = "review" if review else "repair"
        k = _turns(quarter_turns)
        if status_block is None:
            raise ValueError("a repair or review request needs the plan status block")
        block = _status_block(status_block)
        outcome = _last_outcome(last_outcome)
        history = _step_history(step_history)
        if history is not None and review:
            raise ValueError("step_history is sent only with a repair request (a review has no failed step)")
        holding = _pid_or_none(holding, "holding")
        reask = _short_text(reask_error, ISSUES_TEXT_MAX_CHARS, "reask_error")
        marker = _marker_uv(marker_uv)
        external = _protocol_image(external_rgb)
        wrist = rotate_quarter(_protocol_image(wrist_rgb), k)
        wrist_url, drawn = encode_png_data_url(wrist), False
        if marker is not None:
            frame = Image.fromarray(np.array(wrist, dtype=np.uint8, order="C"), mode="RGB")
            drawn = _ring(frame, rotate_pixel(marker, k, wrist.shape[1]))
            if drawn:
                wrist_url = _encode(frame)
        edge = FINGERTIP_EDGE[k]
        wrist_label = (f"Image 2: wrist camera, clean (turned; fingertips at the {edge} edge):" if not drawn
                       else f"Image 2: wrist camera (turned; fingertips at the {edge} edge; the rejected pointed "
                            "point is ringed):")
        images = [("Image 1: external camera, clean:", encode_png_data_url(external)), (wrist_label, wrist_url)]
        fields = {"plan": block, "last_outcome": outcome}
        if not review:
            fields["step_history"] = history              # RL-C1: right after last_outcome (null when none)
        fields.update(holding=holding, previous_answer_rejected=reask)
        meta = {"kind": kind, "stage": None, "camera": None, "pid": None, "text": None, "center_uv": None,
                "origin": None, "crop_px": None, "upsample": None, "quarter_turns": k, "image_size": WRIST_IMAGE_PX,
                "previous_rejected": reask, "replan": False, "review": bool(review), "holding": holding,
                "purpose": None, "marker_uv": marker, "marker_drawn": bool(drawn)}
        _safe_strings(fields, f"{kind} request")
        return kind, {"fields": fields, "images": images}, meta

    def request_repair(self, external_rgb, wrist_rgb, *, quarter_turns: int, status_block: dict,
                       last_outcome: dict | None, holding: str | None, review: bool = False,
                       reask_error: str | None = None, marker_uv=None, step_history: dict | None = None) -> dict:
        """One repair (or review) call; returns the schema-valid answer JSON (``_plan.parse_repair`` builds it).
        ``last_outcome`` carries exactly reason, evidence_text and thresholds [C-45]. ``reask_error`` (the
        StepRunner's issues text after an answer it refused, at most ISSUES_TEXT_MAX_CHARS) is sent as
        previous_answer_rejected; ``marker_uv`` (upright wrist pixels) rings the rejected pointed point on the wrist
        image; ``step_history`` (RL-C1, repair only) follows last_outcome."""
        kind, request, meta = self._repair_request(external_rgb, wrist_rgb, quarter_turns=quarter_turns,
                                                   status_block=status_block, last_outcome=last_outcome,
                                                   holding=holding, review=review, reask_error=reask_error,
                                                   marker_uv=marker_uv, step_history=step_history)
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
    states = [StepState(0, example.steps[0], "done", "placed", {}, {}, 0, 1, []),
              StepState(1, example.steps[1], "failed", "hold_lost", {}, {}, 0, 1, [])]
    block = plan_status_block(example, states, holding=None)
    outcome = {"reason": "hold_lost", "evidence_text": "part C: the gap fell to the hold floor during the turn",
               "thresholds": [["G6_hold_floor_mm", 0.5]]}
    requests = {
        "plan": client._plan_request(grey, grey, quarter_turns=0, fingertip_edge=FINGERTIP_EDGE[0]),
        "point_fine": client._point_request(grey, stage="fine", camera="wrist", text=example.parts[0].text,
                                            pid=example.parts[0].pid, center_uv=centre, quarter_turns=0,
                                            purpose="grasp_point"),
        "point_coarse": client._point_request(grey, stage="coarse", camera="wrist", text=example.parts[4].text,
                                              pid=example.parts[4].pid, quarter_turns=0, purpose="region_seed"),
        "repair": client._repair_request(grey, grey, quarter_turns=0, status_block=block, last_outcome=outcome,
                                         holding=None)}
    return {name: client.build_payload(request, mode=kind) for name, (kind, request, _meta) in requests.items()}
