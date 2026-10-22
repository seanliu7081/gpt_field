"""ORACLE pointing and plan stub for the zero-model-call agent_SpeedUp_v2 smokes (TEST TOOL; contract §C14.5,
spec §6.6). Never imported by a control or model-interface module; refused in live mode.

The stub stands in for the model in ``--sim-smoke relation`` and in the relation smokes: it answers the client's
plan, pointing, repair and review requests (``ScriptedAnswers.answer(kind, meta)``) from evaluator geometry, read
live through the simulator's evaluator-only ``oracle`` op (SIM-EXT, ``agent_speedup_v2_sim``). It may therefore
name objects; it reads evaluator geometry ONLY to choose scripted cells and points, which then pass through the
same client mapping and the same executor answer checks as a model answer would.

- ``ORACLE_PARTS[task_key][pid]``: the evaluator point each plan part id stands for (a ``points`` op item; exact
  body/geom names from SURVEY/tasks/t2_dump.json) plus the body/owner names the smoke reports segment.
- ``ORACLE_PLANS[task_key]``: the spec §3.9 plan (scene cut to <= 80 characters); the ``w`` / ``x`` cells are
  filled at plan time from the live projections (``w`` in the k-turned wrist grid of THAT request, ``x`` in the
  upright external grid). A truth point outside the image gives 0, except when the part is cut by the border
  ([C-38]: the start-view moka pot): then the cell of the part's visible pixel nearest to the projected truth
  (within one cell of it; any distance for a ``whole``-object part), as a correct model would answer.
- Part masks (review BH-3). A part whose ``mask`` item also covers other parts (the handle loop and the pot share
  one body and one visual mesh; the lever and the knob disk share one body) carries ``region``: its own collision
  geoms. Its visible mask is then the ``segment`` mask AND the union of the projected oriented boxes of those
  geoms (per-geom convex hulls, grown by ``REGION_DILATE_PX`` for the visual mesh's overhang). So the border rule
  above, and the self-check's landing / plan-cell checks, see only the part (the loop, not the pot body).
- ``point_fine``: the id of the crop point nearest the projected truth (request meta origin / crop_px /
  quarter_turns), and 0 when the truth lies OUTSIDE the crop, as a correct model would answer [C-37].
- ``point_coarse``: the cell containing the projected truth (0 outside the image).
- Occlusion (review BH-6). Pointing answers use the projected truth even when something renders in front of it.
  Every pointing record and every plan cell therefore carries ``occluded`` / ``occluder`` (``"robot"`` or the
  owner seen at the truth pixel) / ``front_m``, from the ``segment`` depth at the truth pixel against the point's
  depth (occluded beyond ``OCCLUSION_DEPTH_TOL_M``), so a report can count answers more helpful than a model's.
  ``zero_when_robot_occluded=True`` (opt-in; default off = contract behaviour) answers 0 to a pointing request
  whose truth pixel shows the robot in front. A pid outside ``ORACLE_PARTS`` raises ValueError.
- ``repair``: retry; ``review``: hold_still.
- ``perturb_px`` displaces every pointing answer (fine and coarse; never the plan cells) by exactly that many
  pixels in a seeded random direction before quantization; ``wrong_item=pid`` answers every request about that
  pid with the NEXT part's truth point (plan order, cyclic), e.g. the pot body for the handle.

Pointing anchors are not G1 truth (review BH-1; LEAD RULING OPEN). ``ORACLE_PARTS`` fixes one anchor per part
for pointing; its z rule (below) gives C = 1.0292 (the tilted box surface under the anchor xy; the spec's 1.0345
is that box's aabb top, 5.3 mm higher) and D = 1.0518 (the lid knob g16, the highest pot surface at the body's
centre xy; the spec's "body top 1.0331" is g2's aabb top). For pointing these few mm move a projection by
<= ~3 px. G1 heights must NOT be scored against these anchors: ``surface_truth(xy)`` reports, at the MEASURED xy,
the collision surface (``heights``, robot skipped) and each camera's rendered visual surface (segment depth turned
back into a world point), and ``truth_points(detail=True)`` gives the aabb z. Measured on task 2 the visual and
collision surfaces differ by part (lever visual ~4 mm above the collision 0.955; handle ~1-2 mm below; burner
~0), close to the GS-2 3 mm median, so which one GS-2 gates on is for the lead to rule; reports should carry all
three.

Kinematic timing (review BH-2). The oracle op reads the simulator after the step's endpoint refresh, one physics
substep AHEAD of the step's observation (``agent_speedup_v2_sim.V2LiberoWorker.oracle``). At a settled pose they
agree (<= 0.1 px); at full hand speed the wrist projections / renders differ by up to ~4 px. Compare the policy's
observation-derived estimates with ORACLE truth on settled frames only (e.g. HandLocator +-6 px, pointing error),
or budget ~4 px at full speed; the self-check's ``motion_lag`` rows measure it.

Live refusal [C-11]: the constructor raises PermissionError unless the simulator was started with
``monitor_options["oracle"] is True``; the worker refuses the op independently (``agent_speedup_v2_sim``).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from gpt_field.mark_prompting import rotate_pixel, unrotate_pixel  # noqa: E402

ORACLE_VERSION = "agent-speedup-v2-oracle-2"
REQUEST_KINDS = ("plan", "replan", "point_fine", "point_coarse", "repair", "review")
IMAGE_PX = 512
GRID = 8                 # coarse cells and fine points: 8 x 8, ids 1..64 row-major from the top left (v1 marks)
FINE_CROP_PX = 128       # the client's FINE_CROP_PX (V6); the request meta's crop_px wins when present
PLAN_MIN_VISIBLE_FRACTION = 0.01   # a border-cut part counts as visible in a cell from 1 % of its pixels (41 px)
PLAN_BORDER_REACH_PX = 64          # a part point outside the image may take a visible pixel at most one cell away
# A part region (its projected collision boxes) grows by 4 px: the visual mesh overhangs the collision boxes by a
# few mm (the lever top renders ~4 mm above its box, review BH-1), i.e. ~4-7 px at the 0.15-0.25 m wrist range,
# while the loop's nearest pot-body box (g4) is ~1 cm (~15 px) away, so the grown region still excludes the body.
REGION_DILATE_PX = 4
# Occluded = something renders more than 1.5 cm in front of the truth point at its pixel. The visual vs collision
# surface offset at a visible part reaches 7.6 mm along the external ray (the lever top), so 1 cm (the self-check's
# landing tolerance) would sit too close to it; a finger (>= 1.5 cm thick) or another object is well beyond.
OCCLUSION_DEPTH_TOL_M = 0.015
VISUAL_SURFACE_ITERATIONS = 3      # surface_truth: z fixed-point passes (a flat surface converges in 2)
VISUAL_SURFACE_CONVERGED_M = 1e-4

# Part points (evaluator truth): the xy of a geom-aabb point, and z = the highest collision surface at that xy
# (the ``heights`` op, robot skipped) when that surface belongs to the part's owner, else the aabb z. The collision
# geoms are tilted boxes, so an aabb top floats above the surface (the handle box g13: aabb top 1.0345, surface at
# its centre 1.0292); the surface z keeps the point on what a camera sees. A: the knob lever's ridge (geoms g19-g22,
# union aabb top centre, ~93 x 25 mm, top 0.955); B: the burner plate centre (0.9305); C: the moka handle loop's
# outer top segment (g13) at the grasp y; D: the moka body (g2) top centre (the lid knob, 1.052, is the surface
# there); E: the frypan handle (g1) midpoint. ``mask`` is the ``segment`` mask item of the part's visible pixels
# (a body subtree, or the whole object for C/D/E: the handle loop and the pot share one body). ``region`` (A, C, E;
# review BH-3) limits that mask to the part's own collision boxes: A the lever ridge g19-g22 (not the knob disk
# g5-g18), C the handle loop g12-g15 (upper inner, upper outer, outer upright, lower; not the body g2-g8, the root
# g11 or the lid knob g16), E the handle g1 and its root links g2-g3 (not the pan g4-g25). B (its own body) and D
# (the whole pot) need none.
ORACLE_PARTS = {
    "libero_10_t2": {
        "A": {"item": {"geom": ["flat_stove_1_g19", "flat_stove_1_g20", "flat_stove_1_g21", "flat_stove_1_g22"],
                       "where": "aabb_top_centre"},
              "body": "flat_stove_1_button", "owner": "flat_stove_1", "mask": {"body": "flat_stove_1_button"},
              "region": ["flat_stove_1_g19", "flat_stove_1_g20", "flat_stove_1_g21", "flat_stove_1_g22"],
              "what": "rotary part ridge centre top"},
        "B": {"item": {"geom": "flat_stove_1_collision_burner", "where": "aabb_top_centre"},
              "body": "flat_stove_1_burner", "owner": "flat_stove_1", "mask": {"body": "flat_stove_1_burner"},
              "what": "burner plate centre top"},
        "C": {"item": {"geom": "moka_pot_1_g13", "where": "aabb_top_centre"},
              "body": "moka_pot_1_main", "owner": "moka_pot_1", "mask": {"owner": "moka_pot_1"},
              "region": ["moka_pot_1_g12", "moka_pot_1_g13", "moka_pot_1_g14", "moka_pot_1_g15"],
              "what": "handle loop outer segment top"},
        "D": {"item": {"geom": "moka_pot_1_g2", "where": "aabb_top_centre"},
              "body": "moka_pot_1_main", "owner": "moka_pot_1", "mask": {"owner": "moka_pot_1"}, "whole": True,
              "what": "moka body top centre"},
        "E": {"item": {"geom": "chefmate_8_frypan_1_g1", "where": "aabb_top_centre"},
              "body": "chefmate_8_frypan_1_main", "owner": "chefmate_8_frypan_1",
              "mask": {"owner": "chefmate_8_frypan_1"},
              "region": ["chefmate_8_frypan_1_g1", "chefmate_8_frypan_1_g2", "chefmate_8_frypan_1_g3"],
              "what": "frypan handle midpoint"},
    },
}

# Spec §3.9 task-2 plan; w/x are placeholders filled per request from the live projections.
ORACLE_PLANS = {
    "libero_10_t2": {
        "scene": "stove off; knob lever below the stove; moka pot at top edge, handle toward pan",
        "parts": [
            {"id": "A", "text": "stove knob: lever on top of the knob", "move": "hinged", "w": 0, "x": 0},
            {"id": "B", "text": "stove: ring burner plate", "move": "fixed", "w": 0, "x": 0},
            {"id": "C", "text": "moka pot: black handle loop", "move": "free", "w": 0, "x": 0},
            {"id": "D", "text": "moka pot: whole pot", "move": "free", "w": 0, "x": 0},
            {"id": "E", "text": "frying pan: long handle", "move": "free", "w": 0, "x": 0}],
        "hold_true": [],
        "steps": [
            {"do": "turn", "obj": "A", "grip": ["A"], "ref": None, "sense": "any", "keep": [],
             "until": [{"o": "effect", "p": "B", "txt": "burner rings turn red"}, {"o": "stop", "p": None, "txt": None}]},
            {"do": "place_on", "obj": "D", "grip": ["C"], "ref": "B", "sense": "any",
             "keep": [{"c": "upright", "p": "D"}, {"c": "clear_of", "p": "E"}], "until": []}],
    },
}
REPAIR_ANSWER = {"why": "oracle", "action": "retry", "part": None, "step": None}
REVIEW_ANSWER = {"why": "oracle", "action": "hold_still", "part": None, "step": None}


# ---------------------------------------------------------------------------------------------------------
# grids (upright or k-turned pixels: u right, v down, pixel centres at integers)
def in_image(uv, width=IMAGE_PX):
    return uv is not None and -0.5 <= float(uv[0]) < width - 0.5 and -0.5 <= float(uv[1]) < width - 0.5


def coarse_cell_of(uv, width=IMAGE_PX):
    """The 8x8 cell (1..64) whose centre ((col + 0.5) * W/8, v1 ``coarse_cell_center``) is nearest; 0 outside."""
    if not in_image(uv, width):
        return 0
    cell = width / GRID
    col = min(max(int(math.floor(float(uv[0]) / cell)), 0), GRID - 1)
    row = min(max(int(math.floor(float(uv[1]) / cell)), 0), GRID - 1)
    return row * GRID + col + 1


def coarse_cell_centre(cell_id, width=IMAGE_PX):
    row, col = divmod(int(cell_id) - 1, GRID)
    cell = width / GRID
    return (float((col + 0.5) * cell), float((row + 0.5) * cell))


def v1_fine_point_pixel_at(origin, point_id, *, crop_px=FINE_CROP_PX):
    """v1's fine point layout (``mark_prompting.fine_point_pixel``) for a crop of ``crop_px`` at ``origin``."""
    row, col = divmod(int(point_id) - 1, GRID)
    spacing = crop_px / GRID
    return (float(origin[0] + (col + 0.5) * spacing), float(origin[1] + (row + 0.5) * spacing))


def fine_point_mapping():
    """The client's ``fine_point_pixel_at`` when the v2 client exists (Wave 1), else the restated v1 layout."""
    try:
        from gpt_field.agent_speedup_v2_client import fine_point_pixel_at
    except ImportError:
        return v1_fine_point_pixel_at, "v1_layout_restated"
    return fine_point_pixel_at, "agent_speedup_v2_client.fine_point_pixel_at"


def crop_origin(center_uv, crop_px=FINE_CROP_PX, width=IMAGE_PX):
    """A crop of ``crop_px`` centred on ``center_uv``, clamped inside the image (fallback when meta has no origin)."""
    limit = width - crop_px
    return tuple(int(min(max(round(float(c) - crop_px / 2.0), 0), limit)) for c in center_uv)


def in_crop(uv, origin, crop_px):
    return all(origin[i] - 0.5 <= float(uv[i]) < origin[i] + crop_px - 0.5 for i in (0, 1))


def nearest_fine_point(uv, origin, crop_px, mapping):
    best = min(range(1, GRID * GRID + 1),
               key=lambda pid: math.dist(mapping(origin, pid, crop_px=crop_px), (float(uv[0]), float(uv[1]))))
    return best


def convex_hull_2d(points):
    """Monotone-chain convex hull (counter-clockwise, no repeated end point) of 2-D points."""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def fill_polygons(polygons, width=IMAGE_PX, *, dilate_px=0):
    """Boolean width x width mask of the filled polygons (pixel centres at integers), grown by ``dilate_px``."""
    from PIL import Image, ImageDraw, ImageFilter
    canvas = Image.new("L", (width, width), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        polygon = [(float(u), float(v)) for u, v in polygon]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=255)
        elif len(polygon) == 2:
            draw.line(polygon, fill=255, width=1)
        elif polygon:
            draw.point(polygon, fill=255)
    if dilate_px > 0:
        canvas = canvas.filter(ImageFilter.MaxFilter(2 * int(dilate_px) + 1))
    return np.asarray(canvas) > 0


REGION_MIN_DEPTH_M = 0.01          # a region box with a corner closer than 1 cm to the camera plane is skipped


# ---------------------------------------------------------------------------------------------------------
def canonical_task_key(task_key):
    from gpt_field.agent_speedup_v2_tasks import resolve_v2_task
    return resolve_v2_task(task_key).key


def oracle_enabled(sim):
    options = getattr(sim, "monitor_options", None) or {}
    metadata = getattr(sim, "metadata", None) or {}
    return options.get("oracle") is True or metadata.get("oracle_enabled") is True


def require_oracle_sim(sim):
    if not oracle_enabled(sim):
        raise PermissionError("the ORACLE stub needs a simulator started with monitor_options oracle=True "
                              "(zero-call smokes only; never in live mode)")
    return sim


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


class OracleAnswers:
    """``ScriptedAnswers`` from live evaluator geometry (see the module docstring)."""

    def __init__(self, sim, task_key: str, *, perturb_px: float = 0.0, wrong_item: str | None = None, seed: int = 0,
                 zero_when_robot_occluded: bool = False):
        self.task_key = canonical_task_key(task_key)
        if self.task_key not in ORACLE_PARTS or self.task_key not in ORACLE_PLANS:
            raise ValueError(f"no ORACLE parts/plan for task {self.task_key!r}")
        self.sim = require_oracle_sim(sim)
        if isinstance(perturb_px, bool) or not isinstance(perturb_px, (int, float)) or not (
                math.isfinite(perturb_px) and perturb_px >= 0):
            raise ValueError("perturb_px must be a finite number >= 0")
        if type(zero_when_robot_occluded) is not bool:
            raise ValueError("zero_when_robot_occluded must be a boolean")
        self.parts = ORACLE_PARTS[self.task_key]
        if wrong_item is not None and wrong_item not in self.parts:
            raise ValueError(f"wrong_item must be one of {sorted(self.parts)} or None")
        self.perturb_px, self.wrong_item, self.seed = float(perturb_px), wrong_item, int(seed)
        self.zero_when_robot_occluded = zero_when_robot_occluded
        self._rng = np.random.default_rng(self.seed)
        self.mapping, self.mapping_name = fine_point_mapping()
        self.records: list[dict] = []

    # truth ---------------------------------------------------------------------------------------------
    def check_pid(self, pid):
        if not isinstance(pid, str) or pid not in self.parts:
            raise ValueError(f"unknown ORACLE part id {pid!r}; parts of {self.task_key}: {sorted(self.parts)}")
        return pid

    def _source_pid(self, pid):
        self.check_pid(pid)
        if pid != self.wrong_item:
            return pid
        order = list(self.parts)
        return order[(order.index(pid) + 1) % len(order)]

    def truth_points(self, *, detail: bool = False) -> dict:
        """{pid: [x, y, z]} at the current state (module docstring rule); ``detail`` adds the z source."""
        items = [{"name": pid, **spec["item"]} for pid, spec in self.parts.items()]
        points = self.sim.oracle({"kind": "points", "items": items})["points"]
        pids = list(points)
        heights = self.sim.oracle({"kind": "heights", "xy": [points[pid][:2] for pid in pids]})
        out, info = {}, {}
        for pid, z, hit in zip(pids, heights["z"], heights["hit"]):
            on_part = z is not None and hit is not None and hit.get("owner") == self.parts[pid]["owner"]
            out[pid] = [float(points[pid][0]), float(points[pid][1]), float(z if on_part else points[pid][2])]
            info[pid] = {"xyz": out[pid], "aabb_z": float(points[pid][2]), "surface_z": z,
                         "surface_hit": hit, "z_source": "surface" if on_part else "aabb"}
        return info if detail else out

    def surface_truth(self, xy_list, *, cameras=("wrist", "external"), iterations=VISUAL_SURFACE_ITERATIONS):
        """G1 truth candidates at MEASURED xy (review BH-1; which one GS-2 gates on is a lead ruling, so all are
        reported). Per xy: ``surface_z`` / ``surface_hit``, the highest collision surface under it (``heights``,
        robot skipped); ``visual[camera]``, the rendered (visual-mesh) surface that camera sees there: the segment
        depth at the pixel of (x, y, z) turned back into a world point along the camera ray, with z iterated
        (``iterations`` passes, xy kept) -> {"z", "xyz", "xy_offset_mm" (that point's xy minus the queried xy),
        "front_m_first" (collision-point depth minus rendered depth on the first pass), "owner", "geom",
        "robot" (at the pixel), "passes", "converged"}; None when out of view or no collision surface there.
        Oracle kinematics are one substep ahead of the observation: call it on settled frames."""
        rows = []
        xy = [[float(p[0]), float(p[1])] for p in xy_list]
        if not xy:
            return rows
        heights = self.sim.oracle({"kind": "heights", "xy": xy})
        for point, z, hit in zip(xy, heights["z"], heights["hit"]):
            rows.append({"xy": point, "surface_z": z, "surface_hit": hit, "visual": {}})
        for camera in cameras:
            current = {i: float(row["surface_z"]) for i, row in enumerate(rows) if row["surface_z"] is not None}
            result, first_front = {}, {}
            for npass in range(1, max(1, int(iterations)) + 1):
                ids = [i for i in current if not (result.get(i) or {}).get("converged")]
                if not ids:
                    break
                reply = self.sim.oracle({"kind": "project", "camera": camera,
                                         "xyz": [[*rows[i]["xy"], current[i]] for i in ids]})
                centre = np.asarray(reply["cam_xyz"], float)
                seen = [k for k in range(len(ids)) if reply["uv"][k] is not None and reply["in_image"][k]]
                pixels = self.sim.oracle({"kind": "segment", "camera": camera, "masks": [], "mask_png": False,
                                          "uv": [reply["uv"][k] for k in seen]})["at"] if seen else []
                lookup = dict(zip(seen, pixels))
                for k, i in enumerate(ids):
                    pixel, depth = lookup.get(k), reply["depth_m"][k]
                    if pixel is None or not depth > 0:
                        current.pop(i)            # out of view: keep the last pass's result (None on the first)
                        continue
                    point = np.array([*rows[i]["xy"], current[i]])
                    visual = centre + (point - centre) * (float(pixel["depth_m"]) / float(depth))
                    first_front.setdefault(i, float(depth) - float(pixel["depth_m"]))
                    result[i] = {"z": float(visual[2]), "xyz": visual.tolist(),
                                 "xy_offset_mm": 1000.0 * math.hypot(visual[0] - point[0], visual[1] - point[1]),
                                 "front_m_first": first_front[i], "owner": pixel.get("owner"),
                                 "geom": pixel.get("geom"), "robot": bool(pixel.get("robot")), "passes": npass,
                                 "converged": abs(float(visual[2]) - current[i]) < VISUAL_SURFACE_CONVERGED_M}
                    current[i] = float(visual[2])
            for i, row in enumerate(rows):
                row["visual"][camera] = result.get(i)
        return _jsonable(rows)

    def project(self, camera, xyz_list):
        reply = self._project_reply(camera, xyz_list)
        return reply["uv"], reply["in_image"]

    def _project_reply(self, camera, xyz_list):
        return self.sim.oracle({"kind": "project", "camera": camera, "xyz": [list(p) for p in xyz_list]})

    def _truth_uv(self, pid, camera):
        source = self._source_pid(pid)
        xyz = self.truth_points()[source]
        reply = self._project_reply(camera, [xyz])
        return source, xyz, reply["uv"][0], reply["in_image"][0], reply["depth_m"][0]

    def occlusion(self, camera, uvs, depths, width=IMAGE_PX):
        """Per projected point: None when out of view, else {"occluded", "occluder" ("robot" / the owner seen;
        None when not occluded), "front_m" (point depth - rendered depth at its pixel), "pixel_owner"}."""
        out = [None] * len(uvs)
        visible = [k for k, uv in enumerate(uvs) if uv is not None and in_image(uv, width) and depths[k] is not None]
        if not visible:
            return out
        reply = self.sim.oracle({"kind": "segment", "camera": camera, "uv": [list(uvs[k]) for k in visible],
                                 "masks": [], "mask_png": False})
        for k, pixel in zip(visible, reply.get("at") or []):
            if pixel is None:
                continue
            seen = "robot" if pixel.get("robot") else pixel.get("owner")
            front = float(depths[k]) - float(pixel["depth_m"])
            occluded = front > OCCLUSION_DEPTH_TOL_M
            out[k] = {"occluded": occluded, "occluder": seen if occluded else None, "front_m": front,
                      "pixel_owner": seen}
        return out

    def _robot_occluded(self, occlusion):
        return bool(self.zero_when_robot_occluded and occlusion and occlusion["occluded"]
                    and occlusion["occluder"] == "robot")

    def _perturbed(self, uv):
        if uv is None or self.perturb_px == 0.0:
            return uv
        angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        return [float(uv[0]) + self.perturb_px * math.cos(angle), float(uv[1]) + self.perturb_px * math.sin(angle)]

    # ScriptedAnswers -----------------------------------------------------------------------------------
    def answer(self, kind: str, meta: dict) -> dict:
        if kind not in REQUEST_KINDS:
            raise ValueError(f"unknown request kind {kind!r}")
        meta = dict(meta or {})
        if kind in ("plan", "replan"):
            return self._plan(kind, meta)
        if kind == "point_fine":
            return self._point_fine(meta)
        if kind == "point_coarse":
            return self._point_coarse(meta)
        answer = copy.deepcopy(REPAIR_ANSWER if kind == "repair" else REVIEW_ANSWER)
        self.records.append({"i": len(self.records), "kind": kind, "answer": answer})
        return answer

    def _plan(self, kind, meta):
        turns = int(meta.get("quarter_turns", 0) or 0) % 4
        width = int(meta.get("image_size", IMAGE_PX) or IMAGE_PX)
        plan = copy.deepcopy(ORACLE_PLANS[self.task_key])
        truth = self.truth_points()
        pids = [part["id"] for part in plan["parts"]]
        xyz = [truth[self._source_pid(pid)] for pid in pids]
        wrist, external = self._project_reply("wrist", xyz), self._project_reply("external", xyz)
        wrist_uv, wrist_in = wrist["uv"], wrist["in_image"]
        external_uv, external_in = external["uv"], external["in_image"]
        # occlusion is recorded, not acted on: plan cells keep the contract rule (review BH-6)
        wrist_occ = self.occlusion("wrist", wrist_uv, wrist["depth_m"], width)
        external_occ = self.occlusion("external", external_uv, external["depth_m"], width)
        cells = []
        for part, uvw, inw, uvx, inx, occw, occx in zip(plan["parts"], wrist_uv, wrist_in, external_uv, external_in,
                                                        wrist_occ, external_occ):
            source = self._source_pid(part["id"])
            part["w"], w_rule = self._plan_cell(source, "wrist", uvw, inw, turns, width)
            part["x"], x_rule = self._plan_cell(source, "external", uvx, inx, 0, width)
            cells.append({"pid": part["id"], "source_pid": source, "w": part["w"], "w_rule": w_rule,
                          "x": part["x"], "x_rule": x_rule, "truth_uv_wrist": uvw,
                          "truth_uv_wrist_turned": rotate_pixel(uvw, turns, width) if uvw is not None else None,
                          "truth_uv_external": uvx, "w_occlusion": occw, "x_occlusion": occx,
                          "occluded": {"w": occw["occluded"] if occw else None,
                                       "x": occx["occluded"] if occx else None}})
        self.records.append(_jsonable({"i": len(self.records), "kind": kind, "quarter_turns": turns,
                                       "image_size": width, "cells": cells, "answer": plan}))
        return plan

    def _plan_cell(self, pid, camera, uv, inside, turns, width):
        """The cell of the projected truth. A truth outside the image (a part cut by the border, [C-38]) gets the
        cell of the part's visible pixel nearest to it, if that pixel lies within PLAN_BORDER_REACH_PX of the
        projected truth (any distance for a ``whole`` part: any visible pixel of the object is the object) and
        the cell shows >= PLAN_MIN_VISIBLE_FRACTION of the part; otherwise 0, as a correct model would answer."""
        if uv is None:
            return 0, "behind_camera"
        if inside:
            return coarse_cell_of(rotate_pixel(uv, turns, width), width), "truth_in_image"
        mask = self.visible_mask(pid, camera)
        if mask is None:
            return 0, "not_visible"
        vs, us = np.nonzero(mask)
        distance = np.hypot(us - float(uv[0]), vs - float(uv[1]))
        nearest = int(np.argmin(distance))
        reach = math.inf if self.parts[pid].get("whole") else PLAN_BORDER_REACH_PX
        if float(distance[nearest]) > reach:
            return 0, "truth_outside_image"
        candidate = coarse_cell_of(rotate_pixel((float(us[nearest]), float(vs[nearest])), turns, width), width)
        row, col = divmod(candidate - 1, GRID)
        size = width // GRID
        shown = np.rot90(mask, turns)[row * size:(row + 1) * size, col * size:(col + 1) * size]
        if shown.sum() >= PLAN_MIN_VISIBLE_FRACTION * size * size:
            return candidate, "border_cut_visible"
        return 0, "not_visible"

    def visible_mask(self, pid, camera):
        """Boolean upright mask of the part's visible pixels (``segment`` op, limited to the part's ``region``),
        or None when none is visible."""
        import base64
        import io
        from PIL import Image
        spec = self.parts[self.check_pid(pid)]
        reply = self.sim.oracle({"kind": "segment", "camera": camera, "masks": [{"name": pid, **spec["mask"]}]})
        info = reply["masks"][pid]
        if not info["pixels"] or not info.get("png"):
            return None
        raw = base64.b64decode(info["png"].split(",", 1)[1])
        return self.restrict_to_region(pid, camera, np.asarray(Image.open(io.BytesIO(raw))) > 0)

    def region_mask(self, pid, camera, width=IMAGE_PX):
        """Upright boolean mask of the part's ``region`` geoms in that camera: the convex hull of each geom's
        projected oriented-box corners (``points`` op with ``corners``), united and grown by REGION_DILATE_PX.
        None when the part has no ``region`` (its mask item is the part)."""
        geoms = self.parts[self.check_pid(pid)].get("region")
        if not geoms:
            return None
        reply = self.sim.oracle({"kind": "points", "corners": True,
                                 "items": [{"name": pid, "geom": list(geoms), "where": "aabb_centre"}]})
        corner_sets = reply["corners"][pid]
        projected = self._project_reply(camera, [corner for corners in corner_sets for corner in corners])
        polygons, start = [], 0
        for corners in corner_sets:
            uvs = projected["uv"][start:start + len(corners)]
            depths = projected["depth_m"][start:start + len(corners)]
            start += len(corners)
            if any(uv is None for uv in uvs) or min(depths) < REGION_MIN_DEPTH_M:
                continue                   # a box reaching onto / behind the camera plane has no stable outline
            polygons.append(convex_hull_2d(uvs))
        return fill_polygons(polygons, width, dilate_px=REGION_DILATE_PX)

    def restrict_to_region(self, pid, camera, mask):
        """``mask`` AND the part's region (unchanged when the part has none); None when nothing is left."""
        if mask is None:
            return None
        mask = np.asarray(mask, bool)
        region = self.region_mask(pid, camera, width=mask.shape[0])
        if region is None:
            return mask if mask.any() else None
        out = mask & region
        return out if out.any() else None

    def _point_fine(self, meta):
        pid, camera = meta.get("pid"), meta.get("camera", "wrist")
        width = int(meta.get("image_size", IMAGE_PX) or IMAGE_PX)
        turns = int(meta.get("quarter_turns", 0) or 0) % 4
        crop = int(meta.get("crop_px", FINE_CROP_PX) or FINE_CROP_PX)
        source, xyz, uv, inside, depth = self._truth_uv(pid, camera)
        occlusion = self.occlusion(camera, [uv], [depth], width)[0] if inside else None
        robot_zero = self._robot_occluded(occlusion)
        turned = rotate_pixel(uv, turns, width) if uv is not None else None
        origin = meta.get("origin")
        if origin is None:
            center = meta.get("center_uv")
            origin = crop_origin(rotate_pixel(center, turns, width) if center is not None else (width / 2, width / 2),
                                 crop, width)
        origin = (int(origin[0]), int(origin[1]))
        shown = self._perturbed(turned)
        answer_id, answered_uv = 0, None
        in_view = shown is not None and in_image(shown, width) and in_crop(shown, origin, crop)
        hit = in_view and not robot_zero
        if hit:
            answer_id = nearest_fine_point(shown, origin, crop, self.mapping)
            answered_uv = list(unrotate_pixel(self.mapping(origin, answer_id, crop_px=crop), turns, width))
        self.records.append(_jsonable({
            "i": len(self.records), "kind": "point_fine", "pid": pid, "source_pid": source, "camera": camera,
            "quarter_turns": turns, "origin": list(origin), "crop_px": crop, "truth_xyz": xyz, "truth_uv": uv,
            "truth_uv_turned": turned, "shown_uv_turned": shown, "in_image": bool(inside), "in_crop": bool(in_view),
            "occluded": occlusion["occluded"] if occlusion else None, "occlusion": occlusion,
            "answer_rule": "robot_occluded_zero" if in_view and robot_zero else (
                "nearest_fine_point" if hit else "truth_outside_crop"),
            "answer": {"point_id": answer_id}, "answered_uv": answered_uv,
            "error_px": math.dist(answered_uv, uv) if answered_uv is not None else None}))
        return {"point_id": answer_id}

    def _point_coarse(self, meta):
        pid, camera = meta.get("pid"), meta.get("camera", "wrist")
        width = int(meta.get("image_size", IMAGE_PX) or IMAGE_PX)
        turns = int(meta.get("quarter_turns", 0) or 0) % 4
        source, xyz, uv, inside, depth = self._truth_uv(pid, camera)
        occlusion = self.occlusion(camera, [uv], [depth], width)[0] if inside else None
        robot_zero = self._robot_occluded(occlusion)
        turned = rotate_pixel(uv, turns, width) if uv is not None else None
        shown = self._perturbed(turned)
        cell = coarse_cell_of(shown, width) if shown is not None and not robot_zero else 0
        answered_uv = list(unrotate_pixel(coarse_cell_centre(cell, width), turns, width)) if cell else None
        self.records.append(_jsonable({
            "i": len(self.records), "kind": "point_coarse", "pid": pid, "source_pid": source, "camera": camera,
            "quarter_turns": turns, "truth_xyz": xyz, "truth_uv": uv, "truth_uv_turned": turned,
            "shown_uv_turned": shown, "in_image": bool(inside),
            "occluded": occlusion["occluded"] if occlusion else None, "occlusion": occlusion,
            "answer_rule": "robot_occluded_zero" if robot_zero else ("truth_cell" if cell else "truth_outside_image"),
            "answer": {"cell_id": cell},
            "answered_uv": answered_uv, "error_px": math.dist(answered_uv, uv) if answered_uv is not None else None}))
        return {"cell_id": cell}

    def report(self) -> dict:
        by_kind, worst, occluded = {}, {}, {}
        for record in self.records:
            by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
            if record.get("error_px") is not None:
                worst[record["kind"]] = max(worst.get(record["kind"], 0.0), record["error_px"])
            flags = ([record.get("occluded")] if record["kind"].startswith("point_") else
                     [v for cell in record.get("cells", []) for v in (cell.get("occluded") or {}).values()])
            if any(flag is True for flag in flags):
                occluded[record["kind"]] = occluded.get(record["kind"], 0) + 1
        return _jsonable({"version": ORACLE_VERSION, "task_key": self.task_key, "perturb_px": self.perturb_px,
                          "wrong_item": self.wrong_item, "seed": self.seed, "fine_mapping": self.mapping_name,
                          "zero_when_robot_occluded": self.zero_when_robot_occluded,
                          "counts": by_kind, "occluded_answers": occluded, "max_error_px": worst,
                          "answers": copy.deepcopy(self.records)})


def compact_chars(plan) -> int:
    return len(json.dumps(plan, separators=(",", ":")))
