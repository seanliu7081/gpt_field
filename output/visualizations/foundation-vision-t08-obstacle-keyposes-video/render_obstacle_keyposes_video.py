#!/usr/bin/env python3
"""Render saved T08 observations, estimated maps and causal keypose states.

No simulator/model imports. Defaults to previews; full encoding requires
--encode. Relative goals are withheld until the recorded stage entry. Keypose
markers are never joined into an invented planner path.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
W, H, FPS = 1920, 1080, 20
BG, PANEL, FG = (14, 21, 33), (21, 31, 45), (233, 240, 247)
MUTED, GRID = (155, 173, 193), (47, 64, 82)
ORANGE, CYAN, PINK = (255, 167, 79), (75, 209, 239), (238, 115, 186)
GOLD, RED, GREEN, BLUE = (255, 219, 109), (255, 105, 104), (118, 217, 158), (100, 166, 255)
FONT = "/tmp/t04-browser-fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FAILURES = {"route_failure", "transit_stall", "contact_stall", "target_rejection", "segment_stall",
            "target_envelope_rejected", "stage_budget_exhausted", "stage_failed", "stall"}
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]
FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


@lru_cache(None)
def font(size):
    return ImageFont.truetype(FONT, size, index=2)


def text(draw, xy, value, size=22, fill=FG, anchor=None):
    draw.text(xy, str(value), font=font(size), fill=fill, anchor=anchor)


def fit_text(draw, xy, value, max_width, size=20, fill=FG, minimum=15):
    value = str(value)
    while size > minimum and draw.textlength(value, font=font(size)) > max_width:
        size -= 1
    if draw.textlength(value, font=font(size)) > max_width:
        while value and draw.textlength(value + "...", font=font(size)) > max_width:
            value = value[:-1]
        value += "..."
    text(draw, xy, value, size, fill)


def dash(draw, a, b, fill, width=2, on=7, off=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    vector = b - a
    length = np.linalg.norm(vector)
    if length < .1:
        return
    for start in np.arange(0, length, on + off):
        draw.line([tuple(a + vector * start / length),
                   tuple(a + vector * min(length, start + on) / length)], fill=fill, width=width)


def cube(lower, upper):
    return np.array([[x, y, z] for z in (lower[2], upper[2])
                     for y in (lower[1], upper[1]) for x in (lower[0], upper[0])], float)


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("Invalid saved wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def measured_rotation(frame):
    state = frame.get("state8")
    if state is not None:
        vector = np.asarray(state[3:6], float)
        angle = np.linalg.norm(vector)
        if angle < 1e-12:
            return np.eye(3)
        return quaternion_matrix([math.cos(angle / 2), *(vector / angle * math.sin(angle / 2))])
    q = frame.get("grip_quaternion_wxyz") or frame.get("quaternion_wxyz")
    return None if q is None else quaternion_matrix(q)


def step_of(event):
    return int(event.get("frame_index", event.get("control_step", event.get("step", 0))))


def current_call(data, step):
    available = [row for row in data["keyposes"] if row["decision_control_step"] <= step]
    return max((row["call_index"] for row in available), default=0)


def keypose_state(row, step, data):
    """Displayed state is computed only from events already reached in playback."""
    if step < row["decision_control_step"]:
        return "unrevealed"
    entry, end = row.get("entry_control_step"), row.get("end_control_step")
    final = row.get("status", "active")
    if end is not None and step >= end and entry is not None and final not in {"active", "running"}:
        return "completed" if final in {"reached", "completed"} else "stopped"
    if step == row["decision_control_step"]:
        return "pending"
    if entry is not None:
        return "pending" if step <= entry else "active"
    history = row.get("status_history", [])
    for item in reversed(history):
        at = item.get("control_step", item.get("frame_index", item.get("step")))
        if at is not None and step >= at and item.get("status") == "not_executed":
            return "not_executed"
    same_call_ends = [item.get("end_control_step") for item in data["keyposes"]
                      if item["call_index"] == row["call_index"] and item.get("end_control_step") is not None]
    next_decisions = [item["decision_control_step"] for item in data["keyposes"]
                      if item["call_index"] > row["call_index"]]
    known_end = min(next_decisions) if next_decisions else (max(same_call_ends) if same_call_ends else None)
    return "not_executed" if final == "not_executed" and known_end is not None and step >= known_end else "pending"


def visible_goal(row, step):
    if step < row["decision_control_step"] or row.get("is_pose_keypose") is False:
        return None
    if row.get("skill") not in {"move_to_pose", "move_relative", "approach"}:
        return None
    if row.get("skill") == "move_to_pose":
        return row.get("absolute_requested_goal_xyz_m") or row.get("requested_xyz_m")
    entry = row.get("entry_control_step")
    return row.get("resolved_goal_xyz_m") if entry is not None and step >= entry else None


def short_vector(value):
    return "[" + ", ".join(f"{float(component):+.3f}" for component in value) + "]"


class Charts:
    def __init__(self, bounds):
        self.lo, self.hi = np.asarray(bounds[0]), np.asarray(bounds[1])
        self.main = (482, 222, 1355, 737)
        self.xy, self.xz = (1458, 241, 1871, 432), (1458, 548, 1871, 739)
        self.matrix = np.array([[.78, -.63, 0], [.16, .22, -.72]])
        projected = cube(self.lo, self.hi) @ self.matrix.T
        low, high = projected.min(0), projected.max(0)
        self.scale = min((self.main[2] - self.main[0] - 150) / (high[0] - low[0]),
                         (self.main[3] - self.main[1] - 65) / (high[1] - low[1]))
        self.offset = np.array([(self.main[0] + self.main[2]) / 2,
                                (self.main[1] + self.main[3]) / 2]) - self.scale * (low + high) / 2

    def project(self, point):
        return tuple((np.asarray(point) @ self.matrix.T * self.scale + self.offset).tolist())

    def flat(self, point, axes, rectangle):
        a, b = axes
        x0, y0, x1, y1 = rectangle
        return (x0 + (point[a] - self.lo[a]) / (self.hi[a] - self.lo[a]) * (x1 - x0),
                y1 - (point[b] - self.lo[b]) / (self.hi[b] - self.lo[b]) * (y1 - y0))

    def projections(self):
        return (self.project, lambda p: self.flat(p, (0, 1), self.xy),
                lambda p: self.flat(p, (0, 2), self.xz))

    def base(self, image):
        draw = ImageDraw.Draw(image)
        text(draw, (485, 145), "Estimated obstacle map · World coordinates", 26)
        text(draw, (1410, 210), "Top view · XY", 23)
        text(draw, (1410, 515), "Side view · XZ", 23)
        for axis in (0, 1):
            for value in np.arange(math.ceil(self.lo[axis] * 10) / 10, self.hi[axis] + 1e-7, .1):
                p, q = self.lo.copy(), self.lo.copy()
                p[axis] = q[axis] = value
                q[1 - axis] = self.hi[1 - axis]
                draw.line([self.project(p), self.project(q)], fill=GRID)
        for axis, label in enumerate("XYZ"):
            q = self.lo.copy()
            q[axis] = self.hi[axis]
            draw.line([self.project(self.lo), self.project(q)], fill=MUTED, width=2)
            x, y = self.project(q)
            text(draw, (x + 7, y - 5), label + " (m)", 17, MUTED)
        for axes, rectangle in (((0, 1), self.xy), ((0, 2), self.xz)):
            x0, y0, x1, y1 = rectangle
            draw.rectangle(rectangle, outline=GRID)
            for axis in axes:
                for value in np.linspace(self.lo[axis], self.hi[axis], 4):
                    p, q = self.lo.copy(), self.hi.copy()
                    p[axis] = q[axis] = value
                    u, v = self.flat(p, axes, rectangle), self.flat(q, axes, rectangle)
                    draw.line([u, v], fill=GRID)
                    if axis == axes[0]:
                        text(draw, (u[0], y1 + 7), f"{value:.2f}", 15, MUTED, "mt")
                    else:
                        text(draw, (x0 - 9, u[1]), f"{value:.2f}", 15, MUTED, "rm")
            text(draw, ((x0+x1)/2, y1+29), "XYZ"[axes[0]] + " (m)", 16, MUTED, "mt")
            text(draw, (x0-53, (y0+y1)/2), "XYZ"[axes[1]], 17, MUTED, "mm")

    def triad(self, draw, point, rotation, *, length=.045, label=None, muted=False):
        origin = self.project(point)
        for axis, color in enumerate((RED, GREEN, BLUE)):
            endpoint = self.project(np.asarray(point) + rotation[:, axis] * length)
            if muted:
                dash(draw, origin, endpoint, color, width=2, on=3, off=3)
            else:
                draw.line([origin, endpoint], fill=color, width=3)
            vector = np.asarray(endpoint) - origin
            norm = np.linalg.norm(vector)
            if norm > 1:
                along = vector / norm
                side = np.array([-along[1], along[0]])
                tip = np.asarray(endpoint)
                draw.polygon([tuple(tip), tuple(tip - along * 7 + side * 3),
                              tuple(tip - along * 7 - side * 3)], fill=color)
        if label:
            text(draw, (origin[0] + 11, origin[1] + 12), label, 16, MUTED)

    def draw(self, image, boxes, clearance, trajectory, frame, keyposes, step, data, path):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for index, box in enumerate(boxes):
            low, high = np.array(box["lower_xyz_m"]), np.array(box["upper_xyz_m"])
            vertices = [self.project(point) for point in cube(low, high)]
            for face in FACES:
                draw.polygon([vertices[i] for i in face], fill=ORANGE+(13,))
            for i, j in EDGES:
                draw.line([vertices[i], vertices[j]], fill=ORANGE+(205,), width=2)
            expanded = [self.project(point) for point in cube(low-clearance, high+clearance)]
            for i, j in EDGES:
                dash(draw, expanded[i], expanded[j], ORANGE+(100,), 1, 6, 5)
            center = (low+high)/2
            center[2] = high[2]
            x, y = self.project(center)
            label = f"B{index+1}"
            rectangle = draw.textbbox((x, y-16), label, font=font(18), anchor="mm")
            draw.rectangle((rectangle[0]-4, rectangle[1]-2, rectangle[2]+4, rectangle[3]+2), fill=BG+(225,))
            text(draw, (x, y-16), label, 18, ORANGE+(255,), "mm")
            for axes, rectangle in (((0, 1), self.xy), ((0, 2), self.xz)):
                p, q = self.flat(low, axes, rectangle), self.flat(high, axes, rectangle)
                draw.rectangle((min(p[0],q[0]), min(p[1],q[1]), max(p[0],q[0]), max(p[1],q[1])),
                               fill=ORANGE+(30,), outline=ORANGE+(220,), width=2)
                p, q = self.flat(low-clearance, axes, rectangle), self.flat(high+clearance, axes, rectangle)
                corners = [(p[0],p[1]), (q[0],p[1]), (q[0],q[1]), (p[0],q[1])]
                for i in range(4):
                    dash(draw, corners[i], corners[(i+1)%4], ORANGE+(110,), width=1)
        image.alpha_composite(layer)
        draw = ImageDraw.Draw(image)
        for projection in self.projections():
            if len(trajectory) > 1:
                draw.line([projection(p) for p in trajectory], fill=PINK, width=3)
            if len(path) > 1:
                projected = [projection(p) for p in path]
                draw.line(projected, fill=CYAN, width=4)
                for x, y in projected:
                    draw.ellipse((x-2, y-2, x+2, y+2), fill=CYAN)
        for row in keyposes:
            goal = visible_goal(row, step)
            if goal is None:
                continue
            status = keypose_state(row, step, data)
            color = {"pending": MUTED, "active": GOLD, "completed": GREEN,
                     "stopped": RED, "not_executed": (112, 124, 144)}.get(status, MUTED)
            for projection_index, projection in enumerate(self.projections()):
                x, y = projection(goal)
                if status in {"not_executed", "stopped"}:
                    draw.line((x-5, y-5, x+5, y+5), fill=color, width=2)
                    draw.line((x-5, y+5, x+5, y-5), fill=color, width=2)
                else:
                    draw.line((x-6, y, x+6, y), fill=color, width=2)
                    draw.line((x, y-6, x, y+6), fill=color, width=2)
                if projection_index == 0:
                    text(draw, (x+8, y-20-15*(row["segment_index"] % 2)),
                         f"K{row['keypose_number']}", 17, color)
            explicit = row.get("explicit_quaternion_wxyz")
            if explicit is not None and status in {"pending", "active"}:
                self.triad(draw, goal, quaternion_matrix(explicit), length=.04)
        position = frame["grip_xyz_m"]
        for projection in self.projections():
            x, y = projection(position)
            draw.ellipse((x-6, y-6, x+6, y+6), fill=FG, outline=BG, width=2)
        rotation = measured_rotation(frame)
        if rotation is not None:
            self.triad(draw, position, rotation, label="tool")


def load_data(timeline, mapping_path):
    data = json.loads(timeline.read_text())
    mapping = json.loads(mapping_path.read_text())
    frames = data["frames"]
    if len(frames) != len(mapping["control_state_frames"]):
        raise ValueError("Timeline and camera mapping have different control-state counts")
    if len(frames) != 591:
        raise ValueError("This T08 replay must contain initial state plus exactly 590 policy actions")
    data.setdefault("keyposes", [])
    maps = {int(row["revision"]): row for row in data["maps"]}
    maps.setdefault(0, {"revision": 0, "avoid_boxes": [], "clearance_m": 0., "source": "uninitialized"})
    routes = {row["id"]: row for row in data["routes"]}
    points = [row["grip_xyz_m"] for row in frames]
    for obstacle_map in maps.values():
        for box in obstacle_map["avoid_boxes"]:
            points.extend([np.array(box["lower_xyz_m"])-obstacle_map["clearance_m"],
                           np.array(box["upper_xyz_m"])+obstacle_map["clearance_m"]])
    for row in data["keyposes"]:
        for name in ("absolute_requested_goal_xyz_m", "resolved_goal_xyz_m"):
            if row.get(name) is not None:
                points.append(row[name])
    points = np.asarray(points, float)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite saved chart coordinates")
    low = np.floor((points.min(0)-.02)*20)/20
    high = np.ceil((points.max(0)+.02)*20)/20
    high = np.maximum(high, low+.1)
    events = defaultdict(list)
    for event in data["events"]:
        events[step_of(event)].append(event)
    return data, mapping, maps, routes, Charts((low, high)), dict(events)


class CameraReader:
    def __init__(self, mapping):
        self.mapping = mapping
        self.capture = cv2.VideoCapture(str(mapping["source_video"]))
        if not self.capture.isOpened():
            raise RuntimeError("Cannot open recorded source video")
        self.index = -1
        self.last = None

    def get(self, step):
        row = self.mapping["control_state_frames"][step]
        wanted = row["source_frame_index"]
        if wanted == self.index and self.last is not None:
            return self.last
        if wanted < self.index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, wanted)
            self.index = wanted-1
        while self.index < wanted:
            if not self.capture.grab():
                raise RuntimeError(f"Cannot read saved source frame {self.index+1}")
            self.index += 1
        ok, frame = self.capture.retrieve()
        if not ok:
            raise RuntimeError(f"Cannot decode saved source frame {wanted}")
        crops = self.mapping["camera_crops_xyxy_exclusive"]
        images = []
        for name in ("external", "wrist"):
            x0, y0, x1, y1 = crops[name]
            image = Image.fromarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            images.append(image.resize((400,400), Image.Resampling.LANCZOS))
        self.last = images
        return images

    def close(self):
        self.capture.release()


def selected_route(step, frame, keyposes, data, routes):
    """No previous-call route can survive a newer decision boundary."""
    if not keyposes or step == keyposes[0]["decision_control_step"]:
        return []
    active = [row for row in keyposes if keypose_state(row, step, data) == "active"]
    if len(active) != 1 or active[0].get("profile") != "transit":
        return []
    row = active[0]
    route_id = row.get("route_id")
    if route_id is None:
        return []
    route = routes.get(route_id)
    if route is None:
        return []
    return route.get("path_xyz_m", []) or []


def note_for(step, frame, events, total, intro=False):
    if intro:
        return ("Initial observation · Obstacle map not yet initialized",
                "Recorded cameras only. Keyposes and map estimates appear when the corresponding Astra decision arrives.", MUTED)
    if step == total:
        return ("Raw task success · Second pot still held",
                "The last approach was active at termination. No post-release verification is shown; extra diagnostic actions are excluded.", GOLD)
    rows = events.get(step, [])
    first_contact = next((row for row in rows if row["kind"] == "first_forbidden_contact"), None)
    if first_contact:
        return ("First forbidden contact · First transported pot / burner",
                "Red rings mark the logged contact point. This is evaluator contact evidence, independent of the estimated boxes.", RED)
    failure = next((row for row in rows if row["kind"] in FAILURES or "stall" in row["kind"]), None)
    update = next((row for row in rows if row["kind"] == "map_update"), None)
    if failure:
        if step == 528:
            return ("Call 09 K2 stopped · K3 was not executed · Now call 10",
                    "The preceding transit stalled. Call 10 now requests the shown explicit orientation; no previous-call route remains active.", RED)
        suffix = " · New map / decision at this state" if update else " · Control returns to Astra"
        return ("Previous segment stopped" + suffix,
                "A stall is not itself collision proof. The table shows the current call; any later unexecuted segment is not replayed.", RED)
    if update:
        revision = update.get("map_revision", update.get("revision", frame.get("decision_boundary_map_revision", frame["map_revision"])))
        return (f"Astra obstacle-map update · Revision {revision}",
                "Orange boxes are model estimates; dashed extents include clearance. They are not a DA3 reconstruction or a full-robot check.", ORANGE)
    if frame.get("forbidden_contact_delta", 0) > 0:
        return ("Recorded forbidden contact · First transported pot / burner",
                "Contact monitoring is record-only. The pink line tracks the grip site, not the carried object's full swept volume.", RED)
    if frame.get("profile") == "contact":
        return ("Contact-profile motion · Transit route planner not active",
                "Keypose markers are requested goals; no line between keyposes is a planned path. Null orientation requests inherit entry pose.", MUTED)
    if frame.get("route_id"):
        return ("Transit motion · Recorded planner path and measured response",
                "Cyan appears only for the current active transit stage. The model's box map can omit object or robot extents.", CYAN)
    return ("Measured execution · Evidence remains conditional",
            "SAM 3.1 + DA3 produced no usable target 3D centers in this run. Astra's obstacle map remains a model estimate.", MUTED)


def draw_keypose_table(image, keyposes, step, data, call, intro):
    draw = ImageDraw.Draw(image)
    draw.rectangle((472, 797, 1888, 949), fill=PANEL)
    title = "Astra keyposes · Awaiting first decision" if intro else f"Astra call {call:02d} · Requested stages / keyposes"
    text(draw, (488, 800), title, 23)
    text(draw, (1872, 807), "Orientation: inherit unless explicit q is recorded", 17, MUTED, "rt")
    if intro:
        text(draw, (490, 853), "No future keyposes or obstacle map are shown before the first decision.", 20, MUTED)
        return
    headers = ((490,"Key"),(548,"State"),(710,"Stage"),(935,"XYZ world / relative delta (m)"),(1320,"Orientation"),(1510,"Recorded outcome"))
    for x, label in headers:
        text(draw, (x, 833), label, 16, MUTED)
    draw.line((488,855,1872,855), fill=GRID)
    row_height = 27 if len(keyposes) <= 3 else 15
    row_size = 18 if len(keyposes) <= 3 else 14
    for index, row in enumerate(keyposes):
        y = 858 + index*row_height
        status = keypose_state(row, step, data)
        color = {"pending": MUTED,"active": GOLD,"completed": GREEN,"stopped": RED,"not_executed": MUTED}[status]
        label = status.replace("_", " ")
        if step == len(data["frames"])-1 and status == "active":
            label = "active at stop"
        text(draw, (490,y), f"K{row['keypose_number']}", row_size, color)
        text(draw, (548,y), label, row_size, color)
        text(draw, (710,y), row["skill"], row_size)
        goal = visible_goal(row, step)
        delta = row.get("relative_delta_m")
        if goal is not None:
            target_text = short_vector(goal)
        elif delta is not None:
            target_text = "Delta " + short_vector(delta)
        else:
            target_text = "hold entry pose" if row["skill"] == "set_gripper" else "--"
        fit_text(draw, (935,y), target_text, 370, row_size, FG, minimum=14)
        explicit = row.get("explicit_quaternion_wxyz")
        text(draw, (1320,y), "explicit q" if explicit is not None else "inherit", row_size, GOLD if explicit is not None else MUTED)
        entry, end = row.get("entry_control_step"), row.get("end_control_step")
        if status in {"completed", "stopped", "not_executed"}:
            outcome = row.get("status_reason") or status
            if row["skill"] == "set_gripper" and status == "completed":
                outcome = "gap stable; grasp unverified"
        elif status == "active":
            outcome = "in progress; no result yet"
        else:
            outcome = "not started"
        fit_text(draw, (1510,y), outcome, 360, row_size, MUTED, minimum=14)


def render(step, cameras, data, maps, routes, charts, events, *, intro=False):
    frame, total = data["frames"][step], len(data["frames"])-1
    call = 0 if intro else current_call(data, step)
    keyposes = [] if intro else sorted([row for row in data["keyposes"] if row["call_index"] == call],
                                      key=lambda row: row["segment_index"])
    revision = 0 if intro else frame.get("decision_boundary_map_revision", frame["map_revision"])
    obstacle_map = maps[int(revision)]
    path = [] if intro or step == total else selected_route(step, frame, keyposes, data, routes)
    image = Image.new("RGBA", (W,H), BG+(255,))
    draw = ImageDraw.Draw(image)
    text(draw, (32,19), "T08 · Foundation vision, obstacle maps & keyposes", 36)
    text(draw, (32,76), "SAM 3.1 + DA3   ·   0.5x action replay; API waits omitted; key events paused", 21, MUTED)
    text(draw, (1888,24), f"Control step {step:04d} / {total}", 27, FG, "rt")
    text(draw, (1888,73), f"Action time {step/20:05.2f} s   ·   Map {revision}   ·   Clearance {obstacle_map['clearance_m']*100:g} cm", 20, MUTED, "rt")
    draw.line((32,124,1888,124), fill=GRID)
    text(draw, (32,143), "External camera", 23)
    text(draw, (32,601), "Wrist camera", 23)
    image.paste(cameras[0], (32,180))
    image.paste(cameras[1], (32,640))
    charts.base(image)
    draw = ImageDraw.Draw(image)
    for x, label, color, kind in ((490,"Estimated boxes",ORANGE,"box"),(680,"Clearance",ORANGE,"dash"),
        (840,"Planner path",CYAN,"line"),(1018,"Measured grip",PINK,"line"),
        (1224,"Keypose",GOLD,"plus"),(1374,"Contact",RED,"contact"),(1515,"Tool axes: X / Y / Z",MUTED,"axes")):
        y = 190
        if kind == "box": draw.rectangle((x,y,x+20,y+14), outline=color, width=2)
        elif kind == "dash": dash(draw,(x,y+7),(x+23,y+7),color)
        elif kind == "line": draw.line((x,y+7,x+23,y+7), fill=color, width=3)
        elif kind == "plus":
            draw.line((x,y+7,x+22,y+7),fill=color,width=2)
            draw.line((x+11,y-3,x+11,y+17),fill=color,width=2)
        elif kind == "contact": draw.ellipse((x+3,y,x+17,y+14),outline=color,width=2)
        else:
            for offset, axis_color in enumerate((RED,GREEN,BLUE)):
                draw.rectangle((x+offset*7,y+2,x+offset*7+4,y+13),fill=axis_color)
        text(draw, (x+29,y-8), label, 17, color if kind=="axes" else MUTED)
    trajectory = [row["grip_xyz_m"] for row in data["frames"][:step+1]]
    trajectory = trajectory[::max(1,len(trajectory)//500)] + [frame["grip_xyz_m"]]
    charts.draw(image, obstacle_map["avoid_boxes"], obstacle_map["clearance_m"],
                trajectory, frame, keyposes, step, data, path)
    draw = ImageDraw.Draw(image)
    if not intro:
        hit = next((row for row in events.get(step,[]) if row.get("contact_xyz_m")), None)
        if hit:
            for projection in charts.projections():
                x,y = projection(hit["contact_xyz_m"])
                draw.ellipse((x-11,y-11,x+11,y+11),outline=RED,width=3)
                draw.ellipse((x-2,y-2,x+2,y+2),fill=RED)
    text(draw, (490,761), "Triads: tool = measured; K = explicit target orientation · Grip-site path only · No grasp certificate", 17, MUTED)
    draw_keypose_table(image,keyposes,step,data,call,intro)
    draw = ImageDraw.Draw(image)
    title, detail, color = note_for(step,frame,events,total,intro)
    draw.rectangle((472,957,1888,1018),fill=PANEL)
    fit_text(draw,(490,958),title,1378,23,color)
    fit_text(draw,(490,990),detail,1378,18,MUTED)
    x0,x1,y = 490,1870,1033
    draw.line((x0,y,x1,y),fill=GRID,width=5)
    draw.line((x0,y,x0+(x1-x0)*step/total,y),fill=CYAN,width=5)
    for event_step, rows in events.items():
        color = RED if any(row["kind"] in FAILURES or "contact" in row["kind"] for row in rows) else ORANGE
        x = x0+(x1-x0)*event_step/total
        draw.line((x,y-5,x,y+5),fill=color,width=2)
    text(draw,(490,1048),"Model-estimated boxes; no full-robot collision guarantee. SAM/DA3 supported target centers: 0. Post-release diagnostic excluded.",16,MUTED)
    return image.convert("RGB")


def pauses(data, events):
    total = len(data["frames"])-1
    holds = {total:100}
    for step, rows in events.items():
        if any(row["kind"]=="map_update" for row in rows):
            holds[step]=max(holds.get(step,0),40)
        if any(row["kind"]=="first_forbidden_contact" for row in rows):
            holds[step]=max(holds.get(step,0),80)
        if any(row["kind"] in FAILURES or "stall" in row["kind"] for row in rows):
            holds[step]=max(holds.get(step,0),50)
    return holds


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data, mapping, events):
    forbidden_future_relative = []
    boundary_routes = []
    routes={row["id"]:row for row in data["routes"]}
    for step,frame in enumerate(data["frames"]):
        call=current_call(data,step)
        current=[row for row in data["keyposes"] if row["call_index"]==call]
        for row in current:
            if row.get("relative_delta_m") is not None and (row.get("entry_control_step") is None or step<row["entry_control_step"]):
                if visible_goal(row,step) is not None:
                    forbidden_future_relative.append([step,row["id"]])
        if current and step==current[0]["decision_control_step"] and selected_route(step,frame,current,data,routes):
            boundary_routes.append(step)
    if forbidden_future_relative or boundary_routes:
        raise AssertionError("Causal visualization audit failed")
    return {"control_states":len(data["frames"]),"keyposes":len(data["keyposes"]),
            "explicit_target_quaternions":sum(row.get("explicit_quaternion_wxyz") is not None for row in data["keyposes"]),
            "future_relative_goal_leaks":0,"active_routes_on_decision_boundaries":0,
            "map_updates":sum(row["kind"]=="map_update" for rows in events.values() for row in rows),
            "source_camera_mapping_states":len(mapping["control_state_frames"]),
            "diagnostic_actions_included":False}


def embed_chapters(output, chapters, duration):
    lines=[";FFMETADATA1","title=T08 foundation vision - obstacle maps and keyposes",
           "comment=Recorded-only visualization; no model or simulator replay; diagnostic actions excluded."]
    for index,chapter in enumerate(chapters):
        end=chapters[index+1]["video_time_s"] if index+1<len(chapters) else duration
        title=chapter["title"].replace("=","\\=").replace(";","\\;").replace("#","\\#")
        lines.extend(["[CHAPTER]","TIMEBASE=1/1000",f"START={round(chapter['video_time_s']*1000)}",
                      f"END={round(end*1000)}","title="+title])
    metadata=ROOT/"chapters.ffmeta"
    metadata.write_text("\n".join(lines)+"\n")
    temporary=output.with_name(output.stem+".chaptered.mp4")
    subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(output),"-i",str(metadata),
                    "-map_metadata","1","-c","copy","-movflags","+faststart",str(temporary)],check=True)
    temporary.replace(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline",type=Path,default=ROOT/"timeline.json")
    parser.add_argument("--mapping",type=Path,default=ROOT/"control_state_frames.json")
    parser.add_argument("--preview",action="store_true")
    parser.add_argument("--encode",action="store_true")
    parser.add_argument("--preview-steps",nargs="*",type=int)
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"t08-foundation-vision-obstacles-keyposes.mp4")
    args=parser.parse_args()
    if args.encode and args.preview:
        parser.error("Choose either preview or encode")
    data,mapping,maps,routes,charts,events=load_data(args.timeline,args.mapping)
    report=audit(data,mapping,events)
    (ROOT/"renderer_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    if args.audit_only:
        print(json.dumps(report));return
    total=len(data["frames"])-1
    holds=pauses(data,events)
    chosen=sorted(set(args.preview_steps or [0,1,24,93,188,243,297,328,400,500,total,*holds]))
    chosen=[step for step in chosen if 0<=step<=total]
    reader=CameraReader(mapping)
    try:
        if not args.encode:
            render(0,reader.get(0),data,maps,routes,charts,events,intro=True).save(ROOT/"preview-initial.png")
            for step in chosen:
                render(step,reader.get(step),data,maps,routes,charts,events).save(ROOT/f"preview-step-{step:04d}.png")
            print(json.dumps({"previews":chosen,"initial_preview":True,"audit":report}));return
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite an existing encoded video")
        command=["ffmpeg","-hide_banner","-loglevel","error","-f","rawvideo","-vcodec","rawvideo",
                 "-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-an","-c:v","libx264",
                 "-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart","-threads","4",str(args.output)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        chapters=[];outframe=0
        try:
            initial=render(0,reader.get(0),data,maps,routes,charts,events,intro=True).tobytes()
            chapters.append({"control_state":0,"initial_observation":True,"video_time_s":0.,"duration_s":2.,
                             "title":"Initial observation - map not initialized"})
            for _ in range(40):process.stdin.write(initial)
            outframe=40
            for step in range(total+1):
                image=render(step,reader.get(step),data,maps,routes,charts,events)
                count=2+holds.get(step,0)
                if step in holds:
                    chapters.append({"control_state":step,"video_time_s":outframe/FPS,"duration_s":count/FPS,
                                     "title":note_for(step,data['frames'][step],events,total)[0]})
                raw=image.tobytes()
                for _ in range(count):process.stdin.write(raw)
                outframe+=count
                if step%100==0:print(json.dumps({"control_state":step,"encoded_frames":outframe}),flush=True)
        finally:
            process.stdin.close()
        if process.wait():raise RuntimeError("Video encoder failed")
        duration=outframe/FPS
        embed_chapters(args.output,chapters,duration)
        manifest={"video":args.output.name,"width":W,"height":H,"fps":FPS,"frames":outframe,"duration_s":duration,
                  "control_states":total+1,"policy_actions":total,"playback":"Each saved control state 0.1 s; 0.5x action replay; API waits omitted; chapter holds added",
                  "source_video":mapping["source_video"],"timeline":str(args.timeline),"mapping":str(args.mapping),
                  "timeline_sha256":sha(args.timeline),"mapping_sha256":sha(args.mapping),"renderer_sha256":sha(Path(__file__)),
                  "video_sha256":sha(args.output),"size_bytes":args.output.stat().st_size,"chapters":chapters,
                  "embedded_chapters":True,"audit":report,"new_model_calls":0,"new_simulator_steps":0,
                  "keypose_markers_are_not_planner_paths":True,"relative_goals_revealed_only_at_stage_entry":True,
                  "all_target_quaternions_inherited":report["explicit_target_quaternions"]==0,
                  "terminal_semantics":"Raw task success while second pot is held; post-release diagnostic is excluded"}
        (ROOT/"video_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({"video":str(args.output),"duration_s":duration,"frames":outframe}),flush=True)
    finally:
        reader.close()


if __name__=="__main__":
    main()
