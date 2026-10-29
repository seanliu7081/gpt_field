"""Controlled synthetic fixtures. Oracle values test machinery, never Astra ability."""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .baselines import transform_field, waypoint_field
from .field import GridField
from .planner import LocalPlanner, PlannerConfig
from .schema import FieldPrediction, GridSpec


class ReachingScene:
    spacing = 0.01
    lower = -0.30
    size = 61

    def __init__(self, name: str):
        if name not in ("diagonal", "detour"):
            raise ValueError("unknown synthetic scene")
        self.name = name
        self.start = np.array([-0.12, -0.08 if name == "diagonal" else 0.0, 0.5])
        self.target = np.array([0.12, 0.12 if name == "diagonal" else 0.0, 0.5])
        self.axis = self.lower + np.arange(self.size)*self.spacing
        x, y = np.meshgrid(self.axis, self.axis, indexing="ij")
        self.blocked = np.zeros((self.size, self.size), dtype=bool)
        if name == "detour":
            self.blocked = (np.abs(x) <= 0.021) & (y >= -0.131) & (y <= 0.051)
        self.distances = self._solve()

    def _index(self, point):
        return tuple(int(round((float(v)-self.lower)/self.spacing)) for v in point[:2])

    def _solve(self):
        distance = np.full((self.size, self.size), np.inf)
        goal = self._index(self.target)
        distance[goal] = 0
        queue = [(0.0, *goal)]
        while queue:
            value, i, j = heapq.heappop(queue)
            if value != distance[i, j]:
                continue
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                a, b = i+di, j+dj
                if not (0 <= a < self.size and 0 <= b < self.size) or self.blocked[a,b]:
                    continue
                if di and dj and (self.blocked[i,b] or self.blocked[a,j]):
                    continue
                candidate = value + self.spacing*math.hypot(di,dj)
                if candidate < distance[a,b]:
                    distance[a,b] = candidate
                    heapq.heappush(queue, (candidate,a,b))
        return distance

    def field(self, center, grid: GridSpec):
        values, blocked, unknown = [], [], []
        for n, point in enumerate(grid.coordinates(center)):
            i, j = self._index(point)
            if not (0 <= i < self.size and 0 <= j < self.size):
                unknown.append(n)
                values.append(0.0)
            elif self.blocked[i,j]:
                blocked.append(n)
                values.append(0.0)
            elif not math.isfinite(self.distances[i,j]):
                unknown.append(n)
                values.append(0.0)
            else:
                values.append(float(self.distances[i,j]))
        prediction = FieldPrediction.from_dict({
            "stage": "synthetic_fixed_orientation_reaching", "orientation_target_xyzw": [0,0,0,1],
            "gripper_command": "hold", "motion_mode": "move", "values": values,
            "blocked_indices": blocked, "unknown_indices": unknown,
        }, grid)
        return GridField(grid, tuple(center), prediction)

    def segment_is_free(self, a, b):
        for t in np.linspace(0,1,21):
            i,j = self._index(a+t*(b-a))
            if not (0 <= i < self.size and 0 <= j < self.size) or self.blocked[i,j]:
                return False
        return True

    def render(self, path, label: str):
        image = Image.new("RGB", (440, 470), "white")
        draw = ImageDraw.Draw(image)
        def xy(point):
            return (int(20+(point[0]+0.30)/0.60*400), int(420-(point[1]+0.30)/0.60*400))
        draw.rectangle((20,20,420,420), outline="#888888")
        for i,j in np.argwhere(self.blocked):
            px,py = xy((self.axis[i],self.axis[j]))
            draw.rectangle((px-4,py-4,px+4,py+4), fill="#666666")
        if len(path)>1:
            draw.line([xy(p) for p in path], fill="#2266bb", width=3)
        for point,color in ((self.start,"#8888ff"),(self.target,"#22aa55"),(path[-1],"#ee8822")):
            px,py = xy(point)
            draw.ellipse((px-5,py-5,px+5,py+5), fill=color)
        draw.text((20,435), label, fill="black")
        draw.text((20,452), "Synthetic oracle fixture; no Astra predictions", fill="black")
        return image


def run_demo(output: str | Path, *, seed: int = 7, max_steps: int = 80,
             methods=("field", "field_one", "affine", "shuffled", "minimum", "waypoint")) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    grid = GridSpec()
    records = []
    for scene_name in ("diagonal", "detour"):
        scene = ReachingScene(scene_name)
        for method in methods:
            config = PlannerConfig(horizon=1 if method == "field_one" else 3,
                workspace_min=(-0.29,-0.29,0.499999), workspace_max=(0.29,0.29,0.500001))
            planner = LocalPlanner(config)
            position = scene.start.copy()
            path, logs, frames = [position.copy()], [], []
            status, field = "timeout", None
            for step in range(max_steps):
                if np.linalg.norm(position-scene.target) <= 0.011:
                    status = "success"
                    break
                if step % 5 == 0:
                    original = scene.field(position,grid)
                    if method == "waypoint":
                        field = waypoint_field(grid,position,original.prediction.to_dict(),scene.target)
                    else:
                        field = transform_field(original,"field" if method == "field_one" else method,seed+step)
                plan = planner.plan(field,position)
                next_position = position + np.asarray(plan.delta_xyz)
                # Ground-truth collision here is evaluator-only synthetic scoring, never a repair.
                if not scene.segment_is_free(position,next_position):
                    status = "collision"
                    break
                logs.append({"step": step, "position": position.tolist(),
                    "field_center": list(field.center), "status": plan.status,
                    "delta_xyz": list(plan.delta_xyz), "selected_index": plan.selected_index,
                    "predicted_improvement": plan.predicted_improvement})
                position = next_position
                path.append(position.copy())
                frames.append(scene.render(path,f"{scene_name} | {method} | step {step+1}"))
            if np.linalg.norm(position-scene.target) <= 0.011:
                status = "success"
            stem = f"{scene_name}_{method}"
            final = scene.render(path,f"{scene_name} | {method} | {status}")
            final.save(output / f"{stem}.png")
            if frames:
                frames[0].save(output / f"{stem}.gif",save_all=True,append_images=frames[1:]+[final],duration=90,loop=0)
            (output / f"{stem}.json").write_text(json.dumps({"source":"analytic_oracle", "steps":logs,
                "path": [p.tolist() for p in path]},indent=2)+"\n")
            records.append({"suite":"synthetic_reaching", "seed":seed,"task_id":scene_name,"episode_index":0,
                "method":method,"status":status,"steps":len(path)-1,"source":"analytic_oracle",
                "path_length_m":float(sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:]))),
                "final_error_m":float(np.linalg.norm(position-scene.target)),"protocol_hash":"synthetic_v1"})
    (output / "episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    result = {"source":"analytic_oracle", "claim":"Software diagnostics only; these results do not measure Astra.",
              "episodes":records}
    (output / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def compare_cached_field(field: GridField, position, seed: int = 7) -> dict:
    result = {}
    for method in ("field", "field_one", "affine", "shuffled", "minimum"):
        candidate = transform_field(field,"field" if method == "field_one" else method,seed)
        plan = LocalPlanner(PlannerConfig(horizon=1 if method == "field_one" else 3)).plan(candidate,position)
        result[method] = plan.to_dict()
    return result

