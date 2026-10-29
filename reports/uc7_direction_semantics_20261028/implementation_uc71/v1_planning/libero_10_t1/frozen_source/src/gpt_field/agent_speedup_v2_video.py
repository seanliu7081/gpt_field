"""Annotated agent_SpeedUp_v2 playback: relation plan, plan steps, observers, parts and pointing.

A runner-side tool (contract §C1.4, §C13): it subclasses the v1 recorder (spec §8.1 allows the video module to
import v1 drawing helpers), is imported only by scripts/evaluate_agent_speedup_v2.py and never by a control or
model-interface module, and is in neither lint list. It renders only recorded observations, the issued action and
the policy's own diagnostics (contract §C18.2): ``relation``, ``observers``, ``parts_uv``, ``holding``,
``pending_need``, ``api_calls``, ``pointing``, ``outcome`` and the act's ``v2_events`` (the plan from
``relation_plan``, step status from ``step_start`` / ``step_end``). Both camera panels show the UPRIGHT images the
executor and the model use (the legacy policy image mirrored left-right), so part and pointing markers are drawn
at their upright pixels. Evaluator success is displayed but never fed back to the policy. A drawing error in a v2
panel is shown in that panel and counted in the event index; it never ends an episode (video is diagnosis-only).
"""

from __future__ import annotations

from collections import deque
import json
import math

from PIL import Image, ImageDraw

from .agent_speedup_v1_video import AgentSpeedUpV1VideoRecorder, json_safe
from .observation import decode_image
from .video import BG, CYAN, GREEN, HEIGHT, MAGENTA, MUTED, PANEL, RED, TEXT, WIDTH, YELLOW, _font, _short

AGENT_V2_VIDEO_VERSION = "agent-speedup-v2-video-1"
PART_COLOURS = {"A": YELLOW, "B": MAGENTA, "C": CYAN, "D": GREEN, "E": (255, 160, 60), "F": (180, 140, 255),
                "G": (120, 220, 255), "H": (255, 120, 170)}
STATUS_COLOURS = {"done": GREEN, "stays": GREEN, "succeeded": GREEN, "failed": RED, "rejected": RED,
                  "skipped": MUTED, "active": YELLOW, "pending": TEXT}
VERDICT_COLOURS = {"met": GREEN, "violated": RED, "pending": MUTED, "progress": CYAN}
RECENT_EVENTS = 4
PLAN_ROWS = 4
OBSERVER_ITEMS = 6
QUIET_EVENTS = ("observer",)          # frequent; counted, not listed among the recent events
CAMERA_PANELS = (("external", 139, "AGENT VIEW (external)"), ("wrist", 535, "WRIST VIEW"))


def _finite_uv(value):
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        return float(value[0]), float(value[1])
    return None


def _scalar_text(value, limit=40):
    if isinstance(value, float):
        return f"{value:.3g}"
    return _short(value, limit)


def plan_rows(plan, status):
    """[(i, text, status, reason)] for the plan's steps (plan = the relation_plan event's wire JSON)."""
    rows = []
    for i, step in enumerate((plan or {}).get("steps") or []):
        if not isinstance(step, dict):
            continue
        grip = ",".join(str(p) for p in step.get("grip") or [])
        text = f"{i}  {step.get('do')} {step.get('obj')}"
        if grip:
            text += f"  grip [{grip}]"
        if step.get("ref"):
            text += f"  ref {step.get('ref')}"
        until = [f"{item.get('o')}({item.get('p')})" if item.get("p") else str(item.get("o"))
                 for item in step.get("until") or [] if isinstance(item, dict)]
        if until:
            text += "  until " + " or ".join(until)
        keep = [f"{item.get('c')}({item.get('p')})" for item in step.get("keep") or [] if isinstance(item, dict)]
        if keep:
            text += "  keep " + ", ".join(keep)
        state, reason = status.get(i, ("pending", None))
        rows.append((i, text, state, reason))
    return rows


class AgentSpeedUpV2VideoRecorder(AgentSpeedUpV1VideoRecorder):
    """One frame sequence per executed step (the v1 recorder's protocol and constructor), with v2 panels."""

    def __init__(self, path, *, episode_id="agent_speedup_v2", fps=20, offline_smoke=False, grid=None):
        self.plan = None
        self.plan_status: dict[int, tuple] = {}
        self.relation = None
        self.observers = {}
        self.parts_uv = {}
        self.holding = None
        self.pending_need = None
        self.gap_mm = None
        self.model_calls = 0
        self.v2_event_counts: dict[str, int] = {}
        self.recent_events = deque(maxlen=RECENT_EVENTS)
        self.render_errors = 0
        self.last_render_error = None
        self._image_cache: dict[str, tuple] = {}
        super().__init__(path, episode_id=episode_id, fps=fps, offline_smoke=offline_smoke, grid=grid)

    # state --------------------------------------------------------------------------------------------------
    def absorb(self, diagnostics):
        """Update the v2 state from one act's diagnostics (before the frame is rendered)."""
        details = json_safe(dict(diagnostics or {}))
        relation = details.get("relation")
        self.relation = relation if isinstance(relation, dict) else None
        observers = details.get("observers")
        self.observers = observers if isinstance(observers, dict) else {}
        parts = details.get("parts_uv")
        self.parts_uv = parts if isinstance(parts, dict) else {}
        self.holding = details.get("holding")
        self.pending_need = details.get("pending_need")
        self.gap_mm = details.get("finger_gap_mm")
        self.model_calls += len(details.get("api_calls") or details.get("api_calls_in_act") or [])
        for row in details.get("v2_events") or []:
            if not isinstance(row, dict) or not row.get("event"):
                continue
            name = str(row["event"])
            self.v2_event_counts[name] = self.v2_event_counts.get(name, 0) + 1
            if name == "relation_plan" and isinstance(row.get("plan"), dict):
                self.plan, self.plan_status = row["plan"], {}
            elif name == "step_start" and isinstance(row.get("i"), int):
                self.plan_status[row["i"]] = ("active", f"attempt {row.get('attempt')}")
            elif name == "step_end" and isinstance(row.get("i"), int):
                self.plan_status[row["i"]] = (str(row.get("status")), row.get("reason"))
            if name not in QUIET_EVENTS:
                self.recent_events.append(self._event_label(name, row))

    @staticmethod
    def _event_label(name, row):
        bits = [f"{row.get('policy_step', '-')}", name]
        for key in ("i", "do", "obj", "status", "reason", "verdict", "key", "ok", "kind", "pid", "class", "action"):
            value = row.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                bits.append(f"{key}={_scalar_text(value, 24)}")
        return " ".join(bits)

    def decision(self, observation, decision, *, decision_latency_s=None):
        self.absorb(decision.diagnostics)
        super().decision(observation, decision, decision_latency_s=decision_latency_s)

    def _emit(self, event, repeats, **extra):
        v2 = {"policy": "agent_speedup_v2", "video_version": AGENT_V2_VIDEO_VERSION,
              "relation": self.relation, "holding": self.holding, "pending_need": self.pending_need,
              "plan_status": {str(i): list(value) for i, value in self.plan_status.items()},
              "observers": self.observers, "parts_uv": self.parts_uv, "model_calls": self.model_calls,
              "render_errors": self.render_errors}
        super()._emit(event, repeats, **{**v2, **extra})

    # rendering ----------------------------------------------------------------------------------------------
    def _guarded(self, panel, box, draw, render):
        try:
            render()
        except Exception as exc:          # a drawing bug never ends the episode; it is shown and counted
            self.render_errors += 1
            self.last_render_error = f"{panel}: {type(exc).__name__}: {exc}"
            if box is not None:
                draw.rounded_rectangle(box, radius=12, fill=PANEL)
                draw.text((box[0] + 16, box[1] + 12), _short(f"render error in {panel}: {exc}", 110),
                          font=_font(15), fill=RED)

    def render_frame(self):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        title = ("SCRIPTED SMOKE / LIBERO  |  AGENT SPEEDUP V2  |  NO LIVE ASTRA CALLS" if self.offline_smoke
                 else "ASTRA / LIBERO  |  AGENT SPEEDUP V2  |  RELATION PLAN + LOCAL EXECUTOR")
        draw.text((24, 17), title, font=_font(26, True), fill=YELLOW if self.offline_smoke else TEXT)
        draw.text((24, 57), _short(self.episode_id, 110), font=_font(18), fill=MUTED)
        if self.observation is None:
            draw.text((24, 125), "No observation recorded", font=_font(25), fill=MUTED)
            return canvas
        draw.text((24, 94), _short(self.observation.prompt, 135), font=_font(21), fill=TEXT)
        self._guarded("cameras", None, draw, lambda: self._cameras(canvas, draw))
        self._guarded("plan", (425, 139, 1576, 427), draw, lambda: self._chunk_panel(draw))
        self._trace_panel(draw, (425, 443, 974, 651), (0, 1))
        self._trace_panel(draw, (994, 443, 1576, 651), (0, 2))
        self._event_strip_direct(draw)
        self._guarded("feedback", (425, 729, 1576, 916), draw, lambda: self._feedback_panel(draw))
        footer = "Integration diagnostic only; scripted answers are not model performance. " if self.offline_smoke else ""
        footer += "Annotated playback; 20 Hz; API waits omitted. Upright images; letters: parts; rings: pointing."
        draw.text((24, 932), _short(footer, 210), font=_font(14), fill=MUTED)
        return canvas

    def _markers(self, camera):
        marks = []
        for pid, item in sorted(self.parts_uv.items()):
            uv = _finite_uv((item or {}).get(camera)) if isinstance(item, dict) else None
            if uv is not None:
                marks.append(("part", str(pid), uv, PART_COLOURS.get(str(pid), TEXT)))
        for result in self.pointing:
            if not isinstance(result, dict) or result.get("camera", "wrist") != camera:
                continue
            uv = _finite_uv(result.get("uv_upright"))
            if uv is not None:
                marks.append(("point", str(result.get("pid") or "?"), uv, YELLOW))
        return marks

    def _panel_image(self, camera, url):
        """(size, resized upright image, ratio); each observation is drawn twice (its executed frame and the next
        decision frame), so the last decoded image per camera is reused."""
        cached = self._image_cache.get(camera)
        if cached is not None and cached[0] is url:
            return cached[1]
        image = decode_image(url).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        width, height = image.size
        ratio = min(352 / width, 336 / height)
        shown = image.resize((round(width * ratio), round(height * ratio)),
                             Image.Resampling.LANCZOS if ratio < 1 else Image.Resampling.NEAREST)
        value = ((width, height), shown, ratio)
        self._image_cache[camera] = (url, value)
        return value

    def _cameras(self, canvas, draw):
        for camera, y, label in CAMERA_PANELS:
            draw.rounded_rectangle((24, y, 401, y + 381), radius=12, fill=PANEL)
            url = self.observation.image if camera == "external" else self.observation.wrist_image
            (width, height), shown, ratio = self._panel_image(camera, url)
            draw.text((40, y + 6), label, font=_font(16, True), fill=TEXT)
            draw.text((40, y + 26), f"upright (executor / model frame)  |  RGB {width} x {height}",
                      font=_font(12), fill=MUTED)
            x0, y0 = 36 + (352 - shown.width) // 2, y + 42 + (336 - shown.height) // 2
            canvas.paste(shown, (x0, y0))
            for kind, pid, (u, v), colour in self._markers(camera):
                if not (0 <= u < width and 0 <= v < height):
                    continue
                px, py = x0 + (u + .5) * ratio, y0 + (v + .5) * ratio
                if kind == "part":
                    draw.rectangle((px - 4, py - 4, px + 4, py + 4), outline=colour, width=2)
                    draw.text((px + 6, py - 16), pid, font=_font(14, True), fill=colour)
                else:
                    draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=colour, width=2)
                    draw.line((px - 12, py, px + 12, py), fill=colour, width=1)
                    draw.line((px, py - 12, px, py + 12), fill=colour, width=1)

    def _chunk_panel(self, draw):
        draw.rounded_rectangle((425, 139, 1576, 427), radius=12, fill=PANEL)
        relation = self.relation or {}
        label = "SCRIPTED RELATION PLAN" if self.offline_smoke else "ASTRA RELATION PLAN"
        head = (f"{label}  |  phase: {relation.get('phase') or '-'}  |  macro: {self.macro or '-'}  |  "
                f"model calls: {self.model_calls}")
        draw.text((446, 149), _short(head, 105), font=_font(19, True), fill=TEXT)
        rows = plan_rows(self.plan, self.plan_status)
        if not rows:
            scene = None
            routing = self.routing if isinstance(self.routing, dict) else None
            if routing is not None:
                scene = routing.get("situation")
            text = ("Awaiting the relation plan (Calibrate+ runs locally without model calls)" if scene is None
                    else "Plan: " + str(scene))
            draw.text((446, 180), _short(text, 130), font=_font(16), fill=MUTED)
        else:
            scene = (self.plan or {}).get("scene")
            if scene:
                draw.text((446, 178), _short("Scene: " + str(scene), 140), font=_font(14), fill=MUTED)
            for index, (i, text, state, reason) in enumerate(rows[:PLAN_ROWS]):
                colour = STATUS_COLOURS.get(state, TEXT)
                tail = f"  |  {state}" + (f" ({reason})" if reason else "")
                draw.text((446, 198 + 22 * index), _short(text + tail, 128), font=_font(15, state == "active"),
                          fill=colour)
            if len(rows) > PLAN_ROWS:
                draw.text((1480, 198 + 22 * (PLAN_ROWS - 1)), f"+{len(rows) - PLAN_ROWS}", font=_font(14),
                          fill=MUTED)
        executor = f"Executor: {self.primitive or '-'}  |  {self.execution_kind or '-'}"
        if self.macro_step is not None:
            executor += f"  |  macro step {self.macro_step}"
        executor += f"  |  holding {self.holding or '-'}"
        if isinstance(self.gap_mm, (int, float)):
            executor += f"  |  gap {self.gap_mm:.1f} mm"
        if self.pending_need:
            executor += f"  |  waiting: {self.pending_need}"
        draw.text((446, 288), _short(executor, 130), font=_font(15), fill=CYAN)
        outcome = self.last_outcome if isinstance(self.last_outcome, dict) else None
        if outcome is None:
            draw.text((446, 310), "Last outcome: none yet", font=_font(15), fill=MUTED)
        else:
            status = str(outcome.get("status", "?"))
            name = outcome.get("relation") or outcome.get("skill") or "?"
            text = f"Last outcome #{outcome.get('n', '-')}: {name} {status.upper()} ({outcome.get('reason', '-')})"
            if outcome.get("evidence_text"):
                text += "  " + str(outcome["evidence_text"])
            draw.text((446, 310), _short(text, 118), font=_font(15, True), fill=STATUS_COLOURS.get(status, TEXT))
        x = 446
        draw.text((x, 333), "Observers:", font=_font(14, True), fill=TEXT)
        x += 92
        items = sorted(self.observers.items())
        for key, entry in items[:OBSERVER_ITEMS]:
            verdict = (entry or {}).get("verdict") if isinstance(entry, dict) else None
            text = f"{_short(key, 26)}={verdict or '-'}"
            draw.text((x, 333), text, font=_font(13), fill=VERDICT_COLOURS.get(verdict, MUTED))
            x += min(190, 9 * len(text) + 14)
            if x > 1500:
                break
        if not items:
            draw.text((x, 333), "none registered", font=_font(13), fill=MUTED)
        elif len(items) > OBSERVER_ITEMS:
            draw.text((min(x, 1520), 333), f"+{len(items) - OBSERVER_ITEMS}", font=_font(13), fill=MUTED)
        pointing = " -> ".join(self._pointing_line(result) for result in self.pointing) or "none yet"
        calls = ", ".join(map(str, self.api_calls_in_act)) or "none"
        latency = f"{self.last_api_latency_s:.2f} s" if isinstance(self.last_api_latency_s, (int, float)) else "-"
        draw.text((446, 355), _short(f"Pointing (upright px): {pointing}  |  total {self.pointing_count}  |  "
                                     f"calls this act: {calls}  |  last latency {latency}", 145),
                  font=_font(14), fill=TEXT)
        recent = "  ||  ".join(self.recent_events) or "none yet"
        draw.text((446, 377), _short("Events: " + recent, 160), font=_font(13), fill=MUTED)
        history = self.history_stats if isinstance(self.history_stats, dict) else {}
        kinds = history.get("calls_by_kind") if isinstance(history.get("calls_by_kind"), dict) else {}
        totals = ", ".join(f"{key} {value}" for key, value in kinds.items() if value) or "-"
        draw.text((446, 399), _short(f"Requests: {history.get('protocol', 'stateless')} | totals by kind: {totals}"
                                     + (f" | render errors {self.render_errors}" if self.render_errors else ""), 150),
                  font=_font(13), fill=MUTED)

    @staticmethod
    def _pointing_line(result):
        if not isinstance(result, dict):
            return _short(result, 40)
        text = f"{result.get('kind', '?')} {result.get('pid', '?')}@{result.get('camera', '?')}"
        answer = result.get("answer_id")
        text += f" id {answer if answer is not None else 'null'}"
        uv = _finite_uv(result.get("uv_upright"))
        if uv is not None:
            text += f" ({uv[0]:.0f}, {uv[1]:.0f})"
        if result.get("schema_error"):
            text += " schema error"
        return text

    def _event_strip_direct(self, draw):
        labels = ("1 OBSERVE", "2 MODEL CALL", "3 PRIMITIVE ACTION", "4 EXECUTE / FEEDBACK")
        active = {"observation": 0, "skill_decision": 1, "agent_action": 2, "executed": 3}.get(self.event)
        for index, label in enumerate(labels):
            x = 425 + index * 291
            draw.rounded_rectangle((x, 668, x + 277, 711), radius=8,
                                   fill=(32, 91, 90) if index == active else PANEL,
                                   outline=GREEN if index == active else (44, 56, 74), width=2)
            draw.text((x + 10, 682), label, font=_font(14, True), fill=TEXT if index == active else MUTED)

    def summary(self):
        """Recorder-side counts (JSON-safe), for the runner or a test."""
        return json.loads(json.dumps(json_safe({
            "video_version": AGENT_V2_VIDEO_VERSION, "model_calls": self.model_calls,
            "v2_event_counts": dict(self.v2_event_counts), "plan_steps": len(plan_rows(self.plan, {})),
            "render_errors": self.render_errors, "last_render_error": self.last_render_error})))
