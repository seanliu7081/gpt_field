"""Recorded field search with retained history and observable execution feedback.

This recorder only renders supplied observations, field predictions, local plans,
history metadata and measured transitions. It does not start a simulator or make
model requests. The original field and direct-history recorders remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy

from PIL import Image, ImageDraw

from .schema import GridSpec
from .video import (
    BG, CYAN, GREEN, HEIGHT, MAGENTA, MUTED, PANEL, RED, TEXT, WIDTH,
    YELLOW, EpisodeVideoRecorder, _font, _short, _vector,
)


class FieldHistoryVideoRecorder(EpisodeVideoRecorder):
    """Show actual fields, replans and execution feedback as separate events.

    ``offline_smoke`` is the runner-compatible alias of ``offline_mock``. Both
    mark every frame as a scripted diagnostic, including reset and terminal.
    Pass post-feedback ``history_stats`` to ``executed`` for a current snapshot;
    otherwise the displayed counts are explicitly labeled as the decision's.
    """

    def __init__(self, path, task=None, *, grid=None, episode_id="field_history",
                 fps=20, offline_mock=False, offline_smoke=None):
        if offline_smoke is not None and offline_mock and not offline_smoke:
            raise ValueError("offline_mock and offline_smoke labels conflict")
        self.offline_mock = bool(offline_mock if offline_smoke is None else offline_smoke)
        self.offline_smoke = self.offline_mock
        self.task = task
        self.history_stats = None
        self.history_stats_scope = "not_recorded"
        self.feedback = None
        super().__init__(path, grid=grid if grid is not None else GridSpec(),
                         episode_id=episode_id, fps=fps)

    def observe(self, observation, *, step=0, history_stats=None):
        self.history_stats = deepcopy(history_stats)
        self.history_stats_scope = "observation" if history_stats is not None else "not_recorded"
        self.feedback = None
        super().observe(observation, step=step)

    def decision(self, observation, decision, *, decision_latency_s=None):
        self.history_stats = deepcopy(decision.diagnostics.get("history_stats"))
        self.history_stats_scope = "decision" if self.history_stats is not None else "not_recorded"
        self.feedback = deepcopy(decision.diagnostics.get("feedback"))
        # The base emits a field-only refresh before rendering any plan/action.
        super().decision(observation, decision, decision_latency_s=decision_latency_s)

    def executed(self, observation, *, step, success, sim_status=None,
                 feedback=None, history_stats=None):
        self.feedback = deepcopy(feedback)
        if history_stats is not None:
            self.history_stats = deepcopy(history_stats)
            self.history_stats_scope = "after_feedback"
        super().executed(observation, step=step, success=success, sim_status=sim_status)

    def _emit(self, event, repeats, **extra):
        metadata = {
            "policy": "field_history", "offline_mock": self.offline_mock,
            "offline_smoke": self.offline_smoke,
            "history_stats": deepcopy(self.history_stats),
            "history_stats_scope": self.history_stats_scope,
            "feedback": deepcopy(self.feedback),
            "feedback_step": self.feedback.get("step") if self.feedback else None,
            "measured_delta_xyz_m": list(self._measured_delta) if self._measured_delta is not None else None,
            "field_center": list(self.field.center) if self.field is not None else None,
            "candidate_count": self.details.get("plan", {}).get("candidate_count"),
            "valid_candidate_count": self.details.get("plan", {}).get("valid_count"),
        }
        if event == "field_refresh":
            metadata["field_prediction"] = deepcopy(self.details.get("field_prediction"))
        super()._emit(event, repeats, **metadata, **extra)

    def render_frame(self):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        title = ("SCRIPTED SMOKE / LIBERO  |  FIELD + HISTORY  |  NO LIVE MODEL CALLS"
                 if self.offline_mock else "ASTRA / LIBERO  |  FIELD + HISTORY + EXECUTION FEEDBACK")
        draw.text((24, 17), title, font=_font(25, True), fill=YELLOW if self.offline_mock else TEXT)
        draw.text((24, 57), _short(self.episode_id, 110), font=_font(18), fill=MUTED)
        if self.observation is None:
            draw.text((24, 111), _short(self.task or "No observation recorded", 130), font=_font(21), fill=MUTED)
            self._footer(draw)
            return canvas
        draw.text((24, 94), _short(self.observation.prompt, 135), font=_font(21), fill=TEXT)
        self._cameras(canvas, draw)
        self._field_panels(canvas, draw)
        self._event_strip(draw)
        self._details_panel(draw)
        self._footer(draw)
        return canvas

    def _footer(self, draw):
        text = ("SCRIPTED SMOKE: scripted fields, actual local plans; not model performance. "
                if self.offline_mock else "Slowed annotated playback. ")
        text += "Simulator 20 Hz. API waits omitted. Feedback uses robot observations; LIBERO evaluates success separately."
        draw.text((24, 932), text, font=_font(13), fill=MUTED)

    def _field_panels(self, canvas, draw):
        # Preserve the original world units, heatmaps, masks and path geometry.
        super()._field_panels(canvas, draw)
        draw.rectangle((425, 138, 1576, 199), fill=BG)
        label = "SCRIPTED FIELD + ACTUAL LOCAL PLANNER SEARCH" if self.offline_mock else "RECORDED FIELD + ACTUAL LOCAL PLANNER SEARCH"
        draw.text((425, 142), label, font=_font(20, True), fill=TEXT)
        history = self.history_stats
        if history is None:
            line = "History metadata not yet recorded. Camera views and robot state are the actual observation."
        else:
            scope = {"observation": "at observation", "decision": "at decision",
                     "after_feedback": "after feedback"}[self.history_stats_scope]
            number = lambda key: history[key] if key in history else "not recorded"
            line = (f"History {scope}: {number('observations')} observations | "
                    f"{number('feedback_records')} feedback reports | {number('retained_image_pairs')} image pairs | "
                    f"{number('messages')} records")
        draw.text((425, 174), _short(line, 150), font=_font(15), fill=MUTED)
        if self.field is None and self.offline_mock:
            draw.rectangle((444, 237, 1560, 323), fill=PANEL)
            draw.text((450, 245), "No scripted field recorded for this event", font=_font(25), fill=MUTED)
            draw.text((450, 290), "Field panels appear only after a scripted response is recorded.", font=_font(19), fill=MUTED)
        # Projection semantics are also present when history uses the top line.
        if self.field is not None:
            draw.rectangle((425, 619, 1580, 642), fill=BG)
            draw.text((425, 619), "Nearest grid slices; paths projected in world units. Lower potential: teal. Higher: orange. Purple: blocked; gray: unknown.",
                      font=_font(14), fill=MUTED)

    def _event_strip(self, draw):
        labels = ("1 OBSERVE", "2 SCRIPTED FIELD" if self.offline_mock else "2 ASTRA FIELD",
                  "3 LOCAL REPLAN", "4 EXECUTE / FEEDBACK")
        active = {"observation": 0, "field_refresh": 1, "local_replan": 2, "executed": 3}.get(self.event)
        for index, label in enumerate(labels):
            x = 425 + index * 291
            draw.rounded_rectangle((x, 654, x + 277, 695), radius=8,
                                   fill=(32, 91, 90) if index == active else PANEL,
                                   outline=GREEN if index == active else (44, 56, 74), width=2)
            draw.text((x + 11, 666), label, font=_font(14, True), fill=TEXT if index == active else MUTED)
        for x, color, label in ((425, CYAN, "valid candidates"), (638, YELLOW, "selected horizon"),
                                (862, MAGENTA, "measured trace"), (1077, TEXT, "current EEF / square: first move")):
            draw.line((x, 719, x + 22, 719), fill=color, width=3)
            draw.text((x + 30, 709), label, font=_font(14), fill=MUTED)

    def _details_panel(self, draw):
        draw.rounded_rectangle((425, 742, 1576, 916), radius=12, fill=PANEL)
        label = self.status if self.event == "terminal" else self.event
        color = GREEN if self.success else RED if self.event == "terminal" else CYAN
        draw.text((442, 753), f"STEP {self.step:04d}  |  {label.upper().replace('_', ' ')}",
                  font=_font(19, True), fill=color)
        latency = []
        if self.last_api_latency_s is not None:
            latency.append(f"{'scripted response' if self.offline_mock else 'last API'} {self.last_api_latency_s:.2f}s")
        plan_latency = self.details.get("plan_latency_s")
        if plan_latency is not None:
            latency.append(f"plan {plan_latency:.3f}s")
        draw.text((1030, 757), " | ".join(latency) or "No response latency recorded", font=_font(14), fill=MUTED)
        plan = self.details.get("plan", {})
        draw.text((442, 782), "EEF world m: " + _vector(self.observation.state[:3], 4), font=_font(15), fill=TEXT)
        draw.text((1030, 782), f"Search: {plan.get('valid_count', 0)}/{plan.get('candidate_count', 0)} valid; selected {plan.get('selected_index', '-')}",
                  font=_font(15), fill=TEXT)
        draw.text((442, 808), "Action 7D: " + (_vector(self.action, 2) if self.action else "not issued"), font=_font(15), fill=TEXT)
        prediction = self.field.prediction if self.field else None
        auxiliary = f"Field gripper: {prediction.gripper_command}; motion: {prediction.motion_mode}" if prediction else "No field auxiliary commands recorded"
        draw.text((1030, 808), auxiliary, font=_font(15), fill=TEXT)
        line = ("Executed delta m: " + _vector(self._measured_delta, 4) if self._measured_delta is not None else
                "First planned delta m: " + (_vector(plan["delta_xyz"], 4) if "delta_xyz" in plan else "not available"))
        draw.text((442, 834), line, font=_font(15), fill=MAGENTA if self._measured_delta is not None else YELLOW)
        draw.text((1030, 834), "Planner: " + _short(self.details.get("status", "not yet run"), 49), font=_font(14), fill=TEXT)
        feedback = self.feedback
        if feedback is None:
            feedback_line = "Execution feedback not yet recorded. No task-success signal is supplied to the model."
        else:
            vector = lambda key: _vector(feedback[key], 4) if key in feedback else "not recorded"
            feedback_line = (f"Latest feedback, executed step {feedback.get('step', 'not recorded')}: "
                             f"desired {vector('desired_translation_delta_m')} m; "
                             f"measured {vector('measured_translation_delta_m')} m")
        draw.text((442, 863), _short(feedback_line, 145), font=_font(14), fill=MAGENTA if feedback else MUTED)
        if self._error:
            bottom = "Error: " + _short(self._error, 139)
        elif self.event == "terminal":
            bottom = f"Evaluator outcome: {self.status}. Success is checked separately from policy feedback."
        elif feedback:
            progress = feedback.get("rolling_progress", {})
            error = feedback.get("translation_tracking_error_m")
            tracking = f"{error:.5f} m" if error is not None else "not recorded"
            joints = _vector(feedback["gripper_joint_delta"], 4) if "gripper_joint_delta" in feedback else "not recorded"
            bottom = (f"Tracking error: {tracking}; finger joint delta: {joints}. "
                      f"Window {progress.get('window_size', '?')}: {progress.get('low_motion_translation_steps', '?')} low-motion / "
                      f"{progress.get('translation_command_steps', '?')} translation commands.")
        else:
            bottom = "Feedback contains robot measurements only; evaluator success is not model input."
        draw.text((442, 891), _short(bottom, 149), font=_font(13), fill=RED if self._error else MUTED)
