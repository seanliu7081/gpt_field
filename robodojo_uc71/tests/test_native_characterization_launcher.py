"""CPU-only launcher boundary tests. All diagnostic scope fixtures use tmp_path."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_native_characterization as child
from robodojo_uc71.native_characterization import authorization_template
from test_native_characterization import setup


def test_guards_allow_nested_native_chain_only_in_fixed_scope():
    calls = []
    env = NS()
    env.step = lambda: calls.append("step")
    env.take_action_batch = lambda: env.step()
    env.take_action = lambda: env.take_action_batch()
    env.run_eval = lambda: calls.append("forbidden_loop")
    names, permitted = child.install_characterization_guards(env)
    assert set(names) == {"step", "take_action", "take_action_batch", "run_eval"}
    for name in names:
        with pytest.raises(RuntimeError): getattr(env, name)()
    with permitted():
        env.take_action()
        with pytest.raises(RuntimeError, match="loops"): env.run_eval()
        with pytest.raises(RuntimeError, match="Nested"):
            with permitted(): pass
    assert calls == ["step"]
    with pytest.raises(RuntimeError): env.take_action()


def test_fixed_helper_authorization_failure_restores_closed_guard(tmp_path, monkeypatch):
    env, plan, helper, path = setup(tmp_path, monkeypatch)
    env.take_action_batch = lambda: None
    env.step = lambda: None
    _, permitted = child.install_characterization_guards(env)
    with permitted(): helper.dispatch_next()
    scope = json.loads(path.read_text()); scope["diagnostic_native_actions_authorized"] = False
    path.write_text(json.dumps(scope))
    with pytest.raises(Exception, match="authorization"):
        with permitted(): helper.dispatch_next()
    with pytest.raises(RuntimeError, match="fixed-plan"): env.take_action(plan["actions"][1]["action"])
    assert env.take_action_cnt == [1] and env.sim._sim_step_counter == 10
    with pytest.raises(Exception, match="no_retry"):
        with permitted(): helper.dispatch_next()


def scope_fixture(tmp_path):
    binding = {"gpu_uuid": "GPU-synthetic-only", "pci_bus_id": "00000000:01:00.0", "kit_index": 0, "cuda_ordinal": 0}
    plan = {"binding": binding}
    scope = authorization_template(plan)
    scope.update(diagnostic_native_actions_authorized=True, user_message="Synthetic scope only",
        authorization_basis="Test fixture", shutdown_mode="stock-fast", maximum_wall_time_s=900,
        pci_bus_id=binding["pci_bus_id"], attempt_output=str(tmp_path / "attempt"),
        rgb_checkpoint_action_indices=[0,5,14,23,32,37])
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope))
    return binding, plan, scope, path


def test_scope_is_diagnostic_only_and_child_requires_supervisor_flag(tmp_path, monkeypatch):
    binding, plan, scope, path = scope_fixture(tmp_path)
    kwargs = dict(output_parent=tmp_path / "attempt")
    result = child.check_authorization(path, plan, binding, launch=False, **kwargs)
    assert result["maximum_native_actions"] == 37 and result["full_test_authorized"] is False
    with pytest.raises(ValueError, match="supervisor"):
        child.check_authorization(path, plan, binding, launch=True, **kwargs)
    monkeypatch.setenv("UC71_NATIVE_CHARACTERIZATION_AUTHORIZED_CHILD", "1")
    assert child.check_authorization(path, plan, binding, launch=True, **kwargs) == result


@pytest.mark.parametrize("key,value", [
    ("diagnostic_native_actions_authorized", False), ("full_test_authorized", True),
    ("model_calls_authorized", True), ("automatic_retry", True), ("maximum_native_actions", 38),
    ("maximum_wall_time_s", 901), ("shutdown_mode", "full"),
    ("rgb_checkpoint_action_indices", [0,37]), ("pci_bus_id", "00000000:02:00.0"),
    ("attempt_output", "/tmp/different-synthetic-attempt"),
])
def test_scope_rejects_widened_or_different_attempt(tmp_path, key, value):
    binding, plan, scope, path = scope_fixture(tmp_path)
    scope[key] = value; path.write_text(json.dumps(scope))
    with pytest.raises((ValueError, RuntimeError)):
        child.check_authorization(path, plan, binding, launch=False, output_parent=tmp_path / "attempt")


def capture_fixture(monkeypatch):
    env = NS(sim=NS(sim=NS(current_time=1.2), _sim_step_counter=300), take_action_cnt=[0])
    order = []
    pixels = np.arange(480*640*3, dtype=np.uint8).reshape(480,640,3)
    def render(): order.append("render")
    def observation():
        order.append("rgb")
        return [{"vision": {name: {"color": pixels} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                 "evaluator_only_secret": {"reward": 99}}]
    env.obs_manager = NS(render_for_capture=render, get_obs=observation)
    def camera(_):
        assert order[-1] == "render"; order.append("camera")
        return {"native_physics_time_s": 1.2, "scope": "setup_evaluator_only"}
    def robot(_):
        assert order[-1] == "camera"; order.append("robot")
        return {"native_physics_time_s": 1.2, "arms": {}}
    monkeypatch.setattr(child, "capture_camera_inventory", camera)
    monkeypatch.setattr(child, "capture_characterization_state", robot)
    return env, order


def test_render_sync_then_robot_camera_and_exact_clean_rgb(tmp_path, monkeypatch):
    env, order = capture_fixture(monkeypatch)
    capture = child.capture_stage(env, {"native_sim_step_calls": 300}, tmp_path, "before", rgb=True)
    assert order == ["render", "camera", "robot", "rgb"]
    assert len(capture["images"]) == 3 and capture["native_action_counter"] == [0]
    assert capture["physics_unchanged_during_capture"] is True
    for row in capture["images"]:
        assert child.digest(tmp_path / row["path"]) == row["sha256"]
    assert "evaluator_only_secret" not in json.dumps(capture)


@pytest.mark.parametrize("change", ["physics_time", "physics_counter", "action_counter", "step_calls"])
def test_capture_rejects_any_physics_or_action_advance(tmp_path, monkeypatch, change):
    env, order = capture_fixture(monkeypatch)
    counts = {"native_sim_step_calls": 300}
    def render():
        order.append("render")
        if change == "physics_time": env.sim.sim.current_time += .004
        if change == "physics_counter": env.sim._sim_step_counter += 1
        if change == "action_counter": env.take_action_cnt[0] += 1
        if change == "step_calls": counts["native_sim_step_calls"] += 1
    env.obs_manager.render_for_capture = render
    with pytest.raises(RuntimeError, match="advanced physics"):
        child.capture_stage(env, counts, tmp_path, "after", rgb=False)


def test_program_binding_has_exact_transitive_helpers_and_detects_changes(tmp_path, monkeypatch):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("first"); second.write_text("second")
    monkeypatch.setattr(child, "program_files", lambda: [first, second])
    binding = {str(p): child.digest(p) for p in (first, second)}
    path = tmp_path / "binding.json"; path.write_text(json.dumps(binding))
    assert child.verify_program_binding(path) == binding
    first.write_text("changed")
    with pytest.raises(ValueError, match="changed"): child.verify_program_binding(path)
    binding.pop(str(first)); path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="incomplete"): child.verify_program_binding(path)
