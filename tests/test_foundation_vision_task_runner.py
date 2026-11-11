"""Task-wrapper contracts with fake loaders/runners; no model or simulator calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_foundation_vision_task.py"
    spec = importlib.util.spec_from_file_location("evaluate_foundation_vision_task", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def local_models(tmp_path):
    sam = tmp_path / "sam3.1.pt"
    sam.write_bytes(b"test fixture, never loaded")
    da3 = tmp_path / "da3"
    da3.mkdir()
    (da3 / "config.json").write_text("{}")
    (da3 / "model.safetensors").write_bytes(b"test fixture, never loaded")
    return ["--sam-checkpoint", str(sam), "--da3-model", str(da3)]


def fake_runner(main):
    return SimpleNamespace(main=main, execution_options=lambda *a, **k: {
        "reasoning_effort": "high", "billing": "chatgpt_subscription",
        "monitor": {"collision_policy": "record_only"}, "stop_after_steps": None},
        runtime_inventory=lambda *a, **k: {"old_runtime.py": "unchanged"},
        execute_reviewed=lambda *a, **k: pytest.fail("Unexpected execution"))


def test_preflight_preserves_original_cli_and_never_loads_models_or_client(tmp_path, monkeypatch):
    wrapper = module()
    original = ["preflight", "--task", "libero_10_t4", "--episode-index", "0",
                "--task-bddl-override", "/scene.bddl", "--initial-state-override", "/state.npy",
                "--expected-initial-observation-sha256", "a" * 64,
                "--reasoning-effort", "high", "--request-deadline-s", "180", "--stream-progress"]
    runner = fake_runner(lambda argv: argv)
    monkeypatch.setattr(wrapper, "_load_runner", lambda: runner)
    monkeypatch.setattr(wrapper, "_build_vision_engine", lambda _: pytest.fail("Preflight loaded models"))
    monkeypatch.setattr(wrapper, "_live_client_class", lambda: pytest.fail("Preflight imported live client"))
    assert wrapper.main(original + local_models(tmp_path)) == original
    options = runner.execution_options()
    assert options["reasoning_effort"] == "high"
    assert options["billing"] == "chatgpt_subscription"
    assert options["stop_after_steps"] is None
    assert options["foundation_vision"]["mode"] == "vision_conditioned_executive"
    assert options["foundation_vision"]["max_age_s"] == 5.
    assert options["foundation_vision"]["automatic_depth_routing"] is False
    assert options["foundation_vision"]["automatic_action_gate"] is False


def test_preload_failure_precedes_even_loading_task_runner(tmp_path, monkeypatch):
    wrapper = module()
    def fail(_):
        raise RuntimeError("strict local checkpoint load failed")
    monkeypatch.setattr(wrapper, "_build_vision_engine", fail)
    monkeypatch.setattr(wrapper, "_load_runner", lambda: pytest.fail("Reached task runner after preload failure"))
    output = tmp_path / "trial"
    with pytest.raises(RuntimeError, match="strict local checkpoint"):
        wrapper.main(["start", "--output", str(output)] + local_models(tmp_path))
    assert not output.exists()


def test_run_preloads_before_initial_plan_and_injects_single_shared_execution_engine(tmp_path, monkeypatch):
    wrapper = module()
    engine = object()
    events = []
    output = tmp_path / "trial"
    original = ["start", "--task", "libero_10_t4", "--reasoning-effort", "high",
                "--store-dir", "/private/test-store", "--auto-approve", "--output", str(output)]
    def preload(args):
        events.append("preload")
        return engine, {"models": {}, "load_times": {"sam_load_s": 1., "da3_load_s": 2.}}
    class Client:
        def __init__(self, *args, vision_engine, vision_output_dir, vision_extra_targets, **kwargs):
            assert vision_engine is engine
            assert vision_extra_targets == ("can", "handle")
            assert args == ("config",)
            assert kwargs == {"approved_plan": "plan", "lock": "lock", "token_provider": "subscription"}
            assert output.is_dir()
            assert not vision_output_dir.exists()
            vision_output_dir.mkdir()
            events.append("vision_client")
    def execute(actual_output, *, client_factory, **kwargs):
        assert actual_output == output and kwargs == {"reasoning_effort": "high"}
        assert not output.exists()
        output.mkdir()
        client_factory("config", approved_plan="plan", lock="lock", token_provider="subscription")
        return 23
    def main(argv):
        assert argv == original
        assert events == ["preload", "load_runner"]
        events.append("initial_plan")
        assert runner.runtime_inventory() == runner.runtime_inventory()
        return runner.execute_reviewed(output, reasoning_effort="high")
    runner = fake_runner(main)
    runner.execute_reviewed = execute
    def load_runner():
        events.append("load_runner")
        return runner
    monkeypatch.setattr(wrapper, "_build_vision_engine", preload)
    monkeypatch.setattr(wrapper, "_load_runner", load_runner)
    monkeypatch.setattr(wrapper, "_live_client_class", lambda: Client)
    assert wrapper.main(original + local_models(tmp_path)
                        + ["--vision-query", "can", "--vision-query", "handle"]) == 23
    assert runner.execution_options()["foundation_vision"]["extra_target_texts"] == ["can", "handle"]
    assert events == ["preload", "load_runner", "initial_plan", "vision_client"]


def test_inventory_binds_new_sources_and_vision_configuration(tmp_path, monkeypatch):
    wrapper = module()
    runner = fake_runner(lambda _: 0)
    monkeypatch.setattr(wrapper, "_load_runner", lambda: runner)
    wrapper.main(["preflight", "--vision-max-age-s", "7.5"] + local_models(tmp_path))
    first = runner.runtime_inventory()
    assert first["old_runtime.py"] == "unchanged"
    assert wrapper.WRAPPER_PATH in first
    sources = list((wrapper.ROOT / "src/gpt_field/foundation_vision").rglob("*.py"))
    assert sources
    assert all(str(path.relative_to(wrapper.ROOT)) in first for path in sources)
    assert len(first["@foundation_vision"]) == 64
    assert runner.execution_options()["foundation_vision"]["max_age_s"] == 7.5
    # Callers cannot mutate settings used by subsequent preparation/execution hashes.
    runner.execution_options()["foundation_vision"]["max_age_s"] = 999
    assert runner.runtime_inventory() == first


def test_local_model_validation_rejects_remote_ids_before_load(tmp_path, monkeypatch):
    wrapper = module()
    flags = local_models(tmp_path)
    flags[-1] = "depth-anything/DA3-LARGE-1.1"
    monkeypatch.setattr(wrapper, "_build_vision_engine", lambda _: pytest.fail("Loaded invalid model path"))
    monkeypatch.setattr(wrapper, "_load_runner", lambda: pytest.fail("Loaded runner with invalid model path"))
    with pytest.raises(ValueError, match="local directory"):
        wrapper.main(["start"] + flags)


@pytest.mark.parametrize("age", ["0", "-1", "nan", "inf"])
def test_invalid_freshness_rejected_without_model_load(tmp_path, monkeypatch, age):
    wrapper = module()
    monkeypatch.setattr(wrapper, "_build_vision_engine", lambda _: pytest.fail("Loaded invalid config"))
    with pytest.raises(SystemExit):
        wrapper.main(["start", "--vision-max-age-s", age] + local_models(tmp_path))


def test_real_loader_contract_uses_local_only_models_and_loads_both_before_engine(tmp_path, monkeypatch):
    wrapper = module()
    from gpt_field.foundation_vision import backends
    events = []
    class LocalBackend:
        def __init__(self, **kwargs):
            self.name = "sam" if "checkpoint_path" in kwargs else "da3"
            assert kwargs["device"] == "cuda:3" and kwargs["local_files_only"] is True
            assert Path(kwargs.get("checkpoint_path", kwargs.get("model_id"))).exists()
        def _load(self):
            events.append(self.name)
        def provenance(self):
            return {"backend": self.name, "injected": True}
    monkeypatch.setattr(backends, "Sam31Segmenter", LocalBackend)
    monkeypatch.setattr(backends, "DA3DepthEstimator", LocalBackend)
    args, _ = wrapper.build_parser().parse_known_args(
        local_models(tmp_path) + ["--vision-device", "cuda:3"])
    wrapper._validate_local_models(args)
    engine, metadata = wrapper._build_vision_engine(args)
    assert events == ["sam", "da3"]
    assert engine.config.max_age_s == 5.
    assert metadata["models"]["sam"]["injected"] is True
    assert set(metadata["load_times"]) == {"sam_load_s", "da3_load_s"}
