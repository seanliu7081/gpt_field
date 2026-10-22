"""Exercise installed Codex against a loopback fake provider, never real inference.

This is a protocol integration check, not a robot episode or performance test.
No subscription credentials are loaded. No simulator is instantiated.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

import numpy as np

from gpt_field.astranomy_agentic_geometry import extract_geometry_metadata
from gpt_field.astranomy_agentic_motion import ShortTargetExecutor
from gpt_field.observation import image_to_data_url
from .client import CodexPolicyClient, StdioRPC, server_command
from .policy import TASK, SUPPORTED_TASKS


def fixture_request():
    model = SimpleNamespace(cam_fovy=np.array([60., 75.]),
                            camera_name2id=lambda name: {"agentview": 0, "robot0_eye_in_hand": 1}[name])
    data = SimpleNamespace(cam_xpos=np.array([[1., 2., 3.], [-.1, .2, .5]]),
                           cam_xmat=np.array([np.eye(3).ravel(), np.eye(3).ravel()]))
    state = [0., 0., 1., 0., 0., 0., .04, -.04]
    geometry = extract_geometry_metadata(SimpleNamespace(sim=SimpleNamespace(model=model, data=data)),
                                         observation_state=state)
    image = image_to_data_url(np.zeros((512, 512, 3), dtype=np.uint8), size=512, rotate=False)
    return {"external_image": image, "wrist_image": image, "robot_state": [0, 0, 1, 1, 0, 0, 0, 0],
            "joint_positions": [0.] * 7, "geometry": geometry, "step": 0}


def check(output, task_key=TASK):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            body = Path(__file__).with_name("model_catalog.json").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "zstd":
                import zstandard
                raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=20_000_000)
            request = json.loads(raw)
            requests.append(request)
            n = len(requests)
            if n % 2:
                item = {"id": f"fc_{n}", "type": "function_call", "status": "completed",
                        "namespace": "robot", "name": "execute_eef_target", "call_id": f"call_{n}",
                        "arguments": json.dumps({"target": [0, 0, 1.01, 1, 0, 0, 0, 0], "steps": 1})}
            else:
                item = {"id": f"msg_{n}", "type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Target proposed; awaiting actual motion feedback.", "annotations": []}]}
            response = {"id": f"resp_{n}", "object": "response", "model": "gpt-6-astra",
                        "status": "completed", "output": [item],
                        "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
                                  "input_tokens_details": {"cached_tokens": 0},
                                  "output_tokens_details": {"reasoning_tokens": 0}}}
            events = [{"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
                      {"type": "response.output_item.added", "output_index": 0, "item": item},
                      {"type": "response.output_item.done", "output_index": 0, "item": item},
                      {"type": "response.completed", "response": response}]
            body = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    def local_rpc(command, **kwargs):
        local = server_command(command[0], base_url=url, log_dir=output / "codex_logs")
        local += ["-c", "features.enable_request_compression=false"]
        return StdioRPC(local, **kwargs)

    client = CodexPolicyClient(task_key=task_key, token_provider=lambda: "offline-oauth-fixture-not-a-credential",
                              log_dir=output / "rpc", rpc_factory=local_rpc)
    report = {"kind": "offline_codex_protocol_integration", "task_key": task_key, "external_inference_requests": 0,
              "simulator_steps": 0, "subscription_credentials_loaded": False}
    try:
        request = fixture_request()
        proposal = client.predict(request)
        first_id = client.thread_id
        # A synthetic measured transition only. This does not run robot physics.
        state = [0., 0., 1., 0., 0., 0., .04, -.04]
        executor = ShortTargetExecutor(state, **proposal["arguments"])
        action = executor.next_action(state)
        executor.observe_transition(state, action, state)
        client.record_result(executor.feedback(state))
        request["step"] = 1
        client.predict(request)
        assert client.thread_id == first_id
        assert len(requests) == 4, f"Expected two mock requests per turn, saw {len(requests)}"
        declared = requests[0].get("tools", []) + [t for item in requests[0]["input"]
                    if item.get("type") == "additional_tools" for t in item["tools"]]
        names = [t.get("name", t.get("type")) for t in declared]
        assert names == ["robot"], f"Unexpected native tools: {names}"
        for req in requests:
            assert req["model"] == "gpt-6-astra"
            assert req["reasoning"]["effort"] == "high"
        history = json.dumps(requests[2]["input"])
        assert "actual_execution_feedback" in history and "call_1" in history
        assert history.count("data:image/") >= 4, "Previous paired images were not retained"
        report.update(passed=True, persistent_thread_verified=True, full_visual_history_verified=True,
                      native_tools=names, mock_inference_requests=len(requests), **client.budget_snapshot())
        report["billing"] = "offline_mock_no_billing"
    except BaseException as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        (output / "mock_requests.json").write_text(json.dumps(requests, indent=2))
        (output / "result.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default=TASK)
    args = parser.parse_args()
    print(json.dumps(check(args.output, task_key=args.task), indent=2))
