#!/usr/bin/env python3
"""Stock SimulationApp camera diagnostic: procedural scene, no task or model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time


def main() -> int:
    if os.environ.get("UC71_A1_AUTHORIZED_CHILD") != "1":
        raise SystemExit("Use the renewed-authorization-gated A1 launcher")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise SystemExit("A1 stock control forbids CUDA device remapping")
    output = Path(os.environ["UC71_A1_OUTPUT"])

    def progress(phase, **details):
        record = {"phase": phase, "monotonic_s": time.monotonic(), **details}
        (output / "camera_progress.json").write_text(json.dumps(record) + "\n")
        with (output / "camera_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    progress("before_stock_simulation_app_import")
    # Do not import any omni/pxr module before NVIDIA's standard launcher.
    from isaacsim import SimulationApp

    import isaacsim
    package = Path(isaacsim.__file__).resolve().parent
    experience = package / "apps/isaacsim.exp.base.python.kit"
    if not experience.is_file():
        raise RuntimeError("The stock base.python experience is required")
    config = {"headless": True, "active_gpu": 4, "physics_gpu": 4,
              "multi_gpu": False, "max_gpu_count": 1, "width": 640, "height": 480,
              "fast_shutdown": False, "create_new_stage": True,
              "enable_crashreporter": False,
              "extra_args": ["--portable", f"--portable-root={output / 'kit-portable'}",
                             "--/app/settings/persistent=false", "--/app/telemetry/enable=false",
                             "--/telemetry/enableAnonymousData=false", "--/telemetry/enableAnonymousAppName=false",
                             "--/rtx/multiGpu/enabled=false", "--/rtx/multiGpu/maxGpuCount=1"]}
    (output / "simulation_app_configuration.json").write_text(json.dumps({
        "config": config, "experience": str(experience),
        "experience_sha256": hashlib.sha256(experience.read_bytes()).hexdigest(),
        "cuda_visible_devices_present": False, "model_calls": 0, "benchmark_actions": 0}, indent=2) + "\n")
    started = time.monotonic()
    progress("before_stock_simulation_app_startup")
    app = SimulationApp(config, experience=str(experience))
    progress("stock_simulation_app_startup_returned")
    try:
        import carb
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux

        settings = carb.settings.get_settings()
        observed = {key: settings.get(key) for key in (
            "/renderer/activeGpu", "/physics/cudaDevice", "/renderer/multiGpu/enabled",
            "/renderer/multiGpu/maxGpuCount", "/rtx/multiGpu/enabled")}
        (output / "observed_gpu_settings.json").write_text(json.dumps(observed, indent=2) + "\n")
        if observed["/renderer/activeGpu"] != 4 or observed["/physics/cudaDevice"] != 4 or observed["/renderer/multiGpu/enabled"] is not False:
            raise RuntimeError("Stock app GPU settings differ from the authorized GPU4 configuration")
        log = (output / "stdout_stderr.log").read_text(errors="replace")
        active = re.findall(r"^\|\s*(\d+)\s*\|[^|]*\|\s*Yes:\s*\d+[^\n]*\n([^\n]*)", log, flags=re.MULTILINE)
        active = list(dict.fromkeys(active))
        (output / "active_renderer_identity.json").write_text(json.dumps({"active_rows": active,
            "expected_index": 4, "expected_uuid_prefix": "e1a460e9"}, indent=2) + "\n")
        if len(active) != 1 or active[0][0] != "4" or "e1a460e9" not in active[0][1]:
            raise RuntimeError("Kit GPU table did not confirm only the expected GPU4 UUID as active renderer")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Stock SimulationApp did not create its initial stage")
        progress("stock_stage_ready")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        for name, location, color in (
            ("Red", (-0.45, 0.0, 0.3), (0.8, 0.05, 0.05)),
            ("Green", (0.45, 0.0, 0.3), (0.05, 0.8, 0.05)),
        ):
            cube = UsdGeom.Cube.Define(stage, "/World/" + name)
            cube.CreateSizeAttr(0.5)
            cube.AddTranslateOp().Set(Gf.Vec3d(*location))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddScaleOp().Set(Gf.Vec3f(4, 4, 0.05))
        ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.35)])
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(800.0)
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(32.0)
        camera.CreateVerticalApertureAttr(24.0)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(2, -3, 2), Gf.Vec3d(0, 0, 0.25), Gf.Vec3d(0, 0, 1)).GetInverse())
        product = rep.create.render_product("/World/Camera", (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        progress("procedural_camera_product_created")
        frame = None
        for frame_index in range(120):
            app.update()
            data = np.asarray(annotator.get_data())
            if data.ndim == 3 and data.shape[:2] == (480, 640):
                if frame is None:
                    progress("first_rgb_frame_received", update_index=frame_index)
                frame = data[..., :3].copy()
        if frame is None:
            raise RuntimeError("No 640x480 RGB after 120 bounded stock app updates")
        np.save(output / "rgb.npy", frame)
        Image.fromarray(frame).save(output / "rgb.png")
        stats = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                 "per_channel_std": frame.std(axis=(0, 1)).tolist(), "min": int(frame.min()), "max": int(frame.max()),
                 "red_pixels": int(np.sum((frame[..., 0].astype(float) > frame[..., 1] * 1.4) & (frame[..., 0] > 40))),
                 "green_pixels": int(np.sum((frame[..., 1].astype(float) > frame[..., 0] * 1.4) & (frame[..., 1] > 40))),
                 "elapsed_s": time.monotonic() - started, "model_calls": 0, "benchmark_actions": 0,
                 "scope": "stock SimulationApp procedural camera only; native RoboDojo not exercised"}
        stats["passed"] = bool(frame.dtype == np.uint8 and min(stats["per_channel_std"]) > 2
                               and stats["red_pixels"] > 100 and stats["green_pixels"] > 100)
        (output / "camera_result.json").write_text(json.dumps(stats, indent=2) + "\n")
        progress("camera_validation_complete", passed=stats["passed"])
        if not stats["passed"]:
            raise RuntimeError("Procedural RGB failed its nonblank red/green object validation")
        return 0
    finally:
        progress("stock_simulation_app_close_requested")
        app.close()
        progress("stock_simulation_app_close_returned")


if __name__ == "__main__":
    raise SystemExit(main())
