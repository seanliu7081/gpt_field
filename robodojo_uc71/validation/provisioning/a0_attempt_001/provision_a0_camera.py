#!/usr/bin/env python3
"""Bounded render-only Kit diagnostic; called only by provision_a0_launch.py.

No RoboDojo environment, robot, benchmark assets, policy, credentials or LLM.
The launcher checks explicit license/diagnostic authorization before this file
can import Isaac Sim. This diagnostic uses a procedural USD scene only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def main() -> int:
    if os.environ.get("UC71_A0_AUTHORIZED_CHILD") != "1":
        raise SystemExit("Use the authorization-gated provision_a0_launch.py wrapper")
    output = Path(os.environ["UC71_A0_OUTPUT"])
    # This is deliberately the first vendor import, after wrapper authorization.
    import isaacsim
    from kit_app import KitApp

    package = Path(isaacsim.__file__).resolve().parent
    app = KitApp()
    argv = [
        str(output.parent / "a0_camera.kit"),
        "--no-window", "--allow-root",
        "--portable", f"--portable-root={output / 'kit-portable'}",
        "--ext-folder", str(package / "extscache"),
        "--ext-folder", str(package / "exts"),
        "--/app/window/enabled=false",
        "--/app/settings/persistent=false",
        "--/renderer/activeGpu=4",
        "--/renderer/multiGpu/enabled=false",
        "--/rtx/multiGpu/enabled=false",
        "--/rtx/multiGpu/maxGpuCount=1",
        "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
        "--/app/telemetry/enable=false",
    ]
    (output / "kit_arguments.json").write_text(json.dumps(argv, indent=2) + "\n")
    started = time.monotonic()
    app.startup(argv)
    try:
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
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
        ground.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.05))
        ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.35)])
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(800.0)
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(32.0)
        camera.CreateVerticalApertureAttr(24.0)
        matrix = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(2.0, -3.0, 2.0), Gf.Vec3d(0.0, 0.0, 0.25), Gf.Vec3d(0, 0, 1)
        ).GetInverse()
        camera.AddTransformOp().Set(matrix)
        product = rep.create.render_product("/World/Camera", (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        frame = None
        for _ in range(120):
            app.update()
            data = np.asarray(annotator.get_data())
            if data.ndim == 3 and data.shape[:2] == (480, 640):
                frame = data[..., :3].copy()
        if frame is None:
            raise RuntimeError("No native 640x480 RGB frame after 120 bounded updates")
        stats = {
            "shape": list(frame.shape), "dtype": str(frame.dtype),
            "per_channel_std": frame.std(axis=(0, 1)).tolist(),
            "min": int(frame.min()), "max": int(frame.max()),
            "elapsed_s": time.monotonic() - started,
            "model_calls": 0, "benchmark_actions": 0,
            "scope": "procedural RTX camera only; native RoboDojo not tested",
        }
        stats["passed"] = bool(frame.dtype == np.uint8 and min(stats["per_channel_std"]) > 2)
        np.save(output / "rgb.npy", frame)
        Image.fromarray(frame).save(output / "rgb.png")
        (output / "camera_result.json").write_text(json.dumps(stats, indent=2) + "\n")
        if not stats["passed"]:
            raise RuntimeError("Rendered image failed the bounded nonblank variance check")
        print(json.dumps(stats), flush=True)
        return 0
    finally:
        app.shutdown()


if __name__ == "__main__":
    sys.exit(main())
