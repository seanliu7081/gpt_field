#!/usr/bin/env python3
"""Bounded, model-free Isaac Sim camera check for a Conda host with multiple GPUs.

Runs eight procedural captures on GPU 0 without CUDA remapping. The parent checks
the rendered image, process exit, GPU identity and released GPU memory. This does
not run RoboDojo or certify the historical full-task launcher's host bindings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from new_host_camera import renderer_table_identity, write_durable_json
from provision_new_host_camera import gpu_snapshot

BASE = Path(__file__).resolve().parents[1]


def capture(output: Path) -> None:
    binding = json.loads((output / "binding.json").read_text())
    from isaacsim import SimulationApp
    import isaacsim

    experience = Path(isaacsim.__file__).parent / "apps/isaacsim.exp.base.python.kit"
    app = SimulationApp({
        "headless": True, "active_gpu": 0, "physics_gpu": 0,
        "multi_gpu": False, "max_gpu_count": 1, "width": 640, "height": 480,
        "renderer": "RaytracedLighting", "fast_shutdown": True,
        "create_new_stage": True, "enable_crashreporter": False,
        "extra_args": ["--portable", "--portable-root", str(output / "kit"),
                       "--/app/settings/persistent=false", "--/app/telemetry/enable=false",
                       "--/telemetry/enableAnonymousData=false",
                       "--/rtx/multiGpu/enabled=false"],
    }, experience=str(experience))
    import carb
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux

    renderer = renderer_table_identity((output / "stdout_stderr.log").read_text(errors="replace"), binding)
    if not renderer["uuid_prefix_and_pci_bus_match"]:
        raise RuntimeError("Active Kit GPU does not match the selected GPU")
    settings = carb.settings.get_settings()
    assert settings.get("/renderer/activeGpu") == 0
    assert settings.get("/physics/cudaDevice") == 0
    assert settings.get("/renderer/multiGpu/enabled") is False
    # Exercise the pinned CUDA wheel as well as the graphics stack.
    cuda_properties = torch.cuda.get_device_properties(0)
    cuda_uuid = str(cuda_properties.uuid)
    if cuda_uuid.removeprefix("GPU-") != binding["gpu_uuid"].removeprefix("GPU-"):
        raise RuntimeError("CUDA and renderer devices differ")
    value = (torch.ones(4, device="cuda:0") * 2).sum().item()
    assert value == 8.0
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    for name, x, color in (("Red", -.45, (.8, .05, .05)), ("Green", .45, (.05, .8, .05))):
        cube = UsdGeom.Cube.Define(stage, "/World/" + name)
        cube.CreateSizeAttr(.5)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0, .3))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1)
    ground.AddScaleOp().Set(Gf.Vec3f(4, 4, .05))
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.025))
    ground.CreateDisplayColorAttr([Gf.Vec3f(.35, .35, .35)])
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(800)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(24)
    camera.CreateHorizontalApertureAttr(32)
    camera.CreateVerticalApertureAttr(24)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(2, -3, 2), Gf.Vec3d(0, 0, .25), Gf.Vec3d(0, 0, 1)).GetInverse())
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product("/World/Camera", (640, 480))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0,
                              pause_timeline=True, wait_for_render=True)
    frame = np.asarray(annotator.get_data())[..., :3].copy()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
    assert min(frame.std(axis=(0, 1))) > 2
    assert np.sum((frame[..., 0] > frame[..., 1] * 1.4) & (frame[..., 0] > 40)) > 100
    assert np.sum((frame[..., 1] > frame[..., 0] * 1.4) & (frame[..., 1] > 40)) > 100
    Image.fromarray(frame).save(output / "rgb.png")
    annotator.detach()
    product.destroy()
    write_durable_json(output / "capture.json", {
        "passed": True, "shape": list(frame.shape), "captures": 8,
        "renderer": renderer, "cuda_uuid": cuda_uuid, "torch": torch.__version__,
        "cuda_tensor_check": value, "model_calls": 0, "benchmark_actions": 0,
        "shutdown_mode": "stock-fast", "full_plugin_teardown_verified": False,
    })
    app.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic output directory")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.child:
        capture(output)
        return 0
    licence = json.loads((BASE / "validation/provisioning/vendor_authorization.json").read_text())
    if licence.get("nvidia_eula_accepted") is not True:
        parser.error("The preserved NVIDIA EULA acceptance is required")
    before = gpu_snapshot()
    if not before or any(gpu["processes"] for gpu in before):
        parser.error("Run this environment check while the GPUs are idle")
    selected = before[0]
    if int(selected["memory_free"].split()[0]) < 16384:
        parser.error("GPU 0 needs at least 16 GiB free for this check")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("home", "cache", "config", "data"):
        (output / name).mkdir()
    binding = {"gpu_uuid": selected["uuid"], "pci_bus_id": selected["pci_bus_id"],
               "kit_index": 0, "cuda_ordinal": 0}
    write_durable_json(output / "binding.json", binding)
    write_durable_json(output / "gpu_before.json", before)
    icd = next((p for p in (Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
                            Path("/etc/vulkan/icd.d/nvidia_icd.json")) if p.is_file()), None)
    if icd is None:
        parser.error("NVIDIA Vulkan ICD is missing")
    environment = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "LANG": "C.UTF-8",
        "HOME": str(output / "home"), "XDG_CACHE_HOME": str(output / "cache"),
        "XDG_CONFIG_HOME": str(output / "config"), "XDG_DATA_HOME": str(output / "data"),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "VK_ICD_FILENAMES": str(icd),
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    local_libraries = BASE / "runtime/sysroot/usr/lib/x86_64-linux-gnu"
    if local_libraries.is_dir():
        environment["LD_LIBRARY_PATH"] = str(local_libraries)
    timed_out = False
    with (output / "stdout_stderr.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                 "--child", "--output", str(output)],
                                env=environment, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=600)
        except BaseException as error:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            timed_out = True
    # NVML may briefly retain a process after it has exited.
    after = gpu_snapshot()
    for _ in range(10):
        if all(not gpu["processes"] for gpu in after):
            break
        time.sleep(1)
        after = gpu_snapshot()
    captured = output / "capture.json"
    result = {
        "returncode": proc.returncode, "timed_out": timed_out,
        "capture_passed": captured.is_file() and json.loads(captured.read_text())["passed"],
        "gpu_processes_released": all(not gpu["processes"] for gpu in after),
        "gpu_memory_released": all(int(a["memory_used"].split()[0]) <= int(b["memory_used"].split()[0]) + 64
                                   for a, b in zip(after, before)),
        "model_calls": 0, "benchmark_actions": 0,
    }
    result["passed"] = (result["returncode"] == 0 and not timed_out and result["capture_passed"]
                        and result["gpu_processes_released"] and result["gpu_memory_released"])
    write_durable_json(output / "gpu_after.json", after)
    write_durable_json(output / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
