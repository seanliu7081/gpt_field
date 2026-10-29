#!/usr/bin/env python3
"""Bounded single-GPU procedural camera child and CUDA/NVML identity helpers.

Importing this file is side-effect free. Only the gated main constructs CUDA or
Isaac Sim. This creates two procedural cubes; no RoboDojo task or model executes.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


def normalize_pci(value: str) -> str:
    domain, bus, rest = value.lower().split(":")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{rest}"


def renderer_table_identity(log: str, binding: dict) -> dict:
    """Kit 5.1 prints an eight-hex UUID prefix and hex PCI bus, not a full UUID."""
    rows = []
    lines = log.splitlines()
    for index, line in enumerate(lines[:-2]):
        match = re.match(r"^\|\s*(\d+)\s*\|[^|]*\|\s*Yes:\s*\d+", line)
        if match:
            uuid_cells, bus_cells = lines[index + 1].split("|"), lines[index + 2].split("|")
            if len(uuid_cells) < 8 or len(bus_cells) < 7:
                raise RuntimeError("Unrecognized Kit GPU table continuation rows")
            prefix = uuid_cells[7].strip().lower()
            bus = bus_cells[6].strip().lower()
            rows.append({"kit_index": int(match[1]), "uuid_prefix": prefix, "pci_bus_hex": bus})
    rows = list({json.dumps(row, sort_keys=True): row for row in rows}.values())
    expected_prefix = binding["gpu_uuid"].removeprefix("GPU-")[:8].lower() + ".."
    expected_bus = int(normalize_pci(binding["pci_bus_id"]).split(":")[1], 16)
    passed = bool(len(rows) == 1 and rows[0]["kit_index"] == binding["kit_index"]
                  and rows[0]["uuid_prefix"] == expected_prefix
                  and int(rows[0]["pci_bus_hex"], 16) == expected_bus)
    return {"active_rows": rows, "binding": binding, "uuid_prefix_and_pci_bus_match": passed,
            "kit_exposes_full_uuid": False,
            "full_uuid_source": "Full UUID is verified separately against CUDA metadata and NVML"}


def query_cuda_identity(binding: dict) -> dict:
    """Query device metadata only; do not retain or create a CUDA context."""
    driver = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.c_void_p, ctypes.c_int],
        "cuDeviceGetPCIBusId": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
    }
    for name, signature in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = signature, ctypes.c_int

    def checked(name, *args):
        result = getattr(driver, name)(*args)
        if result != 0:
            raise RuntimeError(f"CUDA metadata query {name} failed with code {result}")

    checked("cuInit", 0)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError("The metadata check requires exactly one visible CUDA device")
    device = ctypes.c_int()
    checked("cuDeviceGet", ctypes.byref(device), binding["cuda_ordinal"])
    raw_uuid = (ctypes.c_ubyte * 16)()
    checked("cuDeviceGetUuid", ctypes.byref(raw_uuid), device)
    actual_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
    pci = ctypes.create_string_buffer(64)
    checked("cuDeviceGetPCIBusId", pci, len(pci), device)
    actual_pci = pci.value.decode()
    if actual_uuid != binding["gpu_uuid"] or normalize_pci(actual_pci) != normalize_pci(binding["pci_bus_id"]):
        raise RuntimeError("CUDA ordinal UUID/PCI identity differs from the explicitly bound device")
    return {"gpu_uuid": actual_uuid, "pci_bus_id": actual_pci,
            "cuda_ordinal": binding["cuda_ordinal"], "device_count": count.value,
            "primary_context_created": False, "child_pid": os.getpid(),
            "child_start_ticks": int(Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19])}


def main() -> int:
    if os.environ.get("UC71_CAMERA_AUTHORIZED_CHILD") != "1":
        raise SystemExit("Use provision_new_host_camera.py with a fresh setup authorization record")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise SystemExit("A1 stock control forbids CUDA device remapping")
    output = Path(os.environ["UC71_CAMERA_OUTPUT"])
    binding = json.loads(os.environ["UC71_CAMERA_BINDING"])
    if binding["kit_index"] != 0 or binding["cuda_ordinal"] != 0:
        raise RuntimeError("Single-GPU diagnostic requires explicit zero Kit/CUDA indices")

    def progress(phase, **details):
        record = {"phase": phase, "monotonic_s": time.monotonic(), **details}
        (output / "camera_progress.json").write_text(json.dumps(record) + "\n")
        with (output / "camera_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    identity = query_cuda_identity(binding)
    (output / "cuda_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    progress("cuda_device_metadata_verified", **identity)
    progress("before_stock_simulation_app_import")
    # Do not import any omni/pxr module before NVIDIA's standard launcher.
    from isaacsim import SimulationApp

    import isaacsim
    package = Path(isaacsim.__file__).resolve().parent
    experience = package / "apps/isaacsim.exp.base.python.kit"
    if not experience.is_file():
        raise RuntimeError("The stock base.python experience is required")
    config = {"headless": True, "active_gpu": binding["kit_index"], "physics_gpu": binding["cuda_ordinal"],
              "multi_gpu": False, "max_gpu_count": 1, "width": 640, "height": 480,
              "renderer": "RaytracedLighting", "fast_shutdown": False, "create_new_stage": True,
              "enable_crashreporter": False,
              "extra_args": ["--portable", f"--portable-root={output / 'kit-portable'}",
                             "--/app/settings/persistent=false", "--/app/telemetry/enable=false",
                             "--/telemetry/enableAnonymousData=false", "--/telemetry/enableAnonymousAppName=false",
                             "--/rtx/multiGpu/enabled=false", "--/rtx/multiGpu/maxGpuCount=1"]}
    (output / "simulation_app_configuration.json").write_text(json.dumps({
        "config": config, "experience": str(experience),
        "experience_sha256": hashlib.sha256(experience.read_bytes()).hexdigest(),
        "cuda_visible_devices_present": False,
        "precreated_primary_cuda_context": False,
        "model_calls": 0, "benchmark_actions": 0}, indent=2) + "\n")
    started = time.monotonic()
    progress("before_stock_simulation_app_startup")
    app = SimulationApp(config, experience=str(experience))
    progress("stock_simulation_app_startup_returned")
    annotator = product = None
    attached = False
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
        if observed["/renderer/activeGpu"] != binding["kit_index"] or observed["/physics/cudaDevice"] != binding["cuda_ordinal"] or observed["/renderer/multiGpu/enabled"] is not False:
            raise RuntimeError("Stock app GPU settings differ from the bound single-GPU configuration")
        log = (output / "stdout_stderr.log").read_text(errors="replace")
        renderer_identity = renderer_table_identity(log, binding)
        (output / "active_renderer_identity.json").write_text(json.dumps(renderer_identity, indent=2) + "\n")
        if not renderer_identity["uuid_prefix_and_pci_bus_match"]:
            raise RuntimeError("Kit GPU table did not confirm the bound index, UUID prefix and PCI bus as sole active renderer")
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
        attached = True
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
        try:
            if attached:
                annotator.detach([product])
            if product is not None:
                product.destroy()
        finally:
            app.close()
        progress("stock_simulation_app_close_returned")


if __name__ == "__main__":
    raise SystemExit(main())
