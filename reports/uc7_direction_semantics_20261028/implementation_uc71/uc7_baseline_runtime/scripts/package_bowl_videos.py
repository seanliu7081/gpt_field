#!/usr/bin/env python3
"""Validate and package ten finished bowl episodes without changing run sources.

Usage: .venv/bin/python scripts/package_bowl_videos.py runs/astra_bowl10_001
Use --check-only to validate readiness without writing any output. Packaging
requires ten finalized, decodable MP4s and terminal records; it never creates a
completed-looking package from a live, partial, or missing recording.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import html
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw, ImageFont


INDICES = tuple(range(10))
PENDING = {"pending", "running", "interrupted"}
CONTACT_NAME = "final_observations.png"
GALLERY_NAME = "offline_gallery.html"
PACKAGE_MANIFEST = "video_package_manifest.json"


class NotReady(ValueError):
    """The source artifacts are incomplete, inconsistent, or invalid."""


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise NotReady(f"Cannot read complete JSON artifact: {path}") from exc


def read_jsonl(path):
    try:
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except (OSError, ValueError) as exc:
        raise NotReady(f"Cannot read complete JSONL artifact: {path}") from exc


def local_path(root, relative):
    """Resolve only regular files inside this run, including symlink targets."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise NotReady(f"Artifact path leaves its run directory: {relative}")
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise NotReady(f"Missing or external artifact: {path}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_camera(url):
    prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(prefix):
        raise NotReady("Final observation image is not an embedded RGB PNG")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(url[len(prefix):], validate=True)))
        image.load()
    except (OSError, ValueError) as exc:
        raise NotReady("Final observation contains a damaged PNG") from exc
    if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
        raise NotReady(f"Expected actual RGB 224x224 policy image; got {image.mode} {image.size}")
    return image


def final_observation(trace, steps):
    """Use post-action observations only; use reset only for zero executed actions."""
    reset, last, executed_count = None, None, 0
    try:
        with gzip.open(trace, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "reset":
                    if reset is not None:
                        raise NotReady(f"Multiple resets in one episode trace: {trace}")
                    reset = event
                elif event.get("event") == "executed":
                    if event.get("step") != executed_count:
                        raise NotReady(f"Nonsequential execution steps in {trace}")
                    last = event
                    executed_count += 1
    except (OSError, EOFError, ValueError) as exc:
        raise NotReady(f"Incomplete or invalid execution trace: {trace}") from exc
    if executed_count != steps:
        raise NotReady(f"Trace has {executed_count} executed steps, result has {steps}: {trace}")
    event = last if executed_count else reset
    if event is None or "observation" not in event:
        raise NotReady(f"No actual final observation exists in {trace}")
    observation = event["observation"]
    try:
        images = tuple(decode_camera(observation[key]) for key in
                       ("observation/image", "observation/wrist_image"))
        state = observation["observation/state"]
        if len(state) != 8 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in state):
            raise NotReady(f"Invalid final measured state in {trace}")
    except (KeyError, TypeError) as exc:
        raise NotReady(f"Final observation is incomplete in {trace}") from exc
    if last is not None and last.get("next_state") != state:
        raise NotReady(f"Executed state disagrees with its final observation in {trace}")
    return images, {
        "source_event": event["event"], "source_step": event.get("step"),
        "state": state, "prompt": observation.get("prompt"),
        "observed_success": bool(event.get("success", event.get("sim_status", {}).get("success", False))),
    }


def probe_video(path):
    probe = shutil.which("ffprobe")
    if probe is None:
        raise NotReady("ffprobe is required to validate completed recordings")
    try:
        result = subprocess.run([
            probe, "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_name,codec_type,width,height,nb_read_frames,r_frame_rate,duration:format=duration",
            "-of", "json", str(path),
        ], check=True, capture_output=True, text=True, timeout=120)
        if result.stderr.strip():
            raise NotReady(f"Decoder errors in {path}: {result.stderr.strip()}")
        info = json.loads(result.stdout)
        streams = [stream for stream in info["streams"] if stream.get("codec_type") == "video"]
        if len(streams) != 1:
            raise NotReady(f"Expected exactly one video stream: {path}")
        stream = streams[0]
        frames = int(stream["nb_read_frames"])
        fps = float(Fraction(stream["r_frame_rate"]))
        duration = float(info["format"]["duration"])
        if (stream.get("codec_name") != "h264" or stream.get("width") != 1600
                or stream.get("height") != 960 or frames < 1 or fps != 20
                or abs(duration - frames / fps) > 1 / fps):
            raise NotReady(f"Unexpected or incomplete video stream: {path}")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, ZeroDivisionError) as exc:
        raise NotReady(f"Cannot validate complete MP4: {path}: {exc}") from exc
    return {"codec": stream["codec_name"], "width": stream["width"], "height": stream["height"],
            "frames": frames, "fps": fps, "duration_s": duration}


def validate_events(path, row, media, observed):
    records = read_jsonl(path)
    if not records or records[0].get("event") != "observation" or records[-1].get("event") != "terminal":
        raise NotReady(f"Video events lack initial observation or terminal outcome: {path}")
    position = executed_count = terminal_count = 0
    for event in records:
        count = event.get("frame_count")
        if type(count) is not int or count <= 0 or event.get("frame_start") != position:
            raise NotReady(f"Video event frame spans are not contiguous: {path}")
        if abs(event.get("video_time_s", -1) - position / media["fps"]) > 1e-6:
            raise NotReady(f"Video event times disagree with encoded frames: {path}")
        position += count
        if event.get("event") == "executed":
            if event.get("step") != executed_count:
                raise NotReady(f"Video execution events are not sequential: {path}")
            executed_count += 1
        terminal_count += event.get("event") == "terminal"
    final = records[-1]
    if position != media["frames"] or executed_count != row["steps"] or terminal_count != 1:
        raise NotReady(f"Video frame/event counts disagree with its result: {path}")
    if final.get("status") != row["status"] or final.get("state") != observed["state"]:
        raise NotReady(f"Video terminal outcome/state disagrees with its trace/result: {path}")
    if final.get("success") is not (row["status"] == "success"):
        raise NotReady(f"Video terminal success flag disagrees with its result: {path}")
    if (row["status"] == "success") != observed["observed_success"]:
        raise NotReady(f"Recorded outcome disagrees with last actual simulator observation: {path}")
    return {"event_count": len(records), "executed_events": executed_count,
            "field_refresh_events": sum(record.get("event") == "field_refresh" for record in records)}


def collect(run):
    manifest = read_json(run / "manifest.json")
    summary = read_json(run / "summary.json")
    if manifest.get("configuration", {}).get("episode_indices") != list(INDICES):
        raise NotReady("Expected the ten frozen initial states 0 through 9")
    root_rows = summary.get("episodes", [])
    if len(root_rows) != 10 or sorted(row.get("episode_index", -1) for row in root_rows) != list(INDICES):
        raise NotReady("Summary does not contain exactly ten distinct initial states")
    if summary.get("pending_or_running") != 0 or any(row.get("status") in PENDING for row in root_rows):
        raise NotReady("Evaluation is still running or has missing terminal results")
    partials = list(run.rglob("*.partial.mp4"))
    if partials:
        raise NotReady(f"Unfinished MP4 remains: {partials[0].relative_to(run)}")
    items, all_videos = [], set()
    summary_rows = {row["episode_index"]: row for row in root_rows}
    for index in INDICES:
        directory = run / f"episode_{index:02d}"
        rows_path = local_path(run, directory.relative_to(run) / "episodes.jsonl")
        rows = read_jsonl(rows_path)
        if len(rows) != 1:
            raise NotReady(f"Expected one immutable terminal result for initial state {index}")
        row = rows[0]
        if row.get("episode_index") != index or row.get("task_id") != 0 or row.get("status") in PENDING:
            raise NotReady(f"Terminal result identity/status mismatch for initial state {index}")
        if type(row.get("steps")) is not int or row["steps"] < 0:
            raise NotReady(f"Invalid terminal step count for initial state {index}")
        root_row = summary_rows[index]
        root_status = "incomplete_timeout" if row["status"] == "timeout" and row["steps"] != 220 else row["status"]
        if root_row.get("status") != root_status or root_row.get("steps") != row["steps"]:
            raise NotReady(f"Summary has not yet caught up with initial state {index}")
        video = local_path(directory, row.get("video", "missing-video"))
        if video.suffix != ".mp4" or ".partial." in video.name:
            raise NotReady(f"Result does not reference a finalized MP4: {video}")
        events = local_path(directory, video.with_suffix(".events.jsonl").name)
        trace = local_path(directory, row.get("log", "missing-trace"))
        images, observation = final_observation(trace, row["steps"])
        media = probe_video(video)
        event_info = validate_events(events, row, media, observation)
        all_videos.add(video.resolve())
        items.append({"row": row, "summary_status": root_status, "images": images,
                      "video": video, "events": events, "trace": trace, "directory": directory,
                      "media": media, "observation": observation, "event_info": event_info})
    if len(all_videos) != 10:
        raise NotReady("Ten independent finalized video files are required")
    return manifest, summary, items


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def contact_sheet(items, summary):
    """Five columns, two rows; both final native-resolution cameras per episode."""
    cell_w, cell_h = 492, 352
    sheet = Image.new("RGB", (5 * cell_w + 24, 2 * cell_h + 142), (12, 18, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 18), "ASTRA / LIBERO: ACTUAL FINAL OBSERVATIONS", font=font(30, True), fill=(232, 239, 250))
    draw.text((24, 60), "Each pair is the last executed camera observation; reset images are used only for zero-action episodes.",
              font=font(21), fill=(165, 183, 208))
    outcome = (f"{summary['successes']} / 10 successes" if summary.get("success_rate") is not None
               else f"Provisional evaluation: {summary.get('completed_evaluations', 0)} completed task outcomes; infrastructure outcomes retained")
    draw.text((24, 95), outcome, font=font(20, True), fill=(165, 183, 208))
    for position, item in enumerate(items):
        x, y = 12 + position % 5 * cell_w, 137 + position // 5 * cell_h
        row, obs = item["row"], item["observation"]
        color = (102, 227, 161) if row["status"] == "success" else (249, 174, 119)
        draw.rounded_rectangle((x, y, x + cell_w - 12, y + cell_h - 12), radius=12, fill=(22, 31, 47))
        draw.text((x + 12, y + 12), f"Init {row['episode_index']:02d}  |  {item['summary_status']}", font=font(21, True), fill=color)
        draw.text((x + 12, y + 44), f"{row['steps']} executed steps  |  source: {obs['source_event']}", font=font(17), fill=(232, 239, 250))
        draw.text((x + 12, y + 73), "Agent view / RGB 224 x 224", font=font(15), fill=(165, 183, 208))
        draw.text((x + 246, y + 73), "Wrist view / RGB 224 x 224", font=font(15), fill=(165, 183, 208))
        sheet.paste(item["images"][0], (x + 12, y + 101))
        sheet.paste(item["images"][1], (x + 246, y + 101))
    return sheet


def offline_gallery(items, summary):
    cards = []
    for item in items:
        row = item["row"]
        video = html.escape((Path(item["directory"].name) / item["video"].name).as_posix(), quote=True)
        events = html.escape((Path(item["directory"].name) / item["events"].name).as_posix(), quote=True)
        cards.append(f'<article><h2>Initial state {row["episode_index"]:02d}: {html.escape(item["summary_status"])}</h2>'
                     f'<p>{row["steps"]} executed steps; {item["media"]["duration_s"]:.2f}s annotated playback.</p>'
                     f'<video controls preload="metadata" src="{video}"></video>'
                     f'<p><a href="{video}" download>Download MP4</a> · <a href="{events}">Event index</a></p></article>')
    rate = (f'{summary["successes"]}/10 successes' if summary.get("success_rate") is not None
            else f'Provisional task score; {summary.get("completed_evaluations", 0)}/10 completed task outcomes')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Ten LIBERO bowl recordings</title><style>'
            'body{font:17px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#0c121e;color:#e8effa}'
            'a{color:#73d8ee}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:20px}'
            'article{background:#161f2f;padding:18px;border-radius:12px}h2{font-size:20px}img,video{width:100%;height:auto}'
            '@media(max-width:520px){main{display:block}article{margin-bottom:20px}}</style></head><body>'
            '<h1>Ten Astra / LIBERO bowl recordings</h1>'
            f'<p>{html.escape(summary.get("task", ""))}</p><p><strong>{rate}</strong></p>'
            '<p>Recorded observations, actual model fields, local search, execution and terminal outcomes. '
            'Simulation ran at 20 Hz and paused for API requests; annotated playback slows control events and omits API waits.</p>'
            f'<p><a href="{CONTACT_NAME}">Full-resolution final observations</a> · '
            f'<a href="summary.json">Results</a> · <a href="manifest.json">Frozen protocol</a> · '
            f'<a href="{PACKAGE_MANIFEST}">Media validation and checksums</a></p>'
            f'<img src="{CONTACT_NAME}" alt="Actual final agent and wrist camera views for all ten initial states">'
            '<main>' + ''.join(cards) + '</main></body></html>')


def package(run, output):
    if output.suffix.lower() != ".zip":
        raise NotReady("Package output must have a .zip suffix")
    manifest, summary, items = collect(run)
    destinations = [run / CONTACT_NAME, run / GALLERY_NAME, run / PACKAGE_MANIFEST, output]
    if any(path.exists() for path in destinations):
        raise NotReady("Package outputs already exist; choose a fresh output and preserve prior artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [local_path(run, name) for name in ("index.html", "manifest.json", "summary.json")]
    for item in items:
        files.extend((item["video"], item["events"]))
        for name in ("episodes.jsonl", "manifest.json", "summary.json", "simulator.json", "run_state.json"):
            files.append(local_path(run, item["directory"].relative_to(run) / name))
    with tempfile.TemporaryDirectory(prefix="bowl-video-package-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        sheet = temporary / CONTACT_NAME
        contact_sheet(items, summary).save(sheet)
        gallery = temporary / GALLERY_NAME
        gallery.write_text(offline_gallery(items, summary))
        checksums = {}
        for path in files:
            name = path.relative_to(run).as_posix()
            if name == "index.html":
                name = "coordinator_index.html"
            checksums[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        checksums["index.html"] = {"sha256": sha256(gallery), "bytes": gallery.stat().st_size}
        checksums[CONTACT_NAME] = {"sha256": sha256(sheet), "bytes": sheet.stat().st_size}
        checksums[GALLERY_NAME] = {"sha256": sha256(gallery), "bytes": gallery.stat().st_size}
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_name": run.name,
            "frozen_identity_hash": manifest.get("identity_hash"), "video_count": 10,
            "evaluation_status": summary.get("status"),
            "source_gallery": "coordinator_index.html", "offline_entry_point": "index.html",
            "contact_sheet": CONTACT_NAME,
            "note": "Media completeness does not turn infrastructure errors into completed task outcomes. Traces and raw API logs remain in the source run.",
            "files": checksums,
            "episodes": [{"episode_index": item["row"]["episode_index"], "status": item["summary_status"],
                          "steps": item["row"]["steps"], "video": item["video"].relative_to(run).as_posix(),
                          "events": item["events"].relative_to(run).as_posix(), "ffprobe": item["media"],
                          "event_validation": item["event_info"], "final_observation": item["observation"],
                          "source_trace": item["trace"].relative_to(run).as_posix(),
                          "source_trace_sha256": sha256(item["trace"])} for item in items],
        }
        metadata_path = temporary / PACKAGE_MANIFEST
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        archive = temporary / output.name
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in files:
                relative = path.relative_to(run)
                # Keep the source gallery as evidence; make the richer local gallery the entry point.
                if relative.as_posix() == "index.html":
                    relative = Path("coordinator_index.html")
                bundle.write(path, (Path(run.name) / relative).as_posix(),
                             compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED)
            for path, name in ((sheet, CONTACT_NAME), (gallery, "index.html"),
                               (gallery, GALLERY_NAME), (metadata_path, PACKAGE_MANIFEST)):
                bundle.write(path, (Path(run.name) / name).as_posix())
        with zipfile.ZipFile(archive) as bundle:
            if sum(name.endswith(".mp4") for name in bundle.namelist()) != 10 or bundle.testzip() is not None:
                raise NotReady("Archive integrity check failed")
        # Source files are not replaced. New derived artifacts publish only after successful validation.
        for path, name in ((sheet, CONTACT_NAME), (gallery, GALLERY_NAME), (metadata_path, PACKAGE_MANIFEST)):
            shutil.copyfile(path, run / name)
        shutil.move(str(archive), output)
    return {"archive": str(output), "archive_bytes": output.stat().st_size, "sha256": sha256(output),
            "contact_sheet": str(run / CONTACT_NAME), "gallery": str(run / GALLERY_NAME), "videos": 10}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIRECTORY_videos.zip beside the run directory")
    parser.add_argument("--check-only", action="store_true", help="Validate finished media without writing artifacts")
    args = parser.parse_args(argv)
    run = args.run_directory.resolve()
    try:
        if args.check_only:
            _, summary, items = collect(run)
            result = {"ready": True, "videos": len(items), "evaluation_status": summary.get("status"),
                      "steps": [item["row"]["steps"] for item in items]}
        else:
            output = (args.output or run.with_name(run.name + "_videos.zip")).resolve()
            result = package(run, output)
    except (NotReady, OSError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
