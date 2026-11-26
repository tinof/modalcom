"""A FlashVSR+ job on disk: prepare it, render its seams, finish it.

    <root>/<input_path>                 the source, with <input_path>.scenes.json beside it
    <root>/jobs/<job_id>/job.json       source facts, canvas, plan, parameters, digest
    <root>/jobs/<job_id>/segments/      one mp4 per segment and seam, sidecars, markers
    <root>/jobs/<job_id>/output.mkv     the joined result
    <root>/jobs/<job_id>/report.json    what ran and how fast

No Modal and no torch here: `root` is the io Volume's mount point in a container
and any directory in a test. The Modal functions in plus_episode.py and
plus_service.py only add `reload()` and `commit()` around these calls.

A job is resumable. Planning is deterministic, and a part whose marker carries
the job's digest is skipped, so running the same job again after a failure pays
only for what is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict

from .config import (
    PLUS_CANVAS_MULTIPLE,
    PLUS_ENV,
    PLUS_JOBS_PREFIX,
    PLUS_REPO_SHA,
    PLUS_SCENE_THRESHOLD,
    PLUS_SEAM_OVERLAP_FRAMES,
    PLUS_SEAM_WARMUP_FRAMES,
    PLUS_SEGMENT_SEARCH_SECONDS,
    PLUS_SEGMENT_SECONDS,
    validate_plus_params,
)
from .plus_plan import Canvas, Plan, Seam, plan_canvas, plan_segments, stable_hash
from .plus_video import (
    SourceInfo,
    blend_seam,
    canvas_deliver,
    encode_frames,
    file_digest,
    fingerprint,
    frame_range,
    join_parts,
    probe_source,
    scene_scores,
    seam_files,
    segment_files,
    stream_summary,
)

PLANNING_DEFAULTS = {
    "segment_seconds": PLUS_SEGMENT_SECONDS,
    "search_seconds": PLUS_SEGMENT_SEARCH_SECONDS,
    "scene_threshold": PLUS_SCENE_THRESHOLD,
    "overlap_frames": PLUS_SEAM_OVERLAP_FRAMES,
    "warmup_frames": PLUS_SEAM_WARMUP_FRAMES,
}


def job_dir(root: str, job_id: str) -> str:
    return os.path.join(root, PLUS_JOBS_PREFIX, job_id)


def parts_dir(root: str, job_id: str) -> str:
    return os.path.join(job_dir(root, job_id), "segments")


def load_job(root: str, job_id: str) -> dict:
    with open(os.path.join(job_dir(root, job_id), "job.json")) as handle:
        return json.load(handle)


def marker_matches(marker_path: str, digest: str, required: list[str]) -> dict | None:
    """The saved report of an earlier identical run, if all its files are still there."""
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path) as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return None
    if saved.get("digest") != digest or not all(os.path.exists(path) for path in required):
        return None
    return saved


def publish(local_path: str, final_path: str) -> None:
    """Copy under a temporary name, then rename: a file under its final name is complete."""
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    shutil.copyfile(local_path, f"{final_path}.part")
    os.replace(f"{final_path}.part", final_path)


def publish_json(data: dict, final_path: str) -> None:
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with open(f"{final_path}.part", "w") as handle:
        json.dump(data, handle)
    os.replace(f"{final_path}.part", final_path)


def _load_scores(source_path: str, source_digest: str, frames: int) -> list[float]:
    """Scene scores for the whole source, cached beside it."""
    cache_path = f"{source_path}.scenes.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as handle:
                cached = json.load(handle)
            if cached.get("digest") == source_digest and len(cached.get("scores", [])) == frames:
                return cached["scores"]
        except (OSError, ValueError):
            pass
    started = time.monotonic()
    scores = [round(score, 4) for score in scene_scores(source_path, frames)]
    print(f"Scene pass over {frames} frames took {time.monotonic() - started:.0f}s.")
    publish_json({"digest": source_digest, "scores": scores}, cache_path)
    return scores


def prepare_job(
    root: str,
    job_id: str,
    input_path: str,
    params: dict | None = None,
    planning: dict | None = None,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> dict:
    """Probe the source, plan the segments and write job.json. Returns the job.

    `params` are the model parameters (`validate_plus_params`), `planning`
    overrides the planner's defaults. `start_seconds` and `duration_seconds`
    select an excerpt.
    """
    params = validate_plus_params(**(params or {}))
    unknown = set(planning or {}) - set(PLANNING_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown planning options: {sorted(unknown)}")
    planning = {**PLANNING_DEFAULTS, **(planning or {})}
    if not job_id or "/" in job_id or job_id.startswith("."):
        raise ValueError(f"Invalid job id {job_id!r}.")
    if input_path.startswith("/") or ".." in input_path.split("/"):
        raise ValueError("input_path must be a relative path inside the Volume.")
    source_path = os.path.join(root, input_path)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Input not found on the io Volume: {input_path}")

    info = probe_source(source_path)
    first, last = frame_range(info, start_seconds, duration_seconds)
    source_digest = file_digest(source_path)
    scores = _load_scores(source_path, source_digest, info.frames)
    plan = plan_segments(
        first, last, scores,
        fps=info.fps,
        segment_seconds=float(planning["segment_seconds"]),
        search_seconds=float(planning["search_seconds"]),
        threshold=float(planning["scene_threshold"]),
        overlap_frames=int(planning["overlap_frames"]),
        warmup_frames=int(planning["warmup_frames"]),
    )
    canvas = plan_canvas(
        info.width, info.height, params["scale"],
        multiple=PLUS_CANVAS_MULTIPLE, deliver_scale=params["deliver_scale"],
    )
    # Everything that changes a part's bytes. A marker with another digest
    # belongs to a different job and is rendered again.
    digest = stable_hash({
        "source": source_digest,
        "plan": plan.to_json(),
        "canvas": asdict(canvas),
        "params": params,
        "encoder": {key: value for key, value in PLUS_ENV.items()
                    if "NVENC" in key or "ENCODER" in key},
        "upstream": PLUS_REPO_SHA,
    })
    job = {
        "job_id": job_id,
        "input_path": input_path,
        "digest": digest,
        "params": params,
        "planning": planning,
        "source": info.to_json(),
        "canvas": asdict(canvas),
        "plan": plan.to_json(),
    }
    os.makedirs(parts_dir(root, job_id), exist_ok=True)
    publish_json(job, os.path.join(job_dir(root, job_id), "job.json"))
    hard = len(plan.segments) - 1 - len(plan.seams)
    print(
        f"Job {job_id}: frames [{first}, {last}) of {info.frames}, "
        f"{info.width}x{info.height} -> {canvas.out_w}x{canvas.out_h} on a "
        f"{canvas.canvas_w}x{canvas.canvas_h} canvas, delivered {canvas.deliver_w}x{canvas.deliver_h}; "
        f"{len(plan.segments)} segments ({hard} hard cuts, {len(plan.seams)} soft seams)."
    )
    return job


def render_seam(root: str, job_id: str, seam: dict, digest: str, *, encoder: str, host: str) -> dict:
    """Cross-fade the overlap between two finished segments and encode it.

    The seam must come from the same encoder as the segments around it: a
    stream-copy join needs identical parameter sets.
    """
    import numpy as np

    job = load_job(root, job_id)
    info = SourceInfo.from_json(job["source"])
    canvas = Canvas(**job["canvas"])
    folder = parts_dir(root, job_id)
    item = Seam(**seam)
    names = seam_files(item)
    target = os.path.join(folder, names["video"])
    marker = os.path.join(folder, names["marker"])
    cached = marker_matches(marker, digest, [target])
    if cached is not None:
        return {**cached, "skipped": True}

    segments = {segment.index: segment for segment in Plan.from_json(job["plan"]).segments}
    tail = np.load(os.path.join(folder, segment_files(segments[item.index])["tail"]))
    head = np.load(os.path.join(folder, segment_files(segments[item.index + 1])["head"]))
    if len(tail) != item.frames:
        raise RuntimeError(f"{item.name()}: sidecar has {len(tail)} frames, planned {item.frames}.")
    with tempfile.TemporaryDirectory() as scratch:
        local = os.path.join(scratch, names["video"])
        encode_frames(blend_seam(tail, head), info.fps, local,
                      encoder=encoder, deliver=canvas_deliver(canvas))
        found = fingerprint(local)
        if found.frames != item.frames:
            raise RuntimeError(
                f"{item.name()}: the encoded file has {found.frames} frames, planned {item.frames}."
            )
        report = {"seam": item.name(), "digest": digest, "frames": item.frames,
                  "extradata_md5": found.extradata_md5, "gpu": host, "skipped": False}
        publish(local, target)
    publish_json(report, marker)
    return report


def raise_failures(kind: str, names: list[str], results: list) -> None:
    """Fail the job if any fanned-out call returned an exception."""
    failed = [(name, result) for name, result in zip(names, results) if isinstance(result, BaseException)]
    if failed:
        detail = "; ".join(f"{name}: {type(error).__name__}: {error}" for name, error in failed[:5])
        raise RuntimeError(
            f"{len(failed)} of {len(names)} {kind} failed (finished ones are kept and are "
            f"skipped when the job is run again). {detail}"
        )


def finish_job(root: str, job_id: str, segment_reports: list[dict], seam_reports: list[dict],
               started: float) -> dict:
    """Verify and join the parts, write report.json, return the report."""
    job = load_job(root, job_id)
    info = SourceInfo.from_json(job["source"])
    plan = Plan.from_json(job["plan"])
    base = job_dir(root, job_id)
    output_path = os.path.join(base, "output.mkv")
    partial = os.path.join(base, "output.part.mkv")
    joined = join_parts(plan, parts_dir(root, job_id), partial, info,
                        source_path=os.path.join(root, job["input_path"]))
    os.replace(partial, output_path)
    joined["output"] = os.path.relpath(output_path, root)

    rendered = [report for report in segment_reports if not report.get("skipped")]
    gpu_seconds = sum(report["seconds"] for report in rendered)
    model_frames = sum(report["model_frames"] for report in rendered)
    utilization = [report["gpu_samples"]["utilization_mean"] for report in rendered
                   if report.get("gpu_samples", {}).get("samples")]
    vram = [report["torch_peak_vram_gib"] for report in rendered
            if report.get("torch_peak_vram_gib") is not None]
    report = {
        "job_id": job_id,
        "input_path": job["input_path"],
        "digest": job["digest"],
        "params": job["params"],
        "planning": job["planning"],
        "frames": plan.total_frames,
        "segments": len(plan.segments),
        "soft_seams": len(plan.seams),
        "segments_skipped": len(segment_reports) - len(rendered),
        # Summed over workers: what the job cost, not how long it took.
        "gpu_seconds": round(gpu_seconds, 1),
        "frames_per_gpu_second": round(model_frames / gpu_seconds, 3) if gpu_seconds else None,
        "gpu_utilization_mean": round(sum(utilization) / len(utilization), 1) if utilization else None,
        "peak_vram_gib": max(vram, default=None),
        "megabits_per_second": [report["megabits_per_second"] for report in segment_reports
                                if "megabits_per_second" in report],
        "hosts": sorted({report["gpu"] for report in [*segment_reports, *seam_reports]}),
        "wall_seconds": round(time.monotonic() - started, 1),
        "output": joined,
        "streams": stream_summary(output_path),
        "segment_reports": segment_reports,
    }
    publish_json(report, os.path.join(base, "report.json"))
    print(
        f"Job {job_id} done: {report['frames']} frames, {report['gpu_seconds']} GPU-seconds "
        f"({report['frames_per_gpu_second']} frames per GPU-second), output {joined['output']}."
    )
    return report
