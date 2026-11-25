"""Local client of the deployed FlashVSR+ app (`modal deploy -m modal_app.plus_deploy`).

    python client.py upload film.mkv
    python client.py submit inputs/film.mkv --job film-turbo --mode turbo --wait
    python client.py status fc-...
    python client.py fetch film-turbo /path/to/film.2160p.mkv
    python client.py look-test film.mkv --starts 120 900 2400 --out /path/to/bench

Run it from this directory with a Python that has `modal` and `numpy`
(`uv run --with modal --with numpy python client.py ...`). Every command spends
money on Modal except `status`, `report`, `fetch` and `clean`.

`look-test` is the first thing to run on new material. It cuts short excerpts,
renders each with several presets and scales through the same worker a full job
uses, and downloads the results, so the look and the measured speed are known
before anything long is started.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from modal_app.config import (
    PLUS_APP_NAME,
    PLUS_ATTN_KERNELS,
    PLUS_IO_VOLUME_NAME,
    PLUS_JOBS_PREFIX,
    PLUS_MODES,
    PLUS_TINY_DECODER_MODES,
    VIDEO_EXTENSIONS,
    validate_plus_params,
)

INPUTS_PREFIX = "inputs"


def _modal():
    try:
        import modal
    except ImportError:
        sys.exit("The `modal` package is missing. Try: uv run --with modal --with numpy python client.py ...")
    return modal


def _volume():
    return _modal().Volume.from_name(PLUS_IO_VOLUME_NAME)


def upload(local: Path, remote: str) -> str:
    if local.suffix.lower() not in VIDEO_EXTENSIONS:
        sys.exit(f"Unsupported extension {local.suffix!r}; expected one of {sorted(VIDEO_EXTENSIONS)}.")
    print(f"Uploading {local} ({local.stat().st_size / 1e9:.2f} GB) -> {PLUS_IO_VOLUME_NAME}:{remote}")
    with _volume().batch_upload(force=True) as batch:
        batch.put_file(str(local), remote)
    return remote


def spawn_job(job_id: str, input_path: str, params: dict, planning: dict,
              start: float | None, duration: float | None) -> str:
    run_job = _modal().Function.from_name(PLUS_APP_NAME, "run_job")
    call = run_job.spawn(job_id, input_path, params, planning, start, duration)
    return call.object_id


def poll(call_id: str) -> dict | None:
    """The job's report, or None while it runs. Raises if the job failed."""
    modal = _modal()
    try:
        return modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        return None


def wait_for(call_ids: dict[str, str], interval: float = 30.0) -> dict[str, dict]:
    """Block until every named call has finished; returns reports and errors by name."""
    results: dict[str, dict] = {}
    pending = dict(call_ids)
    while pending:
        for name, call_id in list(pending.items()):
            try:
                report = poll(call_id)
            except Exception as exc:  # noqa: BLE001 - a failed job must not hide the others
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
                del pending[name]
                print(f"{name}: FAILED: {exc}")
                continue
            if report is not None:
                results[name] = report
                del pending[name]
                print(f"{name}: done, {report['frames']} frames, "
                      f"{report['frames_per_gpu_second']} frames per GPU-second")
        if pending:
            time.sleep(interval)
    return results


def fetch(job_id: str, destination: Path) -> None:
    remote = f"{PLUS_JOBS_PREFIX}/{job_id}/output.mkv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    written = 0
    with partial.open("wb") as handle:
        for chunk in _volume().read_file(remote):
            handle.write(chunk)
            written += len(chunk)
    partial.replace(destination)
    print(f"Saved {destination} ({written / 1e9:.2f} GB)")


def read_report(job_id: str) -> dict:
    data = b"".join(_volume().read_file(f"{PLUS_JOBS_PREFIX}/{job_id}/report.json"))
    return json.loads(data)


def _params_from(args: argparse.Namespace) -> dict:
    return validate_plus_params(
        mode=args.mode,
        scale=args.scale,
        deliver_scale=args.deliver_scale,
        seed=args.seed,
        attn_kernel=args.attn_kernel,
        tiny_decoder_mode=args.tiny_decoder_mode,
        color_correction=args.color_correction,
        resident=args.resident,
        rope_float32=args.rope_float32,
        compile_dit=args.compile_dit,
        bypass_model=args.bypass_model,
    )


def _planning_from(args: argparse.Namespace) -> dict:
    planning = {}
    for name in ("segment_seconds", "search_seconds", "scene_threshold"):
        value = getattr(args, name)
        if value is not None:
            planning[name] = value
    return planning


def _summary_row(name: str, report: dict, project_frames: int, gpu_hour_usd: float) -> str:
    if "error" in report:
        return f"{name:<34} FAILED  {report['error'][:80]}"
    rate = report.get("frames_per_gpu_second") or 0.0
    hours = project_frames / rate / 3600 if rate else float("nan")
    rates = report.get("megabits_per_second") or [0.0]
    return (
        f"{name:<34} {rate:>6.2f} fps  util {report.get('gpu_utilization_mean')!s:>5}%  "
        f"VRAM {report.get('peak_vram_gib')!s:>5} GiB  {sum(rates) / len(rates):>5.1f} Mbit/s  "
        f"-> {hours:>5.1f} GPU-h, ${hours * gpu_hour_usd:>6.0f} for {project_frames} frames"
    )


def command_upload(args: argparse.Namespace) -> None:
    local = Path(args.file)
    print(upload(local, f"{INPUTS_PREFIX}/{args.name or local.name}"))


def command_submit(args: argparse.Namespace) -> None:
    params, planning = _params_from(args), _planning_from(args)
    call_id = spawn_job(args.job, args.input, params, planning, args.start, args.duration)
    print(f"job {args.job}: call {call_id}")
    if args.wait:
        report = wait_for({args.job: call_id})[args.job]
        print(json.dumps({key: value for key, value in report.items() if key != "segment_reports"}, indent=1))
        if "error" in report:
            sys.exit(1)


def command_status(args: argparse.Namespace) -> None:
    report = poll(args.call_id)
    if report is None:
        print("pending")
        return
    print(json.dumps({key: value for key, value in report.items() if key != "segment_reports"}, indent=1))


def command_report(args: argparse.Namespace) -> None:
    print(json.dumps(read_report(args.job), indent=1))


def command_fetch(args: argparse.Namespace) -> None:
    fetch(args.job, Path(args.destination))


def command_clean(args: argparse.Namespace) -> None:
    """Remove a job's segment files. The joined output and the report stay."""
    target = f"{PLUS_JOBS_PREFIX}/{args.job}/segments"
    _volume().remove_file(target, recursive=True)
    print(f"Removed {PLUS_IO_VOLUME_NAME}:{target}")


def command_look_test(args: argparse.Namespace) -> None:
    from modal_app.plus_video import probe_source

    source = Path(args.file)
    out = Path(args.out)
    (out / "src").mkdir(parents=True, exist_ok=True)
    info = probe_source(str(source))

    # Each excerpt is re-encoded losslessly and frame-exactly, so this worker and
    # any other upscaler are given the same file with the source's own pixels.
    # A stream copy would be cheaper but leaves timestamp holes at the cut when
    # the source has B-frames, and the worker refuses such a file.
    excerpts: list[tuple[str, Path]] = []
    for number, start in enumerate(args.starts):
        path = out / "src" / f"excerpt_{number}.mkv"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{args.seconds:.3f}",
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "fast", "-qp", "0", "-c:a", "copy", str(path),
            ],
            check=True,
        )
        cut = probe_source(str(path))
        print(f"excerpt {number}: {cut.frames} frames from {start:.2f}s -> {path}")
        excerpts.append((f"excerpt_{number}", path))

    stem = "".join(char if char.isalnum() else "-" for char in source.stem)[:40].strip("-").lower()
    deliver = max(args.scales)
    calls: dict[str, str] = {}
    for name, path in excerpts:
        remote = upload(path, f"{INPUTS_PREFIX}/look/{stem}/{path.name}")
        for mode in args.modes:
            for scale in args.scales:
                params = validate_plus_params(
                    mode=mode, scale=scale,
                    deliver_scale=deliver if scale != deliver else None,
                    resident=args.resident,
                )
                run = f"{mode}-x{scale:g}"
                job_id = f"look-{stem}-{name}-{run}"
                calls[f"{run}/{name}"] = spawn_job(job_id, remote, params, {}, None, None)
    print(f"Started {len(calls)} jobs; waiting.")
    results = wait_for(calls)

    project_frames = args.project_frames or info.frames
    lines = []
    for key in sorted(results):
        run, name = key.split("/")
        report = results[key]
        if "error" not in report:
            fetch(report["job_id"], out / run / f"{name}.mkv")
        lines.append(_summary_row(key, report, project_frames, args.gpu_hour_usd))
    (out / "report.json").write_text(json.dumps(results, indent=1))
    print("\n" + "\n".join(lines))
    print(f"\nReports: {out / 'report.json'}. Projections assume ${args.gpu_hour_usd}/GPU-hour "
          "and the short-clip rate, which includes each stream's start-up.")
    print("Next: run the same files in src/ through rtx-vsr (rtx-vsr/modal_client.py --input ...) "
          "and compare stills with rtx-vsr/scripts/compare_frames.py. The excerpts are lossless "
          "H.264 (High 4:4:4 Predictive), which NVDEC cannot decode: rtx-vsr falls back to its "
          "CPU decoder for them, with the same result.")
    if any("error" in report for report in results.values()):
        sys.exit(1)


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", default="turbo", choices=PLUS_MODES)
    parser.add_argument("--scale", type=float, default=2.0, help="Size the model renders at")
    parser.add_argument("--deliver-scale", type=float, default=None,
                        help="Resample to this scale in the encoder (e.g. render 1.5, deliver 2)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--attn-kernel", default="auto", choices=PLUS_ATTN_KERNELS)
    parser.add_argument("--tiny-decoder-mode", default="auto", choices=PLUS_TINY_DECODER_MODES)
    parser.add_argument("--color-correction", action="store_true",
                        help="Upstream's wavelet colour correction (--cf)")
    parser.add_argument("--no-resident", dest="resident", action="store_false",
                        help="Stock upstream transport: DiT and decoder output go through the CPU")
    parser.add_argument("--rope-float32", action="store_true", help="Benchmark option, changes the output")
    parser.add_argument("--compile-dit", action="store_true", help="Benchmark option: torch.compile the DiT")
    parser.add_argument("--bypass-model", action="store_true",
                        help="Bicubic only: checks the decode/encode path without the model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("upload", help="Copy a local video to the io Volume under inputs/")
    sub.add_argument("file")
    sub.add_argument("--name", help="Name on the Volume (default: the file's name)")
    sub.set_defaults(run=command_upload)

    sub = commands.add_parser("submit", help="Start a job on a file that is on the Volume")
    sub.add_argument("input", help="Path on the Volume, e.g. inputs/film.mkv")
    sub.add_argument("--job", required=True, help="Job id; the same id resumes an interrupted job")
    _add_model_flags(sub)
    sub.add_argument("--start", type=float, default=None, help="Excerpt start in seconds")
    sub.add_argument("--duration", type=float, default=None, help="Excerpt length in seconds")
    sub.add_argument("--segment-seconds", type=float, default=None)
    sub.add_argument("--search-seconds", type=float, default=None)
    sub.add_argument("--scene-threshold", type=float, default=None,
                     help="Above 1 no cut qualifies, so every seam is a cross-faded soft seam")
    sub.add_argument("--wait", action="store_true")
    sub.set_defaults(run=command_submit)

    sub = commands.add_parser("status", help="Poll a call id")
    sub.add_argument("call_id")
    sub.set_defaults(run=command_status)

    sub = commands.add_parser("report", help="Print a finished job's full report")
    sub.add_argument("job")
    sub.set_defaults(run=command_report)

    sub = commands.add_parser("fetch", help="Download a job's output.mkv")
    sub.add_argument("job")
    sub.add_argument("destination")
    sub.set_defaults(run=command_fetch)

    sub = commands.add_parser("clean", help="Delete a job's segment files from the Volume")
    sub.add_argument("job")
    sub.set_defaults(run=command_clean)

    sub = commands.add_parser("look-test", help="Render short excerpts with several presets and scales")
    sub.add_argument("file")
    sub.add_argument("--starts", type=float, nargs="+", required=True, help="Excerpt start times in seconds")
    sub.add_argument("--seconds", type=float, default=8.0)
    sub.add_argument("--modes", nargs="+", default=list(PLUS_MODES), choices=PLUS_MODES)
    sub.add_argument("--scales", type=float, nargs="+", default=[2.0, 1.5],
                     help="Render scales; all are delivered at the largest")
    sub.add_argument("--out", required=True, help="Local folder for excerpts, results and the report")
    sub.add_argument("--no-resident", dest="resident", action="store_false")
    sub.add_argument("--project-frames", type=int, default=None,
                     help="Frames to project the cost for (default: the whole input file)")
    sub.add_argument("--gpu-hour-usd", type=float, default=3.5)
    sub.set_defaults(run=command_look_test)

    args = parser.parse_args()
    try:
        args.run(args)
    except ValueError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
