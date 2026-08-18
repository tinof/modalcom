"""Video I/O helpers extracted from infer.py for Modal deployment.

These functions mirror the original infer.py implementations but are adapted
for serverless execution (paths adjusted for Volumes, NVENC probed at runtime).
"""

import functools
import os
import re
import subprocess

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True when ffmpeg can actually open an NVENC session on this GPU.

    An encoder appearing in `ffmpeg -encoders` only proves the build has the
    code; A100/B200 have no NVENC engine and fail at session open. So probe with
    a real 64x64 encode. Cached per container — the GPU does not change.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=black:size=64x64:duration=0.1",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[encode] NVENC probe failed to run ({exc}); using CPU libx264.")
        return False
    if result.returncode == 0:
        print("[encode] NVENC available: encoding with h264_nvenc.")
        return True
    print(
        "[encode] NVENC unusable on this GPU "
        f"({result.stderr.decode(errors='ignore').strip().splitlines()[-1:] or 'no detail'}); "
        "falling back to CPU libx264 (roughly an order of magnitude slower at 4K)."
    )
    return False


def _video_encoding_args(quality: int) -> list[str]:
    """ffmpeg video-encoder args for the requested 1-10 quality level."""
    crf = int(26 - quality * 0.6)
    if nvenc_available():
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", str(crf),
            "-b:v", "0",
        ]
    return [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-tune", "film",
    ]


def tensor2video(frames: torch.Tensor) -> list[np.ndarray]:
    """Convert tensor (C, T, H, W) or (B, C, T, H, W) to list of numpy arrays (uint8)."""
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [frame for frame in frames]


def natural_key(name: str):
    """Natural sort key for filenames."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str) -> list[str]:
    """List image files with natural sorting."""
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def largest_8n1_leq(n: int) -> int:
    """Find largest 8n+1 <= n."""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def is_video(path: str) -> bool:
    """Check if path is a video file."""
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 2.0, multiple: int = 128) -> tuple[int, int, int, int]:
    """Compute scaled dimensions and target dimensions (padded to multiple)."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    # Pad UP to multiple (no resolution loss)
    tW = ((sW + multiple - 1) // multiple) * multiple
    tH = ((sH + multiple - 1) // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def process_batch_gpu(
    batch_arr,
    sH: int,
    sW: int,
    tH: int,
    tW: int,
    dtype=torch.bfloat16,
    device: str = "cuda",
) -> torch.Tensor:
    """Process a batch of frames on GPU: upscale, center-pad, normalize."""
    if isinstance(batch_arr, list):
        if len(batch_arr) > 0 and isinstance(batch_arr[0], Image.Image):
            batch_arr = np.stack([np.array(img) for img in batch_arr])
        else:
            batch_arr = np.stack(batch_arr)

    t = torch.from_numpy(batch_arr).to(device=device, dtype=dtype)
    t = t.permute(0, 3, 1, 2)  # (B, C, H, W)

    # Upscale
    if t.shape[2] != sH or t.shape[3] != sW:
        t = torch.nn.functional.interpolate(t, size=(sH, sW), mode="bicubic", align_corners=False)

    # Pad to target size
    curr_h, curr_w = t.shape[2], t.shape[3]
    pad_h = tH - curr_h
    pad_w = tW - curr_w

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        t = torch.nn.functional.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    # Normalize [0, 255] -> [-1, 1]
    t = t / 255.0 * 2.0 - 1.0
    return t


def process_frame_gpu(img_or_arr, sH: int, sW: int, tH: int, tW: int, dtype=torch.bfloat16, device: str = "cuda") -> torch.Tensor:
    """Process a single frame on GPU."""
    return process_batch_gpu([img_or_arr], sH, sW, tH, tW, dtype, device).squeeze(0)


def prepare_input_tensor(
    path: str,
    scale: float = 2,
    dtype=torch.bfloat16,
    device: str = "cuda",
) -> tuple:
    """Prepare input tensor from video or image sequence (GPU accelerated).

    Returns:
        (vid, tH, tW, F, fps, input_video_path, total_frames_orig, sH, sW)
    """
    if os.path.isdir(path):
        # Image sequence
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"Input: {w0}x{h0}, {N0} frames")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)

        # Pad to 8n+1
        k = (N0 - 1 + 7) // 8
        F = k * 8 + 1
        pad_len = F - N0
        paths = paths0 + [paths0[-1]] * pad_len

        frames = []
        for p in paths:
            with Image.open(p).convert("RGB") as img:
                frame_tensor = process_frame_gpu(img, sH, sW, tH, tW, dtype, device)
            frames.append(frame_tensor)

        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        fps = 30
        return vid, tH, tW, F, fps, None, N0, sH, sW

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert("RGB")
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass

        fps_val = meta.get("fps", 30)
        fps = float(fps_val) if isinstance(fps_val, (int, float)) else 30.0

        def count_frames(r):
            try:
                nf = meta.get("nframes", None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        # Refine FPS from duration
        duration_sec = meta.get("duration", 0)
        if duration_sec > 0:
            calc_fps = total / duration_sec
            if abs(calc_fps - fps) / (fps + 1e-6) < 0.1:
                fps = calc_fps

        print(f"Input: {w0}x{h0}, {total} frames, {fps:.3f} FPS")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)

        idx = list(range(total))
        k = (total - 1 + 7) // 8
        F = k * 8 + 1
        pad_len = F - total
        idx = idx + [total - 1] * pad_len

        frames = []
        batch_size = 32
        batch_buffer = []

        try:
            for i in idx:
                img_arr = rdr.get_data(i)
                batch_buffer.append(img_arr)

                if len(batch_buffer) >= batch_size:
                    batch_tensor = process_batch_gpu(batch_buffer, sH, sW, tH, tW, dtype, device)
                    frames.append(batch_tensor)
                    batch_buffer = []

            if batch_buffer:
                batch_tensor = process_batch_gpu(batch_buffer, sH, sW, tH, tW, dtype, device)
                frames.append(batch_tensor)
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.cat(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        return vid, tH, tW, F, fps, path, total, sH, sW

    raise ValueError(f"Unsupported input: {path}")


def save_video_with_audio_piped(
    frames: list,
    output_path: str,
    audio_source: str,
    fps: float = 30,
    quality: int = 10,
) -> bool:
    """Save frames as video with audio using an ffmpeg pipe (NVENC when available)."""
    if not frames:
        return False

    if hasattr(frames[0], "shape"):
        h, w = frames[0].shape[:2]
    else:
        w, h = frames[0].size

    encoding_args = _video_encoding_args(quality)

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-i", audio_source,
        "-map", "0:v",
        "-map", "1:a",
        "-pix_fmt", "yuv420p",
        *encoding_args,
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: ffmpeg not found.")
        return False

    for frame in tqdm(frames, desc="Saving"):
        process.stdin.write(np.array(frame).tobytes())

    out, err = process.communicate()

    if process.returncode != 0:
        print(f"FFmpeg Error: {err.decode('utf-8', errors='ignore')}")
        return False

    return True


def save_video(frames: list, save_path: str, fps: float = 30, quality: int = 5):
    """Save frames as video (NVENC when available, else CPU x264 via imageio)."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    crf = int(26 - quality * 0.6)

    if nvenc_available():
        sample_frame = np.array(frames[0])
        height, width = sample_frame.shape[:2]
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-an",
            *_video_encoding_args(quality),
            "-pix_fmt", "yuv420p",
            save_path,
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in tqdm(frames, desc="Saving (NVENC)"):
            process.stdin.write(np.array(f).tobytes())
        _, err = process.communicate()
        if process.returncode == 0:
            return
        # NVENC session errors surface here, not in the tiny probe; fall through
        # to the CPU path so the job still finishes, but never silently.
        print(
            "[encode] NVENC encode failed mid-run, retrying with CPU libx264: "
            f"{err.decode('utf-8', errors='ignore').strip()[-300:]}"
        )

    try:
        w = imageio.get_writer(
            save_path,
            fps=fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            ffmpeg_params=["-preset", "veryfast", "-tune", "film", "-crf", str(crf)],
        )
        for f in tqdm(frames, desc="Saving"):
            w.append_data(np.array(f))
        w.close()
    except Exception as e:
        print(f"Warning: imageio writer failed ({e}), falling back to ffmpeg pipe")
        sample_frame = np.array(frames[0])
        height, width = sample_frame.shape[:2]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "film",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            save_path,
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in tqdm(frames, desc="Streaming to FFmpeg"):
            try:
                process.stdin.write(np.array(f).tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe broken during writing.")
                break

        out, err = process.communicate()
        if process.returncode != 0:
            print(f"FFmpeg Error: {err.decode('utf-8', errors='ignore')}")
