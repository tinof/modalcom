"""Build side-by-side comparison images from an original and an upscaled video.

Frames are matched by index, not timestamp: the worker pipes decoded frames in
order, so frame N of the input is frame N of the output (the tail may be trimmed
by ffmpeg's -shortest, hence the frame-count check below).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
SEPARATOR = 16
LABEL_HEIGHT = 120
LABEL_BG = (24, 24, 28)
LABEL_FG = (255, 255, 255)
SEPARATOR_COLOR = (255, 255, 255)


def extract_frame(video: Path, index: int, dest: Path) -> Image.Image:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"select=eq(n\\,{index})",
            "-frames:v", "1", "-fps_mode", "passthrough",
            str(dest),
        ],
        check=True,
    )
    if not dest.exists():
        raise RuntimeError(f"ffmpeg produced no frame {index} from {video}")
    return Image.open(dest).convert("RGB")


def frame_count(video: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def compose(left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
    width, height = left.size
    canvas = Image.new("RGB", (width * 2 + SEPARATOR, height + LABEL_HEIGHT), SEPARATOR_COLOR)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, canvas.width, LABEL_HEIGHT], fill=LABEL_BG)
    canvas.paste(left, (0, LABEL_HEIGHT))
    canvas.paste(right, (width + SEPARATOR, LABEL_HEIGHT))

    font = ImageFont.truetype(FONT_PATH, max(28, height // 22))
    for label, centre in ((left_label, width // 2), (right_label, width + SEPARATOR + width // 2)):
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (centre - (box[2] - box[0]) // 2, (LABEL_HEIGHT - (box[3] - box[1])) // 2 - box[1]),
            label, fill=LABEL_FG, font=font,
        )
    return canvas


def amplified_difference(baseline: Image.Image, candidate: Image.Image, gain: int) -> Image.Image:
    """Per-pixel |candidate - baseline|, amplified and heat-tinted.

    Blue/black is untouched, yellow/white is where the upscaler changed the most.
    """
    delta = np.abs(
        np.asarray(candidate, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    ).mean(axis=2)
    intensity = np.clip(delta * gain, 0, 255) / 255.0
    heat = np.stack(
        [
            np.clip(intensity * 2.2, 0, 1),
            np.clip(intensity * 1.4, 0, 1),
            np.clip(0.35 - intensity * 0.35, 0, 1) + np.clip(intensity - 0.6, 0, 1),
        ],
        axis=2,
    )
    return Image.fromarray((heat * 255).astype(np.uint8), mode="RGB")


def centre_crop(image: Image.Image, size: int) -> Image.Image:
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--upscaled", required=True, type=Path)
    parser.add_argument("--frames", default="25,100,200,290", help="Comma-separated frame indices")
    parser.add_argument("--outdir", default=Path("outputs/compare"), type=Path)
    parser.add_argument("--crop", type=int, default=480, help="Centre-crop size, in original pixels")
    parser.add_argument("--diff-gain", type=int, default=8, help="Amplification for the difference map")
    parser.add_argument("--no-diff", action="store_false", dest="diff", default=True)
    args = parser.parse_args()

    indices = [int(part) for part in args.frames.split(",") if part.strip()]
    available = min(frame_count(args.original), frame_count(args.upscaled))
    too_high = [n for n in indices if n >= available]
    if too_high:
        sys.exit(f"Frame indices {too_high} exceed the {available} frames common to both files.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    scale = None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index in indices:
            original = extract_frame(args.original, index, tmp_dir / f"o{index}.png")
            upscaled = extract_frame(args.upscaled, index, tmp_dir / f"u{index}.png")

            if scale is None:
                scale = upscaled.width / original.width
                print(f"Original {original.size}, upscaled {upscaled.size} ({scale:g}x)")

            # Match sizes so the halves are comparable; the original side is a
            # plain lanczos resize, labelled as such.
            original_matched = original.resize(upscaled.size, Image.LANCZOS)
            full = compose(
                original_matched, upscaled,
                f"Original {original.height}p (lanczos {scale:g}x)",
                f"RTX VSR {scale:g}x (native {upscaled.height}p)",
            )
            full_path = args.outdir / f"frame{index:04d}_full.png"
            full.save(full_path)

            # Detail crop: resizing a small region is where VSR visibly differs.
            crop_size = int(args.crop * scale)
            crop = compose(
                centre_crop(original, args.crop).resize((crop_size, crop_size), Image.LANCZOS),
                centre_crop(upscaled, crop_size),
                f"Original (lanczos {scale:g}x)",
                f"RTX VSR {scale:g}x",
            )
            crop_path = args.outdir / f"frame{index:04d}_crop.png"
            crop.save(crop_path)
            written = [full_path.name, crop_path.name]

            if args.diff:
                diff = compose(
                    upscaled,
                    amplified_difference(original_matched, upscaled, args.diff_gain),
                    f"RTX VSR {scale:g}x",
                    f"Difference vs lanczos (x{args.diff_gain})",
                )
                diff_path = args.outdir / f"frame{index:04d}_diff.png"
                diff.save(diff_path)
                written.append(diff_path.name)

            print(f"frame {index}: {', '.join(written)}")

    print(f"\nWrote images for {len(indices)} frames to {args.outdir}")


if __name__ == "__main__":
    run()
