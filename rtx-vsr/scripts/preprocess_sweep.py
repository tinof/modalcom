"""Compare preprocess modes side by side and score detail against noise.

The DEBLUR family sharpens every frequency, so it lifts real texture and compression
grain together. Scoring flat regions separately from edges shows which mode adds detail
without just amplifying noise.
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from compare_frames import FONT_PATH, LABEL_BG, LABEL_FG, centre_crop  # noqa: E402

VARIANTS = [
    ("ULTRA only (baseline)", "outputs/jopet_10s_x2_ultra.mp4"),
    ("DEBLUR_LOW", "outputs/jopet_10s_x2_deblur_low.mp4"),
    ("DEBLUR_MEDIUM", "outputs/jopet_10s_x2_deblur_med.mp4"),
    ("DEBLUR_HIGH", "outputs/jopet_10s_x2_deblur.mp4"),
    ("DENOISE_MEDIUM", "outputs/jopet_10s_x2_denoise_med.mp4"),
    ("DENOISE_HIGH", "outputs/jopet_10s_x2_denoise_high.mp4"),
]


def grab(video: str, index: int) -> Image.Image:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", video,
            "-vf", f"select=eq(n\\,{index})",
            "-frames:v", "1", "-fps_mode", "passthrough",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True, check=True,
    )
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def detail_vs_noise(image: Image.Image) -> tuple[float, float]:
    """Laplacian energy at edges (detail) and in flat regions (noise).

    Pixels are split by local gradient magnitude: strong gradients are real structure,
    weak ones are flat areas where any high-frequency energy is noise.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    lap = np.abs(
        4 * grey[1:-1, 1:-1]
        - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
    )
    gy, gx = np.gradient(grey)
    gradient = np.hypot(gx, gy)[1:-1, 1:-1]
    edge = gradient > np.percentile(gradient, 90)
    flat = gradient < np.percentile(gradient, 50)
    return float(lap[edge].mean()), float(lap[flat].mean())


def build_grid(frame: int, crop: int, columns: int, outdir: Path, variants=VARIANTS, name="preprocess_sweep") -> Path:
    panels = []
    for label, path in variants:
        image = centre_crop(grab(path, frame), crop)
        detail, noise = detail_vs_noise(image)
        panels.append((f"{label}   detail {detail:.2f} / noise {noise:.2f}", image))
        print(f"frame {frame} {label:24s} detail={detail:6.2f}  noise={noise:6.2f}")

    label_height = 64
    cell = crop
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (cell * columns, (cell + label_height) * rows), LABEL_BG)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 26)

    for index, (label, image) in enumerate(panels):
        col, row = index % columns, index // columns
        x, y = col * cell, row * (cell + label_height)
        canvas.paste(image, (x, y + label_height))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x + (cell - (box[2] - box[0])) // 2, y + (label_height - (box[3] - box[1])) // 2 - box[1]),
            label, fill=LABEL_FG, font=font,
        )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}_frame{frame:04d}.png"
    canvas.save(path)
    return path


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="100,200")
    parser.add_argument("--crop", type=int, default=700)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--outdir", default=Path("outputs/compare"), type=Path)
    parser.add_argument("--name", default="preprocess_sweep", help="Output filename prefix")
    parser.add_argument(
        "--variants",
        default="",
        help="Semicolon-separated label=path pairs; defaults to the preprocess sweep set",
    )
    args = parser.parse_args()

    variants = VARIANTS
    if args.variants:
        variants = [tuple(pair.split("=", 1)) for pair in args.variants.split(";") if pair.strip()]

    for frame in (int(part) for part in args.frames.split(",") if part.strip()):
        print(build_grid(frame, args.crop, args.columns, args.outdir, variants, args.name))


if __name__ == "__main__":
    run()
