"""Segment planning and frame arithmetic for the FlashVSR+ worker.

Pure standard library on purpose: the CPU orchestrator, the GPU worker and the
local tests all import this, and none of it needs torch, numpy or Modal.

Why segments at all. FlashVSR+ runs at about one frame per second on a 4K
canvas, so a film is tens of GPU-hours. Cutting it into segments lets several
containers work at once and lets a failed run resume at the segment it lost.

Why the cuts sit on scene changes. Every segment restarts the model's stream
(cold KV cache, fresh noise), so two segments never agree on the fine detail
they synthesise. On a hard cut nobody can tell. Where no cut exists near the
target, the two segments overlap by a few frames that are cross-faded, and the
later one first runs some warm-up frames through the model.

Frame numbers are indices into the source's display order. `plus_video.py`
turns them into exact pts for the decoder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Canvas:
    """Geometry of one job: source, model canvas, and what is delivered."""

    src_w: int
    src_h: int
    # The upscaled picture the model is asked to render.
    out_w: int
    out_h: int
    # out_* padded up to the model's multiple; the padding is reflected picture.
    canvas_w: int
    canvas_h: int
    pad_left: int
    pad_top: int
    # What the encoder writes. Equals out_* unless a deliver scale was given.
    deliver_w: int
    deliver_h: int


def _even(value: float) -> int:
    """Nearest even integer, at least 2: yuv420 needs even sides."""
    return max(2, int(round(value / 2.0)) * 2)


def plan_canvas(
    src_w: int,
    src_h: int,
    scale: float,
    *,
    multiple: int = 128,
    deliver_scale: float | None = None,
) -> Canvas:
    """Pad up to the model multiple instead of cropping, so no picture is lost.

    Upstream's own loader centre-crops to the requested size. For 1080p at 2x
    that would throw away picture to reach 3840x2176; padding 8 reflected rows
    top and bottom and cropping them off again loses nothing.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source size {src_w}x{src_h}.")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    out_w, out_h = _even(src_w * scale), _even(src_h * scale)
    canvas_w = -(-out_w // multiple) * multiple
    canvas_h = -(-out_h // multiple) * multiple
    pad_w, pad_h = canvas_w - out_w, canvas_h - out_h
    # Reflect padding needs the pad to be smaller than the picture.
    if pad_w >= out_w or pad_h >= out_h:
        raise ValueError(
            f"Output {out_w}x{out_h} is too small for a {multiple}-pixel canvas multiple."
        )
    if deliver_scale is None:
        deliver_w, deliver_h = out_w, out_h
    else:
        deliver_w, deliver_h = _even(src_w * deliver_scale), _even(src_h * deliver_scale)
    return Canvas(
        src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
        canvas_w=canvas_w, canvas_h=canvas_h,
        pad_left=pad_w // 2, pad_top=pad_h // 2,
        deliver_w=deliver_w, deliver_h=deliver_h,
    )


@dataclass(frozen=True)
class Segment:
    """One stream of the model and the frames it is responsible for.

    The model is fed source frames [model_start, model_end). Its output splits,
    in order, into: `warmup` frames that are dropped, `head` frames that go to a
    sidecar for the seam before this segment, `video` frames that go to this
    segment's video file, and `tail` frames that go to a sidecar for the seam
    after it.
    """

    index: int
    model_start: int
    model_end: int
    warmup: int
    head: int
    video: int
    tail: int

    @property
    def model_frames(self) -> int:
        return self.model_end - self.model_start

    @property
    def video_start(self) -> int:
        """Source frame number of the first frame in this segment's video file."""
        return self.model_start + self.warmup + self.head

    def name(self) -> str:
        return f"seg_{self.index:05d}"


@dataclass(frozen=True)
class Seam:
    """A cross-faded overlap between segment `index` and segment `index + 1`."""

    index: int
    start: int
    frames: int

    def name(self) -> str:
        return f"seam_{self.index:05d}"


@dataclass(frozen=True)
class Plan:
    first_frame: int
    last_frame: int  # exclusive
    segments: tuple[Segment, ...]
    seams: tuple[Seam, ...]

    @property
    def total_frames(self) -> int:
        return self.last_frame - self.first_frame

    def concat_order(self) -> list[tuple[str, int]]:
        """(file stem, frame count) in playback order, zero-length parts left out."""
        seams = {seam.index: seam for seam in self.seams}
        order: list[tuple[str, int]] = []
        for segment in self.segments:
            if segment.video > 0:
                order.append((segment.name(), segment.video))
            seam = seams.get(segment.index)
            if seam is not None:
                order.append((seam.name(), seam.frames))
        return order

    def to_json(self) -> dict:
        return {
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "segments": [asdict(segment) for segment in self.segments],
            "seams": [asdict(seam) for seam in self.seams],
        }

    @staticmethod
    def from_json(data: dict) -> "Plan":
        return Plan(
            first_frame=int(data["first_frame"]),
            last_frame=int(data["last_frame"]),
            segments=tuple(Segment(**item) for item in data["segments"]),
            seams=tuple(Seam(**item) for item in data["seams"]),
        )


def choose_boundaries(
    first_frame: int,
    last_frame: int,
    scene_scores: dict[int, float] | list[float],
    *,
    target_frames: int,
    search_frames: int,
    threshold: float,
    min_frames: int,
) -> list[tuple[int, bool]]:
    """Cut points inside (first_frame, last_frame) as (frame, is_hard_cut).

    A cut at frame S means S is the first frame of the next segment.
    `scene_scores[n]` is the scene-change score of frame n against frame n-1.
    Walking forward from the previous cut, the next one is the strongest scene
    change at or above `threshold` within `search_frames` of the target length.
    If none qualifies the cut is soft and sits exactly on the target.
    """
    if target_frames <= 0 or search_frames < 0 or min_frames <= 0:
        raise ValueError("target_frames and min_frames must be positive, search_frames >= 0")

    def score(frame: int) -> float:
        if isinstance(scene_scores, dict):
            return float(scene_scores.get(frame, 0.0))
        return float(scene_scores[frame]) if 0 <= frame < len(scene_scores) else 0.0

    boundaries: list[tuple[int, bool]] = []
    previous = first_frame
    # Stop once what is left fits in one segment of at most target + search.
    while last_frame - previous > target_frames + search_frames:
        ideal = previous + target_frames
        low = max(previous + min_frames, ideal - search_frames)
        high = min(last_frame - min_frames, ideal + search_frames)
        if low > high:
            break
        candidates = [frame for frame in range(low, high + 1) if score(frame) >= threshold]
        if candidates:
            # Strongest cut wins; among equals the one nearest the target.
            best_frame = max(candidates, key=lambda frame: (score(frame), -abs(frame - ideal)))
            boundaries.append((best_frame, True))
            previous = best_frame
        else:
            soft = min(max(ideal, low), high)
            boundaries.append((soft, False))
            previous = soft
    return boundaries


def plan_segments(
    first_frame: int,
    last_frame: int,
    scene_scores: dict[int, float] | list[float],
    *,
    fps: Fraction,
    segment_seconds: float,
    search_seconds: float,
    threshold: float,
    overlap_frames: int,
    warmup_frames: int,
) -> Plan:
    """Split [first_frame, last_frame) into segments and the seams between them."""
    if last_frame <= first_frame:
        raise ValueError(f"Empty frame range [{first_frame}, {last_frame}).")
    if overlap_frames < 0 or warmup_frames < 0:
        raise ValueError("overlap_frames and warmup_frames must be >= 0")
    target = max(1, int(round(segment_seconds * fps)))
    search = max(0, int(round(search_seconds * fps)))
    # A segment must be longer than the overlaps it may carry on both sides, and
    # long enough that restarting the model for it is worth it.
    min_frames = max(2 * overlap_frames + 16, int(round(2 * fps)))
    boundaries = choose_boundaries(
        first_frame, last_frame, scene_scores,
        target_frames=target, search_frames=search, threshold=threshold, min_frames=min_frames,
    )

    cuts = [first_frame, *(frame for frame, _ in boundaries), last_frame]
    hard = [True, *(is_hard for _, is_hard in boundaries), True]
    segments: list[Segment] = []
    seams: list[Seam] = []
    for index in range(len(cuts) - 1):
        start, end = cuts[index], cuts[index + 1]
        soft_start, soft_end = not hard[index], not hard[index + 1]
        head = overlap_frames if soft_start else 0
        tail = overlap_frames if soft_end else 0
        # Warm-up frames exist only before a soft start, and never reach back
        # past the first frame of the job.
        warmup = min(warmup_frames, start - first_frame) if soft_start else 0
        video = (end - start) - head
        if video < 0:
            raise ValueError(f"Segment {index} [{start}, {end}) is shorter than its overlap.")
        segments.append(Segment(
            index=index,
            model_start=start - warmup,
            model_end=end + tail,
            warmup=warmup, head=head, video=video, tail=tail,
        ))
        # With an overlap of zero a soft cut is a plain cut after a warm-up.
        if soft_end and overlap_frames > 0:
            seams.append(Seam(index=index, start=end, frames=overlap_frames))
    plan = Plan(first_frame, last_frame, tuple(segments), tuple(seams))
    check_plan(plan)
    return plan


def check_plan(plan: Plan) -> None:
    """Fail unless the plan delivers every frame exactly once, in order."""
    position = plan.first_frame
    seams = {seam.index: seam for seam in plan.seams}
    for segment in plan.segments:
        if segment.model_start < plan.first_frame or segment.model_end > plan.last_frame:
            raise ValueError(f"{segment.name()} reads outside the job's frame range.")
        if segment.warmup + segment.head + segment.video + segment.tail != segment.model_frames:
            raise ValueError(f"{segment.name()} parts do not add up to its model frames.")
        # After a seam, `position` already includes the head frames of this segment.
        if segment.video_start != position:
            raise ValueError(f"{segment.name()} does not continue at frame {position}.")
        position += segment.video
        seam = seams.get(segment.index)
        if seam is not None:
            if seam.start != position or seam.frames != segment.tail:
                raise ValueError(f"{seam.name()} does not line up with {segment.name()}.")
            position += seam.frames
        elif segment.tail:
            raise ValueError(f"{segment.name()} has a tail overlap and no seam.")
    if position != plan.last_frame:
        raise ValueError(f"Plan delivers frames up to {position}, expected {plan.last_frame}.")
    if sum(count for _, count in plan.concat_order()) != plan.total_frames:
        raise ValueError("Concat order does not add up to the job's frame count.")


def seam_weights(frames: int) -> list[float]:
    """Weight of the later segment for each cross-faded frame, strictly inside (0, 1)."""
    return [(index + 1) / (frames + 1) for index in range(frames)]


def stable_hash(payload: dict) -> str:
    """Short digest of a JSON-serializable dict, for resume markers."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Frame arithmetic of the original FlashVSR v1.1 streaming pipelines
# (diffsynth/pipelines/flashvsr_tiny*.py). Kept here, torch-free, so the fix for
# the dropped tail frames can be tested without a GPU.
# ---------------------------------------------------------------------------
V1_MIN_PADDED_FRAMES = 25


def v1_emitted_frames(padded_frames: int) -> int:
    """Frames the v1.1 streaming loop emits for an input padded to 8k+1 frames.

    The loop runs (F-1)//8 - 2 iterations; the first emits 21 frames and every
    later one 8, so F-4 in total. For F < 25 it runs zero iterations and the
    pipeline crashes on an empty torch.cat.
    """
    iterations = (padded_frames - 1) // 8 - 2
    return 0 if iterations <= 0 else 21 + 8 * (iterations - 1)


def v1_padded_frame_count(total_frames: int) -> int:
    """Smallest 8k+1 frame count whose emitted frames cover every source frame.

    The old code padded to the smallest 8k+1 >= total. The loop then emitted
    total-4 .. total-1 frames for four of the eight residues of total mod 8, and
    the missing tail was hidden by repeating the last emitted frame.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    padded = max(V1_MIN_PADDED_FRAMES, ((total_frames + 4 - 1 + 7) // 8) * 8 + 1)
    assert v1_emitted_frames(padded) >= total_frames
    return padded
