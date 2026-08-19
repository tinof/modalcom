"""Scene cut detection and window planning for SparkVSR.

Ensures no temporal chunk crosses a scene boundary and every chunk has a valid
reference keyframe without cross-scene ghosting artifacts.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CUT_THRESHOLD,
    DEFAULT_MIN_SCENE_LEN_SEC,
    DEFAULT_OVERLAP,
    MAX_REF_WINDOW_OFFSET_SEC,
    MIN_REF_SPACING,
)


def detect_scene_cuts(
    video_path: str,
    threshold: float = DEFAULT_CUT_THRESHOLD,
    min_scene_len_sec: float = DEFAULT_MIN_SCENE_LEN_SEC,
    fps: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """Detect scene cuts using PySceneDetect AdaptiveDetector.

    Returns a list of shot ranges [(start_frame, end_frame), ...] (0-indexed, half-open).
    """
    from scenedetect import AdaptiveDetector, SceneManager, open_video

    video = open_video(video_path)
    video_fps = video.frame_rate if fps is None else fps
    min_scene_len_frames = max(1, int(min_scene_len_sec * video_fps))

    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_len_frames,
        )
    )

    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # Single continuous shot
        total_frames = video.duration.get_frames() if video.duration else 0
        return [(0, total_frames)] if total_frames > 0 else [(0, 0)]

    shots = []
    for scene in scene_list:
        start_f = scene[0].get_frames()
        end_f = scene[1].get_frames()
        if end_f > start_f:
            shots.append((start_f, end_f))

    return shots if shots else [(0, 0)]


def calculate_window_padding(raw_frames: int) -> Tuple[int, int, int]:
    """Calculate padded frame count to satisfy CogVideoX 8n+1 and minimum 9 frames.

    Returns:
        target_frames: 8n+1 valid length (>= 9)
        pad_before: 0
        pad_after: target_frames - raw_frames
    """
    if raw_frames < 9:
        target = 9
    else:
        target = ((raw_frames - 2) // 8 + 1) * 8 + 1

    pad_before = 0
    pad_after = target - raw_frames
    return target, pad_before, pad_after


def select_window_ref_indices(
    start_frame: int,
    end_frame: int,
    fps: float,
    existing_refs: Optional[List[int]] = None,
) -> List[int]:
    """Assign at least one reference frame inside [start_frame, end_frame).

    Constraints:
    - Located no more than 0.5 seconds into the window
    - Spaced > 4 frames apart from any existing reference
    """
    raw_frames = end_frame - start_frame
    if raw_frames <= 0:
        return []

    max_offset_frames = max(1, int(MAX_REF_WINDOW_OFFSET_SEC * fps))
    # Pick a frame within the first 0.5s (or middle if window is very short)
    offset = min(max_offset_frames // 2, max(0, raw_frames - 1))
    candidate = start_frame + offset

    # Ensure valid index within window
    candidate = max(start_frame, min(candidate, end_frame - 1))

    refs = []
    if existing_refs is None:
        existing_refs = []

    # Check spacing constraint (> 4 frames, i.e. difference >= MIN_REF_SPACING)
    def _conflicts(idx: int) -> bool:
        return any(abs(idx - ref) < MIN_REF_SPACING for ref in existing_refs)

    if _conflicts(candidate):
        alternative = next(
            (alt for alt in range(start_frame, end_frame) if not _conflicts(alt)),
            None,
        )
        if alternative is not None:
            candidate = alternative
        else:
            # No frame in this window is >4 away from an already-assigned
            # reference. Keep the in-window candidate anyway: the reference must
            # live inside the window or the executor drops it (it maps global to
            # window-local indices and discards out-of-range ones), leaving the
            # window unconditioned. Spacing only has to hold *within* a window,
            # because references are encoded into that window's own latent tensor
            # at index // 4 -- references in different windows cannot collide.
            # The global check above is just a spreading heuristic.
            print(
                f"Window [{start_frame}, {end_frame}) sits within 4 frames of an "
                f"existing reference; keeping in-window reference {candidate} "
                "(cross-window proximity is harmless)."
            )

    refs.append(candidate)
    return refs


def plan_cut_aware_windows(
    shots: List[Tuple[int, int]],
    total_frames: int,
    fps: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Dict[str, Any]:
    """Create a cut-aware window plan from shot boundaries.

    No temporal window crosses a shot boundary.
    """
    step = chunk_size - overlap
    windows = []
    all_refs = []
    window_id = 0

    # Ensure full coverage if shots list is empty or doesn't start at 0
    if not shots:
        shots = [(0, total_frames)]

    # PySceneDetect counts frames with its own decoder while `total_frames` comes from
    # decord, and on MKV/VFR sources the two disagree by a frame or two. Left unclamped, an
    # over-counted shot plans a window running past the end of the video, which the
    # executor rejects mid-render — on the parallel path, after every other worker has paid.
    shots = [
        (start, min(end, total_frames))
        for start, end in shots
        if start < total_frames
    ]

    shot_objs = []
    for s_idx, (shot_start, shot_end) in enumerate(shots):
        shot_objs.append({
            "shot_id": s_idx,
            "start_frame": shot_start,
            "end_frame": shot_end,
            "length": shot_end - shot_start,
        })
        shot_len = shot_end - shot_start
        if shot_len <= 0:
            continue

        if shot_len <= chunk_size:
            # Single window covers the whole shot
            target_f, pad_before, pad_after = calculate_window_padding(shot_len)
            refs = select_window_ref_indices(shot_start, shot_end, fps, all_refs)
            all_refs.extend(refs)

            windows.append({
                "window_id": window_id,
                "shot_id": s_idx,
                "start_frame": shot_start,
                "end_frame": shot_end,
                "raw_frames": shot_len,
                "target_frames": target_f,
                "pad_before": pad_before,
                "pad_after": pad_after,
                "ref_indices": refs,
            })
            window_id += 1
        else:
            # Overlapping windows within this shot, following upstream's
            # make_temporal_chunks: stride through the shot, then absorb a runt
            # tail into the previous window instead of appending a nearly
            # duplicate one. Appending would cost an extra inference per shot and
            # give the overlap three contributors whose ramp weights sum above 1.
            spans = [
                (s, min(s + chunk_size, shot_len))
                for s in range(0, max(1, shot_len - overlap), step)
            ]
            if spans[-1][0] + chunk_size < shot_len:
                spans.append((shot_len - chunk_size, shot_len))
            if len(spans) >= 2 and spans[-1][1] - spans[-1][0] < chunk_size:
                tail_end = spans.pop()[1]
                spans[-1] = (spans[-1][0], tail_end)

            for rel_start, rel_end in spans:
                w_start = shot_start + rel_start
                w_end = shot_start + rel_end

                raw_len = w_end - w_start
                target_f, pad_before, pad_after = calculate_window_padding(raw_len)
                refs = select_window_ref_indices(w_start, w_end, fps, all_refs)
                all_refs.extend(refs)

                windows.append({
                    "window_id": window_id,
                    "shot_id": s_idx,
                    "start_frame": w_start,
                    "end_frame": w_end,
                    "raw_frames": raw_len,
                    "target_frames": target_f,
                    "pad_before": pad_before,
                    "pad_after": pad_after,
                    "ref_indices": refs,
                })
                window_id += 1

    unique_refs = sorted(list(set(all_refs)))
    return {
        "cut_aware": True,
        "total_frames": total_frames,
        "fps": fps,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "shots": shot_objs,
        "windows": windows,
        "all_ref_indices": unique_refs,
    }


def plan_fixed_windows(
    total_frames: int,
    fps: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Dict[str, Any]:
    """Fallback planner: fixed overlapping windows ignoring scene cuts."""
    return plan_cut_aware_windows(
        shots=[(0, total_frames)],
        total_frames=total_frames,
        fps=fps,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def build_plan(
    video_path: Optional[str],
    total_frames: int,
    fps: float,
    cut_aware: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    threshold: float = DEFAULT_CUT_THRESHOLD,
    min_scene_len_sec: float = DEFAULT_MIN_SCENE_LEN_SEC,
) -> Dict[str, Any]:
    """Top-level plan builder.

    Runs cut detection if cut_aware is enabled and video_path is provided.
    Falls back to fixed windows if disabled or if cut detection fails.
    """
    if cut_aware and video_path:
        try:
            shots = detect_scene_cuts(
                video_path,
                threshold=threshold,
                min_scene_len_sec=min_scene_len_sec,
                fps=fps,
            )
            return plan_cut_aware_windows(
                shots=shots,
                total_frames=total_frames,
                fps=fps,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except Exception as e:
            print(f"Warning: cut detection failed ({e}), falling back to fixed windows.")

    plan = plan_fixed_windows(
        total_frames=total_frames,
        fps=fps,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    plan["cut_aware"] = False
    return plan
