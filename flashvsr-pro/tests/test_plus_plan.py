"""Segment planner, canvas geometry and the v1.1 frame arithmetic. No ffmpeg, no torch."""

from fractions import Fraction

import pytest

from modal_app.config import validate_plus_params
from modal_app.plus_plan import (
    Plan,
    check_plan,
    choose_boundaries,
    plan_canvas,
    plan_segments,
    seam_weights,
    stable_hash,
    v1_emitted_frames,
    v1_padded_frame_count,
)

FPS = Fraction(25)


def _plan(total, scores):
    return plan_segments(
        0, total, scores, fps=FPS, segment_seconds=60, search_seconds=30, threshold=0.3,
        overlap_frames=8, warmup_frames=32,
    )


def _delivered(plan: Plan) -> list[int]:
    """Source frame numbers in playback order, as the join would produce them."""
    frames: list[int] = []
    seams = {seam.index: seam for seam in plan.seams}
    for segment in plan.segments:
        frames.extend(range(segment.video_start, segment.video_start + segment.video))
        seam = seams.get(segment.index)
        if seam:
            frames.extend(range(seam.start, seam.start + seam.frames))
    return frames


def test_short_input_is_one_segment():
    plan = _plan(1000, {})
    assert len(plan.segments) == 1 and not plan.seams
    segment = plan.segments[0]
    assert (segment.model_start, segment.model_end, segment.video) == (0, 1000, 1000)
    assert (segment.warmup, segment.head, segment.tail) == (0, 0, 0)


def test_hard_cut_is_taken_and_has_no_overlap():
    # Target 1500, window 750..2250. The cut at 1700 qualifies, the one at 1400 is weaker.
    plan = _plan(3000, {1400: 0.5, 1700: 0.9})
    first, second = plan.segments[0], plan.segments[1]
    assert first.model_end == 1700 and first.tail == 0
    assert second.model_start == 1700 and (second.warmup, second.head) == (0, 0)
    assert not plan.seams
    assert _delivered(plan) == list(range(3000))


def test_cut_below_threshold_is_ignored_and_seam_is_soft():
    plan = _plan(3000, {1500: 0.29})
    first, second = plan.segments[0], plan.segments[1]
    # Soft cut exactly on the target.
    assert first.model_end == 1500 + 8 and first.tail == 8
    assert second.model_start == 1500 - 32
    assert (second.warmup, second.head) == (32, 8)
    assert [(seam.index, seam.start, seam.frames) for seam in plan.seams] == [(0, 1500, 8)]
    assert _delivered(plan) == list(range(3000))


def test_cut_outside_the_search_window_is_not_used():
    # 2300 is beyond 1500 + 750.
    plan = _plan(4000, {2300: 0.99})
    assert plan.segments[0].tail == 8
    assert plan.seams[0].start == 1500


def test_equal_scores_prefer_the_cut_nearest_the_target():
    cuts = choose_boundaries(
        0, 4000, {1000: 0.8, 1450: 0.8, 2100: 0.8},
        target_frames=1500, search_frames=750, threshold=0.3, min_frames=50,
    )
    assert cuts[0] == (1450, True)


def test_long_input_covers_every_frame_exactly_once():
    # Scene cuts every 137 frames, with a 4000-frame stretch that has none.
    scores = {frame: 0.6 for frame in range(137, 101_950, 137) if not 40_000 < frame < 44_000}
    plan = _plan(101_950, scores)
    check_plan(plan)
    assert _delivered(plan) == list(range(101_950))
    assert plan.seams, "the cut-free stretch must produce soft seams"
    assert sum(frames for _, frames in plan.concat_order()) == 101_950
    lengths = [segment.model_frames for segment in plan.segments]
    # No segment may approach the model's temporal table limit (about 4,090 frames).
    assert max(lengths) <= 1500 + 750 + 8 + 32
    assert min(lengths) >= 50


def test_last_segment_absorbs_the_remainder():
    plan = _plan(1500 + 750 + 1, {})
    assert len(plan.segments) == 2
    assert plan.segments[-1].model_end == 2251


def test_warmup_never_reaches_before_the_job_start():
    plan = plan_segments(
        500, 4500, {}, fps=FPS, segment_seconds=60, search_seconds=30, threshold=0.3,
        overlap_frames=8, warmup_frames=5000,
    )
    assert all(segment.model_start >= 500 for segment in plan.segments)
    assert _delivered(plan) == list(range(500, 4500))


def test_excerpt_range_and_json_round_trip():
    plan = plan_segments(
        1000, 3000, {1900: 0.7}, fps=FPS, segment_seconds=30, search_seconds=10,
        threshold=0.3, overlap_frames=8, warmup_frames=32,
    )
    assert _delivered(plan) == list(range(1000, 3000))
    assert Plan.from_json(plan.to_json()) == plan


def test_scores_may_be_a_list():
    scores = [0.0] * 4000
    scores[1600] = 0.5
    assert _plan(4000, scores).segments[0].model_end == 1600


def test_check_plan_rejects_a_gap():
    plan = _plan(4000, {1700: 0.9})
    broken = Plan(plan.first_frame, plan.last_frame, plan.segments[1:], plan.seams)
    with pytest.raises(ValueError):
        check_plan(broken)


def test_empty_range_is_rejected():
    with pytest.raises(ValueError):
        _plan(0, {})


def test_seam_weights_are_strictly_inside_the_unit_interval():
    weights = seam_weights(8)
    assert len(weights) == 8
    assert all(0 < weight < 1 for weight in weights)
    assert weights == sorted(weights)
    assert weights[0] == pytest.approx(1 / 9) and weights[-1] == pytest.approx(8 / 9)


def test_canvas_pads_1080p_2x_without_losing_picture():
    canvas = plan_canvas(1920, 1080, 2.0)
    assert (canvas.out_w, canvas.out_h) == (3840, 2160)
    assert (canvas.canvas_w, canvas.canvas_h) == (3840, 2176)
    assert (canvas.pad_left, canvas.pad_top) == (0, 8)
    assert (canvas.deliver_w, canvas.deliver_h) == (3840, 2160)


def test_canvas_one_and_a_half_delivered_at_2x():
    canvas = plan_canvas(1920, 1080, 1.5, deliver_scale=2.0)
    assert (canvas.out_w, canvas.out_h) == (2880, 1620)
    assert (canvas.canvas_w, canvas.canvas_h) == (2944, 1664)
    assert (canvas.pad_left, canvas.pad_top) == (32, 22)
    assert (canvas.deliver_w, canvas.deliver_h) == (3840, 2160)


def test_canvas_sides_are_even_and_multiples_of_128():
    for width, height, scale in [(1917, 1079, 2.0), (720, 576, 2.0), (1280, 534, 3.0)]:
        canvas = plan_canvas(width, height, scale)
        assert canvas.out_w % 2 == 0 and canvas.out_h % 2 == 0
        assert canvas.canvas_w % 128 == 0 and canvas.canvas_h % 128 == 0
        assert canvas.canvas_w >= canvas.out_w and canvas.canvas_h >= canvas.out_h


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


# --- FlashVSR v1.1 frame arithmetic (finding 4) ----------------------------------

def _old_padded(total: int) -> int:
    """What video_io.prepare_input_tensor computed before the fix."""
    return ((total - 1 + 7) // 8) * 8 + 1


def test_old_padding_dropped_real_frames_for_four_residues():
    losing = {total % 8 for total in range(26, 200) if v1_emitted_frames(_old_padded(total)) < total}
    assert losing == {0, 1, 6, 7}


def test_old_padding_emitted_nothing_for_short_clips():
    assert all(v1_emitted_frames(_old_padded(total)) == 0 for total in range(1, 18))


@pytest.mark.parametrize("total", list(range(1, 300)) + [4078, 101_950])
def test_new_padding_covers_every_frame_with_minimal_padding(total):
    padded = v1_padded_frame_count(total)
    assert padded % 8 == 1
    assert v1_emitted_frames(padded) >= total
    # Minimal: one step of 8 fewer would no longer cover the clip.
    assert padded == 25 or v1_emitted_frames(padded - 8) < total


# --- request validation ----------------------------------------------------------

def test_plus_params_defaults():
    params = validate_plus_params()
    assert params["mode"] == "turbo" and params["scale"] == 2.0
    assert params["resident"] is True and params["bypass_model"] is False
    assert params["decode_every"] == 28


@pytest.mark.parametrize("bad", [
    {"mode": "tiny"},
    {"scale": 0.5},
    {"scale": 5},
    {"deliver_scale": 9},
    {"attn_kernel": "flash"},
    {"tiny_decoder_mode": "fast"},
    {"decode_every": 30},
    {"decode_every": 0},
])
def test_plus_params_reject_bad_values(bad):
    with pytest.raises(ValueError):
        validate_plus_params(**bad)


# --- fixes to the v1.1 service -----------------------------------------------------

def test_temp_sibling_never_returns_the_path_itself():
    from modal_app.config import temp_sibling

    assert temp_sibling("/io/out/clip.mp4") == "/io/out/clip_temp.mp4"
    # The old `.replace(".mp4", "_temp.mp4")` returned these unchanged, and the
    # finished output was then removed as the temporary file (finding 8).
    assert temp_sibling("/io/out/clip.mkv") == "/io/out/clip_temp.mkv"
    assert temp_sibling("/io/out/clip.MP4") == "/io/out/clip_temp.MP4"
    for path in ("/io/a.mkv", "/io/a.mov", "/io/a", "/io/a.mp4.mkv"):
        assert temp_sibling(path) != path


def test_v1_request_accepts_only_the_dtype_the_models_are_loaded_in():
    from modal_app.config import validate_request

    assert validate_request(dtype="bf16")["dtype"] == "bf16"
    for dtype in ("fp32", "fp16", "float32"):
        with pytest.raises(ValueError, match="dtype must be one of"):
            validate_request(dtype=dtype)


def test_zero_overlap_gives_warm_up_and_no_seam():
    plan = plan_segments(
        0, 3000, {}, fps=FPS, segment_seconds=60, search_seconds=30, threshold=0.3,
        overlap_frames=0, warmup_frames=32,
    )
    assert not plan.seams
    first, second = plan.segments
    assert (first.tail, second.head, second.warmup) == (0, 0, 32)
    assert second.model_start == 1500 - 32
    assert _delivered(plan) == list(range(3000))
    assert [name for name, _ in plan.concat_order()] == ["seg_00000", "seg_00001"]
