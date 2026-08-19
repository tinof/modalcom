"""Tests for the parallel segmentation used by the fan-out driver.

These run on CPU with no Modal or GPU involvement: `plan_segments` is pure arithmetic
over the cut plan's shot boundaries. They exist because the fan-out path's correctness
rests entirely on segments being contiguous, complete and cut only on shot boundaries —
if any of those breaks, the concatenated output silently loses or duplicates frames.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from modal_app.service import _preflight_plan, plan_segments  # noqa: E402

TARGET = 500


def _shots(*starts):
    return {"shots": [{"start_frame": s} for s in starts]}


def test_segments_are_contiguous_and_cover_every_frame():
    plan = _shots(*[i * 100 for i in range(900)])
    total = 90_000

    segments = plan_segments(plan, total, threshold_frames=0, target_frames=TARGET)

    assert segments[0][0] == 0
    assert segments[-1][1] == total
    for (_, end), (next_start, _) in zip(segments, segments[1:]):
        assert end == next_start


def test_every_boundary_is_a_shot_boundary():
    starts = [i * 100 for i in range(900)]
    plan = _shots(*starts)

    segments = plan_segments(plan, 90_000, threshold_frames=0, target_frames=TARGET)

    admissible = set(starts) | {90_000}
    for start, end in segments:
        assert start in admissible
        assert end in admissible


def test_episode_length_input_produces_many_workers():
    """A 60-minute episode must fan out wide enough to be worth parallelising."""
    plan = _shots(*[i * 100 for i in range(900)])

    segments = plan_segments(plan, 90_000, threshold_frames=0, target_frames=TARGET)

    assert len(segments) > 100


def test_single_shot_input_cannot_be_split():
    """Splitting mid-shot would place a join where the two sides denoised independently."""
    plan = _shots(0)

    segments = plan_segments(plan, 1551, threshold_frames=0, target_frames=TARGET)

    assert segments == [(0, 1551)]


def test_sample_clip_splits_on_its_three_shots():
    """The 12 s sample's measured cuts are at 152 and 206."""
    plan = _shots(0, 152, 206)

    segments = plan_segments(plan, 301, threshold_frames=0, target_frames=100)

    assert segments == [(0, 152), (152, 206), (206, 301)]


def test_a_shot_longer_than_the_target_is_never_split():
    plan = _shots(0, 900)

    segments = plan_segments(plan, 1200, threshold_frames=0, target_frames=TARGET)

    assert segments == [(0, 900), (900, 1200)]


class TestPreflight:
    """The CPU pre-flight is the gate that converts a mid-fan-out GPU failure into a free one."""

    @staticmethod
    def _plan(*windows):
        return {"windows": [
            {"window_id": i, "start_frame": s, "end_frame": e}
            for i, (s, e) in enumerate(windows)
        ]}

    def test_accepts_a_well_formed_plan(self):
        _preflight_plan(self._plan((0, 49), (41, 152), (152, 206)), [(0, 152), (152, 206)], 206)

    def test_rejects_a_window_past_the_end_of_the_video(self):
        with pytest.raises(RuntimeError, match="past the video"):
            _preflight_plan(self._plan((0, 49), (100, 160)), [(0, 150)], 150)

    def test_rejects_a_window_straddling_a_segment_boundary(self):
        with pytest.raises(RuntimeError, match="straddles"):
            _preflight_plan(self._plan((100, 200)), [(0, 152), (152, 300)], 300)

    def test_rejects_non_contiguous_segments(self):
        with pytest.raises(RuntimeError, match="not contiguous"):
            _preflight_plan(self._plan(), [(0, 100), (110, 300)], 300)

    def test_rejects_segments_that_do_not_span_the_video(self):
        with pytest.raises(RuntimeError, match="do not span"):
            _preflight_plan(self._plan(), [(0, 250)], 300)

    def test_rejects_a_segment_too_large_for_container_memory(self):
        """An unbroken shot longer than a worker can decode must fail before the GPU starts."""
        huge = 200_000
        with pytest.raises(RuntimeError, match="longest segment"):
            _preflight_plan(self._plan(), [(0, huge)], huge)
