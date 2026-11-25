"""The whole video path except the model, on real ffmpeg with a CPU encoder.

A synthetic source carries its frame number in three grey blocks. It is cut by
the real planner (one hard cut, one forced soft seam), decoded with the real
decode command, "upscaled" by pixel doubling in place of the model, encoded with
the real encode command (libx265 standing in for hevc_nvenc), cross-faded at the
seam, joined and muxed with the real join command. The frame numbers read back
from the result prove that no frame was lost, repeated or reordered.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from modal_app.plus_plan import plan_canvas, plan_segments
from modal_app.plus_video import (
    SegmentEncoder,
    SegmentReader,
    SegmentSink,
    VideoError,
    VideoFingerprint,
    blend_seam,
    canvas_deliver,
    check_constant_frame_rate,
    check_parts,
    decode_command,
    encode_command,
    encode_frames,
    first_vcl_nal_type,
    frame_range,
    join_parts,
    parse_scene_scores,
    probe_source,
    scene_scores,
    seam_files,
    segment_files,
    stream_summary,
    write_concat_list,
)

from synthetic import (
    CUT,
    FPS,
    FRAMES,
    HEIGHT,
    WIDTH,
    decode_all,
    have_ffmpeg,
    make_source,
    needs_ffmpeg,
    read_index,
    source_frame,
)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> str:
    return make_source(tmp_path_factory.mktemp("source"))


@needs_ffmpeg
def test_probe_reports_every_frame_and_the_keyframes(source):
    info = probe_source(source)
    assert (info.width, info.height, info.fps) == (WIDTH, HEIGHT, FPS)
    assert info.frames == FRAMES
    assert info.frame_pts == sorted(info.frame_pts)
    # 2 s GOP: keyframes every 50 frames.
    assert [info.frame_pts.index(pts) for pts in info.keyframe_pts] == [0, 50, 100, 150, 200]
    assert info.color_space == "bt709"
    assert frame_range(info, None, None) == (0, FRAMES)
    assert frame_range(info, 2.0, 4.0) == (50, 150)
    assert frame_range(info, 9.0, 60.0) == (225, FRAMES)


@needs_ffmpeg
def test_scene_scores_mark_the_cut_frame(source):
    scores = scene_scores(source, FRAMES)
    assert len(scores) == FRAMES
    assert scores.index(max(scores)) == CUT
    assert scores[CUT] > 0.3
    assert max(score for index, score in enumerate(scores) if index != CUT) < 0.3


@needs_ffmpeg
@pytest.mark.parametrize("start,end", [(0, 20), (1, 17), (49, 51), (63, 140), (199, 250), (249, 250)])
def test_decode_is_frame_exact_from_any_start_frame(source, tmp_path, start, end):
    info = probe_source(source)
    reader = SegmentReader(source, info, start, end, scratch_dir=str(tmp_path), chunk_frames=16)
    indices = [read_index(frame) for chunk in reader for frame in chunk]
    reader.finish()
    assert indices == list(range(start, end))


@needs_ffmpeg
def test_reader_fails_when_frames_are_missing(source, tmp_path):
    info = probe_source(source)
    reader = SegmentReader(source, info, 10, 30, scratch_dir=str(tmp_path))
    for _ in reader:
        pass
    reader.expected += 1
    with pytest.raises(VideoError, match="expected 21"):
        reader.finish()


@needs_ffmpeg
def test_round_trip_through_segments_seam_and_join(source, tmp_path):
    info = probe_source(source)
    scores = scene_scores(source, info.frames)
    plan = plan_segments(
        0, info.frames, scores, fps=info.fps, segment_seconds=3, search_seconds=1,
        threshold=0.3, overlap_frames=8, warmup_frames=16,
    )
    # One hard cut on the scene change, then one soft seam where no cut exists.
    assert [(s.model_start, s.model_end, s.warmup, s.head, s.video, s.tail) for s in plan.segments] == [
        (0, 100, 0, 0, 100, 0),
        (100, 183, 0, 0, 75, 8),
        (159, 250, 16, 8, 67, 0),
    ]
    assert [(seam.start, seam.frames) for seam in plan.seams] == [(175, 8)]

    canvas = plan_canvas(info.width, info.height, 2.0)
    parts = tmp_path / "segments"
    parts.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sidecars: dict[str, np.ndarray] = {}
    for segment in plan.segments:
        names = segment_files(segment)
        reader = SegmentReader(source, info, segment.model_start, segment.model_end, scratch_dir=str(scratch))
        encoder = SegmentEncoder(
            canvas.out_w, canvas.out_h, info.fps, str(parts / names["video"]),
            encoder="libx265", deliver=canvas_deliver(canvas),
        )
        sink = SegmentSink(segment, encoder)
        for chunk in reader:
            # The stand-in for the model: 2x by pixel doubling.
            sink.write(chunk.repeat(2, axis=1).repeat(2, axis=2))
        reader.finish()
        head, tail = sink.finish()
        if head is not None:
            sidecars[names["head"]] = head
        if tail is not None:
            sidecars[names["tail"]] = tail

    for seam in plan.seams:
        tail = sidecars[segment_files(plan.segments[seam.index])["tail"]]
        head = sidecars[segment_files(plan.segments[seam.index + 1])["head"]]
        assert tail.shape == (8, canvas.out_h, canvas.out_w, 3)
        encode_frames(blend_seam(tail, head), info.fps, str(parts / seam_files(seam)["video"]), encoder="libx265")

    output = tmp_path / "output.mkv"
    report = join_parts(plan, str(parts), str(output), info, source_path=source)
    assert report["frames"] == FRAMES
    assert report["fps"] == "25/1"
    assert (report["codec"], report["profile"], report["pix_fmt"]) == ("hevc", "Main 10", "yuv420p10le")
    assert report["size"] == "640x360"
    assert report["parts"] == 4 and report["seams"] == 1
    assert report["carried_source_streams"] is True

    # Every frame present, once, in order, across the hard cut and the seam.
    decoded = decode_all(str(output), canvas.out_w, canvas.out_h)
    assert [read_index(frame, scale=2) for frame in decoded] == list(range(FRAMES))

    # Colour survived decode -> RGB -> encode: channel means within one 8-bit level.
    original = decode_all(source, WIDTH, HEIGHT)
    for channel in range(3):
        assert abs(float(decoded[..., channel].mean()) - float(original[..., channel].mean())) < 1.0

    streams = stream_summary(str(output))
    kinds = [stream["codec_type"] for stream in streams]
    assert kinds == ["video", "audio", "subtitle"]
    video, audio, subtitle = streams
    assert video["color_space"] == "bt709"
    assert video["color_primaries"] == "bt709"
    assert video["color_transfer"] == "bt709"
    assert video["color_range"] == "tv"
    assert video["r_frame_rate"] == "25/1"
    assert audio["codec_name"] == "aac" and audio["tags"]["language"] == "fin"
    assert subtitle["codec_name"] == "subrip" and subtitle["tags"]["language"] == "fin"
    assert abs(float(video["start_time"]) - float(audio["start_time"])) < 0.05

    # Subtitle cues keep their times.
    cues = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s:0", "-show_entries", "packet=pts_time", "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert [float(packet["pts_time"]) for packet in json.loads(cues)["packets"]] == [1.0, 6.0]


@needs_ffmpeg
def test_excerpt_job_joins_without_source_streams(source, tmp_path):
    info = probe_source(source)
    plan = plan_segments(
        50, 150, {}, fps=info.fps, segment_seconds=60, search_seconds=30, threshold=0.3,
        overlap_frames=8, warmup_frames=16,
    )
    segment = plan.segments[0]
    parts = tmp_path / "segments"
    parts.mkdir()
    reader = SegmentReader(source, info, segment.model_start, segment.model_end, scratch_dir=str(tmp_path))
    encoder = SegmentEncoder(WIDTH, HEIGHT, info.fps, str(parts / segment_files(segment)["video"]), encoder="libx265")
    sink = SegmentSink(segment, encoder)
    for chunk in reader:
        sink.write(chunk)
    reader.finish()
    sink.finish()
    output = tmp_path / "excerpt.mkv"
    report = join_parts(plan, str(parts), str(output), info, source_path=source)
    assert report["frames"] == 100 and report["carried_source_streams"] is False
    assert [stream["codec_type"] for stream in stream_summary(str(output))] == ["video"]
    assert [read_index(frame) for frame in decode_all(str(output), WIDTH, HEIGHT)] == list(range(50, 150))


@needs_ffmpeg
def test_deliver_size_resamples_in_the_encoder(tmp_path):
    frames = np.stack([source_frame(index) for index in range(10)])
    path = tmp_path / "deliver.mp4"
    encode_frames(frames, FPS, str(path), encoder="libx265", deliver=(480, 270))
    summary = stream_summary(str(path))[0]
    assert (summary["width"], summary["height"]) == (480, 270)


@needs_ffmpeg
def test_join_refuses_a_segment_with_the_wrong_frame_count(source, tmp_path):
    info = probe_source(source)
    plan = plan_segments(
        0, 40, {}, fps=info.fps, segment_seconds=60, search_seconds=30, threshold=0.3,
        overlap_frames=8, warmup_frames=16,
    )
    parts = tmp_path / "segments"
    parts.mkdir()
    frames = np.stack([source_frame(index) for index in range(39)])
    encode_frames(frames, info.fps, str(parts / "seg_00000.mp4"), encoder="libx265")
    with pytest.raises(VideoError, match="39 frames, planned 40"):
        join_parts(plan, str(parts), str(tmp_path / "out.mkv"), info, source_path=None)


@needs_ffmpeg
def test_encoder_reports_ffmpeg_failure(tmp_path):
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(VideoError):
        encode_frames(frames, FPS, str(tmp_path / "x.notaformat"), encoder="libx265")


def test_encoder_rejects_frames_of_the_wrong_shape(tmp_path):
    if not have_ffmpeg():
        pytest.skip("needs ffmpeg")
    encoder = SegmentEncoder(64, 64, FPS, str(tmp_path / "x.mp4"), encoder="libx265")
    try:
        with pytest.raises(ValueError):
            encoder.write(np.zeros((1, 32, 64, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            encoder.write(np.zeros((1, 64, 64, 3), dtype=np.float32))
    finally:
        encoder.abort()


# --- pure helpers ----------------------------------------------------------------

def _info(**overrides):
    from modal_app.plus_video import SourceInfo

    base = SourceInfo(
        width=1920, height=1080, fps=Fraction(25), time_base=Fraction(1, 1000),
        format_start=0.0, video_start=0.0, color_space="bt709", color_range="tv",
        color_transfer="bt709", pix_fmt="yuv420p", field_order="progressive",
        frame_pts=[index * 40 for index in range(500)],
        keyframe_pts=[index * 2000 for index in range(10)],
    )
    return replace(base, **overrides)


def test_decode_command_seeks_before_the_keyframe_and_trims_by_pts():
    command = decode_command("in.mkv", _info(), 130, 200, pts_log="p.log")
    # Frame 130 is pts 5200; the keyframe at or before it is 4000 (4 s); seek 1 s earlier.
    assert command[command.index("-ss") + 1] == "3.000000"
    assert "-noaccurate_seek" in command and "-copyts" in command
    filters = command[command.index("-vf") + 1]
    assert filters.startswith("trim=start_pts=5200:end_pts=8000,")
    assert "in_color_matrix=bt709:in_range=tv" in filters
    assert "accurate_rnd+full_chroma_int" in filters
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-stats_enc_pre") + 1] == "p.log"


def test_decode_command_for_the_start_and_the_end_of_the_file():
    command = decode_command("in.mkv", _info(), 0, 500)
    assert "-ss" not in command
    assert "trim=start_pts=0," in command[command.index("-vf") + 1]
    # A container that starts late: input -ss is relative to its start time.
    late = _info(format_start=1.5)
    command = decode_command("in.ts", late, 130, 200)
    assert command[command.index("-ss") + 1] == "1.500000"


def test_decode_command_picks_the_matrix_for_untagged_sources():
    hd = decode_command("in.mkv", _info(color_space=""), 0, 10)
    assert "in_color_matrix=bt709" in hd[hd.index("-vf") + 1]
    sd = decode_command("in.mkv", _info(color_space="", height=576, color_range="pc"), 0, 10)
    assert "in_color_matrix=smpte170m:in_range=pc" in sd[sd.index("-vf") + 1]


def test_decode_command_rejects_a_range_outside_the_file():
    with pytest.raises(ValueError):
        decode_command("in.mkv", _info(), 10, 501)


def test_nvenc_command_is_main10_level_51_and_tagged():
    command = encode_command(3840, 2160, Fraction(25), "out.mp4", encoder="hevc_nvenc")
    assert command[command.index("-c:v") + 1] == "hevc_nvenc"
    assert command[command.index("-profile:v") + 1] == "main10"
    assert command[command.index("-pix_fmt", command.index("-c:v")) + 1] == "p010le"
    assert command[command.index("-level") + 1] == "5.1"
    assert command[command.index("-tier") + 1] == "high"
    assert command[command.index("-r") + 1] == "25/1"
    assert command[command.index("-cq") + 1] == "20"
    assert command[command.index("-maxrate") + 1] == "100M"
    filters = command[command.index("-vf") + 1]
    assert "out_color_matrix=bt709:out_range=tv" in filters
    assert "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv" in filters


def test_nvenc_level_is_left_to_the_driver_above_level_5_sizes():
    command = encode_command(7680, 4320, Fraction(25), "out.mp4", encoder="hevc_nvenc")
    assert "-level" not in command


def test_ntsc_rate_stays_rational():
    command = encode_command(1920, 1080, Fraction(24000, 1001), "out.mp4", encoder="libx265")
    assert command[command.index("-r") + 1] == "24000/1001"


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError):
        encode_command(64, 64, FPS, "out.mp4", encoder="libx264")


def test_parse_scene_scores_pairs_frames_with_scores():
    text = (
        "frame:0    pts:0       pts_time:0\nlavfi.scene_score=0.000000\n"
        "frame:1    pts:40      pts_time:0.04\nlavfi.scene_score=0.026013\n"
        "frame:3    pts:120     pts_time:0.12\nlavfi.scene_score=0.771414\n"
    )
    assert parse_scene_scores(text) == [0.0, 0.026013, 0.0, 0.771414]


def test_constant_frame_rate_check():
    check_constant_frame_rate([0, 40, 80, 120])
    # 23.976 fps in milliseconds alternates 41 and 42.
    check_constant_frame_rate([0, 42, 83, 125, 167])
    with pytest.raises(VideoError, match="variable frame rate"):
        check_constant_frame_rate([0, 40, 80, 160])
    with pytest.raises(VideoError, match="non-increasing"):
        check_constant_frame_rate([0, 40, 40])


def test_first_vcl_nal_type_skips_parameter_sets():
    def nal(kind: int) -> bytes:
        return b"\x00\x00\x00\x01" + bytes([kind << 1, 1]) + b"\xaa\xbb"

    assert first_vcl_nal_type(nal(32) + nal(33) + nal(34) + nal(39) + nal(19) + nal(1)) == 19
    assert first_vcl_nal_type(nal(32) + nal(33) + nal(34) + nal(21)) == 21
    assert first_vcl_nal_type(nal(32)) is None


def _print(**overrides) -> VideoFingerprint:
    base = VideoFingerprint(
        frames=100, codec="hevc", profile="Main 10", level=153, pix_fmt="yuv420p10le",
        width=3840, height=2160, fps="25/1", extradata_md5="MD5:aa", first_vcl_nal=19,
    )
    return replace(base, **overrides)


def test_check_parts_accepts_matching_segments():
    check_parts([("a", 100, _print()), ("b", 8, _print(frames=8, first_vcl_nal=20))])


@pytest.mark.parametrize("bad,message", [
    (dict(frames=99), "99 frames, planned 100"),
    (dict(first_vcl_nal=21), "not an IDR"),
    (dict(extradata_md5="MD5:bb"), "codec parameters differ"),
    (dict(level=150), "codec parameters differ"),
])
def test_check_parts_rejects(bad, message):
    with pytest.raises(VideoError, match=message):
        check_parts([("a", 100, _print()), ("b", 100, _print(**bad))])


def test_concat_list_carries_exact_durations(tmp_path):
    path = tmp_path / "list.ffconcat"
    write_concat_list(str(path), [("/x/seg 'a'.mp4", 1499), ("/x/b.mp4", 8)], Fraction(24000, 1001))
    lines = path.read_text().splitlines()
    assert lines[0] == "ffconcat version 1.0"
    assert lines[1] == "file '/x/seg '\\''a'\\''.mp4'"
    assert lines[2] == f"duration {1499 * 1001 / 24000:.9f}"
    assert lines[4] == f"duration {8 * 1001 / 24000:.9f}"


def test_blend_seam_fades_from_tail_to_head():
    tail = np.zeros((8, 2, 2, 3), dtype=np.uint8)
    head = np.full((8, 2, 2, 3), 180, dtype=np.uint8)
    mixed = blend_seam(tail, head)
    assert mixed.dtype == np.uint8
    assert [int(frame[0, 0, 0]) for frame in mixed] == [20, 40, 60, 80, 100, 120, 140, 160]
    with pytest.raises(VideoError):
        blend_seam(tail, head[:4])


# --- encoder start-up retries ------------------------------------------------------

class _FlakyEncoder(SegmentEncoder):
    """An encoder whose first `failures` starts die the way a refused NVENC session does."""

    failures = 0

    def _spawn(self):
        if self.starts < self.failures:
            self.starts += 1
            # Reads a little, then exits non-zero: the pipe breaks under the writer.
            return subprocess.Popen(
                ["python3", "-c", "import sys; sys.stdin.buffer.read(16); sys.exit(8)"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
            )
        return super()._spawn()


def _flaky(tmp_path, failures: int, size: int = 64):
    encoder_class = type("Flaky", (_FlakyEncoder,), {"failures": failures})
    return encoder_class(size, size, FPS, str(tmp_path / "flaky.mp4"), encoder="libx265")


def _coded_frames(count: int, size: int = 64) -> np.ndarray:
    levels = (np.arange(count) * 5 % 200 + 20).astype(np.uint8)
    return np.broadcast_to(levels[:, None, None, None], (count, size, size, 3)).copy()


def _levels(path: str, size: int = 64) -> list[int]:
    return [int(round(float(frame.mean()) / 5) * 5) for frame in decode_all(path, size, size)]


@needs_ffmpeg
@pytest.mark.parametrize("failures", [1, 3])
@pytest.mark.parametrize("count", [8, 100])
def test_a_failed_encoder_start_is_retried_without_losing_frames(tmp_path, failures, count):
    encoder = _flaky(tmp_path, failures)
    frames = _coded_frames(count)
    for start in range(0, count, 8):
        encoder.write(frames[start:start + 8])
    encoder.close()
    assert encoder.starts == failures + 1
    assert _levels(str(tmp_path / "flaky.mp4")) == [int(frame[0, 0, 0]) for frame in frames]


@needs_ffmpeg
def test_an_encoder_that_never_starts_fails_with_the_attempt_count(tmp_path):
    encoder = _flaky(tmp_path, failures=99)
    with pytest.raises(VideoError, match="after 6 start"):
        for _ in range(4):
            encoder.write(_coded_frames(8))
        encoder.close()
    encoder.abort()


@needs_ffmpeg
def test_a_failure_after_start_up_is_not_retried(tmp_path):
    encoder = SegmentEncoder(64, 64, FPS, str(tmp_path / "late.mp4"), encoder="libx265")
    for _ in range(8):  # 64 frames: past the replay window
        encoder.write(_coded_frames(8))
    deadline = time.monotonic() + 20
    while encoder._replay is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert encoder._replay is None
    encoder._process.kill()
    with pytest.raises(VideoError):
        for _ in range(50):
            encoder.write(_coded_frames(8))
        encoder.close()
    assert encoder.starts == 1
    encoder.abort()


@needs_ffmpeg
def test_abort_with_a_full_queue_stops_the_writer_thread(tmp_path):
    # A stand-in that never reads: the writer blocks and the queue fills up.
    encoder_class = type("Stuck", (SegmentEncoder,), {
        "_spawn": lambda self: subprocess.Popen(
            ["sleep", "30"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr),
    })
    encoder = encoder_class(512, 512, FPS, str(tmp_path / "stuck.mp4"), encoder="libx265", queue_chunks=2)
    frames = np.zeros((8, 512, 512, 3), dtype=np.uint8)
    encoder.write(frames)
    encoder.write(frames)
    encoder.write(frames)
    started = time.monotonic()
    encoder.abort()
    assert time.monotonic() - started < 10
    assert not encoder._thread.is_alive()


@needs_ffmpeg
def test_resampled_delivery_keeps_the_colours(tmp_path):
    """The 1.5x look-test path: render small, Lanczos up in the encoder."""
    frames = np.stack([source_frame(index) for index in range(0, FRAMES, 5)])
    doubled = frames.repeat(2, axis=1).repeat(2, axis=2)
    plain = tmp_path / "plain.mp4"
    resampled = tmp_path / "resampled.mp4"
    encode_frames(doubled, FPS, str(plain), encoder="libx265")
    encode_frames(frames, FPS, str(resampled), encoder="libx265", deliver=(WIDTH * 2, HEIGHT * 2))
    summary = stream_summary(str(resampled))[0]
    assert (summary["width"], summary["height"]) == (WIDTH * 2, HEIGHT * 2)
    assert (summary["color_space"], summary["color_primaries"], summary["color_transfer"]) == ("bt709",) * 3
    reference = decode_all(str(plain), WIDTH * 2, HEIGHT * 2)
    delivered = decode_all(str(resampled), WIDTH * 2, HEIGHT * 2)
    assert len(delivered) == len(frames)
    for channel in range(3):
        assert abs(float(delivered[..., channel].mean()) - float(reference[..., channel].mean())) < 1.0
        assert abs(float(delivered[..., channel].mean()) - float(frames[..., channel].mean())) < 1.0
