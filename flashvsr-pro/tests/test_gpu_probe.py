"""The probe's pass/fail reading. The checks themselves need a GPU."""

from modal_app.gpu_probe import failures


def test_failures_reads_strings_and_nested_results():
    report = {
        "torch": "2.12.1+cu130",
        "nvenc_h264": "ok",
        "nvenc_hevc_main10": "FAILED: Cannot load libnvidia-encode.so.1",
        "dense_forward": "ok",
        "sparse_forward": {"result": "FAILED", "max_error": 0.9, "tolerance": 0.04, "control_gap": 1.1},
        "weights": "ok",
    }
    assert failures(report) == ["nvenc_hevc_main10", "sparse_forward"]


def test_a_clean_report_has_no_failures():
    assert failures({"nvenc_h264": "ok", "sparse_forward": {"result": "ok"}}) == []
