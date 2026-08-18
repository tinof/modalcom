"""Why does ThreadedDecoder drain after one frame? Isolate the variable.

PopEntries blocks until the batch is available and only returns short once the producer
has signalled done (SPSCBuffer.hpp:86-98), so a 1-frame batch means the decode thread
really did stop. This probe walks the axes one at a time: decoder class, colour type,
buffer/batch size, and whether the container is the problem.

    modal run scripts/probe_nvdec.py
"""

from pathlib import Path

import modal

SAMPLE = Path(__file__).parent.parent / "sample" / "jopet_10s.mkv"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("torch==2.13.0", "PyNvVideoCodec==2.2.0", "numpy==2.5.2")
    .add_local_file(SAMPLE.as_posix(), "/sample.mkv", copy=True)
)

app = modal.App("rtx-probe-nvdec", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def probe() -> None:
    import subprocess

    import PyNvVideoCodec as nvc
    import torch

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", "/sample.mkv",
         "-c:v", "copy", "-an", "/sample.mp4"],
        check=True,
    )

    def drain(label: str, make, batch: int) -> None:
        try:
            dec = make()
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<52} CONSTRUCT FAILED: {str(exc).splitlines()[0][:40]}")
            return
        total, calls, sizes = 0, 0, []
        try:
            while True:
                frames = dec.get_batch_frames(batch)
                calls += 1
                if len(frames) == 0:
                    break
                sizes.append(len(frames))
                total += len(frames)
                if calls > 2000:
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<52} RAISED after {total}: {str(exc).splitlines()[0][:34]}")
            return
        finally:
            with __import__("contextlib").suppress(Exception):
                dec.end()
        print(f"{label:<52} {total:4d} frames  batches={sizes[:6]}{'...' if len(sizes) > 6 else ''}")

    print("expect 301 frames\n")
    print("-- ThreadedDecoder: colour type --")
    for name, ct in [
        ("NATIVE", nvc.OutputColorType.NATIVE),
        ("RGB", nvc.OutputColorType.RGB),
        ("RGBP", nvc.OutputColorType.RGBP),
    ]:
        drain(
            f"ThreadedDecoder mkv {name} buf=12 batch=4",
            lambda ct=ct: nvc.ThreadedDecoder(
                enc_file_path="/sample.mkv", buffer_size=12, gpu_id=0,
                cuda_context=0, cuda_stream=0, use_device_memory=True,
                output_color_type=ct,
            ),
            4,
        )

    print("\n-- ThreadedDecoder: batch size (RGB) --")
    for buf, batch in [(12, 1), (12, 12), (32, 8), (4, 4)]:
        drain(
            f"ThreadedDecoder mkv RGB buf={buf} batch={batch}",
            lambda buf=buf: nvc.ThreadedDecoder(
                enc_file_path="/sample.mkv", buffer_size=buf, gpu_id=0,
                cuda_context=0, cuda_stream=0, use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
            ),
            batch,
        )

    print("\n-- container: mkv vs mp4 --")
    for path in ["/sample.mkv", "/sample.mp4"]:
        drain(
            f"ThreadedDecoder {path} RGB buf=12 batch=4",
            lambda path=path: nvc.ThreadedDecoder(
                enc_file_path=path, buffer_size=12, gpu_id=0,
                cuda_context=0, cuda_stream=0, use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
            ),
            4,
        )

    print("\n-- scanned metadata on --")
    drain(
        "ThreadedDecoder mkv RGB need_scanned=True",
        lambda: nvc.ThreadedDecoder(
            enc_file_path="/sample.mkv", buffer_size=12, gpu_id=0,
            cuda_context=0, cuda_stream=0, use_device_memory=True,
            need_scanned_stream_metadata=True,
            output_color_type=nvc.OutputColorType.RGB,
        ),
        4,
    )

    print("\n-- SimpleDecoder (indexed) --")
    try:
        sd = nvc.SimpleDecoder(
            enc_file_path="/sample.mkv", gpu_id=0, cuda_context=0, cuda_stream=0,
            use_device_memory=True, output_color_type=nvc.OutputColorType.RGB,
            need_scanned_stream_metadata=True,
        )
        print(f"SimpleDecoder len={len(sd)}")
        got = 0
        while True:
            frames = sd.get_batch_frames(8)
            if len(frames) == 0:
                break
            got += len(frames)
        print(f"SimpleDecoder get_batch_frames drained {got} frames")
        t = torch.from_dlpack(sd[0])
        print(f"SimpleDecoder[0] -> {tuple(t.shape)} {t.dtype} {t.device}")
    except Exception as exc:  # noqa: BLE001
        print(f"SimpleDecoder FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")

    print("\n-- low-level Demuxer + Decoder --")
    try:
        dmx = nvc.CreateDemuxer(filename="/sample.mkv")
        low = nvc.CreateDecoder(
            gpuid=0, codec=dmx.GetNvCodecId(), usedevicememory=1,
            outputColorType=nvc.OutputColorType.RGB,
        )
        got = 0
        for packet in dmx:
            for _frame in low.Decode(packet):
                got += 1
        print(f"low-level drained {got} frames")
    except Exception as exc:  # noqa: BLE001
        print(f"low-level FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
