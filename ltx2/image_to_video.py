# ---
# output-directory: "/tmp/ltx2_image_to_video"
# args: ["--prompt", "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background.", "--image-path", "https://modal-cdn.com/example_image_to_video_image.png"]
# ---

# # Animate images with Lightricks LTX-2.3 via CLI, API, and web UI

# This example shows how to run [LTX-2.3](https://github.com/Lightricks/LTX-2) on Modal
# to generate videos from your local command line, via an API, and in a web UI.

# LTX-2.3 is a 22-billion parameter DiT-based video foundation model that produces
# high-quality, coherent videos with full HD (1920×1088) output via its two-stage
# HQ pipeline and second-order Res2s sampler.

# ## Basic setup

import random
import tempfile
import time
from pathlib import Path
from typing import Annotated, Optional

import fastapi
import modal

# All Modal programs need an [`App`](https://modal.com/docs/reference/modal.App) —
# an object that acts as a recipe for the application.

app = modal.App("example-ltx2.3-image-to-video")

# ### Configuring dependencies

# LTX-2.3 uses its own inference packages (`ltx-core` and `ltx-pipelines`)
# instead of Hugging Face `diffusers`. We install these from the
# [LTX-2 GitHub repository](https://github.com/Lightricks/LTX-2).
# We use the two-stage HQ pipeline for production-quality full HD results.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg", "git")
    .run_commands(
        "pip install 'transformers==4.57.3' 'torch==2.7.1'",
        "pip install --no-deps 'ltx-core @ git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core'",
        "pip install --no-deps 'ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines'",
        "pip install torchaudio einops accelerate scipy av",
    )
    .uv_pip_install(
        "torchvision",
        "huggingface-hub[hf_xet]",
        "fastapi[standard]==0.115.8",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "pillow==11.1.0",
    )
)

# ## Storing model weights on Modal

# LTX-2.3 requires several model components:
# - **Checkpoint**: the main 22B transformer weights
# - **Gemma 3 text encoder**: 12B parameter text encoder
# - **Spatial upsampler**: 2x resolution enhancement for the two-stage HQ pipeline
# - **Distilled LoRA**: parameter-efficient adaptation with per-stage strengths

MODEL_REPO = "Lightricks/LTX-2.3"
CHECKPOINT_FILE = "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA_FILE = "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER_FILE = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# We store these on a [Modal Volume](https://modal.com/docs/guide/volumes)
# so they persist across container restarts.

model_volume = modal.Volume.from_name("ltx2.3-model-cache", create_if_missing=True)

MODEL_PATH = "/models"

image = image.env(
    {
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster downloads
        "HF_HUB_CACHE": MODEL_PATH,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # reduce fragmentation
    }
)

# ## Storing model outputs on Modal

OUTPUT_PATH = "/outputs"
output_volume = modal.Volume.from_name("outputs", create_if_missing=True)

# ## Implementing LTX-2.3 inference on Modal

# We wrap the inference logic in a Modal [Cls](https://modal.com/docs/guide/lifecycle-functions)
# that ensures models are loaded and then moved to the GPU once when a new instance
# starts, rather than every time we run it.

# LTX-2.3 uses `TI2VidTwoStagesHQPipeline` for production-quality image-to-video generation.
# The HQ pipeline uses a second-order Res2s sampler (instead of Euler) and applies
# per-stage distilled LoRA strengths for optimal quality at full HD (1920×1088) output.

# We also include a `web` wrapper that makes it possible
# to trigger inference via an API call.
# For details, see the `/docs` route of the URL ending in `inference-web.modal.run`
# that appears when you deploy the app.

with image.imports():
    import imageio
    import torch

    # Workaround: the spatial upsampler and VAE decoder use conv3d under
    # @torch.inference_mode() decorators, which creates "inference tensors"
    # that fail with "cannot be saved for backward". Replacing inference_mode
    # with no_grad before importing ltx packages avoids this issue.
    torch.inference_mode = torch.no_grad

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

MINUTES = 60


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
)
class Inference:
    @modal.enter()
    def load_pipeline(self):
        from huggingface_hub import snapshot_download

        model_dir = f"{MODEL_PATH}/ltx2.3"

        # Download only the files we need (the full repo is large)
        snapshot_download(
            MODEL_REPO,
            local_dir=model_dir,
            allow_patterns=[
                CHECKPOINT_FILE,
                DISTILLED_LORA_FILE,
                SPATIAL_UPSAMPLER_FILE,
                "text_encoder/*",
                "tokenizer/*",
            ],
        )
        model_volume.commit()

        checkpoint_path = f"{model_dir}/{CHECKPOINT_FILE}"
        distilled_lora_path = f"{model_dir}/{DISTILLED_LORA_FILE}"
        upsampler_path = f"{model_dir}/{SPATIAL_UPSAMPLER_FILE}"

        distilled_lora = [
            LoraPathStrengthAndSDOps(
                distilled_lora_path,
                1.0,  # base strength; overridden by per-stage strengths below
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        self.pipeline = TI2VidTwoStagesHQPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            distilled_lora_strength_stage_1=0.25,
            distilled_lora_strength_stage_2=0.5,
            spatial_upsampler_path=upsampler_path,
            gemma_root=model_dir,
            loras=(),
        )

    @modal.method()
    def run(
        self,
        image_bytes: bytes,
        prompt: str,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        width = 960
        height = 544
        num_frames = num_frames or 121  # ~5 seconds at 24fps
        num_inference_steps = num_inference_steps or 15
        cfg_scale = cfg_scale or 3.0
        seed = seed or random.randint(0, 2**32 - 1)
        print(f"Seeding RNG with: {seed}")

        negative_prompt = (
            "worst quality, inconsistent motion, blurry, jittery, distorted"
        )

        # HQ pipeline uses Res2s sampler — no STG needed for optimal quality
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=0.0,
            rescale_scale=0.45,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[],
        )

        # Audio guider params (required by the pipeline, using v2.3 HQ defaults)
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=0.0,
            rescale_scale=1.0,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[],
        )

        # The pipeline expects image file paths, not bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            image_path = f.name

        mp4_name = (
            f"{seed}_{''.join(c if c.isalnum() else '-' for c in prompt[:100])}.mp4"
        )
        output_path = f"{OUTPUT_PATH}/{mp4_name}"

        # Pipeline returns (video_frames_iterator, audio_tensor)
        video_frames_iter, _audio = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=24.0,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=[(image_path, 0, 1.0)],
        )

        # Collect all frames and export to MP4
        frames = [frame for frame in video_frames_iter]
        video_tensor = torch.cat(frames, dim=0) if len(frames) > 1 else frames[0]

        # The two-stage HQ pipeline returns (T, H, W, C) uint8 tensors
        # Output is full HD (1920×1088) after 2x spatial upsampling
        video_np = video_tensor.cpu().numpy()
        imageio.mimwrite(output_path, video_np, fps=24, codec="libx264")

        output_volume.commit()
        torch.cuda.empty_cache()

        Path(image_path).unlink(missing_ok=True)
        return mp4_name

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(
        self,
        image_bytes: Annotated[bytes, fastapi.File()],
        prompt: str,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> fastapi.Response:
        mp4_name = self.run.local(  # run in the same container
            image_bytes=image_bytes,
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        return fastapi.responses.FileResponse(
            path=f"{OUTPUT_PATH}/{mp4_name}",
            media_type="video/mp4",
            filename=mp4_name,
        )


# ## Generating videos from the command line

# We add a [local entrypoint](https://modal.com/docs/reference/modal.App#local_entrypoint)
# that calls the `Inference.run` method to run inference from the command line.
# The function's parameters are automatically turned into a CLI.

# Run it with

# ```bash
# modal run image_to_video.py --prompt "A cat looking out the window at a snowy mountain" --image-path /path/to/cat.jpg
# ```

# You can also pass `--help` to see the full list of arguments.


@app.local_entrypoint()
def entrypoint(
    image_path: str,
    prompt: str,
    num_frames: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    seed: Optional[int] = None,
    twice: bool = True,
):
    import os
    import urllib.request

    print(f"🎥 Generating a video from the image at {image_path}")
    print(f"🎥 using the prompt {prompt}")

    if image_path.startswith(("http://", "https://")):
        image_bytes = urllib.request.urlopen(image_path).read()
    elif os.path.isfile(image_path):
        image_bytes = Path(image_path).read_bytes()
    else:
        raise ValueError(f"{image_path} is not a valid file or URL.")

    inference_service = Inference()

    for _ in range(1 + twice):
        start = time.time()
        mp4_name = inference_service.run.remote(
            image_bytes=image_bytes,
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        duration = time.time() - start
        print(f"🎥 Generated video in {duration:.3f}s")

        output_dir = Path("/tmp/ltx2_image_to_video")
        output_dir.mkdir(exist_ok=True, parents=True)
        local_output_path = output_dir / mp4_name
        local_output_path.write_bytes(b"".join(output_volume.read_file(mp4_name)))
        print(f"🎥 Video saved to {local_output_path}")


# ## Generating videos via an API

# The Modal `Cls` above also included a [`fastapi_endpoint`](https://modal.com/docs/examples/basic_web),
# which adds a simple web API to the inference method.

# To try it out, run

# ```bash
# modal deploy image_to_video.py
# ```

# copy the printed URL ending in `inference-web.modal.run`,
# and add `/docs` to the end. This will bring up the interactive
# Swagger/OpenAPI docs for the endpoint.

# ## Generating videos in a web UI

# Lastly, we add a simple front-end web UI (written in Alpine.js) for
# our image to video backend.

# This is also deployed when you run

# ```bash
# modal deploy image_to_video.py
# ```

# The `Inference` class will serve multiple users from its own auto-scaling pool of warm GPU containers automatically,
# and they will spin down when there are no requests.

frontend_path = Path(__file__).parent / "frontend"

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("jinja2==3.1.5", "fastapi[standard]==0.115.8")
    .add_local_dir(  # mount frontend/client code
        frontend_path, remote_path="/assets"
    )
)


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    import fastapi.templating

    web_app = fastapi.FastAPI()
    templates = fastapi.templating.Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: fastapi.Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Inference().web.get_web_url(),
                "model_name": "LTX-2.3 Image to Video",
                "default_prompt": "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background.",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app
