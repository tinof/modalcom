"""PiSA-SR in-process reference frame super-resolution.

Attribution:
Inference code is vendored verbatim under `pisasr_src/` from
taco-group/SparkVSR (ComfyUI-Spark/sparkvsr_wrapper/pisasr_src), which is itself
adapted from https://github.com/csslc/PiSA-SR. Both are Apache-2.0.

This module is only a thin adapter: it builds the argument object PiSASR_eval
expects, mirrors the official `test_pisasr.py` inference loop (LANCZOS
pre-upscale, multiple-of-8 crop, [-1,1] input, AdaIN color fix against the
source), and frees GPU memory when done.
"""

import gc
import os
from types import SimpleNamespace
from typing import Union

import numpy as np
import torch
from PIL import Image

from ..config import (
    PISASR_LAMBDA_PIX,
    PISASR_LAMBDA_SEM,
    PISASR_SD21_DIR,
    PISASR_WEIGHTS_PATH,
)

# Official PiSA-SR defaults (csslc/PiSA-SR test_pisasr.py)
_VAE_ENCODER_TILED_SIZE = 1024
_VAE_DECODER_TILED_SIZE = 224
_LATENT_TILED_SIZE = 96
_LATENT_TILED_OVERLAP = 32
_MIXED_PRECISION = "fp16"
_ALIGN_METHOD = "adain"


class PiSASRWeightsMissing(RuntimeError):
    """Raised when the operator-supplied pisa_sr.pkl is not on the models Volume."""


class PiSASRModel:
    """PiSA-SR runner that loads on demand and frees GPU memory when closed."""

    def __init__(
        self,
        sd21_dir: str = PISASR_SD21_DIR,
        weights_path: str = PISASR_WEIGHTS_PATH,
        device: str = "cuda",
    ):
        self.sd21_dir = sd21_dir
        self.weights_path = weights_path
        self.device = device
        self.model = None
        self._load()

    def _load(self):
        """Instantiate the vendored PiSASR_eval model."""
        if not os.path.isdir(self.sd21_dir):
            raise PiSASRWeightsMissing(
                f"Stable Diffusion 2.1 base weights not found at {self.sd21_dir}. "
                "Run `modal run -m modal_app.download_weights` to provision them."
            )
        if not os.path.isfile(self.weights_path):
            raise PiSASRWeightsMissing(
                f"PiSA-SR weights not found at {self.weights_path}.\n"
                "pisa_sr.pkl (~32 MB) has no public programmatic download; the upstream "
                "project distributes it via Google Drive only. Fetch it from "
                "https://github.com/csslc/PiSA-SR (Google Drive link in their README) and "
                "upload it once with:\n"
                f"  modal volume put sparkvsr-models /path/to/pisa_sr.pkl pisasr/pisa_sr.pkl\n"
                "Alternatively run with --ref-mode api or --ref-mode no_ref."
            )

        from .pisasr_src import PiSASR_eval

        args = SimpleNamespace(
            device=self.device,
            pretrained_model_path=self.sd21_dir,
            pretrained_path=self.weights_path,
            mixed_precision=_MIXED_PRECISION,
            lambda_pix=PISASR_LAMBDA_PIX,
            lambda_sem=PISASR_LAMBDA_SEM,
            vae_encoder_tiled_size=_VAE_ENCODER_TILED_SIZE,
            vae_decoder_tiled_size=_VAE_DECODER_TILED_SIZE,
            latent_tiled_size=_LATENT_TILED_SIZE,
            latent_tiled_overlap=_LATENT_TILED_OVERLAP,
            default=False,
            align_method=_ALIGN_METHOD,
        )

        print(f"Loading PiSA-SR (SD 2.1 base from {self.sd21_dir}, weights {self.weights_path})...")
        self.model = PiSASR_eval(args)
        self.model.set_eval()
        print("PiSA-SR loaded.")

    @torch.no_grad()
    def upscale(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        target_width: int,
        target_height: int,
    ) -> torch.Tensor:
        """Restore one frame at the target resolution.

        Returns a tensor [C, target_height, target_width] in [0, 1].
        Mirrors the official test_pisasr.py loop.
        """
        from torchvision import transforms
        from torchvision.transforms import functional as TF

        from .pisasr_src.src.my_utils.wavelet_color_fix import (
            adain_color_fix,
            wavelet_color_fix,
        )

        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            pil = Image.fromarray(image).convert("RGB")
        else:
            arr = image
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            pil = Image.fromarray(arr).convert("RGB")

        # Pre-upscale to the target resolution, then crop to a multiple of 8 for the VAE.
        upscaled = pil.resize((target_width, target_height), Image.LANCZOS)
        crop_w = upscaled.width - upscaled.width % 8
        crop_h = upscaled.height - upscaled.height % 8
        if (crop_w, crop_h) != upscaled.size:
            upscaled = upscaled.resize((crop_w, crop_h), Image.LANCZOS)

        c_t = TF.to_tensor(upscaled).unsqueeze(0).to(self.device) * 2 - 1
        _, output_image = self.model(False, c_t, prompt="")

        output_image = torch.clip(output_image * 0.5 + 0.5, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        # Colour alignment against the source keeps broadcast masters from drifting in hue.
        if _ALIGN_METHOD == "adain":
            output_pil = adain_color_fix(target=output_pil, source=upscaled)
        elif _ALIGN_METHOD == "wavelet":
            output_pil = wavelet_color_fix(target=output_pil, source=upscaled)

        if output_pil.size != (target_width, target_height):
            output_pil = output_pil.resize((target_width, target_height), Image.LANCZOS)

        rgb = np.array(output_pil)
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def close(self):
        """Release all GPU memory held by PiSA-SR."""
        if self.model is not None:
            for attr in ("unet", "vae", "text_encoder"):
                if hasattr(self.model, attr):
                    setattr(self.model, attr, None)
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("PiSA-SR reference model unloaded and GPU memory released.")
