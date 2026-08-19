"""Reference keyframe generation for SparkVSR.

Supports three modes:
1. `pisasr` (default): in-service open-source image super-resolution run in-process, then unloaded
2. `api`: fal.ai nano-banana-pro/edit using fixed restoration prompt
3. `no_ref`: reference-free blind restoration baseline
"""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ..config import (
    REF_MODE_API,
    REF_MODE_NO_REF,
    REF_MODE_PISASR,
    VALID_REF_MODES,
)

FAL_RESTORATION_PROMPT = (
    "high quality, extremely detailed, master quality, sharp focus, "
    "4k uhd, 8k resolution, ultra hd, crystal clear, photorealistic, "
    "remove blur, remove compression artifacts, denoise, enhance details"
)


def _load_cached_ref(ref_path: Path) -> Optional[torch.Tensor]:
    """Load cached reference frame if it exists on disk."""
    if ref_path.is_file() and ref_path.stat().st_size > 0:
        try:
            with Image.open(ref_path) as img:
                rgb = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            return tensor
        except Exception as e:
            print(f"Warning: could not read cached reference {ref_path}: {e}")
    return None


def _save_cached_ref(tensor: torch.Tensor, ref_path: Path):
    """Save reference tensor [C, H, W] to disk as PNG."""
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    uint8_img = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(uint8_img).save(ref_path, format="PNG")


def generate_references_fal_api(
    raw_frame_dict: Dict[int, Image.Image],
    target_width: int,
    target_height: int,
    cache_dir: Path,
) -> Dict[int, torch.Tensor]:
    """Generate reference frames via fal.ai nano-banana-pro/edit."""
    import fal_client

    fal_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    if not fal_key:
        raise RuntimeError(
            "ref_mode='api' requires the FAL_KEY environment variable. "
            "Ensure the 'fal' Secret is configured in Modal."
        )

    results = {}
    for frame_idx, pil_img in raw_frame_dict.items():
        ref_path = cache_dir / f"ref_frame_{frame_idx:06d}.png"
        cached = _load_cached_ref(ref_path)
        if cached is not None:
            print(f"Reusing cached reference frame {frame_idx} from {ref_path}")
            results[frame_idx] = cached
            continue

        print(f"Calling fal.ai nano-banana-pro/edit for reference frame {frame_idx}...")
        try:
            # Convert PIL image to PNG bytes for upload
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            image_url = fal_client.upload(img_bytes, "image/png")

            handler = fal_client.submit(
                "fal-ai/nano-banana-pro/edit",
                arguments={
                    "image_url": image_url,
                    "prompt": FAL_RESTORATION_PROMPT,
                    "num_inference_steps": 28,
                    "guidance_scale": 7.5,
                },
            )
            result = handler.get()
            images = result.get("images", [])
            if not images or "url" not in images[0]:
                raise ValueError(f"fal API returned unexpected response: {result}")

            import requests

            resp = requests.get(images[0]["url"], timeout=60)
            resp.raise_for_status()
            restored_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")

            # Resize to target resolution if needed
            if restored_pil.size != (target_width, target_height):
                restored_pil = restored_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)

            rgb = np.array(restored_pil)
            ref_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            _save_cached_ref(ref_tensor, ref_path)
            results[frame_idx] = ref_tensor
        except Exception as e:
            raise RuntimeError(
                f"Reference generation via fal API failed for frame {frame_idx}: {e}"
            ) from e

    return results


def generate_references_pisasr(
    raw_frame_dict: Dict[int, Image.Image],
    target_width: int,
    target_height: int,
    cache_dir: Path,
) -> Dict[int, torch.Tensor]:
    """Generate reference frames using in-process PiSA-SR model, then unload model."""
    from .pisasr import PiSASRModel, PiSASRWeightsMissing

    results = {}
    pending_frames = {}

    for frame_idx, pil_img in raw_frame_dict.items():
        ref_path = cache_dir / f"ref_frame_{frame_idx:06d}.png"
        cached = _load_cached_ref(ref_path)
        if cached is not None:
            print(f"Reusing cached reference frame {frame_idx} from {ref_path}")
            results[frame_idx] = cached
        else:
            pending_frames[frame_idx] = (pil_img, ref_path)

    if pending_frames:
        print(f"Generating {len(pending_frames)} reference frames using PiSA-SR...")
        model = None
        try:
            model = PiSASRModel()
            for frame_idx, (pil_img, ref_path) in pending_frames.items():
                print(f"Upscaling reference frame {frame_idx} -> {target_width}x{target_height}...")
                ref_tensor = model.upscale(pil_img, target_width, target_height)
                _save_cached_ref(ref_tensor, ref_path)
                results[frame_idx] = ref_tensor
        except PiSASRWeightsMissing:
            # Already carries operator instructions; wrapping would bury them.
            raise
        except Exception as e:
            raise RuntimeError(
                f"PiSA-SR reference generation failed: {e}. "
                "Select ref_mode='api' or 'no_ref' to proceed without it."
            ) from e
        finally:
            if model is not None:
                model.close()

    return results


def generate_reference_keyframes(
    ref_mode: str,
    raw_frame_dict: Dict[int, Image.Image],
    target_width: int,
    target_height: int,
    job_dir: Union[str, Path],
) -> Dict[int, torch.Tensor]:
    """Dispatch reference keyframe generation based on chosen ref_mode.

    Args:
        ref_mode: One of "pisasr", "api", "no_ref"
        raw_frame_dict: Dict mapping global frame index -> PIL Image of original source frame
        target_width: Target video output width
        target_height: Target video output height
        job_dir: Root directory for the current job on the IO Volume

    Returns:
        Dict mapping global frame index -> restored reference tensor [C, target_height, target_width] in [0, 1]
    """
    mode = (ref_mode or REF_MODE_PISASR).strip().lower()
    if mode not in VALID_REF_MODES:
        raise ValueError(f"Unknown ref_mode '{ref_mode}'. Must be one of {sorted(VALID_REF_MODES)}.")

    if mode == REF_MODE_NO_REF or not raw_frame_dict:
        print("Reference mode is 'no_ref': continuing with blind super-resolution.")
        return {}

    cache_dir = Path(job_dir) / "references"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if mode == REF_MODE_PISASR:
        return generate_references_pisasr(raw_frame_dict, target_width, target_height, cache_dir)
    elif mode == REF_MODE_API:
        return generate_references_fal_api(raw_frame_dict, target_width, target_height, cache_dir)
    else:
        raise ValueError(f"Unhandled ref_mode '{mode}'")
