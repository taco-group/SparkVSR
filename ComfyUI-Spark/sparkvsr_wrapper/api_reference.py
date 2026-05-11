"""
Nano-Banana / fal.ai API-based reference frame generation for SparkVSR.

Uses fal-ai/nano-banana-pro/edit to super-resolve selected LR frames
and return them as [-1, 1] tensors aligned to the target resolution.
"""

import os
import hashlib
import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Super-Resolution and Restoration Task: Upscale this low-resolution image to "
    "high definition. Restore sharpness and clarity by removing degradation, blur, "
    "noise, grain, and compression artifacts."
)

FAL_MODEL_ID = "fal-ai/nano-banana-pro/edit"


def _file_hash(path: str) -> str:
    """Return a short MD5 hex digest of the file at *path*."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _save_frame_png(frame_tensor: torch.Tensor, path: str) -> None:
    """
    Save a single frame tensor as a PNG file.

    Accepts [C, H, W] in [0, 1], [0, 255], or [-1, 1].
    """
    from PIL import Image
    import numpy as np

    t = frame_tensor.float().cpu()
    if t.min() < 0:  # [-1, 1] → [0, 1]
        t = t.mul(0.5).add(0.5)
    if t.max() > 1.5:  # [0, 255] → [0, 1]
        t = t.div(255.0)
    t = t.clamp(0, 1).mul(255).to(torch.uint8)
    arr = t.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def generate_api_references(
    frames: torch.Tensor,
    ref_indices: List[int],
    api_key: Optional[str] = None,
    api_key_env: str = "NANO_BANANA_API_KEY",
    prompt: Optional[str] = None,
    output_dir: Optional[str] = None,
    cache: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    resolution: str = "1K",
) -> List[Tuple[int, torch.Tensor]]:
    """
    Call the Nano-Banana Pro (fal.ai) API to super-resolve selected frames.

    Args:
        frames:       Video frames. Accepted formats:
                        [F, H, W, C] float in [0, 1]  (ComfyUI)
                        [F, C, H, W] float in [0, 255] (SparkVSR internal)
                      Shape detection: if C <= 4 and last dim >= 3 → ComfyUI.
        ref_indices:  Frame indices to process.
        api_key:      Explicit API key. When empty, the key is read from the
                      environment variable named by *api_key_env* (default
                      ``NANO_BANANA_API_KEY``), then from ``FAL_KEY``.
        api_key_env:  Name of the environment variable that holds the key.
        prompt:       Prompt sent to the model. ``None`` uses the default SR prompt.
        output_dir:   Directory for caching results. ``None`` disables caching.
        cache:        Whether to skip API calls for already-cached frames.
        target_size:  Optional (H, W) to resize results to. Uses bicubic + AR crop.
        resolution:   API resolution hint: ``"1K"``, ``"2K"``, or ``"4K"``.

    Returns:
        List of ``(frame_index, tensor)`` pairs.
        Each tensor is [C, H, W] float32 in [-1, 1].
    """
    try:
        import fal_client
        import requests
        from io import BytesIO
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "API reference mode requires 'fal-client' and 'requests'. "
            "Install with: pip install fal-client requests"
        ) from exc

    from torchvision import transforms as T
    to_tensor = T.ToTensor()

    # ── Resolve API key ───────────────────────────────────────────────────────
    resolved_key = (
        api_key
        or os.environ.get(api_key_env, "")
        or os.environ.get("FAL_KEY", "")
    )
    if not resolved_key:
        raise ValueError(
            f"No API key found. Set the '{api_key_env}' environment variable "
            "or pass api_key= explicitly."
        )
    os.environ["FAL_KEY"] = resolved_key  # fal_client reads FAL_KEY

    # ── Normalise input tensor to [F, C, H, W] in [0, 1] ────────────────────
    if frames.ndim == 4:
        # Distinguish [F, H, W, C] (ComfyUI) from [F, C, H, W] (internal)
        if frames.shape[-1] in (1, 2, 3, 4):
            # Likely ComfyUI: [F, H, W, C] in [0, 1]
            frames_fchw = frames.permute(0, 3, 1, 2).float()
        else:
            frames_fchw = frames.float()
            if frames_fchw.max() > 1.5:
                frames_fchw = frames_fchw.div(255.0)
    else:
        raise ValueError(f"Unexpected frames shape: {frames.shape}")

    # ── Output directory ──────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    active_prompt = prompt or DEFAULT_PROMPT
    results: List[Tuple[int, torch.Tensor]] = []

    for idx in ref_indices:
        frame_tensor = frames_fchw[min(idx, frames_fchw.shape[0] - 1)]  # [C, H, W]

        # ── Cache lookup ───────────────────────────────────────────────────
        cache_path = None
        if cache and output_dir:
            cache_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
            if os.path.exists(cache_path):
                try:
                    img = Image.open(cache_path).convert("RGB")
                    t = to_tensor(img).mul(2.0).sub(1.0)
                    if target_size:
                        from sparkvsr_wrapper.reference import align_ref_frame
                        t = align_ref_frame(t, *target_size)
                    logger.info(f"[API] Cache hit for frame {idx}")
                    results.append((idx, t))
                    continue
                except Exception:
                    pass

        # ── Save LR frame to a temp file ──────────────────────────────────
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _save_frame_png(frame_tensor, tmp_path)

            # ── Upload and call API ───────────────────────────────────────
            logger.info(f"[API] Uploading frame {idx} …")
            image_url = fal_client.upload_file(tmp_path)

            logger.info(f"[API] Calling {FAL_MODEL_ID} for frame {idx} …")
            result = fal_client.run(
                FAL_MODEL_ID,
                arguments={
                    "prompt": active_prompt,
                    "image_urls": [image_url],
                    "num_images": 1,
                    "aspect_ratio": "auto",
                    "output_format": "png",
                    "resolution": resolution,
                },
            )

            output_url = result["images"][0]["url"]
            response = requests.get(output_url, timeout=120)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # ── Cache result ──────────────────────────────────────────────
            if cache_path:
                img.save(cache_path)

            t = to_tensor(img).mul(2.0).sub(1.0)  # [-1, 1]
            if target_size:
                from sparkvsr_wrapper.reference import align_ref_frame
                t = align_ref_frame(t, *target_size)

            logger.info(f"[API] Frame {idx} done → {t.shape[-1]}x{t.shape[-2]}")
            results.append((idx, t))

        except Exception as exc:
            logger.error(f"[API] Failed for frame {idx}: {exc}")
            # Return the upscaled LR frame as a graceful fallback
            fallback = frame_tensor.mul(2.0).sub(1.0)
            if target_size:
                import torch.nn.functional as TF
                fallback = TF.interpolate(
                    fallback.unsqueeze(0),
                    size=target_size,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            results.append((idx, fallback))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return results
