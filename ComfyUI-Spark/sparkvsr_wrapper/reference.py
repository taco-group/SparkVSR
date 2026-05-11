"""
Reference-frame utilities for SparkVSR.

Handles:
  - parsing / auto-selecting reference frame indices
  - preparing external reference images
  - running PiSA-SR to generate HR reference frames (inline or via subprocess)
"""

import os
import logging
import subprocess
import shutil
import tempfile
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Global cache for the inline PiSA-SR model to avoid reloading each run.
_PISASR_MODEL_CACHE: dict = {}

logger = logging.getLogger(__name__)


# ── Index helpers ─────────────────────────────────────────────────────────────

def parse_ref_indices(ref_indices_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse a comma-separated list of frame indices.

    Returns ``None`` when the string is empty / ``None``, which means
    "auto-select".  Raises ``ValueError`` when consecutive indices are
    closer than 4 frames apart.
    """
    if not ref_indices_str or not ref_indices_str.strip():
        return None

    try:
        indices = sorted(set(int(x.strip()) for x in ref_indices_str.split(",") if x.strip()))
    except ValueError:
        raise ValueError(
            f"ref_indices must be comma-separated integers, got '{ref_indices_str}'"
        )

    for i in range(len(indices) - 1):
        gap = indices[i + 1] - indices[i]
        if gap < 4:
            raise ValueError(
                f"Reference frame indices must be at least 4 apart; "
                f"found gap={gap} between {indices[i]} and {indices[i+1]}."
            )
    return indices


def auto_select_ref_indices(
    total_frames: int,
    chunk_len: int = 0,
    overlap_t: int = 8,
    ref_per_chunk: int = 1,
) -> List[int]:
    """
    Automatically choose reference frame indices.

    When ``chunk_len > 0`` uses :func:`~sparkvsr_wrapper.preprocess.compute_adaptive_ref_indices`.
    Otherwise falls back to first / middle / last selection.
    """
    from sparkvsr_wrapper.preprocess import compute_adaptive_ref_indices

    if chunk_len > 0:
        indices, _ = compute_adaptive_ref_indices(
            total_frames, chunk_len, overlap_t, ref_per_chunk
        )
        return indices

    # Simple fallback
    if total_frames <= 0:
        return []
    if total_frames == 1:
        return [0]
    if total_frames == 2:
        return [0, 1]
    # first, middle, last – aligned to multiples of 4
    candidates = [0, (total_frames // 2 // 4) * 4, ((total_frames - 1) // 4) * 4]
    seen, result = set(), []
    for idx in candidates:
        idx = max(0, min(idx, total_frames - 1))
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


# ── Spatial alignment helpers ─────────────────────────────────────────────────

def center_crop_to_aspect_ratio(
    tensor: torch.Tensor, target_h: int, target_w: int
) -> torch.Tensor:
    """
    Center-crop a [C, H, W] tensor to match the aspect ratio of (target_h, target_w).
    """
    _, src_h, src_w = tensor.shape
    target_ar = target_w / target_h
    src_ar = src_w / src_h

    if abs(target_ar - src_ar) < 1e-3:
        return tensor

    if src_ar > target_ar:  # source is wider → crop width
        new_w = int(src_h * target_ar)
        start_w = (src_w - new_w) // 2
        return tensor[:, :, start_w: start_w + new_w]
    else:  # source is taller → crop height
        new_h = int(src_w / target_ar)
        start_h = (src_h - new_h) // 2
        return tensor[:, start_h: start_h + new_h, :]


def align_ref_frame(
    tensor: torch.Tensor, target_h: int, target_w: int
) -> torch.Tensor:
    """
    Align a [C, H, W] reference frame to (target_h, target_w).

    Step 1: center-crop to the target aspect ratio.
    Step 2: bicubic resize to the exact target resolution.
    """
    tensor = center_crop_to_aspect_ratio(tensor, target_h, target_w)
    if tensor.shape[-2:] != (target_h, target_w):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    return tensor


# ── External reference preparation ───────────────────────────────────────────

def prepare_external_references(
    ref_images: torch.Tensor,
    ref_indices: List[int],
    target_h: int,
    target_w: int,
) -> List[torch.Tensor]:
    """
    Prepare externally supplied reference frames.

    Args:
        ref_images:  [N, H, W, C] ComfyUI IMAGE tensor in [0, 1].
        ref_indices: List of N frame indices these images correspond to.
        target_h:    Expected height of the upscaled video.
        target_w:    Expected width  of the upscaled video.

    Returns:
        List of N tensors, each [C, target_h, target_w] in [-1, 1].
    """
    if ref_images.ndim != 4:
        raise ValueError(
            f"external_ref_frames must be [N, H, W, C], got shape {ref_images.shape}"
        )
    N = ref_images.shape[0]
    if N != len(ref_indices):
        raise ValueError(
            f"external_ref_frames has {N} frames but {len(ref_indices)} ref_indices supplied."
        )

    result = []
    for i in range(N):
        # [H, W, C] -> [C, H, W] in [-1, 1]
        t = ref_images[i].permute(2, 0, 1).float().mul(2.0).sub(1.0)
        t = align_ref_frame(t, target_h, target_w)
        result.append(t)
    return result


# ── PiSA-SR reference preparation ────────────────────────────────────────────

def _load_pisasr_model_inline(pisa_config: dict):
    """
    Load (or retrieve from cache) an inline PiSASR_eval instance.

    Cache key is (sd_model_path, checkpoint_path, device).
    """
    from sparkvsr_wrapper.pisasr_src import PiSASR_eval

    sd_model = pisa_config["sd_model_path"]
    ckpt     = pisa_config["checkpoint_path"]
    gpu_id   = str(pisa_config.get("gpu_id", "0")).strip()
    device   = f"cuda:{gpu_id}" if gpu_id.isdigit() else gpu_id
    cache_key = (sd_model, ckpt, device)

    if cache_key in _PISASR_MODEL_CACHE:
        logger.info("[PiSA-SR] Reusing cached inline model.")
        return _PISASR_MODEL_CACHE[cache_key]

    args = SimpleNamespace(
        pretrained_model_path  = sd_model,
        pretrained_path        = ckpt,
        device                 = device,
        seed                   = 42,
        process_size           = 512,
        upscale                = 4,
        align_method           = "adain",
        lambda_pix             = 1.0,
        lambda_sem             = 1.0,
        vae_decoder_tiled_size = 224,
        vae_encoder_tiled_size = 1024,
        latent_tiled_size      = 96,
        latent_tiled_overlap   = 32,
        mixed_precision        = "fp16",
        default                = True,
    )

    logger.info(f"[PiSA-SR] Loading inline model: sd={sd_model!r}  ckpt={ckpt!r}  device={device!r}")
    model = PiSASR_eval(args)
    model.set_eval()
    _PISASR_MODEL_CACHE[cache_key] = model
    return model


def prepare_pisasr_references_inline(
    video_lr: torch.Tensor,
    ref_indices: List[int],
    upscale: int,
    pisa_config: dict,
    target_h: int,
    target_w: int,
) -> List[torch.Tensor]:
    """
    Generate HR reference frames using the bundled PiSA-SR (in-process, no subprocess).

    Same signature and return format as :func:`prepare_pisasr_references`.
    """
    from PIL import Image
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    cache_dir = pisa_config.get("cache_dir") or ""
    if cache_dir:
        if os.path.isfile(cache_dir):
            logger.warning(
                f"[PiSA-SR] cache_dir '{cache_dir}' points to an existing file — ignoring."
            )
            cache_dir = ""
        else:
            os.makedirs(cache_dir, exist_ok=True)

    model = _load_pisasr_model_inline(pisa_config)
    device = model.device
    results = []

    for idx in ref_indices:
        cache_file = None
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"frame_{idx:05d}.png")
            if os.path.exists(cache_file):
                logger.info(f"[PiSA-SR] Cache hit for frame {idx}: {cache_file}")
                img = Image.open(cache_file).convert("RGB")
                t = to_tensor(img).mul(2.0).sub(1.0)
                t = align_ref_frame(t, target_h, target_w)
                results.append(t)
                continue

        # Build input tensor
        frame_np = (
            video_lr[idx].clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        )
        lr_img = Image.fromarray(frame_np)

        # PiSA-SR expects the image pre-upscaled to the target resolution
        ori_w, ori_h = lr_img.size
        scaled_w = ori_w * upscale
        scaled_h = ori_h * upscale
        scaled_w -= scaled_w % 8
        scaled_h -= scaled_h % 8
        lr_img = lr_img.resize((scaled_w, scaled_h), Image.LANCZOS)

        logger.info(f"[PiSA-SR] Running inline inference for frame {idx} ({scaled_w}x{scaled_h}) …")
        try:
            c_t = to_tensor(lr_img).unsqueeze(0).to(device) * 2 - 1
            _, output_image = model(model.args.default, c_t, prompt="")
            output_image = output_image * 0.5 + 0.5
            output_image = torch.clip(output_image, 0, 1)

            out_pil = transforms.ToPILImage()(output_image[0].cpu())

            if cache_file:
                out_pil.save(cache_file)

            t = to_tensor(out_pil).mul(2.0).sub(1.0)
            t = align_ref_frame(t, target_h, target_w)
            logger.info(f"[PiSA-SR] Frame {idx} → {t.shape[-1]}x{t.shape[-2]}")
            results.append(t)
        except Exception as exc:
            logger.error(f"[PiSA-SR] Inline inference failed for frame {idx}: {exc}")
            t = F.interpolate(
                video_lr[idx].unsqueeze(0).div(255.0).mul(2.0).sub(1.0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            results.append(t)

    return results


def prepare_pisasr_references(
    video_lr: torch.Tensor,
    ref_indices: List[int],
    upscale: int,
    pisa_config: dict,
    target_h: int,
    target_w: int,
) -> List[torch.Tensor]:
    """
    Generate HR reference frames using PiSA-SR.

    When ``pisa_config["script_path"]`` is empty the bundled in-process
    implementation is used (recommended for deployment).  When a script path
    is provided the original subprocess call is used as a legacy fallback.

    Required keys in pisa_config: ``sd_model_path``, ``checkpoint_path``.
    Optional keys: ``script_path``, ``python_executable`` (legacy subprocess),
    ``gpu_id``, ``cache_dir``.
    """
    for k in ("sd_model_path", "checkpoint_path"):
        if not pisa_config.get(k):
            if k == "checkpoint_path":
                raise ValueError(
                    "PiSA-SR: pisa_sr.pkl checkpoint not found.\n"
                    "  ➜ Place it at: ComfyUI/models/loras/pisa_sr.pkl  (auto-detected)\n"
                    "  ➜ Or set 'pisa_checkpoint_path' explicitly in the SparkVSR Prepare Reference node.\n"
                    "  Download: https://huggingface.co/jiangyzy/PiSA-SR"
                )
            raise ValueError(
                f"PiSA-SR mode requires pisa_config['{k}'] to be set. "
                f"Got: {pisa_config.get(k)!r}"
            )

    # Sanity-check: checkpoint_path must end with .pkl (not a Python path / dir)
    ckpt = pisa_config["checkpoint_path"]
    if not ckpt.endswith(".pkl"):
        raise ValueError(
            f"pisa_checkpoint_path looks wrong: '{ckpt}'\n"
            "Expected a path ending in .pkl (e.g. /path/to/pisa_sr.pkl).\n"
            "This usually means the workflow widget values are misaligned with the "
            "current node definition. Please reload the SparkVSR default workflow "
            "(File → Load Default or clear the canvas and refresh)."
        )

    script_path = pisa_config.get("script_path") or ""

    # ── Inline mode (default when no script_path is given) ───────────────────
    if not script_path:
        return prepare_pisasr_references_inline(
            video_lr=video_lr,
            ref_indices=ref_indices,
            upscale=upscale,
            pisa_config=pisa_config,
            target_h=target_h,
            target_w=target_w,
        )

    # ── Legacy subprocess mode ────────────────────────────────────────────────
    from PIL import Image
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    cache_dir = pisa_config.get("cache_dir") or ""
    if cache_dir:
        if os.path.isfile(cache_dir):
            logger.warning(
                f"[PiSA-SR] cache_dir '{cache_dir}' points to an existing file — ignoring."
            )
            cache_dir = ""
        else:
            os.makedirs(cache_dir, exist_ok=True)

    gpu_id = str(pisa_config.get("gpu_id", "0"))
    python_exe = pisa_config.get("python_executable") or "python"
    results = []

    for idx in ref_indices:
        cache_file = None
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"frame_{idx:05d}.png")
            if os.path.exists(cache_file):
                logger.info(f"[PiSA-SR] Cache hit for frame {idx}: {cache_file}")
                img = Image.open(cache_file).convert("RGB")
                t = to_tensor(img).mul(2.0).sub(1.0)
                t = align_ref_frame(t, target_h, target_w)
                results.append(t)
                continue

        frame_np = (
            video_lr[idx].clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        )
        lr_img = Image.fromarray(frame_np)

        with tempfile.TemporaryDirectory() as tmpdir:
            lr_path = os.path.join(tmpdir, "input_frame.png")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir, exist_ok=True)
            lr_img.save(lr_path)

            cmd = [
                python_exe, script_path,
                "--input_image", lr_path,
                "--output_dir", out_dir,
                "--pretrained_model_path", pisa_config["sd_model_path"],
                "--pretrained_path", pisa_config["checkpoint_path"],
                "--upscale", str(upscale),
                "--align_method", "adain",
                "--lambda_pix", "1.0",
                "--lambda_sem", "1.0",
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            pisa_cwd = os.path.dirname(script_path)

            logger.info(f"[PiSA-SR] Subprocess for frame {idx} …")
            try:
                subprocess.run(
                    cmd, env=env, check=True,
                    capture_output=True, text=True, cwd=pisa_cwd,
                )
            except subprocess.CalledProcessError as exc:
                logger.error(
                    f"[PiSA-SR] Subprocess failed for frame {idx} "
                    f"(exit {exc.returncode}):\n{exc.stderr}"
                )
                t = F.interpolate(
                    video_lr[idx].unsqueeze(0).div(255.0).mul(2.0).sub(1.0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                results.append(t)
                continue

            out_img_path = os.path.join(out_dir, "input_frame.png")
            if not os.path.exists(out_img_path):
                logger.warning(f"[PiSA-SR] Output missing for frame {idx}; using upscaled LR.")
                t = F.interpolate(
                    video_lr[idx].unsqueeze(0).div(255.0).mul(2.0).sub(1.0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                results.append(t)
                continue

            if cache_file:
                shutil.copy(out_img_path, cache_file)

            img = Image.open(out_img_path).convert("RGB")
            t = to_tensor(img).mul(2.0).sub(1.0)
            t = align_ref_frame(t, target_h, target_w)
            logger.info(f"[PiSA-SR] Frame {idx} → {t.shape[-1]}x{t.shape[-2]}")
            results.append(t)

    return results

