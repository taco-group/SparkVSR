"""Video writing and ComfyUI preview helpers for SparkVSR nodes."""

import os
from datetime import datetime
from typing import Optional, Tuple


def resolve_comfy_dir(kind: str = "output") -> str:
    """Return ComfyUI's output/temp directory, with a local fallback."""
    try:
        import folder_paths

        if kind == "temp":
            return folder_paths.get_temp_directory()
        return folder_paths.get_output_directory()
    except Exception:
        fallback = os.path.join(os.getcwd(), kind)
        os.makedirs(fallback, exist_ok=True)
        return fallback


def unique_video_path(output_dir: str, filename_prefix: str) -> Tuple[str, str]:
    """Build a timestamped, unique MP4 path and return (path, filename)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = filename_prefix.strip() or "sparkvsr"
    filename = f"{safe_prefix}_{timestamp}.mp4"
    out_path = os.path.join(output_dir, filename)

    counter = 0
    while os.path.exists(out_path):
        counter += 1
        filename = f"{safe_prefix}_{timestamp}_{counter:03d}.mp4"
        out_path = os.path.join(output_dir, filename)
    return out_path, filename


def frames_to_uint8_numpy(frames):
    """Convert ComfyUI IMAGE frames [N,H,W,C] in [0,1] to uint8 numpy."""
    import numpy as np
    import torch

    if isinstance(frames, torch.Tensor):
        return frames.float().clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
    arr = np.array(frames)
    if arr.dtype == np.uint8:
        return arr
    max_value = arr.max() if arr.size else 0
    if max_value > 1.5:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def write_video_file(
    frames,
    fps: float,
    out_path: str,
    format: str = "mp4_yuv420p",
) -> None:
    """Write ComfyUI IMAGE frames to an MP4 file."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio with ffmpeg support is required for saving videos. "
            "Install with: pip install imageio[ffmpeg] imageio-ffmpeg"
        ) from exc

    frames_np = frames_to_uint8_numpy(frames)
    pixel_format = "yuv444p" if format == "mp4_yuv444p" else "yuv420p"
    crf = "0" if pixel_format == "yuv444p" else "10"

    iio.imwrite(
        out_path,
        frames_np,
        fps=float(fps),
        codec="libx264",
        pixelformat=pixel_format,
        macro_block_size=None,
        ffmpeg_params=["-crf", crf],
    )


def build_video_preview(
    out_path: str,
    fps: float,
    media_type: str = "output",
    root_dir: Optional[str] = None,
) -> dict:
    """
    Build a preview payload understood by ComfyUI + VideoHelperSuite.

    The key is named ``gifs`` for compatibility with VHS, but the payload format
    is ``video/mp4``.
    """
    if root_dir is None:
        root_dir = resolve_comfy_dir("temp" if media_type == "temp" else "output")

    filename = os.path.basename(out_path)
    out_dir = os.path.dirname(out_path)
    try:
        rel = os.path.relpath(out_dir, root_dir)
        subfolder = "" if rel == "." or rel.startswith("..") else rel
    except Exception:
        subfolder = ""

    return {
        "filename": filename,
        "subfolder": subfolder,
        "type": media_type,
        "format": "video/mp4",
        "frame_rate": float(fps),
        "fullpath": out_path,
    }


def build_video_preview_ui(preview: dict) -> dict:
    """Build a UI payload compatible with both ComfyUI and VHS previews."""
    return {
        "images": [preview],
        "animated": (True,),
        "gifs": [preview],
    }
