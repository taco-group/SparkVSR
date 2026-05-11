"""
Video preprocessing utilities for SparkVSR.

Handles frame format conversion, temporal/spatial padding, upscaling,
and the extra CogVideoX-compatible padding.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def preprocess_frames(
    frames: torch.Tensor,
    upscale: int = 4,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
    upscale_mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, Tuple[int, int]]:
    """
    Preprocess ComfyUI image frames for SparkVSR inference.

    Args:
        frames:      [F, H, W, C] float32 in [0, 1] — ComfyUI IMAGE format.
        upscale:     Integer upscale factor applied to H and W when
                     target_h / target_w are not specified.
        target_h:    Explicit target height (overrides upscale).
        target_w:    Explicit target width  (overrides upscale).
        upscale_mode: Interpolation mode for torch.nn.functional.interpolate.

    Returns:
        video_up:      [1, C, F_pad, H_up_pad, W_up_pad] float32 in [-1, 1].
                       Ready to pass directly to the SparkVSR pipeline.
        video_lr:      [F_pad, C, H_pad, W_pad] float32 in [0, 255].
                       Spatially-padded LR frames used by PiSA-SR.
        pad_f:         Number of temporal frames appended to reach the
                       (F-1) % 8 == 0 requirement.
        remove_pad_h:  Height pixels that must be stripped from the output.
        remove_pad_w:  Width  pixels that must be stripped from the output.
        original_hw:   (H_orig, W_orig) before any padding.
    """
    assert frames.ndim == 4, f"Expected [F, H, W, C], got shape {frames.shape}"

    F_orig, H_orig, W_orig, C = frames.shape
    original_hw = (H_orig, W_orig)

    # [F, H, W, C] -> [F, C, H, W] in [0, 255]
    video = frames.permute(0, 3, 1, 2).float().mul(255.0)

    # ── Temporal padding: (F-1) % 8 == 0 ────────────────────────────────────
    pad_f = 0
    F_cur = F_orig
    remainder = (F_cur - 1) % 8
    if remainder != 0:
        pad_f = 8 - remainder
        last_frame = video[-1:]  # [1, C, H, W]
        video = torch.cat([video, last_frame.expand(pad_f, -1, -1, -1)], dim=0)
        F_cur = F_cur + pad_f

    # ── Spatial padding: H and W multiples of 4 ──────────────────────────────
    pad_h_s4 = (4 - H_orig % 4) % 4
    pad_w_s4 = (4 - W_orig % 4) % 4
    if pad_h_s4 > 0 or pad_w_s4 > 0:
        # pad order: (left, right, top, bottom) for last two dims
        video = F.pad(video, (0, pad_w_s4, 0, pad_h_s4))

    H_pad = H_orig + pad_h_s4
    W_pad = W_orig + pad_w_s4
    video_lr = video  # [F_pad, C, H_pad, W_pad] in [0, 255]

    # ── Determine target resolution ──────────────────────────────────────────
    if target_h is not None and target_w is not None:
        t_h, t_w = int(target_h), int(target_w)
        using_explicit_target = True
    else:
        # Use the spatially-padded dimensions as the upscale base, matching the
        # original inference script which reads H/W after padding to multiples of 4.
        t_h = H_pad * upscale
        t_w = W_pad * upscale
        using_explicit_target = False

    # ── Upscale ──────────────────────────────────────────────────────────────
    interp_kwargs = {"size": (t_h, t_w), "mode": upscale_mode}
    if upscale_mode in ("bilinear", "bicubic"):
        interp_kwargs["align_corners"] = False
    video_up = F.interpolate(video, **interp_kwargs)

    # ── Extra pad to multiples of 16 for CogVideoX patch embedding ───────────
    pad_h_16 = (16 - t_h % 16) % 16
    pad_w_16 = (16 - t_w % 16) % 16
    if pad_h_16 > 0 or pad_w_16 > 0:
        video_up = F.pad(video_up, (0, pad_w_16, 0, pad_h_16))

    # ── Compute how much to strip from the final output ──────────────────────
    if using_explicit_target:
        # The original spatial padding was folded into the crop; only the
        # CogVideoX alignment padding needs to be removed.
        remove_pad_h = pad_h_16
        remove_pad_w = pad_w_16
    else:
        # The upscaled spatial padding *and* the alignment padding must be removed.
        remove_pad_h = pad_h_s4 * upscale + pad_h_16
        remove_pad_w = pad_w_s4 * upscale + pad_w_16

    # ── Normalize to [-1, 1] ─────────────────────────────────────────────────
    video_up = video_up.div(255.0).mul(2.0).sub(1.0)

    # [F_pad, C, H_up, W_up] -> [1, C, F_pad, H_up, W_up]
    video_up = video_up.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

    return video_up, video_lr, pad_f, remove_pad_h, remove_pad_w, original_hw


def remove_padding_and_extra_frames(
    video: torch.Tensor,
    pad_f: int,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    """
    Strip temporal and spatial padding added by :func:`preprocess_frames`.

    Args:
        video: [B, C, F, H, W]
        pad_f: Temporal frames to remove from the end.
        pad_h: Height  pixels to remove from the bottom.
        pad_w: Width   pixels to remove from the right.

    Returns:
        Cropped tensor with the same batch/channel dims.
    """
    if pad_f > 0:
        video = video[:, :, :-pad_f, :, :]
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]
    return video


# ── Tiling helpers (copied verbatim from sparkvsr_inference_script.py) ────────

def make_temporal_chunks(F: int, chunk_len: int, overlap_t: int = 8):
    """Split F frames into overlapping temporal chunks."""
    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap_t")

    chunk_starts = list(range(0, F - overlap_t, effective_stride))
    if chunk_starts[-1] + chunk_len < F:
        chunk_starts.append(F - chunk_len)

    time_chunks = []
    for t_start in chunk_starts:
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk_len:
        # Replace the under-sized last chunk with one that starts at F-chunk_len,
        # ensuring the chunk is exactly chunk_len frames and VAE-aligned ((n-1)%4==0).
        time_chunks[-1] = (F - chunk_len, F)

    return time_chunks


def make_spatial_tiles(H: int, W: int, tile_size_hw, overlap_hw=(32, 32)):
    """Split (H, W) into overlapping spatial tiles."""
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w
    if tile_stride_h <= 0 or tile_stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, H - overlap_h, tile_stride_h))
    if not h_tiles or h_tiles[-1] + tile_height < H:
        h_tiles.append(H - tile_height)
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_height > H:
        h_tiles.pop()

    w_tiles = list(range(0, W - overlap_w, tile_stride_w))
    if not w_tiles or w_tiles[-1] + tile_width < W:
        w_tiles.append(W - tile_width)
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_width > W:
        w_tiles.pop()

    spatial_tiles = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_height, H)
        if h_end + tile_stride_h > H:
            h_end = H
        for w_start in w_tiles:
            w_end = min(w_start + tile_width, W)
            if w_end + tile_stride_w > W:
                w_end = W
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    return spatial_tiles


def get_valid_tile_region(
    t_start, t_end, h_start, h_end, w_start, w_end,
    video_shape, overlap_t, overlap_h, overlap_w,
):
    """Return the valid (non-overlapping) region for a tile and its destination in the output."""
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end   = t_len if t_end == F else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end   = h_len if h_end == H else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end   = w_len if w_end == W else w_len - overlap_w // 2

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_t_start": t_start + valid_t_start, "out_t_end": t_start + valid_t_end,
        "out_h_start": h_start + valid_h_start, "out_h_end": h_start + valid_h_end,
        "out_w_start": w_start + valid_w_start, "out_w_end": w_start + valid_w_end,
    }


def compute_adaptive_ref_indices(total_frames: int, chunk_len: int, overlap_t: int, ref_per_chunk: int = 1):
    """
    Select reference frame indices aligned to chunk kept-regions and latent space
    multiples-of-4 constraint.
    """
    chunks = make_temporal_chunks(total_frames, chunk_len, overlap_t)
    ref_per_chunk = max(1, ref_per_chunk)

    ref_indices = []
    for (t_start, t_end) in chunks:
        t_len = t_end - t_start
        valid_s = 0 if t_start == 0 else overlap_t // 2
        valid_e = t_len if t_end == total_frames else t_len - overlap_t // 2
        keep_start = t_start + valid_s
        keep_end   = t_start + valid_e
        for j in range(ref_per_chunk):
            pos = keep_start if j == 0 else keep_start + (keep_end - keep_start) * j // ref_per_chunk
            pos = (pos // 4) * 4
            pos = max(0, min(pos, total_frames - 1))
            ref_indices.append(pos)

    ref_indices = sorted(set(ref_indices))
    filtered = [ref_indices[0]] if ref_indices else []
    for idx in ref_indices[1:]:
        if idx - filtered[-1] >= 4:
            filtered.append(idx)

    return filtered, chunks
