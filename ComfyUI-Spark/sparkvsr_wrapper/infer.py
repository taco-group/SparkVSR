"""
Core SparkVSR inference functions.

Contains :func:`process_video_ref_i2v` (single-tile forward pass) and
:func:`run_sparkvsr` (full tiled inference loop).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ── Rotary-embedding helpers ──────────────────────────────────────────────────

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    """
    Compute (top-left, bottom-right) crop coordinates for a rotary-embedding
    grid of size *src* = (h, w) to fit a *tgt_height × tgt_width* canvas.
    """
    tw, th = tgt_width, tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width  = int(round(th / h * w))
    else:
        resize_width  = tw
        resize_height = int(round(tw / w * h))
    crop_top  = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width)  / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the RoPE (freqs_cos, freqs_sin) tensors for CogVideoX."""
    from diffusers.models.embeddings import get_3d_rotary_pos_embed

    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width  = width  // (vae_scale_factor_spatial * transformer_config.patch_size)

    p   = transformer_config.patch_size
    p_t = transformer_config.patch_size_t

    base_size_width  = transformer_config.sample_width  // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        base_num_frames = (num_frames + p_t - 1) // p_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(
                max(base_size_height, grid_height),
                max(base_size_width,  grid_width),
            ),
            device=device,
        )

    return freqs_cos, freqs_sin


# ── Single-tile forward pass ──────────────────────────────────────────────────

@torch.no_grad()
def process_video_ref_i2v(
    pipe,
    video: torch.Tensor,
    prompt: str = "",
    ref_frames: List[torch.Tensor] = [],
    ref_indices: List[int] = [],
    chunk_start_idx: int = 0,
    noise_step: int = 0,
    sr_noise_step: int = 399,
    empty_prompt_embedding: Optional[torch.Tensor] = None,
    ref_guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Run a single SparkVSR forward pass on one spatial / temporal tile.

    Args:
        pipe:                  :class:`CogVideoXImageToVideoPipeline` instance.
        video:                 [B, C, F, H, W] float in [-1, 1].
        prompt:                Text conditioning string.
        ref_frames:            List of [C, H, W] reference tensors in [-1, 1].
        ref_indices:           Global frame indices corresponding to ref_frames.
        chunk_start_idx:       Global temporal offset of this tile (t_start).
        noise_step:            LQ noise level (0 = no noise added).
        sr_noise_step:         Denoising timestep (default 399).
        empty_prompt_embedding: Pre-computed empty-prompt embedding (optional).
        ref_guidance_scale:    CFG scale for reference guidance.

    Returns:
        Restored tile [B, C, F, H, W] float in [0, 1].
    """
    video = video.to(pipe.device, dtype=pipe.dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    lq_latent   = latent_dist.sample() * pipe.vae.config.scaling_factor

    batch_size, num_channels, num_frames, height, width = lq_latent.shape
    device = lq_latent.device
    dtype  = lq_latent.dtype

    # ── Build reference latent ────────────────────────────────────────────────
    full_ref_latent = torch.zeros_like(lq_latent)
    for i, idx in enumerate(ref_indices):
        if i >= len(ref_frames):
            break
        local_frame_idx = idx - chunk_start_idx
        target_lat_idx  = local_frame_idx // 4
        if 0 <= target_lat_idx < num_frames:
            r_frame = ref_frames[i].to(device, dtype=dtype)  # [C, H, W]
            chunk   = r_frame.unsqueeze(0).unsqueeze(2).expand(1, -1, 4, -1, -1).clone()
            lat     = pipe.vae.encode(chunk).latent_dist.sample() * pipe.vae.config.scaling_factor
            full_ref_latent[:, :, target_lat_idx, :, :] = lat[0, :, 0, :, :]

    # ── CFG input assembly ────────────────────────────────────────────────────
    do_cfg = abs(ref_guidance_scale - 1.0) > 1e-3
    if do_cfg:
        input_latent_cond   = torch.cat([lq_latent, full_ref_latent], dim=1)
        uncond_ref_latent   = torch.zeros_like(full_ref_latent)
        input_latent_uncond = torch.cat([lq_latent, uncond_ref_latent], dim=1)
        input_latent        = torch.cat([input_latent_uncond, input_latent_cond], dim=0)
    else:
        input_latent = torch.cat([lq_latent, full_ref_latent], dim=1)

    # ── Patch-size-t padding ──────────────────────────────────────────────────
    patch_size_t = pipe.transformer.config.patch_size_t
    ncopy = 0
    if patch_size_t is not None:
        ncopy = input_latent.shape[2] % patch_size_t
        if ncopy > 0:
            first_frame  = input_latent[:, :, :1, :, :]
            input_latent = torch.cat(
                [first_frame.expand(-1, -1, ncopy, -1, -1).clone(), input_latent],
                dim=2,
            )

    # ── Prompt encoding ───────────────────────────────────────────────────────
    if prompt == "" and empty_prompt_embedding is not None:
        prompt_embedding = empty_prompt_embedding.to(device, dtype=dtype)
        if prompt_embedding.shape[0] != batch_size:
            prompt_embedding = prompt_embedding.expand(batch_size, -1, -1).clone()
    else:
        token_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        prompt_embedding = pipe.text_encoder(token_ids.to(device))[0]
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=dtype)

    # [B, C, F, H, W] -> [B, F, C, H, W]
    latents = input_latent.permute(0, 2, 1, 3, 4)

    if do_cfg:
        prompt_embedding = torch.cat([prompt_embedding, prompt_embedding], dim=0)

    # ── Optional LQ noise ─────────────────────────────────────────────────────
    if noise_step != 0:
        lq_part  = latents[:, :, :16, :, :]
        ref_part = latents[:, :, 16:, :, :]
        noise    = torch.randn_like(lq_part)
        ts       = torch.full(
            (latents.shape[0],), fill_value=noise_step, dtype=torch.long, device=device
        )
        lq_part  = pipe.scheduler.add_noise(
            lq_part.transpose(1, 2), noise.transpose(1, 2), ts
        ).transpose(1, 2)
        latents  = torch.cat([lq_part, ref_part], dim=2)

    timesteps = torch.full(
        (latents.shape[0],), fill_value=sr_noise_step, dtype=torch.long, device=device
    )

    # ── Rotary positional embeddings ──────────────────────────────────────────
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    if pipe.transformer.config.use_rotary_positional_embeddings:
        rotary_emb = prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width  * vae_scale_factor_spatial,
            num_frames=latents.shape[1],
            transformer_config=pipe.transformer.config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=device,
        )
    else:
        rotary_emb = None

    # ── OFS embedding ─────────────────────────────────────────────────────────
    ofs = None
    if pipe.transformer.config.ofs_embed_dim is not None:
        ofs = torch.full((latents.shape[0],), fill_value=2.0, device=device, dtype=dtype)

    # ── Transformer forward pass ──────────────────────────────────────────────
    predicted_noise = pipe.transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        ofs=ofs,
        return_dict=False,
    )[0]

    # ── Denoising ─────────────────────────────────────────────────────────────
    predicted_noise_slice = predicted_noise[:, :, :16, :, :].transpose(1, 2)
    lq_sample             = latents[:, :, :16, :, :].transpose(1, 2)

    if do_cfg:
        noise_pred_uncond, noise_pred_cond = predicted_noise_slice.chunk(2)
        predicted_noise_slice = (
            noise_pred_uncond
            + ref_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        )
        lq_sample  = lq_sample.chunk(2)[1]
        timesteps  = timesteps.chunk(2)[0]

    latent_generate = pipe.scheduler.get_velocity(
        predicted_noise_slice, lq_sample, timesteps
    )

    # Remove the prepended temporal-padding frames
    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, :, ncopy:, :, :]
        lq_for_diff    = lq_sample[:, :, ncopy:, :, :]
    else:
        lq_for_diff = lq_sample

    # Diagnostic: log how much the SR latent differs from the LQ latent.
    # A near-zero ratio means the model is doing nothing (bilinear pass-through).
    diff_norm = (latent_generate - lq_for_diff).norm().item()
    lq_norm   = lq_for_diff.norm().item()
    logger.info(
        f"[SR diag] lq_norm={lq_norm:.3f}  "
        f"ref_norm={full_ref_latent.norm().item():.3f}  "
        f"v_pred_norm={predicted_noise_slice.norm().item():.3f}  "
        f"sr_diff={diff_norm:.3f}  "
        f"ratio={diff_norm / (lq_norm + 1e-8):.4f}"
    )

    # ── VAE decode ────────────────────────────────────────────────────────────
    video_generate = pipe.vae.decode(
        latent_generate / pipe.vae.config.scaling_factor
    ).sample
    video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)
    return video_generate


# ── Full tiled inference loop ─────────────────────────────────────────────────

def run_sparkvsr(
    pipe,
    video: torch.Tensor,
    ref_frames_list: List[torch.Tensor],
    ref_indices: List[int],
    chunk_len: int,
    overlap_t: int,
    tile_size_hw: Tuple[int, int],
    overlap_hw: Tuple[int, int],
    ref_guidance_scale: float,
    noise_step: int,
    sr_noise_step: int,
    prompt: str,
    empty_prompt_embedding: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Run the full SparkVSR inference with temporal chunking and spatial tiling.

    Args:
        pipe:                   CogVideoXImageToVideoPipeline.
        video:                  [1, C, F, H, W] float in [-1, 1].
        ref_frames_list:        List of [C, H, W] reference tensors in [-1, 1].
        ref_indices:            Global frame indices for ref_frames_list.
        chunk_len:              Temporal chunk length (0 = whole video).
        overlap_t:              Temporal overlap between chunks.
        tile_size_hw:           (tile_h, tile_w). (0, 0) = no spatial tiling.
        overlap_hw:             (overlap_h, overlap_w).
        ref_guidance_scale:     CFG scale for reference frames.
        noise_step:             LQ noise level.
        sr_noise_step:          Denoising timestep.
        prompt:                 Text conditioning.
        empty_prompt_embedding: Pre-computed empty prompt.

    Returns:
        [1, C, F, H, W] float in [0, 1].
    """
    from sparkvsr_wrapper.preprocess import (
        make_temporal_chunks,
        make_spatial_tiles,
        get_valid_tile_region,
    )

    B, C, F, H, W = video.shape

    eff_overlap_t  = overlap_t  if chunk_len > 0              else 0
    eff_overlap_hw = overlap_hw if (tile_size_hw[0] > 0 and tile_size_hw[1] > 0) else (0, 0)

    time_chunks   = make_temporal_chunks(F, chunk_len, eff_overlap_t)
    spatial_tiles = make_spatial_tiles(H, W, tile_size_hw, eff_overlap_hw)

    output_video = torch.zeros_like(video)

    logger.info(
        f"SparkVSR: F={F} H={H} W={W} | "
        f"chunks={len(time_chunks)} tiles={len(spatial_tiles)}"
    )

    if ref_indices and len(ref_indices) != len(ref_frames_list):
        logger.warning(
            "SparkVSR: ref_indices/ref_frames length mismatch: "
            f"{len(ref_indices)} indices vs {len(ref_frames_list)} frames. "
            "Extra items will be ignored."
        )

    ref_pairs = list(zip(ref_indices, ref_frames_list))

    for t_start, t_end in time_chunks:
        active_ref_pairs = [
            (idx, rf) for idx, rf in ref_pairs if t_start <= idx < t_end
        ]
        active_ref_indices = [idx for idx, _ in active_ref_pairs]
        logger.info(
            f"SparkVSR: chunk [{t_start}, {t_end}) uses "
            f"{len(active_ref_indices)} refs: {active_ref_indices}"
        )

        for h_start, h_end, w_start, w_end in spatial_tiles:
            video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]

            # Crop reference frames to match the current spatial tile
            current_ref_frames = [
                rf[:, h_start:h_end, w_start:w_end] for _, rf in active_ref_pairs
            ]

            tile_out = process_video_ref_i2v(
                pipe=pipe,
                video=video_chunk,
                prompt=prompt,
                ref_frames=current_ref_frames,
                ref_indices=active_ref_indices,
                chunk_start_idx=t_start,
                noise_step=noise_step,
                sr_noise_step=sr_noise_step,
                empty_prompt_embedding=empty_prompt_embedding,
                ref_guidance_scale=ref_guidance_scale,
            )

            region = get_valid_tile_region(
                t_start, t_end, h_start, h_end, w_start, w_end,
                video.shape,
                eff_overlap_t, eff_overlap_hw[0], eff_overlap_hw[1],
            )

            output_video[
                :, :,
                region["out_t_start"]: region["out_t_end"],
                region["out_h_start"]: region["out_h_end"],
                region["out_w_start"]: region["out_w_end"],
            ] = tile_out[
                :, :,
                region["valid_t_start"]: region["valid_t_end"],
                region["valid_h_start"]: region["valid_h_end"],
                region["valid_w_start"]: region["valid_w_end"],
            ]

    return output_video
