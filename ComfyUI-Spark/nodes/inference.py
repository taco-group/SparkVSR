"""
SparkVSR — Inference node.

Takes a SPARKVSR_MODEL and a SPARKVSR_CONDITION and runs the full
tiled / chunked SparkVSR inference pipeline.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class SparkVSRInference:
    """Run SparkVSR super-resolution on the prepared condition."""

    CATEGORY = "SparkVSR"
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "infer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SPARKVSR_MODEL",),
                "condition": ("SPARKVSR_CONDITION",),
                "ref_guidance_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Reference-frame guidance scale (CFG). "
                            "1.0 = base reference conditioning. "
                            "Higher values increase reference-frame influence."
                        ),
                    },
                ),
                "chunk_len": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "tooltip": (
                            "Temporal chunk length. "
                            "0 = process the whole video in one pass. "
                            "49 is recommended for long videos."
                        ),
                    },
                ),
                "overlap_t": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 32,
                        "tooltip": "Temporal overlap between chunks. Ignored when chunk_len=0.",
                    },
                ),
                "tile_size_h": (
                    "INT",
                    {"default": 0, "min": 0, "tooltip": "Spatial tile height. 0 = no tiling."},
                ),
                "tile_size_w": (
                    "INT",
                    {"default": 0, "min": 0, "tooltip": "Spatial tile width.  0 = no tiling."},
                ),
                "overlap_h": ("INT", {"default": 32, "min": 0}),
                "overlap_w": ("INT", {"default": 32, "min": 0}),
                "seed": ("INT", {"default": 42}),
                "sr_noise_step": (
                    "INT",
                    {
                        "default": 399,
                        "min": 0,
                        "max": 999,
                        "tooltip": "Main denoising timestep for SparkVSR.",
                    },
                ),
                "noise_step": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": (
                            "Additional noise level added to LQ latent. "
                            "Blank or 0 = none; valid range is 0..999."
                        ),
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "output_fps": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "tooltip": "0 = inherit fps from the condition.",
                    },
                ),
            }
        }

    # ── Main entry ────────────────────────────────────────────────────────────

    def infer(
        self,
        model,
        condition,
        ref_guidance_scale: float = 1.0,
        chunk_len: int = 0,
        overlap_t: int = 8,
        tile_size_h: int = 0,
        tile_size_w: int = 0,
        overlap_h: int = 32,
        overlap_w: int = 32,
        seed: int = 42,
        sr_noise_step: int = 399,
        noise_step: int = 0,
        prompt: str = "",
        output_fps: float = 0.0,
    ):
        from transformers import set_seed
        from sparkvsr_wrapper.preprocess import remove_padding_and_extra_frames
        from sparkvsr_wrapper.infer import run_sparkvsr

        sr_noise_step = self._coerce_int(
            "sr_noise_step", sr_noise_step, default=399, min_value=0, max_value=999
        )
        noise_step = self._coerce_int(
            "noise_step", noise_step, default=0, min_value=0, max_value=999
        )

        set_seed(seed)

        pipe                   = model["pipe"]
        empty_prompt_embedding = model["empty_prompt_embedding"]

        video_up       = condition["video_up"]          # [1, C, F, H, W]
        ref_frames     = condition["ref_frames_list"]
        ref_indices    = condition["ref_indices"]
        ref_mode       = condition.get("ref_mode", "unknown")
        pad_f          = condition["pad_f"]
        remove_pad_h   = condition["remove_pad_h"]
        remove_pad_w   = condition["remove_pad_w"]
        cond_fps       = condition.get("fps", 0.0)

        logger.info(
            f"[Inference] video={tuple(video_up.shape)} "
            f"ref_mode={ref_mode} refs={len(ref_frames)} "
            f"indices={ref_indices} guidance={ref_guidance_scale} "
            f"chunk={chunk_len} overlap_t={overlap_t} "
            f"tile=({tile_size_h},{tile_size_w}) seed={seed}"
        )
        # ── Run inference ─────────────────────────────────────────────────────
        output_video = run_sparkvsr(
            pipe=pipe,
            video=video_up,
            ref_frames_list=ref_frames,
            ref_indices=ref_indices,
            chunk_len=chunk_len,
            overlap_t=overlap_t,
            tile_size_hw=(tile_size_h, tile_size_w),
            overlap_hw=(overlap_h, overlap_w),
            ref_guidance_scale=ref_guidance_scale,
            noise_step=noise_step,
            sr_noise_step=sr_noise_step,
            prompt=prompt,
            empty_prompt_embedding=empty_prompt_embedding,
        )  # [1, C, F, H, W] in [0, 1]

        # ── Remove padding ────────────────────────────────────────────────────
        output_video = remove_padding_and_extra_frames(
            output_video, pad_f, remove_pad_h, remove_pad_w
        )

        # ── Convert to ComfyUI IMAGE format [N, H, W, C] ─────────────────────
        # output_video: [1, C, F, H, W]  →  [F, H, W, C]
        output_video = output_video[0]                     # [C, F, H, W]
        output_video = output_video.permute(1, 2, 3, 0)   # [F, H, W, C]
        output_video = output_video.float().clamp(0.0, 1.0).cpu()

        fps_out = float(output_fps) if output_fps and output_fps > 0 else float(cond_fps or 24.0)
        logger.info(f"[Inference] done → {tuple(output_video.shape)} fps={fps_out}")

        return (output_video, fps_out)

    @staticmethod
    def _coerce_int(name, value, default=0, min_value=None, max_value=None):
        if value is None or value == "":
            value = default
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer; blank means {default}.") from exc

        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}.")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}.")
        return value
