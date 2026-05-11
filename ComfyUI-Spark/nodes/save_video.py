"""
SparkVSR — Save Video node.

Saves a ComfyUI IMAGE batch (video frames) to an MP4 file using imageio.
"""

import logging

logger = logging.getLogger(__name__)


class SparkVSRSaveVideo:
    """Save super-resolved frames to an MP4 file."""

    CATEGORY = "SparkVSR"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),  # [N, H, W, C] float32 in [0, 1]
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0},
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "sparkvsr"},
                ),
                "format": (
                    ["mp4_yuv420p", "mp4_yuv444p"],
                    {"default": "mp4_yuv420p"},
                ),
            },
            "optional": {
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Leave empty to use the ComfyUI output directory.",
                    },
                ),
            },
        }

    # ── Main entry ────────────────────────────────────────────────────────────

    def save_video(
        self,
        frames,
        fps: float = 24.0,
        filename_prefix: str = "sparkvsr",
        format: str = "mp4_yuv420p",
        output_dir: str = "",
    ):
        from sparkvsr_wrapper.video_io import (
            build_video_preview,
            build_video_preview_ui,
            frames_to_uint8_numpy,
            resolve_comfy_dir,
            unique_video_path,
            write_video_file,
        )

        # ── Resolve output directory ──────────────────────────────────────────
        out_dir = output_dir.strip() if output_dir and output_dir.strip() else ""
        if not out_dir:
            out_dir = resolve_comfy_dir("output")

        # ── Build output file path ────────────────────────────────────────────
        out_path, _filename = unique_video_path(out_dir, filename_prefix)
        frames_np = frames_to_uint8_numpy(frames)

        logger.info(
            f"[SaveVideo] Writing {frames_np.shape[0]} frames @ {fps:.2f} fps "
            f"({format}) → {out_path}"
        )

        write_video_file(frames, fps=fps, out_path=out_path, format=format)

        logger.info(f"[SaveVideo] Saved: {out_path}")
        preview = build_video_preview(out_path, fps=fps, media_type="output", root_dir=out_dir)
        return {"ui": build_video_preview_ui(preview), "result": (out_path,)}
