"""SparkVSR video preview node."""

import logging

logger = logging.getLogger(__name__)


class SparkVSRPreviewVideo:
    """Render an IMAGE batch to a temporary MP4 so it appears in the UI."""

    CATEGORY = "SparkVSR"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "preview_video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0},
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "sparkvsr_preview"},
                ),
                "format": (
                    ["mp4_yuv420p", "mp4_yuv444p"],
                    {"default": "mp4_yuv420p"},
                ),
            },
        }

    def preview_video(
        self,
        frames,
        fps: float = 24.0,
        filename_prefix: str = "sparkvsr_preview",
        format: str = "mp4_yuv420p",
    ):
        from sparkvsr_wrapper.video_io import (
            build_video_preview,
            build_video_preview_ui,
            frames_to_uint8_numpy,
            resolve_comfy_dir,
            unique_video_path,
            write_video_file,
        )

        out_dir = resolve_comfy_dir("temp")
        out_path, _filename = unique_video_path(out_dir, filename_prefix)
        frames_np = frames_to_uint8_numpy(frames)

        logger.info(
            f"[PreviewVideo] Writing {frames_np.shape[0]} preview frames @ "
            f"{fps:.2f} fps ({format}) -> {out_path}"
        )
        write_video_file(frames, fps=fps, out_path=out_path, format=format)

        preview = build_video_preview(out_path, fps=fps, media_type="temp", root_dir=out_dir)
        return {"ui": build_video_preview_ui(preview), "result": (out_path,)}
