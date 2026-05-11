"""
ComfyUI-SparkVSR — Custom node package for SparkVSR video super-resolution.

Registers the ``sparkvsr`` model folder and exports all ComfyUI node classes.
"""

import os
import sys
import logging
import shutil
import subprocess

# Ensure this package's directory is on sys.path so that
# absolute imports like "from nodes.xxx import ..." and
# "from sparkvsr_wrapper.xxx import ..." always resolve correctly,
# regardless of how ComfyUI loads the custom node.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

logger = logging.getLogger(__name__)

# ── Auto-install missing dependencies ────────────────────────────────────────
_REQUIRED_PACKAGES = {
    "peft":   "peft>=0.9.0",
    "einops": "einops>=0.6.0",
}

for _mod, _pkg in _REQUIRED_PACKAGES.items():
    try:
        __import__(_mod)
    except ImportError:
        logger.info(f"[SparkVSR] Installing missing dependency: {_pkg}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", _pkg, "-q"],
                check=True,
            )
            logger.info(f"[SparkVSR] Installed {_pkg}")
        except Exception as _exc:
            logger.warning(f"[SparkVSR] Could not auto-install {_pkg}: {_exc}")
WEB_DIRECTORY = "./web"

# ── Register models/sparkvsr as a searchable folder in ComfyUI ───────────────
try:
    import folder_paths  # type: ignore

    _sparkvsr_model_dir = os.path.join(
        folder_paths.models_dir, "sparkvsr"
    )
    os.makedirs(_sparkvsr_model_dir, exist_ok=True)
    folder_paths.add_model_folder_path("sparkvsr", _sparkvsr_model_dir)
    logger.info(f"[SparkVSR] Registered model folder: {_sparkvsr_model_dir}")

    _sample_video_name = "hitachi_isee5_001.mp4"
    _sample_video_src = os.path.join(_pkg_dir, "sample_videos", _sample_video_name)
    _sample_video_dst = os.path.join(folder_paths.get_input_directory(), _sample_video_name)
    if os.path.exists(_sample_video_src) and not os.path.exists(_sample_video_dst):
        shutil.copy2(_sample_video_src, _sample_video_dst)
        logger.info(f"[SparkVSR] Installed sample video: {_sample_video_dst}")
except ImportError:
    logger.warning(
        "[SparkVSR] folder_paths not available — "
        "running outside of ComfyUI? Model folder registration skipped."
    )
except Exception as exc:
    logger.warning(f"[SparkVSR] Could not register model folder: {exc}")

# ── Import node classes ───────────────────────────────────────────────────────
try:
    from .nodes.load_model import SparkVSRLoadModel
    from .nodes.nano_banana_prompt import SparkVSRNanoBananaPrompt
    from .nodes.prepare_reference import SparkVSRPrepareReference
    from .nodes.inference import SparkVSRInference
    from .nodes.save_video import SparkVSRSaveVideo
    from .nodes.preview_video import SparkVSRPreviewVideo
except ImportError as exc:
    logger.error(
        f"[SparkVSR] Failed to import node classes: {exc}\n"
        "Make sure all dependencies are installed:\n"
        "  pip install -r requirements.txt\n"
        "And that diffusers >= 0.32.0 and transformers >= 4.40.0 are available."
    )
    raise

# ── ComfyUI node registry ─────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "SparkVSRLoadModel":         SparkVSRLoadModel,
    "SparkVSRNanoBananaPrompt":  SparkVSRNanoBananaPrompt,
    "SparkVSRPrepareReference":  SparkVSRPrepareReference,
    "SparkVSRInference":         SparkVSRInference,
    "SparkVSRSaveVideo":         SparkVSRSaveVideo,
    "SparkVSRPreviewVideo":      SparkVSRPreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkVSRLoadModel":        "SparkVSR Load Model",
    "SparkVSRNanoBananaPrompt": "SparkVSR Nano-Banana Pro Prompt",
    "SparkVSRPrepareReference": "SparkVSR Prepare Reference",
    "SparkVSRInference":        "SparkVSR Inference",
    "SparkVSRSaveVideo":        "SparkVSR Save Video",
    "SparkVSRPreviewVideo":     "SparkVSR Preview Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ── Server route: return latest workflow filename ─────────────────────────────
try:
    from aiohttp import web as _aiohttp_web
    from server import PromptServer as _PromptServer  # type: ignore

    _workflows_dir = os.path.join(_pkg_dir, "example_workflows")

    @_PromptServer.instance.routes.get("/sparkvsr/latest_workflow")
    async def _sparkvsr_latest_workflow(request):
        try:
            files = [f for f in os.listdir(_workflows_dir) if f.endswith(".json")]
            if not files:
                return _aiohttp_web.json_response({"error": "no workflows"}, status=404)
            files.sort(
                key=lambda f: os.path.getmtime(os.path.join(_workflows_dir, f)),
                reverse=True,
            )
            return _aiohttp_web.json_response({"filename": files[0]})
        except Exception as exc:
            return _aiohttp_web.json_response({"error": str(exc)}, status=500)

except Exception:
    pass  # Running outside ComfyUI or server not yet available

