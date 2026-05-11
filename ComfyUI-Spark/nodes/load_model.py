"""
SparkVSR — Load Model node.

Loads and caches a CogVideoXImageToVideoPipeline with the SparkVSR weights,
the DPM scheduler, and an optional LoRA adapter.
"""

import os
import logging

logger = logging.getLogger(__name__)


class SparkVSRLoadModel:
    """Load the SparkVSR model pipeline."""

    CATEGORY = "SparkVSR"
    RETURN_TYPES = ("SPARKVSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": "JiongzeYu/SparkVSR",
                        "multiline": False,
                        "tooltip": (
                            "HuggingFace repo-ID (e.g. 'JiongzeYu/SparkVSR') "
                            "or an absolute path to the model directory."
                        ),
                    },
                ),
                "dtype": (
                    ["bfloat16", "float16", "float32"],
                    {"default": "bfloat16"},
                ),
            },
            "optional": {
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional path to LoRA weights (.safetensors or .pkl).",
                    },
                ),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "vae_slicing": ("BOOLEAN", {"default": False}),
                "vae_tiling":  ("BOOLEAN", {"default": False}),
                "device": (
                    "STRING",
                    {
                        "default": "cuda:2",
                        "tooltip": "Target device for the VSR pipeline, e.g. 'cuda:0', 'cuda:2'.",
                    },
                ),
            },
        }

    def load_model(
        self,
        model_path: str,
        dtype: str,
        lora_path: str = "",
        cpu_offload: bool = False,
        vae_slicing: bool = False,
        vae_tiling: bool = False,
        device: str = "cuda:2",
    ):
        resolved_path = self._resolve_model_path(model_path)

        from sparkvsr_wrapper.model_loader import load_sparkvsr_pipeline

        pipe, empty_prompt_embedding = load_sparkvsr_pipeline(
            model_path=resolved_path,
            lora_path=lora_path.strip() or None,
            dtype_str=dtype,
            cpu_offload=cpu_offload,
            vae_slicing=vae_slicing,
            vae_tiling=vae_tiling,
            device=device.strip() or "cuda",
        )

        model = {
            "pipe": pipe,
            "empty_prompt_embedding": empty_prompt_embedding,
        }
        return (model,)

    # ── Path resolution helpers ───────────────────────────────────────────────

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """
        Resolve a model identifier to an absolute path or leave it as a
        HuggingFace repo-ID (diffusers handles those transparently).

        Search order:
          1. Absolute path / relative path that exists on disk.
          2. ``models/sparkvsr/<name>`` via folder_paths.
          3. ``models/diffusion_models/<name>`` via folder_paths.
          4. Pass through as a HuggingFace repo-ID.
        """
        model_path = model_path.strip()

        # Already an absolute path
        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path

        # Relative path from CWD
        if os.path.exists(model_path):
            return os.path.abspath(model_path)

        # Try ComfyUI folder_paths
        try:
            import folder_paths  # type: ignore

            for folder_key in ("sparkvsr", "diffusion_models"):
                try:
                    candidates = folder_paths.get_filename_list(folder_key)
                    for c in candidates:
                        if os.path.basename(c) == model_path or c == model_path:
                            full = folder_paths.get_full_path(folder_key, c)
                            if full and os.path.exists(full):
                                logger.info(
                                    f"Resolved model '{model_path}' → '{full}' "
                                    f"(via folder_paths key '{folder_key}')"
                                )
                                return full
                except Exception:
                    pass

            # Also try as a direct sub-folder
            for folder_key in ("sparkvsr", "diffusion_models"):
                try:
                    dirs = folder_paths.get_folder_paths(folder_key)
                    for d in dirs:
                        candidate = os.path.join(d, model_path)
                        if os.path.isdir(candidate):
                            logger.info(
                                f"Resolved model '{model_path}' → '{candidate}'"
                            )
                            return candidate
                except Exception:
                    pass

        except ImportError:
            pass

        # Fall through: treat as HuggingFace repo-ID / let diffusers decide
        logger.info(
            f"Model path '{model_path}' not found on disk; "
            "treating as a HuggingFace repo-ID."
        )
        return model_path
