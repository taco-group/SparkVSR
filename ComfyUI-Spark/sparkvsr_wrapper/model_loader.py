"""
SparkVSR model loading utilities.

Loads and caches a CogVideoXImageToVideoPipeline with the DPM scheduler
and optional LoRA weights.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Global model cache: (model_path, lora_path, dtype_str) -> (pipe, empty_prompt_embedding)
_MODEL_CACHE: dict = {}

# Paths where the empty-prompt embedding may live (relative to model_path or CWD)
_EMPTY_PROMPT_FILENAME = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
)
_EMPTY_PROMPT_RELPATHS = [
    "pretrained_models/prompt_embeddings/{fn}",
    "../pretrained_models/prompt_embeddings/{fn}",
    "../../pretrained_models/prompt_embeddings/{fn}",
]


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def _find_empty_prompt_embedding(model_path: str) -> Optional[torch.Tensor]:
    """
    Try several candidate paths to locate the pre-computed empty-prompt embedding.
    Returns the embedding tensor or None.
    """
    from safetensors.torch import load_file

    search_roots = [model_path, os.getcwd()]
    for root in search_roots:
        for rel_tmpl in _EMPTY_PROMPT_RELPATHS:
            candidate = Path(root) / rel_tmpl.format(fn=_EMPTY_PROMPT_FILENAME)
            if candidate.exists():
                try:
                    data = load_file(str(candidate))
                    emb = data.get("prompt_embedding")
                    if emb is not None:
                        logger.info(f"Loaded empty-prompt embedding from {candidate}")
                        return emb
                except Exception as exc:
                    logger.warning(f"Failed to load embedding at {candidate}: {exc}")
    logger.info(
        "Empty-prompt embedding not found; an empty string will be tokenised at runtime."
    )
    return None


def load_sparkvsr_pipeline(
    model_path: str,
    lora_path: Optional[str] = None,
    dtype_str: str = "bfloat16",
    cpu_offload: bool = False,
    vae_slicing: bool = False,
    vae_tiling: bool = False,
    device: str = "cuda",
) -> Tuple:
    """
    Load (or retrieve from cache) the SparkVSR pipeline.

    Args:
        model_path:   HuggingFace repo-ID or absolute path to the model directory.
        lora_path:    Optional path to a LoRA `.safetensors` or `.pkl` file.
        dtype_str:    One of ``"bfloat16"``, ``"float16"``, ``"float32"``.
        cpu_offload:  Enable sequential CPU offload to reduce GPU VRAM usage.
        vae_slicing:  Enable VAE slicing.
        vae_tiling:   Enable VAE tiling.
        device:       Target device, e.g. ``"cuda"``, ``"cuda:0"``, ``"cuda:2"``.

    Returns:
        (pipe, empty_prompt_embedding)
        where ``empty_prompt_embedding`` may be ``None``.
    """
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler

    lora_path_key = lora_path or ""
    cache_key = (model_path, lora_path_key, dtype_str, device)
    if cache_key in _MODEL_CACHE:
        logger.info(f"Returning cached SparkVSR pipeline for key {cache_key}")
        return _MODEL_CACHE[cache_key]

    dtype = _resolve_dtype(dtype_str)

    logger.info(f"Loading SparkVSR pipeline from '{model_path}' (dtype={dtype_str}, device={device}) …")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Replace scheduler with CogVideoXDPM (trailing timestep spacing)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    # Optional LoRA
    if lora_path:
        logger.info(f"Loading LoRA weights from '{lora_path}' …")
        pipe.load_lora_weights(lora_path, adapter_name="sparkvsr_lora")
        pipe.fuse_lora(lora_scale=1.0)

    # Device / memory configuration
    if cpu_offload:
        logger.info("Enabling sequential CPU offload …")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    if vae_slicing:
        pipe.vae.enable_slicing()
    if vae_tiling:
        pipe.vae.enable_tiling()

    # Empty-prompt embedding
    empty_prompt_embedding = _find_empty_prompt_embedding(model_path)

    result = (pipe, empty_prompt_embedding)
    _MODEL_CACHE[cache_key] = result
    transformer_config = pipe.transformer.config
    try:
        first_param = next(pipe.transformer.parameters())
        transformer_device = str(first_param.device)
        transformer_dtype = str(first_param.dtype)
    except StopIteration:
        transformer_device = "unknown"
        transformer_dtype = "unknown"
    logger.info(
        "SparkVSR pipeline ready: "
        f"model_path={model_path} "
        f"transformer_in={transformer_config.in_channels} "
        f"transformer_out={transformer_config.out_channels} "
        f"sample_frames={transformer_config.sample_frames} "
        f"dtype={transformer_dtype} device={transformer_device} "
        f"empty_prompt_embedding={'yes' if empty_prompt_embedding is not None else 'no'}"
    )
    return result


def clear_model_cache() -> None:
    """Remove all cached pipeline instances."""
    _MODEL_CACHE.clear()
    logger.info("SparkVSR model cache cleared.")
