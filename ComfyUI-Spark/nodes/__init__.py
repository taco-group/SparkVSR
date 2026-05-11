"""ComfyUI node definitions for SparkVSR."""

from .load_model import SparkVSRLoadModel
from .nano_banana_prompt import SparkVSRNanoBananaPrompt
from .prepare_reference import SparkVSRPrepareReference
from .inference import SparkVSRInference
from .save_video import SparkVSRSaveVideo
from .preview_video import SparkVSRPreviewVideo

__all__ = [
    "SparkVSRLoadModel",
    "SparkVSRNanoBananaPrompt",
    "SparkVSRPrepareReference",
    "SparkVSRInference",
    "SparkVSRSaveVideo",
    "SparkVSRPreviewVideo",
]
