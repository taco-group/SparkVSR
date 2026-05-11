"""
PiSA-SR source bundled for direct import within ComfyUI-SparkVSR.

Exposes PiSASR_eval so the pisa_ref pipeline can run in-process without
a separate conda environment or subprocess call.
"""

from .pisasr_eval import PiSASR_eval

__all__ = ["PiSASR_eval"]
