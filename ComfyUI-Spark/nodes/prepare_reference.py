"""
SparkVSR — Prepare Reference node.

Accepts a batch of video frames (ComfyUI IMAGE format) and produces a
SPARKVSR_CONDITION dict that the inference node consumes.

Supported reference modes:
  no_ref                - No reference frames (model uses learned SR priors only).
  nano-banana-pro-ref   - Nano-Banana Pro (fal.ai) API-generated HD reference frames.
  pisa_ref              - PiSA-SR-generated HD reference frames (local, SD-based).
  external_ref          - User-supplied reference images.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SparkVSRPrepareReference:
    """Preprocess input video frames and prepare reference frames."""

    CATEGORY = "SparkVSR"
    RETURN_TYPES = ("SPARKVSR_CONDITION", "IMAGE")
    RETURN_NAMES = ("condition", "reference_frames")
    FUNCTION = "prepare"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    # [N, H, W, C] float32 in [0, 1] — standard ComfyUI IMAGE
                ),
                "ref_mode": (
                    [
                        "pisa_ref",
                        "no_ref",
                        "nano-banana-pro-ref",
                        "external_ref",
                    ],
                    {"default": "pisa_ref"},
                ),
                "ref_indices": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated frame indices (e.g. '0,24,48'). "
                            "Leave empty for auto-selection."
                        ),
                    },
                ),
                "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
            },
            "optional": {
                "target_width":  ("INT", {"default": 0, "min": 0, "tooltip": "0 = upscale * W"}),
                "target_height": ("INT", {"default": 0, "min": 0, "tooltip": "0 = upscale * H"}),
                "upscale_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "tooltip": "0 = auto-detect from video metadata"},
                ),
                # ── External reference ───────────────────────────────────────
                "external_ref_frames": ("IMAGE",),
                "api_prompt_input": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Prompt from SparkVSR Nano-Banana Pro Prompt. "
                            "Leave the connected prompt node empty to use the built-in default."
                        ),
                    },
                ),
                # ── PiSA-SR ──────────────────────────────────────────────────
                "pisa_sd_model_path": (
                    "STRING",
                    {
                        "default": "Manojb/stable-diffusion-2-1-base",
                        "tooltip": "HuggingFace repo ID or local path to Stable Diffusion 2.1 base model",
                    },
                ),
                "pisa_checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to pisa_sr.pkl. Leave empty to auto-detect from ComfyUI models/pisasr/ or models/loras/.",
                    },
                ),
                "pisa_gpu": ("STRING", {"default": "2"}),
                "pisa_cache_dir": ("STRING", {"default": ""}),
                # ── Nano-Banana Pro API ───────────────────────────────────────
                "api_key_env": (
                    "STRING",
                    {
                        "default": "NANO_BANANA_API_KEY",
                        "tooltip": "Environment variable name that holds the API key",
                    },
                ),
                "api_key": (
                    "STRING",
                    {"default": "", "tooltip": "API key (leave empty to use env var)"},
                ),
                "api_cache_dir": ("STRING", {"default": ""}),
                "external_ref_paths": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional comma/newline-separated reference image paths. "
                            "Relative paths are resolved from ComfyUI's input folder."
                        ),
                    },
                ),
            },
        }

    # ── Main entry ────────────────────────────────────────────────────────────

    def prepare(
        self,
        frames,
        ref_mode: str = "no_ref",
        ref_indices: str = "",
        upscale: int = 4,
        target_width: int = 0,
        target_height: int = 0,
        upscale_mode: str = "bilinear",
        fps: float = 0.0,
        external_ref_frames=None,
        pisa_python_executable: str = "",
        pisa_script_path: str = "",
        pisa_sd_model_path: str = "Manojb/stable-diffusion-2-1-base",
        pisa_checkpoint_path: str = "",
        pisa_gpu: str = "0",
        pisa_cache_dir: str = "",
        api_key_env: str = "NANO_BANANA_API_KEY",
        api_key: str = "",
        api_prompt_input: Optional[str] = None,
        api_cache_dir: str = "",
        external_ref_paths: str = "",
    ):
        import torch
        from sparkvsr_wrapper.preprocess import preprocess_frames
        from sparkvsr_wrapper.reference import (
            parse_ref_indices,
            auto_select_ref_indices,
            prepare_external_references,
            prepare_pisasr_references,
        )

        ref_mode = self._normalize_ref_mode(ref_mode)
        t_h = int(target_height) if target_height and target_height > 0 else None
        t_w = int(target_width)  if target_width  and target_width  > 0 else None

        video_up, video_lr, pad_f, remove_pad_h, remove_pad_w, original_hw = \
            preprocess_frames(
                frames,
                upscale=upscale,
                target_h=t_h,
                target_w=t_w,
                upscale_mode=upscale_mode,
            )

        # Actual padded/upscaled resolution
        _B, _C, _F, vid_h, vid_w = video_up.shape
        F_pad = _F

        logger.info(
            f"[PrepareRef] original={original_hw} → "
            f"upscaled={vid_h}x{vid_w} pad_f={pad_f} "
            f"remove_pad=({remove_pad_h},{remove_pad_w})"
        )

        # ── Reference index selection ─────────────────────────────────────────
        if ref_mode == "no_ref":
            final_ref_indices = []
        else:
            parsed = parse_ref_indices(ref_indices)
            if parsed is None:
                final_ref_indices = auto_select_ref_indices(F_pad)
            else:
                # Clamp to valid range
                clamped = [i for i in parsed if 0 <= i < F_pad]
                if not clamped:
                    logger.warning(
                        "All parsed ref_indices are out of range; falling back to auto."
                    )
                    final_ref_indices = auto_select_ref_indices(F_pad)
                else:
                    # Keep user-supplied indices as-is (matching original script behaviour).
                    # The inference code maps pixel-frame i → latent slot i//4 internally.
                    final_ref_indices = sorted(set(clamped))

        logger.info(f"[PrepareRef] ref_mode={ref_mode} indices={final_ref_indices}")

        # ── Build reference frames ────────────────────────────────────────────
        ref_frames_list = []

        if ref_mode == "no_ref":
            pass  # empty list

        elif ref_mode == "external_ref":
            if external_ref_frames is None and external_ref_paths.strip():
                external_ref_frames = self._load_external_ref_paths(external_ref_paths)
            if external_ref_frames is None:
                raise ValueError(
                    "ref_mode='external_ref' requires either the 'external_ref_frames' "
                    "input or non-empty 'external_ref_paths'."
                )
            ref_frames_list = prepare_external_references(
                external_ref_frames,
                final_ref_indices,
                target_h=vid_h,
                target_w=vid_w,
            )

        elif ref_mode == "pisa_ref":
            pisa_ckpt = pisa_checkpoint_path.strip()
            pisa_sd   = pisa_sd_model_path.strip()
            pisa_scr  = pisa_script_path.strip()

            # Auto-resolve PiSA checkpoint
            if not pisa_ckpt:
                pisa_ckpt = self._find_pisa_checkpoint()
            # Auto-resolve SD model: trigger if empty OR if the value is a bare
            # HuggingFace model ID (no leading '/' and not an existing directory),
            # so a locally-downloaded copy is preferred over a network download.
            if not pisa_sd or (not os.path.isabs(pisa_sd) and not os.path.isdir(pisa_sd)):
                found = self._find_sd_model()
                if found:
                    pisa_sd = found
            if not pisa_scr:
                pisa_scr = self._find_pisa_script()

            pisa_config = {
                "python_executable": pisa_python_executable.strip() or "python",
                "script_path":       pisa_scr,
                "sd_model_path":     pisa_sd,
                "checkpoint_path":   pisa_ckpt,
                "gpu_id":            pisa_gpu.strip(),
                "cache_dir":         pisa_cache_dir.strip(),
            }
            ref_frames_list = prepare_pisasr_references(
                video_lr=video_lr,
                ref_indices=final_ref_indices,
                upscale=upscale,
                pisa_config=pisa_config,
                target_h=vid_h,
                target_w=vid_w,
            )

        elif ref_mode == "nano-banana-pro-ref":
            from sparkvsr_wrapper.api_reference import generate_api_references

            # Resolve API key (never log the key)
            api_key_str = api_key.strip() or ""
            api_prompt_text = self._resolve_api_prompt(api_prompt_input)

            # Auto-detect resolution tier from video size
            max_dim = max(vid_h, vid_w)
            if max_dim <= 1536:
                api_res = "1K"
            elif max_dim <= 3000:
                api_res = "2K"
            else:
                api_res = "4K"

            api_results = generate_api_references(
                frames=frames,         # [F, H, W, C] ComfyUI
                ref_indices=final_ref_indices,
                api_key=api_key_str or None,
                api_key_env=api_key_env.strip() or "NANO_BANANA_API_KEY",
                prompt=api_prompt_text or None,
                output_dir=api_cache_dir.strip() or None,
                cache=True,
                target_size=(vid_h, vid_w),
                resolution=api_res,
            )
            # api_results is [(idx, tensor), ...]
            # Rebuild BOTH ref_frames_list and final_ref_indices from successful
            # results only, so they always stay positionally in sync.
            idx_to_tensor = {i: t for i, t in api_results}
            valid_pairs = [
                (i, idx_to_tensor[i])
                for i in final_ref_indices
                if i in idx_to_tensor
            ]
            if len(valid_pairs) < len(final_ref_indices):
                missing = [i for i in final_ref_indices if i not in idx_to_tensor]
                logger.warning(
                    f"[PrepareRef] API returned no result for indices {missing}; "
                    "those references will be skipped."
                )
            final_ref_indices = [i for i, _ in valid_pairs]
            ref_frames_list   = [t for _, t in valid_pairs]

        else:
            raise ValueError(f"Unknown ref_mode '{ref_mode}'")

        if ref_mode != "no_ref" and not ref_frames_list:
            raise ValueError(
                f"ref_mode='{ref_mode}' selected but no reference frames were prepared. "
                "Check ref_indices, API/PiSA/external-ref settings, and the ComfyUI log above."
            )

        if ref_frames_list:
            shapes = [tuple(ref.shape) for ref in ref_frames_list]
            logger.info(
                f"[PrepareRef] prepared {len(ref_frames_list)} reference frames: "
                f"indices={final_ref_indices} shapes={shapes}"
            )

        # ── Pack condition dict ───────────────────────────────────────────────
        condition = {
            "video_up":      video_up,       # [1, C, F, H, W] in [-1, 1]
            "video_lr":      video_lr,        # [F, C, H, W] in [0, 255]
            "ref_frames_list": ref_frames_list,
            "ref_indices":   final_ref_indices,
            "ref_mode":      ref_mode,
            "pad_f":         pad_f,
            "remove_pad_h":  remove_pad_h,
            "remove_pad_w":  remove_pad_w,
            "fps":           float(fps) if fps and fps > 0 else 0.0,
            "original_h":    original_hw[0],
            "original_w":    original_hw[1],
        }
        ref_preview = self._reference_frames_to_image_batch(ref_frames_list)
        return (condition, ref_preview)

    # ── UI / path helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_ref_mode(ref_mode: str) -> str:
        # Accept legacy names from old saved workflows so they still run.
        aliases = {
            "api_ref": "nano-banana-pro-ref",
            "nano_banana_api_ref": "nano-banana-pro-ref",
            "nano_banana_pro_ref": "nano-banana-pro-ref",
            "pisasr_ref": "pisa_ref",
        }
        return aliases.get(ref_mode, ref_mode)

    @staticmethod
    def _resolve_api_prompt(input_prompt: Optional[str]) -> str:
        return (input_prompt or "").strip()

    @staticmethod
    def _reference_frames_to_image_batch(ref_frames_list):
        """Convert SparkVSR reference tensors to ComfyUI IMAGE preview batch."""
        import torch

        if not ref_frames_list:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        frames = []
        for frame in ref_frames_list:
            # [C,H,W] in [-1,1] -> [H,W,C] in [0,1]
            img = frame.detach().float().mul(0.5).add(0.5).clamp(0.0, 1.0)
            frames.append(img.permute(1, 2, 0).cpu())
        return torch.stack(frames, dim=0)

    @staticmethod
    def _load_external_ref_paths(paths_str: str):
        """Load comma/newline-separated image paths as a ComfyUI IMAGE batch."""
        import re

        import numpy as np
        import torch
        from PIL import Image

        paths = [p.strip() for p in re.split(r"[,\n]+", paths_str) if p.strip()]
        if not paths:
            raise ValueError("external_ref_paths is empty.")

        tensors = []
        for path in paths:
            resolved = SparkVSRPrepareReference._resolve_external_ref_path(path)
            img = Image.open(resolved).convert("RGB")
            arr = np.asarray(img).astype("float32") / 255.0
            tensors.append(torch.from_numpy(arr))
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _resolve_external_ref_path(path: str) -> str:
        if os.path.isabs(path) and os.path.exists(path):
            return path

        try:
            import folder_paths

            annotated = folder_paths.get_annotated_filepath(path)
            if annotated and os.path.exists(annotated):
                return annotated

            candidate = os.path.join(folder_paths.get_input_directory(), path)
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass

        if os.path.exists(path):
            return os.path.abspath(path)
        raise FileNotFoundError(f"External reference image not found: {path}")

    # ── Auto-resolve helpers for PiSA-SR paths ────────────────────────────────

    @staticmethod
    def _find_pisa_checkpoint() -> str:
        """
        Try to locate pisa_sr.pkl, searching in this order:
        1. ComfyUI models/loras/ (recommended location for deployment)
        2. ComfyUI models/sparkvsr/ and models/pisasr/
        3. Alongside this custom node (ComfyUI-SparkVSR/models/)
        4. Any PiSA-SR workspace next to ComfyUI custom_nodes
        5. Broad recursive search within ComfyUI models dir
        """
        _node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ── 1. ComfyUI folder_paths: loras, sparkvsr, pisasr ─────────────────
        try:
            import folder_paths

            for folder_key in ("loras", "sparkvsr", "pisasr"):
                try:
                    names = folder_paths.get_filename_list(folder_key)
                    for c in names:
                        if c.lower().endswith(".pkl") and "pisa" in c.lower():
                            full = folder_paths.get_full_path(folder_key, c)
                            if full and os.path.exists(full):
                                logger.info(f"[PiSA-SR] Auto-resolved checkpoint: {full}")
                                return full
                except Exception:
                    pass

            # ── 2. models_dir sub-folders ─────────────────────────────────
            models_dir = getattr(folder_paths, "models_dir", None)
            if models_dir:
                for sub in ("loras", "pisasr", "sparkvsr", ""):
                    candidate = os.path.join(models_dir, sub, "pisa_sr.pkl")
                    if os.path.exists(candidate):
                        logger.info(f"[PiSA-SR] Auto-resolved checkpoint: {candidate}")
                        return candidate

        except Exception:
            pass

        # ── 3. Custom node bundled models/ folder ─────────────────────────────
        bundled = os.path.join(_node_dir, "models", "pisa_sr.pkl")
        if os.path.exists(bundled):
            logger.info(f"[PiSA-SR] Auto-resolved checkpoint: {bundled}")
            return bundled

        # ── 4. Any PiSA-SR workspace (sibling of custom_nodes or parent dirs) ─
        search_roots = [
            os.path.dirname(_node_dir),                        # custom_nodes/
            os.path.dirname(os.path.dirname(_node_dir)),       # ComfyUI/
            os.path.dirname(os.path.dirname(os.path.dirname(_node_dir))),  # workspace/
        ]
        for root in search_roots:
            for dirpath, _dirs, files in os.walk(root):
                if "pisa_sr.pkl" in files:
                    full = os.path.join(dirpath, "pisa_sr.pkl")
                    logger.info(f"[PiSA-SR] Auto-resolved checkpoint: {full}")
                    return full
                # Don't recurse into unrelated deep trees
                _dirs[:] = [d for d in _dirs if not d.startswith(".") and d not in
                            ("__pycache__", "node_modules", ".git", "venv", "site-packages")]

        logger.warning(
            "[PiSA-SR] Could not auto-detect pisa_sr.pkl.\n"
            "  ➜ Place it at: ComfyUI/models/loras/pisa_sr.pkl\n"
            "  ➜ Or set 'pisa_checkpoint_path' explicitly in the node."
        )
        return ""

    @staticmethod
    def _find_sd_model() -> str:
        """
        Locate a Stable Diffusion 2.1 base model directory.  Search order:
        1. ComfyUI folder_paths (diffusion_models, checkpoints)
        2. PiSA-SR preset/models/ path (common local install)
        3. Sibling directories and parent directories up to workspace level
        """
        _node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _target_names = ("stable-diffusion-2-1-base", "stable-diffusion-2-1")

        # ── 1. ComfyUI folder_paths ───────────────────────────────────────────
        try:
            import folder_paths
            for folder_key in ("diffusion_models", "checkpoints"):
                try:
                    dirs = folder_paths.get_folder_paths(folder_key)
                    for d in dirs:
                        for name in _target_names:
                            candidate = os.path.join(d, name)
                            if os.path.isdir(candidate):
                                logger.info(f"[PiSA-SR] Auto-resolved SD model: {candidate}")
                                return candidate
                except Exception:
                    pass
        except Exception:
            pass

        # ── 2 & 3. Walk up from the custom-node directory ────────────────────
        search_roots = [
            os.path.dirname(_node_dir),                                         # custom_nodes/
            os.path.dirname(os.path.dirname(_node_dir)),                        # ComfyUI/
            os.path.dirname(os.path.dirname(os.path.dirname(_node_dir))),       # workspace/
        ]
        _skip = {"__pycache__", "node_modules", ".git", "venv", "site-packages",
                 "lib", "bin", "include"}
        for root in search_roots:
            for dirpath, dirs, _ in os.walk(root):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _skip]
                for name in _target_names:
                    if os.path.basename(dirpath) == name and os.path.isfile(
                            os.path.join(dirpath, "model_index.json")):
                        logger.info(f"[PiSA-SR] Auto-resolved SD model: {dirpath}")
                        return dirpath

        logger.warning(
            "[PiSA-SR] Could not auto-detect stable-diffusion-2-1-base.\n"
            "  ➜ Download it and place it at:\n"
            "      ComfyUI/models/diffusion_models/stable-diffusion-2-1-base/\n"
            "  ➜ Or set 'pisa_sd_model_path' explicitly in the node."
        )
        return ""

    @staticmethod
    def _find_pisa_script() -> str:
        """Try to find test_pisasr.py in the PiSA-SR workspace."""
        try:
            import folder_paths
            # ComfyUI custom_nodes often sit next to each other
            custom_nodes_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            for root, dirs, files in os.walk(custom_nodes_dir):
                if "test_pisasr.py" in files:
                    p = os.path.join(root, "test_pisasr.py")
                    logger.info(f"[PiSA-SR] Auto-resolved script: {p}")
                    return p
        except Exception:
            pass
        return ""
