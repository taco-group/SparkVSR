# ComfyUI-SparkVSR

ComfyUI custom nodes for **SparkVSR**, a diffusion-based video super-resolution workflow built around CogVideoX-style video generation.

SparkVSR upscales low-resolution video with optional high-resolution reference frames. The default workflow is configured for 4× upscaling and includes `pisa_ref` (default), `nano-banana-pro-ref`, `external_ref`, and `no_ref` modes.

![ComfyUI workflow overview](fig/comfyui_overview.png)

## Features

- One all-in-one ComfyUI workflow: `example_workflows/sparkvsr_all_modes_preview.json`
- **`pisa_ref` is the default mode** — PiSA-SR inference is bundled inside the plugin; no separate environment needed
- Default `upscale=4` with auto-selection of reference frame indices
- Reference modes: `pisa_ref`, `nano-banana-pro-ref`, `external_ref`, `no_ref`
- Standalone `SparkVSR Nano-Banana Pro Prompt` node for editing the Nano-Banana Pro text prompt
- Bundled sample video: `hitachi_isee5_001.mp4`
- Automatic startup workflow loading when the ComfyUI canvas is blank
- Temporal chunking and spatial tiling controls for longer or larger videos
- Enlarged output video preview in `SparkVSR Save Video`

## Prerequisites

You need a working ComfyUI installation before installing this custom node. If you do not have ComfyUI yet, follow the official manual installation guide:

https://docs.comfy.org/installation/manual_install

A minimal Linux/macOS setup looks like this:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch for your GPU/CPU platform first, then:
python -m pip install -r requirements.txt
python main.py
```

After ComfyUI starts successfully, stop it and continue with the plugin installation. For Windows, ComfyUI Desktop or the official portable build is often easier. Whichever ComfyUI installation method you use, install this plugin's Python packages into the same Python environment that runs ComfyUI.

The bundled workflow is configured for a multi-GPU workstation. If your machine has one GPU, set `SparkVSR Load Model -> device` to `cuda:0` and `SparkVSR Prepare Reference -> pisa_gpu` to `0` before queuing the workflow.

## Installation

### 1. Install the plugin

ComfyUI-SparkVSR is published as the `ComfyUI-Spark/` subfolder in the SparkVSR repository:

https://github.com/taco-group/SparkVSR/tree/main/ComfyUI-Spark

Git cannot clone a single GitHub subfolder directly, so clone the SparkVSR repository with sparse checkout and link the plugin folder into ComfyUI's `custom_nodes/` directory:

Linux/macOS:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone --depth 1 --filter=blob:none --sparse https://github.com/taco-group/SparkVSR.git SparkVSR
cd SparkVSR
git sparse-checkout set ComfyUI-Spark
cd ..
ln -s SparkVSR/ComfyUI-Spark ComfyUI-Spark
cd ComfyUI-Spark
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
cd \path\to\ComfyUI\custom_nodes
git clone --depth 1 --filter=blob:none --sparse https://github.com/taco-group/SparkVSR.git SparkVSR
cd SparkVSR
git sparse-checkout set ComfyUI-Spark
cd ..
mklink /D ComfyUI-Spark SparkVSR\ComfyUI-Spark
cd ComfyUI-Spark
python -m pip install -r requirements.txt
```

If `mklink` fails on Windows, run Command Prompt as Administrator or enable Developer Mode.

For updates, run:

```bash
cd /path/to/ComfyUI/custom_nodes/SparkVSR
git pull
```

### 2. Install ComfyUI-VideoHelperSuite

Required for the `VHS_LoadVideo` node used by the default workflow:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
python -m pip install -r ComfyUI-VideoHelperSuite/requirements.txt
```

### 3. Set up PiSA-SR weights (required for default `pisa_ref` mode)

Download `pisa_sr.pkl` (~32 MB) once from the official PiSA-SR Google Drive folder:

https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss

Then place it in ComfyUI's `models/loras/` folder:

```bash
mkdir -p /path/to/ComfyUI/models/loras
cp /path/to/downloads/pisa_sr.pkl /path/to/ComfyUI/models/loras/pisa_sr.pkl
```

Note: `huggingface-cli download jiangyzy/PiSA-SR pisa_sr.pkl` is not a reliable public download path for this file; it currently requires authentication.

The plugin auto-detects the file from `models/loras/pisa_sr.pkl` — no manual path configuration needed.

### 4. SD 2.1 base model (auto-downloaded or local)

PiSA-SR requires SD 2.1 base weights (~5 GB). The plugin defaults to `Manojb/stable-diffusion-2-1-base` (a public mirror). If you have internet access, this downloads automatically on first run. To use a local copy, set `pisa_sd_model_path` in the node to its folder path.

### 5. Python packages (auto-installed)

`peft` and `einops` are listed in `requirements.txt` and are also auto-installed by the plugin on first load if missing. You can also install them manually:

```bash
python -m pip install "peft>=0.9.0" "einops>=0.6.0"
```

### 6. Optional: Nano-Banana Pro API mode

Only needed for `nano-banana-pro-ref`:

```bash
python -m pip install fal-client requests
```

Restart ComfyUI after installation.

## Architecture

ComfyUI-SparkVSR bundles all inference code directly — including a compatibility-patched copy of PiSA-SR's inference logic. Everything runs inside the ComfyUI Python environment. No separate conda environment is required.

| Component | Where it runs | Setup required |
|---|---|---|
| SparkVSR pipeline | ComfyUI env | Auto-downloaded from HuggingFace on first run |
| PiSA-SR inference code | Bundled in this plugin | Nothing — already included |
| PiSA-SR LoRA weights (`pisa_sr.pkl`) | ComfyUI env | **Download once from Google Drive to `models/loras/`** (Step 3 above) |
| SD 2.1 base model | ComfyUI env | Auto-downloaded on first `pisa_ref` run (Step 4 above) |
| `peft`, `einops` packages | ComfyUI env | Auto-installed on plugin load |
| Nano-Banana Pro API | External (fal.ai) | Optional — only for `nano-banana-pro-ref` |

## Startup Workflow

On startup, ComfyUI-SparkVSR auto-loads:

```text
example_workflows/sparkvsr_all_modes_preview.json
```

The workflow is loaded only when the canvas is blank or still on ComfyUI's stock default graph. Existing user workflows are left untouched.

You can disable this behavior in ComfyUI settings:

```text
SparkVSR -> Workflow -> Auto-load SparkVSR workflow on startup
```

The plugin also copies the bundled sample video into ComfyUI's `input` directory if it is missing.

## SparkVSR Model Setup

The default `model_path` is:

```text
JiongzeYu/SparkVSR
```

Diffusers will download the model automatically on first use. To use a manually downloaded copy:

```bash
huggingface-cli download JiongzeYu/SparkVSR --local-dir /path/to/ComfyUI/models/sparkvsr/SparkVSR
```

Then set `model_path` to the folder path in the `SparkVSR Load Model` node.

Optional empty-prompt embedding (improves unconditional inference):

```text
models/sparkvsr/SparkVSR/pretrained_models/prompt_embeddings/
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors
```

## Default Workflow

The only maintained example workflow is:

```text
example_workflows/sparkvsr_all_modes_preview.json
```

It includes:

- `VHS_LoadVideo` with sample video `hitachi_isee5_001.mp4`
- Manual `force_rate=16`, matching the sample video's FPS
- `SparkVSR Prepare Reference` with `ref_mode=pisa_ref` (default) and `upscale=4`
- `SparkVSR Nano-Banana Pro Prompt`, connected to `api_prompt_input`
- `PreviewImage` for prepared reference frames
- `SparkVSR Save Video` with an enlarged output preview

The input video can be previewed directly in `VHS_LoadVideo`; the workflow does not add a second input preview node.

Basic graph:

```text
VHS_LoadVideo -> SparkVSR Prepare Reference -> SparkVSR Inference -> SparkVSR Save Video
                              |
                              -> PreviewImage

SparkVSR Nano-Banana Pro Prompt -> SparkVSR Prepare Reference.api_prompt_input
```

To use the workflow:

1. Open ComfyUI.
2. Let the startup workflow auto-load, or manually load `sparkvsr_all_modes_preview.json`.
3. Confirm `VHS_LoadVideo` points to `hitachi_isee5_001.mp4` or choose your own low-resolution video.
4. Choose `ref_mode` in `SparkVSR Prepare Reference` (default is `pisa_ref`).
5. Set `device` in `SparkVSR Load Model` and `pisa_gpu` in `SparkVSR Prepare Reference` to an available GPU on your machine.
6. Confirm `upscale=4`, or set `target_width` and `target_height` for an explicit output size.
7. Queue the prompt.

## Reference Modes

### `pisa_ref` (default)

Runs PiSA-SR to generate high-resolution reference frames for selected input frames. This is the **default mode** in the workflow.

**How it works:** The PiSA-SR inference code is bundled inside this plugin (`sparkvsr_wrapper/pisasr_src/`) and runs directly in the ComfyUI Python process. You do **not** need a separate conda environment, subprocess, or PiSA-SR repository.

See [Installation](#installation) Step 3–5 for one-time setup.

#### Node parameters

| Parameter | Default | Description |
|---|---|---|
| `pisa_sd_model_path` | `Manojb/stable-diffusion-2-1-base` | HuggingFace repo ID or local path to SD 2.1 base model |
| `pisa_checkpoint_path` | *(auto-detect)* | Path to `pisa_sr.pkl`. Auto-detected from `models/loras/` if left empty |
| `pisa_gpu` | `2` | GPU index to run PiSA-SR on |
| `pisa_cache_dir` | *(empty)* | Optional folder to cache generated reference images across runs |

### `nano-banana-pro-ref`

Uses Nano-Banana Pro through fal.ai to generate high-resolution reference frames for selected input frames.

Required settings:

| Parameter | Description |
|---|---|
| `ref_mode` | Set to `nano-banana-pro-ref` |
| `ref_indices` | Comma-separated frame indices, or empty for auto-selection |
| `api_key_env` | Environment variable that stores the API key, default `NANO_BANANA_API_KEY` |
| `api_key` | Direct API key value. Leave empty to use `api_key_env` |
| `api_prompt_input` | Connected prompt from `SparkVSR Nano-Banana Pro Prompt` |
| `api_cache_dir` | Optional cache directory for generated references |

Set the API key with:

```bash
export NANO_BANANA_API_KEY="your-api-key-here"
```

The code also checks `FAL_KEY` as a fallback.

### `external_ref`

Uses user-provided high-resolution reference images.

| Parameter | Description |
|---|---|
| `external_ref_frames` | ComfyUI `IMAGE` input. Provide one image per reference index |
| `external_ref_paths` | Optional comma- or newline-separated image paths |
| `ref_indices` | Frame indices that correspond to the provided reference images |

### `no_ref`

Runs SparkVSR without any reference frames. The simplest mode — no additional models or API keys needed.

## Nodes

### `SparkVSR Load Model`

Loads the SparkVSR pipeline and caches it for the session.

| Parameter | Description |
|---|---|
| `model_path` | HuggingFace repo ID, model folder name, or absolute path |
| `dtype` | `bfloat16`, `float16`, or `float32` |
| `lora_path` | Optional LoRA path |
| `cpu_offload` | Enable sequential CPU offload |
| `vae_slicing` | Enable VAE slicing |
| `vae_tiling` | Enable VAE tiling |
| `device` | CUDA device string, e.g. `cuda:0`, `cuda:2` (default `cuda:2`) |

### `SparkVSR Prepare Reference`

Preprocesses input video frames, applies 4x upscaling by default, and prepares optional reference frames.

| Parameter | Description |
|---|---|
| `frames` | Input video frames as ComfyUI `IMAGE` |
| `ref_mode` | `no_ref`, `nano-banana-pro-ref`, `pisa_ref`, or `external_ref` |
| `ref_indices` | Reference frame indices. Empty means auto-selection |
| `upscale` | Integer upscale factor. Default `4` |
| `target_width` / `target_height` | Explicit output size. `0` means use `upscale` |
| `upscale_mode` | `bilinear`, `bicubic`, or `nearest` |
| `fps` | Output FPS override. `0` means inherit |

Outputs:

| Output | Description |
|---|---|
| `condition` | Internal SparkVSR condition for inference |
| `reference_frames` | Prepared reference frame preview batch |

### `SparkVSR Nano-Banana Pro Prompt`

Dedicated multiline text prompt node for `nano-banana-pro-ref`.

Connect its `api_prompt` output to `SparkVSR Prepare Reference` -> `api_prompt_input`.

### `SparkVSR Inference`

Runs the SparkVSR diffusion pipeline.

| Parameter | Description |
|---|---|
| `ref_guidance_scale` | Reference-frame guidance scale. `1.0` is the base setting |
| `chunk_len` | Temporal chunk length. `0` processes the full video in one pass |
| `overlap_t` | Temporal overlap between chunks |
| `tile_size_h` / `tile_size_w` | Spatial tile size. `0` disables tiling |
| `overlap_h` / `overlap_w` | Spatial overlap between tiles |
| `seed` | Random seed |
| `sr_noise_step` | Main denoising timestep |
| `noise_step` | Extra noise added to the low-quality latent. Blank or `0` disables it |
| `prompt` | Optional text prompt |
| `output_fps` | Output FPS override. `0` inherits from the condition |

### `SparkVSR Save Video`

Saves the output frames to an MP4 and shows the output preview.

| Parameter | Description |
|---|---|
| `frames` | Output frames as ComfyUI `IMAGE` |
| `fps` | Output frame rate |
| `filename_prefix` | Output filename prefix |
| `format` | `mp4_yuv420p` or `mp4_yuv444p` |
| `output_dir` | Output directory. Empty means ComfyUI's output folder |

## VRAM Tips

Enable `vae_slicing` and `vae_tiling` on `SparkVSR Load Model` for lower VRAM usage. Enable `cpu_offload` only when needed, since it is slower.
