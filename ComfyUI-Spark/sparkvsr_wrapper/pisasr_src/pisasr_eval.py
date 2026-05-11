"""
PiSA-SR evaluation module — bundled inside ComfyUI-SparkVSR.

Evaluation-only subset of PiSA-SR's pisasr.py.
Adapted from https://github.com/csslc/PiSA-SR

Compatibility with diffusers >= 0.27 and peft >= 0.9.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules
from peft.utils.other import transpose

from .src.models.autoencoder_kl import AutoencoderKL
from .src.models.unet_2d_condition import UNet2DConditionModel
from .src.my_utils.vaehook import VAEHook


class PiSASR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = getattr(args, "device", "cuda")
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        # Propagate device to the shared devices module (used by vaehook.py)
        from sparkvsr_wrapper.pisasr_src.src.my_utils import devices as _dev_mod
        _dev_mod.device = torch.device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        self._load_pretrained_weights(args.pretrained_path)

        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size,
        )

        if not args.default:
            self._prepare_lora_deltas(["default_encoder_sem", "default_decoder_sem", "default_others_sem"])
        set_weights_and_activate_adapters(
            self.unet,
            ["default_encoder_sem", "default_decoder_sem", "default_others_sem"],
            [1.0, 1.0, 1.0],
        )
        self.unet.merge_and_unload()

        self._move_models_to_device_and_dtype()

        self.timesteps1 = torch.tensor([1], device=self.device).long()
        self.lambda_pix = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem = torch.tensor([args.lambda_sem], device=self.device)

    def _get_dtype(self, precision):
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _move_models_to_device_and_dtype(self):
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        sd = torch.load(pretrained_path, map_location="cpu")
        self._load_and_save_ckpt_from_state_dict(sd)

    def _prepare_lora_deltas(self, adapter_names):
        self.lora_deltas_sem = {}
        key_list = [key for key, _ in self.unet.named_modules() if "lora_" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.unet, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    for active_adapter in adapter_names:
                        if active_adapter in target.lora_A.keys():
                            weight_A = target.lora_A[active_adapter].weight
                            weight_B = target.lora_B[active_adapter].weight
                            s = target.get_base_layer().weight.size()
                            if s[2:4] == (1, 1):
                                output_tensor = (
                                    weight_B.squeeze(3).squeeze(2)
                                    @ weight_A.squeeze(3).squeeze(2)
                                ).unsqueeze(2).unsqueeze(3) * target.scaling[active_adapter]
                            elif len(s) == 2:
                                output_tensor = (
                                    transpose(weight_B @ weight_A, False)
                                    * target.scaling[active_adapter]
                                )
                            else:
                                output_tensor = F.conv2d(
                                    weight_A.permute(1, 0, 2, 3), weight_B,
                                ).permute(1, 0, 2, 3) * target.scaling[active_adapter]
                            self.lora_deltas_sem[key + ".weight"] = (
                                output_tensor.data.to(dtype=self.weight_dtype, device=self.device)
                            )

    def _apply_lora_delta(self):
        for name, param in self.unet.named_parameters():
            if name in self.lora_deltas_sem:
                param.data = self.lora_deltas_sem[name] + self.ori_unet_weight[name]
            else:
                param.data = self.ori_unet_weight[name]

    def _apply_ori_weight(self):
        for name, param in self.unet.named_parameters():
            param.data = self.ori_unet_weight[name]

    def _load_and_save_ckpt_from_state_dict(self, sd):
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix  = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem  = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix,  adapter_name="default_others_pix")

        for name, param in self.unet.named_parameters():
            if "pix" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        set_weights_and_activate_adapters(
            self.unet,
            ["default_encoder_pix", "default_decoder_pix", "default_others_pix"],
            [1.0, 1.0, 1.0],
        )
        self.unet.merge_and_unload()
        self.ori_unet_weight = {
            name: param.clone().data.to(self.weight_dtype).to(self.device)
            for name, param in self.unet.named_parameters()
        }

        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem,  adapter_name="default_others_sem")

        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        model_pred = self._process_latents(encoded_control, prompt_embeds, default)
        x_denoised = encoded_control - model_pred
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        torch.cuda.synchronize()
        total_time = time.time() - start_time
        return total_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds, default):
        h, w = encoded_control.size()[-2:]
        tile_size = self.args.latent_tiled_size
        tile_overlap = self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        if default:
            return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

        model_pred_sem = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        self._apply_ori_weight()
        model_pred_pix = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        self._apply_lora_delta()
        model_pred_sem -= model_pred_pix
        return self.lambda_pix * model_pred_pix + self.lambda_sem * model_pred_sem

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap):
        _, _, h, w = encoded_control.size()
        tile_size = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

        grid_rows, cur_x = 0, 0
        while cur_x < w:
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1
        grid_cols, cur_y = 0, 0
        while cur_y < h:
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        input_list, noise_preds = [], []
        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size
                input_tile = encoded_control[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols - 1:
                    input_list_t = torch.cat(input_list, dim=0)
                    if default:
                        model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
                    else:
                        model_out_sem = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
                        self._apply_ori_weight()
                        model_out_pix = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
                        self._apply_lora_delta()
                        model_out_sem -= model_out_pix
                        model_out = self.lambda_pix * model_out_pix + self.lambda_sem * model_out_sem
                    input_list = []
                noise_preds.append(model_out)

        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        for row in range(grid_rows):
            for col in range(grid_cols):
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size
                noise_pred[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] += (
                    noise_preds[row * grid_cols + col] * tile_weights
                )
                contributors[:, :, ofs_y:ofs_y + tile_size, ofs_x:ofs_x + tile_size] += tile_weights
        return noise_pred / contributors

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        import numpy as np
        from einops import repeat

        latent_width  = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2
        x_probs = [
            np.exp(-(x - midpoint) ** 2 / (latent_width ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
            for x in range(latent_width)
        ]
        midpoint = latent_height / 2
        y_probs = [
            np.exp(-(y - midpoint) ** 2 / (latent_height ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(
            torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1)
        ).float()

    def _init_tiled_vae(self, encoder_tile_size=1024, decoder_tile_size=224):
        encoder = self.vae.encoder
        decoder = self.vae.decoder
        # Save original forward before replacing (VAEHook calls net.original_forward)
        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward
        encoder.forward = VAEHook(
            encoder,
            encoder_tile_size,
            is_decoder=False,
            fast_decoder=False,
            fast_encoder=False,
            color_fix=False,
        )
        decoder.forward = VAEHook(
            decoder,
            decoder_tile_size,
            is_decoder=True,
            fast_decoder=False,
            fast_encoder=False,
            color_fix=False,
        )
