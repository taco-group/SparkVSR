from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
    CogVideoXImageToVideoPipeline,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from finetune.datasets.ref_real_sr_dataset import RefRealSRDataset 

from ..utils import register
import random
from torchvision import transforms
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from accelerate.logging import get_logger
from finetune.constants import LOG_LEVEL, LOG_NAME
import math

logger = get_logger(LOG_NAME, LOG_LEVEL)

class KFVSRS2RefTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline # I2V Pipeline
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        if components.transformer.config.in_channels != 32:
             logger.warning(f"Expected transformer in_channels to be 32 for I2V, but got {components.transformer.config.in_channels}. Proceeding, but check model path.")

        return components

    @override
    def prepare_dataset(self) -> None:
        from finetune.datasets.ref_real_sr_image_video_dataset import RefRealSRImageVideoDataset
        self.dataset = RefRealSRImageVideoDataset(
            **(self.args.model_dump()),
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )
        
        # Prepare VAE and text encoder 
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        
        # S2 usually sets a custom collate, but we override collate_fn in this class anyway.
        # So standard DataLoader init is fine.
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe
    
    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding
    
    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "hq_videos": [], 
            "lq_videos": [], 
            "hq_images": [], 
            "lq_images": [], 
            "prompts": [], 
            "prompt_embeddings": [], 
            "video_metadatas": [], 
            "ref_frames": [], 
            "ref_indices": [],
            "encoded_lq_videos": [],
            "encoded_hq_videos": []
        }

        for sample in samples:
            # Common fields
            ret["prompts"].append(sample["prompt"])
            ret["prompt_embeddings"].append(sample["prompt_embedding"])
            ret["video_metadatas"].append(sample["video_metadata"])
            
            # S1/Video fields
            if "hq_video" in sample and sample["hq_video"] is not None: ret["hq_videos"].append(sample["hq_video"])
            if "lq_video" in sample and sample["lq_video"] is not None: ret["lq_videos"].append(sample["lq_video"])
            if "ref_video" in sample and sample["ref_video"] is not None: ret["ref_frames"].append(sample["ref_video"])
            else: ret["ref_frames"].append(None) # Append None to keep alignment
            
            if "ref_indices" in sample: ret["ref_indices"].append(sample["ref_indices"])
            else: ret["ref_indices"].append(None)
            
            # S2/Image fields
            if "hq_image" in sample and sample["hq_image"] is not None: ret["hq_images"].append(sample["hq_image"])
            if "lq_image" in sample and sample["lq_image"] is not None: ret["lq_images"].append(sample["lq_image"])

            # Pre-encoded
            if sample.get("encoded_hq_video") is not None:
                ret["encoded_hq_videos"].append(sample["encoded_hq_video"])
            if sample.get("encoded_lq_video") is not None:
                ret["encoded_lq_videos"].append(sample["encoded_lq_video"])

        # Stack Tensors
        if len(ret["hq_videos"]) > 0: ret["hq_videos"] = torch.stack(ret["hq_videos"])
        if len(ret["lq_videos"]) > 0: ret["lq_videos"] = torch.stack(ret["lq_videos"])
        if len(ret["hq_images"]) > 0: ret["hq_images"] = torch.stack(ret["hq_images"])
        if len(ret["lq_images"]) > 0: ret["lq_images"] = torch.stack(ret["lq_images"])
        ret["prompt_embeddings"] = torch.stack(ret["prompt_embeddings"])
        
        if len(ret["encoded_hq_videos"]) > 0:
            ret["encoded_hq_videos"] = torch.stack(ret["encoded_hq_videos"])
        if len(ret["encoded_lq_videos"]) > 0:
            ret["encoded_lq_videos"] = torch.stack(ret["encoded_lq_videos"])

        return ret

    def _prepare_ref_latent_i2v(self, batch, lq_latent_shape, dtype, device, is_image=False):
        batch_size = lq_latent_shape[0]
        C, F, H, W = lq_latent_shape[1], lq_latent_shape[2], lq_latent_shape[3], lq_latent_shape[4]
        
        # Initialize full zero padding
        full_ref_latent = torch.zeros((batch_size, C, F, H, W), device=device, dtype=dtype)
        
        if is_image:
            # For Image: encode encoded image latent directly concat zero latent
            return full_ref_latent 
        
        # For Video: use first frame as ref
        for b in range(batch_size):
            ref_frame = None
            if "hq_videos" in batch and len(batch["hq_videos"]) > b:
                 # hq_videos is [B, C, F, H, W]
                 # We take the first frame [C, 1, H, W]
                 ref_frame = batch["hq_videos"][b, :, 0:1, :, :]
            else:
                 continue

            if ref_frame is None: continue

            ref_frame = ref_frame.to(device=device, dtype=self.components.vae.dtype)
            
            # --- 1. Stronger Augmentation for Reference Frames ---
            # Convert to [0, 1] for torchvision transforms
            ref_aug = (ref_frame * 0.5 + 0.5).clamp(0, 1) # [C, 1, H, W]
            
            # S2: ref_frame is [C, 1, H, W]. Squeeze to [C, H, W] for transforms if needed? 
            # torchvision transforms usually work on [C, H, W] or [B, C, H, W]. [C, 1, H, W] might be treated as 1 frame video or batch depending on dim.
            # Lets squeeze dim 1.
            ref_aug = ref_aug.squeeze(1) # [C, H, W]

            # A. Color Jitter (Moderate)
            # Apply with 50% probability
            if torch.rand(1) < 0.5:
                jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)
                ref_aug = jitter(ref_aug)
            
            # B. Gaussian Blur (Weak-Moderate)
            # Apply with 30% probability
            if torch.rand(1) < 0.3:
                blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ref_aug = blur(ref_aug)
            
            # Back to [-1, 1] and unsqueeze
            ref_frame_aug = (ref_aug * 2.0 - 1.0).clamp(-1, 1).unsqueeze(1)

            # C. Gaussian Noise (Always applied slightly)
            # Noise level: sigma ~ 0.05
            noise = torch.randn_like(ref_frame_aug) * 0.05
            noisy_ref = ref_frame_aug + noise
            
            # VAE Encode
            latent_dist = self.components.vae.encode(noisy_ref.unsqueeze(0)).latent_dist
            lat = latent_dist.sample() * self.components.vae.config.scaling_factor
            
            # Place in full_ref_latent at index 0
            if 0 < F:
                full_ref_latent[b, :, 0, :, :] = lat[0, :, 0, :, :]
                    
        return full_ref_latent

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        is_image_batch = random.random() < self.args.image_ratio
        
        prompt_embedding = batch["prompt_embeddings"]
        
        # 1. Encode Main Content (LQ/HQ)
        if self.args.is_latent:
             # Assuming pre-encoded logic supported if passed
             lq_latent = batch["encoded_lq_videos"]
             hq_latent = batch["encoded_hq_videos"]
        else:
             with torch.no_grad():
                self.components.vae.to(self.accelerator.device)
                
                if is_image_batch:
                    lq_pixels = batch["lq_images"].to(self.accelerator.device)
                    hq_pixels = batch["hq_images"].to(self.accelerator.device)
                    # lq_images is likely [B, C, F=1, H, W] already from dataset
                    if lq_pixels.ndim == 4:
                        lq_pixels = lq_pixels.unsqueeze(2)
                    if hq_pixels.ndim == 4:
                        hq_pixels = hq_pixels.unsqueeze(2)
                else:
                    lq_pixels = batch["lq_videos"].to(self.accelerator.device) 
                    hq_pixels = batch["hq_videos"].to(self.accelerator.device)

                self.components.vae.to(self.accelerator.device)
                
                # Frame-by-frame encoding for LQ
                latents_lq = []
                for i in range(lq_pixels.shape[2]):
                    frame = lq_pixels[:, :, i:i+1, :, :]
                    # Use self.encode_video which handles scaling factor and sampling
                    latent_i = self.encode_video(frame)
                    latents_lq.append(latent_i)
                lq_latent = torch.cat(latents_lq, dim=2)
                
                # Frame-by-frame encoding for HQ
                # latents_hq = []
                # for i in range(hq_pixels.shape[2]):
                #     frame = hq_pixels[:, :, i:i+1, :, :]
                #     latent_i = self.encode_video(frame)
                #     latents_hq.append(latent_i)
                # hq_latent = torch.cat(latents_hq, dim=2)
        
        # 2. Prepare Reference Latent
        ref_latent = self._prepare_ref_latent_i2v(
            batch if not is_image_batch else {}, 
            lq_latent.shape, 
            lq_latent.dtype, 
            self.accelerator.device,
            is_image=is_image_batch
        )
        
        # 3. Concatenate (I2V Input = LQ + Ref)
        input_latent = torch.cat([lq_latent, ref_latent], dim=1) 
        
        # Initialize ncopy
        ncopy = 0
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
             ncopy = input_latent.shape[2] % patch_size_t
             if ncopy > 0:
                  input_first_frame = input_latent[:, :, :1, :, :]
                  input_latent = torch.cat([input_first_frame.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)
                  
                #   hq_first_frame = hq_latent[:, :, :1, :, :]
                #   hq_latent = torch.cat([hq_first_frame.repeat(1, 1, ncopy, 1, 1), hq_latent], dim=2)
             
        input_latent = input_latent.permute(0, 2, 1, 3, 4) 
        # reshape_hq_latent = hq_latent.permute(0, 2, 1, 3, 4)
        
        batch_size, num_frames, num_channels, height, width = input_latent.shape
        
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=input_latent.dtype)
        
        # Noise Addition (I2V Specifics)
        if self.args.noise_step != 0:
             lq_part = input_latent[:, :, :16, :, :]
             ref_part = input_latent[:, :, 16:, :, :]
             
             noise = torch.randn_like(lq_part)
             add_timesteps = torch.full(
                (batch_size,),
                fill_value=self.args.noise_step,
                dtype=torch.long,
                device=self.accelerator.device,
            )
             lq_part_noisy = self.components.scheduler.add_noise(lq_part.transpose(1, 2), noise.transpose(1, 2), add_timesteps).transpose(1, 2)
             input_latent = torch.cat([lq_part_noisy, ref_part], dim=2)
             
        timesteps = torch.full(
            (batch_size,),
            fill_value=self.args.sr_noise_step,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        
        if self.state.transformer_config.ofs_embed_dim is not None:
             ofs = torch.full((batch_size,), fill_value=2.0, device=self.accelerator.device, dtype=input_latent.dtype)
        else:
             ofs = None

        predicted_noise = self.components.transformer(
            hidden_states=input_latent,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            ofs=ofs,
            return_dict=False,
        )[0]
        
        predicted_noise = predicted_noise[:, :, :16, :, :].transpose(1, 2)
        lq_sample = input_latent[:, :, :16, :, :].transpose(1, 2) 
        
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, lq_sample, timesteps
        )
        
        # --- Loss Calculation (S2 Strategy) ---
        use_perceptual = (self.args.ea_dists_weight > 0 or self.args.dists_weight > 0 or 
                          self.args.ea_lpips_weight > 0 or self.args.lpips_weight > 0)
        
        loss_dict = {}
        
        # 1. Decode Prediction (Handle ncopy padding slice first)
        if ncopy > 0:
             latent_pred = latent_pred[:, :, ncopy:, :, :]

        latent_pred_scaled = 1 / self.components.vae.config.scaling_factor * latent_pred
        
        decoded_frames = []
        for i in range(latent_pred_scaled.shape[2]):
             latent_frame = latent_pred_scaled[:, :, i:i+1, :, :]
             frame_decoded = self.components.vae.decode(latent_frame).sample
             decoded_frames.append(frame_decoded)
        video_generate = torch.cat(decoded_frames, dim=2)
        video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)
        
        if 'hq_pixels' not in locals():
             # Decode HQ Latent
             hq_latent_scaled = 1 / self.components.vae.config.scaling_factor * hq_latent
             # Slice padded frames from GT latent too if necessary
             if ncopy > 0:
                  hq_latent_scaled = hq_latent_scaled[:, :, ncopy:, :, :]
             
             decoded_hq = []
             for i in range(hq_latent_scaled.shape[2]):
                  lf = hq_latent_scaled[:, :, i:i+1, :, :]
                  fd = self.components.vae.decode(lf).sample
                  decoded_hq.append(fd)
             hq_pixels = torch.cat(decoded_hq, dim=2)
        
        # Normalize hq_pixels to [0,1]
        hq_pixels_norm = (hq_pixels * 0.5 + 0.5).clamp(0.0, 1.0)
        
        # Decode Latent (Whole Batch)
        video_generate = self.components.vae.decode(latent_pred_scaled).sample
        video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)
        
        # video_generate shape should now be [B, C, F_decoded, H, W]
        # F_decoded should be 17 (or 21 depending on padding).
        # We might need to slice video_generate if it's slightly different from hq_pixels due to VAE padding.
        
        if video_generate.shape[2] != hq_pixels_norm.shape[2]:
             # Usually VAE decoding might produce slightly different frame counts if not careful.
             # But 5 latents -> 17 frames is standard for CogVideoX ( 1 + 4*(5-1) ? No. )
             # Let's align them.
             min_f = min(video_generate.shape[2], hq_pixels_norm.shape[2])
             video_generate = video_generate[:, :, :min_f, :, :]
             hq_pixels_norm = hq_pixels_norm[:, :, :min_f, :, :]
             
        # 3. Compute MSE Loss (Pixel Space)
        mse_loss = F.mse_loss(video_generate.float(), hq_pixels_norm.float(), reduction="mean")
        loss_dict['mse_loss'] = mse_loss.detach().item()
        total_loss = mse_loss
        
        perceptual_loss = torch.tensor(0.0, device=self.accelerator.device)
        frame_diff_loss = torch.tensor(0.0, device=self.accelerator.device)

        if use_perceptual:
             perceptual_accum = 0.0

             for f in range(video_generate.shape[2]):
                  pred_frame = video_generate[:, :, f, :, :].float()
                  gt_frame = hq_pixels_norm[:, :, f, :, :].float()
                  
                  curr_loss = 0.0
                  if self.args.ea_dists_weight > 0:
                       dists = self.dists_loss(pred_frame, gt_frame)
                       edge = self.dists_loss(self.edge_detection_model(pred_frame), self.edge_detection_model(gt_frame))
                       curr_loss = dists + edge
                       curr_loss = curr_loss / (video_generate.shape[2] * 2) * self.args.ea_dists_weight
                  elif self.args.dists_weight > 0:
                       dists = self.dists_loss(pred_frame, gt_frame)
                       curr_loss = dists / video_generate.shape[2] * self.args.dists_weight
                  
                  # Add LPIPS support if needed to match s2_trainer exactly,
                  # assuming self.lpips_loss exists in base Trainer if arguments are enabled.
                  elif self.args.ea_lpips_weight > 0:
                       lpips = self.lpips_loss(pred_frame, gt_frame)
                       edge = self.lpips_loss(self.edge_detection_model(pred_frame), self.edge_detection_model(gt_frame))
                       curr_loss = lpips + edge
                       curr_loss = curr_loss / (video_generate.shape[2] * 2) * self.args.ea_lpips_weight
                  elif self.args.lpips_weight > 0:
                       lpips = self.lpips_loss(pred_frame, gt_frame)
                       curr_loss = lpips / video_generate.shape[2] * self.args.lpips_weight
                       
                  perceptual_accum += curr_loss
             
             perceptual_loss = perceptual_accum
             total_loss = total_loss + perceptual_loss
             loss_dict['perceptual_loss'] = perceptual_loss.detach().item()

        # Frame Diff Loss
        if self.args.frame_diff_weight > 0 and video_generate.shape[2] > 1:
             diff_gen = video_generate[:, :, 1:, :, :] - video_generate[:, :, :-1, :, :]
             diff_gt = hq_pixels_norm[:, :, 1:, :, :] - hq_pixels_norm[:, :, :-1, :, :]
             frame_diff_loss = F.l1_loss(diff_gen, diff_gt) * self.args.frame_diff_weight
             total_loss = total_loss + frame_diff_loss
             loss_dict['frame_diff_loss'] = frame_diff_loss.detach().item()

        loss_dict['loss'] = total_loss.detach().item()
        return total_loss

    @override
    def prepare_for_validation(self):
        super().prepare_for_validation()
        
        api_output_dir = self.args.output_dir / "validation_api_refs"
        self.state.api_ref_dir = api_output_dir
        gt_output_dir = self.args.output_dir / "validation_gt_refs"
        self.state.gt_ref_dir = gt_output_dir 
        
        if self.accelerator.is_main_process:
            from finetune.utils.ref_utils import get_ref_frames_api, save_ref_frames_locally
            from pathlib import Path
            api_output_dir.mkdir(parents=True, exist_ok=True)
            gt_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"[DualVal] Pre-fetching Frames. API: {api_output_dir}, GT: {gt_output_dir}")
            
            video_paths = self.state.validation_videos 
            for video_path in video_paths:
                if not isinstance(video_path, (str, Path)): continue 
                video_name = Path(video_path).stem
                t_frames = None if self.args.raw_test else self.state.train_frames
                
                try:
                    get_ref_frames_api(str(video_path), str(api_output_dir), video_name, t_frames, self.args.raw_test)
                except Exception as e:
                    logger.warning(f"Failed to get API refs for {video_name}: {e}")
                
                try:
                     save_ref_frames_locally(str(video_path), str(gt_output_dir), video_name, t_frames, self.args.raw_test)
                except Exception as e:
                     logger.warning(f"Failed to save GT refs locally for {video_name}: {e}")
                     
        self.accelerator.wait_for_everyone()

    @override
    def validation_step(self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        from torchvision import transforms
        from PIL import Image
        prompt, video = eval_data["prompt"], eval_data["video_tensor"]
        ref_video_gt = eval_data.get("ref_video")
        video_name = eval_data.get("video_name", str(self.accelerator.process_index))
        
        if video.ndim == 4:
            video = video.unsqueeze(0)

        b, c, f, h, w = video.shape
        
        if c > f and c == 3: pass
        elif f == 3 and c > f:
             video = video.permute(0, 2, 1, 3, 4)
             b, c, f, h, w = video.shape

        H_, W_ = h, w
        video_reshaped = video.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w) 
        video_up = torch.nn.functional.interpolate(video_reshaped, size=(H_*4, W_*4), mode="bilinear", align_corners=False)
        video_up = (video_up / 255.0 * 2.0) - 1.0
        video_up = video_up.reshape(b, f, c, H_*4, W_*4).permute(0, 2, 1, 3, 4)
        
        ref_sources = {}
        
        gt_ref_dir_status = "NOT SET"
        loaded_gt_frames = []
        use_cached_gt = False
        
        if hasattr(self.state, 'gt_ref_dir') and self.state.gt_ref_dir is not None:
             gt_ref_dir_status = str(self.state.gt_ref_dir)
             ref_sources["video"] = "GT_CACHED_PLACEHOLDER"
             use_cached_gt = True
        elif ref_video_gt is not None:
            ref_sources["video"] = ref_video_gt 
        else:
            ref_sources["video"] = video_up[0].permute(1, 0, 2, 3) 
            
        api_ref_dir_status = "NOT SET"
        if hasattr(self.state, 'api_ref_dir') and self.state.api_ref_dir is not None:
             api_ref_dir_status = str(self.state.api_ref_dir)
             ref_sources["video_api"] = "API_PLACEHOLDER"
        
        logger.info(f"[DualVal] Sample: {video_name}, GT_Dir: {gt_ref_dir_status}, API_Dir: {api_ref_dir_status}")

        ret_artifacts = []
        
        for ref_type, ref_source in ref_sources.items():
            run_inference = True
            current_ref_frames = None
            
            if ref_type == 'video' and isinstance(ref_source, torch.Tensor):
                 num_frames_video = ref_source.shape[0]
            else:
                 num_frames_video = video_up.shape[2] 
            
            # if num_frames_video >= 3:
            #     ref_idx_list = [0, num_frames_video // 2, num_frames_video - 1]
            # elif num_frames_video == 2:
            #     ref_idx_list = [0, 1]
            # elif num_frames_video == 1:
            #     ref_idx_list = [0]
            # else:
            ref_idx_list = []

            # Loading Logic
            if ref_type == "video_api":
                 api_frames_dir = self.state.api_ref_dir / "temp_frames_output"
                 loaded_frames = []
                 for idx in ref_idx_list:
                      fname = f"{video_name}_frame_{idx:05d}.png"
                      fpath = api_frames_dir / fname
                      if fpath.exists():
                           img = Image.open(fpath).convert("RGB")
                           target_h, target_w = video_up.shape[-2:]
                           if img.size != (target_w, target_h):
                                img = img.resize((target_w, target_h), Image.LANCZOS)
                           t_img = transforms.ToTensor()(img)
                           t_img = (t_img * 2.0) - 1.0
                           loaded_frames.append(t_img.to(self.accelerator.device, dtype=self.components.vae.dtype))
                      else:
                           logger.warning(f"[DualVal] API frame NOT found at {fpath}. Skipping API pass.")
                           run_inference = False
                           break
                 if run_inference:
                      current_ref_frames = loaded_frames
            elif ref_type == "video" and use_cached_gt:
                 gt_frames_dir = self.state.gt_ref_dir
                 loaded_frames = []
                 all_found = True
                 for idx in ref_idx_list:
                      fname = f"{video_name}_frame_{idx:05d}.png"
                      fpath = gt_frames_dir / fname
                      if fpath.exists():
                           img = Image.open(fpath).convert("RGB")
                           target_h, target_w = video_up.shape[-2:]
                           if img.size != (target_w, target_h):
                                img = img.resize((target_w, target_h), Image.LANCZOS)
                           t_img = transforms.ToTensor()(img)
                           t_img = (t_img * 2.0) - 1.0
                           loaded_frames.append(t_img.to(self.accelerator.device, dtype=self.components.vae.dtype))
                      else:
                           all_found = False
                           break
                 if all_found:
                      current_ref_frames = loaded_frames
                 else:
                      if ref_video_gt is not None:
                           current_ref_frames = []
                           for idx in ref_idx_list:
                                r_frame = ref_video_gt[idx].to(self.accelerator.device, dtype=self.components.vae.dtype)
                                if r_frame.max() > 1.05: r_frame = r_frame / 255.0
                                if r_frame.min() >= 0.0 and r_frame.max() <= 1.0: r_frame = (r_frame * 2.0) - 1.0
                                if r_frame.dim() == 3 and r_frame.shape[-2:] != video_up.shape[-2:]:
                                       r_frame = torch.nn.functional.interpolate(r_frame.unsqueeze(0), size=video_up.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                                current_ref_frames.append(r_frame)
                      else:
                           run_inference = False
            else:
                 current_ref_frames = []
                 source_tensor = ref_video_gt if ref_video_gt is not None else ref_source
                 if isinstance(source_tensor, torch.Tensor):
                     for idx in ref_idx_list:
                          r_frame = source_tensor[idx].to(self.accelerator.device, dtype=self.components.vae.dtype)
                          if r_frame.max() > 1.05: r_frame = r_frame / 255.0
                          if r_frame.min() >= 0.0 and r_frame.max() <= 1.0: r_frame = (r_frame * 2.0) - 1.0
                          if r_frame.dim() == 3 and r_frame.shape[-2:] != video_up.shape[-2:]:
                                 r_frame = torch.nn.functional.interpolate(r_frame.unsqueeze(0), size=video_up.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                          current_ref_frames.append(r_frame)
                 else:
                      run_inference = False

            if not run_inference: continue
            
            # Use 'pipe' to run inference actually? S1 uses components manual helper.
            # I should stick to S1 manual helper logic to align with training code structure, 
            # OR use logic similar to 'validation_step' of S1 which used manual calls.
            # Yes, S1 code shown used self.components manually.
            
            self.components.vae.to(self.accelerator.device)
            lq_latent = self.encode_video(video_up.to(self.accelerator.device, dtype=self.components.vae.dtype))
            
            B, C, F_lat, H_lat, W_lat = lq_latent.shape
            full_ref_latent = torch.zeros_like(lq_latent)
            
            for i, idx_in_video in enumerate(ref_idx_list):
                if i >= len(current_ref_frames): break
                r_frame = current_ref_frames[i]
                chunk = r_frame.unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(0) 
                chunk = chunk.transpose(1, 2)
                lat = self.encode_video(chunk)
                target_idx = idx_in_video // 4
                if target_idx < F_lat:
                    full_ref_latent[:, :, target_idx, :, :] = lat[0, :, 0, :, :]
            
            # --- Dual-Pass / CFG Logic ---
            do_classifier_free_guidance = self.args.ref_guidance_scale > 1.0
            
            if do_classifier_free_guidance:
                # Cond
                input_latent_cond = torch.cat([lq_latent, full_ref_latent], dim=1)
                # Uncond
                uncond_ref_latent = torch.zeros_like(full_ref_latent)
                input_latent_uncond = torch.cat([lq_latent, uncond_ref_latent], dim=1)
                
                input_latent = torch.cat([input_latent_uncond, input_latent_cond], dim=0) # [2*B, C*2, F, H, W]
                # B becomes 2*B for the forward pass
            else:
                input_latent = torch.cat([lq_latent, full_ref_latent], dim=1) 
            
            patch_size_t = self.state.transformer_config.patch_size_t
            ncopy = 0
            if patch_size_t is not None:
                 ncopy = input_latent.shape[2] % patch_size_t
                 f_copy = input_latent[:, :, :1, :, :]
                 input_latent = torch.cat([f_copy.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)
                 
            latents = input_latent.permute(0, 2, 1, 3, 4)
            
            self.components.text_encoder.to(self.accelerator.device)
            prompt_emb = self.encode_text(prompt)
            prompt_emb = prompt_emb.view(B, prompt_emb.shape[1], -1).to(dtype=latents.dtype)
            
            if do_classifier_free_guidance:
                 prompt_emb = torch.cat([prompt_emb, prompt_emb], dim=0)

            t = torch.full((latents.shape[0],), self.args.sr_noise_step, dtype=torch.long, device=self.accelerator.device)
            
            rotary_emb = self.prepare_rotary_positional_embeddings(
                height=H_lat * 2 ** (len(self.components.vae.config.block_out_channels) - 1), 
                width=W_lat * 2 ** (len(self.components.vae.config.block_out_channels) - 1),
                num_frames=latents.shape[1],
                transformer_config=self.state.transformer_config,
                vae_scale_factor_spatial=2 ** (len(self.components.vae.config.block_out_channels) - 1),
                device=self.accelerator.device
            ) if self.state.transformer_config.use_rotary_positional_embeddings else None
            
            if self.state.transformer_config.ofs_embed_dim is not None:
                ofs = torch.full((latents.shape[0],), fill_value=2.0, device=self.accelerator.device, dtype=latents.dtype)
            else:
                ofs = None

            predicted_noise = self.components.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_emb,
                timestep=t,
                image_rotary_emb=rotary_emb,
                ofs=ofs,
                return_dict=False
            )[0]
            
            # Predict noise: velocity
            # latents is [B or 2B, F, C_total, H, W]
            lq_sample = latents[:, :, :16, :, :]
            predicted_noise_slice = predicted_noise[:, :, :16, :, :].transpose(1, 2)
            
            # Apply Guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = predicted_noise_slice.chunk(2)
                predicted_noise_slice = noise_pred_uncond + self.args.ref_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Split lq_sample and t as well for scheduler step
                lq_sample = lq_sample.chunk(2)[1] # Take cond part
                t = t.chunk(2)[0]

            generated_latents = self.components.scheduler.get_velocity(predicted_noise_slice, lq_sample.transpose(1, 2), t)
            
            if ncopy > 0:
                generated_latents = generated_latents[:, :, ncopy:, :, :]
            
            video_gen = self.components.vae.decode(generated_latents / self.components.vae.config.scaling_factor).sample
            video_gen = (video_gen / 2 + 0.5).clamp(0, 1)
            video_gen = video_gen.cpu().permute(0, 2, 3, 4, 1).float().numpy() 
            video_gen = video_gen[0] 
            
            frames = [Image.fromarray((frame * 255).astype('uint8')) for frame in video_gen]
            
            artifact_key = "video" if ref_type == "video" else "video_api"
            ret_artifacts.append((artifact_key, frames))
            logger.info(f"[DualVal] Pass {ref_type} completed (Scale={self.args.ref_guidance_scale}).")

        return ret_artifacts

    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def prepare_rotary_positional_embeddings(self, height, width, num_frames, transformer_config, vae_scale_factor_spatial, device):
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)
        p = transformer_config.patch_size
        p_t = transformer_config.patch_size_t
        base_size_width = transformer_config.sample_width // p
        base_size_height = transformer_config.sample_height // p

        if p_t is None:
            grid_crops_coords = self.get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            base_num_frames = (num_frames + p_t - 1) // p_t
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )
        return freqs_cos, freqs_sin

register("sparkvsr-s2", "sft", KFVSRS2RefTrainer)
