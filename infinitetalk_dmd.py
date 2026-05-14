import copy
import gc
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from wan.modules.multitalk_model import WanModel


def get_wan_model(config, checkpoint_dir, weight_files=None, debug=False):
    """
    Load a WanModel from a checkpoint directory.
    The model config (config.json) is read from `checkpoint_dir`; weights are
    merged from `weight_files`.
    """
    import json

    from safetensors.torch import load_file

    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    wan_config = json.load(open(config_path))

    if debug:
        wan_config['num_layers'] = 1

    if "weight_init" in wan_config:
        model = WanModel(**wan_config)
    else:
        # weight_init=False speeds up loading since weights are overwritten below.
        model = WanModel(weight_init=False, **wan_config)

    if debug:
        logging.info("Debug mode enabled: skip weight loading.")
        model.eval()
        return model

    merged_state_dict = {}
    for weight_file in weight_files:
        sd = load_file(weight_file)
        merged_state_dict.update(sd)
    model.load_state_dict(merged_state_dict)
    model.eval()
    return model


class InfiniteTalkDMD(nn.Module):
    """
    DMD (Distribution Matching Distillation) module for InfiniteTalk.

    Roles:
    - Generator (Student): the model being trained for few-step generation.
    - Real Score (Teacher): frozen pretrained model providing the target distribution.
    - Fake Score (Critic): trained alongside the student to model its distribution.

    Initialization modes:
    - stage1:   only generator is materialized (full flow-matching fine-tune).
    - stage2:   generator + real_score + fake_score are materialized.
    - val_only: only generator is materialized for inference.
    """

    def __init__(
        self,
        config,
        device,
        checkpoint_dir,
        infinitetalk_dir,
        stage="stage2",
        debug=False,
    ):
        super().__init__()
        if stage not in ("stage1", "stage2", "val_only"):
            raise ValueError(f"Unknown stage: {stage}")

        self.config = config
        self.device = device
        self.debug = debug
        self.stage = stage
        self.checkpoint_dir = checkpoint_dir
        self.infinitetalk_dir = infinitetalk_dir

        # DMD hyper-parameters
        self.text_guide_scale = getattr(config, 'text_guide_scale', 4.0)
        self.audio_guide_scale = getattr(config, 'audio_guide_scale', 3.0)
        self.num_train_timestep = 1000
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        raw_steps = getattr(config, 'denoising_step_list', [1000, 750, 500, 250])
        self.raw_denoising_step_list = self._parse_raw_step_list(raw_steps)
        self.denoising_step_list = self._parse_denoising_step_list(raw_steps)
        self.keep_k_chunks = getattr(config, 'keep_k_chunks', -1)
        self.dmd_loss_weight = float(getattr(config, 'dmd_loss_weight', 1.0))
        self.temporal_align_weight = float(getattr(config, 'temporal_align_weight', 0.25))

        self._fsdp_wrapped = False
        # Debug helpers: allow saving arbitrary latent batches from the debugger.
        self._debug_processor = None
        self._debug_output_dir = getattr(config, "output_dir", None)
        self._rvm_debug_saved_once = False
        self._rvm_debug_dump_idx = 0
        # Expose latest generator loss components for trainer logging.
        self.latest_gen_loss_components = None

        logging.info(
            "DMD config: stage=%s, "
            "raw_denoising_step_list=%s, denoising_step_list(shifted)=%s, "
            "dmd_loss_weight=%s, temporal_align_weight=%s, "
            "text_guide_scale=%s, audio_guide_scale=%s",
            self.stage,
            self.raw_denoising_step_list,
            self.denoising_step_list,
            self.dmd_loss_weight,
            self.temporal_align_weight,
            self.text_guide_scale,
            self.audio_guide_scale,
        )

        # Resolve initial weight files. init_stage1_full takes priority when set.
        init_stage1_full = getattr(config, 'init_stage1_full', None)
        if debug:
            weight_files = None
        elif init_stage1_full:
            if not os.path.isfile(str(init_stage1_full)):
                raise ValueError(f"init_stage1_full not found: {init_stage1_full}")
            weight_files = [str(init_stage1_full)]
            logging.info("Using init_stage1_full as initial weights: %s", init_stage1_full)
        else:
            weight_files = [
                f"{checkpoint_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{checkpoint_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                f"{checkpoint_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{checkpoint_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{checkpoint_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{checkpoint_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{checkpoint_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                f"{infinitetalk_dir}",
            ]

        if stage == "stage2":
            self._init_separate_models(weight_files)
        else:
            # stage1 / val_only: only generator is materialized.
            self._init_generator_only(weight_files)

    # ==================== Init helpers ====================

    def _init_separate_models(self, weight_files):
        """Initialize generator + real_score (teacher) + fake_score (critic)."""
        logging.info("Loading Generator (Student)...")
        self.generator = get_wan_model(self.config, self.checkpoint_dir, weight_files, debug=self.debug)
        self.generator.requires_grad_(True)

        logging.info("Creating Real Score (Teacher) from Generator...")
        self.real_score = copy.deepcopy(self.generator)
        self.real_score.requires_grad_(False)

        logging.info("Creating Fake Score (Critic) from Generator...")
        self.fake_score = copy.deepcopy(self.generator)
        self.fake_score.requires_grad_(True)

    def _init_generator_only(self, weight_files):
        """Initialize generator only (stage1 training or val_only inference)."""
        logging.info("Loading Generator only (stage=%s)...", self.stage)
        self.generator = get_wan_model(self.config, self.checkpoint_dir, weight_files, debug=self.debug)
        if self.stage == "val_only":
            self.generator.requires_grad_(False)
        else:
            self.generator.requires_grad_(True)
        self.real_score = None
        self.fake_score = None

    # ==================== Training Setup Methods ====================

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on trainable models."""
        if self.fake_score is None:
            logging.info("Enabling gradient checkpointing for Generator only...")
            self.generator.enable_gradient_checkpointing()
        else:
            logging.info("Enabling gradient checkpointing for Generator and Critic...")
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

    def setup_full_finetune(self):
        """Enable full-parameter fine-tuning for generator (+critic when present)."""
        self.generator.requires_grad_(True)
        trainable_count = sum(p.numel() for p in self.generator.parameters())
        logging.info(f"Full fine-tune Generator: {trainable_count:,} trainable parameters.")

        if self.fake_score is not None:
            self.fake_score.requires_grad_(True)
            critic_count = sum(p.numel() for p in self.fake_score.parameters())
            logging.info(f"Full fine-tune Critic: {critic_count:,} trainable parameters.")

    def load_resume_full_before_fsdp(self, resume_dir):
        """
        Load resume checkpoint for full fine-tuning before FSDP wrapping.

        Stage1 saves model_{step}.safetensors; stage2 saves
        generator_{step}.safetensors and critic_{step}.safetensors.
        """
        if not resume_dir:
            return

        training_state_path = os.path.join(resume_dir, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise ValueError(f"Training state file not found: {training_state_path}")

        checkpoint = torch.load(training_state_path, map_location='cpu')
        step = checkpoint['step']

        from safetensors.torch import load_file

        gen_path = os.path.join(resume_dir, f"generator_{step}.safetensors")
        if not os.path.exists(gen_path):
            # Fallback to stage1 naming.
            gen_path = os.path.join(resume_dir, f"model_{step}.safetensors")
        if not os.path.exists(gen_path):
            raise ValueError(f"Generator checkpoint not found in {resume_dir}")
        gen_sd = load_file(gen_path)
        missing, unexpected = self.generator.load_state_dict(gen_sd, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys in generator checkpoint: {unexpected}")
        logging.info(
            "Loaded generator full checkpoint from %s (step=%s, loaded=%d, missing=%d).",
            gen_path, step, len(gen_sd), len(missing),
        )

        critic_path = os.path.join(resume_dir, f"critic_{step}.safetensors")
        if os.path.exists(critic_path) and self.fake_score is not None:
            critic_sd = load_file(critic_path)
            missing, unexpected = self.fake_score.load_state_dict(critic_sd, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected keys in critic checkpoint: {unexpected}")
            logging.info(
                "Loaded critic full checkpoint from %s (step=%s, loaded=%d, missing=%d).",
                critic_path, step, len(critic_sd), len(missing),
            )
        elif os.path.exists(critic_path) and self.fake_score is None:
            logging.warning(
                "Critic checkpoint found at %s but fake_score is not initialized; skip critic resume.",
                critic_path,
            )

    def setup_fsdp(self, sharding_strategy, mixed_precision, wrap_strategy,
                   transformer_module=None, off_load_to_cpu=False):
        """Wrap available submodels with FSDP."""
        from wan.utils.utils import fsdp_wrap

        if self._fsdp_wrapped:
            logging.warning("FSDP already applied, skip.")
            return

        logging.info("Wrapping models with FSDP...")

        self.generator = fsdp_wrap(
            self.generator,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            wrap_strategy=wrap_strategy,
            transformer_module=transformer_module,
            off_load_to_cpu=True,
        )

        if self.real_score is not None:
            self.real_score = fsdp_wrap(
                self.real_score,
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                wrap_strategy=wrap_strategy,
                transformer_module=transformer_module,
                off_load_to_cpu=True,
            )

        if self.fake_score is not None:
            self.fake_score = fsdp_wrap(
                self.fake_score,
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                wrap_strategy=wrap_strategy,
                transformer_module=transformer_module,
                off_load_to_cpu=off_load_to_cpu,
            )

        self._fsdp_wrapped = True

    def setup_for_training(self, args):
        """Single-call setup for training: grad ckpt + full fine-tune + resume + FSDP."""
        if getattr(args, 'gradient_checkpointing', False):
            self.enable_gradient_checkpointing()

        self.setup_full_finetune()

        if getattr(args, 'resume_from', None):
            self.load_resume_full_before_fsdp(args.resume_from)

        if getattr(args, 'use_fsdp', False):
            from wan.modules.multitalk_model import WanAttentionBlock

            transformer_module = None
            if getattr(args, 'fsdp_wrap_strategy', None) == "transformer":
                if getattr(args, 'fsdp_transformer_module', None) == "wan_attention":
                    transformer_module = (WanAttentionBlock,)
            fsdp_cpu_offload = bool(getattr(args, 'fsdp_cpu_offload', False))
            self.setup_fsdp(
                sharding_strategy=args.sharding_strategy,
                mixed_precision=args.mixed_precision,
                wrap_strategy=args.fsdp_wrap_strategy,
                transformer_module=transformer_module,
                off_load_to_cpu=fsdp_cpu_offload,
            )
        else:
            self.to(self.device, dtype=torch.bfloat16)

        return self

    def setup_for_validation(self, args):
        """Minimal setup for validation-only inference (generator only)."""
        self.generator.requires_grad_(False)

        if getattr(args, 'resume_from', None):
            self.load_resume_full_before_fsdp(args.resume_from)

        if getattr(args, 'use_fsdp', False):
            from wan.modules.multitalk_model import WanAttentionBlock

            transformer_module = None
            if getattr(args, 'fsdp_wrap_strategy', None) == "transformer":
                if getattr(args, 'fsdp_transformer_module', None) == "wan_attention":
                    transformer_module = (WanAttentionBlock,)
            fsdp_cpu_offload = bool(getattr(args, 'fsdp_cpu_offload', False))
            self.setup_fsdp(
                sharding_strategy=args.sharding_strategy,
                mixed_precision=args.mixed_precision,
                wrap_strategy=args.fsdp_wrap_strategy,
                transformer_module=transformer_module,
                off_load_to_cpu=fsdp_cpu_offload,
            )
        else:
            self.to(self.device, dtype=torch.bfloat16)

        self.eval()
        return self

    # ==================== Forward Methods ====================

    def forward_wan(self, model, x, t, context, seq_len, clip_fea, y, audio, ref_target_masks=None):
        """
        Wrapper around WanModel forward.
        Signature: (x, t, context, seq_len, clip_fea=None, y=None, audio=None, ...)
        """
        return model(x, t, context, seq_len, clip_fea=clip_fea, y=y, audio=audio, ref_target_masks=ref_target_masks)

    def get_noise_and_timestep(self, latents, generator=False, shift=False):
        """
        Timestep sampler aligned with CausVid/DMD:
        - generator=True:  discrete sampling from [250, 500, 750, 1000].
        - generator=False: continuous sampling from [1, 1000].
        When shift=True, apply timestep_shift.
        """
        bs = latents.shape[0]
        if generator:
            timestep_choices = torch.tensor([250, 500, 750, 1000], device=self.device)
            t = timestep_choices[torch.randint(0, len(timestep_choices), (bs,), device=self.device)].long()
        else:
            t = torch.randint(1, 1001, (bs,), device=self.device, dtype=torch.long)
        t_norm = t.float() / 1000.0

        if shift and hasattr(self.config, 'timestep_shift') and self.config.timestep_shift > 1:
            shift_val = float(self.config.timestep_shift)
            t_norm = shift_val * t_norm / (1.0 + (shift_val - 1.0) * t_norm)
            t = (t_norm * 1000.0).long().clamp(1, 1000)

        t_norm = t_norm.view(bs, 1, 1, 1, 1)
        noise = torch.randn_like(latents)
        return noise, t, t_norm

    def get_x0_from_v(self, x_t, v, t_norm):
        # Linear Flow Matching: x_t = (1-t)x0 + t*x1 (x1=noise), v = x1 - x0 => x0 = x_t - t * v
        return x_t - t_norm * v

    def register_debug_io(self, processor=None, output_dir=None):
        """Register processor and output_dir used by debug-time helpers."""
        if processor is not None:
            self._debug_processor = processor
        if output_dir is not None:
            self._debug_output_dir = output_dir

    def save_debug_latent_batch_as_videos(
        self,
        latent_batch,
        tensor_name,
        processor=None,
        output_dir=None,
        step=None,
        rank=None,
        fps=25,
    ):
        """Decode and save a latent batch as mp4 videos (for interactive debugging)."""
        from wan.utils.utils import cache_video

        proc = processor if processor is not None else self._debug_processor
        if proc is None:
            raise ValueError("processor is required.")
        if not hasattr(proc, "vae") or proc.vae is None:
            raise ValueError("processor.vae is not available.")

        save_root = output_dir if output_dir is not None else self._debug_output_dir
        if save_root is None:
            raise ValueError("output_dir is required.")
        save_root = os.path.abspath(save_root)

        x = latent_batch
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"latent_batch must be Tensor/List[Tensor], got {type(latent_batch)}")
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() != 5:
            raise ValueError(f"latent_batch must have shape [B,C,T,H,W], got {tuple(x.shape)}")

        vae_device = getattr(proc.vae, "device", x.device)
        x = x.detach().to(device=vae_device, dtype=torch.float32)
        latents = [x[idx] for idx in range(x.shape[0])]

        with torch.no_grad():
            decoded_videos = proc.vae.decode(latents)

        if rank is None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

        step_dir = save_root if step is None else os.path.join(save_root, f"iter_{step}")
        os.makedirs(step_dir, exist_ok=True)

        saved_paths = []
        for idx, video in enumerate(decoded_videos):
            sample_dir = os.path.join(step_dir, f"rank_{rank}_sample_{idx + 1}" if rank is not None else f"sample_{idx + 1}")
            os.makedirs(sample_dir, exist_ok=True)
            save_path = os.path.join(sample_dir, f"{tensor_name}.mp4")
            cache_video(
                tensor=video.unsqueeze(0),
                save_file=save_path,
                fps=fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            saved_paths.append(save_path)

        logging.info("Saved debug latent videos '%s': %s", tensor_name, saved_paths)
        return saved_paths

    # ==================== Step list helpers ====================

    def _parse_raw_step_list(self, raw_steps):
        """Parse denoising_step_list into raw integers (without shift)."""
        if isinstance(raw_steps, str):
            steps = [int(x.strip()) for x in raw_steps.split(",") if x.strip()]
        elif isinstance(raw_steps, (list, tuple)):
            steps = [int(x) for x in raw_steps]
        else:
            steps = [1000, 750, 500, 250]
        if len(steps) == 0:
            steps = [1000, 750, 500, 250]
        return [int(max(1, min(1000, s))) for s in steps]

    def _apply_timestep_shift(self, t_raw):
        """Apply timestep shift to a single raw timestep value."""
        t_raw = int(max(1, min(1000, t_raw)))
        if hasattr(self.config, 'timestep_shift') and self.config.timestep_shift > 1:
            shift = float(self.config.timestep_shift)
            t_norm = float(t_raw) / 1000.0
            t_norm = shift * t_norm / (1.0 + (shift - 1.0) * t_norm)
            return int(max(1, min(1000, t_norm * 1000.0)))
        return t_raw

    def _parse_denoising_step_list(self, raw_steps):
        """Parse denoising_step_list and apply timestep_shift, keeping user order."""
        steps = self._parse_raw_step_list(raw_steps)
        return [self._apply_timestep_shift(s) for s in steps]

    def _sample_synced_exit_index(self):
        """Sample a random exit index synchronized across all distributed ranks."""
        max_idx = len(self.denoising_step_list) - 1
        device = self.device
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                idx = random.randint(0, max_idx)
            else:
                idx = 0
            idx_tensor = torch.tensor([idx], device=device, dtype=torch.long)
            dist.broadcast(idx_tensor, src=0)
            idx = int(idx_tensor.item())
        else:
            idx = random.randint(0, max_idx)
        return idx

    # ==================== Prediction helpers ====================

    def _predict_x0_at_step(self, model, x_t, timestep, context, seq_len, clip_fea, cond_latents,
                            audio_emb, ref_target_masks=None):
        """Run one model forward at a fixed timestep and return predicted x0."""
        bs = x_t.shape[0]
        t = torch.full((bs,), int(timestep), device=x_t.device, dtype=torch.long)
        t_norm = t.float().view(bs, 1, 1, 1, 1) / 1000.0
        out_list = self.forward_wan(
            model, x_t, t, context, seq_len, clip_fea, cond_latents, audio_emb,
            ref_target_masks,
        )
        v = torch.stack(out_list) if isinstance(out_list, list) else out_list
        pred_x0 = self.get_x0_from_v(x_t, v, t_norm)
        return pred_x0, t, t_norm

    def _predict_v_at_step(self, model, x_t, timestep, context, seq_len, clip_fea, cond_latents,
                           audio_emb, ref_target_masks=None):
        """Run one model forward at a fixed timestep and return predicted v."""
        bs = x_t.shape[0]
        t = torch.full((bs,), int(timestep), device=x_t.device, dtype=torch.long)
        t_norm = t.float().view(bs, 1, 1, 1, 1) / 1000.0
        out_list = self.forward_wan(
            model, x_t, t, context, seq_len, clip_fea, cond_latents, audio_emb,
            ref_target_masks,
        )
        v = torch.stack(out_list) if isinstance(out_list, list) else out_list
        return v, t, t_norm

    def _simulate_to_exit_step_inject_motion_frames_with_noise(
        self, start_noise, exit_idx, context, seq_len, clip_fea, cond_latents,
        audio_emb, ref_target_masks=None,
        clean_latents=None, inject_motion_frames_num=None,
        add_motion_noise=True,
    ):
        """
        Backward simulation (no_grad):
        Start from Gaussian noise at denoising_step_list[0], iteratively predict v
        and step deterministically until reaching exit_idx. Optionally inject
        motion latents at each step to keep simulation consistent with the exit step.
        """
        x_t = start_noise
        steps = self.denoising_step_list
        if exit_idx <= 0:
            return x_t

        with torch.no_grad():
            for i in range(exit_idx):
                cur_step = steps[i]
                next_step = steps[i + 1]
                cur_t_norm = float(cur_step) / 1000.0
                cur_t_norm_tensor = torch.full(
                    (x_t.shape[0], 1, 1, 1, 1),
                    cur_t_norm,
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                if clean_latents is not None and inject_motion_frames_num is not None:
                    x_t = self._inject_motion_source_with_noise(
                        x_t,
                        clean_latents,
                        inject_motion_frames_num,
                        t_norm=cur_t_norm_tensor,
                        add_noise=add_motion_noise,
                    )
                v_pred, _, _ = self._predict_v_at_step(
                    self.generator, x_t, cur_step, context, seq_len, clip_fea,
                    cond_latents, audio_emb, ref_target_masks=ref_target_masks,
                )
                next_t_norm = float(next_step) / 1000.0
                dt = cur_t_norm - next_t_norm
                x_t = x_t - dt * v_pred
        return x_t

    def _inject_motion_frames(self, noisy_latent, clean_latents, inject_motion_frames_num):
        """Replace the first inject_motion_frames_num positions with clean latents."""
        for i in range(noisy_latent.shape[0]):
            noisy_latent[i, :, :inject_motion_frames_num[i], :, :] = clean_latents[i, :, :inject_motion_frames_num[i], :, :]
        return noisy_latent

    def _inject_motion_source_with_noise(
        self,
        noisy_latent,
        motion_source_latent,
        motion_latent_num,
        t_norm,
        add_noise=True,
    ):
        """
        Inject motion source latent at the start of `noisy_latent`. Optionally
        blend with noise using the same intensity schedule as other positions.
        """
        if motion_source_latent is None:
            return noisy_latent
        if isinstance(motion_latent_num, int):
            motion_latent_num = torch.full(
                (noisy_latent.shape[0],),
                int(motion_latent_num),
                device=noisy_latent.device,
                dtype=torch.long,
            )

        out = noisy_latent.clone()
        B = out.shape[0]
        for i in range(B):
            n = int(motion_latent_num[i].item())
            if n <= 0:
                continue
            n = min(n, int(out.shape[2]), int(motion_source_latent.shape[2]))
            src = motion_source_latent[i : i + 1, :, :n, :, :]
            if add_noise:
                eps = torch.randn_like(src)
                src_to_inject = (1.0 - t_norm[i : i + 1]) * src + t_norm[i : i + 1] * eps
            else:
                src_to_inject = src
            out[i : i + 1, :, :n, :, :] = src_to_inject
        return out

    # ==================== RVM background helpers ====================

    def _compute_rvm_background_overlap_mse(
        self,
        pred_x0_student,
        first_frame_latent,
        motion_source_latent,
        motion_latent_num,
    ):
        """
        Decode latent to pixel space, segment via RVM, then compute temporal
        smoothness loss on adjacent latent frames over conservative background masks.

        The non-motion latent sequence is wrapped by two identical anchor latents
        built by encoding 5 repetitions of the reference frame and taking the
        2nd latent token.
        """
        m = int(motion_latent_num)
        proc = self._debug_processor
        vae = proc.vae
        rvm = proc.rvm
        vae_device = getattr(vae, "device", pred_x0_student.device)

        lat_t_non_motion = int(pred_x0_student.shape[2] - m)

        pred_with_motion = torch.cat(
            [motion_source_latent[:, :, :m], pred_x0_student[:, :, m:]],
            dim=2,
        )
        pred_with_motion = pred_with_motion.detach().to(device=vae_device, dtype=torch.float32)

        with torch.no_grad():
            pred_latents_for_decode = [pred_with_motion[i] for i in range(pred_with_motion.shape[0])]
            pred_videos = vae.decode(pred_latents_for_decode)
        pred_pixels = torch.stack(pred_videos, dim=0)

        # VAE temporal mapping: pixel frames covered by the first m latents = 1 + 4*(m-1) when m>0.
        motion_pixel_t = 1 + 4 * (m - 1)
        pred_pixels_non_motion = pred_pixels[:, :, motion_pixel_t:, :, :]

        with torch.no_grad():
            ref_lat = first_frame_latent.detach().to(device=vae_device, dtype=torch.float32)
            ref_videos = vae.decode([ref_lat[i] for i in range(ref_lat.shape[0])])
            ref_pixels_1 = torch.stack([v[:, :1, :, :] for v in ref_videos], dim=0)

            ref_pixels_5 = ref_pixels_1.repeat(1, 1, 5, 1, 1).contiguous()
            anchor_lat_list = [vae.encode([ref_pixels_5[i]])[0] for i in range(ref_pixels_5.shape[0])]
            anchor_lat_full = torch.stack(anchor_lat_list, dim=0)

            anchor_lat = anchor_lat_full[:, :, 1:2, :, :].to(device=pred_x0_student.device, dtype=pred_x0_student.dtype)

        ref_pixels_4 = ref_pixels_1.repeat(1, 1, 4, 1, 1).contiguous()
        pred_pixels_non_motion_with_anchor = torch.cat([ref_pixels_4, pred_pixels_non_motion, ref_pixels_4], dim=2)
        pred_pixels_01 = ((pred_pixels_non_motion_with_anchor.clamp(-1, 1) + 1.0) * 0.5).float()

        rvm_device = pred_x0_student.device
        with torch.no_grad():
            rec = [None] * 4
            pred_src = pred_pixels_01.detach().permute(0, 2, 1, 3, 4).contiguous().to(device=rvm_device)
            _, pred_pha, *rec = rvm(pred_src, *rec)

        pred_pha_bt1hw = pred_pha.to(device=pred_x0_student.device, dtype=torch.float32)
        bg_mask = (1.0 - pred_pha_bt1hw).clamp(0.0, 1.0)
        overlap_bg = (bg_mask[:, 1:] * bg_mask[:, :-1]).permute(0, 2, 1, 3, 4).contiguous()

        lat_h = int(pred_x0_student.shape[3])
        lat_w = int(pred_x0_student.shape[4])
        lat_t_diff = lat_t_non_motion + 1

        overlap_bg_lat = F.interpolate(
            overlap_bg,
            size=(lat_t_diff, lat_h, lat_w),
            mode="nearest",
        )
        # Blur and threshold the mask so only confident background is kept.
        blur_kernel = int(getattr(self.config, "rvm_bg_blur_kernel_lat", 3))
        blur_pad = blur_kernel // 2
        overlap_bg_lat = F.pad(overlap_bg_lat, pad=(blur_pad, blur_pad, blur_pad, blur_pad, 0, 0), mode="constant", value=1.0)
        overlap_bg_lat = F.avg_pool3d(overlap_bg_lat, kernel_size=(1, blur_kernel, blur_kernel), stride=1, padding=0)
        overlap_bg_lat = (overlap_bg_lat > 0.99).float().to(
            device=pred_x0_student.device, dtype=pred_x0_student.dtype
        )

        if not torch.any(overlap_bg_lat > 0):
            return pred_x0_student[:, :, m:, :, :].sum() * 0.0

        pred_lat_non_motion = pred_x0_student[:, :, m:, :, :]
        pred_lat_for_loss = torch.cat([anchor_lat, pred_lat_non_motion, anchor_lat], dim=2)
        sq_err = (
            pred_lat_for_loss[:, :, 1:, :, :].float()
            - pred_lat_for_loss[:, :, :-1, :, :].float()
        ).pow(2)
        weighted_sq_err = sq_err * overlap_bg_lat.expand_as(sq_err)
        denom = overlap_bg_lat.expand_as(sq_err).sum()
        return weighted_sq_err.sum() / denom

    # ==================== Score helpers (teacher CFG) ====================

    def _sample_teacher_raw_from_student_interval(self, hit_idx):
        """
        For a sampled student step index, sample one raw teacher timestep inside that interval.
        With raw steps [1000,750,500,250]:
          - hit 1000 -> [751,999]
          - hit 750  -> [501,749]
          - hit 500  -> [251,499]
          - hit 250  -> [1,249]
        """
        raw_steps = self.raw_denoising_step_list + [0]
        raw_cur = int(raw_steps[hit_idx])
        raw_next = int(raw_steps[hit_idx + 1])
        lower = raw_next + 1
        upper = raw_cur - 1
        return random.randint(lower, upper)

    # ==================== Motion-source builders ====================

    def _build_motion_source_from_prev_chunk_pixels(self, prev_chunk_pred, motion_frame_num=5):
        """
        Decode prev chunk latent to pixel space, keep the last `motion_frame_num`
        pixel frames, then re-encode to a latent that serves as the motion source
        of the next chunk.
        """
        proc = self._debug_processor
        if proc is None or not hasattr(proc, "vae") or proc.vae is None:
            raise ValueError(
                "Self-forcing++ pixel-space motion extraction requires a registered "
                "processor with VAE. Call register_debug_io(processor=...) before training."
            )

        with torch.no_grad():
            vae_device = getattr(proc.vae, "device", prev_chunk_pred.device)
            latents_for_decode = [
                prev_chunk_pred[i].detach().to(device=vae_device, dtype=torch.float32)
                for i in range(prev_chunk_pred.shape[0])
            ]
            decoded_videos = proc.vae.decode(latents_for_decode)

            pixel_motion_clips = []
            for video in decoded_videos:
                t_video = int(video.shape[1])
                keep = min(motion_frame_num, t_video)
                pixel_motion_clips.append(video[:, -keep:, :, :].to(device=vae_device, dtype=torch.float32))

            motion_latents = [proc.vae.encode([clip])[0] for clip in pixel_motion_clips]
            motion_source = torch.stack(motion_latents).to(
                device=prev_chunk_pred.device, dtype=prev_chunk_pred.dtype
            )
            motion_latent_num = int(motion_source.shape[2])
        return motion_source, motion_latent_num

    # ==================== Self-Forcing++ losses ====================

    def _generator_loss_self_forcing_plus_plus(
        self,
        self_forcing_chunks,
        train_step=None,
        return_visual_dict=False,
    ):
        """
        Self-Forcing++ generator objective:
        - k = len(self_forcing_chunks). Chunks 1..k-1 are simulated under no_grad;
          only chunk k keeps gradient.
        - Within chunk k, only one randomly sampled denoising step keeps gradient.
        - Motion source for chunk 1 is the first-frame latent; for later chunks
          it is built by decoding the previous chunk to pixels, keeping the last
          few frames and re-encoding.
        """
        if self_forcing_chunks is None or len(self_forcing_chunks) == 0:
            raise ValueError("self_forcing_chunks must be a non-empty list.")

        if self.keep_k_chunks > 0:
            k_sample = self.keep_k_chunks
        else:
            k_sample = int(len(self_forcing_chunks))

        prev_chunk_pred = None
        final_visual_dict = None
        loss = None

        for chunk_idx in range(k_sample):
            chunk = self_forcing_chunks[chunk_idx]
            context = chunk["context"]
            clip_fea = chunk["clip_fea"]
            cond_latents = chunk["cond_latents"]
            audio_emb = chunk["audio_emb"]
            seq_len = chunk["seq_len"]
            ref_target_masks = chunk.get("ref_target_masks")
            context_null = chunk.get("context_null")
            audio_null = chunk.get("audio_null")
            first_frame_latent = chunk.get("first_frame_latent")

            bs = int(context.shape[0])
            lat_c = int(first_frame_latent.shape[1])
            lat_t = int(cond_latents.shape[2])
            lat_h = int(cond_latents.shape[3])
            lat_w = int(cond_latents.shape[4])
            device = first_frame_latent.device
            dtype = first_frame_latent.dtype

            if chunk_idx == 0:
                motion_source = first_frame_latent
                motion_latent_num = 1
            else:
                motion_source, motion_latent_num = self._build_motion_source_from_prev_chunk_pixels(
                    prev_chunk_pred,
                    motion_frame_num=5,
                )

            with torch.no_grad():
                start_noise = torch.randn(bs, lat_c, lat_t, lat_h, lat_w, device=device, dtype=dtype)

                if chunk_idx == k_sample - 1:
                    gen_exit_idx = self._sample_synced_exit_index()
                else:
                    gen_exit_idx = len(self.denoising_step_list) - 1

                x_t_student = self._simulate_to_exit_step_inject_motion_frames_with_noise(
                    start_noise,
                    gen_exit_idx,
                    context,
                    seq_len,
                    clip_fea,
                    cond_latents,
                    audio_emb,
                    ref_target_masks=ref_target_masks,
                    clean_latents=motion_source,
                    inject_motion_frames_num=motion_latent_num,
                    add_motion_noise=False,
                )

                gen_exit_step = self.denoising_step_list[gen_exit_idx]
                t_student = torch.full((bs,), int(gen_exit_step), device=start_noise.device, dtype=torch.long)
                t_student_norm = t_student.float().view(bs, 1, 1, 1, 1) / 1000.0

                x_t_student = self._inject_motion_source_with_noise(
                    x_t_student,
                    motion_source_latent=motion_source,
                    motion_latent_num=motion_latent_num,
                    t_norm=t_student_norm,
                    add_noise=False,
                )

            grad_context = nullcontext() if (chunk_idx == k_sample - 1) else torch.no_grad()
            with grad_context:
                pred_x0_student, _, _ = self._predict_x0_at_step(
                    self.generator,
                    x_t_student,
                    gen_exit_step,
                    context,
                    seq_len,
                    clip_fea,
                    cond_latents,
                    audio_emb,
                    ref_target_masks=ref_target_masks,
                )
                m = int(motion_latent_num)
                prev_chunk_pred = torch.cat(
                    [motion_source[:, :, :m], pred_x0_student[:, :, m:].detach()],
                    dim=2,
                )

                # Only the last chunk contributes the DMD objective.
                if not (chunk_idx == k_sample - 1):
                    continue

            with torch.no_grad():
                pred_x0_teacher_input = pred_x0_student
                teacher_t_raw = self._sample_teacher_raw_from_student_interval(gen_exit_idx)
                teacher_t_shifted = self._apply_timestep_shift(teacher_t_raw)
                t_teacher = torch.full((bs,), teacher_t_shifted, device=self.device, dtype=torch.long)
                t_teacher = t_teacher.clamp(self.min_step, self.max_step)
                t_teacher_norm = t_teacher.float().view(bs, 1, 1, 1, 1) / 1000.0
                noise_dmd = torch.randn_like(pred_x0_teacher_input)
                x_t_fake = (1 - t_teacher_norm) * pred_x0_teacher_input + t_teacher_norm * noise_dmd
                x_t_teacher = self._inject_motion_source_with_noise(
                    x_t_fake,
                    motion_source_latent=motion_source,
                    motion_latent_num=motion_latent_num,
                    t_norm=t_teacher_norm,
                    add_noise=False,
                )
                out_teacher_cond = self.forward_wan(
                    self.real_score,
                    x_t_teacher,
                    t_teacher,
                    context,
                    seq_len,
                    clip_fea,
                    cond_latents,
                    audio_emb,
                    ref_target_masks,
                )
                teacher_v_cond = torch.stack(out_teacher_cond) if isinstance(out_teacher_cond, list) else out_teacher_cond
                pred_x0_real_cond = self.get_x0_from_v(x_t_teacher, teacher_v_cond, t_teacher_norm)

                if context_null is not None and audio_null is not None:
                    # pass 2: drop text (keep audio + image)
                    out_drop_text = self.forward_wan(
                        self.real_score, x_t_teacher, t_teacher, context_null, seq_len,
                        clip_fea, cond_latents, audio_emb, ref_target_masks,
                    )
                    v_drop_text = torch.stack(out_drop_text) if isinstance(out_drop_text, list) else out_drop_text
                    pred_x0_drop_text = self.get_x0_from_v(x_t_teacher, v_drop_text, t_teacher_norm)

                    # pass 3: drop text + audio (keep image)
                    out_null_all = self.forward_wan(
                        self.real_score, x_t_teacher, t_teacher, context_null, seq_len,
                        clip_fea, cond_latents, audio_null, ref_target_masks,
                    )
                    v_null_all = torch.stack(out_null_all) if isinstance(out_null_all, list) else out_null_all
                    pred_x0_null_all = self.get_x0_from_v(x_t_teacher, v_null_all, t_teacher_norm)

                    # Dual CFG: null + audio_scale*(drop_text - null) + text_scale*(cond - drop_text)
                    pred_x0_real = (
                        pred_x0_null_all
                        + self.audio_guide_scale * (pred_x0_drop_text - pred_x0_null_all)
                        + self.text_guide_scale * (pred_x0_real_cond - pred_x0_drop_text)
                    )
                else:
                    pred_x0_real = pred_x0_real_cond

                x_t_critic = self._inject_motion_source_with_noise(
                    x_t_fake,
                    motion_source_latent=motion_source,
                    motion_latent_num=motion_latent_num,
                    t_norm=t_teacher_norm,
                    add_noise=False,
                )
                out_critic = self.forward_wan(
                    self.fake_score,
                    x_t_critic,
                    t_teacher,
                    context,
                    seq_len,
                    clip_fea,
                    cond_latents,
                    audio_emb,
                    ref_target_masks,
                )
                critic_v = torch.stack(out_critic) if isinstance(out_critic, list) else out_critic
                pred_x0_fake = self.get_x0_from_v(x_t_critic, critic_v, t_teacher_norm)

                m = motion_latent_num
                grad = pred_x0_fake[:, :, m:, :, :] - pred_x0_real[:, :, m:, :, :]
                p_real = pred_x0_student[:, :, m:, :, :] - pred_x0_real[:, :, m:, :, :]
                normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
                grad_normalized = torch.nan_to_num(
                    grad / normalizer, nan=0.0, posinf=0.0, neginf=0.0
                )

                # Motion-position alignment to teacher with DMD-style per-sample
                # normalization so its per-element gradient stays at O(1).
                motion_diff = (
                    pred_x0_student[:, :, :m, :, :] - pred_x0_real[:, :, :m, :, :]
                )
                motion_normalizer = torch.abs(motion_diff).mean(
                    dim=[1, 2, 3, 4], keepdim=True
                ).clamp(min=1e-6)
                motion_grad_normalized = torch.nan_to_num(
                    motion_diff / motion_normalizer, nan=0.0, posinf=0.0, neginf=0.0
                )

            target = (pred_x0_student[:, :, m:, :, :] - grad_normalized).detach()
            target_no_norm = (pred_x0_student[:, :, m:, :, :] - grad).detach()
            motion_target = (pred_x0_student[:, :, :m, :, :] - motion_grad_normalized).detach()
            motion_target_no_norm = (pred_x0_student[:, :, :m, :, :] - motion_diff).detach()

            dmd_loss = 0.5 * F.mse_loss(pred_x0_student[:, :, m:, :, :].float(), target.float())
            motion_align_loss = 0.5 * F.mse_loss(pred_x0_student[:, :, :m, :, :].float(), motion_target.float())

            motion_weight = float(m) / float(pred_x0_student.shape[2] - m)
            rvm_bg_align_loss = self._compute_rvm_background_overlap_mse(
                pred_x0_student=pred_x0_student,
                first_frame_latent=first_frame_latent,
                motion_source_latent=motion_source,
                motion_latent_num=m,
            )

            loss = (
                self.dmd_loss_weight * dmd_loss
                + self.temporal_align_weight * rvm_bg_align_loss
                + motion_weight * motion_align_loss
            )
            self.latest_gen_loss_components = {
                "dmd_loss_term": float((self.dmd_loss_weight * dmd_loss).detach().item()),
                "motion_align_loss_term": float((motion_weight * motion_align_loss).detach().item()),
                "rvm_bg_align_loss_term": float(
                    (self.temporal_align_weight * rvm_bg_align_loss).detach().item()
                ),
            }

            # Cat'd full-length targets only for visualization parity.
            target = torch.cat([motion_target, target], dim=2)
            target_no_norm = torch.cat([motion_target_no_norm, target_no_norm], dim=2)

            if return_visual_dict:
                full_target = target.detach().clone()
                full_target_no_norm = target_no_norm.detach().clone()
                final_visual_dict = {
                    "pred_x0_teacher_input": pred_x0_teacher_input.detach(),
                    "pred_x0_student": pred_x0_student.detach(),
                    "pred_x0_real": pred_x0_real.detach(),
                    "pred_x0_fake": pred_x0_fake.detach(),
                    "target_no_norm": full_target_no_norm,
                    "target": full_target,
                    "clean_latents": None,
                    "average_target": None,
                    "k": int(k_sample),
                    "self_forcing_k": int(k_sample),
                    "gen_exit_idx": int(gen_exit_idx),
                    "teacher_t_raw": int(teacher_t_raw),
                }

        if loss is None:
            raise RuntimeError("Self-Forcing++ generator loss failed to produce loss.")
        if return_visual_dict:
            return loss, final_visual_dict
        return loss

    def _critic_loss_self_forcing_plus_plus(self, self_forcing_chunks):
        """
        Self-Forcing++ critic objective:
        - Simulate all chunks under no_grad.
        - Train the critic to denoise the student's last-chunk output at a
          teacher-interval timestep.
        """
        if self_forcing_chunks is None or len(self_forcing_chunks) == 0:
            raise ValueError("self_forcing_chunks must be a non-empty list.")

        if self.keep_k_chunks > 0:
            k_sample = self.keep_k_chunks
        else:
            k_sample = int(len(self_forcing_chunks))
        prev_chunk_pred = None
        final_chunk = None
        final_pred_x0_student = None
        final_motion_source = None
        final_motion_latent_num = None
        final_exit_idx = None

        with torch.no_grad():
            for chunk_idx in range(k_sample):
                chunk = self_forcing_chunks[chunk_idx]
                context = chunk["context"]
                clip_fea = chunk["clip_fea"]
                cond_latents = chunk["cond_latents"]
                audio_emb = chunk["audio_emb"]
                seq_len = chunk["seq_len"]
                ref_target_masks = chunk.get("ref_target_masks")
                first_frame_latent = chunk["first_frame_latent"]

                bs = int(context.shape[0])
                lat_c = int(first_frame_latent.shape[1])
                lat_t = int(cond_latents.shape[2])
                lat_h = int(cond_latents.shape[3])
                lat_w = int(cond_latents.shape[4])
                device = first_frame_latent.device
                dtype = first_frame_latent.dtype

                if chunk_idx == 0:
                    motion_source = first_frame_latent
                    motion_latent_num = 1
                else:
                    motion_source, motion_latent_num = self._build_motion_source_from_prev_chunk_pixels(
                        prev_chunk_pred,
                        motion_frame_num=5,
                    )

                start_noise = torch.randn(bs, lat_c, lat_t, lat_h, lat_w, device=device, dtype=dtype)
                if chunk_idx == (k_sample - 1):
                    exit_idx = self._sample_synced_exit_index()
                else:
                    exit_idx = (len(self.denoising_step_list) - 1)

                x_t_student = self._simulate_to_exit_step_inject_motion_frames_with_noise(
                    start_noise,
                    exit_idx,
                    context,
                    seq_len,
                    clip_fea,
                    cond_latents,
                    audio_emb,
                    ref_target_masks=ref_target_masks,
                    clean_latents=motion_source,
                    inject_motion_frames_num=motion_latent_num,
                    add_motion_noise=False,
                )

                exit_step = self.denoising_step_list[exit_idx]
                t_student = torch.full((bs,), int(exit_step), device=start_noise.device, dtype=torch.long)
                t_student_norm = t_student.float().view(bs, 1, 1, 1, 1) / 1000.0
                x_t_student = self._inject_motion_source_with_noise(
                    x_t_student,
                    motion_source_latent=motion_source,
                    motion_latent_num=motion_latent_num,
                    t_norm=t_student_norm,
                    add_noise=False,
                )

                pred_x0_student, _, _ = self._predict_x0_at_step(
                    self.generator,
                    x_t_student,
                    exit_step,
                    context,
                    seq_len,
                    clip_fea,
                    cond_latents,
                    audio_emb,
                    ref_target_masks=ref_target_masks,
                )
                prev_chunk_pred = pred_x0_student.detach()

                if chunk_idx == (k_sample - 1):
                    final_chunk = chunk
                    final_pred_x0_student = prev_chunk_pred
                    final_motion_source = motion_source.detach()
                    final_motion_latent_num = int(motion_latent_num)
                    final_exit_idx = int(exit_idx)

        if final_chunk is None or final_pred_x0_student is None or final_exit_idx is None:
            raise RuntimeError("Self-Forcing++ critic loss failed to produce final chunk prediction.")

        context = final_chunk["context"]
        clip_fea = final_chunk["clip_fea"]
        cond_latents = final_chunk["cond_latents"]
        audio_emb = final_chunk["audio_emb"]
        seq_len = final_chunk["seq_len"]
        ref_target_masks = final_chunk.get("ref_target_masks")

        bs = final_pred_x0_student.shape[0]
        noise_critic = torch.randn_like(final_pred_x0_student)
        critic_t_raw = self._sample_teacher_raw_from_student_interval(final_exit_idx)
        critic_t_shifted = self._apply_timestep_shift(critic_t_raw)
        t_critic = torch.full((bs,), critic_t_shifted, device=self.device, dtype=torch.long)
        t_critic = t_critic.clamp(self.min_step, self.max_step)
        t_critic_norm = t_critic.float().view(bs, 1, 1, 1, 1) / 1000.0
        x_t_critic = (1 - t_critic_norm) * final_pred_x0_student + t_critic_norm * noise_critic

        x_t_critic = self._inject_motion_source_with_noise(
            x_t_critic,
            motion_source_latent=final_motion_source,
            motion_latent_num=final_motion_latent_num,
            t_norm=t_critic_norm,
            add_noise=False,
        )

        out_critic = self.forward_wan(
            self.fake_score,
            x_t_critic,
            t_critic,
            context,
            seq_len,
            clip_fea,
            cond_latents,
            audio_emb,
            ref_target_masks,
        )
        critic_v = torch.stack(out_critic) if isinstance(out_critic, list) else out_critic
        target_v = (noise_critic - final_pred_x0_student).detach()

        loss = 0.5 * F.mse_loss(
            critic_v[:, :, final_motion_latent_num:, :, :].float(),
            target_v[:, :, final_motion_latent_num:, :, :].float(),
        )
        return loss
