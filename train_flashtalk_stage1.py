import argparse
import gc
import itertools
import json
import logging
import os
import random
import re
from datetime import datetime

import src._warning_filters
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from flashtalk_dmd import FlashTalkDMD
from src.data_processor_flashtalk import (
    DataProcessor,
    FlashTalkDataset,
    LmdbBatchReader,
    move_VAE_to_device,
    preprocess_batches_to_payload_files,
)
from wan.utils.utils import (
    barrier,
    cache_video,
    fsdp_state_dict,
    launch_distributed_job,
    set_seed,
    str2bool,
)

# ===================== Args Initialization =====================
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="weights/Wan2.1-I2V-14B-480P", help="Path to Wan2.1 Checkpoint")
parser.add_argument("--vae_checkpoint", type=str, default="Wan2.1_VAE.pth", help="VAE filename")
parser.add_argument("--output_dir", type=str, default="outputs/flashtalk_stage1")
parser.add_argument("--auto_output_dir_name", type=str2bool, default=True,
                   help="Auto append experiment-name slug to output_dir based on key hyper-parameters")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps.")
parser.add_argument("--gen_lr", type=float, default=1e-5, help="Learning rate for stage-1 flow-matching generator.")
parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Max gradient norm for clipping generator")
parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
parser.add_argument("--dist_timeout_minutes", type=int, default=60, help="Distributed process group timeout in minutes.")
parser.add_argument("--config", type=str, default=None, help="Path to stage config yaml. Config values override CLI args.")
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--save_interval", type=int, default=100, help="Interval to save checkpoints.")
parser.add_argument("--resume_replay_sampling", type=str2bool, default=True,
                   help="When resuming, replay random sampling path for finished steps to align RNG with uninterrupted run")
parser.add_argument("--save_generator_latent_videos", type=str2bool, default=True,
                   help="Save stage-1 intermediate latents as decoded mp4 videos after optimizer step")
parser.add_argument("--save_generator_latent_videos_interval", type=int, default=50,
                   help="Interval to save stage-1 intermediate latents as decoded mp4 videos")
parser.add_argument("--warmup_steps", type=int, default=50, help="Linear warmup steps for generator learning rate")
parser.add_argument("--warmup_start_lr", type=float, default=2e-7, help="Warmup start LR for generator")
parser.add_argument("--gradient_checkpointing", type=str2bool, default=True, help="Enable gradient checkpointing to save memory")
parser.add_argument("--text_guide_scale", type=float, default=5.0, help="CFG scale for text guidance")
parser.add_argument("--audio_guide_scale", type=float, default=4.0, help="CFG scale for audio guidance")
parser.add_argument("--use_fixed_reference_frame", type=str2bool, default=False,
                   help="If True, use selected window first frame as reference; otherwise sample one from +/-33 frames around the window")
parser.add_argument("--flow_match_weight", type=float, default=1.0, help="Weight for flow matching velocity loss")
parser.add_argument("--face_loss_weight", type=float, default=1.0, help="Weight for face-region reconstruction loss")
parser.add_argument("--temporal_loss_weight", type=float, default=0.0, help="Weight for temporal latent consistency loss")
parser.add_argument("--face_loss_type", type=str, default="l1", choices=["l1", "l2"], help="Face loss type")
parser.add_argument("--face_det_score_thresh", type=float, default=0.97, help="RetinaFace confidence threshold")
parser.add_argument("--face_det_device", type=str, default="cuda", help="Face detector device (e.g. cuda/cpu)")
parser.add_argument("--enable_background_filter", type=str2bool, default=False,
                   help="Enable RVM+SSIM background-stability filtering during preprocess. "
                        "Only effective when mode=preprocess; ignored otherwise.")
parser.add_argument("--background_ssim_threshold", type=float, default=0.96,
                   help="Minimum background SSIM threshold; samples below this are dropped. "
                        "Only effective when enable_background_filter=true.")
parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.99],
                   metavar=("BETA1", "BETA2"), help="AdamW betas; YAML may use [0.9, 0.999] or '(0.9, 0.999)' string.")

# Extra Args
parser.add_argument("--seed", type=int, default=2026, help="Random seed for training")
parser.add_argument("--wav2vec_dir", type=str, default="weights/chinese-wav2vec2-base")
parser.add_argument("--infinitetalk_dir", type=str, default="weights/InfiniteTalk/single/infinitetalk.safetensors")
parser.add_argument("--size", type=str, default="infinitetalk-480")
parser.add_argument("--training_window_size", type=int, default=33, help="Training window size (frames)")
parser.add_argument("--debug", type=str2bool, default=False, help="Debug mode: Define model structure without loading weights")
parser.add_argument("--timestep_shift", type=float, default=8.0, help="Timestep shift factor for noise scheduling")
parser.add_argument("--n_prompt", type=str,
                   default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                   help="Negative prompt for CFG (aligned with InfiniteTalk inference)")
parser.add_argument("--use_precomputed_audio", type=str2bool, default=False,
                   help="Use precomputed audio embeddings (.pt files) instead of computing from .wav files")
parser.add_argument("--mode", type=str, default="train", choices=["preprocess", "train"],
                   help="Run mode: preprocess=extract per-sample features (LMDB payloads for the train split "
                        "or per-sample feature directories for the val split); train=load samples from a packed LMDB")
parser.add_argument("--preprocess_split", type=str, default="train", choices=["train", "val"],
                   help="When mode=preprocess: choose between extracting LMDB-bound train payloads "
                        "(uses --annotation_file, writes to --payload_dir) and extracting per-sample "
                        "validation feature directories (uses --val_annotation_file, writes to --val_features_dir, "
                        "plus context_null.pt directly inside --val_features_dir). The val split does NOT touch LMDB.")
parser.add_argument("--lmdb_path", type=str, default="",
                   help="LMDB file path for preprocessed batches")
parser.add_argument("--lmdb_num_samples", type=int, default=35000,
                   help="Number of samples to preprocess into LMDB (global, keys start from 0)")
parser.add_argument("--payload_dir", type=str, default=None,
                   help="Stage-A payload directory for preprocess mode. Default: <lmdb_path without ext>.payloads")

# TalkCuts dataset paths (only required when mode=preprocess).
parser.add_argument("--dataset_dir", type=str, default="",
                   help="Root directory that hosts the raw video/audio files referenced by annotation csvs.")
parser.add_argument("--annotation_file", type=str, default="processed_data/example/train_data.csv",
                   help="CSV file with columns video,input_audio,prompt (used when preprocess_split=train).")
parser.add_argument("--val_annotation_file", type=str, default="processed_data/talkcuts/val_data.csv",
                   help="CSV file with columns video,input_audio,prompt (used when preprocess_split=val).")
parser.add_argument("--val_features_dir", type=str, default="processed_data/talkcuts/val/feature",
                   help="Directory of per-sample validation features (context.pt/full_emb.pt/clip_fea.pt/...). "
                        "When preprocess_split=val this is the OUTPUT directory; val features are written to "
                        "<val_features_dir>/<sample_id>/ and a single context_null.pt is written *inside* "
                        "<val_features_dir>/ (next to the sample subdirs). When loaded by val_stage{1,2}.yaml "
                        "this is the INPUT directory.")

# FSDP Args
parser.add_argument("--use_fsdp", type=str2bool, default=True, help="Use FSDP for training")
parser.add_argument("--mixed_precision", type=str2bool, default=True, help="Enable FSDP mixed precision")
parser.add_argument("--sharding_strategy", type=str, default="full", choices=["full", "hybrid_full", "hybrid_zero2", "no_shard"], help="FSDP sharding strategy")
parser.add_argument("--fsdp_cpu_offload", type=str2bool, default=False, help="Enable FSDP CPU offload for trainable model params")
parser.add_argument("--fsdp_wrap_strategy", type=str, default="size", choices=["size", "transformer"], help="FSDP auto wrap strategy")


def _load_stage_config(config_path):
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a key-value yaml mapping: {config_path}")
    return cfg


def _normalize_betas(v):
    """AdamW betas: tuple/list, or YAML string '(0.9, 0.999)' / '0.9, 0.999'."""
    if v is None:
        return (0.9, 0.99)
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"betas must have exactly 2 values, got {v!r}")
        return (float(v[0]), float(v[1]))
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"betas string must be two comma-separated floats, got {v!r}")
        return (float(parts[0]), float(parts[1]))
    raise TypeError(f"Unsupported betas type: {type(v).__name__}")


def _merge_config_over_cli(args, config_path):
    if not config_path:
        return args, {}, set()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_stage_config(config_path)
    overrides = {}
    for k, v in cfg.items():
        if not hasattr(args, k):
            continue
        old_v = getattr(args, k)
        if old_v != v:
            overrides[k] = {"from": old_v, "to": v}
        setattr(args, k, v)
    return args, overrides, set(cfg.keys())


def _format_scientific_short(v):
    s = f"{float(v):.0e}".replace("+", "")
    s = re.sub(r"e-0*(\d+)$", r"e-\1", s)
    s = re.sub(r"e0*(\d+)$", r"e\1", s)
    return s


def _build_experiment_name(args):
    parts = [
        "talkcuts",
        "flashtalk_stage1",
        f"window_{args.training_window_size}",
        f"gen_lr_{_format_scientific_short(args.gen_lr)}",
        f"face_w_{args.face_loss_weight:g}",
        f"temp_w_{args.temporal_loss_weight:g}",
        "full_finetune",
    ]
    return "_".join(parts)


def _compute_face_loss(pred, gt, face_masks, loss_type="l1"):
    # pred/gt: [B, C, T, H, W], face_masks: [B, 1, T, H, W]
    mask = face_masks
    if mask.shape[0] != pred.shape[0] or mask.shape[2] != pred.shape[2]:
        raise ValueError(
            "Face mask temporal shape mismatch. "
            f"mask={tuple(mask.shape)} vs pred={tuple(pred.shape)}"
        )
    if mask.shape[-2:] != pred.shape[-2:]:
        raise ValueError(
            "Face mask spatial shape mismatch. "
            f"mask_hw={tuple(mask.shape[-2:])}, pred_hw={tuple(pred.shape[-2:])}"
        )

    mask = mask.expand(-1, pred.shape[1], -1, -1, -1)
    denom = mask.sum().clamp(min=1.0)

    if loss_type == "l2":
        diff = (pred - gt).pow(2)
    else:
        diff = (pred - gt).abs()
    return (diff * mask).sum() / denom


def _save_stage1_latent_videos(processor, visual_dict, output_dir, step, rank, fps=25):
    """Decode intermediate stage-1 latents and save mp4 videos per sample."""
    tensor_names = [
        "pred_x0",
        "clean_latents",
        "x_t",
    ]
    save_root = os.path.join(output_dir, f"iter_{step}")
    os.makedirs(save_root, exist_ok=True)

    with torch.no_grad():
        bs = visual_dict["clean_latents"].shape[0]
        for name in tensor_names:
            tensor = visual_dict.get(name)
            if tensor is None:
                continue

            latents_list = [tensor[idx] for idx in range(bs)]
            decoded_videos = processor.vae.decode(latents_list)
            for idx, video in enumerate(decoded_videos):
                folder_name = f"rank_{rank}"
                if bs > 1:
                    folder_name += f"_sample_{idx + 1}"

                sample_dir = os.path.join(save_root, folder_name)
                os.makedirs(sample_dir, exist_ok=True)
                t_suffix = ""
                t_norm = visual_dict.get("t_norm")
                if t_norm is not None:
                    t_norm_val = float(t_norm[idx].reshape(-1)[0].item()) if t_norm.shape[0] > idx else float(t_norm.reshape(-1)[0].item())
                    t_suffix = f"_tnorm_{t_norm_val:.4f}"
                else:
                    t_raw = visual_dict.get("t")
                    if t_raw is not None:
                        t_val = float(t_raw[idx].reshape(-1)[0].item()) if t_raw.shape[0] > idx else float(t_raw.reshape(-1)[0].item())
                        t_suffix = f"_t_{t_val:.4f}"

                save_path = os.path.join(sample_dir, f"{name}{t_suffix}.mp4")
                cache_video(
                    tensor=video.unsqueeze(0),
                    save_file=save_path,
                    fps=fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )


def _build_gen_lr_scheduler(optimizer, warmup_steps, warmup_start_lr, base_lr):
    if warmup_steps <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    safe_start_lr = max(float(warmup_start_lr), 1e-7)

    def _lr_lambda(current_step):
        if current_step < warmup_steps:
            alpha = float(current_step) / float(warmup_steps - 1)
            target_lr = safe_start_lr + alpha * (base_lr - safe_start_lr)
            return float(target_lr / base_lr)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def main():
    args = parser.parse_args()
    args, config_overrides, _ = _merge_config_over_cli(args, args.config)
    args.betas = _normalize_betas(args.betas)
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    args.mode = str(args.mode).lower()
    if args.batch_size != 1:
        raise ValueError("preprocess/train mode currently requires batch_size=1")
    args.preprocess_split = str(getattr(args, "preprocess_split", "train")).lower()
    if args.mode == "preprocess":
        if args.preprocess_split == "train":
            if not args.payload_dir:
                raise ValueError("payload_dir is required when mode=preprocess and preprocess_split=train")
            if args.lmdb_num_samples <= 0:
                raise ValueError("lmdb_num_samples must be > 0 when mode=preprocess and preprocess_split=train")
        elif args.preprocess_split == "val":
            if not args.val_annotation_file:
                raise ValueError("val_annotation_file is required when preprocess_split=val")
            if not args.val_features_dir:
                raise ValueError("val_features_dir is required when preprocess_split=val (output directory of per-sample val features)")
    elif args.mode == "train":
        if not args.lmdb_path:
            raise ValueError("lmdb_path is required when mode=train")

    if args.auto_output_dir_name:
        args.output_dir = os.path.join(args.output_dir, _build_experiment_name(args))
    base_output_dir = args.output_dir

    launch_distributed_job(timeout_minutes=args.dist_timeout_minutes)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # Use one shared timestamp across all ranks to avoid split output folders.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
    ts_list = [run_timestamp]
    dist.broadcast_object_list(ts_list, src=0)
    args.output_dir = os.path.join(base_output_dir, ts_list[0])

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

        log_file = os.path.join(args.output_dir, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
        if args.config:
            logging.info("Loaded stage config: %s", args.config)
            if config_overrides:
                logging.info("Config overrides (config > CLI):")
                for key in sorted(config_overrides.keys()):
                    old_v = config_overrides[key]["from"]
                    new_v = config_overrides[key]["to"]
                    logging.info("  %s: %s -> %s", key, old_v, new_v)
            else:
                logging.info("Config had no effective overrides.")
        logging.info("========== Training args ==========")
        for k, v in sorted(vars(args).items()):
            logging.info("  %s: %s", k, v)
        logging.info("===================================")
    else:
        logging.basicConfig(level=logging.ERROR)

    # Load WanModel config from checkpoint directory if available.
    config_path = os.path.join(args.ckpt_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            wan_config_dict = json.load(f)
        class Config: pass
        config = Config()
        for k, v in vars(args).items():
            setattr(config, k, v)
        for k, v in wan_config_dict.items():
            setattr(config, k, v)
    else:
        config = args

    # Background filtering only runs during preprocess; force-disable it in train
    # mode regardless of the user-provided flag, to avoid loading RVM/face models.
    if args.mode != "preprocess":
        config.enable_background_filter = False
    else:
        config.enable_background_filter = bool(args.enable_background_filter)

    if hasattr(args, "seed"):
        set_seed(args.seed + rank)

    processor_init_device = device if args.mode == "preprocess" else ('cpu' if args.use_fsdp else device)

    processor = DataProcessor(
        config,
        args.ckpt_dir,
        processor_init_device,
        n_prompt=args.n_prompt,
        use_precomputed_audio=args.use_precomputed_audio,
        processed_data_dir=None,
        use_fixed_reference_frame=args.use_fixed_reference_frame,
        rank=rank,
        mode=args.mode,
    )

    processor.device = device
    processor.vae.mean = processor.vae.mean.to(device)
    processor.vae.std = processor.vae.std.to(device)
    processor.vae.scale = [processor.vae.mean, 1.0 / processor.vae.std]

    if args.mode == "preprocess" and args.preprocess_split == "val":
        # Validation feature extraction: produce per-sample feature directories
        # under <val_features_dir>/<sample_id>/ and a single context_null.pt
        # inside <val_features_dir>/ (next to the sample subdirs). No LMDB /
        # payload pipeline is involved.
        from src.data_processor_flashtalk import preprocess_validation_features
        os.makedirs(args.val_features_dir, exist_ok=True)
        written, skipped, failed = preprocess_validation_features(
            processor=processor,
            annotation_file=args.val_annotation_file,
            dataset_dir=args.dataset_dir,
            output_dir=args.val_features_dir,
            rank=rank,
            world_size=world_size,
        )
        logging.info(
            "Val features done on rank=%d, written=%d, skipped=%d, failed=%d, output_dir=%s",
            rank, written, skipped, failed, args.val_features_dir,
        )
        if rank == 0:
            logging.info(
                "Stage1 val feature extraction complete: output_dir=%s. "
                "Use these features via config/val_stage{1,2}.yaml -> val_features_dir.",
                args.val_features_dir,
            )
        dist.destroy_process_group()
        return

    if args.mode == "preprocess":
        # Only preprocess mode walks raw video/audio via a DataLoader.
        dataset = FlashTalkDataset(args.dataset_dir, annotation_file=args.annotation_file)
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda x: x)
        dataloader = itertools.cycle(dataloader)

        payload_dir = args.payload_dir
        written, skipped, filtered = preprocess_batches_to_payload_files(
            processor=processor,
            dataloader=dataloader,
            payload_dir=payload_dir,
            num_samples=int(args.lmdb_num_samples),
            rank=rank,
            world_size=world_size,
            model_dtype=torch.bfloat16,
        )
        logging.info(
            "Payload preprocess done on rank=%d, written=%d, skipped=%d, filtered=%d, payload_dir=%s",
            rank,
            written,
            skipped,
            filtered,
            payload_dir,
        )
        barrier()
        if rank == 0:
            logging.info(
                "Stage-A payload preprocess complete: payload_dir=%s, total_samples=%d. "
                "Run tools/payload_files_to_lmdb.py for Stage-B LMDB packing.",
                payload_dir,
                args.lmdb_num_samples,
            )
        dist.destroy_process_group()
        return

    # Stage-1 only needs the generator (full fine-tune flow matching).
    dmd_model = FlashTalkDMD(
        config, device, args.ckpt_dir, args.infinitetalk_dir,
        stage="stage1", debug=args.debug,
    )
    dmd_model.setup_for_training(args)
    dmd_model.register_debug_io(processor=processor, output_dir=args.output_dir)

    gen_params = [p for p in dmd_model.generator.parameters() if p.requires_grad]
    logging.info(f"Trainable params: {len(gen_params)}, total elements: {sum(p.numel() for p in gen_params):,}")

    optimizer_gen = torch.optim.AdamW(gen_params, lr=args.gen_lr, betas=args.betas, weight_decay=args.weight_decay)
    scheduler_gen = _build_gen_lr_scheduler(
        optimizer=optimizer_gen,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        base_lr=args.gen_lr,
    )
    if args.warmup_steps > 0:
        for pg in optimizer_gen.param_groups:
            pg["lr"] = max(float(args.warmup_start_lr), 1e-7)
    logging.info("Gradient clipping max_norm: %.2f", args.max_grad_norm)
    if rank == 0:
        logging.info(
            "Gradient accumulation: grad_accum_steps=%d (effective batch=%d), warmup_steps=%d, warmup_start_lr=%.2e, base_lr=%.2e",
            args.grad_accum_steps,
            args.batch_size * args.grad_accum_steps * world_size,
            args.warmup_steps,
            args.warmup_start_lr,
            args.gen_lr,
        )

    step = 0

    writer = None
    if rank == 0:
        tb_dir = os.path.join(args.output_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            raise ValueError(f"Resume checkpoint directory not found: {args.resume_from}")

        logging.info(f"Resuming training from {args.resume_from}")
        training_state_path = os.path.join(args.resume_from, "training_state.pt")

        if not os.path.exists(training_state_path):
            raise ValueError(f"Training state file not found: {training_state_path}")

        meta = torch.load(training_state_path, map_location='cpu')
        step = meta['step']

        # Each rank saved its own FSDP shard.
        optim_state_path = os.path.join(args.resume_from, f"optim_state_rank{rank}.pt")
        if os.path.exists(optim_state_path):
            optim_ckpt = torch.load(optim_state_path, map_location='cpu')
            optimizer_gen.load_state_dict(optim_ckpt['optimizer_gen'])
            scheduler_gen.load_state_dict(optim_ckpt['scheduler_gen'])
            logging.info(f"Loaded per-rank optimizer state from {optim_state_path}")
            del optim_ckpt
        else:
            if 'optimizer_gen' in meta:
                optimizer_gen.load_state_dict(meta['optimizer_gen'])
                scheduler_gen.load_state_dict(meta['scheduler_gen'])
                logging.info("Loaded optimizer state from legacy training_state.pt (single-rank)")
            else:
                logging.warning("No optimizer state found for rank %d, starting with fresh optimizer.", rank)

        logging.info(f"Resumed training from step {step}")
        logging.info(
            "Generator LR (runtime): %.2e, Max grad norm: %.2f",
            float(optimizer_gen.param_groups[0]["lr"]),
            args.max_grad_norm,
        )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    model_dtype = next(dmd_model.parameters()).dtype
    lmdb_reader = LmdbBatchReader(args.lmdb_path)
    if rank == 0:
        logging.info(
            "LMDB train mode enabled: path=%s, num_samples=%d",
            args.lmdb_path,
            lmdb_reader.num_samples,
        )

    def _prepare_batch_tensors_from_lmdb(step_idx, accum_idx):
        lmdb_key = (step_idx - 1) * world_size * args.grad_accum_steps + accum_idx * world_size + rank
        if lmdb_key >= lmdb_reader.num_samples:
            raise RuntimeError(
                f"LMDB key out of range: key={lmdb_key}, num_samples={lmdb_reader.num_samples}. "
                "Please increase lmdb_num_samples during preprocess."
            )
        return lmdb_reader.get(lmdb_key, device=device, model_dtype=model_dtype)

    def _replay_sampling_history(target_step):
        """Replay pre-resume random draws so next-step RNG matches uninterrupted training."""
        if target_step <= 0:
            return
        if rank == 0:
            logging.info(
                "Resume RNG replay start: replaying %d finished steps x %d accum",
                target_step,
                args.grad_accum_steps,
            )

        with torch.no_grad():
            for replay_step in range(1, target_step + 1):
                for accum_idx in range(args.grad_accum_steps):
                    bt = _prepare_batch_tensors_from_lmdb(replay_step, accum_idx)
                    clean_latents = bt["clean_latents"]
                    dmd_model.get_noise_and_timestep(clean_latents, generator=False, shift=True)
                    random.randint(1, 2)
                    torch.rand((), device=clean_latents.device).item()

                if rank == 0 and (replay_step % 100 == 0 or replay_step == target_step):
                    logging.info("Resume RNG replay progress: %d/%d steps", replay_step, target_step)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            logging.info("Resume RNG replay finished.")

    dmd_model.eval()
    if args.resume_from is not None and args.resume_replay_sampling and step > 0:
        _replay_sampling_history(step)
    while step < args.max_steps:
        step += 1
        current_lr = float(optimizer_gen.param_groups[0]["lr"])
        stage1_visual_dict = None

        optimizer_gen.zero_grad(set_to_none=True)
        flow_loss_sum = 0.0
        face_loss_sum = 0.0
        temporal_loss_sum = 0.0
        total_loss_sum = 0.0

        for accum_idx in range(args.grad_accum_steps):
            bt = _prepare_batch_tensors_from_lmdb(step, accum_idx)

            clean_latents = bt["clean_latents"]
            context = bt["context"]
            clip_fea = bt["clip_fea"]
            cond_latents = bt["cond_latents"]
            audio_emb = bt["audio_emb"]
            seq_len = bt["seq_len"]
            ref_target_masks = bt["ref_target_masks"]
            face_masks = bt["face_masks"]
            context_null = bt["context_null"]
            audio_null = bt["audio_null"]

            noise, t, t_norm = dmd_model.get_noise_and_timestep(clean_latents, generator=False, shift=True)
            x_t = (1.0 - t_norm) * clean_latents + t_norm * noise
            # Inject 1-2 clean motion latents to simulate inference-time conditioning.
            motion_latent_num = random.randint(1, 2)
            x_t[:, :, :motion_latent_num, :, :] = clean_latents[:, :, :motion_latent_num, :, :]

            target_v = noise - clean_latents
            # Do not align loss on injected motion latents.
            loss_start_idx = motion_latent_num

            cond_rand = torch.rand((), device=clean_latents.device).item()
            if cond_rand < 0.05:
                # 5%: drop text + audio (null baseline)
                context_for_wan = context_null
                audio_for_wan = audio_null
            elif cond_rand < 0.10:
                # 5%: drop text only, keep audio + image
                context_for_wan = context_null
                audio_for_wan = audio_emb
            else:
                # 90%: keep all conditions
                context_for_wan = context
                audio_for_wan = audio_emb

            out = dmd_model.forward_wan(
                dmd_model.generator,
                x_t,
                t,
                context_for_wan,
                seq_len,
                clip_fea,
                cond_latents,
                audio_for_wan,
                ref_target_masks=ref_target_masks,
            )

            pred_v = torch.stack(out) if isinstance(out, list) else out
            flow_loss = F.mse_loss(
                pred_v[:, :, loss_start_idx:, :, :].float(),
                target_v[:, :, loss_start_idx:, :, :].float(),
            )
            pred_x0 = dmd_model.get_x0_from_v(x_t, pred_v, t_norm)

            want_visual = (
                args.save_generator_latent_videos
                and step % args.save_generator_latent_videos_interval == 0
                and accum_idx == 0
            )
            if want_visual:
                stage1_visual_dict = {
                    "pred_x0": pred_x0.detach(),
                    "clean_latents": clean_latents.detach(),
                    "x_t": x_t.detach(),
                    "t": t.detach(),
                    "t_norm": t_norm.detach(),
                }

            pred_delta = pred_v[:, :, 1:, :, :] - pred_v[:, :, :-1, :, :]
            gt_delta = target_v[:, :, 1:, :, :] - target_v[:, :, :-1, :, :]
            temporal_start_idx = motion_latent_num

            temporal_loss = F.mse_loss(
                pred_delta[:, :, temporal_start_idx:, :, :].float(),
                gt_delta[:, :, temporal_start_idx:, :, :].float(),
            )

            # face_masks are pre-aligned to latent shape in DataProcessor.
            face_loss = _compute_face_loss(
                pred_v[:, :, loss_start_idx:, :, :].float(),
                target_v[:, :, loss_start_idx:, :, :].float(),
                face_masks.float()[:, :, loss_start_idx:, :, :],
                loss_type=args.face_loss_type,
            )

            total_loss = (
                args.flow_match_weight * flow_loss
                + args.face_loss_weight * face_loss
                + args.temporal_loss_weight * temporal_loss
            )
            (total_loss / args.grad_accum_steps).backward()

            flow_loss_sum += flow_loss.item()
            face_loss_sum += face_loss.item()
            temporal_loss_sum += temporal_loss.item()
            total_loss_sum += total_loss.item()

        torch.nn.utils.clip_grad_norm_(gen_params, max_norm=args.max_grad_norm)
        optimizer_gen.step()
        scheduler_gen.step()
        optimizer_gen.zero_grad(set_to_none=True)

        flow_loss_item = flow_loss_sum / args.grad_accum_steps
        face_loss_item = face_loss_sum / args.grad_accum_steps
        temporal_loss_item = temporal_loss_sum / args.grad_accum_steps
        total_loss_item = total_loss_sum / args.grad_accum_steps

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        if stage1_visual_dict is not None:
            move_VAE_to_device(processor, device)
            _save_stage1_latent_videos(
                processor=processor,
                visual_dict=stage1_visual_dict,
                output_dir=args.output_dir,
                step=step,
                fps=25,
                rank=rank,
            )
            move_VAE_to_device(processor, 'cpu')
            del stage1_visual_dict
            gc.collect()
            torch.cuda.empty_cache()

        if rank == 0:
            writer.add_scalar("loss/total", total_loss_item, step)
            writer.add_scalar("loss/flow_matching", flow_loss_item, step)
            writer.add_scalar("loss/face", face_loss_item, step)
            writer.add_scalar("loss/temporal", temporal_loss_item, step)
            writer.add_scalar("lr/gen", current_lr, step)

        if rank == 0 and step % 10 == 0:
            logging.info(
                "Step %d: Total=%.4f (flow=%.4f, face=%.4f, temporal=%.4f), LR=%.2e, Accum=%d",
                step, total_loss_item, flow_loss_item, face_loss_item, temporal_loss_item,
                current_lr, args.grad_accum_steps,
            )

        if step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_{step}")
            if rank == 0:
                os.makedirs(ckpt_dir, exist_ok=True)

            barrier()

            from safetensors.torch import save_file

            barrier()
            cpu_state = fsdp_state_dict(dmd_model.generator) if args.use_fsdp else dmd_model.generator.state_dict()

            save_path = os.path.join(ckpt_dir, f"model_{step}.safetensors")
            save_sd = {k: v for k, v in cpu_state.items()}

            if rank == 0:
                save_file(save_sd, save_path)
                logging.info(f"Saved checkpoint ({len(save_sd)} params) to {save_path}")

            optim_state_path = os.path.join(ckpt_dir, f"optim_state_rank{rank}.pt")
            torch.save({
                'step': step,
                'optimizer_gen': optimizer_gen.state_dict(),
                'scheduler_gen': scheduler_gen.state_dict(),
            }, optim_state_path)
            barrier()
            if rank == 0:
                training_meta = {'step': step, 'args': vars(args), 'world_size': world_size}
                torch.save(training_meta, os.path.join(ckpt_dir, "training_state.pt"))
                logging.info(f"Checkpoint saved at step {step} in {ckpt_dir}")

    if rank == 0 and writer is not None:
        writer.close()
    if lmdb_reader is not None:
        lmdb_reader.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
