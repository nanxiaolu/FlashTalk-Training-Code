import argparse
import io
import gc
import itertools
import json
import logging
import math
import os
import re
from datetime import datetime
from contextlib import nullcontext

import src._warning_filters  # noqa: F401 - silence noisy 3rd-party warnings
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from infinitetalk_dmd import InfiniteTalkDMD
from src.data_processor_flashtalk import (
    DataProcessor,
    InfiniteTalkDataset,
    LmdbBatchReader,
    _batch_to_cpu,
    load_video_window,
    move_VAE_to_device,
    process_image_to_tensor,
)
from src.validation_inference import run_validation_inference_and_eval  # pyright: ignore[reportMissingImports]
from wan.utils.utils import (barrier, cache_video, fsdp_state_dict,
                             launch_distributed_job, set_seed, str2bool)

# ===================== Args Initialization =====================
# NOTE: train_flashtalk_stage2.py also drives validation of stage1 weights
# (via init_stage1_full + val_only) and stage2 weights (via resume_from + val_only).
# This is intentional and not a typo, see val_stage1.sh / val_stage2.sh.

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="weights/Wan2.1-I2V-14B-480P", help="Path to Wan2.1 Checkpoint")
parser.add_argument("--vae_checkpoint", type=str, default="Wan2.1_VAE.pth", help="VAE filename")
parser.add_argument("--output_dir", type=str, default="outputs/flashtalk_stage2")
parser.add_argument("--auto_output_dir_name", type=str2bool, default=True,
                   help="Auto append experiment-name slug to output_dir based on key hyper-parameters")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gen_grad_accum_steps", type=int, default=8, help="Generator accumulation loops per generator update.")
parser.add_argument("--critic_grad_accum_steps", type=int, default=8, help="Critic accumulation steps.")
parser.add_argument("--gen_lr", type=float, default=2e-6)
parser.add_argument("--critic_lr", type=float, default=4e-7)
parser.add_argument("--gen_betas", nargs=2, type=float, default=None,
                   metavar=("BETA1", "BETA2"),
                   help="Generator AdamW betas; defaults to (0.9, 0.99).")
parser.add_argument("--critic_betas", nargs=2, type=float, default=None,
                   metavar=("BETA1", "BETA2"),
                   help="Critic AdamW betas; defaults to (0.9, 0.99).")
parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Max gradient norm for clipping generator and critic")
parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
parser.add_argument("--init_stage1_full", type=str, default=None,
                   help="Stage-1 full-parameter safetensors path used to initialize generator/real_score/fake_score "
                        "before stage-2 training or stage-1 validation.")
parser.add_argument("--dist_timeout_minutes", type=int, default=60, help="Distributed process group timeout in minutes.")
parser.add_argument("--config", type=str, default=None, help="Path to stage config yaml. Config values override CLI args.")
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--save_interval_first_stage", type=int, default=50,
                   help="Checkpoint interval before save_interval_switch_step.")
parser.add_argument("--save_interval_switch_step", type=int, default=100,
                   help="Step where checkpoint saving switches to the second-stage interval, inclusive.")
parser.add_argument("--save_interval_second_stage", type=int, default=20,
                   help="Checkpoint interval from save_interval_switch_step onward.")

# Validation Args
parser.add_argument("--val_frame_num", type=int, default=33, help="Validation inference clip length.")
parser.add_argument("--val_max_frame_num", type=int, default=1000, help="Validation inference max generated frames.")
parser.add_argument("--val_motion_frame", type=int, default=5, help="Validation inference motion frame count.")
parser.add_argument("--val_color_correction_strength", type=float, default=1.0, help="Validation inference color correction strength.")
parser.add_argument("--val_disable_cfg", type=str2bool, default=False, help="Disable CFG during validation inference.")
parser.add_argument("--val_text_guide_scale", type=float, default=0.0, help="Validation inference text-context guide scale.")
parser.add_argument("--val_audio_guide_scale", type=float, default=0.0, help="Validation inference audio guide scale.")
parser.add_argument("--val_infer_steps", type=int, default=None, help="Optional custom denoising step count for validation inference (T-step, e.g. 40).")
parser.add_argument("--val_only", type=str2bool, default=False, help="Only run validation inference, no training.")

# DMD Args
parser.add_argument("--gradient_checkpointing", type=str2bool, default=True, help="Enable gradient checkpointing to save memory")
parser.add_argument("--text_guide_scale", type=float, default=5.0, help="CFG scale for text guidance")
parser.add_argument("--audio_guide_scale", type=float, default=4.0, help="CFG scale for audio guidance")
parser.add_argument("--denoising_step_list", type=str, default="1000,750,500,250",
                   help="Comma-separated denoising steps for self-forcing simulation")
parser.add_argument("--use_fixed_reference_frame", type=str2bool, default=True,
                   help="If True, always use first frame as reference; otherwise random sample")
parser.add_argument("--save_generator_latent_videos", type=str2bool, default=True,
                   help="Save generator-loss intermediate latents as decoded mp4 videos after generator optimizer step")
parser.add_argument("--save_generator_latent_videos_interval", type=int, default=100,
                   help="Interval to save generator-loss intermediate latents as decoded mp4 videos")
parser.add_argument("--keep_k_chunks", type=int, default=-1, help="Number of chunks to keep for self-forcing++")
parser.add_argument("--dmd_loss_weight", type=float, default=1.0, help="Weight for DMD loss term")
parser.add_argument("--temporal_align_weight", type=float, default=0.25,
                   help="Weight for RVM background temporal alignment loss")

# Extra Args
parser.add_argument("--seed", type=int, default=43, help="Random seed for training")
parser.add_argument("--wav2vec_dir", type=str, default="weights/chinese-wav2vec2-base")
parser.add_argument("--infinitetalk_dir", type=str, default="weights/InfiniteTalk/single/infinitetalk.safetensors")
parser.add_argument("--size", type=str, default="infinitetalk-480")
parser.add_argument("--training_window_size", type=int, default=33, help="Training window size (frames)")
parser.add_argument("--debug", type=str2bool, default=False, help="Debug mode: Define model structure without loading weights")
parser.add_argument("--timestep_shift", type=float, default=5.0, help="Timestep shift factor for noise scheduling")
parser.add_argument("--n_prompt", type=str,
                   default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                   help="Negative prompt for CFG (aligned with InfiniteTalk inference)")
parser.add_argument("--use_precomputed_audio", type=str2bool, default=False,
                   help="Use precomputed audio embeddings (.pt files) instead of computing from .wav files")
parser.add_argument("--mode", type=str, default="train", choices=["preprocess", "train"],
                   help="LMDB mode: preprocess=extract per-sample payload files for later packing; train=load samples from packed LMDB")
parser.add_argument("--lmdb_path", type=str, default="",
                   help="LMDB file path for stage-2 samples")
parser.add_argument("--lmdb_num_samples", type=int, default=17600,
                   help="Number of samples to preprocess (global keys from 0)")
parser.add_argument("--lmdb_map_size_gb", type=int, default=2048,
                   help="LMDB max file size in GB for preprocess mode")
parser.add_argument("--payload_dir", type=str, default=None,
                   help="Stage-A payload directory. Default: <lmdb_path without ext>")
parser.add_argument("--stage2_chunk_frames", type=int, default=33,
                   help="Chunk frame count for stage-2 self-forcing++")
parser.add_argument("--stage2_chunk_overlap_frames", type=int, default=5,
                   help="Chunk overlap frames between adjacent chunks")
parser.add_argument("--stage2_k_max", type=int, default=5,
                   help="Maximum K chunks for self-forcing++")

# TalkCuts dataset paths (only required when mode=preprocess or for val_only).
parser.add_argument("--dataset_dir", type=str, default="",
                   help="Root directory hosting the raw video/audio files referenced by annotation csvs.")
parser.add_argument("--annotation_file", type=str, default="processed_data/example/train_data.csv",
                   help="CSV file with columns video,input_audio,prompt for training set.")
parser.add_argument("--val_annotation_file", type=str, default="processed_data/talkcuts/val_data.csv",
                   help="CSV file for validation set.")
parser.add_argument("--val_features_dir", type=str, default="processed_data/talkcuts/val/feature",
                   help="Directory of per-sample validation features (context.pt/full_emb.pt/clip_fea.pt/...). "
                        "Loaded by val_only inference. Produced by running stage1 preprocess with "
                        "preprocess_split=val; context_null.pt is expected directly inside this directory "
                        "(i.e. <val_features_dir>/context_null.pt).")

# FSDP Args
parser.add_argument("--use_fsdp", type=str2bool, default=True, help="Use FSDP for training")
parser.add_argument("--mixed_precision", type=str2bool, default=True, help="Enable FSDP mixed precision")
parser.add_argument("--sharding_strategy", type=str, default="full", choices=["full", "hybrid_full", "hybrid_zero2", "no_shard"], help="FSDP sharding strategy")
parser.add_argument("--fsdp_wrap_strategy", type=str, default="size", choices=["size", "transformer"], help="FSDP auto wrap strategy")
parser.add_argument("--fsdp_cpu_offload", type=str2bool, default=False, help="FSDP CPU offload")


def _load_stage_config(config_path):
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a key-value yaml mapping: {config_path}")
    return cfg


def _normalize_betas(v):
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


def _checkpoint_save_interval_for_step(args, step):
    if step < args.save_interval_switch_step:
        return args.save_interval_first_stage
    return args.save_interval_second_stage


def _validate_checkpoint_save_args(args):
    for name in ("save_interval_first_stage", "save_interval_switch_step", "save_interval_second_stage"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0, got {getattr(args, name)}")


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
        "flashtalk_stage2",
        f"window_{args.training_window_size}",
        f"gen_lr_{_format_scientific_short(args.gen_lr)}",
        f"critic_lr_{_format_scientific_short(args.critic_lr)}",
        f"dmd_w_{args.dmd_loss_weight:g}",
        f"temp_w_{args.temporal_align_weight:g}",
        f"keep_k_{args.keep_k_chunks}",
        "full_finetune",
    ]
    return "_".join(parts)


def _save_generator_latent_videos(processor, visual_dict, output_dir, step, rank, fps=25):
    """Decode intermediate generator latents and save mp4 videos per sample."""
    tensor_names = [
        "pred_x0_teacher_input",
        "pred_x0_student",
        "pred_x0_real",
        "pred_x0_fake",
        "target_no_norm",
        "target",
        "clean_latents",
        "average_target",
    ]
    save_root = os.path.join(output_dir, f"iter_{step}")
    os.makedirs(save_root, exist_ok=True)
    gen_exit_idx = visual_dict.get("gen_exit_idx")
    teacher_t_raw = visual_dict.get("teacher_t_raw")
    self_forcing_k = visual_dict.get("k", visual_dict.get("self_forcing_k"))

    with torch.no_grad():
        bs = 1
        for name in tensor_names:
            if visual_dict[name] is None:
                continue
            latents_list = [visual_dict[name][idx] for idx in range(bs)]
            decoded_videos = processor.vae.decode(latents_list)
            for idx, video in enumerate(decoded_videos):
                folder_name = "sample" if rank is None else f"rank_{rank}"
                if gen_exit_idx is not None:
                    folder_name += f"_gen_exit_{gen_exit_idx}"
                if teacher_t_raw is not None:
                    folder_name += f"_t_teacher_{teacher_t_raw}"
                if self_forcing_k is not None:
                    folder_name += f"_k_{self_forcing_k}"
                if rank is None and gen_exit_idx is None and teacher_t_raw is None:
                    folder_name = f"{folder_name}_{idx + 1}"
                elif bs > 1:
                    folder_name += f"_sample_{idx + 1}"

                sample_dir = os.path.join(save_root, folder_name)
                os.makedirs(sample_dir, exist_ok=True)
                save_path = os.path.join(sample_dir, f"{name}.mp4")
                cache_video(
                    tensor=video.unsqueeze(0),
                    save_file=save_path,
                    fps=fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )


def _serialize_torch_obj(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _choose_k_with_quota(max_k, k_counts, k_quota):
    max_k = int(max_k)
    for k in range(max_k, 0, -1):
        if k_counts[k] < k_quota:
            return k
    return -1


def _build_stage2_self_forcing_sample(processor, sample, selected_k, args, model_dtype):
    """Build one stage-2 self-forcing sample dict, containing K chunks of conditioning tensors."""
    sample_dir = sample["sample_dir"]
    video_path = sample["video_path"]
    audio_path = sample["audio_path"]
    prompt = sample["prompt"]
    chunk_frames = int(args.stage2_chunk_frames)
    stride = int(args.stage2_chunk_frames - args.stage2_chunk_overlap_frames)
    total_need_frames = int(chunk_frames + (int(selected_k) - 1) * stride)

    if args.size == "infinitetalk-480":
        from wan.utils.multitalk_utils import ASPECT_RATIO_627 as bucket_config
    else:
        from wan.utils.multitalk_utils import ASPECT_RATIO_960 as bucket_config

    # Randomly pick one valid segment satisfying chunk_frames + (k-1)*stride.
    segment_frames, segment_start_idx, _, total_frames = load_video_window(
        sample_dir,
        video_path=video_path,
        frame_num=total_need_frames,
        window_start=None,
    )
    first_frame = segment_frames[0]
    src_h, src_w = first_frame.shape[0], first_frame.shape[1]
    ratio = src_h / src_w
    closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = bucket_config[closest_bucket][0]
    target_size = (target_h, target_w)

    img_ref = Image.fromarray(first_frame)
    img_ref_tensor = process_image_to_tensor(img_ref, target_size=target_size, device=processor.device)
    with torch.no_grad():
        clip_fea = processor.clip.visual(img_ref_tensor[:, :, -1:, :, :]).squeeze(0)
        context = torch.stack(processor.text_encoder([prompt], processor.device)).squeeze(0)
        if processor._context_null_precomputed is not None:
            context_null = processor._context_null_precomputed.to(processor.device).clone().squeeze(0)
        else:
            context_null = torch.stack(processor.text_encoder([processor.n_prompt], processor.device)).squeeze(0)

    # First-frame latent shared by all chunks (chunk-1 motion source).
    with torch.no_grad():
        first_frame_latent = processor.vae.encode([img_ref_tensor.squeeze(0)])[0]
    lat_h = int(first_frame_latent.shape[-2])
    lat_w = int(first_frame_latent.shape[-1])

    padding_frames_pixels = torch.zeros(
        3, chunk_frames, target_h, target_w,
        dtype=img_ref_tensor.dtype,
        device=img_ref_tensor.device,
    )
    padding_frames_pixels[:, 0:1, :, :] = img_ref_tensor[0, :, 0:1, :, :]
    with torch.no_grad():
        y_latent = processor.vae.encode([padding_frames_pixels])[0]
    msk = torch.ones(1, chunk_frames, lat_h, lat_w, device=processor.device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2).to(y_latent.dtype).squeeze(0)
    cond_latents = torch.concat([msk, y_latent], dim=0)

    vae_stride = getattr(processor.config, 'vae_stride', (4, 8, 8))
    patch_size = getattr(processor.config, 'patch_size', (1, 2, 2))
    sp_size = getattr(processor.config, 'sp_size', 1)
    max_seq_len = ((chunk_frames - 1) // vae_stride[0] + 1) * lat_h * lat_w // (patch_size[1] * patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / sp_size)) * sp_size
    ref_target_masks = torch.ones(3, lat_h, lat_w, device=processor.device, dtype=model_dtype)

    full_emb = processor._load_audio_embedding(audio_path, frame_num=chunk_frames)
    window_offsets = torch.tensor([-2, -1, 0, 1, 2], device=processor.device)

    shared_context = context.unsqueeze(0).to(dtype=model_dtype)
    shared_context_null = context_null.unsqueeze(0).to(dtype=model_dtype)
    shared_clip = clip_fea.unsqueeze(0).to(dtype=model_dtype)
    shared_cond = cond_latents.unsqueeze(0).to(dtype=model_dtype)
    shared_first_frame_latent = first_frame_latent.unsqueeze(0).to(dtype=model_dtype)

    chunks = []
    for i in range(int(selected_k)):
        chunk_start_abs = int(segment_start_idx) + i * stride
        frame_indices = torch.arange(chunk_frames, device=processor.device) + chunk_start_abs
        window_indices = frame_indices.unsqueeze(1) + window_offsets.unsqueeze(0)
        window_indices = window_indices.clamp(0, full_emb.shape[0] - 1)
        audio_emb = full_emb[window_indices].unsqueeze(0).to(dtype=model_dtype)
        audio_null = torch.zeros_like(audio_emb)

        chunks.append({
            "first_frame_latent": shared_first_frame_latent,
            "context": shared_context,
            "context_null": shared_context_null,
            "clip_fea": shared_clip,
            "cond_latents": shared_cond,
            "audio_emb": audio_emb,
            "audio_null": audio_null,
            "seq_len": int(max_seq_len),
            "ref_target_masks": ref_target_masks,
        })

    legacy = chunks[-1]
    out = dict(legacy)
    out["self_forcing_chunks"] = chunks
    out["selected_k"] = int(selected_k)
    out["total_frames"] = int(total_frames)
    out["segment_start_idx"] = int(segment_start_idx)
    return out


def preprocess_stage2_payload_files(processor, dataloader, payload_dir, num_samples,
                                     rank, world_size, args, model_dtype=torch.bfloat16,
                                     log_interval=20):
    """Preprocess stage-2 samples into per-sample payload files (one .payload per global key).

    Loop until the globally accumulated count of *useful* samples
    (newly written this run + already existing on disk) reaches ``num_samples``.
    Samples skipped due to no remaining k-quota do not count, so the loop keeps
    iterating to compensate. Key indices are assigned by
    ``local_step * world_size + rank`` and may end up sparse; the LMDB pack
    step renumbers them to a contiguous range later.
    """
    os.makedirs(payload_dir, exist_ok=True)

    k_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    local_step = 0
    written = 0
    skipped_exists = 0
    skipped_no_quota = 0
    target = int(num_samples)
    k_max = int(args.stage2_k_max)
    # Use ceil division so the aggregated quota covers ``target`` even when
    # the count is not divisible by ``k_max``.
    stage2_quota_per_k = (target + k_max - 1) // k_max

    # Rank-0 authoritative global counters across all ranks.
    global_skipped_exists = 0

    while True:
        global_pos = local_step * world_size + rank
        key_idx = int(global_pos)
        batch_raw = next(dataloader)
        sample = batch_raw[0]

        payload_path = os.path.join(payload_dir, f"{key_idx}.payload")
        file_exists = os.path.isfile(payload_path)
        if file_exists:
            max_k = 0
        else:
            _, _, _, total_frames = load_video_window(
                sample["sample_dir"], video_path=sample["video_path"],
                frame_num=1, window_start=0,
            )
            stride = args.stage2_chunk_frames - args.stage2_chunk_overlap_frames
            max_k = min((total_frames - args.stage2_chunk_overlap_frames) // stride, args.stage2_k_max)

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            {"rank": rank, "global_pos": int(global_pos),
             "file_exists": bool(file_exists), "max_k": int(max_k)},
        )

        selected_k_by_rank = None
        stop_now = False
        if rank == 0:
            selected_k_by_rank = {}
            for item in sorted(gathered, key=lambda x: x["global_pos"]):
                r = int(item["rank"])
                if bool(item["file_exists"]):
                    selected_k_by_rank[r] = -2
                    global_skipped_exists += 1
                    continue
                choose_k = _choose_k_with_quota(
                    max_k=item["max_k"], k_counts=k_counts, k_quota=stage2_quota_per_k,
                )
                if int(choose_k) != -1:
                    k_counts[int(choose_k)] += 1
                selected_k_by_rank[r] = int(choose_k)
            global_written = sum(k_counts.values())
            if (global_written + global_skipped_exists) >= target:
                stop_now = True
        bcast_list = [selected_k_by_rank, bool(stop_now)]
        dist.broadcast_object_list(bcast_list, src=0)
        selected_k = int(bcast_list[0][rank])
        stop_now = bool(bcast_list[1])

        if selected_k == -2:
            skipped_exists += 1
        elif selected_k == -1:
            skipped_no_quota += 1
        else:
            batch_tensors = _build_stage2_self_forcing_sample(
                processor=processor, sample=sample, selected_k=selected_k,
                args=args, model_dtype=model_dtype,
            )
            payload = _serialize_torch_obj(_batch_to_cpu(batch_tensors))
            tmp_path = payload_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(payload)
            os.replace(tmp_path, payload_path)
            written += 1
        local_step += 1

        if (written + skipped_exists + skipped_no_quota) % int(log_interval) == 0:
            logging.info(
                "Stage2 payload preprocess rank=%d processed=%d (written=%d, skip_exists=%d, skip_no_quota=%d)",
                rank, written + skipped_exists + skipped_no_quota,
                written, skipped_exists, skipped_no_quota,
            )

        if stop_now:
            break

    dist.barrier()
    if rank == 0:
        logging.info("Stage2 K counts target snapshot: %s", k_counts)
        logging.info(
            "Stage2 payload preprocess summary: written=%d, skip_exists=%d, skip_no_quota=%d",
            written, skipped_exists, skipped_no_quota,
        )
    return written, skipped_exists + skipped_no_quota


def main():
    args = parser.parse_args()
    args, config_overrides, _ = _merge_config_over_cli(args, args.config)
    args.mode = str(args.mode).lower()
    if args.batch_size != 1:
        raise ValueError("preprocess/train mode currently requires batch_size=1")
    if not args.val_only:
        if args.mode == "preprocess":
            if not args.payload_dir:
                raise ValueError("payload_dir is required when mode=preprocess")
            if args.lmdb_num_samples <= 0:
                raise ValueError("lmdb_num_samples must be > 0 when mode=preprocess")
        elif args.mode == "train":
            if not args.lmdb_path:
                raise ValueError("lmdb_path is required when mode=train")
    _validate_checkpoint_save_args(args)
    args.gen_betas = _normalize_betas(args.gen_betas)
    args.critic_betas = _normalize_betas(args.critic_betas)

    if args.init_stage1_full is not None:
        normalized_full = str(args.init_stage1_full).strip()
        if normalized_full.lower() in ("", "none", "null", "false"):
            args.init_stage1_full = None
        else:
            if not os.path.isfile(normalized_full):
                raise FileNotFoundError(f"init_stage1_full not found: {normalized_full}")
            args.init_stage1_full = normalized_full

    args.gen_grad_accum_steps = max(1, int(args.gen_grad_accum_steps))
    args.critic_grad_accum_steps = max(1, int(args.critic_grad_accum_steps))
    if args.val_only and args.use_fsdp:
        # val_only is inference-only; disabling FSDP avoids multi-rank validation tail hangs.
        args.use_fsdp = False

    if args.auto_output_dir_name:
        args.output_dir = os.path.join(args.output_dir, _build_experiment_name(args))
    base_output_dir = args.output_dir

    launch_distributed_job(timeout_minutes=args.dist_timeout_minutes)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

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
        if args.init_stage1_full:
            logging.info("Resolved init_stage1_full path: %s", args.init_stage1_full)
        logging.info("========== Training args ==========")
        for k, v in sorted(vars(args).items()):
            logging.info("  %s: %s", k, v)
        logging.info("===================================")
    else:
        logging.basicConfig(level=logging.ERROR)

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

    # Enable RVM+SSIM background filtering only during preprocess.
    setattr(config, "enable_background_filter", args.mode == "preprocess")

    if hasattr(args, "seed"):
        set_seed(args.seed + rank)

    processor_init_device = device if args.mode == "preprocess" else ('cpu' if args.use_fsdp else device)
    processor = DataProcessor(
        config, args.ckpt_dir, processor_init_device,
        n_prompt=args.n_prompt,
        use_precomputed_audio=args.use_precomputed_audio,
        processed_data_dir=args.val_features_dir,
        use_fixed_reference_frame=args.use_fixed_reference_frame,
        rank=rank,
        mode=args.mode,
    )

    processor.device = device
    processor.vae.mean = processor.vae.mean.to(device)
    processor.vae.std = processor.vae.std.to(device)
    processor.vae.scale = [processor.vae.mean, 1.0 / processor.vae.std]
    if processor.rvm is not None:
        processor.rvm = processor.rvm.to(device)

    if args.mode == "preprocess":
        # Only preprocess mode walks raw video/audio via a DataLoader.
        dataset = InfiniteTalkDataset(args.dataset_dir, annotation_file=args.annotation_file)
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed + rank, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda x: x)
        dataloader = itertools.cycle(dataloader)

        payload_dir = args.payload_dir
        written, skipped = preprocess_stage2_payload_files(
            processor=processor, dataloader=dataloader, payload_dir=payload_dir,
            num_samples=int(args.lmdb_num_samples), rank=rank, world_size=world_size,
            args=args, model_dtype=torch.bfloat16,
        )
        logging.info(
            "Stage2 payload preprocess done on rank=%d, written=%d, skipped=%d, payload_dir=%s",
            rank, written, skipped, payload_dir,
        )
        barrier()
        if rank == 0:
            logging.info(
                "Stage2 payload preprocess complete: payload_dir=%s, total_samples=%d. "
                "Run tools/payload_files_to_lmdb.py for Stage-B LMDB packing.",
                payload_dir, args.lmdb_num_samples,
            )
        dist.destroy_process_group()
        return

    val_dataloader = None
    if args.val_only:
        val_dataset = InfiniteTalkDataset(args.dataset_dir, annotation_file=args.val_annotation_file)
        # Shard validation samples by rank without padding/duplication.
        local_indices = list(range(rank, len(val_dataset), world_size))
        val_subset = Subset(val_dataset, local_indices)
        val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=args.batch_size, collate_fn=lambda x: x)

    # Stage selection:
    # - val_only: only load generator for inference.
    # - training: load generator + real_score + fake_score (three models).
    dmd_stage = "val_only" if args.val_only else "stage2"
    dmd_model = InfiniteTalkDMD(
        config, device, args.ckpt_dir, args.infinitetalk_dir,
        stage=dmd_stage, debug=args.debug,
    )

    if args.val_only:
        dmd_model.setup_for_validation(args)
    else:
        dmd_model.setup_for_training(args)

    dmd_model.register_debug_io(processor=processor, output_dir=args.output_dir)

    if args.val_only:
        if val_dataloader is None:
            raise ValueError("val_only=True requires a val_annotation_file to build val_dataloader.")
        step = 0
        if args.resume_from is not None:
            training_state_path = os.path.join(args.resume_from, "training_state.pt")
            if os.path.exists(training_state_path):
                checkpoint = torch.load(training_state_path, map_location='cpu')
                step = checkpoint.get('step', 0)

        logging.info(
            "val_only=True: run distributed validation inference (rank=%d, local_samples=%d, step=%d).",
            rank, len(val_dataloader.dataset), step,
        )
        run_validation_inference_and_eval(
            dmd_model=dmd_model, processor=processor, val_dataloader=val_dataloader,
            args=args, step=step, rank=rank,
        )
        if rank == 0:
            logging.info("validation inference + eval done (val_only mode).")
        dist.destroy_process_group()
        return

    gen_params = [p for p in dmd_model.generator.parameters() if p.requires_grad]
    critic_params = [p for p in dmd_model.fake_score.parameters() if p.requires_grad]
    logging.info(f"Generator trainable params: {len(gen_params)}, Critic trainable params: {len(critic_params)}")

    optimizer_gen = torch.optim.AdamW(gen_params, lr=args.gen_lr, foreach=False, betas=args.gen_betas)
    optimizer_critic = torch.optim.AdamW(critic_params, lr=args.critic_lr, foreach=False, betas=args.critic_betas)
    logging.info("Gradient clipping max_norm: %.2f", args.max_grad_norm)
    if rank == 0:
        logging.info(
            "Gradient accumulation: gen_grad_accum_steps=%d (effective gen batch=%d), critic_grad_accum_steps=%d",
            args.gen_grad_accum_steps, args.batch_size * args.gen_grad_accum_steps * world_size,
            args.critic_grad_accum_steps,
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
        checkpoint = torch.load(training_state_path, map_location='cpu')
        step = checkpoint['step']

        optim_state_path = os.path.join(args.resume_from, f"optim_state_rank{rank}.pt")
        optim_ckpt = torch.load(optim_state_path, map_location='cpu')
        optimizer_gen.load_state_dict(optim_ckpt['optimizer_gen'])
        if 'optimizer_critic' in optim_ckpt and len(critic_params) > 0:
            optimizer_critic.load_state_dict(optim_ckpt['optimizer_critic'])
        logging.info(f"Loaded per-rank optimizer state from {optim_state_path}")
        del optim_ckpt

        logging.info(f"Resumed training from step {step}")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    model_dtype = next(dmd_model.parameters()).dtype
    lmdb_reader = LmdbBatchReader(args.lmdb_path)
    if rank == 0:
        logging.info(
            "LMDB train mode enabled: path=%s, num_samples=%d",
            args.lmdb_path, lmdb_reader.num_samples,
        )

    def _prepare_batch_tensors_from_lmdb(step_idx, accum_idx, is_gen):
        if is_gen:
            lmdb_key = (step_idx - 1) * world_size * args.gen_grad_accum_steps + accum_idx * world_size + rank
        else:
            lmdb_key = (step_idx - 1) * world_size * args.critic_grad_accum_steps + accum_idx * world_size + rank
        if lmdb_key >= lmdb_reader.num_samples:
            raise RuntimeError(
                f"LMDB key out of range: key={lmdb_key}, num_samples={lmdb_reader.num_samples}. "
                "Please increase lmdb_num_samples during preprocess."
            )
        return lmdb_reader.get(lmdb_key, device=device, model_dtype=model_dtype)

    if args.debug:
        args.save_generator_latent_videos_interval = 1
        args.save_interval_first_stage = 1
        args.save_interval_second_stage = 1
        update_gen_interval = 1
    else:
        update_gen_interval = 5

    move_VAE_to_device(processor, device)
    dmd_model.eval()
    while step < args.max_steps:
        step += 1

        should_update_gen = (step % update_gen_interval == 0)

        # 1. Update Generator
        if should_update_gen:
            optimizer_gen.zero_grad(set_to_none=True)
            gen_visual_dict = None
            loss_gen_sum = 0.0
            dmd_loss_term_sum = 0.0
            motion_align_loss_term_sum = 0.0
            rvm_bg_align_loss_term_sum = 0.0
            gen_term_count = 0

            for gen_accum_idx in range(args.gen_grad_accum_steps):
                cur_batch_tensors = _prepare_batch_tensors_from_lmdb(step, gen_accum_idx, is_gen=True)

                want_visual = args.save_generator_latent_videos and gen_accum_idx == 0
                is_last_accum = (gen_accum_idx == args.gen_grad_accum_steps - 1)
                gen_sync_ctx = nullcontext() if (is_last_accum or not args.use_fsdp) else dmd_model.generator.no_sync()
                with gen_sync_ctx:
                    loss_gen = dmd_model._generator_loss_self_forcing_plus_plus(
                        cur_batch_tensors['self_forcing_chunks'],
                        train_step=max(step - 1, 0),
                        return_visual_dict=want_visual,
                    )
                    if want_visual:
                        loss_gen, gen_visual_dict = loss_gen

                    loss_gen_sum += loss_gen.item()
                    gen_terms = getattr(dmd_model, "latest_gen_loss_components", None)
                    if isinstance(gen_terms, dict):
                        dmd_loss_term_sum += float(gen_terms.get("dmd_loss_term", 0.0))
                        motion_align_loss_term_sum += float(gen_terms.get("motion_align_loss_term", 0.0))
                        rvm_bg_align_loss_term_sum += float(gen_terms.get("rvm_bg_align_loss_term", 0.0))
                        gen_term_count += 1
                    (loss_gen / args.gen_grad_accum_steps).backward()
                del loss_gen
                del cur_batch_tensors

            if args.use_fsdp:
                dmd_model.generator.clip_grad_norm_(max_norm=args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=args.max_grad_norm)
            optimizer_gen.step()
            optimizer_gen.zero_grad(set_to_none=True)
            loss_gen_item = loss_gen_sum / args.gen_grad_accum_steps
            denom = max(gen_term_count, 1)
            dmd_loss_term_item = dmd_loss_term_sum / denom
            motion_align_loss_term_item = motion_align_loss_term_sum / denom
            rvm_bg_align_loss_term_item = rvm_bg_align_loss_term_sum / denom
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            if gen_visual_dict is not None and step % args.save_generator_latent_videos_interval == 0:
                _save_generator_latent_videos(
                    processor=processor, visual_dict=gen_visual_dict,
                    output_dir=args.output_dir, step=step, fps=25, rank=rank,
                )
                del gen_visual_dict
                gc.collect()
                torch.cuda.empty_cache()
        else:
            loss_gen_item = None
            dmd_loss_term_item = None
            motion_align_loss_term_item = None
            rvm_bg_align_loss_term_item = None

        # 2. Update Critic
        loss_critic_item = None
        optimizer_critic.zero_grad(set_to_none=True)
        loss_critic_sum = 0.0
        critic_finite_accum = 0

        for critic_accum_idx in range(args.critic_grad_accum_steps):
            cur_critic_tensors = _prepare_batch_tensors_from_lmdb(step, critic_accum_idx, is_gen=False)

            is_last_critic_accum = (critic_accum_idx == args.critic_grad_accum_steps - 1)
            critic_sync_ctx = nullcontext() if (is_last_critic_accum or not args.use_fsdp) else dmd_model.fake_score.no_sync()
            with critic_sync_ctx:
                loss_critic = dmd_model._critic_loss_self_forcing_plus_plus(
                    cur_critic_tensors['self_forcing_chunks']
                )
                loss_critic_sum += loss_critic.item()
                (loss_critic / args.critic_grad_accum_steps).backward()
            critic_finite_accum += 1
            del loss_critic
            del cur_critic_tensors

        if critic_finite_accum > 0:
            if args.use_fsdp:
                dmd_model.fake_score.clip_grad_norm_(max_norm=args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(critic_params, max_norm=args.max_grad_norm)
            optimizer_critic.step()
        optimizer_critic.zero_grad(set_to_none=True)

        loss_critic_item = (
            loss_critic_sum / critic_finite_accum if critic_finite_accum > 0 else None
        )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        if rank == 0:
            if loss_gen_item is not None:
                writer.add_scalar("gen_loss", loss_gen_item, step)
            if dmd_loss_term_item is not None:
                writer.add_scalar("gen_loss/dmd_loss_term", dmd_loss_term_item, step)
            if motion_align_loss_term_item is not None:
                writer.add_scalar("gen_loss/motion_align_loss_term", motion_align_loss_term_item, step)
            if rvm_bg_align_loss_term_item is not None:
                writer.add_scalar("gen_loss/rvm_bg_align_loss_term", rvm_bg_align_loss_term_item, step)
            if loss_critic_item is not None:
                writer.add_scalar("critic_loss", loss_critic_item, step)
        if rank == 0 and step % 10 == 0:
            gen_loss_str = f"{loss_gen_item:.4f}" if loss_gen_item is not None else "N/A"
            critic_loss_str = f"{loss_critic_item:.4f}" if loss_critic_item is not None else "N/A"
            dmd_loss_term_str = f"{dmd_loss_term_item:.4f}" if dmd_loss_term_item is not None else "N/A"
            motion_align_loss_term_str = f"{motion_align_loss_term_item:.4f}" if motion_align_loss_term_item is not None else "N/A"
            rvm_bg_align_loss_term_str = f"{rvm_bg_align_loss_term_item:.4f}" if rvm_bg_align_loss_term_item is not None else "N/A"
            logging.info(
                "Step %s: Gen Loss=%s (dmd=%s, motion=%s, rvm_bg=%s), Critic Loss=%s, Gen LR=%.2e, Critic LR=%.2e, GenAccum=%d, CriticAccum=%d",
                step, gen_loss_str, dmd_loss_term_str, motion_align_loss_term_str,
                rvm_bg_align_loss_term_str, critic_loss_str, args.gen_lr, args.critic_lr,
                args.gen_grad_accum_steps, args.critic_grad_accum_steps,
            )

        save_interval = _checkpoint_save_interval_for_step(args, step)
        if step % save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_{step}")
            if rank == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
            barrier()

            from safetensors.torch import save_file

            gen_cpu = fsdp_state_dict(dmd_model.generator) if args.use_fsdp else dmd_model.generator.state_dict()
            critic_cpu = fsdp_state_dict(dmd_model.fake_score) if args.use_fsdp else dmd_model.fake_score.state_dict()
            if rank == 0:
                gen_path = os.path.join(ckpt_dir, f"generator_{step}.safetensors")
                critic_path = os.path.join(ckpt_dir, f"critic_{step}.safetensors")
                save_file(gen_cpu, gen_path)
                save_file(critic_cpu, critic_path)
                logging.info(f"Saved Generator to {gen_path}")
                logging.info(f"Saved Critic to {critic_path}")

            optim_state_path = os.path.join(ckpt_dir, f"optim_state_rank{rank}.pt")
            torch.save({
                'step': step,
                'optimizer_gen': optimizer_gen.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict(),
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
