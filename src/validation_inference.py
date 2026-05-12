import csv
import json
import logging
import math
import os
import subprocess
import sys
from typing import Dict, List

import torch
import torch.distributed as dist

from src.data_processor_flashtalk import _get_sample_id_from_video_path, move_VAE_to_device
from wan.multitalk import resize_and_centercrop
from wan.utils.multitalk_utils import match_and_blend_colors, save_video_ffmpeg
from wan.utils.utils import extract_specific_frames


def _to_tensor(out):
    return torch.stack(out) if isinstance(out, list) else out


def _build_audio_windows(full_audio_emb: torch.Tensor, start_idx: int, frame_num: int, device: torch.device) -> torch.Tensor:
    indices = (torch.arange(5, device=device) - 2).view(1, 5)
    center = torch.arange(start_idx, start_idx + frame_num, device=device).view(frame_num, 1)
    win_idx = (center + indices).clamp(0, full_audio_emb.shape[0] - 1)
    return full_audio_emb[win_idx]


def _build_color_reference(video_path: str, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    ref_img = extract_specific_frames(video_path, 0)
    ref_img = resize_and_centercrop(ref_img, (target_h, target_w))
    ref_img = ref_img / 255.0
    ref_img = (ref_img - 0.5) * 2.0
    return ref_img.to(device)


def _append_average_row(result_csv_path: str) -> None:
    if not os.path.isfile(result_csv_path):
        return

    with open(result_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or ["video_path", "Sync-C", "Sync-D", "IQA", "Aesthe"]

    metric_keys = ["Sync-C", "Sync-D", "IQA", "Aesthe"]
    valid_rows = [r for r in rows if r.get("video_path", "") != "__AVERAGE__"]
    sums = {k: 0.0 for k in metric_keys}
    counts = {k: 0 for k in metric_keys}

    for row in valid_rows:
        for key in metric_keys:
            val = row.get(key, "")
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            sums[key] += v
            counts[key] += 1

    avg_row = {"video_path": "__AVERAGE__"}
    for key in metric_keys:
        if counts[key] > 0:
            avg_row[key] = f"{(sums[key] / counts[key]):.6f}"
        else:
            avg_row[key] = ""

    out_rows = [r for r in rows if r.get("video_path", "") != "__AVERAGE__"]
    out_rows.append(avg_row)

    with open(result_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def _build_val_timesteps(dmd_model, args) -> List[float]:
    custom_steps = getattr(args, "val_infer_steps", None)
    if custom_steps is not None:
        custom_steps = int(custom_steps)
        if custom_steps <= 0:
            raise ValueError(f"val_infer_steps must be a positive integer, got {custom_steps}")
        # Build T-step schedule from 1000 -> 0 (inclusive endpoints).
        timesteps = [1000.0 - (1000.0 * i / custom_steps) for i in range(custom_steps)]
        timesteps.append(0.0)
        return timesteps

    timesteps = [float(x) for x in dmd_model.denoising_step_list]
    if len(timesteps) == 0:
        timesteps = [1000.0, 750.0, 500.0, 250.0]
    timesteps = timesteps + [0.0]
    return timesteps


def _infer_single_sample(
    dmd_model,
    processor,
    sample_raw: Dict,
    feature_dir: str,
    args,
    frame_num: int,
    max_frame_num: int,
    motion_frame: int,
    color_correction_strength: float,
    output_mp4_no_ext: str,
    rank: int = 0,
) -> None:
    device = next(dmd_model.parameters()).device
    param_dtype = next(dmd_model.parameters()).dtype

    context = torch.load(os.path.join(feature_dir, "context.pt"), map_location=device).to(dtype=param_dtype)
    if context.dim() == 2:
        context = context.unsqueeze(0)
    clip_fea = torch.load(os.path.join(feature_dir, "clip_fea.pt"), map_location=device).to(dtype=param_dtype).unsqueeze(0)
    first_frame_latent = torch.load(os.path.join(feature_dir, "first_frame_latent.pt"), map_location=device)
    cond_latents = torch.load(os.path.join(feature_dir, "cond_latents.pt"), map_location=device).to(dtype=param_dtype)
    if cond_latents.dim() == 4:
        cond_latents = cond_latents.unsqueeze(0)
    full_audio_emb = torch.load(os.path.join(feature_dir, "full_emb.pt"), map_location=device)
    with open(os.path.join(feature_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    if full_audio_emb.shape[0] <= 0:
        full_audio_emb = torch.zeros(1, 12, 768, device=device, dtype=param_dtype)
    else:
        full_audio_emb = full_audio_emb.to(dtype=param_dtype)

    target_h, target_w = int(meta["target_size"][0]), int(meta["target_size"][1])
    lat_h = first_frame_latent.shape[-2]
    lat_w = first_frame_latent.shape[-1]
    t_lat = (frame_num - 1) // 4 + 1
    cond_t = int(cond_latents.shape[2])
    if cond_t > int(t_lat):
        # Allow using legacy 81-frame cond_latents by truncating to current val_frame_num.
        cond_latents = cond_latents[:, :, :int(t_lat), :, :].contiguous()

    ref_target_masks = torch.ones(3, lat_h, lat_w, device=device, dtype=param_dtype)
    seq_len = ((frame_num - 1) // 4 + 1) * lat_h * lat_w // (2 * 2)
    seq_len = int(math.ceil(seq_len / getattr(dmd_model.config, "sp_size", 1))) * getattr(dmd_model.config, "sp_size", 1)

    timesteps = _build_val_timesteps(dmd_model=dmd_model, args=args)

    context_null = None
    val_features_dir = getattr(args, "val_features_dir", None)
    if val_features_dir:
        # context_null.pt lives inside the val features directory, next to the
        # per-sample feature subdirectories (e.g.
        # processed_data/talkcuts/val/feature/context_null.pt for
        # val_features_dir=processed_data/talkcuts/val/feature).
        context_null_path = os.path.join(val_features_dir, "context_null.pt")
        if os.path.isfile(context_null_path):
            context_null = torch.load(context_null_path, map_location=device).to(dtype=param_dtype)
            if context_null.dim() == 2:
                context_null = context_null.unsqueeze(0)
            elif context_null.dim() == 3 and context_null.shape[0] != 1:
                context_null = context_null[:1]
            elif context_null.dim() != 3:
                raise ValueError(
                    f"context_null.pt has invalid shape {tuple(context_null.shape)}; expected [1,S,D] or [S,D]."
                )
        else:
            logging.warning("context_null.pt not found at %s; fallback to zero context.", context_null_path)
    if context_null is None:
        context_null = torch.zeros_like(context)

    max_frames_target = min(int(max_frame_num), int(max(1, meta.get("audio_total_frames", full_audio_emb.shape[0]))))
    clip_stride = frame_num - motion_frame

    is_first_clip = True
    audio_start_idx = 0
    generated_chunks: List[torch.Tensor] = []
    latent_motion_frames = None
    cur_motion_frames_num = 1
    done = False

    color_reference = None
    if color_correction_strength > 0.0:
        color_reference = _build_color_reference(sample_raw["video_path"], target_h, target_w, device=device)

    while not done:
        y = cond_latents
        cur_motion_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)

        audio_emb = _build_audio_windows(full_audio_emb, audio_start_idx, frame_num, device=device)
        audio_emb = audio_emb.unsqueeze(0).to(dtype=param_dtype)
        audio_null = torch.zeros_like(audio_emb)

        latent = torch.randn(1, 16, t_lat, lat_h, lat_w, dtype=param_dtype, device=device)
        if is_first_clip:
            latent_motion_frames = first_frame_latent.to(device=device, dtype=param_dtype).unsqueeze(0)

        with torch.no_grad():
            for i in range(len(timesteps) - 1):
                cur_t = timesteps[i]
                next_t = timesteps[i + 1]
                t_tensor = torch.tensor([cur_t], device=device)

                latent[:, :, :cur_motion_latent_num] = latent_motion_frames[:, :, :cur_motion_latent_num]

                v_cond = _to_tensor(
                    dmd_model.forward_wan(
                        dmd_model.generator,
                        latent,
                        t_tensor.long(),
                        context,
                        seq_len,
                        clip_fea,
                        y,
                        audio_emb,
                        ref_target_masks=ref_target_masks,
                    )
                )

                if getattr(args, "val_disable_cfg", False):
                    v = v_cond
                else:
                    text_cfg = float(getattr(args, "val_text_guide_scale", 5.0))
                    audio_cfg = float(getattr(args, "val_audio_guide_scale", 4.0))

                    # pass 1: drop text (keep audio + image)
                    v_drop_text = _to_tensor(
                        dmd_model.forward_wan(
                            dmd_model.generator, latent, t_tensor.long(),
                            context_null, seq_len, clip_fea, y, audio_emb,
                            ref_target_masks=ref_target_masks,
                        )
                    )
                    # pass 2: drop text + audio (keep image)
                    v_null_all = _to_tensor(
                        dmd_model.forward_wan(
                            dmd_model.generator, latent, t_tensor.long(),
                            context_null, seq_len, clip_fea, y, audio_null,
                            ref_target_masks=ref_target_masks,
                        )
                    )
                    # Dual CFG: null + audio_scale*(drop_text - null) + text_scale*(cond - drop_text)
                    v = (
                        v_null_all
                        + audio_cfg * (v_drop_text - v_null_all)
                        + text_cfg * (v_cond - v_drop_text)
                    )

                dt = (cur_t - next_t) / 1000.0
                latent = latent - dt * v
                latent[:, :, :cur_motion_latent_num] = latent_motion_frames[:, :, :cur_motion_latent_num]

            decoded = processor.vae.decode([latent[0]])[0].unsqueeze(0).cpu()

        if color_correction_strength > 0.0 and color_reference is not None:
            decoded = match_and_blend_colors(decoded, color_reference, color_correction_strength)

        if is_first_clip:
            generated_chunks.append(decoded)
        else:
            generated_chunks.append(decoded[:, :, cur_motion_frames_num:])

        gen_video_so_far = torch.cat(generated_chunks, dim=2)
        if gen_video_so_far.shape[2] >= max_frames_target:
            done = True
        else:
            is_first_clip = False
            cur_motion_frames_num = motion_frame
            cond_frame = gen_video_so_far[:, :, -cur_motion_frames_num:].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                latent_motion_frames = processor.vae.encode([cond_frame[0]])[0].unsqueeze(0).to(dtype=param_dtype)
            audio_start_idx += clip_stride
            if audio_start_idx >= max_frames_target:
                done = True

    gen_video = torch.cat(generated_chunks, dim=2)[:, :, :max_frames_target].squeeze(0).to(torch.float32)
    # Each rank saves the samples it generated, so distributed runs cover all samples without conflicts.
    save_video_ffmpeg(gen_video, output_mp4_no_ext, [sample_raw["audio_path"]], fps=25, high_quality_save=False)


def run_validation_inference_and_eval(
    dmd_model,
    processor,
    val_dataloader,
    args,
    step: int,
    rank: int,
) -> str:
    frame_num = int(getattr(args, "val_frame_num", 33))
    max_frame_num = int(getattr(args, "val_max_frame_num", 1000))
    motion_frame = int(getattr(args, "val_motion_frame", 9))
    color_correction_strength = float(getattr(args, "val_color_correction_strength", 0.0))

    val_root = os.path.join(args.output_dir, f"val_step_{step}")
    video_dir = os.path.join(val_root, "videos")
    os.makedirs(video_dir, exist_ok=True)

    device = next(dmd_model.parameters()).device
    original_vae_device = getattr(processor.vae, "device", None)
    move_VAE_to_device(processor, device)

    dmd_model.eval()
    with torch.no_grad():
        for val_batch_raw in val_dataloader:
            for sample in val_batch_raw:
                sample_id = _get_sample_id_from_video_path(sample["video_path"])
                feature_dir = os.path.join(args.val_features_dir, sample_id)
                required = [
                    os.path.join(feature_dir, "context.pt"),
                    os.path.join(feature_dir, "clip_fea.pt"),
                    os.path.join(feature_dir, "first_frame_latent.pt"),
                    os.path.join(feature_dir, "cond_latents.pt"),
                    os.path.join(feature_dir, "full_emb.pt"),
                    os.path.join(feature_dir, "metadata.json"),
                ]
                missing = [p for p in required if not os.path.isfile(p)]
                if missing:
                    if rank == 0:
                        logging.warning("Skip val sample %s due to missing features: %s", sample_id, missing)
                    continue

                out_mp4_no_ext = os.path.join(video_dir, sample_id)
                _infer_single_sample(
                    dmd_model=dmd_model,
                    processor=processor,
                    sample_raw=sample,
                    feature_dir=feature_dir,
                    args=args,
                    frame_num=frame_num,
                    max_frame_num=max_frame_num,
                    motion_frame=motion_frame,
                    color_correction_strength=color_correction_strength,
                    output_mp4_no_ext=out_mp4_no_ext,
                    rank=rank,
                )
                if rank == 0:
                    logging.info("Saved val video: %s.mp4", out_mp4_no_ext)

    if original_vae_device == "cpu":
        move_VAE_to_device(processor, "cpu")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        result_csv = os.path.join(video_dir, "result.csv")
        cmd = [
            sys.executable,
            "run_evaluate_gt_standalone.py",
            "--video_path",
            video_dir,
            "--result_file",
            result_csv,
        ]
        # Run eval as a plain single-process job.
        # If WORLD_SIZE is inherited from torchrun, transformers may switch
        # device_map='auto' into TP mode and fail on torch<2.5.
        eval_env = os.environ.copy()
        for k in [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
        ]:
            eval_env.pop(k, None)
        logging.info("Start eval command: %s", " ".join(cmd))
        subprocess.run(
            cmd,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=eval_env,
        )
        _append_average_row(result_csv)
        logging.info("Validation eval finished. Result saved to: %s", result_csv)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return video_dir
