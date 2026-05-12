# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp
import cv2

import imageio
import torch
import torchvision
from PIL import Image
import librosa
import soundfile as sf
import subprocess
from decord import VideoReader, cpu
import gc

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    FullyShardedDataParallel as FSDP
)
import torch.distributed as dist
from functools import partial
import numpy as np
import random
from datetime import timedelta

__all__ = ['cache_video', 'cache_image', 'str2bool']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name



def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
    
def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue

def convert_video_to_h264(input_video_path, output_video_path):
    subprocess.run(
        ['ffmpeg', '-i', input_video_path, '-c:v', 'libx264', '-c:a', 'copy', output_video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def is_video(path):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
    return os.path.splitext(path)[1].lower() in video_exts


def extract_specific_frames(video_path, frame_id):
    if is_video(video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        if frame_id < vr._num_frame:
            frame = vr[frame_id].asnumpy()  # RGB
        else:
            frame = vr[-1].asnumpy()
        del vr
        gc.collect()
        frame = Image.fromarray(frame)
    else:
        frame = Image.open(video_path).convert("RGB")
    return frame

def get_video_codec(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    codec = result.stdout.decode().strip()
    return codec



def split_wav_librosa(wav_path, segments, save_dir):
    y, sr = librosa.load(wav_path, sr=None)
    filename = wav_path.split('/')[-1].split('.')[0]
    save_list = []
    for idx, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        out_path = os.path.join(save_dir, filename + str(start) + '_' + str(end) + '.wav')
        sf.write(out_path, segment, sr)
        print(f"Saved {out_path}: {start}s to {end}s")
        save_list.append(out_path)
    return save_list



def launch_distributed_job(backend: str = "nccl", timeout_minutes: int = 30):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=int(timeout_minutes)),
    )
    torch.cuda.set_device(local_rank)


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        

def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, off_load_to_cpu=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        cpu_offload=CPUOffload(offload_params=off_load_to_cpu),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def barrier():
    if dist.is_initialized():
        dist.barrier()