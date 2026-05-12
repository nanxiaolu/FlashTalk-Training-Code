#!/usr/bin/env python3
"""
Export a single-file generator_*.safetensors checkpoint into HF-style sharded
diffusion_pytorch_model-XXXXX-of-YYYYY.safetensors + index.json.

All floating dtypes (F32/F16/F64/BF16) are stored as BF16; integer/bool dtypes
are kept as-is.

Two modes:
1) Auto (default): bucket tensors by stored bytes; keys are sorted to make the
   split reproducible.
2) --index_path: shard tensors using the weight_map of an existing
   diffusion_pytorch_model.safetensors.index.json (keys must match exactly).
"""
import argparse
import json
import logging
import math
import os
import shutil
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_FLOAT_STO_AS_BF16 = frozenset({"F32", "F16", "BF16", "F64"})

_DTYPE_ELEMENT_NBYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _slice_export_storage_nbytes(sf, key: str) -> int:
    """Stored bytes: floats are counted as BF16 (2 bytes/elem); other dtypes use their native element size."""
    sl = sf.get_slice(key)
    shape = sl.get_shape()
    dtype = str(sl.get_dtype())
    n = 1
    for d in shape:
        n *= int(d)
    if dtype in _FLOAT_STO_AS_BF16:
        return n * 2
    el = _DTYPE_ELEMENT_NBYTES.get(dtype)
    if el is None:
        raise ValueError(f"Unknown dtype {dtype!r} for key {key!r}")
    return n * el


def _to_stored_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16):
        return t.to(dtype=torch.bfloat16)
    return t


def _collect_keys_and_sizes(src_path: str) -> Tuple[List[str], Dict[str, int]]:
    with safe_open(src_path, framework="pt") as sf:
        keys = sorted(sf.keys())
        sizes = {k: _slice_export_storage_nbytes(sf, k) for k in keys}
    return keys, sizes


def _shard_keys_by_size(keys: List[str], sizes: Dict[str, int], num_shards: int) -> List[List[str]]:
    """Bucket sorted keys into shards balanced by stored byte size."""
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    if len(keys) == 0:
        return []
    # Never produce more shards than keys, otherwise some shards would be empty.
    target_shards = min(num_shards, len(keys))
    total_size = sum(sizes[k] for k in keys)
    target_per_shard = int(math.ceil(total_size / float(target_shards)))
    shards: List[List[str]] = []
    cur: List[str] = []
    cur_size = 0
    remaining = len(keys)

    for k in keys:
        v_size = sizes[k]
        # At most close (target_shards - 1) shards early; the last shard takes the remainder.
        can_close_current = len(shards) < (target_shards - 1)
        # Keep the current shard open if remaining keys exactly equal the remaining shard slots.
        must_keep_current_open = can_close_current and remaining <= (target_shards - len(shards))
        if cur and can_close_current and (cur_size + v_size > target_per_shard) and not must_keep_current_open:
            shards.append(cur)
            cur = []
            cur_size = 0
        cur.append(k)
        cur_size += v_size
        remaining -= 1
    if cur:
        shards.append(cur)
    if len(shards) > target_shards:
        raise RuntimeError(f"Internal shard split error: got {len(shards)} shards > target {target_shards}")
    return shards


def _write_shards_from_groups(
    src_path: str,
    output_dir: str,
    key_groups: List[List[str]],
    weight_map: Dict[str, str],
    total_size: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="reshard_diffusion_", dir=output_dir)
    try:
        real_n = len(key_groups)
        for shard_idx, keys in enumerate(key_groups, start=1):
            fname = f"diffusion_pytorch_model-{shard_idx:05d}-of-{real_n:05d}.safetensors"
            shard_sd = {}
            with safe_open(src_path, framework="pt") as sf:
                for k in keys:
                    shard_sd[k] = _to_stored_tensor(sf.get_tensor(k).clone())
            tmp_shard = os.path.join(tmp, fname)
            save_file(shard_sd, tmp_shard, metadata={"format": "pt"})
            logging.info("Built %s (%d tensors)", fname, len(shard_sd))
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(dst):
                os.remove(dst)
            shutil.move(tmp_shard, dst)
            logging.info("Installed %s", dst)

        index_obj = {
            "metadata": {"total_size": int(total_size)},
            "weight_map": weight_map,
        }
        tmp_index = os.path.join(tmp, "diffusion_pytorch_model.safetensors.index.json")
        with open(tmp_index, "w", encoding="utf-8") as f:
            json.dump(index_obj, f, ensure_ascii=False, indent=2)
        final_index = os.path.join(output_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.isfile(final_index):
            os.remove(final_index)
        shutil.move(tmp_index, final_index)
        logging.info("Installed %s", final_index)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def export_auto(src_path: str, output_dir: str, num_shards: int) -> None:
    keys, sizes = _collect_keys_and_sizes(src_path)
    groups = _shard_keys_by_size(keys, sizes, num_shards=num_shards)
    if not groups:
        raise RuntimeError("Empty checkpoint, nothing to export.")
    real_n = len(groups)
    weight_map: Dict[str, str] = {}
    total_size = sum(sizes[k] for k in keys)
    for shard_idx, group in enumerate(groups, start=1):
        fname = f"diffusion_pytorch_model-{shard_idx:05d}-of-{real_n:05d}.safetensors"
        for k in group:
            weight_map[k] = fname
    _write_shards_from_groups(src_path, output_dir, groups, weight_map, total_size)


def reshard_from_existing_index(src_path: str, output_dir: str, index_path: str) -> None:
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]

    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key, fname in weight_map.items():
        shard_to_keys[fname].append(key)

    with safe_open(src_path, framework="pt") as sf:
        src_keys = set(sf.keys())
    missing = set(weight_map.keys()) - src_keys
    if missing:
        raise KeyError(f"{len(missing)} keys in index but missing in src, e.g. {next(iter(missing))!r}")
    extra = src_keys - set(weight_map.keys())
    if extra:
        raise KeyError(f"{len(extra)} keys in src but not in index, e.g. {next(iter(extra))!r}")

    groups = [shard_to_keys[fname] for fname in sorted(shard_to_keys.keys())]
    total_size = 0
    with safe_open(src_path, framework="pt") as sf:
        for k in weight_map:
            total_size += _slice_export_storage_nbytes(sf, k)
    _write_shards_from_groups(src_path, output_dir, groups, dict(weight_map), total_size)


def main():
    parser = argparse.ArgumentParser(
        description="Export generator_*.safetensors to diffusion_pytorch_model-*****-of-*****.safetensors shards."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Single-file generator safetensors checkpoint to export.",
    )
    parser.add_argument("--output_dir", type=str, default="models_inference", help="Output directory for shards + index.")
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="If set, shard tensors exactly as this index weight_map (file must exist).",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="Target shard count when not using --index_path (default 4).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = os.path.abspath(args.output_dir)
    src = os.path.abspath(args.src)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    if args.index_path:
        idx = os.path.abspath(args.index_path)
        if not os.path.isfile(idx):
            raise FileNotFoundError(idx)
        reshard_from_existing_index(src, out, idx)
    else:
        export_auto(src, out, num_shards=int(args.num_shards))


if __name__ == "__main__":
    main()
