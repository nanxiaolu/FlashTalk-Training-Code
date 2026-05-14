import argparse
import io
import os
import re
import sys
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import src._warning_filters  # noqa: F401 - silence noisy 3rd-party warnings

import lmdb
import torch


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def _derive_default_output_path(input_lmdb_path, output_group_size):
    directory = os.path.dirname(input_lmdb_path)
    base = os.path.basename(input_lmdb_path)
    stem, ext = os.path.splitext(base)
    if ext == "":
        ext = ".lmdb"
    stem = re.sub(r"_gs\d+$", "", stem)
    output_name = f"{stem}_gs{int(output_group_size)}{ext}"
    return os.path.join(directory, output_name)


def _remove_existing_lmdb_files(lmdb_path):
    for p in (lmdb_path, f"{lmdb_path}-lock"):
        if os.path.isfile(p):
            os.remove(p)


def _extract_selected_k(raw_bytes, key_idx):
    obj = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(
            f"LMDB key={key_idx} payload must be dict-like, got {type(obj).__name__}"
        )
    if "selected_k" not in obj:
        raise KeyError(f"'selected_k' missing in LMDB key={key_idx}")
    return int(obj["selected_k"])


def repack_stage2_lmdb_group_size(
    input_lmdb_path,
    output_lmdb_path,
    input_group_size,
    output_group_size,
    map_size_gb=2048,
    write_batch_size=128,
    progress_interval=200,
):
    if int(input_group_size) <= 0:
        raise ValueError(f"input_group_size must be > 0, got {input_group_size}")
    if int(output_group_size) <= 0:
        raise ValueError(f"output_group_size must be > 0, got {output_group_size}")
    
    if not os.path.isfile(input_lmdb_path):
        raise FileNotFoundError(f"Input LMDB not found: {input_lmdb_path}")
    if os.path.abspath(input_lmdb_path) == os.path.abspath(output_lmdb_path):
        raise ValueError("output_lmdb_path must be different from input_lmdb_path")

    input_env = lmdb.open(
        input_lmdb_path,
        subdir=False,
        map_size=1,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )
    with input_env.begin(write=False) as txn:
        num_samples_raw = txn.get(b"__num_samples__")
    if num_samples_raw is None:
        input_env.close()
        raise RuntimeError(f"LMDB metadata '__num_samples__' not found in {input_lmdb_path}")
    input_num_samples = int(num_samples_raw.decode("utf-8"))

    print(f"[info] input_lmdb_path={input_lmdb_path}")
    print(f"[info] input_num_samples={input_num_samples}")
    print(f"[info] input_group_size={int(input_group_size)}, output_group_size={int(output_group_size)}")

    os.makedirs(os.path.dirname(output_lmdb_path) or ".", exist_ok=True)
    output_env = lmdb.open(
        output_lmdb_path,
        subdir=False,
        map_size=int(map_size_gb) * 1024 * 1024 * 1024,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )

    pending_kv = []
    written = 0

    def _flush_pending():
        nonlocal pending_kv
        if len(pending_kv) == 0:
            return
        with output_env.begin(write=True) as txn:
            for key_bytes, payload in pending_kv:
                txn.put(key_bytes, payload)
        pending_kv = []

    k_to_buffer = defaultdict(list)
    group_mismatch_count = 0
    first_group_mismatch_msg = None

    with input_env.begin(write=False) as txn:
        current_input_group_k = None
        for idx in range(int(input_num_samples)):
            raw = txn.get(str(int(idx)).encode("utf-8"))
            if raw is None:
                raise KeyError(f"LMDB key {idx} not found in {input_lmdb_path}")

            k_val = _extract_selected_k(raw, key_idx=idx)
            pos_in_group = idx % int(input_group_size)
            if pos_in_group == 0:
                current_input_group_k = k_val
            elif current_input_group_k != k_val:
                group_mismatch_count += 1
                if first_group_mismatch_msg is None:
                    first_group_mismatch_msg = (
                        f"[warn] input group mismatch detected at key={idx}: "
                        f"group start selected_k={current_input_group_k}, current selected_k={k_val}"
                    )

            buf = k_to_buffer[k_val]
            buf.append(raw)
            while len(buf) >= int(output_group_size):
                group_payloads = buf[: int(output_group_size)]
                del buf[: int(output_group_size)]
                for payload in group_payloads:
                    pending_kv.append((str(int(written)).encode("utf-8"), payload))
                    written += 1
                if len(pending_kv) >= int(write_batch_size):
                    _flush_pending()

            if (idx + 1) % int(progress_interval) == 0:
                print(f"[repack] scanned={idx + 1}, written={written}")

    dropped_tail = {}
    for k_val in sorted(k_to_buffer.keys()):
        remain = len(k_to_buffer[k_val])
        if remain > 0:
            dropped_tail[int(k_val)] = int(remain)

    _flush_pending()
    with output_env.begin(write=True) as txn:
        txn.put(b"__num_samples__", str(int(written)).encode("utf-8"))
    output_env.sync()
    output_env.close()
    input_env.close()

    if group_mismatch_count > 0:
        print(first_group_mismatch_msg)
        print(
            f"[warn] total input-group mismatch count={group_mismatch_count}. "
            "Conversion still completed by regrouping with selected_k."
        )

    dropped_total = sum(dropped_tail.values())
    if dropped_total > 0:
        print(
            "[warn] dropped samples because they cannot form a full output group: "
            f"total_dropped={int(dropped_total)}"
        )
        for k_val in sorted(dropped_tail.keys()):
            print(f"[warn] dropped_tail selected_k={k_val}: {dropped_tail[k_val]}")
    else:
        print("[info] no dropped tail samples.")

    print(f"[done] output_lmdb_path={output_lmdb_path}")
    print(f"[done] output_num_samples={int(written)}")
    print(f"[done] output_full_groups={int(written) // int(output_group_size)}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repack a stage2 LMDB from input_group_size to output_group_size by selected_k. "
            "Tail samples that cannot form a full output group are dropped."
        )
    )
    parser.add_argument(
        "--input_lmdb_path",
        type=str,
        required=True,
        help="Input LMDB path (e.g., stage2_sample_6400.lmdb).",
    )
    parser.add_argument(
        "--input_group_size",
        type=int,
        required=True,
        help="Original grouped size used by input LMDB (e.g., 8).",
    )
    parser.add_argument(
        "--output_group_size",
        type=int,
        required=True,
        help="Target grouped size for output LMDB (e.g., 32).",
    )
    parser.add_argument(
        "--output_lmdb_path",
        type=str,
        default=None,
        help=(
            "Output LMDB path. Default: same directory as input, name changed to *_gs<output_group_size>.lmdb."
        ),
    )
    parser.add_argument("--map_size_gb", type=int, default=2048, help="LMDB map_size in GB.")
    parser.add_argument("--progress_interval", type=int, default=200, help="Print progress every N scanned samples.")
    args = parser.parse_args()

    output_lmdb_path = args.output_lmdb_path or _derive_default_output_path(
        input_lmdb_path=args.input_lmdb_path,
        output_group_size=int(args.output_group_size),
    )

    repack_stage2_lmdb_group_size(
        input_lmdb_path=args.input_lmdb_path,
        output_lmdb_path=output_lmdb_path,
        input_group_size=int(args.input_group_size),
        output_group_size=int(args.output_group_size),
        map_size_gb=int(args.map_size_gb),
        write_batch_size=128,
        progress_interval=int(args.progress_interval),
    )


if __name__ == "__main__":
    main()
