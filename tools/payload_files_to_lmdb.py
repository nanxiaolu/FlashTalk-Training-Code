import argparse
import io
import json
import os
import random
import re
import sys
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import src._warning_filters  # noqa: F401 - silence noisy 3rd-party warnings

import lmdb
import torch


def _default_payload_dir_from_lmdb(lmdb_path):
    root, _ = os.path.splitext(lmdb_path)
    return f"{root}.payloads"


def _collect_payload_files(payload_dir):
    if not os.path.isdir(payload_dir):
        raise FileNotFoundError(f"Payload directory not found: {payload_dir}")

    payload_files = []
    for name in os.listdir(payload_dir):
        if not name.endswith(".payload"):
            continue
        key_str = name[: -len(".payload")]
        if not re.fullmatch(r"\d+", key_str):
            continue
        payload_files.append((int(key_str), os.path.join(payload_dir, name)))
    payload_files.sort(key=lambda x: x[0])
    if len(payload_files) == 0:
        raise RuntimeError(f"No '*.payload' files found in {payload_dir}")
    return payload_files


def _renumber_payload_files_contiguous(payload_dir):
    """
    Rename payload files in-place to contiguous keys [0, N-1].
    Example: 0.payload, 4.payload, 9.payload -> 0.payload, 1.payload, 2.payload
    """
    payload_files = _collect_payload_files(payload_dir)
    expected_keys = list(range(len(payload_files)))
    current_keys = [k for k, _ in payload_files]
    if current_keys == expected_keys:
        print(f"[pack] payload keys already contiguous in {payload_dir}, skip renaming.")
        return 0, len(payload_files)

    tmp_records = []
    for old_key, old_path in payload_files:
        tmp_path = os.path.join(payload_dir, f".renumber_tmp_{old_key}.payload")
        os.replace(old_path, tmp_path)
        tmp_records.append((old_key, tmp_path))

    renamed = 0
    for new_key, (old_key, tmp_path) in enumerate(tmp_records):
        new_path = os.path.join(payload_dir, f"{new_key}.payload")
        os.replace(tmp_path, new_path)
        if old_key != new_key:
            renamed += 1

    print(
        f"[pack] renumbered payload files in {payload_dir}: total={len(tmp_records)}, changed={renamed}"
    )
    return renamed, len(tmp_records)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def _deserialize_payload(raw_bytes):
    return torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)


def payload_files_to_lmdb(
    payload_dir,
    output_lmdb_path,
    num_samples=None,
    map_size_gb=2048,
    write_batch_size=128,
    shuffle_k_groups=True,
    group_size=4,
    shuffle_seed=2026,
):
    _renumber_payload_files_contiguous(payload_dir)

    payload_files_all = _collect_payload_files(payload_dir)
    max_key = payload_files_all[-1][0]
    inferred_num_samples = max_key + 1
    if num_samples is not None:
        num_samples = int(num_samples)
        if num_samples > inferred_num_samples:
            raise ValueError(
                f"num_samples={num_samples} is larger than available payload count "
                f"{inferred_num_samples}; preprocess more samples before packing."
            )
    else:
        num_samples = inferred_num_samples

    payload_files = payload_files_all
    if (not shuffle_k_groups) and num_samples < inferred_num_samples:
        # Truncate to the first ``num_samples`` payloads (lowest keys after renumbering).
        # This is the path used when preprocess overshoots and we want the final
        # LMDB sample count to exactly equal ``num_samples``.
        print(
            f"[pack] truncating payload list from {inferred_num_samples} to first "
            f"{num_samples} samples (target __num_samples__)."
        )
        payload_files = payload_files_all[:num_samples]

    if int(group_size) <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")

    os.makedirs(os.path.dirname(output_lmdb_path) or ".", exist_ok=True)
    env = lmdb.open(
        output_lmdb_path,
        subdir=False,
        map_size=int(map_size_gb) * 1024 * 1024 * 1024,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )

    pending = []
    written = 0

    def _flush():
        nonlocal pending
        if len(pending) == 0:
            return
        with env.begin(write=True) as txn:
            for key_bytes, payload in pending:
                txn.put(key_bytes, payload)
        pending = []

    if shuffle_k_groups:
        print(
            f"[pack] shuffle_k_groups enabled: group_size={int(group_size)}, seed={int(shuffle_seed)}. "
            "Reading payload selected_k ..."
        )
        k_to_payloads = defaultdict(list)  # k -> [raw_payload_bytes]
        scanned = 0
        for key_idx, payload_path in payload_files:
            with open(payload_path, "rb") as f:
                payload = f.read()
            obj = _deserialize_payload(payload)
            if "selected_k" not in obj:
                raise KeyError(f"'selected_k' missing in payload: {payload_path}")
            k_val = int(obj["selected_k"])
            k_to_payloads[k_val].append(payload)
            scanned += 1
            if scanned % 500 == 0:
                print(f"[pack] scanned payloads={scanned}")

        all_groups = []
        dropped_tail = {}
        for k_val in sorted(k_to_payloads.keys()):
            entries = k_to_payloads[k_val]
            full_groups = len(entries) // int(group_size)
            for g in range(full_groups):
                start = g * int(group_size)
                all_groups.append((k_val, entries[start:start + int(group_size)]))
            dropped = len(entries) - full_groups * int(group_size)
            if dropped > 0:
                dropped_tail[k_val] = dropped
            print(f"[pack] k={k_val}: total={len(entries)}, full_groups={full_groups}, dropped_tail={dropped}")

        rng = random.Random(int(shuffle_seed))
        rng.shuffle(all_groups)
        print(f"[pack] total_groups={len(all_groups)}")

        # Truncate to ``num_samples`` so the resulting LMDB matches the requested
        # final sample count even when preprocess produced extra payloads.
        target_write = int(num_samples)
        out_key_idx = 0
        stop_writing = False
        for _, group_payloads in all_groups:
            if stop_writing:
                break
            for payload in group_payloads:
                if out_key_idx >= target_write:
                    stop_writing = True
                    break
                pending.append((str(int(out_key_idx)).encode("utf-8"), payload))
                out_key_idx += 1
                if len(pending) >= int(max(1, write_batch_size)):
                    _flush()
                written += 1
                if written % 100 == 0:
                    print(f"[pack] written={written}, latest_key={out_key_idx - 1}")
        if written < target_write:
            raise ValueError(
                f"num_samples={target_write} requested, but only {written} payloads "
                "could be written after k-group shuffling; preprocess more samples or lower num_samples."
            )
        if dropped_tail:
            print(f"[pack] dropped tails (<group_size) by k: {json.dumps(dropped_tail, ensure_ascii=False)}")
    else:
        for key_idx, payload_path in payload_files:
            with open(payload_path, "rb") as f:
                payload = f.read()
            pending.append((str(int(key_idx)).encode("utf-8"), payload))
            if len(pending) >= int(max(1, write_batch_size)):
                _flush()
            written += 1
            if written % 100 == 0:
                print(f"[pack] written={written}, latest_key={key_idx}")

    _flush()
    final_num_samples = int(num_samples)
    if final_num_samples > int(written):
        raise ValueError(
            f"num_samples={final_num_samples} is larger than output payload count {int(written)}"
        )
    with env.begin(write=True) as txn:
        txn.put(b"__num_samples__", str(final_num_samples).encode("utf-8"))
    env.sync()
    env.close()

    print(f"[done] output_lmdb_path={output_lmdb_path}")
    print(f"[done] payload_files_written={written}")
    print(f"[done] __num_samples__={final_num_samples}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage-B: pack Stage-A payload files (<key_idx>.payload) into one LMDB."
    )
    parser.add_argument("--output_lmdb_path", type=str, required=True, help="Output LMDB path.")
    parser.add_argument(
        "--payload_dir",
        type=str,
        default=None,
        help="Payload directory. Default: <output_lmdb_path without ext>.payloads",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional __num_samples__ to write. Default inferred from max payload key + 1.",
    )
    parser.add_argument("--map_size_gb", type=int, default=2048, help="LMDB map_size in GB.")
    parser.add_argument("--write_batch_size", type=int, default=128, help="How many payloads per write transaction.")
    parser.add_argument(
        "--shuffle_k_groups",
        type=_str2bool,
        default=True,
        help="Shuffle writing order by full groups that share selected_k (default: true).",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Group size for selected_k-consistent batching (default: 8).",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=2026,
        help="Random seed used when shuffling k-groups.",
    )
    args = parser.parse_args()

    payload_dir = args.payload_dir or _default_payload_dir_from_lmdb(args.output_lmdb_path)

    payload_files_to_lmdb(
        payload_dir=payload_dir,
        output_lmdb_path=args.output_lmdb_path,
        num_samples=args.num_samples,
        map_size_gb=args.map_size_gb,
        write_batch_size=args.write_batch_size,
        shuffle_k_groups=bool(args.shuffle_k_groups),
        group_size=int(args.group_size),
        shuffle_seed=int(args.shuffle_seed),
    )


if __name__ == "__main__":
    main()
