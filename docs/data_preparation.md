# Preparing your own data

The end-to-end data pipeline has two stages, both implemented with
`torchrun`-spawned workers and synchronized via `torch.distributed`:

```
              ┌───────────────────────────┐         ┌──────────────────────┐
 raw mp4/wav  │ preprocess (per-rank)     │ payload │ pack (single-process)│  LMDB
─────────────►│ writes one .pt per sample ├────────►│ payloads -> LMDB     │──────►
              └───────────────────────────┘  files  └──────────────────────┘
```

A "payload" is a single `.pt` file in
`<payload_dir>/<key>.payload` (e.g.
`processed_data/talkcuts/train/stage1_sample_25030.payloads/<key>.payload`
or `processed_data/example/train/example_stage1.payloads/<key>.payload`)
storing the encoded VAE
latents, CLIP / T5 features, wav2vec audio features, face mask, motion
metadata, and (for Stage 2) the selected K for that sample. The packer
collects these into a single LMDB keyed by `0..N-1` for fast random access
during training.

The same Python entrypoints (`train_flashtalk_stage1.py` and
`train_flashtalk_stage2.py`) drive both preprocess and training; the
config toggles `mode` between `preprocess` and `train`.

---

## Input layout

Your dataset must be described by a CSV with these three columns (no other
columns are read, but extra columns won't hurt):

```
video,input_audio,prompt
relative/or/absolute/path/to/clip_001.mp4,relative/or/absolute/path/to/clip_001.wav,"A man in a blue shirt gestures with his hands while explaining a chart..."
...
```

* `video`: an mp4 file, ~33 frames is what training consumes per window
  (longer is fine; we sub-sample windows internally).
* `input_audio`: a wav file, mono, 16 kHz preferred (we resample if not).
* `prompt`: the text prompt fed to T5; in the released TalkCuts split we
  used Qwen-VL-generated descriptions of the scene.

If `dataset_dir` in your preprocess YAML is **empty** (default), the CSV
paths are resolved relative to the project root. If `dataset_dir` is
non-empty, paths are resolved relative to that directory.

---

## Stage-1 preprocess

```bash
# 1. Copy the example preprocess config and edit it:
#    - lmdb_num_samples:       how many samples you ultimately want in the LMDB
#                              (this is the count AFTER filtering, see below)
#    - payload_dir:            where to dump per-sample .pt files
#    - annotation_file:        your CSV
#    - dataset_dir:            "" to use CSV-relative paths, or a base dir
#    - enable_background_filter: true if you want to drop samples whose
#                                background moves too much (uses RVM + SSIM)
#    - background_ssim_threshold: only effective when the flag above is on
cp config/preprocess_stage1_example.yaml config/preprocess_stage1.yaml
$EDITOR config/preprocess_stage1.yaml

# 2. Launch on 8 GPUs (adjust --nproc_per_node if you have fewer)
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
  train_flashtalk_stage1.py --config config/preprocess_stage1.yaml
```

### About `lmdb_num_samples`

`lmdb_num_samples` is the **final** count of samples that successfully made
it through preprocessing. Each rank loops over its data shard until the
*global* number of "useful" samples (written + already-on-disk) reaches
this target. Samples that are skipped — either because of the background
filter, missing audio, decode errors, etc. — do **not** count.

This means that for `lmdb_num_samples=25030` and 8 ranks with
`barrier_interval=20`, the *actual* on-disk payload count after the run
will be at most `25030 + 8 * 20 = 25190` (over-shoot caused by the
synchronization granularity). The packer truncates to exactly
`num_samples` during stage B.

### Pack stage-1 LMDB

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir      processed_data/talkcuts/train/my_stage1.payloads \
    --output_lmdb_path processed_data/talkcuts/train/my_stage1.lmdb \
    --num_samples      25030 \
    --shuffle_k_groups false
```

For Stage 1, `--shuffle_k_groups false` is correct: there is no K parameter
per sample. The packer renumbers payload keys to `0..N-1` if they are
non-contiguous, and truncates to `--num_samples` if you generated more
than you need.

---

## Stage-2 preprocess

Stage 2 uses *self-forcing chunked* training: each sample stores a
`selected_k ∈ {1..K_max}` value that determines how many 33-frame chunks
the student rolls out during a single training step. Preprocessing must
therefore (a) re-cut each clip into overlapping chunks and (b) assign K
values so that the dataset reaches `quota_per_k` samples per K.

```bash
cp config/preprocess_stage2_example.yaml config/preprocess_stage2.yaml
$EDITOR config/preprocess_stage2.yaml
# Key fields to edit:
#   lmdb_num_samples:           final target count
#   payload_dir:                where to dump per-sample .pt files
#   stage2_k_max:               5 for production datasets; 2 is fine for the demo
#   stage2_chunk_frames:        33 (we never change this)
#   stage2_chunk_overlap_frames:5  (we never change this either)

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
  train_flashtalk_stage2.py --config config/preprocess_stage2.yaml
```

The synchronization mechanism is the same as Stage 1: all ranks
`all_reduce` their "written + skip_exists" counts and continue until the
global total reaches `lmdb_num_samples`. The K distribution snapshot is
also logged (e.g. `Stage2 K counts target snapshot: {1: 1280, 2: 1280, 3:
1280, 4: 1280, 5: 1280}` for `K_max=5` and `num_samples=6400`).

### Pack stage-2 LMDB — **read this**

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir      processed_data/talkcuts/train/my_stage2.payloads \
    --output_lmdb_path processed_data/talkcuts/train/my_stage2.lmdb \
    --num_samples      6400 \
    --shuffle_k_groups true \
    --group_size       8
```

* `--shuffle_k_groups true`: writing order is shuffled at the *group*
  level (not per sample) so that no two adjacent samples in the LMDB share
  audio / motion metadata (which would correlate gradients in a single
  batch).
* `--group_size 8` **must equal the number of GPUs** you intend to train
  on. See [stage2_k_grouping.md](stage2_k_grouping.md) for the full
  rationale. If you re-shard later, repack.

Note: chunked samples that share a `selected_k` are written together in
one group, so the packer reports e.g.
`k=1: total=1280, full_groups=160, dropped_tail=0`. "Dropped tails" are
the last `n < group_size` samples of each K bucket that don't fill a full
group — they're skipped to keep every batch K-homogeneous.

---

## Smoke test on the bundled example

The repo ships with 32 training + 12 validation clips under
`processed_data/example/{train,val}/{video,audio}/`. Use them to
sanity-check the pipeline before pointing it at the full dataset.

```bash
# Preprocess + pack stage 1 on the 32-clip example
bash script/preprocess_stage1.sh
bash script/pack_stage1.sh

# Preprocess + pack stage 2 on the same 32 clips
bash script/preprocess_stage2.sh
bash script/pack_stage2.sh
```

Each preprocess takes ~5 min on 8×A800 (model loading dominates over the
actual 32 samples). Output goes to
`processed_data/example/train/example_stage{1,2}.{payloads/,lmdb}`. These
LMDBs are far too small to train meaningful checkpoints — they exist
only so you can see the pipeline run.
