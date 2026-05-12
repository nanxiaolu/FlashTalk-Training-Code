# Validation

Both Stage-1 and Stage-2 validation are driven by
**`train_flashtalk_stage2.py`** with `val_only: true`. This is intentional
— validation reuses the Stage-2 inference path (CFG handling, denoising
step list, motion injection) regardless of which stage you trained. The
only thing that differs between Stage-1 and Stage-2 validation is **how
the generator weights are loaded**:

| Validation target | YAML | Weight loading mechanism |
|---|---|---|
| Stage-1 generator | `config/val_stage1.yaml` | `init_stage1_full: <path to model_xxx.safetensors>` — loads *only* the generator weights from a single Stage-1 safetensors file. Real-score / fake-score are not used. |
| Stage-2 generator | `config/val_stage2.yaml` | `resume_from: <checkpoint dir>` — loads `generator_xxx.safetensors` (and ignores critic at inference time). |

## Bundled 12-clip val set

`processed_data/example/val/{video,audio}/` ships with 12 TalkCuts clips
and a matching `processed_data/talkcuts/val_data.csv`. The per-sample
precomputed features (loaded from `val_features_dir`) live in
`processed_data/talkcuts/val/feature/<sample_id>/`, and the shared
`context_null.pt` sits at
`processed_data/talkcuts/val/feature/context_null.pt`. Use them to
sanity-check your trained weights with no extra setup:

```bash
# Stage 1
$EDITOR config/val_stage1.yaml   # set init_stage1_full
bash script/val_stage1.sh

# Stage 2
$EDITOR config/val_stage2.yaml   # set resume_from
bash script/val_stage2.sh
```

Outputs:

```
outputs/flashtalk_val_stage{1,2}/<auto_run_name>/<timestamp>/
└── val_step_<step>/videos/<sample_id>.mp4
```

## Validating on your own clips

Edit the `val_annotation_file` in either YAML to point at your own
3-column CSV (`video,input_audio,prompt`) and set `dataset_dir` if your
CSV paths aren't relative to the project root.

## Inference settings cheat-sheet

The 4-step inference targeted by Stage 2:

```yaml
denoising_step_list: "1000,750,500,250"
val_disable_cfg: false        # set true for true CFG-free inference
val_text_guide_scale: 3.0
val_audio_guide_scale: 4.0
val_frame_num: 33
val_motion_frame: 5
use_inject_motion_frames: true
```

To replicate the *teacher* / Stage-1 inference path (~40 steps, full CFG):

```yaml
val_infer_steps: 40
denoising_step_list: ""   # disabled when val_infer_steps is set
```

`val_disable_cfg: true` is useful for measuring how much CFG the model
still relies on after Stage-2 distillation; in a fully CFG-free
checkpoint the disabling should not visibly degrade output.
