OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone train_flashtalk_stage1.py \
  --config config/preprocess_val_example.yaml
