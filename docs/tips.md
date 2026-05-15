# Tips and Common Pitfalls

[Chinese version](tips-zh-CN.md)

This document summarizes the main issues we encountered while reproducing FlashTalk and building its training pipeline. It covers **implementation differences between this repository and the official FlashTalk paper**, explains why those changes were made, and records several practical recommendations for training configuration.

---

## 1. Key differences from the official FlashTalk implementation

During reproduction, we found that following the paper details literally led to several serious generation-quality issues. To achieve stable, high-quality long-form video generation, we introduced the following changes.

### 1.1 Fixing visual jumps at long-video window boundaries

**Problem**: When Stage 2 generates long videos, obvious discontinuities and jumps can appear at the boundary between adjacent windows.

**Improvement**: **Do not add noise to the teacher model's motion latent**. If the teacher motion latent is noised, the denoised teacher motion latent can differ too much from the student motion latent. The teacher real score then does not splice smoothly with the student motion latent, and training transfers this discontinuity to window boundaries. Removing noise from the motion latent sent to the teacher resolves this issue.

### 1.2 Preventing persistent one-direction camera drift

**Problem**: After DMD distillation, the Stage 2 model may generate videos where the camera drifts continuously in one direction, even though the Stage 1 model does not show this behavior.

**Improvement**: **Constrain the student's motion latent to align with the teacher's motion latent**. Although the ablation in Figure 3 of the official paper claims that motion-latent alignment is unnecessary, we found that without it the student's motion latent is almost unconstrained. Drift noise can be amplified during autoregressive rollout and inherited by the next window through temporal consistency. We explicitly add a student-teacher motion-latent alignment constraint, which keeps multi-minute generated videos visually stable.

### 1.3 Improving background detail consistency in minute-long videos

**Problem**: For a small number of out-of-distribution samples, the background texture may change slightly when generating videos lasting several minutes.

**Reason and change**: To further stabilize background details and support truly "infinite" generation, we add a dedicated temporal loss during DMD training.

- **Recommended approach**: Use a portrait matting model to extract the background region for every frame, and expand the portrait contour slightly to ensure the selected region is pure background. Concatenate the reference frame to both ends of the video sequence as anchors, then constrain the difference between adjacent background regions. See `_compute_rvm_background_overlap_mse` in `flashtalk_dmd.py`. This keeps the background texture strictly consistent with the reference while preserving smooth temporal transitions between frames.
- **Pitfall to avoid**: Do not directly align every generated frame's background to the reference-frame background. In our experiments, this destroys temporal continuity and causes severe flickering.

---

## 2. Training configuration and tuning recommendations

Beyond algorithmic changes, we found several important practical rules while tuning training.

### 2.1 Enable static-background filtering for the dataset (required)

- **Problem**: If the generated camera keeps shaking, the original training videos may already contain background motion.
- **Approach**: To keep the training data stable, we use RVM for foreground/background segmentation and keep a sample only when the background SSIM is greater than 0.96. The `enable_background_filter` switch exposes this filtering step in Stage 1 preprocessing.
- **Effect**: Removing samples with shaking backgrounds significantly improves the stability of both Stage 1 and Stage 2.

### 2.2 Reference-frame sampling strategy (adopt InfiniteTalk's approach, required)

- **Problem**: We initially used the first video frame as the reference frame, matching the inference setup. However, during training the motion latent is also the first frame and is noise-free, so the reference frame is quickly "masked out" and cannot anchor the image, which leads to stability issues.
- **Approach**: We adopt the **InfiniteTalk** strategy and randomly select a frame from a neighboring window as the reference frame.
- **Effect**: This greatly improves visual stability.

### 2.3 Do not use LoRA; full-parameter fine-tuning is required

- **Problem**: We spent a long time trying LoRA training. With the current number of iterations (1,000 for Stage 1 and 200 for Stage 2 in the original setup), reducing the learning rate had almost no effect, while increasing it made the model worse.
- **Reason**: LoRA typically needs much longer training schedules to work well, such as LiveAvatar's 27.5k steps with batch size 128. Our goal is fast convergence on a large dataset within roughly 1k + 200 steps, where LoRA is ineffective.
- **Conclusion**: **This setup requires full-parameter fine-tuning**. The default Stage 1 and Stage 2 configurations in this repository both use full-parameter fine-tuning.

### 2.4 Small datasets, such as one-person fine-tuning, need very few steps

- **Observation**: For small one-person datasets of around 500 videos, it is best to continue training from our large-dataset pretrained model and run only a very small number of additional steps. Training from scratch on a small dataset performs poorly.
- **Recommendation**: For example, Stage 1 and Stage 2 usually need only **10 extra iterations** each, equivalent to `(10 + 10 // 5) * batchsize` additional samples. Avoid overtraining, because too many steps often degrade model quality.

### 2.5 Per-GPU batch size

- **Note**: In our training runs, `per-GPU-batchsize` has only been tested with the value **1**.
- **Recommendation**: We do not yet know whether increasing this parameter introduces unknown issues or affects convergence. If you try a larger value, monitor training closely.

### 2.6 Training-step differences from the official FlashTalk paper

- **Explanation**: Because our dataset and some hyperparameters differ from the official FlashTalk setup, the number of iterations needed for convergence also differs from the paper.
- **Recommendation**: When training on your own dataset, do not copy either our default step counts or the paper's step counts blindly. Adjust the total number of iterations according to your dataset size and observed convergence.

