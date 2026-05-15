"""
Data processing module for FlashTalk training.
Contains DataProcessor, FlashTalkDataset, and helper functions for loading/processing data.
"""
import gc
import importlib
import io
import json
import logging
import csv
import math
import os
import random
import re
import sys
import fcntl
from pathlib import Path

import cv2
import librosa
import numpy as np
import pyloudnorm as pyln
import torch
from einops import rearrange
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Wav2Vec2FeatureExtractor

from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE

# Ensure project-root packages (e.g. rvm_model/) are importable when PYTHONPATH only contains src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from rvm_model import MattingNetwork  # pyright: ignore[reportMissingImports]


def move_CLIP_VAE_to_device(processor, device):
    if device == 'cpu':
        processor.clip.model.to('cpu')
        processor.vae.model.to('cpu')
        gc.collect()
    else:
        processor.clip.model.to(device)
        processor.clip.device = device
        processor.vae.model.to(device)
        processor.vae.device = device

def move_VAE_to_device(processor, device):
    if device == 'cpu':
        processor.vae.model.to('cpu')
        gc.collect()
    else:
        processor.vae.model.to(device)
        processor.vae.device = device

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def resize_and_centercrop(cond_image, target_size):
    """
    Resize image or tensor to the target size without padding.
    """
    if isinstance(cond_image, torch.Tensor):
        _, orig_h, orig_w = cond_image.shape
    else:
        orig_h, orig_w = cond_image.height, cond_image.width

    target_h, target_w = target_size
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    scale = max(scale_h, scale_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)

    if isinstance(cond_image, torch.Tensor):
        if len(cond_image.shape) == 3:
            cond_image = cond_image[None]
        resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous()
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor.squeeze(0)
    else:
        resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_image = np.array(resized_image)
        resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor[:, :, None, :, :]

    return cropped_tensor


def process_image_to_tensor(cond_image, target_size, device):
    cond_image = resize_and_centercrop(cond_image, target_size)
    cond_image = cond_image / 255
    cond_image = (cond_image - 0.5) * 2
    cond_image = cond_image.to(device)
    return cond_image


def process_video_frames_to_tensor(frames_np, target_size):
    """
    Convert video frames [T,H,W,C] uint8 to normalized tensor [C,T,H,W] in [-1,1]
    using the same resize+center-crop policy as reference-frame preprocessing.
    """
    if frames_np is None or len(frames_np.shape) != 4:
        raise ValueError(f"Expected frames_np shape [T,H,W,C], got {None if frames_np is None else frames_np.shape}")
    if int(frames_np.shape[-1]) != 3:
        raise ValueError(f"Expected RGB frames with C=3, got shape {frames_np.shape}")

    target_h, target_w = map(int, target_size)
    src_h, src_w = int(frames_np.shape[1]), int(frames_np.shape[2])
    scale_h = target_h / src_h
    scale_w = target_w / src_w
    scale = max(scale_h, scale_w)
    final_h = int(math.ceil(scale * src_h))
    final_w = int(math.ceil(scale * src_w))

    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()  # [T,C,H,W]
    frames_tensor = nn.functional.interpolate(
        frames_tensor, size=(final_h, final_w), mode="bilinear", align_corners=False
    )

    top = max(0, (final_h - target_h) // 2)
    left = max(0, (final_w - target_w) // 2)
    frames_tensor = frames_tensor[:, :, top : top + target_h, left : left + target_w]
    frames_tensor = frames_tensor / 127.5 - 1.0
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
    return frames_tensor


def process_video_masks_to_tensor(mask_np, target_size):
    """
    Convert binary mask frames [T,H,W] to tensor [1,T,H,W] with
    the same resize+center-crop geometry as process_video_frames_to_tensor.
    """
    if mask_np is None or len(mask_np.shape) != 3:
        raise ValueError(f"Expected mask_np shape [T,H,W], got {None if mask_np is None else mask_np.shape}")

    target_h, target_w = map(int, target_size)
    src_h, src_w = int(mask_np.shape[1]), int(mask_np.shape[2])
    scale_h = target_h / src_h
    scale_w = target_w / src_w
    scale = max(scale_h, scale_w)
    final_h = int(math.ceil(scale * src_h))
    final_w = int(math.ceil(scale * src_w))

    masks_tensor = torch.from_numpy(mask_np).unsqueeze(1).float()  # [T,1,H,W]
    masks_tensor = nn.functional.interpolate(
        masks_tensor, size=(final_h, final_w), mode="nearest"
    )
    top = max(0, (final_h - target_h) // 2)
    left = max(0, (final_w - target_w) // 2)
    masks_tensor = masks_tensor[:, :, top : top + target_h, left : left + target_w]
    masks_tensor = masks_tensor.permute(1, 0, 2, 3).contiguous()  # [1,T,H,W]
    masks_tensor = (masks_tensor > 0.5).float()
    return masks_tensor


def align_face_masks_to_latent(face_masks, latent_shape):
    """
    Align face masks from pixel space [1,T,H,W] to latent space [1,T_lat,H_lat,W_lat]
    using actual latent shape produced by WanVAE.
    """
    if face_masks is None or face_masks.ndim != 4:
        raise ValueError(
            f"Expected face_masks shape [1,T,H,W], got {None if face_masks is None else tuple(face_masks.shape)}"
        )
    if len(latent_shape) != 3:
        raise ValueError(f"Expected latent_shape=(T,H,W), got {latent_shape}")

    lat_t, lat_h, lat_w = map(int, latent_shape)
    aligned = nn.functional.interpolate(
        face_masks.unsqueeze(0),
        size=(lat_t, lat_h, lat_w),
        mode="nearest",
    ).squeeze(0)
    return (aligned > 0.5).float()


class FaceMaskExtractor:
    """
    Face-mask extractor aligned with ref_code/face_mask_extraction.py:
    1) Try insightface.FaceAnalysis first.
    2) Fallback to facexlib FaceRestoreHelper detector.
    3) If both fail to detect any face, return full-frame mask with no-face value.
    """

    def __init__(self, det_score_thresh=0.97, device="cuda", det_size=(640, 640), no_face_fill_value=0, rank=0):
        from facexlib.parsing import init_parsing_model
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from insightface.app import FaceAnalysis   # pyright: ignore[reportMissingImports]

        self.det_score_thresh = float(det_score_thresh)
        self.no_face_fill_value = np.uint8(no_face_fill_value)
        self.mask_positive_value = np.uint8(255)

        providers = [("CUDAExecutionProvider", {"device_id": rank}), "CPUExecutionProvider"]
        self.face_app = FaceAnalysis(name="antelopev2", root="weights/insightface", providers=providers)
        self.face_app.prepare(
            ctx_id=rank,
            det_size=(int(det_size[0]), int(det_size[1])),
        )
        
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=device,
            model_rootpath="weights/face_det"
        )
        # Keep ref init behavior in sync, although parsing is not used in mask extraction here.
        self.face_helper.face_parse = init_parsing_model(model_name="bisenet", device=device, model_rootpath="weights/face_det")

    def _build_mask_from_bboxes(self, frame_shape, bboxes):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if bboxes is None:
            return mask

        for bbox in bboxes:
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            x1 = int(max(0, min(w, round(float(x1)))))
            y1 = int(max(0, min(h, round(float(y1)))))
            x2 = int(max(0, min(w, round(float(x2)))))
            y2 = int(max(0, min(h, round(float(y2)))))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = self.mask_positive_value
        return mask

    def _extract_single(self, frame_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 1) insightface first
        try:
            image_info = self.face_app.get(frame_bgr)
        except Exception as e:
            if not hasattr(self, "_insightface_error_logged"):
                logging.warning("InsightFace detection failed, fallback to FaceRestoreHelper. First error: %s", e)
                self._insightface_error_logged = True
            image_info = []

        if image_info is not None and len(image_info) > 0:
            bboxes = [info["bbox"] for info in image_info if "bbox" in info]
            if len(bboxes) > 0:
                return self._build_mask_from_bboxes(frame_rgb.shape, bboxes)

        # 2) FaceRestoreHelper fallback
        try:
            self.face_helper.clean_all()
            with torch.no_grad():
                bboxes = self.face_helper.face_det.detect_faces(frame_bgr, self.det_score_thresh)
        except Exception as e:
            if not hasattr(self, "_face_det_error_logged"):
                logging.warning("FaceRestoreHelper detection failed. First error: %s", e)
                self._face_det_error_logged = True
            bboxes = None

        if bboxes is not None and len(bboxes) > 0:
            return self._build_mask_from_bboxes(frame_rgb.shape, bboxes)

        # 3) no face from both detectors -> full-frame no-face value
        h, w = frame_rgb.shape[:2]
        mask = np.empty((h, w), dtype=np.uint8)
        mask[:] = self.no_face_fill_value
        return mask

    def extract_video_masks(self, frames_np):
        masks = []
        for i in range(frames_np.shape[0]):
            cur = self._extract_single(frames_np[i])
            if cur is None:
                h, w = frames_np[i].shape[:2]
                cur = np.empty((h, w), dtype=np.uint8)
                cur[:] = self.no_face_fill_value
            masks.append(cur)
        return np.stack(masks, axis=0)


class FlashTalkDataset(Dataset):
    def __init__(self, dataset_dir, annotation_file):
        """
        Dataset for loading TalkCuts training data from a CSV annotation file.

        CSV format: video,input_audio,prompt
        - video: relative path under dataset_dir
        - input_audio: relative path under dataset_dir
        - prompt: text description
        """
        self.dataset_dir = dataset_dir
        self.annotation_path = annotation_file
        self.data = []

        if not self.annotation_path or not os.path.exists(self.annotation_path):
            logging.warning(f"Annotation file not found: {self.annotation_path}")
            return

        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    "video": os.path.join(dataset_dir, row.get("video")) if dataset_dir else row.get("video"),
                    "audio": os.path.join(dataset_dir, row.get("input_audio")) if dataset_dir else row.get("input_audio"),
                    "prompt": row.get("prompt", ""),
                })
        logging.info(f"Loaded {len(self.data)} samples from {self.annotation_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item["video"]
        audio_path = item["audio"]
        sample_dir = os.path.dirname(video_path)

        # Relative path uses dataset_dir as base; absolute path stays unchanged.
        video_abs_path = video_path if os.path.isabs(video_path) else os.path.join(self.dataset_dir, video_path)
        audio_abs_path = audio_path if os.path.isabs(audio_path) else os.path.join(self.dataset_dir, audio_path)
        sample_abs_dir = sample_dir if os.path.isabs(sample_dir) else os.path.join(self.dataset_dir, sample_dir)
        return {
            "prompt": item["prompt"],
            "sample_dir": sample_abs_dir,
            "video_path": video_abs_path,
            "audio_path": audio_abs_path,
        }


def load_frames_from_images(images_dir, frame_num=81):
    """
    Load frames from image sequence directory.

    Args:
        images_dir: Path to images directory containing frame_0.png, frame_1.png, ...
        frame_num: Number of frames to load

    Returns:
        numpy array of shape [frame_num, H, W, 3] in RGB format
    """
    frames = []
    for i in range(frame_num):
        img_path = os.path.join(images_dir, f"frame_{i}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            frames.append(np.array(img))
        else:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                logging.warning(f"Frame {i} not found: {img_path}")

    if len(frames) == 0:
        logging.error(f"No frames found in {images_dir}")
        return None

    while len(frames) < frame_num:
        frames.append(frames[-1].copy())

    return np.array(frames[:frame_num])


def _sorted_image_paths(images_dir):
    """Return image paths sorted by numeric index in filename when possible."""
    if not os.path.isdir(images_dir):
        return []
    names = [n for n in os.listdir(images_dir) if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if len(names) == 0:
        return []

    def _key(name):
        match = re.findall(r"(\d+)", name)
        if match:
            return (0, int(match[-1]), name)
        return (1, 0, name)

    names = sorted(names, key=_key)
    return [os.path.join(images_dir, n) for n in names]


def _select_window_indices(total_frames, frame_num=81, window_start=None):
    """Select an inclusive [start_idx, end_idx] window with frame_num length when possible."""
    if total_frames <= 0:
        return 0, 0
    if total_frames <= frame_num:
        return 0, total_frames - 1

    max_start = total_frames - frame_num
    if window_start is None:
        start_idx = random.randint(0, max_start)
    else:
        start_idx = int(max(0, min(window_start, max_start)))
    end_idx = start_idx + frame_num - 1
    return start_idx, end_idx


def select_reference_index(total_frames, start_idx, end_idx, radius=33):
    """
    Sample reference frame from union of:
    [max(start_idx-radius, 0), start_idx] U [end_idx, min(end_idx+radius, total_frames-1)].
    """
    if total_frames <= 0:
        return 0
    left_start = max(start_idx - radius, 0)
    left_end = start_idx
    right_start = end_idx
    right_end = min(end_idx + radius, total_frames - 1)

    candidates = list(range(left_start, left_end))
    candidates.extend(range(right_start + 1, right_end + 1))
    candidates = sorted(set(candidates))
    if len(candidates) == 0:
        return 0
    return random.choice(candidates)


def build_even_indices(src_len, dst_len):
    """
    Build evenly-spaced index mapping from [0, src_len-1] to dst_len samples.
    Example: src_len=81, dst_len=41 -> [0,2,4,...,80].
    """
    if src_len <= 0 or dst_len <= 0:
        raise ValueError(f"Invalid lengths for index mapping: src_len={src_len}, dst_len={dst_len}")
    if dst_len == 1:
        return [0]
    if src_len == dst_len:
        return list(range(src_len))
    indices = np.round(np.linspace(0, src_len - 1, dst_len)).astype(np.int64).tolist()
    # Ensure strict bounds after rounding.
    indices = [max(0, min(src_len - 1, int(x))) for x in indices]
    return indices


def load_video_window(sample_dir, video_path=None, frame_num=81, window_start=None):
    """
    Load one window from video/images and return:
    - frames: [frame_num, H, W, 3] (if source shorter than frame_num, last frame padding)
    - start_idx/end_idx: selected window absolute indices in source timeline
    - total_frames: total frames in source timeline
    """
    images_dir = os.path.join(sample_dir, "images")
    if video_path is None:
        video_path = os.path.join(sample_dir, "sub_clip.mp4")

    if os.path.exists(video_path):
        out = _load_video_window_from_video(video_path, frame_num=frame_num, window_start=window_start)
        if out is not None:
            return out
        logging.warning(f"Failed to load window from video {video_path}, fallback to images.")

    image_paths = _sorted_image_paths(images_dir)
    if len(image_paths) > 0:
        total_frames = len(image_paths)
        start_idx, end_idx = _select_window_indices(total_frames, frame_num=frame_num, window_start=window_start)
        selected = image_paths[start_idx : end_idx + 1]
        frames = [np.array(Image.open(p).convert("RGB")) for p in selected]
        while len(frames) < frame_num and len(frames) > 0:
            frames.append(frames[-1].copy())
        if len(frames) == 0:
            return None
        return np.array(frames[:frame_num]), start_idx, min(end_idx, total_frames - 1), total_frames

    logging.error(f"Failed to load frames from {sample_dir}")
    zero = np.zeros((frame_num, 480, 832, 3), dtype=np.uint8)
    return zero, 0, frame_num - 1, frame_num


def load_reference_frame(sample_dir, video_path=None, frame_idx=0):
    """Load one absolute frame by index from video/images."""
    if video_path is None:
        video_path = os.path.join(sample_dir, "sub_clip.mp4")

    if os.path.isfile(video_path):
        frame = _load_frame_from_video(video_path, frame_idx)
        if frame is not None:
            return frame

    images_dir = os.path.join(sample_dir, "images")
    image_paths = _sorted_image_paths(images_dir)
    if len(image_paths) > 0:
        frame_idx = max(0, min(int(frame_idx), len(image_paths) - 1))
        return np.array(Image.open(image_paths[frame_idx]).convert("RGB"))
    return None


def load_video_frames(sample_dir, video_path=None, frame_num=81):
    """
    Load frames from sample directory.

    Priority:
    1. Try loading from video file (sub_clip.mp4)
    2. Randomly sample one frame_num-sized window from source timeline
    3. Otherwise, fall back to images/ directory with the same window policy
    """
    # Random window sampling for training: window_start=None -> random start index.
    frames, _, _, _ = load_video_window(sample_dir, video_path=video_path, frame_num=frame_num, window_start=None)
    return frames


def _load_video_window_from_video(video_path, frame_num=81, window_start=None):
    """Load one window from video file (decord first, then cv2)."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            del vr
            gc.collect()
            return None
        start_idx, end_idx = _select_window_indices(total_frames, frame_num=frame_num, window_start=window_start)
        indices = np.arange(start_idx, end_idx + 1)
        frames = vr.get_batch(indices).asnumpy()
        del vr
        gc.collect()
        if frames.shape[0] < frame_num and frames.shape[0] > 0:
            pad = np.repeat(frames[-1:], frame_num - frames.shape[0], axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        return frames, start_idx, end_idx, total_frames
    except Exception as e:
        logging.warning(f"Failed to load video window with decord: {e}, trying cv2")


def _load_frame_from_video(video_path, frame_idx):
    """Load one frame by absolute index from video file."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            del vr
            gc.collect()
            return None
        frame_idx = max(0, min(int(frame_idx), total_frames - 1))
        frame = vr[frame_idx].asnumpy()
        del vr
        gc.collect()
        return frame
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        frame_idx = max(0, min(int(frame_idx), total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _get_sample_id(sample_dir, dataset_dir):
    """Get sample_id as relative path from dataset_dir."""
    sample_dir = os.path.normpath(sample_dir)
    dataset_dir = os.path.normpath(dataset_dir)
    if sample_dir.startswith(dataset_dir):
        rel = os.path.relpath(sample_dir, dataset_dir)
        return rel.replace(os.sep, "/")
    return os.path.basename(sample_dir)


def _get_sample_id_from_video_path(video_path):
    """Derive a stable sample_id from a video file path (basename without extension)."""
    base = os.path.basename(str(video_path))
    name, _ = os.path.splitext(base)
    return name

class DataProcessor:
    """
    Helper class to process raw data into tensors.
    Encapsulates VAE, Text Encoders, etc. to be used in collate_fn or Main Loop.
    """
    def __init__(self, config, checkpoint_dir, device, n_prompt="", use_precomputed_audio=False, processed_data_dir=None, use_fixed_reference_frame=False, rank=0, mode='preprocess'):
        self.device = device
        self.config = config
        self.n_prompt = n_prompt
        self.use_precomputed_audio = use_precomputed_audio
        self.processed_data_dir = processed_data_dir
        self.use_fixed_reference_frame = use_fixed_reference_frame
        self.enable_background_filter = bool(getattr(config, "enable_background_filter", True))
        self.background_ssim_threshold = float(getattr(config, "background_ssim_threshold", 0.96))
        default_payload_dir = getattr(config, "payload_dir", None)

        background_ssim_csv_path = getattr(config, "background_ssim_csv_path", None)
        if background_ssim_csv_path:
            self.background_ssim_csv_path = str(background_ssim_csv_path)
        else:
            if default_payload_dir:
                csv_dir = os.path.dirname(default_payload_dir)
            else:
                csv_dir = os.path.join(str(PROJECT_ROOT), "processed_data")
            self.background_ssim_csv_path = os.path.join(csv_dir, "background_min_ssim_scores.csv")
        self.rvm_bg_blur_kernel = 3
        self.rvm_bg_keep_threshold = 0.99
        self._context_null_precomputed = None
        if self.enable_background_filter:
            logging.info("Background SSIM scores will be appended to: %s", self.background_ssim_csv_path)

        logging.info("Loading VAE...")
        vae_path = os.path.join(checkpoint_dir, config.vae_checkpoint) if hasattr(config, 'vae_checkpoint') else os.path.join(checkpoint_dir, "Wan2.1_VAE.pth")
        self.vae = WanVAE(vae_pth=vae_path, device=device)
        self.rvm = MattingNetwork(variant='mobilenetv3').eval().to(device)
        self.rvm.load_state_dict(torch.load('weights/rvm_model/rvm_mobilenetv3.pth'))

        if mode == 'preprocess':
            self._face_mask_extractor = FaceMaskExtractor(
                det_score_thresh=getattr(config, "face_det_score_thresh", 0.97),
                device=getattr(config, "face_det_device", "cuda"),
                det_size=getattr(config, "face_analysis_det_size", (640, 640)),
                no_face_fill_value=getattr(config, "face_no_det_fill_value", 0),
                rank=rank,
            )
            
            t5_path = os.path.join(checkpoint_dir, config.t5_checkpoint) if hasattr(config, 't5_checkpoint') else os.path.join(checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
            t5_tok_path = os.path.join(checkpoint_dir, config.t5_tokenizer) if hasattr(config, 't5_tokenizer') else os.path.join(checkpoint_dir, "google/umt5-xxl")
            logging.info("Loading T5...")
            self.text_encoder = T5EncoderModel(
                text_len=getattr(config, 'text_len', 512),
                dtype=getattr(config, 't5_dtype', torch.bfloat16),
                device=device,
                checkpoint_path=t5_path,
                tokenizer_path=t5_tok_path
            )
            
            logging.info("Loading CLIP...")
            clip_path = os.path.join(checkpoint_dir, config.clip_checkpoint) if hasattr(config, 'clip_checkpoint') else os.path.join(checkpoint_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
            clip_tok_path = os.path.join(checkpoint_dir, config.clip_tokenizer) if hasattr(config, 'clip_tokenizer') else os.path.join(checkpoint_dir, "xlm-roberta-large")
            self.clip = CLIPModel(
                dtype=getattr(config, 'clip_dtype', torch.float16),
                device=device,
                checkpoint_path=clip_path,
                tokenizer_path=clip_tok_path,
            )

            logging.info("Loading Wav2Vec...")
            wav2vec_path = config.wav2vec_dir if hasattr(config, 'wav2vec_dir') else "weights/chinese-wav2vec2-base"
            self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_path, local_files_only=True).to(device)
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path, local_files_only=True)

        else:
            self.text_encoder = None
            self.clip = None
            self.audio_encoder = None
            self.wav2vec_feature_extractor = None
            self._face_mask_extractor = None
            


    def compute_audio_embedding(self, audio_path):
        if self.audio_encoder is None:
            return None
        sr = 16000
        try:
            human_speech_array, _ = librosa.load(audio_path, sr=sr)
        except Exception as e:
            logging.error(f"Failed to load audio {audio_path}: {e}")
            return None

        human_speech_array = loudness_norm(human_speech_array, sr)
        audio_duration = len(human_speech_array) / sr
        video_length = audio_duration * 25

        audio_feature = np.squeeze(self.wav2vec_feature_extractor(human_speech_array, sampling_rate=sr).input_values)
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        if len(audio_feature.shape) == 1:
            audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if not hasattr(embeddings, 'hidden_states') or len(embeddings.hidden_states) == 0:
            logging.error("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "l s d -> s l d")
        return audio_emb

    def _build_rvm_background_masks(self, frames_rgb):
        """
        Build conservative RVM background masks with avg-pool edge smoothing.
        Args:
            frames_rgb: np.ndarray [T,H,W,3], uint8 RGB.
        Returns:
            np.ndarray [T,H,W], bool background mask.
        """
        if frames_rgb is None or frames_rgb.ndim != 4 or int(frames_rgb.shape[-1]) != 3:
            raise ValueError(
                f"Expected frames_rgb with shape [T,H,W,3], got {None if frames_rgb is None else frames_rgb.shape}"
            )

        # RVM expects [B,T,C,H,W] in [0,1].
        src = torch.from_numpy(frames_rgb).to(device=self.device, dtype=torch.float32) / 255.0
        src = src.permute(0, 3, 1, 2).unsqueeze(0).contiguous()

        with torch.no_grad():
            rec = [None] * 4
            _, pha, *rec = self.rvm(src, *rec)  # pha: [B,T,1,H,W]

        bg_mask = (1.0 - pha).clamp(0.0, 1.0)  # [1,T,1,H,W]
        bg_mask = bg_mask.permute(0, 2, 1, 3, 4).contiguous()  # [1,1,T,H,W]

        blur_kernel = max(1, int(self.rvm_bg_blur_kernel))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blur_pad = blur_kernel // 2
        bg_mask = nn.functional.pad(
            bg_mask,
            pad=(blur_pad, blur_pad, blur_pad, blur_pad, 0, 0),
            mode="constant",
            value=1.0,
        )
        bg_mask = nn.functional.avg_pool3d(
            bg_mask,
            kernel_size=(1, blur_kernel, blur_kernel),
            stride=1,
            padding=0,
        )
        bg_mask = (bg_mask > float(self.rvm_bg_keep_threshold)).squeeze(0).squeeze(0)
        return bg_mask.detach().cpu().numpy().astype(bool)

    def _compute_window_background_min_ssim(self, frames):
        """
        Compute min pairwise SSIM on background-only pixels among first/mid/last frame.
        Returns 0.0 when invalid or background area is empty.
        """
        if frames is None or frames.ndim != 4 or int(frames.shape[0]) < 3:
            return 0.0

        first_idx = 0
        mid_idx = int(frames.shape[0] // 2)
        last_idx = int(frames.shape[0] - 1)
        # Use full 33-frame temporal context for RVM recurrent segmentation,
        # then index the first/middle/last masks for SSIM check.
        all_bg_masks = self._build_rvm_background_masks(frames)
        picked = frames[[first_idx, mid_idx, last_idx]]
        bg_masks = all_bg_masks[[first_idx, mid_idx, last_idx]]

        pair_indices = ((0, 1), (1, 2), (0, 2))
        scores = []
        for i0, i1 in pair_indices:
            pair_bg_mask = bg_masks[i0] & bg_masks[i1]
            if int(pair_bg_mask.sum()) <= 0:
                return 0.0
            rgb0 = picked[i0].copy()
            rgb1 = picked[i1].copy()
            rgb0[~pair_bg_mask] = 0
            rgb1[~pair_bg_mask] = 0
            score = ssim(rgb0, rgb1, data_range=255, channel_axis=-1)
            scores.append(float(score))
        return float(np.min(scores)) if len(scores) > 0 else 0.0

    def _append_background_ssim_csv(self, sample_id, min_bg_ssim):
        """
        Append one row to a standalone CSV:
        video_id,min_bg_ssim
        """
        csv_path = self.background_ssim_csv_path
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        with open(csv_path, "a+", encoding="utf-8", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    f.write("video_id,min_bg_ssim\n")
                f.write(f"{sample_id},{float(min_bg_ssim):.6f}\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def process_batch(self, batch, return_unconditional=True, model_dtype=torch.bfloat16, unbind_for_model=False, processed_data_dir=None):
        """
        Process a batch of raw metadata into training tensors.

        When processed_data_dir is set, load full_emb and clean_latents from
        {processed_data_dir}/{sample_id}/full_emb.pt and clean_latents.pt.
        Otherwise compute them on-the-fly.
        """
        prompts = [b['prompt'] for b in batch]
        sample_dirs = [b['sample_dir'] for b in batch]
        video_paths = [b['video_path'] for b in batch]
        audio_paths = [b['audio_path'] for b in batch]
        batch_size = len(prompts)
        frame_num = self.config.training_window_size

        if self.enable_background_filter and batch_size != 1:
            raise ValueError("Background stability filtering requires batch_size=1.")

        size_config = getattr(self.config, 'size', 'infinitetalk-480')
        if size_config == 'infinitetalk-480':
            from wan.utils.multitalk_utils import ASPECT_RATIO_627 as bucket_config
        else:
            from wan.utils.multitalk_utils import ASPECT_RATIO_960 as bucket_config

        vae_stride = getattr(self.config, 'vae_stride', (4, 8, 8))
        patch_size = getattr(self.config, 'patch_size', (1, 2, 2))
        sp_size = getattr(self.config, 'sp_size', 1)

        # Encode text: load precomputed context when available, otherwise run text_encoder.
        dataset_dir_for_id = getattr(self.config, 'dataset_dir', None)
        context_list = []
        with torch.no_grad():
            for j, (sample_dir, prompt) in enumerate(zip(sample_dirs, prompts)):
                sample_id = _get_sample_id(sample_dir, dataset_dir_for_id or sample_dir)
                if processed_data_dir is not None:
                    ctx_path = os.path.join(processed_data_dir, sample_id, "context.pt")
                    if os.path.isfile(ctx_path):
                        ctx = torch.load(ctx_path, map_location=self.device)
                        context_list.append(ctx.squeeze(0))
                else:
                    ctx = self.text_encoder([prompt], self.device)
                    ctx = torch.stack(ctx).squeeze(0)
                    context_list.append(ctx)
            context = torch.stack(context_list)
            if return_unconditional:
                if self._context_null_precomputed is not None:
                    # Precomputed shape is [1, seq, dim]; expand to batch dimension.
                    context_null = self._context_null_precomputed.to(self.device).expand(batch_size, -1, -1).clone()
                else:
                    context_null = self.text_encoder([self.n_prompt] * batch_size, self.device)
                    context_null = torch.stack(context_null)

        # Per-sample visual tensors: clean_latents, cond_latents, clip_fea, ref_target_masks, face_masks (latent-aligned).
        latents_list = []
        cond_latents_list = []
        clip_feas_list = []
        ref_target_masks_list = []
        face_masks_list = []
        window_start_indices = []
        dataset_dir = getattr(self.config, 'dataset_dir', None)

        for i, (sample_dir, video_path) in enumerate(zip(sample_dirs, video_paths)):
            sample_id = _get_sample_id(sample_dir, dataset_dir or sample_dir)

            # Try to load clean_latents and target_size from the precomputed feature directory.
            latents_i = None
            target_size = None
            lat_h = lat_w = None
            if processed_data_dir is not None:
                feat_dir = os.path.join(processed_data_dir, sample_id)
                clean_latents_path = os.path.join(feat_dir, "clean_latents.pt")
                meta_path = os.path.join(feat_dir, "metadata.json")
                if os.path.isfile(clean_latents_path) and os.path.isfile(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    latents_i = torch.load(clean_latents_path, map_location=self.device)
                    target_size = tuple(meta.get("target_size", (480, 832)))
                    lat_h, lat_w = latents_i.shape[-2], latents_i.shape[-1]

            # Load a random video window and record its start index for audio alignment.
            frames, start_idx, end_idx, total_frames = load_video_window(
                sample_dir, video_path=video_path, frame_num=frame_num, window_start=None
            )

            window_start_indices.append(int(start_idx))

            # Pick target_size by aspect-ratio bucket when not loaded from cache.
            if target_size is None:
                src_h, src_w = frames.shape[1], frames.shape[2]
                ratio = src_h / src_w
                closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
                target_h, target_w = bucket_config[closest_bucket][0]
                target_size = (target_h, target_w)

            # Pick reference frame for CLIP/cond:
            # - use_fixed_reference_frame=True: first window frame
            # - otherwise: random sample within +/-training_window_size frames
            #   around the window (defaults to 33).
            if self.use_fixed_reference_frame:
                ref_abs_idx = int(start_idx)
            else:
                ref_abs_idx = int(select_reference_index(
                    total_frames, start_idx, end_idx,
                    radius=int(getattr(self.config, "training_window_size", 33)),
                ))

            # Reuse the in-window frame when the reference index is inside the window.
            if start_idx <= ref_abs_idx <= end_idx:
                ref_frame = frames[ref_abs_idx - start_idx]
            else:
                ref_frame = load_reference_frame(sample_dir, video_path=video_path, frame_idx=ref_abs_idx)
            if ref_frame is None:
                ref_frame = frames[0]

            # Compute CLIP visual feature from the reference frame.
            img_ref = Image.fromarray(ref_frame)
            img_ref_tensor = process_image_to_tensor(img_ref, target_size=target_size, device=self.device)
            with torch.no_grad():
                clip_fea = self.clip.visual(img_ref_tensor[:, :, -1:, :, :]).squeeze(0)
            clip_feas_list.append(clip_fea)

            # Normalize frames to [-1,1] using the same resize + center-crop policy.
            frames_tensor = process_video_frames_to_tensor(frames, target_size=target_size)

            # Background stability filter runs after resize/crop to avoid huge resolutions.
            if self.enable_background_filter:
                try:
                    frames_for_bg = (
                        ((frames_tensor.permute(1, 2, 3, 0).detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
                        .round()
                        .to(torch.uint8)
                        .numpy()
                    )
                    min_bg_ssim = self._compute_window_background_min_ssim(frames_for_bg)
                except Exception as e:
                    self._append_background_ssim_csv(sample_id, float("nan"))
                    logging.warning(
                        "Background SSIM check failed, skip sample_id=%s, error=%s",
                        sample_id,
                        e,
                    )
                    return None
                self._append_background_ssim_csv(sample_id, min_bg_ssim)
                if min_bg_ssim < float(self.background_ssim_threshold):
                    logging.info(
                        "Skip sample due to unstable background: sample_id=%s, min_bg_ssim=%.4f, threshold=%.2f",
                        sample_id,
                        min_bg_ssim,
                        self.background_ssim_threshold,
                    )
                    return None

            # clean_latents: use precomputed if available, otherwise VAE-encode the window.
            if latents_i is None:
                with torch.no_grad():
                    latents_i = self.vae.encode([frames_tensor.to(self.device)])[0]
                lat_h, lat_w = latents_i.shape[-2], latents_i.shape[-1]
                
            face_masks = self._face_mask_extractor.extract_video_masks(frames)
            face_masks = process_video_masks_to_tensor(face_masks, target_size=target_size).to(self.device)
            face_masks = align_face_masks_to_latent(
                face_masks,
                latent_shape=(latents_i.shape[1], latents_i.shape[2], latents_i.shape[3]),
            )

            # Conditional latent (y): place reference frame at index 0, zero pad rest, then VAE-encode.
            padding_frames_pixels = torch.zeros(
                frames_tensor.shape[0], frame_num, frames_tensor.shape[2], frames_tensor.shape[3],
                dtype=frames_tensor.dtype, device=frames_tensor.device
            )
            # Always place the selected reference frame at the first temporal position
            # to align with inference-time conditioning layout.
            ref_frame_tensor = process_video_frames_to_tensor(
                np.expand_dims(ref_frame, axis=0), target_size=target_size
            ).to(frames_tensor.device, dtype=frames_tensor.dtype)
            padding_frames_pixels[:, 0:1, :, :] = ref_frame_tensor[:, 0:1, :, :]
            with torch.no_grad():
                y_latent = self.vae.encode([padding_frames_pixels.to(self.device)])[0]
            # Mask: 1 on first frame, 0 elsewhere; repeat-interleave and concat with y_latent to form cond.
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2).to(latents_i.dtype).squeeze(0)
            y_cond = torch.concat([msk, y_latent], dim=0)
            cond_latents_list.append(y_cond)

            # Single-person scene: full-1 reference target mask.
            ref_mask = torch.ones(3, lat_h, lat_w, device=self.device)
            ref_target_masks_list.append(ref_mask)
            latents_list.append(latents_i)
            face_masks_list.append(face_masks)

        latents = torch.stack(latents_list)
        cond_latents = torch.stack(cond_latents_list)
        clip_feas = torch.stack(clip_feas_list)
        ref_target_masks = torch.stack(ref_target_masks_list)
        face_masks = torch.stack(face_masks_list)

        # Compute sequence length from latent spatial size to match Wan model expectations.
        lat_h, lat_w = latents.shape[-2], latents.shape[-1]
        max_seq_len = ((frame_num - 1) // vae_stride[0] + 1) * lat_h * lat_w // (patch_size[1] * patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / sp_size)) * sp_size

        # Audio: build a 5-frame window (-2..+2) per video frame; prefer precomputed full_emb.
        window_offsets = torch.tensor([-2, -1, 0, 1, 2], device=self.device)
        audio_tensors = []
        for j, audio_path in enumerate(audio_paths):
            sample_dir = sample_dirs[j]
            sample_id = _get_sample_id(sample_dir, dataset_dir or sample_dir)
            if processed_data_dir is not None:
                full_emb_path = os.path.join(processed_data_dir, sample_id, "full_emb.pt")
                if os.path.isfile(full_emb_path):
                    full_emb = torch.load(full_emb_path, map_location=self.device)
                else:
                    # When full_emb.pt is missing in precomputed mode, fall back to zeros (no audio_encoder needed).
                    full_emb = torch.zeros(frame_num, 12, 768, device=self.device)
            else:
                full_emb = self._load_audio_embedding(audio_path, frame_num)

            # Align audio window to the sampled video window start.
            video_start_idx = int(window_start_indices[j]) if j < len(window_start_indices) else 0
            audio_base_idx = video_start_idx if int(full_emb.shape[0]) > int(frame_num) else 0
            frame_indices = torch.arange(frame_num, device=self.device) + int(audio_base_idx)
            window_indices = frame_indices.unsqueeze(1) + window_offsets.unsqueeze(0)
            window_indices = window_indices.clamp(0, full_emb.shape[0] - 1)
            windows = full_emb[window_indices]
            audio_tensors.append(windows)

        audio_batch = torch.stack(audio_tensors)

        # CFG: build a zero audio_null for unconditional pass.
        if return_unconditional:
            audio_null = torch.zeros_like(audio_batch)[-1:]
        else:
            context_null = None
            audio_null = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latents = latents.to(dtype=model_dtype)
        context = context.to(dtype=model_dtype)
        clip_feas = clip_feas.to(dtype=model_dtype)
        cond_latents = cond_latents.to(dtype=model_dtype)
        audio_batch = audio_batch.to(dtype=model_dtype)
        ref_target_masks = ref_target_masks.to(dtype=model_dtype)
        face_masks = face_masks.to(dtype=model_dtype)
        if context_null is not None:
            context_null = context_null.to(dtype=model_dtype)
        if audio_null is not None:
            audio_null = audio_null.to(dtype=model_dtype)

        # When the model expects List[Tensor] instead of a batched tensor, unbind on dim 0.
        if unbind_for_model:
            latents = list(latents.unbind(0))
            context = list(context.unbind(0))
            cond_latents = list(cond_latents.unbind(0))
            if context_null is not None:
                context_null = list(context_null.unbind(0))

        # Wan currently consumes the ref_target_masks of the first sample only.
        ref_target_masks_output = ref_target_masks[0] if ref_target_masks.shape[0] > 0 else None
        return {
            "clean_latents": latents,
            "context": context,
            "clip_fea": clip_feas,
            "cond_latents": cond_latents,
            "audio_emb": audio_batch,
            "ref_target_masks": ref_target_masks_output,
            "seq_len": max_seq_len,
            "context_null": context_null,
            "audio_null": audio_null,
            "face_masks": face_masks,
        }


    def _load_audio_embedding(self, audio_path, frame_num):
        if self.audio_encoder is None:
            return torch.zeros(frame_num, 12, 768, device=self.device)
        if self.use_precomputed_audio:
            pt_path = audio_path.replace('.wav', '.pt').replace('.mp3', '.pt')
            if not os.path.exists(pt_path):
                logging.error(f"Precomputed audio embedding not found: {pt_path}")
                return torch.zeros(frame_num, 12, 768, device=self.device)
            full_emb = torch.load(pt_path, map_location=self.device)
            if torch.isnan(full_emb).any():
                logging.error(f"NaN in precomputed embedding: {pt_path}")
                return torch.zeros(frame_num, 12, 768, device=self.device)
            return full_emb
        if not audio_path or not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return torch.zeros(frame_num, 12, 768, device=self.device)
        full_emb = self.compute_audio_embedding(audio_path)
        if full_emb is None:
            logging.error(f"Failed to compute audio embedding: {audio_path}")
            return torch.zeros(frame_num, 12, 768, device=self.device)
        return full_emb

    def save_validation_features(self, sample_dir, audio_path, video_path, output_dir,
                                 dataset_dir=None, prompt=None, sample_id=None):
        """
        Pre-compute and save validation/inference features into
        ``<output_dir>/<sample_id>/`` for the fixed-first-frame validation
        inference path (see ``src/validation_inference.py``).

        We pin the conditioning frame *and* the reference frame to index 0
        of the clip — this matches the assumption made by the validation
        inference code, so no random window sampling is performed here.

        Files written per sample:
            context.pt              T5 features of ``prompt``: ``[1, seq_len, dim]``.
            full_emb.pt             Full-clip wav2vec features: ``[S, layers, dim]``.
            clip_fea.pt             CLIP visual feature of the first frame.
            first_frame_latent.pt   VAE-encoded first frame: ``[C, 1, H, W]``.
            cond_latents.pt         Fixed-first-frame conditioning latent: ``[C_cond, T, H, W]``.
            metadata.json           target_size, total frame counts, ...

        ``training_window_size`` from the config (default 33) is used as
        ``cond_frame_num`` to match the runtime training/inference window.
        """
        if self.text_encoder is None:
            raise RuntimeError("save_validation_features requires text_encoder, but it is None.")
        if self.clip is None:
            raise RuntimeError("save_validation_features requires clip encoder, but it is None.")

        size_config = getattr(self.config, "size", "infinitetalk-480")
        if size_config == "infinitetalk-480":
            from wan.utils.multitalk_utils import ASPECT_RATIO_627 as bucket_config
        else:
            from wan.utils.multitalk_utils import ASPECT_RATIO_960 as bucket_config

        if sample_id is None:
            sample_id = _get_sample_id(sample_dir, dataset_dir or sample_dir)
        feat_dir = os.path.join(output_dir, sample_id)
        os.makedirs(feat_dir, exist_ok=True)
        cond_latents_path = os.path.join(feat_dir, "cond_latents.pt")
        metadata_path = os.path.join(feat_dir, "metadata.json")

        existing_meta = None
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    existing_meta = json.load(f)
            except Exception:
                existing_meta = None

        if prompt is not None:
            with torch.no_grad():
                ctx = self.text_encoder([prompt], self.device)
                ctx = torch.stack(ctx)
            torch.save(ctx.cpu(), os.path.join(feat_dir, "context.pt"))

        # Fix first frame as both the conditioning and the reference frame.
        frames, start_idx, end_idx, total_frames = load_video_window(
            sample_dir, video_path, frame_num=1, window_start=0
        )
        _ = start_idx, end_idx
        ref_frame = frames[0]

        # Audio: save the entire embedding; inference applies its own sliding window.
        full_emb = self.compute_audio_embedding(audio_path)
        if full_emb is not None:
            torch.save(full_emb.cpu(), os.path.join(feat_dir, "full_emb.pt"))
            audio_total_frames = int(full_emb.shape[0])
        else:
            full_emb = torch.zeros(1, 12, 768, device=self.device)
            torch.save(full_emb.cpu(), os.path.join(feat_dir, "full_emb.pt"))
            audio_total_frames = 0

        target_h = target_w = None
        if existing_meta is not None and existing_meta.get("target_size") is not None:
            target_h, target_w = map(int, existing_meta["target_size"])
        if target_h is None or target_w is None:
            src_h, src_w = ref_frame.shape[0], ref_frame.shape[1]
            ratio = src_h / src_w
            closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
            target_h, target_w = bucket_config[closest_bucket][0]
        target_size = (target_h, target_w)

        # Use the same resize + center-crop as CLIP to keep the first-frame
        # geometry consistent with the CLIP feature.
        img_ref = Image.fromarray(ref_frame)
        img_ref_tensor = process_image_to_tensor(img_ref, target_size=target_size, device=self.device)

        # First-frame latent — at inference we feed this directly without
        # re-encoding through the VAE.
        first_frame_tensor = img_ref_tensor.squeeze(0)
        with torch.no_grad():
            first_frame_latent = self.vae.encode([first_frame_tensor])[0]
        torch.save(first_frame_latent.cpu(), os.path.join(feat_dir, "first_frame_latent.pt"))

        # Cond latent: mask the first frame as the only conditioning slot.
        cond_frame_num = int(getattr(self.config, "training_window_size", 33))
        lat_h, lat_w = first_frame_latent.shape[-2], first_frame_latent.shape[-1]
        padding_frames_pixels = torch.zeros(
            3, cond_frame_num, target_h, target_w,
            dtype=img_ref_tensor.dtype, device=img_ref_tensor.device,
        )
        padding_frames_pixels[:, 0] = img_ref_tensor[0, :, 0]
        with torch.no_grad():
            y_latent = self.vae.encode([padding_frames_pixels.to(self.device)])[0]
        msk = torch.ones(1, cond_frame_num, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2).to(y_latent.dtype).squeeze(0)
        y_cond = torch.concat([msk, y_latent], dim=0)
        torch.save(y_cond.cpu(), cond_latents_path)

        # First-frame CLIP feature.
        with torch.no_grad():
            clip_fea = self.clip.visual(img_ref_tensor[:, :, -1:, :, :]).squeeze(0)
        torch.save(clip_fea.cpu(), os.path.join(feat_dir, "clip_fea.pt"))

        metadata_to_save = {
            "target_size": [target_h, target_w],
            "fps": 25,
            "video_total_frames": int(total_frames),
            "conditioning_frame_idx": 0,
            "reference_frame_idx": 0,
            "audio_total_frames": int(audio_total_frames),
            "feature_type": "validation_inference_first_frame",
            "cond_frame_num": int(cond_frame_num),
        }
        if existing_meta is not None:
            existing_meta.update(metadata_to_save)
            metadata_to_save = existing_meta
        with open(metadata_path, "w") as f:
            json.dump(metadata_to_save, f, indent=2)

    def save_null_context(self, output_path):
        """Encode ``self.n_prompt`` with the text encoder and dump it as a
        single ``context_null.pt`` tensor of shape ``[1, seq_len, dim]``.

        This is the unconditional (CFG) text context used by the Stage-2
        inference path. Run once per dataset; downstream training and
        validation loaders look for
        ``<val_features_dir>/context_null.pt`` (e.g.
        ``processed_data/talkcuts/val/feature/context_null.pt``).
        """
        if self.text_encoder is None:
            raise RuntimeError("save_null_context requires text_encoder, but it is None.")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with torch.no_grad():
            ctx = self.text_encoder([self.n_prompt], self.device)
            ctx = torch.stack(ctx)
        torch.save(ctx.cpu(), output_path)


def _batch_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _batch_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_batch_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_batch_to_cpu(v) for v in obj)
    return obj


def _batch_to_device_and_dtype(obj, device, model_dtype):
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.to(device=device, dtype=model_dtype)
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _batch_to_device_and_dtype(v, device, model_dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_batch_to_device_and_dtype(v, device, model_dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_batch_to_device_and_dtype(v, device, model_dtype) for v in obj)
    return obj


def _serialize_torch_obj(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _deserialize_torch_obj(raw_bytes):
    return torch.load(io.BytesIO(raw_bytes), map_location="cpu")


def _get_lmdb_module():
    try:
        return importlib.import_module("lmdb")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "lmdb is required for preprocess/train LMDB mode. Please install dependencies from requirements.txt"
        ) from e


def preprocess_batches_to_lmdb(
    processor,
    dataloader,
    lmdb_path,
    num_samples,
    rank,
    world_size,
    model_dtype=torch.bfloat16,
    map_size_bytes=1024 * 1024 * 1024 * 1024,
    log_interval=20,
    write_batch_size=32,
    barrier_interval=20,
):
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
    lmdb_mod = _get_lmdb_module()
    env = lmdb_mod.open(
        lmdb_path,
        subdir=False,
        map_size=int(map_size_bytes),
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=True) as txn:
        txn.put(b"__num_samples__", str(int(num_samples)).encode("utf-8"))

    import torch.distributed as dist

    local_step = 0
    written = 0
    pending_kv = []

    def _flush_pending():
        nonlocal pending_kv
        if len(pending_kv) == 0:
            return
        with env.begin(write=True) as txn:
            for k, v in pending_kv:
                txn.put(k, v)
        pending_kv = []

    while True:
        key_idx = local_step * world_size + rank
        if key_idx >= num_samples:
            break

        batch_raw = next(dataloader)
        batch_tensors = processor.process_batch(
            batch_raw,
            return_unconditional=True,
            model_dtype=model_dtype,
            unbind_for_model=False,
            processed_data_dir=None,
        )
        payload = _serialize_torch_obj(_batch_to_cpu(batch_tensors))
        pending_kv.append((str(int(key_idx)).encode("utf-8"), payload))
        if len(pending_kv) >= int(max(1, write_batch_size)):
            _flush_pending()

        local_step += 1
        written += 1
        if written % int(log_interval) == 0:
            logging.info(
                "LMDB preprocess rank=%d wrote %d samples, latest_key=%d",
                rank,
                written,
                key_idx,
            )

        # Synchronize at a lower frequency to avoid per-step stalls.
        if int(barrier_interval) > 0 and (written % int(barrier_interval) == 0):
            dist.barrier()

    _flush_pending()
    dist.barrier()

    env.sync()
    env.close()
    return written


def preprocess_batches_to_payload_files(
    processor,
    dataloader,
    payload_dir,
    num_samples,
    rank,
    world_size,
    model_dtype=torch.bfloat16,
    log_interval=20,
    barrier_interval=20,
):
    """
    Stage-A preprocess:
    Save serialized payload files named by key_idx (e.g., 0.payload, 1.payload, ...).
    Loop until the globally accumulated count of useful samples
    (newly written this run + already existing on disk) reaches ``num_samples``.
    Filtered-out samples (e.g. background filter rejected) do not count, so the
    loop will keep iterating to compensate for them. Key indices stay assigned
    by ``local_step * world_size + rank`` and may end up sparse; the LMDB pack
    step renumbers them to a contiguous range later.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    os.makedirs(payload_dir, exist_ok=True)
    import torch.distributed as dist

    sync_interval = int(barrier_interval) if int(barrier_interval) > 0 else 1
    target = int(num_samples)
    sync_device = getattr(processor, "device", None)
    if not isinstance(sync_device, torch.device):
        sync_device = torch.device(sync_device) if sync_device is not None else torch.device("cpu")

    def _global_useful(local_useful):
        t = torch.tensor([int(local_useful)], device=sync_device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def _trim_payload_dir_to_target():
        payload_records = []
        for name in os.listdir(payload_dir):
            m = re.fullmatch(r"(\d+)\.payload", name)
            if m is None:
                continue
            payload_records.append((int(m.group(1)), os.path.join(payload_dir, name)))
        payload_records.sort(key=lambda x: x[0])

        total = len(payload_records)
        if total == target:
            logging.info("Payload preprocess trim: total=%d already equals target=%d", total, target)
            return 0, total
        if total < target:
            logging.warning(
                "Payload preprocess trim: total=%d < target=%d, nothing removed",
                total,
                target,
            )
            return 0, total

        to_remove = payload_records[target:]
        for _, p in to_remove:
            if os.path.isfile(p):
                os.remove(p)
        removed = len(to_remove)
        final_total = total - removed
        logging.info(
            "Payload preprocess trim: removed=%d (dropped tail keys), final_total=%d, target=%d",
            removed,
            final_total,
            target,
        )
        return removed, final_total

    local_step = 0
    written = 0
    skipped = 0
    filtered = 0

    initial_global_useful = _global_useful(0)
    if initial_global_useful >= target:
        if rank == 0:
            logging.info(
                "Payload preprocess early-stop: initial_global_useful=%d >= target=%d, trimming to exact target.",
                initial_global_useful,
                target,
            )
        dist.barrier()
        if rank == 0:
            _trim_payload_dir_to_target()
        dist.barrier()
        return written, skipped, filtered

    while True:
        key_idx = local_step * world_size + rank

        batch_raw = next(dataloader)

        payload_path = os.path.join(payload_dir, f"{int(key_idx)}.payload")
        if os.path.isfile(payload_path):
            skipped += 1
        else:
            batch_tensors = processor.process_batch(
                batch_raw,
                return_unconditional=True,
                model_dtype=model_dtype,
                unbind_for_model=False,
                processed_data_dir=None,
            )
            if batch_tensors is None:
                filtered += 1
            else:
                payload = _serialize_torch_obj(_batch_to_cpu(batch_tensors))
                tmp_path = payload_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(payload)
                os.replace(tmp_path, payload_path)
                written += 1

        local_step += 1
        processed = written + skipped + filtered
        if processed % int(log_interval) == 0:
            logging.info(
                "Payload preprocess rank=%d processed=%d (written=%d, skipped=%d, filtered=%d), latest_key=%d",
                rank,
                processed,
                written,
                skipped,
                filtered,
                key_idx,
            )

        # Sync the global useful sample count periodically and stop once the
        # target number of final samples has been reached across all ranks.
        if (local_step % sync_interval) == 0:
            global_useful = _global_useful(written + skipped)
            if rank == 0:
                logging.info(
                    "Payload preprocess sync: global_useful=%d / target=%d (local_step=%d)",
                    global_useful, target, local_step,
                )
            if global_useful >= target:
                break

    dist.barrier()
    if rank == 0:
        _trim_payload_dir_to_target()
    dist.barrier()
    return written, skipped, filtered


def preprocess_validation_features(
    processor,
    annotation_file,
    dataset_dir,
    output_dir,
    rank,
    world_size,
    write_context_null=True,
    context_null_path=None,
):
    """
    Walk a validation CSV (columns ``video, input_audio, prompt``) and run
    ``DataProcessor.save_validation_features`` on each sample. Samples are
    sharded across ranks round-robin. Optionally also dumps
    ``context_null.pt`` (rank-0 only) so the validation inference path has
    a precomputed unconditional context.

    ``output_dir`` is the parent directory that will hold per-sample feature
    folders (e.g. ``processed_data/talkcuts/val/feature``).
    ``context_null_path`` defaults to ``<output_dir>/context_null.pt`` (i.e.
    it sits *inside* ``output_dir`` next to the per-sample feature
    subdirectories), matching the layout expected by
    ``src/validation_inference.py``.
    """
    import torch.distributed as dist

    dataset = FlashTalkDataset(dataset_dir, annotation_file=annotation_file)
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped = 0
    failed = 0
    for idx in range(rank, len(dataset), world_size):
        item = dataset[idx]
        sample_dir = item.get("sample_dir") or item.get("video_path")
        sample_id = _get_sample_id(sample_dir, dataset_dir or sample_dir)
        feat_dir = os.path.join(output_dir, sample_id)
        required = ["context.pt", "full_emb.pt", "clip_fea.pt",
                    "first_frame_latent.pt", "cond_latents.pt", "metadata.json"]
        if all(os.path.isfile(os.path.join(feat_dir, f)) for f in required):
            skipped += 1
            continue
        try:
            processor.save_validation_features(
                sample_dir=sample_dir,
                audio_path=item["audio_path"],
                video_path=item["video_path"],
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                prompt=item.get("prompt"),
                sample_id=sample_id,
            )
            written += 1
        except Exception as exc:
            logging.exception("Failed to extract val features for %s: %s", sample_id, exc)
            failed += 1

        if (written + skipped + failed) % 5 == 0:
            logging.info(
                "Val preprocess rank=%d wrote=%d skipped=%d failed=%d latest=%s",
                rank, written, skipped, failed, sample_id,
            )

    if write_context_null and rank == 0:
        if context_null_path is None:
            context_null_path = os.path.join(output_dir, "context_null.pt")
        processor.save_null_context(context_null_path)
        logging.info("Saved context_null.pt to %s", context_null_path)

    dist.barrier()
    return written, skipped, failed


class LmdbBatchReader:
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        lmdb_mod = _get_lmdb_module()
        self.env = lmdb_mod.open(
            lmdb_path,
            subdir=False,
            map_size=1,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            num_samples_raw = txn.get(b"__num_samples__")
        if num_samples_raw is None:
            raise RuntimeError(f"LMDB metadata '__num_samples__' not found in {lmdb_path}")
        self.num_samples = int(num_samples_raw.decode("utf-8"))

    def get(self, key_idx, device, model_dtype):
        with self.env.begin(write=False) as txn:
            raw = txn.get(str(int(key_idx)).encode("utf-8"))
        if raw is None:
            raise KeyError(f"LMDB key {key_idx} not found in {self.lmdb_path}")
        batch_tensors = _deserialize_torch_obj(raw)
        return _batch_to_device_and_dtype(batch_tensors, device=device, model_dtype=model_dtype)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

