#!/usr/bin/env python3
"""
Standalone version of evaluate_gt: self-contained script computing Sync-C,
Sync-D, IQA, Aesthe for a video file or a directory of videos. Does not depend
on the liveavatar package.

Usage:
    python run_evaluate_gt_standalone.py --video_path <video_or_directory>

Required weight files (relative to project root):
    SYNCNET_CKPT     weights/syncnet/syncnet_v2.model
    S3FD_WEIGHT_PATH weights/syncnet/sfd_face.pth
    ONE_ALIGN_PATH   weights/q-align

Dependencies: torch, numpy, cv2, scipy, PIL, transformers,
python_speech_features, scenedetect
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import filecmp
import glob
import math
import os
import pickle
import subprocess
import sys
from shutil import rmtree
from tempfile import TemporaryDirectory

# Use a project-local tmp directory to avoid /tmp being too small.
_PROJECT_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(_PROJECT_TMP, exist_ok=True)

import cv2
import numpy as np
import python_speech_features  # pyright: ignore[reportMissingImports]
import torch
from PIL import Image
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(
    description="Compute Sync-C, Sync-D, IQA, Aesthe for a single video or a directory of videos."
)
parser.add_argument("--video_path", required=True, help="Video file path or directory containing videos.")
parser.add_argument("--fps", type=int, default=25, help="Video frame rate (frames per second).")
parser.add_argument(
    "--sync_clip_seconds",
    type=float,
    default=5.0,
    help="Clip length in seconds used by SyncNet for segmented evaluation.",
)
parser.add_argument(
    "--qalign_chunk_size",
    type=int,
    default=64,
    help="Number of frames per Q-Align inference chunk (smaller=less VRAM, larger=faster).",
)
parser.add_argument(
    "--result_file",
    default=None,
    help="Append evaluation results to this CSV path; default is auto-derived from --video_path.",
)

args = parser.parse_args()


# -----------------------------------------------------------------------------
# AttrDict (replace attrdict package)
# -----------------------------------------------------------------------------
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# -----------------------------------------------------------------------------
# SyncNetModel (S)
# -----------------------------------------------------------------------------
class S(torch.nn.Module):
    def __init__(self, num_layers_in_fc_layers=1024):
        super(S, self).__init__()
        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32
        self.netcnnaud = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            torch.nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            torch.nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
        )
        self.netfcaud = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_layers_in_fc_layers),
        )
        self.netfclip = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_layers_in_fc_layers),
        )
        self.netcnnlip = torch.nn.Sequential(
            torch.nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            torch.nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            torch.nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            torch.nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU(inplace=True),
        )

    def forward_aud(self, x):
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfcaud(mid)
        return out

    def forward_lip(self, x):
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfclip(mid)
        return out

    def forward_lipfeat(self, x):
        mid = self.netcnnlip(x)
        out = mid.view((mid.size()[0], -1))
        return out


# -----------------------------------------------------------------------------
# SyncNetInstance
# -----------------------------------------------------------------------------
def _calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(0, len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(
            feat1[[i], :].repeat(win_size, 1), feat2p[i : i + win_size, :]
        ))
    return dists


class SyncNetInstance(torch.nn.Module):
    def __init__(self, dropout=0, num_layers_in_fc_layers=1024):
        super(SyncNetInstance, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).to(self.device)

    def evaluate(self, opt, videofile):
        self.__S__.eval()
        if os.path.exists(os.path.join(opt.tmp_dir, opt.reference)):
            rmtree(os.path.join(opt.tmp_dir, opt.reference))
        os.makedirs(os.path.join(opt.tmp_dir, opt.reference))
        subprocess.call(
            "ffmpeg -y -i %s -threads 1 -f image2 -v quiet %s"
            % (videofile, os.path.join(opt.tmp_dir, opt.reference, "%06d.jpg")),
            shell=True,
            stdout=None,
        )
        subprocess.call(
            "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -v quiet %s"
            % (videofile, os.path.join(opt.tmp_dir, opt.reference, "audio.wav")),
            shell=True,
            stdout=None,
        )
        flist = sorted(
            glob.glob(os.path.join(opt.tmp_dir, opt.reference, "*.jpg"))
        )
        if len(flist) == 0:
            return None, np.zeros(1), np.zeros(1)
        images = [cv2.imread(f) for f in flist]
        im = np.stack(images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        sample_rate, audio = wavfile.read(
            os.path.join(opt.tmp_dir, opt.reference, "audio.wav")
        )
        mfcc = list(zip(*python_speech_features.mfcc(audio, sample_rate)))
        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        min_length = min(len(images), math.floor(len(audio) / 640))
        lastframe = min_length - 5
        if lastframe <= 0:
            return None, np.zeros(1), np.zeros(1)
        im_feat, cc_feat = [], []
        for i in range(0, lastframe, opt.batch_size):
            im_batch = [
                imtv[:, :, vframe : vframe + 5, :, :]
                for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())
            cc_batch = [
                cct[:, :, :, vframe * 4 : vframe * 4 + 20]
                for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = self.__S__.forward_aud(cc_in.to(self.device))
            cc_feat.append(cc_out.data.cpu())
        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)
        dists = _calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        minval, minidx = torch.min(mdist, 0)
        offset = opt.vshift - minidx
        conf = torch.median(mdist) - minval
        return offset.cpu().numpy(), conf.cpu().numpy(), minval

    def loadParameters(self, path):
        loaded_state = torch.load(
            path, map_location=lambda storage, loc: storage, weights_only=False
        )
        self_state = self.__S__.state_dict()
        for name, param in loaded_state.items():
            self_state[name].copy_(param)


# -----------------------------------------------------------------------------
# S3FD: box_utils + nets + S3FD
# -----------------------------------------------------------------------------
def _nms_dets(dets, thresh):
    x1, y1 = dets[:, 0], dets[:, 1]
    x2, y2 = dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return np.array(keep).astype(int)


def _decode_loc(loc, priors, variances):
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _nms_torch(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        xx1 = torch.clamp(x1[idx], min=x1[i].item())
        yy1 = torch.clamp(y1[idx], min=y1[i].item())
        xx2 = torch.clamp(x2[idx], max=x2[i].item())
        yy2 = torch.clamp(y2[idx], max=y2[i].item())
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        iou = inter / union
        idx = idx[iou.le(overlap)]
    return keep, count


class _PriorBox(object):
    def __init__(
        self,
        input_size,
        feature_maps,
        variance=(0.1, 0.2),
        min_sizes=(16, 32, 64, 128, 256, 512),
        steps=(4, 8, 16, 32, 64, 128),
        clip=False,
    ):
        self.imh, self.imw = input_size[0], input_size[1]
        self.feature_maps = feature_maps
        self.variance = variance
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip

    def forward(self):
        from itertools import product

        mean = []
        for k, fmap in enumerate(self.feature_maps):
            feath, featw = fmap[0], fmap[1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]
                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh
                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh
                mean += [cx, cy, s_kw, s_kh]
        output = torch.FloatTensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class _Detect(object):
    def __init__(
        self,
        num_classes=2,
        top_k=750,
        nms_thresh=0.3,
        conf_thresh=0.05,
        variance=(0.1, 0.2),
        nms_top_k=5000,
    ):
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_top_k = nms_top_k

    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)
        decoded_boxes = _decode_loc(
            loc_data.view(-1, 4), batch_priors, self.variance
        )
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = _nms_torch(
                    boxes_, scores, self.nms_thresh, self.nms_top_k
                )
                count = min(count, self.top_k)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1
                )
        return output


class _L2Norm(torch.nn.Module):
    def __init__(self, n_channels, scale):
        super(_L2Norm, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(n_channels))
        torch.nn.init.constant_(self.weight, scale or 1.0)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        x = x / norm
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x


class _S3FDNet(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(_S3FDNet, self).__init__()
        self.device = device
        self.vgg = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 64, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            torch.nn.Conv2d(256, 512, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, 1024, 3, 1, padding=6, dilation=6),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 1024, 1, 1),
            torch.nn.ReLU(inplace=True),
        ])
        self.L2Norm3_3 = _L2Norm(256, 10)
        self.L2Norm4_3 = _L2Norm(512, 8)
        self.L2Norm5_3 = _L2Norm(512, 5)
        self.extras = torch.nn.ModuleList([
            torch.nn.Conv2d(1024, 256, 1, 1),
            torch.nn.Conv2d(256, 512, 3, 2, padding=1),
            torch.nn.Conv2d(512, 128, 1, 1),
            torch.nn.Conv2d(128, 256, 3, 2, padding=1),
        ])
        self.loc = torch.nn.ModuleList([
            torch.nn.Conv2d(256, 4, 3, 1, padding=1),
            torch.nn.Conv2d(512, 4, 3, 1, padding=1),
            torch.nn.Conv2d(512, 4, 3, 1, padding=1),
            torch.nn.Conv2d(1024, 4, 3, 1, padding=1),
            torch.nn.Conv2d(512, 4, 3, 1, padding=1),
            torch.nn.Conv2d(256, 4, 3, 1, padding=1),
        ])
        self.conf = torch.nn.ModuleList([
            torch.nn.Conv2d(256, 4, 3, 1, padding=1),
            torch.nn.Conv2d(512, 2, 3, 1, padding=1),
            torch.nn.Conv2d(512, 2, 3, 1, padding=1),
            torch.nn.Conv2d(1024, 2, 3, 1, padding=1),
            torch.nn.Conv2d(512, 2, 3, 1, padding=1),
            torch.nn.Conv2d(256, 2, 3, 1, padding=1),
        ])
        self.softmax = torch.nn.Softmax(dim=-1)
        self.detect = _Detect()

    def forward(self, x):
        size = x.size()[2:]
        sources = []
        for k in range(16):
            x = self.vgg[k](x)
        sources.append(self.L2Norm3_3(x))
        for k in range(16, 23):
            x = self.vgg[k](x)
        sources.append(self.L2Norm4_3(x))
        for k in range(23, 30):
            x = self.vgg[k](x)
        sources.append(self.L2Norm5_3(x))
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        for k, v in enumerate(self.extras):
            x = torch.nn.functional.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        loc = []
        conf = []
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())
        features_maps = [[loc[i].size(1), loc[i].size(2)] for i in range(len(loc))]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        with torch.no_grad():
            self.priorbox = _PriorBox(size, features_maps)
            self.priors = self.priorbox.forward()
        output = self.detect.forward(
            loc.view(loc.size(0), -1, 4),
            self.softmax(conf.view(conf.size(0), -1, 2)),
            self.priors.type(type(x.data)).to(self.device),
        )
        return output


S3FD_WEIGHT_PATH = "weights/syncnet/sfd_face.pth"
IMG_MEAN = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")


class S3FD:
    def __init__(self, device="cuda"):
        self.device = device
        self.net = _S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(
            S3FD_WEIGHT_PATH, map_location=self.device, weights_only=False
        )
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.img_mean = torch.FloatTensor([104.0, 117.0, 123.0]).to(self.device)

    def detect_faces(self, image, conf_th=0.8, scales=None):
        if scales is None:
            scales = [1]
        w, h = image.shape[1], image.shape[0]
        bboxes = np.empty(shape=(0, 5))
        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(
                    image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR
                )
                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :].astype("float32")
                scaled_img -= IMG_MEAN
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)
                detections = y.data
                scale = torch.Tensor([w, h, w, h])
                for i in range(detections.size(1)):
                    j = 0
                    while j < detections.size(2) and detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score.item())
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1
            keep = _nms_dets(bboxes, 0.1)
            bboxes = bboxes[keep]
        return bboxes


# -----------------------------------------------------------------------------
# run_syncnet pipeline
# -----------------------------------------------------------------------------
def _bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def _track_shot(opt, scenefaces):
    iouThres = 0.5
    tracks = []
    scenefaces = [list(f) for f in scenefaces]
    while True:
        track = []
        for framefaces in scenefaces:
            for face in list(framefaces):
                if not track:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= opt.num_failed_det:
                    if _bb_iou(face["bbox"], track[-1]["bbox"]) > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                else:
                    break
        if not track:
            break
        if len(track) > opt.min_track:
            framenum = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])
            frame_i = np.arange(framenum[0], framenum[-1] + 1)
            bboxes_i = np.stack(
                [interp1d(framenum, bboxes[:, ij])(frame_i) for ij in range(4)],
                axis=1,
            )
            if max(
                np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]),
            ) > opt.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def _crop_video(opt, track, cropfile):
    flist = sorted(glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg")))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vOut = cv2.VideoWriter(cropfile + "t.avi", fourcc, opt.frame_rate, (224, 224))
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = opt.crop_scale
        bs = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame_pad = np.pad(
            image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110)
        )
        my = dets["y"][fidx] + bsi
        mx = dets["x"][fidx] + bsi
        face = frame_pad[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        vOut.write(cv2.resize(face, (224, 224)))
    audiotmp = os.path.join(opt.tmp_dir, opt.reference, "audio.wav")
    audiostart = track["frame"][0] / opt.frame_rate
    audioend = (track["frame"][-1] + 1) / opt.frame_rate
    vOut.release()
    subprocess.call(
        "ffmpeg -v quiet -y -i %s -ss %.3f -to %.3f %s"
        % (
            os.path.join(opt.avi_dir, opt.reference, "audio.wav"),
            audiostart,
            audioend,
            audiotmp,
        ),
        shell=True,
        stdout=None,
    )
    subprocess.call(
        "ffmpeg -v quiet -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi"
        % (cropfile, audiotmp, cropfile),
        shell=True,
        stdout=None,
    )
    os.remove(cropfile + "t.avi")
    return {"track": track, "proc_track": dets}


def _inference_video(opt):
    DET = S3FD(device="cuda" if torch.cuda.is_available() else "cpu")
    flist = sorted(glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg")))
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
            )
    with open(os.path.join(opt.work_dir, opt.reference, "faces.pckl"), "wb") as f:
        pickle.dump(dets, f)
    return dets


def _scene_detect(opt):
    video_manager = VideoManager(
        os.path.join(opt.avi_dir, opt.reference, "video.avi")
    )
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    with open(os.path.join(opt.work_dir, opt.reference, "scene.pckl"), "wb") as f:
        pickle.dump(
            scene_list
            if scene_list
            else [(video_manager.get_base_timecode(), video_manager.get_current_timecode())],
            f,
        )
    return scene_list if scene_list else [
        (video_manager.get_base_timecode(), video_manager.get_current_timecode())
    ]


_syncnet_inst = None
SYNCNET_CKPT = "weights/syncnet/syncnet_v2.model"


def run_syncnet(videofile, data_dir, start_sec=0.0, clip_sec=5.0):
    if clip_sec <= 0:
        raise ValueError("sync_clip_seconds must be a positive number")

    global _syncnet_inst
    if _syncnet_inst is None:
        _syncnet_inst = SyncNetInstance()
        _syncnet_inst.loadParameters(SYNCNET_CKPT)
    opt = AttrDict({
        "data_dir": data_dir,
        "videofile": videofile,
        "reference": "ref",
        "facedet_scale": 0.25,
        "crop_scale": 0.4,
        "min_track": 25,
        "frame_rate": 25,
        "num_failed_det": 25,
        "min_face_size": 48,
        "vshift": 15,
        "batch_size": 20,
    })
    opt["avi_dir"] = os.path.join(opt["data_dir"], "pyavi")
    opt["tmp_dir"] = os.path.join(opt["data_dir"], "pytmp")
    opt["work_dir"] = os.path.join(opt["data_dir"], "pywork")
    opt["crop_dir"] = os.path.join(opt["data_dir"], "pycrop")
    opt["frames_dir"] = os.path.join(opt["data_dir"], "pyframes")
    for d in [opt["work_dir"], opt["crop_dir"], opt["avi_dir"], opt["frames_dir"], opt["tmp_dir"]]:
        subdir = os.path.join(d, opt["reference"])
        if os.path.exists(subdir):
            rmtree(subdir)
    for d in [opt["work_dir"], opt["crop_dir"], opt["avi_dir"], opt["frames_dir"], opt["tmp_dir"]]:
        os.makedirs(os.path.join(d, opt["reference"]), exist_ok=True)
    out_avi = os.path.join(opt["avi_dir"], opt["reference"], "video.avi")
    subprocess.call(
        "ffmpeg -y -ss %.6f -i %s -t %.6f -qscale:v 2 -async 1 -r 25 -v quiet %s"
        % (start_sec, videofile, clip_sec, out_avi),
        shell=True,
        stdout=None,
    )
    subprocess.call(
        "ffmpeg -y -i %s -qscale:v 2 -threads 1 -v quiet -f image2 %s"
        % (out_avi, os.path.join(opt["frames_dir"], opt["reference"], "%06d.jpg")),
        shell=True,
        stdout=None,
    )
    segment_frames = len(
        glob.glob(os.path.join(opt["frames_dir"], opt["reference"], "*.jpg"))
    )
    if segment_frames == 0:
        subprocess.check_call("rm -rf %s" % data_dir, shell=True)
        return None, np.zeros(1), np.zeros(1), 0
    audio_path = os.path.join(opt["avi_dir"], opt["reference"], "audio.wav")
    ret = subprocess.call(
        "ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 -v quiet %s"
        % (out_avi, audio_path),
        shell=True,
        stdout=None,
    )
    if ret != 0:
        subprocess.call(
            "ffmpeg -y -ss %.6f -i %s -t %.6f -ac 1 -vn -acodec pcm_s16le -ar 16000 -v quiet %s"
            % (start_sec, videofile[:-4] + ".wav", clip_sec, audio_path),
            shell=True,
            stdout=None,
        )
    faces = _inference_video(opt)
    scene = _scene_detect(opt)
    alltracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= opt["min_track"]:
            alltracks.extend(
                _track_shot(opt, faces[shot[0].frame_num : shot[1].frame_num])
            )
    for ii, track in enumerate(alltracks):
        _crop_video(opt, track, os.path.join(opt["crop_dir"], opt["reference"], "%05d" % ii))
    with open(os.path.join(opt["work_dir"], opt["reference"], "tracks.pckl"), "wb") as f:
        pickle.dump(alltracks, f)
    if os.path.exists(os.path.join(opt["tmp_dir"], opt["reference"])):
        rmtree(os.path.join(opt["tmp_dir"], opt["reference"]))
    flist = sorted(glob.glob(os.path.join(opt["crop_dir"], opt["reference"], "0*.avi")))
    if len(flist) == 0:
        subprocess.check_call("rm -rf %s" % data_dir, shell=True)
        return None, np.zeros(1), np.zeros(1), segment_frames
    offset, conf, dist = _syncnet_inst.evaluate(opt, videofile=flist[0])
    subprocess.check_call("rm -rf %s" % data_dir, shell=True)
    return offset, conf, dist, segment_frames


# -----------------------------------------------------------------------------
# evaluate_gt: load_video_with_fps, cal_qalign, evaluate_gt
# -----------------------------------------------------------------------------
def iter_video_with_fps(path, fps=25):
    with TemporaryDirectory(dir=_PROJECT_TMP) as tmp:
        video_cap = cv2.VideoCapture(path)
        original_fps = video_cap.get(cv2.CAP_PROP_FPS)
        if original_fps is None or original_fps <= 0:
            original_fps = fps
        original_fps = math.ceil(original_fps)
        if original_fps != fps:
            tmp_path = os.path.join(tmp, "tmp.mp4")
            subprocess.run(
                ["ffmpeg", "-i", path, "-r", str(fps), tmp_path, "-v", "quiet", "-y"],
                check=True,
            )
            video_cap.release()
            video_cap = cv2.VideoCapture(tmp_path)
        try:
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        finally:
            video_cap.release()


def load_video_with_fps(path, fps=25):
    # Legacy entry: load every frame in one shot.
    return list(iter_video_with_fps(path, fps))


# Use a script-relative absolute path so CWD changes do not turn this into a Hub model id.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONE_ALIGN_PATH = os.path.join(_SCRIPT_DIR, "weights", "q-align")
_qalign_model = None


def _to_1d_float64(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=torch.float64).numpy().reshape(-1)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _iter_chunks(iterator, chunk_size):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def cal_qalign(frames, chunk_size=64):
    global _qalign_model
    if _qalign_model is None:
        _qalign_model = AutoModelForCausalLM.from_pretrained(
            ONE_ALIGN_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
    if chunk_size <= 0:
        raise ValueError("qalign_chunk_size must be a positive integer")

    total_qua = 0.0
    total_aes = 0.0
    total_count = 0

    for frame_chunk in _iter_chunks(frames, chunk_size):
        pil_chunk = [Image.fromarray(f) for f in frame_chunk]
        qua = _to_1d_float64(
            _qalign_model.score(pil_chunk, task_="quality", input_="image")
        )
        aes = _to_1d_float64(
            _qalign_model.score(pil_chunk, task_="aesthetic", input_="image")
        )
        total_qua += float(np.sum(qua))
        total_aes += float(np.sum(aes))
        total_count += len(frame_chunk)

    if total_count == 0:
        raise RuntimeError("No frame decoded from video; cannot compute Q-Align scores.")

    return total_qua / total_count, total_aes / total_count


def evaluate_gt(gt_video_path):
    """Compute Sync-C, Sync-D, IQA, Aesthe for a single video; self-contained for portability."""
    metrics = {}
    if args.sync_clip_seconds <= 0:
        raise ValueError("sync_clip_seconds must be a positive number")

    weighted_sync_c = 0.0
    weighted_sync_d = 0.0
    total_sync_frames = 0
    start_sec = 0.0
    while True:
        with TemporaryDirectory(dir=_PROJECT_TMP) as tmpdir:
            offset, sync_c, sync_d, segment_frames = run_syncnet(
                gt_video_path,
                tmpdir,
                start_sec=start_sec,
                clip_sec=args.sync_clip_seconds,
            )
        if segment_frames <= 0:
            break
        if offset is not None:
            sync_c_val = float(np.asarray(sync_c).item())
            sync_d_val = float(np.asarray(sync_d).item())
            weighted_sync_c += sync_c_val * segment_frames
            weighted_sync_d += sync_d_val * segment_frames
            total_sync_frames += segment_frames
        start_sec += args.sync_clip_seconds

    if total_sync_frames > 0:
        metrics["Sync-C"] = weighted_sync_c / total_sync_frames
        metrics["Sync-D"] = weighted_sync_d / total_sync_frames
    real_frames = iter_video_with_fps(gt_video_path, args.fps)
    qua, aes = cal_qalign(real_frames, chunk_size=args.qalign_chunk_size)
    metrics["IQA"] = qua
    metrics["Aesthe"] = aes
    return metrics


def refresh_average_row(result_file):
    """Recompute the __AVERAGE__ row in result.csv, ignoring empty values to avoid duplicate appends."""
    if not os.path.exists(result_file):
        return

    with open(result_file, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    header = rows[0]
    data_rows = []
    for row in rows[1:]:
        if not row:
            continue
        if row[0].strip() == "__AVERAGE__":
            continue
        data_rows.append(row)

    if not data_rows:
        return

    col_indices = [1, 2, 3, 4]  # Sync-C, Sync-D, IQA, Aesthe
    means = []
    for idx in col_indices:
        values = []
        for row in data_rows:
            if idx >= len(row):
                continue
            cell = row[idx].strip()
            if not cell:
                continue
            try:
                values.append(float(cell))
            except ValueError:
                continue
        means.append(sum(values) / len(values) if values else "")

    avg_row = ["__AVERAGE__"] + [
        ("%.10f" % v) if isinstance(v, float) else "" for v in means
    ]

    with open(result_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
        writer.writerow(avg_row)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    # Derive result_file from --video_path when not provided explicitly.
    if args.result_file is None:
        if os.path.isdir(args.video_path):
            result_file = os.path.join(args.video_path, "result.csv")
        elif os.path.isfile(args.video_path):
            result_file = os.path.join(os.path.dirname(args.video_path), "result.csv")
        else:
            result_file = "result.csv"
    else:
        result_file = args.result_file

    if os.path.isdir(args.video_path):
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}
        candidates = []
        for ext in video_exts:
            candidates.extend(glob.glob(os.path.join(args.video_path, f"*{ext}")))
            candidates.extend(glob.glob(os.path.join(args.video_path, f"*{ext.upper()}")))
        video_paths = sorted(set(candidates))
        if not video_paths:
            print("Error: no video files found in directory:", args.video_path)
            sys.exit(1)
    elif os.path.isfile(args.video_path):
        video_paths = [args.video_path]
    else:
        print("Error: path does not exist:", args.video_path)
        sys.exit(1)

    # Create CSV with header if it does not exist yet.
    file_exists = os.path.exists(result_file)

    for video_path in video_paths:
        metrics = evaluate_gt(video_path)
        lines = [
            "Evaluating: %s" % video_path,
            "========== Result ==========",
        ]
        for k, v in metrics.items():
            lines.append("  %s: %s" % (k, v))
        lines.append("============================")
        lines.append("")
        lines.append("")
        text = "\n".join(lines)
        print(text)

        with open(result_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["video_path", "Sync-C", "Sync-D", "IQA", "Aesthe"])
                file_exists = True
            writer.writerow([
                video_path,
                metrics.get("Sync-C", ""),
                metrics.get("Sync-D", ""),
                metrics.get("IQA", ""),
                metrics.get("Aesthe", ""),
            ])

    refresh_average_row(result_file)


if __name__ == "__main__":
    main()
