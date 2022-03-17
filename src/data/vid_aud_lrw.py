import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.data.transforms import Crop, StatefulRandomHorizontalFlip


class MultiDataset(Dataset):
    def __init__(self, lrw, mode, max_v_timesteps=155, augmentations=False, num_mel_bins=80):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.max_v_timesteps = max_v_timesteps
        self.augmentations = augmentations if mode == 'train' else False
        self.num_mel_bins = num_mel_bins
        self.skip_long_samples = True
        self.file_paths, self.word_list = self.build_file_list(lrw, mode)
        self.word2int = {word: index for index, word in self.word_list.items()}

    def build_file_list(self, lrw, mode):
        file_list = []
        word = {}

        classes = sorted(os.listdir(lrw))
        for i, cla in enumerate(classes):
            word[i] = cla
            modes_dir = os.path.join(lrw, cla)
            modes = sorted(os.listdir(modes_dir))
            for m in modes:
                if mode in m:
                    file_dir = os.path.join(modes_dir, m)
                    files = sorted(os.listdir(file_dir))
                    for file in files:
                        if '.mp4' in file:
                            file_list.append(os.path.join(file_dir, file))

        return file_list, word

    def __len__(self):
        return len(self.file_paths)

    def build_tensor(self, frames):
        if self.augmentations:
            x, y = [random.randint(-5, 5) for _ in range(2)]
        else:
            x, y = 0, 0
        crop = [59 + x, 95 + y, 195 + x, 231 + y]  # 136, 136
        if self.augmentations:
            augmentations1 = transforms.Compose([StatefulRandomHorizontalFlip(0.5)])
        else:
            augmentations1 = transforms.Compose([])

        transform = transforms.Compose([
            transforms.ToPILImage(),
            Crop(crop),  # 83.288
            transforms.Resize([112, 112]),
            augmentations1,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.4136, 0.1700)
        ])

        temporalVolume = torch.zeros(self.max_v_timesteps, 1, 112, 112)
        for i, frame in enumerate(frames):
            temporalVolume[i] = transform(frame)

        ### Random Spatial Erasing ###
        if self.augmentations:
            x_s, y_s = [random.randint(-10, 66) for _ in range(2)]  # starting point
            temporalVolume[:, :, np.maximum(0, y_s):np.minimum(112, y_s + 56), np.maximum(0, x_s):np.minimum(112, x_s + 56)] = 0.

        ### Random Temporal Erasing ###
        if self.augmentations:
            t_s = random.randint(0, 29 - 3)  # starting point
            temporalVolume[t_s:t_s + 3, :, :, :] = 0.

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, T, H, W)
        return temporalVolume

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        content = os.path.split(file_path)[-1].split('_')[0]
        target = self.word2int[content]

        video, aud, info = torchvision.io.read_video(file_path, pts_unit='sec')

        aud_pt_size = aud.size(1) # 1,aud_pt_size
        video_duration = video.size(0) / info['video_fps']  # 1.16sec
        pad_num = (video_duration * info['audio_fps'] - aud_pt_size) // 2    # 64
        if pad_num > 0:
            pad_fr = torch.zeros([1, int(pad_num)])
            pad_bk = torch.zeros([1, int(video_duration * info['audio_fps'] - aud_pt_size - pad_num)])
            aud = torch.cat([pad_fr, aud, pad_bk], 1)
        else:
            cen_aud = aud.size(1) // 2
            aud_st = cen_aud - int((video_duration * info['audio_fps']) // 2)
            aud_bk = cen_aud - int((video_duration * info['audio_fps']) // 2) + int(video_duration * info['audio_fps'])
            aud = aud[:, aud_st:aud_bk]

        assert aud.size(1) == int(video_duration * info['audio_fps'])

        ## Audio ##
        if self.augmentations:
            transform = nn.Sequential(
                torchaudio.transforms.Spectrogram(win_length=400, hop_length=160, power=None, normalized=True), # 100 fps (hop_length 10ms)
                torchaudio.transforms.ComplexNorm(2),
                torchaudio.transforms.MelScale(n_mels=80, sample_rate=16000),
                torchaudio.transforms.AmplitudeToDB(),
                CMVN(),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
                torchaudio.transforms.TimeMasking(time_mask_param=20)
            )
        else:
            transform = nn.Sequential(
                torchaudio.transforms.Spectrogram(win_length=400, hop_length=160, power=None, normalized=True), #100 fps (hop_length 10ms)
                torchaudio.transforms.ComplexNorm(2),
                torchaudio.transforms.MelScale(n_mels=80, sample_rate=16000),
                torchaudio.transforms.AmplitudeToDB(),
                CMVN()
            )

        spec = transform(aud)  # 1, 80, time*100 : C,F,T
        spec_cen = spec.size(2)//2

        ## Video ##
        video = video.permute(0, 3, 1, 2)  # T C H W
        num_v_frames = video.size(0)

        frames = self.build_tensor(video)
        start_spec = np.maximum(0, spec_cen - num_v_frames * 2)
        end_spec = np.minimum(spec.size(2), spec_cen + num_v_frames * 2)
        spec = spec[:, :, start_spec:end_spec]
        num_a_frames = spec.size(2)
        spec = nn.ConstantPad2d((0, self.max_v_timesteps * 4 - num_a_frames, 0, 0), 0.0)(spec)

        return spec, frames, target


class CMVN(torch.jit.ScriptModule):
    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)