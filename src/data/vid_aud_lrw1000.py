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
from PIL import Image
import librosa

info = {}
info['video_fps'] = 25
info['audio_fps'] = 16000

class MultiDataset(Dataset):
    def __init__(self, lrw, mode, max_v_timesteps=57, augmentations=False):
        assert mode in ['train', 'test', 'val']
        self.dir = lrw
        self.max_v_timesteps = max_v_timesteps
        self.augmentations = augmentations if mode == 'train' else False
        self.file_paths, self.annotations, self.word_list = self.build_file_list(lrw, mode)
        self.word2int = {word: index for index, word in enumerate(self.word_list)}

    def build_file_list(self, lrw, mode):
        vid_aud = {}
        with open(os.path.join(lrw, 'info', 'all_audio_video.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for l in lines:
            vid, aud, _, label, start, end = l.strip().split(',')
            start = float(start)
            end = float(end)
            vid_aud[f'{vid}/{label}/{start:.2f}/{end:.2f}'] = aud

        file_list = []
        annotation = {}
        words = []

        if mode == 'train':
            m = 'trn'
        elif mode == 'val':
            m = 'val'
        else:
            m = 'tst'

        with open(os.path.join(lrw, 'info', m + '_1000.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for l in lines:
            vid, _, word, start, end = l.strip().split(',')
            start = float(start)
            end = float(end)
            if (end - start) != 0 and int(end * 25 - start * 25) != 0:
                file_list.append(f'{vid}/{word}/{start:.2f}/{end:.2f}')
                annotation[f'{vid}/{word}/{start:.2f}/{end:.2f}'] = [vid_aud[f'{vid}/{word}/{start:.2f}/{end:.2f}'], word, start, end]
                if word not in words:
                    words.append(word)
        del vid_aud
        return list(np.unique(file_list)), annotation, words

    def __len__(self):
        return len(self.file_paths)

    def gen_video(self, f_name, start_f, end_f):
        files = os.listdir(os.path.join(self.dir, 'images', f_name))
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = list(filter(lambda file: int(os.path.splitext(file)[0]) <= end_f and int(os.path.splitext(file)[0]) >= start_f, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        try:
            array = [np.array(Image.open(os.path.join(self.dir, 'images', f_name, file)).resize([112, 112])) for file in files]
            array = np.stack(array, axis=0)
        except:
            array = np.zeros([1, 112, 112, 1]).astype('uint8')
        return array

    def build_tensor(self, frames):
        if self.augmentations:
            augmentations1 = transforms.Compose([StatefulRandomHorizontalFlip(0.5)])
        else:
            augmentations1 = transforms.Compose([])

        transform = transforms.Compose([
            transforms.ToPILImage(),
            augmentations1,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.4136, 0.1700)
        ])

        temporalVolume = torch.zeros(np.shape(frames)[0], 1, 112, 112)
        for i, frame in enumerate(frames):
            temporalVolume[i] = transform(frame)

        ### Random Erasing ###
        if self.augmentations:
            x_s, y_s = [random.randint(-10, 66) for _ in range(2)]  # starting point
            temporalVolume[:, :, np.maximum(0, y_s):np.minimum(112, y_s + 56), np.maximum(0, x_s):np.minimum(112, x_s + 56)] = 0.

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, T, H, W)
        return temporalVolume

    def __getitem__(self, idx):
        skip = 0
        file_path = self.file_paths[idx]
        aud_file, content, start, end = self.annotations[file_path]
        target = self.word2int[content]

        start_time = np.maximum(0, start - 0.2)
        end_time = end + 0.2

        start_frame = int(start_time * 25.0) + 1
        end_frame = int(end_time * 25.0) + 1

        video = self.gen_video(file_path.split('/')[0], start_frame, end_frame)   #T,H,W,C
        if video.shape[0] == 1:
            skip = 1

        try:
            aud, _ = librosa.load(os.path.join(self.dir, 'audio', aud_file + '.wav'), sr=16000)
            aud = torch.tensor(aud).unsqueeze(0)
        except:
            aud = torch.zeros([1, 100])
            skip = 1

        aud_pt_size = aud.size(1) # 1,aud_pt_size
        video_duration = video.shape[0] / info['video_fps']  # 1.16sec
        pad_num = (video_duration * info['audio_fps'] - aud_pt_size) // 2    # 64
        if pad_num > 0:
            pad_fr = torch.zeros([1, int(pad_num)])
            pad_bk = torch.zeros([1, int(video_duration * info['audio_fps']) - aud_pt_size - int(pad_num)])
            aud = torch.cat([pad_fr, aud, pad_bk], 1)
        elif pad_num == 0 and aud_pt_size < int(video_duration * info['audio_fps']):
            pad = torch.zeros([1, int(video_duration * info['audio_fps']) - aud_pt_size])
            aud = torch.cat([aud, pad], 1)
        else:
            cen_aud = aud.size(1) // 2
            aud_st = cen_aud - int((video_duration * info['audio_fps']) // 2)
            aud_bk = cen_aud - int((video_duration * info['audio_fps']) // 2) + int(video_duration * info['audio_fps'])
            aud = aud[:, aud_st:aud_bk]

        assert aud.size(1) == int(video_duration * info['audio_fps']), f"aud length :{aud.size(1)}, target length :{int(video_duration * info['audio_fps'])}, aud_file :{aud_file}"

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

        ## Video ##
        num_v_frames = video.shape[0]

        frames = self.build_tensor(video)
        spec = spec[:, :, : num_v_frames * 4]
        num_a_frames = spec.size(2)

        return spec, num_a_frames, frames, num_v_frames, target, skip

    def collate(self, batch):
        vid_length = [data[3] for data in batch if data[5] != 1]
        max_vid_length = max(vid_length)

        padded_features = []
        padded_spec = []
        targets = []

        #vid: CTHW
        for i, (spec, spec_len, vid, vid_len, target, skip) in enumerate(batch):
            if not skip == 1:
                vid_pad = torch.zeros([1, max_vid_length - vid_len, 1, 1]).repeat(vid.size(0), 1, vid.size(2), vid.size(3))
                padded_features.append(torch.cat([vid, vid_pad], 1))
                padded_spec.append(nn.ConstantPad2d((0, max_vid_length * 4 - spec_len, 0, 0), 0.)(spec))
                targets.append(target)

        vid = torch.stack(padded_features, 0).float()
        vid_length = torch.IntTensor(vid_length)
        targets = torch.LongTensor(targets)
        spec = torch.stack(padded_spec, 0).float()

        return spec, vid, vid_length, targets

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