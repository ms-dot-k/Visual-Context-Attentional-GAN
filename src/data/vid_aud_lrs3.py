import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.data.transforms import Crop, StatefulRandomHorizontalFlip
from PIL import Image
import librosa
import cv2
from matplotlib import pyplot as plt
import glob
from scipy import signal
import torchvision
from torch.autograd import Variable
from librosa.filters import mel as librosa_mel_fn
from src.data.audio_processing import dynamic_range_compression, dynamic_range_decompression, griffin_lim
from src.data.stft import STFT
import math
log1e5 = math.log(1e-5)

class MultiDataset(Dataset):
    def __init__(self, data, mode, max_v_timesteps=155, window_size=40, augmentations=False, num_mel_bins=80, fast_validate=False, f_min=55., f_max=7600.):
        assert mode in ['pretrain', 'train', 'test', 'val']
        self.data = data
        self.sample_window = True if (mode == 'pretrain' or mode == 'train') else False
        self.fast_validate = fast_validate
        self.max_v_timesteps = window_size if self.sample_window else max_v_timesteps
        self.window_size = window_size
        self.augmentations = augmentations if mode == 'train' else False
        self.num_mel_bins = num_mel_bins
        self.file_paths, self.file_names, self.crops = self.build_file_list(data, mode)
        self.stft = TacotronSTFT(filter_length=640, hop_length=160, win_length=640, n_mel_channels=num_mel_bins, sampling_rate=16000, mel_fmin=f_min, mel_fmax=f_max)

    def build_file_list(self, lrs3, mode):
        file_list, paths = [], []
        crops = {}

        ## LRS3 crop (lip centered axis) load ##
        file = open(f"./data/LRS3/LRS3_crop/preprocess_pretrain.txt", "r")
        content = file.read()
        file.close()
        for i, line in enumerate(content.splitlines()):
            split = line.split(".")
            file = split[0]
            crop_str = split[1][4:]
            crops['pretrain/' + file] = crop_str
        file = open(f"./data/LRS3/LRS3_crop/preprocess_test.txt", "r")
        content = file.read()
        file.close()
        for i, line in enumerate(content.splitlines()):
            split = line.split(".")
            file = split[0]
            crop_str = split[1][4:]
            crops['test/' + file] = crop_str
        file = open(f"./data/LRS3/LRS3_crop/preprocess_trainval.txt", "r")
        content = file.read()
        file.close()
        for i, line in enumerate(content.splitlines()):
            split = line.split(".")
            file = split[0]
            crop_str = split[1][4:]
            crops['trainval/' + file] = crop_str

        ## LRS3 file lists##
        file = open(f"./data/LRS3/lrs3_unseen_{mode}.txt", "r")
        content = file.read()
        file.close()
        for file in content.splitlines():
            if file in crops:
                file_list.append(file)
                paths.append(f"{lrs3}/{file}")

        print(f'Mode: {mode}, File Num: {len(file_list)}')
        return paths, file_list, crops

    def build_tensor(self, frames, crops):
        if self.augmentations:
            s = random.randint(-5, 5)
        else:
            s = 0
        crop = []
        for i in range(0, len(crops), 2):
            left = int(crops[i]) - 40 + s
            upper = int(crops[i + 1]) - 40 + s
            right = int(crops[i]) + 40 + s
            bottom = int(crops[i + 1]) + 40 + s
            crop.append([left, upper, right, bottom])
        crops = crop

        if self.augmentations:
            augmentations1 = transforms.Compose([StatefulRandomHorizontalFlip(0.5)])
        else:
            augmentations1 = transforms.Compose([])

        temporalVolume = torch.zeros(self.max_v_timesteps, 1, 112, 112)
        for i, frame in enumerate(frames):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                Crop(crops[i]),
                transforms.Resize([112, 112]),
                augmentations1,
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(0.4136, 0.1700),
            ])
            temporalVolume[i] = transform(frame)

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, T, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file = self.file_names[idx]
        file_path = self.file_paths[idx]
        crops = self.crops[file].split("/")

        info = {}
        info['video_fps'] = 25
        cap = cv2.VideoCapture(file_path + '.mp4')
        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        cap.release()
        audio, info['audio_fps'] = librosa.load(file_path.replace('LRS3-TED', 'LRS3-TED_audio') + '.wav', sr=16000)
        vid = torch.tensor(np.stack(frames, 0))
        audio = torch.tensor(audio).unsqueeze(0)

        if not 'video_fps' in info:
            info['video_fps'] = 25
            info['audio_fps'] = 16000

        assert vid.size(0) > 5 or audio.size(1) > 5

        ## Audio ##
        audio = audio / torch.abs(audio).max() * 0.9
        aud = torch.FloatTensor(self.preemphasize(audio.squeeze(0))).unsqueeze(0)
        aud = torch.clamp(aud, min=-1, max=1)

        melspec, spec = self.stft.mel_spectrogram(aud)

        ## Video ##
        vid = vid.permute(0, 3, 1, 2)  # T C H W

        if self.sample_window:
            vid, melspec, spec, audio, crops = self.extract_window(vid, melspec, spec, audio, info, crops)
        elif vid.size(0) > self.max_v_timesteps:
            print('Sample is longer than Max video frames! Trimming to the length of ', self.max_v_timesteps)
            vid = vid[:self.max_v_timesteps]
            melspec = melspec[:, :, :int(self.max_v_timesteps * info['audio_fps'] / info['video_fps'] / 160)]
            spec = spec[:, :, :int(self.max_v_timesteps * info['audio_fps'] / info['video_fps'] / 160)]
            audio = audio[:, :int(self.max_v_timesteps * info['audio_fps'] / info['video_fps'])]
            crops = crops[:self.max_v_timesteps * 2]

        num_v_frames = vid.size(0)
        vid = self.build_tensor(vid, crops)

        melspec = self.normalize(melspec)    #0~2 --> -1~1

        spec = self.normalize_spec(spec)    # 0 ~ 1
        spec = self.stft.spectral_normalize(spec)   # log(1e-5) ~ 0 # in log scale
        spec = self.normalize(spec)   # -1 ~ 1

        num_a_frames = melspec.size(2)
        melspec = nn.ConstantPad2d((0, self.max_v_timesteps * 4 - num_a_frames, 0, 0), -1.0)(melspec)
        spec = nn.ConstantPad2d((0, self.max_v_timesteps * 4 - num_a_frames, 0, 0), -1.0)(spec)

        return melspec, spec, vid, num_v_frames, audio.squeeze(0), num_a_frames, file_path.replace(self.data, '')[1:]

    def extract_window(self, vid, mel, spec, aud, info, crops):
        # vid : T,C,H,W
        st_fr = random.randint(0, max(0, vid.size(0) - self.window_size))
        vid = vid[st_fr:st_fr + self.window_size]
        crops = crops[st_fr * 2: st_fr * 2 + self.window_size * 2]

        assert vid.size(0) * 2 == len(crops), f'vid length: {vid.size(0)}, crop length: {len(crops)}'

        st_mel_fr = int(st_fr * info['audio_fps'] / info['video_fps'] / 160)
        mel_window_size = int(self.window_size * info['audio_fps'] / info['video_fps'] / 160)
        mel = mel[:, :, st_mel_fr:st_mel_fr + mel_window_size]
        spec = spec[:, :, st_mel_fr:st_mel_fr + mel_window_size]
        aud = aud[:, st_mel_fr*160:st_mel_fr*160 + mel_window_size*160]
        aud = torch.cat([aud, torch.zeros([1, int(self.window_size / info['video_fps'] * info['audio_fps'] - aud.size(1))])], 1)

        return vid, mel, spec, aud, crops

    def collate_fn(self, batch):
        vid_lengths, spec_lengths, padded_spec_lengths, aud_lengths = [], [], [], []
        for data in batch:
            vid_lengths.append(data[3])
            spec_lengths.append(data[5])
            padded_spec_lengths.append(data[0].size(2))
            aud_lengths.append(data[4].size(0))

        max_aud_length = max(aud_lengths)
        max_spec_length = max(padded_spec_lengths)
        padded_vid = []
        padded_melspec = []
        padded_spec = []
        padded_audio = []
        f_names = []

        for i, (melspec, spec, vid, num_v_frames, audio, spec_len, f_name) in enumerate(batch):
            padded_vid.append(vid)  # B, C, T, H, W
            padded_melspec.append(nn.ConstantPad2d((0, max_spec_length - melspec.size(2), 0, 0), -1.0)(melspec))
            padded_spec.append(nn.ConstantPad2d((0, max_spec_length - spec.size(2), 0, 0), -1.0)(spec))
            padded_audio.append(torch.cat([audio, torch.zeros([max_aud_length - audio.size(0)])], 0))
            f_names.append(f_name)

        vid = torch.stack(padded_vid, 0).float()
        vid_length = torch.IntTensor(vid_lengths)
        melspec = torch.stack(padded_melspec, 0).float()
        spec = torch.stack(padded_spec, 0).float()
        spec_length = torch.IntTensor(spec_lengths)
        audio = torch.stack(padded_audio, 0).float()

        return melspec, spec, vid, vid_length, audio, spec_length, f_names

    def inverse_mel(self, mel, stft):
        if len(mel.size()) < 4:
            mel = mel.unsqueeze(0)  #B,1,80,T

        mel = self.denormalize(mel)
        mel = stft.spectral_de_normalize(mel)
        mel = mel.transpose(2, 3).contiguous()   #B,80,T --> B,T,80
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.matmul(mel, stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(2, 3).squeeze(1)   # B,1,F,T
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        wav = griffin_lim(spec_from_mel, stft.stft_fn, 60).squeeze(1) #B,L
        wav = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
        wavs = []
        for w in wav:
            w = self.deemphasize(w)
            wavs += [w]
        wavs = np.array(wavs)
        wavs = np.clip(wavs, -1, 1)
        return wavs

    def inverse_spec(self, spec, stft):
        if len(spec.size()) < 4:
            spec = spec.unsqueeze(0)  #B,1,321,T

        spec = self.denormalize(spec)  # log1e5 ~ 0
        spec = stft.spectral_de_normalize(spec)  # 0 ~ 1
        spec = self.denormalize_spec(spec)  # 0 ~ 14
        wav = griffin_lim(spec.squeeze(1), stft.stft_fn, 60).squeeze(1) #B,L
        wav = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
        wavs = []
        for w in wav:
            w = self.deemphasize(w)
            wavs += [w]
        wavs = np.array(wavs)
        wavs = np.clip(wavs, -1, 1)
        return wavs

    def preemphasize(self, aud):
        aud = signal.lfilter([1, -0.97], [1], aud)
        return aud

    def deemphasize(self, aud):
        aud = signal.lfilter([1], [1, -0.97], aud)
        return aud

    def normalize(self, melspec):
        melspec = ((melspec - log1e5) / (-log1e5 / 2)) - 1    #0~2 --> -1~1
        return melspec

    def denormalize(self, melspec):
        melspec = ((melspec + 1) * (-log1e5 / 2)) + log1e5
        return melspec

    def normalize_spec(self, spec):
        spec = (spec - spec.min()) / (spec.max() - spec.min())  # 0 ~ 1
        return spec

    def denormalize_spec(self, spec):
        spec = spec * 14. # 0 ~ 14
        return spec

    def plot_spectrogram_to_numpy(self, mels):
        fig, ax = plt.subplots(figsize=(15, 4))
        im = ax.imshow(np.squeeze(mels, 0), aspect="auto", origin="lower",
                       interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = self.save_figure_to_numpy(fig)
        plt.close()
        return data

    def save_figure_to_numpy(self, fig):
        # save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data.transpose(2, 0, 1)

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output, magnitudes
