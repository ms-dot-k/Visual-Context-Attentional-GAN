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
    def __init__(self, grid, mode, max_v_timesteps=155, window_size=40, subject=None, augmentations=False, num_mel_bins=80, fast_validate=False):
        assert mode in ['train', 'test', 'val']
        self.grid = grid
        self.mode = mode
        self.sample_window = True if mode == 'train' else False
        self.fast_validate = fast_validate
        self.max_v_timesteps = window_size if self.sample_window else max_v_timesteps
        self.window_size = window_size
        self.augmentations = augmentations if mode == 'train' else False
        self.num_mel_bins = num_mel_bins
        self.file_paths = self.build_file_list(grid, mode, subject)
        self.f_min = 55.
        self.f_max = 7500.
        self.stft = TacotronSTFT(filter_length=640, hop_length=160, win_length=640, n_mel_channels=80, sampling_rate=16000, mel_fmin=self.f_min, mel_fmax=self.f_max)

    def build_file_list(self, grid, mode, subject):
        file_list = []
        if subject == 'overlap':
            if mode == 'train':
                with open('./data/overlap_train.txt', 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    file = l.strip().replace('mpg_6000/', '') + '.mp4'
                    file_list.append(os.path.join(grid, file))
            else:
                with open('./data/overlap_val.txt', 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    file = l.strip().replace('mpg_6000/', '') + '.mp4'
                    file_list.append(os.path.join(grid, file))
        elif subject == 'unseen':
            with open('./data/unseen_splits.txt', 'r') as f:
                lines = f.readlines()
            for l in lines:
                if mode in l.strip():
                    _, sub, fname = l.strip().split('/')
                    file = f'{sub}/video/{fname}.mp4'
                    if os.path.exists(os.path.join(grid, file)):
                        file_list.append(os.path.join(grid, file))
        else:
            if mode == 'train':
                with open('./data/train_4.txt', 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    file = l.strip()
                    if subject == 'four':
                        file_list.append(os.path.join(grid, file))
                    elif file.split('/')[0] == subject:
                        file_list.append(os.path.join(grid, file))
            elif mode == 'val':
                with open('./data/val_4.txt', 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    file = l.strip()
                    if subject == 'four':
                        file_list.append(os.path.join(grid, file))
                    elif file.split('/')[0] == subject:
                        file_list.append(os.path.join(grid, file))
            else:
                with open('./data/test_4.txt', 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    file = l.strip()
                    if subject == 'four':
                        file_list.append(os.path.join(grid, file))
                    elif file.split('/')[0] == subject:
                        file_list.append(os.path.join(grid, file))
        return file_list

    def build_tensor(self, frames):
        if self.augmentations:
            augmentations1 = transforms.Compose([StatefulRandomHorizontalFlip(0.5)])
        else:
            augmentations1 = transforms.Compose([])
        crop = [59, 95, 195, 231]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            Crop(crop),
            transforms.Resize([112, 112]),
            augmentations1,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.4136, 0.1700)
        ])

        temporalVolume = torch.zeros(self.max_v_timesteps, 1, 112, 112)
        for i, frame in enumerate(frames):
            temporalVolume[i] = transform(frame)

        ### Random Erasing ###
        if self.augmentations:
            x_s, y_s = [random.randint(-10, 66) for _ in range(2)]  # starting point
            temporalVolume[:, :, np.maximum(0, y_s):np.minimum(112, y_s + 56), np.maximum(0, x_s):np.minimum(112, x_s + 56)] = 0.

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, T, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        vid, _, info = torchvision.io.read_video(file_path, pts_unit='sec')
        audio, info['audio_fps'] = librosa.load(file_path.replace('video', 'audio')[:-4] + '.flac', sr=16000)
        audio = torch.FloatTensor(audio).unsqueeze(0)

        if not 'video_fps' in info:
            info['video_fps'] = 25
            info['audio_fps'] = 16000

        if vid.size(0) < 5 or audio.size(1) < 5:
            vid = torch.zeros([1, 112, 112, 3])
            audio = torch.zeros([1, 16000//25])

        ## Audio ##
        aud = audio / torch.abs(audio).max() * 0.9
        aud = torch.FloatTensor(self.preemphasize(aud.squeeze(0))).unsqueeze(0)
        aud = torch.clamp(aud, min=-1, max=1)

        melspec, spec = self.stft.mel_spectrogram(aud)

        ## Video ##
        vid = vid.permute(0, 3, 1, 2)  # T C H W

        if self.sample_window:
            vid, melspec, spec, audio = self.extract_window(vid, melspec, spec, audio, info)

        num_v_frames = vid.size(0)
        vid = self.build_tensor(vid)

        melspec = self.normalize(melspec)

        num_a_frames = melspec.size(2)
        melspec = nn.ConstantPad2d((0, self.max_v_timesteps * 4 - num_a_frames, 0, 0), 0.0)(melspec)
        spec = nn.ConstantPad2d((0, self.max_v_timesteps * 4 - num_a_frames, 0, 0), 0.0)(spec)

        if not self.sample_window:
            audio = audio[:, :self.max_v_timesteps * 4 * 160]
            audio = torch.cat([audio, torch.zeros([1, int(self.max_v_timesteps / info['video_fps'] * info['audio_fps'] - aud.size(1))])], 1)

        if self.mode == 'test':
            return melspec, spec, vid, num_v_frames, audio.squeeze(0), num_a_frames, file_path.replace(self.grid, '')[1:-4]
        else:
            return melspec, spec, vid, num_v_frames, audio.squeeze(0), num_a_frames

    def extract_window(self, vid, mel, spec, aud, info):
        # vid : T,C,H,W
        vid_2_aud = info['audio_fps'] / info['video_fps'] / 160

        st_fr = random.randint(0, vid.size(0) - self.window_size)
        vid = vid[st_fr:st_fr + self.window_size]

        st_mel_fr = int(st_fr * vid_2_aud)
        mel_window_size = int(self.window_size * vid_2_aud)

        mel = mel[:, :, st_mel_fr:st_mel_fr + mel_window_size]
        spec = spec[:, :, st_mel_fr:st_mel_fr + mel_window_size]

        aud = aud[:, st_mel_fr*160:st_mel_fr*160 + mel_window_size*160]
        aud = torch.cat([aud, torch.zeros([1, int(self.window_size / info['video_fps'] * info['audio_fps'] - aud.size(1))])], 1)

        return vid, mel, spec, aud

    def inverse_mel(self, mel, stft):
        if len(mel.size()) < 4:
            mel = mel.unsqueeze(0)  # B,1,80,T

        mel = self.denormalize(mel)
        mel = stft.spectral_de_normalize(mel)
        mel = mel.transpose(2, 3).contiguous()  # B,80,T --> B,T,80
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.matmul(mel, stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(2, 3).squeeze(1)  # B,1,F,T
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        wav = griffin_lim(spec_from_mel, stft.stft_fn, 60).squeeze(1)  # B,L
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
            spec = spec.unsqueeze(0)  # B,1,321,T

        wav = griffin_lim(spec.squeeze(1), stft.stft_fn, 60).squeeze(1)  # B,L
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

    def audio_preprocessing(self, aud):
        fc = self.f_min
        w = fc / (16000 / 2)
        b, a = signal.butter(7, w, 'high')
        aud = aud.squeeze(0).numpy()
        aud = signal.filtfilt(b, a, aud)
        return torch.tensor(aud.copy()).unsqueeze(0)

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
