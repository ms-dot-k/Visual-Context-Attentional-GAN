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
from librosa.filters import mel as librosa_mel_fn
from src.data.stft import STFT
from src.data.audio_processing import dynamic_range_compression, dynamic_range_decompression
import glob, math
from scipy import signal
import librosa

log1e5 = math.log(1e-5)

class MultiDataset(Dataset):
    def __init__(self, lrw, mode, max_v_timesteps=155, augmentations=False, num_mel_bins=80, wav=False):
        self.max_v_timesteps = max_v_timesteps
        self.augmentations = augmentations if mode == 'train' else False
        self.num_mel_bins = num_mel_bins
        self.skip_long_samples = True
        self.wav = wav
        self.file_paths, self.word_list = self.build_file_list(lrw)
        self.word2int = {word: index for index, word in self.word_list.items()}
        self.stft = TacotronSTFT(filter_length=640, hop_length=160, win_length=640, n_mel_channels=80, sampling_rate=16000, mel_fmin=55., mel_fmax=7600.)

    def build_file_list(self, lrw):
        word = {}
        # data_dir: spec_mel (or wav) / class / test (train, val) / class_#.npz(or .wav)
        if self.wav:
            files = sorted(glob.glob(os.path.join(lrw, '*', '*', '*.wav')))
        else:
            files = sorted(glob.glob(os.path.join(lrw, '*', '*', '*.npz')))

        with open('./data/class.txt', 'r') as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            word[i] = l.strip().upper()

        return files, word

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        content = os.path.split(file_path)[-1].split('_')[0].upper()
        target = self.word2int[content]
        
        if self.wav:
            aud, sr = torchaudio.load(file_path)
            if round(sr) != 16000:
                aud = torch.tensor(librosa.resample(aud.squeeze(0).numpy(), sr, 16000)).unsqueeze(0)

            aud = aud / torch.abs(aud).max() * 0.9
            aud = torch.FloatTensor(self.preemphasize(aud.squeeze(0))).unsqueeze(0)
            aud = torch.clamp(aud, min=-1, max=1)

            spec = self.stft.mel_spectrogram(aud)

        else:
            data = np.load(file_path)
            spec = data['mel']
            data.close()

            spec = torch.FloatTensor(self.denormalize(spec))

        spec = spec[:, :, :self.max_v_timesteps * 4]
        num_a_frames = spec.size(2)
        spec = nn.ConstantPad2d((0, self.max_v_timesteps * 4 - num_a_frames, 0, 0), 0.0)(spec)

        assert spec.size(2) == 116
        return spec, target

    def preemphasize(self, aud):
        aud = signal.lfilter([1, -0.97], [1], aud)
        return aud

    def denormalize(self, melspec):
        melspec = ((melspec + 1) * (-log1e5 / 2)) + log1e5
        return melspec

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
        return mel_output