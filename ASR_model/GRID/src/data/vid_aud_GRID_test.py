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
import random
from librosa.filters import mel as librosa_mel_fn
from src.data.audio_processing import dynamic_range_compression, dynamic_range_decompression
from src.data.stft import STFT
import math, glob
from scipy import signal

log1e5 = math.log(1e-5)

letters = ['_', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']


class MultiDataset(Dataset):
    def __init__(self, grid, mode, gtpath, subject, max_v_timesteps=155, max_text_len=150, augmentations=False, num_mel_bins=80, wav=False):
        self.wav = wav
        self.gtpath = gtpath
        self.max_v_timesteps = max_v_timesteps
        self.max_text_len = max_text_len
        self.augmentations = augmentations if mode == 'train' else False
        self.file_paths = self.build_file_list(grid, subject)
        self.int2char = dict(enumerate(letters))
        self.char2int = {char: index for index, char in self.int2char.items()}
        self.stft = TacotronSTFT(filter_length=640, hop_length=160, win_length=640, n_mel_channels=num_mel_bins, sampling_rate=16000, mel_fmin=55., mel_fmax=7500.)

    def build_file_list(self, grid, subject):
        check_list = []
        if subject == 'overlap':
                with open('./../../data/overlap_val.txt', 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    file = l.strip().replace('mpg_6000/', '') + '.mp4'
                    check_list.append(os.path.join(grid, file))
        elif subject == 'unseen':
                with open('./../../data/unseen_splits.txt', 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    if 'test' in l.strip():
                        _, sub, fname = l.strip().split('/')
                        file = f'{sub}/video/{fname}.mp4'
                        if os.path.exists(os.path.join(grid, file)):
                            check_list.append(os.path.join(grid, file))
        else:
            with open('./../../data/test_4.txt', 'r') as f:
                lines = f.readlines()
            for l in lines:
                file = l.strip()
                if subject == 'four':
                    check_list.append(os.path.join(grid, file))
                elif file.split('/')[0] == subject:
                    check_list.append(os.path.join(grid, file))

        if self.wav:
            file_list = sorted(glob.glob(os.path.join(grid, '*', '*.wav')))
        else:
            file_list = sorted(glob.glob(os.path.join(grid, '*', '*.npz')))

        assert len(check_list) == len(file_list), 'The data for testing is not full'
        return file_list

    def __len__(self):
        return len(self.file_paths)

    def build_content(self, content):
        words = []
        with open(content, 'r') as f:
            lines = f.readlines()
            for l in lines:
                word = l.strip().split()[2]
                if not word in ['SIL', 'SP', 'sil', 'sp']:
                    words.append(word)
        return words

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        t, f_name = os.path.split(file_path)
        _, sub = os.path.split(t)

        words = self.build_content(os.path.join(self.gtpath, sub.split('_')[0], 'align', f_name.split('.')[0] + '.align'))
        content = ' '.join(words).upper()

        if self.wav:
            aud, sr = torchaudio.load(file_path)

            if round(sr) != 16000:
                aud = torch.tensor(librosa.resample(aud.squeeze(0).numpy(), sr, 16000)).unsqueeze(0)

            aud = aud / torch.abs(aud).max() * 0.9
            aud = torch.FloatTensor(self.preemphasize(aud.squeeze(0))).unsqueeze(0)
            aud = torch.clamp(aud, min=-1, max=1)

            spec = self.stft.mel_spectrogram(aud)
            num_a_frames = spec.size(2)

        else:
            data = np.load(file_path)
            mel = data['mel']
            data.close()

            spec = torch.FloatTensor(self.denormalize(mel))

            num_a_frames = spec.size(2)

        target, txt_len = self.encode(content)

        spec = nn.ConstantPad2d((0, self.max_v_timesteps * 4 - num_a_frames, 0, 0), 0.0)(spec)

        return spec, target, num_a_frames, txt_len

    def encode(self, content):
        encoded = [self.char2int[c] for c in content]
        if len(encoded) > self.max_text_len:
            print(f"Max output length too short. Required {len(encoded)}")
            encoded = encoded[:self.max_text_len]
        num_txt = len(encoded)
        encoded += [self.char2int['_'] for _ in range(self.max_text_len - len(encoded))]
        return torch.Tensor(encoded), num_txt

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
