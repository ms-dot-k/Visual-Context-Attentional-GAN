import os
import shutil
import glob
import cv2
import torchvision
import torch
from skimage import transform
import numpy as np
from scipy import signal
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
import librosa
import soundfile as sf
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_dir', type=str, default="Data dir of images and audio of GRID")
    parser.add_argument('--Landmark', type=str, default="Data dir of GRID Landmark")
    parser.add_argument('--FPS', type=int, default=25, help="25 for GRID")
    parser.add_argument('--reference', type=str, default='./Ref_face.txt')
    parser.add_argument("--Output_dir", type=str, default='Output dir Ex) ./GRID_processed')
    args = parser.parse_args()
    return args

args = parse_args()
eps = 1e-8

class Crop(object):
    def __init__(self, crop):
        self.crop = crop

    def __call__(self, img):
        return img.crop(self.crop)

######################################################
f = open(args.reference, 'r')
lm = f.readlines()[0]
f.close()
lm = lm.split(':')[-1].split('|')[6]
lms = lm.split(',')
temp_lm = []

for lm in lms:
    x, y = lm.split()
    temp_lm.append([x, y])
temp_lm = np.array(temp_lm, dtype=float)
refer_lm = temp_lm
######################################################


class Preprocessing(Dataset):
    def __init__(self,):
        self.file_paths = self.build_file_list()

    def build_file_list(self):
        file_list = []
        landmarks = sorted(glob.glob(os.path.join(args.Landmark, '*', '*', '*.txt')))
        for lm in landmarks:
            if not os.path.exists(lm.replace(args.Landmark, args.Output_dir)[:-4] + '.mp4'):
                file_list.append(lm)
        return file_list

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        ims = sorted(glob.glob(os.path.join(file_path.replace(args.Landmark, args.Data_dir)[:-4], '*.png')))
        frames = []

        for im in ims:
            frames += [cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)]
        v = np.stack(frames, 0)

        t, f_name = os.path.split(file_path)
        t, m_name = os.path.split(t)
        _, s_name = os.path.split(t)
        save_path = os.path.join(args.Output_dir, s_name, m_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as lf:
                lms = lf.readlines()[0]
        except:
            with open(file_path, 'r', encoding='cp949') as lf:
                lms = lf.readlines()[0]
        lms = lms.split(':')[-1].split('|')
        assert v.shape[0] == len(lms), 'the video frame length differs to the landmark frames'

        aligned_video = []
        for i, frame in enumerate(v):
            lm = lms[i].split(',')
            temp_lm = []
            for l in lm:
                x, y = l.split()
                temp_lm.append([x, y])
            temp_lm = np.array(temp_lm, dtype=float)  # 98,2

            source_lm = temp_lm

            tform.estimate(source_lm, refer_lm)
            mat = tform.params[0:2, :]
            aligned_im = cv2.warpAffine(frame, mat, (np.shape(frame)[0], np.shape(frame)[1]))
            aligned_video += [aligned_im[:256, :256, :]]

        aligned_video = np.array(aligned_video)

        #### audio preprocessing ####
        aud, _ = librosa.load(os.path.join(file_path.replace(args.Landmark, args.Data_dir).replace('video', 'audio')[:-4] + '.wav'), 16000)
        fc = 55.  # Cut-off frequency of the filter
        w = fc / (16000 / 2)  # Normalize the frequency
        b, a = signal.butter(7, w, 'high')
        aud = signal.filtfilt(b, a, aud)

        return torch.tensor(aligned_video), save_path, f_name, torch.tensor(aud.copy())

Data = Preprocessing()
Data_loader = DataLoader(Data, shuffle=False, batch_size=1, num_workers=3)
tform = transform.SimilarityTransform()

for kk, data in enumerate(Data_loader):
    cropped_video, save_path, f_name, aud = data
    aud = aud[0]
    cropped_video = cropped_video[0]
    save_path = save_path[0]
    f_name = f_name[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path.replace('video', 'audio')):
        os.makedirs(save_path.replace('video', 'audio'))
    torchvision.io.write_video(os.path.join(save_path, f_name[:-4] + '.mp4'), video_array=cropped_video, fps=args.FPS)
    sf.write(os.path.join(save_path.replace('video', 'audio'), f_name[:-4] + ".flac"), aud.numpy(), samplerate=16000)
    print('##########', kk + 1, ' / ', len(Data_loader), '##########')
