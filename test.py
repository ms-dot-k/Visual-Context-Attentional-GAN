import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.visual_front import Visual_front
from src.models.generator import Decoder, Discriminator, gan_loss, sync_Discriminator, Postnet
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.vid_aud_grid import MultiDataset
from torch.nn import DataParallel as DP
import torch.nn.parallel
import time
import glob
from torch.autograd import grad
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from matplotlib import pyplot as plt
import copy, librosa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default="Data_dir")
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/GRID')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--subject", type=str, default='overlap')

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--augmentations", default=True)

    parser.add_argument("--window_size", type=int, default=40)
    parser.add_argument("--max_timesteps", type=int, default=75)
    parser.add_argument("--temp", type=float, default=1.0)

    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--gpu", type=str, default='0,1')

    parser.add_argument("--save_mel", default=False, action='store_true')
    parser.add_argument("--save_wav", default=False, action='store_true')

    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    v_front = Visual_front(in_channels=1)
    gen = Decoder()
    post = Postnet()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())

    v_front.load_state_dict(checkpoint['v_front_state_dict'])
    gen.load_state_dict(checkpoint['gen_state_dict'])
    post.load_state_dict(checkpoint['post_state_dict'])
    del checkpoint

    v_front.cuda()
    gen.cuda()
    post.cuda()

    if args.dataparallel:
        v_front = DP(v_front)
        gen = DP(gen)
        post = DP(post)

    _ = test(v_front, gen, post)

def test(v_front, gen, post, fast_validate=False):
    with torch.no_grad():
        v_front.eval()
        gen.eval()
        post.eval()

        val_data = MultiDataset(
            grid=args.grid,
            mode='test',
            subject=args.subject,
            window_size=args.window_size,
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
            fast_validate=fast_validate
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        stft = copy.deepcopy(val_data.stft).cuda()
        stoi_spec_list = []
        estoi_spec_list = []
        pesq_spec_list = []
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        description = 'Check validation step' if fast_validate else 'Validation'
        print(description)
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            mel, spec, vid, vid_len, wav_tr, mel_len, f_name = batch
            vid = vid.cuda()
            phon, sent = v_front(vid)  # S,B,512
            g1, g2, g3 = gen(sent, phon, vid_len)
            g3_temp = g3.clone()

            vid = vid.flip(4)
            phon, sent = v_front(vid)  # S,B,512
            g1, g2, g3 = gen(sent, phon, vid_len)

            g3 = (g3_temp + g3) / 2.
            gs = post(g3)

            wav_spec = val_data.inverse_spec(gs[:, :, :, :mel_len[0]].detach(), stft)

            for b in range(g3.size(0)):
                stoi_spec_list.append(stoi(wav_tr[b][:len(wav_spec[b])], wav_spec[b], 16000, extended=False))
                estoi_spec_list.append(stoi(wav_tr[b][:len(wav_spec[b])], wav_spec[b], 16000, extended=True))
                pesq_spec_list.append(pesq(8000, librosa.resample(wav_tr[b][:len(wav_spec[b])].numpy(), 16000, 8000), librosa.resample(wav_spec[b], 16000, 8000), 'nb'))

                sub_name, _, file_name = f_name[b].split('/')
                if not os.path.exists(f'./test/spec_mel/{sub_name}'):
                    os.makedirs(f'./test/spec_mel/{sub_name}')
                np.savez(f'./test/spec_mel/{sub_name}/{file_name}.npz',
                         mel=g3[b, :, :, :mel_len[b]].detach().cpu().numpy(),
                         spec=gs[b, :, :, :mel_len[b]].detach().cpu().numpy())

                if not os.path.exists(f'./test/wav/{sub_name}'):
                    os.makedirs(f'./test/wav/{sub_name}')
                sf.write(f'./test/wav/{sub_name}/{file_name}.wav', wav_spec[b], 16000, subtype='PCM_16')

            if i >= max_batches:
                break

        print('STOI: ', np.mean(stoi_spec_list))
        print('ESTOI: ', np.mean(estoi_spec_list))
        print('PESQ: ', np.mean(pesq_spec_list))
        with open(f'./test/metric.txt', 'w') as f:
            f.write(f'STOI : {np.mean(stoi_spec_list)}')
            f.write(f'ESTOI : {np.mean(estoi_spec_list)}')
            f.write(f'PESQ : {np.mean(pesq_spec_list)}')

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

