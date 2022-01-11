import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.audio_front import Audio_front
from src.models.classifier import Backend
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.vid_aud_lrw_test import MultiDataset
from torch.nn import DataParallel as DP
import torch.nn.parallel
import math
from matplotlib import pyplot as plt
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="TEST_DIR", help='./../../test/spec_mel')
    parser.add_argument('--wav', default=False, action='store_true')
    parser.add_argument("--checkpoint_dir", type=str, default='./data')
    parser.add_argument("--checkpoint", type=str, default='./data/LRW_acc_0.98464.ckpt')
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--max_timesteps", type=int, default=29)

    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--gpu", type=str, default='0')
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

    a_front = Audio_front(in_channels=1)
    a_back = Backend()

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        a_front.load_state_dict(checkpoint['a_front_state_dict'])
        a_back.load_state_dict(checkpoint['a_back_state_dict'])
        del checkpoint

    a_front.cuda()
    a_back.cuda()

    if args.dataparallel:
        a_front = DP(a_front)
        a_back = DP(a_back)

    _ = validate(a_front, a_back, epoch=0, writer=None)

def validate(a_front, a_back, fast_validate=False, epoch=0, writer=None):
    with torch.no_grad():
        a_front.eval()
        a_back.eval()

        val_data = MultiDataset(
            lrw=args.data,
            mode='test',
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
            wav=args.wav
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        criterion = nn.CrossEntropyLoss().cuda()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        val_loss = []
        tot_cor, tot_v_cor, tot_a_cor, tot_num = 0, 0, 0, 0

        description = 'Check validation step' if fast_validate else 'Validation'
        print(description)
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            a_in, target = batch

            a_feat = a_front(a_in.cuda())  # S,B,51
            a_pred = a_back(a_feat)

            loss = criterion(a_pred, target.long().cuda()).cpu().item()
            prediction = torch.argmax(a_pred.cpu(), dim=1).numpy()
            tot_cor += np.sum(prediction == target.long().numpy())
            tot_num += len(prediction)

            batch_size = a_pred.size(0)
            val_loss.append(loss)

            if i >= max_batches:
                break

        if writer is not None:
            writer.add_scalar('Val/loss', np.mean(np.array(val_loss)), epoch)
            writer.add_scalar('Val/acc', tot_cor / tot_num, epoch)

        a_front.train()
        a_back.train()
        print('test_ACC:', tot_cor / tot_num, 'WER:', 1. - tot_cor / tot_num)
        if fast_validate:
            return {}
        else:
            return np.mean(np.array(val_loss)), tot_cor / tot_num

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

