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
from src.data.vid_aud_GRID_test import MultiDataset
from torch.nn import DataParallel as DP
import torch.nn.parallel
import math
import editdistance
import re
from matplotlib import pyplot as plt
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="TEST_DIR", help='./../../test/spec_mel')
    parser.add_argument('--wav', default=False, action='store_true', help='Is waveform or Mel(.npz) form')
    parser.add_argument('--gtpath', default="GT_path", help='GT transcription path')
    parser.add_argument('--model', default="GRID_CTC")
    parser.add_argument("--checkpoint_dir", type=str, default='./data')
    parser.add_argument("--checkpoint", type=str, default='./data/GRID_4_wer_0.00833_cer_0.00252.ckpt')
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--subject", default='overlap', help=['overlap', 'unseen', 's1', 's2', 's4', 's29', 'four'])

    parser.add_argument("--max_timesteps", type=int, default=75)
    parser.add_argument("--max_text_len", type=int, default=75)

    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    a_front = Audio_front()
    a_back = Backend()

    if args.subject == 'unseen':
        args.checkpoint = './data/GRID_unseen_wer_0.01676_cer_0.00896.ckpt'
    elif args.subject == 'overlap':
        args.checkpoint = './data/GRID_33_wer_0.00368_cer_0.00120.ckpt'
    else:
        args.checkpoint = './data/GRID_4_wer_0.00833_cer_0.00252.ckpt'

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
        a_front.load_state_dict(checkpoint['a_front_state_dict'])
        a_back.load_state_dict(checkpoint['a_back_state_dict'])
        del checkpoint

    a_front.cuda()
    a_back.cuda()

    if args.dataparallel:
        a_front = DP(a_front)
        a_back = DP(a_back)

    wer, cer = validate(a_front, a_back)

def validate(a_front, a_back, fast_validate=False):
    with torch.no_grad():
        a_front.eval()
        a_back.eval()

        val_data = MultiDataset(
            grid=args.data,
            mode='test',
            gtpath=args.gtpath,
            subject=args.subject,
            max_v_timesteps=args.max_timesteps,
            max_text_len=args.max_text_len,
            wav=args.wav,
            augmentations=False
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        wer_sum, cer_sum, tot_num = 0, 0, 0

        description = 'Check validation step' if fast_validate else 'Validation'
        print(description)
        for i, batch in enumerate(dataloader):
            if i % 50 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            a_in, target, aud_len, txt_len = batch

            a_feat = a_front(a_in.cuda())  # B,S,512
            pred = a_back(a_feat)

            cer, wer, sentences = greedy_decode(val_data, F.softmax(pred, dim=2).cpu(), target)

            B, S, _ = a_feat.size()

            tot_num += B
            wer_sum += B * wer
            cer_sum += B * cer
            batch_size = B

            if i % 50 == 0:
                for j in range(2):
                    print('label: ', sentences[j][0])
                    print('prediction: ', sentences[j][1])

            if i >= max_batches:
                break

        a_front.train()
        a_back.train()

        wer = wer_sum / tot_num
        cer = cer_sum / tot_num

        print('test_cer:', cer)
        print('test_wer:', wer)

        if fast_validate:
            return {}
        else:
            return wer, cer

def decode(dataset, label_tokens, pred_tokens):
    label, output = '', ''
    for index in range(len(label_tokens)):
        label += dataset.int2char[int(label_tokens[index])]
    for index in range(len(pred_tokens)):
        output += dataset.int2char[int(pred_tokens[index])]

    output = re.sub(' +', ' ', output)
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)  # remove characters that are repeated more than 2 times
    output = pattern.sub(r"\1", output)

    label = label.replace('_', '')
    output = output.replace('_', '')

    output_words, label_words = output.split(" "), label.split(" ")

    cer = editdistance.eval(output, label) / len(label)
    wer = editdistance.eval(output_words, label_words) / len(label_words)

    return label, output, cer, wer

def greedy_decode(dataset, results, target):
    _, results = results.topk(1, dim=2)
    results = results.squeeze(dim=2)
    cer_sum, wer_sum = 0, 0
    batch_size = results.size(0)
    sentences = []
    for batch in range(batch_size):
        label, output, cer, wer = decode(dataset, target[batch], results[batch])
        sentences.append([label, output])
        cer_sum += cer
        wer_sum += wer

    return cer_sum / batch_size, wer_sum / batch_size, sentences

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

