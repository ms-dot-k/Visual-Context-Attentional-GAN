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
from pesq import pesq
from pystoi import stoi
from matplotlib import pyplot as plt
import copy
import librosa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default="Data_dir")
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/GRID')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=88)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--subject", type=str, default='overlap', help=['overlap', 'unseen', 's1', 's2', 's4', 's29', 'four'])
    parser.add_argument("--eval_step", type=int, default=720)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--augmentations", default=True)

    parser.add_argument("--window_size", type=int, default=40)
    parser.add_argument("--max_timesteps", type=int, default=75)
    parser.add_argument("--temp", type=float, default=1.0)

    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--gpu", type=str, default='0,1,2,3')
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

    train_data = MultiDataset(
        grid=args.grid,
        subject=args.subject,
        mode='train',
        window_size=args.window_size,
        max_v_timesteps=args.max_timesteps,
        augmentations=args.augmentations,
    )

    v_front = Visual_front(in_channels=1)
    gen = Decoder()
    post = Postnet()
    dis1 = Discriminator(phase='1')
    dis2 = Discriminator(phase='2')
    dis3 = Discriminator(phase='3')
    s_dis = sync_Discriminator(temp=args.temp)

    g_params = [{'params': v_front.parameters()}, {'params': gen.parameters()}, {'params': post.parameters()}]
    d_params = [{'params': dis1.parameters()}, {'params': dis2.parameters()}, {'params': dis3.parameters()},
                {'params': s_dis.parameters()}]

    g_optimizer = optim.Adam(g_params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    d_optimizer = optim.Adam(d_params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    g_scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, [500, 800], gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(d_optimizer, [500, 800], gamma=0.1)
    for _ in range(args.start_epoch):
        g_scheduler.step()
        d_scheduler.step()

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())

        v_front.load_state_dict(checkpoint['v_front_state_dict'])
        gen.load_state_dict(checkpoint['gen_state_dict'])
        post.load_state_dict(checkpoint['post_state_dict'])
        dis1.load_state_dict(checkpoint['dis1_state_dict'])
        dis2.load_state_dict(checkpoint['dis2_state_dict'])
        dis3.load_state_dict(checkpoint['dis3_state_dict'])
        s_dis.load_state_dict(checkpoint['s_dis_state_dict'])
        del checkpoint

    v_front.cuda()
    gen.cuda()
    post.cuda()
    dis1.cuda()
    dis2.cuda()
    dis3.cuda()
    s_dis.cuda()

    if args.dataparallel:
        v_front = DP(v_front)
        gen = DP(gen)
        post = DP(post)
        dis1 = DP(dis1)
        dis2 = DP(dis2)
        dis3 = DP(dis3)
        s_dis = DP(s_dis)

    _ = validate(v_front, gen, post, fast_validate=True)
    train(v_front, gen, post, dis1, dis2, dis3, s_dis, train_data, args.epochs, optimizer=[g_optimizer, d_optimizer], scheduler=[g_scheduler, d_scheduler], args=args)

def train(v_front, gen, post, dis1, dis2, dis3, s_dis, train_data, epochs, optimizer, scheduler, args):
    best_val_stoi = 0
    writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])

    v_front.train()
    gen.train()
    post.train()
    dis1.train()
    dis2.train()
    dis3.train()
    s_dis.train()

    g_optimizer, d_optimzier = optimizer
    g_scheduler, d_scheduler = scheduler

    dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    stft = copy.deepcopy(train_data.stft).cuda()

    criterion = nn.L1Loss().cuda()

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = 0
    for epoch in range(args.start_epoch, epochs):
        loss_list = []
        print(f"Epoch [{epoch}/{epochs}]")
        prev_time = time.time()
        for i, batch in enumerate(dataloader):
            step += 1
            if i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %f ********" % (
                    epoch, epochs, (i + 1) * batch_size, samples, iter_time, g_optimizer.param_groups[0]['lr']))
            mel, spec, vid, vid_len, wav_tr, mel_len = batch

            v_front.zero_grad(), gen.zero_grad(), post.zero_grad()

            mel1 = F.interpolate(mel, scale_factor=0.25, mode='bilinear')
            mel2 = F.interpolate(mel, scale_factor=0.5, mode='bilinear')

            phon, sent = v_front(vid.cuda())  # B,S,512, B,512
            g1, g2, g3 = gen(sent, phon, vid_len)  # B,1,20,76, B,1,40,152, B,1,80,304

            mel.requires_grad = True
            mel1.requires_grad = True
            mel2.requires_grad = True

            ################################### DIS ########################################

            ur_lo1, cr_lo1 = dis1(mel1.cuda(), sent.detach(), phon.size(1))
            ur_lo2, cr_lo2 = dis2(mel2.cuda(), sent.detach(), phon.size(1))
            ur_lo3, cr_lo3 = dis3(mel.cuda(), sent.detach(), phon.size(1))

            sync_loss = s_dis(phon, mel.cuda()).mean()  # B*S, 1

            grad_r1 = grad(outputs=ur_lo1.sum(), inputs=mel1, create_graph=True)[0]
            grad_r2 = grad(outputs=ur_lo2.sum(), inputs=mel2, create_graph=True)[0]
            grad_r3 = grad(outputs=ur_lo3.sum(), inputs=mel, create_graph=True)[0]

            grad_p1 = (grad_r1.view(grad_r1.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_p2 = (grad_r2.view(grad_r2.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_p3 = (grad_r3.view(grad_r3.size(0), -1).norm(2, dim=1) ** 2).mean()

            uf_lo1, cf_lo1 = dis1(g1.detach(), sent.detach(), phon.size(1))
            uf_lo2, cf_lo2 = dis2(g2.detach(), sent.detach(), phon.size(1))
            uf_lo3, cf_lo3 = dis3(g3.detach(), sent.detach(), phon.size(1))

            real_loss = 1 / 3 * (gan_loss(ur_lo1, True) + gan_loss(ur_lo2, True) + gan_loss(ur_lo3, True)
                                 + gan_loss(cr_lo1, True) + gan_loss(cr_lo2, True) + gan_loss(cr_lo3, True)) \
                        + 1 / 3 * (grad_p1 + grad_p2 + grad_p3)

            fake_loss = 1 / 3 * (gan_loss(uf_lo1, False) + gan_loss(uf_lo2, False) + gan_loss(uf_lo3, False)
                                 + gan_loss(cf_lo1, False) + gan_loss(cf_lo2, False) + gan_loss(cf_lo3, False))

            dis_loss = real_loss + fake_loss + sync_loss

            d_optimzier.zero_grad()
            dis_loss.backward(retain_graph=True)    #accumulate v_front grad
            d_optimzier.step()

            ################################### GEN ########################################

            gs = post(g3)

            ug_lo1, cg_lo1 = dis1(g1, sent.detach(), phon.size(1))
            ug_lo2, cg_lo2 = dis2(g2, sent.detach(), phon.size(1))
            ug_lo3, cg_lo3 = dis3(g3, sent.detach(), phon.size(1))

            g_sync_loss = s_dis(phon.detach(), g3, True).mean()  # B*S, 1

            g_loss = 1 / 3 * (gan_loss(ug_lo1, True) + gan_loss(ug_lo2, True) + gan_loss(ug_lo3, True)
                              + gan_loss(cg_lo1, True) + gan_loss(cg_lo2, True) + gan_loss(cg_lo3, True)) \
                     + g_sync_loss
            recon_loss = (criterion(train_data.denormalize(g1), train_data.denormalize(mel1.cuda())) +
                          criterion(train_data.denormalize(g2), train_data.denormalize(mel2.cuda())) +
                          criterion(train_data.denormalize(g3), train_data.denormalize(mel.cuda()))) / 3. \
                         + criterion(gs, spec.cuda())

            gen_loss = g_loss + recon_loss * 50.0

            loss_list.append(gen_loss.cpu().item())

            dis1.zero_grad(), dis2.zero_grad(), dis3.zero_grad(), s_dis.zero_grad(), gen.zero_grad(), post.zero_grad()
            gen_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                wav_pred = train_data.inverse_mel(g3.detach()[0], stft)  # 1, 80, T
                wav_spec = train_data.inverse_spec(gs.detach()[0], stft)
                wav_gt = train_data.inverse_mel(mel.cuda().detach()[0], stft)
            else:
                wav_pred = 0
                wav_spec = 0
                wav_gt = 0

            if writer is not None:
                writer.add_scalar('train/gen_loss', g_loss.cpu().item(), step)
                writer.add_scalar('train/recon_loss', recon_loss.cpu().item(), step)
                writer.add_scalar('train/dis_loss', dis_loss.cpu().item(), step)
                writer.add_scalar('train/g_sync_loss', g_sync_loss.cpu(), step)
                writer.add_scalar('train/d_sync_loss', sync_loss.cpu(), step)
                writer.add_scalar('lr/learning_rate', g_optimizer.param_groups[0]['lr'], step)
                if i % 100 == 0:
                    print(f'######## Step(Epoch): {step}({epoch}), Generator Loss: {gen_loss.cpu().item()} #########')
                    writer.add_image('train_mel/g1',
                                     train_data.plot_spectrogram_to_numpy(g1.cpu().detach().numpy()[0]), step)
                    writer.add_image('train_mel/g2',
                                     train_data.plot_spectrogram_to_numpy(g2.cpu().detach().numpy()[0]), step)
                    writer.add_image('train_mel/g3',
                                     train_data.plot_spectrogram_to_numpy(g3.cpu().detach().numpy()[0]), step)
                    writer.add_image('train_mel/gt', train_data.plot_spectrogram_to_numpy(mel.detach().numpy()[0]),
                                     step)
                    writer.add_image('train_spec/gen',
                                     train_data.plot_spectrogram_to_numpy(gs.cpu().detach().numpy()[0]), step)
                    writer.add_image('train_spec/gen_log',
                                     train_data.plot_spectrogram_to_numpy(torch.log(gs).cpu().detach().numpy()[0]),
                                     step)
                    writer.add_image('train_spec/gt',
                                     train_data.plot_spectrogram_to_numpy(spec.detach().numpy()[0]), step)
                    writer.add_image('train_spec/gt_log',
                                     train_data.plot_spectrogram_to_numpy(torch.log(spec).detach().numpy()[0]),
                                     step)
                    writer.add_audio('train_aud/pred_mel', wav_pred, global_step=step, sample_rate=16000)
                    writer.add_audio('train_aud/pred_spec', wav_spec, global_step=step, sample_rate=16000)
                    writer.add_audio('train_aud/gt_mel', wav_gt, global_step=step, sample_rate=16000)
                    writer.add_audio('train_aud/gt_wav', wav_tr[0].numpy(), global_step=step, sample_rate=16000)

            if step % args.eval_step == 0:
                logs = validate(v_front, gen, post, epoch=epoch, writer=writer, fast_validate=True)

                print('VAL_stoi: ', logs[1])
                print('Saving checkpoint: %d' % epoch)
                if args.dataparallel:
                    v_state_dict = v_front.module.state_dict()
                    gen_state_dict = gen.module.state_dict()
                    post_state_dict = post.module.state_dict()
                    dis1_state_dict = dis1.module.state_dict()
                    dis2_state_dict = dis2.module.state_dict()
                    dis3_state_dict = dis3.module.state_dict()
                    s_dis_state_dict = s_dis.module.state_dict()
                else:
                    v_state_dict = v_front.state_dict()
                    gen_state_dict = gen.state_dict()
                    post_state_dict = post.state_dict()
                    dis1_state_dict = dis1.state_dict()
                    dis2_state_dict = dis2.state_dict()
                    dis3_state_dict = dis3.state_dict()
                    s_dis_state_dict = s_dis.state_dict()
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                torch.save({'v_front_state_dict': v_state_dict, 'gen_state_dict': gen_state_dict,
                            'post_state_dict': post_state_dict,
                            'dis1_state_dict': dis1_state_dict, 'dis2_state_dict': dis2_state_dict,
                            'dis3_state_dict': dis3_state_dict, 's_dis_state_dict': s_dis_state_dict},
                           os.path.join(args.checkpoint_dir,
                                        'Epoch_%04d_stoi_%.3f_estoi_%.3f_pesq_%.3f.ckpt' % (
                                        epoch, logs[1], logs[2], logs[3])))

                if logs[1] > best_val_stoi:
                    best_val_stoi = logs[1]
                    bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                    for prev in bests:
                        os.remove(prev)
                    torch.save({'v_front_state_dict': v_state_dict, 'gen_state_dict': gen_state_dict,
                                'post_state_dict': post_state_dict,
                                'dis1_state_dict': dis1_state_dict, 'dis2_state_dict': dis2_state_dict,
                                'dis3_state_dict': dis3_state_dict, 's_dis_state_dict': s_dis_state_dict},
                               os.path.join(args.checkpoint_dir,
                                            'Best_%04d_stoi_%.3f_estoi_%.3f_pesq_%.3f.ckpt' % (
                                            epoch, logs[1], logs[2], logs[3])))

        if scheduler is not None:
            g_scheduler.step()
            d_scheduler.step()

    print('Finishing training')


def validate(v_front, gen, post, fast_validate=False, epoch=0, writer=None):
    with torch.no_grad():
        v_front.eval()
        gen.eval()
        post.eval()

        val_data = MultiDataset(
            grid=args.grid,
            mode='val',
            subject=args.subject,
            window_size=args.window_size,
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
            fast_validate=fast_validate
        )

        dataloader = DataLoader(
            val_data,
            shuffle=True if fast_validate else False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        stft = copy.deepcopy(val_data.stft).cuda()
        criterion = nn.L1Loss().cuda()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(5 * batch_size, int(len(dataloader.dataset)))
            max_batches = 5
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        val_loss = []
        stoi_list = []
        estoi_list = []
        pesq_list = []
        stoi_spec_list = []
        estoi_spec_list = []
        pesq_spec_list = []

        required_iter = (samples // batch_size)

        description = 'Check validation step' if fast_validate else 'Validation'
        print(description)
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            mel, spec, vid, vid_len, wav_tr, mel_len = batch
            phon, sent = v_front(vid.cuda())  # S,B,512
            g1, g2, g3 = gen(sent, phon, vid_len)
            gs = post(g3)

            loss = criterion(g3, mel.cuda()).cpu().item()
            val_loss.append(loss)

            wav_pred = val_data.inverse_mel(g3[:, :, :, :mel_len[0]].detach(), stft)  # B, 1, 80, T
            wav_spec = val_data.inverse_spec(gs[:, :, :, :mel_len[0]].detach(), stft)
            wav_gt = val_data.inverse_mel(mel[:, :, :, :mel_len[0]].cuda().detach(), stft)
            for _ in range(g3.size(0)):
                stoi_list.append(stoi(wav_tr[_][:len(wav_pred[_])].numpy(), wav_pred[_], 16000, extended=False))
                stoi_spec_list.append(stoi(wav_tr[_][:len(wav_spec[_])], wav_spec[_], 16000, extended=False))
                estoi_list.append(stoi(wav_tr[_][:len(wav_pred[_])].numpy(), wav_pred[_], 16000, extended=True))
                estoi_spec_list.append(stoi(wav_tr[_][:len(wav_spec[_])], wav_spec[_], 16000, extended=True))
                try:
                    pesq_list.append(pesq(8000, librosa.resample(wav_tr[_][:len(wav_pred[_])].numpy(), 16000, 8000), librosa.resample(wav_pred[_], 16000, 8000), 'nb'))
                except:
                    pass
                try:
                    pesq_spec_list.append(pesq(8000, librosa.resample(wav_tr[_][:len(wav_spec[_])].numpy(), 16000, 8000), librosa.resample(wav_spec[_], 16000, 8000), 'nb'))
                except:
                    continue

            if i in [int(required_iter // 3), int(2 * (required_iter // 3)), int(3 * (required_iter // 3))]:
                if writer is not None:
                    writer.add_image('val_mel_%d/g1' % i,
                                     val_data.plot_spectrogram_to_numpy(g1.cpu().detach().numpy()[0]), epoch)
                    writer.add_image('val_mel_%d/g2' % i,
                                     val_data.plot_spectrogram_to_numpy(g2.cpu().detach().numpy()[0]), epoch)
                    writer.add_image('val_mel_%d/g3' % i,
                                     val_data.plot_spectrogram_to_numpy(g3.cpu().detach().numpy()[0]), epoch)
                    writer.add_image('val_mel_%d/gt' % i, val_data.plot_spectrogram_to_numpy(mel.detach().numpy()[0]),
                                     epoch)
                    writer.add_image('val_spec_%d/gen' % i,
                                     val_data.plot_spectrogram_to_numpy(gs.cpu().detach().numpy()[0]), epoch)
                    writer.add_image('val_spec_%d/gen_log' % i,
                                     val_data.plot_spectrogram_to_numpy(torch.log(gs).cpu().detach().numpy()[0]), epoch)
                    writer.add_image('val_spec_%d/gt' % i, val_data.plot_spectrogram_to_numpy(spec.detach().numpy()[0]),
                                     epoch)
                    writer.add_image('val_spec_%d/gt_log' % i,
                                     val_data.plot_spectrogram_to_numpy(torch.log(spec).detach().numpy()[0]), epoch)

                    writer.add_audio('val_aud_%d/pred' % i, wav_pred[0], global_step=epoch, sample_rate=16000)
                    writer.add_audio('val_aud_%d/mel' % i, wav_gt[0], global_step=epoch, sample_rate=16000)
                    writer.add_audio('val_aud_%d/spec' % i, wav_spec[0], global_step=epoch, sample_rate=16000)
                    writer.add_audio('val_aud_%d/gt' % i, wav_tr[0], global_step=epoch, sample_rate=16000)
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set(xlim=[0, len(wav_pred[0])], ylim=[-1, 1])
                    ax.plot(wav_pred[0])
                    writer.add_figure('val_wav_%d/pred_mel' % i, fig, epoch)
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set(xlim=[0, len(wav_gt[0])], ylim=[-1, 1])
                    ax.plot(wav_gt[0])
                    writer.add_figure('val_wav_%d/mel' % i, fig, epoch)
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set(xlim=[0, len(wav_spec[0])], ylim=[-1, 1])
                    ax.plot(wav_spec[0])
                    writer.add_figure('val_wav_%d/pred_spec' % i, fig, epoch)
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set(xlim=[0, len(wav_tr[0])], ylim=[-1, 1])
                    ax.plot(wav_tr[0])
                    writer.add_figure('val_wav_%d/gt' % i, fig, epoch)

            if i >= max_batches:
                break

        if writer is not None:
            writer.add_scalar('val/recon_loss', np.mean(np.array(val_loss)), epoch)
            writer.add_scalar('val/mel_stoi', np.mean(np.array(stoi_list)), epoch)
            writer.add_scalar('val/mel_estoi', np.mean(np.array(estoi_list)), epoch)
            writer.add_scalar('val/mel_pesq', np.mean(np.array(pesq_list)), epoch)
            writer.add_scalar('val/postnet_stoi', np.mean(np.array(stoi_spec_list)), epoch)
            writer.add_scalar('val/postnet_estoi', np.mean(np.array(estoi_spec_list)), epoch)
            writer.add_scalar('val/postnet_pesq', np.mean(np.array(pesq_spec_list)), epoch)

        v_front.train()
        gen.train()
        post.train()
        print('val_stoi:', np.mean(np.array(stoi_spec_list)))
        print('val_estoi:', np.mean(np.array(estoi_spec_list)))
        print('val_pesq:', np.mean(np.array(pesq_spec_list)))
        return np.mean(np.array(val_loss)), np.mean(np.array(stoi_spec_list)), np.mean(np.array(estoi_spec_list)), np.mean(np.array(pesq_spec_list))


if __name__ == "__main__":
    args = parse_args()
    train_net(args)

