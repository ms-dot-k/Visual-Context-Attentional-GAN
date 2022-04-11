import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.resnet import BasicBlock

class ResBlk1D(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv1d(dim_in, dim_in, 5, 1, 2)
        self.conv2 = nn.Conv1d(dim_in, dim_out, 5, 1, 2)
        if self.normalize:
            self.norm1 = nn.BatchNorm1d(dim_in)
            self.norm2 = nn.BatchNorm1d(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 5, 1, 2)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 5, 1, 2)
        if self.normalize:
            self.norm1 = nn.BatchNorm2d(dim_in)
            self.norm2 = nn.BatchNorm2d(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class GenResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 5, 1, 2)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 5, 1, 2)
        self.norm1 = nn.BatchNorm2d(dim_in)
        self.norm2 = nn.BatchNorm2d(dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._residual(x)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Avgpool(nn.Module):
    def forward(self, input):
        #input:B,C,H,W
        return input.mean([2, 3])

class AVAttention(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.softmax = nn.Softmax(2)
        self.k = nn.Linear(512, out_dim)
        self.v = nn.Linear(512, out_dim)
        self.q = nn.Linear(2560, out_dim)
        self.out_dim = out_dim
        dim = 20 * 64
        self.mel = nn.Linear(out_dim, dim)

    def forward(self, ph, g, len):
        #ph: B,S,512
        #g: B,C,F,T
        B, C, F, T = g.size()
        k = self.k(ph).transpose(1, 2).contiguous()   # B,256,S
        q = self.q(g.view(B, C * F, T).transpose(1, 2).contiguous())  # B,T,256

        att = torch.bmm(q, k) / math.sqrt(self.out_dim)    # B,T,S
        for i in range(att.size(0)):
            att[i, :, len[i]:] = float('-inf')
        att = self.softmax(att)  # B,T,S

        v = self.v(ph)  # B,S,256
        value = torch.bmm(att, v)  # B,T,256
        out = self.mel(value)  # B, T, 20*64
        out = out.view(B, T, F, -1).permute(0, 3, 2, 1)

        return out  #B,C,F,T

class Postnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.postnet = nn.Sequential(
            nn.Conv1d(80, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResBlk1D(128, 256),
            ResBlk1D(256, 256),
            ResBlk1D(256, 256),
            nn.Conv1d(256, 321, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        # x: B,1,80,T
        x = x.squeeze(1)    # B, 80, t
        x = self.postnet(x)     # B, 321, T
        x = x.unsqueeze(1)  # B, 1, 321, T
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.ModuleList()
        self.g1 = nn.ModuleList()
        self.g2 = nn.ModuleList()
        self.g3 = nn.ModuleList()

        self.att1 = AVAttention(256)
        self.attconv1 = nn.Conv2d(128 + 64, 128, 5, 1, 2)
        self.att2 = AVAttention(256)
        self.attconv2 = nn.Conv2d(64 + 32, 64, 5, 1, 2)

        self.to_mel1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 1, 1, 0),
            nn.Tanh()
        )
        self.to_mel2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Tanh()
        )
        self.to_mel3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Tanh()
        )

        # bottleneck blocks
        self.decode.append(GenResBlk(512 + 128, 512))    # 20,T
        self.decode.append(GenResBlk(512, 256))
        self.decode.append(GenResBlk(256, 256))

        # up-sampling blocks
        self.g1.append(GenResBlk(256, 128))     # 20,T
        self.g1.append(GenResBlk(128, 128))
        self.g1.append(GenResBlk(128, 128))

        self.g2.append(GenResBlk(128, 64, upsample=True))  # 40,2T
        self.g2.append(GenResBlk(64, 64))
        self.g2.append(GenResBlk(64, 64))

        self.g3.append(GenResBlk(64, 32, upsample=True))  # 80,4T
        self.g3.append(GenResBlk(32, 32))
        self.g3.append(GenResBlk(32, 32))

    def forward(self, s, x, len):
        # s: B,512,T x: B,T,512
        s = s.transpose(1, 2).contiguous()
        n = torch.randn([x.size(0), 128, 20, x.size(1)]).cuda()  # B,128,20,T
        x = x.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, 20, 1)  # B, 512, 20, T
        x = torch.cat([x, n], 1)
        for block in self.decode:
            x = block(x)
        for block in self.g1:
            x = block(x)
        g1 = x.clone()
        c1 = self.att1(s, g1, len)
        x = self.attconv1(torch.cat([x, c1], 1))
        for block in self.g2:
            x = block(x)
        g2 = x.clone()
        c2 = self.att2(s, g2, len)
        x = self.attconv2(torch.cat([x, c2], 1))
        for block in self.g3:
            x = block(x)
        return self.to_mel1(g1), self.to_mel2(g2), self.to_mel3(x)

class Discriminator(nn.Module):
    def __init__(self, num_class=1, max_conv_dim=512, phase='1'):
        super().__init__()
        dim_in = 32
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 5, 1, 2)]

        if phase == '1':
            repeat_num = 2
        elif phase == '2':
            repeat_num = 3
        else:
            repeat_num = 4

        for _ in range(repeat_num): # 80,4T --> 40,2T --> 20,T --> 10,T/2 --> 5,T/4
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.main = nn.Sequential(*blocks)

        uncond = []
        uncond += [nn.LeakyReLU(0.2)]
        uncond += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        uncond += [nn.LeakyReLU(0.2)]
        uncond += [Avgpool()]
        uncond += [nn.Linear(dim_out, num_class)]
        self.uncond = nn.Sequential(*uncond)

        cond = []
        cond += [nn.LeakyReLU(0.2)]
        cond += [nn.Conv2d(dim_out + 512, dim_out, 5, 1, 2)]
        cond += [nn.LeakyReLU(0.2)]
        cond += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        cond += [nn.LeakyReLU(0.2)]
        cond += [Avgpool()]
        cond += [nn.Linear(dim_out, num_class)]
        self.cond = nn.Sequential(*cond)

    def forward(self, x, c, vid_max_length):
        # c: B,C,T
        f_len = final_length(vid_max_length)
        c = c.mean(2) #B,C
        c = c.unsqueeze(2).unsqueeze(2).repeat(1, 1, 5, f_len)
        out = self.main(x).clone()
        uout = self.uncond(out)
        out = torch.cat([out, c], dim=1)
        cout = self.cond(out)
        uout = uout.view(uout.size(0), -1)  # (batch, num_domains)
        cout = cout.view(cout.size(0), -1)  # (batch, num_domains)
        return uout, cout

class sync_Discriminator(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )

        self.Res_block = nn.Sequential(
            BasicBlock(256, 256)
        )

        self.Linear = nn.Linear(256 * 20, 512)
        self.temp = temp

    def forward(self, v_feat, aud, gen=False):
        # v_feat: B, S, 512
        a_feat = self.frontend(aud)
        a_feat = self.Res_block(a_feat)
        b, c, f, t = a_feat.size()
        a_feat = a_feat.view(b, c * f, t).transpose(1, 2).contiguous()  # B, T/4, 256 * F/4
        a_feat = self.Linear(a_feat)    # B, S, 512

        if gen:
            sim = torch.abs(F.cosine_similarity(v_feat, a_feat, 2)).mean(1)    #B, S
            loss = 5 * torch.ones_like(sim) - sim
        else:
            v_feat_norm = F.normalize(v_feat, dim=2)    #B,S,512
            a_feat_norm = F.normalize(a_feat, dim=2)    #B,S,512

            sim = torch.bmm(v_feat_norm, a_feat_norm.transpose(1, 2)) / self.temp #B,v_S,a_S

            nce_va = torch.mean(torch.diagonal(F.log_softmax(sim, dim=2), dim1=-2, dim2=-1), dim=1)
            nce_av = torch.mean(torch.diagonal(F.log_softmax(sim, dim=1), dim1=-2, dim2=-1), dim=1)

            loss = -1/2 * (nce_va + nce_av)

        return loss

def gan_loss(inputs, label=None):
    # non-saturating loss with R1 regularization
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()

def final_length(vid_length):
    half = (vid_length // 2)
    quad = (half // 2)
    return quad