from torch import nn
from src.models.resnet import BasicBlock
import torch

class Audio_front(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.in_channels = in_channels

        self.frontend = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.PReLU(32),

            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.Res_block = nn.Sequential(
            BasicBlock(64, 64, relu_type='prelu')
        )

        self.Linear = nn.Linear(64 * 20, 256)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.frontend(x)    #B, 64, F/4, T/4
        x = self.Res_block(x)  #B, 64, F/4, T/4
        b, c, f, t = x.size()
        x = x.view(b, c*f, t).transpose(1, 2).contiguous() #B, T/4, 64 * F/4
        x = self.dropout(x)
        x = self.Linear(x)  #B, T/4, 96
        return x

