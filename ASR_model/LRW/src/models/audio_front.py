from torch import nn
from src.models.resnet import BasicBlock
import torch

class Audio_front(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.in_channels = in_channels

        self.frontend = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
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

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.frontend(x)    #B, 256, F/4, T/4
        x = self.Res_block(x)  #B, 256, F/4, T/4
        b, c, f, t = x.size()
        x = x.view(b, c*f, t).transpose(1, 2).contiguous() #B, T/4, 256 * F/4
        x = self.dropout(x)
        x = self.Linear(x)  #B, T/4, 512
        return x

