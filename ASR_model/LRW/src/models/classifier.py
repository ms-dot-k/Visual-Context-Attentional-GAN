import torch
from torch import nn

class Backend(nn.Module):
    def __init__(self, logits=True):
        super().__init__()
        self.logits = logits

        self.gru = nn.GRU(512, 512, 2, bidirectional=True, dropout=0.3)
        if logits:
            self.fc = nn.Linear(1024, 500)

    def forward(self, x):
        x = x.permute(1, 0, 2).contiguous()  # S,B,512
        self.gru.flatten_parameters()

        x, _ = self.gru(x)
        x = x.mean(0, keepdim=False)

        if self.logits:
            pred = self.fc(x)    # B, 500
            return pred
        else:
            return x

