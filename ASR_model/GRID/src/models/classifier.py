from torch import nn

class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(256, 256, 2, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, 27 + 1)

    def forward(self, x):
        x = x.permute(1, 0, 2).contiguous()  # S,B,96*7*7
        self.gru.flatten_parameters()

        x, _ = self.gru(x)  # S,B,512
        x = x.permute(1, 0, 2).contiguous()
        x = self.fc(x)  # B, S, 28
        return x

