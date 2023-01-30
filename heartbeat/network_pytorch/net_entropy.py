import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,input_size=4):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.bn = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.tanh(self.bn(self.l1(x)))
        x = F.tanh(self.bn(self.l2(x)))
        x = F.tanh(self.bn(self.l3(x)))
        x = F.tanh(self.bn(self.l4(x)))
        out = F.sigmoid(self.out(x))
        return out
    