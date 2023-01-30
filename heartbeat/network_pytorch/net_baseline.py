import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self,input_size=2,num_layers=2):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=8,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        out = nn.functional.sigmoid(out)
        return out


class MLP(nn.Module):
    def __init__(self,input_size=2):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size,64)
        self.bn = nn.BatchNorm1d(64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.bn(self.l1(x)))
        x = F.tanh(self.bn(self.l2(x)))
        x = F.tanh(self.bn(self.l3(x)))
        x = F.tanh(self.bn(self.l4(x)))
        out = F.sigmoid(self.out(x))
        return out    
    