import torch.nn as nn


class RNN(nn.Module):
    def __init__(self,input_size=12,num_layers=2):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=50,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(50, 1)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        out = nn.functional.sigmoid(out)
        return out


class MLP(nn.Module):
    def __init__(self,input_size=12):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size,50)
        self.l2 = nn.Linear(50,50)
        self.out = nn.Linear(50, 1)

    def forward(self, x):
        x = self.l1(x)
        x = nn.functional.tanh(x)
        x = self.l2(x)
        x = nn.functional.tanh(x)
        out = self.out(x)
        out = nn.functional.sigmoid(out)
        return out    
    