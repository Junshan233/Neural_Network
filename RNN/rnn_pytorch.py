import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

TIME_STEP = 50
INPUT_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H_SIZE = 64  # 隐藏单元个数
EPOCHS = 300
h_state = None

steps = np.linspace(0, np.pi * 2, 256, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=INPUT_SIZE,
                          hidden_size=H_SIZE,
                          num_layers=1,
                          batch_first=True)
        self.out = nn.Linear(H_SIZE, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        r_out = r_out.view(-1, H_SIZE)
        outs = self.out(r_out)
        return outs, h_state


rnn = RNN().to(DEVICE)
optimizer = optim.Adam(rnn.parameters())
criterion = nn.MSELoss()
