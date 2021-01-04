#-*-coding:utf-8-*-

import torch
import torch.nn as nn


class NsmcLstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NsmcLstm, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        lstm_out = self.lstm(x)
        linear_out = self.linear(lstm_out[0])
        out = self.sigmoid(linear_out)

        return out
