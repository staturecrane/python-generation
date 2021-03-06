import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, z_dim, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        """
        self.i2h = nn.Linear(z_dim + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(z_dim + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        """

        self.i2h = nn.Linear(z_dim + input_size, hidden_size)
        self.h2h = nn.LSTMCell(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embeddings, input, hidden, state):
        """
        input_combined = torch.cat((embeddings, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        """

        input_combined = torch.cat((embeddings, input), 1)
        x_hat = self.i2h(input_combined)
        hidden, state = self.h2h(x_hat, (hidden, state))
        output = self.h2o(hidden)
        drop = self.dropout(output)

        return drop, hidden, state

    def initHidden(self):
        return (
            Variable(torch.FloatTensor(1, self.hidden_size).cuda().zero_()),
            Variable(torch.FloatTensor(1, self.hidden_size).cuda().zero_())
        )
