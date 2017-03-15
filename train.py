import argparse
import sys
import torch
import torch.nn as nn
import time
import random
from torch.autograd import Variable
from utils.utils import readLines, randomTrainingSet, inputTensor, timeSince
from model.model import RNN
from config import N_LETTERS, ALL_LETTERS

parser = argparse.ArgumentParser(description='Train to generate Python code')
parser.add_argument('data', type=str, help='the file containing training text')

args = parser.parse_args(sys.argv[1:])

programs = readLines(args.data)

rnn = RNN(N_LETTERS, 128, N_LETTERS)
rnn.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.0005
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


def train(input_line_tensor, target_line_tensor):

    hidden  = rnn.initHidden()

    rnn.zero_grad()

    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden  = rnn(input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    optimizer.step()

    return output, loss.data[0] / input_line_tensor.size()[0]

max_length = 100

def sample(start_letter="i"):
    input = Variable(inputTensor(start_letter).cuda())
    hidden = rnn.initHidden()

    output_program = start_letter

    for i in range(max_length):
        output, hidden = rnn(input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == N_LETTERS - 1:
            break
        else:
            letter = ALL_LETTERS[topi]
            output_program += letter
            input = Variable(inputTensor(letter).cuda())
    return output_program


n_epochs = 100000
print_every = 50
plot_every = 500
all_losses = []
total_loss = 0

start = time.time()

for epoch in range(1, n_epochs):
    output, loss = train(*randomTrainingSet(programs))
    total_loss += loss

    if epoch % 100 == 0:
        print("epoch {}: {}".format(epoch, loss))

    if epoch % print_every == 0:
        print(sample())
