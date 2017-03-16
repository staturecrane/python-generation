import argparse
import sys
import torch
import torch.nn as nn
import time
import random
from torch.autograd import Variable
from utils.utils import readLines, randomTrainingSet, inputTensor, timeSince
from model.model import RNN
from config import N_LETTERS, ALL_LETTERS, Z_DIM, SAMPLE_LENGTH

parser = argparse.ArgumentParser(description='Train to generate Python code')
parser.add_argument('data', type=str, help='the file containing training text')

args = parser.parse_args(sys.argv[1:])

programs = readLines(args.data)

rnn = RNN(Z_DIM, N_LETTERS, 128, N_LETTERS)
rnn.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.0005
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


def train(embeddings_tensor, input_line_tensor, target_line_tensor):

    hidden, state  = rnn.initHidden()

    rnn.zero_grad()

    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden, state  = rnn(embeddings_tensor, input_line_tensor[i], hidden,
                state)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    optimizer.step()

    return output, loss.data[0] / input_line_tensor.size()[0]

max_length = SAMPLE_LENGTH

def sample(start_letters="#idaue"):
    start_letter = start_letters[random.randint(1, len(start_letters) - 1)]
    input = Variable(inputTensor(start_letter).cuda())
    embeddings = Variable(torch.zeros(1, Z_DIM).cuda())
    hidden, state = rnn.initHidden()

    output_program = start_letter

    for i in range(max_length):
        output, hidden, state = rnn(embeddings, input[0], hidden, state)
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
print_every = 100
sample_every = 500
save_every = 1000
all_losses = []
total_loss = 0

start = time.time()


for epoch in range(1, n_epochs):
    output, loss = train(*randomTrainingSet(Z_DIM, programs))
    total_loss += loss

    if epoch % print_every == 0:
        print("epoch {}: {}".format(epoch, loss))

    if epoch % sample_every == 0:
        print(sample())

    if epoch % save_every == 0:
        with open('checkpoints/model.pt', 'wb') as f:
            torch.save(rnn, f)
