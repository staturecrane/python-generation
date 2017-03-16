import unicodedata
import random
import torch
import time
import math
from config import ALL_LETTERS, N_LETTERS, SAMPLE_LENGTH
from torch.autograd import Variable


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def readLines(filename):
    program = open(filename).read().split('--------')
    while '' in program:
        program.remove('')
    return [unicodeToAscii(line) for line in program]


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def inputTensor(line):
    length = SAMPLE_LENGTH if len(line) > SAMPLE_LENGTH else len(line)
    tensor = torch.zeros(length, 1, N_LETTERS)
    for li in range(length):
        letter = line[li]
        tensor[li][0][ALL_LETTERS.find(letter)] = 1
    return tensor


def targetTensor(line):
    length = SAMPLE_LENGTH if len(line) > SAMPLE_LENGTH else len(line)
    letter_indexes = [ALL_LETTERS.find(line[li]) for li in range(1,
        length)]
    letter_indexes.append(N_LETTERS - 1)
    return torch.LongTensor(letter_indexes)


def randomTrainingSet(z_dim, l):
    line = randomChoice(l)
    length = SAMPLE_LENGTH if len(line) > SAMPLE_LENGTH else len(line)
    if length > SAMPLE_LENGTH:
        random_idx = torch.randint(0, length - SAMPLE_LENGTH - 1)
        line = line[random_idx : random_idx + SAMPLE_LENGTH + 1]
    embeddings_tensor = Variable(torch.zeros(1, z_dim))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))

    return embeddings_tensor.cuda(), input_line_tensor.cuda(), target_line_tensor.cuda()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "{} {}".format(m, s)
