from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data_loading import Protein_Dataset

from lstm import BiLSTM
from crf import CRF

max_seq_len = 698
batch_size = 64
epochs = 1
START_TAG = 8
STOP_TAG = 9

SetOf7604Proteins_path = os.path.expanduser('../data/SetOf7604Proteins/')
trainList_addr = 'trainList'
validList_addr = 'validList'
testList_addr = 'testList'

train_dataset = Protein_Dataset(SetOf7604Proteins_path, trainList_addr, max_seq_len)
valid_dataset = Protein_Dataset(SetOf7604Proteins_path, validList_addr, max_seq_len)
test_dataset = Protein_Dataset(SetOf7604Proteins_path, testList_addr, max_seq_len, padding=True)
len_train_dataset = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

model = BiLSTM(100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for i, (features, labels, lengths) in enumerate(test_loader):
        features = Variable(features.float().transpose(0, 1))
        labels = Variable(labels.transpose(0, 1))
        features, labels, lengths = sort_by_length(features, labels, lengths)
        # print('features:',features.size())
        # print('labels:', labels.size())
        # print('lengths:', len(lengths))
        output = model(features, lengths)
        optimizer.zero_grad()
        # print(lengths)
        # print(output.size(), labels.size())
        loss = criterion(output.view(-1, 8), labels.view(-1))
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print("Step {}, loss: {:.3f}".format(i, loss.data[0]))