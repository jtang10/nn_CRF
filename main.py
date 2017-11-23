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
from model import LSTM_CRF, BiLSTM
from utils import *

max_seq_len = 698
batch_size = 64
epochs = 30
START_TAG = 8
STOP_TAG = 9

use_cuda = torch.cuda.is_available()

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
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# model = BiLSTM(100, 8)
ix_counter = 0
writer_path = os.path.join("logger", 'test' + str(ix_counter))
while os.path.exists(writer_path):
    ix_counter += 1
    writer_path = os.path.join("logger", 'test' + str(ix_counter))
writer = SummaryWriter(log_dir=writer_path)

model = LSTM_CRF(100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    model = model.cuda()
    criterion =criterion.cuda()


for epoch in range(epochs):
    model.train()
    step_counter = 0
    for i, (features, labels, lengths) in enumerate(train_loader):
        features = cuda_var_wrapper(features.transpose(0, 1))
        labels = cuda_var_wrapper(labels.transpose(0, 1))
        features, labels, lengths = sort_by_length(features, labels, lengths)
        output = model.decode(features, lengths)
        optimizer.zero_grad()
        loss = criterion(output.view(-1, 8), labels.view(-1))
        step_counter += features.size()[1]
        writer.add_scalar('data/loss', loss.data[0], step_counter + epoch * len_train_dataset)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
               %(epoch+1, epochs, i, len_train_dataset//batch_size, loss.data[0]))
        # if i == 5: break

    accuracy_train = evaluate(model, train_loader)
    accuracy_valid = evaluate(model, valid_loader)
    writer.add_scalars('data/accuracy_group', {'accuracy_train': accuracy_train,
                                               'accuracy_valid': accuracy_valid}, (epoch + 1) * len_train_dataset)
    print("Training accuracy {:.4f}; Validation accuracy {:.4f}".format(accuracy_train, accuracy_valid))
