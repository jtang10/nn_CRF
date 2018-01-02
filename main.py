from __future__ import print_function, division
import os
import argparse
import datetime
import time
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_loading import Protein_Dataset
from model import LSTM_CRF, BiLSTM
from utils import *

max_seq_len = 698
START_TAG = 8
STOP_TAG = 9

parser = argparse.ArgumentParser(description='deep learning with CRF')
parser.add_argument('run', metavar='DIR', help='specify the name of summary and model')
parser.add_argument('-e', '--epochs', default=30, type=int, metavar='N', help='number of epochs to run (default: 30)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='batch size of training data (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--dropout', default=0.4, type=float, metavar='N', help='dropout factor. default: 0.4')
parser.add_argument('-c', '--clean', action='store_true', default=False, help="If specified, clear the summary directory first")
parser.add_argument('--hidden', default=100, type=int, metavar='N', help='number of hidden units in RNN (default: 100)')
parser.add_argument('--layers', default=1, type=int, metavar='N', help='number of layers of RFF (default: 1)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# Initialize dataloader
SetOf7604Proteins_path = os.path.expanduser('../data/SetOf7604Proteins/')
train_dataset = Protein_Dataset(SetOf7604Proteins_path, 'trainList', max_seq_len)
valid_dataset = Protein_Dataset(SetOf7604Proteins_path, 'validList', max_seq_len)
test_dataset = Protein_Dataset(SetOf7604Proteins_path, 'testList', max_seq_len, padding=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Save model information
writer_path = os.path.join("logger", args.run)
if os.path.exists(writer_path) and args.clean:
    shutil.rmtree(writer_path, ignore_errors=True)
save_model_dir = os.path.join(os.getcwd(), "saved_model")
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
model_name = ['rnn', datetime.datetime.now().strftime("%b%d_%H:%M"), args.run,
              'epochs', str(args.epochs), 'lr', str(args.lr),
              'hidden', str(args.hidden), 'layers', str(args.layers)]
model_name = '_'.join(model_name)
model_path = os.path.join(save_model_dir, model_name)

writer = SummaryWriter(log_dir=writer_path)

# model = BiLSTM(args.hidden, 8, args.dropout, args.layers)
model = LSTM_CRF(args.hidden, args.layers, args.dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    model = model.cuda()
    criterion =criterion.cuda()

best_accuracy = 0
len_train_dataset = len(train_dataset)
start = time.time()
for epoch in range(args.epochs):
    model.train()
    step_counter = 0
    for i, (features, labels, lengths) in enumerate(train_loader):
        features = cuda_var_wrapper(features.transpose(0, 1))
        labels = cuda_var_wrapper(labels.transpose(0, 1))
        features, labels, lengths = sort_by_length(features, labels, lengths)
        loss = model(features, labels, lengths)
        # output = model(features, lengths)
        # loss = criterion(output.view(-1, 8), labels.view(-1))
        optimizer.zero_grad()
        step_counter += features.size()[1]
        writer.add_scalar('data/loss', loss.data[0], step_counter + epoch * len_train_dataset)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
               %(epoch+1, args.epochs, i, len_train_dataset//args.batch_size, loss.data[0]))

    accuracy_train = evaluate(model, train_loader)
    accuracy_valid = evaluate(model, valid_loader)
    if accuracy_valid > best_accuracy:
        best_accuracy = accuracy_valid
        torch.save(model.state_dict(), model_path)
    writer.add_scalars('data/accuracy_group', {'accuracy_train': accuracy_train,
                                               'accuracy_valid': accuracy_valid}, (epoch + 1) * len_train_dataset)
    print("Training accuracy {:.4f}; Validation accuracy {:.4f}".format(accuracy_train, accuracy_valid))

print("Time spent on training: {:.2f}s".format(time.time() - start))
torch.save(model.state_dict(), model_path + '_2')
writer.close()

model.load_state_dict(torch.load(model_path))
accuracy_test = evaluate(model, test_loader)
print("Model for best validation accuracy:")
print("Test accuracy {:.3f}".format(accuracy_test))
model.load_state_dict(torch.load(model_path + '_2'))
accuracy_test = evaluate(model, test_loader)
print("Model for complete training:")
print("Test accuracy {:.3f}".format(accuracy_test))
