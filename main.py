from __future__ import print_function, division
import os
import argparse
import datetime
import time
import shutil
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter

from data_loading import Protein_Dataset
from model import LSTM_CRF, BiLSTM
from utils import *

max_seq_len = 698
START_TAG = 8
STOP_TAG = 9

parser = argparse.ArgumentParser(description='deep learning with CRF')
parser.add_argument('run', metavar='DIR', help='specify the name of summary and model')
# parser.add_argument('model', choices=['cnn', 'rnn'], help='choose model between cnn and rnn')
parser.add_argument('optimizer', choices=['sgd', 'adam'], default='adam', help='choose optimizer')
parser.add_argument('--crf', action='store_true', default=False, help="If specified, CRF will be added after nn")
parser.add_argument('--load_model', action='store_true', default=False, help="If specified, retrain the given model")
parser.add_argument('-e', '--epochs', default=30, type=int, metavar='N', help='number of epochs to run (default: 30)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='batch size of training data (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--adjust_lr', action='store_true', default=False, help="If specified, adjust lr based on validation set accuracy")
parser.add_argument('--lr_decay', default=0.05, type=float, help='the learning rate decay ratio (default: 0.05)')
# parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate for lr scheduling (default: 1e-6)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd (default: 0.9)')
parser.add_argument('-p', '--optim_patience', default=3, type=int, help='patience for lr scheduler (default: 3)')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout factor. default: 0.5')
parser.add_argument('--patience', default=5, type=int, help='patience for early stopping (default: 5)')
parser.add_argument('--least_iter', default=30, type=int, help='guaranteed number of training iterations (default: 30)')
parser.add_argument('-c', '--clean', action='store_true', default=False, help="If specified, clear the summary directory first")
parser.add_argument('--hidden', default=100, type=int, metavar='N', help='number of hidden units in RNN (default: 100)')
parser.add_argument('--layers', default=1, type=int, metavar='N', help='number of layers of RNN (default: 1)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
else:
    print("Using CPU")

# Initialize dataloader
SetOf7604Proteins_path = os.path.expanduser('../data/ProteinProperty_Project/SetOf7604Proteins/')
CASP11_path = os.path.expanduser('../data/ProteinProperty_Project/CASP11/')
CASP12_path = os.path.expanduser('../data/ProteinProperty_Project/CASP12/')

train_dataset = Protein_Dataset(SetOf7604Proteins_path, 'trainList', max_seq_len)
valid_dataset = Protein_Dataset(SetOf7604Proteins_path, 'validList', max_seq_len)
test_dataset = Protein_Dataset(SetOf7604Proteins_path, 'testList', max_seq_len, padding=True)
CASP11_dataset = Protein_Dataset(CASP11_path, 'proteinList', max_seq_len, padding=True)
CASP12_dataset = Protein_Dataset(CASP12_path, 'proteinList', max_seq_len, padding=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
CASP11_loader = DataLoader(CASP11_dataset, batch_size=64, shuffle=False, num_workers=4)
CASP12_loader = DataLoader(CASP12_dataset, batch_size=64, shuffle=False, num_workers=4)

# Save model information
use_crf = '_crf' if args.crf else ''
model_name = 'rnn' + use_crf
writer_path = os.path.join("logger", model_name, args.run)
if os.path.exists(writer_path) and args.clean:
    shutil.rmtree(writer_path, ignore_errors=True)
save_model_dir = os.path.join(os.getcwd(), "saved_model", model_name)
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
model_name = [model_name, 'epochs', str(args.epochs), args.optimizer,
              'lr', str(args.lr), 'hidden', str(args.hidden),
              'layers', str(args.layers)]
model_name = '_'.join(model_name)
model_path = os.path.join(save_model_dir, model_name)
print('writer_path:', writer_path)
print('save_model_dir:', save_model_dir)
print('model_name:', model_name)


if args.crf:
    model = LSTM_CRF(args.hidden, args.layers, args.dropout)
else:
    model = BiLSTM(args.hidden, 8, args.dropout, args.layers)
    criterion = nn.CrossEntropyLoss()
if args.load_model:
    model.load_state_dict(torch.load(model_path))
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
lr_lambda = lambda epoch: 1 / (1 + (epoch + 1) * args.lr_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

if use_cuda:
    model = model.cuda()
    if not args.crf:
        criterion =criterion.cuda()

best_valid_acc = 0.0
patience = 0
writer = SummaryWriter(log_dir=writer_path)
len_train = len(train_dataset)
len_valid = len(valid_dataset)
start = time.time()

for epoch in range(args.epochs):
    correct_train, total_train = 0, 0
    loss_train = 0

    for i, (features, labels, lengths, _) in enumerate(train_loader):
        model.train()
        features = cuda_var_wrapper(features.transpose(0, 1))
        labels = cuda_var_wrapper(labels.transpose(0, 1))
        features, labels, lengths = sort_by_length(features, labels, lengths)
        if args.crf:
            loss = model(features, labels, lengths)
        else:
            loss = criterion(output.view(-1, 8), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5 == 0: print(loss.data[0])
        correct_batch, total_batch, loss_batch = eval(model, features, labels, lengths, crf=args.crf)
        loss_train += loss_batch
        correct_train += correct_batch
        total_train += total_batch

    accuracy_train = correct_train / total_train
    accuracy_valid, loss_valid = evaluate(model, valid_loader)
    loss_train /= len_train
    loss_valid /= len_valid

    writer.add_scalars('data/accuracy_group', {'accuracy_train': accuracy_train,
                                               'accuracy_valid': accuracy_valid}, epoch + 1)
    writer.add_scalars('data/loss_group', {'loss_train': loss_train,
                                           'loss_valid': loss_valid}, epoch + 1)
    print("Epoch {}: Training loss {:.4f}, accuracy {:.4f}; Validation loss {:.4f}, accuracy {:.4f}" \
          .format(epoch + 1, loss_train, accuracy_train, loss_valid, accuracy_valid))

    if args.adjust_lr:
        scheduler.step(accuracy_valid)
    if accuracy_valid > best_valid_acc:
        patience = 0
        best_valid_acc = accuracy_valid
        torch.save(model.state_dict(), model_path + '_best')
    else:
        patience += 1
        if patience >= args.patience and epoch + 1 >= args.least_iter:
            print("Early stopping activated")
            break

print("Time spent on training: {:.2f}s".format(time.time() - start))
writer.close()
torch.save(model.state_dict(), model_path)

model.load_state_dict(torch.load(model_path + '_best'))
accuracy_test, _ = evaluate(model, test_loader)
accuracy_CASP11, _ = evaluate(model, CASP11_loader)
accuracy_CASP12, _ = evaluate(model, CASP12_loader)
print("Model for best validation accuracy:")
print("Test accuracy {:.3f}".format(accuracy_test))
print("CASP11 accuracy {:.3f}".format(accuracy_CASP11))
print("CASP12 accuracy {:.3f}".format(accuracy_CASP12))

model.load_state_dict(torch.load(model_path))
accuracy_test, _ = evaluate(model, test_loader)
accuracy_CASP11, _ = evaluate(model, CASP11_loader)
accuracy_CASP12, _ = evaluate(model, CASP12_loader)
print("Model for complete training:")
print("Test accuracy {:.3f}".format(accuracy_test))
print("CASP11 accuracy {:.3f}".format(accuracy_CASP11))
print("CASP12 accuracy {:.3f}".format(accuracy_CASP12))
