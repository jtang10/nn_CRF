from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from crf import CRF


class BiLSTM(nn.Module):
    def __init__(self, n_hidden, linear_out, n_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(66, n_hidden // 2, n_layers, bidirectional=True)
        self.linear = nn.Linear(n_hidden, linear_out)

    def forward(self, features, lengths):
        packed_input = pack_padded_sequence(features, lengths)
        output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(output)
        output = self.linear(output)
        return output


class LSTM_CRF(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(LSTM_CRF, self).__init__()
        self.lstm = BiLSTM(n_hidden, 10, n_layers=n_layers)
        self.crf1 = CRF()

    def forward(self, features, labels, lengths):
        lstm_feats = self.lstm(features, lengths)
        labels = labels.data
        score = self.crf1.neg_log_likelihood(lstm_feats, labels)
        return score

    def decode(self, features, lengths):
        lstm_feats = self.lstm(features, lengths)
        _, output = self.crf1(lstm_feats)
        return output