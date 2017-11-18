from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sort_by_length(input_features, input_labels, input_lengths):
        sorted_length, ix = torch.sort(input_lengths, dim=0, descending=True)
        max_length = sorted_length[0]
        sorted_features = input_features[:, ix, :]
        sorted_labels = input_labels[:, ix]
        return sorted_features[:max_length, ...], sorted_labels[:max_length, :], sorted_length.tolist()


class BiLSTM(nn.Module):
    def __init__(self, n_hidden, n_layers=1, linear_out):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(66, n_hidden // 2, n_layers, bidirectional=True)
        self.linear = nn.Linear(n_hidden, linear_out)

    def forward(self, input_features, input_lengths):
        packed_input = pack_padded_sequence(input_features, input_lengths)
        output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(output)
        output = self.linear(output)
        return output
