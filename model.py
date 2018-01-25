from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_loading import Protein_Dataset
from utils import *


use_cuda = torch.cuda.is_available()
START_TAG = 8
STOP_TAG = 9

class LSTM_Multi_CRF(nn.Module):
    def __init__(self, n_hidden, n_layers=1, dropout_rate=0.5):
        super(LSTM_Multi_CRF, self).__init__()
        # self.lstm = BiLSTM(n_hidden, )

class LSTM_CRF(nn.Module):
    def __init__(self, n_hidden, n_layers=1, dropout_rate=0.5):
        super(LSTM_CRF, self).__init__()
        self.lstm = BiLSTM(n_hidden, 10, dropout_rate=dropout_rate, n_layers=n_layers)
        self.crf = CRF()

    def forward(self, features, labels, lengths):
        lstm_feats = self.lstm(features, lengths)
        score = self.crf.neg_log_likelihood(lstm_feats, labels)
        return score

    def decode(self, features, lengths):
        lstm_feats = self.lstm(features, lengths)
        _, output = self.crf(lstm_feats)
        return output


class BiLSTM(nn.Module):
    def __init__(self, n_hidden, linear_out, dropout_rate, n_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(66, n_hidden // 2, n_layers, bidirectional=True)
        self.linear = nn.Linear(n_hidden, linear_out)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features, lengths):
        packed_input = pack_padded_sequence(features, lengths)
        output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(output)
        output = self.dropout(output)
        output = self.linear(output)
        return output


class CRF(nn.Module):
    def __init__(self, tagset_size=10):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size # 8 labels + START_TAG + STOP_TAG
        # transitions[i,j] = the score of transitioning from j to i. Enforcing no 
        # transition from STOP_TAG and to START_TAG
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000


    def _score_sequence(self, feats, labels):
        """Gives the scores of a batch of ground truth tagged sequences
        feats: [sequence_length, batch_size, tagset_size]
        labels: [sequence_length, batch_size]
        """
        batch_size = feats.size()[1]
        score = cuda_var_wrapper(torch.zeros(1, batch_size))
        start_labels = cuda_var_wrapper(torch.LongTensor(1, batch_size).fill_(START_TAG))
        stop_labels = cuda_var_wrapper(torch.LongTensor(1, batch_size).fill_(STOP_TAG))
        emit = feats.gather(2, labels.unsqueeze(2)).squeeze()
        labels = torch.cat((start_labels, labels), 0)
        for i, feat in enumerate(feats):
            trans = self.transitions[labels[i + 1], labels[i]].unsqueeze(0)
            # print('trans.size():', trans.size())
            score += trans
        score = score + self.transitions[stop_labels, labels[-1]] + emit.sum(0,keepdim=True)
        return score.squeeze()


    def _forward_alg(self, feats):
        """Do the forward algorithm to compute the partition function
        feats: [sequence_length, batch_size, tagset_size]
        """
        init_alphas = torch.Tensor(feats.size()[1:]).fill_(-10000.)
        init_alphas[:, START_TAG] = 0.
        forward_var = Variable(init_alphas.unsqueeze(1))
        if use_cuda:
            forward_var = forward_var.cuda()
        # Iterate through the sentence
        for feat in feats:
            # feat.size() = batch_size x tagset_size
            emit_score = feat.unsqueeze(2)
            # [batch, tagset, tagset] = [batch, 1, tagset] + [tagset, tagset] + [batch, tagset, 1]
            tag_var = forward_var + self.transitions + emit_score
            forward_var = log_sum_exp(tag_var).unsqueeze(1)
        terminal_var = (forward_var + self.transitions[STOP_TAG]).squeeze()
        alpha = log_sum_exp(terminal_var)
        return alpha


    def _viterbi_decode(self, feats):
        """Given the nn extracted features [Seq_len x Batch_size x tagset_size], return the Viterbi
           Decoded score and most probable sequence prediction. Input feats is assumed to have
           dimension of 3.
        """
        batch_size = feats.size()[1]
        backpointers = []
        init_vvars = torch.Tensor(feats.size()[1:]).fill_(-10000.)
        init_vvars[:, START_TAG] = 0
        forward_var = Variable(init_vvars.unsqueeze(1)) # [batch x 1 x tagset]
        if use_cuda:
            forward_var = forward_var.cuda()

        for feat in feats:
            # next_tag_var: [batch x tagset x tagset]
            next_tag_var = forward_var + self.transitions
            # viterbi_vars: [batch x tagset]
            viterbi_vars, best_tag_id = next_tag_var.max(2)
            forward_var = (viterbi_vars + feat).unsqueeze(1)
            # list of [batch x tagset], equivalent to [seq_len x batch x tagset]
            backpointers.append(best_tag_id)

        # terminal_var: [batch x tagset]
        terminal_var = (forward_var + self.transitions[STOP_TAG]).squeeze()
        path_score, best_tag_id = terminal_var.max(1)

        best_path = [best_tag_id.unsqueeze(0)] # [batch]
        for backpointer in reversed(backpointers):
            # backpointer: [batch x tagset]
            best_tag_id = backpointer[range(batch_size), best_tag_id.data]
            best_path.append(best_tag_id.unsqueeze(0))
        start = best_path.pop()
        # assert start[:, 0] == START_TAG
        best_path = torch.cat(best_path[::-1], 0)
        return path_score, best_path


    def neg_log_likelihood(self, feats, labels):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sequence(feats, labels)
        return torch.mean(forward_score - gold_score)


    def forward(self, feats):  
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
