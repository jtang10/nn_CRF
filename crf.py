from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loading import Protein_Dataset
from utils import log_sum_exp

use_cuda = torch.cuda.is_available()
START_TAG = 8
STOP_TAG = 9

class CRF(nn.Module):
    def __init__(self):
        super(CRF, self).__init__()
        self.tagset_size = 10 # 8 labels + START_TAG + STOP_TAG
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
        score = Variable(torch.zeros(1, batch_size))
        start_labels = torch.LongTensor(1, batch_size).fill_(START_TAG)
        stop_labels = torch.LongTensor(1, batch_size).fill_(STOP_TAG)
        if use_cuda:
            score = score.cuda()
            start_labels = start_labels.cuda()
            stop_labels = stop_labels.cuda()
        labels = torch.cat((start_labels, labels), 0)
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[labels[i + 1], labels[i]].unsqueeze(0) + feat[:, labels[i + 1]]
        score = score + self.transitions[stop_labels, labels[-1]]
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

    def _viterbi_decode_origin(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                _, best_tag_id = torch.max(next_tag_var, dim=1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        _, best_tag_id = torch.max(terminal_var, dim=1)
        path_score = terminal_var[0][best_tag_id]
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id.data.tolist()[0]]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id.data.tolist()[0]]
            best_path.append(best_tag_id.data.tolist()[0])
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        # assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood(self, feats, labels):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sequence(feats, labels)
        return torch.mean(forward_score - gold_score)


    def forward(self, feats):  
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq

if __name__ == '__main__':
    model = CRF()
    feat = Variable(torch.rand(20, 2, 10))
    score1, path1 = model._viterbi_decode(feat)
    print(path1.size())
    path1 = path1.t().data.tolist()
    score, path = [], []
    for i in range(2):
        score2, path2 = model._viterbi_decode_origin(feat[:, i, :])
        score.append(score2)
        path.append(path2)
    score = torch.cat(score)
    print(score1, score)
    print(path1)
    print(path)
