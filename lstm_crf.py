from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from data_loading import Protein_Dataset


# Helper functions to make the code more readable.


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    """ Given a vector, return log_sum_exp with last dimension reduced.
    """
    last_dimension = len(vec.size()) - 1
    max_score, _ = torch.max(vec, last_dimension)
    _exp = torch.exp(vec - max_score.unsqueeze(last_dimension))
    _sum = torch.sum(_exp, last_dimension)
    _log = max_score + torch.log(_sum)
    return _log

class BiLSTM_CRF(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=1):
        super(BiLSTM_CRF, self).__init__()
        self.tagset_size = 10 # 8 labels + START_TAG + STOP_TAG

        self.lstm = nn.LSTM(n_features, n_hidden // 2, n_layers, bidirectional=True)
        self.linear = nn.Linear(n_hidden, self.tagset_size)

        # transitions[i,j] = the score of transitioning from j to i. Enforcing no 
        # transition from STOP_TAG and to START_TAG
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

    def _get_lstm_features(self, prt_seqs):
        """Given input of batched protein sequences, compute the lstm features.
        Input: [seq_len, batch_size, features]
        Output: [seq_len, batch_size, tagset_size]
        """
        lstm_out, _ = self.lstm(prt_seqs)
        lstm_feats = self.linear(lstm_out)
        return lstm_feats

    def _score_sequence(self, features, labels):
        """Gives the scores of a batch of ground truth tagged sequences
        feats: [sequence_length, batch_size, tagset_size]
        labels: [sequence_length, batch_size]
        """
        batch_size = features.size()[1]
        score = Variable(torch.zeros(batch_size))
        start_labels = torch.LongTensor(1, batch_size).fill_(START_TAG)
        stop_labels = torch.LongTensor(1, batch_size).fill_(STOP_TAG)
        labels = torch.cat((start_labels, labels), 0)
        for i, feat in enumerate(features):
            score += self.transitions[labels[i + 1], labels[i]]# + feat[:, labels[i + 1]]
        score = score + self.transitions[stop_labels, labels[-1]]
        return score


    def _forward_alg(self, feats):
        """Do the forward algorithm to compute the partition function
        feats: [sequence_length, batch_size, tagset_size]
        """
        init_alphas = torch.Tensor(feats.size()[1:]).fill_(-10000.)
        init_alphas[:, START_TAG] = 0.
        forward_var = Variable(init_alphas.unsqueeze(1))

        # Iterate through the sentence
        for feat in feats:
            # feat.size() = batch_size x tagset_size
            emit_score = feat.unsqueeze(2)
            # [batch, tagset, tagset] = [batch, 1, tagset] + [tagset, tagset] + [batch.tagset, 1]
            tag_var = forward_var + self.transitions + emit_score
            forward_var = log_sum_exp(tag_var).unsqueeze(1)
        terminal_var = (forward_var + self.transitions[STOP_TAG]).squeeze()
        alpha = log_sum_exp(terminal_var)
        return alpha


    def _viterbi_decode(self, feats):
        """Given the LSTM extracted features [Seq_len x Batch_size x tagset_size], return the Viterbi
           Decoded score and most probable sequence prediction.
        """
        batch_size = feats.size()[1]
        backpointers = []
        init_vvars = torch.Tensor(feats.size()[1:]).fill_(-10000.)
        init_vvars[:, START_TAG] = 0
        forward_var = Variable(init_vvars.unsqueeze(1)) # [batch x 1 x tagset]

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
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, features, labels):
        feats = self._get_lstm_features(features)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sequence(feats, labels)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__ == '__main__':
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
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = BiLSTM_CRF(66, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for i, (features, labels, lengths) in enumerate(test_loader):
            features = Variable(features.float().transpose(0, 1))
            labels = labels.transpose(0, 1)
            # optimizer.zero_grad()
            feats = model._get_lstm_features(features)

            start = time.time()
            score2, tagged_seq2 = model._viterbi_decode(feats)
            print("Time spent on decoding: {:.3f}s".format(time.time() - start))
            scores, tagged_seqs = [], []
            start = time.time()
            for feat in feats.transpose(0, 1):
                score, tagged_seq = model._viterbi_decode_origin(feat)
                scores.append(score)
                tagged_seqs.append(tagged_seq)
            print("Time spent on decoding original: {:.3f}s".format(time.time() - start))
            print(len(scores), len(tagged_seqs), len(tagged_seqs[0]))
            print(score2)
            print(scores)
            assert tagged_seq2[:, 0].data.tolist() == tagged_seqs[0]
            if i == 0: break
            # loss.backward()
            # if i % 20 == 0:
            #     print('Loss at {}: {}'.format(i, loss.data[0]))
            # optimizer.step()
