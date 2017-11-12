from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter


# Helper functions to make the code more readable.


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, n_features, n_hidden, n_layers=1):
        super(BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.lstm = nn.LSTM(n_features, n_hidden // 2, n_layers, bidirectional=True)
        self.linear = nn.Linear(n_hidden, self.tagset_size)

        # transitions[i,j] = the score of transitioning from j to i. Enforcing no 
        # transition from STOP_TAG and to START_TAG
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _get_lstm_features(self, prt_seqs):
        lstm_out, _ = self.lstm(prt_seqs)
        print("LSTM output size:", lstm_out.size())
        lstm_feats = self.linear(lstm_out)
        print("Linear output size:", lstm_feats.size())
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a ground truth tagged sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def forward(self, prt_seqs):
        return self._get_lstm_features(prt_seqs)

if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = {key: value for (key, value) in \
            zip(['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', START_TAG, STOP_TAG], range(10))}
    model = BiLSTM_CRF(tag_to_ix, 66, 50)
    prt_seqs = Variable(torch.rand(20, 1, 66))
    print("Test input size:", prt_seqs.size())
    lstm_out = model(prt_seqs)
    for feat in lstm_out:
        print(feat.size())
        break
    

