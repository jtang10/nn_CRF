from __future__ import print_function, division
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

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

def sort_by_length(input_features, input_labels, input_lengths):
    """Prepare the input features and labels for pack_padded_sequence. Sort them
    by length in descending order.
    """
    sorted_length, ix = torch.sort(input_lengths, dim=0, descending=True)
    if use_cuda:
        ix = ix.cuda()
    max_length = sorted_length[0]
    sorted_features = input_features[:, ix, :]
    sorted_labels = input_labels[:, ix]
    return sorted_features[:max_length, ...], sorted_labels[:max_length, :], sorted_length.tolist()

def cuda_var_wrapper(var, volatile=False):
    """Use CUDA Variable if GPU available. Also specify volatile for inference.
    """ 
    var = Variable(var, volatile=volatile)
    if use_cuda:
        var = var.cuda()
    return var

def evaluate(model, dataloader, rnn=True):
    model.eval()
    correct = 0.0
    total = 0.0
    for i, (features, labels, lengths) in enumerate(dataloader):
        features = cuda_var_wrapper(features.transpose(0, 1), volatile=True)
        labels = cuda_var_wrapper(labels.transpose(0, 1), volatile=True)
        if rnn:
            features, labels, lengths = sort_by_length(features, labels, lengths)
        output = model.decode(features, lengths)
        correct_batch, total_batch = get_batch_accuracy(labels, output, lengths)
        correct += correct_batch
        total += total_batch
    print("{} out of {} label predictions are correct".format(correct, total))
    return correct / total

def get_batch_accuracy(labels, output, lengths):
    """Get the number of correct predictions within the batch.
    labels: [seq_len x batch_size]
    output: [seq_len x batch_size x 8]
    """
    correct = 0.0
    total = 0.0
    # _, output = torch.max(output, 2)
    correct_matrix = torch.eq(output, labels).data
    for j, length in enumerate(lengths):
            correct += torch.sum(correct_matrix[:length, j])
    total += sum(lengths)
    return correct, total
