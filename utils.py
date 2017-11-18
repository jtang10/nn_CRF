from __future__ import print_function, division
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

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
    if use_cuda:
        var = Variable(var, volatile=volatile).cuda()
    else:
        var = Variable(var, volatile=volatile)
    return var

def evaluate(model, dataloader):
    model.eval()
    correct = 0.0
    total = 0.0
    for i, (features, labels, lengths) in enumerate(dataloader):
        max_length = max(lengths)
        features = cuda_var_wrapper(features[:, :, :max_length], volatile=True)
        labels = cuda_var_wrapper(labels[:, :max_length], volatile=True)
        output = model(features)
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
    _, prediction = torch.max(output, 2)
    correct_matrix = torch.eq(prediction, labels).data
    for j, length in enumerate(lengths):
            correct += torch.sum(correct_matrix[j, :length])
    total += sum(lengths)
    return correct, total