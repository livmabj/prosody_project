# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import os
import pickle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Detaches hidden states from their history,
# to avoid backpropagating when you don't want to
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

# Addition all functions below
def get_batch(source, source_audio, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # adding audio-info
    audio = source_audio[i:i+seq_len]

    # predict the sequences shifted by one word
    target = source[i+1:i+1+seq_len].view(-1)
    return data, audio, target


def get_prediction(source, source_audio, targets, i, seq_length):
    # batching function for fine-tuning, gets batches of sentences
    # Get the sequences of the previous n tokens for data and audio in fine-tuning
    data = source[i-seq_length:i+1]  # i is the index of the current token
    audio = source_audio[i-seq_length:i+1]   
    # Predict the next token
    target = targets[i-seq_length+1:i+2]

    return data, audio, target


def get_freqs(path, type):
    # Returns the frequency data as one long tensor
    with open(os.path.join(path, "freq_" + type + '.pkl'), "rb") as fin:
        data = pickle.load(fin)
    freqs = torch.cat(data, dim=0)
    return freqs


def get_finetuning(data, lens, cuda):
    # Returns the tokenized data into sequences matching sentences
    # depending on the lengths of the sentences in the file
    grouped_data = []
    index = 0
    for num in lens:
        group = data[index : index + num]
        grouped_data.append(group)
        index += num
    if cuda:
        data = [d.cuda() for d in data]
    return grouped_data


def prepare_finetuning(data,audio,targets):
    # Creates targets from input and the first aux of the derived question
    new_targets = []
    for i,d in enumerate(data):
        new_targets.append(torch.cat((d, targets[i]), dim=0))
    return data, audio, new_targets


def get_sentencelengths(path, type, q_type):
    #gets the sentencelengths of fine-tuning input and targets
    with open(os.path.join(path, type+".txt"), "r") as fin:
        sents = fin.readlines()
        sents = [sent.split() for sent in sents]
        lens = [len(sent) for sent in sents]
    with open(os.path.join(path, q_type+".txt"), "r") as fin:
        sents = fin.readlines()
        sents = [sent.split() for sent in sents]
        qlens = [len(sent) for sent in sents]
    return lens, qlens