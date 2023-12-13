# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import math
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from dictionary_corpus import Corpus, tokenize
import os
import audio_model
from lm_argparser import lm_parser
from audio_utils import repackage_hidden, get_batch, get_finetuning, get_freqs, prepare_finetuning, get_sentencelengths, get_prediction, batchify

parser = argparse.ArgumentParser(parents=[lm_parser],
                                 description="Basic training and evaluation for RNN LM")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.info(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

logging.info("Loading data")
start = time.time()
corpus = Corpus(args.data)
logging.info("( %.2f )" % (time.time() - start))
ntokens = len(corpus.dictionary)
logging.info("Vocab size %d", ntokens)
eval_batch_size = 10

# Addition
train_perplexity_values = []
train_loss_values = []
all_train_perplexity_values = []
all_train_loss_values = []

# Addition
logging.info("Batchifying..")
if args.finetune:
    lens, qlens = get_sentencelengths(args.data, "train", "quest_train")
    train_data = get_finetuning(corpus.train, lens, False)
    train_freq_data = get_freqs(args.data, "train")
    train_freq = get_finetuning(train_freq_data, lens, False)
    train_targ_data = tokenize(corpus.dictionary, os.path.join(args.data, 'quest_train.txt'))
    train_targ = get_finetuning(train_targ_data, qlens, False)
    train_data, train_freq, train_targ = prepare_finetuning(train_data,train_freq,train_targ)

    lens, qlens = get_sentencelengths(args.data, "valid", "quest_valid")
    val_data = get_finetuning(corpus.valid, lens, False)
    val_freq_data = get_freqs(args.data, "valid")
    val_freq = get_finetuning(val_freq_data, lens, False)
    val_targ_data = tokenize(corpus.dictionary, os.path.join(args.data, 'quest_valid.txt'))
    val_targ = get_finetuning(val_targ_data, qlens, False)
    val_data, val_freq, val_targ = prepare_finetuning(val_data,val_freq,val_targ)

    lens, qlens = get_sentencelengths(args.data, "test", "quest_test")
    test_data = get_finetuning(corpus.test, lens, False)
    test_freq_data = get_freqs(args.data, "test")
    test_freq = get_finetuning(test_freq_data, lens, False)
    test_targ_data = tokenize(corpus.dictionary, os.path.join(args.data, 'quest_test.txt'))
    test_targ = get_finetuning(test_targ_data, qlens, False)
    test_data, test_freq, test_targ = prepare_finetuning(test_data,test_freq,test_targ)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
else:
    train_data = batchify(corpus.train, args.batch_size, args.cuda)
    train_freq_data = get_freqs(args.data, "train")
    train_freq = batchify(train_freq_data, 20, False)
    val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
    val_freq_data = get_freqs(args.data, "valid")
    val_freq = batchify(val_freq_data, 10, False)
    test_data = batchify(corpus.test, eval_batch_size, args.cuda)
    test_freq_data = get_freqs(args.data, "test")
    test_freq = batchify(test_freq_data, 10, False)
    criterion = nn.CrossEntropyLoss() 



###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")

if args.load == None:
    if args.model == "Transformer":
        model = audio_model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    else:
        model = audio_model.RNNModel(args.model,ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied) #ntokens removed
else:
    with open(args.load, 'rb') as f:
        if args.cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location = lambda storage, loc: storage)

if args.cuda:
    model.cuda()



###############################################################################
# Training code
###############################################################################


def evaluate(data_source, freq_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    if args.model != "Transformer":
        hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, audio, targets = get_batch(data_source, freq_source, i, args.bptt) # Addition line
            if args.model == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, audio, hidden) # Addition line
                output = output.view(-1, ntokens)
                hidden = repackage_hidden(hidden)

            total_loss += len(data) * nn.CrossEntropyLoss()(output, targets).item()

    return total_loss / (len(data_source) - 1)

# Addition function
def evaluate_finetuned(data_source, freq_source, targ_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, sentence in enumerate(data_source):
            if len(data_source) <= 10:
                eval_batch_size = len(data_source)-2 # when sentences are shorter than 10 words 
            else:
                eval_batch_size = 10
            for t in range(eval_batch_size, len(sentence)):
                data, audio, targets = get_prediction(sentence, freq_source[i], targ_source[i], t, eval_batch_size) #d & f tensors of same length, targ, list of tensor +1 ?

                if args.model == "Transformer":
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    hidden = model.init_hidden(1)
                    hidden = repackage_hidden(hidden)
                    output, hidden = model(data.unsqueeze(1), audio.unsqueeze(1), hidden) # reshaping also
                    output = output.view(-1, ntokens)

                total_loss +=  criterion(output, targets).item()

    return total_loss / (len(data_source) - 1)

def finetune():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    # Loop over training set one sentence at a time
    # Additions
    for i, sentence in enumerate(train_data):
        if len(train_data) <= 10:
            eval_batch_size = len(train_data)-2 # when sentences are shorter than 10 words 
        else:
            eval_batch_size = 10
        for t in range(eval_batch_size, len(sentence)):
            data, audio, targets = get_prediction(sentence, train_freq[i], train_targ[i], t, eval_batch_size) # data & freq tensors of same length, targ = list of datatensor +1
            model.zero_grad()
            if args.model == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = model.init_hidden(1)
                hidden = repackage_hidden(hidden)
                output, hidden = model(data.unsqueeze(1), audio.unsqueeze(1), hidden) #reshaping also
                output = output.view(-1, ntokens)

            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_data), lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

def train():
    global train_loss_values, train_perplexity_values
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    if args.model != "Transformer":
        hidden = model.init_hidden(args.batch_size)
    # Loop over training set in chunks of size args.bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, audio, targets = get_batch(train_data, train_freq, i, args.bptt) # Addition line
       
        model.zero_grad()
        
        if args.model == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            # truncated BPP
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, audio, hidden) # Addition line

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            # Addition
            train_perplexity_values.append(math.exp(cur_loss))
            train_loss_values.append(cur_loss)

            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    patience_exhausted = False
    epochs_since_improved = 0
    epoch = 0
    max_epoch = 6 # Addition to speed up training

    while not patience_exhausted and epoch < max_epoch:
        epoch_start_time = time.time()

        if args.finetune:
            finetune()
            val_loss = evaluate_finetuned(val_data, val_freq, val_targ) # Addition line
        else:
            train()
            val_loss = evaluate(val_data, val_freq) # Addition line

        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            epochs_since_improved = 0
        else:
            epochs_since_improved += 1
            if epochs_since_improved >= args.patience:
                patience_exhausted = True

            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            logging.info('| epochs since loss improved: ' + str(epochs_since_improved))
            logging.info('| reducing learning rate to ' + str(lr))

            # Return to the best saved model checkpoint
            with open(args.save, 'rb') as f:
                model = torch.load(f)
        
        logging.info('-' * 89)
        # Addition
        all_train_loss_values.append(train_loss_values)
        all_train_perplexity_values.append(train_perplexity_values)
        epoch += 1
        train_loss_values = []
        train_perplexity_values = []

except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

plt.figure(figsize=(10, 6))

# Addition
# Plotting training loss
plt.subplot(1, 2, 1)
for epoch, loss_values in enumerate(all_train_loss_values, 1):
    plt.plot(loss_values, marker='o', linestyle='-', label='Epoch {}'.format(epoch))
plt.title('Training Loss Over Batches')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plotting training perplexity for all epochs
plt.subplot(1, 2, 2)
for epoch, perplexity_values in enumerate(all_train_perplexity_values, 1):
    plt.plot(perplexity_values, marker='o', linestyle='-', label='Epoch {}'.format(epoch))
plt.title('Training Perplexity Over Batches')
plt.xlabel('Batch')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png')
plt.show()

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
if args.finetune:
    test_loss = evaluate_finetuned(test_data, test_freq, test_targ) # Addition line
else:
    test_loss = evaluate(test_data, test_freq) # Addition line
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
logging.info('=' * 89)