import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from collections import OrderedDict
import itertools

import os
from torchvision import datasets, transforms, models

from fishdataset import SeqDataset, SubsetSampler,collate_seqs
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import numpy as np
from masked_cross_entropy import compute_xe_loss, compute_mse_loss

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--test', action='store_true', help='Test model on test_crops.csv')
parser.add_argument('-lm', '--load-model', type=str, help='Load model from file')
parser.add_argument('-bs', '--batch-size', type=int, default=1, help='Batch size')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('-s', '--suffix',  type=str, default=None, help='Suffix to store checkpoints')

args = parser.parse_args()

class SeqFishNet(nn.Module):
    def __init__(self, max_length, batch_size, n_features, hidden_size = 256, num_layers = 2):
        super(SeqFishNet, self).__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size= self.n_features, 
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bidirectional = True,
            )

        self.rnn_final = nn.LSTM(
            input_size= 2 * self.hidden_size, 
            hidden_size = 1,
            bidirectional = False,
            )


    def forward(self, inputs):

        batch_size = self.batch_size
        n_features = self.n_features
        max_length = self.max_length

        packed_output, _ = self.rnn(inputs)
        packed_output, _ = self.rnn_final(packed_output)
#           packed_output, _ = self.output_linear(packed_output)

        outputs, output_lengths = th.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        #unsorted_lengths = [output_lengths[i] for i in batch_lengths_order]
        #unsorted_outputs = [       outputs[i] for i in batch_lengths_order]

        #print(unsorted_lengths)
        #print(unsorted_outputs)
        return outputs, output_lengths#, th.autograd.Variable(th.FloatTensor([0.]))

        #return unsorted_outputs, unsorted_lengths#, th.autograd.Variable(th.FloatTensor([0.]))
        #return packed_output, _#, th.autograd.Variable(th.FloatTensor([0.]))


TRAIN_X_CSV = 'train_crops_X.csv'
TRAIN_Y_CSV = 'train_crops_Y.csv'

dataset = SeqDataset(
        X_csv_file=TRAIN_X_CSV,
        Y_csv_file=TRAIN_Y_CSV,
        return_numpy=False
        )

model = SeqFishNet(
    max_length = dataset.max_length,
    batch_size = args.batch_size,
    n_features = dataset.n_features,
    )
print(model)

initial_epoch = 0


optimizer = optim.RMSprop([ 
    { 'params': model.parameters() }, 
    ], lr= args.learning_rate, momentum=0.9)

if args.load_model:
    print("Loading checkpoint: " + args.load_model)
    checkpoint = th.load(args.load_model, map_location=lambda storage, loc: storage)
    #print(checkpoint.keys())
    initial_epoch = checkpoint['epoch']
    #print(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if th.cuda.is_available():
    model = model.cuda()
else:
    model = model.cpu()

idx_train, idx_valid = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
train_sampler = SubsetRandomSampler(idx_train[:])
valid_sampler = SubsetRandomSampler(idx_valid[:])

train_loader = DataLoader(dataset,
                      batch_size=args.batch_size,
                      sampler=train_sampler,
                      num_workers=4,
                      collate_fn= collate_seqs,
                      pin_memory=True if th.cuda.is_available() else False)

valid_loader = DataLoader(dataset,
                      batch_size=args.batch_size,
                      sampler=valid_sampler,
                      num_workers=4,
                      collate_fn= collate_seqs,
                      pin_memory=True if th.cuda.is_available() else False)

#dataset.max_length = 200

max_length = dataset.max_length
batch_size = args.batch_size
n_features = dataset.n_features

model.train()
for epoch in range(100):
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        def pack_batches(inputs, requires_grad=True, unpack=False):
            #print(inputs)
            batch_size = len(inputs)
            n_features = inputs[0].size()[1]
            #print(batch_size, n_features)
            batch_in = Variable(th.zeros((max_length, batch_size, n_features)))#, requires_grad=requires_grad)
            if th.cuda.is_available():
                batch_in = batch_in.cuda()
        
            batch_lengths = [input.size()[0] for input in inputs]
            batch_lengths_order = np.argsort(batch_lengths)[::-1]
            sorted_batch_lengths = [batch_lengths[i] for i in batch_lengths_order]
            sorted_inputs        = [       inputs[i] for i in batch_lengths_order]

            for i, (input, length) in enumerate(zip(sorted_inputs, sorted_batch_lengths)):
                batch_in[:length, i, :] = input

            packed_input = th.nn.utils.rnn.pack_padded_sequence(batch_in, sorted_batch_lengths)

            if unpack:
                packed_input, batch_lengths_order = th.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)

            return packed_input, batch_lengths_order

        def chainsaw(x):
            npx = x.numpy()
            _, counts = np.unique(npx, return_counts=True)
            saw = np.empty_like(npx).squeeze(axis=1)
            frame_index = 0
            for count in counts:
                saw[frame_index:frame_index+count] = np.linspace(1.,0.,count)
                frame_index += count
            saw = th.autograd.Variable(th.FloatTensor(np.expand_dims(saw, axis=1)), requires_grad=False)
            return saw


        #print(len(targets))
        #print("--------------------")
        #targets = [target[:,8:9].remainder(2) for target in targets] # keep only fish number mod 2
        targets = [chainsaw(target[:,8:9]) for target in targets] # keep only fish number mod 2
        #print(targets)
        #print("--------------------")
        data,   _ = pack_batches(inputs)
        target, _ = pack_batches(targets, requires_grad=False, unpack=True)

        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        #print(data)
        output, output_lengths = model(data)
        output_lengths = Variable(th.LongTensor(output_lengths))
        #target = th.squeeze(target, dim=2)
        #print(output)
        #print(output.size())
        #print("O --------------------")

        #print(output)
        #print("T --------------------")

        #print(target)
        #print(output_lengths)

        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.

        Returns:
            loss: An average loss value masked by the length.
        """

        #loss = compute_xe_loss(output.cpu(), target.cpu(), output_lengths.cpu())
        loss = compute_mse_loss(output.cpu(), target.cpu(), output_lengths.cpu())
        #loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        if batch_idx % 10 == 0:
            print(target[0,:40])
            print(output[0,:40])

            pass



# ---------------------
