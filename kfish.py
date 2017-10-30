# Keras playground for RNN seq prediction (DID NOT WORK)

from collections import OrderedDict
import itertools
import os
from fishdataset import SeqDataset, SubsetSampler,collate_seqs
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input, Concatenate, GRU, Bidirectional, LSTM, Masking
from keras.layers.core import Dense, Flatten, Lambda, Dropout, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import copy
from keras import backend as K
import random

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--test', action='store_true', help='Test model on test_crops.csv')
parser.add_argument('-lm', '--load-model', type=str, help='Load model from file')
parser.add_argument('-bs', '--batch-size', type=int, default=1, help='Batch size')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='Learning rate')
parser.add_argument('-s', '--suffix',  type=str, default=None, help='Suffix to store checkpoints')

args = parser.parse_args()

def get_model(max_length, n_features):
    boat_seq   = Input(shape=(max_length, n_features ))
    l = Masking(mask_value=-1)(boat_seq)
    l = LSTM(64,return_sequences=True, activation = 'tanh')(l)
#    l = LSTM(64,return_sequences=True, activation = 'relu')(l)
    #l = LSTM(64,return_sequences=True, activation = 'relu')(l)
    l = LSTM(1, activation='sigmoid'  ,return_sequences=True,)(l)

    classification = l

    outputs = [classification]

    model = Model(inputs=[boat_seq], outputs=outputs)
    return model

def gen(dataset, items, batch_size, training=True):

    max_length = dataset.max_length
    n_features = dataset.n_features

    X_seqs = np.zeros((batch_size, max_length, n_features), dtype=np.float32)
    Y_seqs = np.zeros((batch_size, max_length, 1),          dtype=np.float32)
    X_seqs[...] = -1
    Y_seqs[...] = -1
    i = 0

    while True:
        if training:
            random.shuffle(items)

        for item in items:
            X_seq, Y_seq = dataset[item]
            X_seq = X_seq[:, 1:]
            Y_seq = Y_seq[:, 9:10]
            #print(np.unique(Y_seq))
            Y_seq = np.remainder(Y_seq, 2)
            #print(np.unique(Y_seq))
            #print(X_seq, Y_seq)
            #print(X_seq.shape[0])
            X_seqs[i, :min(X_seq.shape[0], max_length),...] = X_seq[:max_length, ...]
            Y_seqs[i, :min(Y_seq.shape[0], max_length),...] = Y_seq[:max_length, ...]

            i += 1

            if i == batch_size:
                yield X_seqs, Y_seqs
                #print(Y_seqs[0])

                X_seqs[...] = -1
                Y_seqs[...] = -1
                i = 0




TRAIN_X_CSV = 'train_crops_X.csv'
TRAIN_Y_CSV = 'train_crops_Y.csv'

dataset = SeqDataset(
        X_csv_file=TRAIN_X_CSV,
        Y_csv_file=TRAIN_Y_CSV,
        )

dataset.max_length = 100
dataset.n_features = 8
model = get_model(dataset.max_length, dataset.n_features)
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=args.learning_rate))
idx_train, idx_valid = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)

model.fit_generator(
        generator        = gen(dataset, idx_train, args.batch_size),
        steps_per_epoch  = len(idx_train) // args.batch_size,
        validation_data  = gen(dataset, idx_valid, args.batch_size ,training=False),
        validation_steps = len(idx_valid) // args.batch_size,
        epochs =100,
        #callbacks = [save_checkpoint, reduce_lr],
        )
