import os
import random
random.seed(961)
import argparse

import numpy as np
import pickle as pkl
import tensorflow as tf

from os.path import join
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import GRU, CuDNNGRU, LSTM, CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

from helpers import *
from rnn_binary_data_generator import DataGenerator

def build_model(embeddings_size):
    tokens_embeddings_input = Input(shape=(None, embeddings_size,))

    lstm = Bidirectional(
        LSTM(
            units=128,
            dropout=args.dropout_rate,
            return_sequences=True,
            kernel_initializer='he_normal'
        )
    )(tokens_embeddings_input)

    lstm = Bidirectional(
        LSTM(
            units=128,
            dropout=args.dropout_rate,
            kernel_initializer='he_normal'
        )
    )(lstm)

    dense = Dropout(args.dropout_rate)(
        Dense(units=256, activation='relu', kernel_initializer='he_normal')(lstm)
    )
    dense = Dropout(args.dropout_rate)(
        Dense(units=128, activation='relu', kernel_initializer='he_normal')(dense)
    )

    output = Dense(units=1, activation='sigmoid', kernel_initializer='he_normal')(dense)

    model = Model(tokens_embeddings_input, output)

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', f1])
    model.summary()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--dropout-rate', default=0.2, type=float)
    parser.add_argument('--dev-split', default=0, type=float)
    args = parser.parse_args()

    # shut up tensorflow and keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    with open(join(args.data_dir, 'train-data-from-bert.pkl'), 'rb') as file:
        train_data = pkl.load(file)

    train_embeddings = list()
    train_labels = list()
    for example in train_data:
        if len(train_data[example][0]) == 0: continue
        train_embeddings.append([item.numpy() for item in train_data[example][0]])
        train_labels.append(int(train_data[example][1]))

    train_data = list(zip(train_embeddings, train_labels))
    random.shuffle(train_data)

    if args.dev_split != 0:
        split_point = int(len(train_embeddings) * args.dev_split)
        dev_data = train_data[:split_point]
        train_data = train_data[split_point:]

        train_data = sorted(train_data, key=lambda item: len(item[0]))
        dev_data = sorted(dev_data, key=lambda item: len(item[0]))

        train_embeddings, train_labels = zip(*train_data)
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)

        dev_embeddings, dev_labels = zip(*dev_data)
        dev_embeddings = np.array(dev_embeddings)
        dev_labels = np.array(dev_labels)
    else:
        train_data = sorted(train_data, key=lambda item: len(item[0]))

        train_embeddings, train_labels = zip(*train_data)
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)

    model = build_model(len(train_embeddings[0][0]))

    train_generator = DataGenerator(train_embeddings, train_labels, args.batch_size)

    if args.dev_split != 0:
        dev_generator = DataGenerator(dev_embeddings, dev_labels, args.batch_size)

    checkpoint_cb = ModelCheckpoint(
        filepath='checkpoints/rnn-binary-epoch{epoch:02d}.h5'
    )

    if args.dev_split != 0:
        model.fit_generator(
            generator=train_generator,
            validation_data=dev_generator,
            epochs=args.epochs,
            callbacks=[checkpoint_cb]
        )
    else:
        model.fit_generator(
            generator=train_generator,
            epochs=args.epochs,
            callbacks=[checkpoint_cb]
        )
