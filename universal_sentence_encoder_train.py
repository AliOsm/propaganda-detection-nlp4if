import os
import csv
import random
random.seed(961)
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from os.path import join
from tqdm import tqdm
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import GRU, CuDNNGRU, LSTM, CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

from helpers import *

def build_model(embeddings_size):
    train_embeddings_input = Input(shape=(embeddings_size,))

    dense = Dropout(args.dropout_rate)(
        Dense(units=256, activation='relu', kernel_initializer='he_normal')(train_embeddings_input)
    )
    dense = Dropout(args.dropout_rate)(
        Dense(units=128, activation='relu', kernel_initializer='he_normal')(dense)
    )

    output = Dense(units=1, activation='sigmoid', kernel_initializer='he_normal')(dense)

    model = Model(train_embeddings_input, output)

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', f1])
    model.summary()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--dropout-rate', default=0.2, type=float)
    parser.add_argument('--dev-split', default=0, type=float)
    args = parser.parse_args()

    # shut up tensorflow and keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    sentences = list()
    train_labels = list()
    with open(join(args.data_dir, 'train-data.tsv'), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)

        for row in reader:
            if len(row[2].strip()) == 0: continue
            sentences.append(row[2])
            train_labels.append(row[3])

    embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/3')

    train_embeddings = list()
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        for i in tqdm(range(0, len(sentences), args.batch_size)):
            train_embeddings.extend(session.run(embed(sentences[i:i + args.batch_size])))

    train_data = list(zip(train_embeddings, train_labels))
    random.shuffle(train_data)

    if args.dev_split != 0:
        split_point = int(len(train_embeddings) * args.dev_split)
        dev_data = train_data[:split_point]
        train_data = train_data[split_point:]

        train_embeddings, train_labels = zip(*train_data)
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)

        dev_embeddings, dev_labels = zip(*dev_data)
        dev_embeddings = np.array(dev_embeddings)
        dev_labels = np.array(dev_labels)
    else:
        train_embeddings, train_labels = zip(*train_data)
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)

    model = build_model(len(train_embeddings[0]))

    checkpoint_cb = ModelCheckpoint(
        filepath='checkpoints/universal-sentence-encoder-epoch{epoch:02d}.h5'
    )

    if args.dev_split != 0:
        model.fit(
            x=train_embeddings,
            y=train_labels,
            validation_data=(dev_embeddings, dev_labels),
            epochs=args.epochs,
            callbacks=[checkpoint_cb]
        )
    else:
        model.fit(
            x=train_embeddings,
            y=train_labels,
            epochs=args.epochs,
            callbacks=[checkpoint_cb]
        )
