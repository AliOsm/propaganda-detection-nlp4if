import os
import csv
import random
random.seed(961)
import argparse

import numpy as np
import pickle as pkl
import tensorflow as tf
import tensorflow_hub as hub

from os.path import join
from tqdm import tqdm
from keras.models import load_model

from helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--dataset-split', default='dev', choices=['dev', 'test'])
    parser.add_argument('--model-path', default='checkpoints/universal-sentence-encoder-binary-epoch05.h5')
    parser.add_argument('--prediction-type', default='binary', choices=['binary', 'probability'])
    args = parser.parse_args()

    # shut up tensorflow and keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    dev_data = list()
    with open(join(args.data_dir, '%s-data.tsv' % args.dataset_split), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)

        for row in reader:
            dev_data.append(row)

    model = load_model(
        args.model_path,
        custom_objects={
            'f1': f1
        }
    )

    embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/3')

    dev_embeddings = list()
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        for i in tqdm(range(0, len(dev_data), 512)):
            sentences = [example[2] for example in dev_data[i:i + 512]]
            dev_embeddings.extend(session.run(embed(sentences)))

    with open(join(args.data_dir, '%s-universal-sentence-encoder-binary-%s.txt' % (args.dataset_split, args.prediction_type)), 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        
        for example, embeddings in tqdm(zip(dev_data, dev_embeddings)):
            prediction = model.predict(np.array([embeddings])).squeeze()

            if args.prediction_type == 'binary':
                if len(example[2].strip()) == 0 or prediction < 0.25:
                    writer.writerow([example[0], example[1], 'non-propaganda'])
                else:
                    writer.writerow([example[0], example[1], 'propaganda'])
            else:
                if len(example[2].strip()) == 0:
                    writer.writerow([example[0], example[1], 0.0])
                else:
                    writer.writerow([example[0], example[1], prediction])
