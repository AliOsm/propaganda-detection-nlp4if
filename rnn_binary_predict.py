import os
import csv
import random
random.seed(961)
import argparse

import numpy as np
import pickle as pkl
import tensorflow as tf

from os.path import join
from tqdm import tqdm
from keras.models import load_model

from helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--model-path', default='checkpoints/rnn-binary-epoch04.h5')
    parser.add_argument('--prediction-type', default='binary', choices=['binary', 'probability'])
    args = parser.parse_args()

    # shut up tensorflow and keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    with open(join(args.data_dir, 'dev-data-from-bert.pkl'), 'rb') as file:
        dev_data = pkl.load(file)

    model = load_model(
        args.model_path,
        custom_objects={
            'f1': f1
        }
    )

    with open(join(args.data_dir, 'rnn-binary-%s.txt' % args.prediction_type), 'w') as file:
        writer = csv.writer(file, delimiter='\t')

        for example in tqdm(dev_data):
            embeddings = [item.numpy() for item in dev_data[example][0]]
            prediction = model.predict(np.array([embeddings])).squeeze()

            if args.prediction_type == 'binary':
                if len(embeddings) == 2 or prediction < 0.25:
                    writer.writerow([example[0], example[1], 'non-propaganda'])
                else:
                    writer.writerow([example[0], example[1], 'propaganda'])
            else:
                if len(embeddings) == 2:
                    writer.writerow([example[0], example[1], 0.0])
                else:
                    writer.writerow([example[0], example[1], prediction])
