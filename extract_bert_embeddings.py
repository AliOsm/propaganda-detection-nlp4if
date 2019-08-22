import csv
import argparse

import torch
import numpy as np
import pickle as pkl

from os.path import join
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--dataset-split', default='train', choices=['train', 'dev'])
    args = parser.parse_args()

    data = list()
    with open(join(args.data_dir, '%s-data.tsv' % args.dataset_split), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)

        for row in reader:
            data.append(row)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    sentences_embeddings = dict()
    for row in tqdm(data):
        marked_text = '[CLS] ' + row[2] + ' [SEP]'
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        token_embeddings = []
        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            for layer_i in range(len(encoded_layers)):
                vec = encoded_layers[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)

        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]

        sentences_embeddings[(row[0], row[1])] = (summed_last_4_layers, row[-1])

    with open(join(args.data_dir, '%s-data-from-bert.pkl' % args.dataset_split), 'wb') as file:
        pkl.dump(sentences_embeddings, file)
