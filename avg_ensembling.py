import csv
import argparse

from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--dataset-split', default='dev', choices=['dev', 'test'])
    parser.add_argument('--predictions-files', nargs='+', default=['%s-rnn-binary-bert-uncased-probability.txt', '%s-rnn-binary-bert-cased-probability.txt', '%s-universal-sentence-encoder-binary-probability.txt'])
    args = parser.parse_args()

    for i in range(len(args.predictions_files)):
        args.predictions_files[i] = args.predictions_files[i] % args.dataset_split

    predictions = dict()

    for prediction_file in args.predictions_files:
        with open(join(args.data_dir, prediction_file), 'r') as file:
            reader = csv.reader(file, delimiter='\t')

            for row in reader:
                try:
                    predictions[(row[0], row[1])] += float(row[-1])
                except:
                    predictions[(row[0], row[1])] = float(row[-1])

    for prediction in predictions:
        predictions[prediction] /= len(args.predictions_files)

    with open(join(args.data_dir, '%s-avg-ensembling.txt' % args.dataset_split), 'w') as file:
        writer = csv.writer(file, delimiter='\t')

        for prediction in predictions:
            if predictions[prediction] < 0.25:
                writer.writerow([prediction[0], prediction[1], 'non-propaganda'])
            else:
                writer.writerow([prediction[0], prediction[1], 'propaganda'])
