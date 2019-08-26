import csv
import argparse

from os import listdir
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--dataset-split', default='train', choices=['train', 'dev'])
    args = parser.parse_args()

    data = dict()

    for file_name in listdir(join(args.data_dir, '%s-labels' % args.dataset_split)):
        file_number = int(''.join(list(filter(str.isdigit, file_name))))

        with open(join(args.data_dir, '%s-labels' % args.dataset_split, file_name), 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if row[2] == 'non-propaganda':
                    data[(file_number, int(row[1]))] = [0]
                elif row[2] == 'propaganda':
                    data[(file_number, int(row[1]))] = [1]
                else:
                    data[(file_number, int(row[1]))] = [-1]

    for file_name in listdir(join(args.data_dir, '%s-articles' % args.dataset_split)):
        file_number = int(''.join(list(filter(str.isdigit, file_name))))

        with open(join(args.data_dir, '%s-articles' % args.dataset_split, file_name), 'r') as file:
            for i, line in enumerate(file):
                data[(file_number, i + 1)].insert(0, line.strip())

    with open(join(args.data_dir, '%s-data.tsv' % args.dataset_split), 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['file_number', 'line_number', 'text', 'label'])

        for file_number, line_number in data:
            writer.writerow([file_number, line_number] + data[(file_number, line_number)])
