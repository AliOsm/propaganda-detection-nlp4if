import csv
import argparse

from os import listdir
from os.path import join

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='data_dir')
	parser.add_argument('--dataset-split', default='train', choices=['train', 'dev', 'test'])
	args = parser.parse_args()

	number_of_articles = 0
	number_of_lines = 0
	number_of_tokens = 0
	unique_tokens = set()

	for file_name in listdir(join(args.data_dir, '%s-articles' % args.dataset_split)):
		number_of_articles += 1

		with open(join(args.data_dir, '%s-articles' % args.dataset_split, file_name), 'r') as file:
			lines = file.readlines()

			number_of_lines += len(lines)
			for line in lines:
				number_of_tokens += len(line.split())
				unique_tokens.update(set(line.split()))

	number_of_unqiue_tokens = len(unique_tokens)
	avg_lines_per_article = round(number_of_lines / number_of_articles, 2)
	avg_tokens_per_article = round(number_of_tokens / number_of_articles, 2)
	avg_tokens_per_line = round(number_of_tokens / number_of_lines, 2)

	with open(join(args.data_dir, '%s-statistics.csv' % args.dataset_split), 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['number_of_articles', number_of_articles])
		writer.writerow(['number_of_lines', number_of_lines])
		writer.writerow(['number_of_tokens', number_of_tokens])
		writer.writerow(['number_of_unqiue_tokens', number_of_unqiue_tokens])
		writer.writerow(['avg_lines_per_article', avg_lines_per_article])
		writer.writerow(['avg_tokens_per_article', avg_tokens_per_article])
		writer.writerow(['avg_tokens_per_line', avg_tokens_per_line])
