import argparse
import sys
import nltk
import re
from pathlib import Path
import spacy


"""
Preprocess document to ensure it has 1 sentence per line, all words in lower case, and
filter out sentences that are too short (< 3 tokens long)
"""

cl_parser = argparse.ArgumentParser(description='Preprocess for Style Transfer')
cl_parser.add_argument('-i', '--input_file', default='interviews_preprocessed_split-by-sentence.txt', help='Input filename')
cl_parser.add_argument('-l', '--lower', type=bool, default=True, help='lower case or not')
cl_parser.add_argument('-d', '--data_dir', type=str, default='../data/interviews', help='Name of data directory')
cl_parser.add_argument('-t', '--threshold', type=int, default=3, help='min number of tokens in a sentence')
args = cl_parser.parse_args()

# in file and out file should be in the same directory 'data'
input_filename = Path(args.input_file).name
lower_case = args.lower
o_filename = str(input_filename) + 'style-preprocessed'


out_filename = Path(__file__).resolve().parent / args.data_dir / o_filename
file_name = Path(__file__).resolve().parent / args.data_dir / input_filename

punctuations = '!"#$%&()*+/<=>?@[\\]^_`{|}~'
additional_punct = '\'´´``„”“’‘'  # to account for foreign quotation marks
stopwords_list = nltk.corpus.stopwords.words('english')
recover_list = {'wa': 'was', 'ha': 'has'}

with open(file_name, mode='r', encoding='utf-8') as f, \
        open(out_filename, mode='w', encoding='utf-8') as out:
    for line in f:
        a_line = [l.lower() for l in line.split() if line]
        if len(a_line) > args.threshold:
            a_line = ' '.join(a_line)
            print(a_line, file=out)
