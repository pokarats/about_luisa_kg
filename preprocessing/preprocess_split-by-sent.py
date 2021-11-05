import argparse
import sys
import nltk
import re
from pathlib import Path
import spacy


"""
Preprocessed manually optimized txt file into 1 document per line and 1 sentence per line.
Casing remains unchanged.
Can be optionally lemmatized.
"""

cl_parser = argparse.ArgumentParser(description='Preprocess Texts')
cl_parser.add_argument('-i', '--input_file', default='articles/articles_optimized.txt', help='Input filename')
cl_parser.add_argument('-l', '--lemmatize', type=bool, default=False, help='Lemmatize or not')
cl_parser.add_argument('-d', '--data_dir', type=str, default='../data', help='Name of data directory')

args = cl_parser.parse_args()
# in file and out file should be in the same directory 'data'
input_filename = Path(args.input_file).name
lemmatized = args.lemmatize
o_filename = str(input_filename) + '_preprocessed'
sent_outfile = str(input_filename) + '_preprocessed_split-by-sentence'


out_filename = Path(__file__).resolve().parent / args.data_dir / o_filename
sent_out = Path(__file__).resolve().parent / args.data_dir / sent_outfile
file_name = Path(__file__).resolve().parent / args.data_dir / input_filename

punctuations = '!"#$%&()*+/<=>?@[\\]^_`{|}~'
additional_punct = '\'´´``„”“’‘'  # to account for foreign quotation marks
stopwords_list = nltk.corpus.stopwords.words('english')
recover_list = {'wa': 'was', 'ha': 'has'}
word_lemmatizer = nltk.WordNetLemmatizer()

# spacy dependency sent tokenizer and dependency parser
parser = spacy.load('en_core_web_sm')


def lemmatize(word):
    """

    :param word: a word token
    :return: lemmatized word
    """
    w = word_lemmatizer.lemmatize(word.lower())
    if w in recover_list:
        return recover_list[w]
    return w


def segment_sent(doc, dest):
    """

    :param dest: output filestream
    :param doc: str multiple sentences
    :return:
    """
    parsed_doc = parser(doc)
    for sent in parsed_doc.sents:
        print(sent, file=dest)


with open(file_name, mode='r', encoding='utf-8') as f, \
        open(out_filename, mode='w', encoding='utf-8') as out, \
        open(sent_out, mode='w', encoding='utf-8') as s_out:
    first_line = f.readline()  # ignore first line
    print(first_line, file=out, end='')
    for line in f:
        stripped = line.translate(str.maketrans("", "", punctuations))
        # no_stop_w_str = ' '.join([w for w in stripped.split() if w not in stopwords_list])
        stripped_line = stripped.translate(str.maketrans("", "", additional_punct))
        stripped_line = re.sub(r'((?<=[0-9])(-|–)(?=[0-9]))|((?<=\s|\t)(-|–)(?=\s|\t))|((?<=[0-9])(-|–))|((-|–)(?=[0-9]))|((-|–)(?=\s|\t))',
                               ' ', stripped_line)  # only strip hyphen when not in between str (inc foreign hyphens)
        if lemmatized:
            stripped_line = ' '.join([lemmatize(w) for w in stripped_line.split()])

        print(stripped_line, file=out, end='')
        segment_sent(stripped_line, dest=s_out)

print(f'Finished pre-processing: {file_name}\n'
      f'Pre-processed version saved as: {out_filename}\n'
      f'Sentencized version saved as: {sent_out}\n'
      f'Lemmatized: {lemmatized}')


