import argparse
import sys
import nltk
import re
from pathlib import Path
import spacy


punctuations = '!"#$%&()*+/<=>?@[\\]^_`{|}~'
additional_punct = '\'´´``„”“’‘'  # to account for foreign quotation marks
stopwords_list = nltk.corpus.stopwords.words('english')
recover_list = {'wa': 'was', 'ha': 'has'}
word_lemmatizer = nltk.WordNetLemmatizer()

# spacy dependency sent tokenizer and dependency parser
nlp = spacy.load('en_core_web_sm')


def cla_parser():
    """
    Parse CL arguments

    :return: parsed args
    """
    parser = argparse.ArgumentParser(description='Preprocess Interiew Texts')
    parser.add_argument('-i', '--input_file', default='interview_cassina.txt', help='Input filename')
    parser.add_argument('-l', '--lemmatize', type=bool, default=False, help='Lemmatize or not')
    parser.add_argument('-d', '--data_dir', type=str, default='../data/interviews', help='Name of data directory')

    return parser.parse_args()


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
    parsed_doc = nlp(doc)
    sentences = []

    for sent in parsed_doc.sents:
        print(sent, file=dest)
        sentences.append(sent)

    return sentences


def strip_line(string, lemmatized=False):
    stripped = string.translate(str.maketrans("", "", punctuations))
    # no_stop_w_str = ' '.join([w for w in stripped.split() if w not in stopwords_list])
    stripped_line = stripped.translate(str.maketrans("", "", additional_punct))
    stripped_line = re.sub(r'((?<=[0-9])(-|–)(?=[0-9]))|((?<=\s|\t)(-|–)(?=\s|\t))|((?<=[0-9])(-|–))|((-|–)(?=[0-9]))|((-|–)(?=\s|\t))',
                           ' ', stripped_line)  # only strip hyphen when not in between str (inc foreign hyphens)
    if lemmatized:
        stripped_line = ' '.join([lemmatize(w) for w in stripped_line.split()])

    return stripped_line


def parse_qa(file_name, out_filename, sent_out, lemmatized=False):
    with open(file_name, mode='r', encoding='utf-8') as f, \
            open(out_filename, mode='w', encoding='utf-8') as out, \
            open(sent_out, mode='w', encoding='utf-8') as s_out:

        questions = []
        answers = []
        answers_sentences = []
        for line in f:
            line = line.strip()
            if line.startswith('Q:'):
                line = line.replace('Q:', '')
                questions.append(strip_line(line, lemmatized))
                continue

            stripped_line = strip_line(line, lemmatized)
            print(stripped_line, file=out)
            answers.append(stripped_line)
            answers_sentences.append(segment_sent(stripped_line, dest=s_out))

    return questions, answers, answers_sentences


def main():
    args = cla_parser()
    # in file and out file should be in the same directory 'data'
    input_filename = Path(args.input_file).name
    lemmatized = args.lemmatize

    o_filename = str(input_filename) + '_preprocesssed'
    sent_outfile = str(input_filename) + '_preprocessed_split-by-sentence'

    out_filename = Path(__file__).resolve().parent / args.data_dir / o_filename
    sent_out = Path(__file__).resolve().parent / args.data_dir / sent_outfile
    file_name = Path(__file__).resolve().parent / args.data_dir / input_filename

    print(args)

    q, a, s = parse_qa(file_name, out_filename, sent_out)


if __name__ == '__main__':
    main()
