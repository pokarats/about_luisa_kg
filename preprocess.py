import sys
import nltk
import re

"""
Pre-process manually edited text file (e.g. luisa_text.txt, luisa_text_optimized.txt) for LDA, getting ride of
punctuations and optionally lemmatize and remove stopwords (English)

output: preprocessed_<input file name>.txt
"""

file_name = sys.argv[1]
lemmatized = True if sys.argv[2] == 'True' else False
out_filename = 'preprocessed_' + file_name

punctuations = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'
additional_punct = '\'´´``„”“’‘'  # to account for foreign quotation marks
stopwords_list = nltk.corpus.stopwords.words('english')
recover_list = {'wa': 'was', 'ha': 'has'}
word_lemmatizer = nltk.WordNetLemmatizer()


def lemmatize(word):
    """

    :param word: a word token
    :return: lemmatized word
    """
    w = word_lemmatizer.lemmatize(word.lower())
    if w in recover_list:
        return recover_list[w]
    return w


with open(file_name, mode='r', encoding='utf-8') as f, \
        open(out_filename, mode='w', encoding='utf-8') as out:
    first_line = f.readline()  # ignore first line
    print(first_line, file=out, end='')
    for line in f:
        stripped = line.lower().translate(str.maketrans("", "", punctuations))
        no_stop_w_str = ' '.join([w for w in stripped.split() if w not in stopwords_list])
        stripped_line = no_stop_w_str.translate(str.maketrans("", "", additional_punct))
        stripped_line = re.sub(r'((?<=[0-9])(-|–)(?=[0-9]))|((?<=\s|\t)(-|–)(?=\s|\t))|((?<=[0-9])(-|–))|((-|–)(?=[0-9]))|((-|–)(?=\s|\t))',
                               ' ', stripped_line)  # only strip hyphen when not in between str (inc foreign hyphens)
        if lemmatized:
            stripped_line = ' '.join([lemmatize(w) for w in stripped_line.split()])
        print(stripped_line, file=out)

print(f'Finished pre-processing: {file_name}\n'
      f'Pre-processed version saved as: {out_filename}\n'
      f'Lemmatized: {lemmatized}')


