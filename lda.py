import argparse
import pickle
import random
from collections import Counter
from functools import wraps
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import logging


"""
Implementation of LDA using Gibbs sampling according to:
Griffiths and Steyvers, 2004
Darling, 2011
and
adaptaions from R code according to https://ethen8181.github.io/machine-learning/clustering_old/topic_model/LDA.html
"""


def timer(func):
    """

    :param func: function whose execution needs to be timed
    :return: Print clock runtime of decorated function
    """

    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        f_result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.getLogger(__name__).info(f"Executed {func.__name__!r} in {end_time - start_time:.4f} secs")
        return f_result

    return wrapper_timer


class Tokenize:
    """
    Read file and tokenize sentences
    1st line in file is the number of documents
    each line is all the words in the document

    """

    def __init__(self, filename):
        self._filename = filename
        self.num_doc = 0
        self.doc_lens = self._doc_lens()
        self.doc_word_counter = [Counter() for _ in range(self.num_doc)]  # word freq dic by doc id
        self.word_counts = self._count()
        self.id_to_word = {}
        self.word_to_id = {}

    def tokenize(self):

        with open(self._filename, 'r') as f:
            # first line is the number of document; only read from line 2 onward
            self.num_doc = int(f.readline())
            return [line.split() for line in f]

    def _count(self):
        word_counts = Counter()

        for doc_id, doc in enumerate(self.tokenize()):
            word_counts.update(doc)
            self.doc_word_counter[doc_id].update(doc)

        return word_counts

    def _doc_lens(self):
        """
        dictionary mapping document ID to document length (num words in the doc)
        :return:
        """
        doc_lens = {}

        for indx, doc in enumerate(self.tokenize()):
            doc_lens[indx] = len(doc)

        return doc_lens

    def _make_unique_indices(self):
        """
        update id_to_word and word_to_id dictionaries class variables mapping each word type to a unique id
        and vice versa

        id starts at 0

        :return:
        """
        for indx, key in enumerate(self.word_counts.keys()):
            self.id_to_word[indx] = key
            self.word_to_id[key] = indx


class LDA:

    def __init__(self, tokenize_class_obj, iterations, alpha, beta, num_topics):
        self.data = tokenize_class_obj
        self.num_doc = self.data.num_doc
        self.vocab_size = len(self.data.word_counts)
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics

        self.doc_word_topic = None  # nested List[{w:topic}]: list of docs, inner dict topic at each word type
        self.doc_topic_counter = [Counter() for _ in range(self.num_doc)]  # each topic is assigned to each document.
        self.topic_word_counter = [Counter() for _ in range(self.num_topics)]  # each word is assigned to each topic.
        self.topic_counts = [0 for _ in range(self.num_topics)]  # The total number of words assigned to each topic.

    def initialize(self):

        # Step 1:
        # initialize each word type in each doc with a topic assignment (topics start from 0 to num_topics)
        # randomly assign topics to words in each doc
        # self.doc_word_topic = List[{word:topic}, {word:topic}]
        # self.doc_word_topic[doc_id] = {each word type in doc vocab}: topic assignment}
        # why would the same word be assigned a different topic in the same document?
        self.doc_word_topic = [{w: random.randrange(self.num_topics) for w in doc} for doc in self.data.tokenize()]

        for doc_id, doc in enumerate(self.data.tokenize()):
            num_words_in_doc = self.data.doc_lens[doc_id]

            # Step 2:
            # increment count for word in each topic in each doc
            # increment count of each word in each topic
            # increment count of total words in each topic
            for word in doc:
                topic = self.doc_word_topic[doc_id][word]
                self.doc_topic_counter[doc_id][topic] += 1
                self.topic_word_counter[topic][word] += 1
                self.topic_counts[topic] += 1

    def _p_w_given_topic(self, topic, word):
        """
        [(C_wt + beta) / (sum_over_w(C_wt) + vocab_size * beta)]

        :param topic:
        :param word:
        :return:
        """
        left_denom = self.vocab_size * self.beta + self.topic_counts[topic]
        return (self.beta + self.topic_word_counter[topic][word]) / left_denom

    def _p_topic_given_doc(self, doc_id, topic):
        """
        [(C_dt + alpha) / (sum_over_t(C_dt) + num_topics * alpha)]

        :param doc_id:
        :param topic:
        :return:
        """
        right_denom = self.alpha * self.num_topics + self.data.doc_lens[doc_id]  # token count in doc_id
        return (self.alpha + self.doc_topic_counter[doc_id][topic]) / right_denom

    @timer
    def gibbs_sampling(self):
        """
        Perform Gibbs Sampling to update doc topic, topic word, and doc word distributions
        :return:
        """

        for _ in tqdm(range(self.iterations)):
            for doc_id in tqdm(range(self.num_doc)):
                for word_type, topic in self.doc_word_topic[doc_id].items():
                    # 1:
                    # remove this word in topic t from the counters
                    # so that it doesn't influence the probability calculation
                    count_this_word_type_in_doc = self.data.doc_word_counter[doc_id][word_type]
                    self.doc_topic_counter[doc_id][topic] -= count_this_word_type_in_doc
                    self.topic_word_counter[topic][word_type] -= count_this_word_type_in_doc
                    self.topic_counts[topic] -= count_this_word_type_in_doc
                    self.data.doc_lens[doc_id] -= count_this_word_type_in_doc

                    # 2:
                    # calculate p_topic_given_word_alpha_beta
                    # according to LDA:
                    # p(topic|w,alpha, beta) = [(C_wt + beta) / (sum_over_w(C_wt) + vocab_size * beta)] *
                    #                          [(C_dt + alpha) / (sum_over_t(C_dt) + num_topics * alpha)]
                    # left_denom = self.vocab_size * self.beta + self.topic_counts[topic]
                    # right_denom = self.alpha * self.num_topics + self.data.doc_lens[doc_id]  # token count in doc_id
                    p_topic_given_word_alpha_beta = np.array([self._p_w_given_topic(topic_i, word_type) *
                                                              self._p_topic_given_doc(doc_id, topic_i)
                                                              for topic_i in range(self.num_topics)])

                    normalized_p_topic_given_word = p_topic_given_word_alpha_beta / p_topic_given_word_alpha_beta.sum()
                    # sampled_topic_i = np.random.choice(self.num_topics, 1, p=normalized_p_topic_given_word)
                    # freq of topics in # experiments == word freq in doc id, take the topic with most frequencies
                    sampled_topic_i = np.random.multinomial(count_this_word_type_in_doc,
                                                            normalized_p_topic_given_word, size=1).argmax()
                    # 3: updating the counts
                    self.doc_word_topic[doc_id][word_type] = sampled_topic_i
                    self.doc_topic_counter[doc_id][sampled_topic_i] += count_this_word_type_in_doc
                    self.topic_word_counter[sampled_topic_i][word_type] += count_this_word_type_in_doc
                    self.topic_counts[sampled_topic_i] += count_this_word_type_in_doc
                    self.data.doc_lens[doc_id] += count_this_word_type_in_doc

    def pickle_model(self, pkl_filename_tw, pkl_filename_dt):
        with open(pkl_filename_tw, mode='wb') as pkl, open(pkl_filename_dt, mode='wb') as pkl2:
            pickle.dump(self.topic_word_counter, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.doc_topic_counter, pkl2, protocol=pickle.HIGHEST_PROTOCOL)

    def print_most_freq_words(self, outfile_filename, num_iterations, num_words, saved_model=False,
                              pkl_filename_tw=None):
        if saved_model:
            logging.getLogger(__name__).info(f'Loading topic_word_counter from saved model...')
            topic_word_counter = pickle.load(open(pkl_filename_tw, mode='rb'))
            assert topic_word_counter == self.topic_word_counter
            self.topic_word_counter = topic_word_counter

        with open(outfile_filename, encoding='utf-8', mode='a+') as out:
            print(f'number of iterations: {num_iterations}', file=out)
            for topic_idx, freq_w_counter in enumerate(self.topic_word_counter):
                print(f'topic: {topic_idx}\tMost frequent words: {freq_w_counter.most_common(num_words)}', file=out)


def main():
    # command line arguments for options
    parser = argparse.ArgumentParser(description='LDA with Gibbs Sampling')

    parser.add_argument('-dd', '--data_dir', type=Path,
                        default=Path(__file__).resolve().parent / 'data',
                        help='Path to data directory')
    parser.add_argument('-f', '--train', default='preprocessed_luisa_text.txt', help='Training file name')
    parser.add_argument('-o', '--output', type=str, default='output.txt', help='Output filename')
    parser.add_argument('-a', '--alpha', type=float, default=0.02,
                        help='alpha hyperparameter for Dirichlet (default=0.02)')
    parser.add_argument('-b', '--beta', type=float, default=0.1,
                        help='beta hyperparameter for Dirichlet (default=0.1)')
    parser.add_argument('-n', '--threshold', type=int, default=1, help='Threshold for how many iterations to run')
    parser.add_argument('-t', '--num_topics', type=int, default=10, help='Number of topics')
    parser.add_argument('-l', '--log', type=str, default='topic_lda.log', help='Logging filename')
    parser.add_argument('-w', '--num_words', type=int, default=5, help='Number of most frequent words to print')
    parser.add_argument('-s', '--saved_model', type=bool, default=False, help='Whether to print results from saved dict')

    p = parser.parse_args()
    path_to_train: Path = p.data_dir / p.train
    path_to_output: Path = p.data_dir / p.output
    path_to_logfile: Path = p.data_dir / p.log

    # set up logging
    program_log = logging.getLogger(__name__)
    filename = str(path_to_logfile)
    logging.basicConfig(filename=filename, format='%(asctime)s %(name)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    program_log.info(f'---------START----------')
    if path_to_train.is_file():
        program_log.info(f'Training from {path_to_train}')
        program_log.info(f'LDA topic modeling output saving to: {p.output}')
        program_log.info(f'LDA Gibbs Sampling with the following parameters:\n'
                         f'alpha: {p.alpha}\tbeta: {p.beta}\n'
                         f'number of topics: {p.num_topics}\titerations: {p.threshold}')
        program_log.info(f'logging outputs saving to: {path_to_logfile}')
    else:
        program_log.error(f'Incorrect path to file!!')
        raise FileNotFoundError('Correct paths to files!!!')

    # pre-processing
    program_log.info(f'pre-processing steps')
    corpus = Tokenize(path_to_train)
    program_log.info(f'corpus has: {corpus.num_doc} documents\n'
                     f'corpus has: {sum(corpus.word_counts.values())} word tokens\n'
                     f'vocab size: {len(corpus.word_counts)}\n')

    # corpus_array = corpus.to_array()
    # print(corpus.word_to_id['plot'], corpus.word_to_id['two'], corpus.word_to_id['teen'], corpus.word_to_id['movies'])
    # print(corpus_array[0, 21])
    # print(corpus_array[1, 3])
    # print(corpus_array[2, :5])
    # print(corpus.word_counts)
    # print(corpus.doc_lens)
    # print(corpus.tokenize()[0])

    # initializing
    program_log.info(f'Initializing...')
    lda = LDA(corpus, iterations=p.threshold, alpha=p.alpha, beta=p.beta, num_topics=p.num_topics)
    lda.initialize()

    # gibbs sampling
    program_log.info(f'Gibbs sampling in progress...')
    lda.gibbs_sampling()

    # most frequent worst per topics
    program_log.info(f'Saving trained distributions to {p.data_dir}')

    if p.saved_model:
        program_log.info(f'Saving trained model dict to {p.data_dir}/*.pkl')
        lda.pickle_model(pkl_filename_tw=p.data_dir / 'c_tw_trained_pickle.pkl',
                         pkl_filename_dt=p.data_dir / 'c_dt_trained_pickle.pkl')

    program_log.info(f'Printing {p.num_words} most frequent words per topic to {path_to_output}')
    lda.print_most_freq_words(outfile_filename=path_to_output, num_iterations=p.threshold, num_words=p.num_words,
                              saved_model=p.saved_model, pkl_filename_tw=p.data_dir / 'c_tw_trained_pickle.pkl')
    program_log.info(f'-----------FINISHED-------------\n')


if __name__ == '__main__':
    main()
