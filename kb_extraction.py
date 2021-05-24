import json
import re
import pandas as pd
import sys
from pathlib import Path
from collections import namedtuple
from itertools import product
import bs4
import requests
import spacy
from spacy import displacy

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx
from networkx.readwrite import json_graph

import matplotlib.pyplot as plt
from tqdm import tqdm

parent_dir = Path(__file__).resolve().parent
parser = spacy.load('en_core_web_sm')

# in file should be in the directory 'data'
try:
    input_filename = Path(sys.argv[1]).name
    lemmatized = True if sys.argv[2] == 'True' else False
except IndexError:
    input_filename = Path("data/case_sentluisa_text_optimized.txt").name

# output/extracted entities and relations will be in 'kb' directory
out_kb_dir = parent_dir / 'kb'
in_filename = parent_dir / 'data' / input_filename


def get_entities(sent):
    """
    Extract 2 entities from each sentence. Function is adapted from analyticsvidhya.com
    Since there may be multiple subj and obj entities in a long sentence, return a list entity pairs

    :param sent: str sentence
    :return: list of subject entities and list of object entities
    """


    # starting chunk
    ent1 = []
    ent2 = []

    prv_tok_dep = ""  # dep tag of prev token in the sent
    prv_tok_text = ""  # prev token in the sent

    prefix = ""
    modifier = ""

    for tok in parser(sent):
        """
        chunk 2: compound and modifiers
        if token is a punct, move on to the next token
        """
        if tok.dep_ != "punct":
            # check for compound word
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    # add current token to previous if previous is also compound
                    prefix = " ".join([prv_tok_text, tok.text])

            # check if token is a modifier
            if tok.dep_.endswith("mod"):
                modifier = tok.text
                # if previous token also a compound, add current token to it
                if prv_tok_dep == "compound" or prv_tok_dep.endswith("mod"):
                    modifier = " ".join([prv_tok_text, tok.text])

            # check if token is an attribute following a modifier
            if tok.dep_.endswith("attr"):
                # if previous token also a compound, add current token to it
                if prv_tok_dep == "compound" or prv_tok_dep.endswith("mod"):
                    modifier = " ".join([prv_tok_text, tok.text])

            # chunk 3: subject of the sentence
            if tok.dep_.endswith("subj"):
                ent1.append(" ".join([modifier, prefix, tok.text]))
                # reset starting var
                prefix = ""
                modifier = ""
                prv_tok_text = ""
                prv_tok_dep = ""

            # chuck 4: object of the sentence
            if tok.dep_.endswith("obj"):
                ent2.append(" ".join([modifier, prefix, tok.text]))

            # update prev with current once we've found subj and obj
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    # to account for multiple entities, make possible product of ent1 (subj) and ent2 (obj) pairs, limited by
    # ent1
    # print(f'ent1: {ent1}\n'
    #      f'ent2: {ent2}\n')

    return ent1, ent2


def get_relations(sentence):
    """

    :param sentence: str sentence
    :return: list of str spans of each predicate
    """

    parsed_sent = parser(sentence)

    # Matcher class obj
    matcher = Matcher(parser.vocab)

    # define pattern for matching
    pattern1 = [{'DEP': 'ROOT'}, {'DEP': 'prep', 'OP': "?"}, {'DEP': 'agent', 'OP': "?"},{'POS': 'ADJ', 'OP': "?"}]
    pattern2 = [{'DEP': 'conj'}, {'DEP': 'det', 'OP': "?"}]

    matcher.add("matching_action", None, pattern1)
    matcher.add("matching_conj", None, pattern2)

    matches = matcher(parsed_sent)
    spans = []
    for match_id, start, end in matches:
        if end - start > 1:
            spans.append(parsed_sent[start:end].text)

    #matched_idx = len(matches) - 1

    #span = parsed_sent[matches[matched_idx][1]:matches[matched_idx][2]]

    return spans


def make_ent_rel_pairs(sentence):
    """

    :param sentence: str sentence
    :return: list of Entpairs where each Entpair is a namedtuple(subj, obj, relation)
    """

    ent_pairs = []
    Entpair = namedtuple("Entpair", "subj obj rel", defaults=["tbd"])
    subjects, objects = get_entities(sentence)
    relations = get_relations(sentence)

    #print(f'subjects: {subjects}, objects: {objects}, relations: {relations}')

    # maximize number of subj-obj-relation pairs, this is a native approach and may create subj-obj-rel entities
    # not found in original corpus
    for subj, obj, rel in product(subjects, objects, relations):
        ent_pairs.append(Entpair(subj.strip(), obj.strip(), rel.strip()))

    return ent_pairs


def text_file_to_csv(infile):
    """

    :param infile: text file where each line is a str sent
    :return:
    """
    sentences = []
    with open(infile, mode='r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip())

    df = pd.DataFrame(list(filter(None, sentences)))  # get rid of empty str in sentences
    df.columns = ['sentence']
    outfile_name = str(infile.stem) + ".csv"
    outfile = parent_dir / 'data' / outfile_name
    df.to_csv(outfile)

    return df


def get_all_entity_pairs(sentences):
    """

    :param sentences: iterable of str sentences
    :return: list of entity pairs, each pair is an Entpair namedtuple with subj, obj, and rel
    """

    ent_pairs = []
    for sent in tqdm(sentences):
        sent_ent = make_ent_rel_pairs(sent)
        if len(sent_ent) >= 1:
            ent_pairs.extend(sent_ent)

    return ent_pairs


# some sample sentences to test kb extraction
try:
    csv_data_file = 'case_sentluisa_text_optimized.csv'
    all_sentences = pd.read_csv(parent_dir / 'data' / csv_data_file)['sentence']
except FileNotFoundError:
    all_sentences = text_file_to_csv(in_filename)['sentence']


# uncomment this next block for viewing a small sample
"""
sample_sentences = all_sentences.sample(5)
print(sample_sentences)

sample_entpairs = get_all_entity_pairs(sample_sentences)

sources = [ent.subj for ent in sample_entpairs]
targets = [ent.obj for ent in sample_entpairs]
edges = [ent.rel for ent in sample_entpairs]

print(pd.Series(edges).value_counts()[:20])

kg_from_df = pd.DataFrame({'source': sources, 'target': targets, 'edge': edges})
G = nx.from_pandas_edgelist(kg_from_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
plt.show(block=True)
"""

# all sentences in corpus
all_ent_pairs = get_all_entity_pairs(all_sentences)
sources = [ent.subj for ent in all_ent_pairs]
targets = [ent.obj for ent in all_ent_pairs]
edges = [ent.rel for ent in all_ent_pairs]

kg_from_df = pd.DataFrame({'source': sources, 'target': targets, 'edge': edges})

top_50 = pd.Series(edges).value_counts()[:50]
print(top_50)
top10_edge_keys = [k for k, v in top_50[:10].items()]
print(top10_edge_keys)


# graph for each of the top 10 relations
for e_key in top10_edge_keys:
    G = nx.from_pandas_edgelist(kg_from_df[kg_from_df['edge'] == e_key], "source", "target", edge_attr=True,
                                create_using=nx.MultiDiGraph())
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.75)  # k determines distance between nodes
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
    save_file_name = "_".join(e_key.split())
    plt.savefig(out_kb_dir / (save_file_name + ".png"), bbox_inches="tight")

    data1 = json_graph.node_link_data(G)

    with open(out_kb_dir / (save_file_name + ".json"), mode='w', encoding='utf-8') as out:
        json.dump(data1, out, ensure_ascii=False, indent=2)

# graph for all subj-obj-relation pairs
G = nx.from_pandas_edgelist(kg_from_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
data_all = json_graph.node_link_data(G)
save_file_name = "KG_all_" + str(in_filename.stem)
with open(out_kb_dir / (save_file_name + ".json"), mode='w', encoding='utf-8') as out:
    json.dump(data_all, out, ensure_ascii=False, indent=4)
