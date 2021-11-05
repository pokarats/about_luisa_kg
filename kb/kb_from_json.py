import argparse
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
cl_parser = argparse.ArgumentParser(description='KB Generation from JSON file')
cl_parser.add_argument('-i', '--input_file', default='kb/KG_presentation_example_social_records_only.json', help='Input filename')
cl_parser.add_argument('-d', '--data_dir', type=str, default='kb', help='Name of data directory')
cl_parser.add_argument('-o', '--out_dir', type=str, default='kb', help='Name of output KG directory')

args = cl_parser.parse_args()
print(args)

input_filename = Path(args.input_file).name

# output/extracted entities and relations will be in 'kb' directory
out_kb_dir = parent_dir / args.out_dir
try:
    out_kb_dir.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f"{out_kb_dir} directory is already there")
else:
    print(f"{out_kb_dir} directory was created")

in_filename = parent_dir / args.data_dir / input_filename

# read in json file
with open(in_filename) as f:
    df = pd.read_json(f, orient='records')
kg_from_df = df.drop(labels='key', axis='columns')

# graph for all subj-obj-relation pairs
G = nx.from_pandas_edgelist(kg_from_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, k=0.75)  # k determines distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
save_file_name = "KG_all_image" + str(in_filename.stem)
plt.savefig(out_kb_dir / (save_file_name + ".png"), bbox_inches="tight")

data_all = json_graph.node_link_data(G)
#save_file_name = "KG_all_" + str(in_filename.stem)
with open(out_kb_dir / (save_file_name + ".json"), mode='w', encoding='utf-8') as out:
    json.dump(data_all, out, ensure_ascii=False, indent=4)
