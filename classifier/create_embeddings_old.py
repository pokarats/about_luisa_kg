# This is deprecated as the bert model no longer works. See 'create_embeddings.py' for new version.

from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import numpy as np
import argparse


def cla_parser():
    """
    Parse commandline arguments

    :return: parsed args
    """
    parser = argparse.ArgumentParser(description='Create Embeddings for Data')
    parser.add_argument('-p', '--pretrained_model', default="sentence-transformers/bert-base-nli-cls-token", help='Pretrained bert model.')
    parser.add_argument('-f', '--file', help='File with annotated question-answer pairs; one pair per line split with <answer>')
    parser.add_argument('-d', '--data_path', default='data/classifier/', help='Path to the data folder. Default: data/classifier/')
    
    return parser.parse_args()


def read_data():
    """
    Reads a given file and turns it into a list of (data, annotation) tuples.
    """
    with open(f"{args.data_path}{args.file}.txt", encoding="UTF-8") as f:
        # create list of [qa-pair, ann] pairs
        qa_pairs = [datapoint.strip().split('#') for datapoint in f.readlines()]
        try:
            # convert [qa-pair, ann] to (qa-pair, ann); read ann as float
            qa_pairs = [(qa_pair, float(ann)) for qa_pair, ann in qa_pairs]
        except:
            print("An error occured in the conversion. Aborting.")
            exit
        return qa_pairs


def encode():
    """
    Encodes the qa-pairs using the pretrained model.
    :return: encoded data; a list of (enc-qa-pair, ann) tuples
    """
    encoded_data = []
    for qa_pair, ann in data:
        # we encode the qa-pairs as a whole as both the question and the answe are relevant for the classification
        tokenized_qa_pair = tokenizer(qa_pair, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded_qa_pair = model(**tokenized_qa_pair)
        # The np.float32 is necessary so that the annotation has the right format for training. Without this, I got an error in training.
        encoded_data.append((encoded_qa_pair[0][0, 0].float(), np.float32(ann)))
    return encoded_data


args = cla_parser()
# loading the pretrained bert-model
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
model = AutoModel.from_pretrained(args.pretrained_model, return_dict=True, output_hidden_states=True)
# we don't want to train new embeddings, we just want to use the existing embeddings
model.train(False)

data = read_data()
data = encode()

with open(f"{args.data_path}{args.file}.pickle", "wb") as f:
    pickler = pickle.Pickler(f)
    pickler.dump(data)
