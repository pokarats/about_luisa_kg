from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import argparse
import torch


def cla_parser():
    """
    Parse commandline arguments

    :return: parsed args
    """
    parser = argparse.ArgumentParser(description='Create Embeddings for Data')
    parser.add_argument('-m', '--pretrained_model', default='sentence-transformers/multi-qa-mpnet-base-dot-v1', help='Pretrained bert model.')
    # this is the model used in the report, however, it is deprecated and the results are a lot better with the new model
    # parser.add_argument('-m', '--pretrained_model', default='sentence-transformers/bert-base-nli-cls-token', help='Pretrained bert model.')
    parser.add_argument('-f', '--file', help='File with annotated question-answer pairs; one pair per line split with <answer>')
    parser.add_argument('-d', '--data_path', default='data/classifier/', help='Path to the data folder. Default: data/classifier/')
    
    return parser.parse_args()


def read_data():
    """
    Reads a given file and turns it into a list of (data, annotation) tuples.
    """
    with open(f"{args.data_path}{args.file}", encoding="UTF-8") as f:
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
        encoded_qa_pair = model.encode(qa_pair)
        # The np.float32 is necessary so that the annotation has the right format for training. Without this, I got an error in training.
        encoded_data.append((torch.from_numpy(encoded_qa_pair), np.float32(ann)))
        # encoded_data.append((encoded_qa_pair, np.float32(ann)))
    return encoded_data


args = cla_parser()

model = SentenceTransformer(args.pretrained_model)
data = read_data()
data = encode()

with open(f"{args.data_path}{args.file}.pickle", "wb") as f:
    pickler = pickle.Pickler(f)
    pickler.dump(data)
