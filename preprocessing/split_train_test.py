from random import sample

from sklearn.model_selection import train_test_split, KFold
import argparse
from pathlib import Path


"""
Split data into train/val/test.
ToDo: kfold cross valdiation for train/val partitions
"""
cl_parser = argparse.ArgumentParser(description='Split Train and Test set')
cl_parser.add_argument('-s', '--src_file', default='style_preprocessed_sentluisa_text_optimized.txt', help='First Input filename')
cl_parser.add_argument('-t', '--tgt_file', default='style_preprocessed_sent_emails_all.txt',
                       help='Second Input filename')
cl_parser.add_argument('-d', '--src_dir', type=str, default='../data', help='Name of src data directory')
cl_parser.add_argument('-e', '--tgt_dir', type=str, default='../data/emails', help='Name of tgt data directory')
cl_parser.add_argument('-o', '--out_dir', type=str, default='train_test', help='Name of train/test directory')
cl_parser.add_argument('-k', '--kfold', type=int, default=5, help='min number of tokens in a sentence')
args = cl_parser.parse_args()


# in file and out file
src_filename = Path(args.src_file).name
tgt_filename = Path(args.tgt_file).name
k_fold = args.kfold
src_out_filename = 'src_'
tgt_out_filename = 'tgt_'

# input files
src = Path(__file__).resolve().parent / args.src_dir / src_filename
tgt = Path(__file__).resolve().parent / args.tgt_dir / tgt_filename

# directory for train/dev/test split
o_dir = Path(__file__).resolve().parent / args.out_dir
o_dir.mkdir(parents=True, exist_ok=True)

src_sentences = []
tgt_sentences = []

with open(src, 'r', encoding='utf-8') as f:
    for line in f:
        src_sentences.append(line.rstrip())

with open(tgt, 'r', encoding='utf-8') as f:
    for line in f:
        tgt_sentences.append(line.rstrip())

num_src = len(src_sentences)
tgt_sentences = sample(tgt_sentences, num_src)

assert len(src_sentences) == len(tgt_sentences)

src_train, src_test = train_test_split(src_sentences, random_state=86, train_size=0.8)
src_train, src_dev = train_test_split(src_train, random_state=86, train_size=0.75)

tgt_train, tgt_test = train_test_split(tgt_sentences, random_state=86, train_size=0.8)
tgt_train, tgt_dev = train_test_split(tgt_train, random_state=86, train_size=0.75)

assert len(src_train) == len(tgt_train)
assert len(src_dev) == len(tgt_dev)

print(f'train size: {len(src_train)}\n'
      f'dev size: {len(src_dev)}\n'
      f'test size: {len(src_test)}\n')

# write to files
src_o_train = o_dir / f'{src_out_filename}train.txt'
tgt_o_train = o_dir / f'{tgt_out_filename}train.txt'

with open(src_o_train, 'w') as str, open(tgt_o_train, 'w') as ttr:
    for s, t in zip(src_train, tgt_train):
        print(s, file=str)
        print(t, file=ttr)

src_o_dev = o_dir / f'{src_out_filename}dev.txt'
src_o_test = o_dir / f'{src_out_filename}test.txt'

tgt_o_dev = o_dir / f'{tgt_out_filename}dev.txt'
tgt_o_test = o_dir / f'{tgt_out_filename}test.txt'

with open(src_o_dev, 'w') as sod, open(src_o_test, 'w') as sot, \
        open(tgt_o_dev, 'w') as tod, open(tgt_o_test, 'w') as tot:
    for s_dev, s_test, t_dev, t_test in zip(src_dev, src_test, tgt_dev, tgt_test):
        print(s_dev, file=sod)
        print(s_test, file=sot)
        print(t_dev, file=tod)
        print(t_test, file=tot)

