from collections import defaultdict
import argparse


def cla_parser():
    """
    Parse commandline arguments

    :return: parsed args
    """
    parser = argparse.ArgumentParser(description='Preprocess Classifier Training Data')
    parser.add_argument('-m', '--unannotated_mode', type=bool, default=False, help='Preprocess annotated or unannotated data.')
    parser.add_argument('-u', '--unann_file', default='qa_data.txt', help='File with question-answer pairs; one pair per line split with <answer>')
    parser.add_argument('-a', '--ann_file', default='qa_data_annotated.csv', help='File with annotated qa-pairs. Form: When were you born? <answer> 1999 [<answer> ...]#0')
    parser.add_argument('-p', '--data_path', default='data/classifier/', help='Path to the data folder. Default: data/classifier/')
    parser.add_argument('--sep_ann', default='#', help='Separator of QA-pairs and annotations. Default: #')
    parser.add_argument('--sep_ans', default='<answer>', help='Separator of questions and answers. Default: <answer>')

    return parser.parse_args()


def write_out_file(filename):
    """
    Write processed question-answer pairs to output file. Filename is specified based on operation mode (annotated or unannotated).
    """
    print("Writing new file")
    with open(f"{args.data_path}{filename}.txt", "w", encoding="UTF-8") as out_file:
        for ques in question_dict.keys():
            for ans,ann in question_dict[ques]:
                # write one question-answer pair with corresponding annotation per line
                out_file.write(f"{ques} {ans}#{ann}\n")


def process_ann_file():
    """
    Reading and converting an annotated csv file. Saves pairs in in the question_dict in form
    {question: [(ans_1, ann), (ans_2, ann), ...]; ...}
    """
    print("Reading annotations")
    with open(f"{args.data_path}{args.ann_file}", encoding="UTF-8", errors="ignore") as ann_file:
        for line in ann_file.readlines():
            try:
                ques_ans_pair, ann = line.split(sep=args.sep_ann)
                ques_ans_pair = ques_ans_pair.split(sep=args.sep_ans)
                # write answer-annotation pairs as a list of tuples
                question_dict[ques_ans_pair[0]] = [(ans.strip(), ann.strip()) for ans in ques_ans_pair[1:]]
            except ValueError:
                print(line)


def process_qa_file():
    """
    Reading and converting a list of QA-pairs. If an annotation file exists and is given, these pairs contained in this file are removed from the defaultdict, 
    such that it only contains unannotated pairs from the corpus.
    """
    print("Reading QA-pairs")
    # part one: read all qa-pairs in corpus
    with open(f"{args.data_path}{args.unann_file}", encoding="UTF-8") as qa_file:
        for line in qa_file.readlines():
            ques_ans_pair = line.split(sep=args.sep_ans)
            try:
                # we add annotation 0 for all pairs for formatting reasons. These are later discarded in prediction.
                question_dict[ques_ans_pair[0]].append((ques_ans_pair[1].strip(), 0))
            except IndexError:
                print(f"Some error occured at: {ques_ans_pair}")
    # part two: double check with annotation file to remove annotated pairs, as these do not need to be predicted.
    try:
        print("Reading annotations")
        with open(f"{args.data_path}{args.ann_file}") as ann_file:
            for line in ann_file.readlines():
                ques_ans_pair, ann = line.split(sep=args.sep_ann)
                ques_ans_pair = ques_ans_pair.split(sep=args.sep_ans)
                # None suppresses the error message if the question is not in the dict
                question_dict.pop(ques_ans_pair[0], None)
    except FileNotFoundError:
        print("Did not find file with annotations or file with annotations was not given. Procede with full qa-pairs list.")


args = cla_parser()
# defaultdict(list) so that I can append new pairs without checking if the entry already exists
question_dict = defaultdict(list)
# modecheck: call different functions if annotated or unannotated questions should be processed
if not args.unannotated_mode:
    output_file_name = "annotated_qa_pairs"
    process_ann_file()
else:
    output_file_name = "unannotated_qa_pairs"
    process_qa_file()
write_out_file(output_file_name)
