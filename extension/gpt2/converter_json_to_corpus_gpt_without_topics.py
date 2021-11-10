#This module converts the final filtered Louisa courpus in the json format to simple txt document
#without key-value pair "topics"
#corpus gpt2_final does not include tags <answer>, <question>

import json
input_file = "corpus_tagged.json"
output_file = "corpus_gpt_final.txt"

def converter(fin, fout):
    line_list=[]
    with open(fin, "r", encoding="utf8") as fin:
        # read the file and lower case
        loaded_file=json.load(fin)
        for item_dict in loaded_file:
            question_line=""
            if "question" in item_dict:
                #question_line=item_dict["question"] + "<answer>"
                question_line = item_dict["question"]
            #topics = " ".join(item_dict['topics'])
            #line = question_line + item_dict['<answer>'] + " " + "<topic>" + " " + topics
            line = question_line + item_dict['<answer>']
            line_list.append(line)

    with open(fout, "w", encoding="utf8") as fout:
        for item in line_list:
            fout.write(item + "\n")

converter(input_file, output_file)

'''
    try:
        for item in loaded_file:
            print(item, type(item))
    except json.decoder.JSONDecodeError as e:
        print(e, item)
        sys.exit(1)
'''