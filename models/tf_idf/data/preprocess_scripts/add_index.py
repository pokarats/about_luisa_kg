# this module updates the index of dictionary entries, because several documents with different indexing have been merged.

import json
import sys

corpus_list = []
def add_index(filename):
    with open(filename, "r") as fin:
        index = 0
        for line in fin:
            index += 1
            line=line.strip()
            if line.endswith(","):
                line = line[:-1]
            #try:
            convertedDict = json.loads(line)
            convertedDict["ID"]=index
            corpus_list.append(convertedDict)
            #except json.decoder.JSONDecodeError as e:
            #    print(e, line)
            #    sys.exit(1)
add_index("corpus_tagged_initial.json")

with open("../../tf-idf/corpus_tagged_new.json", "w") as fout:
    json.dump(corpus_list, fout, ensure_ascii=False, indent=4)

