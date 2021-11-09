__Original files:__
- qa_data.txt: original corpus
- qa_data_annotated.csv: manually annotated qa-pairs
- qa_data_annotated_long.csv: manually annotated qa-pairs, contains more non-extendable pairs

__Classifier pipeline:__

The steps of the classifier pipeline give three files, each annotated and unannotated:
- (un)annotated_qa_pairs.txt: the preprocessed file in the right format for the create_embeddings step. annotated_qa_pairs.txt is created from the annotated corpus and contains annotations (0 or 1). unannotated_qa_pairs.txt contains dummy annotations (0) for formatting reasons.
- (un)annotated_qa_pairs.pickle: contains the encoded qa_pairs; compatible with model.py
- (un)annotated_qa_pairs_predictions.txt: contains the qa-pairs and annotations as predicted by the trained model.

__Other files:__
- paper_data_and_pickels: encoded files, results and data used in the paper. The models used to create these files are now deprecated and can therefore not be reproduced.