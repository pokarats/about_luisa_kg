This directory cointains the following flies:

- corpus_tagged.json - Louisa corpus in a json format

- corpus_final.txt - Louisa corpus in the format needed for tf-idf classifier. Format: paragraph <topic> topic1 topic2 topic3

- converter_json_to_corpus.py - script that converts corpus in json format into corpus needed for tf-idf

- tf_idf.py - script


Experiments with tf_idf model:
- to classify the paragraphs in the Louisa corpus with the zero-shot classifier and to look at the output of the tf_idf model based on this corpus;
- look at the output of tf_idf on the corpus where we annotated the topics of the paragraphs manually;
- to experiment with stop words list for tf_idf. 



