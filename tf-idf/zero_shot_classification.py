from transformers import pipeline
import os.path

'''
# If we use Google Colab, then we mount Google Drive
from google.colab import drive
PROJECT_NAME = 'Louisa'
drive.mount('/content/gdrive')
ROOT_DIR = "/content/gdrive/My Drive/" + PROJECT_NAME
import os
os.makedirs(ROOT_DIR, exist_ok=True)
!ls -la "{ROOT_DIR}"
!head "{ROOT_DIR}/test_corpus.txt"
'''

#ROOT_DIR = os.path.dirname(__file__)
#print(ROOT_DIR)

classifier = pipeline("zero-shot-classification")
# classifier = pipeline("zero-shot-classification", device=0) # to utilize GPU

topics = "exhibition, exhibition organisation, meeting, inspiration, plan,  travelling, production, selling the pieces, current work, invitation, upgrading the body, artificial intelligence, stopping the age, topics of work, art school, artistic idea, fellowship, psychology, sociology, philosophy, human body, boundaries of human, human-machine opposition, historical context, social context, political context, art work description, medium, use of chemical weapons, new human, artificial interventions, war, series, poison."
topics = [s.replace('.', ' ').strip() for s in topics.split(',')]

with open('corpus.txt', 'r', encoding='utf-8') as fin:
    output = []
    count = 0
    for line in fin:
        line = line.strip()
        if len(line) == 0:
            continue
        count += 1
        print("I am still working", count)
        topic = classifier(line, topics)
        output.append((line, topic))


with open('corpus_with_topic.txt', 'w', encoding='utf-8') as fout:
    for paragraph, topic in output:
        fout.write(paragraph + "<topic>" + topic["labels"][0] + "\n")


