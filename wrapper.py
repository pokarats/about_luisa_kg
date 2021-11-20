'''
This wrapper aims to intergrate the Answer Extension module into
the overall pipeline. We receive as an input a question-answer pair retreived by the
Q-A Chatbot module, then extend and paraphrase the answer if appropriate.
'''

'''
Please install the following packages before running the script: 
-nltk
-sklearn
-transformers==2.8.0
-torch
'''

from extension.tf_idf import tf_idf
from extension.t5_model_paraphrase import t5_model_paraphrase

# it is just an example
#question_answer = "What inspires your art? A lot, experiences, texts, songs, films, conversations, observations, the daily life as news."
question_answer = "Are you interested in architecture? Yes I am very interested in architecture"

# TODO: classify the answer

# if the Question Answer pair is classified as extendable:
# generate an extension with tf-idf model and paraphrase it with T5
tf_idf_extension = tf_idf.response(question_answer)
print(tf_idf_extension)
#pass the tf_idf output to the paraphrasing module
#TODO: or should we make paraphrasing optional?
extension_paraphrased = t5_model_paraphrase.paraphrase(tf_idf_extension)
print(extension_paraphrased)


