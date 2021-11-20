# This code is adopted from the code provided by Ernie Chang in https://gitlab.com/erniecyc/basic-bot/-/blob/master/chat.py#L71

# import necessary libraries
import pathlib
import random
# to process standard python strings
import string
import warnings
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

#import os.path
#print(os.path.join(os.path.dirname(__file__), 'file.txt'))

# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves",
    # Our additional stop words
    "yes", "no", "thank", "thanks", "interested", "wish", "hope", "let"
    ])

warnings.filterwarnings("ignore")

# comment the following after the first successful run
nltk.download("popular", quiet=True)  # for downloading packages
nltk.download("punkt")  # first-time use only
nltk.download("wordnet")  # first-time use only

# Reading in the corpus
input_file = pathlib.Path(__file__).parent / "corpus_final.txt"

with open(input_file, "r", encoding="utf8", errors="ignore") as fin:
    # read thу file and lower case
    raw = fin.read().lower()

# Tokenisation
# convert the file into list of sentences splitting by the new line
# list of question - answer pairs
sent_tokens = raw.split("\n")

sent_tokens_without_tags = [token.replace('<answer>', ' ').replace('<topic>', ' ') for token in sent_tokens]

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = (
    "hello",
    "hi",
    "greetings",
    "sup",
    "what's up",
    "hey",
)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        # check if the word in the user response is a greeting
        if word.lower() in GREETING_INPUTS:
            # if it is a greeting, generate a random answer
            return random.choice(GREETING_RESPONSES)

RE_TAG = re.compile(r'<[a-zA-Z]+>')

def split_text(text):
    result = []
    pos = 0
    current_tag = None
    while pos < len(text):
        match = RE_TAG.search(text, pos)
        if match:
            result.append((text[pos:match.start()], current_tag))
            current_tag = match.group()
            # result.append(text[match.start():match.end()])
            pos = match.end()
        else:
            result.append((text[pos:], current_tag))
            break
    return result

def postprocess(response):
    text_and_tag_list = split_text(response)
    for text, tag in text_and_tag_list:
        if tag == '<answer>':
            return text
    return text_and_tag_list[0][0]

    #if '<answer>' in response:
    #    print('FOUND <answer> !!!')
    #if len(response.split("<answer>")) > 1:
    #    # if there are more than 1 answer, return the last element
    #    return response.split("<answer>")[1]
    #else:
    #    # if there is only one answer, return only that answer
    #    return response


# Generating response to the user question
def response(user_question):
    """
    This function can be modified for better matching capability.

    """
    louisa_response = ""
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=ENGLISH_STOP_WORDS)
    tfidf = TfidfVec.fit_transform(sent_tokens_without_tags + [user_question])
    # find a cosine similarity between the last element(user_response) and the rest corpus
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        louisa_response += "I am sorry! I don't understand you"
        return postprocess(louisa_response)
    else:
        louisa_response += sent_tokens[idx]
        return postprocess(louisa_response)


def interactive_main():
    flag = True

    # to be modified
    print("My name is Louisa. I will answer your queries. If you want to exit, type Bye!")
    while flag == True:

        # get user response
        user_response = input()
        user_response = user_response.lower()

        # retrieving bot response
        if user_response != "bye":
            if user_response == "thanks" or user_response == "thank you":
                flag = False
                print("You are welcome..")
            else:
                if greeting(user_response) != None:
                    print(str(greeting(user_response)))
                else:
                    print("", end="")
                    print(response(user_response))
        else:
            flag = False
            print("Bye! take care..")


def main():
    # to preprocess the corpus
    with open("./data/preprocessed/qa_extended_pair.txt", "r") as fin, open("./results/filtered_input/qa.txt", "w") as qa, open("./results/filtered_input/gold_extension.txt", "w") as gold, open("./results/filtered_input/tf_idf.txt", "w") as tf_idf:
        for line in fin:
            question_answer, extension = line.strip().split("<start_extension>")
            print(question_answer.strip(), file=qa)
            print(extension.strip(), file=gold)
            question_answer = question_answer.lower()
            user_response = response(question_answer)
            print(user_response, file=tf_idf)


if __name__ == "__main__":
    main()
