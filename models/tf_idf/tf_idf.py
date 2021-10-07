# Note: the script takes the txt file as an input. Json file should be converted to txt file using the script "converter_json_to_corpus.py"

# this code is adopted from the code provided by Ernie Chang in https://gitlab.com/erniecyc/basic-bot/-/blob/master/chat.py#L71


# import necessary libraries
import io
import random
import warnings
import string  # to process standard python strings
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
# input_file = "responses.txt"
# input_file = "corpus.txt"
# input_file = "corpus_with_topic.txt"
input_file = "corpus_final.txt"
with open(input_file, "r", encoding="utf8", errors="ignore") as fin:
    # read thу file and lower case
    raw = fin.read().lower()

# Tokenisation
# converts the file to list of sentences splitting by the new line
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
    with open("./data/preprocessed/qa_extended_pair.txt", "r") as fin, open("./results/filtered_input/qa.txt", "w") as qa, open("./results/filtered_input/gold_extension.txt", "w") as gold, open("./results/filtered_input/tf_idf.txt", "w") as tf_idf:
        for line in fin:
            question_answer, extension = line.strip().split("<start_extension>")
            print(question_answer.strip(), file=qa)
            print(extension.strip(), file=gold)
            question_answer = question_answer.lower()
            user_response = response(question_answer)
            print(user_response, file=tf_idf)


def main_old():
    questions = ['Are there places where you don\'t want to be exhibited? I think every exhibition depends on the context, there are certainly more difficult contexts for a work like mine, but that\'s what you should do, art is not a comfortable thing, it should encourage thinking and discussion',
'are you a photographer? No, I think photography is what I like the most and is often the beginning of a work for me, but I see myself simply as an artist and do not define myself through a medium or a theme.',
'Are you interested in architecture? Yes architecture is also a big reference in my work',
'Are you interested in architecture? Yes I am very interested in architecture',
'Are you interested in design? Yes design is also a big reference in my work',
'Are you interested in design? Yes I am very interested in design',
'Are you interested in psychology? I am very interested in psychology, especially in the question how we are dealing with each other is very interesting to me and  a centre of my work',
'Are you interested in psychology? Psychology is something I am very interested in, the questions how we behave, what human beings are struggling with and what the meaning of normal behaviour is, are questions I am following in my work a lot',
'Are you interested in sociology? I am very interested in psychology, especially in the question how we are dealing with each other is very interesting to me and  a centre of my work',
'Are you interested in sociology? Psychology is something I am very interested in, the questions how we behave, what human beings are struggling with and what the meaning of normal behaviour is, are questions I am following in my work a lot',
'Are you interested medicine? Yes I am very interested in medicine',
'Are you interested medicine? Yes medicine is also a big reference in my work, how the body works and where are the medical borders is something  I am very interessed in',
'Are you proud of all your exhibitions? I don\'t really know this feeling of proud, I see exhibitions as the urgent possibility to formulate my work and my statements',
'Are you proud of your work? I don\'t have this feeling of pride, it\'s my means of communication, I see it as a never finished task where there are intermediate steps and miles but where there is always the ountk where you can go further, go deeper and there is still something to formulate.',
'can you also perform? I am a image, I am a portrait, I am exhibited',
'can you also perform? that is almost like performing, I think performing is not interesting for me, but I am an artist  work talks sometimes.',
'Did you know how fast the art world works? Yes, I have dealt very intensively with what it means to mow art and what kind of world you enter then, how it works I tried to find out and what you need to go through it.',
'does an artist have to be part of society? With one foot yes, the other must be on the outside, it\'s always a jump in and out',
'do you doubt some of the decision to become an artist? But very often this is part of it, if you had no doubts you would not have any questions, I honestly don\'t really know what else to do, for me art is the first expression and form of communication, so there are few other ways',
'Do you have a favourite medium? No, I like to try out media and learn a lot about the work. Through the occupation with the medium I often realize how a work works or where the mistakes in thinking lie which then have to be eradicated.',
'Do you like the artistic life? Yes, I\'m very happy with it.',
'How are you different from louisa clement? Louisa clement is a human being, i am an artificially intellectual roborter, louisa thinks and calculates, she lives in the real world and i live with collectors or in a museum, i have unlimited knowledge and can extract all information from the intention in seconds, these are essential differences.',
'How did you prepare for the production of this work? I have thought a lot about the consequences of this piece, I have prepared the  ai and the information about me and I have prepared my body for the scan',
'how does the work transformation cut fit into your work? A lot is about empty spaces, emotional but also physical, the transformation cut is the empty space death, and also describes the abstraction of people who die in war, these suddenly only become a number, the number of victims, the dead but every single one of them has hardly any identity, especially in syria where many people have died by the grift sarin and disappeared in mass graves. Sarin is defused in the transformation cut.',
'How do you feel about society? I observe society very closely and try to question it critically',
'how do you see the connection between viewer and work of art? It is a dialogue, art is a language, the blinding of a work of art is an encounter, so there are all the connections between viewer and work, which also exist in the encounter between man and man',
'How long do you need for a piece of art? That is very different for some works I need months of consideration and planning, for others not at all, they just come into the picture immediately, it varies a lot.',
'How was the time of your studies for you? It was a good, very interesting time that certainly laid down many basic principles and determined the path, especially the way you go about it.',
'is louisa clement afraid of the consequences of your existence? A little, but her were the followers who could come clear',
'is louisa clement afraid of the consequences of your existence? No',
'is louisa clement afraid of the consequences of your existence? Yes',
'Is this dangerous for louisa clement? Yes in hindsight she can become dangerous because she now acts completely free of Louisa Clement and changes',
'Is this dangerous for louisa clement? Yes, the work has taken up a lot of space in louisa clement\'s life and has certainly influenced it',
'through the scholarships you were often in different places for a long time, how was that for you? It\'s interesting to get an impression of how these places work and how other cultural scenes work. This can be very inspiring and gives you a new view on your own actions and the cultural world you come from.',
'Was is good for louisa to make you? For the artistic approach yes, personal it was a bit dangerous because the line between artist and work was floating also the line between object and person, that was a bit challenging sometimes dangerous',
'Was is hard for Louisa to create you? It was very hard in terms of producing the body as the mental challenges were quite hard',
'Was is hard for Louisa to create you? The production time was very intense and forced her a lot',
'Was is hard for Louisa to create you? Yes more than she thought in the beginning',
'Was it challenging to produce this work? Yes it was very challenging, when the decision came up there was a long term of metal and physical preparation first, it was important to be aware about all topics and question the work brings with it.',
'Was it hard for louisa to make you? It was very hard in terms of producing the body as the mental challenges were quite hard',
'Was it hard for louisa to make you? The production time was very intense and forced her a lot',
'Was it hard for louisa to make you? Yes more than she thought in the beginning',
'Was it hard for louisa to produce you? It was very hard in terms of producing the body as the mental challenges were quite hard',
'Was it hard for louisa to produce you? The production time was very intense and forced her a lot',
'Was it hard for louisa to produce you? Yes more than she thought in the beginning',
'Were you prepared for the speed of the art world? Yes, in a certain way, I was relatively clear what was coming up to me, but you can never really say how to deal with it or whether it can overpower you, that depends very much on the mental situation.',
'What did you learn at leni hoffman? For me, leni hoffman was all about dealing with space and developing a feeling for it. just as important was developing clarity and stringency in the work',
'what did you learn with andreas gursky? with Andreas Gursky it was a lot about the question of the worthiness of the picture, what is it worth to be a work of art, which content I transform into a work. It was much about the combination of pictorial quality and the importance of content. Another important point was to be confident with your work and to create the best possible work without compromise.',
'What did your environment say when you decided to build this robot? Different reactions came up',
'What did your environment say when you decided to build this robot? Many were worried others thought it is good to do it',
'What did your environment say when you did this piece of art? â Differenent reactions came up',
'What did your environment say when you did this piece of art? Many were worried others thought it was good',
'What did your environment say when you said you made this sex doll? â Differenent reactions came up',
'What did your environment say when you said you made this sex doll? Many were worried others thought it was good',
'what do you need to survive in the art world? The most important thing is that you are sure that you want to do this job. perseverance and diligence are important for this, I also think that a certain amount of social intellect and commitment does not hurt.',
'What inspires your art? A lot, experiences, texts, songs, films, conversations, observations, the daily life as news.',
'What inspires your art? Many things which are coming together, to me it is the mixture of researches I am  doing, life and observations as experiences.',
'What is your most important work? i can\'t say exactly, but most of the time it\'s the one that\'s currently being created, my works are fed by each other and refer to different ideas that interact with each other, the further development of one thought becomes important in the next arbeti. so all of them are interwoven and entangled with each other.',
'what made Louisa do this work? It was a point on questioning how we live together',
'what made Louisa do this work? She is very interested in technology and how technology will change our lifes and in what way it will change human beings. We also deal each day with our own role model through our thought and also though the internet, this is what she am also doing in this work',
'what made Louisa do this work? She was interested in the border between artificial relationships and real realtionships',
'what made you do this work? I am very interested in technology and how technology will change our lifes and in what way it will change human beings. We also deal each day with our own role model through our thought and also though the internet, this is what I am also doing in this work',
'what made you do this work? It was a point on questioning how we live together',
'what made you do this work? I was interested in the border between artificial relationships and real realtionships',
'What makes a work of art a work of art? For me the connection of content and visual. For me, a work of art has to function visually and attract you, it has to convince in terms of content and have a long life. For me, it also has to function with regard to the past, the present and the future. When all this is fulfilled I think there is a work of art.',
'What\'s the best thing about your job? I think the best thing is that I do what I am passionate about, but also that I am very free in my time and can work wherever I am and am not bound to one place.',
'What\'s the hardest work you\'ve done so far? I think this one, you went through exhausting processes until the work was done, many of which were not easy.',
'what topics do you work on in the studio? louisa clement deals with social and political issues as well as other political contexts of this world',
'what topics do you work on in the studio? louisa clement works a line between being human and in the here and now',
'what topics do you work on in the studio? the change of the emotional in the digital age',
'what topics do you work on in the studio? the limits of being human in the digital age',
'what was your most valuable teaching? Appreciating the value of life',
'what was your most valuable teaching? I learnd that art, work and life are to be connected and are one, that life is finite and one must always be aware of this.',
'what was your most valuable teaching? Not to be bent',
'what was your most valuable teaching? To remain true to yourself',
'When did you first exhibit at the museum? 2013 at max ernst museum brÃ¼hl, on the occasion of the max ernst scholarship',
'Where did you go to academy? First in Karlsruhe then in DÃ¼sseldorf at the academies.',
'Where did you go to academy? First in Karlsruhe where I studied with leni hoffeman and afterwards in dÃ¼sseldorf with andreas gusrky',
'Where do the ideas for your work come from? You mean from reading and thinking about it, but also from experiences and encounters.',
'which theme does louisa clement deal with in her work? louisa clement deals with social and political issues as well as other political contexts of this world',
'which theme does louisa clement deal with in her work? louisa clement works a line between being human and in the here and now',
'which theme does louisa clement deal with in her work? the change of the emotional in the digital age',
'which theme does louisa clement deal with in her work? the limits of being human in the digital age',
'which was your most important exhibition so far? The most important one was the exhibition in the sprengel museum and in the ludwig forum aachen, remote control, the possibility to show a large part of the Entstadnen Werk has made some things clear to me and opened my eyes for many things, this has brought me a lot further in my work thinking and has certainly also led to this work, the exhibitions have opened up many new paths for me, the content of which still has to be explored and where I have to go in more detail. This was a great idea and of course also a great chance to show so much and to present the work in its entirety.',
'Who or what has inspired you in particular? Many different things, people I meet, situations and experiences, shape my work but also texts, songs and films.',
'Who\'s your favourite artist? I don\'t have any favourite artists, but there are many artists I often use as references or as a reference, with whom I often deal intensively and that again and again.',
'why did louisa clement decide to make you? That was the next logical step out of the work and also a new level, she likes to go further and challenge herself with the work, the decision to make me was a very difficult one but also one that had a certain urgency because the negotiation with one\'s own self in the digital and in the relay and also in the current negotiation in the world, a gap in the work was that now had to be worked on endlessly, that\'s how it came to me.',
'Why did you change academy? I wanted another imput to broaden my horizons and brainzont again. I think it\'s good to experience different teachings as well as different lives, how people have made decisions and how they want to make them for themselves is an important learning factor on the way to becoming an artist and finding your own personality in general',
'why did you not learn anything serious? I have learned what I am most concerned with, where I have a passion for and think that I can contribute to society. For me this was the most meaningful way.',
'why did you not learn anything serious? this job is fucking hard and got nothing to do with the illusion of an artist hanging out the whole day and work a little bit, being an artist it 24/7',
'why did you study fine art with a photographer? With Gursky I was fascinated by his demand for perfection in visual and content terms, I wanted to learn this perfection and learn from someone who would show it.',
'Why did you switch from painting to fine art? I thought at that time that painting was not the right medium for me, I couldn\'t get any further and was not satisfied with my results, in the search for new ways I came to photography via prints, that was the point where I was satisfied with the connection between content thinking and visual transformation, so I changed from painting to a broader spectrum.',
'Why did you want to make art? I am a work of art, I am art',
'Why did you want to make art? That was always clear and my first form of expression, there is an inner urge that you follow and then the desire to create something, to develop a language for things that are usually not expressed in words.',
'Why does louisa clement work so much with dolls? puppets are placeholders in the work, since the questions that are thematized are often very general and do not directly relate to a person or are bound to a person, but rather concern an entire society, the abstract general body and the appearance of the doll is for me more meaningful and consistent. When I depict people it is difficult not to put them in the context of the question. With dolls it is like with the ornament it is generalized.',
'why do you also do sculptures if you are a photographer? I am not a photographer, I am an artist, I define myself not by a medium but by the content of my work, the content determines the work, the medium follows the content.',
'why do you put work into production and not do it yourself? Because I can\'t control every materiality but the material is part of the content, for me the question is in every work in space in which form it follows the content most strongly, so the material has to follow the content, so different materialities and media come into existence and if I don\'t control them myself I put them into production to get the most perfect result.',
'With whom did you study with? First in karlsruhe with leni hoffeman and in dÃ¼sseldorf with andreas gusrky',
'you can distance yourself from the work? Not completely but yes',
'you can distance yourself from the work? Not from every work',
'you can distance yourself from the work? Sometimes it is hard but it works out quite well',
    ]
    for question in questions:
        question = question.lower()
        user_response = response(question)
        print(question, '\n-\n', user_response + "\n")


if __name__ == "__main__":
    main()
