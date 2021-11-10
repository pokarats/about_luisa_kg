# Given a simple text this module returns it in json document. Json document is a list of dictionaries and each dictionary corresponds to one paragraph of the input text. Each dictionary has the following structure: {"ID"=count, "<answer>"=paragraph, "topics"=topics_list} or  {"ID"=count, "question"=question, "<answer>"=anwer, "topics"=topics_list} for the interviews. 

from collections import defaultdict
import json


text_list="exhibition, exhibition organization, art work description, series, invitation, meeting, plan, travelling, medium, topics of work, current work, production, selling the pieces, informing, exploring, sending pieces, experience, inspiration, background, historical context, social context, political context, psychology, sociology, philosophy, photography, fellowship, artistic idea, art school, use of chemical weapons, war, poison, new human, human body, boundaries of human, human-machine opposition, artificial intelligence, upgrading the body, stopping the age".lstrip().rstrip().split(", ")

def corpus_preprocess(filename):
	dict_list=[]
	with open(filename, "r", encoding="utf-8") as fin:
		count=0
		for line in fin:
			line=line.strip()
			count+=1
			line_dict={}
			line_dict["ID"]=count
			line_dict["<answer>"]=line
			line_dict["topics"]=text_list
			dict_list.append(line_dict)
	return dict_list

my_dict=corpus_preprocess("articles.txt")

with open('articles_tagged_part1.json', 'w') as json_file:
    json.dump(my_dict, json_file)
