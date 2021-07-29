# Knowledge Graph About Luisa's Works

Naive approach to extract concepts for building the knowledge base from articles about Luisa's work

## Stats

After pre-processing, resultant corpus consists of 350 sentences.

## Pre-processing Scripts
- `preprocess.py` is used for preprocessing text for LDA Topic Modeling
- `preprocess_for_sent.py` is used for preprocessing text into sentences for KG generation

## Steps
1. Manually pre-process into paragraphs that discuss the same topic
2. Pre-process into 1 sentence per line, stripping symbols, but keep phrasing punctuation marks and casing
3. Using SpaCy dependency parser, parse each sent for subj-obj entities using a series of rules
4. Parse for relation, again using SpaCy Matcher; using ROOT and conj dep_ as predicates
5. Allow multiple entities per sentence; cartesian product: subj, obj, relation to get maximum number of entity pairs
6. Networkx generates a directed graph from these subj, obj, rel pairs

## Entity and Relation Extraction
1. Follows steps in this [Building KG Tutorial](https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/)
2. Rules for entities and relation predicates are adapted to reflect characteristics of the sentences in our data after analysis

## Graph
1. saved visualization for top 10 edges
2. all nodes and relations also saved to JSON format

## Node
1. Each Node: `subject(source) - object(target) - relation(edge)`

# Topic Modeling

Another approach to quickly grasph the prevalent concepts in the articles about Luisa's work is by using LDA (Latent
Dirichlet Allocation) for topic modeling.

## Steps
1. Follow slightly different pre-processing steps:
2. Experimented with 5 and 10 topics; displaying 5 - 10 frequent words per topic
3. Experimented with 10, 50, 100-500 iterations with **alpha** = [0.02, 0.05, 0,1] and **beta** = 0.1

## Observation
1. Coherent topics emerge after approx 100-200 iterations
2. 5 topics and 10 frequent words seem to be a good balance between getting enough content words and homogeneity
3. Data is too small to rely _only_ on this for knowledge base generation, but ok for initial analysis

# Preprocessing for Style Adaptation
tbc


