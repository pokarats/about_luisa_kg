# Knowledge Graph About Luisa's Works

## Steps
1. manual pre-process into paragraphs that discuss the same topic
2. pre-process into 1 sentence per line, stripping symbols, but keep phrasing punctuation marks and casing
3. Using SpaCy dependency parser, parse each sent for subj-obj entities using a series of rules
4. parse for relation; using ROOT and conj dep_ as predicates
5. allow multiple entities/sent, product combo: subj, obj, relation to get maximum number of entity pairs
6. Networkx these subj, obj, rel to graph

## Entity and Relation Extraction
1. Follows steps in this [Building KG Tutorial](https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/)
2. Rules for entities and relation predicates are adapted to reflect characteristics of the sentences in our data after analysis

## Graph
1. saved visualization for top 10 edges
2. all nodes and relations also saved to JSON format

## Node
1. Each Node: `subject(source) - object(target) - relation(edge)`


