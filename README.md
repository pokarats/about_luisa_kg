# Domain Specific Answer Extension for Conversational Chatbot
This repository contains the implementation of the various experiments and components of the Answer Extension module,
which forms a part of a question and answering chatbot for an art exhibition by Louisa Clement, whose goal is to 
create an interactive experience for the audience that highlights the blurring of boundaries between real and 
artificial experiences.

## Pipeline

As shown in Figures 1 and 2 below, there are three main components that contribute to the avatar’s chatbot functionality
when an exhibition visitor interacts with the avatar. These three modules adapt and extend the functionalities of the 
pre-installed default chat- bot component of the avatar.

1. The visitor’s question in speech form is processed by the Q-A Chatbot module into text; the visitor’s questions are 
   matched to the question-answer pairs from a corpus provided by Louisa Clement
   
2. In the Answer Extension module, a classifier determines the extendability of the matched answer; the generated answer
   extensions give more background information about the artist and her works.
3. In the Voice Assimilation module, either the matched answer from (1) or an extended answer from (2) is passed to 
   the text-to-speech component; a new voice model is trained on recordings of Louisa Clement’s voice and replaces 
   the default voice of the chatbot.

![Project Pipeline](/img/1_pipeline.png)

## Extendability

![Extendability](/img/2_classifier.png)

## Answer Extension

![AnswerExtension](/img/3_ae.png)

## Evaluation 

As shown in the diagram below, our evaluation consists of 2 steps: automatic evaluation with BLEURT and human evaluation
experiments.

First, we compare the tf-idf and GPT-2 extensions against our gold extensions using the 
[BLEURT](https://github.com/google-research/bleurt) automatic evaluation metric. This step allows for a time-efficient 
and objective evaluation; BLEURT is also able to compare semantic similarity between phrases beyond surface similarities 
or lexical overlaps. Subsequently, we select the best-performing model and its paraphrased extensions for human 
evaluation.

![Eval](/img/4_Eval.png)

### BLEURT

BLEURT is a pre-trained regression based model based on BERT ([Devlin et al., 2019](#References)) that takes as input 
two sentences: (1) a reference and (2) a candidate, and returns a score that indicates the extent to which the candidate
sentence conveys the meaning of the reference ([Sellam et al., 2020](References)). 

Higher scores suggest closer meanings and lower scores suggest otherwise. BLEURT scores are interpreted as a relative to
a reference. Although the rating model can be further fine-tuned, the relative nature of the scoring allows the model 
to be used for comparing outputs of different generation models.

`run_bleurt_eval.py` implements an API to calculate the BLEURT score for each reference-candidate pair, the total and 
the average scores as well as the execution time. In order to run this script, first please follow the BLEURT module
installation instructions in [BLEURT GitHub Repository](https://github.com/google-research/bleurt). More information
about this evaluation metric can also be found there.

Results are saved in the `auto_eval_scores` directory. Filenames following this convention: 
`auto_eval_<candidate filename>`.

### Human Evaluation



## Results
![Results](/img/5_results.png)

## Future Work

With more time for data annotation and model implementation, we are interested in exploring the following:

* Incorporate knowledge graph into text generation (e.g. train GPT2 based on the KG)
  
* Extend the classifier to perform topic classification 
  
* Further fine-tuning of the T5 model on Louisa’s corpus for more effective paraphrasing

# Preliminary Experiments and Pre-Processing

Below are steps we explored that may or may not have ended up in the final work of the project.

## Knowledge Graph About Luisa's Works

Naive approach to extract concepts for building the knowledge base from articles about Luisa's work.

### Stats

After pre-processing, resultant corpus consists of 350 sentences.

### Pre-processing Scripts
- `preprocess.py` is used for preprocessing text for LDA Topic Modeling
- `preprocess_for_sent.py` is used for preprocessing text into sentences for KG generation

### Steps

The following steps are implemented in `kb_extraction.py` and `kb_from_json.py`.

1. Manually pre-process into paragraphs that discuss the same topic
2. Pre-process into 1 sentence per line, stripping symbols, but keep phrasing punctuation marks and casing
3. Using SpaCy dependency parser, parse each sent for subj-obj entities using a series of rules
4. Parse for relation, again using SpaCy Matcher; using ROOT and conj dep_ as predicates
5. Allow multiple entities per sentence; cartesian product: subj, obj, relation to get maximum number of entity pairs
6. Networkx generates a directed graph from these subj, obj, rel pairs

### Entity and Relation Extraction
1. Follows steps in this [Building KG Tutorial](https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/)
2. Rules for entities and relation predicates are adapted to reflect characteristics of the sentences in our data after analysis

### Graph
1. saved visualization for top 10 edges
2. all nodes and relations also saved to JSON format
3. Graphs are stored in the ``/kb`` directory

### Node
1. Each Node: `subject(source) - object(target) - relation(edge)`

## Topic Modeling

Another approach to quickly grasp the prevalent concepts in the articles about Luisa's work is by using LDA (Latent
Dirichlet Allocation) for topic modeling.

### Steps

`lda.py` implements the LDA experiemnt. Data generated from the steps below are in `/data` directory.

1. Follow slightly different pre-processing steps:
2. Experimented with 5 and 10 topics; displaying 5 - 10 frequent words per topic
3. Experimented with 10, 50, 100-500 iterations with **alpha** = [0.02, 0.05, 0,1] and **beta** = 0.1

### Observation
1. Coherent topics emerge after approx 100-200 iterations
2. 5 topics and 10 frequent words seem to be a good balance between getting enough content words and homogeneity
3. Data is too small to rely _only_ on this for knowledge base generation, but ok for initial analysis

## Preprocessing for Style Adaptation
`style_transfer_preprocess.py` removes punctuations and re-format foreign punctuation marks.

Data files in our experiments are in the `/data` directory.


# References

Resources that we consulted and/or incorporated in our work.

[Chang, E. Basic bot. Meet Louisa: The Artist.](https://gitlab.com/erniecyc/basic-bot/-/blob/master/chat.py#L71)

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). 
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](http://arxiv.org/abs/1810.04805)

[Goutham, R. T5 Paraphrasing inference.ipynb.](https://colab.research.google.com/drive/1SdUHXDc6V3jxji1ef2NmE-vsStO35Adr?usp=sharing)

Howcroft, D. M., Belz, A., Clinciu, M.-A., Gkatzia, D., Hasan, S. A., Mahamood, S., Mille, S., van Miltenburg, E., 
Santhanam, S., & Rieser, V. (2020). [Twenty Years of Confusion in Human Evaluation: NLG Needs Evaluation Sheets and 
Standardised Definitions.](https://aclanthology.org/2020.inlg-1.23) Proceedings of the 13th International Conference on Natural Language Generation, 169–182.

Iyer, S., Dandekar, N., Csernai, K. (2017). First quora dataset release: Question pairs.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (n.d.). Language Models are Unsupervised Multitask 
Learners. 24.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). 
[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.](http://arxiv.org/abs/1910.10683)

Sellam, T., Das, D., & Parikh, A. P. (2020). [BLEURT: Learning Robust Metrics for Text Generation.](http://arxiv.org/abs/2004.04696)

[GPT2 Language Model Fine-tuning with Texts from Shakespeare.](https://colab.research.google.com/github/fcakyon/gpt2-shakespeare/blob/main/gpt2-shakespeare.ipynb)
