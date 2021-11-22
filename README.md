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
   
Within the Answer Extension module, there are 2 steps: (1) a BERT-based classifier to determine whether an existing Q-A
pair can be extended and (2) an Answer Extension module. The models we experimented with are tf-idf retrieval-based
model, GPT-2, and T5 for paraphrasing the outputs from either the tf-idf or GPT-2 model for increased variability.

![Project Pipeline](/img/1_pipeline.png)

The integration of tasks within this art project predefines some of the main requirements to the Answer Extension 
module. In addition, the key challenges of NLG in close-domain conversational dialogue systems also apply to our module 
considerations. The key requirements of our Answer Extension module include:

1. The extension should only contain factually correct information.
2. The extension should be relevant to Louisa Clement and not contain details she cannot know or too personal details she is not likely to share.
3. The extension should preserve Louisa Clement’s linguistic style.
4. The extension should lend itself to natural and informal human communication.
5. The extended answers to the same question should show sufficient variations and not be identically verbatim with repetitions.
   
Based on these requirements, we can formulate two general criteria extendable questions should fulfill: 
(1) the topics of the questions should be covered by the knowledge base to the task to en- sure the correctness and 
relevance of the content of the generated extension; (2) the question and answer should inherently allow for a longer 
answer. This is necessary to guarantee that the long answer sounds natural and fits into the conversation setting.

## Data

The corpora/datasets used in our project can be found in the `/data` directory. The source corpora for the *Extension
Corpus* are in `/emails` and `/interviews` directories.

The outputs from the generator models are in the `/results` directory of each model in the`/extension` directory.

## Extendability

We trained a simple BERT-based classifier to determine if a Q-A pair is extenable. We use the annotated part of the 
question-answer dataset, which shuffled and randomly split into train and test set (90%/10%). 
The question-answer pairs are then encoded using Sentence-BERT (Reimers and Gurevych, 2019) to be able to employ the 
embedded sentence meaning in the classification process. 

As summarized in the figure below, the classifier itself consists of a two-layer neural network with sigmoid 
non-linearities. We use BCEloss; the classifier is trained for 100 epochs. The implementation of this module can be 
found in the `/classifier` directory.

The question-answer dataset is a small QA corpus consisting of 1000 questions and 1845 individual answers; 1120 of which 
are annotated for the purpose of training the classifier (see Table 1 below). The questions have been curated and answered by Louisa Clement as an addition to the chatbot’s default 
QA component. The corpus covers different questions a visitor might ask from a variety of topics: from interests and 
biographical data, to ‘stereo- typical chatbot questions’, to more detailed questions on the artist’s works and 
inspiration. In the pre-processing steps, the corpus is spell-checked and re-formatted as 1845 question-answer pairs.
Some extendable and non-extendable QA pairs are provided in Table 2.

![Extendability](/img/2_classifier.png)

## Answer Extension

To generate a possible answer extension, we experiment with two different approaches: **a retrieval-based tf-idf model** 
and **a neural GPT-2 model**. Both models take the question-answer pair as a prompt. The goal is to generate an answer 
extension to the extendable short answer that gives additional information to the question and preserves Louisa 
Clement’s style. To further explore the questions of variation and style-adaptation, we also experiment with a T5 model to paraphrase the extension.

As a baseline, we employ a retrieval-based tf-idf model (Salton and Buckley, 1988). This model retrieves a paragraph 
from the Extension Corpus that is consistent with a question and a short answer. We use the paragraphs from the 
*Extension Corpus* as a retrieval base; the model is limited to this corpus and does not generate new text.

Each paragraph in the *Extension Corpus* has been annotated with its main topic words that generalize the content of 
the paragraph. Often they are not expressed explicitly in the paragraph and are introduced by us based on our 
understanding of the paragraph. As shown in Table 3 below, the *Extension Corpus* is composed of three types
of data provided to us by Louisa Clement:

1. *Professional emails* written by Louisa Clement in which she describes the organization of exhibitions, exchanges 
   ideas with other artists, describe her works, talks about her inspiration etc.
2. *Interviews*, in which the artist describes the background of her work, as well as her development as an artist in 
   her own words.
3. *A series of articles* written by critics and journalists about the various works of Louisa Clement, where they 
   place Louisa Clement’s work in the general historical and artistic context and analyze her work.
   
In the original form, this corpus contains 1596 sentences; 1156 from the emails, 330 from the articles, and 110 from 
the interviews.

![AnswerExtension](/img/3_ae.png)

Prior to pre-processing, we identify four main content topics covered by this corpus:

* artwork/exhibition production and description
* sociopolitical background/inspiration
* artistic ideas/concepts
* artistic formation and background

The implementation of the various models we experimented with is in the `extension` directory. The pre-processing steps
perform the tasks listed in Table 3 above.

For our task, the *GPT2LMHeadModel* is fine-tuned on the *Extension Corpus* described above. The extension dataset is 
loaded as a *datasets.Dataset* object. After that we run *GPT2Tokenizer* on the Dataset object. The dataset is split 
into train, dev and testsets. Out of the 364 paragraphs in the extension dataset, 80 % data (292 paragraphs) is assigned to the train set, and 20 % 
are equally divided between dev and test sets (36 paragraphs each). We let the training run for 30 epochs, the final 
eval loss is 4.478, the perplexity is 88.0843.

Since one of our goals is to generate a varied answer extension similar to what a human would produce, we additionally apply the T5 model (Raffelet, 2019) fine-tuned for the question paraphras-ing task (Goutham, 2020) on a large-scale Quora question dataset (Iyer et al., 2017) that consists of over 149,000 samples.  This task is linguistically quite similar to ours, since we also need to express the same meaning of question-style phrases with other words.  Paraphrasing is especially relevant for the extensions retrieved with the tf-idf model, as these lack variations for repeated questions. We did not try out different target lengths for the paraphrase, however, this could be a viable tool to further adapt the functionality of the Answer Extension module.

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

We solicited human evaluators to judge the quality of the extensions. To this end, we randomly select 30 question-answer 
pairs from the *gold-extension* corpus. We then select the best-scoring model from the automatic evaluation --- 
the tf-idf model --- and its paraphrased counterpart --- tf-idf + T5 --- as first and second condition, and generate 
extensions for the 30 samples. We also include our gold extensions as a third condition for comparison.

The extensions are shuffled and randomly distributed across three survey forms (A, B and C), such that each question-
answer pair appears once per survey form and every form contains extensions from each of the three conditions. 
Evaluators, who are blinded from the sample selection process, are asked to judge the quality of the samples with 
respect to 4 different criteria (as described in Table 5 above), on a scale from 1-5; 5 being the ideal score and 1 
the least desirable.

The script to randonly sample and shuffle the question-answer pairs for the human evaluation is `human_eval_samples.py`.
The sample ID's for the survey forms A, B, C are in the `/evaluation` directory. The questions and shuffle answer 
extensions for the 3 forms (A, B, C) are in the `/human_eval_files` directory.

## Results
![Results](/img/5_results.png)

According to the BLEURT scores in Table 7 all models (tf-idf, tf-idf+T5, and GPT-2) score close to -1. Running gold 
against gold gives a relative optimal score of 0.82 for this task. tf-idf extensions have the highest scores (-0.99).

Human evaluators rate the gold extensions higher than our models. *Grammaticality* and *fluency* ratings are higher than
*coherence* ratings, *appropriateness* is rated lowest. Figure 3 shows the correlation between criterion and score. 
Although human evaluators rate extensions from tf-idf and T5 model consistently lower across all criteria, 
the differences in mean rating scores for *grammaticality*, *fluency*, and *coherence* are not statistically significant. 
The only statistically significant difference is in the mean rating scores of the *appropriateness* criterion.

As demonstrated by some examples shown here in Table 8, we notice that the models show the following types of errors:

* grammatical mistakes
* misinterpretation of semantic meanings: semantic ambiguity leads to the wrong use of certain words
* topic-inappropriateness: the extension does not cover the same topic as the original question and answer. 
  
This is the case for all three models and is in accordance with the results from human evaluation.

## Future Work

With more time for data annotation and model implementation, we are interested in exploring the following:

* Incorporate knowledge graph into text generation (e.g. train GPT2 based on the KG)
  
* Extend the classifier to perform topic classification 
  
* Further fine-tuning of the T5 model on Louisa’s corpus for more effective paraphrasing

# Running the code

Below are steps we explored that may or may not have ended up in the final work of the project.

## Classifier

You can find information on how to run the classifier in the classifier README in the corresponding folder. Data needed to run the classifier can be found in the `data` directory.

## Extension

Information on how to run the respective extension models can be found in the folders in `extension`.

## Evaluation

You can find information on how to run our evaluation in the README in the corresponding folder.

## Preliminary Experiments: Knowledge Graph About Luisa's Works

Naive approach to extract concepts for building the knowledge base from articles about Luisa's work.

### Stats

After pre-processing, resultant corpus consists of 350 sentences.

### Pre-processing Scripts
- `preprocess_for_LDA.py` is used for preprocessing text for LDA Topic Modeling
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

## Preliminary Experiments: Topic Modeling

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
`preprocess_style-transfer.py` removes punctuations and re-format foreign punctuation marks.

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
