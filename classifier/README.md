## General information

The classifier now uses 'sentence-transformers/multi-qa-mpnet-base-dot-v1', instead of 'sentence-transformers/bert-base-nli-cls-token' for qa-pair encodings.

The neural model is a simple two-layer model with Tanh and Sigmoid nonlinearities. We use BCEloss. Lower batch size decreases precision, as well as a larger learning rate.

Accuracy >90%; Precision varies between 60%-100%. The classifier has a very small number of false positives.

`models` contains some pretrained models. The `classifier_old_model` was used for the report.

## Classification pipeline

1. Preprocessing: preprocess.py 
2. Encoding: create_embeddings.py 
3. Training: model.py 
4. Predicting: model.py --predict 

### The modules

#### preprocessing.py

```
Preprocess Classifier Training Data

optional arguments:
  -h, --help            show this help message and exit
  --unannotated         Preprocess annotated or unannotated data.
  -u UNANN_FILE, --unann_file UNANN_FILE
                        File with question-answer pairs; one pair per line split with <answer>
  -a ANN_FILE, --ann_file ANN_FILE
                        File with annotated qa-pairs. Form: When were you born? <answer> 1999 [<answer> ...]#0
  -p DATA_PATH, --data_path DATA_PATH
                        Path to the data folder. Default: data/classifier/ (This assumes the classifier is executed from the root directory.)
  --sep_ann SEP_ANN     Separator of QA-pairs and annotations. Default: #
  --sep_ans SEP_ANS     Separator of questions and answers. Default: <answer>
```
This module has to be executed for both the annotated training data, and the unannotated data which is to be predicted. In unannotated mode, duplicate questions, which are also present in the annotated data, are removed if there exists an ANN_FILE csv file with annotations.

#### create_embeddings.py

```
Create Embeddings for Data

optional arguments:
  -h, --help            show this help message and exit
  -m PRETRAINED_MODEL, --pretrained_model PRETRAINED_MODEL
                        Pretrained bert model. Default: 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
  -f FILE, --file FILE  File with preprocessed question-answer pairs; one pair per line split with <answer>
  -d DATA_PATH, --data_path DATA_PATH
                        Path to the data folder. Default: data/classifier/
```

This module has to be executed for both the annotated training data, and the unannotated data which is to be predicted. `create_embeddings_old.py` was used for the paper, but does not work anymore, as the old bert-model is deprecated.

#### model.py

```
Train neural classifier and predict annotations

optional arguments:
  -h, --help            show this help message and exit
  --predict             mode for predicting annotations for unannotated file
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input file. In train mode pickled training data. In predict mode: name of pickle, original text file and output file. Default: annotated_qa_pairs
  -m MODEL_PATH, --model_path MODEL_PATH
                        Output model name. Default: classifier/models/classifier.pt
  -d DATA_PATH, --data_path DATA_PATH
                        Path to the data folder. Default: data/classifier/
```

In train mode, a model is trained based on the `INPUT_FILE` and saved to `MODEL_PATH`. In prediction mode (use `--prediction` flag), a model is loaded from `MODEL_PATH` to predict annotation. __Important:__ For prediction, there have to be both a `INPUT_FILE.pickle` and a `INPUT_FILE.txt` (from the previous steps) at `DATA_PATH`.