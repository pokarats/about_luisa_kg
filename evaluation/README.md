# Preparation for Human Evaluation
## Sample Statistics
- 103 test queries and extensions
- Random sample of 30 test queries for human evaluation
- 3 sets of extensions to choose from, each set representing 1/3 of the
30 test samples for human evaluation
## Selection for Versions to be Compared
We have the following versions of outputs whose qualities we need to determine:
- baseline candidates : 
    - tf-idf
    - gpt-2
- para-phrased version of the base-line outputs (via T5 model)
- gold or reference outputs (from human annotation)

### Automatic Evaluation: BLEURT

**Requirements:** Python 3, Tensorflow (>=1.15) and the library tf-slim (>=1.1).
*See Installation steps in [BLEURT GitHub](https://github.com/google-research/bleurt)*

To select between the baseline candidates we use [BLEURT](https://github.com/google-research/bleurt) 
([Sellam et al., 2020](https://arxiv.org/abs/2004.04696)) to compare the two baseline outputs against the gold outputs. 
The model with the higher average BLEURT score is chosen for further human evaluation.

For details on installation of the BLEURT module, please consult instructions in the 
[BLEURT's GitHub Repo](https://github.com/google-research/bleurt). For a brief explanation of the BLEURT evaluation, 
please refer to [BLEURT's blog post](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html).

The BLEURT scores for all samples are also included in ``auto_eval_scores`` directory.


## Sampling and Shuffling
- For reproducibility, pseudo random sampling was done with a seed == 35
- sample ID's (line numbers) for each of the 3 output sets (baseline, gold, paraphrased) are saved to the following 
  files:
    - ``1.txt``
    - ``2.txt``
    - ``3.txt``
- The whole 30 samples are then shuffled to generate 3 sets of evaluation surveys as follows
    - Eval A, with the sample ID's for each set as: 1=gold, 2=base, 3=paraphrased
    - Eval B, with the sample ID's for each set as: 1=base, 2=paraphrased, 3=gold
    - Eval C, with the sample ID's for each set as: 1=paraphrased, 2=gold, 3=base
- The outputs from the 3 categories (gold, base, paraphrase) are randomly shuffled such that the orders
  between Eval A/B/C are not discernible.
- The order of the queries are consistent across Eval A, B, C; the shuffled ID's are saved in ``shuffled_idx.txt``
- The shuffled samples for the 3 Eval sets are found in ``human_eval_files`` directory.

## Human Evaluation Set-up

- The 3 Evaluation surveys versions (A,B,C) are distributed and collected via Google Forms (links here). 
- Each set of outputs are evaluated X times; total Y times.
- Each evaluator only sees the queries once. i.e. No evaluators are to complete more than 1 version of the Evaluation 
Survey form
- Evaluators are blinded to the fact that there are outputs from different model types in the survey.

## Post-Evaluation Analysis