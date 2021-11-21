### Requirements
To run the model you need an environment with the following packages installed:- transformers
- transformers==2.8.0
- torch
### Directory structure
```
t5_model_paraphrase
│ 
└───data
│   └───gpt2_output.txt
│   └───tf_idf_output.txt
└───README.md
└───requirements.txt
└───results
    └───gpt2_t5_paraphrase.txt
    └───t5_paraphrase_new.txt
    └───t5_paraphrase.txt
└───t5_model_paraphrase.py
```
Files in the `data` folder are required to replicate the results of paraphrasing tf-idf output and gpt2 output.
