### Requirements
To run the model you need an environment with the following packages installed:- transformers
- datasets
- torch
- sentencepiece
- pyyaml
Alternatively, install the packages by running this command: <code> #pip install -U transformers datasets torch sentencepiece pyyaml
</code>
### Directory structure
```
gpt2
│ 
└───converter_json_to_corpus_gpt_without_topics.py   
└───data
│   └───corpus_gpt2_final.txt
│   └───qa.txt  
│
└───gpt2_model_louisa.ipynb
└───gpt2_model_louisa.py
└───README.md
└───results
    └───gpt2_output.txt
```
   
Files in the `data` folder are required to fine-tune the GPT2 model on the Louisa extension data and replicate the results.
