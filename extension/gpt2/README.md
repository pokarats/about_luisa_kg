####Requirements
- transformers
- datasets
- torch
- sentencepiece
- pyyaml
####Directory structure
.
├── converter_json_to_corpus_gpt_without_topics.py
├── data
        ├── corpus_gpt2_final.txt
│       └── qa.txt
├── gpt2_model_louisa.py
├── README.md
├── requirements.txt
└── results
    └── gpt2_output.txt

Files in the data folder are required to fine-tune the GPT2 model on the Louisa extension data and replicate the results.