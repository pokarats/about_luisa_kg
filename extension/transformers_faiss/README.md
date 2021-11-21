### Requirements
The following packages should be installed to run the models: 
- torch==1.8.1
- transformers==3.3.1
- sentence-transformers==0.3.8
- pandas==1.1.2
- faiss-cpu==1.6.1
- numpy==1.19.2
- folium==0.2.1
- streamlit==0.62.0
- -e .
### Directory structure
```
transformers_faiss
├── data
│   ├── corpus_gpt2_final.txt
│   └── qa.txt
├── README.md
├── requirements.txt
├── results
│   └── transformer_faiss_output.txt
└── transformer_faiss_louisa.ipynb
```