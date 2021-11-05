from transformers import AutoTokenizer, AutoModel
import pickle
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-cls-token", return_dict=True, output_hidden_states=True)
model.train(False)
data = []

with open("other_responses.txt", encoding="UTF-8") as f:
    datapoints = [x.strip().split('#') for x in f.readlines()]
    try:
        datapoints = [(txt, float(x)) for txt, x in datapoints]
    except:
        for txt, x in datapoints:
            print(txt)

for a, b in datapoints:
    encoded_input = tokenizer(a, padding=True, truncation=True, max_length=128, return_tensors='pt')
    output = model(**encoded_input)
    data.append((output[0][0, 0].float(), np.float32(b)))
with open("other_reponses_encoded.pickle_tmp", "wb") as f:
    pickler = pickle.Pickler(f)
    pickler.dump(data)