from transformers import AutoTokenizer, AutoModel
import pickle

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-cls-token", return_dict=True, output_hidden_states=True)
model.train(False)
data = []

with open("responses_ann_new.txt", encoding="UTF-8") as f:
    datapoints = f.readlines()
    for i in range(len(datapoints)):
        datapoints[i] = datapoints[i].split(sep=";")

for a, b in datapoints:
    encoded_input = tokenizer(a, padding=True, truncation=True, max_length=128, return_tensors='pt')
    output = model(**encoded_input)
    data.append((output[0][0, 0], b*1))
with open("reponses_encoded.pickle", "wb") as f:
    pickler = pickle.Pickler(f)
    pickler.dump(data)
