from numpy import save
import torch
import torch.nn as nn
import pickle
import random

with open("reponses_encoded.pickle_tmp", "rb") as fread:
    reader = pickle.Unpickler(fread)
    data = reader.load()


class Model(nn.Module):
    def __init__(self, batchSize=99999, learningRate=0.001):
        super().__init__()

        self.layers = [
            nn.Linear(768, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        ]
        self.all_layers = nn.Sequential(*self.layers)
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.all_layers(x)

    def trainModel(self, data, epochs):
        random.shuffle(data)
        dataTrain = data[100:]
        dataDev = data[:100]
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=dataTrain, batch_size=self.batchSize, shuffle=True
        )

        for epoch in range(epochs):
            self.train(True)
            for sample, true_class in self.dataLoader:
                output = self(sample).reshape(-1)
                loss = nn.BCELoss()(output, true_class)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 1 == 0:
                self.train(False)

                hits = 0
                hitsTP = 0
                hitsFP = 0
                totalP = 0
                threshold = 0.5
                for sample, true_class in dataDev:
                    with torch.no_grad():
                        output = self(sample)
                    if (output >= threshold) == (true_class == 1):
                        hits += 1
                    if (output >= threshold) and (true_class == 1):
                        hitsTP += 1
                    if (output >= threshold) and (true_class == 0):
                        hitsFP += 1
                    if true_class == 1:
                        totalP += 1

                precision = 0 if hitsFP == 0 and hitsTP == 0 else hitsTP / (hitsTP + hitsFP)
                print(f"epoch {epoch}, accuracy {hits / len(dataDev) * 100:.2f}%, Precision {precision * 100:.2f}%")

model = Model()
model.trainModel(data, 1000)
torch.save(model, "classifier_model.pt")