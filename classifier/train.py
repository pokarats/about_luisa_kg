import torch
import torch.nn as nn
import pickle

with open("reponses_encoded.pickle", "rb") as fread:
    reader = pickle.Unpickler(fread)
    data = reader.load()


class Model(nn.Module):
    def __init__(self, batchSize=128, learningRate=0.001):
        super().__init__()

        self.layers = [
            nn.Linear(768, 1),
            nn.Sigmoid()
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
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=data, batch_size=self.batchSize, shuffle=True
        )

        for epoch in range(epochs):
            self.train(True)
            for sample, true_class in self.dataLoader:
                output = self(sample).reshape(-1)
                loss = nn.BCELoss()(output, true_class)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.train(False)

                hits = 0
                for sample, true_class in data:
                    with torch.no_grad():
                        output = self(sample)
                    if (output >= 0.5) == (true_class == 1):
                        hits += 1


                    print(f"epoch {epoch}, accuracy {hits / len(data) * 100:.2f}%")

model = Model()
model.trainModel(data, 1000)