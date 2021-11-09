from numpy import save
import torch
import torch.nn as nn
import pickle
import random
import argparse


def cla_parser():
    """
    Parse commandline arguments

    :return: parsed args
    """
    parser = argparse.ArgumentParser(description='Create Embeddings for Data')
    parser.add_argument('--predict', action='store_true', help='mode for predicting annotations for unannotated file')
    parser.add_argument('-i', '--input_file', default='annotated_qa_pairs', help='Input file. In train mode pickled training data. In predict mode: name of pickle, original text file and output file. Default: annotated_qa_pairs')
    parser.add_argument('-m', '--model_path', default='classifier/models/classifier.pt', help='Output model name. Default: classifier/models/classifier.pt')
    parser.add_argument('-d', '--data_path', default='data/classifier/', help='Path to the data folder. Default: data/classifier/')
    
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, batchSize=1000, learningRate=0.01):
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
                hitsFN = 0
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
                    if (output <= threshold) and (true_class == 1):
                        hitsFN += 1
                    if (output >= threshold) and (true_class == 0):
                        hitsFP += 1
                    if true_class == 1:
                        totalP += 1

                precision = 0 if hitsFP == 0 and hitsTP == 0 else hitsTP / (hitsTP + hitsFP)
                print(f"epoch {epoch}, TP {hitsTP} FP {hitsFP} FN {hitsFN} accuracy {hits / len(dataDev) * 100:.2f}%, Precision {precision * 100:.2f}%")

    def predict(self, args, other_data):
        with open(f"{args.data_path}{args.input_file}.txt") as fread:
            with open(f"{args.data_path}{args.input_file}_predictions.txt", "w") as fwrite:
                self.train(False)
                threshold = 0.5
                for sample, dummy_annotation in other_data:
                    with torch.no_grad():
                        output = self(sample)
                    if output >= threshold:
                        ann = "".join((fread.readline()[:-2], "1", "\n"))
                    else:
                        ann = "".join((fread.readline()[:-2], "0", "\n"))
                    fwrite.write(ann)    

def main():
    args = cla_parser()
    with open(f"{args.data_path}{args.input_file}.pickle", "rb") as fread:
        reader = pickle.Unpickler(fread)
        data = reader.load()
    if args.predict:
        model = torch.load(args.model_path)
        model.predict(args, data)
    else:
        model = Model()
        model.trainModel(data, 300)
        torch.save(model, args.model_path)

if __name__ == '__main__':
    main()
