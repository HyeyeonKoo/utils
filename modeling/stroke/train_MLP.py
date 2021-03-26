#-*-coding:utf-8-*-

"""
단순 퍼셉트론 모델로 신경망을 구성해 학습시켰을 때, 
GPU를 사용하더라도 속도가 느렸고 트리 기반 ML 모델들보다 성능이 좋지 않음
층 추가, dropout 등 추가해보았지만 단일 층보다 좋은 결과를 보지 못함
결국 선형회귀임
다른 아키텍처를 사용해 볼 수도 있지만, 이러한 경우에는 단순한 ML을 이용하는 것이 좋을 듯
"""


import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

torch.manual_seed(111)


class StrokeDataset(Dataset):

    def __init__(self, data_path, label_path):
        self.data = self.get_csv(data_path)
        self.label = self.get_csv(label_path)


    def get_csv(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            for line in csv_reader:
                data.append([list(map(float, line))])

        return data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return {
            "input": torch.tensor(self.data[index]),
            "label": torch.tensor(self.label[index], dtype=torch.float)
        }


class StrokeModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(21, 1)

    def forward(self, x):
        x_ = self.linear1(x)
        return torch.sigmoid(x_)


def train(
    train_data_path, train_label_path,
    test_data_path, test_label_path,
    batch_size, epochs, verbose
):
    train_dataset = StrokeDataset(train_data_path, train_label_path)
    test_dataset = StrokeDataset(test_data_path, test_label_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = StrokeModel().cuda()
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print("epoch\tloss\ttrain_acc\ttest_acc")
    for epoch in range(epochs):
        total_loss = 0
        for item in train_dataloader:
            x = item["input"].cuda()
            y = item["label"].cuda()

            optimizer.zero_grad()
            output = model(x)
            
            loss = loss_fn(output, y)
            total_loss += loss.item()
            loss.backward()

            optimizer.step()

        if epoch == 0 or epoch % verbose == 0:
            # epoch
            print(epoch, end="\t")
            
            # train loss
            print(round(total_loss/batch_size, 2), end="\t")
            total_loss = 0
            
            # train acc
            whole_output = model(torch.tensor(train_dataset.data).cuda())
            whole_label = torch.tensor(train_dataset.label).cuda()

            train_acc = round(((whole_output>=0.9)==whole_label).sum().item() / whole_label.shape[0], 2)
            print(train_acc, end="\t\t")

            # test acc
            test_output = model(torch.tensor(test_dataset.data).cuda())
            test_label = torch.tensor(test_dataset.label).cuda()

            test_acc = round(((test_output>=0.9)==test_label).sum().item() / test_label.shape[0], 2)
            print(test_acc, end="\t\t")

            print()

    return model


if __name__=="__main__":
    model = train(
        "data/train_data.csv", "data/train_label.csv",
        "data/test_data.csv", "data/test_label.csv",
        batch_size=8, epochs=200, verbose=10
    )

    torch.save(model, "model/mlp.pt")
