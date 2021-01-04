#-*-coding:utf-8-*-

from data_processing import *
from model import NsmcLstm
import torch.optim as optim
import torch.nn as nn


def get_data(train_path, test_path):
    train_data, train_label = load_data(train_path)
    test_data, test_label = load_data(test_path)

    train_data, test_data = encoding(train_data, test_data, count_limit=3)

    cuda = torch.device("cuda")
    train_data = torch.tensor(train_data, requires_grad=True, dtype=torch.float32, device=cuda)
    test_data = torch.tensor(test_data, dtype=torch.float32, device=cuda)
    train_label = torch.tensor(train_label, dtype=torch.float32, device=cuda)
    test_label = torch.tensor(test_label, dtype=torch.float32, device=cuda)

    return train_data, train_label, test_data, test_label


def get_init_model(input_size, hidden_size):
    model = NsmcLstm(input_size=input_size, hidden_size=hidden_size)
    print(model)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    return model, loss_fn, optimizer


def train(train_path, test_path, epochs=10, verbos=1):
    train_data, train_label, test_data, test_label \
        = get_data(train_path, test_path)

    model, loss_fn, optimizer = get_init_model(
        input_size=train_data.shape[2],
        hidden_size=train_data.shape[2]
    )

    for i in range(epochs):
        model.train()
        output = model(train_data)
        loss = loss_fn(output, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 0 or i % verbos == 0:
            test_result = model(test_data)
            acc = ((test_result>=0.5) == test_label).float().sum() / len(test_label[0])
            print(i, " loss: ", loss.item(), " acc: ", acc.item())

    return model


if __name__ == "__main__":
    train(
        train_path="data/ratings_test.txt",
        test_path="data/ratings_test.txt",
        epochs=50
    )
