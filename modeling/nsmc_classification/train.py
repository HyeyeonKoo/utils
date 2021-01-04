#-*-coding:utf-8-*-

from data_processing import *
from model import NsmcLstm
import torch.optim as optim
import torch.nn as nn


train_data, train_label = load_data("data/ratings_train.txt")
test_data, test_label = load_data("data/ratings_test.txt")

train, test = encoding(train_data, test_data, count_limit=3)

model = NsmcLstm(input_size=train.shape[2], hidden_size=train.shape[2])
print(model)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

epochs = 10
check = 1
for i in range(epochs):
    model.train()
    output = model(train)
    loss = loss_fn(output.view(-1, train.shape[2]), train_label.view(-1).long())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i == 0 or i % check == 0:
        result = output.data.numpy().argmax(axis=2)
        print(i, "loss: ", loss.item())
