#-*-coding:utf-8-*-

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


class NaverNerTrainer:
    def __init__(
        self,
        train_data_loader, model, 
        learning_rate, warmup_step,
        adam_ep, adam_beta1, adam_beta2, weight_decay
    ):
        self.train_data_loader = train_data_loader

        self.device = self.get_device()
        self.model = self.set_model(model)

        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_ep,
            weight_decay=weight_decay
        )

        self.schedule = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda lr_step: (lr_step / warmup_step) * learning_rate \
                if lr_step < warmup_step else learning_rate
        )

        self.loss_fn = nn.NLLLoss(ignore_index=0)


    def get_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        print("device :", device)

        return device


    def set_model(self, model):
        if self.device == "cuda:0":
            model = model.to(self.device)
        
        if torch.cuda.device_count() > 1:
            return nn.parallel.DistributedDataParallel(model)

        return model


    def train(self, epoch, trainable=True, train_verbose_step=100):
        for i, data in enumerate(self.train_data_loader):
            data = {k: v.to(self.device) for k, v in data.items()}

            lr = self.schedule.get_last_lr()[0]
            loss, result = self.model.forward(data["input"], data["label"])

            if trainable:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
            
            if i == 0 or i % train_verbose_step == 0 or i == len(self.train_data_loader) - 1:
                print({
                    "epoch": epoch,
                    "step": i,
                    "lr": lr,
                    "loss": loss.item()
                })

            data = {k: v.to("cpu") for k, v in data.itmes()}


    def save(self, epoch, path):
        torch.save(self.model.cpu(), path)
        self.model.to(self.device)
        print("Epoch :", epoch, "Save :", path)
