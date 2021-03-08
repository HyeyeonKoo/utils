#-*-coding:utf-8-*-

import torch.nn as nn
from torchcrf import CRF


"""
Model class
Model is composed of embedding, feed forward network and conditional random field layer.
Embedding layer is KoBERT model from SKTBrain. (https://github.com/SKTBrain/KoBERT)
CRF layer is from pytorch-crf. (https://pytorch-crf.readthedocs.io/en/stable/)

* Arguments
    - embedding : Embedding model
    - embedding_dropout : Probability of embedding output's dropout.
    - ff_hidden_size :  First shape of feed forward network.
    - output_class_size : Second shape of feed forward network
"""
class NaverNerModel(nn.Module):
    def __init__(
        self,
        embedding,
        embedding_dropout=0.1,
        ff_hidden_size=768,
        output_class_size=29
    ):
        super().__init__()

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.ff_network = nn.Linear(ff_hidden_size, output_class_size)
        self.crf = CRF(num_tags=output_class_size, batch_first=True)


    def forward(self, x, label=None):
        x_ = self.embedding(x)
        x_ = self.embedding_dropout(x_)
        x_ = self.ff_network(x_)
        
        if label:
            x_, result = (-1) * self.crf(x_, label), self.crf.decode(x_)
        else:
            result = self.crf.decode(x_)

        return x_, result
