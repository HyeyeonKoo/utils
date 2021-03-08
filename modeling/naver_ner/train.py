#-*-coding:utf-8-*-

"""
Train Ner using KoBERT, pytorch-crf.
Use Naver Ner dataset.
"""

import torch
import os
from datetime import datetime

from torch.utils.data import DataLoader

from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from data import NaverNerDataset
from model import NaverNerModel
from trainer import NaverNerTrainer


"""
Use KoBERT model, vocabulary and tokenizer as an embedding model.
https://github.com/SKTBrain/KoBERT
"""
embedding_model, vocab = get_pytorch_kobert_model()
token_index = vocab.token_to_idx

tokenizer_ = get_tokenizer()
tokenizer = SentencepieceTokenizer(tokenizer_)


"""
Make train dataset.
"""
start = datetime.now()

train_dataset = NaverNerDataset(
    token_index=token_index,
    tokenizer=tokenizer,
    data_path=os.path.join(os.path.abspath(""), "dataset/naver_ner.txt"),
    max_len=512
)

end = datetime.now()
print("Make Dataset : ", str(end - start))


"""
Create Dataloader.
"""
start = datetime.now()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=5
)

end = datetime.now()
print("Create Dataloader : ", str(end-start))


"""
Initialize Model.
"""
start = datetime.now()

model = NaverNerModel(
    embedding=embedding_model,
    embedding_dropout=0.1,
    ff_hidden_size=768,
    output_class_size=29
)

end = datetime.now()
print("Initialize Model :", str(end-start))
print(model)


"""
Initialize Trainer
"""
start = datetime.now()

trainer = NaverNerTrainer(
    train_data_loader=train_data_loader,
    model=model,
    learning_rate=6e-6,
    warmup_step=100,
    adam_ep=6e-4, adam_beta1=0.9, adam_beta2=0.98, weight_decay=0.01
)

end = datetime.now()
print("Initialize Trainer :", str(end-start))


"""
Train
"""
start = datetime.now()

epoch = 5
for each in range(epoch):
    trainer.train(
        each,
        trainable=True,
        train_verbose_step=100
    )

    trainer.save(os.path.join(os.path.abspath(""), "output", str(each)))
