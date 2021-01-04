#-*-coding:utf-8-*-

from collections import Counter
import torch
from konlpy.tag import Mecab
m = Mecab()


def load_data(path):
    data = []
    label = []

    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            _, data_, label_ = line[:-1].split("\t")
            if data_ == data_:
                data.append(data_)
                label.append([float(label_)])

    return data, [label]


def encoding(train_data, test_data, count_limit=2, len_limit=None, batch_size=1):
    word_index = get_word_index(train_data + test_data, count_limit)

    train_encode = token_encoding(train_data, word_index)
    test_encode = token_encoding(test_data, word_index)

    if len_limit:
        max_len = len_limit
    else:
        max_len = max([len(el) for el in train_encode + test_encode])

    train_pad = padding(train_encode, word_index, max_len)
    test_pad = padding(test_encode, word_index, max_len)

    train_batch = torch.reshape(torch.tensor([train_pad]), 
        (batch_size, int(len(train_pad)/batch_size), max_len))
    test_batch = torch.reshape(torch.tensor([test_pad], dtype=torch.float32), 
        (1, int(len(test_pad)/batch_size), max_len))

    return train_batch, test_batch


def get_word_index(data, limit=2):
    tokens = []
    for sent in data:
        tokens += m.morphs(sent)
    token_count = Counter(tokens)

    word_index = {"[PAD]": 0, "[UNK]": 1}
    index = 2

    for k, v in token_count.items():
        if v >= limit:
            word_index[k] = index
            index += 1

    return word_index


def token_encoding(data, word_index):
    data_encode = []

    for sent in data:
        tokens = m.morphs(sent)

        token_encode = []
        for token in tokens:
            try:
                token_encode.append(word_index[token])
            except KeyError:
                token_encode.append(word_index["[UNK]"])

        data_encode.append(token_encode)

    return data_encode


def padding(data, word_index, max_len):
    data_pad = []

    for sent in data:
        pad_num = max_len - len(sent)

        if pad_num > 0:
            data_pad.append(sent + [word_index["[PAD]"]]*pad_num)
        else:
            data_pad.append(sent[:max_len])

    return data_pad
