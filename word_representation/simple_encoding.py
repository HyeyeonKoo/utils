#-*-coding:utf-8-*-

import numpy as np


def get_word_dictionary(tokenized_sentences):
    word_index, index_word = {}, {}
    index = 0

    for sent in tokenized_sentences:
        tokens = sent.strip().split(" ")

        for token in tokens:
            word_index[token] = index
            index_word[index] = token
            index += 1

    return word_index, index_word


def one_hot_encoding(tokens, word_index):
    encoding = np.zeros((len(tokens), len(word_index.keys())))

    for i in range(len(tokens)):
        try:
            encoding[i][word_index[tokens[i]]] = 1

        except KeyError:
            raise RuntimeError(tokens[i] + " is not in word_index.")

    return encoding