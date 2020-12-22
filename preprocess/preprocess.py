#-*-coding:utf-8-*-

import kss
from tokenizer import MorphTokenizer


def get_sentences(corpus, save=False, save_path="sentences.txt"):
    sentences = kss.split_sentences(corpus)

    if not save:
        return kss.split_sentences(corpus)

    save_iter_data(save_path, sentences)


def get_morphs_sentences(tokenizer, sentences, save=False, save_path="morphs.txt"):
    tokenizer = MorphTokenizer(tokenizer).tokenizer

    result = []
    for sent in sentences:
        result.append(" ".join(tokenizer.morphs(sent)))

    if not save:
        return result

    save_iter_data(save_path, result)


def save_iter_data(save_path, data):
    with open(save_path, "w", encoding="utf-8") as f:
        for element in data:
            f.write(element + "\n") 
