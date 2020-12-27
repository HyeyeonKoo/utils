#-*-coding:utf-8-*-

import numpy as np
import re


def get_tf_idf(tokens, docs):
    tf = get_tf(tokens, docs)
    df = get_df(tokens, docs)

    return tf / df


def get_tf(tokens, docs):
    matrix = np.zeros((len(tokens), len(docs)))

    for i in range(len(tokens)):
        for j in range(len(docs)):
            if tokens[i] in ["^", "$", ".", "*", "+", "?", 
                "[", "]", "{", "}", "(", ")", "|"]:
                matrix[i][j] = len(re.findall("\\" + tokens[i], docs[j]))
            else:
                matrix[i][j] = len(re.findall(tokens[i], docs[j]))

    return matrix


def get_df(tokens, docs):
    matrix = np.zeros((len(tokens), len(docs)))

    for i in range(len(tokens)):
        count = 0

        for j in range(len(docs)):
            if tokens[i] in docs[j]:
                count += 1

        matrix[i] = np.tile(count, len(docs))

    return matrix


def get_context_matrix(word_index, docs, window=2):
    tokens = list(word_index.keys())
    matrix = np.zeros((len(tokens), len(tokens)))

    for i in range(len(tokens)):
        for j in range(len(docs)):
            doc_tokens = docs[j].split(" ")

            for k in range(i - window, i + window + 1):
                if i != k:
                    try:
                        matrix[i][word_index[doc_tokens[k]]] += 1
                    except IndexError:
                        continue
                    except KeyError:
                        continue

    return matrix

                    

