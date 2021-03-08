#-*-coding:utf-8-*-

"""
NaverNerDataset class.

* Arguments
    - token_index : {token1: 0, token2: 1, ...}
    - tokenier : For tokenizing each token or sentence, It is needed.
    - data_path : Read data from data_path
    - max_len : If number of tokens over the max_len, it
"""

from torch.utils.data import Dataset
import torch


class NaverNerDataset(Dataset):
    def __init__(self, token_index, tokenizer, data_path, max_len=512):
        self.token_index = token_index
        self.word_label_sentences = self.get_sentences(data_path)

        self.tokens, labels, label_types = self.get_token_label(tokenizer, max_len)
        self.label_index, self.index_label = self.get_label_index(label_types)
        self.labels = self.get_indexed_label(labels, max_len)


    def get_sentences(self, data_path):
        sentences = []
        with open(data_path, "r", encoding="utf-8") as f:
            sentences = f.read().split("\n\n")
        
        word_label = []        
        for sent in sentences:
            tokens = sent.split("\n")

            each_sentence = []
            for token in tokens:
                elements = token.split("\t")
                try:
                    each_sentence.append((elements[1], elements[2]))
                except:
                    continue

            word_label.append(each_sentence)

        return word_label


    def get_token_label(self, tokenizer, max_len):
        def get_index(token_):
            try:
                return  self.token_index[token_]
            except KeyError:
                return self.token_index["[UNK]"]

        tokens, labels = [], []
        label_types = set()
        for sent in self.word_label_sentences:
            
            sent_tokens, sent_labels = [], []
            for word, tag in sent:
                tokens_ = tokenizer(word)

                if tag == "-":
                    tag = "O"

                for i in range(len(tokens_)):
                    if i == 0:
                        sent_tokens.append(get_index(tokens_[i]))
                        sent_labels.append(tag)
                        label_types.add(tag)
                    else:
                        sent_tokens.append(get_index(tokens_[i]))

                        if tag == "O":
                            sent_labels.append(tag)
                        else:
                            tag_ = tag.replace("_B", "_I")
                            sent_labels.append(tag_)
                            label_types.add(tag_)
            
            tokens.append(self.get_pad_tokens(
                sent_tokens, max_len, self.token_index["[PAD]"]
            ))
            labels.append(sent_labels)

        return tokens, labels, list(label_types)


    def get_pad_tokens(self, sent_tokens, max_len, pad_index):
        if len(sent_tokens) < max_len:
            sent_tokens += [pad_index] * (max_len - len(sent_tokens))
        elif len(sent_tokens) > max_len:
            sent_tokens = sent_tokens[:max_len]

        return sent_tokens


    def get_label_index(self, label_types):
        label_index, index_label = {}, {}
        index = 0
        for label in label_types:
            label_index[label] = index
            index_label[index] = label
            index += 1

        return label_index, index_label


    # 정수 라벨
    def get_indexed_label(self, labels, max_len):
        new_label = []
        for sent in labels:

            sent_label = []
            for label in sent:
                sent_label.append(self.label_index[label])

            new_label.append(self.get_pad_labels(
                sent_label, max_len, self.label_index["O"]
            ))

        return new_label

    # 원-핫 라벨
    # def get_indexed_label(self, labels, max_len):
    #     new_label = []
    #     label_length = len(self.label_index)

    #     pad_label = [0] * label_length
    #     pad_label[self.label_index["O"]] = 1

    #     for sent in labels:

    #         sent_label = []
    #         for label in sent:
    #             new_ = [0] * label_length
    #             new_[self.label_index[label]] = 1
    #             sent_label.append(new_)

    #         new_label.append(self.get_pad_labels(
    #             sent_label, max_len, pad_label
    #         ))

    #     return new_label    


    def get_pad_labels(self, sent_label, max_len, pad_label):
        if len(sent_label) < max_len:
            sent_label += [pad_label] * (max_len - len(sent_label))
        elif len(sent_label) > max_len:
            sent_label = sent_label[:max_len]

        return sent_label


    def __len__(self):
        return len(self.word_label_sentences)


    def __getitem__(self, index):
        return {
            "input": torch.tensor([self.tokens[index]]),
            "label": torch.tensor(self.labels[index])
        }
