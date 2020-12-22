#-*-coding:utf-8-*-

from konlpy.tag import Mecab, Hannanum, Kkma, Komoran, Okt
import sentencepiece


class MorphTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = self.get_tokenizer(tokenizer)


    def get_tokenizer(self, tokenizer):
        tokenizer = tokenizer.lower()

        if tokenizer == "mecab":
            tokenizer = Mecab()

        elif tokenizer == "hannanum":
            tokenizer = Hannanum()

        elif tokenizer == "kkma":
            tokenizer = Kkma()

        elif tokenizer == "komoran":
            tokenizer = Komoran()

        elif tokenizer == "Okt":
            tokenizer = Okt()

        else:
            raise RuntimeError("Tokenizer must be the one of Mecab, Hannanum, Kkma, Komoran, Okt.")

        return tokenizer


class SentencePieceTokenizer:

    def __init__(self, input_f="input.txt", model_prefix="sp_model", vocab_size=32000,
        character_coverage=1.0, model_type="bpe"):
        self.train_param = {
            "input": input_f,
            "model_prefix" : model_prefix,
            "vocab_size": vocab_size,
            "character_coverage": character_coverage,
            "model_type": model_type
        }

        self.model = None


    def train(self):
        command = ("--input=%s --model_prefix=%s --vocab_size=%s --character_coverage=%s --model_type=%s"
            %(self.train_param["input"], self.train_param["model_prefix"], self.train_param["vocab_size"],
            self.train_param["character_coverage"], self.train_param["model_type"]))

        sentencepiece.SentencePieceTrainer.Train(command)

        self.model = sentencepiece.SentencePieceProcessor()
        self.model.Load(self.train_param["model_prefix"] + ".model")

    
    def tokenize(self, sentence):
        return self.model.EncodeAsPieces(sentence)


    def restore(self, segmented_sentence):
        return self.model.DecodePieces(segmented_sentence)


    def encode(self, sentence):
        return self.model.EncodeAsIds(sentence)


    def decode(self, encoded_sentence):
        return self.model.DecodeIds(encoded_sentence)


    # 필요한 기능 차후 추가