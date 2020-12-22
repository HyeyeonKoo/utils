#-*-coding:utf-8-*-

import unittest
from preprocess import *
from tokenizer import SentencePieceTokenizer


class PreprocessTest(unittest.TestCase):
    def test_split_sentence(self):
        sentences = get_sentences("안녕하세요. 반가워요. 오늘 날씨 정말 좋지 않나요?")
        self.assertEqual(['안녕하세요.', '반가워요.', '오늘 날씨     정말 좋지 않나요?'], sentences)


    def  test_morphs_sentences(self):
        sentences = ['안녕하세요.', '반가워요.', '오늘 날씨     정말 좋지 않나요?']
        morphs = get_morphs_sentences("mecab", sentences)
        self.assertEqual(['안녕 하 세요 .', '반가워요 .', '오늘 날씨 정말 좋 지 않 나요 ?'], morphs)


    def test_sentencepiece(self):
        sentences = get_sentences("안녕하세요. 반가워요. 오늘 날씨 정말 좋지 않나요?")
        morphs = get_morphs_sentences("mecab", sentences, save=True)
        
        spm = SentencePieceTokenizer(input_f="morphs.txt", vocab_size=24)
        spm.train()

        self.assertEqual(spm.tokenize("안녕하세요."), ['▁', '안', '녕', '하', '세', '요', '.'])
        self.assertEqual(spm.restore(['▁', '반', '가', '워', '요', '.']), "반가워요.")

        self.assertEqual(spm.encode("오늘 날씨 정말 좋지 않나요?"), 
            [3, 18, 11, 3, 9, 15, 3, 20, 12, 3, 21, 22, 3, 17, 8, 4, 6])
        self.assertEqual(spm.decode([3, 18, 11, 3, 9, 15, 3, 20, 12, 3, 21, 22, 3, 17, 8, 4, 6]), 
            "오늘 날씨 정말 좋지 않나요?")


if __name__ == "__main__":
    unittest.main()