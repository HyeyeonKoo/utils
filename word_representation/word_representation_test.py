#-*-coding:utf-8-*-

import unittest
from simple_encoding import *
from word_doc_matrix import *



class EncodingTest(unittest.TestCase):
    def test_one_hot_encoding(self):
        morphs_sentences = ['안녕 하 세요 .', '반가워요 .', '오늘 날씨 정말 좋 지 않 나요 ?']
        
        word_index, index_word = get_word_dictionary(morphs_sentences)
        self.assertIn("날씨", word_index.keys())
        self.assertEqual(list(word_index.keys())[4], index_word[4])

        encoding = one_hot_encoding(["안녕", "하", "세요"], word_index)
        self.assertEqual(encoding[0][0], 1)


    def test_get_tf_idf(self):
        morphs_sentences = ['안녕 하 세요 .', '반가워요 .', '오늘 날씨 정말 좋 지 않 나요 ?']
        word_index, _ = get_word_dictionary(morphs_sentences)
        tokens = list(word_index.keys())
        
        tf = get_tf(tokens, morphs_sentences)
        self.assertEqual(list(tf[3]), [1, 1, 0])

        df = get_df(tokens, morphs_sentences)
        self.assertEqual(list(df[0]), [1, 1, 1])

        tf_idf = get_tf_idf(tokens, morphs_sentences)
        self.assertEqual(tf_idf[3][0], 0.5)


    def test_get_context_matrix(self):
        morphs_sentences = ['안녕 하 세요 .', '반가워요 .', '오늘 날씨 정말 좋 지 않 나요 ?']
        word_index, index_word = get_word_dictionary(morphs_sentences)

        context = get_context_matrix(word_index, morphs_sentences)
        self.assertEqual(context[0][1], 1)


if __name__ == "__main__":
    unittest.main()