#-*-coding:utf-8-*-

import unittest
from sample_data import *
from ir_eval import MRR
from ir_eval import MAP
from ir_eval import NDCG


class IrEvalTest(unittest.TestCase):

    def test_mrr(self):
        mrr = MRR(query=query, result=result, correct=correct, match="jaccard")
        mrr.eval()

        self.assertEqual(1.0, mrr.score)
        self.assertIn("이순신은 조선 중기의 무신이었다.", 
            mrr.detail["이순신 장군은 누구인가?"]["result"].keys())


    def test_map(self):
        map_ = MAP(query=query, result=result, correct=correct)
        map_.eval()

        self.assertEqual(0.725, map_.score)
        self.assertEqual(0.4, 
            map_.detail["한글은 누가 만들었는가?"]["result"]["세종대왕의 애민정신이 깃든 위대한 유산, 한글"])


    def test_NDCG(self):
        ndcg = NDCG(query=query, result=result, correct=correct, rank_score=rank_score)
        ndcg.eval()

        self.assertEqual(10.27, round(ndcg.correct_dcg, 2))
        self.assertEqual(0.53, round(ndcg.score, 2))
        self.assertEqual(1.55,
            round(ndcg.detail["한글은 누가 만들었는가?"]["result"]["세종대왕의 애민정신이 깃든 위대한 유산, 한글"], 2))


if __name__ == "__main__":
    unittest.main()