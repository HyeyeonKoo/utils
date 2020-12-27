#-*-coding:utf-8-*-

import unittest
import torch
from similarity import *


a = torch.tensor([0.8, 0.5, -0.6])
b = torch.tensor([0.2, -0.7, 0.1])


class EncodingTest(unittest.TestCase):
    def test_l1_dist(self):
        dist = l1_dist(a, b)
        self.assertEqual(float(dist), 2.5)


    def test_l2_dist(self):
       dist = l2_dist(a, b)
       self.assertEqual(round(float(dist), 4), 1.5133)


    def test_inf_dist(self):
        dist = inf_dist(a, b)
        self.assertEqual(float(dist), 1.2)


    def test_cos_sim(self):
        sim = cos_sim(a, b)
        self.assertEqual(round(float(sim), 4), 1.3389)


    def test_jaccard_sim(self):
        sim = jaccard_sim_with_element(a, b)
        self.assertEqual(float(sim), 0)

        sim = jaccard_sim_with_number(a, b)
        self.assertEqual(round(float(sim), 4), -0.7857)


if __name__ == "__main__":
    unittest.main()

