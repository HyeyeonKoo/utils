#-*-coding:utf-8-*-

import torch


def l1_dist(a, b):
    return ((a - b).abs()).sum()


def l2_dist(a, b):
    return ((a - b) ** 2).sum().sqrt()


def inf_dist(a, b):
    return ((a - b).abs()).max()


def cos_sim(a, b):
    return (a - b).sum() / ((a ** 2).sum().sqrt() * (b ** 2).sum().sqrt())


def jaccard_sim_with_element(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def jaccard_sim_with_number(a, b):
    stack = torch.stack([a, b])
    return stack.min(dim=0)[0].sum() / stack.max(dim=0)[0].sum()