#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/8 下午9:21
  @ Author   : Vodka
  @ File     : Skip-gram .py
  @ Software : PyCharm
"""
import collections
import math
import random


def read_data(path):
    """
    读取数据
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]
    counter = collections.Counter([token for st in raw_dataset for token in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    idx_to_token = [token for token, num in counter.items()]  # [idx] = token
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}  # [token] = idx
    dataset = [[token_to_idx[token] for token in st if token in token_to_idx] for st in raw_dataset]
    return dataset, idx_to_token, token_to_idx, counter


def subsample_dataset(dataset):
    """
    原始数据集词频分布差距太大 容易对结果产生影响 因此需要二次采样
    :param dataset:
    :return:
    """
    total_token_num = sum([len(st) for st in dataset])  # 训练集中所有的token数量
    new_dataset = [[idx for idx in st
                    if random.uniform(0, 1) > 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * total_token_num)]
                   # 以右侧概率丢弃该词，越常见词越容易被丢弃从而减少词频
                   for st in dataset]
    return new_dataset


def get_centers_and_contexts(dataset, max_window_size):
    """
    提取中间词 以及其对应的背景词
    :param dataset:
    :param max_window_size:
    :return:
    """
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


dataset, idx_to_token, token_to_idx, counter = read_data('./data/ptb.train.txt')
dataset = subsample_dataset(dataset)
centers, contexts = get_centers_and_contexts(dataset, 5)
