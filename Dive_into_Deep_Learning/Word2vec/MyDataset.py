#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/9 下午9:32
  @ Author   : Vodka
  @ File     : MyDataset .py
  @ Software : PyCharm
"""
import torch


class MyDataset(torch.utils.data.Dataset):
    """
    封装迭代批量读取数据DataLoader
    """
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)
