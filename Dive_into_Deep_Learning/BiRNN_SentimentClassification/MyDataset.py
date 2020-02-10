#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/10 下午7:20
  @ Author   : Vodka
  @ File     : MyDataset .py
  @ Software : PyCharm
"""
import torch


class MyDataset(torch.utils.data.Dataset):
    """
    封装迭代批量读取数据DataLoader
    """
    def __init__(self, contents, labels):
        self.contents = contents
        self.labels = labels

    def __getitem__(self, index):
        return (self.contents[index], self.labels[index])

    def __len__(self):
        return len(self.contents)