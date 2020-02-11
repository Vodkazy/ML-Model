#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/11 下午4:43
  @ Author   : Vodka
  @ File     : Encoder .py
  @ Software : PyCharm
"""
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_hiddens, num_layers, drop_prob=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        """
        :param inputs: (batch, time_steps)
        :param state:
        :return:
        """
        # 互换输入的样本维和时间步维
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)  # (seq_len, batch, input_size)
        return self.rnn(embedding, state)  # GRU的state是h, 而LSTM的是⼀个元组(h, c)
