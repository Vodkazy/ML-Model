#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/11 下午9:52
  @ Author   : Vodka
  @ File     : Decoder .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn

from Dive_into_Deep_Learning.Attention.Simple_attention.AttentionModel import AttentionModel


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_hiddens, num_layers, attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = AttentionModel(2 * num_hiddens, attention_size)
        self.rnn = nn.GRU(num_hiddens + embedding_size, num_hiddens, num_layers,
                          dropout=drop_prob)  # GRU的输⼊包含attention输出的c和实际输⼊, 所以尺⼨是 2*embedding_size
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, current_inputs, state, encoder_states):
        """
        :param current_inputs:  (batch, )
        :param state:  (num_layers, batch, num_hiddens)
        :param encoder_states: (time_step, batch, num_hiddens)
        :return:
        """
        c = self.attention.forward(encoder_states, state[-1])  # 使⽤注意⼒机制计算背景向量，输入为encoder的隐藏状态和decoder上一个时间步的隐藏状态
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, num_hiddens+embedding_size)
        input_and_c = torch.cat((self.embedding(current_inputs), c), dim=1)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(dim=0)
        return output, state
