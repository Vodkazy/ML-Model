#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/11 下午5:14
  @ Author   : Vodka
  @ File     : AttentionModel .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel():
    def __init__(self, input_size, attention_size):
        self.model = nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(attention_size, 1, bias=False))

    def forward(self, encoder_states, decoder_state):
        """
        :param encoder_states: (time_step, batch, num_hiddens)
        :param decoder_state: (batch, num_hiddens)
        :return: (batch, num_hiddens)
        """
        decoder_states = decoder_state.unsqueeze(dim=0).expand_as(encoder_states)  # unsqueeze(dim)表示在dim维上增加一维,原有的维度顺延
        encoder_and_decoder_states = torch.cat((encoder_states, decoder_states), dim=2)  # 将解码器隐藏状态⼴播到和编码器隐藏状态形状相同后进⾏连结
        e = self.model(encoder_and_decoder_states)  # (time_step, batch, 1)
        alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
        return (alpha * encoder_states).sum(dim=0)  # 返回背景向量，每个背景向量的长度等于编码器的隐藏单元个数
