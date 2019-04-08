#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2018/12/17 21:25
  @ Author   : Vodka
  @ File     : model.py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # input to output
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # logsoftmax
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        """
        calculate forward
        :param input:
        :param hidden:
        :return:
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.softmax(self.i2o(combined))
        return output, hidden

    def initHidden(self, hidden_size):
        """
        init the hidden layer
        :param hidden_size:
        :return:
        """
        return (torch.zeros(1, hidden_size))
