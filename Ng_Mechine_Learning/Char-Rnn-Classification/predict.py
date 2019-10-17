#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2018/12/17 21:25
  @ Author   : Vodka
  @ File     : predict.py
  @ Software : PyCharm
"""
from .datapy import *
import torch

rnn = torch.load('char-rnn-classification.pt')
hidden_size = 128


def evaluate(names_tensor):
    """
    the process is similar to that of train , while this doesn't have back-pop
    :param names_tensor:
    :return:
    """
    hidden = rnn.initHidden(hidden_size)
    for i in range(names_tensor.size()[0]):
        output, hidden = rnn(names_tensor[i], hidden)
    return output


def predict(names, n_predictions=3):
    """
    predict which countries the given name belongs to
    :param names:
    :param n_predictions:
    :return:
    """
    output = evaluate(namesToTensor(names))

    top_values, top_indexs = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = top_values[0][i]
        country_index = top_indexs[0][i]
        print('No.%s (%.2f) %s' % (i + 1, value, country_names[country_index]))
        predictions.append([value, country_names[country_index]])

    return predictions


if __name__ == '__main__':
    names = ['kyoma', 'zyyyyy', 'mirror', 'zhangjiatao', 'wavebridge', 'vodkazy']
    for name in names:
        print("The prediction of", name)
        predict(name)
