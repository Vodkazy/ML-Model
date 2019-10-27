#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/27 下午7:49
  @ Author   : Vodka
  @ File     : RNN_scratch .py
  @ Software : PyCharm
"""
import torch
import numpy as np
import torch.nn as nn

num_inputs = 2048  # dim of one input (比如word embedding的维数)
num_hiddens = 256
num_outputs = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


state = None
inputs = []
params = get_params()
outputs, state = rnn(inputs, state, params)
for t in range(10):  # time step
    # 将上一时间步的输出作为当前时间步的输入
    X = inputs[t]  # t时间时候的输入
    # 计算输出和更新隐藏状态
    (Y, state) = rnn(X, state, params)
    output = int(Y[0].argmax(dim=1).item())  # 每一时刻的最大可能的类别是哪个
