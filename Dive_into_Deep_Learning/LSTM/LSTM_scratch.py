#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/27 下午9:10
  @ Author   : Vodka
  @ File     : LSTM_scratch .py
  @ Software : PyCharm
"""
import torch
import numpy as np
import torch.nn as nn

num_inputs = 2048  # dim of one input (比如word embedding的维数)
num_hiddens = 256
num_outputs = 2048  # can be the same as input (看做是单词预测单词)
batch_size = 1
num_steps = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params():
    def _one(shape):
        # 将id转为one hot向量
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
    F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
    O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
    C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
    C = F * C + I * C_tilda
    H = O * C.tanh()
    Y = torch.matmul(H, W_hq) + b_q
    return Y, (H, C)


state = init_lstm_state(batch_size, num_hiddens, device)
inputs = torch.randn(10, 1, 2048)  # 10个单词，batch_size=1， 每个单词2048维
params = get_params()
for t in range(10):  # time step
    # 将上一时间步的输出作为当前时间步的输入
    X = inputs[t, :, :]  # t时间时候的输入
    # 计算输出和更新隐藏状态,这里是用1个lstm单元计算 得到1个输出
    (Y, state) = lstm(X, state, params)
    output = int(Y.argmax(dim=1).item())  # 每一时刻的最大可能的类别是哪个
