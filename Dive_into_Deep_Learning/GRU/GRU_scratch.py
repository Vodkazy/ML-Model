#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/27 下午8:30
  @ Author   : Vodka
  @ File     : GRU_scratch .py
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

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
    R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
    H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
    H = Z * H + (1 - Z) * H_tilda
    Y = torch.matmul(H, W_hq) + b_q
    return Y, (H,)


state = init_gru_state(batch_size, num_hiddens, device)
inputs = torch.randn(10,1,2048) # 10个单词，batch_size=1， 每个单词2048维
params = get_params()
for t in range(10):  # time step
    # 将上一时间步的输出作为当前时间步的输入
    X = inputs[t,:,:]  # t时间时候的输入
    # 计算输出和更新隐藏状态,这里是用1个gru单元计算 得到1个输出
    (Y, state) = gru(X, state, params)
    output = int(Y.argmax(dim=1).item())  # 每一时刻的最大可能的类别是哪个,或者softmax过一个
