#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/27 下午7:49
  @ Author   : Vodka
  @ File     : RNN_scratch .py
  @ Software : PyCharm
"""
import numpy as np
import torch
import torch.nn as nn

num_inputs = 2048  # dim of one input (比如word embedding的维数)
num_hiddens = 256
num_outputs = 2048  # can be the same as input (看做是单词预测单词)
batch_size = 1
num_steps = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 初始化模型参数
def get_params():
    def _one(shape):
        # 将id转为one hot向量
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


# 初始化state全0
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# 单个rnn前向传播一次
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
    Y = torch.matmul(H, W_hq) + b_q
    return Y, (H,)


if __name__ == '__main__':
    state = init_rnn_state(batch_size, num_hiddens, device)
    inputs = torch.randn(10, 1, 2048)  # 10个单词，batch_size=1， 每个单词2048维
    params = get_params()
    for t in range(10):  # time step
        # 将上一时间步的输出作为当前时间步的输入
        X = inputs[t, :, :]  # t时间时候的输入
        # 计算输出和更新隐藏状态,这里是用1个rnn单元计算 得到1个输出
        (Y, state) = rnn(X, state, params)
        output = int(Y.argmax(dim=1).item())  # 每一时刻的最大可能的类别是哪个
